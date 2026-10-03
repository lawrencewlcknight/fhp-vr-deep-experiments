# VR-Deep implementation efficiency audit

These are implementation changes, not a new algorithm configuration. The
Archived Experiment 1 architecture, losses, replay capacities, sampling without
replacement, discounting, optimiser updates, target synchronisation, seeding,
average-policy fitting schedule and playable-policy checkpoint format remain
unchanged. Shared replay/traversal changes also benefit VR-DeepDCFR+; frozen
prediction caching is specific to VR-DeepPDCFR+.

## Changes

- Replay floating-point arrays now use float32, the same dtype previously
  produced at every sampling call. The integer fields remain int64. No
  float16, quantisation, reduced network precision or new abstraction is used.
- Buffers allocate uninitialised storage and read only populated rows. Reset
  reuses allocations. Sampling still calls Python's `random.sample` with the
  same arguments and preserves the reservoir replacement RNG sequence.
- Each sampled index list is converted to a NumPy index array once. Full-data
  reads use prefix slices. These tensors can share replay storage and must be
  treated as read-only; the fitting objectives do not mutate them.
- The critic's next-state column is reconstructed from the next acting
  player's half of the stored next-history tensor. Terminal next states remain
  zero. This is explicitly enabled only for the existing two-player
  concatenated-history encoding. Policy and regret inputs never gain access
  to the opponent's private information.
- Traversal reuses legal actions, masks, information-state and history inputs.
  The two predictive regret forwards share tensor preparation. Explicit tensor
  device placement replaces per-forward default-device contexts. The
  iteration discount power is reused without reassociating multiplication and
  division.
- During cumulative-regret fitting, the previous target network and collected
  rows are fixed. A fit-local cache can hold the resulting regret targets.
- During critic fitting, both players' cumulative and immediate regret
  networks are fixed. A fit-local cache holds the next-state strategies.
  **The target critic is not cached**: its forwards and refreshes at steps
  0, 50, 100, ... remain unchanged. The original inclusive critic update loop
  (`train_steps + 1`) is also retained.

Caches are rebuilt at every fit, including after replay wraparound, refill,
network reinitialisation or a new iteration. They use at most three float32
values per populated row (12 MB at one million rows) for each cache. They are
not retained in snapshots. A cache is skipped when building it would require
at least as many inference batches as direct fitting. Cache construction
consumes no RNG, uses the training batch size and pads the last chunk with
populated rows. A first-minibatch exact comparison falls back to direct
inference if the cache disagrees numerically; this is a runtime spot check,
not a proof for every row or platform.

Caching is enabled on CPU only, the production experiment backend. Other
devices retain direct inference pending equivalence validation. For a
controlled cache-only comparison, set `cache_frozen_predictions = False` on
each predictive regret trainer and on the critic trainer.

We deliberately retain even apparently redundant random model initialisations:
removing them would advance the NumPy RNG differently and change future
trajectories. We also retain all average-policy fits and diagnostics. Removing
intermediate fits would change the observation protocol; their time is already
excluded from the active-training clock.

## Memory

For the current FHP tensors (190 information features, 380 history features,
three actions) and the three one-million-row reservoirs plus one-million-row
critic buffer, backing arrays fall from **11.53 GiB to 5.08 GiB**, approximately
56% less. This excludes network/optimiser memory, temporary minibatches,
fit-local caches, and Python/OpenSpiel overhead. It is not a peak-RSS promise.

## Reproducible equivalence and timing check

Use a full checkout containing the pre-optimisation commit
`4b5d6c8d32126dd34fff887a12d0b9fa88096136`. The audit loads its Python sources
directly from Git into a separate in-memory namespace; it does not alter the
checkout or download anything.

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider
PYTHONDONTWRITEBYTECODE=1 python -m benchmarks.vr_deep_efficiency
```

A larger, still shortened audit uses the production network shape, 2,048-row
minibatches and 750 regret updates:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m benchmarks.vr_deep_efficiency \
  --iterations 2 --traversals 2048 --capacity 16384 --batch 2048 \
  --regret-steps 750 --critic-steps 201 --policy-steps 500 --seeds 0 1 2
```

Both audits require **exact equality**, not merely close final policies, for
model weights, optimiser states, populated replay tensors, RNG states, node
counts and non-timing learning observations. Failure exits nonzero. Tests also
cover both algorithm variants, terminal transitions, replay replacement and
wraparound, partial/full batches, repeated fits, target refreshes, cache
fallback, and two-thread CPU execution. Reference comparisons explicitly skip
when a shallow/source-only checkout lacks the pinned commit.

Initial local validation used macOS ARM, Python 3.12, PyTorch 2.7.0 and NumPy
1.26.4; the cloud environment remains Python 3.11/Linux. Three-seed small and
larger audits matched exactly. The larger audit observed approximately 1.5x
median speed-up, but uses shortened fitting and smaller buffers, and is not a
forecast for a 12/24-hour cloud run. Repeat on the intended VM with populated
production-size replay before projecting wall-clock savings. Identical outputs
are validated at fixed work on the tested backend, not guaranteed across all
hardware, kernels or library versions.

At a fixed active-time budget, a faster implementation intentionally performs
more work, so its final policy need not match the earlier time-limited run.
