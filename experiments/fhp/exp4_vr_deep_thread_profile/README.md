# Experiment 4: CPU computation-thread profiling

Question: on one `n2-standard-16`, which of **1, 2, 4, 8 or 16 computation
threads** gives the fastest fitting and complete iteration of the Experiment 2
VR-DeepPDCFR+ learner? This is a timing study, not another long training run.
There is no exploitability, benchmark-opponent, LBR or head-to-head evaluation.
Experiments 1–3 and their production settings are unchanged.

## Fixed workload

- Use Experiment 2's solver factory and unchanged production configuration:
  lossless 183-value player / 263-value critic inputs, 64–64–64 MLPs, minibatches
  of 2,048, one-million-entry replay capacities, and 10,000 traversals per
  player per iteration. Traversal remains sequential Python/OpenSpiel work.
- Generate genuine replay with source seed 0 and five complete iterations at
  eight threads. Preserve all populated replay rows, models, target/immediate
  networks, Adam states, counters and Python/NumPy/Torch random states. Record
  actual occupancy; this is early-training replay, **not** a claim to reproduce
  the state after 12 or 24 hours. No artificial replay duplication is used.
- During the next reference iteration, capture independent pre-fit fixtures
  for each player's regret fit and critic fit. Capture a separate pre-fit
  average-policy fixture at the iteration boundary. The production fitting
  counts remain 750 regret minibatches per player (each updates both cumulative
  and immediate networks), 10,001 critic updates per player (the existing loop
  is inclusive), and 5,000 average-policy updates.
- Run all five thread settings in three deterministically shuffled rounds:
  **15 fresh child processes, sequentially on the same VM**. These are technical
  timing repeats on a common source state, not three independent learning seeds.
  No other benchmark trial runs concurrently. Each child sets Torch intra-op,
  OpenMP, MKL and OpenBLAS threads before doing measured work; BLAS environment
  variables are set before interpreter imports. Torch inter-op stays at one
  throughout and dynamic OpenMP/MKL thread adjustment is disabled.
- Warm up network kernels and Adam on disposable models, then restore each
  fixture. Construction, loading, hashing fixtures, kernel warm-up, validation,
  JSON writing and upload are outside the measured interval. Average-policy
  fitting is measured separately from normal active iterations.

## How the comparisons are controlled

**Fitting-only:** every setting starts each individual fit from exactly the
same saved networks, optimizer states, replay and random streams. Ordered
minibatch-index/population hashes must match the reference. All production
cache building, resets and updates are included in the timing. No cache or
learning rule is replaced. Existing numerical cache-fallback warnings remain
visible in the child logs.

**Whole iteration:** restore the same pre-iteration fixture and run the normal
iteration body: collection and fitting for player 0, then player 1. Replay the
reference chance/action stream while still consuming the usual random draws.
Execute every normal collection-network forward pass, measure its numerical
drift, then substitute the recorded reference output for constructing the
controlled workload. This keeps the numerical replay data and sampling
probabilities fixed too, avoiding zero-probability actions after numerical
drift. Check node/traversal counts, ordered minibatch hashes and a checksum of
the final numerical replay buffers (the last check is outside timing).
Feature-encoder caches start empty for every trial;
the second player benefits from caching within that iteration normally.

The whole-iteration test is therefore a **trace-controlled timing workload**,
not an autonomous training run from which a policy-quality conclusion can be
drawn. Collected inputs and sampled indices are fixed; fit-derived numerical
targets can differ after thread-dependent updates. Independently restored
fitting tests additionally use identical starting models for each fit. Endpoint
parameter differences from the eight-thread reference and non-finite checks
are recorded; cross-thread bitwise equality is not required.

Sampler hashing and trace instrumentation are inside the timings, consistently
for all settings. Absolute timings include this instrumentation overhead,
particularly during collection; the relative fitting comparison is cleaner.
Reference-generation timings include fixture I/O and are **not** used in speed
comparisons. A promising setting should subsequently be confirmed in an
ordinary, uninstrumented training pilot, ideally also with mature replay.

## Limits and failure handling

One on-demand `n2-standard-16`, 16,000 CPU milli, 62,000 MiB task memory and a
200 GiB `pd-balanced` boot disk. No Ray, extra seed VMs or controller VM.

The profiling work has a **five-hour wall-clock budget** including fixture
preparation and child startup; the Batch task has a **six-hour hard limit**
including installation, the smoke gate and uploads. These are safety caps,
not claimed runtime estimates. Full production fit lengths make this more
representative than timing a few small matrix multiplications. If the budget
is insufficient, the run fails visibly with partial measurements, not a
truncated result marked successful. Automatic retries are disabled.

The cloud workflow first runs regression tests and a tiny end-to-end smoke
profile over all five settings. Only a successful smoke permits full profiling.
The smoke uses one round, smaller replay/minibatches and two/three-update fits;
its timings must not be used to choose production threads.

An independent monitor records CPU, process memory, cgroup/OOM counters and
disk usage every 30 seconds. Completed trial outputs sync to GCS between
trials, outside timed regions. On failure, the finalisation trap attempts to
upload partial files and diagnostics; an abrupt VM-wide loss can still prevent
that final upload. Completion markers are uploaded only after the final
artifact sync succeeds. Fixtures remain in scratch space and are cleaned up;
no multi-gigabyte replay or policy snapshots are uploaded.

## Run on GCP Batch

From the repository root, with `PROJECT_ID`, `REGION`, `BUCKET` and `SA_EMAIL`
already configured. **Commit and push this implementation first**: Batch checks
out the exact pushed SHA, not local uncommitted changes.

```bash
export REPO_REF="$(git rev-parse HEAD)"

# Inspect the job configuration without submitting anything.
export RUN_ID="vr4-check-$(date -u '+%Y%m%d-%H%M%S')"
bash gcp/run_exp4_vr_deep_thread_profile.sh dry-run

# Optional standalone cloud smoke (no full profiling is launched).
export RUN_ID="vr4-smoke-$(date -u '+%Y%m%d-%H%M%S')"
bash gcp/run_exp4_vr_deep_thread_profile.sh smoke-only
bash gcp/run_exp4_vr_deep_thread_profile.sh status

# After smoke succeeds, use a fresh namespace for the full profiling job.
# This job also runs its own smoke gate automatically.
export RUN_ID="vr4-threads-$(date -u '+%Y%m%d-%H%M%S')"
bash gcp/run_exp4_vr_deep_thread_profile.sh run
bash gcp/run_exp4_vr_deep_thread_profile.sh status
```

The laptop can disconnect after submission. Batch shuts down the task when it
finishes; no manual training-process shutdown is needed. A reused GCS run
namespace is rejected to protect previous results.

Download compact results, logs and diagnostics (no models):

```bash
mkdir -p "cloud_outputs/$RUN_ID"
gcloud storage rsync --recursive \
  "${BUCKET%/}/$RUN_ID" "cloud_outputs/$RUN_ID"
```

Expected artifacts under the run prefix:

- `profile/analysis/timings.csv`: median/min/max seconds and paired speedups
  relative to eight threads, for every fit, active fitting combined, and the
  whole iteration. Prefer whole-iteration speed for the first decision.
- `profile/analysis/measurements.csv`: per-repeat raw timings, CPU seconds,
  mean busy cores, process peak RSS and numerical/workload checks.
- `profile/trials/*.json`: full phase timing breakdowns, sampler hashes,
  endpoint/inference drift, trace checks and per-process runtime details.
- `profile/fixture_manifest.json`: source configuration, actual replay
  occupancy, hashes, hardware/runtime information and warm-up history.
- `profile/protocol.json`, `trial_order.json`, `logs/*.txt`, and `SUCCESS.json`
  (or `FAILURE.json` on a handled failure).
- `smoke/`: clearly marked smoke-only outputs; `diagnostics/`: independent
  resource snapshots and failure classification.

Peak RSS is cumulative for the child process, **not** a per-phase memory delta.
The fastest observed setting is descriptive, not statistically established by
three repeats; small differences or overlapping timing ranges warrant more
measurements. No training configuration is automatically changed.

## Local checks

Use the project's Python 3.11 environment:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests/test_exp4_vr_deep_thread_profile.py
bash gcp/run_exp4_vr_deep_thread_profile.sh smoke-local
```

Local smoke outputs go to a newly created temporary directory. Do not interpret
laptop smoke timings as predictions for the GCP machine.
