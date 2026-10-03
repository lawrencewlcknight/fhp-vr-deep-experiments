# FHP VR-Deep Experiments

## Shared policy evaluation

FHP snapshots are evaluated through the sibling `fhp-evaluation-suite`, not
through an algorithm-specific copy. The adapter is
`fhp_vr_deep.evaluation_adapter`. Install it from this directory with
`python -m pip install -e ../../fhp-evaluation-suite`; then run
`fhp-evaluate benchmark SNAPSHOT --deals 10000 --seed 2026` or
`fhp-evaluate lbr SNAPSHOT --deals 1000 --seed 2026`.

Those generic CLI commands apply to raw-input snapshots. For Experiment 2's
encoded snapshots, use `fhp_vr_deep.evaluation_adapter.evaluate_checkpoint` or
its `load_policy_for_evaluation` function with the shared evaluators. The native
versioned loader correctly handles both representations; the generic suite
loader currently assumes raw OpenSpiel inputs.

The benchmark uses both-seat duplicate deals and corrected LooseAggressive
bands `(-300,-100)`. LBR is reported as a lower bound, not exact exploitability.

This is a clean, standalone repository for VR-Deep experiments on two-player
flop hold'em poker (FHP). Its layout and result contracts follow the Leduc
`escher-architecture` repository, while the game definition is identical to the
existing canonical FHP Deep CFR repository in this workspace.

The repository supports both released VR-Deep variants:

- VR-DeepDCFR+;
- VR-DeepPDCFR+.

## Current status

**Experiment 1: selected VR-DeepPDCFR+ on FHP** trains seeds 0, 1 and 2 for
24 active hours each, on three separate `n2-standard-8` VMs. It retains playable
policies at 6, 12, 18 and 24 hours, without large full training-state archives.
A cloud smoke/equivalence gate precedes training and strict aggregation follows.
See [the experiment specification and launch instructions](experiments/fhp/exp1_vr_deep_pdcfr_24h/README.md).

**Experiment 2: lossless input representation** repeats that configuration and
budget with the exact UCV-ESCHER Experiment 2 encoder. It retains the same flat
64–64–64 networks and VR-Deep learning rules. See
[the encoder-transfer specification](experiments/fhp/exp2_vr_deep_lossless_24h/README.md).

**Experiment 3: larger VM, unchanged Experiment 2 learner** uses three
`n2-standard-16` VMs for the same 24 active hours and 6/12/18/24h checkpoints.
Eight fitting threads, sequential traversals, networks and learning settings
are unchanged. There is no performance evaluation or full-state continuation.
See [the VM-only specification and cloud smoke/full-run instructions](experiments/fhp/exp3_vr_deep_lossless_n2_standard16/README.md).

**Experiment 4: computation-thread profiling** compares 1, 2, 4, 8 and 16
threads on one `n2-standard-16`, using shared real replay, matched starting
states/minibatches and three timing repeats. It measures individual fits and
trace-controlled whole iterations, with no poker-performance evaluation.
See [the profiling protocol and cloud smoke/full-run instructions](experiments/fhp/exp4_vr_deep_thread_profile/README.md).

The original two-algorithm Leduc-to-FHP transfer experiment is retained under
the `archieved_` prefix; it is not the active Experiment 1.

Archived Experiment 1 retains the approved transfer configuration for both
VR-Deep variants and paired seeds. To explicitly rerun the archived study:

```bash
python -m experiments.fhp.archieved_exp1_leduc_config_transfer.run
```

The full training, evaluation, smoke-test, GCP Batch, split-job, recovery, and
output contracts are documented in
`experiments/fhp/archieved_exp1_leduc_config_transfer/README.md`.

For one-time Google Cloud setup and the complete step-by-step Batch workflow,
see `docs/GCP_BATCH_EXPERIMENTS.md`.

## Canonical FHP contract

`fhp_vr_deep/game.py` contains the exact OpenSpiel `universal_poker`
parameters used by `fhp-poker-deep-cfr-experiments` at commit `539c05b`:

- two players, two rounds, two private cards, and a three-card flop;
- 13 ranks and four suits;
- blinds `50 100`, fixed raises `100 100`, and at most three raises per round;
- first players `1 2` in the OpenSpiel ACPC parameter convention.

Contract tests pin the complete parameter dictionary and the observed OpenSpiel
game shape so the variant cannot drift silently.

## Analysis contract

FHP's tree is too large for the exact tabular exploitability evaluation used
during Leduc training. This repository therefore retains the same experiment
identity, seed pairing, node/time accounting, CSV/JSON manifests, aggregation,
mean/standard-error reporting, and plot conventions, while policy quality is
measured by sampled, seat-swapped head-to-head play between reloadable policy
snapshots. See `docs/ANALYSIS_CONVENTIONS.md`.

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
python -m pip install -e .
pytest
```

## Provenance

The VR-Deep implementation is adapted from `rpSebastian/DeepPDCFR` commit
`9f156c9fcdac7f8c9bd0debf94c9432d222858d3`, with the correctness corrections
already audited in the Leduc ESCHER-architecture repository. Full details and
the upstream licensing caveat are in `vr_deep_cfr/UPSTREAM.md`.
