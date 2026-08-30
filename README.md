# FHP VR-Deep Experiments

This is a clean, standalone repository for VR-Deep experiments on two-player
flop hold'em poker (FHP). Its layout and result contracts follow the Leduc
`escher-architecture` repository, while the game definition is identical to the
existing canonical FHP Deep CFR repository in this workspace.

The repository supports both released VR-Deep variants:

- VR-DeepDCFR+;
- VR-DeepPDCFR+.

## Current status

Experiment 1 is scaffolded as a direct transfer of the Leduc VR-Deep training
configuration to FHP. Its proposed configuration is intentionally guarded and
cannot start production training until it is approved. Validate it with:

```bash
python -m experiments.fhp.exp1_leduc_config_transfer.run --validate-config
```

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
