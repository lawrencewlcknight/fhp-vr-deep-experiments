# Exact UCV Experiment 2 encoder transfer

`fhp_vr_deep/features.py` vendors the base `FHPFeatureEncoder`, its layout and
its helper functions from the local UCV-ESCHER implementation. It is not an
implementation from the external VR-Deep authors.

- Repository: https://github.com/lawrencewlcknight/fhp-ucv-escher-experiments
- Commit: `e60bd6c82c2a139d87b9606a7cdaddb1a5f32795`
- Source path: `fhp_escher/features.py`
- Original whole-file SHA-256: `9c799a9b24465dce30de5a9d50084893b086d85445f7670425ba2147266cb132`
- Encoder ID/version: `fhp_lossless_suit_canonical_v1`, version 1
- Player input: 183 float32 values; full-state critic input: 263 float32 values.

The feature calculations and metadata are unchanged. The source's later
`StructuredFHPMLP`, feature-extension factory and architecture-specific Torch
imports are excluded. The VR copy adds a strict metadata-to-encoder loader and
attribution docstring. No runtime dependency on the UCV repository is needed.
No feature additions from later UCV experiments are included.

`tests/fixtures/ucv_exp2_encoder_golden.json` contains 222 player/full-state
reference cases produced directly by the original encoder, not the new copy.
The cases cover both players, preflop/flop and multiple betting sequences from
40 deterministic random trajectories (sampling seed 20261003). Each includes
the replayable OpenSpiel history and SHA-256 of the float32 feature bytes.
The cloud tests compare these hashes exactly and separately test all 24 global
suit permutations, opponent-card noninterference, every reachable betting
decision history (with one representative chance deal per history), explicit
next-player replay, and encoded snapshot reloads.

Canonicalisation merges only payoff-equivalent suit names, retaining exact
rank/private/public relationships. Five-card categories are derived from the
player's own two cards plus the three-card public flop; they are not privileged
showdown comparisons. Critic canonicalisation uses both private hands but its
vector never supplies a slice to the regret or average-policy network.

Terminal folds are omitted by the inherited compact betting decoder. This is
safe in its supported scope: encoding decisions and critic transitions whose
terminal next values are masked by the separate done flag. It is not a general
lossless serialisation of terminal game states. Do not reuse the encoder for a
different game without validation of the hard-coded FHP contract.

The full-state vector becomes smaller, but replay must now explicitly retain
next-player features, which raw Experiment 1 could reconstruct by slicing.
Consequently the dimensional reduction is not the same as the reduction in
total replay memory. Worker summaries report allocated replay-array bytes and
network parameter counts; memory diagnostics report actual peak RSS.
