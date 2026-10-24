# UCV Experiment 38 rich AE features

Pinned from UCV commit `6ab85bbcd03627281e411243419d64ce9e9236d5`, the
Experiment 36 source continued by Experiment 38. `provenance.json` records
original and vendored file hashes. All feature arithmetic is unchanged.
Only the unused neural architecture, factories, and Torch imports were removed
from `features.py`; other source files are byte-identical. The AE factory does
not load the unrelated preflop-D asset. No sibling checkout is needed.

Player inputs: 231 float32 values. Critic inputs: 359 float32 values.
All player features use only own cards and public history. Equity is against
1,081 compatible uniform holdings, never the actual opponent holding or range.
Legacy encoder IDs remain separate. This adds features, not UCV learning rules.
