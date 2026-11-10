"""Frozen output-only diagnostic: no new self-play or checkpoint selection."""
import json
from pathlib import Path

PINS = json.loads(Path(__file__).with_name('source_pins.json').read_text())
SEEDS = (0, 1, 2)
REPLICATES = (0, 1)
ARMS = ('native_refit', 'ucv_refit')
VR_BUCKET = 'gs://clever-overview-399515-fhp-vr-deep-results'
UCV_BUCKET = 'gs://clever-overview-399515-fhp-escher-results'


def contract(smoke=False):
    return dict(experiment=10, protocol='vr10_policy_extraction_v1', smoke=smoke,
        seeds=[0] if smoke else list(SEEDS), replicates=[0] if smoke else list(REPLICATES),
        arms=list(ARMS), updates={'native_refit':2 if smoke else 5000,
                                 'ucv_refit':3 if smoke else 20000},
        batch_size=8 if smoke else 2048, gamma=2.,
        learning_rates={'native_refit':.001,'ucv_refit':.003},
        native_layers=[64,64,64], ucv_trunk=[192,192], branch_width=64,
        pairs_per_cell=4 if smoke else 100000, shard_pairs=2 if smoke else 5000,
        init_seed_base=101100, draw_seed_base=102100, deal_seed_base=103100,
        primary='ucv_refit minus archived VR payoff against the fixed UCV cohort',
        fidelity='full-reservoir weighted KL/TV; descriptive training-replay fidelity, not held-out error',
        inference='crossed VR/UCV training-seed bootstrap after averaging fit repetitions; n=3 per cohort',
        scope='FHP fixed 48h Experiment 8 replay; output-fitting package only; no exploitability',
        source_pins=PINS)
