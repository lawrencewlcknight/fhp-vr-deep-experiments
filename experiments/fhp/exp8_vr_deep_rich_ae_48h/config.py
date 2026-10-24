"""Experiment 6 unchanged except UCV Experiment 38 rich AE inputs."""
from copy import deepcopy
from experiments.fhp.exp6_vr_deep_ray8_48h import config as baseline
from fhp_vr_deep.rich_ae import ENCODER_ID, make_encoder, provenance
from fhp_vr_deep.io_utils import canonical_sha256

EXPERIMENT_ID = 8
EXPERIMENT_NAME = "exp8_fhp_vr_deep_rich_ae_48h"
ALGORITHM_ID = "rich_ae_vr_deep_pdcfr_plus_48h"
ALGORITHM_LABEL = "VR-DeepPDCFR+ (rich AE, native 3x64 MLPs, 48h)"
BASELINE_COMMIT = "236bf0c5e94ae90ce57545d82bbabc8bec9b7f07"
SEEDS, THREADS = baseline.SEEDS, baseline.THREADS
REFERENCE_VM = deepcopy(baseline.REFERENCE_VM)
TRAIN_MAX_SECONDS = baseline.TRAIN_MAX_SECONDS
TRAINING_CONFIG = dict(deepcopy(baseline.TRAINING_CONFIG), feature_encoder_id=ENCODER_ID)
CONFIG_SHA256 = canonical_sha256(TRAINING_CONFIG)
CHECKPOINT_HOURS, CHECKPOINT_SECONDS = baseline.CHECKPOINT_HOURS, baseline.CHECKPOINT_SECONDS

def smoke_config():
    return dict(baseline.smoke_config(), feature_encoder_id=ENCODER_ID)

def task_name(index):
    if index not in range(len(SEEDS)):
        raise ValueError("Task index must be 0, 1 or 2")
    return f"task_{index:03d}_{ALGORITHM_ID}_seed_{SEEDS[index]}"

schedule = baseline.schedule

def contract(smoke=False):
    result = baseline.contract(smoke)
    cfg = smoke_config() if smoke else deepcopy(TRAINING_CONFIG)
    result.update(experiment_id=8, experiment_name=EXPERIMENT_NAME,
        algorithm_id=ALGORITHM_ID, algorithm_label=ALGORITHM_LABEL,
        baseline_experiment=baseline.EXPERIMENT_NAME, baseline_commit=BASELINE_COMMIT,
        single_intended_change="rich_AE_inputs_and_necessary_replay_dimensions",
        training_config=cfg, training_config_sha256=canonical_sha256(cfg),
        feature_encoder=make_encoder().metadata(), encoder_source=provenance(),
        input_representation="ucv_exp38_rich_AE_231_player_359_critic",
        ucv_reference_run_id="exp38-rich48-20261019-230016",
        initialisation="fresh_from_scratch_no_warm_start",
        primary_endpoint_hours=48, evaluation="deferred_threeway_all_cross_seed_pairings",
        inference="three training seeds; historical selected UCV reference, not fresh UCV replication")
    result["replay_storage"]["next_player_information"] = "stored_separately_231_values"
    return result
