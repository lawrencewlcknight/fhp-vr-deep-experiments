"""Same scientific configuration and resource/time budget as Experiment 1."""

from copy import deepcopy

from experiments.fhp.exp1_vr_deep_pdcfr_24h import config as baseline
from fhp_vr_deep.features import FHPFeatureEncoder
from fhp_vr_deep.io_utils import canonical_sha256

EXPERIMENT_ID = 2
EXPERIMENT_NAME = "exp2_fhp_vr_deep_lossless_24h"
ALGORITHM_ID = "lossless_vr_deep_pdcfr_plus"
ALGORITHM_LABEL = "VR-DeepPDCFR+ (UCV Exp.2 encoder, unchanged 3x64 MLPs)"
BASELINE_COMMIT = "af3b3b47af54217d0fb22499d9aaf31730a3a7ce"
SEEDS = baseline.SEEDS
CHECKPOINT_HOURS = baseline.CHECKPOINT_HOURS
CHECKPOINT_SECONDS = baseline.CHECKPOINT_SECONDS
SMOKE_SECONDS = baseline.SMOKE_SECONDS
ITERATION_SAFETY_CAP = baseline.ITERATION_SAFETY_CAP
TRAIN_MAX_SECONDS = baseline.TRAIN_MAX_SECONDS
THREADS = baseline.THREADS
REFERENCE_VM = deepcopy(baseline.REFERENCE_VM)
TRAINING_CONFIG = deepcopy(baseline.TRAINING_CONFIG)
CONFIG_SHA256 = canonical_sha256(TRAINING_CONFIG)
ENCODER_SOURCE = dict(
    repository="https://github.com/lawrencewlcknight/fhp-ucv-escher-experiments",
    commit="e60bd6c82c2a139d87b9606a7cdaddb1a5f32795",
    path="fhp_escher/features.py",
    file_sha256="9c799a9b24465dce30de5a9d50084893b086d85445f7670425ba2147266cb132",
)


def smoke_config():
    result = deepcopy(TRAINING_CONFIG)
    for name in ("advantage", "ave_policy", "baseline"):
        result[f"{name}_buffer_size"] = 128
        result[f"{name}_batch_size"] = 2
        result[f"{name}_network_train_steps"] = 2
    result["num_traversals"] = 4
    return result


def task_name(index):
    if index not in range(len(SEEDS)):
        raise ValueError("Task index must be 0, 1 or 2")
    return f"task_{index:03d}_{ALGORITHM_ID}_seed_{SEEDS[index]}"


def schedule(smoke=False):
    return tuple(dict(checkpoint_id=f"time_{h:02d}h", checkpoint_target_hours=h,
                      checkpoint_target_seconds=s)
                 for h, s in zip(CHECKPOINT_HOURS, SMOKE_SECONDS if smoke else CHECKPOINT_SECONDS))


def contract(smoke=False):
    result = baseline.contract(smoke)
    config = smoke_config() if smoke else deepcopy(TRAINING_CONFIG)
    result.update(experiment_id=EXPERIMENT_ID, experiment_name=EXPERIMENT_NAME,
                  algorithm_id=ALGORITHM_ID, algorithm_label=ALGORITHM_LABEL,
                  baseline_experiment=baseline.EXPERIMENT_NAME, baseline_commit=BASELINE_COMMIT,
                  training_config=config, training_config_sha256=canonical_sha256(config),
                  checkpoint_schedule=list(schedule(smoke)),
                  feature_encoder=FHPFeatureEncoder().metadata(), encoder_source=deepcopy(ENCODER_SOURCE),
                  input_representation="ucv_exp2_lossless_suit_canonical_v1",
                  network_architecture="unchanged_flat_mlp_64_64_64")
    result["replay_storage"].update(derive_critic_next_state=False,
                                   next_player_information="stored_separately_183_values")
    return result
