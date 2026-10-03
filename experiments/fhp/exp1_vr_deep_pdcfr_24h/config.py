"""Frozen scientific and resource contract for the new Experiment 1."""

from copy import deepcopy

from experiments.fhp.archieved_exp1_leduc_config_transfer.config import (
    APPROVED_CONFIG as LEDUC_TRANSFER_CONFIG,
    UPSTREAM,
)
from fhp_vr_deep.io_utils import canonical_sha256

EXPERIMENT_ID = 1
EXPERIMENT_NAME = "exp1_fhp_vr_deep_pdcfr_24h"
ALGORITHM_ID = "vr_deep_pdcfr_plus"
ALGORITHM_LABEL = "VR-DeepPDCFR+ (selected Leduc configuration)"
SEEDS = (0, 1, 2)
CHECKPOINT_HOURS = (6, 12, 18, 24)
CHECKPOINT_SECONDS = tuple(h * 3600 for h in CHECKPOINT_HOURS)
SMOKE_SECONDS = (0.001, 0.002, 0.003, 0.004)
ITERATION_SAFETY_CAP = 1_000_000
TRAIN_MAX_SECONDS = 36 * 3600
THREADS = 8
REFERENCE_VM = dict(machine_type="n2-standard-8", cpu_milli=8000,
                    memory_mib=30000, boot_disk_gib=200, boot_disk_type="pd-balanced")

TRAINING_CONFIG = deepcopy(LEDUC_TRANSFER_CONFIG)
TRAINING_CONFIG.update(
    evaluation_frequency=0, max_num_iterations=ITERATION_SAFETY_CAP,
    alpha=2.3, gamma=2.0, reinitialize_imm_regret_networks=True,
)
CONFIG_SHA256 = canonical_sha256(TRAINING_CONFIG)


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
    config = smoke_config() if smoke else deepcopy(TRAINING_CONFIG)
    return dict(
        experiment_id=EXPERIMENT_ID, experiment_name=EXPERIMENT_NAME,
        algorithm_id=ALGORITHM_ID, algorithm_label=ALGORITHM_LABEL,
        is_smoke=bool(smoke), seeds=[0] if smoke else list(SEEDS),
        training_config=config, training_config_sha256=canonical_sha256(config),
        checkpoint_schedule=list(schedule(smoke)), reference_vm=dict(REFERENCE_VM),
        checkpoint_boundary="first_completed_outer_iteration_after_threshold",
        training_clock="solver time excluding policy fitting, checkpoint writes, validation and uploads",
        seed_execution="parallel_batch_task_array_one_vm_per_seed",
        replay_storage=dict(continuous_dtype="float32", integer_dtype="int64",
                            derive_critic_next_state=True, sampling="legacy_global_rng"),
        input_representation="raw_openspiel_information_state",
        frozen_prediction_cache="fit_local_cpu_only_with_first_minibatch_exact_check",
        artifact_retention="all_playable_policies_analysis_metadata_no_full_training_states",
        exact_exploitability=False, upstream=deepcopy(UPSTREAM),
    )
