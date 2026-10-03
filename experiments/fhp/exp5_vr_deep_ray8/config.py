"""Compare with Experiment 3 on identical VMs and central fitting settings."""

from copy import deepcopy

from experiments.fhp.exp3_vr_deep_lossless_n2_standard16 import config as baseline
from fhp_vr_deep.io_utils import canonical_sha256
from vr_deep_cfr.parallel_solver import (RAY_VERSION, WORKER_COUNT, OBJECT_STORE_BYTES,
                                         SMOKE_OBJECT_STORE_BYTES, COLLECTION_TIMEOUT_SECONDS)

EXPERIMENT_ID = 5
EXPERIMENT_NAME = "exp5_fhp_vr_deep_ray8"
ALGORITHM_ID = "lossless_vr_deep_pdcfr_plus_ray8"
ALGORITHM_LABEL = "VR-DeepPDCFR+ (eight Ray traversal workers, unchanged central fitting)"
BASELINE_COMMIT = "96c749410fff8ab89cda29ef28974124cb155d98"
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


def smoke_config():
    result = baseline.smoke_config()
    result["num_traversals"] = 16  # Every actor gets work for both players.
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
    training = smoke_config() if smoke else deepcopy(TRAINING_CONFIG)
    result.update(experiment_id=EXPERIMENT_ID, experiment_name=EXPERIMENT_NAME,
                  algorithm_id=ALGORITHM_ID, algorithm_label=ALGORITHM_LABEL,
                  baseline_experiment=baseline.EXPERIMENT_NAME, baseline_commit=BASELINE_COMMIT,
                  baseline_reference_vm=deepcopy(baseline.REFERENCE_VM),
                  training_config=training, training_config_sha256=canonical_sha256(training),
                  checkpoint_schedule=list(schedule(smoke)), single_intended_change="synchronous_ray_traversals",
                  traversal_execution="eight_ray_actors_frozen_snapshot_per_player_barrier_before_fitting",
                  parallel_traversal_workers=WORKER_COUNT, traversal_worker_threads=1,
                  learner_threads=THREADS, central_fitting="unchanged_serial_player_and_fitter_order",
                  ray_version=RAY_VERSION, ray_object_store_bytes=SMOKE_OBJECT_STORE_BYTES if smoke else OBJECT_STORE_BYTES,
                  collection_timeout_seconds=COLLECTION_TIMEOUT_SECONDS, actor_max_restarts=0, actor_max_task_retries=0,
                  merge_order="ascending_worker_id_regret_then_policy_then_critic",
                  traversal_rng="SeedSequence(run_seed,5102026,worker_id,iteration,player)",
                  equivalence="same_learning_rules_not_bitwise_same_sequential_sampling_stream",
                  training_clock="solver time including snapshot/distribution/collection/validation/merge; "
                                 "excluding Ray startup and checkpoint fitting/writes/validation/uploads")
    return result
