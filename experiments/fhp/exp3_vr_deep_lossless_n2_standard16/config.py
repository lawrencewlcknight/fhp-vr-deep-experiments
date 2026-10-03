"""VM-only replication of the 24-hour lossless VR-DeepPDCFR+ baseline."""

from copy import deepcopy

from experiments.fhp.exp2_vr_deep_lossless_24h import config as baseline
from fhp_vr_deep.io_utils import canonical_sha256

EXPERIMENT_ID = 3
EXPERIMENT_NAME = "exp3_fhp_vr_deep_lossless_n2_standard16"
ALGORITHM_ID = "lossless_vr_deep_pdcfr_plus_n2_16"
ALGORITHM_LABEL = "VR-DeepPDCFR+ (Experiment 2 configuration, n2-standard-16)"
BASELINE_COMMIT = "247b9a3cced860b601f3d6b6d43a36d4a54d06d5"
SEEDS = baseline.SEEDS
CHECKPOINT_HOURS = baseline.CHECKPOINT_HOURS
CHECKPOINT_SECONDS = baseline.CHECKPOINT_SECONDS
SMOKE_SECONDS = baseline.SMOKE_SECONDS
ITERATION_SAFETY_CAP = baseline.ITERATION_SAFETY_CAP
TRAIN_MAX_SECONDS = baseline.TRAIN_MAX_SECONDS
THREADS = baseline.THREADS
TRAINING_CONFIG = deepcopy(baseline.TRAINING_CONFIG)
CONFIG_SHA256 = canonical_sha256(TRAINING_CONFIG)
REFERENCE_VM = deepcopy(baseline.REFERENCE_VM)
# Match UCV-ESCHER Experiment 6's actual Batch allocation, including OS headroom.
REFERENCE_VM.update(machine_type="n2-standard-16", cpu_milli=16000, memory_mib=62000)


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
    training = smoke_config() if smoke else deepcopy(TRAINING_CONFIG)
    result.update(
        experiment_id=EXPERIMENT_ID, experiment_name=EXPERIMENT_NAME,
        algorithm_id=ALGORITHM_ID, algorithm_label=ALGORITHM_LABEL,
        baseline_experiment=baseline.EXPERIMENT_NAME, baseline_commit=BASELINE_COMMIT,
        baseline_reference_vm=deepcopy(baseline.REFERENCE_VM),
        reference_vm=deepcopy(REFERENCE_VM), training_config=training,
        training_config_sha256=canonical_sha256(training), checkpoint_schedule=list(schedule(smoke)),
        single_intended_change="vm_size", learner_threads=THREADS,
        traversal_execution="sequential_single_collector", parallel_traversal_workers=0,
        performance_evaluation="none_deferred_to_separate_experiment",
        expected_speedup="not_assumed_measure_completed_iterations_nodes_and_phase_times",
    )
    return result
