"""Only extend Experiment 5's clock and add final-state persistence."""

from copy import deepcopy
import re

from experiments.fhp.exp5_vr_deep_ray8 import config as baseline
from fhp_vr_deep.io_utils import canonical_sha256

EXPERIMENT_ID = 6
EXPERIMENT_NAME = "exp6_fhp_vr_deep_ray8_48h"
ALGORITHM_ID = "lossless_vr_deep_pdcfr_plus_ray8_48h"
ALGORITHM_LABEL = "VR-DeepPDCFR+ (eight Ray workers, 48-hour baseline)"
BASELINE_COMMIT = "d7001548801a58578bcfe458f3d29eb00fadbc22"
SEEDS = baseline.SEEDS
THREADS = baseline.THREADS
REFERENCE_VM = deepcopy(baseline.REFERENCE_VM)
TRAINING_CONFIG = deepcopy(baseline.TRAINING_CONFIG)
CONFIG_SHA256 = canonical_sha256(TRAINING_CONFIG)
CHECKPOINT_HOURS = tuple(range(6, 49, 6))
CHECKPOINT_SECONDS = tuple(h * 3600 for h in CHECKPOINT_HOURS)
SMOKE_SECONDS = tuple((i + 1) * .001 for i in range(8))
TRAIN_MAX_SECONDS = 72 * 3600


def smoke_config():
    return baseline.smoke_config()


def task_name(index):
    if index not in range(len(SEEDS)):
        raise ValueError("Task index must be 0, 1 or 2")
    return f"task_{index:03d}_{ALGORITHM_ID}_seed_{SEEDS[index]}"


class RunSpec:
    """A new output namespace holds only this segment's policy checkpoints."""

    EXPERIMENT_NAME = EXPERIMENT_NAME
    ALGORITHM_ID = ALGORITHM_ID
    ALGORITHM_LABEL = ALGORITHM_LABEL
    SEEDS = SEEDS
    THREADS = THREADS
    task_name = staticmethod(task_name)

    def __init__(self, start_hours=0, target_hours=48, resume_run_id=None):
        if (not isinstance(start_hours, int) or not isinstance(target_hours, int)
                or start_hours < 0 or start_hours % 6 or target_hours % 6
                or target_hours <= start_hours or target_hours - start_hours > 48):
            raise ValueError("Use increasing six-hour boundaries and at most 48 additional hours")
        if start_hours == 0 and (target_hours != 48 or resume_run_id is not None):
            raise ValueError("Fresh Experiment 6 runs train from scratch to 48 hours")
        if start_hours and (start_hours < 48 or not resume_run_id
                            or not re.fullmatch(r"[a-z][a-z0-9-]{1,34}", resume_run_id)):
            raise ValueError("Continuation requires a source run ID and start_hours >= 48")
        self.start_hours, self.target_hours = start_hours, target_hours
        self.resume_run_id = resume_run_id

    def schedule(self, smoke=False):
        hours = tuple(range(self.start_hours + 6, self.target_hours + 1, 6))
        return tuple(dict(checkpoint_id=f"time_{h:02d}h", checkpoint_target_hours=h,
                          checkpoint_target_seconds=(i + 1) * .001 if smoke else h * 3600)
                     for i, h in enumerate(hours))

    def contract(self, smoke=False):
        if smoke and self.start_hours:
            raise ValueError("Checkpoint smoke is fresh; resume equivalence has a separate test")
        result = baseline.contract(smoke)
        result.update(experiment_id=EXPERIMENT_ID, experiment_name=EXPERIMENT_NAME,
                      algorithm_id=ALGORITHM_ID, algorithm_label=ALGORITHM_LABEL,
                      baseline_experiment=baseline.EXPERIMENT_NAME, baseline_commit=BASELINE_COMMIT,
                      single_intended_change="training_duration_plus_resumable_final_state",
                      checkpoint_schedule=list(self.schedule(smoke)),
                      start_hours=self.start_hours, target_hours=self.target_hours,
                      resume_run_id=self.resume_run_id,
                      artifact_retention="six_hour_policies_and_final_full_training_state",
                      final_state_schema="fhp_vr_ray_training_state_v1",
                      training_clock="cumulative active solver time; excludes Ray startup, downtime, "
                                     "checkpoint policy fitting, persistence, validation and uploads")
        return result


def schedule(smoke=False):
    return RunSpec().schedule(smoke)


def contract(smoke=False):
    return RunSpec().contract(smoke)
