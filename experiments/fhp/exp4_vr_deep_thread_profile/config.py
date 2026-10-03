"""Fixed protocol; the underlying Experiment 2 learner is unchanged."""

from copy import deepcopy
import random

from experiments.fhp.exp2_vr_deep_lossless_24h import config as baseline

THREAD_COUNTS = (1, 2, 4, 8, 16)
REFERENCE_THREADS = 8
REPEATS = 3
WARMUP_ITERATIONS = 5
SOURCE_SEED = 0
PROFILE_MAX_SECONDS = 5 * 3600
COMPONENTS = ("regret_0", "critic_0", "regret_1", "critic_1", "policy", "whole")


def protocol(smoke=False):
    return dict(experiment_id=4, experiment_name="exp4_vr_deep_thread_profile",
                is_smoke=smoke, thread_counts=list(THREAD_COUNTS),
                reference_threads=REFERENCE_THREADS, source_seed=SOURCE_SEED,
                repeats=1 if smoke else REPEATS,
                warmup_iterations=1 if smoke else WARMUP_ITERATIONS,
                training_config=baseline.smoke_config() if smoke else deepcopy(baseline.TRAINING_CONFIG),
                machine_type="n2-standard-16", performance_evaluation=False,
                timing_protocol="independent_prefit_fixtures_and_trace_controlled_iteration_v1",
                checkpoint_policy="none; temporary profiling fixtures are not deployable policies")


def trial_order(repeats):
    """Locally seeded order does not perturb the learner's random streams."""
    rng = random.Random(4102026)
    result = []
    for repeat in range(repeats):
        counts = list(THREAD_COUNTS)
        rng.shuffle(counts)
        result.extend((repeat, threads) for threads in counts)
    return result
