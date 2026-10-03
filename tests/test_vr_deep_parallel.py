"""Pure collection/replay tests; Ray transport is covered by the cloud gate."""

from copy import deepcopy
import random

import numpy as np
import pytest
import torch

from benchmarks.vr_deep_efficiency import assert_exact
from experiments.fhp.exp2_vr_deep_lossless_24h.train import make_solver as sequential_solver, TimedEncodedVRDeepPDCFRPlus
from experiments.fhp.exp5_vr_deep_ray8 import config
from vr_deep_cfr import parallel_solver as p
from vr_deep_cfr.solver import ReservoirBuffer, CircularBuffer, set_seed


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class InlinePool:
    """Test double running the actual actors in deterministic driver order.

    Synthetic PIDs allow checkpoint contract tests without OS processes. Only
    the separately gated real-Ray tests establish actual process isolation.
    """

    def __init__(self, training_config, seed, *, object_store_bytes):
        previous = torch.get_num_threads()
        try:
            maximum = max(p.partition(training_config["num_traversals"]))
            self.workers = [p.TraversalWorker(training_config, seed, i, maximum) for i in range(8)]
            health = [dict(worker.ping(), pid=10000 + i) for i, worker in enumerate(self.workers)]
            self.metadata = dict(backend="ray", ray_version=p.RAY_VERSION, workers=health,
                                 object_store_bytes=object_store_bytes, startup_seconds=.01,
                                 max_restarts=0, max_task_retries=0)
            self.calls, self.closed = [], False
        finally:
            torch.set_num_threads(previous)

    def collect(self, counts, player, iteration, snapshot):
        self.calls.append(deepcopy(snapshot))
        previous = torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            results = [worker.collect(count, player, iteration, snapshot)
                       for worker, count in zip(self.workers, counts) if count]
        finally:
            torch.set_num_threads(previous)
        return results, dict(dispatch_seconds=0., wait_seconds=0.)

    def close(self):
        self.closed = True


def test_counts_and_phase_streams():
    assert p.partition(10000) == [1250] * 8
    assert p.partition(19) == [3, 3, 3, 2, 2, 2, 2, 2]
    assert sum(p.partition(3)) == 3
    before = np.random.get_state()
    seeds = [p.collection_seed(seed, i, iteration, player) for seed in range(3)
             for i in range(8) for iteration in range(1, 5) for player in range(2)]
    assert len(set(seeds)) == len(seeds)
    assert_exact(before, np.random.get_state())
    assert p.collection_seed(0, 1, 2, 0) == p.collection_seed(0, 1, 2, 0)
    with pytest.raises(ValueError):
        p.partition(-1)


@pytest.mark.parametrize("player", [0, 1])
def test_worker_matches_existing_sequential_dfs_and_encoded_replay(player):
    cfg = config.smoke_config()
    source = sequential_solver(0, cfg)
    source.iteration()
    # Distinguish the online critic from its TD target explicitly.
    with torch.no_grad():
        for parameter in source.q_value_trainer.target_model.parameters():
            parameter.fill_(999)
    snapshot = p.inference_snapshot(source, player)
    worker = p.TraversalWorker(cfg, 0, 3, 5)
    actual = worker.collect(5, player, source.num_iteration, snapshot)
    worker_rng = worker.solver._capture_rng_state()
    reference_cfg = deepcopy(cfg)
    reference_cfg.update(num_traversals=5, advantage_buffer_size=140,
                         ave_policy_buffer_size=140, baseline_buffer_size=140)
    reference = sequential_solver(0, reference_cfg)
    for trainer, state in zip(reference.regret_trainers, snapshot["regrets"]):
        for key, value in state.items():
            getattr(trainer, key).load_state_dict({k: torch.from_numpy(v.copy()) for k, v in value.items()})
    reference.q_value_trainer.model.load_state_dict({k: torch.from_numpy(v.copy()) for k, v in snapshot["critic"].items()})
    reference.num_iteration = source.num_iteration
    set_seed(p.collection_seed(0, 3, source.num_iteration, player))
    reference.collect_training_data(player)
    assert actual["nodes"] == reference.nodes_touched and actual["traversals"] == 5
    for key, trainer in (("regret", reference.regret_trainers[player]),
                         ("policy", reference.ave_policy_trainer), ("critic", reference.q_value_trainer)):
        assert_exact(actual[key], p.packed(trainer.buffer))
    assert_exact(worker_rng, reference._capture_rng_state())
    assert actual["critic"]["next_state_buf"].shape[1] == 183
    assert actual["critic"]["history_buf"].shape[1] == 263
    repeat = worker.collect(5, player, source.num_iteration, snapshot)
    for key in ("regret", "policy", "critic"):
        assert_exact(actual[key], repeat[key])
    with pytest.raises(ValueError, match="stale"):
        worker.collect(5, player, source.num_iteration + 1, snapshot)


def test_staging_never_subsamples_or_discards_experience():
    reservoir = p.StagingReservoir(1, 3, 3)
    reservoir.add(np.zeros(3), np.zeros(3), np.ones(3), 1)
    with pytest.raises(RuntimeError, match="drop rows"):
        reservoir.add(np.zeros(3), np.zeros(3), np.ones(3), 1)
    circular = p.StagingCircular(1, 4, 3, 3)
    values = (np.zeros(4), 1, np.zeros(4), np.zeros(3), np.ones(3), 0, 0, 1.)
    circular.add(*values)
    with pytest.raises(RuntimeError, match="overwrite"):
        circular.add(*values)


def test_reservoir_merge_matches_legacy_insertion_and_rng():
    source = ReservoirBuffer(30, 3, 3)
    for i in range(30):
        source.add(np.full(3, i), np.full(3, -i), np.ones(3), i + 1)
    rows = p.packed(source)
    for initial in (0, 3, 14):
        actual, reference = ReservoirBuffer(7, 3, 3), ReservoirBuffer(7, 3, 3)
        for buffer in (actual, reference):
            np.random.seed(21)
            for i in range(initial):
                buffer.add(np.zeros(3), np.zeros(3), np.ones(3), 1)
        np.random.seed(44)
        p.append_reservoir(actual, rows)
        after = np.random.get_state()
        np.random.seed(44)
        for i in range(30):
            reference.add(rows["infostate_buf"][i], rows["q_value_buf"][i],
                          rows["q_value_mask_buf"][i], rows["iteration_buf"][i])
        assert actual.cur_id == reference.cur_id == initial + 30
        assert_exact(p.packed(actual), p.packed(reference))
        assert_exact(after, np.random.get_state())


def test_circular_merge_matches_legacy_wraparound_and_all_columns():
    source = CircularBuffer(19, 5, 3, 3)
    values = [(np.full(5, i), i % 3, np.full(5, -i), np.full(3, 2 * i), np.ones(3), i % 2, i % 2, float(i))
              for i in range(19)]
    for row in values:
        source.add(*row)
    actual, reference = CircularBuffer(7, 5, 3, 3), CircularBuffer(7, 5, 3, 3)
    for buffer in (actual, reference):
        buffer.add(*values[0])
    p.append_circular(actual, p.packed(source))
    for row in values:
        reference.add(*row)
    assert actual.cur_id == reference.cur_id and actual.size == reference.size
    assert_exact(p.packed(actual), p.packed(reference))


def test_central_update_methods_are_inherited_unchanged():
    for method in ("dfs", "iteration", "train_regret", "train_baseline", "train_average_policy"):
        assert getattr(p.ParallelVRDeepPDCFRPlus, method) is getattr(TimedEncodedVRDeepPDCFRPlus, method)


def test_startup_preserves_post_initialisation_rng_even_when_it_fails():
    cfg = config.smoke_config()
    expected_solver = sequential_solver(2, cfg)
    expected = expected_solver._capture_rng_state()

    def noisy(*args, **kwargs):
        random.random()
        np.random.random(10)
        torch.rand(7)
        raise RuntimeError("startup failed")

    with pytest.raises(RuntimeError, match="startup failed"):
        p.make_solver(2, cfg, pool_factory=noisy, smoke=True)
    assert_exact(expected, expected_solver._capture_rng_state())
    solver = p.make_solver(2, cfg, pool_factory=InlinePool, smoke=True)
    try:
        assert_exact(expected, solver._capture_rng_state())
        for actual, reference in zip(solver.regret_trainers, expected_solver.regret_trainers):
            assert_exact(actual.model.state_dict(), reference.model.state_dict())
    finally:
        solver.close()


def test_player_barrier_refreshes_weights_and_budget_is_not_multiplied():
    cfg = config.smoke_config()
    solver = p.make_solver(0, cfg, pool_factory=InlinePool, smoke=True)
    pool = solver.pool
    try:
        solver.iteration()
        assert [tuple(s["token"]) for s in pool.calls] == [(1, 0), (1, 1)]
        before, after = pool.calls
        assert any(not np.array_equal(before["regrets"][0]["model"][k], after["regrets"][0]["model"][k])
                   for k in before["regrets"][0]["model"])
        assert any(not np.array_equal(before["critic"][k], after["critic"][k]) for k in before["critic"])
        assert solver.episode == 2 * cfg["num_traversals"] == 32
        assert solver.parallel_totals["collections"] == 2
        assert solver.phase_seconds["collection"] > 0 and solver.phase_seconds["critic_fitting"] > 0
        assert p.payload_bytes(pool.calls[0]) > 0
    finally:
        solver.close()
    assert pool.closed


@pytest.mark.parametrize("fault", ["missing", "reverse", "stale", "nan", "dropped_rows"])
def test_invalid_phase_aborts_before_replay_merge(fault):
    class BrokenPool(InlinePool):
        def collect(self, *args):
            results, timing = super().collect(*args)
            if fault == "missing":
                results.pop()
            elif fault == "reverse":
                results.reverse()
            elif fault == "stale":
                results[0]["token"] = (0, 0)
            elif fault == "dropped_rows":
                results[0]["critic"] = {key: value[:0] for key, value in results[0]["critic"].items()}
                results[0]["payload_bytes"] = p.payload_bytes(results[0])
            else:
                results[0]["critic"]["history_buf"][0, 0] = np.nan
            return results, timing

    solver = p.make_solver(0, config.smoke_config(), pool_factory=BrokenPool, smoke=True)
    pool = solver.pool
    solver.training_time_checkpoint_seconds = (.001,)
    try:
        with pytest.raises(ValueError):
            solver.solve()
        assert solver.episode == 0 and len(solver.q_value_trainer.buffer) == 0
        assert solver.checkpoint_rows == [] and pool.closed
    finally:
        solver.close()
