"""Numerical/RNG invariants for the implementation-only efficiency changes."""

import random
import subprocess

import numpy as np
import pytest
import torch

from benchmarks.vr_deep_efficiency import assert_exact, compare, load_reference, run_fixed_work
from vr_deep_cfr import variants
from vr_deep_cfr.frozen_cache import build_frozen_cache
from vr_deep_cfr.solver import CircularBuffer, ReservoirBuffer


@pytest.fixture
def reference():
    try:
        return load_reference()
    except subprocess.CalledProcessError:
        pytest.skip("Pinned reference commit requires a full Git checkout")


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("num_samples", [-1, 1, 7, 99])
def test_reservoir_rounding_sampling_overflow_and_reset(reference, num_samples):
    import sys
    old_cls = sys.modules[reference.__package__ + ".solver"].ReservoirBuffer
    buffers = [old_cls(17, 11, 3), ReservoirBuffer(17, 11, 3)]
    for buffer in buffers:
        np.random.seed(201)
        for i in range(71):
            buffer.add(np.arange(11) / 7 + i, [1 / 3, -1 / 7, i / 11], [1, 0, 1], i)
    assert buffers[1].infostate_buf.dtype == np.float32
    for round_index in range(2):
        samples, rngs = [], []
        for buffer in buffers:
            random.seed(71)
            samples.append(buffer.sample(num_samples))
            rngs.append(random.getstate())
        assert_exact(samples[0], samples[1])
        assert rngs[0] == rngs[1]
        for buffer in buffers:
            buffer.reset()
            for i in range(3):
                buffer.add(np.arange(11) / 9 + i, [i, 0, 1], [0, 1, 1], i + 1)


@pytest.mark.parametrize("derive", [False, True])
@pytest.mark.parametrize("num_samples", [-1, 1, 5])
def test_critic_replay_wraparound_and_terminal_reconstruction(reference, derive, num_samples):
    import sys
    old_cls = sys.modules[reference.__package__ + ".solver"].CircularBuffer
    old = old_cls(19, 22, 11, 3)
    new = CircularBuffer(19, 22, 11, 3, derive_next_state=derive)
    rng = np.random.default_rng(7)
    for _ in range(2):
        for i in range(61):
            history, next_history = rng.normal(size=(2, 22))
            player, done = i % 2, int(i % 3 == 0)
            state = np.zeros(11) if done else next_history.reshape(2, 11)[player]
            for buffer in (old, new):
                buffer.add(history, i % 3, next_history, state, [1, 0, 1], player, done, i / 7)
        random.seed(77)
        expected = old.sample(num_samples)
        expected_rng = random.getstate()
        random.seed(77)
        actual = new.sample(num_samples)
        assert_exact(expected, actual)
        assert random.getstate() == expected_rng
        assert old.cur_id == new.cur_id and len(old) == len(new)
        old.reset()
        history_storage = new.history_buf
        new.reset()
        assert new.history_buf is history_storage
        assert new.size == new.cur_id == 0
    if derive:
        assert new.next_state_buf is None


def test_cache_pads_without_rng_or_unpopulated_reads():
    inputs = torch.arange(33, dtype=torch.float32).reshape(11, 3)
    shapes = []
    before = (random.getstate(), np.random.get_state(), torch.random.get_rng_state())
    def predict(indices):
        x = inputs[indices]
        shapes.append(x.shape)
        return x * 2
    cache = build_frozen_cache(11, 4, 20, predict, device="cpu")
    assert shapes == [torch.Size([4, 3])] * 3
    np.testing.assert_array_equal(cache, inputs.numpy() * 2)
    after = (random.getstate(), np.random.get_state(), torch.random.get_rng_state())
    assert_exact(before, after)
    assert build_frozen_cache(11, 4, 2, predict, device="cpu") is None
    assert build_frozen_cache(11, 4, 20, predict, device="cuda") is None
    assert build_frozen_cache(0, 4, 20, predict, device="cpu") is None


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("predictive", [False, True])
def test_fixed_work_matches_original_including_optimizer_replay_rng(reference, seed, predictive):
    settings = dict(seed=seed, iterations=3, traversals=32, capacity=71, batch=16,
                    regret_steps=9, critic_steps=53, policy_steps=9,
                    layers=2, width=16, predictive=predictive)
    old = run_fixed_work(reference, **settings)
    new = run_fixed_work(variants, **settings)
    compare(old, new)
    assert new["replay_array_bytes"] < old["replay_array_bytes"] * 0.5


def test_cached_and_uncached_match_with_multithreaded_full_batches(reference):
    torch.set_num_threads(2)
    settings = dict(seed=12, iterations=3, traversals=32, capacity=71, batch=-1,
                    regret_steps=8, critic_steps=51, policy_steps=8, layers=2, width=16)
    old = run_fixed_work(reference, **settings)
    for cache in (False, True):
        compare(old, run_fixed_work(variants, cache=cache, **settings))


def test_fit_cache_rebuilt_and_target_critic_not_cached(monkeypatch):
    calls = []
    original = variants.build_frozen_cache
    def recording(*args, **kwargs):
        cache = original(*args, **kwargs)
        if cache is not None:
            calls.append(cache.copy())
        return cache
    monkeypatch.setattr(variants, "build_frozen_cache", recording)
    cached = run_fixed_work(variants, seed=21, iterations=3, traversals=32,
                            capacity=71, batch=16, regret_steps=9, critic_steps=101,
                            policy_steps=9, layers=2, width=16)
    # Three iterations x two players x regret and critic fitting: caches are
    # built afresh even when a circular write index has returned to its old value.
    assert len(calls) == 12
    uncached = run_fixed_work(variants, seed=21, iterations=3, traversals=32,
                              capacity=71, batch=16, regret_steps=9, critic_steps=101,
                              policy_steps=9, layers=2, width=16, cache=False)
    compare(cached, uncached)
    # Each fit executes target refreshes at steps 0, 50 and 100. Equal critic
    # weights, best-loss selection and optimiser states guard against freezing
    # the TD target across those refreshes.


def test_production_batch_frozen_predictions_are_exact():
    from vr_deep_cfr.solver import MLP
    torch.manual_seed(81)
    model = MLP(190, [64, 64, 64], 3)
    # A nonzero head is needed: zero initial predictions would be a vacuous test.
    with torch.no_grad():
        for p in model.parameters():
            p.uniform_(-0.1, 0.1)
    inputs = torch.randn(4171, 190)
    cache = build_frozen_cache(4171, 2048, 750, lambda i: model(inputs[i]), device="cpu")
    for _ in range(3):
        indices = torch.randperm(4171)[:2048].numpy()
        with torch.no_grad():
            assert torch.equal(torch.from_numpy(cache[indices]), model(inputs[indices]))


def test_cache_disagreement_falls_back_without_changing_training(monkeypatch):
    settings = dict(seed=51, iterations=2, traversals=32, capacity=71, batch=16,
                    regret_steps=9, critic_steps=53, policy_steps=9, layers=2, width=16)
    expected = run_fixed_work(variants, cache=False, **settings)
    original = variants.build_frozen_cache
    def corrupted(*args, **kwargs):
        cache = original(*args, **kwargs)
        return None if cache is None else cache + 0.1
    monkeypatch.setattr(variants, "build_frozen_cache", corrupted)
    compare(expected, run_fixed_work(variants, **settings))
