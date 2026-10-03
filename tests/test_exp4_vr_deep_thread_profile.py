"""Matched work, complete state, fail-closed reports and isolated cloud allocation."""

from copy import deepcopy
import json
from pathlib import Path
import random
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from benchmarks.vr_deep_efficiency import assert_exact
from experiments.fhp.exp2_vr_deep_lossless_24h import config as baseline
from experiments.fhp.exp4_vr_deep_thread_profile import config, run, workload as w
from gcp import exp4_vr_deep_thread_profile_batch as batch


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_unchanged_production_work_and_repeated_random_order():
    spec = config.protocol()
    assert spec["training_config"] == baseline.TRAINING_CONFIG
    assert spec["warmup_iterations"] == 5 and spec["repeats"] == 3
    assert spec["thread_counts"] == [1, 2, 4, 8, 16]
    assert not spec["performance_evaluation"]
    before = random.getstate()
    order = config.trial_order(3)
    assert random.getstate() == before
    assert order == config.trial_order(3)
    assert len(order) == len(set(order)) == 15
    assert len({tuple(n for r, n in order if r == repeat) for repeat in range(3)}) > 1
    spec["training_config"]["num_traversals"] = -1
    assert baseline.TRAINING_CONFIG["num_traversals"] == 10000


def test_state_round_trip_preserves_all_models_optimizers_replay_and_rng(tmp_path):
    spec = config.protocol(True)
    solver = w.new_solver(spec)
    # Exercise circular wraparound and reservoir replacement, not just fresh buffers.
    for _ in range(9):
        solver.iteration()
    assert len(solver.q_value_trainer.buffer) == 128
    path = tmp_path / "fixture.pt"
    original_models, original_rng = w.model_state(solver), w.rng_state()
    original_buffers = {name: w.buffer_state(trainer.buffer) for name, trainer in w.trainers(solver).items()}
    w.save_fixture(path, solver, w.trainers(solver))
    restored = w.load_fixture(path, spec)
    assert_exact(original_models, w.model_state(restored))
    assert_exact(original_rng, w.rng_state())
    assert_exact(original_buffers, {name: w.buffer_state(trainer.buffer) for name, trainer in w.trainers(restored).items()})
    # One unmodified update is exactly reproducible at the same thread count.
    solver.iteration()
    expected = w.endpoint(solver)
    restored = w.load_fixture(path, spec)
    restored.iteration()
    assert_exact(expected, w.endpoint(restored))


def test_trace_consumes_rng_and_restores_patch_even_on_failure():
    original = np.random.choice
    np.random.seed(18)
    trace = w.ChoiceTrace()
    with trace.observe():
        choices = [np.random.choice(range(3), p=[.2, .3, .5]) for _ in range(12)]
    reference_rng = np.random.get_state()
    np.random.seed(18)
    replay = w.ChoiceTrace(trace.events)
    with replay.observe():
        assert [np.random.choice(range(3), p=[.9, .05, .05]) for _ in range(12)] == choices
    assert replay.divergences > 0
    assert_exact(reference_rng, np.random.get_state())
    with pytest.raises(ValueError, match="population"):
        with w.ChoiceTrace(trace.events).observe():
            np.random.choice(range(4))
    assert np.random.choice is original
    with pytest.raises(ValueError, match="complete"):
        with w.ChoiceTrace(trace.events).observe():
            pass


def test_nested_sampler_audit_and_restoration():
    solver = w.new_solver(config.protocol(True))
    solver.iteration()
    outer, inner = w.BatchDigest(), w.BatchDigest()
    buffer = solver.regret_trainers[0].buffer
    with outer.observe(solver):
        with inner.observe(solver):
            buffer.sample_indices(2)
        assert outer.result() == inner.result()
        buffer.sample_indices(2)
    assert outer.calls == 2 and inner.calls == 1
    assert "sample_indices" not in vars(buffer)


@pytest.fixture
def fixture_run(tmp_path, monkeypatch):
    spec = config.protocol(True)
    scratch, output = tmp_path / "scratch", tmp_path / "output"
    scratch.mkdir()
    output.mkdir()
    from open_spiel.python.algorithms import exploitability

    def forbidden(*args, **kwargs):
        raise AssertionError("Profiling must not evaluate poker policies")

    monkeypatch.setattr(exploitability, "exploitability", forbidden)
    run.prepare(scratch, output, spec)
    return scratch, output, spec


def test_prefit_replay_and_whole_iteration_match_reference(fixture_run):
    scratch, output, spec = fixture_run
    result = run.trial(scratch, output, spec, 1, 0)
    assert set(result["measurements"]) == set(config.COMPONENTS)
    for row in result["measurements"].values():
        assert row["workload_verified"] and row["seconds"] > 0
        assert row["endpoint_max_abs_difference_from_8_threads"] == 0  # Test reference also uses one thread.
    assert result["measurements"]["whole"]["traversals"] == 8
    assert result["measurements"]["whole"]["iterations"] == 1
    assert result["measurements"]["whole"]["choice_replay_divergences"] == 0
    assert result["measurements"]["whole"]["inference_different_outputs"] == 0


def test_inference_replay_keeps_support_and_does_not_skip_computation():
    solver = w.new_solver(config.protocol(True))
    trainer = solver.regret_trainers[0]
    calls = []

    def compute(*args, **kwargs):
        calls.append(True)
        return np.array([0., 1., 0.])

    trainer.get_policy = compute
    trace = w.InferenceTrace([("policy_0", "<f8", [1., 0., 0.])])
    with trace.observe(solver):
        actual = trainer.get_policy(None)
        assert list(actual) == [1., 0., 0.]
        actual[0] = 123  # Must not corrupt the stored trace.
    assert calls == [True] and trace.different_outputs == 1
    assert trace.events[0][2] == [1., 0., 0.]
    assert trainer.get_policy is compute


def test_tampered_fixture_fails_closed(fixture_run):
    scratch, output, spec = fixture_run
    with (scratch / "regret_0.pt").open("ab") as stream:
        stream.write(b"invalid")
    with pytest.raises(ValueError, match="Fixture changed"):
        run.trial(scratch, output, spec, 1, 0)


def test_mismatched_minibatches_fail_closed(fixture_run):
    scratch, output, spec = fixture_run
    reference = run.read(scratch / "reference.json")
    reference["regret_0"]["sampler"]["sha256"] = "bad"
    (scratch / "reference.json").write_text(json.dumps(reference))
    with pytest.raises(ValueError, match="Minibatches differ"):
        run.trial(scratch, output, spec, 1, 0)


def test_summary_requires_all_trials_and_reports_paired_speedup(fixture_run):
    scratch, output, spec = fixture_run
    sample = run.trial(scratch, output, spec, 1, 0)
    with pytest.raises(FileNotFoundError):
        run.summarise(output, spec)
    for threads in config.THREAD_COUNTS:
        result = deepcopy(sample)
        result["threads"] = threads
        for row in result["measurements"].values():
            row["seconds"] = 1 / threads
        (output / "trials" / f"repeat_0_threads_{threads}.json").write_text(json.dumps(result))
    report = run.summarise(output, spec)
    assert report["fastest_observed_whole_iteration_threads"] is None  # Smoke is not a performance claim.
    one = next(row for row in report["timings"] if row["threads"] == 1 and row["component"] == "whole")
    assert one["paired_median_speedup_vs_8"] == .125
    del result["measurements"]["whole"]
    (output / "trials" / "repeat_0_threads_16.json").write_text(json.dumps(result))
    with pytest.raises(ValueError, match="Incomplete"):
        run.summarise(output, spec)


def test_failure_preserves_diagnostics_without_success(tmp_path, monkeypatch):
    def failed(*args, **kwargs):
        raise subprocess.TimeoutExpired("profile", 1)

    monkeypatch.setattr(subprocess, "run", failed)
    output = tmp_path / "failed"
    with pytest.raises(subprocess.TimeoutExpired):
        run.orchestrate(output, smoke=True)
    assert (output / "FAILURE.json").is_file()
    assert not (output / "SUCCESS.json").exists()
    with pytest.raises(FileExistsError):
        run.orchestrate(output, smoke=True)


@pytest.mark.parametrize("threads", config.THREAD_COUNTS)
def test_fresh_process_thread_environment(threads):
    env = run.thread_environment(threads)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        assert env[name] == str(threads)
    assert env["OMP_DYNAMIC"] == env["MKL_DYNAMIC"] == "FALSE"


@pytest.mark.parametrize("smoke", [True, False])
def test_batch_contract_and_shell_syntax(smoke):
    args = SimpleNamespace(project="example", bucket="gs://example-results", region="europe-west1",
                           service_account="batch@example.iam.gserviceaccount.com", repo_ref="a" * 40, run_id="vr4-test")
    job = batch.build_job(args, smoke)
    group = job["taskGroups"][0]
    assert group["taskCount"] == group["parallelism"] == group["taskCountPerNode"] == 1
    assert group["taskSpec"]["computeResource"] == dict(cpuMilli=16000, memoryMib=62000)
    assert group["taskSpec"]["maxRetryCount"] == 0
    assert group["taskSpec"]["maxRunDuration"] == ("7200s" if smoke else "21600s")
    assert job["allocationPolicy"]["instances"][0]["policy"]["machineType"] == "n2-standard-16"
    text = group["taskSpec"]["runnables"][0]["script"]["text"]
    assert ".run smoke " in text
    assert (".run run " in text) is not smoke
    assert "--remote-uri" in text and "batch_diagnostics monitor" in text
    assert "--exclude='(^|/)SUCCESS[.]json$|[.]tmp$'" in text
    subprocess.run(["bash", "-n"], input=text, text=True, check=True)
    subprocess.run(["bash", "-n", str(Path(__file__).parents[1] / "gcp/run_exp4_vr_deep_thread_profile.sh")], check=True)
