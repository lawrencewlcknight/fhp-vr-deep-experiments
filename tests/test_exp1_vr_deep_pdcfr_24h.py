"""New Experiment 1 contract, scheduling, artifact and cloud safety tests."""

import copy
import importlib.util
from pathlib import Path
import random
import shutil
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.fhp.exp1_vr_deep_pdcfr_24h import config
from experiments.fhp.exp1_vr_deep_pdcfr_24h.aggregate import aggregate, safe_path, verify_worker
from experiments.fhp.exp1_vr_deep_pdcfr_24h.run import main
from experiments.fhp.exp1_vr_deep_pdcfr_24h.train import make_solver, publish, run_worker
from fhp_vr_deep.io_utils import read_json
from vr_deep_cfr.policy_snapshots import LoadedVRPolicy, load_policy_snapshot_payload

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("new_vr_batch", ROOT / "gcp/exp1_vr_deep_pdcfr_24h_batch.py")
batch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(batch)


@pytest.fixture(autouse=True)
def one_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


@pytest.fixture
def args():
    return SimpleNamespace(project="test-project", region="europe-west1", bucket="gs://test-bucket",
                           service_account="worker@test-project.iam.gserviceaccount.com",
                           repo_ref="a" * 40, run_id="vr1-24h-test")


@pytest.fixture(scope="module")
def smoke_output(tmp_path_factory):
    root = tmp_path_factory.mktemp("new-vr-exp1")
    before = torch.get_num_threads()
    try:
        assert main(["smoke", "--output-root", str(root), "--threads", "1"]) == 0
    finally:
        torch.set_num_threads(before)
    return root


def test_frozen_scientific_contract():
    actual = config.TRAINING_CONFIG
    original = config.LEDUC_TRANSFER_CONFIG
    assert {k for k in original if original[k] != actual[k]} == {"evaluation_frequency", "max_num_iterations"}
    assert set(actual) - set(original) == {"alpha", "gamma", "reinitialize_imm_regret_networks"}
    assert actual["alpha"] == 2.3 and actual["gamma"] == 2.0
    assert actual["reinitialize_imm_regret_networks"] is True
    assert actual["reinitialize_advantage_networks"] is False
    assert (actual["num_layers"], actual["num_hiddens"]) == (3, 64)
    assert actual["num_traversals"] == 10000
    assert actual["advantage_network_train_steps"] == 750
    assert actual["baseline_network_train_steps"] == 10000
    assert actual["ave_policy_network_train_steps"] == 5000
    for prefix in ("advantage", "ave_policy", "baseline"):
        assert actual[prefix + "_batch_size"] == 2048
        assert actual[prefix + "_buffer_size"] == 1000000
    assert config.SEEDS == (0, 1, 2)
    assert config.CHECKPOINT_SECONDS == (21600, 43200, 64800, 86400)
    assert config.contract()["exact_exploitability"] is False
    assert "no_full_training_states" in config.contract()["artifact_retention"]
    assert config.contract(True)["is_smoke"] is True
    changed = config.contract()
    changed["training_config"]["num_traversals"] = -1
    assert config.TRAINING_CONFIG["num_traversals"] == 10000


def test_time_threshold_does_not_interrupt_either_players_update(monkeypatch):
    solver = make_solver(0, config.smoke_config())
    solver.training_time_checkpoint_seconds = (1, 2, 3, 4)
    calls = []
    elapsed = [0.0]
    monkeypatch.setattr(solver, "_training_elapsed_seconds", lambda: elapsed[0])
    def collect(player):
        calls.append(("collect", player))
        elapsed[0] += 3
        # This hook is normally called on every sampled trajectory.
        solver._maybe_run_training_time_checkpoint()
        assert not solver._stop_requested
        assert not any(c[0] == "checkpoint" for c in calls)
    monkeypatch.setattr(solver, "collect_training_data", collect)
    monkeypatch.setattr(solver, "train_regret", lambda p: calls.append(("regret", p)))
    monkeypatch.setattr(solver, "train_baseline", lambda p: calls.append(("critic", p)))
    # A populated replay is required before a checkpoint may be fitted.
    solver.ave_policy_trainer.buffer.add(np.zeros(190), [1, 0, 0], [1, 1, 1], 1)
    monkeypatch.setattr(solver, "_run_checkpoint", lambda **kw: calls.append(("checkpoint", kw["checkpoint_target_seconds"])))
    solver.solve()
    assert calls == [("collect", 0), ("regret", 0), ("critic", 0),
                     ("collect", 1), ("regret", 1), ("critic", 1)] + [
                         ("checkpoint", x) for x in (1, 2, 3, 4)]
    assert solver.num_iteration == 1 and solver.stop_reason == "training_time_budget"


def test_checkpoint_overhead_and_randomness_do_not_enter_training(monkeypatch):
    import vr_deep_cfr.solver as base
    solver = make_solver(0, config.smoke_config())
    clock = [10.0]
    monkeypatch.setattr(base.time, "perf_counter", lambda: clock[0])
    solver._solve_start_time = 0.0
    solver._checkpoint_overhead_seconds = 0.0
    events = []
    def work():
        clock[0] += 100.0
        random.random()
        np.random.rand()
        torch.rand(4)
        assert solver._training_elapsed_seconds() == 10.0
    monkeypatch.setattr(solver, "train_average_policy", work)
    monkeypatch.setattr(solver, "evaluate", lambda **kw: solver.checkpoint_rows.append(kw))
    solver._post_checkpoint_callback = lambda *_: (events.append("saved"), work())
    rng = copy.deepcopy(solver._capture_rng_state())
    solver._run_checkpoint(checkpoint_target_seconds=10)
    from benchmarks.vr_deep_efficiency import assert_exact
    assert_exact(rng, solver._capture_rng_state())
    assert events == ["saved"]
    assert solver._checkpoint_overhead_seconds == 200.0
    assert solver._training_elapsed_seconds() == 10.0


def test_smoke_outputs_are_complete_playable_relocatable_and_small(smoke_output, tmp_path):
    relocated = tmp_path / "download"
    shutil.copytree(smoke_output, relocated)
    analysis = aggregate(relocated, smoke=True)
    worker = relocated / "workers" / config.task_name(0)
    manifest, summary, snapshots = verify_worker(worker, smoke=True)
    assert manifest["torch_threads"] == 1
    assert summary["stop_reason"] == "training_time_budget"
    assert len(snapshots) == 4
    assert {row["checkpoint_id"] for row in snapshots} == {"time_06h", "time_12h", "time_18h", "time_24h"}
    for row in snapshots:
        payload = load_policy_snapshot_payload(worker / row["path"])
        assert payload["algorithm_id"] == config.ALGORITHM_ID
        assert payload["input_size"] == 190 and payload["policy_network_layers"] == [64] * 3
        assert payload["checkpoint"]["completed_iteration"] == payload["iteration"]
        assert "policy_state_dict" in payload
        assert not {"optimizer", "replay", "training_state", "advantage_state_dict"} & payload.keys()
    assert len(list(worker.rglob("*.pt"))) == 4
    assert sum(p.stat().st_size for p in worker.rglob("*") if p.is_file()) < 2_000_000
    assert (analysis / "nodes_by_training_time.png").is_file()
    assert read_json(analysis / "summary.json")["policy_count"] == 4
    assert read_json(analysis / "checkpoint_aggregate.json")[0]["nodes_touched_se"] is None
    with pytest.raises(FileExistsError):
        main(["smoke", "--output-root", str(relocated), "--threads", "1"])


def test_snapshot_accepts_shared_suite_loader_when_installed(smoke_output):
    # Training/smoke must not require a second repository to be installed.
    from fhp_vr_deep.evaluation_adapter import _import_suite, load_policy_for_evaluation
    try:
        _import_suite()
    except ModuleNotFoundError:
        pytest.skip("Shared evaluation suite is optional for training")
    path = next((smoke_output / "workers").rglob("*.pt"))
    game, target = load_policy_for_evaluation(path)
    native = LoadedVRPolicy(game, path)
    state = game.new_initial_state()
    while not state.is_terminal():
        if state.is_chance_node():
            state.apply_action(state.chance_outcomes()[0][0])
            continue
        actual, expected = target.action_probabilities(state), native.action_probabilities(state)
        assert actual.keys() == expected.keys()
        np.testing.assert_allclose(list(actual.values()), list(expected.values()), rtol=1e-6, atol=1e-7)
        # Check/call reaches the flop and checks both public-card roles.
        state.apply_action(1 if 1 in actual else max(actual))
    assert game.num_players() == 2


def test_aggregate_rejects_smoke_as_production_and_missing_seeds(smoke_output):
    with pytest.raises(ValueError, match="Expected exactly"):
        aggregate(smoke_output)
    with pytest.raises(ValueError, match="wrong training_config"):
        verify_worker(smoke_output / "workers" / config.task_name(0))


def test_corrupt_artifact_rejected(smoke_output, tmp_path):
    dest = tmp_path / "worker"
    shutil.copytree(smoke_output / "workers" / config.task_name(0), dest)
    path = next(dest.rglob("*.pt"))
    path.write_bytes(path.read_bytes() + b"corrupted")
    with pytest.raises(ValueError, match="corrupt"):
        verify_worker(dest, smoke=True)


def test_three_seed_training_and_aggregation_path_with_test_only_small_budget(tmp_path, monkeypatch):
    # Exercise the actual three-worker code path, shortening only this test's
    # frozen contract. No command-line override can silently shorten production.
    monkeypatch.setattr(config, "TRAINING_CONFIG", config.smoke_config())
    monkeypatch.setattr(config, "CHECKPOINT_SECONDS", config.SMOKE_SECONDS)
    for index in range(3):
        run_worker(tmp_path, index)
    result = aggregate(tmp_path)
    summary = read_json(result / "summary.json")
    assert summary["seeds"] == [0, 1, 2] and summary["policy_count"] == 12
    assert summary["is_smoke"] is False
    assert len(list((tmp_path / "workers").rglob("*.pt"))) == 12
    for row in read_json(result / "checkpoint_aggregate.json"):
        assert row["training_seeds"] == 3
        assert row["nodes_touched_se"] is not None


@pytest.mark.parametrize("relative", ["../escape", "/absolute"])
def test_untrusted_manifest_cannot_escape_worker(tmp_path, relative):
    with pytest.raises(ValueError, match="Unsafe"):
        safe_path(tmp_path, relative)


def test_success_is_uploaded_last_and_never_on_incomplete_upload(monkeypatch, tmp_path):
    commands = []
    monkeypatch.setattr(subprocess, "run", lambda command, **_: commands.append(command))
    publish(tmp_path, "gs://bucket/run/worker", success=True)
    assert len(commands) == 2
    assert "rsync" in commands[0] and "--exclude" in commands[0]
    assert commands[1][-1] == "gs://bucket/run/worker/SUCCESS.json"
    def fail(command, **_):
        raise subprocess.CalledProcessError(1, command)
    monkeypatch.setattr(subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        publish(tmp_path, "gs://bucket/run/worker", success=True)


def test_cloud_resources_and_no_training_retries(args):
    job = batch.build_job(args, "train")
    group = job["taskGroups"][0]
    assert group["taskCount"] == group["parallelism"] == 3
    assert group["taskCountPerNode"] == 1
    assert group["taskSpec"]["maxRetryCount"] == 0
    assert group["taskSpec"]["maxRunDuration"] == "129600s"
    assert group["taskSpec"]["computeResource"] == dict(cpuMilli=8000, memoryMib=30000)
    policy = job["allocationPolicy"]["instances"][0]["policy"]
    assert policy["machineType"] == "n2-standard-8"
    assert policy["provisioningModel"] == "STANDARD"
    assert policy["bootDisk"] == dict(sizeGb=200, type="pd-balanced")
    assert "evaluation" not in batch.STAGES


@pytest.mark.parametrize("stage", ["controller", "smoke", "train", "aggregate"])
def test_generated_cloud_scripts_have_valid_shell_and_no_other_algorithm_dependency(args, stage, tmp_path):
    text = batch.script(args, stage)
    path = tmp_path / "job.sh"
    path.write_text(text)
    subprocess.run(["bash", "-n", str(path)], check=True)
    assert args.repo_ref in text
    assert "UCV_EXP1_RUN_ID" not in text and "sd-cfr" not in text
    assert "training_state" not in text
    if stage == "smoke":
        assert "benchmarks.vr_deep_efficiency" in text
        assert "--batch 2048" in text and "--seeds 0 1 2" in text
    if stage == "train":
        assert "--remote-uri" in text
        assert "SUCCESS[.]json" in text


def test_controller_failure_blocks_training(args, monkeypatch):
    calls = []
    monkeypatch.setattr(batch, "cloud", lambda *_a, **_k: None)
    monkeypatch.setattr(batch, "job_state", lambda *_: None)
    monkeypatch.setattr(batch, "submit", lambda _a, stage: calls.append(stage))
    def failed(*_):
        raise RuntimeError("smoke failed")
    monkeypatch.setattr(batch, "wait", failed)
    with pytest.raises(RuntimeError, match="smoke failed"):
        batch.orchestrate(args)
    assert calls == ["smoke"]


def test_controller_reuses_completed_jobs_but_never_restarts_failed_training(args, monkeypatch):
    submitted = []
    monkeypatch.setattr(batch, "cloud", lambda *_a, **_k: None)
    monkeypatch.setattr(batch, "job_state", lambda _a, name: "SUCCEEDED" if name.endswith("smoke") else "FAILED")
    monkeypatch.setattr(batch, "submit", lambda _a, stage: submitted.append(stage))
    monkeypatch.setattr(batch, "wait", lambda *_: None)
    with pytest.raises(RuntimeError, match="automatic training restart"):
        batch.orchestrate(args)
    assert submitted == []


def test_cloud_preflight_distinguishes_missing_outputs_from_permissions(args, monkeypatch):
    def fake(*_a, **kwargs):
        if kwargs.get("capture"):
            return subprocess.CompletedProcess([], 1, "", "ERROR: Permission denied")
    monkeypatch.setattr(batch, "cloud", fake)
    with pytest.raises(RuntimeError, match="Permission denied"):
        batch.preflight(args)
    monkeypatch.setattr(batch, "cloud", lambda *_a, **_k: subprocess.CompletedProcess([], 1, "", "One or more URLs matched no objects."))
    batch.preflight(args)
    monkeypatch.setattr(batch, "cloud", lambda *_a, **_k: subprocess.CompletedProcess([], 0, "gs://existing/output", ""))
    with pytest.raises(ValueError, match="already has stored outputs"):
        batch.preflight(args)


def test_validation_normalises_bucket_and_rejects_bad_identity(args):
    args.bucket = "test-bucket/"
    batch.validate(args)
    assert args.bucket == "gs://test-bucket"
    args.repo_ref = "main"
    with pytest.raises(ValueError, match="full pushed commit"):
        batch.validate(args)
