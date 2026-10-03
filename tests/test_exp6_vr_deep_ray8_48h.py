"""Duration-only control, final archives, strict continuation and real-Ray smoke."""

from copy import deepcopy
import os
import hashlib
import json
import re
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from benchmarks.vr_deep_efficiency import assert_exact
from experiments.fhp.exp5_vr_deep_ray8 import config as baseline
from experiments.fhp.exp6_vr_deep_ray8_48h import config, train, training_state as state
from experiments.fhp.exp6_vr_deep_ray8_48h.aggregate import aggregate, verify_worker
from experiments.fhp.exp6_vr_deep_ray8_48h.run import main
from fhp_vr_deep.io_utils import canonical_sha256, read_json, sha256_file, write_json
from test_vr_deep_parallel import InlinePool
from vr_deep_cfr.parallel_solver import make_solver as exp5_solver

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def one_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def test_frozen_learning_configuration():
    assert config.TRAINING_CONFIG == baseline.TRAINING_CONFIG
    assert config.CONFIG_SHA256 == baseline.CONFIG_SHA256
    assert config.REFERENCE_VM == baseline.REFERENCE_VM
    assert config.SEEDS == (0, 1, 2) and config.THREADS == 8
    assert config.CHECKPOINT_HOURS == (6, 12, 18, 24, 30, 36, 42, 48)
    assert [r["checkpoint_target_seconds"] for r in config.schedule()] == list(config.CHECKPOINT_SECONDS)
    assert config.TRAIN_MAX_SECONDS == 72 * 3600
    spec = config.contract()
    assert spec["parallel_traversal_workers"] == 8 and spec["traversal_worker_threads"] == 1
    assert spec["performance_evaluation"] == "none_deferred_to_separate_experiment"
    assert spec["exact_exploitability"] is False
    continuation = config.RunSpec(48, 72, "vr6-parent")
    assert [r["checkpoint_target_hours"] for r in continuation.schedule()] == [54, 60, 66, 72]
    assert continuation.contract()["training_config"] == baseline.TRAINING_CONFIG


@pytest.mark.parametrize("args", [(0, 24, None), (48, 48, "vr6-parent"), (48, 71, "vr6-parent"),
                                  (24, 48, "vr6-parent"), (48, 72, None), (48, 102, "vr6-parent")])
def test_invalid_segments(args):
    with pytest.raises(ValueError):
        config.RunSpec(*args)


def test_snapshot_requires_completed_iteration(tmp_path):
    cfg = config.smoke_config()
    solver = train.make_solver(0, cfg, smoke=True, pool_factory=InlinePool)
    try:
        with pytest.raises(ValueError, match="completed iteration"):
            state.save_state(tmp_path / "invalid", solver, training_config=cfg, seed=0,
                             target_hours=48, elapsed_seconds=1., is_smoke=True)
    finally:
        solver.close()


def scientific_state(solver):
    payload = state.learner_state(solver)
    # Timing/diagnostic counters necessarily differ across stop/start.
    for key in ("phase_seconds", "parallel_totals", "logger_pending", "logger_history", "checkpoint_rows"):
        payload.pop(key)
    for name, trainer in state.trainers(solver).items():
        buf = trainer.buffer
        size = min(len(buf), buf.buffer_size)
        payload[name + "_replay"] = {key: value[:size].copy() for key, value in vars(buf).items()
                                    if isinstance(value, np.ndarray)}
        payload[name + "_counters"] = {key: getattr(buf, key) for key in ("cur_id", "size") if hasattr(buf, key)}
    return payload


def test_exp5_and_exp6_have_identical_learning_updates():
    cfg = config.smoke_config()
    outputs = []
    for factory in (exp5_solver, train.make_solver):
        solver = factory(0, cfg, smoke=True, pool_factory=InlinePool)
        try:
            solver.iteration()
            solver.iteration()
            solver._run_checkpoint(checkpoint_kind="test")
            outputs.append(scientific_state(solver))
        finally:
            solver.close()
    assert_exact(*outputs)


def restart_equivalence(tmp_path, pool_factory, *, production_batches=False):
    cfg = config.smoke_config()
    cfg.update(num_traversals=32, advantage_buffer_size=71, ave_policy_buffer_size=71, baseline_buffer_size=71,
               advantage_batch_size=16, baseline_batch_size=16, ave_policy_batch_size=16,
               advantage_network_train_steps=9, baseline_network_train_steps=101, ave_policy_network_train_steps=9)
    if production_batches:
        cfg.update(num_traversals=4096, advantage_buffer_size=16384, ave_policy_buffer_size=16384,
                   baseline_buffer_size=16384, advantage_batch_size=2048, baseline_batch_size=2048,
                   ave_policy_batch_size=2048, baseline_network_train_steps=53)
    source = train.make_solver(0, cfg, smoke=True, pool_factory=pool_factory)
    source.logger._verbose = False
    try:
        source.iteration()
        # Exercise policy-checkpoint RNG isolation, not just raw training states.
        source._run_checkpoint(checkpoint_kind="training_time_48h", checkpoint_target_seconds=48 * 3600)
        before = scientific_state(source)
        archive = tmp_path / "training_state"
        state.save_state(archive, source, training_config=cfg, seed=0, target_hours=48,
                         elapsed_seconds=48 * 3600 + .125, is_smoke=True)
        assert_exact(before, scientific_state(source))  # Serialization consumes no learning RNG.
        source.iteration()
        expected = scientific_state(source)
    finally:
        source.close()
    resumed = train.make_solver(0, cfg, smoke=True, pool_factory=pool_factory)
    resumed.logger._verbose = False
    try:
        state.restore_state(archive, resumed, training_config=cfg, seed=0, start_hours=48, is_smoke=True)
        assert_exact(before, scientific_state(resumed))
        assert resumed._training_elapsed_seconds() == 48 * 3600 + .125
        assert resumed.max_wall_clock_seconds == 72 * 3600
        resumed.iteration()
        assert_exact(expected, scientific_state(resumed))
    finally:
        resumed.close()
    return archive, cfg


def test_restart_equivalence_with_ring_wrap_reservoir_overflow_and_target_refreshes(tmp_path):
    archive, cfg = restart_equivalence(tmp_path, InlinePool)
    meta = state.verify_state(archive)
    assert meta["buffers"]["critic"]["populated_rows"] == 71
    assert meta["buffers"]["policy"]["counters"]["cur_id"] > 71
    assert meta["buffers"]["critic"]["arrays"]["next_state_buf"]["shape"][1] == 183


@pytest.fixture
def completed_smoke(tmp_path, monkeypatch):
    original = train.make_solver
    monkeypatch.setattr(train, "make_solver", lambda seed, cfg, smoke=False, pool_factory=InlinePool:
                        original(seed, cfg, smoke=smoke, pool_factory=pool_factory))
    main(["smoke", "--output-root", str(tmp_path), "--threads", "1"])
    return tmp_path


def test_eight_policies_and_one_relocatable_complete_state(completed_smoke, tmp_path):
    worker = completed_smoke / "workers" / config.task_name(0)
    manifest, summary, rows = verify_worker(worker, smoke=True)
    assert summary["checkpoint_count"] == 8 and len(rows) == 8
    assert len(list(worker.glob("checkpoints/*.pt"))) == 8
    assert len(list(worker.glob("training_state/learner.pt"))) == 1
    assert summary["final_training_state_verified"]
    assert summary["final_training_state_bytes"] > 0
    assert read_json(completed_smoke / "analysis/summary.json")["policy_count"] == 8
    solver = train.make_solver(0, config.smoke_config(), smoke=True, pool_factory=InlinePool)
    try:
        state.restore_state(worker / "training_state", solver, training_config=config.smoke_config(),
                            seed=0, start_hours=48, is_smoke=True)
        assert solver.num_iteration == summary["final_outer_iteration"]
        solver.iteration()
        assert solver.num_iteration == summary["final_outer_iteration"] + 1
    finally:
        solver.close()


def test_missing_state_prevents_aggregation(completed_smoke):
    worker = completed_smoke / "workers" / config.task_name(0)
    path = worker / "training_state/learner.pt"
    path.write_bytes(path.read_bytes() + b"corruption")
    with pytest.raises(ValueError, match="Missing/corrupt"):
        aggregate(completed_smoke, smoke=True)


@pytest.mark.parametrize("key,value", [("runtime", {}), ("repository_commit", "bad"), ("threads", 16)])
def test_incompatible_restart_rejected(completed_smoke, key, value):
    directory = completed_smoke / "workers" / config.task_name(0) / "training_state"
    meta = read_json(directory / "manifest.json")
    meta[key] = value
    write_json(directory / "manifest.json", meta)
    solver = train.make_solver(0, config.smoke_config(), smoke=True, pool_factory=InlinePool)
    try:
        with pytest.raises(ValueError, match=key):
            state.restore_state(directory, solver, training_config=config.smoke_config(),
                                seed=0, start_hours=48, is_smoke=True)
    finally:
        solver.close()


def test_final_state_failure_never_writes_success(tmp_path, monkeypatch):
    original = train.make_solver
    monkeypatch.setattr(train, "make_solver", lambda seed, cfg, smoke=False:
                        original(seed, cfg, smoke=smoke, pool_factory=InlinePool))
    def fail(*args, **kwargs):
        raise OSError("simulated final archive failure")
    monkeypatch.setattr(train, "save_state", fail)
    with pytest.raises(OSError, match="archive failure"):
        train.run_worker(tmp_path, 0, smoke=True, threads=1)
    worker = tmp_path / "workers" / config.task_name(0)
    assert not (worker / "SUCCESS.json").exists()
    assert (worker / "failure.json").exists()
    assert len(list(worker.glob("checkpoints/*.pt"))) == 8


def test_publisher_excludes_partial_training_state_directories(tmp_path, monkeypatch):
    from experiments.fhp.exp1_vr_deep_pdcfr_24h import train as runner
    calls = []
    monkeypatch.setattr(runner.subprocess, "run", lambda command, **kwargs: calls.append(command))
    runner.publish(tmp_path, "gs://test-bucket/worker")
    pattern = calls[0][calls[0].index("--exclude") + 1]
    assert re.search(pattern, "training_state.tmp/critic__history_buf.npy")
    assert re.search(pattern, "checkpoints/policy.pt.tmp")
    assert re.search(pattern, "SUCCESS.json")
    assert not re.search(pattern, "training_state/learner.pt")


def test_production_resume_segment_keeps_iteration_and_checkpoint_progress(tmp_path, monkeypatch):
    # Exercise the actual worker/CLI/provenance path with tiny training work and
    # a simulated six-active-hours-per-iteration clock (not a 72-hour test).
    cfg = config.smoke_config()
    original_contract, original_factory = config.RunSpec.contract, train.make_solver
    def tiny_contract(self, smoke=False):
        result = original_contract(self, smoke)
        result.update(training_config=deepcopy(cfg), training_config_sha256=canonical_sha256(cfg))
        return result
    monkeypatch.setattr(config.RunSpec, "contract", tiny_contract)
    monkeypatch.setattr(train, "make_solver", lambda seed, cfg, smoke=False:
                        original_factory(seed, cfg, smoke=smoke, pool_factory=InlinePool))
    monkeypatch.setattr(train.ResumableParallelSolver, "_training_elapsed_seconds",
                        lambda self: float(self.num_iteration * 6 * 3600))
    root = tmp_path / "source"
    worker = train.run_worker(root, 0, threads=8)
    _, old, old_rows = verify_worker(worker)
    assert old["final_outer_iteration"] == 8 and len(old_rows) == 8
    assert old["final_training_elapsed_seconds"] == 48 * 3600
    archive = worker / "training_state"
    old_digest = sha256_file(archive / "manifest.json")
    output = tmp_path / "continued"
    assert main(["resume", "--output-root", str(output), "--resume-from", str(archive),
                 "--resume-run-id", "vr6-source", "--start-hours", "48", "--target-hours", "72"]) == 0
    spec = config.RunSpec(48, 72, "vr6-source")
    _, summary, rows = verify_worker(output / "workers" / config.task_name(0), experiment=spec)
    assert [row["checkpoint_target_hours"] for row in rows] == [54, 60, 66, 72]
    assert [row["iteration"] for row in rows] == [9, 10, 11, 12]
    assert summary["final_training_elapsed_seconds"] == 72 * 3600
    assert summary["final_nodes_touched"] > old["final_nodes_touched"]
    assert summary["continuation_source"]["manifest_sha256"] == old_digest
    assert sha256_file(archive / "manifest.json") == old_digest


def test_state_archive_contains_only_populated_replay(completed_smoke):
    archive = completed_smoke / "workers" / config.task_name(0) / "training_state"
    meta = state.verify_state(archive)
    assert any(buf["populated_rows"] < buf["capacity"] for buf in meta["buffers"].values())
    for buf in meta["buffers"].values():
        for value in buf["arrays"].values():
            assert np.load(archive / value["file"], mmap_mode="r").shape[0] == buf["populated_rows"]


@pytest.mark.parametrize("resume", [False, True])
def test_cloud_resources_continuation_and_shell_syntax(monkeypatch, resume):
    monkeypatch.syspath_prepend(str(ROOT / "gcp"))
    import exp1_vr_deep_pdcfr_24h_batch as shared
    import exp6_vr_deep_ray8_48h_batch as cloud
    env = dict(EXP6_RESUME_RUN_ID="vr6-parent", EXP6_START_HOURS="48", EXP6_TARGET_HOURS="72") if resume else {}
    args = SimpleNamespace(project="test-project", region="europe-west1", bucket="gs://test-bucket",
                           service_account="batch@example.iam.gserviceaccount.com", repo_ref="a" * 40,
                           run_id="vr6-child", experiment=cloud.experiment(env))
    for stage in ("controller", "smoke", "train", "aggregate"):
        job = shared.build_job(args, stage)
        group = job["taskGroups"][0]
        script = group["taskSpec"]["runnables"][0]["script"]["text"]
        subprocess.run(["bash", "-n"], input=script, text=True, check=True)
        if stage in ("train", "smoke"):
            assert job["allocationPolicy"]["instances"][0]["policy"]["machineType"] == "n2-standard-16"
        if stage == "train":
            assert group["taskCount"] == group["parallelism"] == 3
            assert group["taskSpec"]["maxRunDuration"] == "259200s"
            assert group["taskSpec"]["maxRetryCount"] == 0
            assert "batch_diagnostics monitor" in script
            assert ("--resume-from" in script) is resume
            assert ('uv python install "$FHP_STATE_PYTHON"' in script) is resume
            assert "--target-hours 72" in script if resume else "--target-hours 48" in script
        if stage == "controller":
            assert group["taskSpec"]["maxRunDuration"] == "345600s"
            assert "export EXP6_START_HOURS=" in script
        if stage == "smoke":
            assert "export FHP_RUN_RAY_TESTS=1" in script
            assert "test_exp6_vr_deep_ray8_48h.py" in script
        assert "evaluate_checkpoint" not in script


def test_cloud_resume_preflight_checks_all_seed_manifests(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "gcp"))
    import exp6_vr_deep_ray8_48h_batch as cloud
    args = SimpleNamespace(bucket="gs://test-bucket", run_id="vr6-child", repo_ref="a" * 40)
    rows = {}
    for seed in range(3):
        root = f"gs://test-bucket/vr6-source/workers/task_{seed:03d}_{config.ALGORITHM_ID}_seed_{seed}"
        meta = dict(schema=state.SCHEMA, algorithm_id=config.ALGORITHM_ID, seed=seed, is_smoke=False,
                    completed_target_hours=48, repository_commit=args.repo_ref, active_seconds=172801.,
                    runtime={"python": "3.11.9"}, training_config_sha256=config.CONFIG_SHA256,
                    threads=8, game={}, feature_encoder={})
        raw = json.dumps(meta) + "\n"
        rows[root + "/training_state/manifest.json"] = raw
        rows[root + "/SUCCESS.json"] = json.dumps(dict(files={"training_state/manifest.json": hashlib.sha256(raw.encode()).hexdigest()}))
    calls = []
    def fake_cloud(args, *command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout=rows[command[-1]])
    monkeypatch.setattr(cloud, "cloud", fake_cloud)
    spec = cloud.experiment({"EXP6_RESUME_RUN_ID": "vr6-source"})
    spec["preflight_hook"](args)
    assert len(calls) == 6
    args.repo_ref = "b" * 40
    with pytest.raises(ValueError, match="Incompatible"):
        spec["preflight_hook"](args)
    args.run_id = "vr6-source"
    with pytest.raises(ValueError, match="overwrite"):
        spec["preflight_hook"](args)


@pytest.mark.skipif(os.environ.get("FHP_RUN_RAY_TESTS") != "1", reason="Ray process/socket permission")
def test_real_ray_eight_checkpoint_and_restart_smoke(tmp_path):
    import ray
    assert not ray.is_initialized()
    main(["smoke", "--output-root", str(tmp_path / "run"), "--threads", "1"])
    worker = tmp_path / "run/workers" / config.task_name(0)
    verify_worker(worker, smoke=True)
    assert len({r["pid"] for r in read_json(worker / "parallel_runtime.json")["workers"]}) == 8
    assert not ray.is_initialized()
    restart_equivalence(tmp_path / "restart", train.RayTraversalPool)
    assert not ray.is_initialized()


@pytest.mark.skipif(os.environ.get("FHP_RUN_RAY_TESTS") != "1", reason="Ray process/socket permission")
def test_real_ray_restart_with_production_minibatches(tmp_path):
    import ray
    torch.set_num_threads(8)
    restart_equivalence(tmp_path, train.RayTraversalPool, production_batches=True)
    assert not ray.is_initialized()
