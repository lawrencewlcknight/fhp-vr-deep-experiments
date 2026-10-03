"""Experiment identity, artifact safety, allocation and optional real-Ray gate."""

import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest
import torch

from experiments.fhp.exp3_vr_deep_lossless_n2_standard16 import config as baseline
from experiments.fhp.exp5_vr_deep_ray8 import config, train
from experiments.fhp.exp5_vr_deep_ray8.aggregate import aggregate, verify_worker
from experiments.fhp.exp5_vr_deep_ray8.run import main
from fhp_vr_deep.io_utils import read_json, sha256_file, write_json
from vr_deep_cfr import parallel_solver as parallel
from test_vr_deep_parallel import InlinePool

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_only_collection_changes_against_exp3():
    assert config.TRAINING_CONFIG == baseline.TRAINING_CONFIG
    assert config.TRAINING_CONFIG is not baseline.TRAINING_CONFIG
    assert config.CONFIG_SHA256 == baseline.CONFIG_SHA256
    assert config.REFERENCE_VM == baseline.REFERENCE_VM
    assert config.SEEDS == (0, 1, 2) and config.THREADS == baseline.THREADS == 8
    assert config.schedule() == baseline.schedule()
    assert config.CHECKPOINT_HOURS == (6, 12, 18, 24)
    assert config.TRAIN_MAX_SECONDS == 36 * 3600
    spec = config.contract()
    assert spec["parallel_traversal_workers"] == 8 and spec["traversal_worker_threads"] == 1
    assert spec["performance_evaluation"] == "none_deferred_to_separate_experiment"
    assert spec["exact_exploitability"] is False
    assert spec["ray_object_store_bytes"] == 2 * 1024**3
    assert config.smoke_config()["num_traversals"] >= 8


@pytest.fixture
def completed_smoke(tmp_path, monkeypatch):
    def factory(seed, cfg, *, smoke=False):
        return parallel.make_solver(seed, cfg, smoke=smoke, pool_factory=InlinePool)

    monkeypatch.setattr(train, "make_solver", factory)
    assert main(["smoke", "--output-root", str(tmp_path), "--threads", "1"]) == 0
    return tmp_path


def test_checkpoints_runtime_inventory_and_aggregation(completed_smoke):
    root = completed_smoke
    worker = root / "workers" / config.task_name(0)
    manifest, summary, rows = verify_worker(worker, smoke=True)
    assert manifest["parallel_traversal_workers"] == 8
    assert summary["checkpoint_count"] == 4
    assert len(list(root.rglob("*.pt"))) == 4
    assert not (worker / "training_states").exists()
    assert summary["input_sizes"]["policy"] == 183 and summary["input_sizes"]["critic"] == 263
    assert rows[-1]["parallel_traversals"] == rows[-1]["episode"]
    assert rows[-1]["parallel_peak_payload_bytes"] > 0
    assert (root / "analysis/SUCCESS.json").exists()
    assert read_json(root / "analysis/summary.json")["evaluation_status"] == "deferred_to_shared_suite"


@pytest.mark.parametrize("key,value", [("parallel_traversal_workers", 4), ("learner_threads", 16),
                                       ("ray_object_store_bytes", 1), ("merge_order", "finish_order")])
def test_tampered_parallel_contract_rejected(completed_smoke, key, value):
    worker = completed_smoke / "workers" / config.task_name(0)
    path = worker / "run_manifest.json"
    manifest = read_json(path)
    manifest[key] = value
    write_json(path, manifest)
    success = read_json(worker / "SUCCESS.json")
    success["files"][path.name] = sha256_file(path)
    write_json(worker / "SUCCESS.json", success)
    with pytest.raises(ValueError, match=key):
        aggregate(completed_smoke, smoke=True)


def test_thread_override_rejected_before_starting_ray(tmp_path):
    with pytest.raises(ValueError, match="8 Torch CPU threads"):
        train.run_worker(tmp_path, 0, threads=16)


@pytest.mark.parametrize("stage", ["controller", "smoke", "train", "aggregate"])
def test_cloud_resources_gate_dependencies_and_existing_jobs_unchanged(stage, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "gcp"))
    import exp1_vr_deep_pdcfr_24h_batch as shared
    import exp5_vr_deep_ray8_batch as exp5
    args = SimpleNamespace(project="test-project", region="europe-west1", bucket="gs://test-bucket",
                           service_account="batch@example.iam.gserviceaccount.com", repo_ref="a" * 40,
                           run_id="vr5-ray8-test", experiment=exp5.EXPERIMENT)
    job = shared.build_job(args, stage)
    group = job["taskGroups"][0]
    script = group["taskSpec"]["runnables"][0]["script"]["text"]
    subprocess.run(["bash", "-n"], input=script, text=True, check=True)
    if stage in ("smoke", "train"):
        assert job["allocationPolicy"]["instances"][0]["policy"]["machineType"] == "n2-standard-16"
        assert group["taskSpec"]["computeResource"] == dict(cpuMilli=16000, memoryMib=62000)
    if stage != "controller":
        assert "-r requirements-ray.txt" in script
    if stage == "smoke":
        assert "export FHP_RUN_RAY_TESTS=1" in script
        assert "test_vr_deep_parallel.py" in script and "test_exp5_vr_deep_ray8.py" in script
    if stage == "train":
        assert group["taskCount"] == group["parallelism"] == 3
        assert group["taskSpec"]["maxRetryCount"] == 0
        assert group["taskSpec"]["maxRunDuration"] == "129600s"
        assert "OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8" in script
        assert "batch_diagnostics monitor" in script
    assert "evaluate_checkpoint" not in script and "fhp-evaluate" not in script
    args.experiment = shared.DEFAULT_EXPERIMENT
    old = shared.build_job(args, stage)
    old_text = old["taskGroups"][0]["taskSpec"]["runnables"][0]["script"]["text"]
    assert "requirements-ray.txt" not in old_text and "FHP_RUN_RAY_TESTS" not in old_text


@pytest.mark.skipif(os.environ.get("FHP_RUN_RAY_TESTS") != "1", reason="Explicit local permission/cloud gate for Ray processes")
def test_real_eight_actor_checkpoint_smoke(tmp_path):
    import ray
    assert not ray.is_initialized()
    assert main(["smoke", "--output-root", str(tmp_path), "--threads", "1"]) == 0
    worker = tmp_path / "workers" / config.task_name(0)
    verify_worker(worker, smoke=True)
    runtime = read_json(worker / "parallel_runtime.json")
    assert len({row["pid"] for row in runtime["workers"]}) == 8
    assert all(row["pid"] != os.getpid() for row in runtime["workers"])
    assert not ray.is_initialized()


@pytest.mark.skipif(os.environ.get("FHP_RUN_RAY_TESTS") != "1", reason="Explicit local permission/cloud gate for Ray processes")
def test_real_ray_production_minibatch_and_bounded_payload():
    import ray
    cfg = config.smoke_config()
    cfg.update(num_traversals=4096, advantage_buffer_size=16384, ave_policy_buffer_size=16384,
               baseline_buffer_size=16384, advantage_batch_size=2048, ave_policy_batch_size=2048,
               baseline_batch_size=2048, advantage_network_train_steps=16,
               baseline_network_train_steps=53, ave_policy_network_train_steps=16)
    torch.set_num_threads(8)
    solver = parallel.make_solver(0, cfg, smoke=True)
    try:
        solver.iteration()
        assert torch.get_num_threads() == 8  # Actor thread limits must not leak to learner.
        assert solver.episode == solver.parallel_totals["traversals"] == 8192
        assert solver.parallel_totals["collections"] == 2
        assert solver.parallel_totals["peak_payload_bytes"] < parallel.SMOKE_OBJECT_STORE_BYTES
        assert solver.phase_seconds["regret_fitting"] > 0 and solver.phase_seconds["critic_fitting"] > 0
        assert len(solver.q_value_trainer.buffer) >= 2048
        assert len(solver.ave_policy_trainer.buffer) >= 2048
        for trainer in (*solver.regret_trainers, solver.q_value_trainer):
            assert all(torch.isfinite(parameter).all() for parameter in trainer.model.parameters())
    finally:
        solver.close()
    assert not ray.is_initialized()
