"""VM-only scientific contract, artifacts, resource allocation and launch safety."""

from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from experiments.fhp.exp2_vr_deep_lossless_24h import config as baseline
from experiments.fhp.exp2_vr_deep_lossless_24h.train import make_solver as baseline_solver
from experiments.fhp.exp3_vr_deep_lossless_n2_standard16 import config
from experiments.fhp.exp3_vr_deep_lossless_n2_standard16.aggregate import aggregate, verify_worker
from experiments.fhp.exp3_vr_deep_lossless_n2_standard16.run import main
from experiments.fhp.exp3_vr_deep_lossless_n2_standard16.train import make_solver, run_worker
from fhp_vr_deep.io_utils import read_json, sha256_file, write_json
from vr_deep_cfr.policy_snapshots import load_policy_snapshot_payload

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_only_vm_and_experiment_identity_change():
    assert config.TRAINING_CONFIG == baseline.TRAINING_CONFIG
    assert config.TRAINING_CONFIG is not baseline.TRAINING_CONFIG
    assert config.CONFIG_SHA256 == baseline.CONFIG_SHA256
    assert config.smoke_config() == baseline.smoke_config()
    assert config.SEEDS == baseline.SEEDS == (0, 1, 2)
    assert config.THREADS == baseline.THREADS == 8
    assert config.schedule() == baseline.schedule()
    assert config.CHECKPOINT_HOURS == (6, 12, 18, 24)
    assert config.TRAIN_MAX_SECONDS == 36 * 3600
    assert make_solver is baseline_solver  # No substituted learner or traversal.
    before, after = baseline.contract(), config.contract()
    allowed_changes = {"experiment_id", "experiment_name", "algorithm_id", "algorithm_label",
                       "baseline_experiment", "baseline_commit", "reference_vm"}
    assert {key for key in before if before[key] != after[key]} == allowed_changes
    assert after["reference_vm"] == dict(machine_type="n2-standard-16", cpu_milli=16000,
                                        memory_mib=62000, boot_disk_gib=200,
                                        boot_disk_type="pd-balanced")
    assert after["baseline_reference_vm"] == baseline.REFERENCE_VM
    assert after["parallel_traversal_workers"] == 0
    assert after["performance_evaluation"] == "none_deferred_to_separate_experiment"
    assert after["exact_exploitability"] is False
    assert "no_full_training_states" in after["artifact_retention"]
    after["training_config"]["num_traversals"] = -1
    after["reference_vm"]["cpu_milli"] = -1
    assert config.TRAINING_CONFIG == baseline.TRAINING_CONFIG
    assert config.REFERENCE_VM["cpu_milli"] == 16000


@pytest.fixture(scope="module")
def smoke_output(tmp_path_factory):
    root = tmp_path_factory.mktemp("vr-exp3-smoke")
    previous = torch.get_num_threads()
    try:
        with pytest.MonkeyPatch.context() as patch:
            def forbidden(*args, **kwargs):
                raise AssertionError("Performance evaluation must be separate")
            from fhp_vr_deep import evaluation_adapter
            from open_spiel.python.algorithms import exploitability
            patch.setattr(evaluation_adapter, "evaluate_checkpoint", forbidden)
            patch.setattr(exploitability, "exploitability", forbidden)
            assert main(["smoke", "--output-root", str(root), "--threads", "1"]) == 0
    finally:
        torch.set_num_threads(previous)
    return root


def test_policy_only_checkpoints_and_relocatable_aggregation(smoke_output, tmp_path):
    relocated = tmp_path / "download"
    shutil.copytree(smoke_output, relocated)
    worker = relocated / "workers" / config.task_name(0)
    manifest, summary, rows = verify_worker(worker, smoke=True)
    assert manifest["reference_vm"] == config.REFERENCE_VM
    assert summary["checkpoint_count"] == len(rows) == 4
    assert summary["input_sizes"] == {"policy": 183, "critic": 263,
                                      "regret_0": 183, "regret_1": 183}
    assert len(list(worker.rglob("*.pt"))) == 4
    assert not (worker / "training_states").exists()
    for row in rows:
        payload = load_policy_snapshot_payload(worker / row["path"])
        assert payload["version"] == 3
        assert payload["algorithm_id"] == config.ALGORITHM_ID
        assert payload["policy_network_layers"] == [64, 64, 64]
        assert not {"optimizer", "replay", "training_state"} & payload.keys()
    result = aggregate(relocated, smoke=True)
    assert read_json(result / "summary.json")["evaluation_status"] == "deferred_to_shared_suite"
    assert (result / "nodes_by_training_time.png").is_file()
    from experiments.fhp.exp2_vr_deep_lossless_24h.aggregate import verify_worker as verify_exp2
    with pytest.raises(ValueError, match="experiment_name"):
        verify_exp2(worker, smoke=True)
    with pytest.raises(FileExistsError):
        run_worker(relocated, 0, smoke=True)


def test_three_seed_production_path_shortened_only_inside_test(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "TRAINING_CONFIG", config.smoke_config())
    monkeypatch.setattr(config, "CHECKPOINT_SECONDS", config.SMOKE_SECONDS)
    for index in range(3):
        run_worker(tmp_path, index)
    summary = read_json(aggregate(tmp_path) / "summary.json")
    assert summary["seeds"] == [0, 1, 2]
    assert summary["policy_count"] == 12 and not summary["is_smoke"]
    assert len(list((tmp_path / "workers").rglob("*.pt"))) == 12


@pytest.mark.parametrize("key,value", [
    ("reference_vm", baseline.REFERENCE_VM), ("learner_threads", 16),
    ("parallel_traversal_workers", 8), ("performance_evaluation", "enabled"),
])
def test_aggregation_rejects_changed_hardware_contract(smoke_output, tmp_path, key, value):
    worker = tmp_path / "worker"
    shutil.copytree(smoke_output / "workers" / config.task_name(0), worker)
    manifest = read_json(worker / "run_manifest.json")
    manifest[key] = value
    write_json(worker / "run_manifest.json", manifest)
    success = read_json(worker / "SUCCESS.json")
    success["files"]["run_manifest.json"] = sha256_file(worker / "run_manifest.json")
    write_json(worker / "SUCCESS.json", success)
    with pytest.raises(ValueError, match=key):
        verify_worker(worker, smoke=True)


@pytest.mark.parametrize("index", [-1, 3])
def test_bad_seed_rejected(index):
    with pytest.raises(ValueError, match="Task index"):
        config.task_name(index)


def test_production_thread_override_rejected(tmp_path):
    with pytest.raises(ValueError, match="8 Torch CPU threads"):
        run_worker(tmp_path, 0, threads=16)


@pytest.fixture
def cloud_modules(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "gcp"))
    import exp1_vr_deep_pdcfr_24h_batch as batch
    import exp3_vr_deep_lossless_n2_standard16_batch as exp3
    return batch, exp3


@pytest.mark.parametrize("stage", ["controller", "smoke", "train", "aggregate"])
def test_batch_resources_shell_and_no_evaluation(cloud_modules, stage):
    batch, exp3 = cloud_modules
    args = SimpleNamespace(project="test-project", region="europe-west1", bucket="gs://test-bucket",
                           service_account="worker@test.iam.gserviceaccount.com", repo_ref="a" * 40,
                           run_id="vr3-vm16-test", experiment=exp3.EXPERIMENT)
    job = batch.build_job(args, stage)
    group = job["taskGroups"][0]
    task = group["taskSpec"]
    text = task["runnables"][0]["script"]["text"]
    subprocess.run(["bash", "-n"], input=text, text=True, check=True)
    assert "OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8" in text
    assert "exp3_vr_deep_lossless_n2_standard16" in text
    assert "evaluate_checkpoint" not in text and "fhp-evaluate" not in text
    policy = job["allocationPolicy"]["instances"][0]["policy"]
    if stage in ("smoke", "train"):
        assert policy["machineType"] == "n2-standard-16"
        assert task["computeResource"] == {"cpuMilli": 16000, "memoryMib": 62000}
        assert policy["bootDisk"] == {"sizeGb": 200, "type": "pd-balanced"}
    else:
        assert policy["machineType"] == ("e2-small" if stage == "controller" else "n2-standard-8")
    if stage == "train":
        assert group["taskCount"] == group["parallelism"] == 3
        assert group["taskCountPerNode"] == 1
        assert task["maxRunDuration"] == "129600s" and task["maxRetryCount"] == 0
        assert "--requested-memory-mib 62000" in text
        assert "fhp_vr_deep.batch_diagnostics monitor" in text
        assert config.ALGORITHM_ID in text
    if stage == "smoke":
        assert "test_exp3_vr_deep_lossless_n2_standard16.py" in text
        assert "benchmarks.vr_deep_efficiency --threads 8" in text
    assert job["labels"]["experiment"] == "fhp-vr-exp3-24h"
    # Earlier jobs remain at their original allocations.
    args.experiment = batch.DEFAULT_EXPERIMENT
    previous = batch.build_job(args, stage)
    if stage != "controller":
        assert previous["allocationPolicy"]["instances"][0]["policy"]["machineType"] == "n2-standard-8"
        assert previous["taskGroups"][0]["taskSpec"]["computeResource"]["memoryMib"] == 30000


def test_smoke_only_does_not_submit_training(cloud_modules, monkeypatch):
    batch, exp3 = cloud_modules
    calls = []
    monkeypatch.setattr(sys, "argv", ["batch", "smoke-only", "--project", "test-project",
        "--region", "europe-west1", "--bucket", "test-bucket", "--service-account",
        "worker@test.iam.gserviceaccount.com", "--repo-ref", "a" * 40, "--run-id", "vr3-smoke-test"])
    monkeypatch.setattr(batch, "preflight", lambda args: calls.append("preflight"))
    monkeypatch.setattr(batch, "submit", lambda args, stage: calls.append(stage) or "smoke-job")
    batch.main(experiment=exp3.EXPERIMENT)
    assert calls == ["preflight", "smoke"]


def test_launcher_rejects_old_commit_before_cloud_submission(tmp_path):
    import os
    env = dict(os.environ, PROJECT_ID="test-project", REGION="europe-west1",
               BUCKET="gs://test-bucket", SA_EMAIL="worker@test.iam.gserviceaccount.com",
               REPO_REF=config.BASELINE_COMMIT, RUN_ID="vr3-vm16-old-ref")
    for action in ("run", "smoke-only", "aggregate-only"):
        result = subprocess.run(["bash", str(ROOT / "gcp/run_exp3_vr_deep_lossless_n2_standard16.sh"), action],
                                env=env, text=True, capture_output=True)
        assert result.returncode == 2 and "does not contain" in result.stderr
