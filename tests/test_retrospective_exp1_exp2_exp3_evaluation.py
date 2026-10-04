import copy
import importlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.fhp.retrospective_exp1_exp2_exp3_evaluation import run as r
from fhp_evaluation.cohort import execute_tasks, sha256, write_json
from vr_deep_cfr.policy_snapshots import save_policy_snapshot
from vr_deep_cfr.solver import MLP

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("eval123_batch", ROOT / "gcp/retrospective_exp1_exp2_exp3_evaluation_batch.py")
batch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(batch)


@pytest.fixture
def sources(tmp_path, monkeypatch):
    frozen = copy.deepcopy(r.SOURCES)
    for experiment, source in frozen.items():
        config = importlib.import_module(f"experiments.fhp.{r.MODULES[experiment]}.config")
        rows = []
        for seed in range(3):
            worker = tmp_path / source["run_id"] / "workers" / config.task_name(seed)
            manifest = dict(config.contract(), seed=seed, torch_threads=8, repository_commit="a"*40)
            write_json(worker / "run_manifest.json", manifest)
            write_json(worker / "summary.json", dict(stop_reason="training_time_budget"))
            encoder = None
            if experiment != "exp1":
                from fhp_vr_deep.features import encoder_from_metadata
                encoder = encoder_from_metadata(config.contract()["feature_encoder"])
            model = MLP(183 if encoder else 190, [64,64,64], 3)
            solver = SimpleNamespace(ave_policy_trainer=SimpleNamespace(model=model), feature_encoder=encoder,
                                     nodes_touched=100, num_iteration=2, episode=200,
                                     network_layers=[64,64,64], infostate_size=183 if encoder else 190, action_size=3)
            worker_rows = []
            for hour in r.HOURS:
                checkpoint = dict(checkpoint_id=f"time_{hour:02d}h", training_elapsed_seconds=hour*3600)
                path = worker / "checkpoints" / f"{hour}.pt"
                save_policy_snapshot(solver, path, algorithm_id=config.ALGORITHM_ID, algorithm_label="fixture",
                                     seed=seed, config=config.TRAINING_CONFIG, checkpoint_row=checkpoint)
                row = dict(checkpoint, seed=seed, checkpoint_target_hours=hour, algorithm_id=config.ALGORITHM_ID,
                           nodes_touched=100, iteration=2, sha256=sha256(path), size_bytes=path.stat().st_size,
                           path=str(path.relative_to(tmp_path / source["run_id"])))
                rows.append(row)
                worker_rows.append(row)
            write_json(worker / "checkpoint_manifest.json", worker_rows)
            files = {str(p.relative_to(worker)): sha256(p) for p in worker.rglob("*") if p.is_file()}
            write_json(worker / "SUCCESS.json", dict(files=files))
        analysis = tmp_path / source["run_id"] / "analysis"
        write_json(analysis / "snapshot_inventory.json", rows)
        write_json(analysis / "summary.json", dict(is_smoke=False, policy_count=12, seeds=[0,1,2]))
        source["analysis_hashes"] = {name: sha256(analysis / name) for name in source["analysis_hashes"]}
        write_json(analysis / "SUCCESS.json", dict(experiment_name=source["experiment_name"], seeds=[0,1,2],
                                                  files=source["analysis_hashes"]))
    monkeypatch.setattr(r, "SOURCES", frozen)
    return tmp_path


def test_all_sources_validate_and_corruption_fails(sources):
    records = r.discover(sources)
    assert len(records) == 36
    path = Path(records[0]["path"])
    path.write_bytes(path.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="Policy missing"):
        r.discover(sources)


def test_frozen_local_inventories_match_hashes():
    cloud = ROOT / "cloud_outputs"
    if not all((cloud / s["run_id"] / "analysis/SUCCESS.json").exists() for s in r.SOURCES.values()):
        pytest.skip("Optional downloaded real-run metadata")
    for experiment in r.SOURCES:
        _, rows = r.inventory(cloud, experiment)
        assert len(rows) == 12


def test_full_and_pilot_protocol(sources):
    records = r.discover(sources)
    main, lbr = r.build_tasks(records, "main"), r.build_tasks(records, "lbr")
    assert len(main) == 270 and sum(t["deals"] for t in main) == 6300000
    assert {kind: sum(t["kind"] == kind for t in main) for kind in ("rule", "direct", "temporal")} == dict(rule=180,direct=36,temporal=54)
    assert len(lbr) == 900 and sum(t["deals"] for t in lbr) == 9000
    assert all(t["a"]["hours"] == 24 and t["rollouts"] == 4096 for t in lbr)
    probes = r.probe_tasks(lbr, "lbr")
    assert len(probes) == 27 and sum(t["deals"] for t in probes) == 270
    assert len({t["a"]["id"] for t in probes}) == 9
    assert all(t["a"]["seed"] == t["b"]["seed"] for t in main if "b" in t)
    assert {t["evaluation_seed"] for t in probes}.isdisjoint(t["evaluation_seed"] for t in lbr)


def test_real_spawn_raw_encoded_rule_direct_temporal_lbr(sources, tmp_path):
    records = r.discover(sources)
    main = r.build_tasks(records, "main")
    selected = [next(t for t in main if t["kind"] == k) for k in ("rule", "direct", "temporal")]
    selected += [next(t for t in r.build_tasks(records, "lbr") if t["a"]["experiment"] == e) for e in r.SOURCES]
    tasks = [dict(t, deals=2, rollouts=16) for t in selected]
    results = execute_tasks(tasks, tmp_path / "scoring", {}, workers=2, seconds=120)
    assert len(results) == 6
    assert all(np.isfinite(v["summary"]["mean"]) for v in results)


def test_gate_defers_without_full_scoring(sources, monkeypatch, tmp_path):
    records = r.discover(sources)
    calls = []
    def profile(tasks, *a, **kw):
        calls.append(tasks)
        return [dict(task=t, elapsed_seconds=10000) for t in tasks]
    monkeypatch.setattr(r, "execute_tasks", profile)
    args = SimpleNamespace(output=tmp_path / "report", stage="lbr", max_hours=10, workers=16)
    r.stage_run(args, records, {})
    assert len(calls) == 1
    assert json.loads((args.output / "lbr/STATUS.json").read_text())["status"] == "deferred_cost_gate"
    assert not (args.output / "lbr/SUCCESS.json").exists()


def test_lbr_pilot_timeout_leaves_main_success(sources, monkeypatch, tmp_path):
    records = r.discover(sources)
    write_json(tmp_path / "main/SUCCESS.json", {"sentinel": True})
    def timeout(*a, **kw):
        raise TimeoutError("pilot cap")
    monkeypatch.setattr(r, "execute_tasks", timeout)
    r.stage_run(SimpleNamespace(output=tmp_path, stage="lbr", max_hours=10, workers=16), records, {})
    assert json.loads((tmp_path / "main/SUCCESS.json").read_text()) == {"sentinel": True}
    assert json.loads((tmp_path / "lbr/STATUS.json").read_text())["status"] == "deferred_pilot_timeout"


def test_reports_and_completed_stage_reuse(sources, monkeypatch, tmp_path):
    from fhp_evaluation.statistics import sample_summary
    records = r.discover(sources)
    def simulated(tasks, *a, **kw):
        return [dict(task=t, elapsed_seconds=.01, summary=sample_summary([t["a"]["seed"], 2]),
                     paired_mbb=[t["a"]["seed"], 2]) for t in tasks]
    monkeypatch.setattr(r, "execute_tasks", simulated)
    args = SimpleNamespace(output=tmp_path / "plots", stage="main", max_hours=22, workers=16)
    r.stage_run(args, records, {"test": 1})
    assert (args.output / "main/final_head_to_head.png").exists()
    assert len(json.loads((args.output / "main/by_seed.json").read_text())) == 270
    def forbidden(*a, **kw):
        raise AssertionError("Completed scoring must not repeat")
    monkeypatch.setattr(r, "execute_tasks", forbidden)
    r.stage_run(args, records, {"test": 1})
    assert json.loads((args.output / "main/STATUS.json").read_text())["reused"]
    with pytest.raises(ValueError, match="changed"):
        r.stage_run(args, records, {"test": 2})


def test_batch_structure_and_source_allowlist(tmp_path):
    suite = ROOT.parents[1] / "fhp-evaluation-suite"
    args = SimpleNamespace(project="test-project", region="europe-west1", bucket="test-bucket",
                           service_account="worker@test-project.iam.gserviceaccount.com",
                           run_id="vr-eval123-test", suite=suite)
    batch.validate(args)
    checksum, provenance = batch.bundle(suite, tmp_path / "source.tar.gz")
    job = batch.build_job(args, checksum)
    task = job["taskGroups"][0]["taskSpec"]
    assert task["maxRetryCount"] == 0 and task["maxRunDuration"] == "129600s"
    assert task["computeResource"]["cpuMilli"] == 16000
    assert task["runnables"][-1]["alwaysRun"]
    assert task["environment"]["variables"]["OMP_NUM_THREADS"] == "1"
    assert all("cloud_outputs" not in p and not p.endswith(".pt") and "/.git/" not in p for p in provenance["files"])
    script = task["runnables"][0]["script"]["text"]
    assert script.index('STAGE=main\n') < script.index('STAGE=lbr\n')
    assert '"$OUT/main/SUCCESS.json"' in script
    assert '/usr/bin/python3 -I -m venv' in script
    assert batch.build_job(args, checksum, smoke=True)["taskGroups"][0]["taskSpec"]["maxRunDuration"] == "7200s"
    for runnable in task["runnables"]:
        subprocess.run(["bash", "-n"], input=runnable["script"]["text"], text=True, check=True)
    # Exercise the actual source bundle outside either checkout, without network.
    isolated = tmp_path / "unpacked"
    with tarfile.open(tmp_path / "source.tar.gz") as archive:
        archive.extractall(isolated, filter="data")
    env = dict(os.environ, PYTHONPATH=f"{isolated / 'evaluator'}:{isolated / 'native'}")
    subprocess.run([sys.executable, "-m", "experiments.fhp.retrospective_exp1_exp2_exp3_evaluation.run", "--help"],
                   env=env, cwd=isolated, check=True, capture_output=True)


def test_source_paths_cannot_escape(tmp_path):
    for value in ("../bad.pt", "/tmp/bad.pt"):
        with pytest.raises(ValueError, match="escapes"):
            r.contained(tmp_path, value)
