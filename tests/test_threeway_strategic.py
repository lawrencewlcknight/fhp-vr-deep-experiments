"""Offline integration guards. No trained checkpoint or paid cloud job is used."""

import json
from pathlib import Path
import re
import subprocess
from types import SimpleNamespace

import pytest

from experiments.fhp.retrospective_threeway_48h_evaluation import strategic as s
from test_threeway_48h_evaluation import BATCH, batch_args, records


@pytest.fixture
def suite(monkeypatch):
    # Tests needing the suite run in their own pytest process with the shared
    # suite before the vendored packages on PYTHONPATH, like the actual job.
    import fhp_evaluation
    root = Path(__file__).resolve().parents[3] / "fhp-evaluation-suite"
    if Path(fhp_evaluation.__file__).resolve().parent.parent != root:
        pytest.fail("Run strategic tests with the shared fhp-evaluation-suite first on PYTHONPATH")
    return root


@pytest.mark.parametrize("stage,count", [("pilot", 3), ("development", 9)])
def test_spec_uses_same_48h_candidates_and_frozen_panel(records, tmp_path, suite, stage, count):
    repos = {a: tmp_path for a in s.ALGORITHMS}
    spec = s.make_spec(records, repos, tmp_path, stage)
    assert len(spec["candidates"]) == count
    assert len(spec["opponents"]) == 8
    assert spec["reference_policy"] == "exp9_seed0_24h"
    assert spec["range_tilts"] == [0, -2, 2]
    assert {spec["policies"][p]["family"] for p in spec["candidates"]} == {"sd_cfr", "ucv", "vr_deep"}
    assert all(spec["policies"][p]["hours"] == 48 for p in spec["candidates"])
    assert all(spec["policies"][p]["require_suit_invariance"] for p in spec["candidates"])
    from fhp_evaluation.strategic_audit.exp16 import make_spec
    previous = make_spec("development", tmp_path, tmp_path)
    assert spec["opponents"] == previous["opponents"]
    assert {p: spec["policies"][p]["sha256"] for p in spec["opponents"][5:]} == {
        p: previous["policies"][p]["sha256"] for p in previous["opponents"][5:]}


def test_no_assessment_or_changed_checkpoint(records, tmp_path, suite):
    repos = {a: tmp_path for a in s.ALGORITHMS}
    with pytest.raises(ValueError, match="locked"):
        s.make_spec(records, repos, tmp_path, "assessment")
    with pytest.raises(ValueError, match="complete nine"):
        s.make_spec(records[:-1], repos, tmp_path, "pilot")
    Path(records[0]["path"]).write_text("changed")
    with pytest.raises(ValueError, match="Changed"):
        s.make_spec(records, repos, tmp_path, "pilot")


def fake_result(stage):
    seeds = (0,) if stage == "pilot" else range(3)
    spec = dict(candidates=[f"{a}_seed{i}_48h" for a in s.ALGORITHMS for i in seeds],
                opponents=[f"opp{i}" for i in range(8)], range_tilts=[0, -2, 2])
    comparisons = {}
    for ai, algorithm in enumerate(s.ALGORITHMS):
        for seed in seeds:
            for oi, opp in enumerate(spec["opponents"]):
                for tilt in spec["range_tilts"]:
                    comparisons[f"{algorithm}_seed{seed}_48h__vs__{opp}__tilt_{tilt:g}"] = dict(
                        boards=[dict(board=[0, 5, 10], core=True, root_count=50)],
                        mean_root_response_gap_bb=ai + seed*.1 + oi*.01)
    return dict(status="succeeded", provenance=dict(spec=spec), comparisons=comparisons)


def test_equal_panel_weighting_and_independent_seed_summary(tmp_path, suite):
    result = fake_result("development")
    summary = s.write_comparison(tmp_path, result, "development")
    row = summary["endpoints"]["tilt_0"]["sd"]["equal_weight_panel"]
    assert row["n"] == 3
    assert row["mean"] == pytest.approx(.135)
    assert row["seed_values"]["0"] == pytest.approx(.035)
    assert row["standard_error"] == pytest.approx(.1 / 3**.5)
    assert not summary["lbr_evaluated"] and not summary["assessment_evaluated"]
    assert (tmp_path / "threeway_report.html").exists()
    pilot = s.summarise(fake_result("pilot"), "pilot")
    assert not pilot["strength_evidence"]
    assert pilot["endpoints"]["tilt_0"]["sd"]["equal_weight_panel"]["standard_error"] is None


def test_refuse_incomplete_or_different_support():
    result = fake_result("development")
    first = next(iter(result["comparisons"].values()))
    first["boards"][0]["root_count"] = 49
    with pytest.raises(ValueError, match="support differs"):
        s.summarise(result, "development")
    result = fake_result("development")
    result["status"] = "failed"
    with pytest.raises(ValueError, match="complete"):
        s.summarise(result, "development")


@pytest.mark.parametrize("component,hours", [("strategic-pilot", 6), ("strategic-development", 24)])
def test_audit_cloud_job_is_isolated_bounded_and_shell_valid(tmp_path, component, hours):
    args = batch_args(evaluation=component, pilot_reviewed=component.endswith("development"),
                      pilot_run_id="fhp-threeway-audit-pilot" if component.endswith("development") else None)
    BATCH.validate(args)
    job = BATCH.job_config(args)
    task = job["taskGroups"][0]["taskSpec"]
    assert task["maxRunDuration"] == f"{hours*3600}s"
    script = task["runnables"][0]["script"]["text"]
    assert BATCH.DEFAULT_SUITE_REF in script and BATCH.DEFAULT_SD_AUDIT_REF in script
    assert f"-m {BATCH.MODULE} prepare-strategic" in script
    assert f"-m {BATCH.MODULE} main" not in script
    assert f"-m {BATCH.MODULE} lbr" not in script
    assert 'export PYTHONPATH="$WORK/evaluation-suite"' in script
    assert "--unlock-assessment" not in script
    assert "cached_parallel_structured_ucv_escher" in script
    for i, runnable in enumerate(task["runnables"]):
        path = tmp_path / f"audit-{i}.sh"
        path.write_text(runnable["script"]["text"])
        subprocess.run(["bash", "-n", str(path)], check=True)


def test_development_requires_explicit_review_and_matching_pilot(monkeypatch):
    with pytest.raises(ValueError, match="requires"):
        BATCH.validate(batch_args(evaluation="strategic-development"))
    args = batch_args(evaluation="strategic-development", pilot_reviewed=True, pilot_run_id="pilot-frozen")
    BATCH.validate(args)
    refs = BATCH.execution_refs(args)
    contract = dict(protocol=s.PROTOCOL, board_exports=s.export_resources("pilot"), execution_refs=refs, source_identity=dict(
        source_uris=sorted(f"{k}={v}" for k, v in BATCH.source_uris(args).items())))
    success = dict(status="succeeded", stage="pilot", execution_refs=refs,
                   lbr_evaluated=False, assessment_evaluated=False, contract_sha256=s.digest(contract))
    def cloud(_args, *command, **kwargs):
        return SimpleNamespace(stdout=json.dumps(success if command[-1].endswith("SUCCESS.json") else contract))
    monkeypatch.setattr(BATCH, "cloud", cloud)
    BATCH.check_pilot(args)
    success["execution_refs"] = dict(refs, suite_ref="f"*40)
    with pytest.raises(ValueError, match="identical"):
        BATCH.check_pilot(args)


def test_cache_upload_does_not_publish_success_before_data():
    script = BATCH.upload_outputs()
    assert script.count("rsync --recursive") == 2
    assert "|| return 1" in script
    pattern = re.findall(r"--exclude='([^']+)'", script)[1]
    assert not re.search(pattern, "strategic-pilot/tables/sd/board/SUCCESS.json")
    assert not re.search(pattern, "strategic-pilot/board_results/board/SUCCESS.json")
    for path in ("strategic-pilot/analysis/SUCCESS.json", "smoke/SUCCESS.json", "main/SUCCESS.json"):
        assert re.search(pattern, path)
    assert script.rindex("storage cp") > script.rindex("rsync")
    assert script.index('cp "$marker"') < script.index("rsync --recursive")
    assert '"$markers" "$DESTINATION/analysis"' in script


def test_generated_refs_are_valid_json(tmp_path):
    args = batch_args(evaluation="strategic-pilot")
    line = next(line for line in BATCH.worker_script(args, smoke=False).splitlines()
                if "execution_refs.json" in line and line.startswith("printf"))
    subprocess.run(["bash", "-c", line], env={"OUTPUT": str(tmp_path)}, check=True)
    assert json.loads((tmp_path / "execution_refs.json").read_text()) == BATCH.execution_refs(args)


@pytest.mark.parametrize("outcome", ["complete", "deferred", "failed"])
def test_wrapper_uses_shared_workflow_and_checks_completed_report(records, tmp_path, suite, monkeypatch, outcome):
    from fhp_evaluation.strategic_audit import workflow, validation
    from fhp_evaluation.best_response.runner import write_json
    refs = BATCH.execution_refs(batch_args())
    identity = dict(sources=[{k: v for k, v in row.items() if k != "path"} for row in records], lbr_evaluated=False)
    write_json(tmp_path / "sources.json", dict(records=records, identity=identity))
    write_json(tmp_path / "refs.json", refs)
    candidate_spec = s.make_spec(records, {a: tmp_path for a in s.ALGORITHMS}, tmp_path, "pilot")
    monkeypatch.setattr(workflow, "validate_spec", lambda spec: None)  # Synthetic files, not playable checkpoints.
    monkeypatch.setattr(validation, "validate", lambda: dict(status="passed"))
    def git(command, **kwargs):
        return SimpleNamespace(stdout=(refs["suite_ref"] if command[2] == str(suite) else "a"*40) + "\n")
    refs.update(sd_audit_ref="a"*40, ucv_ref="a"*40)
    write_json(tmp_path / "refs.json", refs)
    monkeypatch.setattr(s.subprocess, "run", git)
    calls = []
    def fake_run(spec, library, output, **kwargs):
        calls.append(kwargs)
        if outcome == "deferred":
            from fhp_evaluation.strategic_audit.board_export import AuditDeferred
            raise AuditDeferred("fixture session budget")
        if outcome == "failed":
            raise RuntimeError("fixture invalid policy")
        assert spec == candidate_spec
        assert len([b for b in library["boards"] if b["partition"] == "development"]) == 8
        result = fake_result("pilot")
        for index, name in enumerate(spec["opponents"]):
            for label in list(result["comparisons"]):
                if f"__vs__opp{index}__" in label:
                    result["comparisons"][label.replace(f"__vs__opp{index}__", f"__vs__{name}__")] = result["comparisons"].pop(label)
        result["provenance"]["spec"] = spec
        path = output / "analysis/summary.json"
        write_json(path, result)
        return path
    monkeypatch.setattr(workflow, "run", fake_run)
    output = tmp_path / "output"
    argv = ["--stage", "pilot", "--sources", str(tmp_path / "sources.json"),
            "--execution-refs", str(tmp_path / "refs.json"), "--suite-repo", str(suite),
            "--sd-repo", str(tmp_path), "--ucv-repo", str(tmp_path), "--vr-repo", str(tmp_path),
            "--anchor-root", str(tmp_path), "--output", str(output)]
    if outcome == "failed":
        with pytest.raises(RuntimeError, match="invalid policy"):
            s.main(argv)
    else:
        s.main(argv)
    assert calls == [dict(partition="development", include_reference=True, unlock_assessment=False,
                          resume=False, threads=4, max_seconds=10800, max_rss_mb=52000,
                          board_exports=s.export_resources("pilot"))]
    if outcome != "complete":
        status = s.read(output / "STATUS.json")
        assert status["status"] == ("deferred_time_budget" if outcome == "deferred" else "incomplete")
        assert not (output / "analysis/SUCCESS.json").exists()
        with pytest.raises(ValueError, match="Changed strategic contract"):
            s.main(argv + ["--export-workers", "2"])
        return
    assert s.read(output / "analysis/SUCCESS.json")["status"] == "succeeded"
    s.main(argv)
    assert len(calls) == 1
    (output / "analysis/threeway_summary.json").write_text("changed")
    with pytest.raises(ValueError, match="Changed completed"):
        s.main(argv)


def test_archive_resume_restores_stamps_only_for_identical_chunks(tmp_path, suite):
    from fhp_evaluation.loaders import sha256_file
    chunk = tmp_path / "chunk.npy"
    chunk.write_bytes(b"fixture-not-a-checkpoint")
    sha = sha256_file(chunk)
    spec = dict(policies=dict(sd=dict(family="sd_cfr", checkpoint=str(tmp_path / "48h.json"))))
    saved = dict(spec=spec, policies=dict(sd=dict(native=dict(
        archive_file_stamps=[["chunk.npy", chunk.stat().st_size, 1_000_000_000]], chunk_hashes={"chunk.npy": sha}))))
    (tmp_path / "manifest.json").write_text(json.dumps(saved))
    s.restore_archive_stamps(tmp_path, spec)
    assert chunk.stat().st_mtime_ns == 1_000_000_000
    chunk.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="Changed archive"):
        s.restore_archive_stamps(tmp_path, spec)


@pytest.mark.parametrize("workers", [1, 2, 4])
def test_cloud_worker_resources_are_explicit(workers):
    args = batch_args(evaluation="strategic-pilot", strategic_export_workers=workers)
    BATCH.validate(args)
    assert f"--export-workers {workers}" in BATCH.worker_script(args, smoke=False)
    resources = s.export_resources("pilot", workers)
    assert resources["workers"] * 4 <= resources["cpu_limit"] == 16
    assert resources["session_seconds"] == 16200
    with pytest.raises(ValueError):
        BATCH.validate(batch_args(strategic_export_workers=8))


def test_preflight_refuses_legacy_or_changed_strategic_resume(monkeypatch):
    args = batch_args(evaluation="strategic-pilot", resume=True)
    contract = dict(protocol=s.PROTOCOL, stage="pilot", execution_refs=BATCH.execution_refs(args),
                    board_exports=s.export_resources("pilot"),
                    source_identity=dict(source_uris=sorted(f"{k}={v}" for k, v in BATCH.source_uris(args).items())))
    monkeypatch.setattr(BATCH, "cloud", lambda *a, **k: SimpleNamespace(stdout=json.dumps(contract)))
    BATCH.check_strategic_resume(args)
    for name, value in (("protocol", "fhp_threeway_48h_strategic_v1"),
                        ("board_exports", s.export_resources("pilot", 2)),
                        ("execution_refs", {}), ("stage", "development"), ("source_identity", {})):
        original = contract[name]
        contract[name] = value
        with pytest.raises(ValueError, match="fresh RUN_ID"):
            BATCH.check_strategic_resume(args)
        contract[name] = original


def test_pilot_gate_rejects_changed_parallelism(monkeypatch):
    args = batch_args(evaluation="strategic-development", pilot_reviewed=True, pilot_run_id="pilot-frozen")
    refs = BATCH.execution_refs(args)
    contract = dict(protocol=s.PROTOCOL, execution_refs=refs, board_exports=s.export_resources("pilot", 2),
                    source_identity=dict(source_uris=sorted(f"{k}={v}" for k, v in BATCH.source_uris(args).items())))
    success = dict(status="succeeded", stage="pilot", execution_refs=refs,
                   lbr_evaluated=False, assessment_evaluated=False, contract_sha256=s.digest(contract))
    monkeypatch.setattr(BATCH, "cloud", lambda _a, *c, **k:
                        SimpleNamespace(stdout=json.dumps(success if c[-1].endswith("SUCCESS.json") else contract)))
    with pytest.raises(ValueError, match="same board-export"):
        BATCH.check_pilot(args)
