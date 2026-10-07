import importlib.util
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.fhp.retrospective_threeway_48h_evaluation import run as r

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "threeway_batch", ROOT / "gcp/threeway_48h_evaluation_batch.py"
)
BATCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BATCH)


@pytest.fixture
def records(tmp_path):
    result = []
    for algorithm_index, algorithm in enumerate(r.ALGORITHMS):
        for seed in r.SEEDS:
            path = tmp_path / f"{algorithm}-{seed}.policy"
            path.write_bytes(f"{algorithm}-{seed}".encode())
            result.append(dict(id=f"{algorithm}_s{seed}_48h", algorithm=algorithm,
                label=r.LABELS[algorithm], seed=seed, training_hours=48,
                active_seconds=172800 + seed, nodes_touched=(algorithm_index+1)*1000+seed,
                outer_iteration=100+seed, path=str(path), sha256=r.sha256(path),
                source_run=r.RUN_IDS[algorithm], source_commit=r.SOURCE_COMMITS[algorithm]))
    return result


def fake_result(task):
    value = {"sd": 0.03, "ucv": 0.01, "vr": -0.02}[task["a"]["algorithm"]]
    if task["kind"] == "direct":
        value -= {"sd": 0.03, "ucv": 0.01, "vr": -0.02}[task["b"]["algorithm"]]
    elif task["kind"] == "lbr":
        value = 0.05 - value
    result = dict(num_deal_pairs=task["num_deals"], num_games=2*task["num_deals"],
                  mean_chips_per_hand=value, std_chips_per_pair=1.0,
                  mean_mbb_per_hand=value*10, policy_a_player0_mean_chips=value,
                  policy_a_player1_mean_chips=value)
    return dict(task=r.portable(task), result=result, elapsed_seconds=.01)


def batch_args(**overrides):
    values = dict(project="test-project", region="europe-west1", run_id="fhp-threeway-test",
        bucket="gs://test-output", sd_bucket="gs://test-sd", ucv_bucket="gs://test-ucv",
        vr_bucket="gs://test-vr", sd_run_id=r.RUN_IDS["sd"], ucv_run_id=r.RUN_IDS["ucv"],
        ucv_source10_run_id=r.RUN_IDS["ucv_source10"], vr_run_id=r.RUN_IDS["vr"],
        repo_ref="a"*40, sd_ref=BATCH.DEFAULT_SD_REF, ucv_ref=BATCH.DEFAULT_UCV_REF,
        service_account="runner@test-project.iam.gserviceaccount.com", resume=False)
    values.update(overrides)
    return SimpleNamespace(**values)


def test_prespecified_task_contract(records):
    main = r.build_tasks(records, "production", "implementation")
    lbr = r.build_tasks(records, "lbr", "implementation")
    assert len(main) == 630
    assert sum(task["num_deals"] for task in main) == 3_150_000
    assert sum(task["kind"] == "direct" for task in main) == 540
    assert sum(task["kind"] == "rule" for task in main) == 90
    assert len(lbr) == 900
    assert sum(task["num_deals"] for task in lbr) == 9_000
    assert all(task["rollouts"] == 4096 for task in lbr)
    for left_seed, right_seed in ((0, 0), (1, 2)):
        selected = [task for task in main if task["kind"] == "direct"
                    and task["a"]["seed"] == left_seed and task["b"]["seed"] == right_seed
                    and task["task_id"].endswith("_000")]
        assert len(selected) == 3
        assert len({task["evaluation_seed"] for task in selected}) == 1
    profile = r.build_tasks(records, "profile", "implementation")
    assert {task["evaluation_seed"] for task in profile}.isdisjoint(
        task["evaluation_seed"] for task in main
    )


def test_smoke_exercises_every_source_and_opponent_class(records):
    smoke = r.build_tasks(records, "smoke", "implementation")
    lbr = r.build_tasks(records, "lbr_smoke", "implementation")
    assert len(smoke) == 72
    assert len(lbr) == 9
    assert all(task["num_deals"] == 1 and task["rollouts"] == 16 for task in lbr)
    assert {(task["a"]["algorithm"], task["b"]["algorithm"])
            for task in smoke if task["kind"] == "direct"} == {
                ("sd", "ucv"), ("sd", "vr"), ("ucv", "vr")}
    assert {task["a"]["algorithm"] for task in lbr} == set(r.ALGORITHMS)


def test_reports_require_complete_budgets_and_write_family_intervals(records, tmp_path, monkeypatch):
    def plot_fixture(_matrices, _summaries, output):
        (Path(output) / "final_48h_head_to_head.png").write_bytes(b"fixture")
    monkeypatch.setattr(r, "plots", plot_fixture)
    main_tasks = r.build_tasks(records, "production", "implementation")
    summary = r.report_main([fake_result(task) for task in main_tasks], tmp_path / "main")
    assert len(summary["direct_comparisons"]) == 3
    assert summary["family_confidence"] == pytest.approx(1 - .05/3)
    assert (tmp_path / "main/final_48h_head_to_head.png").is_file()
    assert (tmp_path / "main/analysis_summary.md").is_file()
    rows = (tmp_path / "main/direct_comparison_summary.csv").read_text().splitlines()
    assert len(rows) == 4
    incomplete = [fake_result(task) for task in main_tasks[:-1]]
    with pytest.raises(ValueError, match="Incomplete rule-agent"):
        r.report_main(incomplete, tmp_path / "incomplete")
    lbr = r.report_lbr([fake_result(task) for task in r.build_tasks(
        records, "lbr", "implementation")], tmp_path / "lbr")
    assert lbr["lower_bound_only"] is True and len(lbr["results"]) == 3


def test_crossed_bootstrap_rejects_partial_matrix():
    with pytest.raises(ValueError, match="complete 3x3"):
        r.cluster_interval(np.ones((2, 3)))
    low, high = r.cluster_interval(np.arange(9).reshape(3, 3), confidence=r.FAMILY_CONFIDENCE)
    assert low < high


def test_csv_schema_includes_fields_present_only_in_later_records(tmp_path):
    path = tmp_path / "heterogeneous.csv"
    r.write_csv(path, [dict(algorithm="sd", seed=0),
                       dict(algorithm="ucv", seed=1, continuation_sha256="abc")])
    assert path.read_text().splitlines() == [
        "algorithm,seed,continuation_sha256",
        "sd,0,",
        "ucv,1,abc",
    ]


def test_lbr_pilot_timeout_preserves_completed_main(records, tmp_path, monkeypatch):
    (tmp_path / "main").mkdir()
    r.write_json(tmp_path / "main/SUCCESS.json", {"sentinel": True})
    monkeypatch.setattr(r, "run_tasks", lambda *args, **kwargs: (_ for _ in ()).throw(
        TimeoutError("pilot deadline")
    ))
    args = SimpleNamespace(output=tmp_path, stage="lbr", max_hours=10, workers=16,
                           sd_repo=tmp_path / "sd", ucv_repo=tmp_path / "ucv")
    r.run_stage(args, records, "implementation", {"identity": 1})
    assert r.read_json(tmp_path / "main/SUCCESS.json") == {"sentinel": True}
    assert r.read_json(tmp_path / "lbr/STATUS.json")["status"] == "deferred_pilot_timeout"


def test_batch_job_is_resumable_single_vm_and_shell_valid(tmp_path):
    args = batch_args()
    BATCH.validate(args)
    job = BATCH.job_config(args)
    task = job["taskGroups"][0]["taskSpec"]
    assert task["maxRetryCount"] == 0
    assert task["maxRunDuration"] == "129600s"
    assert task["computeResource"] == {"cpuMilli": 16000, "memoryMib": 62000}
    policy = job["allocationPolicy"]["instances"][0]["policy"]
    assert policy["machineType"] == "n2-standard-16"
    assert policy["bootDisk"]["sizeGb"] == 200
    assert task["runnables"][-1]["alwaysRun"]
    script = task["runnables"][0]["script"]["text"]
    assert script.index(f"-m {BATCH.MODULE} main") < script.index(f"-m {BATCH.MODULE} lbr")
    assert 'if [[ ! -f "$OUTPUT/main/SUCCESS.json" ]]' in script
    assert "time_48h" in script and "training_state" not in script
    assert r.RUN_IDS["sd"] in script and r.RUN_IDS["ucv"] in script and r.RUN_IDS["vr"] in script
    for index, runnable in enumerate(task["runnables"]):
        file = tmp_path / f"runnable-{index}.sh"
        file.write_text(runnable["script"]["text"])
        subprocess.run(["bash", "-n", str(file)], check=True)
    finalizer = task["runnables"][-1]["script"]["text"]
    assert finalizer.index('exec 9>"$WORK/upload.lock"') < finalizer.index("flock -w 180 9")
    assert BATCH.job_config(args, smoke=True)["taskGroups"][0]["taskSpec"]["maxRunDuration"] == "7200s"


def test_batch_rejects_source_namespace_and_unpinned_refs():
    with pytest.raises(ValueError, match="must not reuse"):
        BATCH.validate(batch_args(run_id=r.RUN_IDS["vr"]))
    with pytest.raises(ValueError, match="full pushed"):
        BATCH.validate(batch_args(repo_ref="main"))
    with pytest.raises(ValueError, match="frozen 48-hour"):
        BATCH.validate(batch_args(vr_run_id="another-vr-run"))


def test_source_paths_cannot_escape(tmp_path):
    for value in ("../bad.pt", "/tmp/bad.pt"):
        with pytest.raises(ValueError, match="escapes"):
            r.contained(tmp_path, value)


def test_launcher_is_syntax_valid():
    subprocess.run(["bash", "-n", str(ROOT / "gcp/run_threeway_48h_evaluation.sh")], check=True)
