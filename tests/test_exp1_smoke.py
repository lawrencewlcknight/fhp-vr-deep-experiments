import csv
from pathlib import Path
import shutil

from experiments.fhp.exp1_leduc_config_transfer.run import main
from fhp_vr_deep.io_utils import read_json
from vr_deep_cfr.policy_snapshots import load_policy_snapshot_payload


def _csv_rows(path: Path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_full_experiment_smoke_runs_both_variants_and_analysis(tmp_path):
    assert (
        main(
            [
                "--smoke",
                "--seeds",
                "0",
                "--head-to-head-deals",
                "2",
                "--output-root",
                str(tmp_path),
            ]
        )
        == 0
    )
    run_dirs = list(tmp_path.glob("exp1_fhp_vr_deep_leduc_config_transfer_*"))
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    status = read_json(run_dir / "run_status.json")
    assert status["completed_workers"] == 2
    assert status["failures"] == []
    assert len(_csv_rows(run_dir / "seed_summary.csv")) == 2
    assert len(_csv_rows(run_dir / "checkpoint_manifest.csv")) == 4
    head_to_head_rows = _csv_rows(run_dir / "head_to_head_pairs.csv")
    assert len(head_to_head_rows) == 2
    assert {int(row["num_games"]) for row in head_to_head_rows} == {4}
    assert {row["metric"] for row in head_to_head_rows} == {
        "vr_deep_dcfr_plus_minus_vr_deep_pdcfr_plus"
    }
    assert (run_dir / "nodes_by_training_time.png").is_file()
    assert (run_dir / "head_to_head_by_checkpoint.png").is_file()

    for algorithm_id in ("vr_deep_dcfr_plus", "vr_deep_pdcfr_plus"):
        worker_dir = run_dir / "worker_runs" / f"{algorithm_id}_seed_0"
        worker_summary = read_json(worker_dir / "summary.json")
        assert worker_summary["stop_reason"] == "training_time_budget"
        assert worker_summary["checkpoint_count"] == 2
        snapshot = load_policy_snapshot_payload(worker_dir / "final_policy_snapshot.pt")
        assert snapshot["algorithm_id"] == algorithm_id
        assert snapshot["game"]["name"] == "FHP"

    relocated = tmp_path / "relocated_download"
    shutil.copytree(run_dir, relocated)
    aggregate_root = tmp_path / "reaggregated"
    assert (
        main(
            [
                "--aggregate-run-dir",
                str(relocated),
                "--head-to-head-deals",
                "1",
                "--output-root",
                str(aggregate_root),
            ]
        )
        == 0
    )
    aggregate_dirs = list(aggregate_root.glob("*_aggregated_*"))
    assert len(aggregate_dirs) == 1
    assert len(_csv_rows(aggregate_dirs[0] / "head_to_head_pairs.csv")) == 2
