"""Aggregate training artifacts and evaluate matched FHP policy snapshots."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from fhp_vr_deep.evaluation import evaluate_snapshot_pair
from fhp_vr_deep.io_utils import read_json, stats, write_csv, write_json

from .config import ALGORITHMS


def discover_results(run_dirs: Iterable[str | Path]) -> list[dict[str, object]]:
    """Find relocatable worker results and reject duplicate algorithm/seed pairs."""
    indexed = {}
    for run_dir in (Path(value) for value in run_dirs):
        for result_path in sorted(run_dir.glob("worker_runs/*/result.json")):
            result = read_json(result_path)
            summary = result["summary"]
            key = (str(summary["algorithm_id"]), int(summary["seed"]))
            if key in indexed:
                raise ValueError(f"Duplicate worker result for {key}")
            result["_resolved_worker_dir"] = str(result_path.parent.resolve())
            indexed[key] = result
    return [indexed[key] for key in sorted(indexed)]


def _aggregate_seed_summaries(summary_rows):
    aggregate = {}
    for algorithm_id in ALGORITHMS:
        rows = [row for row in summary_rows if row["algorithm_id"] == algorithm_id]
        numeric_fields = {
            key
            for row in rows
            for key, value in row.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        aggregate[algorithm_id] = {
            field: stats(float(row.get(field, np.nan)) for row in rows)
            for field in sorted(numeric_fields)
            if field != "seed"
        }
    return aggregate


def _time_checkpoint_rows(curves):
    return [
        row
        for row in curves
        if row.get("checkpoint_target_seconds") is not None
        and str(row.get("checkpoint_kind", "")).startswith("training_time")
    ]


def _aggregate_checkpoints(curves):
    grouped = defaultdict(list)
    for row in _time_checkpoint_rows(curves):
        key = (str(row["algorithm_id"]), float(row["checkpoint_target_seconds"]))
        grouped[key].append(row)
    output = []
    metrics = (
        "nodes_touched",
        "training_elapsed_seconds",
        "wall_clock_seconds",
        "average_policy_loss",
        "regret_loss_0",
        "regret_loss_1",
        "baseline_loss_0",
        "baseline_loss_1",
    )
    for (algorithm_id, target_seconds), rows in sorted(grouped.items()):
        output.append(
            {
                "algorithm_id": algorithm_id,
                "algorithm_label": ALGORITHMS[algorithm_id]["algorithm_label"],
                "checkpoint_target_seconds": target_seconds,
                "checkpoint_target_hours": target_seconds / 3600.0,
                "metrics": {
                    metric: stats(float(row.get(metric, np.nan)) for row in rows)
                    for metric in metrics
                },
            }
        )
    return output


def _resolved_snapshot(result: Mapping[str, object], snapshot_row: Mapping[str, object]):
    worker_dir = Path(
        str(result.get("_resolved_worker_dir", result.get("worker_run_dir", "")))
    )
    return worker_dir / str(snapshot_row["snapshot_path"])


def run_head_to_head(results, *, num_deals: int) -> list[dict[str, object]]:
    snapshots = {}
    for result in results:
        summary = result["summary"]
        algorithm_id = str(summary["algorithm_id"])
        seed = int(summary["seed"])
        for snapshot in result["snapshots"]:
            target = float(snapshot["checkpoint_target_seconds"])
            snapshots[(algorithm_id, seed, target)] = _resolved_snapshot(
                result, snapshot
            )

    common_pairs = []
    seeds = sorted({key[1] for key in snapshots})
    targets = sorted({key[2] for key in snapshots})
    for target_index, target in enumerate(targets):
        for seed in seeds:
            dcfr = snapshots.get(("vr_deep_dcfr_plus", seed, target))
            pdcfr = snapshots.get(("vr_deep_pdcfr_plus", seed, target))
            if dcfr is None or pdcfr is None:
                continue
            evaluation_seed = 7_000_000 + 10_000 * target_index + seed
            common_pairs.append(
                evaluate_snapshot_pair(
                    dcfr,
                    pdcfr,
                    training_seed=seed,
                    checkpoint_target_seconds=target,
                    num_deals=num_deals,
                    evaluation_seed=evaluation_seed,
                )
            )
    return common_pairs


def _aggregate_head_to_head(rows):
    output = {}
    targets = sorted({float(row["checkpoint_target_seconds"]) for row in rows})
    for target in targets:
        selected = [
            row for row in rows if float(row["checkpoint_target_seconds"]) == target
        ]
        output[str(target)] = {
            "checkpoint_target_seconds": target,
            "checkpoint_target_hours": target / 3600.0,
            "dcfr_minus_pdcfr_chips": stats(
                float(row["dcfr_seat_averaged_mean_chips"]) for row in selected
            ),
            "dcfr_minus_pdcfr_mbb_per_hand": stats(
                float(row["dcfr_seat_averaged_mean_mbb_per_hand"])
                for row in selected
            ),
        }
    return output


def _plot_training(output_dir: Path, curves) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"vr_deep_dcfr_plus": "#1f77b4", "vr_deep_pdcfr_plus": "#2ca02c"}
    time_rows = _time_checkpoint_rows(curves)
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    for algorithm_id, spec in ALGORITHMS.items():
        selected = [row for row in time_rows if row["algorithm_id"] == algorithm_id]
        for seed in sorted({int(row["seed"]) for row in selected}):
            seed_rows = sorted(
                [row for row in selected if int(row["seed"]) == seed],
                key=lambda row: float(row["checkpoint_target_seconds"]),
            )
            ax.plot(
                [float(row["checkpoint_target_seconds"]) / 3600.0 for row in seed_rows],
                [float(row["nodes_touched"]) for row in seed_rows],
                color=colors[algorithm_id],
                linewidth=1.0,
                alpha=0.2,
            )
        targets = sorted({float(row["checkpoint_target_seconds"]) for row in selected})
        means, errors = [], []
        for target in targets:
            values = [
                float(row["nodes_touched"])
                for row in selected
                if float(row["checkpoint_target_seconds"]) == target
            ]
            metric = stats(values)
            means.append(metric["mean"])
            errors.append(metric["se"])
        ax.errorbar(
            [target / 3600.0 for target in targets],
            means,
            yerr=errors,
            marker="o",
            capsize=4,
            linewidth=2.0,
            color=colors[algorithm_id],
            label=spec["algorithm_label"],
        )
    ax.set_xlabel("Effective training time (hours)")
    ax.set_ylabel("Training nodes touched")
    ax.set_title("Archived Experiment 1: FHP VR-Deep training progress")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "nodes_by_training_time.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_head_to_head(output_dir: Path, rows) -> None:
    if not rows:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    targets = sorted({float(row["checkpoint_target_seconds"]) for row in rows})
    means, errors = [], []
    for target in targets:
        metric = stats(
            float(row["dcfr_seat_averaged_mean_mbb_per_hand"])
            for row in rows
            if float(row["checkpoint_target_seconds"]) == target
        )
        means.append(metric["mean"])
        errors.append(metric["se"])
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    x = np.arange(len(targets))
    ax.bar(x, means, yerr=errors, capsize=5, color="#1f77b4")
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{target / 3600.0:g}h" for target in targets])
    ax.set_xlabel("Training checkpoint")
    ax.set_ylabel("VR-DeepDCFR+ − VR-DeepPDCFR+ (mbb/hand)")
    ax.set_title("Archived Experiment 1: sampled seat-swapped FHP comparison")
    fig.tight_layout()
    fig.savefig(output_dir / "head_to_head_by_checkpoint.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def aggregate_results(
    results: list[dict[str, object]],
    output_dir: str | Path,
    *,
    head_to_head_deals: int,
    run_evaluation: bool,
) -> dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = [dict(result["summary"]) for result in results]
    curve_rows = [dict(row) for result in results for row in result["curves"]]
    snapshot_rows = []
    for result in results:
        summary = result["summary"]
        for row in result["snapshots"]:
            snapshot_rows.append(
                {
                    "algorithm_id": summary["algorithm_id"],
                    "algorithm_label": summary["algorithm_label"],
                    "seed": summary["seed"],
                    **dict(row),
                }
            )
    aggregate = _aggregate_seed_summaries(summary_rows)
    checkpoint_aggregate = _aggregate_checkpoints(curve_rows)
    h2h_rows = (
        run_head_to_head(results, num_deals=head_to_head_deals)
        if run_evaluation
        else []
    )
    h2h_aggregate = _aggregate_head_to_head(h2h_rows)

    write_csv(output_dir / "seed_summary.csv", summary_rows)
    write_csv(output_dir / "checkpoint_curves.csv", curve_rows)
    write_json(output_dir / "checkpoint_curves.json", curve_rows)
    write_csv(output_dir / "checkpoint_manifest.csv", snapshot_rows)
    write_csv(output_dir / "head_to_head_pairs.csv", h2h_rows)
    write_json(output_dir / "aggregate_summary.json", aggregate)
    write_json(output_dir / "checkpoint_aggregate.json", checkpoint_aggregate)
    write_json(output_dir / "head_to_head_summary.json", h2h_aggregate)
    summary = {
        "seed_summary": summary_rows,
        "aggregate": aggregate,
        "checkpoint_aggregate": checkpoint_aggregate,
        "head_to_head": h2h_aggregate,
        "head_to_head_deals_per_pair": head_to_head_deals if run_evaluation else 0,
    }
    write_json(output_dir / "summary.json", summary)
    if curve_rows:
        _plot_training(output_dir, curve_rows)
    if h2h_rows:
        _plot_head_to_head(output_dir, h2h_rows)
    return summary
