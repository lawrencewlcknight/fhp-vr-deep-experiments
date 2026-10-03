"""Strict, relocatable aggregation of the three completed seed workers."""

from pathlib import Path

import numpy as np

from fhp_vr_deep.io_utils import read_json, sha256_file, stats, write_csv
from . import config as default_experiment
from .train import verify_policy, write_json


def safe_path(root, relative):
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe artifact path: {relative}")
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"Artifact escapes worker directory: {relative}")
    return resolved


def verify_worker(worker_dir, *, smoke=False, experiment=default_experiment):
    ALGORITHM_ID, EXPERIMENT_NAME = experiment.ALGORITHM_ID, experiment.EXPERIMENT_NAME
    schedule = experiment.schedule
    worker_dir = Path(worker_dir)
    success = read_json(worker_dir / "SUCCESS.json")
    required = {"run_manifest.json", "summary.json", "checkpoint_manifest.json",
                "checkpoint_rows.json", "training_progress.jsonl"}
    if not required <= set(success["files"]):
        raise ValueError("Incomplete success inventory")
    for relative, digest in success["files"].items():
        path = safe_path(worker_dir, relative)
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"Missing/corrupt artifact: {relative}")
    manifest = read_json(worker_dir / "run_manifest.json")
    summary = read_json(worker_dir / "summary.json")
    snapshots = read_json(worker_dir / "checkpoint_manifest.json")
    expected = experiment.contract(smoke)
    for key in ("experiment_name", "algorithm_id", "training_config", "training_config_sha256",
                "checkpoint_schedule", "is_smoke", "checkpoint_boundary", "artifact_retention",
                "input_representation", "replay_storage"):
        if manifest[key] != expected[key]:
            raise ValueError(f"Worker has wrong {key}")
    if manifest.get("feature_encoder") != expected.get("feature_encoder"):
        raise ValueError("Worker has wrong feature encoder")
    if (summary["seed"] != manifest["seed"] or success["seed"] != manifest["seed"]
            or summary["is_smoke"] != smoke or summary["algorithm_id"] != ALGORITHM_ID
            or summary["checkpoint_count"] != 4
            or summary["experiment_name"] != EXPERIMENT_NAME
            or success["experiment_name"] != EXPERIMENT_NAME
            or success["training_config_sha256"] != expected["training_config_sha256"]
            or summary["stop_reason"] != "training_time_budget" or len(snapshots) != 4):
        raise ValueError("Worker identity or completion state is inconsistent")
    previous_nodes = previous_time = 0
    for index, (row, target) in enumerate(zip(snapshots, schedule(smoke))):
        if any(row[k] != v for k, v in target.items()):
            raise ValueError("Incorrect checkpoint schedule")
        if (row["seed"] != manifest["seed"] or row["algorithm_id"] != ALGORITHM_ID
                or row["checkpoint_index"] != index
                or row["completed_iteration"] != row["iteration"]):
            raise ValueError("Incorrect checkpoint identity")
        if (row["training_elapsed_seconds"] < row["checkpoint_target_seconds"]
                or row["training_elapsed_seconds"] < previous_time
                or row["nodes_touched"] < previous_nodes):
            raise ValueError("Checkpoint clock or node count is inconsistent")
        previous_nodes, previous_time = row["nodes_touched"], row["training_elapsed_seconds"]
        if success["files"].get(row["path"]) != row["sha256"]:
            raise ValueError("Policy absent from verified success inventory")
        metadata = verify_policy(safe_path(worker_dir, row["path"]))
        if metadata.get("feature_encoder") != expected.get("feature_encoder"):
            raise ValueError("Playable policy has wrong feature encoder")
        if (metadata["seed"] != row["seed"] or metadata["algorithm_id"] != ALGORITHM_ID
                or metadata["nodes_touched"] != row["nodes_touched"]
                or metadata["iteration"] != row["completed_iteration"]
                or metadata["authors_parameterisation"] != expected["training_config"]
                or metadata["checkpoint"]["checkpoint_id"] != row["checkpoint_id"]
                or metadata["checkpoint"]["training_elapsed_seconds"] != row["training_elapsed_seconds"]):
            raise ValueError("Playable policy disagrees with checkpoint metadata")
    final = snapshots[-1]
    if (summary["final_policy_snapshot"] != final["path"]
            or summary["final_policy_sha256"] != final["sha256"]
            or summary["final_nodes_touched"] != final["nodes_touched"]
            or summary["final_training_elapsed_seconds"] != final["training_elapsed_seconds"]):
        raise ValueError("Summary does not match final playable checkpoint")
    return manifest, summary, snapshots


def aggregate(output_root, *, smoke=False, experiment=default_experiment):
    ALGORITHM_ID, EXPERIMENT_NAME = experiment.ALGORITHM_ID, experiment.EXPERIMENT_NAME
    SEEDS, schedule, task_name = experiment.SEEDS, experiment.schedule, experiment.task_name
    root = Path(output_root)
    expected_seeds = (0,) if smoke else SEEDS
    expected_dirs = {task_name(SEEDS.index(seed)) for seed in expected_seeds}
    actual_dirs = {p.name for p in (root / "workers").iterdir() if p.is_dir()}
    if actual_dirs != expected_dirs:
        raise ValueError(f"Expected exactly {sorted(expected_dirs)}, found {sorted(actual_dirs)}")
    manifests, summaries, checkpoints, progress = [], [], [], []
    for index, seed in enumerate(expected_seeds):
        worker_dir = root / "workers" / task_name(index)
        manifest, summary, rows = verify_worker(worker_dir, smoke=smoke, experiment=experiment)
        if manifest["seed"] != seed or manifest["task_index"] != index:
            raise ValueError("Worker seed does not match task directory")
        manifests.append(manifest)
        summaries.append(summary)
        for row in rows:
            checkpoints.append(dict(row, path=str((worker_dir / row["path"]).relative_to(root))))
        import json
        progress.extend(json.loads(line) for line in (worker_dir / "training_progress.jsonl").read_text().splitlines())
    if (not all(m["repository_commit"] for m in manifests)
            or len({m["repository_commit"] for m in manifests}) != 1):
        raise ValueError("Cannot combine seeds trained from different code commits")
    analysis = root / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    metrics = ("nodes_touched", "training_elapsed_seconds", "wall_clock_seconds", "overshoot_seconds",
               "average_policy_loss", "regret_loss_0", "regret_loss_1", "baseline_loss_0", "baseline_loss_1")
    aggregated = []
    for target in schedule(smoke):
        rows = [r for r in checkpoints if r["checkpoint_id"] == target["checkpoint_id"]]
        row = dict(target, algorithm_id=ALGORITHM_ID, training_seeds=len(rows))
        for metric in metrics:
            values = [float(r.get(metric, np.nan)) for r in rows]
            values = np.asarray(values)[np.isfinite(values)]
            summary = stats(values)
            # One seed is a functional smoke, not evidence of zero uncertainty.
            row[metric + "_mean"] = summary["mean"]
            row[metric + "_se"] = summary["se"] if len(values) > 1 else None
        aggregated.append(row)
    write_csv(analysis / "seed_summary.csv", [{k: v for k, v in r.items() if not isinstance(v, dict)} for r in summaries])
    write_json(analysis / "seed_summary.json", summaries)
    write_csv(analysis / "checkpoint_seed_metrics.csv", checkpoints)
    write_csv(analysis / "checkpoint_aggregate.csv", aggregated)
    write_json(analysis / "checkpoint_aggregate.json", aggregated)
    write_csv(analysis / "training_progress.csv", [{k: v for k, v in r.items() if not isinstance(v, dict)} for r in progress])
    write_json(analysis / "snapshot_inventory.json", checkpoints)
    write_json(analysis / "summary.json", dict(experiment_name=EXPERIMENT_NAME,
               seeds=list(expected_seeds), policy_count=len(checkpoints), is_smoke=smoke,
               exact_exploitability=False, evaluation_status="deferred_to_shared_suite",
               checkpoint_metrics=aggregated, contract=experiment.contract(smoke)))
    _plot(analysis, checkpoints, aggregated, title=experiment.ALGORITHM_LABEL)
    write_json(analysis / "SUCCESS.json", dict(experiment_name=EXPERIMENT_NAME, seeds=list(expected_seeds),
               files={p.name: sha256_file(p) for p in analysis.iterdir() if p.is_file() and p.name != "SUCCESS.json"}))
    return analysis


def _plot(output, rows, aggregated, *, title="FHP VR-DeepPDCFR+ — 24-hour baseline"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    for seed in sorted({r["seed"] for r in rows}):
        selected = [r for r in rows if r["seed"] == seed]
        ax.plot([r["training_elapsed_seconds"] / 3600 for r in selected],
                [r["nodes_touched"] / 1e6 for r in selected], alpha=0.4, label=f"Seed {seed}")
    ax.errorbar([r["training_elapsed_seconds_mean"] / 3600 for r in aggregated],
                [r["nodes_touched_mean"] / 1e6 for r in aggregated],
                yerr=[(r["nodes_touched_se"] or 0) / 1e6 for r in aggregated],
                color="black", marker="o", capsize=3, label="Mean ± one SE")
    ax.set(xlabel="Actual active training hours", ylabel="Training nodes (millions)",
           title=title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "nodes_by_training_time.png", dpi=160)
    plt.close(fig)
