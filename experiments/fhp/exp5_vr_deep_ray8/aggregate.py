"""Reject silently sequential/misconfigured workers before standard aggregation."""

from pathlib import Path

from experiments.fhp.exp1_vr_deep_pdcfr_24h import aggregate as baseline
from fhp_vr_deep.io_utils import read_json
from . import config


def verify_worker(worker_dir, *, smoke=False, experiment=config):
    manifest, summary, rows = baseline.verify_worker(worker_dir, smoke=smoke, experiment=experiment)
    expected = experiment.contract(smoke)
    for key in ("reference_vm", "baseline_reference_vm", "baseline_commit", "single_intended_change",
                "parallel_traversal_workers", "traversal_worker_threads", "learner_threads", "central_fitting",
                "traversal_execution", "ray_version", "ray_object_store_bytes", "traversal_rng", "merge_order",
                "actor_max_restarts", "actor_max_task_retries", "collection_timeout_seconds", "training_clock"):
        if manifest.get(key) != expected[key]:
            raise ValueError(f"Worker has wrong {key}")
    if not smoke and manifest["torch_threads"] != config.THREADS:
        raise ValueError("Worker has wrong torch_threads")
    success = read_json(Path(worker_dir) / "SUCCESS.json")
    if "parallel_runtime.json" not in success["files"]:
        raise ValueError("Missing verified Ray runtime inventory")
    runtime = read_json(Path(worker_dir) / "parallel_runtime.json")
    actors = runtime["workers"]
    if (runtime["backend"] != "ray" or runtime["ray_version"] != config.RAY_VERSION
            or runtime["object_store_bytes"] != expected["ray_object_store_bytes"]
            or runtime["max_restarts"] != 0 or runtime["max_task_retries"] != 0
            or len(actors) != 8 or len({row["pid"] for row in actors}) != 8
            or [row["worker_id"] for row in actors] != list(range(8))
            or any(row["torch_threads"] != 1 or row["interop_threads"] != 1 for row in actors)):
        raise ValueError("Invalid eight-actor runtime")
    last = rows[-1]
    if (last.get("parallel_workers") != 8 or last.get("parallel_collections") != 2 * last["iteration"]
            or last.get("parallel_traversals") != last["episode"]
            or last["episode"] != 2 * last["iteration"] * expected["training_config"]["num_traversals"]):
        raise ValueError("Incomplete parallel traversal accounting")
    return manifest, summary, rows


def aggregate(output_root, *, smoke=False):
    for index in range(1 if smoke else len(config.SEEDS)):
        verify_worker(Path(output_root) / "workers" / config.task_name(index), smoke=smoke)
    return baseline.aggregate(output_root, smoke=smoke, experiment=config)
