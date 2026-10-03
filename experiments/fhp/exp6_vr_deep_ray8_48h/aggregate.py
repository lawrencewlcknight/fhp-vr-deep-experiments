"""Policies plus a verified, relocatable full training state for each seed."""

from pathlib import Path

from experiments.fhp.exp1_vr_deep_pdcfr_24h import aggregate as baseline
from experiments.fhp.exp5_vr_deep_ray8.aggregate import verify_worker as verify_parallel_worker
from fhp_vr_deep.io_utils import read_json, sha256_file
from . import config
from .training_state import verify_state


def verify_worker(worker_dir, *, smoke=False, experiment=None):
    spec = experiment or config.RunSpec()
    worker_dir = Path(worker_dir)
    manifest, summary, rows = verify_parallel_worker(worker_dir, smoke=smoke, experiment=spec)
    for key in ("start_hours", "target_hours", "resume_run_id", "final_state_schema"):
        if manifest.get(key) != spec.contract(smoke)[key]:
            raise ValueError(f"Worker has wrong {key}")
    state_path = "training_state/manifest.json"
    success = read_json(worker_dir / "SUCCESS.json")
    digest = sha256_file(worker_dir / state_path)
    if (summary.get("final_training_state") != state_path
            or summary.get("final_training_state_sha256") != digest
            or success["files"].get(state_path) != digest
            or summary.get("final_training_state_verified") is not True):
        raise ValueError("Missing verified final training state")
    meta = verify_state(worker_dir / "training_state")
    for filename, record in meta["files"].items():
        if success["files"].get("training_state/" + filename) != record["sha256"]:
            raise ValueError("State file absent from success inventory")
    expected = dict(training_config=manifest["training_config"], seed=manifest["seed"], is_smoke=smoke,
                    completed_target_hours=spec.target_hours, active_seconds=rows[-1]["training_elapsed_seconds"],
                    iteration=rows[-1]["iteration"], episode=rows[-1]["episode"],
                    nodes_touched=rows[-1]["nodes_touched"], repository_commit=manifest["repository_commit"],
                    threads=manifest["torch_threads"], feature_encoder=manifest["feature_encoder"],
                    parent=summary.get("continuation_source"))
    if any(meta.get(key) != value for key, value in expected.items()):
        raise ValueError("Final training state disagrees with completed run")
    if spec.start_hours:
        parent = read_json(worker_dir / "continuation_source.json")
        if (meta["parent"] != parent or parent["run_id"] != spec.resume_run_id
                or parent["completed_target_hours"] != spec.start_hours
                or "continuation_source.json" not in success["files"]):
            raise ValueError("Invalid continuation provenance")
    elif meta["parent"] is not None:
        raise ValueError("Fresh run must not claim a continuation source")
    return manifest, summary, rows


def aggregate(output_root, *, smoke=False, start_hours=0, target_hours=48, resume_run_id=None):
    spec = config.RunSpec(start_hours, target_hours, resume_run_id)
    for index in range(1 if smoke else len(config.SEEDS)):
        verify_worker(Path(output_root) / "workers" / config.task_name(index), smoke=smoke, experiment=spec)
    return baseline.aggregate(output_root, smoke=smoke, experiment=spec)
