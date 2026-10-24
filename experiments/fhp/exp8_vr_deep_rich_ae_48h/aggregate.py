"""Require complete rich-AE policies, Ray accounting and final training states."""
from pathlib import Path
from experiments.fhp.exp1_vr_deep_pdcfr_24h import aggregate as native
from experiments.fhp.exp5_vr_deep_ray8.aggregate import verify_worker as verify_parallel
from experiments.fhp.exp6_vr_deep_ray8_48h.training_state import verify_state
from fhp_vr_deep.io_utils import read_json, sha256_file
from . import config

def verify_worker(worker, *, smoke=False):
    worker = Path(worker)
    manifest, summary, rows = verify_parallel(worker, smoke=smoke, experiment=config)
    expected = config.contract(smoke)
    for key in ("encoder_source", "ucv_reference_run_id", "initialisation", "start_hours", "target_hours", "resume_run_id"):
        if manifest.get(key) != expected[key]:
            raise ValueError(f"Wrong rich-AE contract: {key}")
    meta = verify_state(worker / "training_state", experiment=config)
    for key, value in dict(training_config=expected["training_config"], feature_encoder=expected["feature_encoder"],
        seed=manifest["seed"], is_smoke=smoke, completed_target_hours=48, parent=None,
        active_seconds=rows[-1]["training_elapsed_seconds"], iteration=rows[-1]["iteration"],
        nodes_touched=rows[-1]["nodes_touched"], repository_commit=manifest["repository_commit"]).items():
        if meta.get(key) != value:
            raise ValueError(f"Wrong final state: {key}")
    path = "training_state/manifest.json"
    success = read_json(worker / "SUCCESS.json")
    if (summary.get("final_training_state") != path or not summary.get("final_training_state_verified")
            or summary.get("final_training_state_sha256") != sha256_file(worker / path)):
        raise ValueError("Unverified final state")
    for name, record in meta["files"].items():
        if success["files"].get("training_state/" + name) != record["sha256"]:
            raise ValueError("State file absent from success inventory")
    return manifest, summary, rows

def aggregate(output_root, *, smoke=False):
    for i in range(1 if smoke else len(config.SEEDS)):
        verify_worker(Path(output_root) / "workers" / config.task_name(i), smoke=smoke)
    return native.aggregate(output_root, smoke=smoke, experiment=config)
