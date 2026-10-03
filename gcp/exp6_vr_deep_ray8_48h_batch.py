#!/usr/bin/env python3
"""Stdlib-only 48-hour orchestration and explicit continuation into a fresh run."""

import hashlib
import json
import os
import re

from exp1_vr_deep_pdcfr_24h_batch import cloud, main, q

ALGORITHM_ID = "lossless_vr_deep_pdcfr_plus_ray8_48h"


def experiment(environment=None):
    env = os.environ if environment is None else environment
    source = env.get("EXP6_RESUME_RUN_ID", "")
    start = int(env.get("EXP6_START_HOURS", "48" if source else "0"))
    target = int(env.get("EXP6_TARGET_HOURS", "72" if source else "48"))
    if (start < 0 or start % 6 or target % 6 or not start < target <= start + 48
            or (not source and (start != 0 or target != 48))
            or (source and (start < 48 or not re.fullmatch(r"[a-z][a-z0-9-]{1,34}", source)))):
        raise ValueError("Fresh: 0 to 48 hours. Resume: valid source ID, six-hour boundaries, <=48 additional hours")
    segment = f"--start-hours {start} --target-hours {target}"
    setup, worker = "", segment
    if source:
        if source == env.get("RUN_ID"):
            raise ValueError("Continuation needs a NEW RUN_ID; never overwrite its source")
        segment += f" --resume-run-id {q(source)}"
        setup = ('gcloud storage rsync --recursive '
                 '"${BUCKET%/}/$EXP6_RESUME_RUN_ID/workers/$TASK/training_state" "$WORK/resume-state"')
        worker = segment + ' --resume-from "$WORK/resume-state"'

    def verify_source(args):
        if not source:
            return
        if source == args.run_id:
            raise ValueError("Continuation cannot overwrite its source run")
        source_contract = None
        for seed in (0, 1, 2):
            root = f"{args.bucket}/{source}/workers/task_{seed:03d}_{ALGORITHM_ID}_seed_{seed}"
            success = json.loads(cloud(args, "storage", "cat", root + "/SUCCESS.json", capture=True).stdout)
            raw = cloud(args, "storage", "cat", root + "/training_state/manifest.json", capture=True).stdout
            meta = json.loads(raw)
            if (success["files"].get("training_state/manifest.json") != hashlib.sha256(raw.encode()).hexdigest()
                    or meta["schema"] != "fhp_vr_ray_training_state_v1" or meta["algorithm_id"] != ALGORITHM_ID
                    or meta["seed"] != seed or meta["is_smoke"] or meta["completed_target_hours"] != start
                    or meta["repository_commit"] != args.repo_ref or meta["active_seconds"] >= (start + 6) * 3600):
                raise ValueError(f"Incompatible/incomplete seed {seed} source; use its pinned code and completed target")
            contract = (meta["runtime"], meta["training_config_sha256"], meta["threads"], meta["game"], meta["feature_encoder"])
            if source_contract is not None and contract != source_contract:
                raise ValueError("Source seeds have different runtime/training contracts")
            source_contract = contract

    return dict(
        number=6, module="experiments.fhp.exp6_vr_deep_ray8_48h", algorithm_id=ALGORITHM_ID,
        batch_script="gcp/exp6_vr_deep_ray8_48h_batch.py", label="fhp-vr-exp6-48h",
        worker_resources=dict(machine_type="n2-standard-16", cpu_milli=16000, memory_mib=62000),
        stage_seconds=dict(train=72 * 3600, controller=96 * 3600),
        extra_requirements="requirements-ray.txt", smoke_test_environment=dict(FHP_RUN_RAY_TESTS="1"),
        test_files=("tests/test_exp6_vr_deep_ray8_48h.py", "tests/test_exp5_vr_deep_ray8.py",
                    "tests/test_vr_deep_parallel.py", "tests/test_efficiency.py"),
        extra_environment=dict(EXP6_START_HOURS=str(start), EXP6_TARGET_HOURS=str(target), EXP6_RESUME_RUN_ID=source),
        worker_setup=setup, worker_arguments=worker, aggregate_arguments=segment,
        runtime_manifest=(f"{source}/workers/task_000_{ALGORITHM_ID}_seed_0/training_state/manifest.json" if source else None),
        preflight_hook=verify_source)


if __name__ == "__main__":
    main(experiment=experiment())
