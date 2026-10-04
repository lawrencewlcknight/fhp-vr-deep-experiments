"""Read-only VR-Deep adapters and frozen 1/2/3 evaluation orchestration."""
from __future__ import annotations

import argparse
import importlib
from importlib.metadata import version
import itertools
import json
import os
from pathlib import Path
import platform
import subprocess
import time
import traceback

from fhp_vr_deep.evaluation_adapter import _import_suite, load_policy_for_evaluation

_import_suite()
from fhp_evaluation.cohort import (cost_projection, digest, execute_tasks, portable,
                                  sha256, write_json)
from fhp_evaluation.cohort_report import report
from fhp_evaluation.rule_agents import PUBLISHED_AGENT_NAMES

SOURCES = json.loads(Path(__file__).with_name("sources.json").read_text())
MODULES = {"exp1": "exp1_vr_deep_pdcfr_24h", "exp2": "exp2_vr_deep_lossless_24h",
           "exp3": "exp3_vr_deep_lossless_n2_standard16"}
LOADER = "fhp_vr_deep.evaluation_adapter:load_policy_for_evaluation"
HOURS = (6, 12, 18, 24)
BASE_SEED = 20260922


def contained(root, relative):
    path = Path(root) / relative
    if Path(relative).is_absolute() or ".." in Path(relative).parts or not path.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError("Source path escapes its run")
    return path


def inventory(root, experiment):
    source = SOURCES[experiment]
    root = Path(root) / source["run_id"]
    success = json.loads((root / "analysis/SUCCESS.json").read_text())
    if success["experiment_name"] != source["experiment_name"] or success["seeds"] != [0, 1, 2]:
        raise ValueError("Incorrect source success identity")
    for name, expected in source["analysis_hashes"].items():
        if success["files"].get(name) != expected or sha256(root / "analysis" / name) != expected:
            raise ValueError(f"Source analysis differs from frozen run: {experiment}/{name}")
    summary = json.loads((root / "analysis/summary.json").read_text())
    if summary["is_smoke"] or summary["policy_count"] != 12 or summary["seeds"] != [0, 1, 2]:
        raise ValueError("Expected a complete production source")
    rows = json.loads((root / "analysis/snapshot_inventory.json").read_text())
    keys = [(r["seed"], r["checkpoint_target_hours"]) for r in rows]
    if len(keys) != 12 or set(keys) != set(itertools.product(range(3), HOURS)):
        raise ValueError("Missing or duplicate source checkpoints")
    for row in rows:
        contained(root, row["path"])
    return root, rows


def download(root, bucket):
    """Download only frozen manifests and playable policies, never replay/state."""
    def fetch(uri, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["gcloud", "storage", "cp", uri, str(path)], check=True, timeout=180)
    for experiment, source in SOURCES.items():
        run = source["run_id"]
        for name in ("SUCCESS.json", "summary.json", "snapshot_inventory.json"):
            fetch(f"{bucket}/{run}/analysis/{name}", Path(root) / run / "analysis" / name)
        run_root, rows = inventory(root, experiment)
        workers = {str(Path(row["path"]).parent.parent) for row in rows}
        paths = [row["path"] for row in rows]
        paths += [f"{worker}/{name}" for worker in workers
                  for name in ("SUCCESS.json", "run_manifest.json", "summary.json", "checkpoint_manifest.json")]
        for relative in paths:
            fetch(f"{bucket}/{run}/{relative}", contained(run_root, relative))


def discover(root):
    records = []
    for experiment in SOURCES:
        source_root, rows = inventory(root, experiment)
        config = importlib.import_module(f"experiments.fhp.{MODULES[experiment]}.config")
        expected = config.contract(False)
        commits = set()
        for row in rows:
            path = contained(source_root, row["path"])
            worker = path.parent.parent
            success = json.loads((worker / "SUCCESS.json").read_text())
            for name in ("run_manifest.json", "summary.json", "checkpoint_manifest.json"):
                if sha256(worker / name) != success["files"].get(name):
                    raise ValueError("Corrupt source worker metadata")
            manifest = json.loads((worker / "run_manifest.json").read_text())
            for key in ("experiment_name", "algorithm_id", "training_config", "is_smoke",
                        "checkpoint_schedule", "feature_encoder", "reference_vm"):
                if manifest.get(key) != expected.get(key):
                    raise ValueError(f"Wrong source {key}: {worker}")
            if manifest["seed"] != row["seed"] or manifest["torch_threads"] != 8:
                raise ValueError("Wrong training seed/thread count")
            commits.add(manifest["repository_commit"])
            if (sha256(path) != row["sha256"] or path.stat().st_size != row["size_bytes"]
                    or success["files"].get(str(path.relative_to(worker))) != row["sha256"]):
                raise ValueError("Policy missing from verified worker inventory")
            _, policy = load_policy_for_evaluation(path)
            payload = policy.metadata
            for key in ("seed", "algorithm_id", "nodes_touched", "iteration"):
                if payload[key] != row[key]:
                    raise ValueError(f"Policy metadata mismatch: {key}")
            if (payload["authors_parameterisation"] != expected["training_config"]
                    or payload.get("feature_encoder") != expected.get("feature_encoder")
                    or payload["checkpoint"]["checkpoint_id"] != row["checkpoint_id"]
                    or payload["checkpoint"]["training_elapsed_seconds"] != row["training_elapsed_seconds"]
                    or row["training_elapsed_seconds"] < row["checkpoint_target_hours"] * 3600):
                raise ValueError("Policy configuration/checkpoint clock mismatch")
            records.append(dict(id=f"{experiment}_s{row['seed']}_{row['checkpoint_target_hours']}h",
                                experiment=experiment, seed=row["seed"], hours=row["checkpoint_target_hours"],
                                nodes=row["nodes_touched"], iteration=row["iteration"],
                                active_seconds=row["training_elapsed_seconds"],
                                sha256=row["sha256"], path=str(path.resolve()),
                                source_run=SOURCES[experiment]["run_id"], source_commit=manifest["repository_commit"],
                                source_manifest_sha256=sha256(worker / "run_manifest.json")))
        if len(commits) != 1 or not next(iter(commits)):
            raise ValueError("Mixed or unknown training commits within experiment")
    return records


def build_tasks(records, stage):
    tasks = []
    def add(kind, a, b=None, opponent=None, deals=50000, seed=BASE_SEED):
        match = f"{kind}_{a['id']}_{b['id'] if b else opponent}"
        task = dict(id=match, match_id=match, kind=kind, a=a, loader=LOADER,
                    evaluation_seed=seed, deals=deals)
        if b:
            task["b"] = b
        if opponent:
            task["opponent"] = opponent
        tasks.append(task)
        return task
    lookup = {(r["experiment"], r["seed"], r["hours"]): r for r in records}
    if stage == "lbr":
        for row in records:
            if row["hours"] != 24:
                continue
            for shard in range(100):
                task = add("lbr", row, opponent="local_best_response", deals=10,
                           seed=BASE_SEED + 1000000 + shard)
                task.update(id=task["id"] + f"_{shard:03d}", shard=shard,
                            rollouts=4096, lbr_seed=BASE_SEED + 1500000)
        return tasks
    for row in records:
        for index, name in enumerate(PUBLISHED_AGENT_NAMES):
            add("rule", row, opponent=name, deals=10000, seed=BASE_SEED + 100000 + index)
    for seed in range(3):
        for left, right in (("exp2", "exp1"), ("exp3", "exp1"), ("exp3", "exp2")):
            for hour in HOURS:
                add("direct", lookup[left, seed, hour], lookup[right, seed, hour], seed=BASE_SEED+3000000+hour)
        for experiment in SOURCES:
            for earlier, later in itertools.combinations(HOURS, 2):
                add("temporal", lookup[experiment, seed, later], lookup[experiment, seed, earlier],
                    seed=BASE_SEED+2000000+earlier*1000+later)
    return tasks


def probe_tasks(tasks, stage):
    probes = []
    for task in tasks:
        if stage == "lbr":
            if task["shard"] >= 3:
                continue
            count = 10  # Three independent full-rollout batches per final policy.
        else:
            if task["a"]["hours"] != 24 or (task["kind"] == "temporal" and task["b"]["hours"] != 18):
                continue
            count = 50
        probes.append(dict(task, id="probe_" + task["id"], deals=count,
                           evaluation_seed=task["evaluation_seed"] + 10000000))
    return probes


def evaluator_identity():
    import fhp_evaluation
    root = Path(__file__).resolve().parents[3]
    hashes = {}
    for prefix, directory in (("suite", Path(fhp_evaluation.__file__).parent),
                              ("native", root / "fhp_vr_deep"), ("solver", root / "vr_deep_cfr"),
                              ("experiments", root / "experiments")):
        for path in sorted(directory.rglob("*.py")):
            hashes[f"{prefix}/{path.relative_to(directory)}"] = sha256(path)
    return dict(source_files=hashes, frozen_sources=SOURCES, python=platform.python_version(),
                dependencies={name: version(name) for name in ("torch", "numpy", "open_spiel", "scipy", "matplotlib")},
                deal_protocol="ucv_retrospective_split_arrays_v1", base_seed=BASE_SEED)


def stage_run(args, records, identity):
    output = args.output / args.stage
    tasks = build_tasks(records, args.stage)
    completed = output / "SUCCESS.json"
    if completed.exists() and args.stage != "smoke":
        success = json.loads(completed.read_text())
        if (success["identity_sha256"] != digest(identity) or not success["files"]
                or any(sha256(contained(output, name)) != expected for name, expected in success["files"].items())):
            raise ValueError("Completed stage has changed or corrupt outputs")
        write_json(output / "STATUS.json", dict(status="complete", reused=True, stage=args.stage))
        return
    if args.stage == "smoke":
        # All 36 policies reload above; exercise both encoders, all opponent types,
        # all matchup modes and LBR on each of the three experiment families.
        tasks = [dict(t, deals=2) for t in tasks if t["a"]["seed"] == 0 and t["a"]["hours"] == 24]
        tasks += [dict(t, deals=2, rollouts=16) for t in build_tasks(records, "lbr")
                  if t["a"]["seed"] == 0 and t["shard"] == 0]
        results = execute_tasks(tasks, output, identity, workers=args.workers, seconds=900)
        write_json(output / "SUCCESS.json", dict(smoke=True, tasks=len(results), strength_evidence=False))
        return
    start = time.monotonic()
    deadline = start + args.max_hours * 3600
    write_json(output / "STATUS.json", dict(status="profiling", stage=args.stage))
    probes = probe_tasks(tasks, args.stage)
    try:
        measured = execute_tasks(probes, output / "pilot", identity, workers=args.workers,
                                 seconds=min(900, args.max_hours * 3600))
    except TimeoutError:
        write_json(output / "STATUS.json", dict(status="deferred_pilot_timeout", stage=args.stage,
                   message="Full scoring not started; completed main results are unaffected."))
        return
    estimate = cost_projection(measured, tasks, args.workers)
    remaining = deadline - time.monotonic()
    decision = dict(projected_seconds_with_2x_margin=estimate, available_seconds=remaining,
                    pilot_pairs=sum(t["deals"] for t in probes), workers=args.workers,
                    allowed=estimate <= remaining, full_task_count=len(tasks))
    write_json(output / "cost_gate.json", decision)
    if not decision["allowed"]:
        write_json(output / "STATUS.json", dict(status="deferred_cost_gate", stage=args.stage, **decision))
        return
    write_json(output / "STATUS.json", dict(status="scoring", stage=args.stage))
    results = execute_tasks(tasks, output / "scoring", identity, workers=args.workers,
                            seconds=max(1, remaining - 60))
    summary = report(results, output)
    files = {p.name: sha256(p) for p in output.iterdir() if p.is_file()
             and p.name not in ("STATUS.json", "SUCCESS.json") and not p.name.endswith(".tmp")}
    write_json(output / "SUCCESS.json", dict(status="complete", **summary,
               identity_sha256=digest(identity), files=files))
    write_json(output / "STATUS.json", dict(status="complete", stage=args.stage))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("download", "smoke", "main", "lbr"))
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bucket")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max-hours", type=float, default=22)
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 16 or not 0 < args.max_hours <= 36:
        parser.error("Require 1..16 workers and a positive cap <=36 hours")
    if args.stage == "download":
        if not args.bucket or not args.bucket.startswith("gs://"):
            parser.error("download requires --bucket gs://...")
        download(args.source_root, args.bucket.rstrip("/"))
        return
    try:
        records = discover(args.source_root)
        identity = evaluator_identity()
        args.output.mkdir(parents=True, exist_ok=True)
        provenance = args.output / "evaluation_manifest.json"
        contract = dict(identity=identity, policies=portable(records), workers=args.workers)
        if provenance.exists() and json.loads(provenance.read_text()) != contract:
            raise ValueError("Output already belongs to another evaluation contract")
        write_json(provenance, contract)
        stage_run(args, records, dict(identity, workers=args.workers))
    except Exception:
        write_json(args.output / args.stage / "STATUS.json", dict(status="failed", traceback=traceback.format_exc()))
        raise


if __name__ == "__main__":
    main()
