#!/usr/bin/env python3
"""Repository-owned launch contract over a pinned shared scoring implementation."""
import argparse
from copy import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shlex


def shared(args):
    config = json.loads(args.contract.read_text())
    if hashlib.sha256(args.shared_builder.read_bytes()).hexdigest() != config["shared_builder_sha256"]:
        raise ValueError("Shared Batch builder differs from pinned experiment dependency")
    spec = importlib.util.spec_from_file_location("_shared_representation_builder", args.shared_builder)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return config, module


def validate(args):
    config, module = shared(args)
    module.validate(args)
    if (args.algorithm != config["algorithm"] or config["experiment_id"] != 9
            or config["legacy_evaluation_id"] != {"sd":42,"vr":43}[args.algorithm]
            or config["training_seeds"] != [0,1,2] or config["pairs_per_cell"] != 100000
            or config["shard_pairs"] != 5000 or config["family_confidence"] != .975
            or config["base_seed"] != {"sd":642048,"vr":643048}[args.algorithm]):
        raise ValueError("Experiment contract differs from the frozen evaluation protocol")
    for cohort, pin in config["source_pins"].items():
        expected = module.SOURCES[args.algorithm][cohort]
        if pin["run_id"] != expected["run_id"] or pin["source_names"] != expected["source_names"]:
            raise ValueError("Source cohort differs from pinned shared builder")
    if not re.fullmatch(r"[0-9a-f]{40}", config["shared_evaluator_ref"]):
        raise ValueError("Shared evaluator must be pinned to a full commit")


def source_uris(args):
    return shared(args)[1].source_uris(args)


def metadata_uris(args):
    return shared(args)[1].metadata_uris(args)


def build_job(args):
    validate(args)
    config, module = shared(args)
    shared_args = copy(args)
    shared_args.repo_ref = config["shared_evaluator_ref"]
    job = module.build_job(shared_args)
    script = job["taskGroups"][0]["taskSpec"]["runnables"][0]["script"]["text"]
    q = shlex.quote
    owner = "\n".join([
        'OWNER_REPOSITORY="$WORK/owner"',
        f'git clone --filter=blob:none {q(config["repository_url"])} "$OWNER_REPOSITORY"',
        f'git -C "$OWNER_REPOSITORY" checkout --detach {q(args.repo_ref)}',
        f'[[ "$(git -C "$OWNER_REPOSITORY" rev-parse HEAD)" == {q(args.repo_ref)} ]]',
        f'cp "$OWNER_REPOSITORY/experiments/fhp/{config["package"]}/config.json" "$OUTPUT/experiment_contract.json"',
        f'git -C "$OWNER_REPOSITORY" rev-parse HEAD > "$OUTPUT/experiment_repository_commit.txt"',
        ''])
    marker = 'python -m experiments.fhp.representation_head_to_head.sources'
    if script.count(marker) != 1:
        raise ValueError("Shared worker template changed")
    script = script.replace(marker, owner+marker)
    old = 'python -m experiments.fhp.representation_head_to_head.run'
    if script.count(old) != 1:
        raise ValueError("Shared scoring invocation changed")
    script = script.replace(old, f'python "$OWNER_REPOSITORY/experiments/fhp/{config["package"]}/run.py" --ucv-repo "$REPOSITORY"')
    # The owner wrapper fixes the algorithm; only the metadata CLI retains its flag.
    start = script.index(f'python "$OWNER_REPOSITORY/experiments/fhp/{config["package"]}/run.py"')
    script = script[:start] + script[start:].replace('--algorithm '+args.algorithm+' ', '', 1)
    job["taskGroups"][0]["taskSpec"]["runnables"][0]["script"]["text"] = script
    job["labels"].update(owner_algorithm=args.algorithm, owner_experiment="9")
    return job


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("run-id", "bucket", "service-account", "repo-ref"):
        p.add_argument("--"+key, required=True)
    for name,bucket in (("sd","fhp-deep-cfr"),("vr","fhp-vr-deep")):
        p.add_argument("--"+name+"-bucket", default=f"gs://clever-overview-399515-{bucket}-results")
    for key in ("output", "contract", "shared-builder"):
        p.add_argument("--"+key,type=Path,required=True)
    p.add_argument("--algorithm",choices=("sd","vr"),required=True)
    p.add_argument("--smoke-only",action="store_true")
    p.add_argument("--resume",action="store_true")
    return p


def main():
    args=parser().parse_args()
    job=build_job(args)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(job,indent=2)+"\n")


if __name__ == "__main__":
    main()
