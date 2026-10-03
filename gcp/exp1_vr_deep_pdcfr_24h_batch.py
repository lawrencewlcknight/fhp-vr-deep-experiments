#!/usr/bin/env python3
"""Stdlib-only controller: cloud smoke, three separate seed VMs, aggregation."""

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile
import time

REPO_URL = "https://github.com/lawrencewlcknight/fhp-vr-deep-experiments.git"
MODULE = "experiments.fhp.exp1_vr_deep_pdcfr_24h"
STAGES = ("smoke", "train", "aggregate")
DEFAULT_EXPERIMENT = dict(number=1, module=MODULE, algorithm_id="vr_deep_pdcfr_plus",
                          batch_script="gcp/exp1_vr_deep_pdcfr_24h_batch.py",
                          test_files=("tests/test_exp1_vr_deep_pdcfr_24h.py", "tests/test_efficiency.py"))


def settings(args):
    return getattr(args, "experiment", DEFAULT_EXPERIMENT)


def q(value):
    return shlex.quote(str(value))


def bootstrap(args, *, controller=False):
    number = settings(args)["number"]
    text = f"""#!/usr/bin/env bash
set -Eeuo pipefail
export DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export MPLCONFIGDIR=/tmp/fhp-vr-mpl XDG_CACHE_HOME=/tmp/fhp-vr-cache
if command -v sudo >/dev/null 2>&1; then SUDO=sudo; else SUDO=; fi
$SUDO apt-get update
$SUDO apt-get install -y git curl ca-certificates python3 python3-venv python3-dev build-essential
command -v gcloud >/dev/null || {{ echo 'Batch image must provide gcloud' >&2; exit 1; }}
WORK=/workspace/vr-exp{number}
mkdir -p "$WORK"
git clone --filter=blob:none {q(REPO_URL)} "$WORK/repository"
cd "$WORK/repository"
git checkout --detach {q(args.repo_ref)}
"""
    if controller:
        return text
    text += """
export UV_CACHE_DIR=/tmp/uv-cache UV_PYTHON_INSTALL_DIR=/tmp/uv-python
curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/tmp/uv-bin UV_NO_MODIFY_PATH=1 sh
export PATH="/tmp/uv-bin:$PATH"
uv python install 3.11
uv venv --python 3.11 --seed /tmp/fhp-vr-exp1-venv
source /tmp/fhp-vr-exp1-venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install --no-cache-dir --no-build-isolation -r requirements.txt
python -m pip install --no-deps -e .
python -m pip check
OUT="$WORK/output"
mkdir -p "$OUT"
"""
    if settings(args).get("extra_requirements"):
        text += f"python -m pip install --no-cache-dir -r {q(settings(args)['extra_requirements'])}\npython -m pip check\n"
    return text


def script(args, stage):
    spec = settings(args)
    module = spec["module"]
    remote = f"{args.bucket}/{args.run_id}"
    env = dict(PROJECT_ID=args.project, REGION=args.region, BUCKET=args.bucket,
               SA_EMAIL=args.service_account, REPO_REF=args.repo_ref, RUN_ID=args.run_id)
    exports = "\n".join(f"export {key}={q(value)}" for key, value in env.items()) + "\n"
    if stage == "controller":
        return bootstrap(args, controller=True) + exports + (
            f"exec python3 {q(spec['batch_script'])} orchestrate\n")
    text = bootstrap(args) + exports
    if stage == "smoke":
        text += "".join(f"export {key}={q(value)}\n" for key, value in spec.get("smoke_test_environment", {}).items())
        return text + f"""
finish() {{
  code=$?
  gcloud storage rsync --recursive "$OUT" {q(remote + '/smoke')} || {{ if [[ $code == 0 ]]; then code=1; fi; }}
  exit "$code"
}}
trap finish EXIT
python -m pip install -r requirements-dev.txt
python -m pytest -q -p no:cacheprovider {' '.join(q(p) for p in spec['test_files'])}
python -m benchmarks.vr_deep_efficiency --threads 8 --iterations 2 --traversals 2048 --capacity 16384 \
  --batch 2048 --regret-steps 32 --critic-steps 101 --policy-steps 16 --seeds 0 1 2 \
  | tee "$OUT/equivalence.jsonl"
python -m {module}.run smoke --output-root "$OUT/training" --threads 8
"""
    if stage == "train":
        return text + f"""
INDEX="${{BATCH_TASK_INDEX:?Missing Batch task index}}"
case "$INDEX" in 0|1|2) ;; *) exit 2 ;; esac
TASK="task_$(printf '%03d' "$INDEX")_{spec['algorithm_id']}_seed_$INDEX"
REMOTE={q(remote + '/workers')}/$TASK
DIAGNOSTICS="$WORK/diagnostics"
mkdir -p "$DIAGNOSTICS"
python -m fhp_vr_deep.batch_diagnostics monitor --output "$DIAGNOSTICS/resources.jsonl" \
  --interval-seconds 30 --cloud-log-every 10 &
MONITOR_PID=$!
finish() {{
  code=$?
  trap - EXIT
  kill "$MONITOR_PID" >/dev/null 2>&1 || true
  wait "$MONITOR_PID" >/dev/null 2>&1 || true
  python -m fhp_vr_deep.batch_diagnostics finalize \
    --snapshots "$DIAGNOSTICS/resources.jsonl" --output "$DIAGNOSTICS/summary.json" \
    --status-output "$DIAGNOSTICS/status.json" --failure-root "$OUT" --exit-code "$code" \
    --requested-memory-mib {spec.get('worker_resources', {}).get('memory_mib', 30000)} --job-name {q(args.run_id + '-train')} --bucket-destination "$REMOTE" || true
  gcloud storage rsync --recursive "$DIAGNOSTICS" {q(remote + '/diagnostics')}/"$TASK" || true
  # The trainer publishes the completion marker only after all files land.
  # A failed final upload must never be converted to success by this trap.
  if [[ $code != 0 && -d "$OUT/workers/$TASK" ]]; then
    gcloud storage rsync --recursive --exclude='(^|/)SUCCESS[.]json$|[.]tmp$' \
      "$OUT/workers/$TASK" "$REMOTE" || true
  fi
  exit "$code"
}}
trap finish EXIT
trap 'exit 143' TERM
python -m {module}.run worker --task-index "$INDEX" --output-root "$OUT" --remote-uri "$REMOTE"
"""
    if stage == "aggregate":
        return text + f"""
gcloud storage rsync --recursive {q(remote + '/workers')} "$OUT/workers"
python -m {module}.run aggregate --output-root "$OUT"
gcloud storage rsync --recursive --exclude='(^|/)SUCCESS[.]json$' "$OUT/analysis" {q(remote + '/analysis')}
gcloud storage cp "$OUT/analysis/SUCCESS.json" {q(remote + '/analysis/SUCCESS.json')}
"""
    raise ValueError(stage)


def build_job(args, stage):
    if stage not in ("controller", *STAGES):
        raise ValueError(stage)
    if stage == "controller":
        machine, cpu, memory, disk, seconds = "e2-small", 1000, 1500, 30, 72 * 3600
    else:
        machine, cpu, memory, disk = "n2-standard-8", 8000, 30000, 200
        seconds = dict(smoke=7200, train=36 * 3600, aggregate=14400)[stage]
        # Experiment 3 changes only smoke/training allocation, not learner
        # threads, the controller, aggregation, timeouts or earlier experiments.
        if stage in ("smoke", "train"):
            resources = settings(args).get("worker_resources", {})
            machine = resources.get("machine_type", machine)
            cpu = resources.get("cpu_milli", cpu)
            memory = resources.get("memory_mib", memory)
    count = 3 if stage == "train" else 1
    return dict(taskGroups=[dict(taskSpec=dict(runnables=[dict(script=dict(text=script(args, stage)))],
                computeResource=dict(cpuMilli=cpu, memoryMib=memory), maxRetryCount=0,
                maxRunDuration=f"{seconds}s"), taskCount=count, parallelism=count, taskCountPerNode=1)],
                allocationPolicy=dict(serviceAccount=dict(email=args.service_account),
                instances=[dict(policy=dict(machineType=machine, provisioningModel="STANDARD",
                bootDisk=dict(sizeGb=disk, type="pd-balanced")))]),
                logsPolicy=dict(destination="CLOUD_LOGGING"),
                labels=dict(experiment=f"fhp-vr-exp{settings(args)['number']}-24h", stage=stage))


def cloud(args, *command, capture=False, check=True):
    return subprocess.run(["gcloud", *command, "--project", args.project], check=check,
                          text=True, capture_output=capture)


def job_state(args, name):
    result = cloud(args, "batch", "jobs", "describe", name, "--location", args.region,
                   "--format=value(status.state)", capture=True, check=False)
    if result.returncode == 0:
        return result.stdout.strip()
    if "NOT_FOUND" in result.stderr or "was not found" in result.stderr:
        return None
    raise RuntimeError(result.stderr or "Unable to query Batch job")


def submit(args, stage, *, suffix=""):
    name = f"{args.run_id}-{stage}{suffix}"
    with tempfile.TemporaryDirectory(prefix="fhp-vr-exp1-job-") as temporary:
        path = Path(temporary) / "job.json"
        path.write_text(json.dumps(build_job(args, stage), indent=2))
        try:
            cloud(args, "batch", "jobs", "submit", name, "--location", args.region, "--config", str(path))
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                "Batch submission failed. No later stages were launched. Check quotas and IAM: "
                "the controller service account needs batch.jobsEditor on the project and "
                "iam.serviceAccountUser on the worker service account (itself).") from exc
    return name


def wait(args, name):
    previous = None
    while True:
        state = job_state(args, name)
        if state != previous:
            print(f"{name}: {state}", flush=True)
            previous = state
        if state == "SUCCEEDED":
            return
        if state in {None, "FAILED", "DELETION_IN_PROGRESS"}:
            raise RuntimeError(f"{name}: {state}; later stages were not launched")
        time.sleep(30)


def orchestrate(args):
    cloud(args, "batch", "jobs", "list", "--location", args.region, "--limit=1")
    for stage in STAGES:
        name = f"{args.run_id}-{stage}"
        state = job_state(args, name)
        if state is None:
            submit(args, stage)
        elif state in {"FAILED", "DELETION_IN_PROGRESS"}:
            raise RuntimeError(f"{name} already {state}; refusing an automatic training restart")
        wait(args, name)


def validate(args):
    if not all((args.project, args.region, args.bucket, args.service_account, args.repo_ref, args.run_id)):
        raise ValueError("Set PROJECT_ID, REGION, BUCKET, SA_EMAIL, REPO_REF and RUN_ID")
    if not re.fullmatch(r"[a-z][a-z0-9-]{1,34}", args.run_id):
        raise ValueError("RUN_ID must be 2..35 lowercase Batch-compatible characters")
    if not re.fullmatch(r"[0-9a-f]{40}", args.repo_ref):
        raise ValueError("REPO_REF must be a full pushed commit SHA")
    args.bucket = args.bucket.rstrip("/")
    if not args.bucket.startswith("gs://"):
        args.bucket = "gs://" + args.bucket
    if not re.fullmatch(r"gs://[a-z0-9][a-z0-9._-]+", args.bucket):
        raise ValueError("BUCKET must be a bucket name or gs://bucket, not an object prefix")


def preflight(args):
    cloud(args, "iam", "service-accounts", "describe", args.service_account)
    cloud(args, "storage", "buckets", "describe", args.bucket)
    # A fresh namespace prevents overwrite even if a previous Batch job was deleted.
    result = cloud(args, "storage", "ls", f"{args.bucket}/{args.run_id}/**", capture=True, check=False)
    if result.returncode == 0:
        raise ValueError("RUN_ID already has stored outputs; choose a new RUN_ID")
    if "matched no objects" not in result.stderr and "matched no objects" not in result.stdout:
        raise RuntimeError(result.stderr or "Unable to check destination namespace")


def main(*, experiment=DEFAULT_EXPERIMENT):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "orchestrate", "status", "dry-run", "aggregate-only", "smoke-only"))
    for name, env in (("project", "PROJECT_ID"), ("region", "REGION"), ("bucket", "BUCKET"),
                      ("service-account", "SA_EMAIL"), ("repo-ref", "REPO_REF"), ("run-id", "RUN_ID")):
        parser.add_argument("--" + name, default=os.environ.get(env))
    args = parser.parse_args()
    args.experiment = experiment
    try:
        validate(args)
    except ValueError as exc:
        parser.error(str(exc))
    if args.action == "dry-run":
        print(json.dumps({stage: build_job(args, stage) for stage in ("controller", *STAGES)}, indent=2))
    elif args.action == "status":
        cloud(args, "batch", "jobs", "list", "--location", args.region,
              "--filter", f"name:{args.run_id}", "--format=table(name.basename(),status.state)")
    elif args.action in ("run", "smoke-only"):
        preflight(args)
        name = submit(args, "controller" if args.action == "run" else "smoke")
        print(f"Submitted {name}; the laptop may disconnect. Outputs: {args.bucket}/{args.run_id}")
    elif args.action == "aggregate-only":
        # Recovery repeats analysis, never training; the aggregator validates every seed.
        cloud(args, "iam", "service-accounts", "describe", args.service_account)
        print(submit(args, "aggregate", suffix="-" + time.strftime("%H%M%S", time.gmtime())))
    else:
        orchestrate(args)


if __name__ == "__main__":
    main()
