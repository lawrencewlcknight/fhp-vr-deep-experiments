#!/usr/bin/env python3
"""Submit the resumable final 48-hour three-algorithm FHP evaluation."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile

VR_REPO = "https://github.com/lawrencewlcknight/fhp-vr-deep-experiments.git"
SD_REPO = "https://github.com/lawrencewlcknight/fhp-deep-cfr-experiments.git"
UCV_REPO = "https://github.com/lawrencewlcknight/fhp-ucv-escher-experiments.git"
DEFAULT_SD_REF = "bb689252d6b322adb3de6102d095a49c3ed87250"
DEFAULT_UCV_REF = "cf04f1a710e7b947da9fc1c41ae0159c420c72eb"
MODULE = "experiments.fhp.retrospective_threeway_48h_evaluation.run"
SD_ALGORITHM = "parallel_structured_uniform_sd_cfr_48h"
UCV_ALGORITHM = "hand_board_cached_parallel_ucv_escher"
VR_ALGORITHM = "lossless_vr_deep_pdcfr_plus_ray8_48h"


def q(value):
    return shlex.quote(str(value))


def validate(args):
    for key in ("run_id", "sd_run_id", "ucv_run_id", "ucv_source10_run_id", "vr_run_id"):
        if not re.fullmatch(r"[a-z][a-z0-9-]{0,33}[a-z0-9]", getattr(args, key) or ""):
            raise ValueError(f"{key} must be a 2--35 character lowercase Batch-compatible ID")
    if args.run_id in {args.sd_run_id, args.ucv_run_id, args.ucv_source10_run_id, args.vr_run_id}:
        raise ValueError("Evaluation output must not reuse a source run ID")
    expected_runs = dict(sd_run_id="sdcfr6-48h-20261002-161544",
                         ucv_run_id="exp16-feat48-20261004-182051",
                         ucv_source10_run_id="exp10-features-20261001-161740",
                         vr_run_id="vr6-ray48-20261003-221425")
    if any(getattr(args, key) != value for key, value in expected_runs.items()):
        raise ValueError("This prespecified evaluator accepts only the frozen 48-hour source run IDs")
    for key in ("bucket", "sd_bucket", "ucv_bucket", "vr_bucket"):
        if not re.fullmatch(r"gs://[a-z0-9][a-z0-9._-]+", getattr(args, key)):
            raise ValueError(f"{key} must be a bucket URI without a subdirectory")
    for key in ("repo_ref", "sd_ref", "ucv_ref"):
        if not re.fullmatch(r"[0-9a-f]{40}", getattr(args, key) or ""):
            raise ValueError(f"{key} must be a full pushed commit SHA")
    if not re.fullmatch(r"[^@\s]+@[^@\s]+[.]iam[.]gserviceaccount[.]com", args.service_account or ""):
        raise ValueError("Set the Batch runner service-account email")


def source_uris(args):
    return dict(sd=f"{args.sd_bucket}/{args.sd_run_id}",
                ucv=f"{args.ucv_bucket}/{args.ucv_run_id}",
                ucv_source10=f"{args.ucv_bucket}/{args.ucv_source10_run_id}",
                vr=f"{args.vr_bucket}/{args.vr_run_id}")


def worker_script(args, *, smoke):
    sources = source_uris(args)
    resume = "1" if args.resume else "0"
    smoke_value = "1" if smoke else "0"
    return f'''#!/usr/bin/env bash
set -Eeuo pipefail
export DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/threeway-mpl UV_CACHE_DIR=/tmp/uv-cache UV_PYTHON_INSTALL_DIR=/tmp/uv-python
WORK=/workspace/fhp-threeway-48h
VR_REPOSITORY="$WORK/vr-repository"
SD_REPOSITORY="$WORK/sd-repository"
UCV_REPOSITORY="$WORK/ucv-repository"
INPUT="$WORK/inputs"
OUTPUT="$WORK/output"
DESTINATION={q(args.bucket + '/' + args.run_id)}
mkdir -p "$INPUT" "$OUTPUT"
SYNC_PID=""
sync_outputs() {{
  flock -w 180 9 || return 1
  timeout 300 gcloud storage rsync --recursive --exclude='.*[.]tmp$|(^|/)SUCCESS[.]json$' "$OUTPUT" "$DESTINATION/analysis"
  for marker in "$OUTPUT"/{{smoke,main,lbr}}/SUCCESS.json; do
    [[ -f "$marker" ]] || continue
    timeout 60 gcloud storage cp "$marker" "$DESTINATION/analysis/${{marker#"$OUTPUT/"}}"
  done
}}
cleanup() {{
  result=$?
  trap - EXIT
  if [[ -n "$SYNC_PID" ]]; then kill "$SYNC_PID" 2>/dev/null || true; wait "$SYNC_PID" 2>/dev/null || true; fi
  printf '{{"exit_code":%s}}\n' "$result" > "$OUTPUT/process_exit.json"
  (sync_outputs) 9>"$WORK/upload.lock" || true
  exit "$result"
}}
trap cleanup EXIT
if [[ {resume} -eq 1 ]]; then
  gcloud storage rsync --recursive "$DESTINATION/analysis" "$OUTPUT"
fi
apt-get update -qq
apt-get install -y -qq git curl ca-certificates util-linux
git clone {q(VR_REPO)} "$VR_REPOSITORY"
git -C "$VR_REPOSITORY" checkout --detach {q(args.repo_ref)}
git clone {q(SD_REPO)} "$SD_REPOSITORY"
git -C "$SD_REPOSITORY" checkout --detach {q(args.sd_ref)}
git clone {q(UCV_REPO)} "$UCV_REPOSITORY"
git -C "$UCV_REPOSITORY" checkout --detach {q(args.ucv_ref)}
curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/tmp/uv-bin UV_NO_MODIFY_PATH=1 sh
export PATH="/tmp/uv-bin:$PATH"
uv python install 3.11.17
uv venv --python 3.11.17 --seed "$WORK/venv"
uv pip install --python "$WORK/venv/bin/python" -r "$VR_REPOSITORY/requirements.txt"
"$WORK/venv/bin/python" -m pip check
export PYTHONPATH="$VR_REPOSITORY:$SD_REPOSITORY:$UCV_REPOSITORY"
cd "$VR_REPOSITORY"
"$WORK/venv/bin/python" -m pip freeze > "$OUTPUT/pip_freeze.txt"
git -C "$VR_REPOSITORY" rev-parse HEAD > "$OUTPUT/vr_evaluator_commit.txt"
git -C "$SD_REPOSITORY" rev-parse HEAD > "$OUTPUT/sd_loader_commit.txt"
git -C "$UCV_REPOSITORY" rev-parse HEAD > "$OUTPUT/ucv_loader_commit.txt"

SD_SOURCE={q(sources['sd'])}
UCV_SOURCE={q(sources['ucv'])}
UCV_SOURCE10={q(sources['ucv_source10'])}
VR_SOURCE={q(sources['vr'])}
for seed in 0 1 2; do
  SD_WORKER="task_00${{seed}}_{SD_ALGORITHM}_seed_${{seed}}"
  UCV_WORKER="task_00${{seed}}_{UCV_ALGORITHM}_seed_${{seed}}"
  VR_WORKER="task_00${{seed}}_{VR_ALGORITHM}_seed_${{seed}}"
  mkdir -p "$INPUT/sd/workers/$SD_WORKER/archive" \
           "$INPUT/ucv/workers/$UCV_WORKER/checkpoints" \
           "$INPUT/ucv_source10/workers/$UCV_WORKER" \
           "$INPUT/vr/workers/$VR_WORKER/checkpoints"
  for metadata in run_manifest.json checkpoint_manifest.json SUCCESS.json; do
    gcloud storage cp "$SD_SOURCE/workers/$SD_WORKER/$metadata" "$INPUT/sd/workers/$SD_WORKER/$metadata"
    gcloud storage cp "$UCV_SOURCE/workers/$UCV_WORKER/$metadata" "$INPUT/ucv/workers/$UCV_WORKER/$metadata"
    gcloud storage cp "$UCV_SOURCE10/workers/$UCV_WORKER/$metadata" "$INPUT/ucv_source10/workers/$UCV_WORKER/$metadata"
    gcloud storage cp "$VR_SOURCE/workers/$VR_WORKER/$metadata" "$INPUT/vr/workers/$VR_WORKER/$metadata"
  done
  gcloud storage cp "$UCV_SOURCE/workers/$UCV_WORKER/runtime_manifest.json" "$INPUT/ucv/workers/$UCV_WORKER/runtime_manifest.json"
  gcloud storage cp "$UCV_SOURCE/workers/$UCV_WORKER/continuation_source.json" "$INPUT/ucv/workers/$UCV_WORKER/continuation_source.json"
  gcloud storage cp "$VR_SOURCE/workers/$VR_WORKER/summary.json" "$INPUT/vr/workers/$VR_WORKER/summary.json"
  gcloud storage rsync --recursive "$SD_SOURCE/workers/$SD_WORKER/archive" "$INPUT/sd/workers/$SD_WORKER/archive"
  gcloud storage cp "$UCV_SOURCE/workers/$UCV_WORKER/checkpoints/hand_board_cached_parallel_ucv_escher_seed_${{seed}}_time_48h.pkl" \
    "$INPUT/ucv/workers/$UCV_WORKER/checkpoints/hand_board_cached_parallel_ucv_escher_seed_${{seed}}_time_48h.pkl"
  gcloud storage cp "$VR_SOURCE/workers/$VR_WORKER/checkpoints/lossless_vr_deep_pdcfr_plus_ray8_48h_seed_${{seed}}_time_48h.pt" \
    "$INPUT/vr/workers/$VR_WORKER/checkpoints/lossless_vr_deep_pdcfr_plus_ray8_48h_seed_${{seed}}_time_48h.pt"
done
(while sleep 300; do (sync_outputs) 9>"$WORK/upload.lock" || echo "Periodic upload failed; will retry" >&2; done) &
SYNC_PID=$!
ARGS=(--source-root "$INPUT" --sd-repo "$SD_REPOSITORY" --ucv-repo "$UCV_REPOSITORY"
      --output "$OUTPUT" --workers 16
      --source-uri "sd=$SD_SOURCE" --source-uri "ucv=$UCV_SOURCE"
      --source-uri "ucv_source10=$UCV_SOURCE10" --source-uri "vr=$VR_SOURCE")
"$WORK/venv/bin/python" -m {MODULE} smoke "${{ARGS[@]}}" --max-hours 1
(sync_outputs) 9>"$WORK/upload.lock"
if [[ {smoke_value} -eq 1 ]]; then exit 0; fi
"$WORK/venv/bin/python" -m {MODULE} main "${{ARGS[@]}}" --max-hours 22
(sync_outputs) 9>"$WORK/upload.lock"
if [[ ! -f "$OUTPUT/main/SUCCESS.json" ]]; then
  echo "Main timing gate deferred production scoring; LBR will not start."
  exit 0
fi
"$WORK/venv/bin/python" -m {MODULE} lbr "${{ARGS[@]}}" --max-hours 10
'''


def finalizer_script():
    return '''#!/usr/bin/env bash
set -Eeuo pipefail
WORK=/workspace/fhp-threeway-48h
OUTPUT="$WORK/output"
mkdir -p "$OUTPUT"
df -h > "$OUTPUT/disk_final.txt" || true
free -m > "$OUTPUT/memory_final.txt" || true
flock -w 180 9
timeout 600 gcloud storage rsync --recursive --exclude='.*[.]tmp$|(^|/)SUCCESS[.]json$' "$OUTPUT" "$DESTINATION/analysis"
for marker in "$OUTPUT"/{smoke,main,lbr}/SUCCESS.json; do
  [[ -f "$marker" ]] || continue
  timeout 60 gcloud storage cp "$marker" "$DESTINATION/analysis/${marker#"$OUTPUT/"}"
done
'''


def job_config(args, *, smoke=False):
    seconds = 7200 if smoke else 36 * 3600
    environment = dict(variables={"DESTINATION": f"{args.bucket}/{args.run_id}"})
    return dict(taskGroups=[dict(taskCount=1, parallelism=1, taskSpec=dict(
        runnables=[dict(script=dict(text=worker_script(args, smoke=smoke)), timeout=f"{seconds-1200}s"),
                   dict(alwaysRun=True, script=dict(text=finalizer_script()), timeout="900s")],
        computeResource=dict(cpuMilli=16000, memoryMib=62000), maxRetryCount=0,
        maxRunDuration=f"{seconds}s", environment=environment))],
        allocationPolicy=dict(instances=[dict(policy=dict(machineType="n2-standard-16",
            provisioningModel="STANDARD", bootDisk=dict(sizeGb=200, type="pd-balanced")))],
            serviceAccount=dict(email=args.service_account,
                scopes=["https://www.googleapis.com/auth/cloud-platform"])),
        logsPolicy=dict(destination="CLOUD_LOGGING"),
        labels=dict(workload="fhp-threeway-48h", run=args.run_id,
                    stage="smoke" if smoke else "evaluation"))


def cloud(args, *command, check=True):
    return subprocess.run(["gcloud", *command, "--project", args.project], text=True,
                          capture_output=True, check=check)


def object_exists(args, uri):
    result = cloud(args, "storage", "ls", uri, check=False)
    if result.returncode == 0:
        return bool(result.stdout.strip())
    if "matched no objects" in result.stderr:
        return False
    raise RuntimeError(result.stderr.strip())


def check_sources(args):
    sources = source_uris(args)
    specs = (("sd", SD_ALGORITHM, False, False), ("ucv", UCV_ALGORITHM, True, True),
             ("vr", VR_ALGORITHM, False, False))
    for source, algorithm, runtime, continuation in specs:
        for seed in range(3):
            prefix = f"{sources[source]}/workers/task_{seed:03d}_{algorithm}_seed_{seed}"
            names = ["run_manifest.json", "checkpoint_manifest.json", "SUCCESS.json"]
            if runtime:
                names.append("runtime_manifest.json")
            if continuation:
                names.append("continuation_source.json")
            if source == "vr":
                names.append("summary.json")
            for name in names:
                if not object_exists(args, f"{prefix}/{name}"):
                    raise ValueError(f"Missing source object: {prefix}/{name}")
            if object_exists(args, f"{prefix}/FAILURE.json"):
                raise ValueError(f"Failed source worker: {prefix}")
            final_policy = ({
                "sd": "archive/time_48h.json",
                "ucv": f"checkpoints/hand_board_cached_parallel_ucv_escher_seed_{seed}_time_48h.pkl",
                "vr": f"checkpoints/lossless_vr_deep_pdcfr_plus_ray8_48h_seed_{seed}_time_48h.pt",
            })[source]
            if not object_exists(args, f"{prefix}/{final_policy}"):
                raise ValueError(f"Missing final source policy: {prefix}/{final_policy}")
        if source == "ucv":
            for seed in range(3):
                prefix = f"{sources['ucv_source10']}/workers/task_{seed:03d}_{UCV_ALGORITHM}_seed_{seed}"
                for name in ("run_manifest.json", "checkpoint_manifest.json", "SUCCESS.json"):
                    if not object_exists(args, f"{prefix}/{name}"):
                        raise ValueError(f"Missing UCV lineage object: {prefix}/{name}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "smoke-only", "resume", "status", "dry-run", "check-sources"))
    parser.add_argument("--project", default=os.environ.get("PROJECT_ID"), required=not os.environ.get("PROJECT_ID"))
    parser.add_argument("--region", default=os.environ.get("REGION", "europe-west1"))
    parser.add_argument("--run-id", default=os.environ.get("RUN_ID"), required=not os.environ.get("RUN_ID"))
    parser.add_argument("--bucket", default=os.environ.get("BUCKET", "gs://clever-overview-399515-fhp-vr-deep-results"))
    parser.add_argument("--sd-bucket", default=os.environ.get("SD_BUCKET", "gs://clever-overview-399515-fhp-deep-cfr-results"))
    parser.add_argument("--ucv-bucket", default=os.environ.get("UCV_BUCKET", "gs://clever-overview-399515-fhp-escher-results"))
    parser.add_argument("--vr-bucket", default=os.environ.get("VR_BUCKET", "gs://clever-overview-399515-fhp-vr-deep-results"))
    parser.add_argument("--sd-run-id", default=os.environ.get("SD_RUN_ID", "sdcfr6-48h-20261002-161544"))
    parser.add_argument("--ucv-run-id", default=os.environ.get("UCV_RUN_ID", "exp16-feat48-20261004-182051"))
    parser.add_argument("--ucv-source10-run-id", default=os.environ.get("UCV_SOURCE10_RUN_ID", "exp10-features-20261001-161740"))
    parser.add_argument("--vr-run-id", default=os.environ.get("VR_RUN_ID", "vr6-ray48-20261003-221425"))
    parser.add_argument("--repo-ref", default=os.environ.get("REPO_REF", ""))
    parser.add_argument("--sd-ref", default=os.environ.get("SD_REPO_REF", DEFAULT_SD_REF))
    parser.add_argument("--ucv-ref", default=os.environ.get("UCV_REPO_REF", DEFAULT_UCV_REF))
    parser.add_argument("--service-account", default=os.environ.get("SA_EMAIL",
        "fhp-vr-deep-runner@clever-overview-399515.iam.gserviceaccount.com"))
    parser.add_argument("--output", type=Path, help="JSON output path for dry-run")
    args = parser.parse_args(argv)
    for name in ("bucket", "sd_bucket", "ucv_bucket", "vr_bucket"):
        setattr(args, name, "gs://" + getattr(args, name).removeprefix("gs://").rstrip("/"))
    args.resume = args.action == "resume"
    validate(args)
    return parser, args


def main(argv=None):
    parser, args = parse_args(argv)
    if args.action == "status":
        print(cloud(args, "batch", "jobs", "list", "--location", args.region,
                    "--filter", f"labels.workload=fhp-threeway-48h AND labels.run={args.run_id}").stdout)
        return
    if args.action == "dry-run":
        if not args.output:
            parser.error("dry-run requires --output")
        args.output.write_text(json.dumps(job_config(args), indent=2) + "\n")
        print(f"Wrote {args.output}; no cloud calls or jobs submitted")
        return
    check_sources(args)
    if args.action == "check-sources":
        print("All source worker metadata is present.")
        return
    cloud(args, "iam", "service-accounts", "describe", args.service_account)
    destination = f"{args.bucket}/{args.run_id}"
    occupied = object_exists(args, destination + "/**")
    if occupied and not args.resume:
        raise ValueError("Output prefix exists; choose a new RUN_ID or use resume")
    if args.resume and not object_exists(args, destination + "/analysis/evaluation_manifest.json"):
        raise ValueError("No compatible evaluation manifest exists to resume")
    active = cloud(args, "batch", "jobs", "list", "--location", args.region,
                   "--filter", f"labels.workload=fhp-threeway-48h AND labels.run={args.run_id}", "--format=json")
    if any(job["status"]["state"] not in {"SUCCEEDED", "FAILED"} for job in json.loads(active.stdout)):
        raise ValueError("An active job already uses this output prefix")
    smoke = args.action == "smoke-only"
    suffix = "-smoke" if smoke else ("-r" + datetime.now(timezone.utc).strftime("%H%M%S") if args.resume else "")
    job_name = args.run_id + suffix
    with tempfile.TemporaryDirectory(prefix="threeway-48h-submit-") as temporary:
        config = Path(temporary) / "job.json"
        config.write_text(json.dumps(job_config(args, smoke=smoke)))
        result = cloud(args, "batch", "jobs", "submit", job_name, "--location", args.region,
                       "--config", str(config))
    print(result.stdout)
    print(f"Submitted {job_name}; laptop may disconnect. Outputs: {destination}/analysis/")


if __name__ == "__main__":
    main()
