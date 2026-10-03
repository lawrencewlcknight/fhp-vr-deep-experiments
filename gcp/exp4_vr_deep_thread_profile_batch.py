#!/usr/bin/env python3
"""One on-demand VM, serial thread trials, bounded profiling and no evaluation."""

import argparse
import json
import os
from pathlib import Path
import tempfile

try:
    from . import exp1_vr_deep_pdcfr_24h_batch as shared
except ImportError:
    import exp1_vr_deep_pdcfr_24h_batch as shared

MODULE = "experiments.fhp.exp4_vr_deep_thread_profile.run"


def script(args, smoke_only=False):
    args.experiment = dict(number=4)
    remote = f"{args.bucket}/{args.run_id}"
    text = shared.bootstrap(args) + f'''
REMOTE={shared.q(remote)}
mkdir -p "$WORK/fixtures" "$OUT/diagnostics"
python -m fhp_vr_deep.batch_diagnostics monitor --output "$OUT/diagnostics/resources.jsonl" \\
  --interval-seconds 30 --cloud-log-every 10 &
MONITOR_PID=$!
finish() {{
  code=$?
  trap - EXIT
  kill "$MONITOR_PID" >/dev/null 2>&1 || true
  wait "$MONITOR_PID" >/dev/null 2>&1 || true
  python -m fhp_vr_deep.batch_diagnostics finalize \\
    --snapshots "$OUT/diagnostics/resources.jsonl" --output "$OUT/diagnostics/summary.json" \\
    --status-output "$OUT/diagnostics/status.json" --failure-root "$OUT" --exit-code "$code" \\
    --requested-memory-mib 62000 --job-name {shared.q(args.run_id + '-profile')} --bucket-destination "$REMOTE" || true
  gcloud storage rsync --recursive --exclude='(^|/)SUCCESS[.]json$|[.]tmp$' "$OUT" "$REMOTE" || code=1
  if [[ $code == 0 ]]; then
    for stage in smoke profile; do
      if [[ -f "$OUT/$stage/SUCCESS.json" ]]; then
        gcloud storage cp "$OUT/$stage/SUCCESS.json" "$REMOTE/$stage/SUCCESS.json" || code=1
      fi
    done
  fi
  exit "$code"
}}
trap finish EXIT
trap 'exit 143' TERM
python -m pip install -r requirements-dev.txt
python -m pytest -q -p no:cacheprovider tests/test_exp4_vr_deep_thread_profile.py
python -m {MODULE} smoke --output-root "$OUT/smoke" --scratch-root "$WORK/fixtures" \\
  --remote-uri "$REMOTE/smoke"
'''
    if not smoke_only:
        text += f'''python -m {MODULE} run --output-root "$OUT/profile" --scratch-root "$WORK/fixtures" \\
  --remote-uri "$REMOTE/profile"
'''
    return text


def build_job(args, smoke_only=False):
    return dict(taskGroups=[dict(taskSpec=dict(runnables=[dict(script=dict(text=script(args, smoke_only)))],
                computeResource=dict(cpuMilli=16000, memoryMib=62000), maxRetryCount=0,
                maxRunDuration="7200s" if smoke_only else "21600s"),
                taskCount=1, parallelism=1, taskCountPerNode=1)],
                allocationPolicy=dict(serviceAccount=dict(email=args.service_account),
                instances=[dict(policy=dict(machineType="n2-standard-16", provisioningModel="STANDARD",
                bootDisk=dict(sizeGb=200, type="pd-balanced")))]),
                logsPolicy=dict(destination="CLOUD_LOGGING"),
                labels=dict(experiment="fhp-vr-exp4-thread-profile", stage="smoke" if smoke_only else "profile"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "smoke-only", "dry-run", "status"))
    for name, env in (("project", "PROJECT_ID"), ("region", "REGION"), ("bucket", "BUCKET"),
                      ("service-account", "SA_EMAIL"), ("repo-ref", "REPO_REF"), ("run-id", "RUN_ID")):
        parser.add_argument("--" + name, default=os.environ.get(env))
    args = parser.parse_args()
    try:
        shared.validate(args)
    except ValueError as exc:
        parser.error(str(exc))
    if args.action == "dry-run":
        print(json.dumps(build_job(args), indent=2))
    elif args.action == "status":
        shared.cloud(args, "batch", "jobs", "list", "--location", args.region,
                     "--filter", f"name:{args.run_id}", "--format=table(name.basename(),status.state)")
    else:
        shared.preflight(args)
        name = args.run_id + ("-smoke" if args.action == "smoke-only" else "-profile")
        with tempfile.TemporaryDirectory(prefix="vr-exp4-batch-") as temporary:
            path = Path(temporary) / "job.json"
            path.write_text(json.dumps(build_job(args, args.action == "smoke-only"), indent=2))
            shared.cloud(args, "batch", "jobs", "submit", name, "--location", args.region, "--config", str(path))
        print(f"Submitted {name}; laptop may disconnect. Outputs: {args.bucket}/{args.run_id}")


if __name__ == "__main__":
    main()
