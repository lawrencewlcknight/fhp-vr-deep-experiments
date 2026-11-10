#!/usr/bin/env python3
"""One bounded VM: three concurrent refit workers, then eight scoring workers."""
import argparse
import json
from pathlib import Path
import re
import shlex


def build_job(run_id,bucket,service_account,repo_ref,*,profile_only=False,resume=False):
    if not re.fullmatch('[a-z][a-z0-9-]{0,62}',run_id):raise ValueError('Invalid run ID')
    if not re.fullmatch('[0-9a-f]{40}',repo_ref):raise ValueError('Use an exact pushed commit')
    if not re.fullmatch(r'gs://[a-z0-9][a-z0-9._-]+',bucket):raise ValueError('Bucket root required')
    if not re.fullmatch(r'[A-Za-z0-9._-]+@[A-Za-z0-9.-]+\.gserviceaccount\.com',service_account):raise ValueError('Service account required')
    q=shlex.quote;module='experiments.fhp.exp10_vr_deep_policy_extraction.run'
    script=f'''#!/bin/bash
set -euo pipefail
WORK=/workspace/vr10
mkdir -p "$WORK"
export DEBIAN_FRONTEND=noninteractive
apt_with_retry() {{
  for attempt in {{1..30}}; do
    if apt-get -o DPkg::Lock::Timeout=10 "$@"; then return 0; fi
    sleep 10
  done
  return 1
}}
apt_with_retry update -qq
apt_with_retry install -y -qq git curl ca-certificates python3-dev build-essential util-linux
export PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
REPO="$WORK/repo"
INPUT="$WORK/inputs"
OUTPUT="$WORK/output"
REMOTE={q(bucket+'/'+run_id)}
mkdir -p "$OUTPUT"
git clone --filter=blob:none https://github.com/lawrencewlcknight/fhp-vr-deep-experiments.git "$REPO"
git -C "$REPO" checkout --detach {q(repo_ref)}
[[ "$(git -C "$REPO" rev-parse HEAD)" == {q(repo_ref)} ]]
curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/tmp/uv-bin UV_NO_MODIFY_PATH=1 sh
export PATH="/tmp/uv-bin:$PATH" UV_PYTHON_INSTALL_DIR=/tmp/uv-python
uv python install 3.11.16
uv venv --python 3.11.16 --seed "$WORK/venv"
source "$WORK/venv/bin/activate"
cd "$REPO"
python -m pip install --retries 3 -r requirements.txt
publish() (
  flock 9
  # Capture completed fit receipts before copying data, so a fit finishing
  # during rsync cannot advertise files omitted from that rsync's scan.
  markers=$(mktemp -d "$WORK/markers.XXXXXX")
  for marker in "$OUTPUT"/fits/*/SUCCESS.json; do
    if [[ -f "$marker" ]]; then
      seed=$(basename "$(dirname "$marker")")
      mkdir -p "$markers/$seed"
      cp "$marker" "$markers/$seed/SUCCESS.json"
    fi
  done
  timeout 600 gcloud storage rsync --recursive --exclude '(^|/)(fixtures)(/|$)|SUCCESS[.]json$|[.]tmp$' "$OUTPUT" "$REMOTE" || return 1
  for marker in "$markers"/*/SUCCESS.json; do
    [[ ! -f "$marker" ]] || timeout 60 gcloud storage cp "$marker" "$REMOTE/fits/$(basename "$(dirname "$marker")")/SUCCESS.json" || return 1
  done
) 9>"$WORK/publish.lock"
SYNC_PID=""
cleanup() {{
  result=$?
  trap - EXIT
  if [[ -n "$SYNC_PID" ]]; then kill "$SYNC_PID" 2>/dev/null || true; wait "$SYNC_PID" 2>/dev/null || true; fi
  publish || true
  exit "$result"
}}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
# Resume only the small outputs. Inputs are allowlisted by pinned checksums.
if [[ {int(resume)} == 1 ]]; then gcloud storage rsync --recursive "$REMOTE" "$OUTPUT"; fi
python -m pip freeze > "$OUTPUT/pip_freeze.txt"
git rev-parse HEAD > "$OUTPUT/repository_commit.txt"
(while sleep 300; do publish || echo 'Periodic upload failed; will retry' >&2; done) &
SYNC_PID=$!
python -m unittest discover -s tests -p test_exp10_policy_extraction.py
python -m {module} smoke --output "$WORK/smoke" --workers 2
python -m {module} fetch --input "$INPUT"
python -m {module} fit --input "$INPUT" --output "$OUTPUT" --workers 3
publish
python -m {module} profile-matches --input "$INPUT" --output "$OUTPUT" --workers 8
'''
    if not profile_only:
        script+=f'''python -m {module} evaluate --input "$INPUT" --output "$OUTPUT" --workers 8
python -m {module} aggregate --output "$OUTPUT"
publish
gcloud storage cp "$OUTPUT/analysis/SUCCESS.json" "$REMOTE/analysis/SUCCESS.json"
'''
    return dict(taskGroups=[dict(taskCount=1,parallelism=1,taskSpec=dict(
        runnables=[dict(script=dict(text=script),timeout='82800s')],
        computeResource=dict(cpuMilli=16000,memoryMib=62000),maxRunDuration='86400s',maxRetryCount=0))],
        allocationPolicy=dict(instances=[dict(policy=dict(machineType='n2-standard-16',provisioningModel='STANDARD',
            bootDisk=dict(type='pd-balanced',sizeGb=150)))],serviceAccount=dict(email=service_account,scopes=['https://www.googleapis.com/auth/cloud-platform'])),
        logsPolicy=dict(destination='CLOUD_LOGGING'),labels=dict(experiment='vr10-policy-extraction',run=run_id))


def main():
    p=argparse.ArgumentParser()
    for k in ['run-id','bucket','service-account','repo-ref']:p.add_argument('--'+k,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--profile-only',action='store_true');p.add_argument('--resume',action='store_true');a=p.parse_args()
    job=build_job(a.run_id,a.bucket,a.service_account,a.repo_ref,profile_only=a.profile_only,resume=a.resume)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(job,indent=2)+'\n')


if __name__=='__main__':main()
