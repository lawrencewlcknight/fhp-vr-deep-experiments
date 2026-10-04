#!/usr/bin/env bash
set -Eeuo pipefail
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin${PATH:+:$PATH}"
export DEBIAN_FRONTEND=noninteractive
WORK=/workspace/vr-eval123
OUT="$WORK/output"
mkdir -p "$OUT" "$WORK/source" "$WORK/inputs"
# Restore before starting background uploads: never race a download with a sync.
if [[ "$VR_EVAL_RESUME" == 1 ]]; then
  timeout 600 gcloud storage rsync --recursive "$VR_EVAL_DESTINATION/analysis" "$OUT"
fi
exec > >(tee -a "$OUT/bootstrap.log") 2>&1
STAGE=bootstrap
MONITOR_PID=""
trap 'printf "ERROR stage=%s line=%s exit_code=%s\n" "$STAGE" "$LINENO" "$?" >&2' ERR
sync_outputs() {
  (
    flock -w 180 9 || exit 1
    timeout 120 gcloud storage rsync --recursive \
      --exclude='.*[.]tmp$|(^|/)SUCCESS[.]json$' "$OUT" "$VR_EVAL_DESTINATION/analysis" || exit 1
    for marker in "$OUT"/{smoke,main,lbr}/SUCCESS.json; do
      [[ -f "$marker" ]] || continue
      timeout 60 gcloud storage cp "$marker" "$VR_EVAL_DESTINATION/analysis/${marker#"$OUT/"}" || exit 1
    done
  ) 9>"$WORK/upload.lock"
}
cleanup() {
  code=$?
  set +e
  kill "$SYNC_PID" ${MONITOR_PID:+"$MONITOR_PID"} 2>/dev/null
  wait "$SYNC_PID" ${MONITOR_PID:+"$MONITOR_PID"} 2>/dev/null
  printf '{"exit_code":%s,"stage":"%s"}\n' "$code" "$STAGE" > "$OUT/process_exit.json"
  if [[ -x "$WORK/venv/bin/python" ]]; then
    "$WORK/venv/bin/python" -m fhp_vr_deep.batch_diagnostics finalize \
      --snapshots "$OUT/resources.jsonl" --output "$OUT/resource_summary.json" \
      --status-output "$OUT/resource_status.json" --failure-root "$OUT" --exit-code "$code" \
      --requested-memory-mib 62000 --job-name "${BATCH_JOB_UID:-vr-eval123}" \
      --bucket-destination "$VR_EVAL_DESTINATION/analysis" || true
  fi
  sync_outputs || echo 'Exit upload failed; final Batch runnable will retry.'
  exit "$code"
}
while sleep 300; do sync_outputs || echo 'WARNING: periodic upload failed'; done &
SYNC_PID=$!
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
timeout 180 gcloud storage cp "$VR_EVAL_DESTINATION/inputs/source.tar.gz" "$WORK/source.tar.gz"
printf '%s  %s\n' "$VR_EVAL_BUNDLE_SHA256" "$WORK/source.tar.gz" | sha256sum -c -
tar -xzf "$WORK/source.tar.gz" -C "$WORK/source"
cp "$WORK/source/source_manifest.json" "$OUT/source_manifest.json"
timeout --signal=TERM --kill-after=15 1800 /bin/bash -Eeuo pipefail <<'VR_SETUP'
apt-get update
apt-get install -y python3 python3-venv ca-certificates
/usr/bin/python3 -I -m venv /workspace/vr-eval123/bootstrap-venv
/workspace/vr-eval123/bootstrap-venv/bin/pip install uv==0.8.22
export UV_PYTHON_INSTALL_DIR=/workspace/vr-eval123/python
/workspace/vr-eval123/bootstrap-venv/bin/uv python install 3.11.13
/workspace/vr-eval123/bootstrap-venv/bin/uv venv --python 3.11.13 --seed /workspace/vr-eval123/venv
/workspace/vr-eval123/venv/bin/pip install --no-cache-dir -r /workspace/vr-eval123/source/native/requirements.txt
/workspace/vr-eval123/venv/bin/pip check
VR_SETUP
PYTHON="$WORK/venv/bin/python"
export PYTHONPATH="$WORK/source/evaluator:$WORK/source/native"
cd "$WORK/source/native"
"$PYTHON" -m pip freeze > "$OUT/pip_freeze.txt"
"$PYTHON" -m fhp_vr_deep.batch_diagnostics monitor --output "$OUT/resources.jsonl" \
  --interval-seconds 30 --cloud-log-every 10 &
MONITOR_PID=$!
MODULE=experiments.fhp.retrospective_exp1_exp2_exp3_evaluation.run
ARGS=(--source-root "$WORK/inputs" --output "$OUT" --workers 16)
STAGE=download
timeout --kill-after=15 1800 "$PYTHON" -m "$MODULE" download "${ARGS[@]}" --bucket "$VR_EVAL_BUCKET"
STAGE=smoke
timeout --kill-after=15 1200 "$PYTHON" -m "$MODULE" smoke "${ARGS[@]}"
if [[ "$VR_EVAL_SMOKE" == 1 ]]; then STAGE=smoke_complete; exit 0; fi
STAGE=main
timeout --kill-after=15 81000 "$PYTHON" -m "$MODULE" main "${ARGS[@]}" --max-hours 22
# Make the successful main comparison durable BEFORE constructing LBR.
sync_outputs
if [[ ! -f "$OUT/main/SUCCESS.json" ]]; then
  echo 'Main timing gate deferred full evaluation. Inspect main/STATUS.json.'
  STAGE=main_deferred
  exit 0
fi
STAGE=lbr
timeout --kill-after=15 36600 "$PYTHON" -m "$MODULE" lbr "${ARGS[@]}" --max-hours 10
"$PYTHON" -c 'import json,sys; print("LBR outcome:", json.load(open(sys.argv[1]))["status"])' "$OUT/lbr/STATUS.json"
STAGE=complete
