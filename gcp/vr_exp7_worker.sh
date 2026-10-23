#!/usr/bin/env bash
set -Eeuo pipefail
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin${PATH:+:$PATH}"
export DEBIAN_FRONTEND=noninteractive
case "${BATCH_TASK_INDEX:?}" in 0) COMPONENT=matches;; 1) COMPONENT=strategic;; *) exit 2;; esac
export WORK=/workspace/vr-exp7
export OUT="$WORK/output" COMPONENT
mkdir -p "$OUT" "$WORK/source" "$WORK/inputs"
if [[ "$VR7_RESUME" == 1 ]]; then
  timeout 1800 gcloud storage rsync --recursive "$VR7_DEST/analysis/$COMPONENT" "$OUT"
fi
exec > >(tee -a "$OUT/bootstrap.log") 2>&1
STAGE=bootstrap
MONITOR_PID=""
SYNC_PID=""
sync_outputs() {
  [[ -f "$WORK/source/native/gcp/vr_exp7_sync.sh" ]] || return 0
  bash "$WORK/source/native/gcp/vr_exp7_sync.sh"
}
cleanup() {
  code=$?
  set +e
  kill ${SYNC_PID:+"$SYNC_PID"} ${MONITOR_PID:+"$MONITOR_PID"} 2>/dev/null
  wait ${SYNC_PID:+"$SYNC_PID"} ${MONITOR_PID:+"$MONITOR_PID"} 2>/dev/null
  printf '{"exit_code":%s,"stage":"%s","component":"%s"}\n' "$code" "$STAGE" "$COMPONENT" > "$OUT/process_exit.json"
  if [[ -x "$WORK/venv/bin/python" ]]; then
    PYTHONPATH="$WORK/source/native" "$WORK/venv/bin/python" -m fhp_vr_deep.batch_diagnostics finalize \
      --snapshots "$OUT/resources.jsonl" --output "$OUT/resource_summary.json" \
      --status-output "$OUT/resource_status.json" --failure-root "$OUT" --exit-code "$code" \
      --requested-memory-mib 62000 --job-name "${BATCH_JOB_UID:-vr-exp7}" \
      --bucket-destination "$VR7_DEST/analysis/$COMPONENT" || true
  fi
  sync_outputs || echo 'Exit upload failed; final Batch runnable will retry.'
  exit "$code"
}
trap cleanup EXIT
trap 'printf "ERROR stage=%s line=%s exit=%s\n" "$STAGE" "$LINENO" "$?" >&2' ERR
trap 'exit 143' TERM
trap 'exit 130' INT
timeout 180 gcloud storage cp "$VR7_DEST/inputs/source.tar.gz" "$WORK/source.tar.gz"
printf '%s  %s\n' "$VR7_SHA256" "$WORK/source.tar.gz" | sha256sum -c -
tar -xzf "$WORK/source.tar.gz" -C "$WORK/source"
cp "$WORK/source/source_manifest.json" "$OUT/source_manifest.json"
while sleep 300; do sync_outputs || echo 'WARNING: periodic upload failed'; done &
SYNC_PID=$!
timeout --signal=TERM --kill-after=15 1800 /bin/bash -Eeuo pipefail <<'VR7_SETUP'
apt-get update
apt-get install -y python3 python3-venv ca-certificates
/usr/bin/python3 -I -m venv /workspace/vr-exp7/bootstrap-venv
/workspace/vr-exp7/bootstrap-venv/bin/pip install uv==0.8.22
export UV_PYTHON_INSTALL_DIR=/workspace/vr-exp7/python
/workspace/vr-exp7/bootstrap-venv/bin/uv python install 3.11.13
/workspace/vr-exp7/bootstrap-venv/bin/uv venv --python 3.11.13 --seed /workspace/vr-exp7/venv
/workspace/vr-exp7/venv/bin/pip install --no-cache-dir -r /workspace/vr-exp7/source/native/requirements.txt
/workspace/vr-exp7/venv/bin/pip check
VR7_SETUP
PYTHON="$WORK/venv/bin/python"
NATIVE="$WORK/source/native"
SUITE="$WORK/source/evaluator"
UCV="$WORK/source/ucv"
EXPERIMENT="$NATIVE/experiments/fhp/exp7_vr_deep_24h_48h_evaluation"
export PYTHONPATH="$SUITE:$NATIVE"
cd "$NATIVE"
"$PYTHON" -m pip freeze > "$OUT/pip_freeze.txt"
"$PYTHON" -m fhp_vr_deep.batch_diagnostics monitor --output "$OUT/resources.jsonl" \
  --interval-seconds 30 --cloud-log-every 10 &
MONITOR_PID=$!
STAGE=download
"$PYTHON" "$EXPERIMENT/sources.py" download --root "$WORK/inputs" --component "$COMPONENT" \
  --vr-bucket "$VR7_VR_BUCKET" --ucv-bucket "$VR7_UCV_BUCKET" --output "$OUT/sources.json"
STAGE=validate
"$PYTHON" "$EXPERIMENT/sources.py" validate-native --root "$WORK/inputs" --component "$COMPONENT" \
  --family vr_deep --output "$OUT/native_vr_validation.json"
if [[ "$COMPONENT" == strategic ]]; then
  (cd "$UCV" && PYTHONPATH="$UCV" "$PYTHON" "$EXPERIMENT/sources.py" validate-native \
    --root "$WORK/inputs" --component strategic --family ucv --output "$OUT/native_ucv_validation.json")
fi
run_component() {
  local mode="$1"
  local extra=()
  [[ "$mode" == smoke ]] && extra+=(--smoke)
  if [[ "$COMPONENT" == matches ]]; then
    "$PYTHON" -m experiments.fhp.exp7_vr_deep_24h_48h_evaluation.matches \
      --input "$WORK/inputs" --output "$OUT/$mode" --bundle-manifest "$WORK/source/source_manifest.json" \
      --workers 16 --max-hours 40 "${extra[@]}"
  else
    (cd "$SUITE" && PYTHONPATH="$SUITE" "$PYTHON" "$EXPERIMENT/strategic.py" \
      --input "$WORK/inputs" --output "$OUT/$mode" --bundle-manifest "$WORK/source/source_manifest.json" \
      --suite-repo "$SUITE" --vr-repo "$NATIVE" --ucv-repo "$UCV" "${extra[@]}")
  fi
}
STAGE=smoke
if [[ "$COMPONENT" == matches ]]; then
  "$PYTHON" -m experiments.fhp.exp7_vr_deep_24h_48h_evaluation.matches \
    --input "$WORK/inputs" --output "$OUT/smoke" --bundle-manifest "$WORK/source/source_manifest.json" \
    --smoke --workers 16 --max-hours 1
else
  run_component smoke
fi
sync_outputs
if [[ "$VR7_SMOKE" == 1 ]]; then STAGE=smoke_complete; exit 0; fi
STAGE=full
run_component full
[[ -f "$OUT/full/SUCCESS.json" ]]
STAGE=complete
sync_outputs
