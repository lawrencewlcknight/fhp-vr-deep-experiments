#!/usr/bin/env bash
set -Eeuo pipefail
export PYTHONDONTWRITEBYTECODE=1
cd "$(dirname "$0")/.."
ACTION="${1:-run}"
if [[ "$ACTION" == smoke-local ]]; then
  OUT="$(mktemp -d "${TMPDIR:-/tmp}/fhp-vr-exp3-smoke.XXXXXX")"
  exec "${PYTHON:-python3}" -m experiments.fhp.exp3_vr_deep_lossless_n2_standard16.run smoke --output-root "$OUT" --threads 1
fi
: "${PROJECT_ID:?Set PROJECT_ID}"
: "${REGION:?Set REGION}"
: "${BUCKET:?Set BUCKET}"
: "${SA_EMAIL:?Set SA_EMAIL to the FHP VR-Deep Batch service account}"
export REPO_REF="${REPO_REF:-$(git rev-parse HEAD)}"
if [[ "$ACTION" == run || "$ACTION" == dry-run || "$ACTION" == smoke-only ]]; then
  export RUN_ID="${RUN_ID:-vr3-vm16-$(date -u '+%Y%m%d-%H%M%S')}"
else
  : "${RUN_ID:?Set RUN_ID to the existing run}"
fi
if [[ "$ACTION" == run || "$ACTION" == smoke-only || "$ACTION" == aggregate-only ]]; then
  for required in experiments/fhp/exp3_vr_deep_lossless_n2_standard16/run.py \
                  experiments/fhp/exp3_vr_deep_lossless_n2_standard16/config.py \
                  gcp/exp3_vr_deep_lossless_n2_standard16_batch.py fhp_vr_deep/features.py \
                  vr_deep_cfr/encoded_solver.py tests/fixtures/ucv_exp2_encoder_golden.json \
                  tests/test_exp3_vr_deep_lossless_n2_standard16.py; do
    if ! git cat-file -e "$REPO_REF:$required" 2>/dev/null; then
      echo "REPO_REF=$REPO_REF does not contain $required." >&2
      echo 'Commit and push the new experiment, then export REPO_REF to that full commit SHA.' >&2
      exit 2
    fi
  done
fi
exec "${PYTHON:-python3}" gcp/exp3_vr_deep_lossless_n2_standard16_batch.py "$ACTION"
