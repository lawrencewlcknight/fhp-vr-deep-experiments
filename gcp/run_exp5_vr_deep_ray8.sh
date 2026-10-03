#!/usr/bin/env bash
set -Eeuo pipefail
export PYTHONDONTWRITEBYTECODE=1
cd "$(dirname "$0")/.."
ACTION="${1:-run}"
if [[ "$ACTION" == smoke-local ]]; then
  OUT="$(mktemp -d "${TMPDIR:-/tmp}/fhp-vr-exp5-smoke.XXXXXX")"
  echo "Ray smoke output: $OUT"
  exec "${PYTHON:-python3}" -m experiments.fhp.exp5_vr_deep_ray8.run smoke --output-root "$OUT" --threads 1
fi
: "${PROJECT_ID:?Set PROJECT_ID}"
: "${REGION:?Set REGION}"
: "${BUCKET:?Set BUCKET}"
: "${SA_EMAIL:?Set SA_EMAIL}"
export REPO_REF="${REPO_REF:-$(git rev-parse HEAD)}"
if [[ "$ACTION" == run || "$ACTION" == dry-run || "$ACTION" == smoke-only ]]; then
  export RUN_ID="${RUN_ID:-vr5-ray8-$(date -u '+%Y%m%d-%H%M%S')}"
else
  : "${RUN_ID:?Set RUN_ID to the existing run}"
fi
if [[ "$ACTION" == run || "$ACTION" == smoke-only || "$ACTION" == aggregate-only ]]; then
  for required in experiments/fhp/exp5_vr_deep_ray8/{run,config,train,aggregate}.py \
                  vr_deep_cfr/parallel_solver.py requirements-ray.txt \
                  gcp/exp5_vr_deep_ray8_batch.py tests/test_exp5_vr_deep_ray8.py tests/test_vr_deep_parallel.py; do
    if ! git cat-file -e "$REPO_REF:$required" 2>/dev/null; then
      echo "REPO_REF=$REPO_REF lacks $required. Commit and push Experiment 5 before submitting." >&2
      exit 2
    fi
  done
fi
exec "${PYTHON:-python3}" gcp/exp5_vr_deep_ray8_batch.py "$ACTION"
