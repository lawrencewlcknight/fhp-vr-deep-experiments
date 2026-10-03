#!/usr/bin/env bash
set -Eeuo pipefail
export PYTHONDONTWRITEBYTECODE=1
cd "$(dirname "$0")/.."
ACTION="${1:-run}"
if [[ "$ACTION" == smoke-local ]]; then
  SCRATCH="$(mktemp -d "${TMPDIR:-/tmp}/fhp-vr-exp4-smoke.XXXXXX")"
  echo "Profiling smoke output: $SCRATCH/output"
  exec "${PYTHON:-python3}" -m experiments.fhp.exp4_vr_deep_thread_profile.run smoke --output-root "$SCRATCH/output"
fi
: "${PROJECT_ID:?Set PROJECT_ID}"
: "${REGION:?Set REGION}"
: "${BUCKET:?Set BUCKET}"
: "${SA_EMAIL:?Set SA_EMAIL}"
export REPO_REF="${REPO_REF:-$(git rev-parse HEAD)}"
if [[ "$ACTION" == status ]]; then
  : "${RUN_ID:?Set RUN_ID to the existing run}"
else
  export RUN_ID="${RUN_ID:-vr4-threads-$(date -u '+%Y%m%d-%H%M%S')}"
fi
if [[ "$ACTION" == run || "$ACTION" == smoke-only ]]; then
  for required in experiments/fhp/exp4_vr_deep_thread_profile/{run,config,workload}.py \
                  gcp/exp4_vr_deep_thread_profile_batch.py tests/test_exp4_vr_deep_thread_profile.py; do
    if ! git cat-file -e "$REPO_REF:$required" 2>/dev/null; then
      echo "REPO_REF=$REPO_REF lacks $required. Commit and push Experiment 4 before submitting." >&2
      exit 2
    fi
  done
fi
exec "${PYTHON:-python3}" gcp/exp4_vr_deep_thread_profile_batch.py "$ACTION"
