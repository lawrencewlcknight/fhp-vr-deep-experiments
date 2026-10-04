#!/usr/bin/env bash
set -Eeuo pipefail
export PYTHONDONTWRITEBYTECODE=1
cd "$(dirname "$0")/.."
ACTION="${1:-run}"
case "$ACTION" in
  run|dry-run|smoke-only) export RUN_ID="${RUN_ID:-vr-eval123-$(date -u '+%Y%m%d-%H%M%S')}" ;;
  resume|status) : "${RUN_ID:?Set RUN_ID to the existing evaluation}" ;;
  *) echo 'Use run, smoke-only, dry-run, resume or status' >&2; exit 2 ;;
esac
exec "${PYTHON:-python3}" gcp/retrospective_exp1_exp2_exp3_evaluation_batch.py "$ACTION"
