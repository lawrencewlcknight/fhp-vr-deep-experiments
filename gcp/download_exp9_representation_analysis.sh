#!/usr/bin/env bash
set -Eeuo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${BUCKET:?Set BUCKET}" "${RUN_ID:?Set the completed evaluation RUN_ID}"
EVALUATION_REPO="${EVALUATION_REPO:-$REPO_DIR/../../fhp_ucv_escher/fhp-ucv-escher-experiments}"
TEMP_DIR="$(mktemp -d /tmp/fhp-vr-exp9-download.XXXXXX)"
if ! git -C "$EVALUATION_REPO" cat-file -e "b6eabbeeeca6d8ae8f07f383a5296492c2cd2376^{commit}" 2>/dev/null; then
  EVALUATION_REPO="$TEMP_DIR/evaluator"
  git clone --filter=blob:none https://github.com/lawrencewlcknight/fhp-ucv-escher-experiments.git "$EVALUATION_REPO"
fi
git -C "$EVALUATION_REPO" show "b6eabbeeeca6d8ae8f07f383a5296492c2cd2376:gcp/download_representation_analysis.py" > "$TEMP_DIR/download.py"
python3 "$TEMP_DIR/download.py" --bucket "gs://${BUCKET#gs://}" --run-id "$RUN_ID" \
  --output "${OUTPUT_DIR:-$REPO_DIR/cloud_outputs/$RUN_ID/analysis}"
