#!/usr/bin/env bash
set -Eeuo pipefail
ACTION="${1:-dry-run}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
: "${PROJECT_ID:?Set PROJECT_ID}"
: "${REGION:?Set REGION}"
: "${BUCKET:?Set BUCKET}"
: "${SA_EMAIL:?Set SA_EMAIL}"
: "${RUN_ID:?Set a fresh VR_DEEP Experiment 9 RUN_ID (reuse only for resume/status)}"
BUCKET="gs://${BUCKET#gs://}"
BUCKET="${BUCKET%/}"
if [[ "$ACTION" == status ]]; then
  gcloud batch jobs list --project "$PROJECT_ID" --location "$REGION" \
    --filter="labels.workload=fhp-exp43 AND labels.run=$RUN_ID" --format='table(name.basename(),status.state,createTime)'
  echo "Completion requires $BUCKET/$RUN_ID/analysis/SUCCESS.json with smoke=false."
  exit 0
fi
case "$ACTION" in run|resume|smoke-cloud|preflight|dry-run) ;; *) echo "Usage: $0 [run|resume|smoke-cloud|preflight|dry-run|status]" >&2; exit 2 ;; esac
: "${REPO_REF:?Set REPO_REF to the full pushed VR_DEEP Experiment 9 commit}"
if [[ ! "$REPO_REF" =~ ^[0-9a-f]{40}$ ]] || ! git -C "$REPO_DIR" cat-file -e "$REPO_REF^{commit}"; then
  echo "REPO_REF must be a full commit available in this checkout." >&2; exit 2
fi
for file in experiments/fhp/exp9_vr_deep_representation_48h/config.json experiments/fhp/exp9_vr_deep_representation_48h/run.py gcp/exp9_representation_batch.py; do
  git -C "$REPO_DIR" cat-file -e "$REPO_REF:$file" || { echo "Commit and push this repository's Experiment 9 first." >&2; exit 2; }
done
VR_BUCKET="${VR_BUCKET:-gs://clever-overview-399515-fhp-vr-deep-results}"
SD_BUCKET="${SD_BUCKET:-gs://clever-overview-399515-fhp-deep-cfr-results}"
TEMP_DIR="$(mktemp -d /tmp/fhp-vr-exp9.XXXXXX)"
git -C "$REPO_DIR" show "$REPO_REF:gcp/exp9_representation_batch.py" > "$TEMP_DIR/builder.py"
git -C "$REPO_DIR" show "$REPO_REF:experiments/fhp/exp9_vr_deep_representation_48h/config.json" > "$TEMP_DIR/contract.json"
EVALUATOR_REF="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["shared_evaluator_ref"])' "$TEMP_DIR/contract.json")"
EVALUATION_REPO="${EVALUATION_REPO:-$REPO_DIR/../../fhp_ucv_escher/fhp-ucv-escher-experiments}"
if ! git -C "$EVALUATION_REPO" cat-file -e "$EVALUATOR_REF^{commit}" 2>/dev/null; then
  EVALUATION_REPO="$TEMP_DIR/evaluator"
  git clone --filter=blob:none https://github.com/lawrencewlcknight/fhp-ucv-escher-experiments.git "$EVALUATION_REPO"
fi
git -C "$EVALUATION_REPO" show "$EVALUATOR_REF:gcp/representation_head_to_head_batch.py" > "$TEMP_DIR/shared_builder.py"
git -C "$EVALUATION_REPO" show "$EVALUATOR_REF:gcp/representation_access_preflight.py" > "$TEMP_DIR/preflight.py"
EXTRA_ARGS=()
JOB_NAME="$RUN_ID"
if [[ "$ACTION" == resume ]]; then
  EXTRA_ARGS+=(--resume)
  JOB_NAME="${RUN_ID}-r$(date -u '+%H%M%S')"
fi
if [[ "$ACTION" == smoke-cloud || "${EXP9_SMOKE_ONLY:-0}" == 1 ]]; then EXTRA_ARGS+=(--smoke-only); fi
BUILDER_ARGS=(--algorithm vr --output "$TEMP_DIR/job.json" --run-id "$RUN_ID"
  --bucket "$BUCKET" --vr-bucket "$VR_BUCKET" --sd-bucket "$SD_BUCKET"
  --service-account "$SA_EMAIL" --repo-ref "$REPO_REF"
  --contract "$TEMP_DIR/contract.json" --shared-builder "$TEMP_DIR/shared_builder.py")
python3 "$TEMP_DIR/builder.py" "${BUILDER_ARGS[@]}" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
if [[ "$ACTION" == dry-run ]]; then
  echo "Prepared $TEMP_DIR/job.json; no cloud job submitted."; exit 0
fi
gcloud iam service-accounts describe "$SA_EMAIL" --project "$PROJECT_ID" --format='value(email)'
python3 "$TEMP_DIR/preflight.py" --builder "$TEMP_DIR/builder.py" \
  --project "$PROJECT_ID" --receipt "$TEMP_DIR/access_preflight.json" "${BUILDER_ARGS[@]}"
if [[ "$ACTION" == preflight ]]; then
  echo "Read-only preflight complete; no cloud job submitted."; exit 0
fi
ACTIVE_STATES="$(gcloud batch jobs list --project "$PROJECT_ID" --location "$REGION" \
  --filter="labels.workload=fhp-exp43 AND labels.run=$RUN_ID" --format='value(status.state)')"
for state in $ACTIVE_STATES; do
  case "$state" in SUCCEEDED|FAILED) ;; *) echo "An active job uses this output namespace." >&2; exit 2 ;; esac
done
if [[ "$ACTION" == resume ]]; then
  gcloud storage ls "$BUCKET/$RUN_ID/analysis/evaluation_manifest.json" >/dev/null
else
  if gcloud storage ls "$BUCKET/$RUN_ID/**" >"$TEMP_DIR/objects.txt" 2>"$TEMP_DIR/storage_error.txt"; then
    echo "Output prefix exists; use a fresh RUN_ID or resume." >&2; exit 2
  fi
  if [[ "$( < "$TEMP_DIR/storage_error.txt")" != *"matched no objects"* ]]; then
    cat "$TEMP_DIR/storage_error.txt" >&2
    echo "Cannot confirm unused output prefix; refusing submission." >&2; exit 2
  fi
fi
gcloud batch jobs submit "$JOB_NAME" --project "$PROJECT_ID" --location "$REGION" --config "$TEMP_DIR/job.json"
echo "Submitted $JOB_NAME. Laptop may disconnect. Results: $BUCKET/$RUN_ID/analysis/"
