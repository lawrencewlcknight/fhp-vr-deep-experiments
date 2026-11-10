#!/usr/bin/env bash
set -Eeuo pipefail
ACTION="${1:-dry-run}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
: "${PROJECT_ID:?Set PROJECT_ID}"
: "${REGION:?Set REGION}"
: "${BUCKET:?Set BUCKET}"
: "${SA_EMAIL:?Set SA_EMAIL}"
: "${RUN_ID:?Set a fresh VR Experiment 10 RUN_ID}"
: "${REPO_REF:?Set REPO_REF to the full pushed commit containing Experiment 10}"
BUCKET="gs://${BUCKET#gs://}"; BUCKET="${BUCKET%/}"
case "$ACTION" in dry-run|run|profile|resume) ;; *) echo 'Use dry-run, run, profile, or resume' >&2; exit 2 ;; esac
if [[ ! "$REPO_REF" =~ ^[0-9a-f]{40}$ ]]; then echo 'Full commit SHA required' >&2; exit 2; fi
TEMP_DIR="$(mktemp -d /tmp/vr10-launch.XXXXXX)"
git -C "$REPO_DIR" show "$REPO_REF:gcp/exp10_policy_extraction_batch.py" > "$TEMP_DIR/builder.py"
git -C "$REPO_DIR" cat-file -e "$REPO_REF:experiments/fhp/exp10_vr_deep_policy_extraction/run.py"
EXTRA=(); JOB_NAME="$RUN_ID"
if [[ "$ACTION" == profile ]]; then EXTRA+=(--profile-only); fi
if [[ "$ACTION" == resume ]]; then EXTRA+=(--resume); JOB_NAME="${RUN_ID}-r$(date -u '+%H%M%S')"; fi
python3 "$TEMP_DIR/builder.py" --run-id "$RUN_ID" --bucket "$BUCKET" \
  --service-account "$SA_EMAIL" --repo-ref "$REPO_REF" --output "$TEMP_DIR/job.json" ${EXTRA[@]+"${EXTRA[@]}"}
if [[ "$ACTION" == dry-run ]]; then echo "Prepared $TEMP_DIR/job.json; no job submitted."; exit 0; fi
if (( ${#JOB_NAME} > 63 )); then echo 'Job name exceeds 63 characters; use a shorter RUN_ID' >&2; exit 2; fi
ACTIVE_STATES="$(gcloud batch jobs list --project "$PROJECT_ID" --location "$REGION" \
  --filter="labels.experiment=vr10-policy-extraction AND labels.run=$RUN_ID" --format='value(status.state)')"
for state in $ACTIVE_STATES; do
  case "$state" in SUCCEEDED|FAILED) ;; *) echo 'An active job uses this output prefix' >&2; exit 2 ;; esac
done
if [[ "$ACTION" == resume ]]; then
  gcloud storage ls "$BUCKET/$RUN_ID/experiment.json" >/dev/null
else
  if gcloud storage ls "$BUCKET/$RUN_ID/**" >"$TEMP_DIR/objects" 2>"$TEMP_DIR/storage_error"; then
    echo 'Output exists; use a fresh RUN_ID or resume' >&2; exit 2
  fi
  if [[ "$(< "$TEMP_DIR/storage_error")" != *"matched no objects"* ]]; then
    cat "$TEMP_DIR/storage_error" >&2; exit 2
  fi
fi
gcloud batch jobs submit "$JOB_NAME" --project "$PROJECT_ID" --location "$REGION" --config "$TEMP_DIR/job.json"
echo "Submitted $JOB_NAME. Complete analysis requires $BUCKET/$RUN_ID/analysis/SUCCESS.json."
