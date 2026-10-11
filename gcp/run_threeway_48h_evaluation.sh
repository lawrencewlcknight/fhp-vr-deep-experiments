#!/usr/bin/env bash
set -Eeuo pipefail
ACTION="${1:-run}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
if [[ "$ACTION" == status ]]; then
  exec "${PYTHON:-python3}" "$SCRIPT_DIR/threeway_48h_evaluation_batch.py" "$@"
fi
: "${REPO_REF:?Set REPO_REF to the full pushed VR-Deep evaluation commit SHA}"
if [[ ! "$REPO_REF" =~ ^[0-9a-f]{40}$ ]]; then
  echo "REPO_REF must be a full pushed commit SHA" >&2
  exit 2
fi
# Scheduling-only revisions may use a newer launcher while native/evaluator
# checkouts stay at the original REPO_REF required by saved cache identities.
LAUNCHER_REF="${LAUNCHER_REF:-$REPO_REF}"
if [[ ! "$LAUNCHER_REF" =~ ^[0-9a-f]{40}$ ]]; then
  echo "LAUNCHER_REF must be a full pushed commit SHA" >&2
  exit 2
fi
for file in experiments/fhp/retrospective_threeway_48h_evaluation/run.py \
            gcp/threeway_48h_evaluation_batch.py; do
  if ! git -C "$REPO_DIR" cat-file -e "$REPO_REF:$file"; then
    echo "REPO_REF predates the three-way evaluator. Commit and push it first." >&2
    exit 2
  fi
done
if ! git -C "$REPO_DIR" cat-file -e "$LAUNCHER_REF:gcp/threeway_48h_evaluation_batch.py"; then
  echo "LAUNCHER_REF does not contain the three-way Batch launcher. Commit and push it first." >&2
  exit 2
fi
TEMP_DIR="$(mktemp -d /tmp/fhp-threeway-48h-launch.XXXXXX)"
trap 'rm -f "$TEMP_DIR/builder.py" "$TEMP_DIR/strategic_contract.py"; rmdir "$TEMP_DIR"' EXIT
git -C "$REPO_DIR" show "$LAUNCHER_REF:gcp/threeway_48h_evaluation_batch.py" > "$TEMP_DIR/builder.py"
CONTRACT_FILE="experiments/fhp/retrospective_threeway_48h_evaluation/strategic.py"
if git -C "$REPO_DIR" cat-file -e "$REPO_REF:$CONTRACT_FILE" 2>/dev/null; then
  git -C "$REPO_DIR" show "$REPO_REF:$CONTRACT_FILE" > "$TEMP_DIR/strategic_contract.py"
fi
PYTHONDONTWRITEBYTECODE=1 "${PYTHON:-python3}" "$TEMP_DIR/builder.py" "$@"
