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
for file in experiments/fhp/retrospective_threeway_48h_evaluation/run.py \
            gcp/threeway_48h_evaluation_batch.py; do
  if ! git -C "$REPO_DIR" cat-file -e "$REPO_REF:$file"; then
    echo "REPO_REF predates the three-way evaluator. Commit and push it first." >&2
    exit 2
  fi
done
TEMP_DIR="$(mktemp -d /tmp/fhp-threeway-48h-launch.XXXXXX)"
trap 'rm -f "$TEMP_DIR/builder.py"; rmdir "$TEMP_DIR"' EXIT
git -C "$REPO_DIR" show "$REPO_REF:gcp/threeway_48h_evaluation_batch.py" > "$TEMP_DIR/builder.py"
"${PYTHON:-python3}" "$TEMP_DIR/builder.py" "$@"
