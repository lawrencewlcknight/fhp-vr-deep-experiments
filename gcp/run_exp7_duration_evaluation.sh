#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${PYTHON:-python3}" "$ROOT/gcp/exp7_duration_evaluation_batch.py" "${@:-run}"
