#!/usr/bin/env bash
# Called periodically and by Batch's alwaysRun finalizer. Capture immutable
# completion markers BEFORE copying data; never publish a marker ahead of data.
set -Eeuo pipefail
: "${WORK:?}" "${OUT:?}" "${COMPONENT:?}" "${VR7_DEST:?}"
exec 9>"$WORK/upload.lock"
flock -w 180 9
SNAPSHOT="$(mktemp -d "$WORK/markers.XXXXXX")"
trap 'rm -r -- "$SNAPSHOT"' EXIT
while IFS= read -r -d '' marker; do
  relative="${marker#"$OUT/"}"
  mkdir -p "$SNAPSHOT/$(dirname "$relative")"
  cp "$marker" "$SNAPSHOT/$relative"
done < <(find "$OUT" -type f -name SUCCESS.json -print0)
timeout 600 gcloud storage rsync --recursive \
  --exclude='.*[.]tmp$|(^|/)SUCCESS[.]json$|(^|/)[.]lock$' "$OUT" "$VR7_DEST/analysis/$COMPONENT"
# All snapshots refer to data already covered by the successful rsync.
timeout 120 gcloud storage rsync --recursive "$SNAPSHOT" "$VR7_DEST/analysis/$COMPONENT"
