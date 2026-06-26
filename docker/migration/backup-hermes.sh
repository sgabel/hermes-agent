#!/usr/bin/env bash
# PRD-033 FR-12 — pre-cutover backup bundle of ~/.hermes. Run FIRST, before any rewrite
# or service stop. Produces a restorable tarball under ~/hermes/backups/.
set -euo pipefail
HERMES_HOME="${HERMES_HOME_HOST:-$HOME/.hermes}"
DEST="${BACKUP_DIR:-$HOME/hermes/backups}"
TS="$(date +%Y%m%d-%H%M%S)"
mkdir -p "$DEST"
OUT="$DEST/hermes-home-precutover-$TS.tar.gz"
# Exclude only regenerable caches; keep state.db, sessions, mem0, configs, creds.
# tar exits 1 on the benign "file changed as we read it" warning (harmless when the
# agent is stopped first, but tolerate it regardless); exit >=2 is a real failure.
set +e
tar --warning=no-file-changed --exclude='./logs/*.log' -czf "$OUT" -C "$(dirname "$HERMES_HOME")" "$(basename "$HERMES_HOME")"
rc=$?
set -e
if [ "$rc" -ge 2 ]; then echo "ERROR: tar failed (rc=$rc)" >&2; exit "$rc"; fi
# Integrity smoke: the bundle is the only rollback-to-data safety net — fail loudly if
# it is truncated/unreadable rather than discovering it during a rollback.
tar -tzf "$OUT" >/dev/null 2>&1 || { echo "ERROR: backup unreadable: $OUT" >&2; exit 1; }
echo "backup: $OUT ($(du -h "$OUT" | cut -f1)) (tar rc=$rc, integrity OK)"
echo "$OUT" > "$DEST/.last-precutover-backup"
