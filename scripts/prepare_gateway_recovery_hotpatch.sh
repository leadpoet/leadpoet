#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-/private/tmp/leadpoet-gateway-recovery-hotpatch.tgz}"

FILES=(
  "fulfillment/api.py"
  "research_lab/admin.py"
  "research_lab/arweave_audit.py"
  "research_lab/chain.py"
  "research_lab/maintenance.py"
  "research_lab/scoring_worker.py"
  "research_lab/trajectory_projector.py"
  "research_lab/worker.py"
  "tasks/hourly_batch.py"
  "utils/logger.py"
  "utils/tee_client.py"
  "gateway/__init__.py"
)

for file in "${FILES[@]}"; do
  if [[ ! -f "$ROOT_DIR/gateway/$file" ]]; then
    echo "missing gateway/$file" >&2
    exit 1
  fi
done

COPYFILE_DISABLE=1 tar --disable-copyfile -C "$ROOT_DIR/gateway" -czf "$OUT" "${FILES[@]}"
ls -lh "$OUT"
echo
echo "Prepared recovery hotpatch bundle: $OUT"
echo "Production copy/apply/restart still requires explicit operator approval."
