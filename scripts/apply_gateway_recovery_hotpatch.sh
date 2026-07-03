#!/usr/bin/env bash
set -euo pipefail

HOST="${1:-leadpoet-gateway}"
BUNDLE="${2:-/private/tmp/leadpoet-gateway-recovery-hotpatch.tgz}"
REMOTE_BUNDLE="/tmp/leadpoet-gateway-recovery-hotpatch.tgz"
REMOTE_ROOT="/home/ec2-user/gateway"

if [[ "${LEADPOET_PROD_WRITE_APPROVED:-}" != "yes" ]]; then
  cat >&2 <<'MSG'
Refusing to mutate production.

Set LEADPOET_PROD_WRITE_APPROVED=yes only after the operator has explicitly
approved production code mutation. This script copies files to the gateway and
extracts them over the live checkout after creating a hotpatch backup.
MSG
  exit 2
fi

if [[ ! -f "$BUNDLE" ]]; then
  "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/prepare_gateway_recovery_hotpatch.sh" "$BUNDLE"
fi

echo "Copying hotpatch bundle to $HOST:$REMOTE_BUNDLE"
scp "$BUNDLE" "$HOST:$REMOTE_BUNDLE"

echo "Applying hotpatch on $HOST"
ssh -o BatchMode=yes "$HOST" 'bash -s' <<'REMOTE'
set -euo pipefail

REMOTE_ROOT="/home/ec2-user/gateway"
REMOTE_BUNDLE="/tmp/leadpoet-gateway-recovery-hotpatch.tgz"

cd "$REMOTE_ROOT"
test -f "$REMOTE_BUNDLE"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
BACKUP_DIR="hotpatch-backups/$STAMP"
mkdir -p "$BACKUP_DIR"
tar -tzf "$REMOTE_BUNDLE" > "$BACKUP_DIR/file-list.txt"

while IFS= read -r file; do
  if [[ -e "$file" ]]; then
    mkdir -p "$BACKUP_DIR/$(dirname "$file")"
    cp -a "$file" "$BACKUP_DIR/$file"
  else
    echo "remote file will be created: $file"
  fi
done < "$BACKUP_DIR/file-list.txt"

tar -xzf "$REMOTE_BUNDLE"

python3 -m py_compile \
  fulfillment/api.py \
  research_lab/admin.py \
  research_lab/arweave_audit.py \
  research_lab/chain.py \
  research_lab/maintenance.py \
  research_lab/scoring_worker.py \
  research_lab/trajectory_projector.py \
  research_lab/worker.py \
  tasks/hourly_batch.py \
  utils/logger.py \
  gateway/__init__.py

echo "Applied gateway recovery hotpatch."
echo "Backup: $REMOTE_ROOT/$BACKUP_DIR"
echo "Restart gateway/workers separately, then run health and requeue dry-runs."
REMOTE
