#!/bin/bash
#
# Stage top-level Research Lab runtime packages into the gateway Docker context.
# The Nitro enclave build only sees ~/gateway, so runtime dependencies imported
# as top-level packages must be copied under gateway/_attested_runtime first.

set -euo pipefail

GATEWAY_ROOT="${GATEWAY_ROOT:-$HOME/gateway}"
SOURCE_ROOT="${RESEARCH_LAB_RUNTIME_SOURCE_ROOT:-$HOME}"
DEST_ROOT="$GATEWAY_ROOT/_attested_runtime"
TMP_ROOT="$GATEWAY_ROOT/.attested_runtime.tmp"
PACKAGES=(
  "research_lab"
  "leadpoet_verifier"
  "schemas"
  "leadpoet_canonical"
  "qualification"
  "validator_models"
)

echo "Staging attested Research Lab runtime packages"
echo "  Gateway root: $GATEWAY_ROOT"
echo "  Source root:  $SOURCE_ROOT"
echo "  Dest root:    $DEST_ROOT"

if ! command -v rsync >/dev/null 2>&1; then
  echo "ERROR: rsync is required to stage attested runtime packages" >&2
  exit 1
fi

rm -rf "$TMP_ROOT"
mkdir -p "$TMP_ROOT"

for package in "${PACKAGES[@]}"; do
  source_dir="$SOURCE_ROOT/$package"
  dest_dir="$TMP_ROOT/$package"
  if [ ! -d "$source_dir" ]; then
    echo "ERROR: required runtime package missing: $source_dir" >&2
    exit 1
  fi
  mkdir -p "$dest_dir"
  rsync -a --delete \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='*.pyo' \
    --exclude='*.pyd' \
    --exclude='*.log' \
    --exclude='*.bak' \
    --exclude='*.bak-*' \
    --exclude='*.backup*' \
    --exclude='*.tmp' \
    --exclude='*.pem' \
    --exclude='*.key' \
    --exclude='*.jwk' \
    --exclude='.DS_Store' \
    --exclude='.env' \
    "$source_dir/" "$dest_dir/"
done

find "$TMP_ROOT" -type d -name '__pycache__' -prune -exec rm -rf {} + 2>/dev/null || true
find "$TMP_ROOT" -type f \( -name '*.pyc' -o -name '*.pyo' -o -name '*.pyd' \) -delete 2>/dev/null || true

rm -rf "$DEST_ROOT"
mv "$TMP_ROOT" "$DEST_ROOT"

echo "Attested runtime staged:"
find "$DEST_ROOT" -type f | wc -l | awk '{print "  files: " $1}'
