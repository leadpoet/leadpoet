#!/bin/bash
# Produce three clean validator EIF build-evidence records on one parent host.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VALIDATOR_TEE_DIR="$(dirname "$SCRIPT_DIR")"
REPO_ROOT="$(dirname "$VALIDATOR_TEE_DIR")"
BUILDER_DOMAIN="${VALIDATOR_V2_BUILDER_DOMAIN:-}"
BUILDER_ID="${VALIDATOR_V2_BUILDER_ID:-}"
OUTPUT_DIR="${VALIDATOR_V2_BUILD_EVIDENCE_DIR:-$VALIDATOR_TEE_DIR/release-evidence-v2}"

case "$BUILDER_DOMAIN" in
  gateway|validator) ;;
  *) echo "ERROR: VALIDATOR_V2_BUILDER_DOMAIN must be gateway or validator" >&2; exit 1 ;;
esac
if [[ ! "$BUILDER_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$ ]]; then
  echo "ERROR: VALIDATOR_V2_BUILDER_ID is invalid" >&2
  exit 1
fi
if [ -n "$(git -C "$REPO_ROOT" status --porcelain --untracked-files=no)" ]; then
  echo "ERROR: validator release evidence requires a clean tracked checkout" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
rm -f "$OUTPUT_DIR"/validator-"$BUILDER_DOMAIN"-*.json

for ordinal in 1 2 3; do
  echo "Building clean validator V2 release evidence ${ordinal}/3 on ${BUILDER_DOMAIN}"
  docker rmi -f \
    validator-base:v1 \
    validator-tee-enclave:raw \
    validator-tee-enclave:latest 2>/dev/null || true
  docker builder prune -af >/dev/null
  rm -f "$REPO_ROOT/.validator-base.dockerfile.sha256"
  bash "$SCRIPT_DIR/build_enclave.sh"
  python3 -m validator_tee.host.verify_release_gate_v2 \
    --emit-evidence \
    --local-release "$VALIDATOR_TEE_DIR/validator-v2-release.json" \
    --builder-domain "$BUILDER_DOMAIN" \
    --builder-id "$BUILDER_ID" \
    --build-ordinal "$ordinal" \
    --output "$OUTPUT_DIR/validator-${BUILDER_DOMAIN}-${ordinal}.json"
done

python3 - "$OUTPUT_DIR" "$BUILDER_DOMAIN" <<'PY'
import json
import sys
from pathlib import Path

from validator_tee.host.release_v2 import DETERMINISTIC_RELEASE_FIELDS

root = Path(sys.argv[1])
domain = sys.argv[2]
records = [
    json.loads((root / f"validator-{domain}-{ordinal}.json").read_text())
    for ordinal in (1, 2, 3)
]
releases = [record["release"] for record in records]
for field in DETERMINISTIC_RELEASE_FIELDS:
    if len({release[field] for release in releases}) != 1:
        raise SystemExit(f"three clean validator builds diverged at {field}")
print("validator_v2_three_build_release_hash=" + releases[0]["release_hash"])
PY

echo "Three clean ${BUILDER_DOMAIN} validator build records are ready in $OUTPUT_DIR"
