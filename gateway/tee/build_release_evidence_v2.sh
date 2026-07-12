#!/bin/bash
# Produce the twelve gateway-role build records for one independent parent.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILDER_DOMAIN="${GATEWAY_V2_BUILDER_DOMAIN:-}"
BUILDER_ID="${GATEWAY_V2_BUILDER_ID:-}"
REVISION="${GATEWAY_V2_BUILD_REVISION:-HEAD}"
OUTPUT_DIR="${GATEWAY_V2_BUILD_EVIDENCE_DIR:-$HOME/.cache/leadpoet/gateway-release-evidence-v2}"
WORK_ROOT="${GATEWAY_V2_BUILD_WORK_ROOT:-$HOME/.cache/leadpoet/gateway-release-build-v2}"
CACHE_FILE="${GATEWAY_V2_PCR0_CACHE_FILE:-$HOME/.cache/leadpoet/gateway-pcr0-cache.json}"

case "$BUILDER_DOMAIN" in
  gateway|validator) ;;
  *) echo "ERROR: GATEWAY_V2_BUILDER_DOMAIN must be gateway or validator" >&2; exit 1 ;;
esac
if [[ ! "$BUILDER_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$ ]]; then
  echo "ERROR: GATEWAY_V2_BUILDER_ID is invalid" >&2
  exit 1
fi
if [ -n "$(git -C "$REPO_ROOT" status --porcelain --untracked-files=no)" ]; then
  echo "ERROR: gateway release evidence requires a clean tracked checkout" >&2
  exit 1
fi

COMMIT="$(git -C "$REPO_ROOT" rev-parse "${REVISION}^{commit}")"
mkdir -p "$OUTPUT_DIR" "$WORK_ROOT"
RESULT_FILE="$OUTPUT_DIR/gateway-${BUILDER_DOMAIN}-build-result.json"
EVIDENCE_FILE="$OUTPUT_DIR/gateway-${BUILDER_DOMAIN}-evidence.json"
TEMP_RESULT="$(mktemp "$OUTPUT_DIR/.gateway-build-result.XXXXXX")"
TEMP_EVIDENCE="$(mktemp "$OUTPUT_DIR/.gateway-evidence.XXXXXX")"
trap 'rm -f "$TEMP_RESULT" "$TEMP_EVIDENCE"' EXIT

echo "Building every gateway V2 role three times on ${BUILDER_DOMAIN} parent ${BUILDER_ID}"
PYTHONPATH="$REPO_ROOT" python3 -m validator_tee.host.gateway_pcr0_builder \
  --repo-root "$REPO_ROOT" \
  --revision "$COMMIT" \
  --work-root "$WORK_ROOT/$COMMIT" \
  --cache-file "$CACHE_FILE" \
  --repetitions 3 \
  --builder-domain "$BUILDER_DOMAIN" \
  --builder-id "$BUILDER_ID" \
  --all-roles \
  > "$TEMP_RESULT"

PYTHONPATH="$REPO_ROOT" python3 - \
  "$TEMP_RESULT" "$TEMP_EVIDENCE" "$COMMIT" "$BUILDER_DOMAIN" "$BUILDER_ID" <<'PY'
import json
import os
import sys
from pathlib import Path

from gateway.tee.release_manifest_v2 import normalize_build_evidence
from gateway.tee.topology import ROLE_SPECS

source, destination, commit, domain, builder_id = sys.argv[1:]
value = json.loads(Path(source).read_text(encoding="utf-8"))
if not isinstance(value, list) or len(value) != len(ROLE_SPECS):
    raise SystemExit("gateway builder did not return every physical role")
records = []
for result in value:
    if not isinstance(result, dict) or result.get("verified_build_count") != 3:
        raise SystemExit("gateway role lacks three verified builds")
    evidence = result.get("build_evidence")
    if not isinstance(evidence, list) or len(evidence) != 3:
        raise SystemExit("gateway role evidence is incomplete")
    records.extend(normalize_build_evidence(item) for item in evidence)
expected = {
    (role, ordinal)
    for role in ROLE_SPECS
    for ordinal in (1, 2, 3)
}
observed = {(item["physical_role"], item["build_ordinal"]) for item in records}
if observed != expected or len(records) != len(expected):
    raise SystemExit("gateway role/domain/ordinal evidence coverage is incomplete")
if {
    (item["commit_sha"], item["builder_domain"], item["builder_id"])
    for item in records
} != {(commit, domain, builder_id)}:
    raise SystemExit("gateway evidence identity differs from requested build")
Path(destination).write_text(
    json.dumps(records, sort_keys=True, indent=2) + "\n",
    encoding="utf-8",
)
os.chmod(destination, 0o600)
PY

mv -f "$TEMP_RESULT" "$RESULT_FILE"
mv -f "$TEMP_EVIDENCE" "$EVIDENCE_FILE"
trap - EXIT
echo "Gateway V2 evidence ready: $EVIDENCE_FILE"
echo "Gateway V2 build commit: $COMMIT"
