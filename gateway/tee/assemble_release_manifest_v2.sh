#!/bin/bash
# Combine exact gateway- and validator-parent evidence into one approved release.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
GATEWAY_EVIDENCE="${1:-}"
VALIDATOR_EVIDENCE="${2:-}"
OUTPUT="${3:-}"
ACCEPTANCE_SIGNER_PUBKEY_HASH="${GATEWAY_V2_ACCEPTANCE_SIGNER_PUBKEY_HASH:-}"

if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <gateway-evidence.json> <validator-evidence.json> <release-manifest.json>" >&2
  exit 1
fi
test -s "$GATEWAY_EVIDENCE" || {
  echo "ERROR: gateway-parent build evidence is unavailable" >&2
  exit 1
}
test -s "$VALIDATOR_EVIDENCE" || {
  echo "ERROR: validator-parent build evidence is unavailable" >&2
  exit 1
}
if [[ ! "$ACCEPTANCE_SIGNER_PUBKEY_HASH" =~ ^sha256:[0-9a-f]{64}$ ]]; then
  echo "ERROR: GATEWAY_V2_ACCEPTANCE_SIGNER_PUBKEY_HASH is required" >&2
  exit 1
fi

mkdir -p "$(dirname "$OUTPUT")"
TEMP_OUTPUT="$(mktemp "$(dirname "$OUTPUT")/.gateway-release.XXXXXX")"
trap 'rm -f "$TEMP_OUTPUT"' EXIT
PYTHONPATH="$REPO_ROOT" python3 -m gateway.tee.release_manifest_v2 \
  --evidence "$GATEWAY_EVIDENCE" \
  --evidence "$VALIDATOR_EVIDENCE" \
  --acceptance-signer-pubkey-hash "$ACCEPTANCE_SIGNER_PUBKEY_HASH" \
  --output "$TEMP_OUTPUT"
PYTHONPATH="$REPO_ROOT" python3 -m gateway.tee.release_manifest_v2 \
  --verify "$TEMP_OUTPUT"
chmod 600 "$TEMP_OUTPUT"
mv -f "$TEMP_OUTPUT" "$OUTPUT"
trap - EXIT
echo "Approved gateway V2 release manifest: $OUTPUT"
