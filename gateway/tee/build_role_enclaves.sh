#!/bin/bash
# Build every gateway V2 role EIF from one normalized, clean Git source tree.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GATEWAY_ROOT="${GATEWAY_ROOT:-/home/ec2-user/gateway}"
EIF_ROOT="${GATEWAY_TEE_EIF_ROOT:-/home/ec2-user/tee}"
RELEASE_MANIFEST="${GATEWAY_V2_RELEASE_MANIFEST:-$EIF_ROOT/gateway-v2-release-manifest.json}"
RELEASE_ARCHIVE_ROOT="${GATEWAY_V2_RELEASE_ARCHIVE_ROOT:-$EIF_ROOT/releases-v2}"
TOPOLOGY_MODE="${GATEWAY_TEE_TOPOLOGY_MODE:-full}"
ROLES=(
  gateway_coordinator
  gateway_scoring_a
  gateway_scoring_b
  gateway_autoresearch
)

python3 "$SCRIPT_DIR/topology.py" --verify "$SCRIPT_DIR/topology.json"
if [ "$TOPOLOGY_MODE" = "full" ]; then
  test -s "$RELEASE_MANIFEST" || {
    echo "ERROR: approved six-build V2 release manifest is missing: $RELEASE_MANIFEST" >&2
    exit 1
  }
  python3 "$SCRIPT_DIR/release_manifest_v2.py" --verify "$RELEASE_MANIFEST"
fi

if [ "${GATEWAY_TEE_SKIP_STAGE:-0}" != "1" ]; then
  bash "$SCRIPT_DIR/stage_attested_runtime.sh"
fi

mkdir -p "$EIF_ROOT"
for role in "${ROLES[@]}"; do
  image="tee-enclave:${role}"
  output="$EIF_ROOT/tee-enclave-${role}.eif"
  measurements="$EIF_ROOT/enclave-build-${role}.json"
  rm -f "$output" "$measurements"
  sudo docker build \
    --no-cache \
    --build-arg "LEADPOET_ENCLAVE_ROLE=${role}" \
    -f "$GATEWAY_ROOT/tee/Dockerfile.enclave" \
    -t "$image" \
    "$GATEWAY_ROOT/"
  sudo docker image inspect -f '{{.Id}}' "$image" \
    > "$EIF_ROOT/enclave-image-${role}.txt"
  sudo nitro-cli build-enclave \
    --docker-uri "$image" \
    --output-file "$output" \
    | tee "$measurements"
done

if [ "$TOPOLOGY_MODE" = "full" ]; then
  python3 "$SCRIPT_DIR/verify_release_artifacts_v2.py" \
    --release-manifest "$RELEASE_MANIFEST" \
    --gateway-root "$GATEWAY_ROOT" \
    --eif-root "$EIF_ROOT" \
    --output "$EIF_ROOT/gateway-v2-local-verification.json"
  PYTHONPATH="${GATEWAY_ROOT%/gateway}" python3 -m gateway.tee.release_archive_v2 \
    --archive \
    --release-manifest "$RELEASE_MANIFEST" \
    --gateway-root "$GATEWAY_ROOT" \
    --eif-root "$EIF_ROOT" \
    --archive-root "$RELEASE_ARCHIVE_ROOT" \
    --retain 3
fi

echo "Built role EIFs:"
for role in "${ROLES[@]}"; do
  test -s "$EIF_ROOT/tee-enclave-${role}.eif"
  printf '  %s: ' "$role"
  grep -m1 'PCR0' "$EIF_ROOT/enclave-build-${role}.json" || true
done
