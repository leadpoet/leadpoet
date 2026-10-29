#!/bin/bash
# Both verifier and protected publisher run this trusted exact-main helper.
set -Eeuo pipefail

test -n "${BUILD_COMMIT:?}"
test -n "${GITHUB_RUN_ID:?}"
test -n "${GITHUB_RUN_ATTEMPT:?}"
test -n "${RUNNER_TEMP:?}"
expected_args=()
if [ -n "${EXPECTED_PCR0:-}" ]; then
  expected_args=(--expected-pcr0 "$EXPECTED_PCR0")
fi
python3 .github/scripts/verify_attested_release_evidence.py \
  --gateway-artifact "$RUNNER_TEMP/evidence/gateway" \
  --validator-artifact "$RUNNER_TEMP/evidence/validator" \
  --gateway-builder-id "gateway-parent-${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}" \
  --validator-builder-id "validator-parent-${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}" \
  --commit "$BUILD_COMMIT" \
  --acceptance-signer-pubkey-hash "${ACCEPTANCE_SIGNER_PUBKEY_HASH:?}" \
  "${expected_args[@]}"

mkdir -p "$RUNNER_TEMP/release"
GATEWAY_V2_ACCEPTANCE_SIGNER_PUBKEY_HASH="$ACCEPTANCE_SIGNER_PUBKEY_HASH" \
  bash gateway/tee/assemble_release_manifest_v2.sh \
    "$RUNNER_TEMP/evidence/gateway/gateway-gateway-evidence.json" \
    "$RUNNER_TEMP/evidence/validator/gateway-validator-evidence.json" \
    "$RUNNER_TEMP/release/gateway-v2-release-manifest.json"
python3 -m gateway.tee.release_channel_v2 \
  --build \
  --gateway-manifest "$RUNNER_TEMP/release/gateway-v2-release-manifest.json" \
  --output "$RUNNER_TEMP/release/release-channel-v2.json"
python3 -m gateway.tee.release_channel_v2 \
  --verify "$RUNNER_TEMP/release/release-channel-v2.json" \
  --expected-commit "$BUILD_COMMIT"
