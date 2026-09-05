#!/bin/bash
# Build one exact local identity for each gateway role and the validator.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CANDIDATE_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPOSITORY=""
REVISION=""
GATEWAY_OUTPUT=""
VALIDATOR_OUTPUT=""

while [ "$#" -gt 0 ]; do
  case "$1" in
    --repository) REPOSITORY="${2:-}"; shift 2 ;;
    --revision) REVISION="${2:-}"; shift 2 ;;
    --gateway-output) GATEWAY_OUTPUT="${2:-}"; shift 2 ;;
    --validator-output) VALIDATOR_OUTPUT="${2:-}"; shift 2 ;;
    *) echo "ERROR: unsupported local release argument: $1" >&2; exit 2 ;;
  esac
done

if [ -z "$REPOSITORY" ] || [ -z "$REVISION" ] \
    || [ -z "$GATEWAY_OUTPUT" ] || [ -z "$VALIDATOR_OUTPUT" ] \
    || ! [[ "$REVISION" =~ ^[0-9a-f]{40}$ ]]; then
  echo "ERROR: local release inputs are incomplete" >&2
  exit 2
fi
test "$(git -C "$REPOSITORY" rev-parse "$REVISION^{commit}")" = "$REVISION"

if [ "${LEADPOET_LOCAL_RELEASE_FROZEN_REVISION:-}" != "$REVISION" ]; then
  FROZEN_SOURCE_ROOT="$(mktemp -d /tmp/leadpoet-local-release-source.XXXXXX)"
  cleanup_frozen_source() {
    chmod -R u+w "$FROZEN_SOURCE_ROOT" 2>/dev/null || true
    rm -rf -- "$FROZEN_SOURCE_ROOT"
  }
  trap cleanup_frozen_source EXIT
  chmod 700 "$FROZEN_SOURCE_ROOT"
  git -C "$REPOSITORY" archive "$REVISION" | tar -xf - -C "$FROZEN_SOURCE_ROOT"
  test "$(git -C "$REPOSITORY" hash-object --no-filters \
    "$FROZEN_SOURCE_ROOT/gateway/tee/build_local_release_v2.sh")" \
    = "$(git -C "$REPOSITORY" rev-parse \
      "$REVISION:gateway/tee/build_local_release_v2.sh")"
  LEADPOET_LOCAL_RELEASE_FROZEN_REVISION="$REVISION" \
    /bin/bash "$FROZEN_SOURCE_ROOT/gateway/tee/build_local_release_v2.sh" \
      --repository "$REPOSITORY" \
      --revision "$REVISION" \
      --gateway-output "$GATEWAY_OUTPUT" \
      --validator-output "$VALIDATOR_OUTPUT"
  exit
fi
cd "$CANDIDATE_ROOT"

WORK_ROOT="${GATEWAY_V2_BUILD_WORK_ROOT:-$HOME/.cache/leadpoet/gateway-release-build-v2}"
TEMPORARY_ROOT="$(mktemp -d /tmp/leadpoet-local-release.XXXXXX)"
. "$CANDIDATE_ROOT/validator_tee/scripts/docker_operation_lock_v2.sh"
leadpoet_acquire_docker_operation_lock_v2
cleanup() {
  rm -f -- \
    "$CANDIDATE_ROOT/.validator-base.dockerfile.sha256" \
    "$CANDIDATE_ROOT/validator_tee/validator-enclave.eif" \
    "$CANDIDATE_ROOT/validator_tee/enclave_build_output.txt" \
    "$CANDIDATE_ROOT/validator_tee/validator-v2-release.json"
  rm -rf -- "$CANDIDATE_ROOT/.validator-tee-artifacts"
  rm -rf -- "$TEMPORARY_ROOT"
  leadpoet_release_docker_operation_lock_v2 || true
}
trap cleanup EXIT

echo "Building one local gateway identity per role for $REVISION"
PYTHONPATH="$CANDIDATE_ROOT" python3 -m validator_tee.host.gateway_pcr0_builder \
  --repo-root "$REPOSITORY" \
  --revision "$REVISION" \
  --work-root "$WORK_ROOT/$REVISION-local" \
  --cache-file "$TEMPORARY_ROOT/gateway-cache.json" \
  --repetitions 1 \
  --builder-domain local \
  --builder-id local-restart \
  --all-roles \
  --output-file "$TEMPORARY_ROOT/gateway-builds.json"

echo "Building one local validator identity for $REVISION"
VALIDATOR_V2_BUILD_COMMIT="$REVISION" \
  bash "$CANDIDATE_ROOT/validator_tee/scripts/build_enclave.sh"

PYTHONPATH="$CANDIDATE_ROOT" python3 -m gateway.tee.local_release_v2 \
  --gateway-build-results "$TEMPORARY_ROOT/gateway-builds.json" \
  --validator-release "$CANDIDATE_ROOT/validator_tee/validator-v2-release.json" \
  --gateway-output "$GATEWAY_OUTPUT" \
  --validator-output "$VALIDATOR_OUTPUT"
