#!/bin/bash
# Execute only on a fresh isolated runner; never reclaim a production Docker host.
set -Eeuo pipefail

test "$(uname -m)" = x86_64
case "${GATEWAY_V2_BUILDER_DOMAIN:?}" in
  gateway)
    test "${RUNNER_ENVIRONMENT:-}" = github-hosted
    ;;
  validator)
    test "$(aws sts get-caller-identity --query Account --output text)" = 899952150628
    ;;
  *) echo 'ERROR: unexpected build domain' >&2; exit 1 ;;
esac
python3 .github/scripts/verify_attested_builder_isolation.py
# Neither build needs AWS credentials after the isolated-account check.
unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN AWS_PROFILE
export AWS_EC2_METADATA_DISABLED=true
export AWS_SHARED_CREDENTIALS_FILE=/dev/null
export AWS_CONFIG_FILE=/dev/null

test -n "${RUNNER_TEMP:?}"
export PATH="$RUNNER_TEMP/nitro-toolchain/bin:$PATH"
export NITRO_CLI_BLOBS="$RUNNER_TEMP/nitro-toolchain/blobs"
export NITRO_CLI_ARTIFACTS="$RUNNER_TEMP/nitro-artifacts"
export NITRO_CLI_LOGS_PATH="$RUNNER_TEMP/nitro-logs"
export NITRO_CLI_SOCKETS_PATH="$RUNNER_TEMP/nitro-sockets"
mkdir -p "$NITRO_CLI_ARTIFACTS" "$NITRO_CLI_LOGS_PATH" "$NITRO_CLI_SOCKETS_PATH"
python3 .github/scripts/verify_attested_nitro_toolchain.py \
  --nitro-cli "$RUNNER_TEMP/nitro-toolchain/bin/nitro-cli" \
  --blobs "$NITRO_CLI_BLOBS"

existing_containers="$(docker ps -aq)"
test -z "$existing_containers" || {
  echo 'ERROR: release builder must start without Docker containers' >&2
  exit 1
}
registry_mirrors="$(docker info --format '{{json .RegistryConfig.Mirrors}}')"
case "$registry_mirrors" in
  '[]'|'null') ;;
  *) echo 'ERROR: release builder must not use a Docker registry cache mirror' >&2; exit 1 ;;
esac
docker version --format 'Client={{.Client.Version}} Server={{.Server.Version}}'
free -m
df -h "$RUNNER_TEMP"

export GATEWAY_V2_BUILD_REVISION="${BUILD_COMMIT:?}"
export GATEWAY_V2_BUILDER_ID="${GATEWAY_V2_BUILDER_DOMAIN}-parent-${GITHUB_RUN_ID:?}-${GITHUB_RUN_ATTEMPT:?}"
export GATEWAY_V2_OFFLINE_ARTIFACT_ROOT="$RUNNER_TEMP/offline-artifacts"
export GATEWAY_V2_BUILD_EVIDENCE_DIR="$RUNNER_TEMP/gateway-evidence"
export GATEWAY_V2_BUILD_WORK_ROOT="$RUNNER_TEMP/gateway-build-work"
export GATEWAY_V2_PCR0_CACHE_FILE="$RUNNER_TEMP/gateway-pcr0-cache.json"
export LEADPOET_DOCKER_OPERATION_LOCK_FILE="$RUNNER_TEMP/docker-operation.lock"
bash gateway/tee/prepare_offline_enclave_dependencies.sh
. validator_tee/scripts/docker_operation_lock_v2.sh
leadpoet_acquire_docker_operation_lock_v2
trap 'leadpoet_release_docker_operation_lock_v2' EXIT
/usr/bin/time -v bash gateway/tee/build_release_evidence_v2.sh
free -m
df -h "$RUNNER_TEMP"
