#!/bin/bash
# Build every gateway V2 role EIF from one normalized, clean Git source tree.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GATEWAY_ROOT="${GATEWAY_ROOT:-/home/ec2-user/gateway}"
REPO_ROOT="${GATEWAY_ROOT%/gateway}"
EIF_ROOT="${GATEWAY_TEE_EIF_ROOT:-/home/ec2-user/tee}"
RELEASE_MANIFEST="${GATEWAY_V2_RELEASE_MANIFEST:-$EIF_ROOT/gateway-v2-release-manifest.json}"
RELEASE_ARCHIVE_ROOT="${GATEWAY_V2_RELEASE_ARCHIVE_ROOT:-$EIF_ROOT/releases-v2}"
LAST_GOOD_MANIFEST="${GATEWAY_LAST_GOOD_MANIFEST:-/home/ec2-user/.config/leadpoet/deployments/gateway-last-good.json}"
TOPOLOGY_MODE="${GATEWAY_TEE_TOPOLOGY_MODE:-full}"
ROLES=(
  gateway_coordinator
  gateway_scoring
)

publish_built_eif_for_verification() {
  local artifact="$1"
  local owner
  owner="$(id -u):$(id -g)"

  test -f "$artifact" && test ! -L "$artifact" && test -s "$artifact" || {
    echo "ERROR: built gateway EIF is unavailable or unsafe: $artifact" >&2
    return 1
  }
  sudo chown --no-dereference -- "$owner" "$artifact"
  chmod 0600 -- "$artifact"
  test -f "$artifact" && test ! -L "$artifact" && test -s "$artifact" \
    && test -O "$artifact" && test -G "$artifact" && test -r "$artifact" || {
    echo "ERROR: built gateway EIF is not privately readable by the verifier: $artifact" >&2
    return 1
  }
}

. "$REPO_ROOT/validator_tee/scripts/docker_operation_lock_v2.sh"
leadpoet_acquire_docker_operation_lock_v2

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
test -d "$EIF_ROOT" && test ! -L "$EIF_ROOT" && test -O "$EIF_ROOT" || {
  echo "ERROR: gateway EIF root is unavailable or unsafe: $EIF_ROOT" >&2
  exit 1
}
COLD_BUILD_ROOT=""
cleanup_cold_build_root() {
  if [ -z "$COLD_BUILD_ROOT" ]; then
    return 0
  fi
  case "$COLD_BUILD_ROOT" in
    "$EIF_ROOT"/.gateway-eif-cold-build.*)
      test -d "$COLD_BUILD_ROOT" && test ! -L "$COLD_BUILD_ROOT" \
        && test -O "$COLD_BUILD_ROOT" || return 1
      rm -rf -- "$COLD_BUILD_ROOT"
      ;;
    *)
      echo "ERROR: gateway EIF cold-build root is unsafe" >&2
      return 1
      ;;
  esac
}
trap cleanup_cold_build_root EXIT

# A prior interrupted cold build can leave root-owned final-path files behind.
# Make only the fixed, regular artifacts readable so the transactional
# restore path can retain or replace the complete previous set.
for role in "${ROLES[@]}"; do
  live_output="$EIF_ROOT/tee-enclave-${role}.eif"
  if [ -e "$live_output" ] && ! test -O "$live_output"; then
    publish_built_eif_for_verification "$live_output"
  fi
done

RESTORED_EXACT_RELEASE=0
if [ "$TOPOLOGY_MODE" = "full" ]; then
  echo "Checking for a verified exact-release gateway EIF archive"
  if PYTHONPATH="${GATEWAY_ROOT%/gateway}" python3 \
      -m gateway.tee.release_archive_v2 \
      --restore \
      --release-manifest "$RELEASE_MANIFEST" \
      --gateway-root "$GATEWAY_ROOT" \
      --eif-root "$EIF_ROOT" \
      --archive-root "$RELEASE_ARCHIVE_ROOT"; then
    RESTORED_EXACT_RELEASE=1
    echo "Restored the exact verified gateway EIF release"
  else
    restore_status="$?"
    if [ "$restore_status" -ne 3 ]; then
      echo "ERROR: exact gateway EIF archive verification or restore failed" >&2
      exit "$restore_status"
    fi
    echo "No exact gateway EIF archive is retained; performing a cold build"
  fi
fi

if [ "$RESTORED_EXACT_RELEASE" != "1" ]; then
  BUILD_EIF_ROOT="$EIF_ROOT"
  if [ "$TOPOLOGY_MODE" = "full" ]; then
    COLD_BUILD_ROOT="$(mktemp -d "$EIF_ROOT/.gateway-eif-cold-build.XXXXXXXX")"
    chmod 0700 "$COLD_BUILD_ROOT"
    BUILD_EIF_ROOT="$COLD_BUILD_ROOT"
  fi
  for role in "${ROLES[@]}"; do
    image="tee-enclave:${role}"
    raw_image="${image}-raw"
    output="$BUILD_EIF_ROOT/tee-enclave-${role}.eif"
    measurements="$BUILD_EIF_ROOT/enclave-build-${role}.json"
    rm -f "$output" "$measurements"
    sudo env \
      DOCKER_BUILDKIT=1 \
      BUILDX_NO_DEFAULT_ATTESTATIONS=1 \
      docker build \
      --pull \
      --no-cache \
      --build-arg "SOURCE_DATE_EPOCH=0" \
      --build-arg "LEADPOET_ENCLAVE_ROLE=${role}" \
      -f "$GATEWAY_ROOT/tee/Dockerfile.enclave" \
      -t "$raw_image" \
      "$GATEWAY_ROOT/"
    sudo python3 \
      "${GATEWAY_ROOT%/gateway}/validator_tee/host/docker_image_normalizer_v2.py" \
      --source-image "$raw_image" \
      --normalized-image "$image"
    sudo docker image inspect -f '{{.Id}}' "$image" \
      > "$BUILD_EIF_ROOT/enclave-image-${role}.txt"
    sudo nitro-cli build-enclave \
      --docker-uri "$image" \
      --output-file "$output" \
      | tee "$measurements"
    # nitro-cli runs under sudo and creates the EIF as root:root 0600.  The
    # release verifier and archive writer intentionally run without sudo, so
    # publish the completed regular file to that same invoking identity before
    # either consumer reads it.
    publish_built_eif_for_verification "$output"
    nitro-cli describe-eif --eif-path "$output" >/dev/null
    sudo docker rmi -f "$raw_image" >/dev/null 2>&1 || true
  done

  if [ "$TOPOLOGY_MODE" = "full" ]; then
    python3 "$SCRIPT_DIR/verify_release_artifacts_v2.py" \
      --release-manifest "$RELEASE_MANIFEST" \
      --gateway-root "$GATEWAY_ROOT" \
      --eif-root "$BUILD_EIF_ROOT" \
      --output "$BUILD_EIF_ROOT/gateway-v2-local-verification.json"
    PYTHONPATH="${GATEWAY_ROOT%/gateway}" python3 -m gateway.tee.release_archive_v2 \
      --archive \
      --release-manifest "$RELEASE_MANIFEST" \
      --gateway-root "$GATEWAY_ROOT" \
      --eif-root "$BUILD_EIF_ROOT" \
      --archive-root "$RELEASE_ARCHIVE_ROOT" \
      --last-good-manifest "$LAST_GOOD_MANIFEST" \
      --retain 3
    PYTHONPATH="${GATEWAY_ROOT%/gateway}" python3 -m gateway.tee.release_archive_v2 \
      --restore \
      --release-manifest "$RELEASE_MANIFEST" \
      --gateway-root "$GATEWAY_ROOT" \
      --eif-root "$EIF_ROOT" \
      --archive-root "$RELEASE_ARCHIVE_ROOT"
    cleanup_cold_build_root
    COLD_BUILD_ROOT=""
  fi
fi

echo "Built role EIFs:"
for role in "${ROLES[@]}"; do
  test -s "$EIF_ROOT/tee-enclave-${role}.eif"
  printf '  %s: ' "$role"
  grep -m1 'PCR0' "$EIF_ROOT/enclave-build-${role}.json" || true
done
