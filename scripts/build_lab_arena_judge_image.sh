#!/usr/bin/env bash
# Build and publish the Lab Arena judge image; print the digest-pinned reference.
#
#   bash scripts/build_lab_arena_judge_image.sh <registry>/<repository> [--no-push]
#
# The image is built for linux/amd64 from lab_arena/judge/Dockerfile at the
# current commit, without provenance or SBOM attestations so the pushed
# reference is a single-platform manifest. The last line of output is the
# value for LAB_ARENA_SCORER_IMAGE; the service resolves it at startup and
# pins its digest and entry command on every round configuration.
set -euo pipefail

REPOSITORY="${1:?usage: $0 <registry>/<repository> [--no-push]}"
PUSH=1
if [[ "${2:-}" == "--no-push" ]]; then
  PUSH=0
fi
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMMIT="$(git -C "$ROOT" rev-parse HEAD)"
TAG="${REPOSITORY}:judge-${COMMIT:0:12}"

# Shared gateway/validator builders use the existing host maintenance contract.
# Local Docker builds on macOS and unmanaged Linux hosts keep their behavior.
if [[ "$(uname -s)" == Linux ]] && {
  [[ -n "${LEADPOET_DOCKER_OPERATION_LOCK_FILE:-}" ]] \
    || [[ -d /home/ec2-user/.config/leadpoet ]];
}; then
  . "$ROOT/validator_tee/scripts/docker_operation_lock_v2.sh"
  leadpoet_acquire_docker_operation_lock_v2 >&2
  PYTHONPATH="$ROOT" python3 - \
    "$LEADPOET_DOCKER_OPERATION_LOCK_FILE" \
    "$LEADPOET_DOCKER_OPERATION_ADMISSION_LOCK_FILE" "$$" <<'PY'
from pathlib import Path
import sys
from validator_tee.host.docker_zero_runtime_reconciler_v2 import verify_docker_operation_locks_v2

verify_docker_operation_locks_v2(
    lock_file=Path(sys.argv[1]), admission_lock_file=Path(sys.argv[2]),
    owner_pid=int(sys.argv[3]),
)
PY
  python3 "$ROOT/validator_tee/host/docker_operation_guard_v2.py" \
    --wait --timeout-seconds 1800 --interval-seconds 3 >&2
  BUILD_INVENTORY_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/leadpoet-arena-judge.XXXXXX")"
  BUILD_IMAGE_INVENTORY="$BUILD_INVENTORY_ROOT/images.txt"
  BUILD_INVENTORY_READY=0
  cleanup_build() {
    local build_status="$?" cleanup_status=0
    trap - EXIT
    if [[ "$BUILD_INVENTORY_READY" == 1 ]]; then
      # A successful --no-push build intentionally leaves its local output.
      if [[ "$PUSH" == 1 || "$build_status" != 0 ]]; then
        bash "$ROOT/validator_tee/scripts/reclaim_release_build_images_v2.sh" \
          cleanup "$BUILD_IMAGE_INVENTORY" >&2 || cleanup_status="$?"
      fi
      if [[ "$cleanup_status" == 0 ]]; then
        docker builder prune --all --force >&2 || cleanup_status="$?"
      fi
    fi
    rm -rf -- "$BUILD_INVENTORY_ROOT" || cleanup_status="$?"
    leadpoet_release_docker_operation_lock_v2 >&2 || cleanup_status="$?"
    if [[ "$cleanup_status" != 0 ]]; then
      echo "ERROR: Arena judge build cleanup failed ($cleanup_status)" >&2
    fi
    if [[ "$build_status" != 0 ]]; then
      exit "$build_status"
    fi
    exit "$cleanup_status"
  }
  trap cleanup_build EXIT
  bash "$ROOT/validator_tee/scripts/reclaim_release_build_images_v2.sh" \
    snapshot "$BUILD_IMAGE_INVENTORY"
  BUILD_INVENTORY_READY=1
fi

# BuildKit attaches provenance and SBOM attestations by default, which turn
# the pushed reference into an index; the legacy builder never attaches them
# and does not know the flags.
ATTESTATION_FLAGS=()
if docker buildx version >/dev/null 2>&1; then
  ATTESTATION_FLAGS=(--provenance=false --sbom=false)
fi

docker build \
  --platform linux/amd64 \
  ${ATTESTATION_FLAGS[@]+"${ATTESTATION_FLAGS[@]}"} \
  --file "$ROOT/lab_arena/judge/Dockerfile" \
  --build-arg "LEADPOET_BUILD_COMMIT=${COMMIT}" \
  --tag "$TAG" \
  "$ROOT"

if [[ "$PUSH" == "1" ]]; then
  docker push "$TAG"
  REFERENCE="$(docker inspect --format '{{index .RepoDigests 0}}' "$TAG")"
  echo "LAB_ARENA_SCORER_IMAGE=${REFERENCE}"
else
  echo "built ${TAG} without pushing; a digest-pinned reference exists only after a push"
fi
