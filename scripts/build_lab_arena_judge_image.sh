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
