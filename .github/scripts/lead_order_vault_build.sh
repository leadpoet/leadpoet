#!/usr/bin/env bash
# Reproducible build of the lead-order key vault image.
#   lead_order_vault_build.sh        build twice, compare digests, write dist/
#   lead_order_vault_build.sh push   push the same build and check its digest
# Anyone can run the first form on the same commit and must get the same digest.
set -euo pipefail

IMAGE="${IMAGE:-ghcr.io/leadpoet/lead-order-vault}"
PLATFORM=linux/amd64
OUT=dist/lead-order-vault
# A different BuildKit can produce a different digest, so the builder is pinned.
BUILDKIT_VERSION=v0.35.1
BUILDKIT="moby/buildkit:$BUILDKIT_VERSION@sha256:baf6138eb8c37644800c7c913fba051e840ebb9ba13eaf92d92ef9a1712d0b4c"
BUILDER="lead-order-vault-$BUILDKIT_VERSION"
SOURCE_DATE_EPOCH="$(git log -1 --format=%ct)"
export SOURCE_DATE_EPOCH

builder() {
  docker buildx inspect "$BUILDER" >/dev/null 2>&1 \
    || docker buildx create --name "$BUILDER" --driver docker-container \
         --driver-opt "image=$BUILDKIT" >/dev/null
}

build() {  # $1: --output value, then any extra buildx flags
  out="$1"; shift
  docker buildx build --builder "$BUILDER" --platform "$PLATFORM" --no-cache \
    --provenance=false --sbom=false --build-arg SOURCE_DATE_EPOCH \
    -f lead_order_vault/Dockerfile --output "$out" "$@" .
}

digest_of() {  # manifest digest from an OCI layout tarball
  tar -xOf "$1" index.json | python3 -c 'import json,sys; print(json.load(sys.stdin)["manifests"][0]["digest"])'
}

builder
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

if [ "${1:-}" = "push" ]; then
  : "${DIGEST:?DIGEST from the compared build is required}" "${GHCR_TOKEN:?}"
  echo "$GHCR_TOKEN" | docker login ghcr.io -u "${GITHUB_ACTOR:-leadpoet}" --password-stdin >/dev/null
  # Pushed by digest only, with the same OCI media types as the compared
  # builds, so a mismatch leaves no tag behind.
  build "type=image,name=$IMAGE,push-by-digest=true,push=true,oci-mediatypes=true,rewrite-timestamp=true" \
    --metadata-file "$work/push.json"
  pushed="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["containerimage.digest"])' "$work/push.json")"
  [ "$pushed" = "$DIGEST" ] || { echo "pushed digest $pushed differs from built $DIGEST" >&2; exit 1; }
  echo "pushed $IMAGE@$DIGEST"
  exit 0
fi

mkdir -p "$OUT"
build "type=oci,dest=$work/a.tar,rewrite-timestamp=true"
build "type=oci,dest=$work/b.tar,rewrite-timestamp=true"
first="$(digest_of "$work/a.tar")"
second="$(digest_of "$work/b.tar")"
[ "$first" = "$second" ] || { echo "not reproducible: $first vs $second" >&2; exit 1; }

sed "s#^\(    image: \).*#\1$IMAGE@$first#" lead_order_vault/docker-compose.yml > "$OUT/docker-compose.yml"
grep -q "$IMAGE@$first" "$OUT/docker-compose.yml"
printf '{"commit":"%s","image":"%s@%s","source_date_epoch":%s,"buildkit":"%s"}\n' \
  "$(git rev-parse HEAD)" "$IMAGE" "$first" "$SOURCE_DATE_EPOCH" "$BUILDKIT" > "$OUT/release.json"
echo "reproducible digest $first"
if [ -n "${GITHUB_OUTPUT:-}" ]; then
  echo "digest=$first" >> "$GITHUB_OUTPUT"
fi
