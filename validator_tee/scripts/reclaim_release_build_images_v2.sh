#!/bin/bash
# Remove only Docker images created by one release-evidence build.

set -euo pipefail

MODE="${1:-}"
INVENTORY_PATH="${2:-}"
MAX_INVENTORY_BYTES=1048576

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

require_exclusive_lock() {
  local lock_file observed_lock

  lock_file="${LEADPOET_DOCKER_OPERATION_LOCK_FILE:-}"
  [ "${LEADPOET_DOCKER_OPERATION_LOCK_HELD:-0}" = "1" ] \
    || fail "release image cleanup requires the Docker operation lock"
  [ "${LEADPOET_DOCKER_OPERATION_ADMISSION_LOCK_HELD:-0}" = "1" ] \
    || fail "release image cleanup requires Docker maintenance admission"
  [ -n "$lock_file" ] \
    || fail "release image cleanup lock path is unavailable"
  observed_lock="$(readlink "/proc/$$/fd/7" 2>/dev/null || true)"
  [ "$observed_lock" = "$lock_file" ] \
    || fail "release image cleanup has no inherited Docker lock descriptor"
}

validate_inventory() {
  local path="$1"
  local size

  [ -f "$path" ] && [ ! -L "$path" ] \
    || fail "release image inventory is unavailable"
  size="$(wc -c < "$path" | tr -d '[:space:]')"
  [[ "$size" =~ ^[0-9]+$ ]] && [ "$size" -le "$MAX_INVENTORY_BYTES" ] \
    || fail "release image inventory exceeds its size bound"
  if grep -Ev '^sha256:[0-9a-f]{64}$' "$path" | grep -q .; then
    fail "release image inventory contains an invalid image id"
  fi
  LC_ALL=C sort -c -u "$path" \
    || fail "release image inventory is not canonical"
}

write_current_inventory() {
  local destination="$1"

  docker image ls --all --no-trunc --quiet \
    | LC_ALL=C sort -u > "$destination"
  validate_inventory "$destination"
}

require_exclusive_lock
[ -n "$INVENTORY_PATH" ] \
  || fail "release image inventory path is required"
[[ "$INVENTORY_PATH" = /* ]] \
  || fail "release image inventory path must be absolute"
[ -d "$(dirname "$INVENTORY_PATH")" ] \
  || fail "release image inventory directory is unavailable"

case "$MODE" in
  snapshot)
    [ ! -e "$INVENTORY_PATH" ] && [ ! -L "$INVENTORY_PATH" ] \
      || fail "release image inventory already exists"
    temporary="$(mktemp "$(dirname "$INVENTORY_PATH")/.release-images.XXXXXX")"
    trap 'rm -f -- "$temporary"' EXIT
    write_current_inventory "$temporary"
    chmod 600 "$temporary"
    mv -- "$temporary" "$INVENTORY_PATH"
    trap - EXIT
    ;;
  cleanup)
    validate_inventory "$INVENTORY_PATH"
    current="$(mktemp "$(dirname "$INVENTORY_PATH")/.release-images-current.XXXXXX")"
    created="$(mktemp "$(dirname "$INVENTORY_PATH")/.release-images-created.XXXXXX")"
    after="$(mktemp "$(dirname "$INVENTORY_PATH")/.release-images-after.XXXXXX")"
    missing="$(mktemp "$(dirname "$INVENTORY_PATH")/.release-images-missing.XXXXXX")"
    trap 'rm -f -- "$current" "$created" "$after" "$missing"' EXIT
    write_current_inventory "$current"
    LC_ALL=C comm -13 "$INVENTORY_PATH" "$current" > "$created"
    validate_inventory "$created"
    while IFS= read -r image_id; do
      [ -n "$image_id" ] || continue
      if grep -Fqx -- "$image_id" "$INVENTORY_PATH"; then
        fail "release image cleanup selected a pre-existing image"
      fi
      docker image rm --force -- "$image_id"
    done < "$created"
    write_current_inventory "$after"
    LC_ALL=C comm -23 "$INVENTORY_PATH" "$after" > "$missing"
    if [ -s "$missing" ]; then
      fail "release build removed a pre-existing Docker image"
    fi
    ;;
  *)
    fail "usage: reclaim_release_build_images_v2.sh snapshot|cleanup /absolute/inventory"
    ;;
esac
