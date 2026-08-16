#!/bin/bash

# Coordinate destructive Docker maintenance and image builds on shared hosts.
# The lock is inherited by child processes so one restart can cover its full
# Docker lifecycle without allowing a self-hosted runner to enter mid-build.

leadpoet_canonical_docker_lock_path_v2() {
  python3 - "$1" <<'PY'
import os
import sys

path = os.path.expanduser(str(sys.argv[1] or "").strip())
if not path or not os.path.isabs(path):
    raise SystemExit(1)
print(os.path.realpath(path))
PY
}

leadpoet_acquire_docker_operation_lock_v2() {
  local raw_lock_file raw_admission_lock_file lock_file admission_lock_file
  local lock_timeout lock_started_seconds
  local lock_elapsed_seconds lock_remaining_seconds observed_fd observed_admission_fd
  local owner_pid owner_fd owner_admission_fd

  raw_lock_file="${LEADPOET_DOCKER_OPERATION_LOCK_FILE:-/home/ec2-user/.config/leadpoet/docker-operation-v2.lock}"
  command -v python3 >/dev/null 2>&1 || {
    echo "ERROR: python3 is required for canonical Docker lock paths" >&2
    return 1
  }
  if ! lock_file="$(leadpoet_canonical_docker_lock_path_v2 "$raw_lock_file")"; then
    echo "ERROR: LEADPOET_DOCKER_OPERATION_LOCK_FILE must be an absolute path" >&2
    return 1
  fi
  raw_admission_lock_file="${LEADPOET_DOCKER_OPERATION_ADMISSION_LOCK_FILE:-${lock_file}.admission}"
  if ! admission_lock_file="$(leadpoet_canonical_docker_lock_path_v2 "$raw_admission_lock_file")"; then
    echo "ERROR: LEADPOET_DOCKER_OPERATION_ADMISSION_LOCK_FILE must be an absolute path" >&2
    return 1
  fi
  lock_timeout="${LEADPOET_DOCKER_OPERATION_LOCK_TIMEOUT_SECONDS:-3600}"

  if ! [[ "$lock_timeout" =~ ^[0-9]+$ ]] || [ "$lock_timeout" -lt 1 ]; then
    echo "ERROR: LEADPOET_DOCKER_OPERATION_LOCK_TIMEOUT_SECONDS must be a positive integer" >&2
    return 1
  fi
  if [ "$admission_lock_file" = "$lock_file" ]; then
    echo "ERROR: Docker operation admission and resource locks must differ" >&2
    return 1
  fi
  command -v flock >/dev/null 2>&1 || {
    echo "ERROR: flock is required for Docker operation coordination" >&2
    return 1
  }

  if [ "${LEADPOET_DOCKER_OPERATION_LOCK_HELD:-0}" = "1" ]; then
    observed_fd="$(readlink /proc/$$/fd/7 2>/dev/null || true)"
    if [ "$observed_fd" = "$lock_file" ]; then
      if [ "${LEADPOET_DOCKER_OPERATION_ADMISSION_LOCK_HELD:-0}" = "1" ]; then
        observed_admission_fd="$(readlink /proc/$$/fd/8 2>/dev/null || true)"
        if [ "$observed_admission_fd" != "$admission_lock_file" ]; then
          echo "ERROR: inherited Docker admission lock marker has no live lock owner" >&2
          return 1
        fi
      fi
      return 0
    fi
    owner_pid="${LEADPOET_DOCKER_OPERATION_LOCK_OWNER_PID:-}"
    if [[ "$owner_pid" =~ ^[0-9]+$ ]]; then
      owner_fd="$(readlink "/proc/$owner_pid/fd/7" 2>/dev/null || true)"
      if [ "$owner_fd" = "$lock_file" ]; then
        if [ "${LEADPOET_DOCKER_OPERATION_ADMISSION_LOCK_HELD:-0}" = "1" ]; then
          owner_admission_fd="$(readlink "/proc/$owner_pid/fd/8" 2>/dev/null || true)"
          if [ "$owner_admission_fd" != "$admission_lock_file" ]; then
            echo "ERROR: inherited Docker admission lock marker has no live lock owner" >&2
            return 1
          fi
        fi
        return 0
      fi
    fi
    echo "ERROR: inherited Docker operation lock marker has no live lock owner" >&2
    return 1
  fi

  mkdir -p "$(dirname "$lock_file")" "$(dirname "$admission_lock_file")"
  chmod 700 "$(dirname "$lock_file")"
  chmod 700 "$(dirname "$admission_lock_file")"
  exec 8>>"$admission_lock_file"
  chmod 600 "$admission_lock_file"
  lock_started_seconds="$SECONDS"
  echo "Waiting for exclusive Docker maintenance admission: $admission_lock_file"
  if ! flock -w "$lock_timeout" 8; then
    echo "ERROR: timed out waiting for exclusive Docker maintenance admission" >&2
    exec 8>&-
    return 1
  fi

  exec 7>>"$lock_file"
  chmod 600 "$lock_file"
  echo "Waiting for exclusive Docker build/maintenance access: $lock_file"
  lock_elapsed_seconds=$((SECONDS - lock_started_seconds))
  lock_remaining_seconds=$((lock_timeout - lock_elapsed_seconds))
  if [ "$lock_remaining_seconds" -lt 1 ]; then
    echo "ERROR: timed out waiting for exclusive Docker build/maintenance access" >&2
    exec 7>&-
    flock -u 8 || true
    exec 8>&-
    return 1
  fi
  if ! flock -w "$lock_remaining_seconds" 7; then
    echo "ERROR: timed out waiting for exclusive Docker build/maintenance access" >&2
    exec 7>&-
    flock -u 8 || true
    exec 8>&-
    return 1
  fi

  LEADPOET_DOCKER_OPERATION_LOCK_FILE="$lock_file"
  LEADPOET_DOCKER_OPERATION_ADMISSION_LOCK_FILE="$admission_lock_file"
  LEADPOET_DOCKER_OPERATION_LOCK_HELD=1
  LEADPOET_DOCKER_OPERATION_ADMISSION_LOCK_HELD=1
  LEADPOET_DOCKER_OPERATION_LOCK_OWNED=1
  LEADPOET_DOCKER_OPERATION_LOCK_OWNER_PID="$$"
  export LEADPOET_DOCKER_OPERATION_LOCK_FILE
  export LEADPOET_DOCKER_OPERATION_ADMISSION_LOCK_FILE
  export LEADPOET_DOCKER_OPERATION_LOCK_HELD
  export LEADPOET_DOCKER_OPERATION_ADMISSION_LOCK_HELD
  export LEADPOET_DOCKER_OPERATION_LOCK_OWNER_PID
  echo "Exclusive Docker build/maintenance access acquired"
}

leadpoet_ensure_post_activation_docker_operation_lock_v2() {
  local raw_lock_file raw_admission_lock_file lock_file admission_lock_file
  local observed_fd observed_admission_fd

  raw_lock_file="${LEADPOET_DOCKER_OPERATION_LOCK_FILE:-/home/ec2-user/.config/leadpoet/docker-operation-v2.lock}"
  if ! lock_file="$(leadpoet_canonical_docker_lock_path_v2 "$raw_lock_file")"; then
    echo "ERROR: LEADPOET_DOCKER_OPERATION_LOCK_FILE must be an absolute path" >&2
    return 1
  fi
  raw_admission_lock_file="${LEADPOET_DOCKER_OPERATION_ADMISSION_LOCK_FILE:-${lock_file}.admission}"
  if ! admission_lock_file="$(leadpoet_canonical_docker_lock_path_v2 "$raw_admission_lock_file")"; then
    echo "ERROR: LEADPOET_DOCKER_OPERATION_ADMISSION_LOCK_FILE must be an absolute path" >&2
    return 1
  fi
  observed_fd="$(readlink /proc/$$/fd/7 2>/dev/null || true)"
  observed_admission_fd="$(readlink /proc/$$/fd/8 2>/dev/null || true)"

  if [ "${LEADPOET_DOCKER_OPERATION_LOCK_HELD:-0}" = "1" ] \
      && [ "$observed_fd" = "$lock_file" ]; then
    if [ "${LEADPOET_DOCKER_OPERATION_ADMISSION_LOCK_HELD:-0}" = "1" ] \
        && [ "$observed_admission_fd" != "$admission_lock_file" ]; then
      echo "ERROR: post-activation Docker admission descriptor targets an unexpected file" >&2
      return 1
    fi
    return 0
  fi
  if [ -n "$observed_fd" ]; then
    echo "ERROR: post-activation Docker lock descriptor targets an unexpected file" >&2
    return 1
  fi
  if [ -n "$observed_admission_fd" ]; then
    echo "ERROR: post-activation Docker admission descriptor targets an unexpected file" >&2
    return 1
  fi

  # An older installed launcher may activate a newer checkout without having
  # opened fd 7. Acquire the exact same exclusive lock before the new script
  # performs any Docker work; never trust a stale inherited marker by itself.
  unset LEADPOET_DOCKER_OPERATION_LOCK_HELD
  unset LEADPOET_DOCKER_OPERATION_ADMISSION_LOCK_HELD
  unset LEADPOET_DOCKER_OPERATION_LOCK_OWNED
  unset LEADPOET_DOCKER_OPERATION_LOCK_OWNER_PID
  echo "Post-activation Docker lock was not inherited; acquiring it now"
  leadpoet_acquire_docker_operation_lock_v2
}

leadpoet_release_docker_operation_lock_v2() {
  if [ "${LEADPOET_DOCKER_OPERATION_LOCK_OWNED:-0}" != "1" ]; then
    if [ "${LEADPOET_DOCKER_OPERATION_LOCK_HELD:-0}" != "1" ] \
        || [ "${LEADPOET_DOCKER_OPERATION_LOCK_OWNER_PID:-}" != "$$" ] \
        || [ "$(readlink /proc/$$/fd/7 2>/dev/null || true)" != "${LEADPOET_DOCKER_OPERATION_LOCK_FILE:-}" ]; then
      return 0
    fi
  fi
  flock -u 7
  exec 7>&-
  if [ "${LEADPOET_DOCKER_OPERATION_ADMISSION_LOCK_HELD:-0}" = "1" ] \
      && [ "$(readlink /proc/$$/fd/8 2>/dev/null || true)" = "${LEADPOET_DOCKER_OPERATION_ADMISSION_LOCK_FILE:-}" ]; then
    flock -u 8
    exec 8>&-
  fi
  unset LEADPOET_DOCKER_OPERATION_LOCK_HELD
  unset LEADPOET_DOCKER_OPERATION_ADMISSION_LOCK_HELD
  unset LEADPOET_DOCKER_OPERATION_LOCK_OWNED
  unset LEADPOET_DOCKER_OPERATION_LOCK_OWNER_PID
  echo "Exclusive Docker build/maintenance access released"
}
