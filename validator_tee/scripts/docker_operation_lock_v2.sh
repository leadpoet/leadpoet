#!/bin/bash

# Coordinate destructive Docker maintenance and image builds on shared hosts.
# The lock is inherited by child processes so one restart can cover its full
# Docker lifecycle without allowing a self-hosted runner to enter mid-build.

leadpoet_acquire_docker_operation_lock_v2() {
  local lock_file lock_timeout observed_fd owner_pid owner_fd

  lock_file="${LEADPOET_DOCKER_OPERATION_LOCK_FILE:-/home/ec2-user/.config/leadpoet/docker-operation-v2.lock}"
  lock_timeout="${LEADPOET_DOCKER_OPERATION_LOCK_TIMEOUT_SECONDS:-3600}"

  if ! [[ "$lock_timeout" =~ ^[0-9]+$ ]] || [ "$lock_timeout" -lt 1 ]; then
    echo "ERROR: LEADPOET_DOCKER_OPERATION_LOCK_TIMEOUT_SECONDS must be a positive integer" >&2
    return 1
  fi
  command -v flock >/dev/null 2>&1 || {
    echo "ERROR: flock is required for Docker operation coordination" >&2
    return 1
  }

  if [ "${LEADPOET_DOCKER_OPERATION_LOCK_HELD:-0}" = "1" ]; then
    observed_fd="$(readlink /proc/$$/fd/7 2>/dev/null || true)"
    if [ "$observed_fd" = "$lock_file" ]; then
      return 0
    fi
    owner_pid="${LEADPOET_DOCKER_OPERATION_LOCK_OWNER_PID:-}"
    if [[ "$owner_pid" =~ ^[0-9]+$ ]]; then
      owner_fd="$(readlink "/proc/$owner_pid/fd/7" 2>/dev/null || true)"
      if [ "$owner_fd" = "$lock_file" ]; then
        return 0
      fi
    fi
    echo "ERROR: inherited Docker operation lock marker has no live lock owner" >&2
    return 1
  fi

  mkdir -p "$(dirname "$lock_file")"
  chmod 700 "$(dirname "$lock_file")"
  exec 7>>"$lock_file"
  chmod 600 "$lock_file"
  echo "Waiting for exclusive Docker build/maintenance access: $lock_file"
  if ! flock -w "$lock_timeout" 7; then
    echo "ERROR: timed out waiting for exclusive Docker build/maintenance access" >&2
    exec 7>&-
    return 1
  fi

  LEADPOET_DOCKER_OPERATION_LOCK_FILE="$lock_file"
  LEADPOET_DOCKER_OPERATION_LOCK_HELD=1
  LEADPOET_DOCKER_OPERATION_LOCK_OWNED=1
  LEADPOET_DOCKER_OPERATION_LOCK_OWNER_PID="$$"
  export LEADPOET_DOCKER_OPERATION_LOCK_FILE
  export LEADPOET_DOCKER_OPERATION_LOCK_HELD
  export LEADPOET_DOCKER_OPERATION_LOCK_OWNER_PID
  echo "Exclusive Docker build/maintenance access acquired"
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
  unset LEADPOET_DOCKER_OPERATION_LOCK_HELD
  unset LEADPOET_DOCKER_OPERATION_LOCK_OWNED
  unset LEADPOET_DOCKER_OPERATION_LOCK_OWNER_PID
  echo "Exclusive Docker build/maintenance access released"
}
