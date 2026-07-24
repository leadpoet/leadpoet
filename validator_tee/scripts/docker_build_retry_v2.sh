#!/bin/bash

# Retry only Docker daemon/transport interruptions. Build, dependency, and
# Dockerfile failures remain deterministic hard failures.

leadpoet_is_transient_docker_build_failure_v2() {
  local log_path="$1"
  grep -Eiq \
    'rpc error: code = Unavailable|error reading from server: EOF|failed to receive status|Cannot connect to the Docker daemon|connection reset by peer|unexpected EOF|transport is closing' \
    "$log_path"
}

leadpoet_run_docker_build_with_retry_v2() {
  local label="$1"
  shift

  local max_attempts backoff_seconds attempt status log_path errexit_enabled
  max_attempts="${VALIDATOR_DOCKER_BUILD_MAX_ATTEMPTS:-3}"
  backoff_seconds="${VALIDATOR_DOCKER_BUILD_RETRY_BACKOFF_SECONDS:-5}"

  if ! [[ "$max_attempts" =~ ^[0-9]+$ ]] || [ "$max_attempts" -lt 1 ]; then
    echo "ERROR: VALIDATOR_DOCKER_BUILD_MAX_ATTEMPTS must be a positive integer" >&2
    return 1
  fi
  if ! [[ "$backoff_seconds" =~ ^[0-9]+$ ]]; then
    echo "ERROR: VALIDATOR_DOCKER_BUILD_RETRY_BACKOFF_SECONDS must be a non-negative integer" >&2
    return 1
  fi
  if [ "$#" -eq 0 ]; then
    echo "ERROR: Docker build retry requires a command" >&2
    return 1
  fi

  errexit_enabled=0
  case "$-" in
    *e*) errexit_enabled=1 ;;
  esac

  for attempt in $(seq 1 "$max_attempts"); do
    log_path="$(mktemp /tmp/leadpoet-docker-build.XXXXXX.log)"
    echo "Running $label (attempt $attempt/$max_attempts)"
    set +e
    "$@" 2>&1 | tee "$log_path"
    status="${PIPESTATUS[0]}"
    if [ "$errexit_enabled" -eq 1 ]; then
      set -e
    fi

    if [ "$status" -eq 0 ]; then
      rm -f "$log_path"
      return 0
    fi
    if [ "$attempt" -ge "$max_attempts" ] \
        || ! leadpoet_is_transient_docker_build_failure_v2 "$log_path"; then
      rm -f "$log_path"
      return "$status"
    fi

    echo "Transient Docker daemon/build transport failure detected; retrying $label" >&2
    rm -f "$log_path"
    if ! docker info >/dev/null 2>&1; then
      sudo systemctl start containerd.service docker.service
      for _ready_attempt in $(seq 1 30); do
        if docker info >/dev/null 2>&1; then
          break
        fi
        if [ "$_ready_attempt" -eq 30 ]; then
          echo "ERROR: Docker daemon did not recover before build retry" >&2
          return "$status"
        fi
        sleep 1
      done
    fi
    sleep "$((backoff_seconds * attempt))"
  done
}
