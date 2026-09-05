#!/bin/bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "ERROR: usage: verify_pinned_gateway_release_v2.sh <gateway-url> <commit-sha>" >&2
  exit 2
fi

GATEWAY_URL="${1%/}"
EXPECTED_COMMIT="$2"
MAX_ATTEMPTS="${VALIDATOR_PINNED_GATEWAY_MAX_ATTEMPTS:-12}"
RETRY_DELAY_SECONDS=3
COORDINATION_FILE="${VALIDATOR_PINNED_GATEWAY_COORDINATION_FILE:-}"
COORDINATION_MAX_ATTEMPTS="${VALIDATOR_PINNED_GATEWAY_COORDINATION_MAX_ATTEMPTS:-3000}"
TOTAL_TIMEOUT_SECONDS="${VALIDATOR_PINNED_GATEWAY_TIMEOUT_SECONDS:-9300}"

if ! [[ "$EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]]; then
  echo "ERROR: pinned gateway commit must be a lowercase full Git SHA" >&2
  exit 2
fi
if ! [[ "$MAX_ATTEMPTS" =~ ^[1-9][0-9]*$ ]] \
    || [ "$MAX_ATTEMPTS" -gt 3000 ]; then
  echo "ERROR: VALIDATOR_PINNED_GATEWAY_MAX_ATTEMPTS must be between 1 and 3000" >&2
  exit 2
fi
if [ -n "$COORDINATION_FILE" ] && [[ "$COORDINATION_FILE" != /* ]]; then
  echo "ERROR: VALIDATOR_PINNED_GATEWAY_COORDINATION_FILE must be absolute" >&2
  exit 2
fi
if ! [[ "$COORDINATION_MAX_ATTEMPTS" =~ ^[1-9][0-9]*$ ]] \
    || [ "$COORDINATION_MAX_ATTEMPTS" -gt 3000 ]; then
  echo "ERROR: VALIDATOR_PINNED_GATEWAY_COORDINATION_MAX_ATTEMPTS must be between 1 and 3000" >&2
  exit 2
fi
if ! [[ "$TOTAL_TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]] \
    || [ "$TOTAL_TIMEOUT_SECONDS" -gt 10800 ]; then
  echo "ERROR: VALIDATOR_PINNED_GATEWAY_TIMEOUT_SECONDS must be between 1 and 10800" >&2
  exit 2
fi

# Use a process-group timeout around the whole verifier, including the
# coordination wait and every child request. Python is already a mandatory
# validator-host dependency and keeps this bound portable across developer and
# Amazon Linux environments.
if [ "${VALIDATOR_PINNED_GATEWAY_TIMEOUT_ACTIVE:-0}" != "1" ]; then
  exec python3 - "$TOTAL_TIMEOUT_SECONDS" "$0" "$@" <<'PY'
import os
import signal
import subprocess
import sys

timeout_seconds = int(sys.argv[1])
command = ["bash", sys.argv[2], *sys.argv[3:]]
environment = dict(os.environ)
environment["VALIDATOR_PINNED_GATEWAY_TIMEOUT_ACTIVE"] = "1"
process = subprocess.Popen(
    command,
    env=environment,
    start_new_session=True,
)


def terminate_group(signum):
    try:
        os.killpg(process.pid, signum)
    except ProcessLookupError:
        return


def forward_signal(signum, _frame):
    terminate_group(signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        terminate_group(signal.SIGKILL)
        process.wait()
    raise SystemExit(128 + signum)


for forwarded in (signal.SIGHUP, signal.SIGINT, signal.SIGTERM):
    signal.signal(forwarded, forward_signal)

try:
    status = process.wait(timeout=timeout_seconds)
except subprocess.TimeoutExpired:
    terminate_group(signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        terminate_group(signal.SIGKILL)
        process.wait()
    print(
        "ERROR: pinned gateway release verification exceeded %ss"
        % timeout_seconds,
        file=sys.stderr,
    )
    status = 124
raise SystemExit(status)
PY
fi

# Bound the coordination wait, live HTTP probes, retries, and response
# validation as one operation. Each HTTP request consumes only the remaining
# overall budget in addition to the outer process-group timeout.
VERIFIER_STARTED_AT="$SECONDS"

remaining_timeout_seconds() {
  local elapsed remaining
  elapsed=$((SECONDS - VERIFIER_STARTED_AT))
  remaining=$((TOTAL_TIMEOUT_SECONDS - elapsed))
  if [ "$remaining" -le 0 ]; then
    return 1
  fi
  printf '%s\n' "$remaining"
}

fail_if_timed_out() {
  if ! remaining_timeout_seconds >/dev/null; then
    echo "ERROR: pinned gateway release verification exceeded ${TOTAL_TIMEOUT_SECONDS}s" >&2
    exit 124
  fi
}

bounded_retry_sleep() {
  local delay remaining
  remaining="$(remaining_timeout_seconds)" || return 124
  delay="$RETRY_DELAY_SECONDS"
  if [ "$remaining" -lt "$delay" ]; then
    delay="$remaining"
  fi
  sleep "$delay"
}

bounded_gateway_request() {
  local connect_timeout label max_time remaining request_status url
  label="$1"
  url="$2"
  remaining="$(remaining_timeout_seconds)" || return 124
  max_time=35
  if [ "$remaining" -lt "$max_time" ]; then
    max_time="$remaining"
  fi
  connect_timeout=5
  if [ "$max_time" -lt "$connect_timeout" ]; then
    connect_timeout="$max_time"
  fi
  request_status=0
  curl --fail --silent --show-error \
    --connect-timeout "$connect_timeout" --max-time "$max_time" \
    "$url" || request_status=$?
  if [ "$request_status" -ne 0 ]; then
    echo "pinned_gateway_request_failed endpoint=${label} curl_status=${request_status}" >&2
    return "$request_status"
  fi
}

require_current_coordination_success() {
  local coordination_value
  if [ -z "$COORDINATION_FILE" ]; then
    return 0
  fi
  if [ ! -r "$COORDINATION_FILE" ]; then
    echo "ERROR: coordinated gateway success marker disappeared" >&2
    return 1
  fi
  coordination_value="$(cat "$COORDINATION_FILE")"
  case "$coordination_value" in
    "$EXPECTED_COMMIT")
      return 0
      ;;
    "failed:$EXPECTED_COMMIT")
      echo "ERROR: coordinated gateway restart failed for the selected commit" >&2
      ;;
    failed:*)
      echo "ERROR: coordinated gateway failure marker differs from the selected commit" >&2
      ;;
    *)
      echo "ERROR: coordinated gateway success marker differs from the selected commit" >&2
      ;;
  esac
  return 1
}

if [ -n "$COORDINATION_FILE" ]; then
  coordination_ready=0
  for attempt in $(seq 1 "$COORDINATION_MAX_ATTEMPTS"); do
    fail_if_timed_out
    if [ -r "$COORDINATION_FILE" ]; then
      coordination_value="$(cat "$COORDINATION_FILE")"
      case "$coordination_value" in
        "$EXPECTED_COMMIT")
          coordination_ready=1
          break
          ;;
        "failed:$EXPECTED_COMMIT")
          echo "ERROR: coordinated gateway restart failed for the selected commit" >&2
          exit 1
          ;;
        failed:*)
          echo "ERROR: coordinated gateway failure marker differs from the selected commit" >&2
          exit 1
          ;;
        *)
          if [[ "$coordination_value" =~ ^[0-9a-f]{40}$ ]]; then
            echo "ERROR: coordinated gateway success marker differs from the selected commit" >&2
            exit 1
          fi
          ;;
      esac
    fi
    if [ "$attempt" -lt "$COORDINATION_MAX_ATTEMPTS" ]; then
      if [ "$attempt" -eq 1 ] || [ $((attempt % 20)) -eq 0 ]; then
        echo "Waiting for the coordinated gateway restart to complete ($attempt/$COORDINATION_MAX_ATTEMPTS)" >&2
      fi
      if ! bounded_retry_sleep; then
        fail_if_timed_out
      fi
    fi
  done
  if [ "$coordination_ready" != "1" ]; then
    echo "ERROR: coordinated gateway restart did not complete after $COORDINATION_MAX_ATTEMPTS attempts" >&2
    exit 1
  fi
fi

for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
  fail_if_timed_out
  gateway_health=""
  gateway_build_info=""
  gateway_release_evidence=""
  request_failed=0

  if ! gateway_health="$(
    bounded_gateway_request \
      "v2_authority" \
      "$GATEWAY_URL/health/v2-authority"
  )"; then
    request_failed=1
  fi
  if [ "$request_failed" -eq 0 ] && ! gateway_build_info="$(
    bounded_gateway_request \
      "build_info" \
      "$GATEWAY_URL/build-info"
  )"; then
    request_failed=1
  fi
  if [ "$request_failed" -eq 0 ] && ! gateway_release_evidence="$(
    bounded_gateway_request \
      "release_evidence" \
      "$GATEWAY_URL/weights/v2/release-evidence/$EXPECTED_COMMIT"
  )"; then
    request_failed=1
  fi

  alignment_result=""
  if [ "$request_failed" -eq 0 ] && alignment_result="$(
    python3 - \
        "$EXPECTED_COMMIT" \
        "$gateway_health" \
        "$gateway_build_info" \
        "$gateway_release_evidence" <<'PY'
import json
import sys

expected_commit = str(sys.argv[1] or "").lower()


def load_json(raw, label):
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        raise SystemExit("pinned gateway %s is invalid JSON" % label)
    if not isinstance(value, dict):
        raise SystemExit("pinned gateway %s is not an object" % label)
    return value


health = load_json(sys.argv[2], "V2 authority")
if (
    health.get("status") != "ready"
    or str(health.get("commit_sha") or "").lower() != expected_commit
):
    raise SystemExit("pinned gateway V2 authority is not ready on the selected commit")

build_info = load_json(sys.argv[3], "build-info")
if str(build_info.get("git_commit") or "").lower() != expected_commit:
    raise SystemExit("pinned gateway build-info differs from the selected commit")

release = load_json(sys.argv[4], "release evidence")
release_schema = release.get("schema_version")
if str(release.get("commit_sha") or "").lower() != expected_commit:
    raise SystemExit("pinned gateway release evidence differs from the selected commit")
if release_schema == "leadpoet.auditor_local_release_evidence.v1":
    from leadpoet_canonical.auditor_v2 import fetch_locked_release_identity_cache

    try:
        identity_cache = fetch_locked_release_identity_cache(release)
    except Exception:
        raise SystemExit(
            "pinned gateway local release evidence is invalid"
        ) from None
    if any(
        str(entry.get("commit_sha") or "").lower() != expected_commit
        for entry in identity_cache.get("entries", [])
    ):
        raise SystemExit(
            "pinned gateway local release evidence differs from the selected commit"
        )
elif release_schema != "leadpoet.auditor_release_evidence.v2":
    raise SystemExit("pinned gateway release evidence schema is invalid")

print(json.dumps({
    "status": "pinned_gateway_release_aligned",
    "commit_sha": expected_commit,
}, sort_keys=True))
PY
  )"; then
    # Re-read the marker after the HTTP contract checks so a concurrent
    # operator failure/cancellation cannot be mistaken for authorization.
    if ! require_current_coordination_success; then
      exit 1
    fi
    printf '%s\n' "$alignment_result"
    exit 0
  fi

  if [ "$attempt" -lt "$MAX_ATTEMPTS" ]; then
    echo "Pinned gateway release is not aligned yet; retrying ($attempt/$MAX_ATTEMPTS)" >&2
    if ! bounded_retry_sleep; then
      fail_if_timed_out
    fi
  fi
done

echo "ERROR: pinned gateway release did not align after $MAX_ATTEMPTS attempts" >&2
exit 1
