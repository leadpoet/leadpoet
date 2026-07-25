#!/bin/bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "ERROR: usage: verify_pinned_gateway_release_v2.sh <gateway-url> <commit-sha>" >&2
  exit 2
fi

GATEWAY_URL="${1%/}"
EXPECTED_COMMIT="$2"
MAX_ATTEMPTS=12
RETRY_DELAY_SECONDS=3

if ! [[ "$EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]]; then
  echo "ERROR: pinned gateway commit must be a lowercase full Git SHA" >&2
  exit 2
fi

for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
  gateway_health=""
  gateway_build_info=""
  gateway_release_evidence=""
  request_failed=0

  if ! gateway_health="$(
    curl --fail --silent --show-error \
      --connect-timeout 5 --max-time 35 \
      "$GATEWAY_URL/health/v2-authority"
  )"; then
    request_failed=1
  fi
  if [ "$request_failed" -eq 0 ] && ! gateway_build_info="$(
    curl --fail --silent --show-error \
      --connect-timeout 5 --max-time 35 \
      "$GATEWAY_URL/build-info"
  )"; then
    request_failed=1
  fi
  if [ "$request_failed" -eq 0 ] && ! gateway_release_evidence="$(
    curl --fail --silent --show-error \
      --connect-timeout 5 --max-time 35 \
      "$GATEWAY_URL/weights/v2/release-evidence/$EXPECTED_COMMIT"
  )"; then
    request_failed=1
  fi

  if [ "$request_failed" -eq 0 ] && python3 - \
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
if (
    release.get("schema_version") != "leadpoet.auditor_release_evidence.v2"
    or str(release.get("commit_sha") or "").lower() != expected_commit
):
    raise SystemExit("pinned gateway release evidence differs from the selected commit")

print(json.dumps({
    "status": "pinned_gateway_release_aligned",
    "commit_sha": expected_commit,
}, sort_keys=True))
PY
  then
    exit 0
  fi

  if [ "$attempt" -lt "$MAX_ATTEMPTS" ]; then
    echo "Pinned gateway release is not aligned yet; retrying ($attempt/$MAX_ATTEMPTS)" >&2
    sleep "$RETRY_DELAY_SECONDS"
  fi
done

echo "ERROR: pinned gateway release did not align after $MAX_ATTEMPTS attempts" >&2
exit 1
