#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GATEWAY_KEY="${LEADPOET_GATEWAY_SSH_KEY:-$HOME/Downloads/leadpoet-gateway-tee-main.pem}"
VALIDATOR_KEY="${LEADPOET_VALIDATOR_SSH_KEY:-$HOME/Downloads/leadpoet-validator.pem}"
GATEWAY_HOST="${LEADPOET_GATEWAY_SSH_HOST:-ec2-user@52.91.135.79}"
VALIDATOR_HOST="${LEADPOET_VALIDATOR_SSH_HOST:-ec2-user@100.59.201.156}"
GATEWAY_RESTART="${LEADPOET_GATEWAY_RESTART_PATH:-/home/ec2-user/gw_restart.sh}"
VALIDATOR_RESTART="${LEADPOET_VALIDATOR_RESTART_PATH:-/home/ec2-user/validator_restart.sh}"
VALIDATOR_COORDINATION_ATTEMPTS=600

commit=""
component="all"
validator_job=""
temporary_root=""
coordination_file=""

usage() {
  cat <<'EOF'
Usage:
  bash scripts/restart_attested_release_local.sh \
    --commit <full-40-character-sha> \
    [--component all|gateway|validator]

The default "all" mode starts both exact-commit restarts in one invocation.
A single-component restart is accepted only when the other component is
already running the selected commit.
EOF
}

cleanup() {
  local status="$?"
  if [ -n "$validator_job" ] && kill -0 "$validator_job" 2>/dev/null; then
    kill -TERM "$validator_job" 2>/dev/null || true
    wait "$validator_job" 2>/dev/null || true
  fi
  if [ -n "$coordination_file" ]; then
    ssh "${ssh_common[@]}" -i "$VALIDATOR_KEY" "$VALIDATOR_HOST" \
      "rm -f -- '$coordination_file'" >/dev/null 2>&1 || true
  fi
  if [ -n "$temporary_root" ]; then
    rm -rf -- "$temporary_root"
  fi
  return "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

while [ "$#" -gt 0 ]; do
  case "$1" in
    --commit)
      if [ "$#" -lt 2 ]; then
        echo "ERROR: --commit requires a full 40-character SHA" >&2
        exit 2
      fi
      commit="$2"
      shift 2
      ;;
    --commit=*)
      commit="${1#--commit=}"
      shift
      ;;
    --component)
      if [ "$#" -lt 2 ]; then
        echo "ERROR: --component requires all, gateway, or validator" >&2
        exit 2
      fi
      component="$2"
      shift 2
      ;;
    --component=*)
      component="${1#--component=}"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unsupported attested restart argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ! [[ "$commit" =~ ^[0-9a-f]{40}$ ]]; then
  echo "ERROR: --commit must be a lowercase full 40-character SHA" >&2
  exit 2
fi
case "$component" in
  all|gateway|validator) ;;
  *)
    echo "ERROR: --component must be all, gateway, or validator" >&2
    exit 2
    ;;
esac
for key in "$GATEWAY_KEY" "$VALIDATOR_KEY"; do
  if [ ! -r "$key" ]; then
    echo "ERROR: SSH key is unavailable: $key" >&2
    exit 1
  fi
done

temporary_root="$(mktemp -d /tmp/leadpoet-attested-restart.XXXXXX)"
helper="$temporary_root/exact_commit_restart_v2.py"

echo "Fetching current public V2 compatibility authority"
git -C "$ROOT" fetch origin main
branch_commit="$(git -C "$ROOT" rev-parse --verify origin/main^{commit})"
git -C "$ROOT" cat-file -e "$commit^{commit}"
git -C "$ROOT" show \
  origin/main:Leadpoet/utils/exact_commit_restart_v2.py > "$helper"
compatibility_floor="$(
  git -C "$ROOT" show origin/main:gw_restart.sh \
    | sed -n \
      's/^V2_DEPLOYMENT_COMPATIBILITY_FLOOR_SHA="\([0-9a-f]\{40\}\)"$/\1/p' \
    | head -n 1
)"
if ! [[ "$compatibility_floor" =~ ^[0-9a-f]{40}$ ]]; then
  echo "ERROR: current V2 rollback compatibility floor is unavailable" >&2
  exit 1
fi
python3 "$helper" \
  --repo-root "$ROOT" \
  --selected-commit "$commit" \
  --branch-ref origin/main \
  --compatibility-floor "$compatibility_floor"
echo "Selected release is compatible with current public auditors: $commit"
echo "Current public V2 authority commit: $branch_commit"

ssh_common=(
  -n
  -o BatchMode=yes
  -o ConnectTimeout=15
  -o ServerAliveInterval=30
  -o ServerAliveCountMax=20
)

gateway_active_commit() {
  ssh "${ssh_common[@]}" -i "$GATEWAY_KEY" "$GATEWAY_HOST" \
    "curl -fsS --connect-timeout 5 --max-time 15 \
      http://127.0.0.1:8000/build-info \
      | python3 -c 'import json,sys; print(json.load(sys.stdin)[\"git_commit\"])'"
}

validator_active_commit() {
  ssh "${ssh_common[@]}" -i "$VALIDATOR_KEY" "$VALIDATOR_HOST" \
    "docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' \
      leadpoet-validator-main \
      | sed -n 's/^VALIDATOR_V2_DEPLOY_COMMIT=//p'"
}

run_gateway_restart() {
  ssh -tt "${ssh_common[@]}" -i "$GATEWAY_KEY" "$GATEWAY_HOST" \
    "exec bash '$GATEWAY_RESTART' --commit '$commit'"
}

run_validator_restart() {
  local coordination_environment=""
  local command=()
  if [ -n "$coordination_file" ]; then
    coordination_environment="$coordination_file"
  fi
  command=(
    ssh -tt "${ssh_common[@]}" -i "$VALIDATOR_KEY" "$VALIDATOR_HOST"
    "exec env \
      VALIDATOR_PINNED_GATEWAY_COORDINATION_FILE='$coordination_environment' \
      VALIDATOR_PINNED_GATEWAY_COORDINATION_MAX_ATTEMPTS='$VALIDATOR_COORDINATION_ATTEMPTS' \
      bash '$VALIDATOR_RESTART' --commit '$commit'"
  )
  if [ "${VALIDATOR_RESTART_EXEC_SSH:-0}" = "1" ]; then
    exec "${command[@]}"
  fi
  "${command[@]}"
}

verify_gateway_release() {
  ssh "${ssh_common[@]}" -i "$GATEWAY_KEY" "$GATEWAY_HOST" \
    "python3 - '$commit' <<'PY'
import json
import sys
import urllib.request

expected = sys.argv[1]

def get(path):
    with urllib.request.urlopen(
        'http://127.0.0.1:8000' + path,
        timeout=35,
    ) as response:
        value = json.load(response)
    if not isinstance(value, dict):
        raise SystemExit('gateway response is not an object: ' + path)
    return value

health = get('/health/v2-authority')
build = get('/build-info')
release = get('/weights/v2/release-evidence/' + expected)
if health.get('status') != 'ready':
    raise SystemExit('gateway V2 authority is not ready')
if str(health.get('commit_sha') or '').lower() != expected:
    raise SystemExit('gateway V2 authority commit differs')
if str(build.get('git_commit') or '').lower() != expected:
    raise SystemExit('gateway build-info commit differs')
if release.get('schema_version') != 'leadpoet.auditor_release_evidence.v2':
    raise SystemExit('gateway release evidence schema differs')
if str(release.get('commit_sha') or '').lower() != expected:
    raise SystemExit('gateway release evidence commit differs')
print(json.dumps({
    'commit_sha': expected,
    'status': 'gateway_exact_release_ready',
}, sort_keys=True))
PY"
}

verify_validator_release() {
  ssh "${ssh_common[@]}" -i "$VALIDATOR_KEY" "$VALIDATOR_HOST" \
    "python3 - '$commit' <<'PY'
import json
import subprocess
import sys

expected = sys.argv[1]
running = subprocess.check_output(
    ['docker', 'inspect', '-f', '{{.State.Running}}', 'leadpoet-validator-main'],
    text=True,
).strip()
restarts = subprocess.check_output(
    ['docker', 'inspect', '-f', '{{.RestartCount}}', 'leadpoet-validator-main'],
    text=True,
).strip()
environment = subprocess.check_output(
    [
        'docker',
        'inspect',
        '-f',
        '{{range .Config.Env}}{{println .}}{{end}}',
        'leadpoet-validator-main',
    ],
    text=True,
).splitlines()
values = dict(
    line.split('=', 1)
    for line in environment
    if '=' in line
)
if running != 'true' or restarts != '0':
    raise SystemExit('validator coordinator is not cleanly running')
if values.get('VALIDATOR_V2_DEPLOY_COMMIT') != expected:
    raise SystemExit('validator coordinator commit differs')
if values.get('VALIDATOR_WEIGHT_PROTOCOL') != 'authoritative_v2':
    raise SystemExit('validator coordinator is not authoritative V2')
print(json.dumps({
    'commit_sha': expected,
    'status': 'validator_exact_release_ready',
}, sort_keys=True))
PY"
}

case "$component" in
  gateway)
    observed_validator="$(validator_active_commit)"
    if [ "$observed_validator" != "$commit" ]; then
      echo "ERROR: validator is on $observed_validator, not $commit; use --component all" >&2
      exit 1
    fi
    verify_validator_release
    run_gateway_restart
    ;;
  validator)
    observed_gateway="$(gateway_active_commit)"
    if [ "$observed_gateway" != "$commit" ]; then
      echo "ERROR: gateway is on $observed_gateway, not $commit; use --component all" >&2
      exit 1
    fi
    verify_gateway_release
    run_validator_restart
    ;;
  all)
    validator_log="$temporary_root/validator-restart.log"
    coordination_file="/tmp/leadpoet-coordinated-restart.$(basename "$temporary_root").ready"
    ssh "${ssh_common[@]}" -i "$VALIDATOR_KEY" "$VALIDATOR_HOST" \
      "rm -f -- '$coordination_file'"
    echo "Starting validator restart so it captures the official window before waiting for the gateway"
    (
      VALIDATOR_RESTART_EXEC_SSH=1 run_validator_restart
    ) > >(tee "$validator_log") 2>&1 &
    validator_job="$!"

    capture_ready=0
    for _ in $(seq 1 120); do
      if grep -Fq \
          "Acquiring the independently built V2 release channel" \
          "$validator_log"; then
        capture_ready=1
        break
      fi
      if ! kill -0 "$validator_job" 2>/dev/null; then
        wait "$validator_job"
        exit 1
      fi
      sleep 1
    done
    if [ "$capture_ready" != "1" ]; then
      echo "ERROR: validator did not capture a valid restart start" >&2
      exit 1
    fi

    echo "Validator restart start captured; restarting the gateway on the same release"
    run_gateway_restart
    ssh "${ssh_common[@]}" -i "$VALIDATOR_KEY" "$VALIDATOR_HOST" \
      "set -Eeuo pipefail
       umask 077
       marker='$coordination_file.tmp'
       printf '%s\\n' '$commit' > \"\$marker\"
       mv -f -- \"\$marker\" '$coordination_file'"
    echo "Gateway exact release is ready; waiting for the paired validator restart"
    wait "$validator_job"
    validator_job=""
    ;;
esac

verify_gateway_release
verify_validator_release
echo "SUCCESS: gateway and validator are aligned on attested release $commit"
