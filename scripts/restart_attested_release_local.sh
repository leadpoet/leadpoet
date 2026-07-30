#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GATEWAY_KEY="${LEADPOET_GATEWAY_SSH_KEY:-$HOME/Downloads/leadpoet-2026-07-28.pem}"
VALIDATOR_KEY="${LEADPOET_VALIDATOR_SSH_KEY:-$HOME/Downloads/leadpoet-2026-07-28.pem}"
GATEWAY_HOST="${LEADPOET_GATEWAY_SSH_HOST:-ec2-user@52.91.135.79}"
VALIDATOR_HOST="${LEADPOET_VALIDATOR_SSH_HOST:-ec2-user@100.59.201.156}"
GATEWAY_RESTART="${LEADPOET_GATEWAY_RESTART_PATH:-/home/ec2-user/gw_restart.sh}"
VALIDATOR_RESTART="${LEADPOET_VALIDATOR_RESTART_PATH:-/home/ec2-user/validator_restart.sh}"
VALIDATOR_REPO_ROOT="${LEADPOET_VALIDATOR_REPO_ROOT:-/home/ec2-user/leadpoet/leadpoet}"
VALIDATOR_COORDINATION_ATTEMPTS=600
VALIDATOR_FAILURE_CLEANUP_ATTEMPTS=60
VALIDATOR_FAILURE_MARKER_ATTEMPTS=20

commit=""
component="all"
validator_job=""
failure_marker_job=""
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
  local failure_marker_command=""
  set +e
  if [ -n "$validator_job" ] && kill -0 "$validator_job" 2>/dev/null; then
    # Cancellation is the authority revocation. Signal the remote validator
    # before attempting the durable failure marker so a slow SSH connection
    # cannot delay cleanup past the late activation boundary.
    kill -TERM "$validator_job" 2>/dev/null || true
    if [ -n "$coordination_file" ]; then
      echo "Signaling the paired validator to clean up after restart failure" >&2
      failure_marker_command="$(
        coordination_remote_command "failed:$commit"
      )"
      ssh "${ssh_common[@]}" -i "$VALIDATOR_KEY" "$VALIDATOR_HOST" \
        "$failure_marker_command" >/dev/null 2>&1 &
      failure_marker_job="$!"
    fi
    for _ in $(seq 1 "$VALIDATOR_FAILURE_CLEANUP_ATTEMPTS"); do
      if ! kill -0 "$validator_job" 2>/dev/null; then
        break
      fi
      sleep 1
    done
    if kill -0 "$validator_job" 2>/dev/null; then
      kill -KILL "$validator_job" 2>/dev/null || true
    fi
    wait "$validator_job" 2>/dev/null || true
  fi
  if [ -n "$failure_marker_job" ]; then
    for _ in $(seq 1 "$VALIDATOR_FAILURE_MARKER_ATTEMPTS"); do
      if ! kill -0 "$failure_marker_job" 2>/dev/null; then
        break
      fi
      sleep 1
    done
    if kill -0 "$failure_marker_job" 2>/dev/null; then
      kill -TERM "$failure_marker_job" 2>/dev/null || true
      sleep 1
    fi
    if kill -0 "$failure_marker_job" 2>/dev/null; then
      kill -KILL "$failure_marker_job" 2>/dev/null || true
    fi
    wait "$failure_marker_job" 2>/dev/null || true
    failure_marker_job=""
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
python3 "$helper" \
  --repo-root "$ROOT" \
  --selected-commit "$commit" \
  --branch-ref origin/main
echo "Selected release is compatible with current public auditors: $commit"
echo "Current public V2 authority commit: $branch_commit"

ssh_common=(
  -n
  -o BatchMode=yes
  -o ConnectTimeout=15
  -o ServerAliveInterval=30
  -o ServerAliveCountMax=20
)

coordination_remote_command() {
  local value="$1"
  case "$value" in
    "$commit"|"failed:$commit") ;;
    *)
      echo "ERROR: invalid coordinated restart marker value" >&2
      return 2
      ;;
  esac
  printf '%s\n' \
    "set -Eeuo pipefail
     umask 077
     marker='$coordination_file.tmp'
     printf '%s\\n' '$value' > \"\$marker\"
     mv -f -- \"\$marker\" '$coordination_file'"
}

publish_coordination_value() {
  local remote_command
  remote_command="$(coordination_remote_command "$1")"
  ssh "${ssh_common[@]}" -i "$VALIDATOR_KEY" "$VALIDATOR_HOST" \
    "$remote_command"
}

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
  local expected_forward_commit=""
  local launcher_command=""
  local launcher_command_quoted=""
  local restart_arguments=""
  local command=()
  if [ -n "$coordination_file" ]; then
    coordination_environment="$coordination_file"
  fi
  if [ "$commit" != "$branch_commit" ]; then
    # Historical rollback releases retain the installed controller and use
    # its exact-commit compatibility handoff. A forward release at the frozen
    # public tip uses the normal N-1 -> N pull/re-exec path so the candidate
    # launcher can prepare in parallel and install itself after success.
    restart_arguments="--commit '$commit'"
    launcher_command="exec bash '$VALIDATOR_RESTART' $restart_arguments"
  else
    expected_forward_commit="$commit"
    # On the first N-1 -> N attempt, the installed launcher owns the Git
    # handoff. If a prior attempt already completed that handoff but failed
    # before installing N, retry the exact candidate launcher from its clean
    # Git blob instead of silently falling back to the stale installed copy.
    launcher_command="
      candidate_launcher='$VALIDATOR_REPO_ROOT/validator_restart.sh'
      observed_head=\$(git -C '$VALIDATOR_REPO_ROOT' rev-parse --verify HEAD)
      if [ \"\$observed_head\" = '$commit' ]; then
        if [ ! -r \"\$candidate_launcher\" ] \
            || ! git -C '$VALIDATOR_REPO_ROOT' diff --quiet '$commit' -- validator_restart.sh; then
          echo 'ERROR: selected validator launcher is not the exact candidate Git blob' >&2
          exit 1
        fi
        exec bash \"\$candidate_launcher\"
      fi
      exec bash '$VALIDATOR_RESTART'"
  fi
  printf -v launcher_command_quoted '%q' "$launcher_command"
  command=(
    ssh -tt "${ssh_common[@]}" -i "$VALIDATOR_KEY" "$VALIDATOR_HOST"
    "exec env \
      VALIDATOR_PINNED_GATEWAY_COORDINATION_FILE='$coordination_environment' \
      VALIDATOR_PINNED_GATEWAY_COORDINATION_MAX_ATTEMPTS='$VALIDATOR_COORDINATION_ATTEMPTS' \
      VALIDATOR_COORDINATED_EXPECTED_COMMIT='$expected_forward_commit' \
      bash -c $launcher_command_quoted"
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
    echo "Starting validator preparation in parallel with the paired gateway restart"
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

    echo "Validator restart start captured; restarting the gateway while validator preparation continues"
    run_gateway_restart
    publish_coordination_value "$commit"
    echo "Gateway restart completed; releasing exact-SHA validator activation"
    wait "$validator_job"
    validator_job=""
    ;;
esac

verify_gateway_release
verify_validator_release
echo "SUCCESS: gateway and validator are aligned on attested release $commit"
