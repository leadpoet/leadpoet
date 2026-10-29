#!/bin/bash
# Canonical normal-Arena local-wallet restart. No hardware signer is required.
set -euo pipefail
umask 077
export PYTHONDONTWRITEBYTECODE=1

SOURCE_ROOT="${VALIDATOR_ROOT:-/home/ec2-user/leadpoet/leadpoet}"
RELEASE_ROOT="${VALIDATOR_RELEASE_ROOT:-/home/ec2-user/leadpoet/validator-releases}"
CURRENT_LINK="${VALIDATOR_CURRENT_LINK:-/home/ec2-user/leadpoet/validator-current}"
ENV_FILE="${VALIDATOR_ENV_FILE:-/home/ec2-user/.config/leadpoet/arena-validator.env}"
SERVICE_ENV="${VALIDATOR_SERVICE_ENV:-/home/ec2-user/.config/leadpoet/arena-validator-service.env}"
SERVICE="${VALIDATOR_SERVICE_NAME:-leadpoet-arena-validator.service}"
UNIT_PATH="${VALIDATOR_SERVICE_UNIT_PATH:-/etc/systemd/system/$SERVICE}"
PYTHON=""
# Leave room for supervised retries when the public chain RPC is rate limited.
# Every retry still passes the full readiness check and ten-second PID stability.
READY_TIMEOUT="${VALIDATOR_READY_TIMEOUT_SECONDS:-300}"
STOP_TIMEOUT="${VALIDATOR_STOP_TIMEOUT_SECONDS:-9300}"
LOCK_FILE="${VALIDATOR_RESTART_LOCK_FILE:-/home/ec2-user/.config/leadpoet/arena-validator-restart.lock}"
TARGET_REQUEST="${VALIDATOR_DEPLOY_COMMIT:-origin/main}"
STAGE=""
ROLLBACK=""
CANDIDATE_SERVICE_ENV=""
SNAPSHOT_SERVICE_ENV=""
OLD_ACTIVE=0
DRAINED=0
PROMOTED=0
ACTIVATED=0
PREVIOUS_CURRENT_TARGET=""
fail() { echo "ERROR: $*" >&2; exit 1; }

select_validator_python() {
  if [ "${VALIDATOR_PYTHON_BIN+x}" = x ]; then
    PYTHON="$VALIDATOR_PYTHON_BIN"
    return
  fi
  local metadata key value seen="|" count=0 load_state="" fragment="" drop_ins="" reload=""
  metadata="$(systemctl show "$SERVICE" --all --property=LoadState,FragmentPath,DropInPaths,NeedDaemonReload)" \
    || fail "cannot inspect installed validator unit"
  while IFS='=' read -r key value; do
    [[ "$seen" != *"|$key|"* ]] || fail "ambiguous installed validator unit metadata"
    seen+="$key|"
    count=$((count + 1))
    case "$key" in
      LoadState) load_state="$value" ;;
      FragmentPath) fragment="$value" ;;
      DropInPaths) drop_ins="$value" ;;
      NeedDaemonReload) reload="$value" ;;
      *) fail "unexpected installed validator unit metadata" ;;
    esac
  done <<< "$metadata"
  [ "$count" -eq 4 ] && [ -z "$drop_ins" ] && [ "$reload" = no ] \
    || fail "installed validator unit has overrides or requires reload"
  if [ ! -e "$UNIT_PATH" ] && [ ! -L "$UNIT_PATH" ]; then
    [ "$load_state" = not-found ] && [ -z "$fragment" ] \
      || fail "missing validator unit is still installed in systemd"
    PYTHON=/home/ec2-user/arena-validator-venv311/bin/python3
    return
  fi
  [ -f "$UNIT_PATH" ] && [ ! -L "$UNIT_PATH" ] && [ -r "$UNIT_PATH" ] \
    || fail "installed validator unit must be a readable regular file"
  [ "$load_state" = loaded ] && [ "$fragment" = "$UNIT_PATH" ] \
    || fail "installed validator unit differs from systemd fragment"
  local text_status=0
  LC_ALL=C grep -aq '[^[:print:][:space:]]' "$UNIT_PATH" || text_status=$?
  [ "$text_status" -eq 1 ] || fail "installed validator unit must be plain text"
  local section="" line start="" pre="" start_count=0 pre_count=0 environment_file
  while IFS= read -r line || [ -n "$line" ]; do
    line="${line#"${line%%[![:space:]]*}"}"
    line="${line%"${line##*[![:space:]]}"}"
    case "$line" in
      ''|\#*|\;*) continue ;;
      *\\*|.include*) fail "unsupported validator unit continuation or include" ;;
      \[*\]) section="$line" ;;
      ExecStartPre=*)
        [ "$section" = '[Service]' ] || fail "validator start command outside Service section"
        pre_count=$((pre_count + 1)); pre="${line#ExecStartPre=}" ;;
      ExecStart=*)
        [ "$section" = '[Service]' ] || fail "validator start command outside Service section"
        start_count=$((start_count + 1)); start="${line#ExecStart=}" ;;
      ExecStart*) fail "unsupported installed validator start command" ;;
    esac
  done < "$UNIT_PATH"
  [ "$start_count" -eq 1 ] && [ "$pre_count" -eq 1 ] \
    || fail "installed validator unit must have one start and one preflight command"
  local start_pattern='^(/[A-Za-z0-9_./-]+)[[:space:]]+scripts/run_arena_validator[.]py[[:space:]]+--environment-file[[:space:]]+(/[A-Za-z0-9_./-]+)$'
  local pre_pattern='^(/[A-Za-z0-9_./-]+)[[:space:]]+scripts/run_arena_validator[.]py[[:space:]]+--environment-file[[:space:]]+(/[A-Za-z0-9_./-]+)[[:space:]]+--check-only$'
  [[ "$start" =~ $start_pattern ]] || fail "unsupported installed validator start command"
  PYTHON="${BASH_REMATCH[1]}"; environment_file="${BASH_REMATCH[2]}"
  [[ "$pre" =~ $pre_pattern ]] || fail "unsupported installed validator preflight command"
  [ "${BASH_REMATCH[1]}" = "$PYTHON" ] && [ "${BASH_REMATCH[2]}" = "$environment_file" ] \
    || fail "installed validator start and preflight commands differ"
}

validate_validator_stop_policy() {
  # systemctl stop uses the loaded old unit, not the candidate installed later.
  # Keep this separate from interpreter selection, including explicit overrides.
  local metadata key value seen="|" count=0 load_state="" kill_mode="" kill_signal="" stop_timeout=""
  metadata="$(systemctl show "$SERVICE" --all --property=LoadState,KillMode,KillSignal,TimeoutStopUSec)" \
    || fail "cannot inspect installed validator stop policy"
  while IFS='=' read -r key value; do
    [[ "$seen" != *"|$key|"* ]] || fail "ambiguous installed validator stop policy"
    seen+="$key|"; count=$((count + 1))
    case "$key" in
      LoadState) load_state="$value" ;;
      KillMode) kill_mode="$value" ;;
      KillSignal) kill_signal="$value" ;;
      TimeoutStopUSec) stop_timeout="$value" ;;
      *) fail "unexpected installed validator stop policy" ;;
    esac
  done <<< "$metadata"
  # A first installation has no old service to stop.
  [ "$load_state" != not-found ] || return 0
  [ "$count" -eq 4 ] && [ "$load_state" = loaded ] && [ "$kill_mode" = mixed ] \
    && { [ "$kill_signal" = 15 ] || [ "$kill_signal" = SIGTERM ]; } \
    || fail "installed validator stop policy cannot safely drain workers"
  # systemctl formats TimeoutStopUSec as a compound duration. Require at least
  # the committed unit's 9300-second drain budget; unknown formats fail closed.
  "$PYTHON" - "$stop_timeout" <<'PY' \
    || fail "installed validator stop timeout is insufficient or invalid"
import re
import sys
from fractions import Fraction

duration = sys.argv[1].strip()
if duration == "infinity":
    raise SystemExit(0)
units = {"us": Fraction(1, 1000000), "µs": Fraction(1, 1000000),
         "μs": Fraction(1, 1000000), "ms": Fraction(1, 1000), "s": 1,
         "min": 60, "h": 3600, "d": 86400, "w": 604800}
if not duration or len(duration) > 256:
    raise SystemExit(1)
end = 0
seconds = Fraction(0)
for match in re.finditer(r"([0-9]+(?:\.[0-9]+)?)(us|µs|μs|ms|s|min|h|d|w)", duration):
    if duration[end:match.start()].strip():
        raise SystemExit(1)
    seconds += Fraction(match[1]) * units[match[2]]
    end = match.end()
if not end or duration[end:].strip() or seconds < 9300:
    raise SystemExit(1)
PY
}

cleanup() {
  status=$?
  trap - EXIT
  if [ "$status" -ne 0 ] && [ "$DRAINED" -eq 1 ] && [ "$ACTIVATED" -eq 0 ]; then
    sudo systemctl stop "$SERVICE" >/dev/null 2>&1 || true
    if [ "$PROMOTED" -eq 1 ]; then
      if [ -n "$PREVIOUS_CURRENT_TARGET" ]; then
        ln -sfn "$PREVIOUS_CURRENT_TARGET" "$CURRENT_LINK.new"
        mv -Tf "$CURRENT_LINK.new" "$CURRENT_LINK"
      else
        [ ! -L "$CURRENT_LINK" ] || unlink "$CURRENT_LINK"
      fi
      if sudo test -f "$ROLLBACK/service.env"; then
        sudo install -m 0600 -o root -g root "$ROLLBACK/service.env" "$SERVICE_ENV"
      else
        sudo rm -f -- "$SERVICE_ENV"
      fi
      if sudo test -f "$ROLLBACK/service.unit"; then
        sudo install -m 0644 "$ROLLBACK/service.unit" "$UNIT_PATH"
      else
        sudo rm -f -- "$UNIT_PATH"
      fi
      sudo systemctl daemon-reload
    fi
    if [ "$OLD_ACTIVE" -eq 1 ]; then
      sudo systemctl start "$SERVICE" >/dev/null 2>&1 \
        || echo "ERROR: previous Arena validator service recovery failed" >&2
    fi
    echo "Restart failed; prior release and private configuration retained at $ROLLBACK" >&2
  fi
  if [ -n "$STAGE" ] && [[ "$STAGE" = "$RELEASE_ROOT"/.candidate.* ]]; then
    rm -rf -- "$STAGE"
  fi
  if [ -n "$CANDIDATE_SERVICE_ENV" ]; then
    sudo rm -f -- "$CANDIDATE_SERVICE_ENV"
  fi
  if [ -n "$SNAPSHOT_SERVICE_ENV" ]; then
    rm -f -- "$SNAPSHOT_SERVICE_ENV"
  fi
  exit "$status"
}
trap cleanup EXIT

for path in "$SOURCE_ROOT" "$RELEASE_ROOT" "$CURRENT_LINK" "$ENV_FILE" "$SERVICE_ENV" "$UNIT_PATH" "$LOCK_FILE"; do
  [[ "$path" =~ ^/[A-Za-z0-9_./-]+$ ]] && [ "$path" != / ] || fail "restart paths must be explicit absolute paths"
done
[[ "$SERVICE" =~ ^[A-Za-z0-9_.-]+\.service$ ]] || fail "invalid service name"
[[ "$READY_TIMEOUT" =~ ^[1-9][0-9]*$ ]] && [[ "$STOP_TIMEOUT" =~ ^[1-9][0-9]*$ ]] || fail "invalid restart timeout"
mkdir -p "$(dirname "$LOCK_FILE")" "$RELEASE_ROOT"
exec 9>"$LOCK_FILE"
flock -n 9 || fail "another validator restart owns the controller lock"
select_validator_python
[[ "$PYTHON" =~ ^/[A-Za-z0-9_./-]+$ ]] && [ "$PYTHON" != / ] \
  || fail "validator interpreter must be an explicit absolute path"
[ -x "$PYTHON" ] || fail "validator interpreter is unavailable"
cd "$SOURCE_ROOT"
git rev-parse --git-dir >/dev/null || fail "installed validator checkout is unavailable"
git diff --quiet && git diff --cached --quiet || fail "installed checkout has tracked changes"
git fetch --no-tags origin main
TARGET_SHA="$(git rev-parse --verify "$TARGET_REQUEST^{commit}")"
[[ "$TARGET_SHA" =~ ^[0-9a-f]{40}$ ]] || fail "target commit is invalid"
git merge-base --is-ancestor "$TARGET_SHA" origin/main || fail "target commit is not on origin/main"

STAGE="$(mktemp -d "$RELEASE_ROOT/.candidate.XXXXXX")"
git archive --format=tar "$TARGET_SHA" | tar -xf - -C "$STAGE"
printf '%s\n' "$TARGET_SHA" > "$STAGE/.release-commit"
test -r "$STAGE/lab_arena/local_weight_signer.py" && test -r "$STAGE/deploy/leadpoet-arena-validator.service" || fail "candidate local-wallet release is incomplete"
RELEASE="$RELEASE_ROOT/$TARGET_SHA"
if [ -d "$RELEASE" ]; then
  diff -qr "$STAGE" "$RELEASE" >/dev/null || fail "existing release differs from exact Git source"
else
  mv "$STAGE" "$RELEASE"; STAGE=""
fi

# Exact source does not update the selected environment. Check its actual trust
# dependencies before wallet/chain readiness or any change to the running service.
"$PYTHON" -I -B "$RELEASE/scripts/check_validator_python_security.py" \
  || fail "selected validator interpreter failed Python trust preflight"

[ -f "$ENV_FILE" ] && [ ! -L "$ENV_FILE" ] || fail "operator environment must be a regular file"
[ "$(stat -c %u "$ENV_FILE")" = "$(id -u)" ] || fail "operator Arena environment owner differs from restart identity"
[ "$(stat -c %a "$ENV_FILE")" = "600" ] || fail "operator Arena environment must have mode 0600"
CANDIDATE_SERVICE_ENV="${SERVICE_ENV}.candidate.$$"
SNAPSHOT_SERVICE_ENV="$RELEASE_ROOT/.service-env-snapshot.$$"
( cd "$RELEASE" && PYTHONPATH="$RELEASE" "$PYTHON" - "$ENV_FILE" "$SNAPSHOT_SERVICE_ENV" <<'PY'
import sys
from pathlib import Path
from scripts.run_arena_validator import write_environment_snapshot

write_environment_snapshot(Path(sys.argv[1]), Path(sys.argv[2]))
PY
)
sudo install -m 0600 -o root -g root "$SNAPSHOT_SERVICE_ENV" "$CANDIDATE_SERVICE_ENV"
rm -f -- "$SNAPSHOT_SERVICE_ENV"
SNAPSHOT_SERVICE_ENV=""

# Parse configuration as data, and create only the named durable directories.
read -r STATE_PATH RUNNER_PATH < <(
  cd "$RELEASE"
  PYTHONPATH="$RELEASE" "$PYTHON" - "$ENV_FILE" <<'PY'
import os,sys
from pathlib import Path
from scripts.run_arena_validator import load_environment
load_environment(Path(sys.argv[1]))
paths = [os.environ.get(name, "") for name in ("LAB_ARENA_VALIDATOR_STATE_DIR", "LAB_ARENA_RUNNER_WORK_DIR")]
for value in paths:
    path = Path(value)
    if not path.is_absolute() or str(path) in ("/", "/home", "/root", "/var", "/var/lib") or any(c.isspace() for c in value):
        raise SystemExit("Arena durable directory is unsafe")
print(*paths)
PY
)
for durable in "$STATE_PATH" "$RUNNER_PATH"; do
  [[ "$durable" = /* ]] || fail "Arena durable directory is not absolute"
  if sudo test -e "$durable" || sudo test -L "$durable"; then
    sudo test -d "$durable" && sudo test ! -L "$durable" || fail "Arena durable directory is unsafe"
  fi
  sudo install -d -m 0700 -o root -g root "$durable"
done

# Local key access, public signing-key pin and finalized chain identity must
# pass before the old service is touched. Scoring/runsc setup does not gate this.
( cd "$RELEASE" && sudo env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$RELEASE" \
  "$PYTHON" scripts/run_arena_validator.py --environment-file "$CANDIDATE_SERVICE_ENV" --check-only )

if [ -e "$CURRENT_LINK" ] || [ -L "$CURRENT_LINK" ]; then
  [ -L "$CURRENT_LINK" ] || fail "current release path is not a symlink"
  PREVIOUS_CURRENT_TARGET="$(readlink -f "$CURRENT_LINK")"
  [ -d "$PREVIOUS_CURRENT_TARGET" ] || fail "previous release is unavailable"
fi
ROLLBACK="$(mktemp -d "$RELEASE_ROOT/.rollback.XXXXXX")"
if sudo test -f "$SERVICE_ENV"; then sudo install -m 0600 "$SERVICE_ENV" "$ROLLBACK/service.env"; fi
if sudo test -f "$UNIT_PATH"; then sudo install -m 0644 "$UNIT_PATH" "$ROLLBACK/service.unit"; fi
sudo systemctl is-active --quiet "$SERVICE" && OLD_ACTIVE=1
validate_validator_stop_policy
# The paired coordinator supplies only a private nonce-bound permit, never
# gateway service credentials. Validate/consume after all normal preflight and
# immediately before the destructive step. The default standalone path is unchanged.
if [ -n "${VALIDATOR_PAIR_PERMIT:-}" ]; then
  [ "${VALIDATOR_PAIR_AUTHORITY_COMMIT:-}" = "$TARGET_SHA" ] || fail "paired restart authority differs"
  [[ "${VALIDATOR_PAIR_GENERATION:-}" =~ ^[1-9][0-9]*$ ]] || fail "paired restart generation is invalid"
  PYTHONPATH="$RELEASE" "$PYTHON" -B \
    "$RELEASE/scripts/lab_arena_restart_guard_handoff.py" consume-permit \
    --path "$VALIDATOR_PAIR_PERMIT" --candidate "$TARGET_SHA" \
    --invocation "${VALIDATOR_PAIR_INVOCATION:-}" --scope all \
    --nonce "${VALIDATOR_PAIR_NONCE:-}" --authority-commit "$TARGET_SHA" \
    --expected-generation "$VALIDATOR_PAIR_GENERATION" >/dev/null
fi
# Only now drain existing scoring work and stop the old weight process.
DRAINED=1
if [ "$(sudo systemctl show -p LoadState --value "$SERVICE")" != not-found ]; then
  timeout "$STOP_TIMEOUT" sudo systemctl stop "$SERVICE"
  [ "$(sudo systemctl show -p MainPID --value "$SERVICE")" = 0 ] || fail "old validator has not stopped"
fi
PROMOTED=1
ln -sfn "$RELEASE" "$CURRENT_LINK.new"
mv -Tf "$CURRENT_LINK.new" "$CURRENT_LINK"
sudo mv -f "$CANDIDATE_SERVICE_ENV" "$SERVICE_ENV"
CANDIDATE_SERVICE_ENV=""
sed -e "s|^WorkingDirectory=.*|WorkingDirectory=$CURRENT_LINK|" \
    -e "s|/home/ec2-user/.config/leadpoet/arena-validator-service.env|$SERVICE_ENV|g" \
    -e "s|^ExecStartPre=/usr/bin/python3 |ExecStartPre=$PYTHON |" \
    -e "s|^ExecStart=/usr/bin/python3 |ExecStart=$PYTHON |" \
    "$RELEASE/deploy/leadpoet-arena-validator.service" > "$ROLLBACK/candidate.unit"
sudo install -m 0644 "$ROLLBACK/candidate.unit" "$UNIT_PATH"
sudo systemctl daemon-reload
sudo systemctl enable "$SERVICE" >/dev/null
if ! sudo systemctl start "$SERVICE"; then
  echo "Initial Arena validator start failed; waiting for supervised systemd retries" >&2
fi
deadline=$((SECONDS + READY_TIMEOUT))
stable_pid=""
stable_since=$SECONDS
while [ "$SECONDS" -lt "$deadline" ]; do
  main_pid="$(sudo systemctl show -p MainPID --value "$SERVICE")"
  if sudo systemctl is-active --quiet "$SERVICE" && [[ "$main_pid" =~ ^[1-9][0-9]*$ ]]; then
    if [ "$main_pid" != "$stable_pid" ]; then stable_pid="$main_pid"; stable_since=$SECONDS; fi
    if [ "$((SECONDS - stable_since))" -ge 10 ]; then ACTIVATED=1; break; fi
  else
    stable_pid=""; stable_since=$SECONDS
  fi
  sleep 2
done
[ "$ACTIVATED" -eq 1 ] || fail "normal Arena validator did not remain ready"
echo "SUCCESS: normal Arena local-wallet validator is supervised at exact commit $TARGET_SHA"
echo "Previous release configuration retained at $ROLLBACK; finalized weight verification remains required."
