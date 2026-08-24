#!/bin/bash
# Gateway liveness watchdog.
#
# The gateway is launched by gw_restart.sh as a detached `setsid python -m
# gateway.main` background process. That script watches the new process only
# until its startup health check passes; once it exits, nothing on the host
# supervises the gateway. If gateway.main dies afterwards — a crash, an
# out-of-memory kill, an unhandled exception in a worker task — the host is
# left with nothing listening on :8000 and no automatic recovery, so the outage
# lasts until a person notices and re-runs gw_restart.sh by hand.
#
# In the three weeks to 2026-08-24 that happened five times, for 62 minutes,
# 95 minutes, 4h13m, 9h38m and (2026-08-24 17:08 UTC) the one this script was
# written for. Every one of them ended with a manual restart.
#
# This script is one liveness check, meant to be run every minute by
# gateway-liveness-watchdog.timer. It deliberately does NOT change how the
# gateway is launched: recovery is the supported path, `gw_restart.sh`, exactly
# what an operator would run.
#
# It is a stopgap, not a supervisor. It is built to fail safe rather than to
# recover aggressively:
#
#   * It requires REQUIRED_FAILURES consecutive bad checks before acting, so a
#     restart already in progress or a momentary stall is never interrupted.
#   * It never acts while the gateway restart lock is held — a deploy in flight
#     owns the host.
#   * It restarts at most once per COOLDOWN_SECONDS, and at most MAX_RESTARTS
#     times in WINDOW_SECONDS. A gateway that keeps dying is a bug for a human
#     to read, not something to relaunch in a loop against a script that
#     rebuilds enclaves.
#
# Exit status is always 0 unless the script itself is misconfigured; a failed
# liveness check is a normal outcome and is recorded in the log.

set -euo pipefail

LEADPOET_REPO_ROOT="${LEADPOET_REPO_ROOT:-/home/ec2-user/leadpoet_repo}"
GATEWAY_RESTART_SCRIPT="${GATEWAY_RESTART_SCRIPT:-$LEADPOET_REPO_ROOT/gw_restart.sh}"
GATEWAY_PYTHON_BIN="${GATEWAY_PYTHON_BIN:-/home/ec2-user/venv311/bin/python3}"
GATEWAY_HEALTH_URL="${GATEWAY_HEALTH_URL:-http://localhost:8000/health}"
GATEWAY_RESTART_LOCK_FILE="${GATEWAY_RESTART_LOCK_FILE:-/home/ec2-user/.config/leadpoet/gateway-restart.lock}"

WATCHDOG_ROOT="${GATEWAY_WATCHDOG_ROOT:-/home/ec2-user/.config/leadpoet/watchdog}"
WATCHDOG_STATE_FILE="$WATCHDOG_ROOT/state"
WATCHDOG_LOG_FILE="${GATEWAY_WATCHDOG_LOG_FILE:-/home/ec2-user/gateway/watchdog.log}"
WATCHDOG_LOCK_FILE="$WATCHDOG_ROOT/watchdog.lock"

# Consecutive failed checks before a restart is attempted. At a 60s timer this
# is three minutes of confirmed-dead before we touch anything.
REQUIRED_FAILURES="${GATEWAY_WATCHDOG_REQUIRED_FAILURES:-3}"
# Seconds a health probe may take before it counts as a failure.
HEALTH_TIMEOUT_SECONDS="${GATEWAY_WATCHDOG_HEALTH_TIMEOUT_SECONDS:-5}"
# Minimum gap between two watchdog-initiated restarts.
COOLDOWN_SECONDS="${GATEWAY_WATCHDOG_COOLDOWN_SECONDS:-1800}"
# Circuit breaker: at most MAX_RESTARTS watchdog restarts per WINDOW_SECONDS.
MAX_RESTARTS="${GATEWAY_WATCHDOG_MAX_RESTARTS:-3}"
WINDOW_SECONDS="${GATEWAY_WATCHDOG_WINDOW_SECONDS:-21600}"

mkdir -p "$WATCHDOG_ROOT" "$(dirname "$WATCHDOG_LOG_FILE")"
chmod 700 "$WATCHDOG_ROOT"

log() {
  printf '%s watchdog: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >>"$WATCHDOG_LOG_FILE"
}

# One watchdog at a time. A restart attempt runs far longer than the timer
# interval, so overlapping invocations are expected and must not stack.
exec 8>"$WATCHDOG_LOCK_FILE"
chmod 600 "$WATCHDOG_LOCK_FILE"
if ! flock -n 8; then
  exit 0
fi

# State: "<consecutive_failures> <last_restart_epoch> <restart_epochs_csv>"
consecutive_failures=0
last_restart_at=0
restart_history=""
if [ -r "$WATCHDOG_STATE_FILE" ]; then
  # shellcheck disable=SC2162
  read consecutive_failures last_restart_at restart_history <"$WATCHDOG_STATE_FILE" || true
  consecutive_failures="${consecutive_failures:-0}"
  last_restart_at="${last_restart_at:-0}"
  restart_history="${restart_history:-}"
  [ "$restart_history" = "-" ] && restart_history=""
fi

now="$(date -u +%s)"

save_state() {
  printf '%s %s %s\n' "$consecutive_failures" "$last_restart_at" "${restart_history:--}" \
    >"$WATCHDOG_STATE_FILE"
  chmod 600 "$WATCHDOG_STATE_FILE"
}

gateway_process_present() {
  pgrep -f "^$GATEWAY_PYTHON_BIN -u -m gateway[.]main$" >/dev/null 2>&1
}

gateway_health_ok() {
  timeout "$HEALTH_TIMEOUT_SECONDS" \
    curl -fsS --max-time "$HEALTH_TIMEOUT_SECONDS" "$GATEWAY_HEALTH_URL" >/dev/null 2>&1
}

restart_in_progress() {
  # gw_restart.sh holds this lock for the whole of a deploy. If we cannot take
  # it, a restart already owns the host and the gateway being down is expected.
  [ -e "$GATEWAY_RESTART_LOCK_FILE" ] || return 1
  # Probe in a subshell so the descriptor and the lock die with it either way.
  ( flock -n 9 ) 9>>"$GATEWAY_RESTART_LOCK_FILE" 2>/dev/null && return 1
  return 0
}

# Watchdog-initiated restarts inside the rolling window.
recent_restart_count() {
  local kept="" stamp count=0
  local cutoff=$((now - WINDOW_SECONDS))
  local IFS=,
  for stamp in $restart_history; do
    [ -n "$stamp" ] || continue
    if [ "$stamp" -gt "$cutoff" ]; then
      kept="${kept:+$kept,}$stamp"
      count=$((count + 1))
    fi
  done
  restart_history="$kept"
  printf '%s' "$count"
}

process_present=no
health_ok=no
if gateway_process_present; then process_present=yes; fi
if gateway_health_ok; then health_ok=yes; fi

if [ "$process_present" = yes ] && [ "$health_ok" = yes ]; then
  if [ "$consecutive_failures" -ne 0 ]; then
    log "gateway healthy again after $consecutive_failures failed check(s)"
  fi
  consecutive_failures=0
  save_state
  exit 0
fi

consecutive_failures=$((consecutive_failures + 1))
log "liveness check failed ($consecutive_failures/$REQUIRED_FAILURES): process_present=$process_present health_ok=$health_ok"

if [ "$consecutive_failures" -lt "$REQUIRED_FAILURES" ]; then
  save_state
  exit 0
fi

if restart_in_progress; then
  log "a gateway restart already holds the restart lock — standing down"
  save_state
  exit 0
fi

if [ $((now - last_restart_at)) -lt "$COOLDOWN_SECONDS" ]; then
  log "within the $((COOLDOWN_SECONDS / 60))m cooldown since the last watchdog restart — standing down"
  save_state
  exit 0
fi

recent="$(recent_restart_count)"
if [ "$recent" -ge "$MAX_RESTARTS" ]; then
  log "circuit breaker open: $recent watchdog restarts in the last $((WINDOW_SECONDS / 3600))h. The gateway is not staying up and needs a human — not restarting."
  save_state
  exit 0
fi

if [ ! -x "$GATEWAY_RESTART_SCRIPT" ]; then
  log "ERROR: $GATEWAY_RESTART_SCRIPT is not executable — cannot recover"
  save_state
  exit 0
fi

log "gateway down for $consecutive_failures consecutive checks — running $GATEWAY_RESTART_SCRIPT"
last_restart_at="$now"
restart_history="${restart_history:+$restart_history,}$now"
consecutive_failures=0
save_state

if "$GATEWAY_RESTART_SCRIPT" >>"$WATCHDOG_LOG_FILE" 2>&1; then
  log "restart completed"
else
  log "ERROR: restart exited non-zero — see the output above and the gateway log"
fi

exit 0
