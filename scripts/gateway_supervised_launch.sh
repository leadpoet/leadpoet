#!/bin/bash
# ExecStart for leadpoet-gateway.service.
#
# gw_restart.sh builds the gateway's launch environment across ~4,000 lines of
# secret loading, release pinning and preflight. Rather than re-derive any of
# that in a unit file, the restart script snapshots the exact environment it
# was about to launch with, and this script replays that snapshot and execs the
# gateway. `exec` means systemd supervises gateway.main itself: the process
# systemd watches, and the process `pgrep -f "^<python> -u -m gateway.main$"`
# finds, are the same PID they were before this unit existed.
#
# Everything below either succeeds or exits non-zero; with Restart=always a
# non-zero exit is retried, so a transient failure here self-heals.

set -uo pipefail

LAUNCH_ENV="${GATEWAY_SUPERVISED_LAUNCH_ENV:-/home/ec2-user/.config/leadpoet/gateway-launch-env.sh}"

if [ ! -r "$LAUNCH_ENV" ]; then
  echo "gateway_supervised_launch: no launch environment at $LAUNCH_ENV;" \
    "run gw_restart.sh to deploy the gateway and write one" >&2
  exit 78
fi

# The snapshot is `export -p` output: a sequence of `declare -x NAME=value`
# lines, safe to source under `set +u`.
set +u
# shellcheck disable=SC1090
. "$LAUNCH_ENV"
set -u

# Variables that exist only for the duration of a restart: helper paths, proof
# descriptors and release-handoff state. gw_restart.sh strips these from its
# own direct launch with `env -u`, so strip them here too. Keeping this list in
# sync with that one is what tests/test_gateway_supervision.py checks.
unset GATEWAY_MINER_MAINTENANCE_PROOF_FD
unset GATEWAY_REBENCHMARK_RETRY_RECONCILIATION_HELPER
unset GATEWAY_GIT_HELPER
unset GATEWAY_EXACT_COMMIT_HELPER
unset GATEWAY_HOST_MEMORY_GUARD_PATH
unset GATEWAY_RESTART_AUTHORITY_ROOT
unset GATEWAY_RESTART_AUTHORITY_COMMIT
unset GATEWAY_ACTIVE_RELEASE_RESTART_INVOCATION_ID
unset GATEWAY_PAIRED_ACTIVE_RELEASE_REQUIRED
unset GATEWAY_ACTIVE_RELEASE_FALLBACK_CONTEXT
unset GATEWAY_PAIRED_DESTRUCTIVE_HANDOFF_FILE
unset GATEWAY_PAIRED_DESTRUCTIVE_HANDOFF_NONCE
unset GATEWAY_PAIRED_DESTRUCTIVE_HANDOFF_TIMEOUT_SECONDS
unset GATEWAY_VALIDATOR_RELEASE_REQUIREMENTS
unset GATEWAY_COUNTERPART_RELEASE_LINEAGE

GATEWAY_PYTHON_BIN="${GATEWAY_PYTHON_BIN:-/home/ec2-user/venv311/bin/python3}"
LEADPOET_REPO_ROOT="${LEADPOET_REPO_ROOT:-/home/ec2-user/leadpoet_repo}"
GATEWAY_LOG_FILE="${GATEWAY_LOG_FILE:-/home/ec2-user/gateway/gateway.log}"

if [ ! -x "$GATEWAY_PYTHON_BIN" ]; then
  echo "gateway_supervised_launch: $GATEWAY_PYTHON_BIN is not executable" >&2
  exit 78
fi

cd "$LEADPOET_REPO_ROOT" || exit 78

mkdir -p "$(dirname "$GATEWAY_LOG_FILE")" 2>/dev/null || true
# Append rather than truncate: under supervision the log spans restarts, and
# `tail -f $GATEWAY_LOG_FILE` is how the crash before a restart gets read.
exec >>"$GATEWAY_LOG_FILE" 2>&1 </dev/null

exec "$GATEWAY_PYTHON_BIN" -u -m gateway.main
