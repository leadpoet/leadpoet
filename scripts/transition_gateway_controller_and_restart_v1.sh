#!/bin/bash
set -Eeuo pipefail

if [ "$#" -ne 2 ] || [ "$1" != "--commit" ] \
    || ! [[ "$2" =~ ^[0-9a-f]{40}$ ]]; then
  echo "usage: $0 --commit <40-character origin/main SHA>" >&2
  exit 2
fi

candidate="$2"
repository="${LEADPOET_REPO_ROOT:-/home/ec2-user/leadpoet_repo}"
host_restart="${GATEWAY_HOST_RESTART_SCRIPT:-/home/ec2-user/gw_restart.sh}"
controller_root="${GATEWAY_RESTART_CONTROLLER_ROOT:-/home/ec2-user/.config/leadpoet/restart-controller/gateway}"
transition_root="$(mktemp -d /tmp/gateway-controller-transition.XXXXXXXX)"
restart_lock="${GATEWAY_RESTART_LOCK_FILE:-/home/ec2-user/.config/leadpoet/gateway-restart.lock}"
mkdir -p "$(dirname "$restart_lock")"
exec 9>"$restart_lock"
chmod 600 "$restart_lock"
flock -n 9 || { echo "ERROR: another gateway restart is active" >&2; exit 1; }

cleanup() {
  git -C "$repository" worktree remove --force "$transition_root" >/dev/null 2>&1 || true
}
trap cleanup EXIT

git -C "$repository" fetch --no-tags origin main
test "$(git -C "$repository" rev-parse origin/main)" = "$candidate"
git -C "$repository" worktree add --detach "$transition_root" "$candidate"
test "$(git -C "$transition_root" rev-parse HEAD)" = "$candidate"
test -z "$(git -C "$transition_root" status --porcelain=v1 --untracked-files=all)"

python3 "$transition_root/scripts/install_gateway_controller_transition_v1.py" \
  --repo "$transition_root" \
  --commit "$candidate" \
  --controller-root "$controller_root" \
  --host-restart "$host_restart" \
  --lock "$restart_lock" \
  --lock-fd 9

migration_hash="$(sha256sum "$transition_root/scripts/203-retire-legacy-incentive-weight-bridge.sql" | awk '{print $1}')"
cleanup
trap - EXIT

export GATEWAY_MIGRATION_203_SQL_SHA256="$migration_hash"
export GATEWAY_RESTART_LOCK_FILE="$restart_lock"
export GATEWAY_RESTART_LOCK_HELD=1
exec "$host_restart" --commit "$candidate"
