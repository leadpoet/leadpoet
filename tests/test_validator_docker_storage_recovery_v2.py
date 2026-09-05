import os
from pathlib import Path
import subprocess
import time
from typing import Optional

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _write_executable(path: Path, source: str) -> None:
    path.write_text(source, encoding="utf-8")
    path.chmod(0o755)


def _run_recovery(
    tmp_path: Path,
    *,
    available: int,
    containers: int = 0,
    images: int = 0,
    volumes: int = 0,
    layerdb_images: int = 0,
    layerdb_mounts: int = 0,
    overlay_directories: int = 0,
    layerdb_image_directory_exists: bool = True,
    layerdb_mount_directory_exists: bool = True,
    overlay_directory_exists: bool = True,
    containerd_containers: int = 0,
    containerd_tasks: int = 0,
    containerd_running_tasks: int = 0,
    running_tasks_clear_after: int = -1,
    task_probe_failures: int = 0,
    system_prune_failures: int = 0,
    non_moby_namespaces: int = 0,
    moby_shims: int = 0,
    live_runtime_min_free_bytes: int = 18_000_000_000,
    stale_overlay_mounts: int = 0,
    stale_layer_records: int = 0,
    stale_mount_records: int = 0,
    stale_overlay_directories: int = 0,
    stale_overlay_links: int = 0,
    available_after_stale_reclaim: Optional[int] = None,
    available_after_builder_prune: Optional[int] = None,
    images_after_builder_prune: Optional[int] = None,
    live_restore_enabled: bool = True,
    host_gateway_live: bool = False,
    host_gateway_live_after_inventory: bool = False,
    allow_live_host_gateway_prune: bool = False,
    require_zero_runtime_reconcile: Optional[str] = None,
    host_gateway_exits_during_live_prune: bool = False,
    host_gateway_start_time_changes_during_live_prune: bool = False,
    host_gateway_exits_during_daemon_reconcile: bool = False,
    host_gateway_appears_during_daemon_reconcile: bool = False,
    runtime_appears_after_stale_reclaim: bool = False,
    docker_root: str = "/var/lib/docker",
    docker_root_after_builder_prune: Optional[str] = None,
    fail_shim_probe: bool = False,
    malformed_shim_probe: bool = False,
    hang_shim_probe: bool = False,
    zero_shim_probe_succeeds: bool = False,
    nonzero_shim_probe_reports_no_match: bool = False,
    fail_stale_reclaimer: bool = False,
    malformed_stale_reclaimer: bool = False,
    pre_reclaim_audit_bool_field: Optional[str] = None,
    reclaim_count_mismatch_field: Optional[str] = None,
    post_reclaim_stale_field: Optional[str] = None,
    post_reclaim_active_count_field: Optional[str] = None,
    change_audit_manifest_after_stale_reclaim: bool = False,
    fail_zero_runtime_reconciler: bool = False,
    zero_runtime_reconcile_clears_orphan_metadata: bool = False,
    malformed_zero_runtime_reconciler: bool = False,
    zero_runtime_boolean_field: Optional[str] = None,
    change_audit_manifest_after_reconcile: bool = False,
    change_audit_manifest_after_post_prune: bool = False,
    post_reconcile_active_count_field: Optional[str] = None,
    post_prune_active_count_field: Optional[str] = None,
    stale_audit_before_reconcile: bool = False,
    stale_audit_after_reconcile: bool = False,
    fail_umount: bool = False,
    fail_systemctl_stop: bool = False,
    fail_daemon_recovery: bool = False,
    hang_daemon_probe: bool = False,
    hang_ctr_probe: bool = False,
    hang_systemctl_start: bool = False,
    hang_systemctl_stop: bool = False,
    prune_attempts: Optional[int] = None,
    daemon_ready_attempts: int = 3,
    daemon_probe_timeout_seconds: int = 1,
) -> tuple[subprocess.CompletedProcess[str], str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    if host_gateway_live:
        gateway_proc = proc_root / "200"
        gateway_proc.mkdir()
        (gateway_proc / "status").write_text(
            "Name:\tpython3\nPPid:\t1\n",
            encoding="utf-8",
        )
        (gateway_proc / "cmdline").write_bytes(
            b"/home/ec2-user/venv311/bin/python3\0"
            b"-u\0-m\0gateway.main\0"
        )
        (gateway_proc / "stat").write_text(
            "200 (python3) " + " ".join(["S", *(["0"] * 18), "12345"]) + "\n",
            encoding="utf-8",
        )
    sudo_log = tmp_path / "sudo.log"

    _write_executable(
        bin_dir / "df",
        """#!/bin/bash
available="$FAKE_AVAILABLE"
if [ -f "$FAKE_STALE_RECLAIM_MARKER" ]; then
  available="$FAKE_AVAILABLE_AFTER_STALE_RECLAIM"
elif [ -f "$FAKE_BUILDER_PRUNE_MARKER" ]; then
  available="$FAKE_AVAILABLE_AFTER_BUILDER_PRUNE"
fi
printf 'Avail\\n%s\\n' "$available"
""",
    )
    _write_executable(
        bin_dir / "docker",
        """#!/bin/bash
emit_rows() {
  local count="$1"
  local index=0
  while [ "$index" -lt "$count" ]; do
    printf 'row-%s\\n' "$index"
    index=$((index + 1))
  done
}
emit_image_rows() {
  local count="$1"
  local index=0
  while [ "$index" -lt "$count" ]; do
    printf 'sha256:%064x\n' "$((index + 1))"
    index=$((index + 1))
  done
}
printf 'docker %s\n' "$*" >> "$FAKE_OPERATION_LOG"
if [ "${1:-}" = "info" ] && [ -f "$FAKE_DAEMONS_STOPPED_MARKER" ]; then
  if [ "$FAKE_HANG_DAEMON_PROBE" = "1" ]; then
    exec /bin/sleep 60
  fi
  probe=0
  if [ -f "$FAKE_DAEMON_READY_PROBE_STATE" ]; then
    probe="$(cat "$FAKE_DAEMON_READY_PROBE_STATE")"
  fi
  printf '%s\n' "$((probe + 1))" > "$FAKE_DAEMON_READY_PROBE_STATE"
  exit 1
fi
case "${1:-}:${2:-}" in
  info:)
    exit 0
    ;;
  image:prune)
    touch "$FAKE_IMAGE_PRUNE_MARKER"
    exit 0
    ;;
  builder:prune)
    touch "$FAKE_BUILDER_PRUNE_MARKER"
    prune_count=0
    if [ -f "$FAKE_BUILDER_PRUNE_STATE" ]; then
      prune_count="$(cat "$FAKE_BUILDER_PRUNE_STATE")"
    fi
    printf '%s\n' "$((prune_count + 1))" > "$FAKE_BUILDER_PRUNE_STATE"
    if [ "$FAKE_HOST_GATEWAY_EXITS_DURING_LIVE_PRUNE" = "1" ]; then
      rm -rf "$LEADPOET_PROC_ROOT/200"
    fi
    if [ "$FAKE_HOST_GATEWAY_START_TIME_CHANGES_DURING_LIVE_PRUNE" = "1" ]; then
      printf '200 (python3) S 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 54321\n' \
        > "$LEADPOET_PROC_ROOT/200/stat"
    fi
    exit 0
    ;;
  system:prune)
    attempt=0
    if [ -f "$FAKE_SYSTEM_PRUNE_STATE" ]; then
      attempt="$(cat "$FAKE_SYSTEM_PRUNE_STATE")"
    fi
    attempt=$((attempt + 1))
    printf '%s\\n' "$attempt" > "$FAKE_SYSTEM_PRUNE_STATE"
    if [ "$attempt" -le "$FAKE_SYSTEM_PRUNE_FAILURES" ]; then
      exit 1
    fi
    exit 0
    ;;
  ps:-aq)
    container_count="$FAKE_CONTAINERS"
    if [ -f "$FAKE_STALE_RECLAIM_MARKER" ] \
        && [ "$FAKE_RUNTIME_APPEARS_AFTER_STALE_RECLAIM" = "1" ]; then
      container_count=1
    fi
    emit_rows "$container_count"
    exit 0
    ;;
  inspect:--format)
    shift 3
    for container_id in "$@"; do
      index="${container_id#row-}"
      printf '/var/lib/docker/overlay2/%064x/merged\n' "$index"
    done
    exit 0
    ;;
  image:ls)
    image_count="$FAKE_IMAGES"
    if [ -f "$FAKE_BUILDER_PRUNE_MARKER" ] \
        && [ "$FAKE_IMAGES_AFTER_BUILDER_PRUNE" -ge 0 ]; then
      image_count="$FAKE_IMAGES_AFTER_BUILDER_PRUNE"
    fi
    emit_image_rows "$image_count"
    exit 0
    ;;
  volume:ls)
    emit_rows "$FAKE_VOLUMES"
    exit 0
    ;;
  info:--format)
    case "${3:-}" in
      *LiveRestoreEnabled*) printf '%s\\n' "$FAKE_LIVE_RESTORE_ENABLED" ;;
      *)
        docker_root="$FAKE_DOCKER_ROOT"
        if [ -f "$FAKE_BUILDER_PRUNE_MARKER" ]; then
          docker_root="$FAKE_DOCKER_ROOT_AFTER_BUILDER_PRUNE"
        fi
        printf '%s\\n' "$docker_root"
        ;;
    esac
    exit 0
    ;;
  info:)
    # Bare `docker info` availability/readiness probes (pre-inventory preamble
    # and the post-reset readiness loop) — the fake daemon is always healthy.
    exit 0
    ;;
esac
exit 2
""",
    )
    _write_executable(
        bin_dir / "findmnt",
        """#!/bin/bash
index=0
while [ "$index" -lt "$FAKE_CONTAINERS" ]; do
  printf '/var/lib/docker/overlay2/%064x/merged\\n' "$index"
  index=$((index + 1))
done
if [ ! -f "$FAKE_STALE_RECLAIM_MARKER" ]; then
  index=0
  while [ "$index" -lt "$FAKE_STALE_OVERLAY_MOUNTS" ]; do
    printf '/var/lib/docker/overlay2/%064x/merged\\n' "$((65536 + index))"
    index=$((index + 1))
  done
fi
exit 0
""",
    )
    _write_executable(
        bin_dir / "sleep",
        """#!/bin/bash
exit 0
""",
    )
    _write_executable(
        bin_dir / "flock",
        """#!/bin/bash
# Lock semantics are covered separately; this recovery fixture has one process.
exit 0
""",
    )
    _write_executable(
        bin_dir / "sudo",
        """#!/bin/bash
command="$1"
shift
printf 'sudo %s %s\n' "$command" "$*" >> "$FAKE_OPERATION_LOG"
emit_rows() {
  local count="$1"
  local index=0
  while [ "$index" -lt "$count" ]; do
    printf '/fake/%s\\n' "$index"
    index=$((index + 1))
  done
}
case "$command" in
  python3)
    if [ -f "$FAKE_DAEMON_RECONCILE_MARKER" ] \
        && [ "$FAKE_ZERO_RUNTIME_RECONCILE_CLEARS_ORPHAN_METADATA" = "1" ]; then
      exit 0
    fi
    [ "$FAKE_LAYERDB_IMAGES" -eq 0 ] \
      && [ "$FAKE_LAYERDB_MOUNTS" -eq 0 ] \
      && [ "$FAKE_OVERLAY_DIRECTORIES" -eq 0 ]
    exit $?
    ;;
  env)
    printf '%s %s\n' "$command" "$*" >> "$FAKE_SUDO_LOG"
    case "$*" in
      *docker_zero_runtime_reconciler_v2.py*)
        if [ "$FAKE_FAIL_ZERO_RUNTIME_RECONCILER" = "1" ]; then
          exit 1
        fi
        touch "$FAKE_DAEMON_RECONCILE_MARKER"
        if [ "$FAKE_HOST_GATEWAY_EXITS_DURING_DAEMON_RECONCILE" = "1" ]; then
          rm -rf "$LEADPOET_PROC_ROOT/200"
        fi
        if [ "$FAKE_HOST_GATEWAY_APPEARS_DURING_DAEMON_RECONCILE" = "1" ]; then
          mkdir -p "$LEADPOET_PROC_ROOT/202"
          printf 'Name:\tpython3\nPPid:\t1\n' \
            > "$LEADPOET_PROC_ROOT/202/status"
          printf '/home/ec2-user/venv311/bin/python3\\0-u\\0-m\\0gateway.main\\0' \
            > "$LEADPOET_PROC_ROOT/202/cmdline"
          printf '202 (python3) S 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 78901\n' \
            > "$LEADPOET_PROC_ROOT/202/stat"
        fi
        if [ "$FAKE_MALFORMED_ZERO_RUNTIME_RECONCILER" = "1" ]; then
          printf '{}\n'
          exit 0
        fi
        container_count=0
        containerd_container_count=0
        containerd_task_count=0
        image_count="$FAKE_IMAGES"
        moby_shim_count=0
        root_device=1
        root_inode=2
        case "$FAKE_ZERO_RUNTIME_BOOLEAN_FIELD" in
          container_count) container_count=true ;;
          containerd_container_count) containerd_container_count=true ;;
          containerd_task_count) containerd_task_count=true ;;
          image_count) image_count=true ;;
          moby_shim_count) moby_shim_count=true ;;
          root_device) root_device=true ;;
          root_inode) root_inode=true ;;
        esac
        printf '{"container_count":%s,"containerd_container_count":%s,"containerd_task_count":%s,"docker_root":"/var/lib/docker","image_count":%s,"image_manifest_hash":"sha256:%064d","moby_shim_count":%s,"restart_performed":true,"root_device":%s,"root_inode":%s,"schema_version":"leadpoet.docker_zero_runtime_reconcile.v1","status":"ready"}\n' \
          "$container_count" "$containerd_container_count" "$containerd_task_count" "$image_count" 0 "$moby_shim_count" "$root_device" "$root_inode"
        ;;
      *docker_live_restore_reconciler_v2.py*)
        printf '{"config_changed":true,"container_count":%s,"image_count":%s,"manifest_hash":"sha256:%064d","schema_version":"leadpoet.docker_live_restore_reconcile.v1","status":"ready"}\n' \
          "$FAKE_CONTAINERS" "$FAKE_IMAGES" 0
        ;;
      *docker_stale_mount_reclaimer_v2.py*--audit-only*)
        manifest_digit=0
        active_container_count=0
        active_image_count="$FAKE_IMAGES"
        active_layer_count="$FAKE_LAYERDB_IMAGES"
        active_mount_count=0
        active_overlay_dir_count="$FAKE_LAYERDB_IMAGES"
        mounted_overlay_count="$FAKE_STALE_OVERLAY_MOUNTS"
        stale_layer_record_count="$FAKE_STALE_LAYER_RECORDS"
        stale_mount_record_count="$FAKE_STALE_MOUNT_RECORDS"
        stale_overlay_dir_count="$FAKE_STALE_OVERLAY_DIRECTORIES"
        stale_overlay_link_count="$FAKE_STALE_OVERLAY_LINKS"
        if [ -f "$FAKE_STALE_RECLAIM_MARKER" ]; then
          mounted_overlay_count=0
          stale_layer_record_count=0
          stale_mount_record_count=0
          stale_overlay_dir_count=0
          stale_overlay_link_count=0
        fi
        if [ -f "$FAKE_STALE_RECLAIM_MARKER" ] \
            && [ ! -f "$FAKE_DAEMON_RECONCILE_MARKER" ] \
            && [ "$FAKE_CHANGE_AUDIT_MANIFEST_AFTER_STALE_RECLAIM" = "1" ]; then
          manifest_digit=1
        fi
        if [ -f "$FAKE_DAEMON_RECONCILE_MARKER" ] \
            && [ "$FAKE_CHANGE_AUDIT_MANIFEST_AFTER_RECONCILE" = "1" ]; then
          manifest_digit=1
        fi
        if [ -f "$FAKE_BUILDER_PRUNE_STATE" ] \
            && [ "$(cat "$FAKE_BUILDER_PRUNE_STATE")" -ge 2 ] \
            && [ "$FAKE_CHANGE_AUDIT_MANIFEST_AFTER_POST_PRUNE" = "1" ]; then
          manifest_digit=1
        fi
        if [ -f "$FAKE_DAEMON_RECONCILE_MARKER" ] \
            && [ "$FAKE_STALE_AUDIT_AFTER_RECONCILE" = "1" ]; then
          stale_layer_record_count=1
        fi
        if [ -f "$FAKE_STALE_RECLAIM_MARKER" ] \
            && [ ! -f "$FAKE_DAEMON_RECONCILE_MARKER" ] \
            && [ "$FAKE_STALE_AUDIT_BEFORE_RECONCILE" = "1" ]; then
          stale_layer_record_count=1
        fi
        if [ -f "$FAKE_STALE_RECLAIM_MARKER" ] \
            && [ ! -f "$FAKE_DAEMON_RECONCILE_MARKER" ]; then
          case "$FAKE_POST_RECLAIM_STALE_FIELD" in
            stale_layer_record_count) stale_layer_record_count=1 ;;
            mounted_overlay_count) mounted_overlay_count=1 ;;
            stale_mount_record_count) stale_mount_record_count=1 ;;
            stale_overlay_dir_count) stale_overlay_dir_count=1 ;;
            stale_overlay_link_count) stale_overlay_link_count=1 ;;
          esac
          case "$FAKE_POST_RECLAIM_ACTIVE_COUNT_FIELD" in
            active_container_count) active_container_count=1 ;;
            active_image_count) active_image_count=$((active_image_count + 1)) ;;
            active_layer_count) active_layer_count=$((active_layer_count + 1)) ;;
            active_mount_count) active_mount_count=1 ;;
            active_overlay_dir_count) active_overlay_dir_count=$((active_overlay_dir_count + 1)) ;;
          esac
        fi
        if [ -f "$FAKE_DAEMON_RECONCILE_MARKER" ]; then
          case "$FAKE_POST_RECONCILE_ACTIVE_COUNT_FIELD" in
            active_container_count) active_container_count=1 ;;
            active_image_count) active_image_count=$((active_image_count + 1)) ;;
            active_layer_count) active_layer_count=$((active_layer_count + 1)) ;;
            active_mount_count) active_mount_count=1 ;;
            active_overlay_dir_count) active_overlay_dir_count=$((active_overlay_dir_count + 1)) ;;
          esac
        fi
        if [ -f "$FAKE_BUILDER_PRUNE_STATE" ] \
            && [ "$(cat "$FAKE_BUILDER_PRUNE_STATE")" -ge 2 ]; then
          case "$FAKE_POST_PRUNE_ACTIVE_COUNT_FIELD" in
            active_container_count) active_container_count=1 ;;
            active_image_count) active_image_count=$((active_image_count + 1)) ;;
            active_layer_count) active_layer_count=$((active_layer_count + 1)) ;;
            active_mount_count) active_mount_count=1 ;;
            active_overlay_dir_count) active_overlay_dir_count=$((active_overlay_dir_count + 1)) ;;
          esac
        fi
        if [ ! -f "$FAKE_STALE_RECLAIM_MARKER" ]; then
          case "$FAKE_PRE_RECLAIM_AUDIT_BOOL_FIELD" in
            active_container_count) active_container_count=true ;;
            active_image_count) active_image_count=true ;;
            active_layer_count) active_layer_count=true ;;
            active_mount_count) active_mount_count=true ;;
            active_overlay_dir_count) active_overlay_dir_count=true ;;
            mounted_overlay_count) mounted_overlay_count=true ;;
            stale_layer_record_count) stale_layer_record_count=true ;;
            stale_mount_record_count) stale_mount_record_count=true ;;
            stale_overlay_dir_count) stale_overlay_dir_count=true ;;
            stale_overlay_link_count) stale_overlay_link_count=true ;;
          esac
        fi
        printf '{"active_container_count":%s,"active_image_count":%s,"active_layer_count":%s,"active_manifest_hash":"sha256:%064d","active_mount_count":%s,"active_overlay_dir_count":%s,"docker_root":"/var/lib/docker","mounted_overlay_count":%s,"schema_version":"leadpoet.docker_stale_mount_audit.v3","stale_layer_record_count":%s,"stale_mount_record_count":%s,"stale_overlay_dir_count":%s,"stale_overlay_link_count":%s,"status":"ready"}\n' \
          "$active_container_count" "$active_image_count" "$active_layer_count" "$manifest_digit" "$active_mount_count" "$active_overlay_dir_count" "$mounted_overlay_count" "$stale_layer_record_count" "$stale_mount_record_count" "$stale_overlay_dir_count" "$stale_overlay_link_count"
        ;;
      *docker_stale_mount_reclaimer_v2.py*)
        if [ "$FAKE_FAIL_STALE_RECLAIMER" = "1" ]; then
          exit 1
        fi
        touch "$FAKE_STALE_RECLAIM_MARKER"
        if [ "$FAKE_MALFORMED_STALE_RECLAIMER" = "1" ]; then
          printf '{}\n'
          exit 0
        fi
        active_container_count="$FAKE_CONTAINERS"
        active_image_count="$FAKE_IMAGES"
        active_layer_count="$FAKE_LAYERDB_IMAGES"
        active_mount_count="$FAKE_CONTAINERS"
        mounted_overlay_count="$((FAKE_CONTAINERS + FAKE_STALE_OVERLAY_MOUNTS))"
        reclaimed_layer_record_count="$FAKE_STALE_LAYER_RECORDS"
        reclaimed_mount_count="$FAKE_STALE_OVERLAY_MOUNTS"
        reclaimed_mount_record_count="$FAKE_STALE_MOUNT_RECORDS"
        reclaimed_overlay_dir_count="$FAKE_STALE_OVERLAY_DIRECTORIES"
        reclaimed_overlay_link_count="$FAKE_STALE_OVERLAY_LINKS"
        case "$FAKE_RECLAIM_COUNT_MISMATCH_FIELD" in
          active_container_count) active_container_count=$((active_container_count + 1)) ;;
          active_image_count) active_image_count=$((active_image_count + 1)) ;;
          active_layer_count) active_layer_count=$((active_layer_count + 1)) ;;
          active_mount_count) active_mount_count=$((active_mount_count + 1)) ;;
          mounted_overlay_count) mounted_overlay_count=$((mounted_overlay_count + 1)) ;;
          reclaimed_layer_record_count) reclaimed_layer_record_count=$((reclaimed_layer_record_count + 1)) ;;
          reclaimed_mount_count) reclaimed_mount_count=$((reclaimed_mount_count + 1)) ;;
          reclaimed_mount_record_count) reclaimed_mount_record_count=$((reclaimed_mount_record_count + 1)) ;;
          reclaimed_overlay_dir_count) reclaimed_overlay_dir_count=$((reclaimed_overlay_dir_count + 1)) ;;
          reclaimed_overlay_link_count) reclaimed_overlay_link_count=$((reclaimed_overlay_link_count + 1)) ;;
        esac
        printf '{"active_container_count":%s,"active_image_count":%s,"active_layer_count":%s,"active_mount_count":%s,"docker_root":"/var/lib/docker","mounted_overlay_count":%s,"reclaimed_layer_record_count":%s,"reclaimed_mount_count":%s,"reclaimed_mount_record_count":%s,"reclaimed_overlay_dir_count":%s,"reclaimed_overlay_link_count":%s,"schema_version":"leadpoet.docker_stale_mount_reclaim.v3","status":"ready"}\n' \
          "$active_container_count" "$active_image_count" "$active_layer_count" "$active_mount_count" "$mounted_overlay_count" "$reclaimed_layer_record_count" "$reclaimed_mount_count" "$reclaimed_mount_record_count" "$reclaimed_overlay_dir_count" "$reclaimed_overlay_link_count"
        ;;
      *) exit 2 ;;
    esac
    ;;
  test)
    if [ "${1:-}" != "-d" ]; then
      exit 2
    fi
    case "${2:-}" in
      */image/overlay2/layerdb/sha256)
        if [ -f "$FAKE_DAEMON_RECONCILE_MARKER" ] \
            && [ "$FAKE_ZERO_RUNTIME_RECONCILE_CLEARS_ORPHAN_METADATA" = "1" ]; then
          exit 1
        fi
        [ -f "$FAKE_DAEMON_RECONCILE_MARKER" ] \
          || [ "$FAKE_LAYERDB_IMAGE_DIRECTORY_EXISTS" = "1" ]
        ;;
      */image/overlay2/layerdb/mounts)
        if [ -f "$FAKE_DAEMON_RECONCILE_MARKER" ] \
            && [ "$FAKE_ZERO_RUNTIME_RECONCILE_CLEARS_ORPHAN_METADATA" = "1" ]; then
          exit 1
        fi
        [ -f "$FAKE_DAEMON_RECONCILE_MARKER" ] \
          || [ "$FAKE_LAYERDB_MOUNT_DIRECTORY_EXISTS" = "1" ]
        ;;
      */overlay2)
        if [ -f "$FAKE_DAEMON_RECONCILE_MARKER" ] \
            && [ "$FAKE_ZERO_RUNTIME_RECONCILE_CLEARS_ORPHAN_METADATA" = "1" ]; then
          exit 1
        fi
        [ -f "$FAKE_DAEMON_RECONCILE_MARKER" ] \
          || [ "$FAKE_OVERLAY_DIRECTORY_EXISTS" = "1" ]
        ;;
      *)
        exit 2
        ;;
    esac
    ;;
  du)
    printf '%s\\t/var/lib/docker\\n' "$FAKE_DOCKER_ROOT_BYTES"
    ;;
  find)
    case "$1" in
      */layerdb/sha256)
        if [ -f "$FAKE_DAEMON_RECONCILE_MARKER" ] \
            && [ "$FAKE_ZERO_RUNTIME_RECONCILE_CLEARS_ORPHAN_METADATA" = "1" ]; then
          emit_rows 0
        else
          emit_rows "$FAKE_LAYERDB_IMAGES"
        fi
        ;;
      */layerdb/mounts)
        if [ -f "$FAKE_DAEMON_RECONCILE_MARKER" ] \
            && [ "$FAKE_ZERO_RUNTIME_RECONCILE_CLEARS_ORPHAN_METADATA" = "1" ]; then
          emit_rows 0
        else
          emit_rows "$FAKE_LAYERDB_MOUNTS"
        fi
        ;;
      */overlay2)
        if [ -f "$FAKE_DAEMON_RECONCILE_MARKER" ] \
            && [ "$FAKE_ZERO_RUNTIME_RECONCILE_CLEARS_ORPHAN_METADATA" = "1" ]; then
          emit_rows 0
        else
          emit_rows "$FAKE_OVERLAY_DIRECTORIES"
        fi
        ;;
      *) exit 2 ;;
    esac
    ;;
  ctr)
    if [ "$FAKE_HANG_CTR_PROBE" = "1" ]; then
      exec /bin/sleep 60
    fi
    if [ -f "$FAKE_DAEMONS_STOPPED_MARKER" ]; then
      exit 1
    fi
    if [ -f "$FAKE_CONTAINERD_RESET_MARKER" ]; then
      exit 0
    fi
    if [ "${1:-}" = "namespaces" ]; then
      emit_rows "$FAKE_NON_MOBY_NAMESPACES"
      exit 0
    fi
    case "${3:-}:${4:-}" in
      containers:list)
        container_count="$FAKE_CONTAINERD_CONTAINERS"
        if [ -f "$FAKE_STALE_RECLAIM_MARKER" ] \
            && [ "$FAKE_RUNTIME_APPEARS_AFTER_STALE_RECLAIM" = "1" ]; then
          container_count=1
        fi
        emit_rows "$container_count"
        ;;
      tasks:list)
        if [ "${5:-}" = "-q" ]; then
          task_count="$FAKE_CONTAINERD_TASKS"
          if [ -f "$FAKE_STALE_RECLAIM_MARKER" ] \
              && [ "$FAKE_RUNTIME_APPEARS_AFTER_STALE_RECLAIM" = "1" ]; then
            task_count=1
          fi
          emit_rows "$task_count"
        else
          probe=0
          if [ -f "$FAKE_RUNNING_TASK_PROBE_STATE" ]; then
            probe="$(cat "$FAKE_RUNNING_TASK_PROBE_STATE")"
          fi
          probe=$((probe + 1))
          printf '%s\\n' "$probe" > "$FAKE_RUNNING_TASK_PROBE_STATE"
          if [ "$probe" -le "$FAKE_TASK_PROBE_FAILURES" ]; then
            exit 1
          fi
          printf 'TASK PID STATUS\\n'
          index=0
          running="$FAKE_CONTAINERD_RUNNING_TASKS"
          if [ "$FAKE_RUNNING_TASKS_CLEAR_AFTER" -ge 0 ] \
              && [ "$probe" -gt "$FAKE_RUNNING_TASKS_CLEAR_AFTER" ]; then
            running=0
          fi
          while [ "$index" -lt "$running" ]; do
            printf 'task-%s 100 RUNNING\\n' "$index"
            index=$((index + 1))
          done
        fi
        ;;
      *) exit 2 ;;
    esac
    ;;
  umount)
    printf '%s %s\\n' "$command" "$*" >> "$FAKE_SUDO_LOG"
    if [ "$FAKE_FAIL_UMOUNT" = "1" ]; then
      exit 1
    fi
    touch "$FAKE_STALE_RECLAIM_MARKER"
    ;;
  rm)
    printf '%s %s\\n' "$command" "$*" >> "$FAKE_SUDO_LOG"
    if [ "$*" = "-rf --one-file-system /var/lib/containerd" ]; then
      touch "$FAKE_CONTAINERD_RESET_MARKER"
    fi
    ;;
  systemctl)
    printf '%s %s\\n' "$command" "$*" >> "$FAKE_SUDO_LOG"
    case "${1:-}" in
      stop)
        touch "$FAKE_DAEMONS_STOPPED_MARKER"
        if [ "$FAKE_HANG_SYSTEMCTL_STOP" = "1" ]; then
          exec /bin/sleep 60
        fi
        if [ "$FAKE_FAIL_SYSTEMCTL_STOP" = "1" ]; then
          exit 1
        fi
        ;;
      start)
        if [ "$FAKE_HANG_SYSTEMCTL_START" = "1" ]; then
          exec /bin/sleep 60
        fi
        if [ "$FAKE_FAIL_DAEMON_RECOVERY" != "1" ]; then
          rm -f "$FAKE_DAEMONS_STOPPED_MARKER"
        fi
        ;;
    esac
    ;;
  install|pkill)
    printf '%s %s\\n' "$command" "$*" >> "$FAKE_SUDO_LOG"
    ;;
  *)
    exit 2
    ;;
esac
""",
    )
    _write_executable(
        bin_dir / "pgrep",
        """#!/bin/bash
if [ "$FAKE_HANG_SHIM_PROBE" = "1" ]; then
  exec /bin/sleep 60
fi
if [ "$FAKE_FAIL_SHIM_PROBE" = "1" ]; then
  exit 2
fi
if [ "$FAKE_MALFORMED_SHIM_PROBE" = "1" ]; then
  printf 'not-a-number\\n'
  exit 0
fi
if [ "$FAKE_HOST_GATEWAY_LIVE_AFTER_INVENTORY" = "1" ] \
    && [ ! -f "$FAKE_LATE_HOST_GATEWAY_MARKER" ]; then
  mkdir -p "$LEADPOET_PROC_ROOT/201"
  printf 'Name:\tpython3\nPPid:\t1\n' \
    > "$LEADPOET_PROC_ROOT/201/status"
  printf '/home/ec2-user/venv311/bin/python3\\0-u\\0-m\\0gateway.main\\0' \
    > "$LEADPOET_PROC_ROOT/201/cmdline"
  printf '201 (python3) S 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 67890\n' \
    > "$LEADPOET_PROC_ROOT/201/stat"
  touch "$FAKE_LATE_HOST_GATEWAY_MARKER"
fi
shim_count="$FAKE_MOBY_SHIMS"
if [ -f "$FAKE_CONTAINERD_RESET_MARKER" ]; then
  shim_count=0
elif [ -f "$FAKE_STALE_RECLAIM_MARKER" ] \
    && [ "$FAKE_RUNTIME_APPEARS_AFTER_STALE_RECLAIM" = "1" ]; then
  shim_count=1
fi
printf '%s\\n' "$shim_count"
if [ "$shim_count" -eq 0 ]; then
  if [ "$FAKE_ZERO_SHIM_PROBE_SUCCEEDS" = "1" ]; then
    exit 0
  fi
  exit 1
fi
if [ "$FAKE_NONZERO_SHIM_PROBE_REPORTS_NO_MATCH" = "1" ]; then
  exit 1
fi
exit 0
""",
    )

    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "VALIDATOR_DOCKER_ALLOW_DATA_ROOT_RESET": "1",
        "VALIDATOR_DOCKER_ALLOW_LIVE_HOST_GATEWAY_PRUNE": (
            "1" if allow_live_host_gateway_prune else "0"
        ),
        "VALIDATOR_DOCKER_MIN_FREE_BYTES": "30000000000",
        "VALIDATOR_DOCKER_LIVE_RUNTIME_MIN_FREE_BYTES": str(
            live_runtime_min_free_bytes
        ),
        "LEADPOET_DOCKER_OPERATION_LOCK_FILE": str(
            tmp_path / "docker-operation.lock"
        ),
        "LEADPOET_PROC_ROOT": str(proc_root),
        "LEADPOET_DOCKER_ALLOW_NONSTANDARD_PROC_ROOT": "1",
        "FAKE_AVAILABLE": str(available),
        "FAKE_AVAILABLE_AFTER_STALE_RECLAIM": str(
            available
            if available_after_stale_reclaim is None
            else available_after_stale_reclaim
        ),
        "FAKE_AVAILABLE_AFTER_BUILDER_PRUNE": str(
            available
            if available_after_builder_prune is None
            else available_after_builder_prune
        ),
        "FAKE_LIVE_RESTORE_ENABLED": (
            "true" if live_restore_enabled else "false"
        ),
        "FAKE_CONTAINERS": str(containers),
        "FAKE_IMAGES": str(images),
        "FAKE_IMAGES_AFTER_BUILDER_PRUNE": str(
            -1
            if images_after_builder_prune is None
            else images_after_builder_prune
        ),
        "FAKE_VOLUMES": str(volumes),
        "FAKE_LAYERDB_IMAGES": str(layerdb_images),
        "FAKE_LAYERDB_MOUNTS": str(layerdb_mounts),
        "FAKE_OVERLAY_DIRECTORIES": str(overlay_directories),
        "FAKE_LAYERDB_IMAGE_DIRECTORY_EXISTS": (
            "1" if layerdb_image_directory_exists else "0"
        ),
        "FAKE_LAYERDB_MOUNT_DIRECTORY_EXISTS": (
            "1" if layerdb_mount_directory_exists else "0"
        ),
        "FAKE_OVERLAY_DIRECTORY_EXISTS": (
            "1" if overlay_directory_exists else "0"
        ),
        "FAKE_CONTAINERD_CONTAINERS": str(containerd_containers),
        "FAKE_CONTAINERD_TASKS": str(containerd_tasks),
        "FAKE_CONTAINERD_RUNNING_TASKS": str(containerd_running_tasks),
        "FAKE_RUNNING_TASKS_CLEAR_AFTER": str(running_tasks_clear_after),
        "FAKE_RUNNING_TASK_PROBE_STATE": str(tmp_path / "running-task-probes"),
        "FAKE_TASK_PROBE_FAILURES": str(task_probe_failures),
        "FAKE_SYSTEM_PRUNE_FAILURES": str(system_prune_failures),
        "FAKE_SYSTEM_PRUNE_STATE": str(tmp_path / "system-prune-attempts"),
        "FAKE_NON_MOBY_NAMESPACES": str(non_moby_namespaces),
        "FAKE_MOBY_SHIMS": str(moby_shims),
        "FAKE_FAIL_SHIM_PROBE": "1" if fail_shim_probe else "0",
        "FAKE_MALFORMED_SHIM_PROBE": "1" if malformed_shim_probe else "0",
        "FAKE_HANG_SHIM_PROBE": "1" if hang_shim_probe else "0",
        "FAKE_ZERO_SHIM_PROBE_SUCCEEDS": (
            "1" if zero_shim_probe_succeeds else "0"
        ),
        "FAKE_NONZERO_SHIM_PROBE_REPORTS_NO_MATCH": (
            "1" if nonzero_shim_probe_reports_no_match else "0"
        ),
        "FAKE_STALE_OVERLAY_MOUNTS": str(stale_overlay_mounts),
        "FAKE_STALE_LAYER_RECORDS": str(stale_layer_records),
        "FAKE_STALE_MOUNT_RECORDS": str(stale_mount_records),
        "FAKE_STALE_OVERLAY_DIRECTORIES": str(stale_overlay_directories),
        "FAKE_STALE_OVERLAY_LINKS": str(stale_overlay_links),
        "FAKE_STALE_RECLAIM_MARKER": str(tmp_path / "stale-mounts-reclaimed"),
        "FAKE_BUILDER_PRUNE_MARKER": str(tmp_path / "builder-pruned"),
        "FAKE_BUILDER_PRUNE_STATE": str(tmp_path / "builder-prune-count"),
        "FAKE_IMAGE_PRUNE_MARKER": str(tmp_path / "image-pruned"),
        "FAKE_CONTAINERD_RESET_MARKER": str(tmp_path / "containerd-reset"),
        "FAKE_DAEMONS_STOPPED_MARKER": str(tmp_path / "daemons-stopped"),
        "FAKE_DAEMON_READY_PROBE_STATE": str(
            tmp_path / "daemon-ready-probes"
        ),
        "FAKE_FAIL_UMOUNT": "1" if fail_umount else "0",
        "FAKE_FAIL_STALE_RECLAIMER": "1" if fail_stale_reclaimer else "0",
        "FAKE_MALFORMED_STALE_RECLAIMER": (
            "1" if malformed_stale_reclaimer else "0"
        ),
        "FAKE_PRE_RECLAIM_AUDIT_BOOL_FIELD": (
            "" if pre_reclaim_audit_bool_field is None else pre_reclaim_audit_bool_field
        ),
        "FAKE_RECLAIM_COUNT_MISMATCH_FIELD": (
            "" if reclaim_count_mismatch_field is None else reclaim_count_mismatch_field
        ),
        "FAKE_POST_RECLAIM_STALE_FIELD": (
            "" if post_reclaim_stale_field is None else post_reclaim_stale_field
        ),
        "FAKE_POST_RECLAIM_ACTIVE_COUNT_FIELD": (
            ""
            if post_reclaim_active_count_field is None
            else post_reclaim_active_count_field
        ),
        "FAKE_CHANGE_AUDIT_MANIFEST_AFTER_STALE_RECLAIM": (
            "1" if change_audit_manifest_after_stale_reclaim else "0"
        ),
        "FAKE_FAIL_ZERO_RUNTIME_RECONCILER": (
            "1" if fail_zero_runtime_reconciler else "0"
        ),
        "FAKE_MALFORMED_ZERO_RUNTIME_RECONCILER": (
            "1" if malformed_zero_runtime_reconciler else "0"
        ),
        "FAKE_ZERO_RUNTIME_BOOLEAN_FIELD": (
            "" if zero_runtime_boolean_field is None else zero_runtime_boolean_field
        ),
        "FAKE_ZERO_RUNTIME_RECONCILE_CLEARS_ORPHAN_METADATA": (
            "1" if zero_runtime_reconcile_clears_orphan_metadata else "0"
        ),
        "FAKE_CHANGE_AUDIT_MANIFEST_AFTER_RECONCILE": (
            "1" if change_audit_manifest_after_reconcile else "0"
        ),
        "FAKE_CHANGE_AUDIT_MANIFEST_AFTER_POST_PRUNE": (
            "1" if change_audit_manifest_after_post_prune else "0"
        ),
        "FAKE_POST_RECONCILE_ACTIVE_COUNT_FIELD": (
            ""
            if post_reconcile_active_count_field is None
            else post_reconcile_active_count_field
        ),
        "FAKE_POST_PRUNE_ACTIVE_COUNT_FIELD": (
            "" if post_prune_active_count_field is None else post_prune_active_count_field
        ),
        "FAKE_STALE_AUDIT_AFTER_RECONCILE": (
            "1" if stale_audit_after_reconcile else "0"
        ),
        "FAKE_STALE_AUDIT_BEFORE_RECONCILE": (
            "1" if stale_audit_before_reconcile else "0"
        ),
        "FAKE_DAEMON_RECONCILE_MARKER": str(
            tmp_path / "daemon-reconciled"
        ),
        "FAKE_FAIL_SYSTEMCTL_STOP": "1" if fail_systemctl_stop else "0",
        "FAKE_FAIL_DAEMON_RECOVERY": "1" if fail_daemon_recovery else "0",
        "FAKE_HANG_DAEMON_PROBE": "1" if hang_daemon_probe else "0",
        "FAKE_HANG_CTR_PROBE": "1" if hang_ctr_probe else "0",
        "FAKE_HANG_SYSTEMCTL_START": "1" if hang_systemctl_start else "0",
        "FAKE_HANG_SYSTEMCTL_STOP": "1" if hang_systemctl_stop else "0",
        "FAKE_HOST_GATEWAY_LIVE_AFTER_INVENTORY": (
            "1" if host_gateway_live_after_inventory else "0"
        ),
        "FAKE_HOST_GATEWAY_EXITS_DURING_LIVE_PRUNE": (
            "1" if host_gateway_exits_during_live_prune else "0"
        ),
        "FAKE_HOST_GATEWAY_START_TIME_CHANGES_DURING_LIVE_PRUNE": (
            "1" if host_gateway_start_time_changes_during_live_prune else "0"
        ),
        "FAKE_HOST_GATEWAY_EXITS_DURING_DAEMON_RECONCILE": (
            "1" if host_gateway_exits_during_daemon_reconcile else "0"
        ),
        "FAKE_HOST_GATEWAY_APPEARS_DURING_DAEMON_RECONCILE": (
            "1" if host_gateway_appears_during_daemon_reconcile else "0"
        ),
        "FAKE_RUNTIME_APPEARS_AFTER_STALE_RECLAIM": (
            "1" if runtime_appears_after_stale_reclaim else "0"
        ),
        "FAKE_DOCKER_ROOT": docker_root,
        "FAKE_DOCKER_ROOT_AFTER_BUILDER_PRUNE": (
            docker_root
            if docker_root_after_builder_prune is None
            else docker_root_after_builder_prune
        ),
        "FAKE_LATE_HOST_GATEWAY_MARKER": str(
            tmp_path / "late-host-gateway"
        ),
        "FAKE_DOCKER_ROOT_BYTES": "229720371200",
        "FAKE_SUDO_LOG": str(sudo_log),
        "FAKE_OPERATION_LOG": str(tmp_path / "operations.log"),
        "VALIDATOR_DOCKER_PRUNE_ATTEMPTS": str(
            system_prune_failures + 1
            if prune_attempts is None
            else prune_attempts
        ),
        "VALIDATOR_DOCKER_DAEMON_READY_ATTEMPTS": str(
            daemon_ready_attempts
        ),
        "VALIDATOR_DOCKER_DAEMON_PROBE_TIMEOUT_SECONDS": str(
            daemon_probe_timeout_seconds
        ),
        "VALIDATOR_DOCKER_DAEMON_CONTROL_TIMEOUT_SECONDS": str(
            daemon_probe_timeout_seconds
        ),
        "VALIDATOR_DOCKER_SETTLE_ATTEMPTS": str(
            max(
                1,
                task_probe_failures + 1,
                3 if running_tasks_clear_after >= 0 else 1,
            )
        ),
    }
    env.pop("REQUIRE_ZERO_RUNTIME_RECONCILE", None)
    if require_zero_runtime_reconcile is not None:
        env["REQUIRE_ZERO_RUNTIME_RECONCILE"] = require_zero_runtime_reconcile
    result = subprocess.run(
        ["bash", str(ROOT / "validator_tee/scripts/reclaim_docker_storage_v2.sh")],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    return result, sudo_log.read_text(encoding="utf-8") if sudo_log.exists() else ""


def test_validator_docker_recovery_is_guarded_and_runs_after_shutdown():
    recovery = (
        ROOT / "validator_tee" / "scripts" / "reclaim_docker_storage_v2.sh"
    ).read_text(encoding="utf-8")
    restart = (ROOT / "validator_restart.sh").read_text(encoding="utf-8")
    workflow = (
        ROOT / ".github" / "workflows" / "attested-v2-release.yml"
    ).read_text(encoding="utf-8")

    assert (
        'CONTAINER_COUNT="$(run_bounded_daemon_inventory docker ps -aq'
        in recovery
    )
    assert 'if [ "$CONTAINER_COUNT" -ne 0 ]' in recovery
    assert 'if [ "$IMAGE_COUNT" -ne 0 ]' in recovery
    assert 'if [ "$VOLUME_COUNT" -ne 0 ]' in recovery
    assert 'if [ "$DOCKER_ROOT" != "/var/lib/docker" ]' in recovery
    assert "ORPHANED_DOCKER_STATE=1" in recovery
    assert 'OVERLAY_DIRECTORY_COUNT="$(\n' in recovery
    assert "systemctl stop docker.service docker.socket containerd.service" in recovery
    assert 'rm -rf --one-file-system "$DOCKER_ROOT"' in recovery
    assert 'rm -rf --one-file-system "$CONTAINERD_ROOT"' in recovery
    assert "docker system prune --all --force --volumes" not in recovery

    remove_containers = restart.index("| xargs -r docker rm")
    journal_cleanup = restart.index(
        'sudo journalctl \\\n          --vacuum-size="$VALIDATOR_JOURNAL_VACUUM_SIZE"'
    )
    pip_cleanup = restart.index('rm -rf -- "$HOME/.cache/pip"')
    reclaim = restart.index("reclaim_docker_storage_v2.sh")
    build = restart.index("bash validator_tee/scripts/build_enclave.sh")
    assert remove_containers < journal_cleanup < pip_cleanup < reclaim < build
    cleanup = restart[journal_cleanup:reclaim]
    for protected_path in (
        ".bittensor",
        ".config/leadpoet",
        "leadpoet-legacy",
        "validator_weights",
        "validator-releases-v2",
        "actions-runner",
        "leadpoet-v2-artifacts",
        "drand-cabi-v2",
    ):
        assert protected_path not in cleanup
    assert "VALIDATOR_DOCKER_ALLOW_DATA_ROOT_RESET=1" in restart
    assert "VALIDATOR_DOCKER_ALLOW_DATA_ROOT_RESET=1" in workflow

    subprocess.run(
        [
            "bash",
            "-n",
            str(
                ROOT
                / "validator_tee"
                / "scripts"
                / "reclaim_docker_storage_v2.sh"
            ),
        ],
        check=True,
    )


def test_nonstandard_proc_root_requires_explicit_rehearsal_flag(
    tmp_path: Path,
) -> None:
    result = subprocess.run(
        ["bash", str(ROOT / "validator_tee/scripts/reclaim_docker_storage_v2.sh")],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "LEADPOET_PROC_ROOT": str(tmp_path / "empty-proc"),
            "LEADPOET_DOCKER_OPERATION_LOCK_FILE": str(tmp_path / "lock"),
        },
    )

    assert result.returncode == 2
    assert "only in explicit rehearsal" in result.stderr


def test_validator_docker_recovery_resets_orphaned_empty_root_above_floor(
    tmp_path: Path,
) -> None:
    result, sudo_log = _run_recovery(
        tmp_path,
        available=40_000_000_000,
        layerdb_images=3_904,
        layerdb_mounts=322,
        overlay_directories=4_559,
    )

    assert result.returncode == 0, result.stderr
    assert "orphaned=1" in result.stdout
    assert "systemctl stop docker.service docker.socket containerd.service" in sudo_log
    assert "rm -rf --one-file-system /var/lib/docker" in sudo_log
    assert "rm -rf --one-file-system /var/lib/containerd" in sudo_log


def test_failed_reset_recovery_bounds_a_hanging_daemon_probe(
    tmp_path: Path,
) -> None:
    started = time.monotonic()
    result, sudo_log = _run_recovery(
        tmp_path,
        available=40_000_000_000,
        layerdb_images=1,
        stale_overlay_mounts=1,
        fail_umount=True,
        fail_daemon_recovery=True,
        hang_daemon_probe=True,
        daemon_ready_attempts=1,
        daemon_probe_timeout_seconds=1,
    )

    assert result.returncode != 0
    assert time.monotonic() - started < 4
    assert "Recovering Docker/containerd readiness" in result.stderr
    assert "systemctl start containerd.service docker.service" in sudo_log


def test_initial_readiness_bounds_a_hanging_ctr_probe(
    tmp_path: Path,
) -> None:
    started = time.monotonic()
    result, sudo_log = _run_recovery(
        tmp_path,
        available=40_000_000_000,
        hang_ctr_probe=True,
        daemon_ready_attempts=1,
        daemon_probe_timeout_seconds=1,
    )

    assert result.returncode != 0
    assert time.monotonic() - started < 4
    assert "did not recover before storage inventory" in result.stderr
    assert "systemctl start containerd.service docker.service" in sudo_log


def test_initial_recovery_bounds_a_hanging_systemctl_start(
    tmp_path: Path,
) -> None:
    started = time.monotonic()
    result, sudo_log = _run_recovery(
        tmp_path,
        available=40_000_000_000,
        hang_ctr_probe=True,
        hang_systemctl_start=True,
        daemon_ready_attempts=1,
        daemon_probe_timeout_seconds=1,
    )

    assert result.returncode != 0
    assert time.monotonic() - started < 4
    assert "systemctl start containerd.service docker.service" in sudo_log


def test_reset_bounds_hanging_systemctl_stop_and_recovers_daemons(
    tmp_path: Path,
) -> None:
    started = time.monotonic()
    result, sudo_log = _run_recovery(
        tmp_path,
        available=40_000_000_000,
        layerdb_images=1,
        hang_systemctl_stop=True,
        daemon_ready_attempts=1,
        daemon_probe_timeout_seconds=1,
    )

    assert result.returncode != 0
    assert time.monotonic() - started < 4
    assert "systemctl stop docker.service docker.socket containerd.service" in sudo_log
    assert "systemctl start containerd.service docker.service" in sudo_log
    assert "readiness recovered" in result.stderr


def test_validator_docker_recovery_resets_stale_containerd_state(
    tmp_path: Path,
) -> None:
    result, sudo_log = _run_recovery(
        tmp_path,
        available=40_000_000_000,
        containerd_containers=22,
        containerd_tasks=7,
        moby_shims=20,
    )

    assert result.returncode == 0, result.stderr
    assert "containerd_containers=22" in result.stdout
    assert "containerd_tasks=7" in result.stdout
    assert "moby_shims=20" in result.stdout
    assert "orphaned=1" in result.stdout
    assert "systemctl stop docker.service docker.socket containerd.service" in sudo_log


def test_validator_docker_recovery_defers_reset_for_exact_live_host_gateway(
    tmp_path: Path,
) -> None:
    result, sudo_log = _run_recovery(
        tmp_path,
        available=25_000_000_000,
        layerdb_images=1,
        host_gateway_live=True,
    )

    assert result.returncode == 0, result.stderr
    assert '"status": "live"' in result.stdout
    assert "runtime_mode=host-gateway-live" in result.stdout
    assert "storage maintenance deferred while the exact host gateway is live" in result.stdout
    assert not (tmp_path / "system-prune-attempts").exists()
    assert not (tmp_path / "builder-pruned").exists()
    assert "docker_stale_mount_reclaimer_v2.py" not in sudo_log
    assert "docker_zero_runtime_reconciler_v2.py" not in sudo_log
    assert "systemctl stop" not in sudo_log
    assert "rm -rf" not in sudo_log


def test_required_zero_runtime_reconcile_runs_with_ample_space(
    tmp_path: Path,
) -> None:
    result, sudo_log = _run_recovery(
        tmp_path,
        available=25_000_000_000,
        images=14,
        layerdb_images=65,
        overlay_directories=65,
        stale_overlay_links=2,
        host_gateway_live=True,
        allow_live_host_gateway_prune=True,
        require_zero_runtime_reconcile="1",
    )

    assert result.returncode == 0, result.stderr
    assert "storage maintenance deferred" not in result.stdout
    assert "phase=pre-stale-reclaim containers=0" in result.stdout
    assert "phase=pre-daemon-reconcile containers=0" in result.stdout
    assert "phase=post-daemon-reconcile containers=0" in result.stdout
    assert "phase=post-reconcile-builder-prune containers=0" in result.stdout
    assert "phase=post-reclaim containers=0" in result.stdout
    assert "Docker storage ready after bounded online reclaim" in result.stdout
    assert "docker_zero_runtime_reconciler_v2.py" in sudo_log
    assert (tmp_path / "stale-mounts-reclaimed").is_file()
    assert '"active_image_count":14' in result.stdout
    assert '"active_layer_count":65' in result.stdout
    assert '"active_overlay_dir_count":65' in result.stdout
    assert '"stale_overlay_link_count":2' in result.stdout
    assert '"reclaimed_overlay_link_count":2' in result.stdout
    assert (tmp_path / "builder-prune-count").read_text().strip() == "2"
    operations = (tmp_path / "operations.log").read_text().splitlines()
    relevant_operations = []
    for operation in operations:
        if operation == "docker builder prune --all --force":
            relevant_operations.append("builder_prune")
        elif "docker_zero_runtime_reconciler_v2.py" in operation:
            relevant_operations.append("dockerd_reconcile")
        elif "docker_stale_mount_reclaimer_v2.py" in operation:
            relevant_operations.append(
                "stale_audit" if operation.endswith(" --audit-only") else "stale_reclaim"
            )
    assert relevant_operations == [
        "builder_prune",
        "stale_audit",
        "stale_reclaim",
        "stale_audit",
        "dockerd_reconcile",
        "stale_audit",
        "builder_prune",
        "stale_audit",
    ]
    assert not (tmp_path / "image-pruned").exists()
    assert not (tmp_path / "system-prune-attempts").exists()
    assert "systemctl" not in sudo_log
    assert "rm " not in sudo_log
    assert "pkill" not in sudo_log


def test_required_live_reconcile_initializes_a_fully_empty_metadata_root(
    tmp_path: Path,
) -> None:
    result, sudo_log = _run_recovery(
        tmp_path,
        available=25_000_000_000,
        layerdb_image_directory_exists=False,
        layerdb_mount_directory_exists=False,
        overlay_directory_exists=False,
        host_gateway_live=True,
        allow_live_host_gateway_prune=True,
        require_zero_runtime_reconcile="1",
        change_audit_manifest_after_reconcile=True,
    )

    assert result.returncode == 0, result.stderr
    assert "phase=pre-stale-reclaim containers=0" in result.stdout
    assert "phase=post-daemon-reconcile containers=0" in result.stdout
    assert "Docker storage ready after bounded online reclaim" in result.stdout
    assert "docker_stale_mount_reclaimer_v2.py" in sudo_log
    assert "docker_zero_runtime_reconciler_v2.py" in sudo_log
    assert (tmp_path / "daemon-reconciled").is_file()
    assert (tmp_path / "builder-prune-count").read_text().strip() == "2"


def test_required_live_reconcile_rejects_partial_metadata_before_audit(
    tmp_path: Path,
) -> None:
    result, sudo_log = _run_recovery(
        tmp_path,
        available=25_000_000_000,
        layerdb_image_directory_exists=False,
        host_gateway_live=True,
        allow_live_host_gateway_prune=True,
        require_zero_runtime_reconcile="1",
    )

    assert result.returncode == 1
    assert "metadata layout is partial or inconsistent" in result.stderr
    assert "docker_stale_mount_reclaimer_v2.py" not in sudo_log
    assert "docker_zero_runtime_reconciler_v2.py" not in sudo_log


def test_required_live_reconcile_rejects_absent_metadata_with_images(
    tmp_path: Path,
) -> None:
    result, sudo_log = _run_recovery(
        tmp_path,
        available=25_000_000_000,
        images=1,
        layerdb_image_directory_exists=False,
        layerdb_mount_directory_exists=False,
        overlay_directory_exists=False,
        host_gateway_live=True,
        allow_live_host_gateway_prune=True,
        require_zero_runtime_reconcile="1",
    )

    assert result.returncode == 1
    assert "metadata layout is partial or inconsistent" in result.stderr
    assert "docker_stale_mount_reclaimer_v2.py" not in sudo_log
    assert "docker_zero_runtime_reconciler_v2.py" not in sudo_log


@pytest.mark.parametrize(
    "stale_state",
    [
        pytest.param({}, id="all-zero"),
        pytest.param({"stale_layer_records": 1}, id="layer-record"),
        pytest.param({"stale_overlay_mounts": 1}, id="mounted-overlay"),
        pytest.param({"stale_mount_records": 1}, id="mount-record"),
        pytest.param({"stale_overlay_directories": 1}, id="overlay-directory"),
        pytest.param({"stale_overlay_links": 1}, id="overlay-link"),
    ],
)
def test_required_zero_runtime_reconcile_applies_every_stale_dimension_and_zero(
    tmp_path: Path,
    stale_state: dict[str, int],
) -> None:
    result, sudo_log = _run_recovery(
        tmp_path,
        available=25_000_000_000,
        images=2,
        layerdb_images=3,
        host_gateway_live=True,
        allow_live_host_gateway_prune=True,
        require_zero_runtime_reconcile="1",
        **stale_state,
    )

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "stale-mounts-reclaimed").is_file()
    assert (tmp_path / "daemon-reconciled").is_file()
    assert (tmp_path / "builder-prune-count").read_text().strip() == "2"
    operations = (tmp_path / "operations.log").read_text().splitlines()
    sequence = []
    for operation in operations:
        if operation == "docker builder prune --all --force":
            sequence.append("builder_prune")
        elif "docker_zero_runtime_reconciler_v2.py" in operation:
            sequence.append("dockerd_reconcile")
        elif "docker_stale_mount_reclaimer_v2.py" in operation:
            sequence.append(
                "stale_audit" if operation.endswith(" --audit-only") else "stale_reclaim"
            )
    assert sequence == [
        "builder_prune",
        "stale_audit",
        "stale_reclaim",
        "stale_audit",
        "dockerd_reconcile",
        "stale_audit",
        "builder_prune",
        "stale_audit",
    ]
    assert "docker image prune" not in "\n".join(operations)
    assert "docker system prune" not in "\n".join(operations)
    assert "systemctl" not in sudo_log


@pytest.mark.parametrize(
    "field",
    [
        "active_container_count",
        "active_image_count",
        "active_layer_count",
        "active_mount_count",
        "mounted_overlay_count",
        "reclaimed_layer_record_count",
        "reclaimed_mount_count",
        "reclaimed_mount_record_count",
        "reclaimed_overlay_dir_count",
        "reclaimed_overlay_link_count",
    ],
)
def test_required_zero_runtime_reconcile_rejects_reclaim_count_mismatch(
    tmp_path: Path,
    field: str,
) -> None:
    result, sudo_log = _run_recovery(
        tmp_path,
        available=25_000_000_000,
        images=2,
        layerdb_images=3,
        stale_layer_records=1,
        stale_overlay_mounts=1,
        stale_mount_records=1,
        stale_overlay_directories=1,
        stale_overlay_links=1,
        host_gateway_live=True,
        allow_live_host_gateway_prune=True,
        require_zero_runtime_reconcile="1",
        reclaim_count_mismatch_field=field,
    )

    assert result.returncode == 1
    assert "reclaim returned malformed evidence" in result.stderr
    assert "docker_zero_runtime_reconciler_v2.py" not in sudo_log
    assert not (tmp_path / "daemon-reconciled").exists()
    assert (tmp_path / "builder-prune-count").read_text().strip() == "1"


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        ("fail_stale_reclaimer", "validated stale Docker overlay reclaim failed"),
        ("malformed_stale_reclaimer", "reclaim returned malformed evidence"),
    ],
)
def test_required_zero_runtime_reconcile_stale_apply_failure_is_terminal(
    tmp_path: Path,
    failure: str,
    message: str,
) -> None:
    result, sudo_log = _run_recovery(
        tmp_path,
        available=25_000_000_000,
        host_gateway_live=True,
        allow_live_host_gateway_prune=True,
        require_zero_runtime_reconcile="1",
        **{failure: True},
    )

    assert result.returncode == 1
    assert message in result.stderr
    assert "docker_zero_runtime_reconciler_v2.py" not in sudo_log
    assert not (tmp_path / "daemon-reconciled").exists()
    assert (tmp_path / "builder-prune-count").read_text().strip() == "1"


@pytest.mark.parametrize(
    "field",
    [
        "stale_layer_record_count",
        "mounted_overlay_count",
        "stale_mount_record_count",
        "stale_overlay_dir_count",
        "stale_overlay_link_count",
    ],
)
def test_required_zero_runtime_reconcile_rejects_residual_stale_state(
    tmp_path: Path,
    field: str,
) -> None:
    result, sudo_log = _run_recovery(
        tmp_path,
        available=25_000_000_000,
        host_gateway_live=True,
        allow_live_host_gateway_prune=True,
        require_zero_runtime_reconcile="1",
        post_reclaim_stale_field=field,
    )

    assert result.returncode == 1
    assert "audit was invalid before daemon reconciliation" in result.stderr
    assert "docker_zero_runtime_reconciler_v2.py" not in sudo_log
    assert not (tmp_path / "daemon-reconciled").exists()
    assert (tmp_path / "builder-prune-count").read_text().strip() == "1"


@pytest.mark.parametrize(
    "field",
    [
        "active_container_count",
        "active_image_count",
        "active_layer_count",
        "active_mount_count",
        "active_overlay_dir_count",
    ],
)
def test_required_zero_runtime_reconcile_rejects_post_reclaim_active_change(
    tmp_path: Path,
    field: str,
) -> None:
    result, sudo_log = _run_recovery(
        tmp_path,
        available=25_000_000_000,
        images=2,
        layerdb_images=3,
        host_gateway_live=True,
        allow_live_host_gateway_prune=True,
        require_zero_runtime_reconcile="1",
        post_reclaim_active_count_field=field,
    )

    assert result.returncode == 1
    assert "docker_zero_runtime_reconciler_v2.py" not in sudo_log
    assert not (tmp_path / "daemon-reconciled").exists()
    assert (tmp_path / "builder-prune-count").read_text().strip() == "1"


@pytest.mark.parametrize(
    "field",
    [
        "active_container_count",
        "active_image_count",
        "active_layer_count",
        "active_mount_count",
        "active_overlay_dir_count",
        "mounted_overlay_count",
        "stale_layer_record_count",
        "stale_mount_record_count",
        "stale_overlay_dir_count",
        "stale_overlay_link_count",
    ],
)
def test_required_zero_runtime_reconcile_rejects_boolean_pre_audit_counts(
    tmp_path: Path,
    field: str,
) -> None:
    result, sudo_log = _run_recovery(
        tmp_path,
        available=25_000_000_000,
        host_gateway_live=True,
        allow_live_host_gateway_prune=True,
        require_zero_runtime_reconcile="1",
        pre_reclaim_audit_bool_field=field,
    )

    assert result.returncode == 1
    assert "audit was invalid before guarded stale reclaim" in result.stderr
    assert not (tmp_path / "stale-mounts-reclaimed").exists()
    assert "docker_zero_runtime_reconciler_v2.py" not in sudo_log
    assert (tmp_path / "builder-prune-count").read_text().strip() == "1"


def test_required_zero_runtime_reconcile_rejects_manifest_change_during_reclaim(
    tmp_path: Path,
) -> None:
    result, sudo_log = _run_recovery(
        tmp_path,
        available=25_000_000_000,
        images=2,
        layerdb_images=3,
        host_gateway_live=True,
        allow_live_host_gateway_prune=True,
        require_zero_runtime_reconcile="1",
        change_audit_manifest_after_stale_reclaim=True,
    )

    assert result.returncode == 1
    assert "identity changed during guarded stale reclaim" in result.stderr
    assert "docker_zero_runtime_reconciler_v2.py" not in sudo_log
    assert not (tmp_path / "daemon-reconciled").exists()
    assert (tmp_path / "builder-prune-count").read_text().strip() == "1"


@pytest.mark.parametrize(
    "field",
    [
        "container_count",
        "containerd_container_count",
        "containerd_task_count",
        "image_count",
        "moby_shim_count",
        "root_device",
        "root_inode",
    ],
)
def test_required_zero_runtime_reconcile_rejects_boolean_helper_counts(
    tmp_path: Path,
    field: str,
) -> None:
    result, sudo_log = _run_recovery(
        tmp_path,
        available=25_000_000_000,
        images=2,
        layerdb_images=3,
        host_gateway_live=True,
        allow_live_host_gateway_prune=True,
        require_zero_runtime_reconcile="1",
        zero_runtime_boolean_field=field,
    )

    assert result.returncode == 1
    assert "reconciliation returned malformed evidence" in result.stderr
    assert "docker_zero_runtime_reconciler_v2.py" in sudo_log
    assert (tmp_path / "daemon-reconciled").is_file()
    assert (tmp_path / "builder-prune-count").read_text().strip() == "1"


@pytest.mark.parametrize(
    "field",
    [
        "active_container_count",
        "active_image_count",
        "active_layer_count",
        "active_mount_count",
        "active_overlay_dir_count",
    ],
)
def test_required_zero_runtime_reconcile_rejects_post_helper_active_change(
    tmp_path: Path,
    field: str,
) -> None:
    result, sudo_log = _run_recovery(
        tmp_path,
        available=25_000_000_000,
        images=2,
        layerdb_images=3,
        host_gateway_live=True,
        allow_live_host_gateway_prune=True,
        require_zero_runtime_reconcile="1",
        post_reconcile_active_count_field=field,
    )

    assert result.returncode == 1
    assert "docker_zero_runtime_reconciler_v2.py" in sudo_log
    assert (tmp_path / "daemon-reconciled").is_file()
    assert (tmp_path / "builder-prune-count").read_text().strip() == "1"


@pytest.mark.parametrize(
    "field",
    [
        "active_container_count",
        "active_image_count",
        "active_layer_count",
        "active_mount_count",
        "active_overlay_dir_count",
    ],
)
def test_required_zero_runtime_reconcile_rejects_post_prune_active_change(
    tmp_path: Path,
    field: str,
) -> None:
    result, sudo_log = _run_recovery(
        tmp_path,
        available=25_000_000_000,
        images=2,
        layerdb_images=3,
        host_gateway_live=True,
        allow_live_host_gateway_prune=True,
        require_zero_runtime_reconcile="1",
        post_prune_active_count_field=field,
    )

    assert result.returncode == 1
    assert "docker_zero_runtime_reconciler_v2.py" in sudo_log
    assert (tmp_path / "daemon-reconciled").is_file()
    assert (tmp_path / "builder-prune-count").read_text().strip() == "2"


@pytest.mark.parametrize("value", ["", "yes", "2", "-1"])
def test_required_zero_runtime_reconcile_flag_is_strict(
    tmp_path: Path,
    value: str,
) -> None:
    result, sudo_log = _run_recovery(
        tmp_path,
        available=25_000_000_000,
        host_gateway_live=True,
        allow_live_host_gateway_prune=True,
        require_zero_runtime_reconcile=value,
    )

    assert result.returncode == 2
    assert "REQUIRE_ZERO_RUNTIME_RECONCILE must be 0 or 1" in result.stderr
    assert sudo_log == ""
    assert not (tmp_path / "builder-pruned").exists()


def test_required_zero_runtime_reconcile_requires_live_prune_admission(
    tmp_path: Path,
) -> None:
    result, sudo_log = _run_recovery(
        tmp_path,
        available=25_000_000_000,
        host_gateway_live=True,
        require_zero_runtime_reconcile="1",
    )

    assert result.returncode == 2
    assert "requires VALIDATOR_DOCKER_ALLOW_LIVE_HOST_GATEWAY_PRUNE=1" in result.stderr
    assert sudo_log == ""
    assert not (tmp_path / "builder-pruned").exists()


def test_required_zero_runtime_reconcile_accepts_stably_absent_gateway(
    tmp_path: Path,
) -> None:
    result, sudo_log = _run_recovery(
        tmp_path,
        available=40_000_000_000,
        layerdb_image_directory_exists=False,
        layerdb_mount_directory_exists=False,
        overlay_directory_exists=False,
        allow_live_host_gateway_prune=True,
        require_zero_runtime_reconcile="1",
    )

    assert result.returncode == 0, result.stderr
    assert "exact host gateway absent" in result.stdout
    assert "runtime_mode=empty" in result.stdout
    assert "phase=pre-absent-daemon-reconcile containers=0" in result.stdout
    assert "phase=post-absent-daemon-reconcile containers=0" in result.stdout
    assert "docker_zero_runtime_reconciler_v2.py" in sudo_log
    assert "docker_stale_mount_reclaimer_v2.py" not in sudo_log
    assert (tmp_path / "daemon-reconciled").is_file()
    assert (tmp_path / "builder-prune-count").read_text().strip() == "1"
    assert (tmp_path / "image-pruned").is_file()
    assert (tmp_path / "system-prune-attempts").read_text().strip() == "1"
    assert " rm " not in sudo_log


def test_required_zero_runtime_reconcile_rejects_absent_to_live_gateway_race(
    tmp_path: Path,
) -> None:
    result, sudo_log = _run_recovery(
        tmp_path,
        available=40_000_000_000,
        allow_live_host_gateway_prune=True,
        require_zero_runtime_reconcile="1",
        host_gateway_live_after_inventory=True,
    )

    assert result.returncode == 1
    assert "exact host gateway state changed during absent-runtime" in result.stderr
    assert "docker_zero_runtime_reconciler_v2.py" not in sudo_log
    assert (tmp_path / "builder-prune-count").read_text().strip() == "1"
    assert (tmp_path / "image-pruned").is_file()
    assert (tmp_path / "system-prune-attempts").read_text().strip() == "1"
    assert "systemctl" not in sudo_log


def test_required_absent_zero_runtime_reconcile_helper_failure_is_terminal(
    tmp_path: Path,
) -> None:
    result, sudo_log = _run_recovery(
        tmp_path,
        available=40_000_000_000,
        allow_live_host_gateway_prune=True,
        require_zero_runtime_reconcile="1",
        fail_zero_runtime_reconciler=True,
    )

    assert result.returncode == 1
    assert "guarded empty-runtime dockerd reconciliation failed" in result.stderr
    assert "docker_zero_runtime_reconciler_v2.py" in sudo_log
    assert not (tmp_path / "daemon-reconciled").exists()
    assert " rm " not in sudo_log


def test_required_absent_zero_runtime_reconcile_rejects_gateway_start_during_helper(
    tmp_path: Path,
) -> None:
    result, sudo_log = _run_recovery(
        tmp_path,
        available=40_000_000_000,
        allow_live_host_gateway_prune=True,
        require_zero_runtime_reconcile="1",
        host_gateway_appears_during_daemon_reconcile=True,
    )

    assert result.returncode == 1
    assert "exact host gateway state changed during absent-runtime" in result.stderr
    assert "docker_zero_runtime_reconciler_v2.py" in sudo_log
    assert (tmp_path / "daemon-reconciled").is_file()
    assert " rm " not in sudo_log


@pytest.mark.parametrize(
    "runtime",
    [
        {"containers": 1},
        {"containerd_containers": 1},
        {"containerd_tasks": 1},
        {"moby_shims": 1},
    ],
)
def test_required_zero_runtime_reconcile_rejects_nonempty_runtime(
    tmp_path: Path,
    runtime: dict[str, int],
) -> None:
    result, sudo_log = _run_recovery(
        tmp_path,
        available=25_000_000_000,
        host_gateway_live=True,
        allow_live_host_gateway_prune=True,
        require_zero_runtime_reconcile="1",
        **runtime,
    )

    assert result.returncode == 1
    assert "phase=pre-prune" in result.stderr
    assert "exact container runtime is empty" in result.stderr
    assert "docker_zero_runtime_reconciler_v2.py" not in sudo_log
    assert not (tmp_path / "builder-pruned").exists()


def test_required_zero_runtime_reconcile_rejects_stale_pre_audit(
    tmp_path: Path,
) -> None:
    result, sudo_log = _run_recovery(
        tmp_path,
        available=25_000_000_000,
        host_gateway_live=True,
        allow_live_host_gateway_prune=True,
        require_zero_runtime_reconcile="1",
        stale_audit_before_reconcile=True,
    )

    assert result.returncode == 1
    assert "audit was invalid before daemon reconciliation" in result.stderr
    assert "docker_zero_runtime_reconciler_v2.py" not in sudo_log
    assert (tmp_path / "builder-prune-count").read_text().strip() == "1"


def test_required_zero_runtime_reconcile_helper_failure_is_terminal(
    tmp_path: Path,
) -> None:
    result, sudo_log = _run_recovery(
        tmp_path,
        available=25_000_000_000,
        host_gateway_live=True,
        allow_live_host_gateway_prune=True,
        require_zero_runtime_reconcile="1",
        fail_zero_runtime_reconciler=True,
    )

    assert result.returncode == 1
    assert "dockerd reconciliation failed" in result.stderr
    assert "docker_zero_runtime_reconciler_v2.py" in sudo_log
    assert (tmp_path / "builder-prune-count").read_text().strip() == "1"


def test_validator_docker_recovery_fails_closed_for_low_space_host_gateway(
    tmp_path: Path,
) -> None:
    result, sudo_log = _run_recovery(
        tmp_path,
        available=17_999_999_999,
        layerdb_images=1,
        host_gateway_live=True,
    )

    assert result.returncode == 1
    assert "exact host gateway runtime has only 17999999999 free bytes" in result.stderr
    assert "refusing Docker maintenance, daemon stop, or data-root reset" in result.stderr
    assert not (tmp_path / "system-prune-attempts").exists()
    assert "systemctl stop" not in sudo_log
    assert "rm -rf" not in sudo_log


def test_live_host_gateway_online_reclaim_preserves_images_and_reaches_floor(
    tmp_path: Path,
) -> None:
    result, sudo_log = _run_recovery(
        tmp_path,
        available=15_000_000_000,
        available_after_stale_reclaim=40_000_000_000,
        images=9,
        layerdb_images=1_562,
        layerdb_mounts=133,
        overlay_directories=1_834,
        stale_overlay_mounts=133,
        host_gateway_live=True,
        allow_live_host_gateway_prune=True,
    )

    assert result.returncode == 0, result.stderr
    assert "online prune admitted under the exclusive operation lock" in result.stdout
    assert "phase=pre-prune containers=0" in result.stdout
    assert "phase=post-builder-prune containers=0" in result.stdout
    assert "phase=pre-stale-reclaim containers=0" in result.stdout
    assert "phase=pre-daemon-reconcile containers=0" in result.stdout
    assert "phase=post-daemon-reconcile containers=0" in result.stdout
    assert "phase=post-reconcile-builder-prune containers=0" in result.stdout
    assert "phase=post-reclaim containers=0" in result.stdout
    assert "Docker storage ready after bounded online reclaim" in result.stdout
    assert "docker_stale_mount_reclaimer_v2.py" in sudo_log
    assert "docker_zero_runtime_reconciler_v2.py" in sudo_log
    assert "docker_live_restore_reconciler_v2.py" not in sudo_log
    assert not (tmp_path / "image-pruned").exists()
    assert not (tmp_path / "system-prune-attempts").exists()
    assert "systemctl" not in sudo_log
    assert "rm " not in sudo_log
    assert "pkill" not in sudo_log


@pytest.mark.parametrize(
    ("failure_option", "message"),
    [
        ("fail_zero_runtime_reconciler", "dockerd reconciliation failed"),
        (
            "malformed_zero_runtime_reconciler",
            "reconciliation returned malformed evidence",
        ),
        (
            "change_audit_manifest_after_reconcile",
            "image/layer identity changed during daemon reconciliation",
        ),
        (
            "stale_audit_after_reconcile",
            "Docker audit was invalid after daemon reconciliation",
        ),
        (
            "change_audit_manifest_after_post_prune",
            "image/layer identity changed during reconciled builder prune",
        ),
    ],
)
def test_live_host_gateway_reconciliation_failures_remain_fail_closed(
    tmp_path: Path,
    failure_option: str,
    message: str,
) -> None:
    result, sudo_log = _run_recovery(
        tmp_path,
        available=15_000_000_000,
        available_after_stale_reclaim=40_000_000_000,
        images=4,
        stale_overlay_mounts=2,
        host_gateway_live=True,
        allow_live_host_gateway_prune=True,
        **{failure_option: True},
    )

    assert result.returncode == 1
    assert message in result.stderr
    assert "docker_zero_runtime_reconciler_v2.py" in sudo_log
    assert not (tmp_path / "image-pruned").exists()
    assert not (tmp_path / "system-prune-attempts").exists()
    assert "systemctl" not in sudo_log
    assert "rm " not in sudo_log
    assert "pkill" not in sudo_log


def test_live_host_gateway_reconciliation_rejects_gateway_exit(
    tmp_path: Path,
) -> None:
    result, sudo_log = _run_recovery(
        tmp_path,
        available=15_000_000_000,
        available_after_stale_reclaim=40_000_000_000,
        images=4,
        stale_overlay_mounts=2,
        host_gateway_live=True,
        allow_live_host_gateway_prune=True,
        host_gateway_exits_during_daemon_reconcile=True,
    )

    assert result.returncode == 1
    assert "exact host gateway identity changed" in result.stderr
    assert "docker_zero_runtime_reconciler_v2.py" in sudo_log
    assert not (tmp_path / "image-pruned").exists()
    assert not (tmp_path / "system-prune-attempts").exists()
    assert "systemctl" not in sudo_log
    assert "rm " not in sudo_log


def test_live_host_gateway_online_reclaim_rejects_malformed_free_space(
    tmp_path: Path,
) -> None:
    result, sudo_log = _run_recovery(
        tmp_path,
        available="not-a-number",  # type: ignore[arg-type]
        host_gateway_live=True,
        allow_live_host_gateway_prune=True,
    )

    assert result.returncode == 1
    assert "filesystem free-space state is malformed" in result.stderr
    assert "integer expression expected" not in result.stderr
    assert not (tmp_path / "builder-pruned").exists()
    assert not (tmp_path / "image-pruned").exists()
    assert not (tmp_path / "system-prune-attempts").exists()
    assert "systemctl" not in sudo_log
    assert "rm " not in sudo_log


def test_live_host_gateway_online_reclaim_rejects_overflowing_free_space(
    tmp_path: Path,
) -> None:
    result, sudo_log = _run_recovery(
        tmp_path,
        available="999999999999999999999999999999",  # type: ignore[arg-type]
        host_gateway_live=True,
        allow_live_host_gateway_prune=True,
    )

    assert result.returncode == 1
    assert "filesystem free-space state is malformed" in result.stderr
    assert "integer expression expected" not in result.stderr
    assert not (tmp_path / "builder-pruned").exists()
    assert not (tmp_path / "image-pruned").exists()
    assert not (tmp_path / "system-prune-attempts").exists()
    assert "systemctl" not in sudo_log
    assert "rm " not in sudo_log


def test_live_host_gateway_online_reclaim_rejects_overflowing_floor(
    tmp_path: Path,
) -> None:
    result, sudo_log = _run_recovery(
        tmp_path,
        available=15_000_000_000,
        live_runtime_min_free_bytes=999_999_999_999_999_999_999_999,
        host_gateway_live=True,
        allow_live_host_gateway_prune=True,
    )

    assert result.returncode == 2
    assert "positive bounded integer byte count" in result.stderr
    assert "integer expression expected" not in result.stderr
    assert not (tmp_path / "builder-pruned").exists()
    assert not (tmp_path / "image-pruned").exists()
    assert not (tmp_path / "system-prune-attempts").exists()
    assert "systemctl" not in sudo_log
    assert "rm " not in sudo_log


def test_live_host_gateway_online_reclaim_rejects_overflowing_retry_bound(
    tmp_path: Path,
) -> None:
    result, sudo_log = _run_recovery(
        tmp_path,
        available=15_000_000_000,
        prune_attempts=999_999_999_999_999_999_999_999,
        host_gateway_live=True,
        allow_live_host_gateway_prune=True,
    )

    assert result.returncode == 2
    assert "PRUNE_ATTEMPTS must be between 1 and 300" in result.stderr
    assert "integer expression expected" not in result.stderr
    assert not (tmp_path / "builder-pruned").exists()
    assert not (tmp_path / "image-pruned").exists()
    assert not (tmp_path / "system-prune-attempts").exists()
    assert "systemctl" not in sudo_log
    assert "rm " not in sudo_log


def test_live_host_gateway_online_builder_prune_can_reach_floor_without_raw_reclaim(
    tmp_path: Path,
) -> None:
    result, sudo_log = _run_recovery(
        tmp_path,
        available=17_500_000_000,
        available_after_builder_prune=19_000_000_000,
        images=3,
        host_gateway_live=True,
        allow_live_host_gateway_prune=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Docker storage ready after bounded online reclaim" in result.stdout
    assert "docker_stale_mount_reclaimer_v2.py" not in sudo_log
    assert not (tmp_path / "image-pruned").exists()
    assert not (tmp_path / "system-prune-attempts").exists()


def test_live_host_gateway_online_reclaim_accepts_zero_count_success_status(
    tmp_path: Path,
) -> None:
    result, sudo_log = _run_recovery(
        tmp_path,
        available=17_500_000_000,
        available_after_builder_prune=19_000_000_000,
        images=3,
        host_gateway_live=True,
        allow_live_host_gateway_prune=True,
        zero_shim_probe_succeeds=True,
    )

    assert result.returncode == 0, result.stderr
    assert "moby_shims=0" in result.stdout
    assert "Docker storage ready after bounded online reclaim" in result.stdout
    assert "docker_stale_mount_reclaimer_v2.py" not in sudo_log
    assert not (tmp_path / "image-pruned").exists()
    assert not (tmp_path / "system-prune-attempts").exists()


def test_live_host_gateway_online_reclaim_fails_if_floor_remains_unmet(
    tmp_path: Path,
) -> None:
    result, sudo_log = _run_recovery(
        tmp_path,
        available=15_000_000_000,
        available_after_stale_reclaim=17_999_999_999,
        images=3,
        host_gateway_live=True,
        allow_live_host_gateway_prune=True,
    )

    assert result.returncode == 1
    assert "bounded online Docker reclaim left only 17999999999" in result.stderr
    assert "refusing daemon stop or data-root reset" in result.stderr
    assert "docker_stale_mount_reclaimer_v2.py" in sudo_log
    assert not (tmp_path / "image-pruned").exists()
    assert not (tmp_path / "system-prune-attempts").exists()
    assert "systemctl" not in sudo_log
    assert "rm " not in sudo_log


def test_live_host_gateway_online_reclaim_refuses_nonempty_runtime(
    tmp_path: Path,
) -> None:
    result, sudo_log = _run_recovery(
        tmp_path,
        available=15_000_000_000,
        containers=1,
        containerd_containers=1,
        containerd_tasks=1,
        moby_shims=1,
        host_gateway_live=True,
        allow_live_host_gateway_prune=True,
    )

    assert result.returncode == 1
    assert (
        "refusing online Docker prune unless the exact container runtime is empty"
        in result.stderr
    )
    assert not (tmp_path / "builder-pruned").exists()
    assert "docker_stale_mount_reclaimer_v2.py" not in sudo_log
    assert "systemctl" not in sudo_log


def test_live_host_gateway_online_reclaim_refuses_nonzero_no_match_status(
    tmp_path: Path,
) -> None:
    result, sudo_log = _run_recovery(
        tmp_path,
        available=15_000_000_000,
        moby_shims=1,
        host_gateway_live=True,
        allow_live_host_gateway_prune=True,
        nonzero_shim_probe_reports_no_match=True,
    )

    assert result.returncode == 1
    assert "moby shim inventory status is inconsistent" in result.stderr
    assert not (tmp_path / "builder-pruned").exists()
    assert "docker_stale_mount_reclaimer_v2.py" not in sudo_log
    assert "systemctl" not in sudo_log


def test_offline_recovery_rejects_nonzero_no_match_shim_status(
    tmp_path: Path,
) -> None:
    result, sudo_log = _run_recovery(
        tmp_path,
        available=40_000_000_000,
        moby_shims=1,
        nonzero_shim_probe_reports_no_match=True,
    )

    assert result.returncode == 1
    assert "moby shim inventory status is inconsistent" in result.stderr
    assert "systemctl" not in sudo_log
    assert "pkill" not in sudo_log
    assert "docker_stale_mount_reclaimer_v2.py" not in sudo_log
    assert "rm " not in sudo_log


def test_live_host_gateway_online_reclaim_rejects_failed_shim_inventory(
    tmp_path: Path,
) -> None:
    result, sudo_log = _run_recovery(
        tmp_path,
        available=15_000_000_000,
        host_gateway_live=True,
        allow_live_host_gateway_prune=True,
        fail_shim_probe=True,
    )

    assert result.returncode == 1
    assert "moby shim inventory is unreadable" in result.stderr
    assert not (tmp_path / "builder-pruned").exists()
    assert "docker_stale_mount_reclaimer_v2.py" not in sudo_log
    assert "systemctl" not in sudo_log
    assert "rm " not in sudo_log


def test_live_host_gateway_online_reclaim_rejects_malformed_shim_inventory(
    tmp_path: Path,
) -> None:
    result, sudo_log = _run_recovery(
        tmp_path,
        available=15_000_000_000,
        host_gateway_live=True,
        allow_live_host_gateway_prune=True,
        malformed_shim_probe=True,
    )

    assert result.returncode == 1
    assert "moby shim inventory is malformed" in result.stderr
    assert "integer expression expected" not in result.stderr
    assert not (tmp_path / "builder-pruned").exists()
    assert "docker_stale_mount_reclaimer_v2.py" not in sudo_log
    assert "systemctl" not in sudo_log
    assert "rm " not in sudo_log


def test_live_host_gateway_online_reclaim_bounds_hung_shim_inventory(
    tmp_path: Path,
) -> None:
    started = time.monotonic()
    result, sudo_log = _run_recovery(
        tmp_path,
        available=15_000_000_000,
        host_gateway_live=True,
        allow_live_host_gateway_prune=True,
        hang_shim_probe=True,
        daemon_probe_timeout_seconds=1,
    )
    elapsed = time.monotonic() - started

    assert result.returncode == 1
    assert elapsed < 8
    assert "moby shim inventory is unreadable" in result.stderr
    assert not (tmp_path / "builder-pruned").exists()
    assert "docker_stale_mount_reclaimer_v2.py" not in sudo_log
    assert "systemctl" not in sudo_log
    assert "rm " not in sudo_log


def test_live_host_gateway_online_reclaim_refuses_gateway_identity_change(
    tmp_path: Path,
) -> None:
    result, sudo_log = _run_recovery(
        tmp_path,
        available=15_000_000_000,
        images=3,
        host_gateway_live=True,
        allow_live_host_gateway_prune=True,
        host_gateway_exits_during_live_prune=True,
    )

    assert result.returncode == 1
    assert "exact host gateway identity changed during online Docker reclaim" in result.stderr
    assert "docker_stale_mount_reclaimer_v2.py" not in sudo_log
    assert not (tmp_path / "image-pruned").exists()
    assert "systemctl" not in sudo_log


def test_live_host_gateway_online_reclaim_refuses_image_identity_change(
    tmp_path: Path,
) -> None:
    result, sudo_log = _run_recovery(
        tmp_path,
        available=15_000_000_000,
        images=3,
        images_after_builder_prune=2,
        host_gateway_live=True,
        allow_live_host_gateway_prune=True,
    )

    assert result.returncode == 1
    assert "Docker image identity changed during online reclaim" in result.stderr
    assert "docker_stale_mount_reclaimer_v2.py" not in sudo_log
    assert not (tmp_path / "image-pruned").exists()
    assert "systemctl" not in sudo_log


def test_live_host_gateway_online_reclaim_never_recovers_unreadable_daemon(
    tmp_path: Path,
) -> None:
    result, sudo_log = _run_recovery(
        tmp_path,
        available=15_000_000_000,
        host_gateway_live=True,
        allow_live_host_gateway_prune=True,
        hang_ctr_probe=True,
        daemon_ready_attempts=1,
        daemon_probe_timeout_seconds=1,
    )

    assert result.returncode == 1
    assert "refusing daemon maintenance" in result.stderr
    assert "systemctl" not in sudo_log
    assert not (tmp_path / "builder-pruned").exists()


def test_live_host_gateway_online_reclaim_rejects_unexpected_docker_root(
    tmp_path: Path,
) -> None:
    result, sudo_log = _run_recovery(
        tmp_path,
        available=15_000_000_000,
        docker_root="/srv/docker",
        host_gateway_live=True,
        allow_live_host_gateway_prune=True,
    )

    assert result.returncode == 1
    assert "refusing online prune for unexpected Docker data-root" in result.stderr
    assert not (tmp_path / "builder-pruned").exists()
    assert not (tmp_path / "image-pruned").exists()
    assert not (tmp_path / "system-prune-attempts").exists()
    assert "systemctl" not in sudo_log
    assert "rm " not in sudo_log


def test_live_host_gateway_online_reclaim_rejects_docker_root_change(
    tmp_path: Path,
) -> None:
    result, sudo_log = _run_recovery(
        tmp_path,
        available=15_000_000_000,
        docker_root_after_builder_prune="/srv/docker",
        host_gateway_live=True,
        allow_live_host_gateway_prune=True,
    )

    assert result.returncode == 1
    assert "refusing online prune for unexpected Docker data-root" in result.stderr
    assert (tmp_path / "builder-pruned").exists()
    assert "docker_stale_mount_reclaimer_v2.py" not in sudo_log
    assert not (tmp_path / "image-pruned").exists()
    assert not (tmp_path / "system-prune-attempts").exists()
    assert "systemctl" not in sudo_log
    assert "rm " not in sudo_log


def test_live_host_gateway_online_reclaim_fails_closed_on_helper_error(
    tmp_path: Path,
) -> None:
    result, sudo_log = _run_recovery(
        tmp_path,
        available=15_000_000_000,
        images=3,
        host_gateway_live=True,
        allow_live_host_gateway_prune=True,
        fail_stale_reclaimer=True,
    )

    assert result.returncode == 1
    assert "validated stale Docker overlay reclaim failed" in result.stderr
    assert "docker_stale_mount_reclaimer_v2.py" in sudo_log
    assert not (tmp_path / "image-pruned").exists()
    assert not (tmp_path / "system-prune-attempts").exists()
    assert "systemctl" not in sudo_log


def test_live_host_gateway_online_reclaim_rejects_pid_reuse_identity(
    tmp_path: Path,
) -> None:
    result, sudo_log = _run_recovery(
        tmp_path,
        available=15_000_000_000,
        images=3,
        host_gateway_live=True,
        allow_live_host_gateway_prune=True,
        host_gateway_start_time_changes_during_live_prune=True,
    )

    assert result.returncode == 1
    assert "exact host gateway identity changed during online Docker reclaim" in result.stderr
    assert "docker_stale_mount_reclaimer_v2.py" not in sudo_log
    assert "systemctl" not in sudo_log


def test_live_host_gateway_online_reclaim_rejects_post_reclaim_runtime_race(
    tmp_path: Path,
) -> None:
    result, sudo_log = _run_recovery(
        tmp_path,
        available=15_000_000_000,
        available_after_stale_reclaim=40_000_000_000,
        images=3,
        stale_overlay_mounts=1,
        host_gateway_live=True,
        allow_live_host_gateway_prune=True,
        runtime_appears_after_stale_reclaim=True,
    )

    assert result.returncode == 1
    assert "phase=pre-daemon-reconcile" in result.stderr
    assert "refusing online Docker prune unless the exact container runtime is empty" in result.stderr
    assert "docker_stale_mount_reclaimer_v2.py" in sudo_log
    assert "docker_zero_runtime_reconciler_v2.py" not in sudo_log
    assert not (tmp_path / "image-pruned").exists()
    assert "systemctl" not in sudo_log


def test_live_host_gateway_online_reclaim_rejects_malformed_helper_evidence(
    tmp_path: Path,
) -> None:
    result, sudo_log = _run_recovery(
        tmp_path,
        available=15_000_000_000,
        images=3,
        host_gateway_live=True,
        allow_live_host_gateway_prune=True,
        malformed_stale_reclaimer=True,
    )

    assert result.returncode == 1
    assert "returned malformed evidence" in result.stderr
    assert "docker_stale_mount_reclaimer_v2.py" in sudo_log
    assert not (tmp_path / "image-pruned").exists()
    assert "systemctl" not in sudo_log


def test_validator_docker_recovery_rechecks_gateway_immediately_before_stop(
    tmp_path: Path,
) -> None:
    result, sudo_log = _run_recovery(
        tmp_path,
        available=40_000_000_000,
        layerdb_images=1,
        host_gateway_live_after_inventory=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.count('"status": "absent"') == 1
    assert result.stdout.count('"status": "live"') == 1
    assert "runtime_mode=host-gateway-live phase=pre-reset" in result.stdout
    assert "storage maintenance deferred while the exact host gateway is live" in result.stdout
    assert "systemctl stop" not in sudo_log
    assert "rm -rf" not in sudo_log


def test_validator_docker_recovery_late_gateway_low_space_fails_closed(
    tmp_path: Path,
) -> None:
    result, sudo_log = _run_recovery(
        tmp_path,
        available=17_999_999_999,
        layerdb_images=1,
        host_gateway_live_after_inventory=True,
    )

    assert result.returncode == 1
    assert "runtime_mode=host-gateway-live phase=pre-reset" in result.stdout
    assert "exact host gateway runtime has only 17999999999 free bytes" in result.stderr
    assert "systemctl stop" not in sudo_log
    assert "rm -rf" not in sudo_log


def test_validator_docker_recovery_restores_daemons_after_unmount_failure(
    tmp_path: Path,
) -> None:
    result, sudo_log = _run_recovery(
        tmp_path,
        available=40_000_000_000,
        layerdb_images=1,
        stale_overlay_mounts=1,
        fail_umount=True,
    )

    assert result.returncode == 1
    assert "Recovering Docker/containerd readiness after failed data-root reset" in result.stderr
    assert "readiness recovered after failed data-root reset" in result.stderr
    stop = sudo_log.index(
        "systemctl stop docker.service docker.socket containerd.service"
    )
    failed_unmount = sudo_log.index("umount /var/lib/docker/overlay2/")
    recovery_start = sudo_log.index(
        "systemctl start containerd.service docker.service",
        failed_unmount,
    )
    assert stop < failed_unmount < recovery_start
    assert not (tmp_path / "daemons-stopped").exists()


def test_validator_docker_recovery_restores_after_partial_stop_failure(
    tmp_path: Path,
) -> None:
    result, sudo_log = _run_recovery(
        tmp_path,
        available=40_000_000_000,
        layerdb_images=1,
        fail_systemctl_stop=True,
    )

    assert result.returncode == 1
    assert "readiness recovered after failed data-root reset" in result.stderr
    stop = sudo_log.index(
        "systemctl stop docker.service docker.socket containerd.service"
    )
    recovery_start = sudo_log.index(
        "systemctl start containerd.service docker.service",
        stop,
    )
    assert stop < recovery_start
    assert not (tmp_path / "daemons-stopped").exists()


def test_validator_docker_recovery_bounds_failed_exit_readiness_attempts(
    tmp_path: Path,
) -> None:
    result, sudo_log = _run_recovery(
        tmp_path,
        available=40_000_000_000,
        layerdb_images=1,
        fail_systemctl_stop=True,
        fail_daemon_recovery=True,
        daemon_ready_attempts=2,
    )

    assert result.returncode == 1
    assert "recovery remained unavailable after failed data-root reset" in result.stderr
    assert "systemctl start containerd.service docker.service" in sudo_log
    assert (tmp_path / "daemon-ready-probes").read_text(encoding="utf-8") == "2\n"


def test_validator_docker_recovery_refuses_running_containerd_task(
    tmp_path: Path,
) -> None:
    result, sudo_log = _run_recovery(
        tmp_path,
        available=40_000_000_000,
        containerd_containers=1,
        containerd_tasks=1,
        containerd_running_tasks=1,
    )

    assert result.returncode == 1
    assert "refusing containerd reset while 1 moby task(s) are running" in result.stderr
    assert "systemctl stop" not in sudo_log


def test_validator_docker_recovery_waits_for_transient_containerd_task(
    tmp_path: Path,
) -> None:
    result, sudo_log = _run_recovery(
        tmp_path,
        available=40_000_000_000,
        containerd_running_tasks=1,
        running_tasks_clear_after=1,
    )

    assert result.returncode == 0, result.stderr
    assert "Waiting for Docker teardown to settle" in result.stderr
    assert "systemctl stop" not in sudo_log


def test_validator_docker_recovery_retries_transient_system_prune(
    tmp_path: Path,
) -> None:
    result, sudo_log = _run_recovery(
        tmp_path,
        available=40_000_000_000,
        system_prune_failures=1,
    )

    assert result.returncode == 0, result.stderr
    assert "Docker system prune did not settle; retrying (1/2)" in result.stderr
    assert "systemctl stop" not in sudo_log


def test_validator_docker_recovery_retries_unreadable_teardown_state(
    tmp_path: Path,
) -> None:
    result, sudo_log = _run_recovery(
        tmp_path,
        available=40_000_000_000,
        task_probe_failures=1,
    )

    assert result.returncode == 0, result.stderr
    assert "Waiting for Docker teardown state to become readable" in result.stderr
    assert "systemctl stop" not in sudo_log


def test_validator_docker_recovery_preserves_healthy_live_runtime(
    tmp_path: Path,
) -> None:
    result, sudo_log = _run_recovery(
        tmp_path,
        available=40_000_000_000,
        containers=11,
        containerd_containers=11,
        containerd_tasks=11,
        containerd_running_tasks=11,
        moby_shims=11,
    )

    assert result.returncode == 0, result.stderr
    assert "containers=11" in result.stdout
    assert "containerd_running_tasks=11" in result.stdout
    assert "orphaned=0" in result.stdout
    assert "runtime_mode=live required_free_bytes=18000000000" in result.stdout
    assert "Waiting for Docker teardown" not in result.stderr
    assert "systemctl stop" not in sudo_log
    assert "rm -rf" not in sudo_log
    assert "docker_live_restore_reconciler_v2" not in sudo_log


def test_validator_docker_recovery_reconciles_prior_raw_cleanup_state(
    tmp_path: Path,
) -> None:
    result, sudo_log = _run_recovery(
        tmp_path,
        available=40_000_000_000,
        containers=11,
        containerd_containers=11,
        containerd_tasks=11,
        containerd_running_tasks=11,
        moby_shims=11,
        live_restore_enabled=False,
    )

    assert result.returncode == 0, result.stderr
    assert '"reclaimed_mount_count":0' in result.stdout
    assert "Reconciling dockerd metadata" in result.stdout
    assert "docker_live_restore_reconciler_v2" in sudo_log
    assert "systemctl stop" not in sudo_log


def test_validator_docker_recovery_allows_one_recovery_build_after_failed_deploy(
    tmp_path: Path,
) -> None:
    result, sudo_log = _run_recovery(
        tmp_path,
        available=25_000_000_000,
        containers=11,
        containerd_containers=11,
        containerd_tasks=11,
        containerd_running_tasks=11,
        moby_shims=11,
        layerdb_images=3_889,
        layerdb_mounts=312,
        overlay_directories=4_514,
    )

    assert result.returncode == 0, result.stderr
    assert "runtime_mode=live required_free_bytes=18000000000" in result.stdout
    assert "Docker storage ready: free_bytes=25000000000" in result.stdout
    assert "systemctl stop" not in sudo_log
    assert "rm -rf" not in sudo_log


def test_validator_docker_recovery_refuses_live_runtime_below_recovery_reserve(
    tmp_path: Path,
) -> None:
    result, sudo_log = _run_recovery(
        tmp_path,
        available=17_999_999_999,
        containers=11,
        containerd_containers=11,
        containerd_tasks=11,
        containerd_running_tasks=11,
        moby_shims=11,
    )

    assert result.returncode == 1
    assert "live validator runtime has only 17999999999 free bytes" in result.stderr
    assert "refusing to stop containers or reset Docker storage" in result.stderr
    assert "systemctl stop" not in sudo_log
    assert "rm -rf" not in sudo_log


def test_validator_docker_recovery_reclaims_stale_nitro_mounts_before_floor(
    tmp_path: Path,
) -> None:
    result, sudo_log = _run_recovery(
        tmp_path,
        available=14_738_382_848,
        available_after_stale_reclaim=40_000_000_000,
        containers=11,
        containerd_containers=11,
        containerd_tasks=11,
        containerd_running_tasks=11,
        moby_shims=11,
        layerdb_images=4_147,
        layerdb_mounts=336,
        overlay_directories=4_819,
        stale_overlay_mounts=325,
    )

    assert result.returncode == 0, result.stderr
    assert '"active_mount_count":11' in result.stdout
    assert '"mounted_overlay_count":336' in result.stdout
    assert '"reclaimed_mount_count":325' in result.stdout
    assert "Docker storage ready: free_bytes=40000000000" in result.stdout
    assert "docker_stale_mount_reclaimer_v2" in sudo_log
    assert "docker_live_restore_reconciler_v2" in sudo_log
    assert "PYTHONSAFEPATH=1 python3" in sudo_log
    assert "-m validator_tee.host" not in sudo_log
    assert "systemctl stop" not in sudo_log
    assert "rm -rf" not in sudo_log


def test_validator_docker_recovery_keeps_strict_empty_runtime_floor(
    tmp_path: Path,
) -> None:
    result, sudo_log = _run_recovery(
        tmp_path,
        available=25_000_000_000,
    )

    assert result.returncode == 1
    assert "runtime_mode=empty required_free_bytes=30000000000" in result.stdout
    assert "Docker data-root reset left only 25000000000 free bytes" in result.stderr
    assert "systemctl stop docker.service docker.socket containerd.service" in sudo_log


def test_validator_docker_recovery_leaves_clean_root_above_floor_untouched(
    tmp_path: Path,
) -> None:
    result, sudo_log = _run_recovery(
        tmp_path,
        available=40_000_000_000,
    )

    assert result.returncode == 0, result.stderr
    assert "orphaned=0" in result.stdout
    assert "systemctl stop" not in sudo_log
    assert "rm -rf" not in sudo_log


def test_validator_docker_recovery_reconciles_orphans_without_deleting_volume(
    tmp_path: Path,
) -> None:
    result, sudo_log = _run_recovery(
        tmp_path,
        available=40_000_000_000,
        volumes=1,
        layerdb_images=1,
        zero_runtime_reconcile_clears_orphan_metadata=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Reconciling orphaned dockerd metadata before considering a data-root reset" in result.stdout
    assert "Docker storage ready after orphaned metadata reconciliation: free_bytes=40000000000" in result.stdout
    assert "docker_zero_runtime_reconciler_v2.py" in sudo_log
    assert "systemctl stop docker.service docker.socket containerd.service" not in sudo_log
    assert "rm -rf" not in sudo_log


@pytest.mark.parametrize(
    "case,expected",
    [("missing", 0), ("empty", 0), ("link_index", 0), ("record", 1),
     ("invalid_parent", 1), ("image_inventory_failure", 1), ("sudo_failure", 1)],
)
def test_reconciled_metadata_inventory_fails_closed(
    tmp_path: Path, case: str, expected: int,
) -> None:
    source = (ROOT / "validator_tee/scripts/reclaim_docker_storage_v2.sh").read_text()
    function = source[source.index("empty_runtime_metadata_is_clear() {"):
                      source.index("\nonline_image_ids() {")]
    docker_root = tmp_path / "docker"
    docker_root.mkdir()
    if case in ("empty", "link_index", "record"):
        (docker_root / "overlay2").mkdir()
    if case == "link_index":
        (docker_root / "overlay2/l").mkdir()
    if case == "record":
        (docker_root / "overlay2/stale-record").touch()
    if case == "invalid_parent":
        (docker_root / "image").touch()
    driver = tmp_path / "inventory.sh"
    driver.write_text(
        "set -euo pipefail\n"
        "run_bounded_daemon_inventory() { return "
        + ("7" if case == "image_inventory_failure" else "0") + "; }\n"
        + ('sudo() { return 1; }\n' if case == "sudo_failure"
           else 'sudo() { "$@"; }\n')
        + function
        + "\nif empty_runtime_metadata_is_clear; then exit 0; else exit 1; fi\n"
    )
    result = subprocess.run(
        ["bash", str(driver)], capture_output=True, text=True, timeout=10,
        env={**os.environ, "DOCKER_ROOT": str(docker_root)},
    )
    assert result.returncode == expected, result.stdout + result.stderr


def test_validator_docker_recovery_keeps_a_volume_when_metadata_persists(
    tmp_path: Path,
) -> None:
    result, sudo_log = _run_recovery(
        tmp_path,
        available=40_000_000_000,
        volumes=1,
        layerdb_images=1,
    )

    assert result.returncode == 1
    assert "Docker metadata is not proven empty after guarded reconciliation" in result.stderr
    assert "refusing Docker data-root reset while 1 volume(s) remain" in result.stderr
    assert "docker_zero_runtime_reconciler_v2.py" in sudo_log
    assert "systemctl stop" not in sudo_log
    assert "rm -rf" not in sudo_log


def test_validator_docker_recovery_preserves_any_named_volume(
    tmp_path: Path,
) -> None:
    result, sudo_log = _run_recovery(
        tmp_path,
        available=20_000_000_000,
        volumes=1,
    )

    assert result.returncode == 1
    assert "refusing Docker data-root reset while 1 volume(s) remain" in result.stderr
    assert "systemctl stop" not in sudo_log
    assert "rm -rf" not in sudo_log
