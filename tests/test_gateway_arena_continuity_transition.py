"""Exercise the controller's real Arena cutover blocks without host writes."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import textwrap

import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "gw_restart.sh").read_text(encoding="utf-8")
SHA = "a" * 40


def _between(start: str, end: str) -> str:
    assert SOURCE.count(start) == 1, start
    return SOURCE.split(start, 1)[1].split(end, 1)[0]


def _function(name: str) -> str:
    start = SOURCE.index(f"{name}() {{")
    rest = SOURCE[start:]
    # These functions have no nested function declarations. The matching
    # first column brace ends the shell function, including any heredocs.
    return rest.split("\n}\n", 1)[0] + "\n}\n"


OVERLAP = 'if [ "$ARENA_CONTINUITY_ENABLED" = 1 ]; then\n' + _between(
    'if [ "$ARENA_CONTINUITY_ENABLED" = 1 ]; then\n'
    '  GATEWAY_DEPLOY_STAGE="arena_candidate_overlap"',
    '\nGATEWAY_DEPLOY_STAGE="lab_arena_claim_drain"',
).join(['  GATEWAY_DEPLOY_STAGE="arena_candidate_overlap"', ''])
DRAIN_AND_STOP = 'GATEWAY_DEPLOY_STAGE="lab_arena_claim_drain"' + _between(
    'GATEWAY_DEPLOY_STAGE="lab_arena_claim_drain"',
    '\nrecord_gateway_restart_timing "shutdown_complete"',
)
BACKEND_CHECK = 'GATEWAY_DEPLOY_STAGE="gateway_health_check"' + _between(
    'GATEWAY_DEPLOY_STAGE="gateway_health_check"',
    '\nGATEWAY_DEPLOY_STAGE="host_restart_script_install"',
)
PUBLISH = 'GATEWAY_DEPLOY_STAGE="host_restart_script_install"' + _between(
    'GATEWAY_DEPLOY_STAGE="host_restart_script_install"',
    '\nLAB_ARENA_RESTART_GUARD_GENERATION=""',
)


def _runtime_clone(tmp_path: Path, origin: str) -> str:
    """Run the controller's real environment projections with stale state."""
    stale = {
        "GATEWAY_DESTRUCTIVE_PHASE_STARTED": "1",
        "GATEWAY_CONTROLLER_PROCESS_STATE_FILE": "/stale/other-worker.json",
        "GATEWAY_PREFLIGHT_TREE": "",
        "GATEWAY_RESTART_AUTHORITY_ROOT": "/stale/authority",
        "GATEWAY_RESTART_AUTHORITY_COMMIT": "b" * 40,
        "SAFE_RUNTIME_VALUE": "runtime value with spaces",
        "SUPABASE_SERVICE_ROLE_KEY": "fixture-only-service-key",
        "LAB_ARENA_OPENROUTER_API_KEY": "fixture-only-provider-key",
    }

    def project(marker: str, *args: Path, inherited: Path | None = None) -> None:
        block = _between(marker + " <<'PY'\n", "\nPY\n")
        if inherited is not None:
            block = block.replace('f"/proc/{pid}/environ"', repr(str(inherited)))
        subprocess.run([sys.executable, "-", *(str(arg) for arg in args)],
                       input=block, text=True, check=True, capture_output=True, timeout=5)

    clone = tmp_path / "projected-clone.sh"
    if origin == "live":
        inherited = tmp_path / "inherited-environ"
        inherited.write_bytes(b"\0".join(f"{key}={value}".encode()
                                          for key, value in stale.items()) + b"\0")
        project('python3 - "$PID" "$ENV_CLONE"', inherited, clone, inherited=inherited)
    else:
        secret = tmp_path / "secret-document"
        raw = (json.dumps(stale) if origin == "secret-json" else
               "\n".join(f"export {key}={shlex.quote(value)}" for key, value in stale.items()))
        secret.write_text(raw)
        cache = tmp_path / "gateway.env"
        if origin == "secret-cache":
            cache.write_text(raw)
        else:
            project('python3 - "$SECRET_TMP" "$GATEWAY_ENV_FILE"', secret, cache)
            # The canonical cache also feeds scoped service environment loaders.
            assert "GATEWAY_DESTRUCTIVE_PHASE_STARTED" not in cache.read_text()
            assert "GATEWAY_CONTROLLER_PROCESS_STATE_FILE" not in cache.read_text()
            assert "GATEWAY_PREFLIGHT_TREE" not in cache.read_text()
        project('python3 - "$GATEWAY_ENV_FILE" "$ENV_SECRET"', cache, clone)
    result = clone.read_text()
    for key in stale:
        if key.startswith("GATEWAY_"):
            assert key not in result
    assert "SAFE_RUNTIME_VALUE='runtime value with spaces'" in result
    assert "SUPABASE_SERVICE_ROLE_KEY=fixture-only-service-key" in result
    assert "LAB_ARENA_OPENROUTER_API_KEY=fixture-only-provider-key" in result
    return result


def _run(tmp_path: Path, *, fail: str = "", sha: str = SHA,
         runtime_clone: str = "", review_rows: list[dict] | None = None) -> subprocess.CompletedProcess[str]:
    preflight = tmp_path / "preflight"
    preflight.mkdir(exist_ok=True)
    (preflight / "marker").write_text("candidate\n", encoding="utf-8")
    clone = tmp_path / "env-clone"
    clone.write_text(runtime_clone, encoding="utf-8")
    ingress = tmp_path / "ingress"
    if not ingress.exists():
        ingress.write_text("legacy 0 - legacy\n", encoding="utf-8")
    if review_rows is not None:
        (tmp_path / "review-rows.json").write_text(json.dumps(review_rows))
        (tmp_path / "idle-review-probe.py").write_text(
            "import json, os, sys\n"
            "from datetime import datetime, timedelta, timezone\n"
            "from types import SimpleNamespace\n"
            f"sys.path.insert(0, {str(ROOT)!r})\n"
            "from scripts.arena_first_cutover_idle import require_idle\n"
            "rows=json.load(open(os.environ['IDLE_REVIEW_FILE']))\n"
            "now=datetime(2026, 10, 15, tzinfo=timezone.utc)\n"
            "service=SimpleNamespace(active_rounds=lambda:[{'round_id':'r'}],\n"
            " _round=lambda _id:{'round_id':'r','status':'stage2'},\n"
            " _submission_replacement_cutoff=lambda _round:now+timedelta(hours=3),\n"
            " _is_daily_baseline=lambda _row,_round:False,\n"
            " _store=SimpleNamespace(list_submissions=lambda _id,*,status:[row for row in rows if row['status']==status]))\n"
            "require_idle(service, now=now)\n",
        )
    python = tmp_path / "mock-python"
    python.write_text(
        '#!/bin/bash\n'
        'printf "python:%s\\n" "$*" >> "$EVENTS"\n'
        'if [ "${FAIL_IDLE:-}" = 1 ] && [[ "$*" == *arena_first_cutover_idle.py* ]]; '
        'then exit 1; fi\n'
        'if [ -n "${IDLE_REVIEW_FILE:-}" ] && [[ "$*" == *arena_first_cutover_idle.py* ]]; '
        'then exec python3 "$TASK_TMP/idle-review-probe.py"; fi\n'
        'if [ "${FAIL:-}" = domain_gap ] && [[ "$*" == *gateway-process.json* ]]; then exit 1; fi\n'
        'if [[ "$*" == *"manage_owned_process_group.py stop "* ]] || '
        '[[ "$*" == *"manage_owned_process_group.py verify "* ]] || '
        '[[ "$*" == *"manage_owned_process_group.py verify-invocation "* ]]; then\n'
        '  state_file=""; prior=""\n'
        '  for item in "$@"; do if [ "$prior" = --state-file ]; then state_file="$item"; break; fi; prior="$item"; done\n'
        '  if [[ "$state_file" == "$TASK_TMP/runtime/"* ]]; then\n'
        '    if [[ "$*" == *"manage_owned_process_group.py stop "* ]]; then rm -f "$state_file"; '
        'else [ -f "$state_file" ] || exit 1; fi\n'
        '  fi\n'
        '  if [ "${FAIL:-}" = disowned ] && [[ "$*" == *"manage_owned_process_group.py verify "* ]]; '
        'then exit 1; fi\n'
        'fi\n'
        'if [[ "$*" == *" verify-invocation "* ]]; then '
        'printf "gateway-1792000000-123\\n"; exit 0; fi\n'
        'if [[ "$*" == *"lab_arena_restart_claim_guard.py owned "* ]]; then\n'
        '  requested=""\n'
        '  while [ "$#" -gt 1 ]; do if [ "$1" = --candidate ]; then requested="$2"; break; fi; shift; done\n'
        '  if [ "${FAIL:-}" = owner_mismatch ]; then requested="cccccccccccccccccccccccccccccccccccccccc"; fi\n'
        '  active=true; if [ "${FAIL:-}" = expired_guard ]; then active=false; fi\n'
        '  printf \'{"schema_version":"leadpoet.lab_arena.restart_guard_state.v1",'
        '"guard_present":true,"guard_active":%s,"candidate_commit":"%s",'
        '"restart_scope":"gateway","restart_phase":"gateway_destructive",'
        '"guard_generation":1}\\n\' "$active" "$requested"\n'
        '  exit 0\n'
        'fi\n'
        'if [ "${FAIL:-}" = occupied ] && [ "$1" = - ] && [ "$2" = 8794 ]; then exit 1; fi\n'
        'if [ "$1" = - ]; then exec python3 "$@"; fi\n',
        encoding="utf-8",
    )
    python.chmod(0o700)
    harness = tmp_path / "cutover.sh"
    functions = "\n".join(_function(name) for name in (
        "arena_slot_ports", "arena_runtime_tree", "prepare_arena_runtime_tree",
        "retire_arena_runtime_tree",
        "arena_ingress_transition", "arena_component_command",
        "rollback_arena_candidate_before_destructive",
        "recover_arena_restart_owner", "restore_arena_recovery_guard_before_destructive",
        "wait_for_gateway_v2_authority", "wait_for_arena_driver_startup",
        "wait_for_arena_current_readiness",
        "abort_lab_arena_restart_guard_before_destructive",
    ))
    functions += _function("stop_arena_component").replace(
        "stop_arena_component() {", "actual_stop_arena_component() {", 1,
    )
    harness.write_text(
        textwrap.dedent("""\
        #!/bin/bash
        set -euo pipefail
        record() { printf '%s\\n' "$*" >> "$EVENTS"; }
        arena_ingress_tool() {
          local root="$1"; shift
          if [ "$1" = --inspect ]; then cat "$INGRESS_FILE"; return; fi
          [ "$1" = --activate ] || return 1
          shift
          local old_mode= old_slot= old_sha= old_backend=
          local new_mode= new_slot= new_sha= new_backend=
          while [ "$#" -gt 0 ]; do
            case "$1" in
              --expect-mode) old_mode="$2";; --expect-slot) old_slot="$2";;
              --expect-sha) old_sha="$2";; --expect-backend) old_backend="$2";;
              --mode) new_mode="$2";; --slot) new_slot="$2";;
              --sha) new_sha="$2";; --backend) new_backend="$2";;
              *) return 1;;
            esac
            shift 2
          done
          [ "$(cat "$INGRESS_FILE")" = "$old_mode $old_slot $old_sha $old_backend" ] || return 1
          printf '%s %s %s %s\\n' "$new_mode" "$new_slot" "$new_sha" "$new_backend" > "$INGRESS_FILE"
          record "ingress:$new_mode:$new_slot:$new_backend"
          if [ "${FAIL:-}" = route_gap ] && [ "$new_mode" = domain ]; then return 1; fi
          if [ "${FAIL:-}" = dual_gap ] && [ "$new_mode" = dual ] && [ "$new_backend" = closed ]; then return 1; fi
        }
        preflight_gateway_restart_disk() { record disk; }
        verify_controller_process_helper() { record helper; }
        start_arena_component() {
          record "start:$1:$2"
          if [ "${FAIL:-}" != driver_died ] || [ "$1" != driver ]; then
            touch "$ARENA_RUNTIME_ROOT/$1-slot$2-process.json"
          fi
          if [ "$1" = driver ] && [ "${FAIL:-}" != driver_marker ]; then
            mkdir -p "$GATEWAY_LOG_ROOT"
            printf 'ARENA_SERVICE_LIFECYCLE event=startup outcome=ok source_sha=%s physical_role=arena-driver invocation_id=%s\n' \
              "$GATEWAY_DEPLOY_SHA" "$GATEWAY_RESTART_INVOCATION_ID" \
              >> "$GATEWAY_LOG_ROOT/arena-driver-slot$2.log"
          fi
        }
        stop_arena_component() {
          record "stop:$1:$2"
          actual_stop_arena_component "$@"
        }
        record_gateway_restart_timing() { record "timing:$1"; }
        drain_lab_arena_for_restart() {
          record drain
          if [ "${FAIL:-}" = drain ]; then return 1; fi
          if [ "${FAIL:-}" = conflict ]; then
            printf '%s\\n' "dual 0 bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb ready" > "$INGRESS_FILE"
            return 1
          fi
          record guard:drain
          printf 'draining\\n' > "$TASK_TMP/guard"
        }
        run_lab_arena_restart_guard() {
          record "guard:$2"
          case "$2" in
            state)
              if [ "${FAIL:-}" = bad_guard_state ]; then printf '{"guard_present":"false"}\n'; return; fi
              if [ -f "$TASK_TMP/guard" ]; then present=true; else present=false; fi
              printf '{"schema_version":"leadpoet.lab_arena.restart_guard_state.v1",'
              printf '"guard_present":%s}\n' "$present"
              ;;
            drain) printf 'draining\\n' > "$TASK_TMP/guard" ;;
            authorize) printf 'gateway_destructive\\n' > "$TASK_TMP/guard" ;;
            abort)
              if [ -f "$TASK_TMP/guard" ] && [ "$(cat "$TASK_TMP/guard")" != draining ]; then
                record guard:abort-refused
                return 1
              fi
              rm -f "$TASK_TMP/guard" ;;
            release)
              if [ "${FAIL:-}" = ready_gap ]; then return 1; fi
              rm -f "$TASK_TMP/guard" ;;
          esac
        }
        timeout() { shift; "$@"; }
        # The privileged cache helper is covered separately. Return its report
        # here while the real shell function validates it and removes source.
        sudo() {
          local slot= sha=
          while [ "$#" -gt 0 ]; do
            case "$1" in
              --slot) slot="$2"; shift;;
              --sha) sha="$2"; shift;;
            esac
            shift
          done
          record "retire:$slot:$sha"
          printf '{"archive":"%s","cache_paths":0,"status":"cache_cleanup_complete","applied":true}\\n' \
            "$ARENA_RUNTIME_ROOT/slot$slot-$sha"
        }
        curl() {
          local url="${!#}"
          record "curl:$url"
          if [ "$url" = http://localhost:8000/health/v2-authority ]; then return 1; fi
          if [ "${FAIL:-}" = backend ] && [[ "$url" == *:8001/health ]]; then return 1; fi
          if [ "${FAIL:-}" = current_read ] && [ "${GATEWAY_DEPLOY_STAGE:-}" = lab_arena_service_start ] && [[ "$url" == */arena/v1/current ]]; then return 1; fi
          if [[ "$url" == */build-info ]]; then printf '{"git_commit":"%s"}\\n' "$GATEWAY_DEPLOY_SHA"; fi
        }
        sleep() { :; }
        pgrep() { printf '123\\n'; }
        tail() { :; }
        mv() {
          if [ "$1" = -T ]; then
            shift 2
            # BSD rename rejects a write-protected source directory. GNU mv
            # on the production host accepts the controller's exact call.
            chmod u+w "$1"
          fi
          command mv "$@"
        }
        install_successful_restart_script() { record backend:install; }
        finalize_deployment_record() { record deployment:finalize; }
        """)
        + functions
        + textwrap.dedent("""\
        GATEWAY_PREFLIGHT_TREE="$TASK_TMP/preflight"
        LEADPOET_REPO_ROOT="$TASK_TMP/repo"
        ARENA_RUNTIME_ROOT="$TASK_TMP/runtime"
        GATEWAY_PYTHON_BIN="$TASK_TMP/mock-python"
        GATEWAY_ENV_FILE="$TASK_TMP/env-clone"
        GATEWAY_LOG_ROOT="$TASK_TMP/logs"
        ENV_CLONE="$TASK_TMP/env-clone"
        GATEWAY_CONTROLLER_PROCESS_HELPER="$TASK_TMP/helper"
        GATEWAY_CONTROLLER_PROCESS_STATE_FILE="$TASK_TMP/old-worker.json"
        GATEWAY_ACTIVE_RELEASE_COMPONENT=gateway
        GATEWAY_RESTART_GUARD_SCOPE=gateway
        GATEWAY_PAIRED_RESTART=0
        GATEWAY_ACTIVE_RELEASE_RESTART_INVOCATION_ID=gateway-1792000000-123
        GATEWAY_RESTART_INVOCATION_ID=gateway-1792000000-456
        GATEWAY_LOG_FILE="$TASK_TMP/log"
        GATEWAY_V2_HEALTH_MAX_ATTEMPTS=2
        GATEWAY_V2_HEALTH_RETRY_SECONDS=0
        GATEWAY_V2_HEALTH_DEADLINE_SECONDS=10
        MIN_FREE_KB=1
        PREPARED_GATEWAY_SHA="$TEST_SHA"
        GATEWAY_DEPLOY_SHA="$TEST_SHA"
        LAB_ARENA_RESTART_GUARD_GENERATION=1
        ARENA_CONTINUITY_ENABLED=1
        ARENA_INGRESS_SWITCHED=0
        ARENA_CANDIDATE_TREE=
        GATEWAY_DESTRUCTIVE_PHASE_STARTED=0
        GATEWAY_BACKEND_HEALTH_URL=http://127.0.0.1:8001
        GATEWAY_DEPLOY_STAGE=preflight
        ARENA_RECOVERING=0
        ARENA_REUSE_PREVIOUS=0
        trap 'status=$?; record "controller:$GATEWAY_DESTRUCTIVE_PHASE_STARTED:$GATEWAY_CONTROLLER_PROCESS_STATE_FILE:$GATEWAY_PREFLIGHT_TREE"
        if [ "$status" -ne 0 ]; then
          if [ "$ARENA_RECOVERING" = 1 ] && [ "$GATEWAY_DESTRUCTIVE_PHASE_STARTED" != 1 ]; then
            if rollback_arena_candidate_before_destructive; then
              restore_arena_recovery_guard_before_destructive || true
            fi
          else
            if rollback_arena_candidate_before_destructive; then
              abort_lab_arena_restart_guard_before_destructive
            fi
          fi
        fi' EXIT
        """)
        + OVERLAP + "\n" + DRAIN_AND_STOP + "\n"
        + 'record_gateway_restart_timing "shutdown_complete"\n'
        + BACKEND_CHECK + "\n" + PUBLISH + "\n",
        encoding="utf-8",
    )
    env = {**os.environ, "TASK_TMP": str(tmp_path), "INGRESS_FILE": str(ingress),
           "EVENTS": str(tmp_path / "events"), "TEST_SHA": sha, "FAIL": fail,
           "FAIL_IDLE": "1" if fail == "idle" else "0",
           "IDLE_REVIEW_FILE": str(tmp_path / "review-rows.json") if review_rows is not None else ""}
    return subprocess.run(["bash", str(harness)], cwd=tmp_path, env=env,
                          capture_output=True, text=True, timeout=20)


def _events(tmp_path: Path) -> list[str]:
    return (tmp_path / "events").read_text(encoding="utf-8").splitlines()


@pytest.mark.parametrize("decision", ["passed", "rejected"])
def test_first_cutover_accepts_completed_audit_claims_before_authorization(tmp_path: Path, decision: str):
    rows = [{"status": "frozen", "code_review_status": decision,
             "code_review_claim": f"retained-finish-token-{index}", "code_review_expires_at": None,
             "code_review_started_at": "2026-10-14T23:00:00+00:00"}
            for index in range(24)]
    result = _run(tmp_path, review_rows=rows)
    assert result.returncode == 0, result.stderr
    events = _events(tmp_path)
    idle = next(i for i, event in enumerate(events) if "arena_first_cutover_idle.py" in event)
    assert events.index("guard:drain") < idle < events.index("guard:authorize")
    assert events.index("guard:authorize") < events.index("timing:shutdown_started")
    assert "guard:release" in events
    assert not (tmp_path / "guard").exists()


@pytest.mark.parametrize("review", [
    {"code_review_status": "reviewing", "code_review_expires_at": "2026-10-15T01:00:00+00:00"},
    {"code_review_status": "pending", "code_review_expires_at": None},
    {"code_review_status": "error", "code_review_expires_at": None},
    {"code_review_status": "passed"},
    {"code_review_status": "passed", "code_review_expires_at": "2026-10-15T01:00:00+00:00"},
])
def test_real_idle_refusal_aborts_only_draining_guard_before_shutdown(tmp_path: Path, review: dict):
    live_worker = tmp_path / "old-worker.json"
    live_worker.write_text("live legacy worker\n")
    row = {"status": "frozen", "code_review_claim": "claimed-review", **review}
    result = _run(tmp_path, review_rows=[row])
    assert result.returncode != 0
    assert "legacy Arena paid review" in result.stderr
    events = _events(tmp_path)
    assert "guard:drain" in events and "guard:abort" in events
    assert "guard:abort-refused" not in events
    assert "guard:authorize" not in events
    assert "timing:shutdown_started" not in events
    assert not any(event.startswith("python:") and
                   ("old-worker.json" in event or "gateway-process.json" in event)
                   for event in events)
    assert live_worker.read_text() == "live legacy worker\n"
    assert (tmp_path / "ingress").read_text().strip() == "legacy 0 - legacy"
    assert not (tmp_path / "guard").exists()
    assert not (tmp_path / "runtime" / f"slot1-{SHA}").exists()


def test_failure_after_real_stop_keeps_destructive_guard(tmp_path: Path):
    result = _run(tmp_path, fail="backend", review_rows=[{
        "status": "frozen", "code_review_claim": "retained-finish-token",
        "code_review_status": "passed", "code_review_expires_at": None,
    }])
    assert result.returncode != 0
    events = _events(tmp_path)
    assert "timing:shutdown_started" in events
    assert "guard:authorize" in events
    assert "guard:abort" not in events and "guard:release" not in events
    assert (tmp_path / "guard").read_text().strip() == "gateway_destructive"
    assert (tmp_path / "ingress").read_text().strip() == f"dual 1 {SHA} closed"


@pytest.mark.parametrize("origin", ["live", "secret-json", "secret-dotenv", "secret-cache"])
@pytest.mark.parametrize("failure", ["occupied", "drain"])
def test_stale_runtime_state_cannot_suppress_owned_candidate_cleanup_and_retry(
    tmp_path: Path, origin: str, failure: str,
):
    clone = _runtime_clone(tmp_path, origin)
    live_worker = tmp_path / "old-worker.json"
    live_worker.write_text("live legacy worker\n")
    unrelated = tmp_path / "runtime" / f"slot0-{'c' * 40}"
    unrelated.mkdir(parents=True)
    (unrelated / "marker").write_text("unrelated retained archive\n")
    first = _run(tmp_path, fail=failure, runtime_clone=clone)
    assert first.returncode != 0
    events = _events(tmp_path)
    assert f"controller:0:{live_worker}:{tmp_path}/preflight" in events
    assert "guard:authorize" not in events
    assert "timing:shutdown_started" not in events
    assert not any(event.startswith("python:") and
                   ("old-worker.json" in event or "gateway-process.json" in event)
                   for event in events)
    assert live_worker.read_text() == "live legacy worker\n"
    assert (unrelated / "marker").read_text() == "unrelated retained archive\n"
    assert (tmp_path / "ingress").read_text().strip() == "legacy 0 - legacy"
    assert not (tmp_path / "guard").exists()
    assert not (tmp_path / "runtime" / f"slot1-{SHA}").exists()
    assert not (tmp_path / "runtime" / "api-slot1-process.json").exists()
    assert not (tmp_path / "runtime" / "edge-slot1-process.json").exists()
    assert "stop:edge:1" in events and "stop:api:1" in events

    retry = _run(tmp_path, runtime_clone=clone)
    assert retry.returncode == 0, retry.stderr
    assert (tmp_path / "ingress").read_text().strip() == f"dual 1 {SHA} ready"


def test_occupied_edge_port_refuses_before_process_launch_and_cleans_archive(tmp_path: Path):
    # Hold the real selected edge port while the production bind-check runs.
    # This must not affect the existing ingress, worker, or another archive.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as occupied:
        occupied.bind(("127.0.0.1", 8891))
        occupied.listen()
        result = _run(tmp_path, runtime_clone=_runtime_clone(tmp_path, "live"))
        assert result.returncode != 0
        assert "candidate Arena port is already occupied" in result.stderr
        assert not any(event.startswith("start:") for event in _events(tmp_path))
        assert (tmp_path / "ingress").read_text().strip() == "legacy 0 - legacy"
        assert not (tmp_path / "runtime" / f"slot1-{SHA}").exists()
        assert not (tmp_path / "guard").exists()
    assert _run(tmp_path).returncode == 0


def test_stale_environment_failure_preserves_previous_serving_guard(tmp_path: Path):
    assert _run(tmp_path, fail="backend").returncode != 0
    previous = tmp_path / "runtime" / f"slot1-{SHA}"
    prior_count = len(_events(tmp_path))
    result = _run(tmp_path, fail="drain", runtime_clone=_runtime_clone(tmp_path, "live"))
    assert result.returncode != 0
    events = _events(tmp_path)[prior_count:]
    assert "guard:abort" not in events
    assert (tmp_path / "guard").exists()
    assert (tmp_path / "ingress").read_text().strip() == f"dual 1 {SHA} closed"
    assert previous.is_dir()
    assert (tmp_path / "runtime" / "api-slot1-process.json").exists()
    assert (tmp_path / "runtime" / "edge-slot1-process.json").exists()
    assert not (tmp_path / "runtime" / f"slot0-{SHA}").exists()
    assert "stop:api:1" not in events and "stop:edge:1" not in events
    assert _run(tmp_path).returncode == 0


def test_first_migration_and_same_sha_warm_restart(tmp_path: Path):
    first = _run(tmp_path)
    assert first.returncode == 0, first.stderr
    events = _events(tmp_path)
    assert events.index("guard:authorize") < events.index("ingress:domain:1:legacy")
    old_worker_stop = next(i for i, event in enumerate(events)
                           if event.startswith("python:") and "old-worker.json" in event)
    assert events.index("guard:authorize") < old_worker_stop
    assert events.index("start:api:0") < events.index("ingress:dual:1:closed")
    assert events.index("stop:api:0") < events.index("ingress:dual:1:closed")
    assert events.index("ingress:dual:1:ready") < events.index("guard:release")
    assert "curl:http://127.0.0.1:8001/health/v2-authority" in events
    assert (tmp_path / "ingress").read_text().strip() == f"dual 1 {SHA} ready"
    assert (tmp_path / "runtime" / f"slot1-{SHA}").is_dir()

    second = _run(tmp_path)
    assert second.returncode == 0, second.stderr
    later = _events(tmp_path)[len(events):]
    assert "start:api:0" in later and "start:edge:0" in later
    assert later.index("guard:authorize") < later.index("ingress:dual:0:ready")
    assert "stop:driver:1" in later
    assert later.index("ingress:dual:0:closed") < later.index("stop:edge:1")
    assert later.index("stop:edge:1") < later.index("stop:api:1")
    assert later.index("stop:api:1") < later.index("guard:release")
    assert (tmp_path / "ingress").read_text().strip() == f"dual 0 {SHA} ready"
    assert (tmp_path / "runtime" / f"slot0-{SHA}").is_dir()
    assert not (tmp_path / "runtime" / f"slot1-{SHA}").exists()


@pytest.mark.parametrize("failure", ["idle", "drain", "route_gap", "conflict", "occupied", "disowned"])
def test_before_shutdown_failure_rolls_back_or_retains_conflicting_route(tmp_path: Path, failure: str):
    result = _run(tmp_path, fail=failure)
    assert result.returncode != 0
    events = _events(tmp_path)
    assert ("guard:abort" in events) is (failure != "conflict")
    assert not any(event.startswith("python:") and
                   ("old-worker.json" in event or "gateway-process.json" in event)
                   for event in events)
    if failure in ("idle", "drain", "route_gap", "occupied", "disowned"):
        assert (tmp_path / "ingress").read_text().strip() == "legacy 0 - legacy"
        assert not (tmp_path / "runtime" / f"slot1-{SHA}").exists()
        assert "stop:edge:1" in events and "stop:api:1" in events
    else:
        assert (tmp_path / "ingress").read_text().startswith("dual 0 ")
        assert (tmp_path / "runtime" / f"slot1-{SHA}").exists()
        assert "stop:edge:1" not in events


def test_backend_failure_keeps_arena_serving_and_protected_routes_closed(tmp_path: Path):
    result = _run(tmp_path, fail="backend")
    assert result.returncode != 0
    events = _events(tmp_path)
    assert "ingress:dual:1:closed" in events
    assert "ingress:dual:1:ready" not in events
    assert "guard:release" not in events
    assert "stop:edge:1" not in events and "stop:api:1" not in events
    assert (tmp_path / "ingress").read_text().strip() == f"dual 1 {SHA} closed"
    assert (tmp_path / "runtime" / f"slot1-{SHA}").is_dir()


def test_second_backend_failure_can_retry_with_original_guard_owner(tmp_path: Path):
    first = _run(tmp_path, fail="backend")
    assert first.returncode != 0
    assert (tmp_path / "ingress").read_text().strip() == f"dual 1 {SHA} closed"
    first_count = len(_events(tmp_path))

    second = _run(tmp_path, fail="backend")
    assert second.returncode != 0, second.stderr
    second_events = _events(tmp_path)[first_count:]
    assert any("verify-invocation" in event for event in second_events)
    assert any("lab_arena_restart_claim_guard.py owned" in event for event in second_events)
    assert "ingress:dual:0:closed" in second_events
    assert "stop:edge:1" in second_events and "stop:api:1" in second_events
    assert (tmp_path / "ingress").read_text().strip() == f"dual 0 {SHA} closed"
    assert not (tmp_path / "runtime" / f"slot1-{SHA}").exists()
    second_count = len(_events(tmp_path))

    third = _run(tmp_path)
    assert third.returncode == 0, third.stderr
    third_events = _events(tmp_path)[second_count:]
    assert any("verify-invocation" in event for event in third_events)
    assert "ingress:dual:1:ready" in third_events
    assert third_events.index("stop:api:0") < third_events.index("guard:release")
    assert (tmp_path / "ingress").read_text().strip() == f"dual 1 {SHA} ready"


def test_first_migration_partial_stop_recovers_only_same_release(tmp_path: Path):
    first = _run(tmp_path, fail="domain_gap")
    assert first.returncode != 0, first.stderr
    first_events = _events(tmp_path)
    assert "ingress:domain:1:legacy" in first_events
    assert any("old-worker.json" in event for event in first_events)
    assert "ingress:dual:1:closed" not in first_events
    assert (tmp_path / "ingress").read_text().strip() == f"domain 1 {SHA} legacy"
    assert (tmp_path / "runtime" / f"slot1-{SHA}").is_dir()

    different = _run(tmp_path, sha="b" * 40)
    assert different.returncode != 0
    assert "exact release" in different.stderr
    assert (tmp_path / "ingress").read_text().strip() == f"domain 1 {SHA} legacy"
    assert not (tmp_path / "runtime" / f"slot0-{'b' * 40}").exists()

    prior_count = len(_events(tmp_path))
    retry = _run(tmp_path)
    assert retry.returncode == 0, retry.stderr
    events = _events(tmp_path)[prior_count:]
    assert any("verify-invocation" in event for event in events)
    assert "start:api:1" not in events and "start:edge:1" not in events
    assert "start:api:0" not in events
    assert "ingress:dual:1:closed" in events
    assert "ingress:dual:1:ready" in events
    assert (tmp_path / "ingress").read_text().strip() == f"dual 1 {SHA} ready"


def test_direct_port_handoff_failure_retries_without_stale_temp_api(tmp_path: Path):
    first = _run(tmp_path, fail="dual_gap")
    assert first.returncode != 0, first.stderr
    events = _events(tmp_path)
    assert events.index("stop:api:0") < events.index("ingress:dual:1:closed")
    assert not (tmp_path / "runtime" / "api-slot0-process.json").exists()
    assert (tmp_path / "ingress").read_text().strip() == f"dual 1 {SHA} closed"

    prior_count = len(events)
    retry = _run(tmp_path)
    assert retry.returncode == 0, retry.stderr
    retry_events = _events(tmp_path)[prior_count:]
    assert "start:api:0" in retry_events and "start:edge:0" in retry_events
    assert "ingress:dual:0:ready" in retry_events


def test_ready_ingress_retained_guard_uses_exact_owner(tmp_path: Path):
    first = _run(tmp_path, fail="ready_gap")
    assert first.returncode != 0, first.stderr
    assert (tmp_path / "ingress").read_text().strip() == f"dual 1 {SHA} ready"
    assert (tmp_path / "guard").exists()
    prior_count = len(_events(tmp_path))

    retry = _run(tmp_path)
    assert retry.returncode == 0, retry.stderr
    events = _events(tmp_path)[prior_count:]
    assert "guard:state" in events
    assert any("verify-invocation" in event for event in events)
    assert any("lab_arena_restart_claim_guard.py owned" in event for event in events)
    assert events.index("guard:drain") < events.index("ingress:dual:0:ready")
    assert not (tmp_path / "guard").exists()


def test_ready_ingress_rejects_malformed_guard_state_before_candidate_write(tmp_path: Path):
    assert _run(tmp_path).returncode == 0
    prior_count = len(_events(tmp_path))
    retry = _run(tmp_path, fail="bad_guard_state")
    assert retry.returncode != 0
    assert "restart guard state is invalid" in retry.stderr
    events = _events(tmp_path)[prior_count:]
    assert not any(event.startswith("start:") for event in events)
    assert not (tmp_path / "runtime" / f"slot0-{SHA}").exists()


def test_ready_ingress_retained_guard_rejects_wrong_owner(tmp_path: Path):
    assert _run(tmp_path, fail="ready_gap").returncode != 0
    prior_count = len(_events(tmp_path))
    retry = _run(tmp_path, fail="owner_mismatch")
    assert retry.returncode != 0
    events = _events(tmp_path)[prior_count:]
    assert any("verify-invocation" in event for event in events)
    assert not any(event.startswith("start:") for event in events)
    assert (tmp_path / "ingress").read_text().strip() == f"dual 1 {SHA} ready"


def test_expired_retained_guard_is_renewed_by_canonical_drain(tmp_path: Path):
    assert _run(tmp_path, fail="backend").returncode != 0
    prior_count = len(_events(tmp_path))
    retry = _run(tmp_path, fail="expired_guard")
    assert retry.returncode == 0, retry.stderr
    events = _events(tmp_path)[prior_count:]
    assert any("lab_arena_restart_claim_guard.py owned" in event for event in events)
    assert "guard:drain" in events and "guard:authorize" in events


@pytest.mark.parametrize("failure", ["driver_marker", "driver_died"])
def test_driver_must_be_live_with_new_startup_marker_before_guard_release(
    tmp_path: Path, failure: str,
):
    first = _run(tmp_path, fail=failure)
    assert first.returncode != 0
    events = _events(tmp_path)
    assert "start:driver:1" in events
    assert "ingress:dual:1:ready" not in events
    assert "guard:release" not in events
    assert (tmp_path / "ingress").read_text().strip() == f"dual 1 {SHA} closed"


def test_stale_same_sha_driver_marker_does_not_release_guard(tmp_path: Path):
    assert _run(tmp_path).returncode == 0
    assert _run(tmp_path).returncode == 0
    prior_count = len(_events(tmp_path))
    third = _run(tmp_path, fail="driver_marker")
    assert third.returncode != 0
    events = _events(tmp_path)[prior_count:]
    assert "start:driver:1" in events
    assert "guard:release" not in events
    assert (tmp_path / "ingress").read_text().strip() == f"dual 1 {SHA} closed"


def test_first_migration_recovery_rejects_missing_edge_receipt(tmp_path: Path):
    assert _run(tmp_path, fail="domain_gap").returncode != 0
    (tmp_path / "runtime" / "edge-slot1-process.json").unlink()
    prior_count = len(_events(tmp_path))
    retry = _run(tmp_path)
    assert retry.returncode != 0
    events = _events(tmp_path)[prior_count:]
    assert any("verify-invocation" in event for event in events)
    assert not any(event.startswith("start:") for event in events)
    assert (tmp_path / "ingress").read_text().strip() == f"domain 1 {SHA} legacy"


def test_recovery_owner_mismatch_fails_before_candidate_write(tmp_path: Path):
    assert _run(tmp_path, fail="backend").returncode != 0
    prior_count = len(_events(tmp_path))
    retry = _run(tmp_path, fail="owner_mismatch")
    assert retry.returncode != 0
    events = _events(tmp_path)[prior_count:]
    assert any("verify-invocation" in event for event in events)
    assert not any(event.startswith("start:") for event in events)
    assert (tmp_path / "ingress").read_text().strip() == f"dual 1 {SHA} closed"
    assert not (tmp_path / "runtime" / f"slot0-{SHA}").exists()


def test_previous_sha_uses_candidate_verified_stop_helper(tmp_path: Path):
    assert _run(tmp_path).returncode == 0
    prior_count = len(_events(tmp_path))
    next_sha = "b" * 40
    result = _run(tmp_path, sha=next_sha)
    assert result.returncode == 0, result.stderr
    events = _events(tmp_path)[prior_count:]
    old_driver_stop = next(event for event in events
                           if event.startswith("python:") and "driver-slot1-process" in event)
    assert f"/slot0-{next_sha}/scripts/manage_owned_process_group.py" in old_driver_stop
    assert f"--cwd {tmp_path}/runtime/slot1-{SHA}" in old_driver_stop
    assert (tmp_path / "ingress").read_text().strip() == f"dual 0 {next_sha} ready"


def _assert_listener_alive(listener: socket.socket) -> None:
    # A real connection proves the unrelated listener survived the cutovers.
    client = socket.create_connection(listener.getsockname(), timeout=1)
    connection, _ = listener.accept()
    try:
        # Close the client first so TIME_WAIT belongs to its ephemeral port,
        # not the fixed listener needed by the next controller test.
        client.close()
        assert connection.recv(1) == b""
    finally:
        client.close()
        connection.close()


def test_old_edge_listener_survives_first_and_second_dedicated_port_cutovers(tmp_path: Path):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as old_listener:
        old_listener.bind(("127.0.0.1", 8801))
        old_listener.listen()
        old_listener.settimeout(1)
        first = _run(tmp_path)
        assert first.returncode == 0, first.stderr
        first_events = _events(tmp_path)
        assert "python:- 8794 8891" in first_events
        assert "curl:http://127.0.0.1:8891/arena/v1/current" in first_events
        assert (tmp_path / "ingress").read_text().strip() == f"dual 1 {SHA} ready"
        _assert_listener_alive(old_listener)

        second_sha = "b" * 40
        second = _run(tmp_path, sha=second_sha)
        assert second.returncode == 0, second.stderr
        later = _events(tmp_path)[len(first_events):]
        assert "python:- 8792 8890" in later
        assert "curl:http://127.0.0.1:8890/arena/v1/current" in later
        assert (tmp_path / "ingress").read_text().strip() == f"dual 0 {second_sha} ready"
        _assert_listener_alive(old_listener)


@pytest.mark.parametrize("candidate_slot,edge_port", [(1, 8891), (0, 8890)])
def test_occupied_dedicated_edge_rejects_candidate_before_drain(
    tmp_path: Path, candidate_slot: int, edge_port: int,
):
    if candidate_slot == 0:
        assert _run(tmp_path).returncode == 0
    prior_ingress = ((tmp_path / "ingress").read_text()
                     if (tmp_path / "ingress").exists() else "legacy 0 - legacy\n")
    prior_count = len(_events(tmp_path)) if (tmp_path / "events").exists() else 0
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as foreign_listener:
        foreign_listener.bind(("127.0.0.1", edge_port))
        foreign_listener.listen()
        foreign_listener.settimeout(1)
        result = _run(tmp_path, sha="b" * 40)
        assert result.returncode != 0
        assert "candidate Arena port is already occupied" in result.stderr
        events = _events(tmp_path)[prior_count:]
        assert "drain" not in events and "guard:authorize" not in events
        assert not any(event.startswith("start:") for event in events)
        assert not any(event.startswith("ingress:") for event in events)
        assert (tmp_path / "ingress").read_text() == prior_ingress
        assert not (tmp_path / "runtime" / f"slot{candidate_slot}-{'b' * 40}").exists()
        _assert_listener_alive(foreign_listener)


def test_failed_current_read_never_releases_guard(tmp_path: Path):
    result = _run(tmp_path, fail="current_read")
    assert result.returncode != 0
    assert (tmp_path / "guard").read_text().strip() == "gateway_destructive"
    events = (tmp_path / "events").read_text()
    assert "guard:release" not in events
    assert "backend:install" not in events
