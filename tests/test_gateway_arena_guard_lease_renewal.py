"""The single-gateway restart must hold its Arena guard lease to the release.

The drain renews the guard lease only while it runs. Everything after it --
shutdown, activation, enclave builds, service start -- used to run with no
renewal at all, so a restart longer than one lease left an expired guard
behind and Arena review work stayed off until a later restart cleared it.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import time

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "gw_restart.sh").read_text()

CANDIDATE = "c" * 40
INVOCATION = "restart-renewal"
GENERATION = 7
LEASE_SECONDS = 2


def _function(name: str) -> str:
    return SOURCE[SOURCE.index(name + "() {"):].split("\n}\n", 1)[0] + "\n}\n"


def _helper(base: Path, *, drift_after: int | None = None, fail_after: int | None = None) -> str:
    """A guard helper whose lease really expires when nobody renews it."""
    return textwrap.dedent(
        f"""
        import argparse, json, time
        from pathlib import Path

        DRIFT_AFTER = {drift_after!r}
        FAIL_AFTER = {fail_after!r}

        BASE = Path({str(base)!r})
        STATE = BASE / "guard.json"
        CALLS = BASE / "calls.log"

        parser = argparse.ArgumentParser()
        parser.add_argument("action")
        parser.add_argument("--scope")
        parser.add_argument("--generation", type=int, default=0)
        parser.add_argument("--timeout-seconds", type=float, default=0.0)
        parser.add_argument("--lease-seconds", type=int, default=900)
        parser.add_argument("--candidate")
        parser.add_argument("--invocation")
        parser.add_argument("--environment-file")
        args = parser.parse_args()

        state = json.loads(STATE.read_text())
        live = time.time() < state["expires_at"]
        if args.action == "probe":
            print("live" if live else "expired")
            raise SystemExit(0)

        calls = [line for line in CALLS.read_text().splitlines() if line]
        calls.append(json.dumps(
            {{"action": args.action, "scope": args.scope, "generation": args.generation,
              "timeout_seconds": args.timeout_seconds}}, sort_keys=True))
        CALLS.write_text("\\n".join(calls) + "\\n")

        if FAIL_AFTER is not None and len(calls) > FAIL_AFTER:
            raise SystemExit("transient Arena restart guard RPC failure")
        if not live:
            # An abandoned guard is exactly what the lease is for. Renewing one
            # would acquire a fresh guard, which a restart may never do.
            raise SystemExit("Arena restart guard lease has expired")

        generation = state["generation"]
        if DRIFT_AFTER is not None and len(calls) > DRIFT_AFTER:
            generation += 1
        else:
            state["expires_at"] = time.time() + state["lease"]
            STATE.write_text(json.dumps(state))

        print(json.dumps({{
            "schema_version": "leadpoet.lab_arena.restart_quiescence.v1",
            "guard_active": True,
            "guard_generation": generation,
            "restart_scope": state["scope"],
            "restart_phase": "gateway_destructive",
            "preserved": True,
        }}, sort_keys=True))
        """
    )


def _setup(base: Path, **helper_options) -> Path:
    source_root = base / "preflight"
    (source_root / "scripts").mkdir(parents=True)
    (source_root / "scripts" / "lab_arena_restart_claim_guard.py").write_text(
        _helper(base, **helper_options))
    (base / "calls.log").write_text("")
    (base / "env.json").write_text("{}")
    (base / "guard.json").write_text(json.dumps({
        "generation": GENERATION,
        "scope": "gateway",
        "lease": LEASE_SECONDS,
        "expires_at": time.time() + LEASE_SECONDS,
    }))
    return source_root


def _run(base: Path, source_root: Path, body: str, *, paired: str = "0") -> subprocess.CompletedProcess:
    script = f"""
set -euo pipefail
GATEWAY_PYTHON_BIN={sys.executable!r}
GATEWAY_ENV_FILE={str(base / "env.json")!r}
PREPARED_GATEWAY_SHA={CANDIDATE!r}
GATEWAY_ACTIVE_RELEASE_RESTART_INVOCATION_ID={INVOCATION!r}
GATEWAY_RESTART_GUARD_SCOPE=gateway
GATEWAY_PAIRED_RESTART={paired!r}
LAB_ARENA_RESTART_GUARD_GENERATION={GENERATION!r}
LAB_ARENA_RESTART_RENEWAL_PID=""
LAB_ARENA_RESTART_RENEWAL_ROOT=""
LAB_ARENA_RESTART_RENEWAL_INTERVAL_SECONDS=1
LAB_ARENA_RESTART_RENEWAL_MAX_SECONDS=120
SOURCE_ROOT={str(source_root)!r}
BASE={str(base)!r}
{_function("run_lab_arena_restart_guard")}
{_function("renew_lab_arena_restart_guard_once")}
{_function("start_lab_arena_restart_renewal")}
{_function("stop_lab_arena_restart_renewal")}
probe() {{
  "$GATEWAY_PYTHON_BIN" "$SOURCE_ROOT/scripts/lab_arena_restart_claim_guard.py" probe \
    --candidate "$PREPARED_GATEWAY_SHA" --invocation "$GATEWAY_ACTIVE_RELEASE_RESTART_INVOCATION_ID"
}}
{body}
"""
    return subprocess.run(["bash", "-c", script], text=True, capture_output=True,
                          timeout=60, env=os.environ.copy())


def _calls(base: Path) -> list[dict]:
    return [json.loads(line) for line in (base / "calls.log").read_text().splitlines() if line]


def test_renewal_holds_the_lease_past_its_expiry_and_releases_it_on_stop(tmp_path: Path):
    """Without renewal the guard would be abandoned before the restart ends."""
    source_root = _setup(tmp_path)
    # A restart window several leases long: a drain-only renewal expires inside it.
    result = _run(tmp_path, source_root, """
start_lab_arena_restart_renewal "$SOURCE_ROOT"
sleep 6
probe > "$BASE/during"
stop_lab_arena_restart_renewal
sleep 4
probe > "$BASE/after"
""")
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "during").read_text().strip() == "live"
    # The lease is still the bound on an abandoned restart: once the controller
    # stops renewing, the guard expires instead of withholding review forever.
    assert (tmp_path / "after").read_text().strip() == "expired"
    calls = _calls(tmp_path)
    assert len(calls) >= 4
    assert all(call == {"action": "drain", "scope": "gateway", "generation": GENERATION,
                        "timeout_seconds": 30.0} for call in calls)


def test_renewal_stops_without_reacquiring_when_the_guard_generation_changes(tmp_path: Path):
    source_root = _setup(tmp_path, drift_after=2)
    result = _run(tmp_path, source_root, """
start_lab_arena_restart_renewal "$SOURCE_ROOT"
sleep 6
stop_lab_arena_restart_renewal
""")
    assert result.returncode == 0, result.stderr
    assert "lease renewal stopped" in result.stderr
    # It stopped at the third call instead of renewing a guard it no longer owns.
    assert len(_calls(tmp_path)) == 3


def test_renewal_retries_a_transient_failure_rather_than_abandoning_the_guard(tmp_path: Path):
    source_root = _setup(tmp_path, fail_after=1)
    result = _run(tmp_path, source_root, """
start_lab_arena_restart_renewal "$SOURCE_ROOT"
sleep 4
stop_lab_arena_restart_renewal
""")
    assert result.returncode == 0, result.stderr
    assert "lease renewal failed; retrying" in result.stderr
    assert len(_calls(tmp_path)) >= 3


def test_renewal_survives_the_removal_of_the_tree_it_was_started_from(tmp_path: Path):
    """Shutdown deletes the preflight tree and activation rewrites the checkout."""
    source_root = _setup(tmp_path)
    result = _run(tmp_path, source_root, """
start_lab_arena_restart_renewal "$SOURCE_ROOT"
rm -rf "$SOURCE_ROOT"
sleep 6
probe_root="$LAB_ARENA_RESTART_RENEWAL_ROOT"
stop_lab_arena_restart_renewal
[ -n "$probe_root" ] && [ ! -e "$probe_root" ]
""")
    assert result.returncode == 0, result.stderr
    assert len(_calls(tmp_path)) >= 4


def test_paired_restart_leaves_renewal_to_its_coordinator(tmp_path: Path):
    source_root = _setup(tmp_path)
    result = _run(tmp_path, source_root, """
start_lab_arena_restart_renewal "$SOURCE_ROOT"
sleep 3
stop_lab_arena_restart_renewal
""", paired="1")
    assert result.returncode == 0, result.stderr
    assert _calls(tmp_path) == []


def test_renewal_spans_the_whole_destructive_window_of_the_restart():
    """Source ordering: started once the destructive phase is authorized,
    stopped only at the guard release that ends the restart."""
    started = SOURCE.index('start_lab_arena_restart_renewal "$GATEWAY_PREFLIGHT_TREE"')
    authorized = SOURCE.index('--phase gateway_destructive >/dev/null; then')
    released = SOURCE.index('GATEWAY_DEPLOY_STAGE="lab_arena_claim_guard_release"')
    shutdown = SOURCE.index("--term-seconds 6600 --no-kill", started)
    stopped = SOURCE.index("\nstop_lab_arena_restart_renewal\n", started)
    assert authorized < started < shutdown < stopped < released
    assert SOURCE.index("stop_lab_arena_restart_renewal", SOURCE.index("on_gateway_restart_exit() {")) > 0
    assert "stop_lab_arena_restart_renewal" in _function("on_gateway_restart_signal")
