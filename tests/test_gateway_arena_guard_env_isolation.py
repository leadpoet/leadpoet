from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import shlex
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "gw_restart.sh"
GUARD_AUTHORITY_KEYS = {
    "PREPARED_GATEWAY_SHA",
    "LAB_ARENA_RESTART_GUARD_GENERATION",
}


def _shell_function_source(script: str, name: str) -> str:
    lines = script.splitlines()
    start = lines.index(f"{name}() {{")
    for end in range(start + 1, len(lines)):
        if lines[end] == "}":
            return "\n".join(lines[start : end + 1])
    raise AssertionError(f"unterminated shell function: {name}")


def _python_set_after(script: str, marker: str, name: str) -> set[str]:
    marker_offset = script.index(marker)
    start = script.index(f"{name} = {{", marker_offset) + len(f"{name} = ")
    end = script.index("\n}", start) + 2
    value = ast.literal_eval(script[start:end])
    assert isinstance(value, set)
    return value


def test_guard_helper_ignores_stale_cloned_restart_authority(tmp_path: Path) -> None:
    script = SCRIPT_PATH.read_text(encoding="utf-8")
    run_guard = _shell_function_source(script, "run_lab_arena_restart_guard")
    source_root = tmp_path / "source"
    helper = source_root / "scripts" / "lab_arena_restart_claim_guard.py"
    helper.parent.mkdir(parents=True)
    helper.write_text("# exact helper path fixture\n", encoding="utf-8")

    stale_candidate = "a" * 40
    current_candidate = "b" * 40
    stale_clone = tmp_path / "stale-env-clone.sh"
    stale_clone.write_text(
        "\n".join(
            (
                f"export PREPARED_GATEWAY_SHA={stale_candidate}",
                "export LAB_ARENA_RESTART_GUARD_GENERATION=101",
                "export GATEWAY_ACTIVE_RELEASE_RESTART_INVOCATION_ID=restart-stale",
                "export GATEWAY_ACTIVE_RELEASE_COMPONENT=gateway",
                "export LAB_ARENA_SUPABASE_URL=https://stale.invalid",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    canonical_env = tmp_path / "gateway.env"
    canonical_env.write_text(
        "LAB_ARENA_SUPABASE_URL=https://fresh.invalid\n"
        "LAB_ARENA_SUPABASE_ANON_KEY=fresh-anon\n"
        "LAB_ARENA_SERVICE_KEY=sb_secret_fresh\n",
        encoding="utf-8",
    )

    capture = tmp_path / "helper-capture.json"
    python = tmp_path / "record-python"
    python.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "Path(os.environ['CAPTURE']).write_text(json.dumps(sys.argv[1:]))\n",
        encoding="utf-8",
    )
    python.chmod(0o755)
    state = tmp_path / "shell-state.json"
    harness = tmp_path / "guard-harness.sh"
    harness.write_text(
        "#!/bin/bash\nset -euo pipefail\n"
        + run_guard
        + "\n"
        + f"ENV_CLONE={shlex.quote(str(stale_clone))}\n"
        + f"GATEWAY_ENV_FILE={shlex.quote(str(canonical_env))}\n"
        + f"GATEWAY_PYTHON_BIN={shlex.quote(str(python))}\n"
        + f"PREPARED_GATEWAY_SHA={current_candidate}\n"
        + "LAB_ARENA_RESTART_GUARD_GENERATION=202\n"
        + "GATEWAY_ACTIVE_RELEASE_RESTART_INVOCATION_ID=restart-current\n"
        + "GATEWAY_ACTIVE_RELEASE_COMPONENT=all\n"
        + f"run_lab_arena_restart_guard {shlex.quote(str(source_root))} authorize "
        + '--generation "$LAB_ARENA_RESTART_GUARD_GENERATION" '
        + "--phase gateway_destructive\n"
        + "python3 - "
        + f"{shlex.quote(str(state))} "
        + '"$PREPARED_GATEWAY_SHA" "$LAB_ARENA_RESTART_GUARD_GENERATION" '
        + '"$GATEWAY_ACTIVE_RELEASE_RESTART_INVOCATION_ID" '
        + '"$GATEWAY_ACTIVE_RELEASE_COMPONENT" <<\'PY\'\n'
        + "import json, sys\n"
        + "open(sys.argv[1], 'w').write(json.dumps(sys.argv[2:]))\n"
        + "PY\n",
        encoding="utf-8",
    )
    harness.chmod(0o755)

    completed = subprocess.run(
        ["bash", str(harness)],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
        env={**os.environ, "CAPTURE": str(capture)},
    )
    assert completed.returncode == 0, completed.stderr
    helper_argv = json.loads(capture.read_text(encoding="utf-8"))
    assert helper_argv == [
        str(helper),
        "authorize",
        "--generation",
        "202",
        "--phase",
        "gateway_destructive",
        "--environment-file",
        str(canonical_env),
        "--candidate",
        current_candidate,
        "--invocation",
        "restart-current",
    ]
    assert json.loads(state.read_text(encoding="utf-8")) == [
        current_candidate,
        "202",
        "restart-current",
        "all",
    ]


def test_restart_authority_is_excluded_from_every_runtime_projection() -> None:
    script = SCRIPT_PATH.read_text(encoding="utf-8")
    projected_sets = (
        _python_set_after(
            script,
            'python3 - "$SECRET_TMP" "$GATEWAY_ENV_FILE"',
            "restart_only_keys",
        ),
        _python_set_after(
            script,
            'python3 - "$GATEWAY_ENV_FILE" "$ENV_SECRET"',
            "skip_keys",
        ),
        _python_set_after(
            script,
            'python3 - "$PID" "$ENV_CLONE"',
            "skip_keys",
        ),
    )
    for excluded in projected_sets:
        assert GUARD_AUTHORITY_KEYS <= excluded


def test_restart_authority_is_not_inherited_by_long_lived_services() -> None:
    script = SCRIPT_PATH.read_text(encoding="utf-8")
    arena_launch = _shell_function_source(script, "start_lab_arena_service")
    gateway_start = script.index('echo "Relaunching gateway with cloned runtime env"')
    gateway_launch = script[
        gateway_start : script.index(
            'setsid "$GATEWAY_PYTHON_BIN" -u -m gateway.main', gateway_start
        )
    ]
    for variable in GUARD_AUTHORITY_KEYS:
        unset = f"-u {variable}"
        assert unset in arena_launch
        assert unset in gateway_launch
