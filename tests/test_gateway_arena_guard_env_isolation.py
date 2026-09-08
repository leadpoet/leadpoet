from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


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
        str(stale_clone),
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


def test_guard_helper_uses_file_credentials_over_conflicting_ambient_values(
    tmp_path: Path,
) -> None:
    script = SCRIPT_PATH.read_text(encoding="utf-8")
    run_guard = _shell_function_source(script, "run_lab_arena_restart_guard")
    capture = tmp_path / "selected-credentials.json"
    wrapper = tmp_path / "capture-python"
    wrapper.write_text(
        "#!/usr/bin/env python3\n"
        "import importlib.util, json, os, sys\n"
        "from pathlib import Path\n"
        "helper, *arguments = sys.argv[1:]\n"
        "spec = importlib.util.spec_from_file_location('guard_under_test', helper)\n"
        "module = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(module)\n"
        "class Response:\n"
        "    status = 200\n"
        "    def read(self, limit):\n"
        "        value = {\n"
        "            'schema_version': 'leadpoet.lab_arena.restart_guard_state.v1',\n"
        "            'paused': False, 'operator_paused': False,\n"
        "            'guard_present': False, 'guard_active': False,\n"
        "            'guard_generation': 0,\n"
        "            'drain': {\n"
        "                'schema_version': 'leadpoet.lab_arena.restart_drain_state.v1',\n"
        "                'captured_count': 0, 'accepted_receipt_count': 0,\n"
        "                'reported_terminal_receipt_count': 0,\n"
        "                'still_leased_count': 0, 'lost_or_mutated_count': 0,\n"
        "                'current_leased_count': 0, 'pending_retry_count': 0,\n"
        "                'preserved': True,\n"
        "            },\n"
        "        }\n"
        "        return json.dumps(value).encode()\n"
        "class Connection:\n"
        "    def __init__(self, hostname, port, timeout):\n"
        "        Path(os.environ['CAPTURE']).write_text(json.dumps({\n"
        "            'hostname': hostname,\n"
        "            **{name: os.environ.get(name) for name in (\n"
        "                'LAB_ARENA_SUPABASE_URL',\n"
        "                'LAB_ARENA_SUPABASE_ANON_KEY',\n"
        "                'LAB_ARENA_SERVICE_KEY',\n"
        "                'LAB_ARENA_SERVICE_JWT',\n"
        "            )},\n"
        "        }))\n"
        "    def request(self, *args, **kwargs): pass\n"
        "    def getresponse(self): return Response()\n"
        "    def close(self): pass\n"
        "module.http.client.HTTPSConnection = Connection\n"
        "sys.argv = [helper, *arguments]\n"
        "raise SystemExit(module.main())\n",
        encoding="utf-8",
    )
    wrapper.chmod(0o755)
    source_root = tmp_path / "source"
    helper = source_root / "scripts" / "lab_arena_restart_claim_guard.py"
    helper.parent.mkdir(parents=True)
    helper.write_bytes((ROOT / "scripts/lab_arena_restart_claim_guard.py").read_bytes())
    environment_file = tmp_path / "gateway.env"
    file_values = {
        "LAB_ARENA_SUPABASE_URL": "https://file-authority.invalid",
        "LAB_ARENA_SUPABASE_ANON_KEY": "file-anon",
        "LAB_ARENA_SERVICE_KEY": "sb_secret_file",
        "LAB_ARENA_SERVICE_JWT": "file.legacy.jwt",
    }
    environment_file.write_text(
        "\n".join(f"export {name}={value}" for name, value in file_values.items())
        + "\n",
        encoding="utf-8",
    )
    harness = tmp_path / "guard-harness.sh"
    harness.write_text(
        "#!/bin/bash\nset -euo pipefail\n"
        + run_guard
        + "\n"
        + f"ENV_CLONE={shlex.quote(str(environment_file))}\n"
        + f"GATEWAY_PYTHON_BIN={shlex.quote(str(wrapper))}\n"
        + f"PREPARED_GATEWAY_SHA={'b' * 40}\n"
        + "GATEWAY_ACTIVE_RELEASE_RESTART_INVOCATION_ID=restart-current\n"
        + f"run_lab_arena_restart_guard {shlex.quote(str(source_root))} state\n",
        encoding="utf-8",
    )
    harness.chmod(0o755)
    ambient = {
        "LAB_ARENA_SUPABASE_URL": "https://ambient-authority.invalid",
        "LAB_ARENA_SUPABASE_ANON_KEY": "ambient-anon",
        "LAB_ARENA_SERVICE_KEY": "sb_secret_ambient",
        "LAB_ARENA_SERVICE_JWT": "ambient.legacy.jwt",
    }

    completed = subprocess.run(
        ["bash", str(harness)],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
        env={**os.environ, **ambient, "CAPTURE": str(capture)},
    )

    assert completed.returncode == 0, completed.stderr
    selected = json.loads(capture.read_text(encoding="utf-8"))
    assert selected == {"hostname": "file-authority.invalid", **file_values}


def test_guard_helper_without_file_or_ambient_credentials_fails_closed(
    tmp_path: Path,
) -> None:
    script = SCRIPT_PATH.read_text(encoding="utf-8")
    run_guard = _shell_function_source(script, "run_lab_arena_restart_guard")
    source_root = tmp_path / "source"
    helper = source_root / "scripts" / "lab_arena_restart_claim_guard.py"
    helper.parent.mkdir(parents=True)
    helper.write_bytes((ROOT / "scripts/lab_arena_restart_claim_guard.py").read_bytes())
    harness = tmp_path / "missing-credentials.sh"
    harness.write_text(
        "#!/bin/bash\nset -euo pipefail\n"
        + run_guard
        + "\n"
        + f"ENV_CLONE={shlex.quote(str(tmp_path / 'missing.env'))}\n"
        + f"GATEWAY_PYTHON_BIN={shlex.quote(sys.executable)}\n"
        + f"PREPARED_GATEWAY_SHA={'b' * 40}\n"
        + "GATEWAY_ACTIVE_RELEASE_RESTART_INVOCATION_ID=restart-current\n"
        + f"run_lab_arena_restart_guard {shlex.quote(str(source_root))} state\n",
        encoding="utf-8",
    )
    environment = dict(os.environ)
    for name in (
        "LAB_ARENA_SUPABASE_URL",
        "LAB_ARENA_SUPABASE_ANON_KEY",
        "LAB_ARENA_SERVICE_KEY",
        "LAB_ARENA_SERVICE_JWT",
    ):
        environment.pop(name, None)

    completed = subprocess.run(
        ["bash", str(harness)],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
        env=environment,
    )

    assert completed.returncode == 1
    assert "database authority is unavailable" in completed.stderr


def test_guard_helper_does_not_fall_back_to_ambient_credentials_when_file_exists(
    tmp_path: Path,
) -> None:
    script = SCRIPT_PATH.read_text(encoding="utf-8")
    run_guard = _shell_function_source(script, "run_lab_arena_restart_guard")
    source_root = tmp_path / "source"
    helper = source_root / "scripts" / "lab_arena_restart_claim_guard.py"
    helper.parent.mkdir(parents=True)
    helper.write_bytes((ROOT / "scripts/lab_arena_restart_claim_guard.py").read_bytes())
    environment_file = tmp_path / "incomplete-gateway.env"
    environment_file.write_text("export LAB_ARENA_MODE=live\n", encoding="utf-8")
    network_marker = tmp_path / "network-attempted"
    wrapper = tmp_path / "reject-network-python"
    wrapper.write_text(
        "#!/usr/bin/env python3\n"
        "import importlib.util, os, sys\n"
        "from pathlib import Path\n"
        "helper, *arguments = sys.argv[1:]\n"
        "spec = importlib.util.spec_from_file_location('guard_under_test', helper)\n"
        "module = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(module)\n"
        "class Connection:\n"
        "    def __init__(self, *args, **kwargs):\n"
        "        Path(os.environ['NETWORK_MARKER']).write_text('attempted')\n"
        "        raise AssertionError('network must not be reached')\n"
        "module.http.client.HTTPSConnection = Connection\n"
        "sys.argv = [helper, *arguments]\n"
        "raise SystemExit(module.main())\n",
        encoding="utf-8",
    )
    wrapper.chmod(0o755)
    harness = tmp_path / "incomplete-authority.sh"
    harness.write_text(
        "#!/bin/bash\nset -euo pipefail\n"
        + run_guard
        + "\n"
        + f"ENV_CLONE={shlex.quote(str(environment_file))}\n"
        + f"GATEWAY_PYTHON_BIN={shlex.quote(str(wrapper))}\n"
        + f"PREPARED_GATEWAY_SHA={'b' * 40}\n"
        + "GATEWAY_ACTIVE_RELEASE_RESTART_INVOCATION_ID=restart-current\n"
        + f"run_lab_arena_restart_guard {shlex.quote(str(source_root))} state\n",
        encoding="utf-8",
    )
    ambient = {
        "LAB_ARENA_SUPABASE_URL": "https://ambient-authority.invalid",
        "LAB_ARENA_SUPABASE_ANON_KEY": "ambient-anon",
        "LAB_ARENA_SERVICE_KEY": "sb_secret_ambient",
        "LAB_ARENA_SERVICE_JWT": "ambient.legacy.jwt",
    }

    completed = subprocess.run(
        ["bash", str(harness)],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
        env={**os.environ, **ambient, "NETWORK_MARKER": str(network_marker)},
    )

    assert completed.returncode == 1
    assert "database authority is unavailable" in completed.stderr
    assert not network_marker.exists()


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
