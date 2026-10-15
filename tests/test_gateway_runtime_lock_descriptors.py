from __future__ import annotations

import fcntl
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "gw_restart.sh"


def _launch_blocks() -> dict[str, str]:
    source = SCRIPT.read_text(encoding="utf-8")
    blocks: dict[str, str] = {}

    # Use the real command blocks from the controller. The test supplies only
    # their runtime variables and stub executables.
    arena = source[source.index("start_arena_component() {"):source.index("\nstop_arena_component() {")]
    for name, marker in (("arena-api", '  else\n'),
                         ("arena-edge", '  if [ "$component" = edge ]; then\n')):
        branch = arena.index(marker)
        start = arena.index("    env ", branch)
        end = arena.index("&\n", start) + 2
        blocks[name] = arena[start:end]

    legacy_start = source.index("start_lab_arena_service() {")
    legacy = source[legacy_start:source.index('  pid="$!"', legacy_start)]
    start = legacy.index("  env -u ")
    legacy_end = legacy.index("&\n", start) + 2
    blocks["legacy-service"] = legacy[start:legacy_end]

    start = source.index("env -u ", source.index("# Release the build lock"))
    end = source.index("&\n", start) + 2
    blocks["gateway"] = source[start:end]
    return blocks


BLOCKS = _launch_blocks()


def _write(path: Path, contents: str, executable: bool = False) -> None:
    path.write_text(contents, encoding="utf-8")
    if executable:
        path.chmod(0o755)


def _exercise(tmp_path: Path, block: str, *, unsafe: bool = False) -> bool:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    ready = tmp_path / "serving.ready"
    pid_file = tmp_path / "serving.pid"
    lock7, lock8 = tmp_path / "resource.lock", tmp_path / "admission.lock"

    # setsid is not installed on every supported developer platform.
    _write(bin_dir / "setsid", "#!/usr/bin/env python3\nimport os,sys\nos.setsid()\nos.execvp(sys.argv[1], sys.argv[1:])\n", True)
    # Stand in for the long-running serving interpreter, regardless of the
    # production module/script arguments passed by the launch block.
    _write(bin_dir / "serving-python", "#!/usr/bin/env python3\nimport os,time\nfrom pathlib import Path\nmarkers=('LEADPOET_DOCKER_OPERATION_LOCK_HELD','LEADPOET_DOCKER_OPERATION_ADMISSION_LOCK_HELD','LEADPOET_DOCKER_OPERATION_LOCK_OWNER_PID')\nassert all(name not in os.environ for name in markers), {name:os.environ.get(name) for name in markers}\nbase=Path(__file__).resolve().parent.parent\n(base/'serving.pid').write_text(str(os.getpid()))\n(base/'serving.ready').write_text('ready')\nwhile True: time.sleep(.05)\n", True)
    actual = block
    if unsafe:
        actual = re.sub(r"\s+7>&- 8>&-", "", actual)
    prelude = f'''exec 7>"{lock7}" 8>"{lock8}"
python3 -c 'import fcntl; fcntl.flock(7, fcntl.LOCK_EX); fcntl.flock(8, fcntl.LOCK_EX)'
PATH="{bin_dir}:$PATH"
GATEWAY_PYTHON_BIN="{bin_dir / 'serving-python'}"
LEADPOET_REPO_ROOT="{tmp_path}"
GATEWAY_ROOT="{tmp_path}"
GATEWAY_LOG_ROOT="{tmp_path}"
GATEWAY_LOG_FILE="{tmp_path / 'gateway.log'}"
LAB_ARENA_SERVICE_LOG_FILE="{tmp_path / 'legacy.log'}"
GATEWAY_ENV_FILE="{tmp_path / 'env'}"
GATEWAY_RESTART_INVOCATION_ID=test
GATEWAY_ACTIVE_RELEASE_RESTART_INVOCATION_ID=test
GATEWAY_BIND_PORT=8001
ARENA_API_PORT=18001 ARENA_EDGE_PORT=18002
LAB_ARENA_RELEASE_ROOT="{tmp_path}" LAB_ARENA_RELEASE_COMMIT=test GITHUB_SHA=test
ARENA_COMPONENT_LOG_FILE="{tmp_path / 'arena.log'}" sha=test tree="{tmp_path}"
ARENA_COMPONENT_COMMAND=("{bin_dir / 'serving-python'}" -u scripts/stub.py)
'''
    # Execute a launch block, then let this controller shell exit. Its held
    # lock descriptions should not be retained by the serving process.
    controller = subprocess.run(
        ["bash", "-c", prelude + "\n" + actual],
        cwd=tmp_path,
        env={
            **os.environ,
            "LEADPOET_DOCKER_OPERATION_LOCK_HELD": "1",
            "LEADPOET_DOCKER_OPERATION_ADMISSION_LOCK_HELD": "1",
            "LEADPOET_DOCKER_OPERATION_LOCK_OWNER_PID": "12345",
        },
        capture_output=True,
        text=True,
        timeout=5,
    )
    try:
        assert controller.returncode == 0, controller.stderr
        deadline = time.monotonic() + 3
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert ready.exists(), controller.stderr
        serving_pid = int(pid_file.read_text())
        os.kill(serving_pid, 0)
        with lock7.open("r+") as first, lock8.open("r+") as second:
            try:
                fcntl.flock(first.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(second.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                return True
            except BlockingIOError:
                return False
    finally:
        if pid_file.exists():
            try:
                os.kill(int(pid_file.read_text()), 9)
            except (ProcessLookupError, ValueError):
                pass


@pytest.mark.parametrize("launch", ["arena-api", "arena-edge", "legacy-service", "gateway"])
def test_serving_children_do_not_retain_controller_locks(tmp_path: Path, launch: str) -> None:
    assert _exercise(tmp_path, BLOCKS[launch]) is True


def test_unsafe_launch_without_fd_closures_retains_locks(tmp_path: Path) -> None:
    assert _exercise(tmp_path, BLOCKS["gateway"], unsafe=True) is False
