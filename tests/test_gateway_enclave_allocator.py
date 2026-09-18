import os
from pathlib import Path
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def _write_command(path: Path, body: str) -> None:
    path.write_text("#!/bin/bash\nset -euo pipefail\n" + body, encoding="utf-8")
    path.chmod(0o755)


def test_allocator_transitions_old_two_role_capacity_to_coordinator(tmp_path: Path):
    tee = tmp_path / "tee"
    tee.mkdir()
    shutil.copy2(ROOT / "gateway/tee/configure_allocator.sh", tee)
    shutil.copy2(ROOT / "gateway/tee/topology.py", tee)
    shutil.copy2(ROOT / "gateway/tee/topology.json", tee)
    commands = tmp_path / "commands"
    commands.mkdir()
    log = tmp_path / "sudo.log"
    allocator = tmp_path / "allocator.yaml"
    allocator.write_text("---\nmemory_mib: 65536\ncpu_count: 8\n", encoding="utf-8")
    _write_command(commands / "getconf", 'printf "16\\n"\n')
    _write_command(commands / "awk", 'printf "131072\\n"\n')
    _write_command(
        commands / "sudo",
        'printf "%s\\n" "$*" >> "$ALLOCATOR_TEST_LOG"\n'
        'case "$1" in\n'
        '  cmp) exec /usr/bin/cmp "${@:2}" ;;\n'
        '  install) destination="${@: -1}"; source="${@: -2:1}"; '
        'mkdir -p "$(dirname "$destination")"; cp "$source" "$destination" ;;\n'
        '  nitro-cli) exit 0 ;;\n'
        '  systemctl) exit 0 ;;\n'
        '  *) exit 91 ;;\n'
        'esac\n',
    )
    result = subprocess.run(
        ["bash", str(tee / "configure_allocator.sh")],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PATH": str(commands) + os.pathsep + os.environ["PATH"],
            "ALLOCATOR_TEST_LOG": str(log),
            "NITRO_ENCLAVES_ALLOCATOR_CONFIG": str(allocator),
        },
    )
    assert result.returncode == 0, result.stderr
    assert allocator.read_text(encoding="utf-8") == "---\nmemory_mib: 8192\ncpu_count: 2\n"
    calls = log.read_text(encoding="utf-8")
    assert calls.index("nitro-cli terminate-enclave --all") < calls.index(
        "systemctl restart nitro-enclaves-allocator.service"
    )


def test_allocator_rejects_noncanonical_roles_before_host_mutation(tmp_path: Path):
    tee = tmp_path / "tee"
    tee.mkdir()
    script = ROOT / "gateway/tee/configure_allocator.sh"
    shutil.copy2(script, tee)
    shutil.copy2(ROOT / "gateway/tee/topology.py", tee)
    topology = (ROOT / "gateway/tee/topology.json").read_text(encoding="utf-8")
    (tee / "topology.json").write_text(
        topology.replace(
            '"roles": {',
            '"roles": {"gateway_scoring": {"cid": 17, "memory_mib": 8192, '
            '"service_role": "gateway_scoring", "vcpus": 2},',
            1,
        ),
        encoding="utf-8",
    )
    commands = tmp_path / "commands"
    commands.mkdir()
    mutation = tmp_path / "host-mutated"
    _write_command(
        commands / "sudo",
        f"touch {mutation!s}\nexit 90\n",
    )
    result = subprocess.run(
        ["bash", str(tee / "configure_allocator.sh")],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PATH": str(commands) + os.pathsep + os.environ["PATH"],
        },
    )
    assert result.returncode != 0
    assert not mutation.exists()
    assert 'set(roles or {}) != {"gateway_coordinator"}' in script.read_text(
        encoding="utf-8"
    )
