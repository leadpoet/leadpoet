from __future__ import annotations

from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "scripts" / "research_lab_admin_wrapper_runtime.sh"
CANONICAL_CUTOVER_PATH = (
    "/home/ec2-user/.config/leadpoet/stateful-epoch-cutover.json"
)
UNSET = "__UNSET__"
ADMIN_ARGS = ("maintenance", "status", "--json")


def _isolated_wrapper(tmp_path: Path, canonical_path: Path) -> Path:
    source = WRAPPER.read_text(encoding="utf-8")
    assert source.count(CANONICAL_CUTOVER_PATH) == 1
    wrapper = tmp_path / "research_lab_admin_wrapper_runtime.sh"
    wrapper.write_text(
        source.replace(CANONICAL_CUTOVER_PATH, str(canonical_path)),
        encoding="utf-8",
    )
    wrapper.chmod(0o700)
    return wrapper


def _run_wrapper(
    tmp_path: Path,
    *,
    wrapper: Path = WRAPPER,
    env_lines: tuple[str, ...] = (),
) -> tuple[subprocess.CompletedProcess[str], Path, Path]:
    repo = tmp_path / "repo"
    (repo / "gateway").mkdir(parents=True)
    env_file = tmp_path / "gateway.env"
    env_file.write_text("\n".join(env_lines), encoding="utf-8")
    capture = tmp_path / "captured.invocation"
    python_bin = tmp_path / "python3"
    python_bin.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "{\n"
        "  printf 'cwd=%s\\n' \"$PWD\"\n"
        f"  printf 'path=%s\\n' \"${{LEADPOET_SUBNET_EPOCH_CUTOVER_PATH-{UNSET}}}\"\n"
        f"  printf 'json=%s\\n' \"${{LEADPOET_SUBNET_EPOCH_CUTOVER_JSON-{UNSET}}}\"\n"
        f"  printf 'pythonpath=%s\\n' \"${{PYTHONPATH-{UNSET}}}\"\n"
        f"  printf 'env_file=%s\\n' \"${{GATEWAY_ENV_FILE-{UNSET}}}\"\n"
        "  printf 'arg=%s\\n' \"$@\"\n"
        "} > \"$CAPTURE\"\n",
        encoding="utf-8",
    )
    python_bin.chmod(0o700)
    completed = subprocess.run(
        ["bash", str(wrapper), *ADMIN_ARGS],
        check=False,
        capture_output=True,
        text=True,
        env={
            "PATH": "/usr/bin:/bin",
            "HOME": str(tmp_path),
            "LEADPOET_REPO": str(repo),
            "GATEWAY_ENV_FILE": str(env_file),
            "GATEWAY_PYTHON_BIN": str(python_bin),
            "CAPTURE": str(capture),
        },
    )
    return completed, capture, repo


@pytest.mark.parametrize("authority", ("canonical", "path", "json"))
def test_admin_wrapper_selects_one_cutover_authority_and_executes_exactly(
    tmp_path: Path,
    authority: str,
) -> None:
    canonical_path = tmp_path / "stateful-epoch-cutover.json"
    wrapper = WRAPPER
    env_lines: tuple[str, ...] = ()
    expected_path = UNSET
    expected_json = UNSET
    if authority == "canonical":
        canonical_path.write_text("{}", encoding="utf-8")
        wrapper = _isolated_wrapper(tmp_path, canonical_path)
        expected_path = str(canonical_path)
    elif authority == "path":
        explicit_path = tmp_path / "operator-cutover.json"
        explicit_path.write_text("{}", encoding="utf-8")
        env_lines = (f"LEADPOET_SUBNET_EPOCH_CUTOVER_PATH={explicit_path}",)
        expected_path = str(explicit_path)
    else:
        expected_json = '{"schema_version":"future.v1","opaque":"a=b"}'
        env_lines = (f"LEADPOET_SUBNET_EPOCH_CUTOVER_JSON={expected_json}",)

    completed, capture, repo = _run_wrapper(
        tmp_path,
        wrapper=wrapper,
        env_lines=env_lines,
    )

    assert completed.returncode == 0, completed.stderr
    assert capture.read_text(encoding="utf-8").splitlines() == [
        f"cwd={repo.resolve()}",
        f"path={expected_path}",
        f"json={expected_json}",
        f"pythonpath={repo}",
        "env_file=/dev/null",
        "arg=-m",
        "arg=gateway.research_lab.admin",
        *(f"arg={value}" for value in ADMIN_ARGS),
    ]


@pytest.mark.parametrize("failure", ("both", "missing", "empty"))
def test_admin_wrapper_rejects_ambiguous_or_unusable_authority_before_python(
    tmp_path: Path,
    failure: str,
) -> None:
    canonical_path = tmp_path / "stateful-epoch-cutover.json"
    wrapper = WRAPPER
    env_lines: tuple[str, ...] = ()
    expected_error = "set only one subnet epoch cutover authority form"
    if failure == "both":
        env_lines = (
            "LEADPOET_SUBNET_EPOCH_CUTOVER_PATH=/operator/cutover.json",
            'LEADPOET_SUBNET_EPOCH_CUTOVER_JSON={"schema_version":"test"}',
        )
    else:
        if failure == "empty":
            canonical_path.write_text("", encoding="utf-8")
        wrapper = _isolated_wrapper(tmp_path, canonical_path)
        expected_error = (
            "canonical subnet epoch cutover manifest is not a regular nonempty file"
        )

    completed, capture, _ = _run_wrapper(
        tmp_path,
        wrapper=wrapper,
        env_lines=env_lines,
    )

    assert completed.returncode == 2
    assert expected_error in completed.stderr
    assert not capture.exists()
