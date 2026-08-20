import json
import os
from pathlib import Path
import subprocess
import sys

from validator_tee.host import vsock_client


ROOT = Path(__file__).resolve().parents[1]


def _run_environment_cid_probe(source: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["ENCLAVE_CID"] = "23"
    environment["PYTHONPATH"] = os.pathsep.join(
        value
        for value in (str(ROOT), environment.get("PYTHONPATH"))
        if value
    )
    return subprocess.run(
        [sys.executable, "-c", source],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )


def test_environment_cid_diagnostic_does_not_pollute_stdout():
    result = _run_environment_cid_probe(
        "from validator_tee.host.vsock_client import get_enclave_cid; "
        "raise SystemExit(0 if get_enclave_cid() == 23 else 2)"
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
    assert result.stderr == "[vsock] Using ENCLAVE_CID from environment: 23\n"


def test_environment_cid_keeps_operator_json_stdout_machine_readable():
    result = _run_environment_cid_probe(
        "import json; "
        "from validator_tee.host.vsock_client import get_enclave_cid; "
        "print(json.dumps({'cid': get_enclave_cid()}, separators=(',', ':')))"
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"cid": 23}
    assert result.stdout == '{"cid":23}\n'
    assert result.stderr == "[vsock] Using ENCLAVE_CID from environment: 23\n"


def test_invalid_environment_cid_diagnostic_stays_on_stderr(
    monkeypatch,
    capsys,
):
    monkeypatch.setenv("ENCLAVE_CID", "not-a-cid")
    monkeypatch.setattr(
        vsock_client.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            args=["nitro-cli", "describe-enclaves"],
            returncode=0,
            stdout='[{"State":"RUNNING","EnclaveCID":29}]',
            stderr="",
        ),
    )

    assert vsock_client.get_enclave_cid() == 29
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "[vsock] Invalid ENCLAVE_CID: not-a-cid\n"


def test_cid_discovery_error_diagnostic_stays_on_stderr(monkeypatch, capsys):
    monkeypatch.delenv("ENCLAVE_CID", raising=False)

    def fail_discovery(*_args, **_kwargs):
        raise OSError("nitro-cli unavailable")

    monkeypatch.setattr(vsock_client.subprocess, "run", fail_discovery)

    assert vsock_client.get_enclave_cid() is None
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == (
        "[vsock] Error getting enclave CID: nitro-cli unavailable\n"
    )
