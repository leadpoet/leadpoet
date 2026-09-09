from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import pytest

from scripts.lab_arena_restart_claim_guard import _commitments, _identity


SCRIPT = Path("scripts/lab_arena_restart_guard_handoff.py")
CANDIDATE = "a" * 40
AUTHORITY = "b" * 40
INVOCATION = "restart-handoff-test"
NONCE = "c" * 64
ENV = {
    **os.environ,
    "PYTHONPATH": os.pathsep.join(
        value for value in (str(Path.cwd()), os.environ.get("PYTHONPATH", "")) if value
    ),
}
for _credential_name in (
    "LAB_ARENA_SUPABASE_URL",
    "LAB_ARENA_SUPABASE_ANON_KEY",
    "LAB_ARENA_SERVICE_KEY",
    "LAB_ARENA_SERVICE_JWT",
):
    ENV.pop(_credential_name, None)


@pytest.fixture()
def handoff_root():
    root = Path(tempfile.mkdtemp(prefix="leadpoet-handoff-test.", dir="/tmp"))
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _args(action: str, path: Path, *, nonce: str = NONCE) -> list[str]:
    return [
        sys.executable,
        str(SCRIPT),
        action,
        "--path",
        str(path),
        "--candidate",
        CANDIDATE,
        "--invocation",
        INVOCATION,
        "--scope",
        "all",
        "--nonce",
        nonce,
        "--authority-commit",
        AUTHORITY,
    ]


def _authorized_state(*, generation: int = 7, expiry: str | None = None) -> dict:
    guard, owner = _identity(CANDIDATE, INVOCATION)
    guard_commitment, owner_commitment = _commitments(guard, owner)
    return {
        "schema_version": "leadpoet.lab_arena.restart_guard_state.v1",
        "paused": True,
        "operator_paused": False,
        "guard_present": True,
        "guard_active": True,
        "guard_commitment": guard_commitment,
        "owner_commitment": owner_commitment,
        "guard_generation": generation,
        "guard_expires_at": expiry
        or (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
        "candidate_commit": CANDIDATE,
        "restart_scope": "all",
        "restart_phase": "validator_destructive",
        "drain": {
            "schema_version": "leadpoet.lab_arena.restart_drain_state.v1",
            "captured_count": 2,
            "accepted_receipt_count": 1,
            "reported_terminal_receipt_count": 1,
            "still_leased_count": 0,
            "lost_or_mutated_count": 0,
            "current_leased_count": 0,
            "pending_retry_count": 1,
            "preserved": True,
        },
    }


def test_request_and_offset_expiry_permit_round_trip_without_nonce_output(
    handoff_root: Path,
) -> None:
    request = handoff_root / "leadpoet-request.json"
    permit = handoff_root / "leadpoet-permit.json"
    written = subprocess.run(
        _args("write-request", request), capture_output=True, text=True, check=True, env=ENV
    )
    checked = subprocess.run(
        _args("validate-request", request), capture_output=True, text=True, check=True, env=ENV
    )
    permit_written = subprocess.run(
        [*_args("write-permit", permit), "--expected-generation", "7"],
        input=json.dumps(_authorized_state()),
        capture_output=True,
        text=True,
        check=True,
        env=ENV,
    )
    permit_checked = subprocess.run(
        _args("validate-permit", permit), capture_output=True, text=True, check=True, env=ENV
    )

    for result in (written, checked, permit_written, permit_checked):
        assert NONCE not in result.stdout
        assert "handoff_nonce_commitment" in result.stdout
    assert json.loads(request.read_text())["handoff_nonce"] == NONCE
    assert json.loads(permit.read_text())["guard_generation"] == 7


def test_permit_rejects_changed_authorization_generation(handoff_root: Path) -> None:
    result = subprocess.run(
        [
            *_args("write-permit", handoff_root / "leadpoet-permit.json"),
            "--expected-generation",
            "8",
        ],
        input=json.dumps(_authorized_state(generation=7)),
        capture_output=True,
        text=True,
        env=ENV,
    )

    assert result.returncode == 1
    assert NONCE not in result.stderr
    assert not (handoff_root / "leadpoet-permit.json").exists()


def test_request_and_permit_reject_changed_nonce_before_use(handoff_root: Path) -> None:
    request = handoff_root / "leadpoet-request.json"
    permit = handoff_root / "leadpoet-permit.json"
    subprocess.run(_args("write-request", request), check=True, capture_output=True, env=ENV)
    request_result = subprocess.run(
        _args("validate-request", request, nonce="d" * 64),
        capture_output=True,
        text=True,
        env=ENV,
    )
    subprocess.run(
        [*_args("write-permit", permit), "--expected-generation", "7"],
        input=json.dumps(_authorized_state()),
        check=True,
        capture_output=True,
        text=True,
        env=ENV,
    )
    permit_result = subprocess.run(
        _args("validate-permit", permit, nonce="d" * 64),
        capture_output=True,
        text=True,
        env=ENV,
    )

    assert request_result.returncode == 1
    assert permit_result.returncode == 1
    assert NONCE not in request_result.stderr
    assert NONCE not in permit_result.stderr


def test_request_rejects_malformed_document(handoff_root: Path) -> None:
    request = handoff_root / "leadpoet-request.json"
    request.write_text("{not-json", encoding="utf-8")
    request.chmod(0o600)

    result = subprocess.run(
        _args("validate-request", request),
        capture_output=True,
        text=True,
        env=ENV,
    )

    assert result.returncode == 1
    assert "handoff file is invalid" in result.stderr


def test_permit_rejects_unpreserved_drain_arithmetic(handoff_root: Path) -> None:
    state = _authorized_state()
    state["drain"]["accepted_receipt_count"] = 0
    result = subprocess.run(
        _args("write-permit", handoff_root / "leadpoet-permit.json"),
        input=json.dumps(state),
        capture_output=True,
        text=True,
        env=ENV,
    )

    assert result.returncode == 1
    assert "authorization is invalid" in result.stderr


@pytest.mark.parametrize("unsafe_kind", ["mode", "symlink", "fifo"])
def test_handoff_reader_rejects_unsafe_files(
    handoff_root: Path, unsafe_kind: str,
) -> None:
    request = handoff_root / "leadpoet-request.json"
    subprocess.run(_args("write-request", request), check=True, capture_output=True, env=ENV)
    if unsafe_kind == "mode":
        request.chmod(0o644)
    elif unsafe_kind == "symlink":
        target = handoff_root / "leadpoet-target.json"
        request.rename(target)
        request.symlink_to(target)
    else:
        request.unlink()
        os.mkfifo(request, 0o600)

    result = subprocess.run(
        _args("validate-request", request), capture_output=True, text=True, timeout=2, env=ENV
    )

    assert result.returncode == 1
    assert "handoff file is unavailable" in result.stderr


def test_permit_rejects_expired_or_naive_timestamp(handoff_root: Path) -> None:
    for index, expiry in enumerate(("2000-01-01T00:00:00Z", "2099-01-01T00:00:00")):
        result = subprocess.run(
            _args("write-permit", handoff_root / f"leadpoet-permit-{index}.json"),
            input=json.dumps(_authorized_state(expiry=expiry)),
            capture_output=True,
            text=True,
            env=ENV,
        )
        assert result.returncode == 1
        assert "expiry" in result.stderr or "expired" in result.stderr
