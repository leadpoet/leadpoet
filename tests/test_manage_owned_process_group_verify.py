import argparse
import hashlib
import json
import os
import stat
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts import manage_owned_process_group as process_group


def _fixture(tmp_path, *, mode: int = 0o600) -> tuple[argparse.Namespace, dict]:
    cwd = os.path.realpath(tmp_path)
    argv = ["/srv/arena/old-release", "--role", "gateway"]
    state = {
        "version": process_group.STATE_VERSION,
        "pid": 4242,
        "pgid": 4242,
        "start_time_ticks": 987654,
        "uid": os.getuid(),
        "cwd": cwd,
        "argv": argv,
    }
    state_file = tmp_path / "owned-process.json"
    state_file.write_text(json.dumps(state, sort_keys=True) + "\n", encoding="utf-8")
    state_file.chmod(mode)
    args = argparse.Namespace(
        state_file=state_file,
        cwd=cwd,
        uid=os.getuid(),
        process_argv=argv,
    )
    return args, state


def _patch_live_process(monkeypatch: pytest.MonkeyPatch, state: dict, **changes: object) -> None:
    process = {
        **state,
        "state": "S",
        "session": state["pgid"],
        **changes,
    }
    monkeypatch.setattr(process_group, "_read_process", lambda _pid: process)
    monkeypatch.setattr(process_group, "_matching_processes", lambda **_kwargs: [process])
    monkeypatch.setattr(process_group, "_active_group_members", lambda _pgid: [process])


def test_verify_emits_only_bounded_hash_metadata_and_does_not_mutate(
    tmp_path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    args, state = _fixture(tmp_path)
    _patch_live_process(monkeypatch, state)
    before_bytes = args.state_file.read_bytes()
    before_mode = stat.S_IMODE(args.state_file.stat().st_mode)
    monkeypatch.setattr(
        process_group.os,
        "killpg",
        lambda *_args: pytest.fail("verify must never signal a process"),
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "manage_owned_process_group.py",
            "verify",
            "--state-file",
            str(args.state_file),
            "--cwd",
            args.cwd,
            "--uid",
            str(args.uid),
            "--",
            *args.process_argv,
        ],
    )
    assert process_group.main() == 0

    result = json.loads(capsys.readouterr().out)
    assert result["action"] == "verify"
    assert result["ok"] is True
    assert result["pid"] == state["pid"]
    assert result["pgid"] == state["pgid"]
    assert result["live_group"] is True
    assert result["live_group_member_count"] == 1
    assert len(result["state_sha256"]) == 64
    assert len(result["cwd_sha256"]) == 64
    assert len(result["argv_sha256"]) == 64
    assert "argv" not in result
    assert "cwd" not in result
    assert result["state_sha256"] == hashlib.sha256(before_bytes).hexdigest()
    assert args.state_file.read_bytes() == before_bytes
    assert stat.S_IMODE(args.state_file.stat().st_mode) == before_mode


def test_verify_rejects_reused_pid_without_mutation(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, state = _fixture(tmp_path)
    _patch_live_process(monkeypatch, state, start_time_ticks=state["start_time_ticks"] + 1)
    before = args.state_file.read_bytes()

    with pytest.raises(process_group.OwnershipError, match="identity changed"):
        process_group._verify(args)

    assert args.state_file.read_bytes() == before


def test_verify_rejects_non_owner_state_file(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    args, state = _fixture(tmp_path, mode=0o640)
    _patch_live_process(monkeypatch, state)
    before = args.state_file.read_bytes()

    with pytest.raises(process_group.OwnershipError, match="owner-only"):
        process_group._verify(args)

    assert args.state_file.read_bytes() == before


def test_verify_rejects_state_replacement_during_read(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, state = _fixture(tmp_path)
    _patch_live_process(monkeypatch, state)
    real_fstat = process_group.os.fstat
    calls = 0

    def changing_fstat(descriptor: int):
        nonlocal calls
        calls += 1
        metadata = real_fstat(descriptor)
        if calls == 2:
            return SimpleNamespace(
                st_dev=metadata.st_dev,
                st_ino=metadata.st_ino,
                st_mode=metadata.st_mode,
                st_uid=metadata.st_uid,
                st_gid=metadata.st_gid,
                st_size=metadata.st_size + 1,
                st_mtime_ns=metadata.st_mtime_ns,
                st_ctime_ns=metadata.st_ctime_ns,
            )
        return metadata

    monkeypatch.setattr(process_group.os, "fstat", changing_fstat)
    before = args.state_file.read_bytes()

    with pytest.raises(process_group.OwnershipError, match="changed while it was read"):
        process_group._verify(args)

    assert args.state_file.read_bytes() == before


def test_verify_rejects_path_replacement_during_read(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, state = _fixture(tmp_path)
    _patch_live_process(monkeypatch, state)
    real_lstat = process_group.os.lstat
    calls = 0
    target = os.fspath(args.state_file)

    def changing_lstat(path):
        nonlocal calls
        metadata = real_lstat(path)
        if os.fspath(path) != target:
            return metadata
        calls += 1
        if calls == 2:
            return SimpleNamespace(
                st_dev=metadata.st_dev,
                st_ino=metadata.st_ino + 1,
                st_mode=metadata.st_mode,
                st_uid=metadata.st_uid,
                st_gid=metadata.st_gid,
                st_size=metadata.st_size,
                st_mtime_ns=metadata.st_mtime_ns,
                st_ctime_ns=metadata.st_ctime_ns,
            )
        return metadata

    monkeypatch.setattr(process_group.os, "lstat", changing_lstat)
    before = args.state_file.read_bytes()

    with pytest.raises(process_group.OwnershipError, match="replaced"):
        process_group._verify(args)

    assert args.state_file.read_bytes() == before


def test_verify_rejects_duplicate_exact_processes(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, state = _fixture(tmp_path)
    first = {**state, "state": "S", "session": state["pgid"]}
    second = {
        **first,
        "pid": state["pid"] + 1,
        "pgid": state["pgid"] + 1,
        "session": state["pgid"] + 1,
    }
    monkeypatch.setattr(process_group, "_all_processes", lambda: [first, second])
    before = args.state_file.read_bytes()

    with pytest.raises(process_group.OwnershipError, match="exactly one"):
        process_group._verify(args)

    assert args.state_file.read_bytes() == before


def test_verify_rejects_detached_session_mismatch(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, state = _fixture(tmp_path)
    process = {**state, "state": "S", "session": state["pgid"] + 1}
    monkeypatch.setattr(process_group, "_all_processes", lambda: [process])

    with pytest.raises(process_group.OwnershipError, match="exactly one"):
        process_group._verify(args)


def test_verify_rejects_symlink_state(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    args, state = _fixture(tmp_path)
    target = tmp_path / "target-state.json"
    target.write_bytes(args.state_file.read_bytes())
    args.state_file.unlink()
    args.state_file.symlink_to(target)
    _patch_live_process(monkeypatch, state)

    with pytest.raises(process_group.OwnershipError, match="regular file"):
        process_group._verify(args)


def test_verify_rejects_fifo_state_without_blocking(tmp_path) -> None:
    args, _state = _fixture(tmp_path)
    args.state_file.unlink()
    os.mkfifo(args.state_file, 0o600)

    command = [
        sys.executable,
        os.fspath(process_group.__file__),
        "verify",
        "--state-file",
        os.fspath(args.state_file),
        "--cwd",
        args.cwd,
        "--uid",
        str(args.uid),
        "--",
        *args.process_argv,
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=1)

    assert result.returncode == 1
    assert "regular file" in result.stderr


def test_verify_rejects_fifo_replaced_after_initial_check(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, _state = _fixture(tmp_path)
    replacement = tmp_path / "replacement-fifo"
    os.mkfifo(replacement, 0o600)
    real_open = process_group.os.open
    swapped = False

    def replacing_open(path, flags, mode=0o777):
        nonlocal swapped
        if not swapped and os.fspath(path) == os.fspath(args.state_file):
            swapped = True
            saved = tmp_path / "saved-state.json"
            os.replace(args.state_file, saved)
            os.replace(replacement, args.state_file)
        return real_open(path, flags, mode)

    monkeypatch.setattr(process_group.os, "open", replacing_open)

    with pytest.raises(process_group.OwnershipError, match="regular file"):
        process_group._verify(args)

    assert swapped is True


def test_verify_rejects_wrong_owner_from_opened_file(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, state = _fixture(tmp_path)
    _patch_live_process(monkeypatch, state)
    real_fstat = process_group.os.fstat

    def wrong_owner(descriptor: int):
        metadata = real_fstat(descriptor)
        return SimpleNamespace(
            st_dev=metadata.st_dev,
            st_ino=metadata.st_ino,
            st_mode=metadata.st_mode,
            st_uid=metadata.st_uid + 1,
            st_gid=metadata.st_gid,
            st_size=metadata.st_size,
            st_mtime_ns=metadata.st_mtime_ns,
            st_ctime_ns=metadata.st_ctime_ns,
        )

    monkeypatch.setattr(process_group.os, "fstat", wrong_owner)
    with pytest.raises(process_group.OwnershipError, match="owned by"):
        process_group._verify(args)


@pytest.mark.parametrize("identity_field", ["version", "pid", "pgid", "start_time_ticks", "uid"])
def test_verify_rejects_bool_identity_values(tmp_path, identity_field: str) -> None:
    args, _state = _fixture(tmp_path)
    state = json.loads(args.state_file.read_text(encoding="utf-8"))
    state[identity_field] = True
    args.state_file.write_text(json.dumps(state, sort_keys=True) + "\n", encoding="utf-8")
    args.state_file.chmod(0o600)
    before = args.state_file.read_bytes()

    with pytest.raises(process_group.OwnershipError, match="invalid schema"):
        process_group._verify(args)

    assert args.state_file.read_bytes() == before
