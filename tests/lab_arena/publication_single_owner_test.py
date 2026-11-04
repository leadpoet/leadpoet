"""The gateway publication gate must work across independent processes."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from lab_arena.publication_lock import PublicationLockError, publication_round_lock
from lab_arena.service import ArenaService, ServiceError


_PUBLISH_CHILD = """
import json, sys
from pathlib import Path
from types import SimpleNamespace
from lab_arena.service import ArenaService
s = ArenaService.__new__(ArenaService)
s._config = SimpleNamespace(publication_lock_dir=Path(sys.argv[1]))
def assessment(round_id):
    Path(sys.argv[2]).write_text('assessed')
    if sys.argv[4] == 'hold':
        sys.stdin.readline()
    return {'status': 'stale', 'round_status': 'published'}
s._publish_locked = assessment
print(json.dumps(s.publish(sys.argv[3])), flush=True)
"""


def _child(directory: Path, marker: Path, round_id: str, mode: str, *, wait=False):
    command = [sys.executable, "-c", _PUBLISH_CHILD, str(directory),
               str(marker), round_id, mode]
    if wait:
        return subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True)
    return subprocess.run(command, capture_output=True, text=True, timeout=5, check=True)


def _wait_for(path: Path):
    until = time.monotonic() + 5
    while time.monotonic() < until:
        if path.exists():
            return
        time.sleep(0.01)
    raise AssertionError("publication assessment did not start")


def test_one_expensive_assessment_per_round_and_different_round_independent(tmp_path):
    tmp_path.chmod(0o700)
    first_marker = tmp_path / "first"
    first = _child(tmp_path, first_marker, "arena-2026-11-03", "hold", wait=True)
    try:
        _wait_for(first_marker)
        duplicate_marker = tmp_path / "duplicate"
        start = time.monotonic()
        duplicate = _child(tmp_path, duplicate_marker, "arena-2026-11-03", "quick")
        assert time.monotonic() - start < 3
        assert json.loads(duplicate.stdout) == {
            "status": "retry", "round_status": "scored", "reason": "publication_busy",
        }
        assert not duplicate_marker.exists()
        other_marker = tmp_path / "other"
        other = _child(tmp_path, other_marker, "arena-2026-11-04", "quick")
        assert json.loads(other.stdout)["status"] == "stale"
        assert other_marker.exists()
    finally:
        if first.poll() is None:
            first.stdin.write("\n")
            first.stdin.flush()
        stdout, stderr = first.communicate(timeout=5)
        assert first.returncode == 0, stderr
        assert json.loads(stdout)["status"] == "stale"
    replay_marker = tmp_path / "replay"
    assert json.loads(_child(tmp_path, replay_marker, "arena-2026-11-03", "quick").stdout)["status"] == "stale"
    assert replay_marker.exists()


def test_process_death_releases_lock(tmp_path):
    tmp_path.chmod(0o700)
    first = _child(tmp_path, tmp_path / "first", "arena-2026-11-03", "hold", wait=True)
    try:
        _wait_for(tmp_path / "first")
        first.kill()
        first.communicate(timeout=5)
        result = _child(tmp_path, tmp_path / "after", "arena-2026-11-03", "quick")
        assert json.loads(result.stdout)["status"] == "stale"
        assert (tmp_path / "after").exists()
    finally:
        if first.poll() is None:
            first.kill()
            first.communicate(timeout=5)


def test_exception_releases_lock_and_bad_lock_files_fail_closed(tmp_path):
    tmp_path.chmod(0o700)
    with pytest.raises(RuntimeError, match="assessment failed"):
        with publication_round_lock(tmp_path, "round") as acquired:
            assert acquired
            raise RuntimeError("assessment failed")
    with publication_round_lock(tmp_path, "round") as acquired:
        assert acquired
    # A killed creator may leave its temporary name hard-linked to the final
    # inode. The final lock must remain usable after that crash.
    round_name = "arena-publication-" + hashlib.sha256(b"round").hexdigest() + ".lock"
    os.link(tmp_path / round_name, tmp_path / "orphaned-temporary")
    with publication_round_lock(tmp_path, "round") as acquired:
        assert acquired
    name = "arena-publication-" + hashlib.sha256(b"bad").hexdigest() + ".lock"
    (tmp_path / name).symlink_to(tmp_path / "round")
    with pytest.raises(PublicationLockError):
        with publication_round_lock(tmp_path, "bad"):
            pass
    (tmp_path / name).unlink()
    (tmp_path / name).write_text("unsafe")
    (tmp_path / name).chmod(0o644)
    with pytest.raises(PublicationLockError):
        with publication_round_lock(tmp_path, "bad"):
            pass
    tmp_path.chmod(0o777)
    with pytest.raises(PublicationLockError):
        with publication_round_lock(tmp_path, "round"):
            pass


def test_service_refuses_to_assess_when_lock_unavailable(tmp_path):
    service = ArenaService.__new__(ArenaService)
    service._config = type("Config", (), {"publication_lock_dir": tmp_path / "missing"})()
    service._publish_locked = lambda _round: pytest.fail("assessment bypassed lock")
    with pytest.raises(ServiceError) as failure:
        service.publish("arena-2026-11-03")
    assert failure.value.code == "publication_lock_unavailable"
