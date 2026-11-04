"""Only one gateway process may prepare a round's publication at a time."""

import fcntl
import hashlib
import multiprocessing
import os
from pathlib import Path
import stat
from types import SimpleNamespace

import pytest

from lab_arena import publication_lock
from lab_arena.service import ArenaService
from tests.lab_arena.publication_stale_exit_test import publication


ROUND = "arena-2026-11-03"


def _stale_service(root, *, status="published"):
    service = object.__new__(ArenaService)
    service._config = SimpleNamespace(publication_lock_dir=root)
    service._round = lambda _: {"status": status}
    return service


def _hold_publication(root, ready, release):
    service = _stale_service(root)

    def read(_round_id):
        ready.set()
        if not release.wait(20):
            raise AssertionError("test publisher was not released")
        return {"status": "published"}

    service._round = read
    assert service.publish(ROUND) == {"status": "stale", "round_status": "published"}


@pytest.fixture
def owner(tmp_path):
    context = multiprocessing.get_context("spawn")
    ready, release = context.Event(), context.Event()
    process = context.Process(target=_hold_publication, args=(tmp_path / "locks", ready, release))
    process.start()
    try:
        assert ready.wait(10), "publisher process did not enter publication"
        yield process, release, tmp_path / "locks"
    finally:
        # A killed waiter can leave multiprocessing.Event's own semaphore
        # unusable. Do not touch that test signal after the process is dead.
        if process.is_alive():
            release.set()
            process.join(10)
        if process.is_alive():
            process.kill()
            process.join(5)
            pytest.fail("test publisher process did not stop")


def test_busy_publication_returns_before_any_round_read(tmp_path):
    root = tmp_path / "locks"
    root.mkdir(mode=0o700)
    path = root / (hashlib.sha256(ROUND.encode()).hexdigest() + ".lock")
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        service = object.__new__(ArenaService)
        service._config = SimpleNamespace(publication_lock_dir=root)

        def unexpected_read(_round_id):
            raise AssertionError("busy publisher started publication reads")

        service._round = unexpected_read
        assert service.publish(ROUND) == {
            "status": "retry", "round_status": "scored",
            "reason": "publication_in_progress",
        }
    finally:
        os.close(descriptor)


def test_separate_processes_exclude_same_round_but_allow_another_round(owner):
    process, release, root = owner
    contender = _stale_service(root)
    reads = []
    contender._round = lambda round_id: reads.append(round_id) or {"status": "published"}
    assert contender.publish(ROUND)["reason"] == "publication_in_progress"
    assert reads == []
    assert contender.publish(ROUND + "-other") == {"status": "stale", "round_status": "published"}
    assert reads == [ROUND + "-other"]
    release.set()
    process.join(10)
    assert process.exitcode == 0
    # A later attempt reads the owner's committed state before any heavy work.
    assert contender.publish(ROUND) == {"status": "stale", "round_status": "published"}
    assert reads == [ROUND + "-other", ROUND]


def test_abrupt_owner_exit_releases_kernel_lock_without_cleanup(owner):
    process, _release, root = owner
    assert _stale_service(root).publish(ROUND)["reason"] == "publication_in_progress"
    process.kill()
    process.join(10)
    assert process.exitcode != 0
    assert _stale_service(root).publish(ROUND) == {"status": "stale", "round_status": "published"}
    assert len(list(root.iterdir())) == 1


def test_exception_and_retry_release_lock(tmp_path):
    root = tmp_path / "locks"
    service = _stale_service(root)

    def failure(_round_id):
        raise ValueError("publication evidence unavailable")

    service._round = failure
    with pytest.raises(ValueError, match="evidence unavailable"):
        service.publish(ROUND)
    with publication_lock.round_lock(root, ROUND) as acquired:
        assert acquired
    service._publish_owned = lambda _: {"status": "retry", "reason": "judge_incomplete"}
    assert service.publish(ROUND)["reason"] == "judge_incomplete"
    with publication_lock.round_lock(root, ROUND) as acquired:
        assert acquired


def test_publication_holds_lock_through_assessments_and_final_cas(publication, tmp_path):
    publication.service._config = SimpleNamespace(publication_lock_dir=tmp_path / "locks")
    contender = _stale_service(publication.service._config.publication_lock_dir)
    transitions = publication.service._store.transition_round
    busy = []

    def attempt(_submission):
        busy.append(contender.publish("old")["reason"])

    def commit(*args):
        attempt(None)
        return transitions(*args)

    publication.after_assessment = attempt
    publication.service._store.transition_round = commit
    assert publication.service.publish("old")["status"] == "ok"
    assert busy == ["publication_in_progress"] * 3
    assert publication.assessed == ["baseline", "challenger"]
    assert len(publication.transitions) == 1
    with publication_lock.round_lock(publication.service._config.publication_lock_dir, "old") as acquired:
        assert acquired


@pytest.mark.parametrize("status", ["published", "cancelled"])
def test_host_lock_preserves_final_cas_against_external_state_change(publication, tmp_path, status):
    publication.service._config = SimpleNamespace(publication_lock_dir=tmp_path / "locks")
    transition = publication.service._store.transition_round

    def competing_transition(*args):
        publication.row.update(status=status, publication_doc={"owner": "other"})
        return transition(*args)

    publication.service._store.transition_round = competing_transition
    assert publication.service.publish("old")["status"] == "stale"
    assert publication.row["publication_doc"] == {"owner": "other"}
    assert publication.assessed == ["baseline", "challenger"]
    with publication_lock.round_lock(publication.service._config.publication_lock_dir, "old") as acquired:
        assert acquired


def test_busy_driver_and_admin_use_same_guard_and_driver_reaches_next_round(publication, tmp_path, capsys):
    from lab_arena.driver import _advance_active
    from scripts import lab_arena_admin

    root = tmp_path / "locks"
    publication.service._config = SimpleNamespace(publication_lock_dir=root)
    with publication_lock.round_lock(root, "old") as acquired:
        assert acquired
        assert "failed" not in _advance_active(publication.service)
        assert publication.next_opened == ["next"]
        args = SimpleNamespace(command="advance", round="old", expect_status="scored", dry_run=False)
        assert lab_arena_admin.run(args, publication.service) == 0
    assert "publication_in_progress" in capsys.readouterr().out
    assert publication.assessed == []
    assert publication.transitions == []


def test_lock_file_is_persistent_private_and_not_inherited(tmp_path, monkeypatch):
    root = tmp_path / "locks"
    observed = []
    flock = publication_lock.fcntl.flock

    def inspect(descriptor, operation):
        observed.append(os.get_inheritable(descriptor))
        return flock(descriptor, operation)

    monkeypatch.setattr(publication_lock.fcntl, "flock", inspect)
    with publication_lock.round_lock(root, ROUND) as acquired:
        assert acquired
        files = list(root.iterdir())
        assert len(files) == 1
        before = files[0].stat()
        assert stat.S_IMODE(before.st_mode) == 0o600
        assert before.st_uid == os.geteuid()
        assert stat.S_IMODE(root.stat().st_mode) == 0o700
    with publication_lock.round_lock(root, ROUND) as acquired:
        assert acquired
    assert list(root.iterdir()) == files
    assert files[0].stat().st_ino == before.st_ino
    assert observed == [False, False]


@pytest.mark.parametrize("unsafe", ["parent_symlink", "directory_symlink", "directory_public", "file_symlink", "file_hardlink", "file_public", "file_fifo"])
def test_unsafe_lock_paths_fail_closed_before_publication(tmp_path, unsafe):
    root = tmp_path / "locks"
    root.mkdir(mode=0o700)
    target = tmp_path / "target"
    path = root / (hashlib.sha256(ROUND.encode()).hexdigest() + ".lock")
    if unsafe == "parent_symlink":
        link = tmp_path / "alias"
        link.symlink_to(root, target_is_directory=True)
        root = link / "child"
    elif unsafe == "directory_symlink":
        root.rmdir()
        target.mkdir(mode=0o700)
        root.symlink_to(target, target_is_directory=True)
    elif unsafe == "directory_public":
        root.chmod(0o755)
    elif unsafe == "file_symlink":
        target.touch(mode=0o600)
        path.symlink_to(target)
    elif unsafe == "file_hardlink":
        target.touch(mode=0o600)
        os.link(target, path)
    elif unsafe == "file_public":
        path.touch(mode=0o644)
    elif unsafe == "file_fifo":
        os.mkfifo(path, mode=0o600)
    service = _stale_service(root)
    service._round = lambda _: pytest.fail("unsafe publication lock allowed a round read")
    from lab_arena.service import ServiceError

    with pytest.raises(ServiceError) as failure:
        service.publish(ROUND)
    assert failure.value.code == "publication_lock_unavailable"


def test_namespaces_use_stable_home_public_endpoint_and_mode(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    directory = publication_lock.production_lock_directory
    expected = directory("https://db.example", "live")
    assert expected.parent == tmp_path / ".config" / "leadpoet" / "arena-publication-locks"
    assert directory("https://DB.example:443/", "live") == expected
    assert directory("https://db.example/", "live") == expected
    assert directory("https://db.example", "shadow") != expected
    assert directory("https://other.example", "live") != expected
    assert directory("https://db.example/other", "live") != expected
    assert directory("https://private:credential@db.example?token=private", "live") == expected
    assert "db.example" not in str(expected)


def test_production_wiring_always_enables_stable_shared_lock(monkeypatch, tmp_path):
    # This constructs the real wiring with all external boundaries faked.
    from tests.lab_arena.broker_round_lease_wiring_test import _wired_factory
    from lab_arena import wiring

    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    first, _store, _provider = _wired_factory(monkeypatch)
    second, _store, _provider = _wired_factory(monkeypatch)
    expected = publication_lock.production_lock_directory("test", "shadow")
    assert first.config.publication_lock_dir == second.config.publication_lock_dir == expected
    # Runtime working-directory changes do not split ownership across slots.
    slot = tmp_path / "slot"
    slot.mkdir()
    monkeypatch.chdir(slot)
    third, _store, _provider = _wired_factory(monkeypatch)
    assert third.config.publication_lock_dir == expected
    live, _app = wiring.build_service_from_environment("live")
    live_again, _app = wiring.build_service_from_environment("live")
    assert live.config.publication_lock_dir == live_again.config.publication_lock_dir
    assert live.config.publication_lock_dir == publication_lock.production_lock_directory("test", "live")
    assert live.config.publication_lock_dir != expected
    assert wiring.production_lock_directory is publication_lock.production_lock_directory


def test_guard_outcomes_keep_existing_telemetry_vocabulary(tmp_path):
    from gateway.observability.otel_bootstrap import configure_arena_otel
    from lab_arena import telemetry
    from lab_arena.service import ServiceError
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    recorder = configure_arena_otel(None, span_exporter=exporter)
    root = tmp_path / "locks"
    service = _stale_service(root)
    telemetry.install_recorder(recorder)
    try:
        with publication_lock.round_lock(root, ROUND):
            with telemetry.stage("advance_publish"):
                assert service.publish(ROUND)["reason"] == "publication_in_progress"
        root.chmod(0o755)
        with pytest.raises(ServiceError):
            with telemetry.stage("advance_publish"):
                service.publish(ROUND)
        # These are process-local outcomes, not new request-denial attributes.
        for code in ("publication_in_progress", "publication_lock_unavailable"):
            recorder.note_denial(code)
            assert recorder.take_denial() == "-"
    finally:
        telemetry.install_recorder(None)
    spans = exporter.get_finished_spans()
    assert len(spans) == 2
    assert [span.attributes["arena.outcome"] for span in spans] == ["ok", "failed"]
    assert spans[1].attributes["arena.error_type"] == "ServiceError"
    assert str(root) not in repr(spans)
