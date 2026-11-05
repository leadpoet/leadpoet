"""Source identity survives cache hits, reloads and active-run transitions."""

import gzip
import io
import os
import tarfile
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from lab_arena import runner, source_bundle
from tests.lab_arena.test_lab_arena_runner import _source_archive_with_requirements


def archive(version, *, compressed=False):
    fixture = _source_archive_with_requirements("fixture-package==1.0.0\n")
    contents = []
    with tarfile.open(fileobj=io.BytesIO(fixture), mode="r:gz") as source:
        for member in source:
            body = source.extractfile(member).read()
            if member.name == "harness.py":
                body = b'VERSION = "' + version.encode() + b'"\n' + body
            contents.append((member.name, body))
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w") as target:
        for name, body in contents:
            member = tarfile.TarInfo(name)
            member.size = len(body)
            target.addfile(member, io.BytesIO(body))
    payload = gzip.compress(
        raw.getvalue(), compresslevel=9 if compressed else 0, mtime=0
    )
    source_bundle.validate_source_archive(payload)
    return payload


@pytest.fixture
def sources(tmp_path, monkeypatch):
    payloads = {
        "old": archive("old"),
        "new": archive("new"),
        "small": archive("new", compressed=True),
    }
    assert payloads["old"] != payloads["new"] and len(payloads["old"]) == len(
        payloads["new"]
    )
    assert len(payloads["old"]) != len(payloads["small"])
    fetches = []
    installs = []
    current = {}

    def fetch(run_id, _token):
        fetches.append(run_id)
        current["run"] = run_id
        return payloads[run_id]

    def install(_requirements, target):
        installs.append(current["run"])
        (target / "installed.py").write_text(current["run"])

    remove = runner._remove_cache_path

    def remove_as_owner(path):
        # Production runs as root. Unlock only this disposable fixture's
        # read-only directories before invoking the unchanged cleanup helper.
        for directory, _names, _files in os.walk(path):
            os.chmod(directory, 0o700)
        remove(path)

    monkeypatch.setattr(runner, "_remove_cache_path", remove_as_owner)
    root = tmp_path / "sources"

    def make(**kwargs):
        return runner.SourceCache(root, fetch, dependency_installer=install, **kwargs)

    return make, root, payloads, fetches, installs


def acquire(cache, payloads, run_id, ref, submission="sub-fixed"):
    return cache.acquire(
        run_id, "fixture-token", ref, submission, len(payloads[run_id])
    )


@pytest.mark.parametrize("restart", [False, True])
def test_changed_reference_with_equal_size_rebuilds_source_and_dependencies(
    sources, restart
):
    make, root, payloads, fetches, installs = sources
    cache = make()
    with acquire(cache, payloads, "old", "old/source.tar.gz"):
        pass
    if restart:
        cache = make()
    with acquire(cache, payloads, "new", "new/source.tar.gz") as (source, deps):
        assert 'VERSION = "new"' in (source / "harness.py").read_text()
        assert (deps / "installed.py").read_text() == "new"
    assert fetches == installs == ["old", "new"]
    assert (root / "submission-sub-fixed" / ".ready").read_text() == "new/source.tar.gz"


@pytest.mark.parametrize("restart", [False, True])
def test_same_reference_remains_a_cache_hit(sources, restart):
    make, _root, payloads, fetches, installs = sources
    cache = make()
    with acquire(cache, payloads, "old", "same/source.tar.gz"):
        pass
    if restart:
        cache = make()
    with acquire(cache, payloads, "old", "same/source.tar.gz"):
        pass
    assert fetches == installs == ["old"]


@pytest.mark.parametrize("marker", [b"", b"\xff", b"x" * 4097, b"x" * 1025])
def test_legacy_or_invalid_identity_rebuilds_on_reload(sources, marker):
    make, root, payloads, fetches, installs = sources
    cache = make()
    with acquire(cache, payloads, "old", "old/source.tar.gz"):
        pass
    ready = root / "submission-sub-fixed" / ".ready"
    os.chmod(ready, 0o600)
    ready.write_bytes(marker)
    os.chmod(ready, 0o400)
    cache = make()
    with acquire(cache, payloads, "new", "new/source.tar.gz") as (source, _deps):
        assert 'VERSION = "new"' in (source / "harness.py").read_text()
    assert fetches == installs == ["old", "new"]


def test_reference_is_compared_exactly_without_whitespace_normalization(sources):
    make, _root, payloads, fetches, _installs = sources
    cache = make()
    with acquire(cache, payloads, "old", "source.tar.gz"):
        pass
    with acquire(cache, payloads, "new", "source.tar.gz ") as (source, _deps):
        assert 'VERSION = "new"' in (source / "harness.py").read_text()
    assert fetches == ["old", "new"]


def test_changed_size_and_different_submission_keep_existing_miss_behavior(sources):
    make, _root, payloads, fetches, installs = sources
    cache = make()
    with acquire(cache, payloads, "old", "source.tar.gz"):
        pass
    with acquire(cache, payloads, "small", "source.tar.gz") as (source, _deps):
        assert 'VERSION = "new"' in (source / "harness.py").read_text()
    with acquire(cache, payloads, "new", "new/source.tar.gz", "sub-other"):
        pass
    assert fetches == installs == ["old", "small", "new"]


def test_new_identity_waits_for_active_source_without_deleting_it(sources, monkeypatch):
    make, _root, payloads, fetches, _installs = sources
    cache = make()
    waiting = threading.Event()
    wait = cache._condition.wait

    def observe_wait(*args, **kwargs):
        waiting.set()
        return wait(*args, **kwargs)

    monkeypatch.setattr(cache._condition, "wait", observe_wait)

    def new_run():
        with acquire(cache, payloads, "new", "new/source.tar.gz") as (source, deps):
            return (source / "harness.py").read_text(), (
                deps / "installed.py"
            ).read_text()

    with ThreadPoolExecutor(max_workers=1) as pool:
        with acquire(cache, payloads, "old", "old/source.tar.gz") as (source, _deps):
            pending = pool.submit(new_run)
            assert waiting.wait(2)
            assert not pending.done() and fetches == ["old"]
            assert 'VERSION = "old"' in (source / "harness.py").read_text()
            # Parallel users of the still-active identity are not blocked.
            with acquire(cache, payloads, "old", "old/source.tar.gz"):
                pass
        code, installed = pending.result(timeout=3)
    assert 'VERSION = "new"' in code and installed == "new"
    assert fetches == ["old", "new"] and not cache._in_use and not cache._building


def test_failed_preparation_releases_waiters_and_keeps_retry_source_bound(
    tmp_path, monkeypatch
):
    payload = archive("new")
    started = threading.Event()
    release = threading.Event()
    waiting = threading.Event()
    waiting_threads = set()
    calls = []

    def fetch(run_id, _token):
        calls.append(run_id)
        if run_id == "first":
            started.set()
            assert release.wait(3)
            raise runner.RunnerError("fixture download failed")
        return payload

    cache = runner.SourceCache(
        tmp_path / "sources", fetch, dependency_installer=lambda *_: None
    )
    wait = cache._condition.wait

    def observe_wait(*args, **kwargs):
        waiting_threads.add(threading.get_ident())
        if len(waiting_threads) == 2:
            waiting.set()
        return wait(*args, **kwargs)

    monkeypatch.setattr(cache._condition, "wait", observe_wait)

    def use(run_id):
        with cache.acquire(
            run_id, "fixture-token", "new/source.tar.gz", "sub-fixed", len(payload)
        ) as (source, _deps):
            return (source / "harness.py").read_text()

    with ThreadPoolExecutor(max_workers=3) as pool:
        first = pool.submit(use, "first")
        assert started.wait(2)
        second = pool.submit(use, "second")
        third = pool.submit(use, "third")
        assert waiting.wait(2)
        release.set()
        with pytest.raises(runner.RunnerError, match="fixture download failed"):
            first.result(timeout=3)
        assert 'VERSION = "new"' in second.result(timeout=3)
        assert 'VERSION = "new"' in third.result(timeout=3)
    assert calls[0] == "first" and len(calls) == 2
    assert calls[1] in {"second", "third"}
    assert not cache._building and not cache._in_use


@pytest.mark.parametrize("reference", ["sources/café.tar.gz", "\U00010400" * 1024])
def test_exact_utf8_reference_survives_reload_at_character_limit(sources, reference):
    make, root, payloads, fetches, _installs = sources
    cache = make()
    with acquire(cache, payloads, "old", reference):
        pass
    assert (root / "submission-sub-fixed" / ".ready").read_bytes() == reference.encode(
        "utf-8"
    )
    cache = make()
    with acquire(cache, payloads, "old", reference):
        pass
    assert fetches == ["old"]


def test_marker_read_is_bounded_before_decoding():
    requested = []

    class Stream(io.BytesIO):
        def read(self, size=-1):
            assert 0 < size <= 4097
            requested.append(size)
            return super().read(size)

    class Marker:
        def open(self, mode):
            assert mode == "rb"
            return Stream(b"x" * 100_000)

    assert runner.SourceCache._ready_source_ref(Marker()) is None
    assert len(requested) == 1
