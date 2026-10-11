"""Published result bytes join only while the same object GET is in flight."""

import asyncio
import concurrent.futures
import json
import threading

import httpx
import pytest

from lab_arena import contracts, service as service_module
from lab_arena.api import create_app
from lab_arena.service import ArenaService, _PublicResultInFlightReads, ServiceError
from tests.lab_arena.scoring_validator_attribution_test import (
    ROUND_ID, SUBMISSION_ID, _execution, _service,
)


def _observed_future(monkeypatch):
    joined = threading.Event()

    class ObservedFuture(concurrent.futures.Future):
        def result(self, timeout=None):
            joined.set()
            return super().result(timeout)

    monkeypatch.setattr(service_module, "Future", ObservedFuture)
    return joined


def test_two_concurrent_asgi_results_share_only_the_output_get(monkeypatch):
    joined = _observed_future(monkeypatch)
    document = json.dumps({
        "schema_version": contracts.OUTPUT_DOCUMENT_SCHEMA_VERSION,
        "companies": [],
    }).encode()

    class Objects:
        def __init__(self):
            self.entered = threading.Event()
            self.release = threading.Event()
            self.calls = 0

        def get_bounded(self, ref, max_bytes):
            assert (ref, max_bytes) == ("public-output", service_module.MAX_OUTPUT_BYTES)
            self.calls += 1
            self.entered.set()
            assert self.release.wait(3)
            return document

    execution = _execution("execute-public", 0)
    execution["output_ref"] = "public-output"
    service = _service([execution], public_positions={0})
    objects = Objects()
    service._objects = objects
    service._public_result_reads = _PublicResultInFlightReads(objects)
    checks = {"round": 0, "disclosure": 0}
    checks_lock = threading.Lock()
    original_round = service._round
    original_disclosure = service._public_icp_disclosure

    def counted_round(round_id):
        with checks_lock:
            checks["round"] += 1
        return original_round(round_id)

    def counted_disclosure(row):
        with checks_lock:
            checks["disclosure"] += 1
        return original_disclosure(row)

    service._round = counted_round
    service._public_icp_disclosure = counted_disclosure
    path = f"/arena/v1/rounds/{ROUND_ID}/results/{SUBMISSION_ID}"

    async def calls():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_app(service)),
            base_url="http://private",
        ) as client:
            first = asyncio.create_task(client.get(path))
            assert await asyncio.to_thread(objects.entered.wait, 3)
            second = asyncio.create_task(client.get(path))
            assert await asyncio.to_thread(joined.wait, 3)
            objects.release.set()
            return await asyncio.gather(first, second)

    first, second = asyncio.run(calls())
    assert first.status_code == second.status_code == 200
    assert first.content == second.content
    assert first.json()["outputs"]["execute-public"]["companies"] == []
    assert objects.calls == 1
    assert checks == {"round": 2, "disclosure": 2}
    # A later request rechecks the round and issues a fresh object GET.
    async def later():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_app(service)),
            base_url="http://private",
        ) as client:
            return await client.get(path)

    assert asyncio.run(later()).status_code == 200
    assert objects.calls == 2
    assert checks == {"round": 3, "disclosure": 3}


def test_concurrent_asgi_owner_error_fails_closed_then_retries_fresh(monkeypatch):
    joined = _observed_future(monkeypatch)
    document = json.dumps({
        "schema_version": contracts.OUTPUT_DOCUMENT_SCHEMA_VERSION,
        "companies": [],
    }).encode()

    class Objects:
        def __init__(self):
            self.entered = threading.Event()
            self.release = threading.Event()
            self.fail = True
            self.calls = 0

        def get_bounded(self, ref, max_bytes):
            assert (ref, max_bytes) == ("public-output", service_module.MAX_OUTPUT_BYTES)
            self.calls += 1
            self.entered.set()
            assert self.release.wait(3)
            if self.fail:
                raise FileNotFoundError("object missing")
            return document

    execution = _execution("execute-public", 0)
    execution["output_ref"] = "public-output"
    service = _service([execution], public_positions={0})
    objects = Objects()
    service._objects = objects
    service._public_result_reads = _PublicResultInFlightReads(objects)
    path = f"/arena/v1/rounds/{ROUND_ID}/results/{SUBMISSION_ID}"

    async def concurrent_failure():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_app(service)),
            base_url="http://private",
        ) as client:
            first = asyncio.create_task(client.get(path))
            assert await asyncio.to_thread(objects.entered.wait, 3)
            second = asyncio.create_task(client.get(path))
            assert await asyncio.to_thread(joined.wait, 3)
            objects.release.set()
            return await asyncio.gather(first, second)

    for response in asyncio.run(concurrent_failure()):
        assert response.status_code == 503
        assert response.headers["cache-control"] == "no-store"
        assert response.json() == {
            "status": "rejected", "code": "public_output_unavailable",
        }
    assert objects.calls == 1
    assert not service._public_result_reads._pending

    objects.fail = False

    async def retry():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_app(service)),
            base_url="http://private",
        ) as client:
            return await client.get(path)

    response = asyncio.run(retry())
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert response.json()["outputs"]["execute-public"]["companies"] == []
    assert objects.calls == 2


def test_later_change_missing_and_oversize_are_fresh_failures():
    class Objects:
        def __init__(self):
            self.values = {"ref": b"first"}
            self.calls = []

        def get(self, ref):
            self.calls.append(("get", ref))
            return self.values[ref]

        def get_bounded(self, ref, max_bytes):
            self.calls.append(("get_bounded", ref, max_bytes))
            value = self.values[ref]
            if len(value) > max_bytes:
                raise ValueError("oversize")
            return value

    objects = Objects()
    reads = _PublicResultInFlightReads(objects)
    assert reads.get_bounded("ref", 10) == b"first"
    objects.values["ref"] = b"changed"
    assert reads.get_bounded("ref", 10) == b"changed"
    del objects.values["ref"]
    with pytest.raises(KeyError):
        reads.get_bounded("ref", 10)
    objects.values["ref"] = b"too long"
    with pytest.raises(ValueError, match="oversize"):
        reads.get_bounded("ref", 2)
    assert reads.get_bounded("ref", 10) == b"too long"
    assert len(objects.calls) == 5
    assert not reads._pending


def test_benchmark_validation_joins_only_its_object_get(monkeypatch):
    joined = _observed_future(monkeypatch)
    row = {
        "benchmark_ref": "public-benchmark",
        "configuration_doc": {"stage_1_icp_count": 1, "stage_2_icp_count": 1},
    }
    icps = [{"position": 0}, {"position": 1}]

    class Objects:
        def __init__(self):
            self.entered = threading.Event()
            self.release = threading.Event()
            self.calls = 0

        def get(self, ref):
            assert ref == "public-benchmark"
            self.calls += 1
            self.entered.set()
            assert self.release.wait(3)
            return json.dumps({
                "schema_version": "leadpoet.lab_arena.benchmark.v1",
                "round_id": ROUND_ID,
                "icps": icps,
            }).encode()

    objects = Objects()
    reads = _PublicResultInFlightReads(objects)
    service = object.__new__(ArenaService)
    service._objects = objects

    def read():
        return service._benchmark_icps_from_row(
            ROUND_ID, row, get_object=reads.get,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(read)
        assert objects.entered.wait(3)
        second = pool.submit(read)
        assert joined.wait(3)
        objects.release.set()
        assert first.result(timeout=3) == second.result(timeout=3) == icps
    assert objects.calls == 1


def test_method_ref_and_bound_are_separate_inflight_keys():
    barrier = threading.Barrier(4, timeout=3)

    class Objects:
        def __init__(self):
            self.calls = []
            self.lock = threading.Lock()

        def get(self, ref):
            with self.lock:
                self.calls.append(("get", ref, None))
            barrier.wait()
            return b"get"

        def get_bounded(self, ref, max_bytes):
            with self.lock:
                self.calls.append(("get_bounded", ref, max_bytes))
            barrier.wait()
            return b"bounded"

    objects = Objects()
    reads = _PublicResultInFlightReads(objects)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [
            pool.submit(reads.get, "ref"),
            pool.submit(reads.get_bounded, "ref", 1),
            pool.submit(reads.get_bounded, "ref", 2),
            pool.submit(reads.get_bounded, "other", 1),
        ]
        assert [future.result(timeout=3) for future in futures] == [
            b"get", b"bounded", b"bounded", b"bounded",
        ]
    assert set(objects.calls) == {
        ("get", "ref", None), ("get_bounded", "ref", 1),
        ("get_bounded", "ref", 2), ("get_bounded", "other", 1),
    }


def test_owner_baseexception_releases_waiter_then_retry_reads_again(monkeypatch):
    joined = _observed_future(monkeypatch)

    class FailedRead(BaseException):
        pass

    class Objects:
        def __init__(self):
            self.entered = threading.Event()
            self.release = threading.Event()
            self.fail = True
            self.calls = 0

        def get(self, ref):
            self.calls += 1
            self.entered.set()
            assert self.release.wait(3)
            if self.fail:
                raise FailedRead("source failed")
            return b"recovered"

    objects = Objects()
    reads = _PublicResultInFlightReads(objects)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(reads.get, "ref")
        assert objects.entered.wait(3)
        second = pool.submit(reads.get, "ref")
        assert joined.wait(3)
        objects.release.set()
        for call in (first, second):
            with pytest.raises(FailedRead, match="source failed"):
                call.result(timeout=3)
    assert objects.calls == 1
    assert not reads._pending
    objects.fail = False
    assert reads.get("ref") == b"recovered"
    assert objects.calls == 2


def test_unpublished_and_undisclosed_results_do_not_fetch_objects():
    class Objects:
        def get(self, _ref):
            pytest.fail("private benchmark read")

        def get_bounded(self, _ref, _max_bytes):
            pytest.fail("private output read")

    execution = _execution("execute-private", 0)
    execution["output_ref"] = "private-output"
    for positions, status in (({0}, "stage1"), (None, "published")):
        service = _service([execution], public_positions=positions, status=status)
        objects = Objects()
        service._objects = objects
        service._public_result_reads = _PublicResultInFlightReads(objects)
        if status != "published":
            with pytest.raises(ServiceError, match="results_not_public"):
                service.public_results(ROUND_ID, SUBMISSION_ID)
        else:
            result = service.public_results(ROUND_ID, SUBMISSION_ID)
            assert result["outputs"] == {}
            assert result["public_icp_status"] == "pending"
