"""Bounded review recovery through replacement, scoring, and publication."""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pytest

from lab_arena import broker, code_review, contracts, source_bundle
from lab_arena.service import ServiceError
from lab_arena.store import new_lease_token
from tests.lab_arena.test_lab_arena_service_round import (
    CANARY_KEYS,
    CANARY_OPENROUTER_KEY,
    CANARY_OPENROUTER_MANAGEMENT_KEY,
    FakeProviderTransport,
    Harness,
    connect,
    database,
    flavor_source_archive,
    keypair,
)


def _iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _signed_reserve(harness: Harness, round_id: str, flavor: str, miner_label: str):
    """Reserve an upload through the same signed API used by a real miner."""

    miner = keypair("svc-miner-" + miner_label)
    payload = flavor_source_archive(flavor)
    facts = source_bundle.validate_source_archive(payload)
    request = contracts.build_signed_request(
        scope=contracts.SCOPE_SUBMISSION_PRESIGN,
        round_id=round_id,
        hotkey=miner.ss58_address,
        body={
            "source_size_bytes": facts["source_size_bytes"],
            "consent": {"public_rerun": True},
        },
        timestamp=int(harness.clock().timestamp()),
        sign_message=lambda message: miner.sign(message.encode()).hex(),
    )
    target = harness.service.handle_submission_presign(request)
    harness.flavors[target["submission_id"]] = flavor
    return miner, payload, facts, target


def _signed_finalize(harness: Harness, round_id: str, reserved) -> str:
    """Upload and finalize one prior signed reservation."""

    miner, payload, facts, target = reserved
    harness.objects.put(target["source_ref"], payload)
    request = contracts.build_signed_request(
        scope=contracts.SCOPE_SUBMISSION_FINALIZE,
        round_id=round_id,
        hotkey=miner.ss58_address,
        body={
            "submission_id": target["submission_id"],
            "source_ref": target["source_ref"],
            "source_size_bytes": facts["source_size_bytes"],
            "credentials": {
                "openrouter_api_key": CANARY_OPENROUTER_KEY,
                "openrouter_management_key": CANARY_OPENROUTER_MANAGEMENT_KEY,
                "deepline_api_key": CANARY_KEYS["deepline"],
            },
        },
        timestamp=int(harness.clock().timestamp()),
        sign_message=lambda message: miner.sign(message.encode()).hex(),
    )
    result = harness.service.handle_submission_finalize(
        target["submission_id"], request
    )
    assert result["status"] == "accepted"
    return str(target["submission_id"])


def _set_round_schedule(connect, round_id: str, field: str, value: str) -> None:
    """Retime only this disposable round and restore its write guard."""

    connection = connect()
    try:
        with connection, connection.cursor() as cursor:
            cursor.execute(
                "ALTER TABLE public.lab_arena_rounds "
                "DISABLE TRIGGER lab_arena_rounds_write_once"
            )
            cursor.execute(
                "UPDATE public.lab_arena_rounds SET configuration_doc = "
                "pg_catalog.jsonb_set(configuration_doc, %s::TEXT[], "
                "pg_catalog.to_jsonb(%s::TEXT)) WHERE round_id=%s",
                (["schedule", field], value, round_id),
            )
            assert cursor.rowcount == 1
            cursor.execute(
                "ALTER TABLE public.lab_arena_rounds "
                "ENABLE TRIGGER lab_arena_rounds_write_once"
            )
    finally:
        connection.close()


def _make_review_retry_ready(connect, submission_id: str) -> None:
    """Move one disposable error timestamp beyond the largest test backoff."""

    connection = connect()
    try:
        with connection, connection.cursor() as cursor:
            cursor.execute(
                "UPDATE public.lab_arena_submissions "
                "SET code_review_started_at = "
                "pg_catalog.date_trunc('second', pg_catalog.clock_timestamp()) "
                "- INTERVAL '1 hour' WHERE submission_id=%s "
                "AND code_review_status='error'",
                (submission_id,),
            )
            assert cursor.rowcount == 1
    finally:
        connection.close()


class RecoveryReviewTransport(FakeProviderTransport):
    """Fail selected review calls without making a real provider request."""

    def __init__(self) -> None:
        super().__init__()
        self.attempts: dict[str, int] = {}
        self.first_dispatch = threading.Event()
        self.release_first_dispatch = threading.Event()

    @staticmethod
    def _flavor(request: dict) -> str:
        submitted = json.loads(request["messages"][1]["content"])
        return next(
            str(item["content"])
            for item in submitted["submission_files"]
            if item["path"] == "flavor.txt"
        )

    def send(self, **request):
        body = json.loads(request["body"].decode("utf-8"))
        if body.get("model") != code_review.DEFAULT_REVIEW_MODEL:
            return super().send(**request)
        flavor = self._flavor(body)
        attempt = self.attempts.get(flavor, 0) + 1
        self.attempts[flavor] = attempt
        if flavor == "RecoveredReplacement":
            if attempt == 1:
                self.first_dispatch.set()
                assert self.release_first_dispatch.wait(timeout=5)
            if attempt in (1, 3):
                return broker.ProviderResponse(
                    404 if attempt == 1 else 503,
                    {"content-type": "application/json"},
                    b'{"error":"temporary"}',
                )
            if attempt in (2, 4):
                raise broker.ProviderTransportError("controlled transport failure")
        if flavor == "PermanentAuthFailure":
            return broker.ProviderResponse(
                401,
                {"content-type": "application/json"},
                b'{"error":"invalid key"}',
            )
        return super().send(**request)


def _ledger_kinds(harness: Harness, submission_id: str) -> list[str]:
    return [
        str(row["entry_kind"])
        for row in harness.service.store.list_ledger(submission_id=submission_id)
        if row.get("operation_id") == "openrouter.code_review"
    ]


def test_transient_review_recovers_after_replacement_and_only_it_is_evaluated(
    connect, tmp_path
):
    harness = Harness(connect, tmp_path, challengers=[], runners=["alpha"])
    harness.chain.epoch = 39016
    transport = RecoveryReviewTransport()
    reviewer = harness.service.config.code_reviewer
    reviewer._transport = transport
    harness.review_transport = transport
    assert harness.service.store.code_review_schema()["max_transient_attempts"] == 6

    # A UTC-midnight round fixes the source replacement boundary at 23:00.
    now = datetime.now(timezone.utc)
    harness.clock.now = now
    cutoff = (now + timedelta(days=1)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    # The test later brings the cutoff into the current UTC day so PostgreSQL
    # can exercise retry timing without sleep. Keep the daily ID on that day.
    round_id = "arena-" + now.date().isoformat()
    configuration = harness.service.create_round(cutoff, round_id=round_id)
    harness.round_id = round_id
    replacement_freeze = harness.service._submission_replacement_cutoff(
        {"configuration_doc": configuration}
    )
    assert (replacement_freeze.hour, replacement_freeze.minute) == (23, 0)

    original_reserved = _signed_reserve(
        harness, round_id, "SupersededOriginal", "review-recovery"
    )
    original_id = _signed_finalize(harness, round_id, original_reserved)
    original = harness.service.store.get_submission(original_id)
    original_source_ref = original["source_ref"]
    original_source = harness.objects.get(original_source_ref)

    replacement_reserved = _signed_reserve(
        harness, round_id, "RecoveredReplacement", "review-recovery"
    )
    replacement_id = _signed_finalize(harness, round_id, replacement_reserved)
    permanent_id = _signed_finalize(
        harness,
        round_id,
        _signed_reserve(
            harness, round_id, "PermanentAuthFailure", "permanent-review-failure"
        ),
    )
    unadmitted = _signed_reserve(
        harness, round_id, "NeverAdmitted", "never-admitted"
    )[3]["submission_id"]

    original = harness.service.store.get_submission(original_id)
    replacement = harness.service.store.get_submission(replacement_id)
    assert (
        original["status"],
        original["rejection_rule"],
        original["replaced_by_submission_id"],
    ) == ("rejected", "source_replaced", replacement_id)
    assert replacement["replaces_submission_id"] == original_id
    assert replacement["source_ref"] != original_source_ref
    assert harness.objects.get(original_source_ref) == original_source

    # Neither the worker nor SQL can spend before the 23:00 source freeze.
    assert harness.service.review_pending_submissions() == {"reviewed": 0}
    deferred = harness.service.store.begin_submission_review(
        replacement_id,
        replacement["miner_hotkey"],
        new_lease_token(),
        50_000,
        code_review.DEFAULT_REVIEW_MODEL,
        3,
        replacement["source_size_bytes"],
    )
    assert deferred["status"] == "deferred"
    assert harness.service.store.list_ledger(submission_id=replacement_id) == []

    # Move only the disposable round beyond replacement freeze. The benchmark
    # deadline remains open while review retries are exercised.
    review_cutoff = datetime.now(timezone.utc) + timedelta(minutes=30)
    _set_round_schedule(
        connect,
        round_id,
        "submission_open",
        _iso(review_cutoff - timedelta(days=1)),
    )
    _set_round_schedule(connect, round_id, "submission_cutoff", _iso(review_cutoff))
    original_deadline = configuration["schedule"]["benchmark_deadline"]

    # Two workers race the same pending row. One claim dispatches attempt 1;
    # the other sees the durable lease and creates no duplicate ledger spend.
    pending = harness.service.store.get_submission(replacement_id)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(reviewer.review, pending)
        assert transport.first_dispatch.wait(timeout=5)
        second = pool.submit(reviewer.review, pending)
        duplicate = second.result(timeout=5)
        transport.release_first_dispatch.set()
        failed = first.result(timeout=5)
    assert duplicate["status"] == "busy"
    assert failed["status"] == "error"
    assert transport.attempts["RecoveredReplacement"] == 1
    first_error = harness.service.store.get_submission(replacement_id)[
        "code_review_doc"
    ]
    assert first_error["error_code"] == "code_review_provider_unavailable"
    assert first_error["provider_http_status"] == 404
    assert first_error["retryable"] is True
    first_kinds = _ledger_kinds(harness, replacement_id)
    assert first_kinds.count("reservation") == 1
    assert first_kinds.count("dispatch") == 1

    immediate = reviewer.review(harness.service.store.get_submission(replacement_id))
    assert immediate["status"] == "backoff"
    assert _ledger_kinds(harness, replacement_id) == first_kinds

    # A ready retry is still refused at the benchmark deadline and does not
    # consume an attempt. Restore the real deadline before recovery continues.
    _make_review_retry_ready(connect, replacement_id)
    _set_round_schedule(
        connect,
        round_id,
        "benchmark_deadline",
        _iso(datetime.now(timezone.utc) - timedelta(seconds=1)),
    )
    at_deadline = harness.service.store.begin_submission_review(
        replacement_id,
        replacement["miner_hotkey"],
        new_lease_token(),
        50_000,
        code_review.DEFAULT_REVIEW_MODEL,
        3,
        replacement["source_size_bytes"],
    )
    assert at_deadline["status"] == "deadline"
    assert harness.service.store.get_submission(replacement_id)[
        "code_review_attempts"
    ] == 1
    assert _ledger_kinds(harness, replacement_id) == first_kinds
    _set_round_schedule(
        connect, round_id, "benchmark_deadline", original_deadline
    )

    # Attempts 2-4 are retryable transport/503 failures. Attempt 5 passes,
    # proving recovery beyond the former three-attempt limit and below cap 6.
    outcomes = []
    for _attempt in range(2, 6):
        _make_review_retry_ready(connect, replacement_id)
        outcome = reviewer.review(
            harness.service.store.get_submission(replacement_id)
        )
        outcomes.append(outcome["status"])
    assert outcomes == ["error", "error", "error", "passed"]
    recovered = harness.service.store.get_submission(replacement_id)
    assert recovered["code_review_status"] == "passed"
    assert recovered["code_review_attempts"] == 5
    assert recovered["code_review_doc"]["cost_status"] == "settled"
    assert recovered["code_review_doc"]["cost_microusd"] == 0
    assert recovered["code_review_doc"]["review_cost_microusd"] > 0
    assert transport.attempts["RecoveredReplacement"] == 5
    recovered_kinds = _ledger_kinds(harness, replacement_id)
    assert recovered_kinds.count("reservation") == 5
    assert recovered_kinds.count("dispatch") == 5
    assert reviewer.review(recovered)["status"] == "existing"
    assert transport.attempts["RecoveredReplacement"] == 5
    assert _ledger_kinds(harness, replacement_id) == recovered_kinds

    permanent = reviewer.review(
        harness.service.store.get_submission(permanent_id)
    )
    assert permanent["status"] == "error"
    permanent_row = harness.service.store.get_submission(permanent_id)
    assert permanent_row["code_review_doc"]["error_code"] == (
        "code_review_provider_authentication"
    )
    assert permanent_row["code_review_doc"]["retryable"] is False
    assert reviewer.review(permanent_row)["status"] == "exhausted"
    assert transport.attempts["PermanentAuthFailure"] == 1
    assert _ledger_kinds(harness, permanent_id).count("dispatch") == 1

    assert harness.service.store.list_ledger(submission_id=original_id) == []
    assert harness.service.store.list_ledger(submission_id=unadmitted) == []

    # Freeze and publish the full round. Only the selected replacement enters
    # evaluation beside the baseline, and it receives all 20 ICP assignments.
    harness.clock.advance_to(_iso(review_cutoff))
    committed = harness.service.advance_round(round_id)
    assert committed["status"] == "ok"
    participants = harness.service.store.get_round(round_id)["participants"]
    challenger_ids = {
        row["submission_id"] for row in participants if not row["is_king"]
    }
    assert challenger_ids == {replacement_id}
    assert harness.service.store.get_submission(replacement_id)["status"] == "frozen"
    assert harness.service.store.get_submission(original_id)["status"] == "rejected"
    assert harness.service.store.get_submission(permanent_id)["status"] == "rejected"

    harness.advance_until("published", runners=1)
    execute_runs = harness.service.store.list_runs(
        round_id, submission_id=replacement_id, kind="execute"
    )
    assert len(execute_runs) == contracts.BENCHMARK_ICP_COUNT == 20
    assert {run["icp_position"] for run in execute_runs} == set(range(20))
    assert harness.service.store.list_runs(
        round_id, submission_id=original_id
    ) == []
    assert harness.service.store.list_runs(
        round_id, submission_id=permanent_id
    ) == []
    assert harness.service.store.list_runs(
        round_id, submission_id=unadmitted
    ) == []
    public_results = harness.service.public_results(round_id, replacement_id)
    assert len(
        public_results["scores"]["stage_1"]
        + public_results["scores"]["stage_2"]
    ) == 20
    assert public_results["submission_scores"]["final"] is not None

    public = harness.service.public_submissions(round_id)["submissions"]
    public_by_id = {row["submission_id"]: row for row in public}
    assert replacement_id in public_by_id
    assert original_id not in public_by_id
    assert unadmitted not in public_by_id
    excluded = public_by_id[permanent_id]
    assert excluded["status"] == "review_failed"
    assert excluded["stage1_score"] is None
    assert excluded["final_score"] is None
    assert excluded["is_champion"] is False
    assert excluded["code"]["available"] is False
    assert excluded["code"]["url"] is None
    assert excluded["code_review"]["status"] == "error"
    assert excluded["code_review"]["retryable"] is False
    assert excluded["code_review"]["error_code"] == (
        "code_review_provider_authentication"
    )
    assert "source_ref" not in excluded

    object_reads = []
    real_get_bounded = harness.objects.get_bounded
    harness.objects.get_bounded = lambda *args, **kwargs: (
        object_reads.append(args) or real_get_bounded(*args, **kwargs)
    )
    try:
        with pytest.raises(ServiceError) as source_error:
            harness.service.public_submission_code(permanent_id)
    finally:
        harness.objects.get_bounded = real_get_bounded
    assert (source_error.value.code, source_error.value.status) == (
        "source_not_public",
        403,
    )
    assert object_reads == []

    run_reads = []
    real_list_runs = harness.service.store.list_runs
    harness.service.store.list_runs = lambda *args, **kwargs: (
        run_reads.append(args) or real_list_runs(*args, **kwargs)
    )
    try:
        with pytest.raises(ServiceError) as results_error:
            harness.service.public_results(round_id, permanent_id)
    finally:
        harness.service.store.list_runs = real_list_runs
    assert (results_error.value.code, results_error.value.status) == (
        "submission_missing",
        404,
    )
    assert run_reads == []
