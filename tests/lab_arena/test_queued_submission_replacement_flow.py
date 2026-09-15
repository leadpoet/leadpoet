"""Complete queued replacement flow on disposable PostgreSQL and local objects.

Installed admission, review, execution, scoring, and publication code is used.
A fixture-only schedule shift makes database cutoff checks deterministic without
waiting one hour or changing any production clock behavior.
"""
from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from lab_arena import contracts, icp_disclosure, service as svc, source_bundle
from tests.lab_arena.lab_arena_pg_harness import CURRENT_SERVICE_MIGRATIONS, database_with_lab_arena_migration
from tests.lab_arena.test_lab_arena_service_round import (
    Harness, FixtureObjectStore, CANARY_DEEPLINE_KEY, CANARY_OPENROUTER_KEY,
    CANARY_OPENROUTER_MANAGEMENT_KEY, _run_stage_one_to_scoring,
    flavor_source_archive, keypair,
)


class ChecksumObjects(FixtureObjectStore):
    def presign_put(self, ref, *, size_bytes, content_type, expires_seconds, source_content_md5=None):
        result = super().presign_put(ref, size_bytes=size_bytes, content_type=content_type, expires_seconds=expires_seconds)
        if source_content_md5:
            result["upload_headers"]["content-md5"] = source_content_md5
        return result


@pytest.fixture(scope="module")
def database():
    yield from database_with_lab_arena_migration(CURRENT_SERVICE_MIGRATIONS)


@pytest.fixture(scope="module")
def connect(database):
    psycopg2, dsn = database
    return lambda: psycopg2.connect(**dsn)


def signed(h, miner, round_id, scope, body):
    return contracts.build_signed_request(
        scope=scope, round_id=round_id, hotkey=miner.ss58_address, body=body,
        timestamp=int(h.clock().timestamp()),
        sign_message=lambda message: miner.sign(message.encode()).hex(),
    )


def reserve(h, round_id, miner, source):
    checksum = base64.b64encode(hashlib.md5(source, usedforsecurity=False).digest()).decode()
    fact = source_bundle.validate_source_archive(source)
    row = h.service.handle_submission_presign(signed(h, miner, round_id, contracts.SCOPE_SUBMISSION_PRESIGN, {
        "source_size_bytes": fact["source_size_bytes"], "source_content_md5": checksum,
        "consent": {"public_rerun": True},
    }))
    assert row["status"] == "upload_ready"
    assert row["upload_headers"]["content-md5"] == checksum
    h.objects.put(row["source_ref"], source)
    return row


def finalize(h, round_id, miner, row):
    return h.service.handle_submission_finalize(row["submission_id"], signed(
        h, miner, round_id, contracts.SCOPE_SUBMISSION_FINALIZE, {
            "submission_id": row["submission_id"], "source_ref": row["source_ref"],
            "source_size_bytes": len(h.objects.get(row["source_ref"])),
            "credentials": {
                "openrouter_api_key": CANARY_OPENROUTER_KEY,
                "openrouter_management_key": CANARY_OPENROUTER_MANAGEMENT_KEY,
                "deepline_api_key": CANARY_DEEPLINE_KEY,
            },
        },
    ))


def expect_error(call, code, status):
    with pytest.raises(svc.ServiceError) as caught:
        call()
    assert (caught.value.code, caught.value.status) == (code, status)


def retime_disposable_round(connect, h, round_id, cutoff):
    configuration = dict(h.service.store.get_round(round_id)["configuration_doc"])
    configuration["schedule"] = h.service.build_schedule(cutoff)
    contracts.validate_round_configuration(configuration)
    connection = connect()
    try:
        with connection.cursor() as cursor:
            # The fixture change is transactional and restores the write-once
            # trigger before any service call resumes.
            cursor.execute("ALTER TABLE public.lab_arena_rounds DISABLE TRIGGER lab_arena_rounds_write_once")
            cursor.execute("UPDATE public.lab_arena_rounds SET configuration_doc=%s::jsonb WHERE round_id=%s AND status='open'", (json.dumps(configuration), round_id))
            assert cursor.rowcount == 1
            cursor.execute("ALTER TABLE public.lab_arena_rounds ENABLE TRIGGER lab_arena_rounds_write_once")
        connection.commit()
    finally:
        connection.close()
    return configuration["schedule"]


def test_latest_valid_replacement_is_the_only_evaluated_source(connect, tmp_path):
    h = Harness(connect, tmp_path, challengers=[], runners=["alpha"])
    h.objects = ChecksumObjects(h.objects_root)
    h.service = h.build_service()
    # The tested flow is faster than the production per-hotkey throttle.
    h.service._submission_request_limiter = SimpleNamespace(check=lambda _hotkey: SimpleNamespace(allowed=True))
    h.chain.epoch = 39871
    h.service.config.defaults = replace(h.service.config.defaults, benchmark_disclosure_from="2026-01-01T00:00:00Z")
    round_id = "arena-2026-11-01"
    h.round_id = round_id
    configuration = h.service.create_round(datetime.now(timezone.utc) + timedelta(hours=12), round_id=round_id)
    assert configuration["benchmark_disclosure_policy"] == icp_disclosure.DELAYED_DISCLOSURE_POLICY
    expected_freeze = (datetime.fromisoformat(configuration["schedule"]["submission_cutoff"].replace("Z", "+00:00")) - timedelta(hours=1)).isoformat().replace("+00:00", "Z")
    assert h.service.public_round(round_id)["submission_replacement_cutoff"] == expected_freeze
    assert h.service.public_current()["open_round"]["submission_replacement_cutoff"] == expected_freeze
    h.clock.now = datetime.now(timezone.utc)
    miner = keypair("queued-flow-owner")
    newcomer = keypair("queued-flow-new-owner")

    source_a = flavor_source_archive("SourceA")
    a = reserve(h, round_id, miner, source_a)
    assert finalize(h, round_id, miner, a)["status"] == "accepted"
    assert h.service.store.get_submission(a["submission_id"])["status"] == "accepted"
    assert h.service.review_pending_submissions()["reviewed"] == 0
    assert not h.review_transport.review_requests

    bad = reserve(h, round_id, miner, flavor_source_archive("NoLicense", include_license=False))
    with pytest.raises(svc.ServiceError) as invalid:
        finalize(h, round_id, miner, bad)
    assert invalid.value.status == 400 and invalid.value.code.startswith("submission_rejected:")
    assert h.service.store.get_submission(bad["submission_id"])["status"] == "rejected"
    assert h.service.store.get_submission(a["submission_id"])["status"] == "accepted"
    assert h.objects.get(a["source_ref"]) == source_a

    source_b = flavor_source_archive("SourceB")
    b = reserve(h, round_id, miner, source_b)
    assert b["submission_id"] not in {a["submission_id"], bad["submission_id"]}
    assert h.service.store.get_submission(a["submission_id"])["status"] == "accepted"
    assert h.service.store.get_submission(b["submission_id"])["status"] == "uploading"
    assert finalize(h, round_id, miner, b)["status"] == "accepted"
    assert h.service.store.get_submission(b["submission_id"])["status"] == "accepted"
    replaced = h.service.store.get_submission(a["submission_id"])
    assert (replaced["status"], replaced["rejection_rule"]) == ("rejected", "source_replaced")
    expect_error(lambda: finalize(h, round_id, miner, a), "submission_superseded", 409)
    assert h.objects.get(b["source_ref"]) == source_b
    assert h.service.review_pending_submissions()["reviewed"] == 0
    assert not h.review_transport.review_requests

    pending = reserve(h, round_id, miner, flavor_source_archive("PendingC"))
    assert h.service.store.get_submission(b["submission_id"])["status"] == "accepted"
    # The SQL clock is now beyond this fixture round's new one-hour boundary.
    schedule = retime_disposable_round(connect, h, round_id, datetime.now(timezone.utc) + timedelta(seconds=30))
    freeze_at = datetime.fromisoformat(schedule["submission_cutoff"].replace("Z", "+00:00")) - timedelta(hours=1)
    h.clock.now = freeze_at
    expect_error(lambda: reserve(h, round_id, miner, flavor_source_archive("LateDifferent")), "submission_replacement_closed", 409)
    expect_error(lambda: finalize(h, round_id, miner, pending), "submission_replacement_closed", 409)
    assert h.service.store.get_submission(b["submission_id"])["status"] == "accepted"
    first = reserve(h, round_id, newcomer, flavor_source_archive("FirstTime"))
    assert finalize(h, round_id, newcomer, first)["status"] == "accepted"
    # Review claims open at the replacement freeze (23), after the winning
    # source is fixed and before the benchmark admission cutoff (00).
    assert h.service.review_pending_submissions()["reviewed"] == 2
    assert len(h.review_transport.review_requests) == 2
    expect_error(lambda: h.service.public_benchmark(round_id), "benchmark_not_public", 403)

    # Advance only the disposable round's schedule. The installed SQL review
    # and commitment guards observe real elapsed database time.
    schedule = retime_disposable_round(connect, h, round_id, datetime.now(timezone.utc) - timedelta(seconds=2))
    h.clock.now = datetime.now(timezone.utc)
    expect_error(lambda: reserve(h, round_id, keypair("queued-flow-too-late"), flavor_source_archive("TooLate")), "submission_window_closed", 409)
    assert h.service.review_pending_submissions()["reviewed"] == 0
    assert len(h.review_transport.review_requests) == 2
    assert h.service.store.get_submission(a["submission_id"])["code_review_status"] != "passed"
    assert h.service.store.get_submission(b["submission_id"])["code_review_status"] == "passed"
    assert h.service.store.get_submission(first["submission_id"])["code_review_status"] == "passed"
    assert h.service.advance_round(round_id)["status"] == "ok"
    committed = h.service.store.get_round(round_id)
    selected = {item["submission_id"] for item in committed["participants"]}
    assert {b["submission_id"], first["submission_id"]} <= selected
    assert a["submission_id"] not in selected and pending["submission_id"] not in selected
    assert h.service.store.get_submission(b["submission_id"])["status"] == "frozen"
    assert h.service.store.get_submission(pending["submission_id"])["status"] == "rejected"
    assert committed["configuration_doc"]["benchmark_disclosure_policy"] == icp_disclosure.DELAYED_DISCLOSURE_POLICY
    expect_error(lambda: h.service.public_benchmark(round_id), "benchmark_not_public", 403)

    for item in committed["participants"]:
        h.flavors.setdefault(item["submission_id"], "PublicBaseline")
    h.flavors[b["submission_id"]] = "SourceB"
    h.flavors[first["submission_id"]] = "FirstTime"
    _run_stage_one_to_scoring(h, len(selected), runners=1)
    h.advance_until("published", runners=1)
    publication = h.service.store.get_round(round_id)
    assert publication["status"] == "published"
    all_runs = h.service.store.list_runs(round_id)
    assert {run["submission_id"] for run in all_runs} <= selected
    assert not {a["submission_id"], pending["submission_id"]} & {run["submission_id"] for run in all_runs}
    execute_b = h.service.store.list_runs(round_id, submission_id=b["submission_id"], kind="execute")
    assert len(execute_b) == contracts.BENCHMARK_ICP_COUNT
    assert all(run["per_icp_score"] is not None for run in execute_b)
    ranking_ids = {item["submission_id"] for item in publication["publication_doc"]["final_ranking"]}
    assert b["submission_id"] in ranking_ids and a["submission_id"] not in ranking_ids
    expect_error(lambda: h.service.public_benchmark(round_id), "benchmark_not_public", 403)
    h.clock.now = datetime.fromisoformat(schedule["submission_cutoff"].replace("Z", "+00:00")) + timedelta(hours=24, seconds=1)
    public = h.service.public_benchmark(round_id)
    assert public["disclosure_policy"] == icp_disclosure.DELAYED_DISCLOSURE_POLICY
    assert len(public["icps"]) == contracts.BENCHMARK_ICP_COUNT
    results = h.service.public_results(round_id, b["submission_id"])
    assert len(results["scores"]["stage_1"] + results["scores"]["stage_2"]) == contracts.BENCHMARK_ICP_COUNT
    assert len(results["outputs"]) == contracts.BENCHMARK_ICP_COUNT
    preview = h.service.public_submission_code(b["submission_id"])
    assert {file["path"]: file["content"] for file in preview["files"]}["flavor.txt"] == "SourceB"
    expect_error(lambda: h.service.public_submission_code(a["submission_id"]), "source_not_public", 403)
    expect_error(lambda: h.service.public_results(round_id, a["submission_id"]), "submission_missing", 404)

    # This miner can start a new round; its prior replacement is round-local.
    h.clock.now = datetime.now(timezone.utc)
    next_id = "arena-2026-11-02"
    h.service.create_round(datetime.now(timezone.utc) + timedelta(hours=12), round_id=next_id)
    fresh = reserve(h, next_id, miner, flavor_source_archive("NextRound"))
    assert fresh["submission_id"] not in {a["submission_id"], b["submission_id"]}
    assert finalize(h, next_id, miner, fresh)["status"] == "accepted"
    assert h.service.store.get_submission(fresh["submission_id"])["round_id"] == next_id
    assert h.service.store.get_submission(b["submission_id"])["round_id"] == round_id
    h.service.cancel(next_id, "operator")
