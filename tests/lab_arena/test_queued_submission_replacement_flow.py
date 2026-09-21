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
from fastapi.testclient import TestClient
from lab_arena import contact_policy, contracts, credentials as credentials_module, icp_disclosure, service as svc, source_bundle
from lab_arena.api import create_app
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


def reserve(h, round_id, miner, source, *, uploaded_source=None):
    checksum = base64.b64encode(hashlib.md5(source, usedforsecurity=False).digest()).decode()
    fact = source_bundle.validate_source_archive(source)
    row = h.service.handle_submission_presign(signed(h, miner, round_id, contracts.SCOPE_SUBMISSION_PRESIGN, {
        "source_size_bytes": fact["source_size_bytes"], "source_content_md5": checksum,
        "consent": {"public_rerun": True},
    }))
    assert row["status"] == "upload_ready"
    assert row["upload_headers"]["content-md5"] == checksum
    h.objects.put(row["source_ref"], source if uploaded_source is None else uploaded_source)
    return row


def finalize(h, round_id, miner, row, *, credentials=None):
    credentials = credentials or {
        "openrouter_api_key": CANARY_OPENROUTER_KEY,
        "openrouter_management_key": CANARY_OPENROUTER_MANAGEMENT_KEY,
        "deepline_api_key": CANARY_DEEPLINE_KEY,
    }
    return h.service.handle_submission_finalize(row["submission_id"], signed(
        h, miner, round_id, contracts.SCOPE_SUBMISSION_FINALIZE, {
            "submission_id": row["submission_id"], "source_ref": row["source_ref"],
            "source_size_bytes": len(h.objects.get(row["source_ref"])),
            "credentials": credentials,
        },
    ))


def expect_error(call, code, status):
    with pytest.raises(svc.ServiceError) as caught:
        call()
    assert (caught.value.code, caught.value.status) == (code, status)


def expect_replacement_limit(h, round_id, miner, source):
    expect_error(
        lambda: reserve(h, round_id, miner, source),
        "submission_replacement_limit_reached", 409,
    )


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


def test_one_replacement_attempt_per_miner_preserves_fallbacks_and_evaluates_only_winners(
    connect, tmp_path, monkeypatch,
):
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
    assert configuration["benchmark_disclosure_policy"] == icp_disclosure.CUTOFF_PUBLIC_POLICY
    expected_freeze = (datetime.fromisoformat(configuration["schedule"]["submission_cutoff"].replace("Z", "+00:00")) - timedelta(hours=1)).isoformat().replace("+00:00", "Z")
    public_round = h.service.public_round(round_id)
    public_current = h.service.public_current()["open_round"]
    for view in (public_round, public_current):
        assert view["submission_replacement_cutoff"] == expected_freeze
        assert view["max_replacement_attempts"] == 1
        assert view["output_schema_version"] == contact_policy.output_schema(configuration)
        assert "icps" not in view
        for policy in ("integrity_policy", "contact_policy", "company_quality_policy", "intent_details_policy"):
            assert view.get(policy) == configuration.get(policy)
    h.clock.now = datetime.now(timezone.utc)
    miner = keypair("queued-flow-successful-owner")
    license_miner = keypair("queued-flow-license-owner")
    credential_miner = keypair("queued-flow-credential-owner")
    abandoned_miner = keypair("queued-flow-abandoned-owner")
    newcomer = keypair("queued-flow-new-owner")

    source_a = flavor_source_archive("SourceA")
    a = reserve(h, round_id, miner, source_a)
    assert finalize(h, round_id, miner, a)["status"] == "accepted"
    assert h.service.store.get_submission(a["submission_id"])["status"] == "accepted"
    assert h.service.review_pending_submissions()["reviewed"] == 0
    assert not h.review_transport.review_requests

    source_b = flavor_source_archive("SourceB")
    b = reserve(h, round_id, miner, source_b)
    assert b["submission_id"] != a["submission_id"]
    assert h.service.store.get_submission(a["submission_id"])["status"] == "accepted"
    assert h.service.store.get_submission(b["submission_id"])["status"] == "uploading"
    assert reserve(h, round_id, miner, source_b)["submission_id"] == b["submission_id"]
    expect_error(lambda: finalize(h, round_id, newcomer, b), "submission_missing", 404)
    assert finalize(h, round_id, miner, b)["status"] == "accepted"
    assert finalize(h, round_id, miner, b)["status"] == "accepted"
    assert reserve(h, round_id, miner, source_b)["submission_id"] == b["submission_id"]
    assert h.service.store.get_submission(b["submission_id"])["status"] == "accepted"
    replaced = h.service.store.get_submission(a["submission_id"])
    assert (replaced["status"], replaced["rejection_rule"]) == ("rejected", "source_replaced")
    expect_error(lambda: finalize(h, round_id, miner, a), "submission_superseded", 409)
    expect_replacement_limit(h, round_id, miner, flavor_source_archive("SecondReplacement"))
    assert h.objects.get(b["source_ref"]) == source_b

    license_source = flavor_source_archive("LicenseFallback")
    license_original = reserve(h, round_id, license_miner, license_source)
    assert finalize(h, round_id, license_miner, license_original)["status"] == "accepted"
    bad_license_source = flavor_source_archive("NoLicense", include_license=False)
    bad_license = reserve(h, round_id, license_miner, bad_license_source)
    with pytest.raises(svc.ServiceError) as invalid:
        finalize(h, round_id, license_miner, bad_license)
    assert invalid.value.status == 400 and invalid.value.code.startswith("submission_rejected:source_license_")
    assert h.service.store.get_submission(bad_license["submission_id"])["status"] == "rejected"
    assert h.service.store.get_submission(license_original["submission_id"])["status"] == "accepted"
    expect_replacement_limit(h, round_id, license_miner, bad_license_source)
    expect_replacement_limit(h, round_id, license_miner, flavor_source_archive("AfterBadLicense"))
    assert h.objects.get(license_original["source_ref"]) == license_source

    credential_source = flavor_source_archive("CredentialFallback")
    credential_original = reserve(h, round_id, credential_miner, credential_source)
    assert finalize(h, round_id, credential_miner, credential_original)["status"] == "accepted"
    bad_credential_source = flavor_source_archive("BadCredentials")
    bad_credential = reserve(h, round_id, credential_miner, bad_credential_source)
    manager = h.service.config.credential_manager
    normal_encrypt = manager.validate_and_encrypt

    def validate_test_credentials(values, *, submission_id, miner_hotkey):
        if values["openrouter_api_key"] == "invalid-openrouter-key":
            raise credentials_module.CredentialError("openrouter_api_key_invalid")
        return normal_encrypt(values, submission_id=submission_id, miner_hotkey=miner_hotkey)

    monkeypatch.setattr(manager, "validate_and_encrypt", validate_test_credentials)
    invalid_credentials = {
        "openrouter_api_key": "invalid-openrouter-key",
        "openrouter_management_key": CANARY_OPENROUTER_MANAGEMENT_KEY,
        "deepline_api_key": CANARY_DEEPLINE_KEY,
    }
    expect_error(
        lambda: finalize(h, round_id, credential_miner, bad_credential, credentials=invalid_credentials),
        "submission_rejected:openrouter_api_key_invalid", 400,
    )
    assert h.service.store.get_submission(bad_credential["submission_id"])["status"] == "rejected"
    assert h.service.store.get_submission(credential_original["submission_id"])["status"] == "accepted"
    expect_replacement_limit(h, round_id, credential_miner, bad_credential_source)
    expect_replacement_limit(h, round_id, credential_miner, flavor_source_archive("AfterBadCredentials"))

    abandoned_source = flavor_source_archive("AbandonedFallback")
    abandoned_original = reserve(h, round_id, abandoned_miner, abandoned_source)
    assert finalize(h, round_id, abandoned_miner, abandoned_original)["status"] == "accepted"
    pending_source = flavor_source_archive("PendingReplacement")
    pending = reserve(h, round_id, abandoned_miner, pending_source)
    assert reserve(h, round_id, abandoned_miner, pending_source)["submission_id"] == pending["submission_id"]
    expect_replacement_limit(h, round_id, abandoned_miner, flavor_source_archive("AfterAbandoned"))
    assert h.service.store.get_submission(abandoned_original["submission_id"])["status"] == "accepted"
    assert h.service.review_pending_submissions()["reviewed"] == 0
    assert not h.review_transport.review_requests

    # The SQL clock is now beyond this fixture round's new one-hour boundary.
    schedule = retime_disposable_round(connect, h, round_id, datetime.now(timezone.utc) + timedelta(seconds=30))
    freeze_at = datetime.fromisoformat(schedule["submission_cutoff"].replace("Z", "+00:00")) - timedelta(hours=1)
    h.clock.now = freeze_at
    expect_error(lambda: reserve(h, round_id, license_miner, flavor_source_archive("LateDifferent")), "submission_replacement_closed", 409)
    expect_error(lambda: finalize(h, round_id, abandoned_miner, pending), "submission_replacement_closed", 409)
    assert h.service.store.get_submission(b["submission_id"])["status"] == "accepted"
    assert h.service.store.get_submission(abandoned_original["submission_id"])["status"] == "accepted"
    first = reserve(h, round_id, newcomer, flavor_source_archive("FirstTime"))
    assert finalize(h, round_id, newcomer, first)["status"] == "accepted"
    expect_error(lambda: reserve(h, round_id, newcomer, flavor_source_archive("LateReplacement")), "submission_replacement_closed", 409)
    # Review claims open at the replacement freeze (23), after the winning
    # source is fixed and before the benchmark admission cutoff (00).
    assert h.service.review_pending_submissions()["reviewed"] == 5
    assert len(h.review_transport.review_requests) == 5
    reviewed_flavors = {
        next(file["content"] for file in json.loads(request["messages"][1]["content"])["submission_files"]
             if file["path"] == "flavor.txt")
        for request in h.review_transport.review_requests
    }
    assert reviewed_flavors == {
        "SourceB", "LicenseFallback", "CredentialFallback",
        "AbandonedFallback", "FirstTime",
    }
    expect_error(lambda: h.service.public_benchmark(round_id), "benchmark_not_public", 403)

    # Advance only the disposable round's schedule. The installed SQL review
    # and commitment guards observe real elapsed database time.
    schedule = retime_disposable_round(connect, h, round_id, datetime.now(timezone.utc) - timedelta(seconds=2))
    h.clock.now = datetime.now(timezone.utc)
    expect_error(lambda: reserve(h, round_id, keypair("queued-flow-too-late"), flavor_source_archive("TooLate")), "submission_window_closed", 409)
    assert h.service.review_pending_submissions()["reviewed"] == 0
    assert len(h.review_transport.review_requests) == 5
    assert h.service.store.get_submission(a["submission_id"])["code_review_status"] != "passed"
    selected_challengers = {
        b["submission_id"], license_original["submission_id"],
        credential_original["submission_id"], abandoned_original["submission_id"],
        first["submission_id"],
    }
    discarded = {
        a["submission_id"], bad_license["submission_id"],
        bad_credential["submission_id"], pending["submission_id"],
    }
    for submission_id in selected_challengers:
        assert h.service.store.get_submission(submission_id)["code_review_status"] == "passed"
    for submission_id in discarded:
        assert h.service.store.get_submission(submission_id)["code_review_status"] != "passed"
    assert h.service.advance_round(round_id)["status"] == "ok"
    committed = h.service.store.get_round(round_id)
    selected = {item["submission_id"] for item in committed["participants"]}
    assert selected_challengers <= selected
    assert selected.isdisjoint(discarded)
    for submission_id in selected_challengers:
        assert h.service.store.get_submission(submission_id)["status"] == "frozen"
    assert (h.service.store.get_submission(pending["submission_id"])["status"],
            h.service.store.get_submission(pending["submission_id"])["rejection_rule"]) == (
                "rejected", "source_upload_incomplete",
            )
    assert committed["configuration_doc"]["benchmark_disclosure_policy"] == icp_disclosure.CUTOFF_PUBLIC_POLICY
    cutoff_bank = h.service.public_benchmark(round_id)
    assert cutoff_bank["disclosure_policy"] == icp_disclosure.CUTOFF_PUBLIC_POLICY
    assert len(cutoff_bank["icps"]) == contracts.BENCHMARK_ICP_COUNT
    assert h.service.public_submission_code(b["submission_id"])["files"]
    for submission_id in discarded:
        expect_error(
            lambda submission_id=submission_id: h.service.public_submission_code(
                submission_id
            ),
            "source_not_public",
            403,
        )
    expect_error(
        lambda: h.service.public_results(round_id, b["submission_id"]),
        "results_not_public",
        403,
    )

    for item in committed["participants"]:
        h.flavors.setdefault(item["submission_id"], "PublicBaseline")
    h.flavors[b["submission_id"]] = "SourceB"
    h.flavors[license_original["submission_id"]] = "LicenseFallback"
    h.flavors[credential_original["submission_id"]] = "CredentialFallback"
    h.flavors[abandoned_original["submission_id"]] = "AbandonedFallback"
    h.flavors[first["submission_id"]] = "FirstTime"
    _run_stage_one_to_scoring(h, len(selected), runners=1)
    h.advance_until("published", runners=1)
    publication = h.service.store.get_round(round_id)
    assert publication["status"] == "published"
    all_runs = h.service.store.list_runs(round_id)
    assert {run["submission_id"] for run in all_runs} <= selected
    assert discarded.isdisjoint({run["submission_id"] for run in all_runs})
    for submission_id in discarded:
        assert h.service.store.list_ledger(submission_id=submission_id) == []
    execute_b = h.service.store.list_runs(round_id, submission_id=b["submission_id"], kind="execute")
    assert len(execute_b) == contracts.BENCHMARK_ICP_COUNT
    assert all(run["per_icp_score"] is not None for run in execute_b)
    ranking_ids = {item["submission_id"] for item in publication["publication_doc"]["final_ranking"]}
    assert selected_challengers <= ranking_ids
    assert ranking_ids.isdisjoint(discarded)
    public = h.service.public_benchmark(round_id)
    assert public["disclosure_policy"] == icp_disclosure.CUTOFF_PUBLIC_POLICY
    assert len(public["icps"]) == contracts.BENCHMARK_ICP_COUNT
    results = h.service.public_results(round_id, b["submission_id"])
    assert len(results["scores"]["stage_1"] + results["scores"]["stage_2"]) == contracts.BENCHMARK_ICP_COUNT
    assert len(results["outputs"]) == contracts.BENCHMARK_ICP_COUNT
    preview = h.service.public_submission_code(b["submission_id"])
    assert {file["path"]: file["content"] for file in preview["files"]}["flavor.txt"] == "SourceB"
    expect_error(lambda: h.service.public_submission_code(a["submission_id"]), "source_not_public", 403)
    expect_error(lambda: h.service.public_results(round_id, a["submission_id"]), "submission_missing", 404)

    # The one replacement allowance resets in the next round with the active
    # integrity, contact, and Intent Details policy shape.
    h.clock.now = datetime.now(timezone.utc)
    next_id = "arena-2026-11-02"
    source_miner = keypair("queued-flow-source-validation-owner")
    h.chain.owned[miner.ss58_address] = [miner.ss58_address]
    h.chain.owned[source_miner.ss58_address] = [source_miner.ss58_address]
    h.service.config.defaults = replace(
        h.service.config.defaults,
        integrity_from="2026-01-01T00:00:00Z",
        contacts_from="2026-01-01T00:00:00Z",
        intent_details_from="2026-01-01T00:00:00Z",
    )
    next_configuration = h.service.create_round(
        datetime.now(timezone.utc) + timedelta(hours=12), round_id=next_id,
    )
    assert {"integrity_policy", "contact_policy", "intent_details_policy"} <= next_configuration.keys()
    assert "company_quality_policy" not in next_configuration
    with TestClient(create_app(h.service)) as http:
        round_response = http.get("/arena/v1/rounds/%s" % next_id)
        current_response = http.get("/arena/v1/current")
    assert round_response.status_code == current_response.status_code == 200
    for view in (round_response.json(), current_response.json()["open_round"]):
        assert view["max_replacement_attempts"] == 1
        assert view["output_schema_version"] == "leadpoet.lab_arena.output.v5"
        assert "company_quality_policy" not in view
        assert "icps" not in view
        for policy in ("integrity_policy", "contact_policy", "intent_details_policy"):
            assert view[policy] == next_configuration[policy]
    fresh = reserve(h, next_id, miner, flavor_source_archive("NextRound"))
    assert fresh["submission_id"] not in {a["submission_id"], b["submission_id"]}
    assert finalize(h, next_id, miner, fresh)["status"] == "accepted"
    next_replacement = reserve(h, next_id, miner, flavor_source_archive("NextRoundReplacement"))
    expect_error(lambda: finalize(h, next_id, newcomer, next_replacement), "submission_missing", 404)
    assert finalize(h, next_id, miner, next_replacement)["status"] == "accepted"
    expect_replacement_limit(h, next_id, miner, flavor_source_archive("NextRoundSecondReplacement"))
    assert h.service.review_pending_submissions()["reviewed"] == 0
    assert h.service.store.list_runs(next_id) == []
    assert h.service.store.list_ledger(submission_id=next_replacement["submission_id"]) == []
    assert h.service.store.get_submission(fresh["submission_id"])["round_id"] == next_id
    assert h.service.store.get_submission(fresh["submission_id"])["rejection_rule"] == "source_replaced"
    assert h.service.store.get_submission(b["submission_id"])["round_id"] == round_id

    source_fallback = reserve(h, next_id, source_miner, flavor_source_archive("SourceValidationFallback"))
    assert finalize(h, next_id, source_miner, source_fallback)["status"] == "accepted"
    source_attempt_bytes = flavor_source_archive("ChecksumFailure")
    source_attempt = reserve(
        h, next_id, source_miner, source_attempt_bytes,
        uploaded_source=b"x" * len(source_attempt_bytes),
    )
    expect_error(
        lambda: finalize(h, next_id, source_miner, source_attempt),
        "submission_rejected:source_checksum_mismatch", 400,
    )
    assert h.service.store.get_submission(source_attempt["submission_id"])["status"] == "rejected"
    assert h.service.store.get_submission(source_fallback["submission_id"])["status"] == "accepted"
    expect_replacement_limit(h, next_id, source_miner, source_attempt_bytes)
    expect_replacement_limit(h, next_id, source_miner, flavor_source_archive("AfterBadChecksum"))
    h.service.cancel(next_id, "operator")
