"""Bounded release gates for admission and the real Arena queue at 256 miners.

Prior accepted entries are bulk fixture setup, not admission evidence. The
86->87 and 255->256 boundaries use signed HTTP and the real serialized SQL
finalization. Full execution uses the supported two-ICP configuration; the
separate baseline-first ten-ICP gate proves 10 baseline and 2,560 miner assignments exist.
Provider transport and sandbox processes use the existing controlled fixtures.
"""

from __future__ import annotations

import hashlib
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from threading import Barrier

from fastapi.testclient import TestClient

from lab_arena import contracts, runner as rn, source_bundle, submission_similarity
from lab_arena.api import create_app
from tests.lab_arena.icp_fixtures import daily_icps
from tests.lab_arena.baseline_scored_first_postgres_test import _enable_verified_proxy_runtime
from tests.lab_arena.parallel_twenty_runtime_e2e_test import bounded_socket_shutdown_poll  # noqa: F401
from tests.lab_arena.test_lab_arena_service_round import (
    CANARY_DEEPLINE_KEY, CANARY_OPENROUTER_KEY,
    CANARY_OPENROUTER_MANAGEMENT_KEY, Harness, assert_canary_absent,
    connect, database, flavor_source_archive, keypair,
)


def _harness(connect, tmp_path, *, count, suffix, cutoff_hours=0.5, baseline_first=False):
    harness = Harness(connect, tmp_path, challengers=[], runners=["alpha", "beta"])
    harness.clock.now = datetime.now(timezone.utc)
    harness.chain.epoch = 67000 + count
    harness.service.config.defaults = replace(
        harness.service.config.defaults, benchmark_icp_count=count,
        max_challengers=256, runner_slot_ceiling=10,
        execution_sequence_from="2000-01-01T00:00:00Z" if baseline_first else None,
    )
    harness.service.config.daily_icp_source = lambda **kw: {
        "status": "ready", "set_id": int(kw["set_id"]), "icps": daily_icps()[:count],
    }
    config = harness.service.create_round(
        datetime.now(timezone.utc) + timedelta(hours=cutoff_hours),
        round_id="arena-2026-10-30-" + suffix,
    )
    assert config["max_challengers"] == 256
    harness.round_id = config["round_id"]
    if baseline_first:
        _enable_verified_proxy_runtime(harness)
    return harness


def _signed(harness, miner, scope, body):
    return contracts.build_signed_request(
        scope=scope, round_id=harness.round_id, hotkey=miner.ss58_address,
        body=body, timestamp=int(harness.clock().timestamp()),
        sign_message=lambda message: miner.sign(message.encode()).hex(),
    )


def _upload(harness, http, flavor, *, miner_label=None):
    miner = keypair("svc-miner-" + (miner_label or flavor))
    payload = flavor_source_archive(flavor)
    facts = source_bundle.validate_source_archive(payload, require_license=True)
    response = http.post("/arena/v1/submissions/presign", json=_signed(
        harness, miner, contracts.SCOPE_SUBMISSION_PRESIGN,
        {"source_size_bytes": facts["source_size_bytes"], "consent": {"public_rerun": True}},
    ))
    if response.status_code != 200:
        return response, None
    target = response.json()
    harness.objects.put(target["source_ref"], payload)
    harness.flavors[target["submission_id"]] = flavor
    finalize = _signed(harness, miner, contracts.SCOPE_SUBMISSION_FINALIZE, {
        "submission_id": target["submission_id"], "source_ref": target["source_ref"],
        "source_size_bytes": facts["source_size_bytes"], "credentials": {
            "openrouter_api_key": CANARY_OPENROUTER_KEY,
            "openrouter_management_key": CANARY_OPENROUTER_MANAGEMENT_KEY,
            "deepline_api_key": CANARY_DEEPLINE_KEY,
        },
    })
    return target, finalize


def _finalize(http, upload):
    target, envelope = upload
    assert isinstance(target, dict), getattr(target, "text", target)
    return http.post("/arena/v1/submissions/%s/finalize" % target["submission_id"], json=envelope)


def _bulk_prior_accepted(harness, template_id, indexes):
    """Seed historical accepted entries; leave all guards and SQL enabled."""
    from psycopg2.extras import Json, execute_values

    overrides = []
    for index in indexes:
        flavor = "Load-%03d" % index
        submission_id = "load-%s-%03d" % (harness.round_id, index)
        # Distinct valid hotkeys without deriving hundreds of expensive wallets.
        # The load fixture never signs as these historical miners.
        from bittensor_wallet import Keypair
        hotkey = Keypair(public_key="0x" + hashlib.sha256(flavor.encode()).hexdigest(), ss58_format=42).ss58_address
        payload = flavor_source_archive(flavor)
        facts = source_bundle.validate_source_archive(payload, require_license=True)
        identity = submission_similarity.inspect_archive(payload)
        source_ref = "arena/%s/sources/%s.tar.gz" % (harness.round_id, submission_id)
        harness.objects.put(source_ref, payload)
        harness.flavors[submission_id] = flavor
        overrides.append((Json({
            "submission_id": submission_id, "miner_hotkey": hotkey,
            "source_ref": source_ref, "source_size_bytes": facts["source_size_bytes"],
            "submission_doc": {"source_ref": source_ref, "source_size_bytes": facts["source_size_bytes"], "consent": {"public_rerun": True}},
            "source_archive_sha256": identity.archive_sha256,
            "source_normalized_sha256": identity.normalized_sha256,
        }), template_id))
    with harness.connect() as conn, conn.cursor() as cur:
        execute_values(cur, """
            INSERT INTO public.lab_arena_submissions
            SELECT (jsonb_populate_record(NULL::public.lab_arena_submissions,
                to_jsonb(s) || v.doc::jsonb)).*
            FROM public.lab_arena_submissions s
            JOIN (VALUES %s) AS v(doc,template_id)
                ON s.submission_id=v.template_id
        """, overrides, page_size=256)
        cur.execute("""
            INSERT INTO public.lab_arena_submission_credentials
                (submission_id, miner_hotkey, provider, ciphertext)
            SELECT s.submission_id,s.miner_hotkey,c.provider,c.ciphertext
            FROM public.lab_arena_submissions s
            CROSS JOIN public.lab_arena_submission_credentials c
            WHERE s.round_id=%s AND s.submission_id LIKE 'load-%%'
                AND c.submission_id=%s
            ON CONFLICT DO NOTHING
        """, (harness.round_id, template_id))


def _accepted(harness):
    return [r for r in harness.service.store.list_submissions(harness.round_id, status="accepted") if not r["is_king"]]


def _admit_at_cap(harness, http):
    first = harness.submit("Load-Original-" + str(contracts.benchmark_icp_count(harness.service.store.get_round(harness.round_id)["configuration_doc"])), harness.round_id)
    _bulk_prior_accepted(harness, first, range(1, 86))
    assert len(_accepted(harness)) == 86
    at87 = _upload(harness, http, "Load-real-87-" + harness.round_id)
    assert _finalize(http, at87).status_code == 200
    assert len(_accepted(harness)) == 87
    duplicate = _upload(harness, http, harness.flavors[first], miner_label="Load-exact-duplicate-" + harness.round_id)
    refused = _finalize(http, duplicate)
    assert refused.status_code == 409 and refused.json()["code"] == "submission_rejected:duplicate_submission"
    assert len(_accepted(harness)) == 87
    _bulk_prior_accepted(harness, first, range(86, 254))
    assert len(_accepted(harness)) == 255
    uploads = [_upload(harness, http, "Load-boundary-%d-%s" % (i, harness.round_id)) for i in range(4)]
    barrier = Barrier(4)
    manager = harness.service.config.credential_manager
    original = manager.validate_and_encrypt

    def synchronized(*args, **kwargs):
        encrypted = original(*args, **kwargs)
        barrier.wait(timeout=15)
        return encrypted

    manager.validate_and_encrypt = synchronized
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            responses = list(pool.map(lambda upload: _finalize(http, upload), uploads))
    finally:
        manager.validate_and_encrypt = original
    assert [r.status_code for r in responses].count(200) == 1
    rejected = [r for r in responses if r.status_code != 200]
    assert len(rejected) == 3
    assert all(r.status_code == 409 and r.json()["code"] == "submission_rejected:capacity.round_full" for r in rejected)
    assert len(_accepted(harness)) == 256
    with harness.connect() as conn, conn.cursor() as cur:
        cur.execute("""SELECT count(*) FROM public.lab_arena_submission_credentials c
            JOIN public.lab_arena_submissions s USING(submission_id)
            WHERE s.round_id=%s AND s.status='rejected'""", (harness.round_id,))
        assert cur.fetchone() == (0,)
    harness.service.review_pending_submissions()
    assert all(row["code_review_status"] == "passed" for row in _accepted(harness))
    return first


def _freeze_and_open(harness):
    harness.clock.advance_to(harness.schedule()["submission_cutoff"])
    assert harness.service.advance_round(harness.round_id)["participants"] == 257
    for p in harness.service.store.get_round(harness.round_id)["participants"]:
        harness.flavors.setdefault(p["submission_id"], "PublicBaseline")
    harness.clock.advance_to(harness.schedule()["stage_1_start"])
    return harness.service.advance_round(harness.round_id)


def test_256_admission_ten_icp_queue_and_concurrent_claims(connect, tmp_path):
    harness = _harness(connect, tmp_path, count=10, suffix="queue", baseline_first=True)
    with TestClient(create_app(harness.service)) as http:
        _admit_at_cap(harness, http)
        assert _freeze_and_open(harness)["assignments"] == 10
        harness.api_factory = lambda: rn.HttpArenaApiClient("http://localhost", client=http)
        harness.advance_until("stage1_scored", runners=2)
        baseline = harness.service.store.list_runs(harness.round_id, stage=1, kind="execute")
        assert len(baseline) == 10 and all(r["per_icp_score"] > 0 for r in baseline)
        assert harness.service.advance_round(harness.round_id)["assignments"] == 10 * 256
        runs = harness.service.store.list_runs(harness.round_id, stage=2, kind="execute")
        expected = {(p["submission_id"], position) for p in harness.service.store.get_round(harness.round_id)["participants"] for position in range(10) if not p["is_king"]}
        assert len(runs) == len({r["assignment_id"] for r in runs}) == len(expected)
        assert {(r["submission_id"], r["icp_position"]) for r in runs} == expected
        harness.api_factory = lambda: rn.HttpArenaApiClient("http://localhost", client=http)
        scheduled = harness.clock.now
        harness.clock.now = datetime.now(timezone.utc)
        runners = [harness.runner(i, parallel=4) for i in range(2)]
        try:
            with ThreadPoolExecutor(max_workers=2) as pool:
                leases = list(pool.map(lambda runner: [runner.claim_one() for _ in range(4)], runners))
            leases = [lease for group in leases for lease in group]
            assert all(lease["status"] == "leased" for lease in leases)
            assert len({lease["run_id"] for lease in leases}) == 8
            for runner in runners:
                refused = runner.claim_one()
                assert refused["status"] == "no_free_slot", refused
                assert refused["active_leases"] == refused["slot_limit"] == 4
            rows = harness.service.store.list_runs(harness.round_id, stage=2, kind="execute")
            assert sum(r["status"] == "leased" for r in rows) == 8
            assert sum(r["status"] == "pending" for r in rows) == 10 * 256 - 8
            assert all(r["lease_generation"] == 1 and r["lease_token_hash"] for r in rows if r["status"] == "leased")
        finally:
            for runner in runners:
                runner.close()
            harness.clock.now = scheduled
    harness.service.store.cancel_round(harness.round_id, "operator_abort")


def test_256_backlog_executes_scores_and_publishes_over_http(connect, tmp_path):
    started = time.monotonic()
    harness = _harness(connect, tmp_path, count=2, suffix="publication", baseline_first=True)
    with TestClient(create_app(harness.service)) as http:
        example = _admit_at_cap(harness, http)
        harness.api_factory = lambda: rn.HttpArenaApiClient("http://localhost", client=http)
        assert _freeze_and_open(harness)["assignments"] == 2
        # Keep actual runners and their caches across the full competition.
        runners = [harness.runner(i) for i in range(2)]
        # Cache eviction has separate tests. Keep these small fixture archives
        # resident so a non-root macOS host need not remove read-only source.
        for runner in runners:
            runner._config.source_cache._max_entries = 257
        try:
            for _ in range(80):
                status = harness.status()
                if status == "published":
                    break
                if status in ("stage1", "stage1_scoring", "stage2", "stage2_scoring"):
                    scheduled = harness.clock.now
                    harness.clock.now = datetime.now(timezone.utc)
                    try:
                        while any(runner.run_once() for runner in runners):
                            pass
                    finally:
                        harness.clock.now = scheduled
                    assert not [c for runner in runners for c in runner.completed if c.get("error")]
                if status == "stage1_scored":
                    harness.clock.advance_to(harness.schedule()["stage_2_start"])
                step = harness.service.advance_round(harness.round_id)
                assert step.get("status") not in ("cancelled", "terminal", "retry", "stale"), step
            else:
                raise AssertionError("round did not publish")
        finally:
            for runner in runners:
                runner.close()
        row = harness.service.store.get_round(harness.round_id)
        assert row["status"] == "published" and row["cancel_reason"] is None
        assert len(row["publication_doc"]["stage1_ranking"]) == 256
        runs = harness.service.store.list_runs(harness.round_id)
        assert all(r["status"] == "accepted" for r in runs)
        assert len({(r["assignment_id"], r["attempt"]) for r in runs}) == len(runs)
        assert all(r["lease_generation"] == 1 for r in runs)
        execution = [r for r in runs if r["kind"] == "execute"]
        assert len(execution) == 257 * 2
        assert len(row["publication_doc"]["final_ranking"]) == 257
        assert all(r["per_icp_score"] is not None for r in execution)
        assert all(r["per_icp_score"] > 0 for r in execution)
        public = http.get("/arena/v1/rounds/%s" % harness.round_id)
        results = http.get("/arena/v1/rounds/%s/results/%s" % (harness.round_id, example))
        assert public.status_code == results.status_code == 200
        assert len(results.json()["scores"]["stage_2"]) == 2
        assert_canary_absent(harness, connect)
    print("\nADMISSION LOAD FULL PATH seconds=%.2f" % (time.monotonic() - started))


def test_http_cutoff_accepts_last_second_and_rejects_at_boundary(connect, tmp_path):
    harness = _harness(connect, tmp_path, count=2, suffix="cutoff", cutoff_hours=10 / 3600)
    cutoff = datetime.strptime(harness.schedule()["submission_cutoff"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    with TestClient(create_app(harness.service)) as http:
        accepted_upload = _upload(harness, http, "Load-last-second")
        late_upload = _upload(harness, http, "Load-sql-at-cutoff")
        # Use PostgreSQL's real wall clock, without changing frozen schedule or
        # disabling a trigger. The first signed request finishes one second
        # before cutoff; the second reaches SQL after it despite a stale gateway.
        time.sleep(max(0, cutoff.timestamp() - time.time() - 1))
        harness.clock.now = cutoff - timedelta(seconds=1)
        assert _finalize(http, accepted_upload).status_code == 200
        assert datetime.now(timezone.utc) < cutoff
        time.sleep(max(0, cutoff.timestamp() - time.time() + 0.1))
        refused = _finalize(http, late_upload)
        assert refused.status_code == 409 and refused.json()["code"] == "submission_window_closed"
        late_id = late_upload[0]["submission_id"]
        assert harness.service.store.get_submission(late_id)["status"] == "uploading"
        with connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM public.lab_arena_submission_credentials WHERE submission_id=%s", (late_id,))
            assert cur.fetchone() == (0,)
        before = harness.service.store.list_submissions(harness.round_id)
        harness.clock.now = cutoff
        rejected, envelope = _upload(harness, http, "Load-at-cutoff")
        assert envelope is None
        assert rejected.status_code == 409 and rejected.json()["code"] == "submission_window_closed"
        assert harness.service.store.list_submissions(harness.round_id) == before
    harness.service.store.cancel_round(harness.round_id, "operator_abort")


def test_same_hotkey_replacement_preserves_full_round_slot(connect, tmp_path):
    harness = _harness(connect, tmp_path, count=2, suffix="replacement", cutoff_hours=12)
    with TestClient(create_app(harness.service)) as http:
        first = _upload(harness, http, "Load-replacement-owner")
        assert _finalize(http, first).status_code == 200
        _bulk_prior_accepted(harness, first[0]["submission_id"], range(1, 256))
        assert len(_accepted(harness)) == 256
        replacement = _upload(harness, http, "Load-replacement-new", miner_label="Load-replacement-owner")
        assert _finalize(http, replacement).status_code == 200
        assert len(_accepted(harness)) == 256
        assert harness.service.store.get_submission(first[0]["submission_id"])["rejection_rule"] == "source_replaced"
        assert harness.service.store.get_submission(replacement[0]["submission_id"])["replaces_submission_id"] == first[0]["submission_id"]
    harness.service.store.cancel_round(harness.round_id, "operator_abort")
