"""Database guard for the baseline cost-efficiency score consequence."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from lab_arena import contracts, integrity, verify
from lab_arena.store import ArenaStoreError
from qualification.scoring.arena_integrity import canonical_company_identity
from tests.lab_arena import test_lab_arena_service_round as fixtures
from tests.lab_arena.lab_arena_pg_harness import (
    CURRENT_SERVICE_MIGRATIONS,
    database_with_lab_arena_migration,
)
from tests.lab_arena.test_integrity_round import IntegrityHarness


@pytest.fixture(scope="module")
def database():
    yield from database_with_lab_arena_migration(CURRENT_SERVICE_MIGRATIONS)


def _score(companies, icp, _reference):
    indexes, _ = verify.bucket_skip(icp, companies)
    rows = []
    for index in indexes:
        baseline = companies[index]["company_name"].startswith("PublicBaseline")
        rows.append(
            {
                "final_score": 5.0 if baseline else 1.0,
                "company_index": index,
                "company_identity_key": canonical_company_identity(
                    companies[index]
                ).key,
                "company_qualified": True,
                "duplicate_company": False,
                "verifier_gate_receipts": [
                    {"gate": "company_fit", "decision": "match"}
                ],
                "intent_signals_detail": [],
                "failure_reason": "",
            }
        )
    return rows


def _ranking(publication, submission_id):
    return next(
        row
        for row in publication["final_ranking"]
        if row["submission_id"] == submission_id
    )


def test_guard_requires_receipt_backed_zero_and_migration_replays_without_history_changes(
    database, tmp_path, monkeypatch
):
    psycopg2, dsn = database
    connect = lambda: psycopg2.connect(**dsn)
    monkeypatch.setattr(fixtures, "deterministic_scorer", _score)
    harness = IntegrityHarness(
        connect, tmp_path, challengers=["Threshold"], runners=["alpha"]
    )
    harness.service.config.defaults = replace(
        harness.service.config.defaults,
        cost_per_company_microusd=1,
        execution_cap_microusd=80_000_000,
    )
    harness.chain.epoch = 25_090
    harness.clock.now = datetime.now(timezone.utc)
    round_id = "arena-2026-11-02-costzero"
    configuration = harness.service.create_round(
        harness.clock.now + timedelta(hours=12), round_id=round_id
    )
    assert configuration["integrity_policy"] == integrity.POLICY
    harness.round_id = round_id
    challenger_id = harness.submit("Threshold", round_id)
    harness.clock.advance_to(harness.schedule()["submission_cutoff"])
    harness.advance_until("scored")

    round_row = harness.service.store.get_round(round_id)
    baseline_id = next(
        row["submission_id"] for row in round_row["participants"] if row["is_king"]
    )
    captured = {}
    transition = harness.service.store.transition_round

    def capture_transition(captured_round_id, expected, next_status, patch):
        captured["args"] = (
            captured_round_id,
            expected,
            next_status,
            deepcopy(patch),
        )
        return {"status": "captured"}

    monkeypatch.setattr(
        harness.service.store, "transition_round", capture_transition
    )
    assert harness.service.publish(round_id)["status"] == "captured"
    efficient_patch = captured["args"][3]
    efficient_baseline = _ranking(
        efficient_patch["publication_doc"], baseline_id
    )
    assert efficient_baseline["eligible"] is True
    assert efficient_baseline["final_score"] == 5.0
    monkeypatch.setattr(harness.service.store, "transition_round", transition)

    forged_eligible_zero = deepcopy(efficient_patch)
    _ranking(
        forged_eligible_zero["publication_doc"], baseline_id
    )["final_score"] = 0.0
    with pytest.raises(ArenaStoreError, match="lab_arena_final_score_mismatch"):
        transition(round_id, "scored", "published", forged_eligible_zero)

    forged_baseline_null = deepcopy(efficient_patch)
    _ranking(
        forged_baseline_null["publication_doc"], baseline_id
    )["final_score"] = None
    with pytest.raises(ArenaStoreError, match="lab_arena_final_score_mismatch"):
        transition(round_id, "scored", "published", forged_baseline_null)

    # The regular fake provider is free. Add one successful paid sourcing
    # terminal after scoring so the publication guard must apply the frozen
    # one-microdollar-per-qualified-pair rule. This is test-only ledger setup;
    # the accepted score rows remain untouched.
    connection = connect()
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT run_id,miner_hotkey,stage FROM public.lab_arena_runs "
                "WHERE round_id=%s AND submission_id=%s AND kind='execute' "
                "ORDER BY stage,icp_position LIMIT 1",
                (round_id, baseline_id),
            )
            run_id, miner_hotkey, stage = cursor.fetchone()
            cursor.execute(
                "SELECT public.lab_arena__integrity_eligibility(%s,%s,%s)",
                (round_id, baseline_id, list(range(20))),
            )
            current = cursor.fetchone()[0]
            amount = int(current["eligibility_cap_microusd"]) + 1
            cursor.execute(
                "INSERT INTO public.lab_arena_ledger "
                "(entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,"
                "call_identity,provider,operation_id,funding_source,"
                "amount_microusd,entry_doc,terminal_response) VALUES "
                "('settlement',%s,%s,%s,%s,%s,%s,'openrouter',"
                "'openrouter.chat','host',%s,'{}'::jsonb,%s::jsonb)",
                (
                    miner_hotkey,
                    round_id,
                    baseline_id,
                    run_id,
                    stage,
                    contracts.document_hash({"baseline-cost": round_id}),
                    amount,
                    '{"status":200,"call_succeeded":true}',
                ),
            )
    finally:
        connection.close()
    monkeypatch.setattr(
        harness.service.store, "transition_round", capture_transition
    )
    assert harness.service.publish(round_id)["status"] == "captured"
    monkeypatch.setattr(harness.service.store, "transition_round", transition)

    args = captured["args"]
    valid_patch = args[3]
    valid_publication = valid_patch["publication_doc"]
    baseline = _ranking(valid_publication, baseline_id)
    challenger = _ranking(valid_publication, challenger_id)
    assert baseline["final_score"] == 0.0
    assert baseline["eligibility_reason"] == "cost_per_company_exceeded"
    assert challenger["final_score"] == 1.0
    assert valid_publication["king_decision"]["winner_submission_id"] == (
        challenger_id
    )

    forged_raw_baseline = deepcopy(valid_patch)
    _ranking(
        forged_raw_baseline["publication_doc"], baseline_id
    )["final_score"] = 5.0
    with pytest.raises(ArenaStoreError, match="lab_arena_final_score_mismatch"):
        transition(round_id, "scored", "published", forged_raw_baseline)

    forged_challenger = deepcopy(valid_patch)
    _ranking(
        forged_challenger["publication_doc"], challenger_id
    )["final_score"] = 0.0
    with pytest.raises(ArenaStoreError, match="lab_arena_final_score_mismatch"):
        transition(round_id, "scored", "published", forged_challenger)

    assert transition(*args)["status"] == "ok"
    published = harness.service.store.get_round(round_id)
    assert _ranking(published["publication_doc"], baseline_id)["final_score"] == 0

    migration = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "254-lab-arena-baseline-cost-eligibility.sql"
    ).read_text(encoding="utf-8")
    connection = connect()
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT md5(row_to_json(rounds)::text) "
                "FROM public.lab_arena_rounds AS rounds WHERE round_id=%s",
                (round_id,),
            )
            before = cursor.fetchone()[0]
            cursor.execute(migration)
            cursor.execute(migration)
            cursor.execute(
                "SELECT md5(row_to_json(rounds)::text) "
                "FROM public.lab_arena_rounds AS rounds WHERE round_id=%s",
                (round_id,),
            )
            assert cursor.fetchone()[0] == before
            cursor.execute(
                "SELECT public.lab_arena_baseline_cost_eligibility_schema_v1()"
            )
            assert cursor.fetchone()[0] == {
                "schema_version": (
                    "leadpoet.lab_arena.baseline_cost_eligibility_schema.v1"
                ),
                "version": 254,
                "policy": "zero_baseline_cost_ineligible_v1",
            }
            for role, expected in (
                ("lab_arena_service", True),
                ("anon", False),
                ("authenticated", False),
                ("service_role", False),
            ):
                cursor.execute(
                    "SELECT has_function_privilege(%s, "
                    "'public.lab_arena_baseline_cost_eligibility_schema_v1()', "
                    "'EXECUTE')",
                    (role,),
                )
                assert cursor.fetchone()[0] is expected
    finally:
        connection.close()
        harness.service.store.close()
