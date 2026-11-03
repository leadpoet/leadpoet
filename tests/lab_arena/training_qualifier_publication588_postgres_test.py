"""The publication trigger permits unknown noncontenders, never unknown winners."""

from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
from pathlib import Path
from threading import Barrier

import pytest

from scripts.verify_arena_full_thinking_e2e import full_thinking_database
from tests.lab_arena.training_publication569_postgres_test import _verify, _training
from tests.lab_arena.baseline_scored_first_postgres_test import (
    _baseline_first_judge, _enable_verified_proxy_runtime,
)
from tests.lab_arena.test_integrity_round import IntegrityHarness
from tests.lab_arena import test_lab_arena_service_round as fixtures


MIGRATION = (Path(__file__).resolve().parents[2] /
             "scripts/588-lab-arena-qualifier-training-publication.sql")
MARKER = {"status": "not_assessed", "eligible": None, "reasons": []}


@pytest.fixture()
def database():
    yield from full_thinking_database()


def _try_publish(cursor, round_id, publication, *, expected_error=None):
    cursor.execute("SAVEPOINT qualifier_publication_attempt")
    try:
        cursor.execute("UPDATE public.lab_arena_rounds SET status='published', "
                       "publication_doc=%s::jsonb WHERE round_id=%s",
                       (json.dumps(publication), round_id))
        cursor.execute("RELEASE SAVEPOINT qualifier_publication_attempt")
        return True
    except Exception as exc:
        cursor.execute("ROLLBACK TO SAVEPOINT qualifier_publication_attempt")
        cursor.execute("RELEASE SAVEPOINT qualifier_publication_attempt")
        assert exc.pgcode == "22023", str(exc)
        if expected_error is not None:
            assert expected_error in str(exc)
        return False


def test_exact_unknown_marker_and_winner_guard_preserve_scored_costs(
    database, tmp_path, monkeypatch,
):
    psycopg2, dsn = database
    harness = IntegrityHarness(lambda: psycopg2.connect(**dsn), tmp_path,
                               challengers=["Miner0"], runners=["alpha"])
    harness.service.config.defaults = replace(
        harness.service.config.defaults,
        execution_sequence_from="2000-01-01T00:00:00Z",
        per_icp_cost_policy=False, flat_per_icp_cost_policy=True,
        rewards_enabled=True,
    )
    _enable_verified_proxy_runtime(harness)

    def judge(companies, icp, reference):
        rows = _baseline_first_judge(companies, icp, reference)
        for row in rows:
            row["final_score"] = 80.0
        return rows

    monkeypatch.setattr(fixtures, "deterministic_scorer", judge)
    assert fixtures._start_round(harness, day=16, epoch=62416) == 2
    harness.clock.advance_to(harness.schedule()["stage_1_start"])
    harness.advance_until("scored", runners=1)
    round_id = harness.round_id
    with psycopg2.connect(**dsn) as connection, connection.cursor() as cursor:
        # The established full-thinking fixture freezes this policy at scoring
        # completion while retaining its accepted runs and ledger rows.
        cursor.execute("SET LOCAL session_replication_role=replica")
        cursor.execute("UPDATE public.lab_arena_rounds SET configuration_doc="
                       "configuration_doc || %s::jsonb WHERE round_id=%s",
                       (json.dumps({
                           "reasoning_capture_policy": "full_generated_thinking_v1",
                           "execution_icp_cap_microusd": 6_000_000,
                           "cost_per_company_microusd": 6_000_000,
                           "execution_cap_microusd": 120_000_000,
                       }),
                        round_id))
        cursor.execute("SET LOCAL session_replication_role=origin")
    # No contender can cross the margin. No source, capture, or export read is
    # permitted, even though the full SQL cost and score proof still runs.
    monkeypatch.setattr(harness.service, "_submission_training_eligibility",
                        lambda *_args, **_kwargs: pytest.fail("unexpected training export"))
    with pytest.raises(Exception, match="lab_arena_publication_training_report_mismatch"):
        harness.service.publish(round_id)  # Migration 569 rejects the new marker.
    assert harness.service.store.get_round(round_id)["status"] == "scored"
    with psycopg2.connect(**dsn) as connection, connection.cursor() as cursor:
        cursor.execute(MIGRATION.read_text())
        cursor.execute(MIGRATION.read_text())  # Exact idempotence.
    assert harness.service.publish(round_id)["status"] == "ok"
    publication = harness.service.store.get_round(round_id)["publication_doc"]
    assert publication["king_decision"]["outcome"] == "no_king"
    assert harness.service.activate_reward(round_id)["status"] == "activated"
    assert harness.service.activate_reward(round_id)["status"] == "existing"
    with psycopg2.connect(**dsn) as connection, connection.cursor() as cursor:
        baseline = next(row for row in publication["final_ranking"] if row["is_baseline"])
        miner = next(row for row in publication["final_ranking"] if not row["is_baseline"])
        low = deepcopy(miner)
        low["training_data_eligibility"] = MARKER
        assert _verify(cursor, round_id, low)
        baseline_unknown = deepcopy(baseline)
        baseline_unknown["training_data_eligibility"] = MARKER
        assert _verify(cursor, round_id, baseline_unknown)
        malformed = deepcopy(low)
        malformed["training_data_eligibility"] = {**MARKER, "raw_trace": "private"}
        assert not _verify(cursor, round_id, malformed)
        cost_forgery = deepcopy(low)
        cost_forgery["cost_summary"]["per_icp"][0]["competition_sourcing_microusd"] += 1
        assert not _verify(cursor, round_id, cost_forgery)
        eligibility_forgery = deepcopy(low)
        eligibility_forgery.update(eligible=False, eligibility_reason="cost_per_company_exceeded")
        assert not _verify(cursor, round_id, eligibility_forgery)

        original = deepcopy(publication)
        for row in original["final_ranking"]:
            row["training_data_eligibility"] = deepcopy(MARKER)
        cursor.execute("SET LOCAL session_replication_role=replica")
        cursor.execute("UPDATE public.lab_arena_rounds SET status='scored', "
                       "publication_doc=NULL, reward_activated_at=NULL, reward_basis_doc=NULL "
                       "WHERE round_id=%s", (round_id,))
        cursor.execute("SET LOCAL session_replication_role=origin")
        assert _try_publish(cursor, round_id, original)

        # At equality with margin zero, the same measured miner can crown.
        # The outer guard rejects an unassessed possible winner even though
        # the row validator accepts the exact marker.
        cursor.execute("SET LOCAL session_replication_role=replica")
        cursor.execute("UPDATE public.lab_arena_rounds SET status='scored', "
                       "publication_doc=NULL, configuration_doc="
                       "jsonb_set(configuration_doc,'{promotion_margin}','0'::jsonb) "
                       "WHERE round_id=%s", (round_id,))
        cursor.execute("SET LOCAL session_replication_role=origin")
        assert not _try_publish(cursor, round_id, original,
                                expected_error="lab_arena_publication_training_report_mismatch")
        full = deepcopy(original)
        for row in full["final_ranking"]:
            row["training_data_eligibility"] = _training(True)
        full["king_decision"] = {
            "outcome": "crowned", "winner_submission_id": miner["submission_id"],
            "king_submission_id": miner["submission_id"],
            "king_hotkey": next(p["miner_hotkey"] for p in full["participants"]
                                if p["submission_id"] == miner["submission_id"]),
        }
        assert _try_publish(cursor, round_id, full)
        cursor.execute("SET LOCAL session_replication_role=replica")
        cursor.execute("UPDATE public.lab_arena_rounds SET status='scored', "
                       "publication_doc=NULL WHERE round_id=%s", (round_id,))
        cursor.execute("SET LOCAL session_replication_role=origin")
        full["final_ranking"] = [
            {**row, "training_data_eligibility": deepcopy(MARKER)}
            if row["is_baseline"] else row for row in full["final_ranking"]
        ]
        assert not _try_publish(
            cursor, round_id, full,
            expected_error="lab_arena_publication_training_report_mismatch",
        )  # Reward baseline stays audited.
        for row in full["final_ranking"]:
            if row["is_baseline"]:
                row["training_data_eligibility"] = _training(True)
        # Reopen only this disposable fixture, including the prior canonical
        # publication authority. Production write-once guards stay enabled for
        # both competing RPCs below.
        cursor.execute("SET LOCAL session_replication_role=replica")
        cursor.execute(
            "UPDATE public.lab_arena_rounds SET status='scored', publication_doc=NULL, "
            "published_at=NULL, king_outcome=NULL, king_hotkey=NULL, "
            "reward_activated_at=NULL, reward_basis_doc=NULL, reward_basis_hash=NULL, "
            "effective_reward_epoch=NULL, king_start_epoch=NULL WHERE round_id=%s",
            (round_id,),
        )
        cursor.execute("SET LOCAL session_replication_role=origin")

    # Both transactions start against the same scored row. The validated
    # publication transition is atomic: one commits and one sees stale state.
    barrier = Barrier(2)

    def publish_once():
        with psycopg2.connect(**dsn) as connection, connection.cursor() as cursor:
            cursor.execute("SET LOCAL statement_timeout='10s'")
            barrier.wait(timeout=10)
            cursor.execute("SET LOCAL ROLE lab_arena_service")
            cursor.execute(
                "SELECT public.lab_arena_transition_round(%s,'scored','published',%s::jsonb)",
                (round_id, json.dumps({"publication_doc": full,
                                      "published_at": full["published_at"]})),
            )
            return cursor.fetchone()[0]["status"]

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _index: publish_once(), range(2)))
    assert sorted(results) == ["ok", "stale"]
