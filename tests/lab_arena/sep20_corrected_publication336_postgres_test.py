"""Real PostgreSQL proof for the inactive Sep20 corrected publication release."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

from lab_arena import weight_state
from leadpoet_canonical import arena_weights
from tests.lab_arena import sep20_authority_preserving_rejudge332_postgres_test as rerun332
from tests.lab_arena import sep20_scored_authority_rejudge337_postgres_test as rerun337
from tests.lab_arena.lab_arena_pg_harness import CURRENT_SERVICE_MIGRATIONS


ROUND = rerun332.ROUND
ARCHIVE = ROUND + "-r337archive"
SCORE_NAMESPACE = "rerun337"
TEMPLATE = (
    Path(__file__).parents[2]
    / "scripts/336-arena-2026-09-20-corrected-publication-release.sql.template"
)


@pytest.fixture()
def database():
    yield from rerun332.database.__wrapped__()


def _digest(cursor, expression: str, parameters=()) -> str:
    cursor.execute(
        "SELECT encode(extensions.digest((" + expression + ")::text,'sha256'),'hex')",
        parameters,
    )
    return cursor.fetchone()[0]


def _capture_publication(service) -> dict:
    captured = {}
    transition = service.store.transition_round

    def capture(round_id, expected_status, next_status, patch=None):
        assert (round_id, expected_status, next_status) == (
            ROUND,
            "scored",
            "published",
        )
        captured.update(patch or {})
        return {"status": "captured", "round_status": "scored"}

    service.store.transition_round = capture
    try:
        assert service.publish(ROUND)["status"] == "captured"
    finally:
        service.store.transition_round = transition
    return captured["publication_doc"]


def _render(cursor, publication: dict) -> str:
    values = {
        "__ARCHIVE_ROUND_ID__": ARCHIVE,
        "__ARCHIVE_ROUND_SHA256__": _digest(
            cursor,
            "SELECT to_jsonb(r) FROM public.lab_arena_rounds r WHERE round_id=%s",
            (ARCHIVE,),
        ),
        "__SCORED_STABLE_ROUND_SHA256__": _digest(
            cursor,
            "SELECT jsonb_build_object("
            "'configuration_doc',configuration_doc,'rewards_enabled',rewards_enabled,"
            "'participants',participants,"
            "'benchmark_ref',benchmark_ref,'evaluation_date',evaluation_date,"
            "'icp_set_date',icp_set_date,'stage1_scoring_plan_doc',stage1_scoring_plan_doc,"
            "'stage2_scoring_plan_doc',stage2_scoring_plan_doc,'finalists',finalists,"
            "'stage3_scoring_plan_doc',stage3_scoring_plan_doc,"
            "'confirmation_bank_ref',confirmation_bank_ref,"
            "'confirmation_bank_hash',confirmation_bank_hash,"
            "'confirmation_cohort',confirmation_cohort,"
            "'champion_funding_frozen',champion_funding_frozen,"
            "'champion_submission_id',champion_submission_id,"
            "'champion_hotkey',champion_hotkey,"
            "'champion_fallback_providers',champion_fallback_providers,"
            "'effective_reward_epoch',effective_reward_epoch,"
            "'reward_basis_hash',reward_basis_hash,'reward_basis_doc',reward_basis_doc,"
            "'signing_key_doc',signing_key_doc,'reward_activated_at',reward_activated_at,"
            "'king_start_epoch',king_start_epoch,'promotion_required',promotion_required,"
            "'promotion_doc',promotion_doc,'baseline_promoted_at',baseline_promoted_at) "
            "FROM public.lab_arena_rounds WHERE round_id=%s",
            (ROUND,),
        ),
        "__CORRECTED_KING_DECISION_SHA256__": _digest(
            cursor, "%s::jsonb", (json.dumps(publication["king_decision"]),)
        ),
        "__SCORE_RUN_COUNT__": str(
            int(
                rerun332._scalar(
                    cursor,
                    "SELECT count(*) FROM public.lab_arena_runs "
                    "WHERE round_id=%s AND kind='score'",
                    (ROUND,),
                )
            )
        ),
        "__SCORE_NAMESPACE__": SCORE_NAMESPACE,
    }
    for marker, table, order in (
        ("__SUBMISSIONS_SHA256__", "lab_arena_submissions", "submission_id"),
        ("__RUNS_SHA256__", "lab_arena_runs", "run_id"),
        ("__LEDGER_SHA256__", "lab_arena_ledger", "entry_id"),
    ):
        values[marker] = rerun332._scalar(
            cursor,
            "SELECT encode(extensions.digest(coalesce(string_agg(encode("
            "extensions.digest(to_jsonb(item)::text,'sha256'),'hex'),'' ORDER BY "
            + order
            + "),''),'sha256'),'hex') FROM public."
            + table
            + " item WHERE round_id=%s",
            (ROUND,),
        )
    cursor.execute(
        "SELECT pg_get_functiondef("
        "'public.lab_arena_rounds_write_once_v1()'::regprocedure)"
    )
    definition = cursor.fetchone()[0]
    values["__WRITE_ONCE_DEFINITION_SHA256__"] = hashlib.sha256(
        definition.encode()
    ).hexdigest()
    cursor.execute(
        "SELECT king_outcome,coalesce(king_hotkey,'') FROM public.lab_arena_rounds "
        "WHERE round_id=%s",
        (ROUND,),
    )
    old_outcome, old_hotkey = cursor.fetchone()
    decision = publication["king_decision"]
    values.update(
        {
            "__OLD_KING_OUTCOME__": old_outcome,
            "__OLD_KING_HOTKEY__": old_hotkey,
            "__CORRECTED_KING_OUTCOME__": decision["outcome"],
            "__CORRECTED_KING_HOTKEY__": decision.get("king_hotkey") or "",
            "__CORRECTED_KING_DECISION_JSON__": json.dumps(
                decision, sort_keys=True, separators=(",", ":")
            ).replace("'", "''"),
        }
    )
    rendered = TEMPLATE.read_text()
    for marker, value in values.items():
        rendered = rendered.replace(marker, value)
    assert not re.search(r"__[A-Z0-9_]+__", rendered)
    return rendered


def _reward_promotion_snapshot(cursor):
    cursor.execute(
        "SELECT rewards_enabled,effective_reward_epoch,reward_basis_hash,"
        "reward_basis_doc,signing_key_doc,reward_activated_at,king_start_epoch,"
        "promotion_required,promotion_doc,baseline_promoted_at "
        "FROM public.lab_arena_rounds WHERE round_id=%s",
        (ROUND,),
    )
    return cursor.fetchone()


def _weight_snapshot(cursor):
    cursor.execute(
        "SELECT jsonb_agg(to_jsonb(w) ORDER BY network,netuid,epoch) "
        "FROM public.lab_arena_accepted_weight_states w"
    )
    return cursor.fetchone()[0]


def _prepare_scored337(connection, harness, monkeypatch):
    rerun337._prepare_scored335(connection, harness, monkeypatch)
    with connection.cursor() as cursor:
        rendered337, _, _ = rerun337._render337(
            cursor, rerun332._schedule(), harness.objects, monkeypatch
        )
        cursor.execute(rendered337)
    connection.commit()
    rerun332._drive_rejudge_cycle(
        harness.service, harness.objects, harness.runner_keys[0]
    )
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT active_round.status,count(*) FILTER(WHERE run.kind='score'),"
            "count(*) FILTER(WHERE run.kind='score' AND run.status='accepted' "
            "AND run.terminal_cause='accepted' "
            "AND run.assignment_id LIKE '%%:score:rerun337') "
            "FROM public.lab_arena_rounds active_round "
            "JOIN public.lab_arena_runs run USING(round_id) "
            "WHERE active_round.round_id=%s GROUP BY active_round.status",
            (ROUND,),
        )
        assert cursor.fetchone() == ("scored", 98, 98)
    connection.commit()


def _add_superseded_failed_score_attempt(
    cursor, *, linked=True, terminal_cause="judge_error"
):
    cursor.execute(
        "SELECT run_id FROM public.lab_arena_runs WHERE round_id=%s "
        "AND kind='score' AND status='accepted' ORDER BY run_id LIMIT 1",
        (ROUND,),
    )
    accepted_run_id = cursor.fetchone()[0]
    cursor.execute("ALTER TABLE public.lab_arena_runs DISABLE TRIGGER USER")
    cursor.execute(
        "UPDATE public.lab_arena_runs SET attempt=2,updated_at=clock_timestamp() "
        "WHERE run_id=%s RETURNING assignment_id",
        (accepted_run_id,),
    )
    assignment_id = cursor.fetchone()[0]
    failed_assignment_id = assignment_id if linked else assignment_id.replace(
        ":score:rerun337", ":orphan:score:rerun337"
    )
    cursor.execute(
        "INSERT INTO public.lab_arena_runs SELECT (jsonb_populate_record("
        "NULL::public.lab_arena_runs,to_jsonb(failed)||jsonb_build_object("
        "'run_id',failed.run_id||':failed','assignment_id',%s,'attempt',1,"
        "'status','failed','terminal_cause',%s,'result_doc',NULL,"
        "'claim_request_id',NULL,'claim_request_hash',NULL,'claim_response',NULL,"
        "'created_at',clock_timestamp(),'updated_at',clock_timestamp()))).* "
        "FROM public.lab_arena_runs failed WHERE failed.run_id=%s",
        (failed_assignment_id, terminal_cause, accepted_run_id),
    )
    cursor.execute("ALTER TABLE public.lab_arena_runs ENABLE TRIGGER USER")
    cursor.execute(
        "SELECT count(*),count(*) FILTER(WHERE status='accepted'),"
        "count(*) FILTER(WHERE status='failed') FROM public.lab_arena_runs "
        "WHERE round_id=%s AND kind='score' AND assignment_id=%s",
        (ROUND, assignment_id),
    )
    assert cursor.fetchone() == ((2, 1, 1) if linked else (1, 1, 0))


def test_changed_decision_publishes_without_rewriting_reward_or_weights(
    database, tmp_path, monkeypatch
):
    psycopg2, dsn = database
    harness = rerun332.IsolatedHarness(
        lambda: psycopg2.connect(**dsn), tmp_path, challengers=[], runners=["alpha"]
    )
    harness.round_id = ROUND
    with psycopg2.connect(**dsn) as connection:
        _prepare_scored337(connection, harness, monkeypatch)
        publication = _capture_publication(harness.service)
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT king_outcome,king_hotkey FROM public.lab_arena_rounds "
                "WHERE round_id=%s",
                (ARCHIVE,),
            )
            archived_decision = cursor.fetchone()
            assert archived_decision != (
                publication["king_decision"]["outcome"],
                publication["king_decision"].get("king_hotkey") or None,
            )
            historical = _reward_promotion_snapshot(cursor)
            weights = _weight_snapshot(cursor)
            rendered336 = _render(cursor, publication)
            cursor.execute(rendered336)
            cursor.execute(
                "SELECT king_outcome,king_hotkey FROM public.lab_arena_rounds "
                "WHERE round_id=%s",
                (ROUND,),
            )
            assert cursor.fetchone() == (
                publication["king_decision"]["outcome"],
                publication["king_decision"].get("king_hotkey") or None,
            )
            cursor.execute("BEGIN")
            cursor.execute("SAVEPOINT reward_drift")
            with pytest.raises(
                psycopg2.Error,
                match="write-once",
            ):
                cursor.execute(
                    "UPDATE public.lab_arena_rounds SET status='published',"
                    "status_generation=status_generation+1,publication_doc=%s::jsonb,"
                    "published_at=%s::timestamptz,king_outcome=%s,king_hotkey=%s,"
                    "reward_basis_hash='sha256:'||repeat('0',64) WHERE round_id=%s",
                    (
                        json.dumps(publication),
                        publication["published_at"],
                        publication["king_decision"]["outcome"],
                        publication["king_decision"].get("king_hotkey") or None,
                        ROUND,
                    ),
                )
            cursor.execute("ROLLBACK TO SAVEPOINT reward_drift")
        connection.commit()

        assert harness.service.publish(ROUND)["status"] == "ok"
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT status,publication_doc,king_outcome,king_hotkey "
                "FROM public.lab_arena_rounds WHERE round_id=%s",
                (ROUND,),
            )
            status, saved, outcome, hotkey = cursor.fetchone()
            assert status == "published"
            assert saved["king_decision"] == publication["king_decision"]
            assert outcome == publication["king_decision"]["outcome"]
            assert hotkey == (publication["king_decision"].get("king_hotkey") or None)
            assert _reward_promotion_snapshot(cursor) == historical
            assert _weight_snapshot(cursor) == weights
            assert harness.service.public_reward_basis(int(historical[1])) == historical[3]
        rows = harness.service.store.published_reward_bases(
            mode="live", network_name="finney", netuid=71, limit=200
        )
        usable = harness.service._usable_reward_bases(rows)
        assert historical[3] in usable
        assert harness.service._latest_miner_basis(usable) == historical[3]
        burn_hotkey = rerun332.fixtures.keypair("correction336-burn").ss58_address
        accepted = weight_state.build_accepted_weight_state(
            harness.signer,
            network="finney",
            genesis_hash="1" * 64,
            netuid=71,
            epoch=int(historical[1]),
            valid_from_block=1,
            valid_until_block=360,
            reward_basis=historical[3],
            burn_hotkey=burn_hotkey,
            issued_at=historical[3]["published_at"],
        )
        assert arena_weights.verify_accepted_weight_state_signature(
            accepted,
            public_key_der=harness.signer.public_key_der,
            expected_public_key_hash=harness.signer.public_key_hash,
        ) == accepted["state_hash"]
        vector = arena_weights.derive_arena_weights(
            accepted, [historical[3]["king_hotkey"], burn_hotkey]
        )
        assert vector["champion_share_ppb"] > 0
        competition = harness.service.public_competition()
        completed = competition["latest_completed_round"]
        assert completed["round_id"] == ROUND
        if publication["king_decision"]["outcome"] == "no_king":
            assert completed["champion"] is None
        else:
            assert completed["champion"]["submission_id"] == publication[
                "king_decision"
            ]["winner_submission_id"]
        assert harness.service.promote_baseline(ROUND)["status"] == "existing"
        assert harness.service.activate_reward(ROUND)["status"] == "existing"
        with psycopg2.connect(**dsn) as retry:
            with retry.cursor() as cursor:
                cursor.execute(rendered336)
            retry.commit()


def test_template_is_inactive_and_separates_publication_from_reward():
    body = TEMPLATE.read_text()
    assert TEMPLATE.name not in CURRENT_SERVICE_MIGRATIONS
    assert body.count("UPDATE public.lab_arena_rounds SET") == 1
    assert "king_outcome=corrected_outcome,king_hotkey=corrected_hotkey" in body
    assert "INSERT INTO" not in body and "DELETE FROM" not in body
    assert "assignment_id NOT LIKE '%:score:__SCORE_NAMESPACE__'" in body
    assert "round_id='__ARCHIVE_ROUND_ID__'" in body
    assert "accepted_retry.attempt>score_run.attempt" in body
    assert "count(DISTINCT scored_run_id)" in body
    assert "archived.king_outcome IS DISTINCT FROM '__OLD_KING_OUTCOME__'" in body
    assert "__CORRECTED_KING_DECISION_SHA256__" in body
    assert "DISABLE TRIGGER USER" in body
    assert "CREATE OR REPLACE FUNCTION" not in body
    assert "stable_after IS DISTINCT FROM stable_before" in body
    assert "lab_arena_000_sep20_rejudge332_publication_stop" in body


def test_superseded_failed_score_attempt_is_preserved_and_hash_bound(
    database, tmp_path, monkeypatch
):
    psycopg2, dsn = database
    harness = rerun332.IsolatedHarness(
        lambda: psycopg2.connect(**dsn), tmp_path, challengers=[], runners=["alpha"]
    )
    harness.round_id = ROUND
    with psycopg2.connect(**dsn) as connection:
        _prepare_scored337(connection, harness, monkeypatch)
        publication = _capture_publication(harness.service)
        with connection.cursor() as cursor:
            _add_superseded_failed_score_attempt(cursor)
            rendered = _render(cursor, publication)
            assert "<>99" in rendered
            cursor.execute("SAVEPOINT failed_history_hash")
            cursor.execute("ALTER TABLE public.lab_arena_runs DISABLE TRIGGER USER")
            cursor.execute(
                "UPDATE public.lab_arena_runs SET terminal_doc='{}'::jsonb "
                "WHERE round_id=%s AND kind='score' AND status='failed'",
                (ROUND,),
            )
            cursor.execute("ALTER TABLE public.lab_arena_runs ENABLE TRIGGER USER")
            with pytest.raises(
                psycopg2.Error, match="corrected publication preimage differs"
            ):
                cursor.execute(rendered)
            cursor.execute("ROLLBACK TO SAVEPOINT failed_history_hash")
            cursor.execute(rendered)
            cursor.execute(
                "SELECT count(*),count(*) FILTER(WHERE status='accepted'),"
                "count(*) FILTER(WHERE status='failed') FROM public.lab_arena_runs "
                "WHERE round_id=%s AND kind='score'",
                (ROUND,),
            )
            assert cursor.fetchone() == (99, 98, 1)
        connection.commit()


def test_failed_score_without_later_accepted_attempt_is_rejected(
    database, tmp_path, monkeypatch
):
    psycopg2, dsn = database
    harness = rerun332.IsolatedHarness(
        lambda: psycopg2.connect(**dsn), tmp_path, challengers=[], runners=["alpha"]
    )
    harness.round_id = ROUND
    with psycopg2.connect(**dsn) as connection:
        _prepare_scored337(connection, harness, monkeypatch)
        publication = _capture_publication(harness.service)
        with connection.cursor() as cursor:
            _add_superseded_failed_score_attempt(cursor, linked=False)
            rendered = _render(cursor, publication)
            with pytest.raises(
                psycopg2.Error, match="corrected publication preimage differs"
            ):
                cursor.execute(rendered)
        connection.rollback()


def test_non_judge_failure_with_later_accepted_attempt_is_rejected(
    database, tmp_path, monkeypatch
):
    psycopg2, dsn = database
    harness = rerun332.IsolatedHarness(
        lambda: psycopg2.connect(**dsn), tmp_path, challengers=[], runners=["alpha"]
    )
    harness.round_id = ROUND
    with psycopg2.connect(**dsn) as connection:
        _prepare_scored337(connection, harness, monkeypatch)
        publication = _capture_publication(harness.service)
        with connection.cursor() as cursor:
            _add_superseded_failed_score_attempt(
                cursor, terminal_cause="provider_error"
            )
            rendered = _render(cursor, publication)
            with pytest.raises(
                psycopg2.Error, match="corrected publication preimage differs"
            ):
                cursor.execute(rendered)
        connection.rollback()


def test_no_king_alignment_keeps_historical_reward_authority(
    database, tmp_path, monkeypatch
):
    psycopg2, dsn = database
    harness = rerun332.IsolatedHarness(
        lambda: psycopg2.connect(**dsn), tmp_path, challengers=[], runners=["alpha"]
    )
    harness.round_id = ROUND
    with psycopg2.connect(**dsn) as connection:
        _prepare_scored337(connection, harness, monkeypatch)
        publication = _capture_publication(harness.service)
        publication["king_decision"] = {"outcome": "no_king", "king_hotkey": ""}
        with connection.cursor() as cursor:
            historical = _reward_promotion_snapshot(cursor)
            weights = _weight_snapshot(cursor)
            cursor.execute(_render(cursor, publication))
            cursor.execute(
                "SELECT status,king_outcome,king_hotkey,reward_basis_hash "
                "FROM public.lab_arena_rounds WHERE round_id=%s",
                (ROUND,),
            )
            assert cursor.fetchone() == (
                "scored", "no_king", None, historical[2]
            )
            assert _reward_promotion_snapshot(cursor) == historical
            assert _weight_snapshot(cursor) == weights
        connection.commit()


def test_disabled_user_trigger_is_rejected_without_changing_operator_state(
    database, tmp_path, monkeypatch
):
    psycopg2, dsn = database
    harness = rerun332.IsolatedHarness(
        lambda: psycopg2.connect(**dsn), tmp_path, challengers=[], runners=["alpha"]
    )
    harness.round_id = ROUND
    with psycopg2.connect(**dsn) as connection:
        _prepare_scored337(connection, harness, monkeypatch)
        publication = _capture_publication(harness.service)
        with connection.cursor() as cursor:
            rendered336 = _render(cursor, publication)
            cursor.execute(
                "ALTER TABLE public.lab_arena_rounds DISABLE TRIGGER "
                "lab_arena_rounds_write_once"
            )
        connection.commit()
        with pytest.raises(
            psycopg2.Error, match="USER trigger state differs"
        ):
            with connection.cursor() as cursor:
                cursor.execute(rendered336)
        connection.rollback()
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT tgenabled FROM pg_trigger WHERE tgrelid="
                "'public.lab_arena_rounds'::regclass "
                "AND tgname='lab_arena_rounds_write_once'"
            )
            assert cursor.fetchone() == ("D",)
            cursor.execute(
                "ALTER TABLE public.lab_arena_rounds ENABLE TRIGGER "
                "lab_arena_rounds_write_once"
            )
        connection.commit()
