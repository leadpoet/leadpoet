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
from tests.lab_arena import sep20_scored_authority_rejudge335_postgres_test as rerun335
from tests.lab_arena.lab_arena_pg_harness import CURRENT_SERVICE_MIGRATIONS


ROUND = rerun332.ROUND
ARCHIVE = ROUND + "-r335archive"
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


def test_changed_decision_publishes_without_rewriting_reward_or_weights(
    database, tmp_path, monkeypatch
):
    psycopg2, dsn = database
    harness = rerun332.IsolatedHarness(
        lambda: psycopg2.connect(**dsn), tmp_path, challengers=[], runners=["alpha"]
    )
    harness.round_id = ROUND
    with psycopg2.connect(**dsn) as connection:
        schedule = rerun335._prepare_scored_rerun332(
            connection, harness, monkeypatch
        )
        with connection.cursor() as cursor:
            rendered335, _, _ = rerun335._render335(
                cursor, schedule, harness.objects, monkeypatch
            )
            cursor.execute(rendered335)
        connection.commit()

        rerun332._drive_rejudge_cycle(
            harness.service, harness.objects, harness.runner_keys[0]
        )
        publication = _capture_publication(harness.service)
        with connection.cursor() as cursor:
            archived = rerun332._json(
                cursor,
                "SELECT publication_doc->'king_decision' FROM public.lab_arena_rounds "
                "WHERE round_id=%s",
                (ARCHIVE,),
            )
            assert publication["king_decision"] != archived
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
    assert "assignment_id NOT LIKE '%:score:rerun335'" in body
    assert "__CORRECTED_KING_DECISION_SHA256__" in body
    assert "DISABLE TRIGGER USER" in body
    assert "CREATE OR REPLACE FUNCTION" not in body
    assert "stable_after IS DISTINCT FROM stable_before" in body
    assert "lab_arena_000_sep20_rejudge332_publication_stop" in body


def test_disabled_user_trigger_is_rejected_without_changing_operator_state(
    database, tmp_path, monkeypatch
):
    psycopg2, dsn = database
    harness = rerun332.IsolatedHarness(
        lambda: psycopg2.connect(**dsn), tmp_path, challengers=[], runners=["alpha"]
    )
    harness.round_id = ROUND
    with psycopg2.connect(**dsn) as connection:
        schedule = rerun335._prepare_scored_rerun332(
            connection, harness, monkeypatch
        )
        with connection.cursor() as cursor:
            rendered335, _, _ = rerun335._render335(
                cursor, schedule, harness.objects, monkeypatch
            )
            cursor.execute(rendered335)
        connection.commit()
        rerun332._drive_rejudge_cycle(
            harness.service, harness.objects, harness.runner_keys[0]
        )
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
