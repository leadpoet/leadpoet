"""Saved-result verification of integrity rounds on disposable PostgreSQL."""
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from lab_arena import contracts, integrity, service as svc, verify, weight_state
from lab_arena.chain import MetagraphSnapshot
from lab_arena.store import ArenaStoreError
from leadpoet_canonical import arena_weights
from qualification.scoring.arena_integrity import canonical_company_identity
from tests.lab_arena import test_lab_arena_service_round as fixtures
from tests.lab_arena.lab_arena_pg_harness import (
    CURRENT_SERVICE_MIGRATIONS,
    DEFAULT_MIGRATIONS,
    database_with_lab_arena_migration,
)

MIGRATIONS = DEFAULT_MIGRATIONS + (
    "211-lab-arena-owner-admission.sql", "212-lab-arena-accepted-judgment-cache.sql", "213-lab-arena-score-integrity.sql", "214-lab-arena-prior-credential-refusal.sql",
)


@pytest.fixture()
def database():
    yield from database_with_lab_arena_migration(CURRENT_SERVICE_MIGRATIONS)


@pytest.fixture()
def database_before_251():
    yield from database_with_lab_arena_migration(
        CURRENT_SERVICE_MIGRATIONS[
            :CURRENT_SERVICE_MIGRATIONS.index(
                "251-lab-arena-twenty-icp-promotion.sql"
            )
        ]
    )


@pytest.fixture()
def historical_database():
    yield from database_with_lab_arena_migration(
        MIGRATIONS
        + (
            "223-lab-arena-cancelled-call-late-settlement.sql",
            "225-lab-arena-openrouter-delayed-cost-reconciliation.sql",
            "227-lab-arena-champion-funding.sql",
            "229-lab-arena-successful-call-cost-eligibility.sql",
            "230-lab-arena-successful-call-cost-permissions.sql",
        )
    )


def test_integrity_migration_replays_and_keeps_private_function_grants(
    historical_database,
):
    psycopg2, dsn = historical_database
    migration = Path(__file__).resolve().parents[2] / "scripts" / "213-lab-arena-score-integrity.sql"
    with psycopg2.connect(**dsn) as connection, connection.cursor() as cursor:
        cursor.execute(migration.read_text())
        cursor.execute("SELECT public.lab_arena_integrity_schema_v1()")
        assert cursor.fetchone()[0] == {"schema_version": "leadpoet.lab_arena.integrity_schema.v1", "version": 213}
        for function in ("public.lab_arena_prepare_confirmation_bank(text,text,text)", "public.lab_arena_open_confirmation(text,jsonb)", "public.lab_arena_integrity_schema_v1()"):
            cursor.execute("SELECT has_function_privilege('lab_arena_service', %s, 'EXECUTE'), has_function_privilege('anon', %s, 'EXECUTE'), has_function_privilege('authenticated', %s, 'EXECUTE')", (function, function, function))
            assert cursor.fetchone() == (True, False, False)
        cursor.execute("SELECT has_function_privilege('lab_arena_service', 'public.lab_arena__integrity_eligibility(text,text,integer[])', 'EXECUTE')")
        assert cursor.fetchone()[0] is False
        for role in ("lab_arena_service", "anon", "authenticated"):
            cursor.execute("SELECT has_function_privilege(%s, 'public.lab_arena__confirmation_account_failure(text,text)', 'EXECUTE')", (role,))
            assert cursor.fetchone()[0] is False


def test_legacy_rounds_still_publish_and_promote_after_integrity_migrations(database, tmp_path):
    psycopg2, dsn = database
    fixtures.test_full_round_publishes_results_and_next_day_uses_the_public_baseline(
        lambda: psycopg2.connect(**dsn), tmp_path,
    )


class IntegrityHarness(fixtures.Harness):
    def objects_key(self):
        return "integrity-round"

    def build_service(self):
        service = super().build_service()
        service._config.defaults = replace(service._config.defaults, integrity_from="2026-01-01T00:00:00Z")
        def metagraph(*, finalized=True):
            assert finalized
            hotkeys = tuple(key.ss58_address for key in fixtures.KEYS.values())
            return MetagraphSnapshot(netuid=71, block_number=123, block_hash="0x" + "a" * 64,
                hotkeys=hotkeys, coldkeys=hotkeys,
                validator_permit=tuple(key in self.runner_keys for key in hotkeys),
                stake=tuple(100_000.0 if key in self.runner_keys else 0.0 for key in hotkeys),
                active=tuple(key in self.runner_keys for key in hotkeys))
        self.chain.metagraph = metagraph
        return service

    def advance_until(self, target, *, runners=1, max_steps=80):
        for _ in range(max_steps):
            status = self.status()
            if status == target:
                return self.service.store.get_round(self.round_id)
            if status in ("stage1", "stage1_scoring", "stage2", "stage2_scoring"):
                self.run_stage_with_runners(runners)
            outcome = self.service.advance_round(self.round_id)
            assert outcome.get("status") not in ("cancelled", "terminal", "retry", "stale"), (status, outcome)
        raise AssertionError((target, self.status()))


def test_closed_judge_failures_keep_billing_evidence_without_harming_healthy_work(
    database, tmp_path, monkeypatch
):
    psycopg2, dsn = database
    harness = IntegrityHarness(lambda: psycopg2.connect(**dsn), tmp_path,
        challengers=["Outage", "Healthy"], runners=["alpha"])
    def judge(companies, icp, reference):
        indexes, _ = verify.bucket_skip(icp, companies)
        return [{"final_score": 40.0, "company_index": index,
            "company_identity_key": canonical_company_identity(companies[index]).key,
            "company_qualified": True, "duplicate_company": False,
            "verifier_gate_receipts": [{"gate": "company_fit", "decision": "match"}],
            "intent_signals_detail": [], "failure_reason": ""} for index in indexes]
    monkeypatch.setattr(fixtures, "deterministic_scorer", judge)
    round_id = "arena-2026-11-01-outage"
    harness.clock.now = datetime.now(timezone.utc)
    harness.service.create_round(harness.clock.now + timedelta(minutes=30), round_id=round_id)
    harness.round_id = round_id
    submission = harness.submit("Outage", round_id)
    healthy_submission = harness.submit("Healthy", round_id)
    harness.clock.advance_to(harness.schedule()["submission_cutoff"])
    harness.advance_until("stage1_scoring")
    credentials = harness.service.config.credential_manager
    original_key = credentials.runtime_key
    monkeypatch.setattr(credentials, "runtime_key", lambda row, provider:
        "miner-outage" if row["submission_id"] == submission else original_key(row, provider))
    original_send = fixtures.FakeProviderTransport.send

    outage_active = True

    def unavailable(self, **kwargs):
        if outage_active and any(
            "miner-outage" in str(value) for value in kwargs["headers"].values()
        ):
            return fixtures.br.ProviderResponse(503, {"content-type": "application/json"}, b'{"error":"temporarily unavailable"}')
        return original_send(self, **kwargs)

    monkeypatch.setattr(fixtures.FakeProviderTransport, "send", unavailable)
    harness.run_stage_with_runners(1)
    runs = [
        run
        for run in harness.service.store.list_runs(round_id, stage=1, kind="score")
        if run["submission_id"] == submission
    ]
    failed = [run for run in runs if run["terminal_cause"] == "judge_error"]
    assert len(failed) == contracts.STAGE_1_ICP_COUNT * 2
    assert all(run["status"] == "failed" for run in failed)
    assert {run["attempt"] for run in failed} == {1, 2}
    assert {run["icp_position"] for run in failed} == set(
        contracts.stage_positions(1)
    )
    assert not any(run["status"] == "pending" for run in runs)
    assert not any(run["terminal_cause"] == "credential_error" for run in runs)
    healthy_runs = [
        run
        for run in harness.service.store.list_runs(round_id, stage=1, kind="score")
        if run["submission_id"] == healthy_submission
    ]
    assert healthy_runs and all(run["status"] == "accepted" for run in healthy_runs)

    candidates = harness.service.store.list_deepline_cost_reconciliations(
        round_id, run_id=failed[0]["run_id"]
    )
    assert len(candidates) == 1
    candidate = candidates[0]
    ledger_before = harness.service.store.list_ledger(
        call_identity=candidate["call_identity"]
    )
    assert ledger_before[-1]["entry_kind"] == "uncertain"
    settled = harness.service.store.reconcile_deepline_cost(
        round_id=round_id,
        run_id=failed[0]["run_id"],
        call_identity=candidate["call_identity"],
        uncertain_entry_id=candidate["uncertain_entry_id"],
        request_id=candidate["request_id"],
        operation=candidate["operation"],
        credential_fingerprint=candidate["credential_fingerprint"],
        actual_microusd=2_000,
        cost_units="0.02",
    )
    assert settled["status"] == "settled"
    ledger_after = harness.service.store.list_ledger(
        call_identity=candidate["call_identity"]
    )
    assert ledger_after[:-1] == ledger_before
    assert ledger_after[-1]["entry_kind"] == "settlement"
    assert ledger_after[-1]["amount_microusd"] == 2_000
    assert harness.service.store.list_deepline_cost_reconciliations(
        round_id, run_id=failed[0]["run_id"]
    ) == []
    retained = harness.service.store.get_run(failed[0]["run_id"])
    assert retained["status"] == "failed"
    assert retained["terminal_cause"] == "judge_error"


def test_complete_twenty_icp_winner_cannot_be_vetoed_by_retired_stage(
    database, tmp_path, monkeypatch,
):
    psycopg2, dsn = database
    connect = lambda: psycopg2.connect(**dsn)
    harness = IntegrityHarness(
        connect, tmp_path, challengers=["Challenger"], runners=["alpha"]
    )

    def judge(companies, icp, reference):
        assert not reference and "verified_example_company" not in icp
        assert not str(icp["icp_id"]).startswith("confirmation_")
        indexes, _ = verify.bucket_skip(icp, companies)
        rows = []
        for index in indexes:
            baseline = companies[index]["company_name"].startswith("PublicBaseline")
            score = 40.0 if baseline else 80.0
            rows.append({"final_score": score, "company_index": index,
                "company_identity_key": canonical_company_identity(companies[index]).key,
                "company_qualified": True,
                "duplicate_company": False,
                "verifier_gate_receipts": [{"gate": "company_fit", "decision": "match"}],
                "intent_signals_detail": [], "failure_reason": ""})
        return rows

    monkeypatch.setattr(fixtures, "deterministic_scorer", judge)
    harness.chain.epoch = 25_080
    harness.clock.now = datetime.now(timezone.utc)
    harness.service.config.defaults = replace(
        harness.service.config.defaults, rewards_enabled=True
    )
    round_id = "arena-2026-11-01-twentyicp"
    config = harness.service.create_round(
        datetime.now(timezone.utc) + timedelta(minutes=30), round_id=round_id
    )
    harness.round_id = round_id
    assert config["integrity_policy"] == integrity.POLICY
    assert not {
        "stage_3_start", "stage_3_close", "stage_3_scoring_close"
    }.intersection(config["schedule"])
    challenger = harness.submit("Challenger", round_id)
    harness.clock.advance_to(harness.schedule()["submission_cutoff"])
    harness.advance_until("scored")
    harness.service = harness.build_service()
    captured = {}
    transition = harness.service.store.transition_round

    def capture_transition(round_id, expected, next_status, patch):
        captured["args"] = (round_id, expected, next_status, deepcopy(patch))
        return {"status": "captured"}

    monkeypatch.setattr(
        harness.service.store, "transition_round", capture_transition
    )
    assert harness.service.publish(round_id)["status"] == "captured"
    transition_args = captured["args"]
    publication = transition_args[3]["publication_doc"]
    challenger_row = next(
        row for row in publication["final_ranking"]
        if row["submission_id"] == challenger
    )
    adversarial = []
    wrong_score = deepcopy(transition_args[3])
    next(
        row for row in wrong_score["publication_doc"]["final_ranking"]
        if row["submission_id"] == challenger
    )["final_score"] += 1
    adversarial.append((wrong_score, "final_score_mismatch"))
    missing_finalist = deepcopy(transition_args[3])
    missing_finalist["publication_doc"]["final_ranking"] = [
        row for row in missing_finalist["publication_doc"]["final_ranking"]
        if row["submission_id"] != challenger
    ]
    adversarial.append((missing_finalist, "publication_ranking_incomplete"))
    wrong_cost = deepcopy(transition_args[3])
    next(
        row for row in wrong_cost["publication_doc"]["final_ranking"]
        if row["submission_id"] == challenger
    )["cost_summary"]["qualified_company_count"] -= 1
    adversarial.append((wrong_cost, "publication_cost_report_mismatch"))
    wrong_winner = deepcopy(transition_args[3])
    wrong_winner["publication_doc"]["king_decision"]["winner_submission_id"] = (
        "forged-submission"
    )
    adversarial.append((wrong_winner, "publication_winner_invalid"))
    for bad_patch, error in adversarial:
        with pytest.raises(ArenaStoreError, match=error):
            transition(round_id, "scored", "published", bad_patch)

    monkeypatch.setattr(harness.service.store, "transition_round", transition)
    accepted = transition(*transition_args)
    assert accepted["status"] == "ok"
    published = harness.service.store.get_round(round_id)
    assert published["king_outcome"] == "crowned"
    ranking = {
        row["submission_id"]: row
        for row in published["publication_doc"]["final_ranking"]
    }
    assert ranking[challenger]["final_score"] == 80.0
    assert challenger_row["final_score"] == 80.0
    assert ranking[challenger]["cost_summary"]["qualified_company_count"] == 100
    assert published["confirmation_bank_ref"] is None
    assert published["confirmation_bank_hash"] is None
    assert published["confirmation_cohort"] is None
    assert published["stage3_scoring_plan_doc"] is None
    all_runs = harness.service.store.list_runs(round_id)
    assert all(int(run["stage"]) in (1, 2) for run in all_runs)
    assert all(int(run["icp_position"]) <= 19 for run in all_runs)
    public = harness.service.public_results(round_id, challenger)
    assert set(public["scores"]) == {"stage_1", "stage_2"}
    assert len(public["outputs"]) == contracts.BENCHMARK_ICP_COUNT
    fixtures.assert_canary_absent(harness, connect)

    from lab_arena.promotion import GitPromoter
    repository_root = tmp_path / "promotion-repository"
    repository_root.mkdir()
    remote = fixtures.promotion_repository(repository_root)
    harness.service._config.baseline_promoter_factory = lambda: GitPromoter(
        str(remote), tmp_path / "promotion-objects"
    )
    harness.clock.now = datetime.fromisoformat(
        str(published["published_at"]).replace("Z", "+00:00")
    )
    assert harness.service.promote_pending_baselines() == {
        "status": "ok", "promoted": 1,
    }
    assert harness.service.activate_reward(round_id)["status"] == "activated"
    rewarded = harness.service.store.get_round(round_id)
    basis = rewarded["reward_basis_doc"]
    burn_hotkey = fixtures.keypair("twenty-icp-burn").ss58_address
    accepted = weight_state.build_accepted_weight_state(
        harness.signer,
        network="finney",
        genesis_hash="1" * 64,
        netuid=71,
        epoch=int(basis["effective_reward_epoch"]),
        valid_from_block=1,
        valid_until_block=360,
        reward_basis=basis,
        burn_hotkey=burn_hotkey,
        issued_at=basis["published_at"],
    )
    arena_weights.verify_accepted_weight_state_signature(
        accepted,
        public_key_der=harness.signer.public_key_der,
        expected_public_key_hash=harness.signer.public_key_hash,
    )
    vector = arena_weights.derive_arena_weights(
        accepted, [basis["king_hotkey"], burn_hotkey]
    )
    assert vector["champion_share_ppb"] > 0


def test_cutover_keeps_inert_bank_and_finishes_open_round_on_twenty_icps(
    database_before_251, tmp_path, monkeypatch,
):
    psycopg2, dsn = database_before_251
    connect = lambda: psycopg2.connect(**dsn)
    harness = IntegrityHarness(
        connect, tmp_path, challengers=["CutoverWinner"], runners=["alpha"]
    )
    def judge(companies, icp, reference):
        indexes, _ = verify.bucket_skip(icp, companies)
        return [{
            "final_score": 40.0 if companies[index]["company_name"].startswith(
                "PublicBaseline"
            ) else 80.0,
            "company_index": index,
            "company_identity_key": canonical_company_identity(companies[index]).key,
            "company_qualified": True,
            "duplicate_company": False,
            "verifier_gate_receipts": [{"gate": "company_fit", "decision": "match"}],
            "intent_signals_detail": [],
            "failure_reason": "",
        } for index in indexes]

    monkeypatch.setattr(fixtures, "deterministic_scorer", judge)
    harness.clock.now = datetime.now(timezone.utc)
    harness.service._require_integrity_schema = lambda: None
    round_id = "arena-2026-11-02-cutover"
    config = harness.service.create_round(
        harness.clock.now + timedelta(minutes=30), round_id=round_id
    )
    harness.round_id = round_id
    winner = harness.submit("CutoverWinner", round_id)
    confirmation_ref = f"arena/{round_id}/confirmation/{'c' * 64}.json"
    confirmation_hash = "sha256:" + "c" * 64
    with connect() as connection, connection.cursor() as cursor:
        cursor.execute(
            "ALTER TABLE public.lab_arena_rounds "
            "DISABLE TRIGGER lab_arena_rounds_write_once"
        )
        cursor.execute(
            """UPDATE public.lab_arena_rounds
               SET configuration_doc = jsonb_set(
                     configuration_doc, '{schedule}',
                     configuration_doc -> 'schedule' || jsonb_build_object(
                       'stage_3_start', %s,
                       'stage_3_close', %s,
                       'stage_3_scoring_close', %s
                     )
                   ),
                   confirmation_bank_ref = %s,
                   confirmation_bank_hash = %s
               WHERE round_id = %s""",
            (
                config["schedule"]["final_scoring_close"],
                config["schedule"]["publication_deadline"],
                config["schedule"]["publication_deadline"],
                confirmation_ref,
                confirmation_hash,
                round_id,
            ),
        )
        cursor.execute(
            "ALTER TABLE public.lab_arena_rounds "
            "ENABLE TRIGGER lab_arena_rounds_write_once"
        )
        migration = Path(__file__).resolve().parents[2] / (
            "scripts/251-lab-arena-twenty-icp-promotion.sql"
        )
        cursor.execute(migration.read_text(encoding="utf-8"))

    harness.service = harness.build_service()
    cutover = harness.service.store.get_round(round_id)
    assert cutover["confirmation_bank_ref"] == confirmation_ref
    assert cutover["confirmation_bank_hash"] == confirmation_hash
    assert not {
        "stage_3_start", "stage_3_close", "stage_3_scoring_close"
    }.intersection(cutover["configuration_doc"]["schedule"])
    harness.clock.advance_to(harness.schedule()["submission_cutoff"])
    published = harness.advance_until("published")
    assert published["king_outcome"] == "crowned"
    ranking = {
        item["submission_id"]: item
        for item in published["publication_doc"]["final_ranking"]
    }
    assert ranking[winner]["final_score"] > 0
    assert published["confirmation_bank_ref"] == confirmation_ref
    assert published["confirmation_bank_hash"] == confirmation_hash
    assert published["confirmation_cohort"] is None
    assert published["stage3_scoring_plan_doc"] is None
    runs = harness.service.store.list_runs(round_id)
    assert runs
    assert all(run["stage"] in (1, 2) for run in runs)
    assert max(run["icp_position"] for run in runs) == 19

    # Replaying the exact cutover after publication must retain all evidence.
    with connect() as connection, connection.cursor() as cursor:
        def evidence():
            result = []
            for table, key in (
                ("lab_arena_rounds", "round_id"),
                ("lab_arena_runs", "run_id"),
                ("lab_arena_ledger", "entry_id"),
            ):
                cursor.execute(
                    f"SELECT md5(COALESCE(string_agg(row_to_json(r)::text, '' "
                    f"ORDER BY {key}), '')) FROM public.{table} r "
                    "WHERE round_id = %s", (round_id,),
                )
                result.append(cursor.fetchone()[0])
            return result

        before = evidence()
        cursor.execute(migration.read_text(encoding="utf-8"))
        cursor.execute(migration.read_text(encoding="utf-8"))
        assert evidence() == before
        cursor.execute("SELECT public.lab_arena_twenty_icp_promotion_schema_v1()")
        assert cursor.fetchone()[0]["version"] == 251
        for signature in (
            "public.lab_arena_prepare_confirmation_bank(text,text,text)",
            "public.lab_arena_open_confirmation(text,jsonb)",
            "public.lab_arena__confirmation_account_failure(text,text)",
        ):
            cursor.execute("SELECT to_regprocedure(%s)", (signature,))
            assert cursor.fetchone()[0] is None
        cursor.execute(
            "SELECT has_function_privilege('lab_arena_service', "
            "'public.lab_arena_twenty_icp_promotion_schema_v1()', 'EXECUTE'), "
            "has_function_privilege('anon', "
            "'public.lab_arena_twenty_icp_promotion_schema_v1()', 'EXECUTE')"
        )
        assert cursor.fetchone() == (True, False)
