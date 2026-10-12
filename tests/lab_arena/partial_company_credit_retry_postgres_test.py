"""Replenished credit resumes validated partial scoring under existing gates."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from lab_arena import contracts
from lab_arena.service import ServiceError
from lab_arena.store import ArenaStore, PsycopgTransport
from tests.lab_arena.lab_arena_pg_harness import POSTGREST_MIGRATIONS, database_with_lab_arena_migration
from tests.lab_arena.partial_company_judgments_postgres_test import _setup, _request
from tests.lab_arena.test_lab_arena_migration_postgres import claim, hotkey
from tests.lab_arena.test_lab_arena_service_round import keypair

SCRIPTS = Path(__file__).parents[2] / "scripts"


@pytest.fixture()
def database():
    yield from database_with_lab_arena_migration(POSTGREST_MIGRATIONS + ("403-lab-arena-credit-failure-retry.sql", "465-lab-arena-partial-company-judgments.sql"))


@pytest.fixture()
def store(database):
    psycopg, dsn = database
    return ArenaStore(PsycopgTransport(lambda: psycopg.connect(**dsn)))


def _prepare(store, database, *, credit_status=402, credit_charge=0, credit_proof=True):
    from tests.lab_arena.company_judgments_postgres_test import round_config
    def credit_config(*args, **kwargs):
        config = round_config(*args, **kwargs)
        config["schedule"]["stage_1_scoring_close"] = "2030-01-01T23:00:00Z"
        return config
    with patch("tests.lab_arena.company_judgments_postgres_test.round_config", credit_config):
        service, round_id, runner_key, leased, token, refs = _setup(store, "credential_error")
    assert service.handle_complete(_request(round_id, runner_key, leased, token, terminal="credential_error"))["status"] == "failed"
    source = store.get_run(leased["run_id"])
    assert source["output_ref"]
    psycopg, dsn = database
    with psycopg.connect(**dsn) as connection:
        with connection.cursor() as cursor:
            for label, amount, response in (
                ("paid-sibling", 100, {"status": 200, "call_succeeded": True}),
                ("credit-stop", credit_charge, {"status": credit_status, "call_succeeded": False}),
            ):
                if label == "credit-stop" and credit_proof:
                    response["credit_failure_proof"] = {
                        "schema_version": "leadpoet.lab_arena.credit_failure_proof.v1",
                        "provider": "openrouter", "reason": "out_of_credit",
                        "provider_status": credit_status, "actual_microusd": credit_charge,
                    }
                cursor.execute("INSERT INTO public.lab_arena_ledger (entry_kind, miner_hotkey, round_id, submission_id, run_id, stage, call_identity, provider, operation_id, funding_source, amount_microusd, terminal_response) VALUES ('settlement', %s, %s, %s, %s, 1, %s, 'openrouter', 'openrouter.chat', 'miner_key', %s, %s::jsonb)", (leased["miner_hotkey"], round_id, leased["submission_id"], leased["run_id"], contracts.document_hash(label), amount, json.dumps(response)))
    owner = keypair(round_id[-2:] + "-miner-0")
    assert owner.ss58_address == leased["miner_hotkey"]
    service._invalidate_hot_round()
    return service, round_id, owner, leased, refs, source


def _retry_request(round_id, owner, submission_id):
    return contracts.build_signed_request(scope=contracts.SCOPE_SUBMISSION_CREDIT_RETRY,
        round_id=round_id, hotkey=owner.ss58_address,
        body={"submission_id": submission_id}, timestamp=int(datetime.now(timezone.utc).timestamp()),
        sign_message=lambda message: owner.sign(message.encode()).hex())


def test_owner_credit_retry_fails_old_gate_and_reuses_completed_companies_after_upgrade(store, database):
    service, round_id, owner, leased, refs, source = _prepare(store, database)
    envelope = _retry_request(round_id, owner, leased["submission_id"])
    psycopg, dsn = database
    with psycopg.connect(**dsn) as connection:
        with connection.cursor() as cursor:
            # Reinstall the exact previous RPC to prove the specific old fault.
            cursor.execute((SCRIPTS / "403-lab-arena-credit-failure-retry.sql").read_text())
    assert service.handle_submission_credit_retry(leased["submission_id"], envelope)["status"] == "no_eligible"
    with psycopg.connect(**dsn) as connection:
        with connection.cursor() as cursor:
            cursor.execute((SCRIPTS / "465-lab-arena-partial-company-judgments.sql").read_text())
            cursor.execute((SCRIPTS / "465-lab-arena-partial-company-judgments.sql").read_text())
    before_ledger = store.list_ledger(run_id=leased["run_id"])
    before_cache = [store.get_company_judgment(ref["cache_key"], 0) for ref in refs[:2]]
    receipt = service.handle_submission_credit_retry(leased["submission_id"], envelope)
    assert receipt == {"status": "queued", "requeued_count": 1}
    assert service.handle_submission_credit_retry(leased["submission_id"], envelope) == {"status": "replayed", "requeued_count": 1}
    retry_runner = hotkey("partial-credit-retry-validator")
    retry, _, _, _ = claim(store, round_id, retry_runner, excluded=[retry_runner])
    assert retry["status"] == "leased" and retry["attempt"] == 2
    assert sorted(row["company_index"] for row in retry["company_judgment_cache"]["hits"]) == [0, 1]
    assert [row["company_index"] for row in retry["company_judgment_cache"]["misses"]] == [2]
    assert store.get_run(leased["run_id"])["result_doc"] == source["result_doc"]
    assert store.list_ledger(run_id=leased["run_id"]) == before_ledger
    assert [store.get_company_judgment(ref["cache_key"], 0) for ref in refs[:2]] == before_cache
    assert sum(row["amount_microusd"] for row in before_ledger) == 100
    assert service.handle_submission_credit_retry(leased["submission_id"], _retry_request(round_id, owner, leased["submission_id"]))["status"] == "no_eligible"


@pytest.mark.parametrize("change", ["wrong_owner", "unproved_credit", "charged_credit", "completion_stamp", "claim_binding", "scope_binding", "deadline", "source_hash"])
def test_partial_credit_retry_retains_authority_cost_and_deadline_gates(store, database, change):
    service, round_id, owner, leased, refs, _ = _prepare(store, database,
        credit_status=401 if change == "unproved_credit" else 402,
        credit_charge=7 if change == "charged_credit" else 0,
        credit_proof=change != "unproved_credit")
    if change == "wrong_owner":
        owner = keypair("wrong-partial-credit-owner")
    elif change == "source_hash":
        source = store.get_run(leased["run_id"])
        service._objects._client.objects[source["output_ref"]] = b'{"tampered":true}'
    elif change in ("completion_stamp", "claim_binding", "scope_binding", "deadline"):
        psycopg, dsn = database
        with psycopg.connect(**dsn) as connection:
            with connection.cursor() as cursor:
                # These injected negatives represent a missing/unbound stored
                # authority. Production terminal/cache rows stay immutable.
                cursor.execute("SET session_replication_role=replica")
                if change == "completion_stamp":
                    cursor.execute("UPDATE public.lab_arena_runs SET result_doc = result_doc - 'company_judgment_completion_request_hash' WHERE run_id=%s", (leased["run_id"],))
                elif change == "claim_binding":
                    cursor.execute("UPDATE public.lab_arena_company_judgments SET evidence_doc = jsonb_set(evidence_doc, '{source_claim_request_hash}', %s::jsonb) WHERE source_score_run_id=%s", (json.dumps("sha256:" + "b" * 64), leased["run_id"]))
                elif change == "scope_binding":
                    cursor.execute("UPDATE public.lab_arena_runs SET company_judgment_refs = '[]'::jsonb WHERE run_id=%s", (leased["run_id"],))
                else:
                    cursor.execute("UPDATE public.lab_arena_rounds SET configuration_doc = jsonb_set(configuration_doc, '{schedule,stage_1_scoring_close}', '\"2020-01-01T23:00:00Z\"'::jsonb) WHERE round_id=%s", (round_id,))
                cursor.execute("SET session_replication_role=origin")
        service._invalidate_hot_round()
    envelope = _retry_request(round_id, owner, leased["submission_id"])
    if change == "wrong_owner":
        with pytest.raises(ServiceError, match="credit_retry_owner_required"):
            service.handle_submission_credit_retry(leased["submission_id"], envelope)
    elif change in ("claim_binding", "source_hash"):
        with pytest.raises(ServiceError, match="credit_retry_source_unavailable"):
            service.handle_submission_credit_retry(leased["submission_id"], envelope)
    else:
        assert service.handle_submission_credit_retry(leased["submission_id"], envelope)["status"] == "no_eligible"
    assert not any(run["attempt"] == 2 for run in store.list_runs(round_id, kind="score"))
