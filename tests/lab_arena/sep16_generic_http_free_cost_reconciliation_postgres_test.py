"""Disposable PostgreSQL proof for migration 271's exact zero settlements."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.lab_arena.lab_arena_pg_harness import (
    CURRENT_SERVICE_MIGRATIONS,
    database_with_lab_arena_migration,
)


ROOT = Path(__file__).resolve().parents[2]
MIGRATION = (
    ROOT / "scripts/271-reconcile-sep16-generic-http-free-costs.sql"
)
ROUND = "arena-2026-09-16"
SUBMISSION = "baseline-2026-09-16"
HOTKEY = "5FNVgRnrxMibhcBGEAaajGrYjsaCn441a5HuGUBUNnxEBLo9"
CREDENTIAL = (
    "sha256:6337fb7ec65645a89f951013ad29797a"
    "c8ef19a0542b666f741e77837e0743ee"
)
BASIS = "deepline_authenticated_catalog_free_per_call_2026-09-16"
PRICING_PROOF = (
    "sha256:6be6c06a52fe7c3133d95325ef183a0"
    "896f1278129ce6a48fcf2ac0a2b4c6f73"
)
TOOL_CONTRACT = (
    "sha256:e1ab8fdbe0b81511c7f2b8e9ab509ff9"
    "52d640a257f0665a894a879d2e5e7d63"
)
CATALOG_RESPONSE = (
    "sha256:7b29488b6399ca8a445311a9ad935017"
    "27871a7f5f3920f0fbf66b3da3c4bd73"
)
TARGETS = (
    {
        "uncertain_id": 445520,
        "reservation_id": 445500,
        "dispatch_id": 445501,
        "run_id": f"{ROUND}:{SUBMISSION}:1:0:rerun269:1",
        "assignment_id": f"{ROUND}:{SUBMISSION}:1:0:rerun269",
        "position": 0,
        "attempt": 1,
        "status": "accepted",
        "terminal_cause": "accepted",
        "identity": (
            "sha256:04e80ee1554b329921dc21e603e96517"
            "e1e1cb44703efe503facefa54d73780e"
        ),
        "request_id": "ctx-tool-04e80ee1554b329921dc21e603e96517",
        "job_id": "iad1::9hjtk-1789588271601-58dfa6a48e84",
        "body_bytes": 168069,
    },
    {
        "uncertain_id": 447528,
        "reservation_id": 447505,
        "dispatch_id": 447506,
        "run_id": f"{ROUND}:{SUBMISSION}:1:5:rerun269:2",
        "assignment_id": f"{ROUND}:{SUBMISSION}:1:5:rerun269",
        "position": 5,
        "attempt": 2,
        "status": "failed",
        "terminal_cause": "provider_error",
        "identity": (
            "sha256:669b75511203146a5ff6316fcc1837d4"
            "74185bd974fc87981279030c2e299efd"
        ),
        "request_id": "ctx-tool-669b75511203146a5ff6316fcc1837d4",
        "job_id": "iad1::5wn7l-1789590248594-ac5d12110a16",
        "body_bytes": 25723,
    },
)


@pytest.fixture(scope="module")
def database():
    yield from database_with_lab_arena_migration(CURRENT_SERVICE_MIGRATIONS)


def _ledger_snapshot(connection):
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT entry_id,entry_kind,miner_hotkey,round_id,submission_id,"
            "run_id,stage,call_identity,provider,operation_id,funding_source,"
            "amount_microusd,entry_doc,terminal_response,created_at "
            "FROM public.lab_arena_ledger "
            "WHERE entry_doc->>'sep16_generic_http_catalog_reconciliation' "
            "IS DISTINCT FROM 'true' ORDER BY entry_id"
        )
        return cursor.fetchall()


def _cost_state(connection):
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT public.lab_arena__successful_call_cost_state(%s,'execute',NULL)",
            (SUBMISSION,),
        )
        return cursor.fetchone()[0]


def _eligibility(connection):
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT public.lab_arena__successful_call_eligibility(%s,%s,2)",
            (ROUND, SUBMISSION),
        )
        return cursor.fetchone()[0]


def _seed(connection):
    with connection.cursor() as cursor:
        cursor.execute("SET session_replication_role=replica")
        cursor.execute(
            "TRUNCATE public.lab_arena_ledger,public.lab_arena_runs,"
            "public.lab_arena_submissions,public.lab_arena_rounds "
            "RESTART IDENTITY CASCADE"
        )
        cursor.execute(
            "INSERT INTO public.lab_arena_rounds("
            "round_id,status,status_generation,stage_generation,cancel_reason,"
            "configuration_doc) VALUES (%s,'cancelled',16,12,'operator',%s::jsonb)",
            (
                ROUND,
                json.dumps(
                    {
                        "sourcing_cost_eligibility_policy": "successful_calls_v1",
                        "execution_cap_microusd": 80_000_000,
                        "cost_per_company_microusd": 10_000_000,
                    }
                ),
            ),
        )
        cursor.execute(
            "INSERT INTO public.lab_arena_submissions("
            "submission_id,round_id,miner_hotkey,status,is_king) "
            "VALUES (%s,%s,%s,'frozen',true)",
            (SUBMISSION, ROUND, HOTKEY),
        )
        for target in TARGETS:
            cursor.execute(
                "INSERT INTO public.lab_arena_runs("
                "run_id,assignment_id,round_id,submission_id,miner_hotkey,"
                "stage,icp_position,attempt,kind,status,terminal_cause) "
                "VALUES (%s,%s,%s,%s,%s,1,%s,%s,'execute',%s,%s)",
                (
                    target["run_id"],
                    target["assignment_id"],
                    ROUND,
                    SUBMISSION,
                    HOTKEY,
                    target["position"],
                    target["attempt"],
                    target["status"],
                    target["terminal_cause"],
                ),
            )
            reservation_doc = {
                "deepline_request_id": target["request_id"],
                "tool": "generic_http_request",
                "credential_fingerprint": CREDENTIAL,
            }
            call_doc = {
                "reason": "missing_provider_cost",
                "call_succeeded": True,
                "provider_status": 200,
                "body_bytes": target["body_bytes"],
                "body_is_mapping": True,
                "usage_present": False,
                "billing_present": False,
                "top_level_job_status": "completed",
                "deepline_request_id": target["request_id"],
                "deepline_job_id": target["job_id"],
                "deepline_operation": "generic_http_request",
                "credential_fingerprint": CREDENTIAL,
            }
            for entry_id, kind, entry_doc in (
                (target["reservation_id"], "reservation", reservation_doc),
                (target["dispatch_id"], "dispatch", {}),
                (
                    target["uncertain_id"],
                    "uncertain",
                    {"reason": "worker_reported", "call": call_doc},
                ),
            ):
                cursor.execute(
                    "INSERT INTO public.lab_arena_ledger("
                    "entry_id,entry_kind,miner_hotkey,round_id,submission_id,"
                    "run_id,stage,call_identity,provider,operation_id,"
                    "funding_source,amount_microusd,entry_doc) VALUES ("
                    "%s,%s,%s,%s,%s,%s,1,%s,'deepline','deepline.execute',"
                    "'host',0,%s::jsonb)",
                    (
                        entry_id,
                        kind,
                        HOTKEY,
                        ROUND,
                        SUBMISSION,
                        target["run_id"],
                        target["identity"],
                        json.dumps(entry_doc),
                    ),
                )

        # A failed generic call remains an uncertainty but is not a successful
        # unresolved charge. Migration 271 must leave every row in this chain.
        failed_identity = "sha256:" + "f" * 64
        for entry_id, kind, entry_doc in (
            (
                447600,
                "reservation",
                {
                    "deepline_request_id": "ctx-tool-" + "f" * 32,
                    "tool": "generic_http_request",
                    "credential_fingerprint": CREDENTIAL,
                },
            ),
            (447601, "dispatch", {}),
            (
                447602,
                "uncertain",
                {
                    "reason": "worker_reported",
                    "call": {
                        "reason": "missing_provider_cost",
                        "call_succeeded": False,
                        "provider_status": 503,
                        "body_is_mapping": True,
                        "billing_present": False,
                        "usage_present": False,
                        "body_bytes": 20,
                        "deepline_request_id": "ctx-tool-" + "f" * 32,
                        "deepline_operation": "generic_http_request",
                        "credential_fingerprint": CREDENTIAL,
                    },
                },
            ),
        ):
            cursor.execute(
                "INSERT INTO public.lab_arena_ledger("
                "entry_id,entry_kind,miner_hotkey,round_id,submission_id,"
                "run_id,stage,call_identity,provider,operation_id,"
                "funding_source,amount_microusd,entry_doc) VALUES ("
                "%s,%s,%s,%s,%s,%s,1,%s,'deepline','deepline.execute',"
                "'host',0,%s::jsonb)",
                (
                    entry_id,
                    kind,
                    HOTKEY,
                    ROUND,
                    SUBMISSION,
                    TARGETS[0]["run_id"],
                    failed_identity,
                    json.dumps(entry_doc),
                ),
            )
        cursor.execute("SET session_replication_role=origin")
        cursor.execute(
            "SELECT setval(pg_get_serial_sequence("
            "'public.lab_arena_ledger','entry_id'),"
            "(SELECT max(entry_id) FROM public.lab_arena_ledger))"
        )
    connection.commit()


@pytest.fixture
def connection(database):
    psycopg2, dsn = database
    result = psycopg2.connect(**dsn)
    _seed(result)
    try:
        yield result
    finally:
        result.close()


def test_exact_reconciliation_is_append_only_idempotent_and_clears_success_cost(
    connection,
):
    before = _ledger_snapshot(connection)
    state = _cost_state(connection)
    assert state["successful_calls"] == 0
    assert state["success_unresolved_calls"] == 2
    assert state["uncertain_calls"] == 3
    assert _eligibility(connection)["eligibility_reason"] == (
        "provider_cost_uncertain"
    )

    for _ in range(2):
        with connection.cursor() as cursor:
            cursor.execute(MIGRATION.read_text(encoding="utf-8"))

    assert _ledger_snapshot(connection) == before
    state = _cost_state(connection)
    assert state["successful_calls"] == 2
    assert state["successful_microusd"] == 0
    assert state["success_unresolved_calls"] == 0
    assert state["uncertain_calls"] == 1
    assert _eligibility(connection)["eligibility_reason"] == "eligible"

    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT entry_doc,terminal_response FROM public.lab_arena_ledger "
            "WHERE entry_doc->>'sep16_generic_http_catalog_reconciliation'="
            "'true' ORDER BY (entry_doc->>'reconciled_uncertainty_entry_id')::bigint"
        )
        settlements = cursor.fetchall()
        assert len(settlements) == 2
        for target, (entry_doc, terminal) in zip(TARGETS, settlements):
            assert entry_doc["reconciled_uncertainty_entry_id"] == target[
                "uncertain_id"
            ]
            assert entry_doc["deepline_request_id"] == target["request_id"]
            assert entry_doc["deepline_job_id"] == target["job_id"]
            assert entry_doc["provider_response_payload_retained"] is False
            assert entry_doc["catalog_evidence"] == {
                "authenticated_endpoint": "/api/v2/tools",
                "checked_at": "2026-09-16T20:33:19Z",
                "tool_id": "generic_http_request",
                "provider": "generic_http",
                "operation": "generic_http_request",
                "operation_id": "generic_http_request",
                "operation_aliases": ["generic_http_request", "request"],
                "credits_per_unit": 0,
                "usd_per_unit": 0,
                "currency": "USD",
                "unit": "call",
                "display_text": "Free",
                "pricing_proof_sha256": PRICING_PROOF,
                "tool_contract_sha256": TOOL_CONTRACT,
                "response_sha256": CATALOG_RESPONSE,
            }
            assert terminal["status"] == 200
            assert terminal["call_succeeded"] is True
            assert terminal["provider_cost"] == {
                "basis": BASIS,
                "units": "0",
                "unit_name": "credits",
                "operation": "generic_http_request",
                "request_id": target["job_id"],
            }
        cursor.execute(
            "SELECT entry_kind,entry_doc#>>'{call,call_succeeded}' "
            "FROM public.lab_arena_ledger WHERE entry_id=447602"
        )
        assert cursor.fetchone() == ("uncertain", "false")


@pytest.mark.parametrize(
    ("mutation", "error"),
    (
        (
            "UPDATE public.lab_arena_rounds SET cancel_reason=NULL "
            f"WHERE round_id='{ROUND}'",
            "round_or_submission_mismatch",
        ),
        (
            "UPDATE public.lab_arena_ledger SET entry_doc=entry_doc #- "
            "'{call,billing_present}' WHERE entry_id=445520",
            "head_mismatch:445520",
        ),
        (
            "UPDATE public.lab_arena_ledger SET entry_doc="
            "jsonb_set(entry_doc,'{call,call_succeeded}','null'::jsonb) "
            "WHERE entry_id=445520",
            "unresolved_set_mismatch",
        ),
        (
            "UPDATE public.lab_arena_ledger SET entry_doc="
            "entry_doc-'deepline_request_id' WHERE entry_id=445500",
            "chain_mismatch:445520",
        ),
        (
            "UPDATE public.lab_arena_runs SET assignment_id="
            "assignment_id||':changed' WHERE run_id='"
            + TARGETS[1]["run_id"]
            + "'",
            "run_mismatch:447528",
        ),
        (
            "INSERT INTO public.lab_arena_ledger("
            "entry_id,entry_kind,miner_hotkey,round_id,submission_id,run_id,"
            "stage,call_identity,provider,operation_id,funding_source,"
            "amount_microusd,entry_doc) VALUES (447702,'uncertain','"
            + HOTKEY
            + "','"
            + ROUND
            + "','"
            + SUBMISSION
            + "','"
            + TARGETS[0]["run_id"]
            + "',1,'sha256:"
            + "e" * 64
            + "','deepline','deepline.execute','host',0,"
            "'{\"reason\":\"worker_reported\",\"call\":{"
            "\"reason\":\"missing_provider_cost\","
            "\"call_succeeded\":true,"
            "\"deepline_operation\":\"generic_http_request\"}}')",
            "unresolved_set_mismatch",
        ),
    ),
)
def test_reconciliation_aborts_atomically_on_any_identity_or_set_mismatch(
    connection, mutation, error
):
    with connection.cursor() as cursor:
        cursor.execute("SET session_replication_role=replica")
        cursor.execute(mutation)
        cursor.execute("SET session_replication_role=origin")
    connection.commit()
    before = _ledger_snapshot(connection)

    with pytest.raises(Exception, match=error):
        with connection.cursor() as cursor:
            cursor.execute(MIGRATION.read_text(encoding="utf-8"))
    connection.rollback()

    assert _ledger_snapshot(connection) == before
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT count(*) FROM public.lab_arena_ledger WHERE "
            "entry_doc->>'sep16_generic_http_catalog_reconciliation'='true'"
        )
        assert cursor.fetchone() == (0,)
