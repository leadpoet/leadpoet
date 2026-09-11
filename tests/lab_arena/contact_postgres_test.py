"""PostgreSQL guards for opt-in Arena contact qualification."""

from __future__ import annotations

import json
from pathlib import Path

import psycopg2
import pytest

from lab_arena import contracts, scoring
from lab_arena.owner_admission import OwnerAdmission
from lab_arena.store import ArenaStore, PsycopgTransport
from tests.lab_arena.lab_arena_pg_harness import database_with_lab_arena_migration
from tests.lab_arena.test_integrity_round import MIGRATIONS
from tests.lab_arena.test_lab_arena_migration_postgres import (
    commit_round,
    encrypted_runtime_credentials,
    hotkey,
    pass_code_review,
    round_config,
    source_submission_doc,
)

MIGRATION = "215-lab-arena-contacts.sql"


@pytest.fixture(scope="module")
def database():
    yield from database_with_lab_arena_migration(MIGRATIONS + (MIGRATION,))


@pytest.fixture()
def store(database):
    psycopg2, dsn = database
    transport = PsycopgTransport(lambda: psycopg2.connect(**dsn))
    yield ArenaStore(transport)
    transport.close()


def _configuration(round_id: str, *, contacts: bool) -> dict:
    config = round_config(
        round_id,
        [hotkey(f"{round_id}-runner")],
        cost_per_company_microusd=100_000,
    )
    config["integrity_policy"] = "arena_integrity_v1"
    config["scorer_policy"] = scoring.build_scorer_policy(
        scoring_adapter_version=(
            "qualification_contacts_v3"
            if contacts
            else "qualification_integrity_v2"
        )
    )
    if contacts:
        config["contact_policy"] = "contacts_v1"
    return config


def _open_run(store: ArenaStore, round_id: str, *, contacts: bool) -> str:
    assert store.create_round(round_id, _configuration(round_id, contacts=contacts))[
        "status"
    ] == "created"
    digest = "sha256:" + "a" * 64
    assert store.prepare_confirmation_bank(
        round_id,
        f"arena/{round_id}/confirmation/{digest.removeprefix('sha256:')}.json",
        digest,
    )["status"] == "ok"
    prefix = round_id.replace("arena-", "")
    miner = hotkey(f"{prefix}-miner")
    submission_id = f"{prefix}-sub"
    registered = store.register_submission(
        round_id,
        submission_id,
        miner,
        source_submission_doc(round_id, submission_id),
        owner_admission=OwnerAdmission(
            coldkey=hotkey(f"{prefix}-owner"),
            block_number=123,
            block_hash="0x" + "b" * 64,
        ),
    )
    assert registered["status"] == "registered"
    assert store.accept_submission_with_credentials(
        round_id,
        submission_id,
        miner,
        encrypted_runtime_credentials(submission_id),
    )["status"] == "ok"
    pass_code_review(store, submission_id, miner)
    assert store.update_submission(
        round_id, submission_id, "accepted", "frozen", {"is_king": False}
    )["status"] == "ok"
    participants = [
        {
            "submission_id": submission_id,
            "miner_hotkey": miner,
            "is_king": False,
        }
    ]
    commit_round(store, round_id, participants)
    assert store.open_stage(
        round_id, 1, participants, list(contracts.stage_positions(1))
    )["status"] == "ok"
    return str(store.list_runs(round_id, stage=1, kind="execute")[0]["run_id"])


def _qualification_doc(*, company: bool, contact: bool | None) -> dict:
    row = {
        "company_index": 0,
        "company_identity_key": "domain:example.com|name:example",
        "company_qualified": company,
        "duplicate_company": False,
    }
    if contact is not None:
        row["contact_qualified"] = contact
    return {"companies": [row]}


def _save_score(database, run_id: str, score: float, document: dict) -> None:
    psycopg2, dsn = database
    with psycopg2.connect(**dsn) as connection, connection.cursor() as cursor:
        cursor.execute(
            "UPDATE public.lab_arena_runs "
            "SET per_icp_score=%s, qualification_doc=%s::jsonb WHERE run_id=%s",
            (score, json.dumps(document), run_id),
        )


def test_contact_migration_replays_and_keeps_private_acl(database) -> None:
    psycopg2, dsn = database
    migration = Path(__file__).resolve().parents[2] / "scripts" / MIGRATION
    with psycopg2.connect(**dsn) as connection, connection.cursor() as cursor:
        cursor.execute(migration.read_text(encoding="utf-8"))
        cursor.execute(migration.read_text(encoding="utf-8"))
        cursor.execute("SELECT public.lab_arena_contact_schema_v1()")
        assert cursor.fetchone()[0] == {
            "schema_version": "leadpoet.lab_arena.contact_schema.v1",
            "version": 215,
        }
        function = "public.lab_arena_contact_schema_v1()"
        cursor.execute(
            "SELECT has_function_privilege('lab_arena_service', %s, 'EXECUTE'), "
            "has_function_privilege('anon', %s, 'EXECUTE'), "
            "has_function_privilege('authenticated', %s, 'EXECUTE')",
            (function, function, function),
        )
        assert cursor.fetchone() == (True, False, False)
        cursor.execute(
            "SELECT has_function_privilege('lab_arena_service', "
            "'public.lab_arena__qualification_doc_valid(jsonb)', 'EXECUTE')"
        )
        assert cursor.fetchone()[0] is False


def test_old_and_contact_rounds_require_their_exact_receipt_shape(
    database, store
) -> None:
    old_run = _open_run(store, "arena-2099-01-01-cpo", contacts=False)
    _save_score(database, old_run, 10.0, _qualification_doc(company=True, contact=None))

    old_contact_run = _open_run(store, "arena-2099-01-02-cpm", contacts=False)
    with pytest.raises(
        psycopg2.Error, match="lab_arena_contact_receipt_policy_mismatch"
    ):
        _save_score(
            database,
            old_contact_run,
            10.0,
            _qualification_doc(company=True, contact=True),
        )

    contact_missing_run = _open_run(
        store, "arena-2099-01-03-cnm", contacts=True
    )
    with pytest.raises(
        psycopg2.Error, match="lab_arena_contact_receipt_policy_mismatch"
    ):
        _save_score(
            database,
            contact_missing_run,
            10.0,
            _qualification_doc(company=True, contact=None),
        )


def test_contact_false_cannot_receive_credit(database, store) -> None:
    denied_run = _open_run(store, "arena-2099-01-04-ccd", contacts=True)
    with pytest.raises(psycopg2.Error, match="lab_arena_contact_credit_invalid"):
        _save_score(
            database,
            denied_run,
            10.0,
            _qualification_doc(company=False, contact=False),
        )

    zero_run = _open_run(store, "arena-2099-01-05-ccz", contacts=True)
    _save_score(
        database,
        zero_run,
        0.0,
        _qualification_doc(company=False, contact=False),
    )
    positive_run = _open_run(store, "arena-2099-01-06-ccp", contacts=True)
    _save_score(
        database,
        positive_run,
        10.0,
        _qualification_doc(company=True, contact=True),
    )

    contradictory_run = _open_run(
        store, "arena-2099-01-07-cct", contacts=True
    )
    contradictory = {
        "companies": [
            _qualification_doc(company=True, contact=True)["companies"][0],
            {
                "company_index": 1,
                "company_identity_key": "domain:other.example|name:other",
                "company_qualified": False,
                "duplicate_company": False,
                "contact_qualified": True,
            },
        ]
    }
    with pytest.raises(
        psycopg2.Error, match="lab_arena_qualification_receipt_required"
    ):
        _save_score(database, contradictory_run, 10.0, contradictory)
