"""Real PostgreSQL proof for the one-use Sep20 reward correction."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from lab_arena import rewards, signing, weight_state
from leadpoet_canonical import arena_weights
from tests.lab_arena.lab_arena_pg_harness import (
    CURRENT_SERVICE_MIGRATIONS,
    database_with_lab_arena_migration,
)


ROOT = Path(__file__).parents[2]
INSTALL_TEMPLATE = ROOT / "scripts/340-arena-2026-09-20-invalid-reward-correction.sql.template"
CLEANUP_TEMPLATE = ROOT / "scripts/341-arena-2026-09-20-invalid-reward-correction-cleanup.sql.template"
ROUND = "arena-2026-09-20"
ARCHIVE = "arena-2026-09-20-rewardhistory340"
OLD_EPOCH = 25_288
LAST_EPOCH = 25_293
NEW_EPOCH = LAST_EPOCH + 1
CHAMPION = "5FqcBT8JJ2Sr4KHWkpJG4nza9QtwGMSWXucQXeotVUEeM7VX"
BURN = "5FNVgRnrxMibhcBGEAaajGrYjsaCn441a5HuGUBUNnxEBLo9"
PUBLISHED_AT = "2026-09-20T16:00:00Z"
CONSTANTS = {
    "pool_percent": 25,
    "pool_basis": "total_emissions",
    "king_pool_share_percent_by_week": [100, 80, 60, 40, 20],
    "epochs_per_reward_week": 140,
    "eligibility_max_epochs": 45,
}


@pytest.fixture()
def database():
    yield from database_with_lab_arena_migration(CURRENT_SERVICE_MIGRATIONS)


def _pg_sha(cursor, expression: str, parameters=()) -> str:
    cursor.execute(
        "SELECT encode(extensions.digest((" + expression + ")::text,'sha256'),'hex')",
        parameters,
    )
    return cursor.fetchone()[0]


def _signed_basis(signer, *, outcome: str, epoch: int, hotkey: str) -> dict:
    document = rewards.reward_basis_document(
        round_id=ROUND,
        published_at=PUBLISHED_AT,
        finalized_epoch=epoch - 1,
        king_outcome=outcome,
        king_hotkey=hotkey,
        reward_constants=CONSTANTS,
    )
    return signing.sign_document(signer, document, hash_field="reward_basis_hash")


def _seed(cursor):
    signer = signing.LocalSigner.generate()
    key_doc = signing.signing_key_document(signer.public_key_der)
    old_basis = _signed_basis(
        signer, outcome="crowned", epoch=OLD_EPOCH, hotkey=CHAMPION
    )
    replacement = _signed_basis(
        signer, outcome="no_king", epoch=NEW_EPOCH, hotkey=""
    )
    configuration = {
        "schema_version": "leadpoet.lab_arena.round_configuration.v1",
        "round_id": ROUND,
        "mode": "live",
        "rewards_enabled": True,
        "network_name": "finney",
        "netuid": 71,
        "baseline_hotkey": BURN,
        "reward_constants": CONSTANTS,
    }
    publication = {
        "schema_version": "leadpoet.lab_arena.publication.v1",
        "round_id": ROUND,
        "published_at": PUBLISHED_AT,
        "king_decision": {
            "outcome": "no_king",
            "king_hotkey": "",
            "king_submission_id": None,
            "winner_submission_id": None,
        },
    }
    cursor.execute(
        "INSERT INTO public.lab_arena_rounds("
        "round_id,status,configuration_doc,rewards_enabled,publication_doc,"
        "king_outcome,king_hotkey,king_start_epoch,effective_reward_epoch,"
        "reward_basis_hash,reward_basis_doc,signing_key_doc,reward_activated_at,"
        "published_at,promotion_required) VALUES("
        "%s,'published',%s::jsonb,true,%s::jsonb,'no_king',NULL,%s,%s,%s,"
        "%s::jsonb,%s::jsonb,clock_timestamp(),%s::timestamptz,false)",
        (
            ROUND,
            json.dumps(configuration),
            json.dumps(publication),
            old_basis["king_start_epoch"],
            OLD_EPOCH,
            old_basis["reward_basis_hash"],
            json.dumps(old_basis),
            json.dumps(key_doc),
            PUBLISHED_AT,
        ),
    )
    cursor.execute(
        "INSERT INTO public.lab_arena_submissions("
        "submission_id,round_id,miner_hotkey,status,is_king,submission_doc) "
        "VALUES('sep20-correction-sentinel',%s,%s,'frozen',false,'{}'::jsonb)",
        (ROUND, CHAMPION),
    )
    cursor.execute(
        "INSERT INTO public.lab_arena_runs("
        "run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,"
        "icp_position,attempt,kind,status) VALUES("
        "'sep20-correction-run','sep20-correction-assignment',%s,"
        "'sep20-correction-sentinel',%s,1,0,1,'execute','pending')",
        (ROUND, CHAMPION),
    )
    cursor.execute(
        "INSERT INTO public.lab_arena_ledger("
        "entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,"
        "call_identity,provider,operation_id,funding_source,amount_microusd,"
        "entry_doc) VALUES('refusal',%s,%s,'sep20-correction-sentinel',"
        "'sep20-correction-run',1,'sha256:'||repeat('3',64),'openrouter',"
        "'sep20-correction-operation','host',0,'{}'::jsonb)",
        (CHAMPION, ROUND),
    )
    accepted = weight_state.build_accepted_weight_state(
        signer,
        network="finney",
        genesis_hash="1" * 64,
        netuid=71,
        epoch=LAST_EPOCH,
        valid_from_block=1,
        valid_until_block=360,
        reward_basis=old_basis,
        burn_hotkey=BURN,
        issued_at=PUBLISHED_AT,
    )
    cursor.execute(
        "INSERT INTO public.lab_arena_accepted_weight_states("
        "network,netuid,epoch,state_hash,state_doc) VALUES("
        "'finney',71,%s,%s,%s::jsonb)",
        (LAST_EPOCH, accepted["state_hash"], json.dumps(accepted)),
    )
    return signer, key_doc, old_basis, replacement, accepted


def _render_install(cursor, key_doc, old_basis, replacement) -> str:
    values = {
        "__REPLACEMENT_EFFECTIVE_REWARD_EPOCH__": str(NEW_EPOCH),
        "__LAST_IMMUTABLE_EPOCH__": str(LAST_EPOCH),
        "__REPLACEMENT_REWARD_BASIS_DOC_SHA256__": _pg_sha(
            cursor, "%s::jsonb", (json.dumps(replacement),)
        ),
        "__REPLACEMENT_REWARD_BASIS_HASH__": replacement["reward_basis_hash"],
        "__PINNED_SIGNING_PUBLIC_KEY_HASH__": key_doc["public_key_hash"],
        "__ACTIVE_ROUND_SHA256__": _pg_sha(
            cursor,
            "SELECT to_jsonb(active) FROM public.lab_arena_rounds active "
            "WHERE round_id=%s",
            (ROUND,),
        ),
        "__PUBLICATION_DOC_SHA256__": _pg_sha(
            cursor,
            "SELECT publication_doc FROM public.lab_arena_rounds WHERE round_id=%s",
            (ROUND,),
        ),
        "__SIGNING_KEY_DOC_SHA256__": _pg_sha(
            cursor,
            "SELECT signing_key_doc FROM public.lab_arena_rounds WHERE round_id=%s",
            (ROUND,),
        ),
        "__REVOKED_REWARD_BASIS_HASH__": old_basis["reward_basis_hash"],
        "__REVOKED_EFFECTIVE_REWARD_EPOCH__": str(OLD_EPOCH),
    }
    body = INSTALL_TEMPLATE.read_text(encoding="utf-8")
    for marker, value in values.items():
        body = body.replace(marker, value)
    assert not re.search(r"__[A-Z0-9_]+__", body)
    return body


def _render_cleanup(cursor, old_basis, replacement) -> str:
    values = {
        "__REPLACEMENT_EFFECTIVE_REWARD_EPOCH__": str(NEW_EPOCH),
        "__REPLACEMENT_REWARD_BASIS_DOC_SHA256__": _pg_sha(
            cursor, "%s::jsonb", (json.dumps(replacement),)
        ),
        "__REPLACEMENT_REWARD_BASIS_HASH__": replacement["reward_basis_hash"],
        "__REVOKED_REWARD_BASIS_HASH__": old_basis["reward_basis_hash"],
        "__REVOKED_EFFECTIVE_REWARD_EPOCH__": str(OLD_EPOCH),
    }
    body = CLEANUP_TEMPLATE.read_text(encoding="utf-8")
    for marker, value in values.items():
        body = body.replace(marker, value)
    assert not re.search(r"__[A-Z0-9_]+__", body)
    return body


def _preserved_snapshot(cursor):
    result = {}
    for table, order in (
        ("lab_arena_accepted_weight_states", "network,netuid,epoch"),
        ("lab_arena_submissions", "submission_id"),
        ("lab_arena_runs", "run_id"),
        ("lab_arena_ledger", "entry_id"),
    ):
        result[table] = _pg_sha(
            cursor,
            "SELECT coalesce(jsonb_agg(to_jsonb(item) ORDER BY "
            + order
            + "),'[]'::jsonb) FROM public."
            + table
            + " item",
        )
    return result


def _call(cursor, payload):
    cursor.execute(
        "SELECT public.lab_arena_sep20_revoke_invalid_reward340(%s::jsonb)",
        (json.dumps(payload),),
    )
    return cursor.fetchone()[0]


def test_correction_is_prospective_atomic_private_and_cleanup_is_final(database):
    psycopg2, dsn = database
    with psycopg2.connect(**dsn) as connection, connection.cursor() as cursor:
        signer, key_doc, old_basis, replacement, accepted = _seed(cursor)
        install = _render_install(cursor, key_doc, old_basis, replacement)
        cursor.execute(install)
        cursor.execute("BEGIN")

        cursor.execute("SAVEPOINT unauthorized")
        cursor.execute("SET LOCAL ROLE lab_arena_service")
        with pytest.raises(psycopg2.Error, match="permission denied"):
            _call(cursor, replacement)
        cursor.execute("ROLLBACK TO SAVEPOINT unauthorized")

        cursor.execute("SAVEPOINT future_state")
        future = weight_state.build_accepted_weight_state(
            signer,
            network="finney",
            genesis_hash="1" * 64,
            netuid=71,
            epoch=NEW_EPOCH,
            valid_from_block=361,
            valid_until_block=720,
            reward_basis=old_basis,
            burn_hotkey=BURN,
            issued_at=PUBLISHED_AT,
        )
        cursor.execute(
            "INSERT INTO public.lab_arena_accepted_weight_states("
            "network,netuid,epoch,state_hash,state_doc) VALUES("
            "'finney',71,%s,%s,%s::jsonb)",
            (NEW_EPOCH, future["state_hash"], json.dumps(future)),
        )
        cursor.execute("SET LOCAL ROLE service_role")
        with pytest.raises(psycopg2.Error, match="preimage differs"):
            _call(cursor, replacement)
        cursor.execute("ROLLBACK TO SAVEPOINT future_state")

        before = _preserved_snapshot(cursor)
        cursor.execute("SET LOCAL ROLE service_role")
        result = _call(cursor, replacement)
        cursor.execute("RESET ROLE")
        assert result == {
            "status": "corrected",
            "archived_reward_basis_hash": old_basis["reward_basis_hash"],
            "active_reward_basis_hash": replacement["reward_basis_hash"],
        }
        assert _preserved_snapshot(cursor) == before

        cursor.execute(
            "SELECT status,publication_doc,promotion_required,rewards_enabled,"
            "configuration_doc->>'mode',reward_basis_doc,signing_key_doc,"
            "effective_reward_epoch,reward_activated_at IS NOT NULL "
            "FROM public.lab_arena_rounds WHERE round_id=%s",
            (ARCHIVE,),
        )
        archive = cursor.fetchone()
        assert archive[:5] == ("cancelled", None, False, True, "live")
        assert archive[5] == old_basis
        assert archive[6] == key_doc
        assert archive[7:] == (OLD_EPOCH, True)

        cursor.execute(
            "SELECT status,publication_doc#>>'{king_decision,outcome}',"
            "king_outcome,king_hotkey,reward_basis_doc,effective_reward_epoch "
            "FROM public.lab_arena_rounds WHERE round_id=%s",
            (ROUND,),
        )
        active = cursor.fetchone()
        assert active[:4] == ("published", "no_king", "no_king", None)
        assert active[4:] == (replacement, NEW_EPOCH)
        cursor.execute(
            "SELECT reward_basis_doc FROM public.lab_arena_rounds "
            "WHERE configuration_doc->>'mode'='live' AND rewards_enabled "
            "AND reward_activated_at IS NOT NULL "
            "AND arena_network_name='finney' AND arena_netuid=71 "
            "ORDER BY effective_reward_epoch"
        )
        assert [row[0] for row in cursor.fetchall()] == [old_basis, replacement]

        assert rewards.governing_reward_basis([old_basis, replacement], LAST_EPOCH) == old_basis
        assert rewards.governing_reward_basis([old_basis, replacement], NEW_EPOCH) == replacement
        assert arena_weights.derive_arena_weights(
            accepted, [CHAMPION, BURN]
        )["champion_share_ppb"] > 0
        replacement_state = weight_state.build_accepted_weight_state(
            signer,
            network="finney",
            genesis_hash="1" * 64,
            netuid=71,
            epoch=NEW_EPOCH,
            valid_from_block=361,
            valid_until_block=720,
            reward_basis=replacement,
            burn_hotkey=BURN,
            issued_at=PUBLISHED_AT,
        )
        replacement_vector = arena_weights.derive_arena_weights(
            replacement_state, [CHAMPION, BURN]
        )
        assert replacement_vector["champion_share_ppb"] == 0
        assert replacement_vector["burned_residual_ppb"] == 1_000_000_000

        cursor.execute("SET LOCAL ROLE service_role")
        assert _call(cursor, replacement)["status"] == "existing"
        cursor.execute("RESET ROLE")
        cursor.execute(
            "SELECT count(*) FROM pg_catalog.pg_trigger "
            "WHERE tgrelid='public.lab_arena_rounds'::regclass "
            "AND NOT tgisinternal AND tgenabled<>'O'"
        )
        assert cursor.fetchone()[0] == 0

        cleanup = _render_cleanup(cursor, old_basis, replacement)
        cursor.execute(cleanup)
        cursor.execute(cleanup)
        cursor.execute(
            "SELECT to_regprocedure("
            "'public.lab_arena_sep20_revoke_invalid_reward340(jsonb)')"
        )
        assert cursor.fetchone()[0] is None


def test_templates_commit_no_signed_payload_and_keep_scope_exact():
    install = INSTALL_TEMPLATE.read_text(encoding="utf-8")
    cleanup = CLEANUP_TEMPLATE.read_text(encoding="utf-8")
    assert '"signature_b64":"' not in install + cleanup
    assert not re.search(r'"signature_b64"\s*:\s*"[A-Za-z0-9+/=]+"', install + cleanup)
    assert "arena-2026-09-20-rewardhistory340" in install
    assert "arena-2026-09-20-reward-history340" not in install
    assert "SECURITY DEFINER" in install
    assert "TO service_role" in install
    assert "FROM PUBLIC, lab_arena_service" in install
    assert "accepted.epoch >= replacement_epoch" in install
    assert "outcome.epoch >= replacement_epoch" in install
    assert "DISABLE TRIGGER USER" in install
    assert "ENABLE TRIGGER USER" in install
    assert "DELETE FROM" not in install + cleanup
    assert "DROP FUNCTION IF EXISTS" in cleanup
