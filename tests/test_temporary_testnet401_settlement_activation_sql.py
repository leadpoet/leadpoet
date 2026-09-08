from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from tests.lab_arena.lab_arena_pg_harness import database_with_lab_arena_migration


ROOT = Path(__file__).resolve().parents[1]
MIGRATION = ROOT / "scripts/193-activate-temporary-testnet401-chain-realized-settlement.sql"
MIGRATION_SQL = MIGRATION.read_text(encoding="utf-8")
VALIDATOR = "5CJyMxw6YJJvLhPf58gSpMB7mvSKSCMx9RXhXJum6cNfqMEz"


SCHEMA_SQL = r"""
CREATE TABLE public.research_lab_compact_weight_authorities_v2 (
    bundle_hash TEXT NOT NULL,
    netuid INTEGER NOT NULL CHECK (netuid > 0),
    epoch_id BIGINT NOT NULL CHECK (epoch_id >= 0),
    validator_hotkey TEXT NOT NULL,
    authority_stage TEXT NOT NULL CHECK (
        authority_stage IN ('published', 'finalized')
    ),
    schema_version TEXT NOT NULL CHECK (
        schema_version = 'leadpoet.compact_published_weight_authority.v2'
    ),
    compact_finalization_hash TEXT,
    finalization_receipt_hash TEXT,
    authority_doc JSONB NOT NULL CHECK (jsonb_typeof(authority_doc) = 'object'),
    PRIMARY KEY (bundle_hash, authority_stage),
    UNIQUE (netuid, epoch_id, validator_hotkey, authority_stage)
);
CREATE FUNCTION public.prevent_research_lab_attested_v2_mutation()
RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION 'immutable';
END;
$$;
CREATE TABLE public.research_lab_chain_realized_settlement_activation_v1 (
    netuid INTEGER PRIMARY KEY CHECK (netuid > 0),
    schema_version TEXT NOT NULL CHECK (
        schema_version =
        'leadpoet.research_lab_chain_realized_settlement_activation.v1'
    ),
    first_epoch_id INTEGER NOT NULL CHECK (first_epoch_id >= 0),
    source_bundle_hash TEXT NOT NULL CHECK (
        source_bundle_hash ~ '^sha256:[0-9a-f]{64}$'
    ),
    source_bundle_epoch_id INTEGER NOT NULL CHECK (
        source_bundle_epoch_id = first_epoch_id
    ),
    source_finalized_block BIGINT NOT NULL CHECK (source_finalized_block >= 0),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE TRIGGER prevent_research_lab_chain_settlement_activation_v1_mutation
BEFORE UPDATE OR DELETE
ON public.research_lab_chain_realized_settlement_activation_v1
FOR EACH ROW EXECUTE FUNCTION public.prevent_research_lab_attested_v2_mutation();
"""


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _seed_finalized(
    cursor,
    *,
    netuid: int,
    epoch: int,
    marker: str,
    finalized_block: int,
    validator: str = VALIDATOR,
    stage: str = "finalized",
):
    bundle_hash = f"sha256:{_sha(marker + ':bundle')}"
    cursor.execute(
        """
        INSERT INTO public.research_lab_compact_weight_authorities_v2 VALUES (
            %s, %s, %s, %s, %s,
            'leadpoet.compact_published_weight_authority.v2',
            %s, %s,
            jsonb_build_object(
                'finalization', jsonb_build_object(
                    'compact_submission', jsonb_build_object(
                        'finalization', jsonb_build_object(
                            'finalized_block', %s
                        )
                    )
                )
            )
        )
        """,
        (
            bundle_hash,
            netuid,
            epoch,
            validator,
            stage,
            f"sha256:{_sha(marker + ':finalization')}",
            f"sha256:{_sha(marker + ':receipt')}",
            finalized_block,
        ),
    )
    return bundle_hash


def _activation(cursor, netuid: int):
    cursor.execute(
        """
        SELECT netuid, schema_version, first_epoch_id, source_bundle_hash,
               source_bundle_epoch_id, source_finalized_block, created_at
        FROM public.research_lab_chain_realized_settlement_activation_v1
        WHERE netuid = %s
        """,
        (netuid,),
    )
    return cursor.fetchone()


def _delete_activation(cursor, netuid: int):
    cursor.execute(
        """
        ALTER TABLE public.research_lab_chain_realized_settlement_activation_v1
            DISABLE TRIGGER USER;
        DELETE FROM public.research_lab_chain_realized_settlement_activation_v1
        WHERE netuid = %s;
        ALTER TABLE public.research_lab_chain_realized_settlement_activation_v1
            ENABLE TRIGGER USER;
        """,
        (netuid,),
    )


def _run_expect_error(connection, psycopg2, error: str):
    with pytest.raises(psycopg2.Error, match=error):
        with connection.cursor() as cursor:
            cursor.execute(MIGRATION_SQL)
    with connection.cursor() as cursor:
        cursor.execute("ROLLBACK")


def test_activation_migration_uses_first_finalized_compact_testnet401_authority():
    database = database_with_lab_arena_migration(())
    psycopg2, dsn = next(database)
    connection = psycopg2.connect(**dsn)
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            cursor.execute(SCHEMA_SQL)
            finney_hash = f"sha256:{_sha('finney-existing')}"
            cursor.execute(
                """
                INSERT INTO public.research_lab_chain_realized_settlement_activation_v1
                VALUES (
                    71,
                    'leadpoet.research_lab_chain_realized_settlement_activation.v1',
                    24073, %s, 24073, 9000000
                )
                """,
                (finney_hash,),
            )
            _seed_finalized(
                cursor, netuid=71, epoch=1, marker="finney-compact", finalized_block=1
            )
            _seed_finalized(
                cursor,
                netuid=401,
                epoch=1,
                marker="wrong-validator",
                finalized_block=2,
                validator="not-the-approved-validator",
            )
            _seed_finalized(
                cursor,
                netuid=401,
                epoch=2,
                marker="published-only",
                finalized_block=3,
                stage="published",
            )
            later_hash = _seed_finalized(
                cursor,
                netuid=401,
                epoch=22061,
                marker="later",
                finalized_block=7959505,
            )
            first_hash = _seed_finalized(
                cursor,
                netuid=401,
                epoch=22060,
                marker="first",
                finalized_block=7959145,
            )
            finney_before = _activation(cursor, 71)

            cursor.execute(MIGRATION_SQL)
            first_activation = _activation(cursor, 401)
            assert first_activation[0:6] == (
                401,
                "leadpoet.research_lab_chain_realized_settlement_activation.v1",
                22060,
                first_hash,
                22060,
                7959145,
            )
            assert first_activation[3] != later_hash
            assert _activation(cursor, 71) == finney_before

            cursor.execute(MIGRATION_SQL)
            assert _activation(cursor, 401) == first_activation
            assert _activation(cursor, 71) == finney_before

            _delete_activation(cursor, 401)
            conflicting_hash = f"sha256:{_sha('conflicting-activation')}"
            cursor.execute(
                """
                INSERT INTO public.research_lab_chain_realized_settlement_activation_v1
                VALUES (
                    401,
                    'leadpoet.research_lab_chain_realized_settlement_activation.v1',
                    22061, %s, 22061, 7959505
                )
                """,
                (conflicting_hash,),
            )
        _run_expect_error(
            connection, psycopg2, "testnet401_settlement_activation_conflicts"
        )
        with connection.cursor() as cursor:
            assert _activation(cursor, 71) == finney_before
            _delete_activation(cursor, 401)
            cursor.execute(
                "DELETE FROM public.research_lab_compact_weight_authorities_v2 "
                "WHERE netuid = 401"
            )
        _run_expect_error(
            connection,
            psycopg2,
            "testnet401_first_finalized_compact_authority_unavailable",
        )
        with connection.cursor() as cursor:
            assert _activation(cursor, 71) == finney_before
    finally:
        connection.close()
        database.close()


def test_activation_migration_has_no_guessed_source_or_mutation_path():
    assert "MIN(authority.epoch_id)" in MIGRATION_SQL
    assert "authority.authority_stage = 'finalized'" in MIGRATION_SQL
    assert f"'{VALIDATOR}'" in MIGRATION_SQL
    assert "FROM public.research_lab_compact_weight_authorities_v2 authority" in MIGRATION_SQL
    assert "ON CONFLICT" not in MIGRATION_SQL
    assert "UPDATE public.research_lab_chain_realized_settlement_activation_v1" not in MIGRATION_SQL
    assert "DELETE FROM public.research_lab_chain_realized_settlement_activation_v1" not in MIGRATION_SQL
    assert "22060" not in MIGRATION_SQL
    assert "7959145" not in MIGRATION_SQL
