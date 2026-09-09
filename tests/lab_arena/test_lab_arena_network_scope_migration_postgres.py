"""PostgreSQL proof for queryable, backwards-compatible Arena chain scope."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from lab_arena import signing
from lab_arena.service import ArenaService
from lab_arena.store import ArenaStore, PsycopgTransport
from tests.lab_arena.lab_arena_pg_harness import (
    DEFAULT_MIGRATIONS,
    LAB_ARENA_NETWORK_SCOPE_MIGRATION,
    database_with_lab_arena_migration,
)
from tests.lab_arena.test_lab_arena_reward_migration_postgres import (
    MINER_A,
    MINER_B,
    _basis,
    _publish,
)
from tests.lab_arena.test_lab_arena_promotion_migration_postgres import _plan
from tests.postgres_migration_harness import SCRIPTS


def test_network_scope_migration_is_idempotent_and_keeps_legacy_rows_on_finney():
    generator = database_with_lab_arena_migration(
        DEFAULT_MIGRATIONS[
            : DEFAULT_MIGRATIONS.index(LAB_ARENA_NETWORK_SCOPE_MIGRATION)
        ]
    )
    psycopg2, dsn = next(generator)
    connection = psycopg2.connect(**dsn)
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "INSERT INTO public.lab_arena_rounds "
                "(round_id, status, configuration_doc, rewards_enabled) VALUES "
                "(%s, 'open', %s::jsonb, FALSE)",
                ("arena-2026-09-07-legacy", json.dumps({"mode": "live"})),
            )
            migration = (SCRIPTS / LAB_ARENA_NETWORK_SCOPE_MIGRATION).read_text(
                encoding="utf-8"
            )
            cursor.execute(migration)
            cursor.execute(migration)
            cursor.execute(
                "SELECT arena_network_name, arena_netuid "
                "FROM public.lab_arena_rounds WHERE round_id=%s",
                ("arena-2026-09-07-legacy",),
            )
            assert cursor.fetchone() == ("finney", 71)
            cursor.execute("SELECT public.lab_arena_schema_version_v1()")
            assert cursor.fetchone()[0]["version"] == 189

            cursor.execute(
                "INSERT INTO public.lab_arena_rounds "
                "(round_id, status, configuration_doc, rewards_enabled) VALUES "
                "(%s, 'open', %s::jsonb, TRUE)",
                (
                    "arena-2026-09-07-testnet",
                    json.dumps({
                        "mode": "live", "network_name": "test", "netuid": 401,
                    }),
                ),
            )
            cursor.execute(
                "SELECT round_id FROM public.lab_arena_rounds "
                "WHERE arena_network_name='test' AND arena_netuid=401"
            )
            assert cursor.fetchall() == [("arena-2026-09-07-testnet",)]
            with pytest.raises(psycopg2.Error):
                cursor.execute(
                    "INSERT INTO public.lab_arena_rounds "
                    "(round_id, status, configuration_doc, rewards_enabled) VALUES "
                    "(%s, 'open', %s::jsonb, TRUE)",
                    (
                        "arena-2026-09-07-unpaired",
                        json.dumps({"mode": "live", "network_name": "test"}),
                    ),
                )
    finally:
        connection.close()
        generator.close()


@pytest.mark.parametrize("other_network,other_netuid", [("test", 401), ("test", 71), ("finney", 401)])
def test_reward_activation_history_and_epochs_are_chain_scoped(other_network, other_netuid):
    generator = database_with_lab_arena_migration(DEFAULT_MIGRATIONS[:-1])
    psycopg2, dsn = next(generator)
    control = psycopg2.connect(**dsn)
    control.autocommit = True
    transport = PsycopgTransport(lambda: psycopg2.connect(**dsn))
    store = ArenaStore(transport)
    signer = signing.LocalSigner.generate()
    key = signing.signing_key_document(signer.public_key_der)

    def promote(round_id, miner, seed):
        with control.cursor() as cursor:
            cursor.execute(
                "INSERT INTO public.lab_arena_submissions "
                "(submission_id, round_id, miner_hotkey, status, is_king) "
                "VALUES (%s, %s, %s, 'frozen', FALSE)",
                (round_id + "-miner", round_id, miner),
            )
        plan = _plan(seed)
        assert store.prepare_promotion(round_id, plan)["status"] == "prepared"
        assert store.complete_promotion(round_id, plan)["status"] == "promoted"

    try:
        test_winner = "arena-2026-09-08-testwinner"
        published_at = _publish(
            store,
            control,
            test_winner,
            miner=MINER_A,
            baseline_score=50,
            miner_score=60,
            crowned=True,
            network_name=other_network,
            netuid=other_netuid,
        )
        promote(test_winner, MINER_A, "a")
        assert store.activate_reward(
            test_winner,
            _basis(signer, test_winner, published_at, 100, "crowned", MINER_A),
            key,
        )["status"] == "activated"

        test_pending = "arena-2026-09-09-testpending"
        test_pending_at = _publish(
            store,
            control,
            test_pending,
            miner=MINER_B,
            baseline_score=50,
            miner_score=40,
            crowned=False,
            network_name=other_network,
            netuid=other_netuid,
        )
        test_blocked = "arena-2026-09-09-testblocked"
        _publish(
            store, control, test_blocked, miner=MINER_B, baseline_score=50,
            miner_score=60, crowned=True, network_name=other_network, netuid=other_netuid,
        )
        # Upgrade a database with an existing foreign king and pending rows.
        migration = (SCRIPTS / "197-lab-arena-reward-chain-scope.sql").read_text()
        with control.cursor() as cursor:
            cursor.execute(migration)

        # The older unactivated testnet publication does not block Finney. The
        # testnet king also does not become Finney's predecessor.
        finney = "arena-2026-09-09-finney"
        finney_at = _publish(
            store,
            control,
            finney,
            miner=MINER_B,
            baseline_score=50,
            miner_score=40,
            crowned=False,
        )
        finney_basis = _basis(signer, finney, finney_at, 100, "no_king", "")
        assert store.activate_reward(finney, finney_basis, key)["status"] == "activated"

        # The same effective epoch is valid on both chains, and the next
        # testnet no-winner round carries only the prior testnet king.
        promote(test_blocked, MINER_B, "b")
        defended = _basis(
            signer, test_pending, test_pending_at, 101, "defended", MINER_A, 100
        )
        assert store.activate_reward(test_pending, defended, key)["status"] == "activated"
        with control.cursor() as cursor:
            cursor.execute(
                "SELECT round_id, arena_network_name, arena_netuid, king_outcome, "
                "king_hotkey, king_start_epoch FROM public.lab_arena_reward_basis_v1 "
                "WHERE round_id = ANY(%s) ORDER BY arena_netuid, effective_reward_epoch",
                ([test_winner, test_pending, finney],),
            )
            assert set(cursor.fetchall()) == {
                (finney, "finney", 71, "no_king", None, 0),
                (test_winner, other_network, other_netuid, "crowned", MINER_A, 100),
                (test_pending, other_network, other_netuid, "defended", MINER_A, 100),
            }
            # Reapplying the exact migration preserves signed historical rows.
            migration = (SCRIPTS / "197-lab-arena-reward-chain-scope.sql").read_text()
            cursor.execute(migration)
            cursor.execute(migration)
            cursor.execute("SELECT public.lab_arena_schema_version_v1()")
            assert cursor.fetchone()[0]["version"] == 197
            cursor.execute("SELECT reward_basis_doc FROM public.lab_arena_rounds WHERE round_id=%s", (finney,))
            assert cursor.fetchone()[0] == finney_basis
        assert [row["round_id"] for row in store.published_reward_bases(
            mode="live", network_name="finney", netuid=71,
        )] == [finney]
        assert store.activate_reward(finney, finney_basis, key)["status"] == "existing"
        service = object.__new__(ArenaService)
        service._store = store
        service._config = SimpleNamespace(mode="live", network_name="finney", netuid=71)
        assert service.public_reward_basis(99) is None
        assert service.public_reward_basis(101) == finney_basis
        service._config = SimpleNamespace(mode="live", network_name=other_network, netuid=other_netuid)
        assert service.public_reward_basis(101) == defended
    finally:
        transport.close()
        control.close()
        generator.close()


def test_reward_scope_upgrade_under_hosted_role_preserves_access_controls():
    generator = database_with_lab_arena_migration(DEFAULT_MIGRATIONS[:-1])
    psycopg2, dsn = next(generator)
    control = psycopg2.connect(**dsn)
    control.autocommit = True
    try:
        with control.cursor() as cursor:
            cursor.execute(
                'CREATE ROLE reward_migrator LOGIN NOSUPERUSER PASSWORD %s',
                (dsn.get('password', 'disposable-test-password'),),
            )
            cursor.execute('GRANT lab_arena_owner TO reward_migrator')
            cursor.execute('ALTER SCHEMA public OWNER TO reward_migrator')
        admin = psycopg2.connect(**dict(dsn, user='reward_migrator'))
        admin.autocommit = True
        try:
            with admin.cursor() as cursor:
                sql = (SCRIPTS / '197-lab-arena-reward-chain-scope.sql').read_text()
                cursor.execute(sql)
                cursor.execute(sql)
                cursor.execute("SELECT has_schema_privilege('lab_arena_owner','public','CREATE'), "
                               "has_schema_privilege('lab_arena_service','public','CREATE')")
                assert cursor.fetchone() == (False, False)
                for role, allowed in [('anon', False), ('authenticated', False), ('service_role', True), ('lab_arena_service', True)]:
                    cursor.execute("SELECT has_table_privilege(%s,'public.lab_arena_reward_basis_v1','SELECT')", (role,))
                    assert cursor.fetchone()[0] is allowed
                for role in ('anon', 'authenticated'):
                    cursor.execute("SELECT has_function_privilege(%s,'public.lab_arena_activate_reward(text,jsonb,jsonb)','EXECUTE')", (role,))
                    assert cursor.fetchone()[0] is False
        finally:
            admin.close()
    finally:
        control.close()
        generator.close()
