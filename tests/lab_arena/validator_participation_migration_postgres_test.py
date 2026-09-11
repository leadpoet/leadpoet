"""N-1 compatibility for the additive Arena participation migration."""

from pathlib import Path

from lab_arena.store import ArenaStore, PsycopgTransport, hash_lease_token
from tests.lab_arena.lab_arena_pg_harness import (
    POSTGREST_MIGRATIONS,
    database_with_lab_arena_migration,
)
from tests.lab_arena.test_lab_arena_migration_postgres import (
    claim,
    complete,
    open_round,
)


MIGRATION = (
    Path(__file__).resolve().parents[2]
    / "scripts/216-lab-arena-validator-participation.sql"
)


def test_n_minus_one_lease_completes_after_additive_migration():
    migrations = tuple(name for name in POSTGREST_MIGRATIONS if name != MIGRATION.name)
    database = database_with_lab_arena_migration(migrations)
    psycopg2, dsn = next(database)
    transport = PsycopgTransport(lambda: psycopg2.connect(**dsn))
    store = ArenaStore(transport)
    admin = psycopg2.connect(**dsn)
    admin.autocommit = True
    try:
        round_id = "arena-2026-09-11-n1"
        runners, _ = open_round(store, round_id, prefix="nminus", participants=1)
        runner = runners[0]
        leased, token, _, _ = claim(store, round_id, runner)
        run_id, lease_hash = leased["run_id"], hash_lease_token(token)
        assert "participation_accepted_at" not in store.get_run(run_id)

        with admin.cursor() as cursor:
            cursor.execute(MIGRATION.read_text())
        assert (
            complete(store, run_id, lease_hash, "accepted", output_ref="output")[
                "status"
            ]
            == "accepted"
        )
        with admin.cursor() as cursor:
            cursor.execute(
                "SELECT participation_accepted_at FROM public.lab_arena_runs "
                "WHERE run_id = %s",
                (run_id,),
            )
            timestamp = cursor.fetchone()[0]
            cursor.execute(
                "SELECT public.lab_arena_has_recent_participation_v1(%s, %s, %s)",
                ("finney", 71, runner),
            )
            eligible = cursor.fetchone()[0]
        assert timestamp is not None
        assert eligible == {"eligible": True}

        with admin.cursor() as cursor:
            cursor.execute(MIGRATION.read_text())
        repeated = complete(
            store, run_id, lease_hash, "accepted", output_ref="output"
        )
        assert repeated["status"] == "accepted" and repeated["idempotent"] is True
        with admin.cursor() as cursor:
            cursor.execute(
                "SELECT participation_accepted_at FROM public.lab_arena_runs "
                "WHERE run_id = %s",
                (run_id,),
            )
            assert cursor.fetchone()[0] == timestamp
    finally:
        admin.close()
        transport.close()
        database.close()
