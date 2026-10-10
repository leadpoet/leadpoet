"""Priority metadata reads do not grant accounting or recovery authority."""
from lab_arena.store import ArenaStore, PsycopgTransport
from tests.lab_arena import settled_score_host_recovery441_postgres_test as recovery

# Use the original committed security/expiry migration chain.
database = recovery.database
migrated = recovery.migrated


def test_real_abandoned_selector_preserves_all_settled_reclaim_gate(database, migrated):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        lease = recovery._seed(conn, settled=False)
        before = recovery._audit(conn, lease)
        transport = PsycopgTransport(lambda: psycopg.connect(**dsn))
        store = ArenaStore(transport)
        try:
            assert store.list_abandoned_billing_runs(recovery.recovery.prior.ROUND) == [lease['run_id']]
            assert recovery._audit(conn, lease) == before
            # Scheduling a billing read does not authorize release of a dispatch.
            assert recovery._expire(conn) == {'status': 'ok', 'expired': 0, 'retried': 0}
            assert recovery._settle(conn, lease)['status'] == 'settled'
            assert recovery._expire(conn) == {'status': 'ok', 'expired': 1, 'retried': 1}
            assert store.list_abandoned_billing_runs(recovery.recovery.prior.ROUND) == []
        finally:
            transport.close()
