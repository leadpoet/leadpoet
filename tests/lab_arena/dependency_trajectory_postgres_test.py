"""Private dependency diagnostics survive the existing PostgreSQL append RPC."""

import json

import pytest

from lab_arena import runner, trajectory
from lab_arena.service import ArenaService, ServiceError
from tests.lab_arena.trajectory_postgres_test import (
    LEASE, RUN, _connect_store, database, seeded,
)


def test_dependency_details_persist_privately_with_existing_lease_authority(seeded):
    psycopg2, dsn = seeded
    store = _connect_store(psycopg2, dsn)
    gateway = object.__new__(ArenaService)
    gateway._store = store
    gateway._require_round_ownership = lambda _round_id: None
    failures = [
        runner.AgentDependencyError("invalid requirements"),
        *[runner.DependencyInstallInfrastructureError(kind, detail=(
            "failure " + LEASE + " Bearer secret-bearer api_key=plain-secret "
            "https://user:pass@private.test/pip?token=query-secret " + "x" * 1000
        )) for kind in ("network_error", "timeout", "installer_error")],
    ]
    events = [trajectory.event("runtime.error", {
        "status": "model_error" if isinstance(failure, runner.AgentDependencyError) else "provider_error",
        "failure_stage": "setup", "error_class": type(failure).__name__,
        "failure_kind": getattr(failure, "failure_kind", "source_error"),
        "diagnostic_detail": failure.diagnostic_detail,
    }) for failure in failures]
    try:
        with pytest.raises(ServiceError, match="lease_stale"):
            gateway.handle_trajectory(RUN, "wrong-lease", {"events": events})
        assert store.list_trajectory_events(RUN) == []
        result = gateway.handle_trajectory(RUN, LEASE, {"events": events})
        assert result["inserted"] == 4
        replay = gateway.handle_trajectory(RUN, LEASE, {"events": events})
        assert replay["inserted"] == 0 and replay["existing"] == 4
        rows = store.list_trajectory_events(RUN)
        assert len(rows) == 4
        assert {row["content"]["failure_kind"] for row in rows} == {
            "source_error", "network_error", "timeout", "installer_error",
        }
        assert all(len(row["content"]["diagnostic_detail"]) <= runner.MAX_JUDGE_DIAGNOSTIC_CHARS
                   for row in rows)
        serialized = json.dumps(rows)
        for secret in (LEASE, "secret-bearer", "plain-secret", "user:pass", "query-secret"):
            assert secret not in serialized
        with psycopg2.connect(**dsn) as connection, connection.cursor() as cursor:
            cursor.execute("SELECT count(*) FROM public.lab_arena_ledger WHERE run_id=%s", (RUN,))
            assert cursor.fetchone()[0] == 0
            cursor.execute("SET ROLE authenticated")
            with pytest.raises(psycopg2.Error):
                cursor.execute("SELECT content FROM public.lab_arena_trajectory_events")
    finally:
        store.close()
