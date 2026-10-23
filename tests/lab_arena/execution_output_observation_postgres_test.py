"""Existing private trajectory persistence retains bounded host output observations."""
import json

from lab_arena import runner, runtime, trajectory
from tests.lab_arena.trajectory_postgres_test import (
    LEASE, RUN, _connect_store, database, seeded,
)
from tests.lab_arena.trajectory_test import _service_for_trajectory


def test_private_output_observation_round_trips_without_result_contract_changes(seeded):
    psycopg2, dsn = seeded
    raw = b'{"api_key":"never-store-this-output-body"'
    observed = runner._execution_output_observation(
        runtime.fake_result(output_bytes=raw), None, "invalid_output",
        {"error_class": "execution_output_invalid", "reason": "invalid_json"},
    )
    event = trajectory.event("runtime.finished", {
        "status": "provider_error", "company_count": 0, "execution_output": observed,
    })
    assert len(json.dumps(event).encode()) < trajectory.MAX_EVENT_BYTES
    store = _connect_store(psycopg2, dsn)
    try:
        service = _service_for_trajectory(store)
        assert service.handle_trajectory(RUN, LEASE, {"events": [event]})["inserted"] == 1
        assert service.handle_trajectory(RUN, LEASE, {"events": [event]})["existing"] == 1
        with psycopg2.connect(**dsn) as connection, connection.cursor() as cursor:
            cursor.execute("SELECT content FROM public.lab_arena_trajectory_events "
                           "WHERE run_id=%s AND event_id=%s", (RUN, event["event_id"]))
            content = cursor.fetchone()[0]
            assert content["execution_output"] == observed
            assert "never-store-this-output-body" not in json.dumps(content)
            cursor.execute("SELECT result_doc FROM public.lab_arena_runs WHERE run_id=%s", (RUN,))
            assert cursor.fetchone()[0] is None
        assert observed["company_count"] is None and observed["state"] == "invalid"
    finally:
        store.close()
