"""The paid local runner must retain the same trusted capture as production."""

import io
import tarfile
from pathlib import Path

from lab_arena import runner, training_export, training_input, training_observations
from scripts.verify_arena_full_thinking_e2e import Evidence, FullThinkingCaptureApi, FullThinkingLocalRuntime
from tests.lab_arena.test_lab_arena_runner import FakeApi, lease, make_config
from tests.lab_arena.parallel_twenty_icp_execution_postgres_test import _verified_test_pool
from tests.lab_arena.training_input_test import CLAIM_HASH, TOKEN, service_for
from tests.lab_arena.training_observations_test import observation, response


def test_local_runner_capture_routes_persist_real_service_objects(tmp_path):
    service, objects = service_for(tmp_path)
    api = FullThinkingCaptureApi(service)
    lease = {"run_id": "r1", "kind": "execute", "lease_generation": 1,
             "claim_request_hash": CLAIM_HASH}
    document = {"icp": {"prompt": "Find verified companies"}}

    assert api.training_input("r1", TOKEN, document)["status"] == "stored"
    assert training_input.read_input(objects, "r1", "execute", 1, CLAIM_HASH)["input"] == document

    state = runner.RunState(lease=lease, lease_token=TOKEN)
    with state.lock:
        state.worker_observations.append(observation())
    journal = runner._ObservationJournal(api, lease, TOKEN, state)
    assert journal._prepare() and journal._send_pending()
    segments, partial, error = training_observations.read_journal_prefix(
        objects, "r1", "execute", 1, CLAIM_HASH)
    assert error is None and len(segments) == 1 and partial["observation_count"] == 1

    final = state.worker_observations.document()
    assert api.training_observations("r1", TOKEN, final)["status"] == "stored"
    assert training_observations.read_snapshot(
        objects, "r1", "execute", 1, CLAIM_HASH)["payload"] == final


def test_local_runner_persists_fixture_agent_ack_and_phase_states(tmp_path, monkeypatch):
    """Exercise the staged agent, worker socket, and real capture API together."""
    service, objects = service_for(tmp_path)
    events = []
    def save_trajectory(_run_id, _lease_hash, batch):
        events.extend(batch)
        return {"status": "accepted", "accepted": len(batch), "inserted": len(batch), "existing": 0}
    service._store.append_trajectory_events = save_trajectory
    capture_api = FullThinkingCaptureApi(service)
    answer = {"choices": [{"finish_reason": "stop", "message": {
        "role": "assistant", "content": '{"companies":[]}',
        "reasoning": "No company has verified evidence in this scripted test.",
    }}]}

    class ScriptedApi(FakeApi):
        def __init__(self):
            claimed = lease()
            claimed["lease_token"] = TOKEN
            claimed["parallel_twenty_icp_execution"] = True
            super().__init__([claimed], broker_documents=[response(200, answer, outcome="settled")])

        def training_input(self, run_id, token, document):
            return capture_api.training_input(run_id, token, document)

        def training_observations(self, run_id, token, document):
            return capture_api.training_observations(run_id, token, document)

        def training_observation_segment(self, run_id, token, document):
            return capture_api.training_observation_segment(run_id, token, document)

        def trajectory(self, run_id, token, batch):
            return capture_api.trajectory(run_id, token, batch)

    fixture = Path(__file__).parent / "fixtures" / "full_thinking_sourcing_e2e" / "harness.py"
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w:gz") as bundle:
        bundle.add(fixture, arcname="harness.py")
        bundle.add(Path(__file__).parents[2] / "LICENSE", arcname="LICENSE")
    api = ScriptedApi()
    api.source_payload = archive.getvalue()
    api.leases[0]["source_size_bytes"] = len(api.source_payload)
    deps = tmp_path / "deps"
    deps.mkdir()
    monkeypatch.setenv("LAB_ARENA_PAID_MODEL_DEPS", str(deps))
    monkeypatch.setenv("LAB_ARENA_FULL_E2E_MODEL_DEPS", str(deps))
    evidence = Evidence(tmp_path / "evidence.json")
    sandbox = FullThinkingLocalRuntime(tmp_path / "processes", [], evidence)
    (tmp_path / "work").mkdir()
    config = make_config(tmp_path, api, sandbox)
    config.proxy_worker_pool = _verified_test_pool(2)
    worker = runner.Runner(config)
    try:
        assert worker.run_once() == 1
        assert worker.abandoned == 0, worker.completed
    finally:
        worker.close()

    assert api.completions[0]["body"]["result"]["terminal_status"] == "accepted"
    assert len(api.provider_frames) == 1
    assert {"runtime.started", "runtime.finished"} <= {item["kind"] for item in events}
    saved = training_observations.read_snapshot(objects, "r1", "execute", 1, CLAIM_HASH)
    payload = saved["payload"]
    assert payload["complete"] is True
    receipts = [row for row in payload["agent_observations"] if row["kind"] == "response_received"]
    assert len(receipts) == 1
    assert receipts[0]["data"]["call_identity"] == payload["response_receipts"][0]["call_identity"]
    assert receipts[0]["privacy"]["redacted"] is False
    phases = {row["data"]["name"] for row in payload["agent_observations"]
              if row["kind"] == "state_snapshot"}
    assert {"after_discovery", "after_verification", "after_admission", "final",
            "training_capture_summary", "automatic_capture_summary"} <= phases
    assert training_export._reported_phase_coverage(payload) is True
