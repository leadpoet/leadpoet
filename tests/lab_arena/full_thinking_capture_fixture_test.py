"""The paid local runner must retain the same trusted capture as production."""

from lab_arena import runner, training_input, training_observations
from scripts.verify_arena_full_thinking_e2e import FullThinkingCaptureApi
from tests.lab_arena.training_input_test import CLAIM_HASH, TOKEN, service_for
from tests.lab_arena.training_observations_test import observation


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
