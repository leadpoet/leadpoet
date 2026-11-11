"""URL redaction must preserve separate pages in a judge's local trace."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile

import pytest

from lab_arena import contracts, lab_arena_checkpoint as checkpoint
from lab_arena import training_capture, training_observations as obs
from tests.lab_arena.judge_local_tool_test import event, frame, worker
from tests.lab_arena.training_export_test import Objects


def pages():
    # Production's document URLs use different KEY values on the same path.
    # Keep the privacy rule: a field called KEY is still redacted.
    base = "https://docs.example/doc?KEY="
    return {base + "DOC_A": {"final_url": base + "DOC_A", "text": "First complete source."},
            base + "DOC_B": {"final_url": base + "DOC_B", "text": "Second complete source."}}


@pytest.mark.parametrize("submission", ["baseline-2026-11-11", "sub-control"])
def test_real_judge_socket_preserves_redacted_page_collisions(monkeypatch, submission):
    with tempfile.TemporaryDirectory(prefix="source-capture-", dir="/tmp") as short:
        server, state, api = worker(Path(short))
        state.lease["submission_id"] = submission
        server.start()
        original = event(result={"claims": {"industry": "UNPROVEN"},
                                 "_server_fetched_pages": pages()})
        before = deepcopy(original)
        try:
            monkeypatch.setenv(checkpoint.WORKER_SOCKET_ENV, str(Path(short) / "worker.sock"))
            monkeypatch.setenv(checkpoint.JUDGE_CAPTURE_POLICY_ENV, checkpoint.JUDGE_CAPTURE_POLICY)
            assert checkpoint.record_judge_tool_event(original) == 0
            assert server.wait_for_handlers()
            assert original == before
            document = state.worker_observations.document()
            assert document["complete"] and document["agent_observation_count"] == 1
            row = document["agent_observations"][0]
            retained = row["data"]["result"]["_server_fetched_pages"]
            assert [p["page"]["text"] for p in retained] == [p["text"] for p in pages().values()]
            assert len(retained) == 2 and retained[0]["url"] == retained[1]["url"]
            assert row["privacy"] == {"redacted": True, "redactions": ["credential"]}
            assert "DOC_A" not in json.dumps(row) and "DOC_B" not in json.dumps(row)
            assert row["data"]["result"]["claims"] == before["result"]["claims"]
            assert len(state.calls) == len(api.provider_frames) == 0
            snapshot, compressed = obs.build_snapshot("score:1", "score", document,
                lease_generation=1, claim_request_hash="sha256:" + "a" * 64)
            objects = Objects()
            objects.put(obs.object_ref("score:1", 1, "sha256:" + "a" * 64), compressed)
            restored = obs.read_snapshot(objects, "score:1", "score", 1, "sha256:" + "a" * 64)
            assert restored == snapshot
            assert restored["payload"]["agent_observations"][0] == row
        finally:
            server.stop()


def test_known_secrets_are_redacted_in_page_urls_text_and_other_fields(tmp_path):
    server, state, _ = worker(tmp_path)
    value = event(arguments="host-lease-token", result={"_server_fetched_pages": {
        "https://example.test/host-lease-token": {"text": "host-lease-token", "password": "private"}}})
    assert json.loads(server._handle_training_control(json.dumps(frame(value)).encode()))["recorded"]
    row = state.worker_observations.agent_observations[0]
    assert "host-lease-token" not in json.dumps(row) and '"private"' not in json.dumps(row)
    assert row["privacy"]["redacted"] is True


def test_historical_map_snapshot_hash_and_readback_are_unchanged():
    original = event(result={"_server_fetched_pages": {"https://example.test/": {"text": "source"}}})
    row = {"id": 0, "kind": "judge_local_tool", "observed_at": "2026-11-11T00:00:00Z",
           "after_action_sequence": None, "depends_on": [], "data": original,
           "privacy": {"redacted": False, "redactions": []}}
    document = obs.Buffer(agent_observations=[row], agent_observation_count=1).document()
    before = contracts.document_hash(document)
    snapshot, _ = obs.build_snapshot("old:1", "score", document,
        lease_generation=1, claim_request_hash="sha256:" + "b" * 64)
    assert snapshot["payload"] == document
    assert snapshot["payload_sha256"] == before
    assert isinstance(snapshot["payload"]["agent_observations"][0]["data"]["result"]["_server_fetched_pages"], dict)


def test_unrelated_key_collisions_still_fail_closed(tmp_path):
    server, state, _ = worker(tmp_path)
    with pytest.raises(training_capture.CaptureError, match="request_redaction_key_collision"):
        training_capture.sanitize_training_value(pages())
    assert not json.loads(server._handle_training_control(json.dumps(frame(event(result=pages()))).encode()))["recorded"]
    assert state.worker_observations.document()["complete"] is False


def test_malformed_mapping_keys_still_fail_closed():
    row = {"id": 0, "kind": "judge_local_tool", "observed_at": "2026-11-11T00:00:00Z",
           "after_action_sequence": None, "depends_on": [],
           "data": event(result={"_server_fetched_pages": {1: {"text": "invalid key"}}}),
           "privacy": {"redacted": False, "redactions": []}}
    buffer = obs.Buffer()
    assert buffer.append_agent_observation(row) is None
    assert not buffer.document()["complete"]


def test_source_page_records_export_without_becoming_training_targets(tmp_path):
    from lab_arena import training_thinking
    from tests.lab_arena.judge_thinking_export_test import judge_example, records
    from tests.lab_arena.training_steps_test import seal
    server, state, _ = worker(tmp_path)
    source_pages = {"https://example.test/a": {"text": "First full evidence"},
                    "https://example.test/b": {"text": "Second full evidence"}}
    assert json.loads(server._handle_training_control(json.dumps(frame(event(result={
        "_server_fetched_pages": source_pages}))).encode()))["recorded"]
    example = judge_example()
    before, _ = records(example, role="judge")
    artifact = example["worker_observation_artifacts"][example["score_run_id"]]
    rows = artifact["payload"]["agent_observations"]
    captured = deepcopy(state.worker_observations.agent_observations[0])
    captured["id"] = len(rows)
    rows.append(captured)
    artifact["payload_sha256"] = contracts.document_hash(artifact["payload"])
    for call in example["provider_calls"]:
        if call["run_id"] == example["score_run_id"]:
            for ref in call["response_provenance"]["agent_receipt_refs"] + [call["response_provenance"]["worker_receipt_ref"]]:
                ref["payload_sha256"] = artifact["payload_sha256"]
    example = seal(example)
    after, _ = records(example, role="judge")
    assert len(before) == len(after) == 1
    assert before[0]["generated_thinking"] == after[0]["generated_thinking"]
    assert after[0]["quality"] == {"status": "unreviewed", "correct_action": "not_established"}
    local = [r for r in training_thinking.readable_trace(example)["observations"] if r["kind"] == "judge_local_tool"]
    assert local[0]["value"]["data"]["result"]["_server_fetched_pages"] == [
        {"url": url, "page": page} for url, page in source_pages.items()]


@pytest.mark.parametrize("kind,marker", [("execute", True), ("score", False)])
def test_capture_remains_frozen_judge_only(tmp_path, kind, marker):
    server, state, api = worker(tmp_path, kind=kind, marker=marker)
    assert not json.loads(server._handle_training_control(json.dumps(frame(event(result={"_server_fetched_pages": pages()}))).encode()))["recorded"]
    assert state.worker_observations.agent_observations == [] and api.provider_frames == []
