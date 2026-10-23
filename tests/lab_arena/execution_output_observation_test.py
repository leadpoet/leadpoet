"""Private host observations distinguish output parsing from final classification."""
import base64
import json
import os

import pytest

from lab_arena import contracts, runner, runtime, scoring, shim, trajectory
from tests.lab_arena.test_lab_arena_runner import (
    FakeApi, lease, make_config, scoring_lease, valid_company,
)
from tests.lab_arena.output_diagnostics_test import CheckpointSequenceRuntime, checkpoint_lease


EMPTY = b'{"companies":[]}'
SECRET = "do-not-persist-output-secret"


class CaptureApi(FakeApi):
    def __init__(self, leases):
        super().__init__(leases)
        self.events = []

    def trajectory(self, run_id, lease_token, events):
        self.events.extend(trajectory.validate_batch({"events": events}))
        return {"accepted": len(events)}

    def provider(self, run_id, lease_token, frame):
        self.provider_frames.append(dict(frame))
        body = b'{"error":{"code":"provider_unavailable"}}'
        return {"status": 502, "headers": {"content-type": "application/json"},
                "body_b64": base64.b64encode(body).decode(),
                "call": {"call_identity": contracts.document_hash(["failure", frame["action_sequence"]]),
                         "operation_id": frame["operation_id"], "action_sequence": frame["action_sequence"],
                         "provider": "deepline", "funding_source": "miner_key",
                         "provider_status": 502, "outcome": "uncertain",
                         "error_code": "provider_unavailable"}}


class SelectedOutputRuntime:
    def __init__(self, raw, *, failure=True, output_error=None, timed_out=False):
        self.raw, self.failure = raw, failure
        self.output_error, self.timed_out = output_error, timed_out

    def run_icp(self, spec, **_kwargs):
        if self.failure:
            os.environ[shim.WORKER_SOCKET_ENV] = str(spec.socket_path)
            try:
                assert shim.dispatch("deepline.execute", {"tool": "exa_search", "payload": {}}, 5000)[0] == 502
            finally:
                os.environ.pop(shim.WORKER_SOCKET_ENV, None)
        return runtime.fake_result(exit_code=0, output_bytes=self.raw,
                                   output_error=self.output_error, timed_out=self.timed_out)


def execute(tmp_path, sandbox, selected_lease=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "work").mkdir()
    api = CaptureApi([selected_lease or lease()])
    assert runner.Runner(make_config(tmp_path, api, sandbox)).run_once() == 1
    finish = [event for event in api.events if event["kind"] == "runtime.finished"]
    assert len(finish) == 1
    assert len(json.dumps(finish[0], sort_keys=True, separators=(",", ":")).encode()) <= trajectory.MAX_EVENT_BYTES
    return api, finish[0]["content"]


@pytest.mark.parametrize("raw,read_error,state,terminal,code", [
    (EMPTY, None, "validated_empty", "accepted", None),
    (None, None, "missing", "model_error", "missing_output"),
    (("{\"private\":\"" + SECRET).encode(), None, "invalid", "invalid_output", "invalid_json"),
    ((json.dumps({"companies": [{"private_payload": SECRET}]})).encode(), None,
     "invalid", "invalid_output", "output_contract_violation"),
    (b'{"schema_version":"unsupported","companies":[]}', None,
     "invalid", "invalid_output", "output_schema_mismatch"),
    (None, SECRET, "unreadable", "invalid_output", "sandbox_output_error"),
])
def test_preclassification_output_survives_provider_override(tmp_path, raw, read_error, state, terminal, code):
    api, finish = execute(tmp_path, SelectedOutputRuntime(raw, output_error=read_error))
    completion = api.completions[0]["body"]
    assert completion["result"]["terminal_status"] == "provider_error"
    assert completion["output"] is None and "failure_diagnostic" not in completion["result"]
    assert finish["company_count"] == 0  # Existing post-override field keeps its meaning.
    observed = finish["execution_output"]
    assert observed == {"state": state, "terminal_before_classification": terminal,
                        "failure_code": code, "company_count": 0 if state == "validated_empty" else None,
                        "output_bytes": len(raw) if raw is not None else None,
                        "output_sha256": contracts.hash_bytes(raw) if raw is not None else None,
                        "checkpoint_output_invalid": False}
    assert SECRET not in json.dumps(api.events) and SECRET not in json.dumps(completion)
    assert len(json.dumps(observed).encode()) < 512
    contracts.validate_run_result(completion["result"])


def test_nonempty_result_and_provider_frames_are_unchanged(tmp_path, monkeypatch):
    raw = json.dumps({"companies": [valid_company(1)]}).encode()
    api, finish = execute(tmp_path / "observed", SelectedOutputRuntime(raw))
    assert finish["execution_output"]["state"] == "validated_nonempty"
    assert finish["execution_output"]["company_count"] == 1
    assert finish["execution_output"]["output_sha256"] == contracts.hash_bytes(raw)
    monkeypatch.setattr(runner, "_execution_output_observation", lambda *_: None)
    control, _ = execute(tmp_path / "control", SelectedOutputRuntime(raw))
    assert api.provider_frames == control.provider_frames
    assert api.completions[0]["body"] == control.completions[0]["body"]
    assert api.completions[0]["body"]["result"]["terminal_status"] == "accepted"


def test_present_but_unvalidated_timeout_does_not_claim_parse_success(tmp_path):
    api, finish = execute(tmp_path, SelectedOutputRuntime(EMPTY, failure=False, timed_out=True))
    assert api.completions[0]["body"]["result"]["terminal_status"] == "model_timeout"
    assert finish["execution_output"]["state"] == "not_validated"
    assert finish["execution_output"]["company_count"] is None
    assert finish["execution_output"]["output_sha256"] == contracts.hash_bytes(EMPTY)


def test_checkpoint_observation_hashes_selected_valid_bytes_not_rejected_final_file(tmp_path):
    sandbox = CheckpointSequenceRuntime(tmp_path / "runtime-work", mode="fast")
    sandbox.work_dir.mkdir()
    api, finish = execute(tmp_path, sandbox, checkpoint_lease())
    assert sandbox.result.output_bytes == EMPTY and sandbox.result.checkpoint_output_invalid
    observed = finish["execution_output"]
    assert observed["state"] == "validated_empty" and observed["checkpoint_output_invalid"] is True
    assert observed["output_sha256"] == contracts.hash_bytes(EMPTY)
    assert api.completions[0]["body"]["result"]["terminal_status"] == "accepted"
    assert api.completions[0]["body"]["result"]["failure_diagnostic"]["reason"] == "invalid_final_checkpoint"


def test_private_observation_failure_cannot_change_completion(tmp_path, monkeypatch):
    def broken(*_args):
        raise ValueError(SECRET)
    monkeypatch.setattr(runner, "_execution_output_observation", broken)
    api, finish = execute(tmp_path, SelectedOutputRuntime(EMPTY, failure=False))
    assert api.completions[0]["body"]["result"]["terminal_status"] == "accepted"
    assert "execution_output" not in finish and SECRET not in json.dumps(api.events)


def test_shared_judge_path_does_not_observe_execution_output(tmp_path, monkeypatch):
    observed = []
    monkeypatch.setattr(runner, "_execution_output_observation", lambda *args: observed.append(args))
    output = scoring.build_scoring_output("r1", [
        {"final_score": 71.0, "failure_reason": ""},
        {"final_score": 44.5, "failure_reason": ""},
    ])
    api, finish = execute(tmp_path, SelectedOutputRuntime(json.dumps(output).encode(), failure=False), scoring_lease())
    assert observed == [] and "execution_output" not in finish
    assert api.completions[0]["body"]["output"] == output


def test_observation_cannot_displace_existing_finish_event_at_size_limit(tmp_path, monkeypatch):
    observe = runner._execution_output_observation
    monkeypatch.setattr(runner, "_execution_output_observation", lambda *_: None)
    control, _ = execute(tmp_path / "control", SelectedOutputRuntime(EMPTY, failure=False))
    original = next(event for event in control.events if event["kind"] == "runtime.finished")
    original_bytes = len(json.dumps(original, sort_keys=True, separators=(",", ":")).encode())
    monkeypatch.setattr(runner, "_execution_output_observation", observe)
    monkeypatch.setattr(trajectory, "MAX_EVENT_BYTES", original_bytes + 32)
    api, finish = execute(tmp_path / "bounded", SelectedOutputRuntime(EMPTY, failure=False))
    assert api.completions[0]["body"] == control.completions[0]["body"]
    assert "execution_output" not in finish
