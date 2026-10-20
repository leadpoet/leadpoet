"""Automatic observations remain private, exact, bounded and non-authoritative."""

import json
import os
import shutil
import tempfile
from dataclasses import replace
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from lab_arena import (
    agent_entrypoint, automatic_capture, contracts, lab_arena_checkpoint,
    runner, runtime, shim, training_observations,
)
from lab_arena.api import create_app
from lab_arena.output import output_document_from_bytes
from tests.lab_arena.test_lab_arena_runner import BridgingRuntime, FakeApi, lease, make_config, valid_company
from tests.lab_arena.test_lab_arena_runtime import make_spec
from tests.lab_arena.training_input_test import CLAIM_HASH, TOKEN, service_for
from tests.lab_arena.training_observations_test import IDENTITY, response


def capture(text, *, encoding="utf-8", complete=True):
    raw = text.encode(encoding)
    return {
        "source": "runtime_http_v1", "complete": complete, "encoding": encoding,
        "received_body_sha256": contracts.hash_bytes(raw), "received_body_bytes": len(raw),
    }


def web_row(text, *, context=True, complete=True):
    data = {
        "url": "https://example.com/page", "final_url": "https://example.com/page",
        "retrieved_at": "2026-10-20T12:00:00Z", "status": 200, "content_type": "text/html",
        "body": text, "body_ref": None, "body_sha256": contracts.hash_bytes(text.encode()),
        "body_bytes": len(text.encode()), "stored_body_sha256": None, "stored_body_bytes": None,
    }
    if context:
        data["capture"] = capture(text, complete=complete)
    return {
        "id": 0, "kind": "web_fetch", "observed_at": "2026-10-20T12:00:00Z",
        "after_action_sequence": None, "depends_on": [], "data": data,
        "privacy": {"redacted": False, "redactions": []},
    }


@pytest.mark.parametrize("context", [False, True])
@pytest.mark.parametrize("complete", [False, True])
def test_existing_and_runtime_web_records_roundtrip(context, complete, tmp_path):
    service, objects = service_for(tmp_path)
    buffer = training_observations.Buffer()
    assert buffer.append_agent_observation(web_row("Exact page \x00 café", context=context, complete=complete)) == 0
    document = buffer.document()
    assert document["complete"] is (complete or not context)
    result = TestClient(create_app(service)).post(
        "/arena/v1/runs/r1/training-observations", json=document,
        headers={"x-lab-arena-lease": TOKEN},
    )
    assert result.status_code == 200
    saved = training_observations.read_snapshot(objects, "r1", "execute", 1, CLAIM_HASH)
    assert saved["payload"] == document
    assert saved["schema_version"] == training_observations.SCHEMA_VERSION


@pytest.mark.parametrize("field,value", [
    ("source", "trusted"), ("complete", 1), ("encoding", "base64_codec"),
    ("encoding", "not-a-codec"), ("received_body_sha256", "sha256:" + "0" * 64),
    ("received_body_bytes", True), ("received_body_bytes", 1),
])
def test_runtime_metadata_cannot_claim_different_bytes(field, value):
    row = web_row("actual bytes")
    row["data"]["capture"][field] = value
    with pytest.raises(training_observations.ObservationError):
        training_observations.clean_agent_observation(row)


def test_non_utf8_bytes_are_reversible_and_privacy_stays_redacted():
    text = "café " + TOKEN
    row = web_row(text)
    row["data"]["capture"] = capture(text, encoding="iso8859-1")
    cleaned = training_observations.clean_agent_observation(row, secrets=(TOKEN,))
    assert TOKEN not in json.dumps(cleaned)
    assert cleaned["privacy"] == {"redacted": True, "redactions": ["credential"]}
    assert cleaned["data"]["capture"] == row["data"]["capture"]
    # A strict reader can verify the redacted artifact without pretending to
    # reconstruct the original secret-bearing body.
    assert training_observations.clean_agent_observation(cleaned) == cleaned


def test_body_dedup_preserves_distinct_runtime_context_and_partial_status():
    buffer = training_observations.Buffer()
    assert buffer.append_agent_observation(web_row("same body")) == 0
    second = web_row("same body", complete=False)
    second["id"] = 1
    assert buffer.append_agent_observation(second) == 1
    document = buffer.document()
    assert document["agent_observations"][1]["data"]["body_ref"] == 0
    assert document["agent_observations"][1]["data"]["capture"]["complete"] is False
    assert document["complete"] is False
    document["complete"] = True
    document["failures"] = []
    with pytest.raises(training_observations.ObservationError, match="completeness_invalid"):
        training_observations.build_snapshot(
            "r1", "execute", document, lease_generation=1, claim_request_hash=CLAIM_HASH,
        )


def test_failed_automatic_summary_marks_existing_artifact_incomplete():
    row = web_row("")
    row["kind"] = "state_snapshot"
    row["data"] = {
        "name": "automatic_capture_summary",
        "value": {"complete": False, "counts": {"queue_overflow": 1}},
    }
    buffer = training_observations.Buffer()
    assert buffer.append_agent_observation(row) == 0
    assert buffer.document()["failures"] == ["capture_incomplete"]


def test_unmodified_httpx_provider_client_gets_one_exact_receipt(tmp_path, monkeypatch):
    document = response(200, {"answer": "unchanged"}, outcome="settled")
    expected = b'{"answer": "unchanged"}'
    state = runner.RunState(lease=lease(), lease_token="private-test-lease")
    api = FakeApi([], broker_documents=[document])
    directory = Path(tempfile.mkdtemp(prefix="arena-auto-", dir="/tmp"))
    socket_path = directory / "w.sock"
    server = runner.WorkerSocketServer(socket_path, api, state)
    server.start()
    monkeypatch.setenv(shim.WORKER_SOCKET_ENV, str(socket_path))
    recorder = automatic_capture.install(lab_arena_checkpoint)
    try:
        with httpx.Client(transport=httpx.HTTPTransport(uds=str(socket_path)), trust_env=False) as client:
            result = client.post(
                "http://openrouter.ai/api/v1/chat/completions",
                json={"model": "openai/gpt-4o-mini", "messages": [{"role": "user", "content": "check"}], "max_tokens": 20},
            )
        assert result.status_code == 200 and result.content == expected
        # Existing agents may still explicitly acknowledge the same consumed
        # response. This must not duplicate the runtime's acknowledgement.
        assert lab_arena_checkpoint.record_response_received(
            call_identity=IDENTITY, status=200, body=expected,
        ) is not None
        recorder.close()
        acknowledgements = [
            row for row in state.worker_observations.agent_observations
            if row["kind"] == "response_received"
        ]
        assert len(acknowledgements) == 1
        assert acknowledgements[0]["data"]["body_sha256"] == contracts.hash_bytes(expected)
        assert state.action_sequence == len(api.provider_frames) == 1
        assert state.worker_observations.document()["complete"] is True
    finally:
        recorder.close()
        server.stop()
        shutil.rmtree(directory)


def test_runtime_mount_is_read_only_and_never_added_to_scorers(tmp_path):
    checkpoint = tmp_path / "checkpoint.py"
    checkpoint.write_text("pass\n")
    checkpoint.chmod(0o444)
    capture_path = tmp_path / "automatic_capture.py"
    capture_path.write_text("pass\n")
    capture_path.chmod(0o444)
    spec = make_spec(tmp_path, checkpoint_module_path=checkpoint, automatic_capture_path=capture_path)
    runtime.require_safe_agent_mounts(spec)
    mount = next(row for row in runtime.oci_spec(spec)["mounts"]
                 if row["destination"] == runtime.SANDBOX_AUTOMATIC_CAPTURE_PATH)
    assert mount["source"] == str(capture_path)
    assert "ro" in mount["options"] and "noexec" in mount["options"]
    capture_path.chmod(0o644)
    with pytest.raises(runtime.SandboxSpecError, match="read-only"):
        runtime.require_safe_agent_mounts(spec)
    with pytest.raises(runtime.SandboxSpecError, match="trusted source"):
        make_spec(tmp_path, source_dir=None, dependency_dir=None, agent_entrypoint_path=None,
                  entry_command=runtime.SCORER_ENTRY_COMMAND, automatic_capture_path=capture_path)


def test_entrypoint_installs_before_import_and_flushes_after_output(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    input_path = tmp_path / "input.json"
    input_path.write_text(json.dumps({"icp": {"prompt": "unchanged"}}))
    output = tmp_path / "companies.json"
    recorder = tmp_path / "automatic_capture.py"
    recorder.write_text(
        "import os\nfrom pathlib import Path\n"
        "class Recorder:\n"
        "    def close(self):\n"
        "        assert Path(os.environ['TEST_CAPTURE_OUTPUT']).is_file()\n"
        "        os.environ['TEST_CAPTURE_CLOSED']='yes'\n"
        "        return {'source':'runtime_http_v1','complete':False,'pending_queue':1,'inflight':1}\n"
        "def install(checkpoint):\n"
        "    os.environ['TEST_CAPTURE_INSTALLED']='yes'\n"
        "    return Recorder()\n"
    )
    (source / "harness.py").write_text(
        "import os\nassert os.environ['TEST_CAPTURE_INSTALLED']=='yes'\n"
        "def run_icp(icp): return [{'company_name':icp['prompt']}]\n"
    )
    monkeypatch.setattr(agent_entrypoint, "AUTOMATIC_CAPTURE_PATH", recorder)
    monkeypatch.setenv("TEST_CAPTURE_OUTPUT", str(output))
    monkeypatch.delenv("TEST_CAPTURE_INSTALLED", raising=False)
    monkeypatch.delenv("TEST_CAPTURE_CLOSED", raising=False)
    agent_entrypoint.run(source_dir=source, input_path=input_path, output_path=output)
    assert os.environ["TEST_CAPTURE_CLOSED"] == "yes"
    assert json.loads(output.read_text()) == {"companies": [{"company_name": "unchanged"}]}
    summary = json.loads((tmp_path / runtime.AUTOMATIC_CAPTURE_SUMMARY_NAME).read_text())
    assert summary == {
        "source": "runtime_http_v1", "complete": False, "pending_queue": 1, "inflight": 1,
    }


@pytest.mark.parametrize("failure", ["import", "install", "close"])
def test_capture_failure_does_not_replace_agent_output(tmp_path, monkeypatch, failure):
    source = tmp_path / "source"
    source.mkdir()
    (source / "harness.py").write_text("def run_icp(icp): return [{'company_name':'unchanged'}]\n")
    input_path = tmp_path / "input.json"
    input_path.write_text('{"icp":{}}')
    recorder = tmp_path / "automatic_capture.py"
    recorder.write_text({
        "import": "raise RuntimeError('capture unavailable')\n",
        "install": "def install(checkpoint): raise RuntimeError('capture unavailable')\n",
        "close": "class Recorder:\n    def close(self): raise RuntimeError('capture unavailable')\ndef install(checkpoint): return Recorder()\n",
    }[failure])
    monkeypatch.setattr(agent_entrypoint, "AUTOMATIC_CAPTURE_PATH", recorder)
    output = tmp_path / "companies.json"
    agent_entrypoint.run(source_dir=source, input_path=input_path, output_path=output)
    assert json.loads(output.read_text()) == {"companies": [{"company_name": "unchanged"}]}
    summary = json.loads((tmp_path / runtime.AUTOMATIC_CAPTURE_SUMMARY_NAME).read_text())
    assert summary["complete"] is False
    assert summary["failure"] == ("recorder_close_failed" if failure == "close" else "recorder_unavailable")


@pytest.mark.parametrize("kind", ["missing", "invalid", "list", "bool", "oversize", "symlink", "fifo"])
def test_bad_summary_is_bounded_and_does_not_change_output(tmp_path, kind):
    spec = make_spec(
        tmp_path, checkpoint_module_path=tmp_path / "checkpoint.py",
        automatic_capture_path=tmp_path / "capture.py",
    )
    spec.output_dir.mkdir()
    output = b'{"companies":[]}'
    spec.output_path.write_bytes(output)
    path = spec.output_dir / runtime.AUTOMATIC_CAPTURE_SUMMARY_NAME
    if kind == "invalid":
        path.write_bytes(b"\xff")
    elif kind == "list":
        path.write_text("[]")
    elif kind == "bool":
        path.write_text('{"source":"runtime_http_v1","complete":1}')
    elif kind == "oversize":
        path.write_bytes(b" " * 16_385)
    elif kind == "symlink":
        path.symlink_to(spec.output_path)
    elif kind == "fifo":
        os.mkfifo(path)
    summary = runtime.read_automatic_capture_summary(spec)
    assert summary == {"source": "runtime_http_v1", "complete": False, "failure": "summary_unavailable"}
    assert runtime.read_output(spec) == output


def test_valid_summary_roundtrips_but_scorer_does_not_read_it(tmp_path):
    spec = make_spec(
        tmp_path, checkpoint_module_path=tmp_path / "checkpoint.py",
        automatic_capture_path=tmp_path / "capture.py",
    )
    spec.output_dir.mkdir()
    summary = {
        "source": "runtime_http_v1", "complete": False, "pending_queue": 1,
        "inflight": 1, "counts": {"close_flush_timeout": 1},
    }
    (spec.output_dir / runtime.AUTOMATIC_CAPTURE_SUMMARY_NAME).write_text(json.dumps(summary))
    assert runtime.read_automatic_capture_summary(spec) == summary
    scorer = make_spec(
        tmp_path, source_dir=None, dependency_dir=None, agent_entrypoint_path=None,
        entry_command=runtime.SCORER_ENTRY_COMMAND,
    )
    assert runtime.read_automatic_capture_summary(scorer) is None


@pytest.mark.parametrize("complete", [True, False])
def test_runner_persists_exit_summary_without_changing_output_or_cost(tmp_path, complete):
    summary = {"source": "runtime_http_v1", "complete": complete, "pending_queue": 0 if complete else 1}

    class CapturedApi(FakeApi):
        def training_observations(self, run_id, token, document):
            self.saved = document
            return {"status": "stored", "payload_sha256": contracts.document_hash(document)}

    class CapturedRuntime(BridgingRuntime):
        def run_icp(self, spec, **kwargs):
            assert spec.automatic_capture_path.read_bytes() == runner.AUTOMATIC_CAPTURE_PATH.read_bytes()
            assert spec.automatic_capture_path.stat().st_mode & 0o222 == 0
            return replace(super().run_icp(spec, **kwargs), automatic_capture_summary=summary)

    api = CapturedApi([lease()])
    expected = {"companies": [valid_company(1)]}
    sandbox = CapturedRuntime(output=expected, calls=1)
    (tmp_path / "work").mkdir()
    worker = runner.Runner(make_config(tmp_path, api, sandbox))
    assert worker.run_once() == 1
    assert worker.abandoned == 0
    completion = api.completions[0]["body"]
    assert completion["output"] == output_document_from_bytes(json.dumps(expected).encode())
    result = completion["result"]
    assert result["terminal_status"] == "accepted"
    assert result["resource_summary"]["provider_call_count"] == len(api.provider_frames) == 1
    summaries = [row for row in api.saved["agent_observations"] if row["kind"] == "state_snapshot"]
    assert len(summaries) == 1 and summaries[0]["data"]["value"] == summary
    assert api.saved["complete"] is complete
    assert (not complete) == ("capture_incomplete" in api.saved["failures"])
