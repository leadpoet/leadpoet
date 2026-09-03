"""Runner: claims, local slots, socket bridge, per-ICP execution, receipts (labarena.md 9, 10, 18.2)."""

from __future__ import annotations

import base64
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pytest
from bittensor_wallet import Keypair

from lab_arena import contracts, runner as rn, runtime, shim
from lab_arena.output import output_document_from_bytes

RUNNER = Keypair.create_from_uri("//Runner")
MINER = Keypair.create_from_uri("//Miner").ss58_address
ROUND = "arena-2026-09-02"
IMAGE = "sha256:" + "a" * 64


def sign(message: str) -> str:
    return RUNNER.sign(message.encode("utf-8")).hex()


def verify(hotkey: str, signature: str, message: str) -> bool:
    raw = bytes.fromhex(signature[2:] if signature.startswith("0x") else signature)
    return bool(Keypair(ss58_address=hotkey).verify(message.encode("utf-8"), raw))


def valid_company(index: int) -> Dict[str, Any]:
    return {
        "company_name": "Co %d" % index,
        "company_website": "https://co%d.example.com" % index,
        "industry": "Software",
        "employee_count": "51-200",
        "country": "United States",
        "intent_signals": [{"source": "news", "description": "Raised a round", "url": "https://news.example.com/%d" % index, "date": "2026-08-01", "snippet": "Co raised money", "matched_icp_signal": 0}],
    }


class FakeApi:
    """In-process stand-in for the service's runner endpoints."""

    def __init__(self, leases: List[Dict[str, Any]], *, broker_documents=None):
        self.leases = list(leases)
        self.claims: List[Dict[str, Any]] = []
        self.provider_frames: List[Dict[str, Any]] = []
        self.events: Dict[str, List[Dict[str, Any]]] = {}
        self.completions: List[Dict[str, Any]] = []
        self.cursor: Dict[str, Dict[str, Any]] = {}
        self.broker_documents = list(broker_documents or [])
        self.lock = threading.Lock()

    def claim(self, envelope):
        self.claims.append(envelope)
        assert envelope["scope"] == contracts.SCOPE_CLAIM and verify(envelope["hotkey"], envelope["signature"], contracts.signed_request_message(envelope))
        if not self.leases:
            return {"status": "no_pending"}
        lease = self.leases.pop(0)
        self.cursor[lease["run_id"]] = {"event_cursor": 0, "event_head_hash": ""}
        return lease

    def provider(self, run_id, lease_token, frame):
        self.provider_frames.append(dict(frame))
        with self.lock:
            state = self.cursor[run_id]
            payload = {"call_identity": contracts.document_hash(["call", frame["action_sequence"]]), "operation_id": frame["operation_id"], "reserved_microusd": 5000, "actual_microusd": 5000, "outcome": "settled", "status": 200, "request_hash": contracts.document_hash(frame["parameters"]), "response_hash": contracts.document_hash("resp")}
            event = contracts.build_private_event(event_type="provider_call", sequence=state["event_cursor"], prev_hash=state["event_head_hash"], timestamp="2026-09-02T01:00:00Z", payload=payload)
            self.events.setdefault(run_id, []).append(event)
            state.update(event_cursor=event["sequence"] + 1, event_head_hash=event["event_hash"])
            call = dict(payload, event_cursor=state["event_cursor"], event_head_hash=state["event_head_hash"])
        body = json.dumps({"results": [{"url": "https://co1.example.com"}]}).encode()
        return {"status": 200, "headers": {"content-type": "application/json", "content-length": str(len(body))}, "body_b64": base64.b64encode(body).decode(), "call": call}

    def append_events(self, run_id, lease_token, events):
        with self.lock:
            state = self.cursor[run_id]
            for event in events:
                assert event["sequence"] == state["event_cursor"] and event["prev_hash"] == state["event_head_hash"]
                self.events.setdefault(run_id, []).append(event)
                state.update(event_cursor=event["sequence"] + 1, event_head_hash=event["event_hash"])
            return {"status": "appended", "event_cursor": state["event_cursor"], "event_head_hash": state["event_head_hash"]}

    def complete(self, envelope):
        self.completions.append(envelope)
        return {"status": "accepted"}


def lease(run_id="r1", position=0):
    return {
        "status": "leased", "run_id": run_id, "assignment_id": "%s:s1:1:%d" % (ROUND, position), "submission_id": "s1", "miner_hotkey": MINER,
        "image_digest": IMAGE, "stage": 1, "icp_position": position, "icp_hash": contracts.document_hash({"icp": position}), "attempt": 1,
        "lease_generation": 1, "stage_generation": 1, "lease_expires_at": "2026-09-02T01:07:00+00:00",
        "event_cursor": 0, "event_head_hash": "", "lease_token": "tok-" + run_id, "icp": {"icp_id": "arena:x", "prompt": "p", "max_companies": 5},
    }


class BridgingRuntime:
    """A fake sandbox that behaves like a model: talks to the worker socket, writes output."""

    def __init__(self, *, output: Any = None, timed_out=False, exit_code=0, calls=1):
        self.output = output
        self.timed_out = timed_out
        self.exit_code = exit_code
        self.calls = calls
        self.specs: List[runtime.SandboxSpec] = []

    def run_icp(self, spec, **_):
        self.specs.append(spec)
        os.environ[shim.WORKER_SOCKET_ENV] = str(spec.socket_path)
        try:
            for _ in range(self.calls):
                status, headers, body = shim.dispatch("deepline.execute", {"tool": "exa_search", "payload": {"query": "fintech"}}, 5000)
                assert status == 200 and json.loads(body)["results"]
        finally:
            os.environ.pop(shim.WORKER_SOCKET_ENV, None)
        if self.output is not None:
            spec.output_path.write_bytes(self.output if isinstance(self.output, bytes) else json.dumps(self.output).encode())
        return runtime.fake_result(exit_code=self.exit_code, timed_out=self.timed_out, output_bytes=runtime.read_output(spec), stdout=b"model log line\n", stderr=b"")


def make_config(tmp_path, api, sandbox_runtime, *, parallel=1):
    cache = rn.ImageCache(tmp_path / "images", lambda digest, target: (target / "rootfs").mkdir())
    return rn.RunnerConfig(
        round_id=ROUND, identity=rn.RunnerIdentity(hotkey=RUNNER.ss58_address, sign=sign), api=api, sandbox_runtime=sandbox_runtime,
        image_cache=cache, worker_release_hash=contracts.document_hash("worker"), work_dir=tmp_path / "work", max_parallel_runs=parallel,
        evaluation_date="2026-09-02", clock=lambda: datetime(2026, 9, 2, 1, 0, tzinfo=timezone.utc),
    )


def test_accepted_run_bridges_provider_calls_and_signs_a_bound_receipt(tmp_path):
    api = FakeApi([lease()])
    sandbox = BridgingRuntime(output={"companies": [valid_company(1), valid_company(2)]}, calls=2)
    (tmp_path / "work").mkdir()
    runner_ = rn.Runner(make_config(tmp_path, api, sandbox))
    assert runner_.run_once() == 1
    envelope = api.completions[0]
    assert envelope["scope"] == contracts.SCOPE_COMPLETE and verify(envelope["hotkey"], envelope["signature"], contracts.signed_request_message(envelope))
    receipt = envelope["body"]["receipt"]
    receipt_doc = {k: v for k, v in receipt.items() if k != "run_id"}
    validated = contracts.validate_icp_receipt(receipt_doc, verify_signature=verify)
    assert validated["terminal_status"] == "accepted" and validated["runner_hotkey"] == RUNNER.ss58_address
    assert validated["resource_summary"]["provider_call_count"] == 2 and len(envelope["body"]["calls"]) == 2
    frames = api.provider_frames
    assert [f["action_sequence"] for f in frames] == [0, 1] and all(set(f) == {"operation_id", "parameters", "timeout_ms", "action_sequence"} for f in frames)
    events = api.events["r1"]
    assert [e["event_type"] for e in events] == ["process_started", "provider_call", "provider_call", "stdout", "stderr", "process_finished", "output_validated"]
    assert contracts.private_event_root(events) == validated["private_event_root"]
    assert validated["provider_call_root"] == rn.provider_call_root(envelope["body"]["calls"])
    assert validated["cost_root"] == rn.cost_root(envelope["body"]["calls"])
    output = envelope["body"]["output"]
    assert validated["output_hash"] == contracts.document_hash(output) and len(output["companies"]) == 2
    spec = sandbox.specs[0]
    assert spec.rootfs_path == tmp_path / "images" / IMAGE.replace(":", "-") and spec.wall_clock_seconds == contracts.ICP_WALL_CLOCK_SECONDS
    assert not spec.input_dir.exists()  # run directory cleaned
    assert runner_.abandoned == 0


@pytest.mark.parametrize("kind,expected", [
    ("timeout", "model_timeout"),
    ("invalid_json", "invalid_output"),
    ("unknown_field", "invalid_output"),
    ("no_output", "model_error"),
])
def test_model_failures_map_to_terminal_causes_with_no_output_hash(tmp_path, kind, expected):
    api = FakeApi([lease()])
    if kind == "timeout":
        sandbox = BridgingRuntime(output=None, timed_out=True, exit_code=None)
    elif kind == "invalid_json":
        sandbox = BridgingRuntime(output=b"{not json")
    elif kind == "unknown_field":
        sandbox = BridgingRuntime(output={"companies": [dict(valid_company(1), extra_field="x")]})
    else:
        sandbox = BridgingRuntime(output=None, exit_code=3)
    (tmp_path / "work").mkdir()
    rn.Runner(make_config(tmp_path, api, sandbox)).run_once()
    receipt = api.completions[0]["body"]["receipt"]
    assert receipt["terminal_status"] == expected
    assert api.completions[0]["body"]["output"] is None
    kinds = [e["event_type"] for e in api.events["r1"]]
    assert "output_validated" not in kinds and ("output_rejected" in kinds or kind == "timeout")


def test_local_slots_bound_claims_and_images_export_once(tmp_path):
    leases = [lease("r%d" % i, i) for i in range(5)]
    api = FakeApi(leases)
    exports = []
    cache = rn.ImageCache(tmp_path / "images", lambda digest, target: exports.append(digest) or (target / "rootfs").mkdir())
    (tmp_path / "work").mkdir()
    config = make_config(tmp_path, api, BridgingRuntime(output={"companies": [valid_company(1)]}), parallel=3)
    config.image_cache = cache
    runner_ = rn.Runner(config)
    assert runner_.run_once() == 3  # three local slots, then the loop waits for completions
    assert exports == [IMAGE]
    assert runner_.run_once() == 2 and len(api.completions) == 5
    assert runner_.run_once() == 0
    assert all(c["body"]["declared_parallelism"] == 3 for c in api.claims)


def test_frame_validation_rejects_identity_fields_and_unknown_operations(tmp_path):
    api = FakeApi([])
    state = rn.RunState(lease=lease(), lease_token="tok", event_cursor=0, event_head_hash="")
    api.cursor["r1"] = {"event_cursor": 0, "event_head_hash": ""}
    server = rn.WorkerSocketServer(tmp_path / "worker.sock", api, state, rn.EventWriter(api, state, lambda: datetime(2026, 9, 2, tzinfo=timezone.utc)))
    good = shim.build_operation_frame("deepline.execute", {"tool": "exa_search", "payload": {"query": "x"}}, 1000)
    response = json.loads(server.handle_frame(good))
    assert response["status"] == 200 and state.action_sequence == 1 and state.event_cursor == 1
    for hostile in (
        json.dumps({"schema_version": shim.OPERATION_FRAME_SCHEMA_VERSION, "operation_id": "deepline.execute", "parameters": {"tool": "exa_search", "payload": {"query": "x"}}, "timeout_ms": 1000, "lease_token": "steal"}),
        json.dumps({"schema_version": shim.OPERATION_FRAME_SCHEMA_VERSION, "operation_id": "deepline.play", "parameters": {}, "timeout_ms": 1000}),
        json.dumps({"schema_version": shim.OPERATION_FRAME_SCHEMA_VERSION, "operation_id": "deepline.execute", "parameters": {"tool": "exa_search", "payload": {"query": "x"}, "headers": {"x-api-key": "k"}}, "timeout_ms": 1000}),
        b"not json",
    ):
        error = json.loads(server.handle_frame(hostile.encode() if isinstance(hostile, str) else hostile))
        assert set(error) == {"error"}
    assert state.action_sequence == 1 and len(api.provider_frames) == 1


def test_event_append_failure_fails_the_run_closed(tmp_path):
    api = FakeApi([lease()])

    def broken(run_id, lease_token, events):
        return {"status": "stale"}

    api.append_events = broken
    (tmp_path / "work").mkdir()
    runner_ = rn.Runner(make_config(tmp_path, api, BridgingRuntime(output={"companies": [valid_company(1)]})))
    runner_.run_once()
    assert runner_.abandoned == 1 and api.completions == []


def test_worker_release_identity_and_parallelism_env():
    identity = rn.worker_release_identity(repository_commit="a" * 40, runtime_lock_hash=contracts.document_hash("lock"))
    assert identity["shim_hash"] == shim.shim_source_hash() and identity["worker_release_hash"].startswith("sha256:")
    assert rn.max_parallel_runs_from_environment({}) == 1 and rn.max_parallel_runs_from_environment({rn.MAX_PARALLEL_ENV: "4"}) == 4
    with pytest.raises(rn.RunnerError):
        rn.max_parallel_runs_from_environment({rn.MAX_PARALLEL_ENV: "0"})
    with pytest.raises(rn.RunnerError):
        rn.HttpArenaApiClient("http://arena.example.com")
    document = output_document_from_bytes(json.dumps([valid_company(1)]).encode())
    assert document["schema_version"] == contracts.OUTPUT_DOCUMENT_SCHEMA_VERSION and len(document["companies"]) == 1


def test_runner_refuses_a_round_whose_release_identity_differs():
    from lab_arena import operations

    release = rn.worker_release_identity(repository_commit="a" * 40, runtime_lock_hash=contracts.document_hash("lock"))
    configuration = {"release": {"worker_release_hash": release["worker_release_hash"], "runsc_lock_hash": contracts.document_hash("lock"), "shim_hash": shim.shim_source_hash()}, "operation_table_hash": operations.OPERATION_TABLE_HASH}
    assert rn.verify_release_against_round(configuration, worker_release_hash=release["worker_release_hash"], runtime_lock_hash=contracts.document_hash("lock"))["shim_hash"] == shim.shim_source_hash()
    for broken in (
        {**configuration, "release": {**configuration["release"], "worker_release_hash": contracts.document_hash("other")}},
        {**configuration, "release": {**configuration["release"], "runsc_lock_hash": contracts.document_hash("other")}},
        {**configuration, "release": {**configuration["release"], "shim_hash": contracts.document_hash("other")}},
        {**configuration, "operation_table_hash": contracts.document_hash("other")},
        {"operation_table_hash": operations.OPERATION_TABLE_HASH},
    ):
        with pytest.raises(rn.RunnerError):
            rn.verify_release_against_round(broken, worker_release_hash=release["worker_release_hash"], runtime_lock_hash=contracts.document_hash("lock"))


# ---------------------------------------------------------------------------
# Scoring assignments: the validator runs the pinned judge image on one output
# ---------------------------------------------------------------------------

SCORER_IMAGE = "sha256:" + "5" * 64


def scoring_lease(run_id="r9", position=3):
    from lab_arena import scoring

    base = lease(run_id, position)
    base.update({
        "assignment_id": "%s:s1:1:%d:score" % (ROUND, position), "kind": "score", "work_item_id": contracts.document_hash("item-%d" % position),
        "scored_run_id": "r1", "image_digest": SCORER_IMAGE, "scored_output": {"companies": [valid_company(1), valid_company(2)]}, "scorer_policy": scoring.build_scorer_policy(),
    })
    return base


def test_scoring_lease_runs_the_judge_image_in_trusted_mode_and_signs_a_breakdown_receipt(tmp_path):
    from lab_arena import scoring

    work_item_id = contracts.document_hash("item-3")
    breakdowns = [{"final_score": 71.0, "failure_reason": ""}, {"final_score": 44.5, "failure_reason": ""}]
    output = scoring.build_scoring_output(work_item_id, breakdowns)
    api = FakeApi([scoring_lease()])
    sandbox = BridgingRuntime(output=output, calls=1)
    (tmp_path / "work").mkdir()
    runner_ = rn.Runner(make_config(tmp_path, api, sandbox))
    assert runner_.run_once() == 1 and runner_.abandoned == 0
    envelope = api.completions[0]
    receipt = {k: v for k, v in envelope["body"]["receipt"].items() if k != "run_id"}
    validated = contracts.validate_icp_receipt(receipt, verify_signature=verify)
    assert validated["kind"] == "score" and validated["terminal_status"] == "accepted" and validated["image_digest"] == SCORER_IMAGE
    assert envelope["body"]["output"] == output and validated["output_hash"] == contracts.document_hash(output)
    spec = sandbox.specs[0]
    assert spec.entry_file == rn.SCORER_ENTRY_FILE and spec.extra_environment[shim.TRUSTED_SCORER_ENV] == "1"
    assert spec.rootfs_path == tmp_path / "images" / SCORER_IMAGE.replace(":", "-")
    # The judge sandbox read a scoring input, not an ICP input.
    events = api.events["r9"]
    assert [e["event_type"] for e in events][-1] == "output_validated" and events[-1]["payload"]["breakdown_count"] == 2
    assert validated["provider_call_root"] == rn.provider_call_root(envelope["body"]["calls"])


@pytest.mark.parametrize("output, timed_out, expected", [
    ({"schema_version": "leadpoet.lab_arena.scoring_output.v1", "work_item_id": contracts.document_hash("item-3"), "failure": "judge_key_refused"}, False, "judge_key_refused"),
    ({"schema_version": "leadpoet.lab_arena.scoring_output.v1", "work_item_id": contracts.document_hash("item-3"), "failure": "judge_error"}, False, "judge_error"),
    (None, True, "judge_timeout"),
    (b"not json", False, "judge_error"),
    ({"companies": [valid_company(1)]}, False, "judge_error"),
])
def test_judge_failures_map_to_judge_causes_with_no_output(tmp_path, output, timed_out, expected):
    api = FakeApi([scoring_lease()])
    sandbox = BridgingRuntime(output=output, timed_out=timed_out, calls=0)
    (tmp_path / "work").mkdir()
    runner_ = rn.Runner(make_config(tmp_path, api, sandbox))
    assert runner_.run_once() == 1 and runner_.abandoned == 0
    receipt = {k: v for k, v in api.completions[0]["body"]["receipt"].items() if k != "run_id"}
    validated = contracts.validate_icp_receipt(receipt, verify_signature=verify)
    assert validated["terminal_status"] == expected and validated["kind"] == "score"
    assert api.completions[0]["body"].get("output") in (None, {}) and validated["output_hash"] == contracts.document_hash(None)


class RefusingApi(FakeApi):
    """The broker refuses the scored miner's key or quota: a 402 body, and the call summary keeps the code."""

    def __init__(self, leases, *, code="budget_refused"):
        super().__init__(leases)
        self.code = code

    def provider(self, run_id, lease_token, frame):
        self.provider_frames.append(dict(frame))
        with self.lock:
            state = self.cursor[run_id]
            payload = {"call_identity": contracts.document_hash(["call", frame["action_sequence"]]), "operation_id": frame["operation_id"], "reserved_microusd": 0, "actual_microusd": 0, "outcome": "refused", "error_code": self.code}
            event = contracts.build_private_event(event_type="provider_call", sequence=state["event_cursor"], prev_hash=state["event_head_hash"], timestamp="2026-09-02T01:00:00Z", payload=payload)
            self.events.setdefault(run_id, []).append(event)
            state.update(event_cursor=event["sequence"] + 1, event_head_hash=event["event_hash"])
            call = dict(payload, event_cursor=state["event_cursor"], event_head_hash=state["event_head_hash"])
        body = b'{"error":{"code":"%s"}}' % self.code.encode()
        return {"status": 402, "headers": {"content-type": "application/json", "content-length": str(len(body))}, "body_b64": base64.b64encode(body).decode(), "call": call}


class RefusedJudgeRuntime(BridgingRuntime):
    """The real judge folds a refused provider call into its own failure document."""

    def run_icp(self, spec, **_):
        self.specs.append(spec)
        os.environ[shim.WORKER_SOCKET_ENV] = str(spec.socket_path)
        try:
            status, _headers, _body = shim.dispatch("openrouter.chat", {"model": "openai/gpt-4o-mini", "messages": [{"role": "user", "content": "judge"}]}, 5000)
        finally:
            os.environ.pop(shim.WORKER_SOCKET_ENV, None)
        assert status == 402
        spec.output_path.write_bytes(json.dumps(self.output).encode())
        return runtime.fake_result(exit_code=0, output_bytes=runtime.read_output(spec))


@pytest.mark.parametrize("code, expected", [("budget_refused", "judge_key_refused"), ("budget_exhausted", "judge_key_refused"), ("call_refused", "judge_key_refused"), ("provider_unavailable", "judge_error")])
def test_a_refused_key_or_quota_during_judging_is_the_scored_miners_outcome(tmp_path, code, expected):
    """The worker reclassifies a judge error from the refusal it recorded; any other broker error stays a judge error."""

    from lab_arena import scoring

    failure = scoring.build_scoring_failure(contracts.document_hash("item-3"), "judge_error", detail="provider error")
    api = RefusingApi([scoring_lease()], code=code)
    (tmp_path / "work").mkdir()
    runner_ = rn.Runner(make_config(tmp_path, api, RefusedJudgeRuntime(output=failure)))
    assert runner_.run_once() == 1 and runner_.abandoned == 0
    receipt = {k: v for k, v in api.completions[0]["body"]["receipt"].items() if k != "run_id"}
    validated = contracts.validate_icp_receipt(receipt, verify_signature=verify)
    assert validated["terminal_status"] == expected and validated["kind"] == "score"
    reasons = [e["payload"].get("reason") for e in api.events["r9"] if e["event_type"] == "output_rejected"]
    assert ("judge_key_refused:" + code in reasons) == (expected == "judge_key_refused")
    assert api.completions[0]["body"]["calls"][0]["error_code"] == code  # the refusal travels with the receipt's calls
