"""Saved synthetic live probes exercise three linked model/tool continuations."""
import base64
import copy
import json
from pathlib import Path

import pytest

from lab_arena import contracts, reasoning_capture, training_capture, training_thinking
from lab_arena.training_export import _calls
from tests.lab_arena.training_steps_test import bind_selected_output, fixture, seal

PROBES = sorted((Path(__file__).parents[2] / "docs/evidence/full-thinking").glob("*price-bound.json"))


def historical_registry():
    """Replay immutable probes with their original frozen 8192-token policy."""
    supplemental = Path(__file__).parents[2] / "docs/evidence/full-thinking/chat-output-cap-32768-20261025.json"
    return reasoning_capture.validate_registry(json.loads(supplemental.read_text())["original_registry"])


class Objects:
    def __init__(self):
        self.values = {}

    def put(self, ref, raw):
        self.values[ref] = raw

    def get_bounded(self, ref, maximum):
        raw = self.values[ref]
        assert len(raw) <= maximum
        return raw


def _example(probe):
    base = fixture()
    final = probe["attempts"][-1]
    response = final["response"]
    selected = [row for row in response["openrouter_metadata"]["endpoints"]["available"]
                if row.get("selected") is True]
    assert len(selected) == 1
    registry = historical_registry()
    route = next(row for row in registry["approved_routes"] if row["model_id"] == probe["model"])
    _, annotation = reasoning_capture.apply("openrouter.chat", {"model": probe["model"]},
                                              policy=reasoning_capture.POLICY, registry=registry)
    objects = Objects()
    ledger, worker_receipts, acks = [], [], []
    for index, attempt in enumerate(probe["attempts"]):
        request_body = copy.deepcopy(attempt["request"])
        response_body = copy.deepcopy(attempt["response"])
        hashed = {"model": probe["model"], "turn": index}
        request_hash = contracts.document_hash(hashed)
        identity = "probe-call-%d" % index
        admitted = copy.deepcopy(request_body)
        # These saved diagnostic wires already contain the organizer's pin;
        # it was never part of the admitted miner request or synthetic hash.
        admitted.pop("provider", None)
        capture_doc = {"operation_id": "openrouter.chat", "request_hash": request_hash,
                       "hashed_parameters": hashed, "admitted_parameters": admitted,
                       "effective_parameters": {"model": probe["model"]},
                       "reasoning_capture": annotation,
                       "outbound": {"method": "POST", "host": "openrouter.ai",
                                    "path": "/api/v1/chat/completions", "body": request_body,
                                    "body_sha256": contracts.document_hash(request_body)}}
        pointer = training_capture.RequestCapture(objects)(capture_doc)
        assert pointer["status"] == "stored" and not pointer["redacted"]
        saved_request = training_capture.read_request(objects, pointer)
        assert saved_request["outbound"]["body"] == request_body
        wire = contracts.canonical_json(response_body).encode()
        wire_hash = contracts.hash_bytes(wire)
        observation = reasoning_capture.route_observation(response_body, annotation)
        assert observation["status"] == "matched"
        data = {"call_identity": identity, "status": 200,
                "body_sha256": wire_hash, "body_bytes": len(wire)}
        worker_receipts.append(dict(data, ordinal=index + 1, write_state="socket_written",
                                    operation_id="openrouter.chat", action_sequence=index))
        acks.append({"kind": "response_received", "id": "ack-%d" % index,
                     "privacy": {"redacted": False}, "data": data})
        assert saved_request["request_hash"] == request_hash
        context = {"run_id": "run", "round_id": base["round_id"],
                   "submission_id": base["submission_id"], "provider": "openrouter",
                   "operation_id": "openrouter.chat", "call_identity": identity,
                   "funding_source": "miner"}
        reserve = dict(context, entry_id=10 * index + 1, entry_kind="reservation",
                       amount_microusd=0,
                       entry_doc={"request_hash": request_hash, "action_sequence": index,
                                  "base_call_identity": identity, "provider_attempt": 1,
                                  "request_capture": pointer, "reasoning_capture": copy.deepcopy(annotation)})
        terminal = dict(context, entry_id=10 * index + 3, entry_kind="settlement",
                        amount_microusd=1,
                        terminal_response={"status": 200, "headers": {},
                                           "body_b64": base64.b64encode(wire).decode(),
                                           "call_succeeded": True,
                                           "reasoning_route_observation": observation})
        ledger.extend([reserve, dict(context, entry_id=10 * index + 2,
                                     entry_kind="dispatch", amount_microusd=0,
                                     entry_doc={}), terminal])
    calls, gaps = _calls(ledger, {"run"}, objects, {"run": 0})
    assert not gaps
    assert len(calls) == 3
    for index, call in enumerate(calls):
        assert call["response_sha256"] == contracts.document_hash(probe["attempts"][index]["response"])
    payload = {"response_receipts": worker_receipts, "agent_observations": acks}
    payload_hash = contracts.document_hash(payload)
    for index, call in enumerate(calls):
        call["response_provenance"] = {
            "agent_receipt": "reported_exact_body_received", "delivery_to_worker": "socket_written",
            "worker_received_receipt_entry_ids": [10 * index + 3],
            "worker_receipt_ref": {"run_id": "run", "ordinal": index + 1,
                                   "payload_sha256": payload_hash},
            "agent_receipt_refs": [{"run_id": "run", "id": "ack-%d" % index,
                                    "payload_sha256": payload_hash}]}
    base.update(provider_calls=calls, configuration={"reasoning_capture_policy": reasoning_capture.POLICY,
                                                      "reasoning_route_registry": registry},
                worker_observation_artifacts={"run": {"payload": payload,
                                                       "payload_sha256": payload_hash}},
                completeness={"complete": True, "incomplete_reasons": []})
    return bind_selected_output(base)


@pytest.mark.parametrize("probe", [json.loads(path.read_text()) for path in PROBES],
                         ids=lambda probe: probe["model"])
def test_live_three_turn_body_through_private_storage_and_strict_export(probe):
    example = _example(probe)
    splits, manifest = training_thinking.build([example])
    records = splits["train"] + splits["eval"]
    expected = sum(bool(attempt["response"]["choices"][0]["message"].get("reasoning"))
                   for attempt in probe["attempts"])
    assert expected > 0
    assert len(records) == expected
    for record in records:
        ref = record["source_refs"][0]
        original = probe["attempts"][int(ref["call_identity"].split("-")[-1])]["response"]
        text = original["choices"][0]["message"].get("reasoning")
        assert record["generated_thinking"][0]["text"] == text
        assert len(record["generated_thinking"]) == 1  # Alias and typed block are the same text.
        assert len(record["generated_thinking"][0]["source_paths"]) == 2
        assert ref["response_sha256"] == contracts.document_hash(original)
    assert manifest["coverage"]["trajectories"]["complete_full_thinking"] == 1
    assert manifest["coverage"]["returned_classes"].get("requested_empty", 0) == 3 - expected
