"""Host paragraph schema order survives the real canonical shim and broker."""

from __future__ import annotations

import asyncio
import copy
import json
from dataclasses import replace

import pytest
import httpx

from lab_arena import broker as br
from lab_arena import contracts, operations, scoring, shim, training_capture
from lab_arena.service import LocalObjectStore
from qualification.scoring import intent_details, verification_helpers
from tests.lab_arena.test_lab_arena_broker import (
    CONTEXT, FakeTransport, make_broker, price_table,
)
from tests import test_arena_intent_details_grounding as grounding


OLD_NAME = "arena_intent_details_review"
NEW_NAME = "arena_intent_details_review_evidence_first_v1"


def _parameters(*, old=False):
    response_format = copy.deepcopy(intent_details._RESPONSE_FORMAT)
    response_format["json_schema"]["name"] = OLD_NAME if old else NEW_NAME
    return {
        "model": "openai/gpt-4o-mini",
        "messages": [{"role": "system", "content": intent_details._SYSTEM},
                     {"role": "user", "content": "{}"}],
        "max_tokens": 800,
        "response_format": response_format,
    }


def _through_shim(parameters):
    return shim.decode_operation_frame(shim.build_operation_frame(
        "openrouter.chat", parameters, 45000,
    ))[1]


def _schema(body):
    return json.loads(body)["response_format"]["json_schema"]["schema"]


def test_new_scorer_explicitly_selects_versioned_evidence_first_protocol():
    assert intent_details._RESPONSE_FORMAT["json_schema"]["name"] == NEW_NAME
    assert contracts.document_hash(intent_details._RESPONSE_FORMAT) == (
        operations._INTENT_DETAILS_EVIDENCE_FIRST_FORMAT_HASH
    )


def test_host_schema_order_survives_canonical_shim_without_changing_json_values():
    parameters = _through_shim(_parameters())
    canonical = operations.build_outbound_request("openrouter.chat", parameters)
    ordered = operations.build_outbound_request(
        "openrouter.chat", parameters, host_score=True,
    )
    assert list(_schema(canonical.body)["properties"]) == sorted(
        _schema(canonical.body)["properties"]
    )
    assert list(_schema(ordered.body)["properties"]) == _schema(ordered.body)["required"]
    assert ordered.body != canonical.body
    assert json.loads(ordered.body) == json.loads(canonical.body)
    # Only the one properties object's order changes. The parsed document's
    # canonical encoding, and hence all existing hashes, remain unchanged.
    assert contracts.canonical_json(json.loads(ordered.body)).encode() == canonical.body
    def encode(value):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()
    assert ordered.body.replace(
        encode(_schema(ordered.body)["properties"]),
        encode(_schema(canonical.body)["properties"]), 1,
    ) == canonical.body
    assert contracts.document_hash(parameters) == contracts.document_hash(
        _through_shim(_parameters())
    )
    nested = _schema(ordered.body)["properties"]["unit_grounding"]["items"]["properties"]
    assert list(nested) == sorted(nested)


@pytest.mark.parametrize("change", ["old_name", "unknown_name", "schema", "required", "strict", "repair"])
def test_unrecognized_or_historical_formats_keep_exact_canonical_bytes(change):
    parameters = _parameters()
    declared = parameters["response_format"]["json_schema"]
    if change == "old_name":
        declared["name"] = OLD_NAME
    elif change == "unknown_name":
        declared["name"] = "miner_owned_schema"
    elif change == "schema":
        declared["schema"]["properties"]["connects_icp"] = {"type": "string"}
    elif change == "required":
        declared["schema"]["required"].reverse()
    elif change == "strict":
        declared["strict"] = False
    else:
        parameters["response_format"] = copy.deepcopy(intent_details._SEMANTIC_REPAIR_RESPONSE_FORMAT)
    parameters = _through_shim(parameters)
    ordinary = operations.build_outbound_request("openrouter.chat", parameters)
    host = operations.build_outbound_request("openrouter.chat", parameters, host_score=True)
    assert host.body == ordinary.body
    assert host.body == contracts.canonical_json(json.loads(host.body)).encode()


def test_miner_cannot_supply_the_host_score_switch_in_parameters_or_frame():
    parameters = {**_parameters(), "host_score": True}
    with pytest.raises(operations.OperationRequestError):
        _through_shim(parameters)
    frame = json.loads(shim.build_operation_frame("openrouter.chat", _parameters(), 45000))
    frame["host_score"] = True
    with pytest.raises(shim.OperationFrameError):
        shim.validate_operation_frame(frame)


@pytest.mark.parametrize("kind", ["score", "execute"])
@pytest.mark.parametrize("funding_source", ["host", "miner_key"])
@pytest.mark.parametrize("capture", [False, True, "broken"])
@pytest.mark.parametrize("old", [False, True])
def test_broker_uses_host_owned_run_kind_for_both_payers_and_capture_paths(kind, funding_source, capture, old, tmp_path):
    transport = FakeTransport([(200, {"usage": {"cost": "0.000001"}})])
    seen = []

    writer = training_capture.RequestCapture(LocalObjectStore(tmp_path))

    def capture_request(document, **kwargs):
        seen.append(document)
        if capture == "broken":
            raise RuntimeError("synthetic capture failure")
        return writer.prepare(document, **kwargs)

    provider, store, _ = make_broker(
        transport=transport, judge_models=("openai/gpt-4o-mini",),
        provider_funding_source_for=lambda *_: funding_source,
        capture_request=capture_request if capture else None,
    )
    result = provider.execute(
        replace(CONTEXT, kind=kind), operation_id="openrouter.chat",
        parameters=_through_shim(_parameters(old=old)), action_sequence=0, timeout_ms=45000,
    )
    assert result.status == 200
    assert result.call["funding_source"] == funding_source
    assert len(transport.sent) == 1
    body = transport.sent[0]["body"]
    schema = _schema(body)
    assert list(schema["properties"]) == (
        schema["required"] if kind == "score" and not old else sorted(schema["properties"])
    )
    if kind == "execute" or old:
        assert body == contracts.canonical_json(json.loads(body)).encode()
    assert [row["kind"] for row in store.calls.values()] == ["settlement"]
    if capture:
        assert len(seen) == 1
        assert seen[0]["outbound"]["body_sha256"] == contracts.hash_bytes(body)
        assert list(seen[0]["outbound"]["body"]["response_format"]["json_schema"]["schema"]["properties"]) == list(schema["properties"])


def test_protocol_name_and_wire_order_both_separate_judgment_cache_identity():
    policy = scoring.build_scorer_policy(
        judge_models={"intent_details_review": "openai/gpt-4o-mini"},
        scoring_adapter_version="qualification_integrity_v2", intent_details=True,
    )
    context = replace(CONTEXT, kind="score", round_id="arena-2026-09-02", judgment_cache_scope={
        "round_id": "arena-2026-09-02", "evaluation_date": "2026-09-02",
        "scorer_image_digest": "sha256:" + "a" * 64,
        "scorer_image_reference": "registry/scorer@sha256:" + "a" * 64,
        "scorer_policy": policy,
    })
    identities = []
    hashes = []
    for old, host_score in [(True, True), (False, False), (False, True)]:
        normalized = _through_shim(_parameters(old=old))
        outbound = operations.build_outbound_request("openrouter.chat", normalized, host_score=host_score)
        material = br._judgment_cache_material(
            context, requested_operation_id="openrouter.chat", effective_operation_id="openrouter.chat",
            normalized_request=normalized, outbound_body=outbound.body,
        )
        assert material is not None
        identities.append(material[0])
        hashes.append(contracts.document_hash(normalized))
    assert len(set(identities)) == 3
    assert hashes[0] != hashes[1] == hashes[2]


@pytest.mark.parametrize("connects_icp", [True, False])
def test_native_reviewer_client_shim_broker_and_parser_preserve_verdicts(monkeypatch, connects_icp):
    """Exercise the real client and parser; only the paid provider is a fake."""
    from tests.lab_arena.test_lab_arena_broker import FakeLedgerStore, HOST_KEYS

    inputs = grounding.inputs()
    document = intent_details.review_evidence(*inputs)
    checks = {name: True for name in intent_details._CHECKS}
    checks["connects_icp"] = connects_icp
    verdict = grounding._review_response(
        checks, [{"matched_icp_signal": i, "covered": True} for i in (0, 1)], document,
    )
    transport = FakeTransport([(200, {
        "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(verdict)}}],
        "usage": {"cost": "0.000001"},
    })])
    prices = price_table()
    prices["models"][intent_details.REVIEW_MODEL] = prices["models"]["openai/gpt-4o-mini"]
    provider = br.Broker(
        store=FakeLedgerStore(), key_for=lambda name: HOST_KEYS[name],
        transport=transport, price_table=prices, judge_models=(intent_details.REVIEW_MODEL,),
    )

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            pass

        async def post(self, url, *, json, **kwargs):
            result = provider.execute(
                replace(CONTEXT, kind="score"), operation_id="openrouter.chat",
                parameters=_through_shim(json), action_sequence=0, timeout_ms=45000,
            )
            return httpx.Response(result.status, content=result.body, request=httpx.Request("POST", url))

    monkeypatch.setattr(verification_helpers.httpx, "AsyncClient", Client)
    monkeypatch.setattr(verification_helpers, "OPENROUTER_API_KEY", "test-only-key")
    receipt = asyncio.run(intent_details.review_intent_details(*inputs))
    assert receipt["decision"] == ("match" if connects_icp else "mismatch")
    assert receipt["input_hash"] == contracts.document_hash(document)
    assert len(transport.sent) == 1
    wire = json.loads(transport.sent[0]["body"])
    assert wire["model"] == intent_details.REVIEW_MODEL
    assert wire["reasoning"] == {"effort": "low"}
    assert wire["messages"][0]["content"] == intent_details._SYSTEM
    assert json.loads(wire["messages"][1]["content"]) == document
    assert list(_schema(transport.sent[0]["body"])["properties"]) == list(intent_details._RESPONSE_FORMAT["json_schema"]["schema"]["properties"])
