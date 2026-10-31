"""The frozen industry role retains its declared structured-output sequence."""
from copy import deepcopy
from dataclasses import replace
import json

import pytest

from qualification.scoring import industry_classification as industry
from lab_arena import broker as br, contracts, operations, scoring
from tests.lab_arena.test_industry_judge_provider_route import MODEL, fixture
from tests.lab_arena.test_lab_arena_broker import CONTEXT


def request(research=False):
    packet = industry.prepare(
        dict(industry="Software", subindustry="Support", product_service="Sells support software"),
        dict(name="Supplier", website="https://supplier.example", linkedin=""),
        [dict(source_id="s0", source_url="https://supplier.example/product",
              body="Supplier sells support software to business customers.")],
        evaluation_date='2026-10-24',
    )
    return industry.request_payload(packet, model=MODEL, research=research)


@pytest.mark.parametrize("research", [False, True])
@pytest.mark.parametrize("funding", ["host", "miner_key"])
@pytest.mark.parametrize("capture", [False, True])
def test_industry_wire_keeps_declared_fact_before_verdict_order(research, funding, capture, monkeypatch):
    broker, store, transport = fixture(roles={"industry_evidence_review": MODEL}, funding=funding)
    captures, cache_bodies = [], []
    if capture:
        broker._capture_request = lambda document, **kwargs: captures.append(document)
    material = br._judgment_cache_material
    def observe_material(*args, **kwargs):
        cache_bodies.append(kwargs["outbound_body"])
        return material(*args, **kwargs)
    monkeypatch.setattr(br, "_judgment_cache_material", observe_material)
    parameters = request(research)
    original = deepcopy(parameters)
    schema = parameters["response_format"]["json_schema"]["schema"]
    assert list(schema["properties"]) == schema["required"]
    result = broker.execute(replace(CONTEXT, kind="score"), operation_id="openrouter.chat",
                            parameters=parameters, action_sequence=0, timeout_ms=65000)
    assert result.status == 200
    sent = json.loads(transport.sent[0]["body"])
    actual = sent["response_format"]["json_schema"]["schema"]
    assert list(actual["properties"]) == schema["required"]
    assert actual == schema  # Only order changes, never constraints or fields.
    assert sent["messages"] == parameters["messages"]
    assert parameters == original
    assert result.call["funding_source"] == funding
    assert store.calls[result.call["call_identity"]]["actual"] == 200
    assert cache_bodies == [transport.sent[0]["body"]]
    if capture:
        assert captures[0]["outbound"]["body_sha256"] == contracts.hash_bytes(cache_bodies[0])
        assert list(captures[0]["outbound"]["body"]["response_format"]["json_schema"]
                    ["schema"]["properties"]) == schema["required"]


@pytest.mark.parametrize("roles,kind", [
    ({"company_fit_reverification": MODEL}, "score"),
    ({"industry_evidence_review": MODEL}, "execute"),
])
def test_older_policies_and_execution_keep_existing_wire_order(roles, kind):
    broker, _, transport = fixture(roles=roles)
    result = broker.execute(replace(CONTEXT, kind=kind), operation_id="openrouter.chat",
                            parameters=request(True), action_sequence=0, timeout_ms=65000)
    assert result.status == 200
    schema = json.loads(transport.sent[0]["body"])["response_format"]["json_schema"]["schema"]
    assert list(schema["properties"]) == sorted(schema["properties"])


def wires(parameters):
    instance, _, _ = fixture(roles={"industry_evidence_review": MODEL})
    route = br._openrouter_host_route(
        kind="score", operation_id="openrouter.chat", model=MODEL,
        pricing=instance._price_table["models"][MODEL], parameters=parameters,
        industry_review_model=MODEL,
    )
    return tuple(operations.build_outbound_request(
        "openrouter.chat", parameters, openrouter_provider_policy=route.provider_policy,
        industry_schema_order=enabled,
    ).body for enabled in (False, True))


@pytest.mark.parametrize("schema", [True, {},
    {"properties": {"z": {}, "a": {}}, "required": ["z"]},
    {"properties": {"z": {}, "a": {}}, "required": ["z", "z"]},
    {"properties": {"z": {}, "a": {}}, "required": [{}, "a"]},
])
def test_other_schema_shapes_keep_identical_bytes(schema):
    parameters = request()
    parameters["response_format"]["json_schema"]["schema"] = schema
    before, after = wires(parameters)
    assert after == before


def test_shared_model_with_another_schema_keeps_identical_bytes():
    parameters = request()
    parameters["response_format"]["json_schema"]["name"] = "another_verifier"
    before, after = wires(parameters)
    assert after == before


def test_wire_order_changes_judgment_cache_identity_without_changing_request_identity():
    parameters = request()
    before, after = wires(parameters)
    assert before != after and json.loads(before) == json.loads(after)
    policy = scoring.build_scorer_policy(scoring_adapter_version="qualification_integrity_v2")
    policy["judge_models"]["industry_evidence_review"] = MODEL
    context = replace(CONTEXT, kind="score", judgment_cache_scope={
        "round_id": CONTEXT.round_id, "evaluation_date": "2026-10-24",
        "scorer_image_digest": "sha256:" + "a" * 64,
        "scorer_image_reference": "registry/scorer@sha256:" + "a" * 64,
        "scorer_policy": policy,
    })
    keys = [br._judgment_cache_material(context, requested_operation_id="openrouter.chat",
        effective_operation_id="openrouter.chat", normalized_request=parameters,
        outbound_body=body) for body in (before, after)]
    assert all(keys) and keys[0][0] != keys[1][0]
    assert json.loads(keys[0][1])["request_hash"] == json.loads(keys[1][1])["request_hash"]
    assert json.loads(keys[1][1])["outbound_body_hash"] == contracts.hash_bytes(after)
