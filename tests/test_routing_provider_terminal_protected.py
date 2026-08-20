"""Adversarial tests for the protected routing provider terminal normalizer."""

from __future__ import annotations

import base64
from copy import deepcopy
from dataclasses import replace
import json
from typing import Any, Mapping

import pytest

from gateway.research_lab.routing_execution_authorization import (
    RoutingProviderCallAuthorizationV2,
    execute_routing_provider_call_authorization_v2,
    routing_provider_dispatch_job_id_v2,
)
from gateway.research_lab.routing_provider_bindings import (
    DEEPLINE_COMPILER_FAMILY,
    ROUTING_BINDING_CATALOG_SCHEMA,
    ROUTING_UNIT_DATASET_SCHEMA,
    ReviewedDeeplineActionCompiler,
    RoutingBindingManifest,
    VerifiedRoutingBindingCatalog,
    VerifiedRoutingUnitDataset,
)
from gateway.research_lab.routing_provider_terminal_protected import (
    ProtectedRoutingProviderTerminalError,
    build_routing_budget_reservation_v3,
    execute_protected_routing_provider_terminal_v2,
    routing_budget_reservation_proof_v3,
    validate_routing_budget_reservation_result_v3,
    validate_routing_budget_reservation_v3,
)
from gateway.tee.provider_broker_v2 import PROVIDER_BROKER_SCHEMA_VERSION
from gateway.tee.provider_evidence_v2 import create_signed_provider_evidence_record
from leadpoet_canonical.attested_v2 import (
    EMPTY_ARTIFACT_ROOT,
    EMPTY_HOST_OPERATION_ROOT,
    EMPTY_TRANSPORT_ROOT,
    build_boot_identity_body,
    build_execution_receipt_body,
    build_transport_attempt,
    create_boot_identity,
    create_signed_execution_receipt,
    sha256_bytes,
)
from research_lab.canonical import sha256_json
from research_lab.eval.provider_evidence_cache import canonical_request_fingerprint
from research_lab.routing_experiments import ProviderBindingIdentity


def _h(char: str) -> str:
    return "sha256:" + char * 64


def _keys():
    cryptography = pytest.importorskip("cryptography")
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    key = Ed25519PrivateKey.generate()
    pubkey = key.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    ).hex()
    return key, pubkey


def _binding() -> ProviderBindingIdentity:
    return ProviderBindingIdentity(
        binding_id="binding-deepline-bloomberry-jobs-v2",
        provider_id="bloomberry_jobs",
        tool_id="intent.source_add.bloomberry_jobs",
        source_lineage_id="deepline.bloomberry.jobs",
        adapter_version="v2",
        manifest_hash=_h("1"),
        capability_hash=_h("2"),
        execution_contract_hash=_h("3"),
        cost_model_hash=_h("4"),
    )


def _compiler() -> tuple[ReviewedDeeplineActionCompiler, ProviderBindingIdentity]:
    binding = _binding()
    unit = {
        "company_domain": "example.com",
        "job_keyword": "machine learning",
        "countries": "US;CA",
        "minimum_date": "2026-08-01",
        "maximum_date": "2026-08-31",
    }
    unit_set_hash = sha256_json(
        {
            "schema_version": ROUTING_UNIT_DATASET_SCHEMA,
            "units": [{"unit_ref": "company-1", "input": unit}],
        }
    )
    manifest = RoutingBindingManifest(
        binding=binding,
        compiler_family=DEEPLINE_COMPILER_FAMILY,
        transport_id="deepline",
        action_id="bloomberry_search_job_postings",
        input_projection={
            "domain": "company_domain",
            "keyword": "job_keyword",
            "countries": "countries",
            "minimum_date": "minimum_date",
            "maximum_date": "maximum_date",
        },
        input_constants={},
        model_binding_requirements_hash=_h("0"),
        output_contract_hash=_h("5"),
        evidence_contract_hash=_h("6"),
        retry_policy_hash=_h("7"),
        max_results=1,
        timeout_ms=5_000,
        credit_ceiling_microunits=90_000,
    )
    catalog = VerifiedRoutingBindingCatalog(
        manifest_uri="s3://lab-routing/bindings/catalog-001.json",
        manifest_hash=_h("8"),
        signature_ref="s3://lab-routing/signatures/catalog-001.sig",
        signing_key_id="routing-kms-key",
        catalog_version="catalog-001",
        bindings={manifest.identity_key(): manifest},
    )
    dataset = VerifiedRoutingUnitDataset(
        manifest_uri="s3://lab-routing/units/dataset-001.json",
        manifest_hash=_h("9"),
        signature_ref="s3://lab-routing/signatures/dataset-001.sig",
        signing_key_id="routing-kms-key",
        unit_set_hash=unit_set_hash,
        provenance_hash=_h("a"),
        units={"company-1": unit},
    )
    return ReviewedDeeplineActionCompiler(
        binding_catalog=catalog, unit_dataset=dataset
    ), binding


def _authorization_and_request(compiler, binding):
    manifest = compiler.binding_catalog.resolve(binding)
    prepared = compiler.prepare(
        binding=binding,
        unit_ref="company-1",
        authorization_credit_microunits=manifest.credit_ceiling_microunits,
        authorization_timeout_ms=manifest.timeout_ms,
        expected_model_binding_requirements_hash=_h("0"),
    )
    core_fingerprint = _h("a")
    grant = RoutingProviderCallAuthorizationV2(
        admission_job_id=_h("b"),
        experiment_hash=_h("b"),
        experiment_id="experiment-1",
        purpose="research_lab.routing_provider_evidence.v2",
        envelope_hash=_h("e"),
        admission_bundle_hash=_h("f"),
        protected_release_hash=_h("0"),
        protected_boot_identity_hash=_h("1"),
        variant_id="candidate",
        stage="intent_evidence",
        artifact_lineage_hash=_h("2"),
        pointer_document_hash=_h("3"),
        model_artifact_hash=_h("4"),
        manifest_hash=_h("5"),
        image_digest="registry.example/router@sha256:" + "6" * 64,
        commit_sha="7" * 40,
        build_id="build-1",
        routing_contract_hash=_h("7"),
        routing_catalog_hash=_h("8"),
        routing_policy_hash=_h("9"),
        feature_schema_hash=_h("a"),
        verifier_contract_hash=_h("b"),
        binding=binding,
        transport_id=prepared.transport_id,
        binding_catalog_manifest_hash=prepared.binding_catalog_manifest_hash,
        binding_catalog_version=prepared.binding_catalog_version,
        action_id=prepared.action_id,
        unit_ref=prepared.unit_ref,
        unit_input_hash=prepared.unit_input_hash,
        unit_dataset_manifest_hash=prepared.unit_dataset_manifest_hash,
        unit_set_hash=prepared.unit_set_hash,
        model_binding_observation_receipt_hash=_h("c"),
        attempt=0,
        core_request_fingerprint=core_fingerprint,
        request_body_hash=prepared.request_body_hash,
        retry_policy_hash=prepared.retry_policy_hash,
        credit_cap_microunits=prepared.credit_ceiling_microunits,
        timeout_ms=prepared.timeout_ms,
        claim_key=_h("d"),
        claim_generation=1,
        claim_fence_hash=_h("e"),
    )
    # Replace the compact fields in the broker request with the actual signed
    # authorization identity.  The compiler only owns the transport shape.
    expected = execute_routing_provider_call_authorization_v2(
        grant.to_dict(), authorization_job_id="routing-authorization-job"
    )
    key, pubkey = _keys()
    auth_boot = create_boot_identity(
        body=build_boot_identity_body(
            role="gateway_scoring",
            physical_role="gateway_scoring",
            commit_sha="7" * 40,
            pcr0="8" * 96,
            build_manifest_hash=_h("9"),
            dependency_lock_hash=_h("a"),
            config_hash=_h("b"),
            boot_nonce="d" * 32,
            signing_pubkey=pubkey,
            transport_pubkey="e" * 64,
            transport_certificate_hash=_h("f"),
            attestation_user_data_hash=_h("0"),
            issued_at="2026-08-19T12:00:00Z",
        ),
        attestation_document_b64=base64.b64encode(b"test-attestation").decode(),
    )
    receipt = create_signed_execution_receipt(
        body=build_execution_receipt_body(
            role="gateway_scoring",
            purpose=grant.purpose,
            job_id="routing-authorization-job",
            epoch_id=1,
            sequence=1,
            commit_sha="7" * 40,
            pcr0="8" * 96,
            build_manifest_hash=_h("9"),
            dependency_lock_hash=_h("a"),
            config_hash=_h("b"),
            boot_identity_hash=auth_boot["boot_identity_hash"],
            input_root=grant.authorization_hash(),
            output_root=expected["output_root"],
            transport_root_hash=EMPTY_TRANSPORT_ROOT,
            host_operation_root_hash=EMPTY_HOST_OPERATION_ROOT,
            artifact_root=EMPTY_ARTIFACT_ROOT,
            parent_receipt_hashes=(),
            status="succeeded",
            failure_code=None,
            issued_at="2026-08-19T12:00:00Z",
        ),
        enclave_pubkey=pubkey,
        sign_digest=key.sign,
    )
    proof = {
        "authorization_hash": grant.authorization_hash(),
        "authorization_request_hash": grant.authorization_hash(),
        "authorization_proof_hash": receipt["receipt_hash"],
        "request_body_hash": grant.request_body_hash,
        "action_id": grant.action_id,
        "credit_cap_microunits": grant.credit_cap_microunits,
        "timeout_ms": grant.timeout_ms,
        "authorization": grant.to_dict(),
        "authorization_result": expected,
        "authorization_receipt": receipt,
    }
    request = compiler.broker_request(
        prepared=prepared,
        experiment_hash=_h("b"),
        dispatch_job_id=routing_provider_dispatch_job_id_v2(proof),
        variant_id="candidate",
        attempt_number=0,
        core_request_fingerprint=core_fingerprint,
        authorization_hash=proof["authorization_hash"],
        authorization_proof_hash=proof["authorization_proof_hash"],
    )
    request = dict(request)
    request["routing_authorization"] = proof
    request["job_id"] = routing_provider_dispatch_job_id_v2(proof)
    request["provider_id"] = grant.transport_id
    return prepared, request, proof, key, pubkey, auth_boot


def _call_fixture(
    response: Mapping[str, Any],
    *,
    http_status: int = 200,
    compiler_binding: tuple[ReviewedDeeplineActionCompiler, ProviderBindingIdentity]
    | None = None,
):
    compiler, binding = compiler_binding or _compiler()
    prepared, request, proof, key, pubkey, auth_boot = _authorization_and_request(
        compiler, binding
    )
    response_body = json.dumps(response, sort_keys=True, separators=(",", ":")).encode()
    request_body = base64.b64decode(request["body_b64"], validate=True)
    attempt = build_transport_attempt(
        request_id="0123456789abcdef0123456789abcdef",
        logical_operation_id=request["logical_operation_id"],
        job_id=request["job_id"],
        purpose=request["purpose"],
        provider_id=request["provider_id"],
        attempt_number=request["attempt_number"],
        method="POST",
        destination_host="code.deepline.com",
        destination_port=443,
        path_hash=_h("1"),
        nonsecret_headers_hash=_h("2"),
        body_hash=sha256_bytes(request_body),
        credential_ref_hash=_h("3"),
        retry_policy_hash=request["retry_policy_hash"],
        timeout_ms=request["timeout_ms"],
        started_at="2026-08-19T12:00:00Z",
        terminal_status="authenticated_response",
        http_status=http_status,
        response_hash=sha256_bytes(response_body),
        request_artifact_hash=_h("4"),
        response_artifact_hash=_h("5"),
        tls_peer_chain_hash=_h("6"),
        tls_protocol="TLSv1.3",
        failure_code=None,
        completed_at="2026-08-19T12:00:01Z",
    )
    boot = {"boot_identity_hash": _h("7"), "signing_pubkey": pubkey}
    record = create_signed_provider_evidence_record(
        body={
            "coordinator_boot_identity_hash": boot["boot_identity_hash"],
            "request_hash": attempt["request_hash"],
            # Provider evidence identifies the canonical transport request;
            # the authorization's core fingerprint identifies the router
            # retry identity and is intentionally a different namespace.
            "request_fingerprint": canonical_request_fingerprint(
                "POST", request["url"], request_body
            ),
            "evidence": "recorded",
            "status": http_status,
            "body_hash": sha256_bytes(response_body),
            "encrypted_request_artifact_id": _h("8"),
            "encrypted_response_artifact_id": _h("9"),
            "transport_attempt_hash": attempt["attempt_hash"],
            "source_record_hash": "",
            "issued_at": "2026-08-19T12:00:00Z",
        },
        coordinator_pubkey=pubkey,
        sign_digest=key.sign,
    )
    result = {
        "terminal_status": "authenticated_response",
        "http_status": http_status,
        "headers": {},
        "body_b64": base64.b64encode(response_body).decode(),
        "encrypted_request_artifact_id": _h("8"),
        "encrypted_artifact_id": _h("9"),
        "transport_attempt": attempt,
    }
    return compiler, prepared, request, proof, result, record, boot, response_body, key, auth_boot


def _run(response, *, http_status=200):
    fixture = _call_fixture(response, http_status=http_status)
    compiler, prepared, request, proof, result, record, boot, response_body, _key, _auth_boot = fixture
    return execute_protected_routing_provider_terminal_v2(
        authorization_proof=proof,
        prepared_call=prepared,
        broker_request=request,
        broker_result=result,
        provider_record=record,
        trusted_coordinator_boot_identity=boot,
        raw_response_body=response_body,
        binding_catalog=compiler.binding_catalog,
        unit_dataset=compiler.unit_dataset,
    ), fixture


def _direct_terminal_fixture(
    *,
    tool_id: str,
    action_id: str,
    unit: Mapping[str, Any],
    projection: Mapping[str, str],
    max_results: int,
    credit_cap: int,
    response: Mapping[str, Any],
    http_status: int = 200,
):
    """Use the same reviewed direct compiler fixtures for terminal coverage."""

    from tests.test_routing_provider_bindings import _direct_call

    compiler, prepared, _request = _direct_call(
        tool_id=tool_id,
        action_id=action_id,
        unit=dict(unit),
        projection=dict(projection),
        max_results=max_results,
        credit_cap=credit_cap,
    )
    return _call_fixture(
        response,
        http_status=http_status,
        compiler_binding=(compiler, prepared.binding),
    )


def _normalize_fixture(fixture):
    (
        compiler,
        prepared,
        request,
        proof,
        result,
        record,
        boot,
        response_body,
        _key,
        _auth_boot,
    ) = fixture
    output = execute_protected_routing_provider_terminal_v2(
        authorization_proof=proof,
        prepared_call=prepared,
        broker_request=request,
        broker_result=result,
        provider_record=record,
        trusted_coordinator_boot_identity=boot,
        raw_response_body=response_body,
        binding_catalog=compiler.binding_catalog,
        unit_dataset=compiler.unit_dataset,
    )
    return output, prepared


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        (
            {
                "result": {
                    "data": {
                        "jobs": [
                            {
                                "id": "job-1",
                                "title": "Machine Learning Engineer",
                                "company_domain": "example.com",
                                "snapshot_date": "2026-08-19",
                                "inactive": False,
                                "displayed_url": "https://jobs.example.com/job-1",
                            }
                        ]
                    }
                },
                "billing": {"credits_charged": 0.09},
            },
            "verified",
        ),
        (
            {"result": {"data": {"jobs": []}}, "billing": {"credits_charged": 0}},
            "source_miss",
        ),
        (
            {"error": "provider rejected request", "billing": {"credits_charged": 0.09}},
            "adapter_failure",
        ),
    ],
)
def test_terminal_derives_success_miss_and_billed_failure(response, expected):
    output, fixture = _run(response, http_status=402 if expected == "adapter_failure" else 200)
    (
        _compiler,
        prepared,
        request,
        _proof,
        _result,
        _record,
        _boot,
        _body,
        _key,
        _auth_boot,
    ) = fixture
    assert prepared.binding.provider_id == "bloomberry_jobs"
    assert prepared.transport_id == "deepline"
    assert request["provider_id"] == "deepline"
    assert output["projection"]["outcome"] == expected
    assert output["projection"]["billing_state"] == "known"
    assert output["provider_receipt"]["outcome"] == expected
    assert "standard_receipt_commitments" not in output
    assert "input_commitment" not in output


@pytest.mark.parametrize(
    (
        "tool_id",
        "action_id",
        "unit",
        "projection",
        "max_results",
        "credit_cap",
        "response",
        "expected",
    ),
    [
        (
            "intent.source_add.podscan",
            "podscan_episodes_search",
            {
                "query": '"Acme Corp" expansion',
                "company": "Acme Corp",
                "minimum": "2026-08-01",
                "maximum": "2026-08-31",
            },
            {
                "query": "query",
                "expected_company": "company",
                "minimum_date": "minimum",
                "maximum_date": "maximum",
            },
            1,
            140_000,
            {
                "data": {
                    "episodes": [
                        {
                            "episode_id": "episode-17",
                            "metadata": {
                                "guests": [
                                    {"guest_name": "Jane", "guest_company": "Acme Corp"}
                                ]
                            },
                            "_search_highlight": "Acme Corp discusses expansion",
                            "posted_at": "2026-08-18",
                            "episode_url": "https://pod.example/episodes/acme-expansion",
                        }
                    ]
                },
                "billing": {"credits_charged": 0.14},
            },
            "verified",
        ),
        (
            "intent.source_add.predictleads_financing",
            "predictleads_company_financing_events",
            {
                "domain": "example.com",
                "minimum": "2026-08-01",
                "maximum": "2026-08-31",
            },
            {
                "domain": "domain",
                "begin_date": "minimum",
                "end_date": "maximum",
                "minimum_date": "minimum",
                "maximum_date": "maximum",
            },
            25,
            560_000,
            {
                "result": {
                    "data": [
                        {
                            "type": "financing_event",
                            "id": "event-1",
                            "attributes": {
                                "effective_date": "2026-08-18",
                                "financing_type": "series_a",
                                "source_urls": ["https://news.example/funding"],
                            },
                            "relationships": {
                                "company": {"data": {"type": "company", "id": "co-1"}}
                            },
                        }
                    ],
                    "included": [
                        {
                            "type": "company",
                            "id": "co-1",
                            "attributes": {"domain": "example.com"},
                        }
                    ],
                },
                "billing": {"credits_charged": 0.56},
            },
            "verified",
        ),
        (
            "intent.source_add.predictleads_jobs",
            "predictleads_company_job_openings",
            {
                "domain": "example.com",
                "role": "Machine Learning",
                "minimum": "2026-08-01",
                "maximum": "2026-08-31",
            },
            {
                "domain": "domain",
                "role_keyword": "role",
                "begin_date": "minimum",
                "end_date": "maximum",
                "minimum_date": "minimum",
                "maximum_date": "maximum",
            },
            25,
            560_000,
            {
                "result": {
                    "data": [
                        {
                            "type": "job_opening",
                            "id": "event-1",
                            "attributes": {
                                "status": None,
                                "title": "Machine Learning Engineer",
                                "posted_at": "2026-08-18",
                                "last_seen_at": "2026-08-31",
                                "url": "https://jobs.example.com/ml-engineer",
                            },
                            "relationships": {
                                "company": {"data": {"type": "company", "id": "co-1"}}
                            },
                        }
                    ],
                    "included": [
                        {
                            "type": "company",
                            "id": "co-1",
                            "attributes": {"domain": "example.com"},
                        }
                    ],
                },
                "billing": {"credits_charged": 0.56},
            },
            "verified",
        ),
    ],
    ids=("podscan", "predictleads-financing", "predictleads-jobs"),
)
def test_terminal_projects_reviewed_podscan_and_predictleads_sources(
    tool_id,
    action_id,
    unit,
    projection,
    max_results,
    credit_cap,
    response,
    expected,
):
    fixture = _direct_terminal_fixture(
        tool_id=tool_id,
        action_id=action_id,
        unit=unit,
        projection=projection,
        max_results=max_results,
        credit_cap=credit_cap,
        response=response,
    )
    output, prepared = _normalize_fixture(fixture)
    assert prepared.transport_id == "deepline"
    assert output["projection"]["outcome"] == expected
    assert output["projection"]["billing_state"] == "known"


@pytest.mark.parametrize(
    ("tool_id", "action_id", "unit", "projection", "max_results", "credit_cap", "empty"),
    [
        (
            "intent.source_add.podscan",
            "podscan_episodes_search",
            {
                "query": '"Acme Corp" expansion',
                "company": "Acme Corp",
                "minimum": "2026-08-01",
                "maximum": "2026-08-31",
            },
            {"query": "query", "expected_company": "company", "minimum_date": "minimum", "maximum_date": "maximum"},
            1,
            140_000,
            {"data": {"episodes": []}, "billing": {"credits_charged": 0}},
        ),
        (
            "intent.source_add.predictleads_financing",
            "predictleads_company_financing_events",
            {"domain": "example.com", "minimum": "2026-08-01", "maximum": "2026-08-31"},
            {"domain": "domain", "begin_date": "minimum", "end_date": "maximum", "minimum_date": "minimum", "maximum_date": "maximum"},
            25,
            560_000,
            {"result": {"data": [], "included": []}, "billing": {"credits_charged": 0.56}},
        ),
        (
            "intent.source_add.predictleads_jobs",
            "predictleads_company_job_openings",
            {"domain": "example.com", "role": "Machine Learning", "minimum": "2026-08-01", "maximum": "2026-08-31"},
            {"domain": "domain", "role_keyword": "role", "begin_date": "minimum", "end_date": "maximum", "minimum_date": "minimum", "maximum_date": "maximum"},
            25,
            560_000,
            {"result": {"data": [], "included": []}, "billing": {"credits_charged": 0.56}},
        ),
    ],
    ids=("podscan-miss", "predictleads-financing-miss", "predictleads-jobs-miss"),
)
def test_terminal_projects_reviewed_source_misses(
    tool_id, action_id, unit, projection, max_results, credit_cap, empty
):
    fixture = _direct_terminal_fixture(
        tool_id=tool_id,
        action_id=action_id,
        unit=unit,
        projection=projection,
        max_results=max_results,
        credit_cap=credit_cap,
        response=empty,
    )
    output, _prepared = _normalize_fixture(fixture)
    assert output["projection"]["outcome"] == "source_miss"
    assert output["projection"]["billing_state"] == "known"


@pytest.mark.parametrize(
    ("tool_id", "action_id", "unit", "projection", "max_results", "credit_cap", "response", "credit"),
    [
        (
            "intent.source_add.podscan",
            "podscan_episodes_search",
            {"query": '"Acme Corp" expansion', "company": "Acme Corp", "minimum": "2026-08-01", "maximum": "2026-08-31"},
            {"query": "query", "expected_company": "company", "minimum_date": "minimum", "maximum_date": "maximum"},
            1,
            140_000,
            {"error": "provider rejected request", "billing": {"credits_charged": 0.14}},
            0.14,
        ),
        (
            "intent.source_add.predictleads_financing",
            "predictleads_company_financing_events",
            {"domain": "example.com", "minimum": "2026-08-01", "maximum": "2026-08-31"},
            {"domain": "domain", "begin_date": "minimum", "end_date": "maximum", "minimum_date": "minimum", "maximum_date": "maximum"},
            25,
            560_000,
            {"error": "provider rejected request", "billing": {"credits_charged": 0.56}},
            0.56,
        ),
        (
            "intent.source_add.predictleads_jobs",
            "predictleads_company_job_openings",
            {"domain": "example.com", "role": "Machine Learning", "minimum": "2026-08-01", "maximum": "2026-08-31"},
            {"domain": "domain", "role_keyword": "role", "begin_date": "minimum", "end_date": "maximum", "minimum_date": "minimum", "maximum_date": "maximum"},
            25,
            560_000,
            {"error": "provider rejected request", "billing": {"credits_charged": 0.56}},
            0.56,
        ),
    ],
    ids=("podscan-billed-failure", "predictleads-financing-billed-failure", "predictleads-jobs-billed-failure"),
)
def test_terminal_projects_billed_failures(
    tool_id, action_id, unit, projection, max_results, credit_cap, response, credit
):
    fixture = _direct_terminal_fixture(
        tool_id=tool_id,
        action_id=action_id,
        unit=unit,
        projection=projection,
        max_results=max_results,
        credit_cap=credit_cap,
        response=response,
        http_status=402,
    )
    output, _prepared = _normalize_fixture(fixture)
    assert output["projection"]["outcome"] == "adapter_failure"
    assert output["projection"]["credit_microunits"] == int(credit * 1_000_000)
    assert output["projection"]["billing_state"] == "known"


@pytest.mark.parametrize(
    "response",
    [
        {"result": {"data": {"jobs": []}}, "billing": {"credits_charged": True}},
        {"result": {"data": {"jobs": []}}},
    ],
)
def test_terminal_rejects_uncertain_or_boolean_billing(response):
    fixture = _call_fixture(response)
    compiler, prepared, request, proof, result, record, boot, body, _key, _auth_boot = fixture
    with pytest.raises(ProtectedRoutingProviderTerminalError):
        execute_protected_routing_provider_terminal_v2(
            authorization_proof=proof,
            prepared_call=prepared,
            broker_request=request,
            broker_result=result,
            provider_record=record,
            trusted_coordinator_boot_identity=boot,
            raw_response_body=body,
            binding_catalog=compiler.binding_catalog,
            unit_dataset=compiler.unit_dataset,
        )


def test_terminal_rejects_body_attempt_and_identity_substitution():
    output, fixture = _run(
        {"result": {"data": {"jobs": []}}, "billing": {"credits_charged": 0}}
    )
    del output
    compiler, prepared, request, proof, result, record, boot, body, _key, _auth_boot = fixture
    for mutate in (
        lambda: result.update({"body_b64": base64.b64encode(b'{"substituted":true}').decode()}),
        lambda: result["transport_attempt"].update({"attempt_number": 1}),
        lambda: request.update({"provider_id": "other-provider"}),
        lambda: record.update({"record_hash": _h("f")}),
    ):
        forged_result = deepcopy(result)
        forged_request = deepcopy(request)
        forged_record = deepcopy(record)
        result, request, record = forged_result, forged_request, forged_record
        mutate()
        with pytest.raises(ProtectedRoutingProviderTerminalError):
            execute_protected_routing_provider_terminal_v2(
                authorization_proof=proof,
                prepared_call=prepared,
                broker_request=request,
                broker_result=result,
                provider_record=record,
                trusted_coordinator_boot_identity=boot,
                raw_response_body=body,
                binding_catalog=compiler.binding_catalog,
                unit_dataset=compiler.unit_dataset,
            )


def test_terminal_rejects_admission_job_reused_as_authorization_job():
    fixture = _call_fixture({"result": {"data": {"jobs": []}}})
    compiler, prepared, request, proof, result, record, boot, body, _key, _auth_boot = fixture
    forged_request = dict(request)
    forged_request["job_id"] = proof["authorization"]["admission_job_id"]
    forged_request["routing_authorization"] = proof
    with pytest.raises(ProtectedRoutingProviderTerminalError, match="authorization proof"):
        execute_protected_routing_provider_terminal_v2(
            authorization_proof=proof,
            prepared_call=prepared,
            broker_request=forged_request,
            broker_result=result,
            provider_record=record,
            trusted_coordinator_boot_identity=boot,
            raw_response_body=body,
            binding_catalog=compiler.binding_catalog,
            unit_dataset=compiler.unit_dataset,
        )


def test_terminal_rejects_model_transport_identity_mismatch_and_malformed_response():
    _output, fixture = _run(
        {"result": {"data": {"jobs": []}}, "billing": {"credits_charged": 0}}
    )
    compiler, prepared, request, proof, result, record, boot, body, _key, _auth_boot = fixture
    malformed = b"not-json"
    result = dict(result)
    result["body_b64"] = base64.b64encode(malformed).decode()
    with pytest.raises(ProtectedRoutingProviderTerminalError):
        execute_protected_routing_provider_terminal_v2(
            authorization_proof=proof,
            prepared_call=replace(prepared, action_id="other_action"),
            broker_request=request,
            broker_result=result,
            provider_record=record,
            trusted_coordinator_boot_identity=boot,
            raw_response_body=malformed,
            binding_catalog=compiler.binding_catalog,
            unit_dataset=compiler.unit_dataset,
        )

    with pytest.raises(ProtectedRoutingProviderTerminalError):
        execute_protected_routing_provider_terminal_v2(
            authorization_proof=proof,
            prepared_call=replace(prepared, transport_id="wrong-transport"),
            broker_request=request,
            broker_result=fixture[4],
            provider_record=record,
            trusted_coordinator_boot_identity=boot,
            raw_response_body=fixture[7],
            binding_catalog=compiler.binding_catalog,
            unit_dataset=compiler.unit_dataset,
        )


def test_terminal_rejects_prepared_call_absent_from_reviewed_catalog():
    compiler, prepared, request, proof, result, record, boot, body, _key, _auth_boot = (
        _call_fixture(
            {"result": {"data": {"jobs": []}}, "billing": {"credits_charged": 0}}
        )
    )
    catalog_without_binding = replace(compiler.binding_catalog, bindings={})
    with pytest.raises(ProtectedRoutingProviderTerminalError):
        execute_protected_routing_provider_terminal_v2(
            authorization_proof=proof,
            prepared_call=prepared,
            broker_request=request,
            broker_result=result,
            provider_record=record,
            trusted_coordinator_boot_identity=boot,
            raw_response_body=body,
            binding_catalog=catalog_without_binding,
            unit_dataset=compiler.unit_dataset,
        )


def test_terminal_rejects_authorization_and_provider_signatures():
    compiler, prepared, request, proof, result, record, boot, body, _key, _auth_boot = (
        _call_fixture(
            {"result": {"data": {"jobs": []}}, "billing": {"credits_charged": 0}}
        )
    )
    forged_proof = deepcopy(proof)
    forged_proof["authorization_receipt"] = dict(proof["authorization_receipt"])
    forged_proof["authorization_receipt"]["enclave_signature"] = "0" * 128
    forged_request = dict(request)
    forged_request["routing_authorization"] = forged_proof
    with pytest.raises(ProtectedRoutingProviderTerminalError):
        execute_protected_routing_provider_terminal_v2(
            authorization_proof=forged_proof,
            prepared_call=prepared,
            broker_request=forged_request,
            broker_result=result,
            provider_record=record,
            trusted_coordinator_boot_identity=boot,
            raw_response_body=body,
            binding_catalog=compiler.binding_catalog,
            unit_dataset=compiler.unit_dataset,
        )


def test_terminal_rejects_trusted_coordinator_identity_substitution():
    compiler, prepared, request, proof, result, record, boot, body, _key, _auth_boot = (
        _call_fixture(
            {"result": {"data": {"jobs": []}}, "billing": {"credits_charged": 0}}
        )
    )
    for field, replacement in (
        ("boot_identity_hash", _h("f")),
        ("signing_pubkey", "0" * 64),
    ):
        forged_boot = dict(boot)
        forged_boot[field] = replacement
        with pytest.raises(ProtectedRoutingProviderTerminalError):
            execute_protected_routing_provider_terminal_v2(
                authorization_proof=proof,
                prepared_call=prepared,
                broker_request=request,
                broker_result=result,
                provider_record=record,
                trusted_coordinator_boot_identity=forged_boot,
                raw_response_body=body,
                binding_catalog=compiler.binding_catalog,
                unit_dataset=compiler.unit_dataset,
            )

    forged_record = dict(record)
    forged_record["coordinator_signature"] = "0" * 128
    with pytest.raises(ProtectedRoutingProviderTerminalError):
        execute_protected_routing_provider_terminal_v2(
            authorization_proof=proof,
            prepared_call=prepared,
            broker_request=request,
            broker_result=result,
            provider_record=forged_record,
            trusted_coordinator_boot_identity=boot,
            raw_response_body=body,
            binding_catalog=compiler.binding_catalog,
            unit_dataset=compiler.unit_dataset,
        )


def test_budget_reservation_is_exactly_bound_and_proof_is_redacted():
    compiler, prepared, _request, proof, _result, _record, _boot, _body, _key, _auth_boot = (
        _call_fixture(
            {"result": {"data": {"jobs": []}}, "billing": {"credits_charged": 0}}
        )
    )
    authorization = RoutingProviderCallAuthorizationV2.from_mapping(
        proof["authorization"]
    )
    reservation = build_routing_budget_reservation_v3(
        authorization=authorization,
        prepared_call=prepared,
        lease_seconds=5,
    )
    assert set(reservation) == {
        "schema_version",
        "event_key",
        "reservation_id",
        "experiment_hash",
        "binding_id",
        "claim_key",
        "claim_generation",
        "claim_fence_hash",
        "credit_microunits",
        "lease_seconds",
        "event_doc",
    }
    assert set(reservation["event_doc"]) == {
        "schema_version",
        "reservation_id",
        "binding_id",
        "call_grant_hash",
        "action_id",
        "tool_id",
        "variant_id",
        "unit_ref",
        "attempt",
        "request_fingerprint",
        "request_body_hash",
        "binding_catalog_manifest_hash",
    }
    with pytest.raises(ProtectedRoutingProviderTerminalError):
        validate_routing_budget_reservation_v3(
            {**reservation, "credit_microunits": reservation["credit_microunits"] + 1},
            authorization=authorization,
            prepared_call=prepared,
        )

    result = {
        "schema_version": "leadpoet.research_lab.routing_budget_reservation_result.v3",
        "reserved": True,
        "idempotent": False,
        "reservation_id": reservation["reservation_id"],
        "event_key": reservation["event_key"],
        "experiment_hash": reservation["experiment_hash"],
        "binding_id": reservation["binding_id"],
        "claim_key": reservation["claim_key"],
        "claim_generation": reservation["claim_generation"],
        "credit_microunits": reservation["credit_microunits"],
        "lease_expires_at": "2099-01-01T00:00:00+00:00",
    }
    normalized = validate_routing_budget_reservation_result_v3(
        result, reservation=reservation
    )
    proof_projection = routing_budget_reservation_proof_v3(
        reservation_result=normalized,
        response_hash=_h("a"),
        transport_attempt_hash=_h("b"),
    )
    assert set(proof_projection) == {
        "schema_version",
        "reservation_id",
        "event_key",
        "experiment_hash",
        "binding_id",
        "claim_key",
        "claim_generation",
        "credit_microunits",
        "lease_expires_at",
        "response_hash",
        "transport_attempt_hash",
    }
    assert "body_b64" not in proof_projection
    assert "credential_ref" not in proof_projection
    assert "event_doc" not in proof_projection
