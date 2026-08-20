from __future__ import annotations

import base64
from dataclasses import replace
import json

import pytest

from gateway.research_lab.routing_provider_bindings import (
    DEEPLINE_COMPILER_FAMILY,
    ROUTING_BINDING_CATALOG_SCHEMA,
    ROUTING_UNIT_DATASET_SCHEMA,
    ReviewedDeeplineActionCompiler,
    PreparedRoutingProviderWorkflow,
    ReviewedDeeplineWorkflowCompiler,
    RoutingBindingManifest,
    RoutingProviderBindingError,
    SignedRoutingBindingCatalogLoader,
    SignedRoutingUnitDatasetLoader,
    VerifiedRoutingBindingCatalog,
    VerifiedRoutingUnitDataset,
)
from gateway.research_lab.routing_execution_authorization import (
    RoutingProviderCallAuthorizationV2,
    execute_routing_provider_call_authorization_v2,
    routing_provider_dispatch_job_id_v2,
)
from gateway.research_lab.routing_experiment_runtime import (
    AttestedScoringV2RoutingProviderCallAuthority,
)
from gateway.research_lab.routing_predictleads_workflows import (
    ROUTE_CONNECTIONS,
    ROUTE_ACTION_ORDER,
    ROUTE_CREDIT_CEILINGS,
    ROUTE_NEWS,
    ROUTE_MAX_CALLS,
    ROUTE_TECHNOLOGY,
    workflow_manifest,
)
from research_lab.canonical import sha256_json
from research_lab.routing_experiments import ProviderBindingIdentity
from leadpoet_canonical.attested_v2 import (
    build_execution_receipt_body,
    create_signed_execution_receipt,
)


def _hash(char: str) -> str:
    return "sha256:" + char * 64


_MODEL_REQUIREMENTS_HASH = _hash("0")


def _binding(*, tool_id: str = "intent.source_add.bloomberry_jobs") -> ProviderBindingIdentity:
    provider_id = {
        "intent.source_add.bloomberry_jobs": "bloomberry_jobs",
        "intent.source_add.bloomberry": "bloomberry",
        "intent.source_add.podscan": "podscan",
        "intent.source_add.predictleads_financing": "predictleads_financing",
        "intent.source_add.predictleads_jobs": "predictleads_jobs",
        "intent.source_add.predictleads_connections": "predictleads_connections",
        "intent.source_add.predictleads_news": "predictleads_news",
        "intent.source_add.predictleads_technology": "predictleads_technology",
        "intent.source_add.builtwith": "builtwith",
        "intent.source_add.sumble": "sumble",
    }.get(tool_id, "unreviewed")
    return ProviderBindingIdentity(
        binding_id="deepline-bloomberry-jobs-v1",
        provider_id=provider_id,
        tool_id=tool_id,
        source_lineage_id="deepline.bloomberry.jobs",
        adapter_version="v1",
        manifest_hash=_hash("1"),
        capability_hash=_hash("2"),
        execution_contract_hash=_hash("3"),
        cost_model_hash=_hash("4"),
    )


def _signed(payload: dict) -> dict:
    value = dict(payload)
    value["manifest_hash"] = sha256_json(value)
    return value


def _verifier(document, key_id):
    return {
        "verified": True,
        "manifest_hash": document["manifest_hash"],
        "signature_ref": document["signature_ref"],
        "key_id": key_id,
        "signing_algorithm": "ECDSA_SHA_256",
    }


def _authorities(*, action_id="bloomberry_search_job_postings", billing_cap=90_000):
    binding = _binding()
    catalog_uri = "s3://lab-routing/bindings/catalog-001.json"
    unit_uri = "s3://lab-routing/units/dataset-001.json"
    catalog = _signed(
        {
            "schema_version": ROUTING_BINDING_CATALOG_SCHEMA,
            "manifest_uri": catalog_uri,
            "catalog_version": "catalog-001",
            "bindings": [
                {
                    "binding": binding.to_dict(),
                    "compiler_family": DEEPLINE_COMPILER_FAMILY,
                    "transport_id": "deepline",
                    "execution_kind": "direct_action",
                    "action_id": action_id,
                    "workflow_id": None,
                    "workflow_manifest_hash": None,
                    "input_projection": {
                        "domain": "company_domain",
                        "keyword": "job_keyword",
                        "countries": "countries",
                        "minimum_date": "minimum_date",
                        "maximum_date": "maximum_date",
                    },
                    "input_constants": {},
                    "model_binding_requirements_hash": _MODEL_REQUIREMENTS_HASH,
                    "output_contract_hash": _hash("5"),
                    "evidence_contract_hash": _hash("6"),
                    "retry_policy_hash": _hash("7"),
                    "max_results": 1,
                    "timeout_ms": 5_000,
                    "credit_ceiling_microunits": billing_cap,
                }
            ],
            "signature_ref": "s3://lab-routing/signatures/catalog-001.sig",
        }
    )
    units = {
        "company-1": {
            "company_domain": "example.com",
            "job_keyword": "machine learning",
            "countries": "US;CA",
            "minimum_date": "2026-08-01",
            "maximum_date": "2026-08-31",
        }
    }
    unit_set_hash = sha256_json(
        {
            "schema_version": ROUTING_UNIT_DATASET_SCHEMA,
            "units": [{"unit_ref": "company-1", "input": units["company-1"]}],
        }
    )
    dataset = _signed(
        {
            "schema_version": ROUTING_UNIT_DATASET_SCHEMA,
            "manifest_uri": unit_uri,
            "units": units,
            "unit_set_hash": unit_set_hash,
            "provenance_hash": _hash("8"),
            "signature_ref": "s3://lab-routing/signatures/dataset-001.sig",
        }
    )
    documents = {catalog_uri: catalog, unit_uri: dataset}
    loader = documents.__getitem__
    catalog_value = SignedRoutingBindingCatalogLoader(
        manifest_uri=catalog_uri,
        key_id="kms-binding-key",
        loader=loader,
        verifier=_verifier,
    ).load()
    unit_value = SignedRoutingUnitDatasetLoader(
        manifest_uri=unit_uri,
        key_id="kms-unit-key",
        loader=loader,
        verifier=_verifier,
    ).load(
        expected_unit_refs=("company-1",),
        expected_unit_set_hash=unit_set_hash,
    )
    return binding, catalog_value, unit_value


def _prepared():
    binding, catalog, units = _authorities()
    compiler = ReviewedDeeplineActionCompiler(
        binding_catalog=catalog,
        unit_dataset=units,
    )
    prepared = compiler.prepare(
        binding=binding,
        unit_ref="company-1",
        authorization_credit_microunits=90_000,
        authorization_timeout_ms=5_000,
        expected_model_binding_requirements_hash=_MODEL_REQUIREMENTS_HASH,
    )
    request = compiler.broker_request(
        prepared=prepared,
        experiment_hash=_hash("9"),
        dispatch_job_id="routing-dispatch-test-job",
        variant_id="candidate",
        attempt_number=0,
        core_request_fingerprint=_hash("a"),
        authorization_hash=_hash("b"),
        authorization_proof_hash=_hash("c"),
    )
    return compiler, prepared, request


def _broker_result(request, response):
    body = json.dumps(response, sort_keys=True, separators=(",", ":")).encode()
    return {
        "terminal_status": "authenticated_response",
        "http_status": 200,
        "headers": {},
        "body_b64": base64.b64encode(body).decode(),
        "encrypted_request_artifact_id": _hash("d"),
        "encrypted_artifact_id": _hash("e"),
        "transport_attempt": {
            "logical_operation_id": request["logical_operation_id"],
            "job_id": request["job_id"],
            "purpose": request["purpose"],
            "provider_id": request["provider_id"],
            "attempt_number": request["attempt_number"],
            "method": "POST",
            "timeout_ms": request["timeout_ms"],
            "retry_policy_hash": request["retry_policy_hash"],
            "started_at": "2026-08-19T12:00:00+00:00",
            "completed_at": "2026-08-19T12:00:00.025000+00:00",
        },
        "evidence_artifact_hashes": [_hash("f")],
    }


def _direct_call(
    *,
    tool_id: str,
    action_id: str,
    unit: dict,
    projection: dict,
    max_results: int,
    credit_cap: int,
    phase: str = "initial",
    execution_mode: str = "measured_lab",
):
    binding = _binding(tool_id=tool_id)
    manifest = RoutingBindingManifest(
        binding=binding,
        compiler_family=DEEPLINE_COMPILER_FAMILY,
        transport_id="deepline",
        action_id=action_id,
        input_projection=projection,
        input_constants={},
        model_binding_requirements_hash=_MODEL_REQUIREMENTS_HASH,
        output_contract_hash=_hash("5"),
        evidence_contract_hash=_hash("6"),
        retry_policy_hash=_hash("7"),
        max_results=max_results,
        timeout_ms=5_000,
        credit_ceiling_microunits=credit_cap,
    )
    unit_set_hash = sha256_json(
        {
            "schema_version": ROUTING_UNIT_DATASET_SCHEMA,
            "units": [{"unit_ref": "company-1", "input": unit}],
        }
    )
    compiler = ReviewedDeeplineActionCompiler(
        binding_catalog=VerifiedRoutingBindingCatalog(
            manifest_uri="s3://lab-routing/bindings/direct.json",
            manifest_hash=_hash("a"),
            signature_ref="s3://lab-routing/signatures/direct.sig",
            signing_key_id="kms-binding-key",
            catalog_version="direct-v1",
            bindings={manifest.identity_key(): manifest},
        ),
        unit_dataset=VerifiedRoutingUnitDataset(
            manifest_uri="s3://lab-routing/units/direct.json",
            manifest_hash=_hash("b"),
            signature_ref="s3://lab-routing/signatures/units.sig",
            signing_key_id="kms-unit-key",
            unit_set_hash=unit_set_hash,
            provenance_hash=_hash("c"),
            units={"company-1": unit},
        ),
    )
    prepared = compiler.prepare(
        binding=binding,
        unit_ref="company-1",
        authorization_credit_microunits=credit_cap,
        authorization_timeout_ms=5_000,
        expected_model_binding_requirements_hash=_MODEL_REQUIREMENTS_HASH,
        phase=phase,
        execution_mode=execution_mode,
    )
    request = compiler.broker_request(
        prepared=prepared,
        experiment_hash=_hash("9"),
        dispatch_job_id="routing-dispatch-test-job",
        variant_id="candidate",
        attempt_number=0,
        core_request_fingerprint=_hash("a"),
        authorization_hash=_hash("b"),
        authorization_proof_hash=_hash("c"),
    )
    return compiler, prepared, request


def _project(triple, response):
    compiler, prepared, request = triple
    return compiler.project_result(
        prepared=prepared,
        broker_request=request,
        broker_result=_broker_result(request, response),
        core_request_fingerprint=_hash("a"),
    )


def _composite_compiler(*, tool_id: str, route: str, unit: dict):
    binding = _binding(tool_id=tool_id)
    manifest = workflow_manifest(route)
    projection = {
        field: field
        for field in (
            ("company_domain", "minimum_date", "maximum_date")
            if route == ROUTE_CONNECTIONS
            else (
                "company_domain",
                "intent_category",
                "minimum_date",
                "maximum_date",
            )
            if route == ROUTE_NEWS
            else ("company_domain", "technology", "minimum_date", "maximum_date")
        )
    }
    binding_manifest = RoutingBindingManifest(
        binding=binding,
        compiler_family=DEEPLINE_COMPILER_FAMILY,
        transport_id="deepline",
        action_id="",
        input_projection=projection,
        input_constants={},
        model_binding_requirements_hash=_MODEL_REQUIREMENTS_HASH,
        output_contract_hash=_hash("5"),
        evidence_contract_hash=_hash("6"),
        retry_policy_hash=_hash("7"),
        max_results=1,
        timeout_ms=manifest.timeout_ms,
        credit_ceiling_microunits=manifest.credit_ceiling_microcredits,
        execution_kind="composite_workflow",
        workflow_id=route,
        workflow_manifest_hash=manifest.manifest_hash,
    )
    unit_set_hash = sha256_json(
        {
            "schema_version": ROUTING_UNIT_DATASET_SCHEMA,
            "units": [{"unit_ref": "company-1", "input": unit}],
        }
    )
    catalog = VerifiedRoutingBindingCatalog(
        manifest_uri="s3://lab-routing/bindings/composite.json",
        manifest_hash=_hash("a"),
        signature_ref="s3://lab-routing/signatures/composite.sig",
        signing_key_id="kms-binding-key",
        catalog_version="composite-v1",
        bindings={binding_manifest.identity_key(): binding_manifest},
    )
    units = VerifiedRoutingUnitDataset(
        manifest_uri="s3://lab-routing/units/composite.json",
        manifest_hash=_hash("b"),
        signature_ref="s3://lab-routing/signatures/units.sig",
        signing_key_id="kms-unit-key",
        unit_set_hash=unit_set_hash,
        provenance_hash=_hash("c"),
        units={"company-1": unit},
    )
    return binding, ReviewedDeeplineWorkflowCompiler(
        binding_catalog=catalog,
        unit_dataset=units,
    )


def _composite_row(*, tool_id: str, route: str) -> dict:
    binding = _binding(tool_id=tool_id)
    reviewed = workflow_manifest(route)
    fields = (
        ("company_domain", "company_domain", "minimum_date", "minimum_date", "maximum_date", "maximum_date")
        if route == ROUTE_CONNECTIONS
        else (
            ("company_domain", "company_domain", "intent_category", "intent_category", "minimum_date", "minimum_date", "maximum_date", "maximum_date")
            if route == ROUTE_NEWS
            else ("company_domain", "company_domain", "technology", "technology", "minimum_date", "minimum_date", "maximum_date", "maximum_date")
        )
    )
    projection = dict(zip(fields[::2], fields[1::2]))
    return {
        "binding": binding.to_dict(),
        "compiler_family": DEEPLINE_COMPILER_FAMILY,
        "transport_id": "deepline",
        "execution_kind": "composite_workflow",
        "action_id": None,
        "workflow_id": route,
        "workflow_manifest_hash": reviewed.manifest_hash,
        "input_projection": projection,
        "input_constants": {},
        "model_binding_requirements_hash": _MODEL_REQUIREMENTS_HASH,
        "output_contract_hash": _hash("5"),
        "evidence_contract_hash": _hash("6"),
        "retry_policy_hash": _hash("7"),
        "max_results": 1,
        "timeout_ms": reviewed.timeout_ms,
        "credit_ceiling_microunits": reviewed.credit_ceiling_microcredits,
    }


def _direct_catalog_row() -> dict:
    binding, catalog, _units = _authorities()
    manifest = next(iter(catalog.bindings.values()))
    return {
        "binding": binding.to_dict(),
        "compiler_family": DEEPLINE_COMPILER_FAMILY,
        "transport_id": "deepline",
        "execution_kind": "direct_action",
        "action_id": manifest.action_id,
        "workflow_id": None,
        "workflow_manifest_hash": None,
        "input_projection": dict(manifest.input_projection),
        "input_constants": dict(manifest.input_constants),
        "model_binding_requirements_hash": manifest.model_binding_requirements_hash,
        "output_contract_hash": manifest.output_contract_hash,
        "evidence_contract_hash": manifest.evidence_contract_hash,
        "retry_policy_hash": manifest.retry_policy_hash,
        "max_results": manifest.max_results,
        "timeout_ms": manifest.timeout_ms,
        "credit_ceiling_microunits": manifest.credit_ceiling_microunits,
    }


def _composite_document(row: dict | list[dict], uri: str = "s3://lab-routing/bindings/composite-catalog.json"):
    rows = row if isinstance(row, list) else [row]
    return _signed(
        {
            "schema_version": ROUTING_BINDING_CATALOG_SCHEMA,
            "manifest_uri": uri,
            "catalog_version": "composite-catalog-1",
            "bindings": rows,
            "signature_ref": "s3://lab-routing/signatures/composite-catalog.sig",
        }
    )


@pytest.mark.parametrize(
    ("tool_id", "route"),
    [
        ("intent.source_add.predictleads_connections", ROUTE_CONNECTIONS),
        ("intent.source_add.predictleads_technology", ROUTE_TECHNOLOGY),
    ],
)
def test_composite_workflow_compiler_prepares_atomic_signed_route_without_dispatch(
    tool_id, route
):
    unit = {
        "company_domain": "example.com",
        "minimum_date": "2026-08-01",
        "maximum_date": "2026-08-31",
        "technology": "Snowflake",
    }
    binding, compiler = _composite_compiler(tool_id=tool_id, route=route, unit=unit)
    prepared = compiler.prepare(
        binding=binding,
        unit_ref="company-1",
        authorization_credit_microunits=ROUTE_CREDIT_CEILINGS[route],
        authorization_timeout_ms=30_000,
        expected_model_binding_requirements_hash=_MODEL_REQUIREMENTS_HASH,
    )
    assert isinstance(prepared, PreparedRoutingProviderWorkflow)
    assert prepared.workflow_id == route
    assert prepared.ordered_actions == ROUTE_ACTION_ORDER[route]
    assert prepared.branch_optional_actions == workflow_manifest(route).branch_optional_actions
    assert prepared.credit_ceiling_microunits == ROUTE_CREDIT_CEILINGS[route]
    assert prepared.max_calls == ROUTE_MAX_CALLS[route]
    assert prepared.workflow_input["company_domain"] == "example.com"
    assert "url" not in prepared.authorization_projection()
    assert "payload" not in prepared.authorization_projection()


def test_composite_news_is_offline_only_because_exa_price_is_calculated_at_execution():
    binding, compiler = _composite_compiler(
        tool_id="intent.source_add.predictleads_news",
        route=ROUTE_NEWS,
        unit={
            "company_domain": "example.com",
            "intent_category": "PARTNERSHIP",
            "minimum_date": "2026-08-01",
            "maximum_date": "2026-08-31",
        },
    )
    with pytest.raises(RoutingProviderBindingError, match="no provider-enforced cost cap"):
        compiler.prepare(
            binding=binding,
            unit_ref="company-1",
            authorization_credit_microunits=ROUTE_CREDIT_CEILINGS[ROUTE_NEWS],
            authorization_timeout_ms=30_000,
            expected_model_binding_requirements_hash=_MODEL_REQUIREMENTS_HASH,
        )
    prepared = compiler.prepare(
        binding=binding,
        unit_ref="company-1",
        authorization_credit_microunits=ROUTE_CREDIT_CEILINGS[ROUTE_NEWS],
        authorization_timeout_ms=30_000,
        expected_model_binding_requirements_hash=_MODEL_REQUIREMENTS_HASH,
        execution_mode="replay",
    )
    assert prepared.branch_optional_actions == ("exa_search",)


@pytest.mark.parametrize(
    ("tool_id", "route"),
    [
        ("intent.source_add.predictleads_connections", ROUTE_CONNECTIONS),
        ("intent.source_add.predictleads_technology", ROUTE_TECHNOLOGY),
    ],
)
def test_signed_catalog_round_trips_composite_workflow_manifest(tool_id, route):
    uri = "s3://lab-routing/bindings/composite-catalog.json"
    document = _composite_document(_composite_row(tool_id=tool_id, route=route), uri)
    catalog = SignedRoutingBindingCatalogLoader(
        manifest_uri=uri,
        key_id="kms-binding-key",
        loader=lambda _uri: document,
        verifier=_verifier,
    ).load_composite_workflows()
    binding = _binding(tool_id=tool_id)
    manifest = catalog.resolve(binding)
    assert manifest.execution_kind == "composite_workflow"
    assert manifest.action_id == ""
    assert manifest.workflow_id == route
    assert manifest.workflow_manifest_hash == workflow_manifest(route).manifest_hash
    assert manifest.credit_ceiling_microunits == ROUTE_CREDIT_CEILINGS[route]


def test_signed_catalog_round_trips_mixed_direct_and_composite_bindings():
    uri = "s3://lab-routing/bindings/mixed-catalog.json"
    direct_row = _direct_catalog_row()
    composite_row = _composite_row(
        tool_id="intent.source_add.predictleads_technology",
        route=ROUTE_TECHNOLOGY,
    )
    document = _composite_document([direct_row, composite_row], uri)
    catalog = SignedRoutingBindingCatalogLoader(
        manifest_uri=uri,
        key_id="kms-binding-key",
        loader=lambda _uri: document,
        verifier=_verifier,
    ).load_reviewed_bindings()
    direct_binding = _binding()
    composite_binding = _binding(tool_id="intent.source_add.predictleads_technology")
    assert catalog.resolve(direct_binding).execution_kind == "direct_action"
    assert catalog.resolve(composite_binding).execution_kind == "composite_workflow"


def test_signed_mixed_catalog_rejects_duplicate_binding_identity():
    uri = "s3://lab-routing/bindings/mixed-duplicate.json"
    row = _direct_catalog_row()
    document = _composite_document([row, dict(row)], uri)
    with pytest.raises(RoutingProviderBindingError, match="duplicated"):
        SignedRoutingBindingCatalogLoader(
            manifest_uri=uri,
            key_id="kms-binding-key",
            loader=lambda _uri: document,
            verifier=_verifier,
        ).load_reviewed_bindings()


@pytest.mark.parametrize("field", ("max_results", "timeout_ms", "credit_ceiling_microunits"))
def test_signed_composite_catalog_rejects_boolean_numeric_limits(field):
    uri = "s3://lab-routing/bindings/composite-bool-limit.json"
    row = _composite_row(
        tool_id="intent.source_add.predictleads_technology",
        route=ROUTE_TECHNOLOGY,
    )
    row[field] = True
    document = _composite_document([_direct_catalog_row(), row], uri)
    with pytest.raises(RoutingProviderBindingError, match="limits"):
        SignedRoutingBindingCatalogLoader(
            manifest_uri=uri,
            key_id="kms-binding-key",
            loader=lambda _uri: document,
            verifier=_verifier,
        ).load_reviewed_bindings()


@pytest.mark.parametrize(
    "mutation",
    ("wrong_hash", "wrong_action", "wrong_order_cap", "endpoint", "branch"),
)
def test_signed_catalog_rejects_composite_substitution_or_branch(mutation):
    uri = "s3://lab-routing/bindings/composite-adversarial.json"
    row = _composite_row(
        tool_id="intent.source_add.predictleads_connections",
        route=ROUTE_CONNECTIONS,
    )
    if mutation == "wrong_hash":
        row["workflow_manifest_hash"] = "0" * 64
    elif mutation == "wrong_action":
        row["action_id"] = "predictleads_company_connections"
    elif mutation == "wrong_order_cap":
        row["credit_ceiling_microunits"] = 560_000
    elif mutation == "endpoint":
        row["input_projection"]["endpoint"] = "company_domain"
    else:
        row["branch"] = "exa_search"
    document = _composite_document([_direct_catalog_row(), row], uri)
    with pytest.raises(RoutingProviderBindingError):
        SignedRoutingBindingCatalogLoader(
            manifest_uri=uri,
            key_id="kms-binding-key",
            loader=lambda _uri: document,
            verifier=_verifier,
        ).load_reviewed_bindings()


@pytest.mark.parametrize(
    "mutation",
    [
        "workflow_hash",
        "workflow_id",
        "credit",
        "missing_unit",
        "extra_input",
    ],
)
def test_composite_workflow_rejects_substitutions_unknown_caps_and_unsigned_input(mutation):
    binding, compiler = _composite_compiler(
        tool_id="intent.source_add.predictleads_connections",
        route=ROUTE_CONNECTIONS,
        unit={
            "company_domain": "example.com",
            "minimum_date": "2026-08-01",
            "maximum_date": "2026-08-31",
        },
    )
    manifest = compiler.binding_catalog.bindings[next(iter(compiler.binding_catalog.bindings))]
    if mutation == "workflow_hash":
        manifest = replace(manifest, workflow_manifest_hash=_hash("d"))
    elif mutation == "workflow_id":
        manifest = replace(manifest, workflow_id=ROUTE_TECHNOLOGY)
    elif mutation == "credit":
        manifest = replace(manifest, credit_ceiling_microunits=560_000)
    elif mutation == "missing_unit":
        manifest = replace(manifest, input_projection={"company_domain": "company_domain", "minimum_date": "minimum_date", "maximum_date": "absent"})
    else:
        manifest = replace(manifest, input_projection={**manifest.input_projection, "endpoint": "company_domain"})
    catalog = replace(
        compiler.binding_catalog,
        bindings={manifest.identity_key(): manifest},
    )
    changed = ReviewedDeeplineWorkflowCompiler(
        binding_catalog=catalog,
        unit_dataset=compiler.unit_dataset,
    )
    with pytest.raises(RoutingProviderBindingError):
        changed.prepare(
            binding=binding,
            unit_ref="company-1",
            authorization_credit_microunits=ROUTE_CREDIT_CEILINGS[ROUTE_CONNECTIONS],
            authorization_timeout_ms=30_000,
            expected_model_binding_requirements_hash=_MODEL_REQUIREMENTS_HASH,
        )


def test_signed_catalog_and_unit_dataset_compile_exact_deepline_request():
    _compiler, prepared, request = _prepared()
    assert request["url"] == (
        "https://code.deepline.com/api/v2/integrations/"
        "bloomberry_search_job_postings/execute"
    )
    assert request["headers"] == {
        "Content-Type": "application/json",
        "x-deepline-execute-response-intent": "raw",
    }
    body = json.loads(base64.b64decode(request["body_b64"]))
    assert body == {
        "provider": "bloomberry",
        "operation": "bloomberry_search_job_postings",
        "payload": {
            "active_only": True,
            "countries": "US;CA",
            "domain": "example.com",
            "exact_match": True,
            "keyword": "machine learning",
            "limit": 1,
            "show_facets": False,
        },
    }
    assert prepared.unit_input_hash.startswith("sha256:")
    assert "company_domain" not in request
    assert "payload" not in request


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda row: row.update(action_id="generic_http_request"), "unreviewed action"),
        (lambda row: row["input_projection"].update(endpoint="company_domain"), "not reviewed"),
        (lambda row: row.update(timeout_ms=60_000), "limits exceed"),
    ],
)
def test_signed_binding_cannot_select_arbitrary_endpoint_body_or_limit(mutation, message):
    binding = _binding()
    uri = "s3://lab-routing/bindings/bad.json"
    row = {
        "binding": binding.to_dict(),
        "compiler_family": DEEPLINE_COMPILER_FAMILY,
        "transport_id": "deepline",
        "execution_kind": "direct_action",
        "action_id": "bloomberry_search_job_postings",
        "workflow_id": None,
        "workflow_manifest_hash": None,
        "input_projection": {"domain": "company_domain"},
        "input_constants": {},
        "model_binding_requirements_hash": _MODEL_REQUIREMENTS_HASH,
        "output_contract_hash": _hash("5"),
        "evidence_contract_hash": _hash("6"),
        "retry_policy_hash": _hash("7"),
        "max_results": 1,
        "timeout_ms": 5_000,
        "credit_ceiling_microunits": 90_000,
    }
    mutation(row)
    document = _signed(
        {
            "schema_version": ROUTING_BINDING_CATALOG_SCHEMA,
            "manifest_uri": uri,
            "catalog_version": "bad",
            "bindings": [row],
            "signature_ref": "s3://lab-routing/signatures/bad.sig",
        }
    )
    with pytest.raises(RoutingProviderBindingError, match=message):
        SignedRoutingBindingCatalogLoader(
            manifest_uri=uri,
            key_id="kms-binding-key",
            loader=lambda _uri: document,
            verifier=_verifier,
        ).load()


def test_signed_binding_rejects_malformed_nested_model_identity():
    binding = _binding()
    uri = "s3://lab-routing/bindings/malformed-identity.json"
    raw_binding = binding.to_dict()
    raw_binding["capability_hash"] = "not-a-hash"
    document = _signed(
        {
            "schema_version": ROUTING_BINDING_CATALOG_SCHEMA,
            "manifest_uri": uri,
            "catalog_version": "malformed",
            "bindings": [
                {
                    "binding": raw_binding,
                    "compiler_family": DEEPLINE_COMPILER_FAMILY,
                    "transport_id": "deepline",
                    "execution_kind": "direct_action",
                    "action_id": "bloomberry_search_job_postings",
                    "workflow_id": None,
                    "workflow_manifest_hash": None,
                    "input_projection": {"domain": "domain"},
                    "input_constants": {},
                    "model_binding_requirements_hash": _MODEL_REQUIREMENTS_HASH,
                    "output_contract_hash": _hash("5"),
                    "evidence_contract_hash": _hash("6"),
                    "retry_policy_hash": _hash("7"),
                    "max_results": 1,
                    "timeout_ms": 5_000,
                    "credit_ceiling_microunits": 90_000,
                }
            ],
            "signature_ref": "s3://lab-routing/signatures/malformed.sig",
        }
    )
    with pytest.raises(RoutingProviderBindingError, match="binding is invalid"):
        SignedRoutingBindingCatalogLoader(
            manifest_uri=uri,
            key_id="kms-binding-key",
            loader=lambda _uri: document,
            verifier=_verifier,
        ).load()


def test_missing_or_extra_signed_unit_ref_fails_before_provider_compilation():
    binding, catalog, units = _authorities()
    compiler = ReviewedDeeplineActionCompiler(binding_catalog=catalog, unit_dataset=units)
    with pytest.raises(RoutingProviderBindingError, match="absent"):
        compiler.prepare(
            binding=binding,
            unit_ref="company-2",
            authorization_credit_microunits=1_800_000,
            authorization_timeout_ms=5_000,
            expected_model_binding_requirements_hash=_MODEL_REQUIREMENTS_HASH,
        )


def test_deepline_success_projects_only_reviewed_evidence_and_authoritative_billing():
    compiler, prepared, request = _prepared()
    broker_result = _broker_result(
        request,
        {
            "result": {
                "data": {
                    "jobs": [
                        {
                            "id": 17,
                            "title": "Machine Learning Engineer",
                            "company_domain": "example.com",
                            "active": True,
                            "snapshot_date": "2026-08-18",
                            "displayed_url": "https://jobs.example.com/ml-engineer",
                            "description": "must never enter the evidence projection",
                        }
                    ]
                }
            },
            "billing": {"credits_charged": 0.09, "cost_usd": 0.009},
        },
    )
    result = compiler.project_result(
        prepared=prepared,
        broker_request=request,
        broker_result=broker_result,
        core_request_fingerprint=_hash("a"),
    )
    assert result["outcome"] == "verified"
    assert result["credit_microunits"] == 90_000
    assert result["billing_state"] == "known"
    assert result["latency_ms"] == 25
    assert set(result) == {
        "outcome", "evidence_hash", "credit_microunits", "latency_ms",
        "billing_state", "binding_id", "provider_id", "tool_id", "request_fingerprint",
    }


def test_direct_action_success_without_exact_deepline_billing_stays_uncertain():
    compiler, prepared, request = _prepared()
    result = compiler.project_result(
        prepared=prepared,
        broker_request=request,
        broker_result=_broker_result(
            request,
            {
                "result": {
                    "data": {
                        "jobs": [
                            {
                                "id": 17,
                                "title": "Machine Learning Engineer",
                                "company_domain": "example.com",
                                "active": True,
                                "snapshot_date": "2026-08-18",
                                "displayed_url": "https://jobs.example.com/ml-engineer",
                            }
                        ]
                    }
                }
            },
        ),
        core_request_fingerprint=_hash("a"),
    )
    assert result["outcome"] == "verified"
    assert result["billing_state"] == "uncertain"
    assert result["credit_microunits"] == 0


def test_broker_identity_mismatch_is_rejected_even_with_valid_body():
    compiler, prepared, request = _prepared()
    result = _broker_result(request, {"result": {"data": {"jobs": []}}, "billing": {"credits_charged": 0}})
    result["transport_attempt"]["provider_id"] = "generic_http"
    with pytest.raises(RoutingProviderBindingError, match="identity differs"):
        compiler.project_result(
            prepared=prepared,
            broker_request=request,
            broker_result=result,
            core_request_fingerprint=_hash("a"),
        )


@pytest.mark.parametrize(
    ("tool_id", "action_id", "unit", "projection", "credit_cap"),
    (
        (
            "intent.source_add.bloomberry_jobs",
            "bloomberry_search_job_postings",
            {
                "company_domain": "example.com",
                "job_keyword": "machine learning",
                "minimum_date": "2026-08-01",
                "maximum_date": "2026-08-31",
            },
            {
                "domain": "company_domain",
                "keyword": "job_keyword",
                "minimum_date": "minimum_date",
                "maximum_date": "maximum_date",
            },
            90_000,
        ),
        (
            "intent.source_add.predictleads_financing",
            "predictleads_company_financing_events",
            {
                "company_domain": "example.com",
                "begin_date": "2026-08-01",
                "end_date": "2026-08-31",
                "minimum_date": "2026-08-01",
                "maximum_date": "2026-08-31",
            },
            {
                "domain": "company_domain",
                "begin_date": "minimum_date",
                "end_date": "maximum_date",
                "minimum_date": "minimum_date",
                "maximum_date": "maximum_date",
            },
            560_000,
        ),
    ),
)
def test_compiler_authority_broker_binds_model_provider_to_deepline_transport(
    tool_id,
    action_id,
    unit,
    projection,
    credit_cap,
):
    from tests.test_provider_broker_v2 import FakeTransport, _broker

    compiler, prepared, _unused_request = _direct_call(
        tool_id=tool_id,
        action_id=action_id,
        unit=unit,
        projection=projection,
        max_results=1 if tool_id.endswith("bloomberry_jobs") else 25,
        credit_cap=credit_cap,
    )
    assert prepared.transport_id == "deepline"
    assert prepared.binding.provider_id != prepared.transport_id

    cryptography = pytest.importorskip("cryptography")
    serialization = pytest.importorskip(
        "cryptography.hazmat.primitives.serialization"
    )
    key = cryptography.hazmat.primitives.asymmetric.ed25519.Ed25519PrivateKey.generate()
    pubkey = key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    ).hex()
    grant = RoutingProviderCallAuthorizationV2(
        admission_job_id="routing-job-1",
        experiment_hash=_hash("a"),
        experiment_id="experiment-1",
        purpose="research_lab.routing_provider_evidence.v2",
        envelope_hash=_hash("b"),
        admission_bundle_hash=_hash("c"),
        protected_release_hash=_hash("d"),
        protected_boot_identity_hash=_hash("e"),
        variant_id="candidate",
        stage="intent_evidence",
        artifact_lineage_hash=_hash("f"),
        pointer_document_hash=_hash("0"),
        model_artifact_hash=_hash("1"),
        manifest_hash=_hash("2"),
        image_digest="registry.example/router@sha256:" + "3" * 64,
        commit_sha="4" * 40,
        build_id="build-1",
        routing_contract_hash=_hash("5"),
        routing_catalog_hash=_hash("6"),
        routing_policy_hash=_hash("7"),
        feature_schema_hash=_hash("8"),
        verifier_contract_hash=_hash("9"),
        binding=prepared.binding,
        transport_id=prepared.transport_id,
        binding_catalog_manifest_hash=prepared.binding_catalog_manifest_hash,
        binding_catalog_version=prepared.binding_catalog_version,
        action_id=prepared.action_id,
        unit_ref=prepared.unit_ref,
        unit_input_hash=prepared.unit_input_hash,
        unit_dataset_manifest_hash=prepared.unit_dataset_manifest_hash,
        unit_set_hash=prepared.unit_set_hash,
        model_binding_observation_receipt_hash=_hash("a"),
        attempt=0,
        core_request_fingerprint=_hash("b"),
        request_body_hash=prepared.request_body_hash,
        retry_policy_hash=prepared.retry_policy_hash,
        credit_cap_microunits=prepared.credit_ceiling_microunits,
        timeout_ms=prepared.timeout_ms,
        claim_key=_hash("c"),
        claim_generation=1,
        claim_fence_hash=_hash("d"),
    )

    def authorize_call(request):
        expected = execute_routing_provider_call_authorization_v2(
            request["payload"], authorization_job_id="routing-authorization-job"
        )
        receipt = create_signed_execution_receipt(
            body=build_execution_receipt_body(
                role="gateway_scoring",
                purpose=grant.purpose,
                job_id="routing-authorization-job",
                epoch_id=4,
                sequence=1,
                commit_sha="4" * 40,
                pcr0="5" * 96,
                build_manifest_hash=_hash("6"),
                dependency_lock_hash=_hash("7"),
                config_hash=_hash("8"),
                boot_identity_hash=_hash("9"),
                input_root=grant.authorization_hash(),
                output_root=expected["output_root"],
                transport_root_hash=_hash("a"),
                host_operation_root_hash=_hash("b"),
                artifact_root=_hash("c"),
                parent_receipt_hashes=(),
                status="succeeded",
                failure_code=None,
                issued_at="2026-08-19T12:00:00Z",
            ),
            enclave_pubkey=pubkey,
            sign_digest=lambda digest: key.sign(digest),
        )
        return {
            "status": "succeeded",
            "operation": request["operation"],
            "purpose": request["purpose"],
            "result": expected,
            "execution_receipt": receipt,
        }

    authorization_response = authorize_call(
        {
            "operation": "attest_routing_provider_call_v2",
            "purpose": grant.purpose,
            "payload": grant.to_dict(),
        }
    )
    authorization_receipt = authorization_response["execution_receipt"]
    proof = {
        "authorization_hash": grant.authorization_hash(),
        "authorization_request_hash": grant.authorization_hash(),
        "authorization_proof_hash": authorization_receipt["receipt_hash"],
        "request_body_hash": grant.request_body_hash,
        "action_id": grant.action_id,
        "credit_cap_microunits": grant.credit_cap_microunits,
        "timeout_ms": grant.timeout_ms,
        "authorization": grant.to_dict(),
        "authorization_result": authorization_response["result"],
        "authorization_receipt": authorization_receipt,
    }
    request = dict(
        compiler.broker_request(
            prepared=prepared,
            experiment_hash=grant.experiment_hash,
            dispatch_job_id=routing_provider_dispatch_job_id_v2(proof),
            variant_id=grant.variant_id,
            attempt_number=grant.attempt,
            core_request_fingerprint=grant.core_request_fingerprint,
            authorization_hash=grant.authorization_hash(),
            authorization_proof_hash=str(proof["authorization_proof_hash"]),
        )
    )
    request["routing_authorization"] = proof
    assert request["provider_id"] == "deepline"
    assert grant.binding.provider_id != request["provider_id"]
    assert request["job_id"] == routing_provider_dispatch_job_id_v2(proof)
    assert request["purpose"] == grant.purpose
    assert request["retry_policy_hash"] == grant.retry_policy_hash
    assert request["attempt_number"] == grant.attempt

    transport = FakeTransport()
    broker = _broker(transport)
    broker.retry_policy_hashes["deepline"] = prepared.retry_policy_hash
    result = broker.execute(request)
    assert result["terminal_status"] == "authenticated_response"
    assert len(transport.calls) == 1


def test_full_binding_identity_not_only_tool_id_is_required():
    binding, catalog, units = _authorities()
    compiler = ReviewedDeeplineActionCompiler(binding_catalog=catalog, unit_dataset=units)
    changed = ProviderBindingIdentity(**{**binding.to_dict(), "capability_hash": _hash("9")})
    with pytest.raises(RoutingProviderBindingError, match="absent"):
        compiler.prepare(
            binding=changed,
            unit_ref="company-1",
            authorization_credit_microunits=1_800_000,
            authorization_timeout_ms=5_000,
            expected_model_binding_requirements_hash=_MODEL_REQUIREMENTS_HASH,
        )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda row: row.update(company_domain="other.example"),
        lambda row: row.update(active=False, inactive=True),
        lambda row: row.update(snapshot_date="2026-07-31"),
        lambda row: row.update(title="Account Executive"),
        lambda row: row.update(displayed_url="https://bloomberry.com/job/17"),
    ],
    ids=("employer", "inactive", "stale", "role", "provider-url"),
)
def test_bloomberry_jobs_rejects_every_unqualified_evidence_dimension(mutation):
    triple = _direct_call(
        tool_id="intent.source_add.bloomberry_jobs",
        action_id="bloomberry_search_job_postings",
        unit={
            "domain": "example.com",
            "role": "Machine Learning",
            "minimum": "2026-08-01",
            "maximum": "2026-08-31",
        },
        projection={
            "domain": "domain",
            "keyword": "role",
            "minimum_date": "minimum",
            "maximum_date": "maximum",
        },
        max_results=1,
        credit_cap=90_000,
    )
    row = {
        "id": 17,
        "title": "Machine Learning Engineer",
        "company_domain": "example.com",
        "active": True,
        "snapshot_date": "2026-08-18",
        "displayed_url": "https://jobs.example.com/ml-engineer",
    }
    mutation(row)
    assert _project(
        triple,
        {"result": {"data": {"jobs": [row]}}, "billing": {"credits_charged": 0.09}},
    )["outcome"] == "source_miss"


def test_bloomberry_jobs_accepts_live_string_inactive_zero():
    triple = _direct_call(
        tool_id="intent.source_add.bloomberry_jobs",
        action_id="bloomberry_search_job_postings",
        unit={
            "domain": "openai.com",
            "role": "intelligence analyst",
            "minimum": "2026-08-01",
            "maximum": "2026-08-20",
        },
        projection={
            "domain": "domain",
            "keyword": "role",
            "minimum_date": "minimum",
            "maximum_date": "maximum",
        },
        max_results=1,
        credit_cap=90_000,
    )
    row = {
        "id": "website_jobs:118433561",
        "title": "Protective Intelligence & Threat Analyst",
        "normalized_job_title": "intelligence analyst",
        "company_domain": "openai.com",
        "inactive": "0",
        "snapshot_date": "2026-08-20",
        "displayed_url": (
            "https://jobs.ashbyhq.com/openai/"
            "4ef8e8f4-a1ef-4d83-92cb-76463ec0a151"
        ),
    }
    result = _project(
        triple,
        {"result": {"data": {"jobs": [row]}}, "billing": {"credits_charged": 0.09}},
    )
    assert result["outcome"] == "verified"
    assert result["credit_microunits"] == 90_000


def test_bloomberry_jobs_requires_domain_role_and_freshness_before_dispatch():
    base = {
        "domain": "example.com",
        "role": "Machine Learning",
        "minimum": "2026-08-01",
        "maximum": "2026-08-31",
    }
    projection = {
        "domain": "domain",
        "keyword": "role",
        "minimum_date": "minimum",
        "maximum_date": "maximum",
    }
    for missing in ("domain", "role", "minimum", "maximum"):
        with pytest.raises(RoutingProviderBindingError, match="required input|context"):
            _direct_call(
                tool_id="intent.source_add.bloomberry_jobs",
                action_id="bloomberry_search_job_postings",
                unit={key: value for key, value in base.items() if key != missing},
                projection=projection,
                max_results=1,
                credit_cap=90_000,
            )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda row: row.update(vendor_name="HubSpot"),
        lambda row: row.update(company_domain="other.example"),
        lambda row: row.update(change_date="2026-07-31"),
        lambda row: row.update(country="CA"),
        lambda row: row.update(vendor_url=""),
    ],
    ids=("selector", "domain", "date", "country", "source"),
)
def test_bloomberry_technology_change_requires_exact_selector_and_evidence(mutation):
    triple = _direct_call(
        tool_id="intent.source_add.bloomberry",
        action_id="bloomberry_get_tech_stack_changes",
        unit={
            "technology": "Salesforce",
            "domain": "example.com",
            "country": "US",
            "minimum": "2026-08-01",
            "maximum": "2026-08-31",
        },
        projection={
            "technology_name": "technology",
            "expected_domain": "domain",
            "expected_country": "country",
            "minimum_date": "minimum",
            "maximum_date": "maximum",
        },
        max_results=1,
        credit_cap=4_210_000,
        execution_mode="replay",
    )
    row = {
        "company_domain": "example.com",
        "vendor_name": "Salesforce",
        "change_date": "2026-08-18",
        "country": "US",
        "vendor_source": "vendor-release",
        "vendor_url": "https://example.com/releases/salesforce",
    }
    mutation(row)
    assert _project(
        triple,
        {"result": {"data": {"signals": [row]}}, "billing": {"credits_charged": 4.21}},
    )["outcome"] == "source_miss"


def test_bloomberry_technology_change_accepts_exact_gated_evidence():
    triple = _direct_call(
        tool_id="intent.source_add.bloomberry",
        action_id="bloomberry_get_tech_stack_changes",
        unit={
            "technology": "Salesforce",
            "domain": "example.com",
            "country": "US",
            "minimum": "2026-08-01",
            "maximum": "2026-08-31",
        },
        projection={
            "technology_name": "technology",
            "expected_domain": "domain",
            "expected_country": "country",
            "minimum_date": "minimum",
            "maximum_date": "maximum",
        },
        max_results=1,
        credit_cap=4_210_000,
        execution_mode="replay",
    )
    assert _project(
        triple,
        {
            "result": {
                "data": {
                    "signals": [
                        {
                            "company_domain": "example.com",
                            "vendor_name": "Salesforce",
                            "change_date": "2026-08-18",
                            "country": "US",
                            "vendor_source": "vendor-release",
                            "vendor_url": "https://example.com/releases/salesforce",
                        }
                    ]
                }
            },
            "billing": {"credits_charged": 4.21},
        },
    )["outcome"] == "verified"


def test_bloomberry_technology_change_is_unavailable_before_measured_dispatch():
    with pytest.raises(RoutingProviderBindingError, match="no domain-scoped action"):
        _direct_call(
            tool_id="intent.source_add.bloomberry",
            action_id="bloomberry_get_tech_stack_changes",
            unit={
                "technology": "Salesforce",
                "domain": "example.com",
                "minimum": "2026-08-01",
                "maximum": "2026-08-31",
            },
            projection={
                "technology_name": "technology",
                "expected_domain": "domain",
                "minimum_date": "minimum",
                "maximum_date": "maximum",
            },
            max_results=1,
            credit_cap=4_210_000,
        )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda row: row["metadata"]["guests"][0].update(guest_company="Other Inc"),
        lambda row: row.update(_search_highlight="Other Inc discusses markets"),
        lambda row: row.update(posted_at="2026-07-31"),
        lambda row: row.update(episode_url="https://podscan.fm/episodes/17"),
    ],
    ids=("guest", "highlight", "date", "directory-url"),
)
def test_podscan_requires_attributable_fresh_public_episode(mutation):
    triple = _direct_call(
        tool_id="intent.source_add.podscan",
        action_id="podscan_episodes_search",
        unit={
            "query": '"Acme Corp" expansion',
            "company": "Acme Corp",
            "minimum": "2026-08-01",
            "maximum": "2026-08-31",
        },
        projection={
            "query": "query",
            "expected_company": "company",
            "minimum_date": "minimum",
            "maximum_date": "maximum",
        },
        max_results=1,
        credit_cap=140_000,
    )
    row = {
        "episode_id": "episode-17",
        "metadata": {"guests": [{"guest_name": "Jane", "guest_company": "Acme Corp"}]},
        "_search_highlight": "Acme Corp discusses expansion",
        "posted_at": "2026-08-18",
        "episode_url": "https://pod.example/episodes/acme-expansion",
    }
    mutation(row)
    assert _project(
        triple,
        {"data": {"episodes": [row]}, "billing": {"credits_charged": 0.14}},
    )["outcome"] == "source_miss"


def test_podscan_accepts_attributable_fresh_public_episode():
    triple = _direct_call(
        tool_id="intent.source_add.podscan",
        action_id="podscan_episodes_search",
        unit={
            "query": '"Acme Corp" expansion',
            "company": "Acme Corp",
            "minimum": "2026-08-01",
            "maximum": "2026-08-31",
        },
        projection={
            "query": "query",
            "expected_company": "company",
            "minimum_date": "minimum",
            "maximum_date": "maximum",
        },
        max_results=1,
        credit_cap=140_000,
    )
    assert _project(
        triple,
        {
            "data": {
                "episodes": [
                    {
                        "episode_id": "episode-17",
                        "episode_title": "Acme Corp expansion interview",
                        "metadata": {
                            "guests": [
                                {"guest_name": "Jane", "guest_company": "Acme Corp"}
                            ]
                        },
                        "podcast": {
                            "podcast_name": "Growth Stories",
                            "podcast_url": "https://pod.example/shows/growth-stories",
                        },
                        "_search_highlight": "Acme Corp discusses expansion",
                        "posted_at": "2026-08-18",
                        "episode_url": "https://pod.example/episodes/acme-expansion",
                    }
                ]
            },
            "billing": {"credits_charged": 0.14},
        },
    )["outcome"] == "verified"


def test_podscan_accepts_live_nested_highlight_with_emphasis_markup():
    triple = _direct_call(
        tool_id="intent.source_add.podscan",
        action_id="podscan_episodes_search",
        unit={
            "query": '"Sam Altman"',
            "person": "Sam Altman",
            "minimum": "2026-04-01",
            "maximum": "2026-04-30",
        },
        projection={
            "query": "query",
            "expected_person": "person",
            "minimum_date": "minimum",
            "maximum_date": "maximum",
        },
        max_results=1,
        credit_cap=140_000,
    )
    row = {
        "episode_id": "ep-live-shape",
        "episode_title": "A conversation with Sam Altman",
        "metadata": {
            "guests": [{"guest_name": "Sam Altman", "guest_company": "OpenAI"}]
        },
        "_search_highlight": {
            "transcription": "A conversation with <em>Sam</em> <em>Altman</em>."
        },
        "posted_at": "2026-04-25T01:56:05+00:00",
        "episode_url": "https://pod.example/episodes/sam-altman",
    }
    result = _project(
        triple,
        {"episodes": [row], "billing": {"credits_charged": 0.14}},
    )
    assert result["outcome"] == "verified"
    assert result["credit_microunits"] == 140_000


@pytest.mark.parametrize("query", ("Acme Corp expansion", '"Other Inc" expansion'))
def test_podscan_rejects_unquoted_or_unrelated_query_before_dispatch(query):
    with pytest.raises(RoutingProviderBindingError, match="signed subject"):
        _direct_call(
            tool_id="intent.source_add.podscan",
            action_id="podscan_episodes_search",
            unit={
                "query": query,
                "company": "Acme Corp",
                "minimum": "2026-08-01",
                "maximum": "2026-08-31",
            },
            projection={
                "query": "query",
                "expected_company": "company",
                "minimum_date": "minimum",
                "maximum_date": "maximum",
            },
            max_results=1,
            credit_cap=140_000,
        )


def _predictleads_direct(*, jobs: bool):
    action = (
        "predictleads_company_job_openings"
        if jobs
        else "predictleads_company_financing_events"
    )
    tool = (
        "intent.source_add.predictleads_jobs"
        if jobs
        else "intent.source_add.predictleads_financing"
    )
    unit = {
        "domain": "example.com",
        "minimum": "2026-08-01",
        "maximum": "2026-08-31",
    }
    projection = {
        "domain": "domain",
        "begin_date": "minimum",
        "end_date": "maximum",
        "minimum_date": "minimum",
        "maximum_date": "maximum",
    }
    if jobs:
        unit["role"] = "Machine Learning"
        projection["role_keyword"] = "role"
    return _direct_call(
        tool_id=tool,
        action_id=action,
        unit=unit,
        projection=projection,
        max_results=25,
        credit_cap=560_000,
    )


@pytest.mark.parametrize(
    ("jobs", "mutation"),
    [
        (False, lambda row, included: included[0]["attributes"].update(domain="other.example")),
        (False, lambda row, included: row.update(type="job_opening")),
        (False, lambda row, included: row["attributes"].update(effective_date="2026-07-31")),
        (False, lambda row, included: row["attributes"].update(source_urls=[])),
        (True, lambda row, included: included[0]["attributes"].update(domain="other.example")),
        (True, lambda row, included: row.update(type="financing_event")),
        (True, lambda row, included: row["attributes"].update(status="closed")),
        (True, lambda row, included: row["attributes"].update(posted_at="2026-07-31")),
        (True, lambda row, included: row["attributes"].update(title="Account Executive")),
        (True, lambda row, included: row["attributes"].update(url="")),
    ],
    ids=(
        "finance-company", "finance-cross-type", "finance-date", "finance-source",
        "jobs-company", "jobs-cross-type", "jobs-status", "jobs-date", "jobs-role", "jobs-url",
    ),
)
def test_predictleads_direct_tools_require_exact_relationship_and_evidence(jobs, mutation):
    triple = _predictleads_direct(jobs=jobs)
    attributes = (
        {
            # PredictLeads uses null for a currently open job.  The only
            # non-null status in its live output contract is "closed".
            "status": None,
            "title": "Machine Learning Engineer",
            "posted_at": "2026-08-18",
            "last_seen_at": "2026-08-31",
            "url": "https://jobs.example.com/ml-engineer",
        }
        if jobs
        else {
            "effective_date": "2026-08-18",
            "financing_type": "series_a",
            "source_urls": ["https://news.example/funding"],
        }
    )
    row = {
        "type": "job_opening" if jobs else "financing_event",
        "id": "event-1",
        "attributes": attributes,
        "relationships": {"company": {"data": {"type": "company", "id": "co-1"}}},
    }
    included = [{"type": "company", "id": "co-1", "attributes": {"domain": "example.com"}}]
    mutation(row, included)
    assert _project(
        triple,
        {"result": {"data": [row], "included": included}, "billing": {"credits_charged": 0.56}},
    )["outcome"] == "source_miss"


@pytest.mark.parametrize("jobs", (False, True), ids=("financing", "jobs"))
def test_predictleads_direct_tools_accept_exact_gated_evidence(jobs):
    triple = _predictleads_direct(jobs=jobs)
    attributes = (
        {
            "status": None,
            "title": "Machine Learning Engineer",
            "posted_at": "2026-08-18",
            "last_seen_at": "2026-08-31",
            "url": "https://jobs.example.com/ml-engineer",
        }
        if jobs
        else {
            "effective_date": "2026-08-18",
            "financing_type": "series_a",
            "source_urls": ["https://news.example/funding"],
        }
    )
    row = {
        "type": "job_opening" if jobs else "financing_event",
        "id": "event-1",
        "attributes": attributes,
        "relationships": {"company": {"data": {"type": "company", "id": "co-1"}}},
    }
    assert _project(
        triple,
        {
            "result": {
                "data": [row],
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
    )["outcome"] == "verified"


def test_predictleads_jobs_accepts_live_deepline_response_shape():
    """Keep the measured Deepline/PredictLeads shape in the contract matrix."""

    triple = _direct_call(
        tool_id="intent.source_add.predictleads_jobs",
        action_id="predictleads_company_job_openings",
        unit={
            "domain": "openai.com",
            "role": "Growth",
            "minimum": "2026-08-01",
            "maximum": "2026-08-19",
        },
        projection={
            "domain": "domain",
            "begin_date": "minimum",
            "end_date": "maximum",
            "minimum_date": "minimum",
            "maximum_date": "maximum",
            "role_keyword": "role",
        },
        max_results=1,
        credit_cap=560_000,
    )
    row = {
        "type": "job_opening",
        "id": "f4d86c7f-d0c9-4ac9-be0b-493b23fd8c7e",
        "attributes": {
            "status": None,
            "title": "New Geography and International Growth Lead",
            "normalized_title": "New Geography and International Growth Lead",
            "posted_at": None,
            "first_seen_at": "2026-08-14T10:00:00Z",
            "last_seen_at": "2026-08-19T16:00:00Z",
            "url": "https://www.linkedin.com/jobs/view/4450932685/",
        },
        "relationships": {
            "company": {
                "data": {
                    "type": "company",
                    "id": "62b85623-6793-55e1-a71b-4c3333599b2d",
                }
            }
        },
    }
    result = _project(
        triple,
        {
            "result": {
                "data": [row],
                "included": [
                    {
                        "type": "company",
                        "id": "62b85623-6793-55e1-a71b-4c3333599b2d",
                        "attributes": {"domain": "openai.com"},
                    }
                ],
            },
            "billing": {"credits_charged": 0.56, "cost_usd": 0.056},
        },
    )
    assert result["outcome"] == "verified"
    assert result["credit_microunits"] == 560_000


@pytest.mark.parametrize(
    ("status", "last_seen_at", "expected_outcome"),
    [
        (None, "2026-08-31", "verified"),
        (None, "2026-08-26", "verified"),
        (None, "2026-08-25", "source_miss"),
        (None, "2026-09-01", "source_miss"),
        ("closed", "2026-08-31", "source_miss"),
    ],
    ids=("cutoff", "cutoff-minus-five", "cutoff-minus-six", "after-cutoff", "closed"),
)
def test_predictleads_jobs_uses_signed_cutoff_for_current_provider_records(
    status, last_seen_at, expected_outcome
):
    triple = _predictleads_direct(jobs=True)
    row = {
        "type": "job_opening",
        "id": "event-1",
        "attributes": {
            "status": status,
            "title": "Machine Learning Engineer",
            "posted_at": "2026-08-18",
            "last_seen_at": last_seen_at,
            "url": "https://jobs.example.com/ml-engineer",
        },
        "relationships": {"company": {"data": {"type": "company", "id": "co-1"}}},
    }
    included = [{"type": "company", "id": "co-1", "attributes": {"domain": "example.com"}}]
    assert _project(
        triple,
        {"result": {"data": [row], "included": included}, "billing": {"credits_charged": 0.56}},
    )["outcome"] == expected_outcome


def test_predictleads_aliases_and_caps_are_exact():
    _compiler, prepared, _request = _predictleads_direct(jobs=True)
    assert prepared.payload["limit"] == 25
    assert prepared.payload["first_seen_at_from"] == "2026-08-01"
    assert prepared.payload["first_seen_at_until"] == "2026-08-31"
    assert "first_seen_at_to" not in prepared.payload
    assert prepared.payload["active_only"] is True
    assert prepared.payload["not_closed"] is True


def test_builtwith_is_confirmation_only_and_requires_exact_history_and_billing():
    kwargs = dict(
        tool_id="intent.source_add.builtwith",
        action_id="builtwith_domain_lookup",
        unit={
            "domain": "example.com",
            "technology": "Salesforce",
            "parent": _hash("d"),
            "first_from": "2025-01-01",
            "first_until": "2025-12-31",
            "last_from": "2026-08-01",
            "last_until": "2026-08-31",
        },
        projection={
            "domain": "domain",
            "requested_technology": "technology",
            "parent_intent_event_hash": "parent",
            "first_detected_from": "first_from",
            "first_detected_until": "first_until",
            "last_detected_from": "last_from",
            "last_detected_until": "last_until",
        },
        max_results=1,
        credit_cap=5_000_000,
    )
    with pytest.raises(RoutingProviderBindingError, match="confirmation-only"):
        _direct_call(**kwargs)
    with pytest.raises(RoutingProviderBindingError, match="no provider-enforced cost cap"):
        _direct_call(**kwargs, phase="conditional_confirmation")
    triple = _direct_call(
        **kwargs,
        phase="conditional_confirmation",
        execution_mode="replay",
    )
    row = {
        "Lookup": "example.com",
        "Result": {
            "Paths": [
                {
                    "Domain": "example.com",
                    "Technologies": [
                        {
                            "Name": "Salesforce",
                            "FirstDetected": "2025-01-01",
                            "LastDetected": "2026-08-18",
                        }
                    ],
                }
            ]
        },
    }
    assert _project(
        triple,
        {"result": {"data": {"Results": [row]}}, "billing": {"credits_charged": 0.2}},
    )["outcome"] == "verified"
    row["Result"]["Paths"][0]["Technologies"][0]["Name"] = "HubSpot"
    assert _project(
        triple,
        {"result": {"data": {"Results": [row]}}, "billing": {"credits_charged": 0.2}},
    )["outcome"] == "source_miss"


def test_builtwith_rejects_history_outside_signed_detection_range():
    triple = _direct_call(
        tool_id="intent.source_add.builtwith",
        action_id="builtwith_domain_lookup",
        unit={
            "domain": "example.com",
            "technology": "Salesforce",
            "parent": _hash("d"),
            "first_from": "2025-01-01",
            "first_until": "2025-12-31",
            "last_from": "2026-08-01",
            "last_until": "2026-08-31",
        },
        projection={
            "domain": "domain",
            "requested_technology": "technology",
            "parent_intent_event_hash": "parent",
            "first_detected_from": "first_from",
            "first_detected_until": "first_until",
            "last_detected_from": "last_from",
            "last_detected_until": "last_until",
        },
        max_results=1,
        credit_cap=5_000_000,
        phase="conditional_confirmation",
        execution_mode="replay",
    )
    row = {
        "Lookup": "example.com",
        "Result": {
            "Paths": [
                {
                    "Domain": "example.com",
                    "Technologies": [
                        {
                            "Name": "Salesforce",
                            "FirstDetected": "2024-12-31",
                            "LastDetected": "2026-08-18",
                        }
                    ],
                }
            ]
        },
    }
    assert _project(
        triple,
        {"result": {"data": {"Results": [row]}}, "billing": {"credits_charged": 0.2}},
    )["outcome"] == "source_miss"


@pytest.mark.parametrize(
    ("tool_id", "message"),
    (
        ("intent.source_add.predictleads_connections", "unreviewed action"),
        ("intent.source_add.predictleads_news", "unreviewed action"),
        ("intent.source_add.predictleads_technology", "unreviewed action"),
        ("intent.source_add.sumble", "explicitly unavailable"),
        ("candidate.source_add.unreviewed", "unreviewed action"),
    ),
)
def test_composite_and_unknown_source_tools_are_rejected_at_catalog_admission(
    tool_id, message
):
    binding = _binding(tool_id=tool_id)
    uri = "s3://lab-routing/bindings/unavailable.json"
    row = {
        "binding": binding.to_dict(),
        "compiler_family": DEEPLINE_COMPILER_FAMILY,
        "transport_id": "deepline",
        "execution_kind": "direct_action",
        "action_id": "predictleads_company_connections",
        "workflow_id": None,
        "workflow_manifest_hash": None,
        "input_projection": {"domain": "domain"},
        "input_constants": {},
        "model_binding_requirements_hash": _MODEL_REQUIREMENTS_HASH,
        "output_contract_hash": _hash("5"),
        "evidence_contract_hash": _hash("6"),
        "retry_policy_hash": _hash("7"),
        "max_results": 1,
        "timeout_ms": 5_000,
        "credit_ceiling_microunits": 560_000,
    }
    document = _signed(
        {
            "schema_version": ROUTING_BINDING_CATALOG_SCHEMA,
            "manifest_uri": uri,
            "catalog_version": "unavailable",
            "bindings": [row],
            "signature_ref": "s3://lab-routing/signatures/unavailable.sig",
        }
    )
    with pytest.raises(RoutingProviderBindingError, match=message):
        SignedRoutingBindingCatalogLoader(
            manifest_uri=uri,
            key_id="kms-binding-key",
            loader=lambda _uri: document,
            verifier=_verifier,
        ).load()


@pytest.mark.parametrize("fixed_field", ("limit", "active_only", "include_transcript", "no_pii"))
def test_signed_projection_cannot_override_code_fixed_inputs(fixed_field):
    binding = _binding()
    uri = "s3://lab-routing/bindings/fixed-override.json"
    action = (
        "podscan_episodes_search"
        if fixed_field == "include_transcript"
        else "builtwith_domain_lookup"
        if fixed_field == "no_pii"
        else "bloomberry_search_job_postings"
    )
    tool = (
        "intent.source_add.podscan"
        if action.startswith("podscan")
        else "intent.source_add.builtwith"
        if action.startswith("builtwith")
        else "intent.source_add.bloomberry_jobs"
    )
    binding = _binding(tool_id=tool)
    row = {
        "binding": binding.to_dict(),
        "compiler_family": DEEPLINE_COMPILER_FAMILY,
        "transport_id": "deepline",
        "execution_kind": "direct_action",
        "action_id": action,
        "workflow_id": None,
        "workflow_manifest_hash": None,
        "input_projection": {fixed_field: "caller_value"},
        "input_constants": {},
        "model_binding_requirements_hash": _MODEL_REQUIREMENTS_HASH,
        "output_contract_hash": _hash("5"),
        "evidence_contract_hash": _hash("6"),
        "retry_policy_hash": _hash("7"),
        "max_results": 1,
        "timeout_ms": 5_000,
        "credit_ceiling_microunits": 90_000,
    }
    document = _signed(
        {
            "schema_version": ROUTING_BINDING_CATALOG_SCHEMA,
            "manifest_uri": uri,
            "catalog_version": "fixed",
            "bindings": [row],
            "signature_ref": "s3://lab-routing/signatures/fixed.sig",
        }
    )
    with pytest.raises(RoutingProviderBindingError, match="not reviewed"):
        SignedRoutingBindingCatalogLoader(
            manifest_uri=uri,
            key_id="kms-binding-key",
            loader=lambda _uri: document,
            verifier=_verifier,
        ).load()
