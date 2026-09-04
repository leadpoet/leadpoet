import base64
import time
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from pydantic import ValidationError

from gateway.research_lab import api, source_add_catalog
from gateway.research_lab.models import (
    AttestedCredentialCiphertextV2,
    ResearchLabSourceAdapterProbeConfigureRequest,
    ResearchLabSourceAdapterProvisionRequest,
    ResearchLabSourceAdapterSubmissionRequest,
    ResearchLabSourceMetadata,
    ResearchLabSourceAddCredentialRecipientRequest,
)
from gateway.research_lab.source_add_catalog import (
    PROVISION_STATUS_APPROVED_PENDING,
    PROVISION_STATUS_ELIGIBLE,
    provider_registry_entries_from_provisioned_rows,
    probe_endpoints_from_provisioned_rows,
    sanitize_source_add_doc,
    source_add_row_credential_ready,
)
from gateway.research_lab.source_add_llm_judge import _parse_verdict
from gateway.research_lab.source_add_provenance import PRECHECK_MANUAL, PRECHECK_PASSED
from research_lab.source_add_execution import SourceAddRejectionReason, intake_source_add_submission
from research_lab.source_add_identity import (
    normalize_source_add_domain,
    normalize_source_add_provider_origin,
    normalize_source_add_url,
    source_documentation_identity_hash,
    source_identity_alias_hashes_from_metadata,
    source_identity_hash,
    source_provider_origin_hash,
)


def _manifest_doc(**overrides):
    doc = {
        "adapter_id": "adapter:test-source",
        "miner_ref": "miner:hotkey",
        "source_name": "Test Source",
        "source_kind": "news",
        "declared_base_domains": ["api.test-source.example"],
        "output_schema_ref": "schema:source-add-output:v1",
        "allowed_output_fields": ["evidence_refs", "snapshot_refs", "content_hashes", "normalized_text_hashes"],
        "submitted_artifact_ref": "artifact:test",
        "code_bundle_hash": "sha256:" + "a" * 64,
        "sandbox_policy_ref": "policy:sandbox-v1",
        "max_trial_cost_cents": 500,
        "max_request_cost_cents": 5,
        "max_latency_ms": 30_000,
        "fixture_refs": ["fixture:test"],
        "credential_policy": "no_credentials",
    }
    doc.update(overrides)
    return doc


def _source_metadata_doc(**overrides):
    doc = {
        "api_base_url": "https://api.test-source.example",
        "documentation_url": "https://docs.test-source.example/api",
        "auth_type": "none",
        "endpoint_examples": [
            {
                "method": "GET",
                "path": "/search",
                "purpose": "Search current source records",
                "example_query": "q=test",
            }
        ],
        "rate_limit_notes": "Use conservative request pacing.",
        "data_provenance_notes": "Official source records.",
        "third_party_refs": [],
    }
    doc.update(overrides)
    return doc


FAKE_BUILTWITH_CREDENTIAL = "FAKE_BUILTWITH_VALUE_12345"


@pytest.mark.asyncio
async def test_public_source_add_credential_recipient_is_retired(monkeypatch):
    monkeypatch.setattr(
        api.ResearchLabGatewayConfig,
        "from_env",
        staticmethod(lambda: SimpleNamespace(api_enabled=True, source_add_enabled=True)),
    )
    monkeypatch.setattr(api, "_verify_signed_miner", lambda _payload: _async_none())
    async def fail_recipient(**_kwargs):
        raise AssertionError("public miner route must not create a recipient")

    monkeypatch.setattr(api, "_source_add_credential_recipient", fail_recipient)
    payload = ResearchLabSourceAddCredentialRecipientRequest(
        miner_hotkey="miner-hotkey-value",
        signature="signature-value-123",
        timestamp=int(time.time()),
        idempotency_key="recipient-request-1",
        adapter_id="adapter:test-source",
    )
    with pytest.raises(api.HTTPException) as exc_info:
        await api.create_source_add_credential_recipient(payload)
    assert exc_info.value.status_code == 410
    assert exc_info.value.detail == "SOURCE_ADD miner credentials are not accepted"


async def _async_none():
    return None


async def _async_value(value):
    return value


@pytest.mark.asyncio
async def test_source_add_rejects_plaintext_and_v2_miner_credentials():
    common = {
        "miner_hotkey": "miner-hotkey-value",
        "signature": "signature-value-123",
        "timestamp": int(time.time()),
        "idempotency_key": "source-submit-1",
        "manifest": _manifest_doc(),
        "source_metadata": _source_metadata_doc(),
    }
    with pytest.raises(
        ValidationError,
        match="credential material|miners must not submit",
    ):
        ResearchLabSourceAdapterSubmissionRequest(
            **common,
            adapter_credential="plaintext-secret-value",
        )
    encrypted = AttestedCredentialCiphertextV2(
        request_id="sha256:" + "8" * 64,
        ciphertext_b64=base64.b64encode(b"x" * 384).decode(),
    )
    with pytest.raises(
        ValidationError,
        match="credential material|miners must not submit",
    ):
        ResearchLabSourceAdapterSubmissionRequest(
            **common,
            adapter_credential_v2=encrypted,
        )


@pytest.mark.parametrize(
    "body",
    (
        {
            "miner_hotkey": "miner-hotkey-value",
            "signature": "signature-value-123",
            "timestamp": "not-an-integer",
            "idempotency_key": "source-submit-malformed-builtwith",
            "manifest": _manifest_doc(),
            "source_metadata": {
                **_source_metadata_doc(),
                "documentation_url": (
                    "https://api.builtwith.com/docs?"
                    f"KEY={FAKE_BUILTWITH_CREDENTIAL}"
                ),
            },
        },
        b'{"source_metadata":{"documentation_url":"KEY='
        + FAKE_BUILTWITH_CREDENTIAL.encode("utf-8"),
    ),
)
def test_source_add_http_validation_errors_are_generic_and_never_echo_credentials(
    body,
):
    app = FastAPI()
    app.include_router(api.router)
    client = TestClient(app, raise_server_exceptions=False)

    if isinstance(body, bytes):
        response = client.post(
            "/research-lab/source-adapters",
            content=body,
            headers={"Content-Type": "application/json"},
        )
    else:
        response = client.post("/research-lab/source-adapters", json=body)

    assert response.status_code == 400
    assert response.json() == {"detail": "Submission failed"}
    assert FAKE_BUILTWITH_CREDENTIAL not in response.text


@pytest.mark.parametrize(
    "credential_location",
    (
        "top_level_unknown",
        "metadata_unknown",
        "endpoint_unknown",
    ),
)
def test_source_add_unknown_credential_fields_fail_before_route_execution(
    monkeypatch,
    credential_location,
):
    async def fail_verify(_payload):
        raise AssertionError("credential request must not reach signature verification")

    monkeypatch.setattr(api, "_verify_signed_miner", fail_verify)
    body = {
        "miner_hotkey": "5" + "A" * 47,
        "signature": "signature-value-123",
        "timestamp": int(time.time()),
        "idempotency_key": f"source-unknown-{credential_location}",
        "manifest": _manifest_doc(),
        "source_brief": "Official provider documentation.",
        "source_metadata": _source_metadata_doc(),
    }
    if credential_location == "top_level_unknown":
        body["xApiKey"] = FAKE_BUILTWITH_CREDENTIAL
    elif credential_location == "metadata_unknown":
        body["source_metadata"]["xApiKey"] = FAKE_BUILTWITH_CREDENTIAL
    else:
        body["source_metadata"]["endpoint_examples"][0][
            "xApiKey"
        ] = FAKE_BUILTWITH_CREDENTIAL

    app = FastAPI()
    app.include_router(api.router)
    response = TestClient(app, raise_server_exceptions=False).post(
        "/research-lab/source-adapters",
        json=body,
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "Submission failed"}
    assert FAKE_BUILTWITH_CREDENTIAL not in response.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "credential_location",
    (
        "manifest_nested",
        "manifest_authorization_header",
        "manifest_common_api_header",
        "manifest_camel_case_api_header",
        "source_brief",
        "endpoint_example",
    ),
)
async def test_builtwith_key_is_rejected_generically_after_signature_before_catalog_or_rpc(
    monkeypatch,
    credential_location,
):
    monkeypatch.setattr(
        api.ResearchLabGatewayConfig,
        "from_env",
        staticmethod(
            lambda: SimpleNamespace(
                api_enabled=True,
                production_writes_enabled=True,
                miner_submissions_enabled=True,
                source_add_enabled=True,
                source_add_max_concurrent_per_hotkey=3,
                source_add_max_per_day_per_hotkey=5,
                source_add_max_per_30d_per_hotkey=10,
            )
        ),
    )
    monkeypatch.setattr(
        api,
        "source_add_control_state",
        lambda *a, **k: _async_value({"paused": False, "status": "active"}),
    )

    manifest = _manifest_doc()
    source_brief = "Official BuiltWith API documentation."
    metadata = _source_metadata_doc(
        api_base_url="https://api.builtwith.com/v21",
        documentation_url="https://api.builtwith.com/free-api",
        auth_type="api_key_query",
    )
    marker = f"KEY={FAKE_BUILTWITH_CREDENTIAL}"
    if credential_location == "manifest_nested":
        manifest["submission_notes"] = {"example": marker}
    elif credential_location == "manifest_authorization_header":
        manifest["request_headers"] = {
            "Authorization": f"Bearer {FAKE_BUILTWITH_CREDENTIAL}"
        }
    elif credential_location == "manifest_common_api_header":
        manifest["request_headers"] = {
            "X-RapidAPI-Key": FAKE_BUILTWITH_CREDENTIAL
        }
    elif credential_location == "manifest_camel_case_api_header":
        manifest["request_headers"] = {
            "xRapidApiKey": FAKE_BUILTWITH_CREDENTIAL
        }
    elif credential_location == "source_brief":
        source_brief = f"BuiltWith request example: {marker}"
    else:
        metadata["endpoint_examples"][0]["example_query"] = (
            f"{marker}&LOOKUP=example.com"
        )

    payload = ResearchLabSourceAdapterSubmissionRequest.model_construct(
        miner_hotkey="miner-hotkey-value",
        signature="signature-value-123",
        timestamp=int(time.time()),
        idempotency_key=f"source-submit-builtwith-{credential_location}",
        manifest=manifest,
        source_brief=source_brief,
        source_metadata=ResearchLabSourceMetadata.model_validate(metadata),
    )

    async def fail_verify_signed_miner(_value):
        raise AssertionError(
            "credential-bearing request must be rejected before signature logging"
        )

    def fail_builtin(*_args, **_kwargs):
        raise AssertionError("credential-bearing request must not reach catalog lookup")

    async def fail_rpc(*_args, **_kwargs):
        raise AssertionError("credential-bearing request must not reach persistence")

    monkeypatch.setattr(api, "_verify_signed_miner", fail_verify_signed_miner)
    monkeypatch.setattr(
        source_add_catalog,
        "source_add_api_is_current_builtin_sync",
        fail_builtin,
    )
    monkeypatch.setattr(api, "_source_add_rpc", fail_rpc)

    with pytest.raises(api.HTTPException) as exc_info:
        await api.submit_research_lab_source_adapter(payload)

    body = JSONResponse(
        status_code=exc_info.value.status_code,
        content={"detail": exc_info.value.detail},
    ).body
    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Submission failed"
    assert body == b'{"detail":"Submission failed"}'
    assert FAKE_BUILTWITH_CREDENTIAL.encode() not in body


def test_source_add_sanitizer_recursively_redacts_builtwith_key_values():
    marker = f"KEY={FAKE_BUILTWITH_CREDENTIAL}"
    sanitized = sanitize_source_add_doc(
        {
            "manifest": {
                "submission_notes": {
                    "example": f"/free1/api.json/{marker}/LOOKUP=example.com"
                }
            },
            "source_metadata": {
                "documentation_url": (
                    f"https://api.builtwith.com/docs?{marker}"
                ),
                "endpoint_examples": [
                    {
                        "example_query": f"{marker}&LOOKUP=example.com",
                        "purpose": "Return key technology signals",
                    }
                ],
            },
            "safe_summary": "Key technology signals are documented.",
        }
    )

    assert FAKE_BUILTWITH_CREDENTIAL not in str(sanitized)
    assert sanitized["manifest"]["submission_notes"]["example"] == "[redacted]"
    assert sanitized["source_metadata"]["documentation_url"] == "[redacted]"
    assert (
        sanitized["source_metadata"]["endpoint_examples"][0]["example_query"]
        == "[redacted]"
    )
    assert sanitized["safe_summary"] == "Key technology signals are documented."


def test_source_add_sanitizer_redacts_generic_auth_header_key():
    sanitized = sanitize_source_add_doc(
        {"headers": {"X-Custom-Auth": FAKE_BUILTWITH_CREDENTIAL}}
    )

    assert sanitized == {"headers": {"X-Custom-Auth": "[redacted]"}}


def test_intake_rejects_duplicate_source_identity_hash():
    identity = source_identity_hash(
        api_base_url="https://api.test-source.example/v1",
        documentation_url="https://docs.test-source.example",
        declared_base_domains=["api.test-source.example"],
    )
    record, errors = intake_source_add_submission(
        _manifest_doc(),
        miner_hotkey="hk-test",
        source_identity_ref=identity,
        existing_source_identity_hashes=[identity],
    )
    assert record is None
    assert SourceAddRejectionReason.DUPLICATE_SOURCE in errors


def test_v2_api_identity_cannot_be_bypassed_by_changing_documentation():
    original = source_identity_hash(
        api_base_url="https://API.test-source.example/v1/",
        documentation_url="https://docs.test-source.example/docs/quickstart",
        declared_base_domains=["api.test-source.example"],
    )
    changed_docs = source_identity_hash(
        api_base_url="https://api.test-source.example/v1",
        documentation_url="https://attacker.example/reference",
        declared_base_domains=["attacker.example"],
    )
    different_api_path = source_identity_hash(
        api_base_url="https://api.test-source.example/v2",
        documentation_url="https://docs.test-source.example/docs",
    )

    assert original == changed_docs
    assert original != different_api_path


def test_provider_origin_identity_is_path_independent_but_subdomain_specific():
    first_path = source_provider_origin_hash(
        "https://API.test-source.example/v1"
    )
    second_path = source_provider_origin_hash(
        "https://api.test-source.example/v2/search"
    )
    distinct_subdomain = source_provider_origin_hash(
        "https://data.test-source.example/v1"
    )

    assert normalize_source_add_provider_origin(
        "https://api.test-source.example/v3"
    ) == "api.test-source.example"
    assert first_path == second_path
    assert first_path != distinct_subdomain


def test_current_model_gate_uses_exact_submitted_and_tested_origin(monkeypatch):
    from gateway.research_lab import provider_evidence_proxy

    monkeypatch.setattr(
        provider_evidence_proxy,
        "reserved_builtin_provider_domains_sync",
        lambda: {"openrouter.ai"},
    )

    assert source_add_catalog.source_add_api_is_current_builtin_sync(
        "https://openrouter.ai/api/v1",
        tested_base_url="https://OPENROUTER.ai:443/api/v2",
    ) is True
    assert source_add_catalog.source_add_api_is_current_builtin_sync(
        "https://api.new-source.example/v1",
        tested_base_url="https://api.new-source.example/v2",
    ) is False
    with pytest.raises(ValueError, match="submitted/tested provider origin differs"):
        source_add_catalog.source_add_api_is_current_builtin_sync(
            "https://api.new-source.example/v1",
            tested_base_url="https://other.new-source.example/v1",
        )


@pytest.mark.parametrize(
    ("value", "expected"),
    (
        ("https://api.test-source.example:443/v1", "api.test-source.example"),
        ("https://www.api.test-source.example./v1", "api.test-source.example"),
        ("https://[2001:0db8:0000::1]/v1", "2001:db8::1"),
        ("https://[::ffff:192.0.2.1]/v1", ""),
        ("https://[::ffff:c000:201]/v1", ""),
        ("https://192.0.2.1/v1", "192.0.2.1"),
        ("https://b\N{LATIN SMALL LETTER U WITH DIAERESIS}cher.example/v1", ""),
        ("https://localhost/v1", ""),
        ("https://api.test-source.example:8443/v1", ""),
        ("https://127.1/v1", ""),
        ("https://api.test-source.example/path with space", ""),
    ),
)
def test_provider_origin_normalization_has_one_strict_exact_host_contract(
    value, expected
):
    assert normalize_source_add_provider_origin(value) == expected
    assert bool(source_provider_origin_hash(value)) is bool(expected)


def test_v2_documentation_alias_is_reserved_separately_and_stably():
    first = source_documentation_identity_hash(
        "https://docs.test-source.example/docs/quickstart"
    )
    moved = source_documentation_identity_hash(
        "https://docs.test-source.example/docs/reference/auth"
    )
    metadata_aliases = source_identity_alias_hashes_from_metadata(
        {"documentation_url": "https://docs.test-source.example/docs/latest"}
    )

    assert first == moved
    assert metadata_aliases == (first,)


def test_source_identity_normalizes_ipv6_without_truncating_host():
    assert normalize_source_add_domain("https://[2001:db8::1]/v1") == "2001:db8::1"
    assert normalize_source_add_domain("2001:db8::1") == "2001:db8::1"
    assert normalize_source_add_url("https://[2001:db8::1]/v1/") == (
        "https://[2001:db8::1]/v1"
    )


@pytest.mark.asyncio
async def test_submission_delegates_identity_and_limits_to_atomic_rpc(monkeypatch):
    monkeypatch.setattr(
        api.ResearchLabGatewayConfig,
        "from_env",
        staticmethod(
            lambda: SimpleNamespace(
                api_enabled=True,
                production_writes_enabled=True,
                miner_submissions_enabled=True,
                source_add_enabled=True,
                source_add_max_concurrent_per_hotkey=3,
                source_add_max_per_day_per_hotkey=5,
                source_add_max_per_30d_per_hotkey=10,
            )
        ),
    )
    monkeypatch.setattr(api, "_verify_signed_miner", lambda _payload: _async_none())
    monkeypatch.setattr(
        api,
        "source_add_control_state",
        lambda *a, **k: _async_value({"paused": False, "status": "active"}),
    )
    monkeypatch.setattr(
        source_add_catalog,
        "source_add_api_is_current_builtin_sync",
        lambda *_args, **_kwargs: False,
    )
    observed = {}

    async def fake_rpc(name, params):
        observed["name"] = name
        observed["params"] = dict(params)
        return {
            "status": "admitted",
            "stage": "provenance_queued",
            "work_id": params["p_work_id"],
        }

    monkeypatch.setattr(api, "_source_add_rpc", fake_rpc)
    payload = ResearchLabSourceAdapterSubmissionRequest(
        miner_hotkey="miner-hotkey-value",
        signature="signature-value-123",
        timestamp=int(time.time()),
        idempotency_key="source-submit-atomic-1",
        manifest=_manifest_doc(),
        source_metadata=_source_metadata_doc(),
    )

    response = await api.submit_research_lab_source_adapter(payload)

    assert response.stage == "provenance_queued"
    assert observed["name"] == "research_lab_source_add_admit_v3"
    assert observed["params"]["p_max_open"] == 3
    assert observed["params"]["p_max_day"] == 5
    assert observed["params"]["p_max_30d"] == 10
    assert observed["params"]["p_cooldown_seconds"] == (
        api._SOURCE_ADD_SUBMISSION_COOLDOWN_SECONDS
    )
    assert observed["params"]["p_documentation_identity_hash"].startswith(
        "sha256:"
    )
    assert observed["params"]["p_provider_origin_hash"].startswith("sha256:")
    assert observed["params"]["p_record_doc"]["provider_origin_host"] == (
        "api.test-source.example"
    )
    assert observed["params"]["p_record_doc"]["provider_origin_hash"] == (
        observed["params"]["p_provider_origin_hash"]
    )
    assert observed["params"]["p_record_doc"]["manifest"]["credential_policy"] == "no_credentials"
    assert observed["params"]["p_record_doc"]["credential_envelope"] == {}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalid_case",
    (
        "miner_credential_field",
        "declared_domains",
        "source_identity",
        "manifest_intake",
    ),
)
async def test_invalid_source_add_route_failures_are_generic_and_never_persist(
    monkeypatch,
    invalid_case,
):
    monkeypatch.setattr(
        api.ResearchLabGatewayConfig,
        "from_env",
        staticmethod(
            lambda: SimpleNamespace(
                api_enabled=True,
                production_writes_enabled=True,
                source_add_enabled=True,
                source_add_max_concurrent_per_hotkey=3,
                source_add_max_per_day_per_hotkey=5,
                source_add_max_per_30d_per_hotkey=10,
            )
        ),
    )
    monkeypatch.setattr(api, "_verify_signed_miner", lambda _payload: _async_none())
    monkeypatch.setattr(
        api,
        "source_add_control_state",
        lambda *a, **k: _async_value({"paused": False, "status": "active"}),
    )
    monkeypatch.setattr(
        source_add_catalog,
        "source_add_api_is_current_builtin_sync",
        lambda *_args, **_kwargs: False,
    )

    async def fail_rpc(*_args, **_kwargs):
        raise AssertionError("invalid SOURCE_ADD submissions must not persist")

    monkeypatch.setattr(api, "_source_add_rpc", fail_rpc)
    manifest = _manifest_doc()
    source_metadata = ResearchLabSourceMetadata.model_validate(
        _source_metadata_doc()
    )
    payload_fields = {}
    if invalid_case == "miner_credential_field":
        payload_fields["adapter_credential"] = "redacted-test-value"
    elif invalid_case == "declared_domains":
        manifest["declared_base_domains"] = "api.test-source.example"
    elif invalid_case == "source_identity":
        source_metadata = source_metadata.model_copy(
            update={"api_base_url": "not-a-provider-origin"}
        )
    else:
        manifest["source_kind"] = "unsupported-source-kind"

    payload = ResearchLabSourceAdapterSubmissionRequest.model_construct(
        miner_hotkey="miner-hotkey-value",
        signature="signature-value-123",
        timestamp=int(time.time()),
        idempotency_key=f"source-submit-invalid-{invalid_case}",
        manifest=manifest,
        source_metadata=source_metadata,
        **payload_fields,
    )

    with pytest.raises(api.HTTPException) as exc_info:
        await api.submit_research_lab_source_adapter(payload)

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == api.SOURCE_ADD_SUBMISSION_FAILED_DETAIL
    assert JSONResponse(
        status_code=exc_info.value.status_code,
        content={"detail": exc_info.value.detail},
    ).body == b'{"detail":"Submission failed"}'


@pytest.mark.asyncio
async def test_duplicate_submission_response_is_exact_and_private(monkeypatch):
    monkeypatch.setattr(
        api.ResearchLabGatewayConfig,
        "from_env",
        staticmethod(
            lambda: SimpleNamespace(
                api_enabled=True,
                production_writes_enabled=True,
                miner_submissions_enabled=True,
                source_add_enabled=True,
                source_add_max_concurrent_per_hotkey=3,
                source_add_max_per_day_per_hotkey=5,
                source_add_max_per_30d_per_hotkey=10,
            )
        ),
    )
    monkeypatch.setattr(api, "_verify_signed_miner", lambda _payload: _async_none())
    monkeypatch.setattr(
        api,
        "source_add_control_state",
        lambda *a, **k: _async_value({"paused": False, "status": "active"}),
    )
    monkeypatch.setattr(
        source_add_catalog,
        "source_add_api_is_current_builtin_sync",
        lambda *_args, **_kwargs: False,
    )
    async def duplicate_rpc(*_args, **_kwargs):
        return {"status": "duplicate"}

    monkeypatch.setattr(api, "_source_add_rpc", duplicate_rpc)
    payload = ResearchLabSourceAdapterSubmissionRequest(
        miner_hotkey="miner-hotkey-value",
        signature="signature-value-123",
        timestamp=int(time.time()),
        idempotency_key="source-submit-duplicate-1",
        manifest=_manifest_doc(),
        source_metadata=_source_metadata_doc(),
    )

    with pytest.raises(api.HTTPException) as exc_info:
        await api.submit_research_lab_source_adapter(payload)

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == "Submission failed"
    assert JSONResponse(
        status_code=exc_info.value.status_code,
        content={"detail": exc_info.value.detail},
    ).body == b'{"detail":"Submission failed"}'


@pytest.mark.asyncio
async def test_distinct_source_cooldown_response_remains_generic_429(monkeypatch):
    monkeypatch.setattr(
        api.ResearchLabGatewayConfig,
        "from_env",
        staticmethod(
            lambda: SimpleNamespace(
                api_enabled=True,
                production_writes_enabled=True,
                miner_submissions_enabled=True,
                source_add_enabled=True,
                source_add_max_concurrent_per_hotkey=3,
                source_add_max_per_day_per_hotkey=5,
                source_add_max_per_30d_per_hotkey=10,
            )
        ),
    )
    monkeypatch.setattr(api, "_verify_signed_miner", lambda _payload: _async_none())
    monkeypatch.setattr(
        api,
        "source_add_control_state",
        lambda *a, **k: _async_value({"paused": False, "status": "active"}),
    )
    monkeypatch.setattr(
        source_add_catalog,
        "source_add_api_is_current_builtin_sync",
        lambda *_args, **_kwargs: False,
    )

    async def cooldown_rpc(name, params):
        assert name == "research_lab_source_add_admit_v3"
        assert (
            params["p_cooldown_seconds"]
            == api._SOURCE_ADD_SUBMISSION_COOLDOWN_SECONDS
        )
        return {
            "status": "route_cooldown",
            "cooldown_seconds": api._SOURCE_ADD_SUBMISSION_COOLDOWN_SECONDS,
            "wait_seconds": 7,
        }

    monkeypatch.setattr(api, "_source_add_rpc", cooldown_rpc)
    payload = ResearchLabSourceAdapterSubmissionRequest(
        miner_hotkey="miner-hotkey-value",
        signature="signature-value-123",
        timestamp=int(time.time()),
        idempotency_key="source-submit-distinct-cooldown-1",
        manifest=_manifest_doc(),
        source_metadata=_source_metadata_doc(),
    )

    with pytest.raises(api.HTTPException) as exc_info:
        await api.submit_research_lab_source_adapter(payload)

    assert exc_info.value.status_code == 429
    assert exc_info.value.detail == {
        "code": "research_lab_rate_limited",
        "route": "source_adapters",
        "message": (
            "Please wait 7 seconds before submitting another lead "
            "(anti-spam cooldown)."
        ),
        "stats": {
            "limit_type": "cooldown",
            "cooldown_seconds": api._SOURCE_ADD_SUBMISSION_COOLDOWN_SECONDS,
            "wait_seconds": 7,
        },
    }


@pytest.mark.asyncio
async def test_current_builtin_provider_is_rejected_generically_before_admission(
    monkeypatch,
):
    monkeypatch.setattr(
        api.ResearchLabGatewayConfig,
        "from_env",
        staticmethod(
            lambda: SimpleNamespace(
                api_enabled=True,
                production_writes_enabled=True,
                miner_submissions_enabled=True,
                source_add_enabled=True,
                source_add_max_concurrent_per_hotkey=3,
                source_add_max_per_day_per_hotkey=5,
                source_add_max_per_30d_per_hotkey=10,
            )
        ),
    )
    monkeypatch.setattr(api, "_verify_signed_miner", lambda _payload: _async_none())
    monkeypatch.setattr(
        api,
        "source_add_control_state",
        lambda *a, **k: _async_value({"paused": False, "status": "active"}),
    )
    monkeypatch.setattr(
        source_add_catalog,
        "source_add_api_is_current_builtin_sync",
        lambda *_args, **_kwargs: True,
    )

    async def fail_rpc(*_args, **_kwargs):
        raise AssertionError("built-in provider rejection must not persist")

    monkeypatch.setattr(api, "_source_add_rpc", fail_rpc)
    payload = ResearchLabSourceAdapterSubmissionRequest(
        miner_hotkey="miner-hotkey-value",
        signature="signature-value-123",
        timestamp=int(time.time()),
        idempotency_key="source-submit-openrouter-1",
        manifest=_manifest_doc(
            source_name="OpenRouter",
            declared_base_domains=["attacker-declared.example"],
        ),
        source_metadata=_source_metadata_doc(
            api_base_url="https://openrouter.ai/api/v1",
            documentation_url="https://openrouter.ai/docs/quickstart",
            auth_type="bearer",
        ),
    )

    with pytest.raises(api.HTTPException) as exc_info:
        await api.submit_research_lab_source_adapter(payload)

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == "Submission failed"
    assert JSONResponse(
        status_code=exc_info.value.status_code,
        content={"detail": exc_info.value.detail},
    ).body == b'{"detail":"Submission failed"}'


@pytest.mark.asyncio
async def test_builtin_provider_catalog_failure_blocks_admission(monkeypatch):
    monkeypatch.setattr(
        api.ResearchLabGatewayConfig,
        "from_env",
        staticmethod(
            lambda: SimpleNamespace(
                api_enabled=True,
                production_writes_enabled=True,
                miner_submissions_enabled=True,
                source_add_enabled=True,
                source_add_max_concurrent_per_hotkey=3,
                source_add_max_per_day_per_hotkey=5,
                source_add_max_per_30d_per_hotkey=10,
            )
        ),
    )
    monkeypatch.setattr(api, "_verify_signed_miner", lambda _payload: _async_none())
    monkeypatch.setattr(
        api,
        "source_add_control_state",
        lambda *a, **k: _async_value({"paused": False, "status": "active"}),
    )
    monkeypatch.setattr(
        source_add_catalog,
        "source_add_api_is_current_builtin_sync",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("catalog unavailable")
        ),
    )

    async def fail_rpc(*_args, **_kwargs):
        raise AssertionError("catalog failure must not persist")

    monkeypatch.setattr(api, "_source_add_rpc", fail_rpc)
    payload = ResearchLabSourceAdapterSubmissionRequest(
        miner_hotkey="miner-hotkey-value",
        signature="signature-value-123",
        timestamp=int(time.time()),
        idempotency_key="source-submit-catalog-failure-1",
        manifest=_manifest_doc(),
        source_metadata=_source_metadata_doc(),
    )

    with pytest.raises(api.HTTPException) as exc_info:
        await api.submit_research_lab_source_adapter(payload)

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == "SOURCE_ADD workflow temporarily unavailable"


@pytest.mark.asyncio
async def test_atomic_source_add_rpc_fails_closed_without_leaking_storage_error(monkeypatch):
    async def failed_rpc(_name, _params):
        raise RuntimeError("private duplicate table unavailable")

    monkeypatch.setattr(api, "call_rpc", failed_rpc)
    with pytest.raises(api.HTTPException) as exc_info:
        await api._source_add_rpc("research_lab_source_add_admit", {})

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == "SOURCE_ADD workflow temporarily unavailable"
    assert "duplicate" not in exc_info.value.detail.lower()


@pytest.mark.asyncio
async def test_exact_operator_probe_config_is_one_logical_work_across_retries(monkeypatch):
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "service-role-test")
    monkeypatch.setattr(
        api.ResearchLabGatewayConfig,
        "from_env",
        staticmethod(
            lambda: SimpleNamespace(
                api_enabled=True,
                production_writes_enabled=True,
                source_add_enabled=True,
            )
        ),
    )
    submission_id = "source_add_submission:" + "d" * 16
    call_count = 0

    async def fake_select_one(table, **_kwargs):
        nonlocal call_count
        assert table == "research_lab_source_add_submission_current"
        call_count += 1
        return {
            "submission_id": submission_id,
            "adapter_id": "adapter:test-source",
            "miner_hotkey": "hk-owner",
            "stage": "provenance_precheck_passed" if call_count == 1 else "functional_probe_passed",
            "seq": 3 if call_count == 1 else 99,
            "submission_doc": {
                "manifest": _manifest_doc(),
                "source_metadata": _source_metadata_doc(),
            },
            "precheck_status": PRECHECK_PASSED,
            "precheck_doc": {},
            "source_identity_hash": "sha256:" + "1" * 64,
        }

    work_ids = []

    async def fake_rpc(name, params):
        assert name == "research_lab_source_add_configure_probe_v3"
        work_ids.append(params["p_work_id"])
        if len(work_ids) == 3:
            return {"status": "final_approval_frozen"}
        return {
            "status": "queued" if len(work_ids) == 1 else "already_configured",
            "stage": "functional_probe_queued" if len(work_ids) == 1 else "functional_probe_passed",
            "work_id": params["p_work_id"],
        }

    monkeypatch.setattr(api, "select_one", fake_select_one)
    monkeypatch.setattr(api, "_source_add_rpc", fake_rpc)
    payload = ResearchLabSourceAdapterProbeConfigureRequest(
        base_url="https://api.test-source.example",
        auth_kind="none",
        probes=[
            {
                "method": "GET",
                "path": "/search",
                "query": {"q": "test"},
                "body_json": None,
            }
        ],
    )

    first = await api.configure_research_lab_source_adapter_test(
        submission_id, payload, authorization="Bearer service-role-test"
    )
    second = await api.configure_research_lab_source_adapter_test(
        submission_id, payload, authorization="Bearer service-role-test"
    )
    with pytest.raises(api.HTTPException) as frozen:
        await api.configure_research_lab_source_adapter_test(
            submission_id, payload, authorization="Bearer service-role-test"
        )

    assert work_ids[0] == work_ids[1] == work_ids[2]
    assert first.queue_status == "queued"
    assert second.queue_status == "already_configured"
    assert second.stage == "functional_probe_passed"
    assert frozen.value.status_code == 409
    assert frozen.value.detail == "SOURCE_ADD final approval is frozen"


@pytest.mark.asyncio
async def test_operator_probe_config_rejects_frozen_final_approval_stage(
    monkeypatch,
):
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "service-role-test")
    monkeypatch.setattr(
        api.ResearchLabGatewayConfig,
        "from_env",
        staticmethod(
            lambda: SimpleNamespace(
                api_enabled=True,
                production_writes_enabled=True,
                source_add_enabled=True,
            )
        ),
    )
    monkeypatch.setattr(
        api,
        "select_one",
        lambda *_args, **_kwargs: _async_value({"stage": "accepted"}),
    )

    async def fail_rpc(*_args, **_kwargs):
        raise AssertionError("frozen final approval must not reach configuration RPC")

    monkeypatch.setattr(api, "_source_add_rpc", fail_rpc)
    payload = ResearchLabSourceAdapterProbeConfigureRequest(
        base_url="https://api.test-source.example",
        auth_kind="none",
        probes=[
            {
                "method": "GET",
                "path": "/search",
                "query": {"q": "test"},
                "body_json": None,
            }
        ],
    )

    with pytest.raises(api.HTTPException) as frozen:
        await api.configure_research_lab_source_adapter_test(
            "source_add_submission:" + "d" * 16,
            payload,
            authorization="Bearer service-role-test",
        )

    assert frozen.value.status_code == 409
    assert frozen.value.detail == "SOURCE_ADD final approval is frozen"


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ("leg1_queued", "leg1_created"))
async def test_operator_probe_config_allows_leg1_stage(monkeypatch, stage):
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "service-role-test")
    monkeypatch.setattr(
        api.ResearchLabGatewayConfig,
        "from_env",
        staticmethod(
            lambda: SimpleNamespace(
                api_enabled=True,
                production_writes_enabled=True,
                source_add_enabled=True,
            )
        ),
    )
    submission_id = "source_add_submission:" + "d" * 16
    monkeypatch.setattr(
        api,
        "select_one",
        lambda *_args, **_kwargs: _async_value(
            {
                "submission_id": submission_id,
                "adapter_id": "adapter:test-source",
                "miner_hotkey": "hk-owner",
                "stage": stage,
                "seq": 9,
                "submission_doc": {
                    "manifest": _manifest_doc(),
                    "source_metadata": _source_metadata_doc(),
                },
                "precheck_status": PRECHECK_PASSED,
                "precheck_doc": {"reasons": ["provenance_reference_backed"]},
                "source_identity_hash": "sha256:" + "1" * 64,
            }
        ),
    )
    observed = {}

    async def fake_rpc(name, params):
        observed.update({"name": name, "params": params})
        return {
            "status": "queued",
            "stage": "functional_probe_queued",
            "work_id": params["p_work_id"],
        }

    monkeypatch.setattr(api, "_source_add_rpc", fake_rpc)
    payload = ResearchLabSourceAdapterProbeConfigureRequest(
        base_url="https://api.test-source.example",
        auth_kind="none",
        probes=[
            {
                "method": "GET",
                "path": "/search",
                "query": {"q": "test"},
                "body_json": None,
            }
        ],
    )

    response = await api.configure_research_lab_source_adapter_test(
        submission_id,
        payload,
        authorization="Bearer service-role-test",
    )

    assert response.queue_status == "queued"
    assert observed["name"] == "research_lab_source_add_configure_probe_v3"


@pytest.mark.asyncio
async def test_owner_provision_rejects_frozen_final_approval_stage(
    monkeypatch,
):
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "service-role-test")
    monkeypatch.setattr(
        api.ResearchLabGatewayConfig,
        "from_env",
        staticmethod(
            lambda: SimpleNamespace(
                api_enabled=True,
                production_writes_enabled=True,
                source_add_enabled=True,
            )
        ),
    )
    monkeypatch.setattr(
        api,
        "select_one",
        lambda *_args, **_kwargs: _async_value({"stage": "accepted"}),
    )

    async def fail_rpc(*_args, **_kwargs):
        raise AssertionError("frozen final approval must not reach provisioning RPC")

    monkeypatch.setattr(api, "_source_add_rpc", fail_rpc)

    with pytest.raises(api.HTTPException) as frozen:
        await api.provision_research_lab_source_adapter(
            "source_add_submission:" + "d" * 16,
            ResearchLabSourceAdapterProvisionRequest(
                registry_provider_id="test_source",
                provision_status=PROVISION_STATUS_APPROVED_PENDING,
            ),
            authorization="Bearer service-role-test",
        )

    assert frozen.value.status_code == 409
    assert frozen.value.detail == "SOURCE_ADD final approval is frozen"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stage",
    ("provenance_precheck_passed", "leg1_queued", "leg1_created"),
)
async def test_owner_provision_requires_exact_functional_pass_and_finalizes_atomically(
    monkeypatch,
    stage,
):
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "service-role-test")
    monkeypatch.setattr(
        api.ResearchLabGatewayConfig,
        "from_env",
        staticmethod(
            lambda: SimpleNamespace(
                api_enabled=True,
                production_writes_enabled=True,
                source_add_enabled=True,
                source_add_functional_probes_enabled=False,
                evaluation_epoch=0,
                source_add_probe_timeout_seconds=45,
            )
        ),
    )
    submission_doc = {
        "adapter_id": "adapter:test-source",
        "miner_hotkey": "hk-owner",
        "manifest": _manifest_doc(),
        "source_metadata": _source_metadata_doc(),
        "source_identity_hash": "sha256:" + "1" * 64,
    }
    select_one_calls = []

    async def fake_select_one(table, **_kwargs):
        select_one_calls.append(table)
        if table == "research_lab_source_add_submission_current":
            return {
                "submission_id": "source_add_submission:" + "a" * 16,
                "adapter_id": "adapter:test-source",
                "miner_hotkey": "hk-owner",
                "stage": stage,
                "submission_doc": submission_doc,
                "precheck_status": PRECHECK_PASSED,
                "precheck_doc": {"reasons": ["provenance_reference_backed"]},
                "source_identity_hash": "sha256:" + "1" * 64,
            }
        if table == "research_lab_source_add_probe_config_current":
            return {
                "config_ref": "source_add_probe_config:0123456789abcdef",
                "config_status": "active",
                "probe_doc": {
                    "schema_version": "leadpoet.source_add_probe_config.v2",
                    "provider_id": "sourceadd_0123456789abcdef",
                    "base_url": "https://api.test-source.example",
                    "auth_kind": "none",
                    "auth_name": "",
                    "request_headers": {},
                    "probes": [
                        {
                            "method": "GET",
                            "path": "/search",
                            "query": {"q": "test"},
                            "body_json": None,
                        }
                    ],
                },
                "credential_envelope": {},
            }
        if table == "research_lab_source_add_functional_probe_current":
            return {
                "result_status": "passed",
                "config_ref": "source_add_probe_config:0123456789abcdef",
            }
        if table == "research_lab_source_add_provisioning_current":
            return None
        if table == "research_lab_source_catalog":
            return None
        return None

    finalized = {}

    async def fake_rpc(name, params):
        assert name == "research_lab_source_add_finalize_provision_v3"
        finalized.update(params)
        return {
            "status": "provisioned",
            "catalog_id": params["p_catalog_row"]["catalog_id"],
            "provision_ref": params["p_provision_row"]["provision_ref"],
        }

    monkeypatch.setattr(api, "select_one", fake_select_one)
    monkeypatch.setattr(api, "_source_add_rpc", fake_rpc)
    monkeypatch.setattr(api, "reserved_builtin_provider_ids_sync", lambda: set())

    response = await api.provision_research_lab_source_adapter(
        "source_add_submission:" + "a" * 16,
        ResearchLabSourceAdapterProvisionRequest(
            registry_provider_id="test_source",
            provider_alias="Test source discovery",
            provision_status=PROVISION_STATUS_APPROVED_PENDING,
            cost_model={"est_cost_microusd_per_call": 250_000},
            routing_contract={
                "stage": "candidate_acquisition",
                "execution_mode": "observe",
                "priority": 91,
                "capabilities": [
                    "candidate.provider_discovery",
                    "intent.monitoring",
                ],
                "idempotency": "resume_safe",
                "cost_class": "paid",
                "unit_cost": 0.25,
                "max_calls": 2,
                "max_results": 25,
                "timeout_seconds": 12.3456,
                "intent_categories": ["hiring"],
                "evidence_types": ["provider_database", "job_posting"],
                "category_contracts": [
                    {
                        "category": "HIRING",
                        "capabilities": ["intent.monitoring"],
                        "evidence_types": ["job_posting"],
                        "requirements": ["receipt_only"],
                    }
                ],
                "binding_requirements": ["receipt_only"],
                "best_for": ["icp.structured_eligible", "intent.hiring"],
                "avoid_when": ["provider.unhealthy"],
                "best_for_description": "Use for reviewed hiring discovery.",
                "avoid_when_description": "Avoid when provider health fails.",
            },
            probe_endpoints=[
                {
                    "endpoint_id": "test_source.search",
                    "provider_id": "test_source",
                    "method": "GET",
                    "path": "/search",
                    "params": [{"name": "q", "type": "string", "required": True, "location": "query"}],
                }
            ],
        ),
        authorization="Bearer service-role-test",
    )

    assert response.adapter_id == "adapter:test-source"
    assert response.provision_status == PROVISION_STATUS_APPROVED_PENDING
    assert finalized["p_smoke_attempt"] == {}
    registry_entry = finalized["p_provision_row"]["provision_doc"][
        "provider_registry_entry"
    ]
    assert registry_entry["id"] == "test_source"
    planner = registry_entry["planner_summary"]
    assert planner["provider_alias"] == "Test source discovery"
    assert planner["stage"] == "candidate_acquisition"
    assert planner["execution_mode"] == "observe"
    assert planner["idempotency"] == "resume_safe"
    assert planner["cost_class"] == "paid"
    assert planner["unit_cost"] == 0.25
    assert planner["max_calls"] == 2
    assert planner["timeout_seconds"] == 12.346
    assert planner["intent_categories"] == ["HIRING"]
    assert planner["category_contracts"] == [
        {
            "category": "HIRING",
            "capabilities": ["intent.monitoring"],
            "evidence_types": ["job_posting"],
            "requirements": ["receipt_only"],
        }
    ]
    assert planner["binding_requirements"] == ["receipt_only"]
    assert planner["best_for_features"] == [
        "icp.structured_eligible",
        "intent.hiring",
    ]
    assert planner["avoid_when_features"] == ["provider.unhealthy"]
    assert finalized["p_provision_row"]["credential_envelope"] == {}
    assert "api_credential" not in str(finalized)


@pytest.mark.asyncio
async def test_owner_eligible_provision_creates_pending_then_queues_exact_smoke(monkeypatch):
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "service-role-test")
    monkeypatch.setattr(
        api.ResearchLabGatewayConfig,
        "from_env",
        staticmethod(
            lambda: SimpleNamespace(
                api_enabled=True,
                production_writes_enabled=True,
                source_add_enabled=True,
                source_add_functional_probes_enabled=True,
            )
        ),
    )
    submission_id = "source_add_submission:" + "b" * 16
    config_ref = "source_add_probe_config:0123456789abcdef"
    probe_doc = {
        "schema_version": "leadpoet.source_add_probe_config.v2",
        "provider_id": "sourceadd_0123456789abcdef",
        "base_url": "https://api.test-source.example",
        "auth_kind": "none",
        "auth_name": "",
        "request_headers": {},
        "probes": [
            {
                "method": "GET",
                "path": "/search",
                "query": {"q": "test"},
                "body_json": None,
            }
        ],
    }

    async def fake_select_one(table, **_kwargs):
        if table == "research_lab_source_add_submission_current":
            return {
                "submission_id": submission_id,
                "adapter_id": "adapter:test-source",
                "miner_hotkey": "hk-owner",
                "stage": "functional_probe_passed",
                "submission_doc": {
                    "manifest": _manifest_doc(),
                    "source_metadata": _source_metadata_doc(),
                },
                "precheck_status": PRECHECK_PASSED,
                "precheck_doc": {},
                "source_identity_hash": "sha256:" + "1" * 64,
            }
        if table == "research_lab_source_add_probe_config_current":
            return {
                "config_ref": config_ref,
                "config_status": "active",
                "probe_doc": probe_doc,
                "credential_envelope": {},
            }
        if table == "research_lab_source_add_functional_probe_current":
            return {"result_status": "passed", "config_ref": config_ref}
        return None

    rpc_calls = []
    freeze_final_approval = {"enabled": True}

    async def fake_rpc(name, params):
        rpc_calls.append((name, params))
        if name == "research_lab_source_add_finalize_provision_v3":
            if freeze_final_approval["enabled"]:
                return {"status": "final_approval_frozen"}
            assert params["p_provision_row"]["provision_status"] == (
                PROVISION_STATUS_APPROVED_PENDING
            )
            assert params["p_provision_row"]["provision_doc"][
                "provider_registry_entry"
            ]["active"] is False
            assert params["p_smoke_attempt"] == {}
            return {
                "status": "provisioned",
                "catalog_id": params["p_catalog_row"]["catalog_id"],
                "provision_ref": params["p_provision_row"]["provision_ref"],
            }
        assert name == "research_lab_source_add_enqueue_provision_smoke_v2"
        assert params["p_config_ref"] == config_ref
        assert params["p_provision_row"]["provision_status"] == (
            PROVISION_STATUS_ELIGIBLE
        )
        assert params["p_provision_row"]["provision_doc"][
            "provider_registry_entry"
        ]["active"] is True
        return {
            "status": "queued",
            "work_id": params["p_work_id"],
            "work_status": "queued",
        }

    monkeypatch.setattr(api, "select_one", fake_select_one)
    monkeypatch.setattr(api, "_source_add_rpc", fake_rpc)
    monkeypatch.setattr(api, "reserved_builtin_provider_ids_sync", lambda: set())

    with pytest.raises(api.HTTPException) as frozen:
        await api.provision_research_lab_source_adapter(
            submission_id,
            ResearchLabSourceAdapterProvisionRequest(
                registry_provider_id="test_source",
                provision_status=PROVISION_STATUS_ELIGIBLE,
                probe_endpoints=[
                    {
                        "endpoint_id": "test_source.search",
                        "provider_id": "test_source",
                        "method": "GET",
                        "path": "/search",
                        "params": [
                            {
                                "name": "q",
                                "type": "string",
                                "required": True,
                                "location": "query",
                            }
                        ],
                    }
                ],
            ),
            authorization="Bearer service-role-test",
        )
    assert frozen.value.status_code == 409
    assert frozen.value.detail == "SOURCE_ADD final approval is frozen"

    freeze_final_approval["enabled"] = False
    rpc_calls.clear()
    response = await api.provision_research_lab_source_adapter(
        submission_id,
        ResearchLabSourceAdapterProvisionRequest(
            registry_provider_id="test_source",
            provision_status=PROVISION_STATUS_ELIGIBLE,
            probe_endpoints=[
                {
                    "endpoint_id": "test_source.search",
                    "provider_id": "test_source",
                    "method": "GET",
                    "path": "/search",
                    "params": [
                        {
                            "name": "q",
                            "type": "string",
                            "required": True,
                            "location": "query",
                        }
                    ],
                }
            ],
        ),
        authorization="Bearer service-role-test",
    )

    assert [name for name, _params in rpc_calls] == [
        "research_lab_source_add_finalize_provision_v3",
        "research_lab_source_add_enqueue_provision_smoke_v2",
    ]
    assert response.provision_status == PROVISION_STATUS_APPROVED_PENDING
    assert response.requested_provision_status == PROVISION_STATUS_ELIGIBLE
    assert response.queue_status == "queued"
    assert response.work_id and response.work_id.startswith("source_add_work:")


def test_owner_process_environment_credentials_are_retired():
    with pytest.raises(ValidationError, match="process-environment credentials are retired"):
        ResearchLabSourceAdapterProvisionRequest(
            registry_provider_id="test_source_auth",
            provision_status=PROVISION_STATUS_ELIGIBLE,
            credential_env_refs=["SYNTHETIC_SOURCE_CREDENTIAL"],
        )


def test_source_add_provider_alias_rejects_secret_material():
    with pytest.raises(ValidationError, match="raw provider secret material"):
        ResearchLabSourceAdapterProvisionRequest(
            registry_provider_id="test_source",
            provider_alias="".join(("sk", "-or-", "synthetic")),
        )


def test_source_add_v8_routing_contract_round_trips_as_json():
    request = ResearchLabSourceAdapterProvisionRequest(
        registry_provider_id="test_source",
        provider_alias="Test source",
        routing_contract={
            "stage": "intent_evidence",
            "execution_mode": "invoke",
            "priority": 40,
            "capabilities": ["intent.provider_evidence"],
            "idempotency": "idempotent",
            "cost_class": "free",
            "unit_cost": 0.0,
            "max_calls": 1,
            "max_results": 10,
            "timeout_seconds": 15.0,
            "intent_categories": ["funding"],
            "evidence_types": ["funding_event"],
            "category_contracts": [
                {
                    "category": "FUNDING",
                    "capabilities": ["intent.provider_evidence"],
                    "evidence_types": ["funding_event"],
                    "requirements": [],
                }
            ],
            "binding_requirements": [],
            "best_for": ["intent.funding"],
            "avoid_when": [],
        },
    )
    dumped = request.model_dump(mode="json")
    assert ResearchLabSourceAdapterProvisionRequest.model_validate(
        dumped
    ).model_dump(mode="json") == dumped


@pytest.mark.parametrize(
    "provider_id",
    ("Test_Source", "1test_source", " test_source", "test_source "),
)
def test_source_add_provision_rejects_noncanonical_v8_provider_id(provider_id):
    with pytest.raises(
        ValidationError,
        match="canonical lowercase slug",
    ):
        ResearchLabSourceAdapterProvisionRequest(
            registry_provider_id=provider_id,
        )


def test_source_add_provision_rejects_invalid_explicit_v8_contract():
    request = ResearchLabSourceAdapterProvisionRequest(
        registry_provider_id="test_source",
        routing_contract={
            "stage": "candidate_acquisition",
            "cost_class": "free",
            "unit_cost": 1.0,
        },
    )
    with pytest.raises(ValueError, match="cost_class and unit_cost differ"):
        api.normalize_source_add_planner_contract(
            request.registry_provider_id,
            request.routing_contract,
        )


def test_provisioned_rows_build_provider_and_probe_catalog_entries():
    row = {
        "adapter_id": "adapter:test-source",
        "miner_hotkey": "hk-owner",
        "provision_doc": {
            "provider_registry_entry": {
                "id": "test_source",
                "base_url": "https://api.test-source.example",
                "auth_kind": "none",
                "credential_ref": [],
                "cost_model": {"est_cost_microusd_per_call": 1000},
            },
            "probe_endpoints": [
                {
                    "endpoint_id": "test_source.search",
                    "provider_id": "test_source",
                    "method": "GET",
                    "path": "/search",
                    "params": [{"name": "q", "type": "string", "required": True, "location": "query"}],
                }
            ],
        },
    }
    providers = provider_registry_entries_from_provisioned_rows([row])
    probes = probe_endpoints_from_provisioned_rows([row])
    assert providers[0].id == "test_source"
    assert providers[0].base_url == "https://api.test-source.example"
    assert probes[0].endpoint_id == "test_source.search"


def test_provisioned_source_loader_paginates_beyond_postgrest_default(monkeypatch):
    source_rows = [
        {
            "adapter_id": f"adapter:source-{index}",
            "provision_status": PROVISION_STATUS_ELIGIBLE,
            "provision_doc": {
                "provider_registry_entry": {
                    "id": f"source_{index}",
                    "base_url": f"https://source-{index}.invalid",
                    "auth_kind": "none",
                    "credential_ref": [],
                }
            },
            "credential_envelope": {},
        }
        for index in range(750)
    ]
    ranges = []

    class Response:
        def __init__(self, data):
            self.data = data

    class Query:
        def __init__(self):
            self.start = 0
            self.end = 0

        def select(self, *_args):
            return self

        def eq(self, *_args):
            return self

        def range(self, start, end):
            self.start, self.end = start, end
            ranges.append((start, end))
            return self

        def execute(self):
            return Response(source_rows[self.start : self.end + 1])

    class Client:
        def table(self, table):
            assert table == "research_lab_source_add_provisioning_current"
            return Query()

    monkeypatch.setattr(source_add_catalog, "get_write_client", lambda: Client())
    loaded = source_add_catalog.load_provisioned_source_rows_sync(raise_on_error=True)
    assert len(loaded) == 750
    assert ranges == [(0, 499), (500, 999)]


def test_source_add_encrypted_credential_envelope_must_be_well_formed():
    row = {
        "provision_doc": {
            "provider_registry_entry": {
                "auth_kind": "header",
                "credential_ref": ["encrypted_ref:source_add:synthetic"],
            }
        },
        "credential_envelope": {
            "ciphertext_b64": "not-base64",
            "kms_key_id": "alias/synthetic",
            "credential_ref": "encrypted_ref:source_add:synthetic",
        },
    }
    assert source_add_row_credential_ready(row) is False
    row["credential_envelope"]["ciphertext_b64"] = base64.b64encode(b"encrypted-payload").decode()
    assert source_add_row_credential_ready(row) is True


def test_llm_judge_verdict_parser_accepts_helped_json():
    verdict = _parse_verdict(
        '{"verdict":"helped","confidence":0.9,"source_used":true,'
        '"adapter_id":"adapter:test-source","registry_provider_id":"test_source",'
        '"evidence_summary":"Used source","reason_codes":["matched_api"]}',
        model_id="openai/gpt-5.6-sol",
        provider_usage={"model": "openai/gpt-5.6-sol"},
    )
    assert verdict.passed is True
    assert verdict.trigger_evidence()["llm_judge_passed"] is True


def test_llm_judge_verdict_parser_rejects_string_source_used():
    with pytest.raises(ValueError, match="non-boolean source_used"):
        _parse_verdict(
            '{"verdict":"helped","confidence":0.9,"source_used":"false"}',
            model_id="openai/gpt-5.6-sol",
            provider_usage={},
        )


@pytest.mark.asyncio
async def test_owner_recheck_only_queues_provenance_and_never_creates_leg1(monkeypatch):
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "service-role-test")
    monkeypatch.setattr(
        api.ResearchLabGatewayConfig,
        "from_env",
        staticmethod(
            lambda: SimpleNamespace(
                api_enabled=True,
                production_writes_enabled=True,
                source_add_enabled=True,
            )
        ),
    )
    submission_id = "source_add_submission:" + "b" * 16
    submission_doc = {
        "submission_id": submission_id,
        "adapter_id": "adapter:test-source",
        "miner_hotkey": "hk-owner",
        "manifest": _manifest_doc(),
        "stage": PRECHECK_MANUAL,
        "stage_history": ["submitted", "manifest_validated", PRECHECK_MANUAL],
        "source_metadata": _source_metadata_doc(),
        "precheck_status": PRECHECK_MANUAL,
        "precheck_doc": {"reasons": ["low_docs_completeness"]},
        "source_identity_hash": "sha256:" + "1" * 64,
    }

    async def fake_select_one(table, **_kwargs):
        assert table == "research_lab_source_add_submission_current"
        return {
            "submission_id": submission_id,
            "adapter_id": "adapter:test-source",
            "miner_hotkey": "hk-owner",
            "stage": PRECHECK_MANUAL,
            "submission_doc": submission_doc,
            "precheck_status": PRECHECK_MANUAL,
            "precheck_doc": {"reasons": ["low_docs_completeness"]},
            "source_identity_hash": "sha256:" + "1" * 64,
        }

    queued = {}

    async def fake_rpc(name, params):
        assert name == "research_lab_source_add_requeue_provenance_v2"
        queued.update(params)
        return {
            "status": "queued",
            "stage": "provenance_queued",
            "work_id": params["p_work_id"],
        }

    monkeypatch.setattr(api, "select_one", fake_select_one)
    monkeypatch.setattr(api, "_source_add_rpc", fake_rpc)

    response = await api.recheck_research_lab_source_adapter_provenance(
        submission_id,
        authorization="Bearer service-role-test",
    )

    assert response.precheck_status == PRECHECK_MANUAL
    assert response.stage == "provenance_queued"
    assert response.leg1_reward_status == "not_evaluated"
    assert queued["p_submission_id"] == submission_id
    assert queued["p_identity_hash"].startswith("sha256:")
    assert queued["p_provider_origin_hash"].startswith("sha256:")


@pytest.mark.asyncio
async def test_owner_recheck_refuses_legacy_submission_without_structured_metadata(monkeypatch):
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "service-role-test")
    monkeypatch.setattr(
        api.ResearchLabGatewayConfig,
        "from_env",
        staticmethod(
            lambda: SimpleNamespace(
                api_enabled=True,
                production_writes_enabled=True,
                source_add_enabled=True,
            )
        ),
    )

    async def legacy_row(*_args, **_kwargs):
        return {
            "submission_id": "source_add_submission:" + "c" * 16,
            "adapter_id": "adapter:legacy-source",
            "miner_hotkey": "hk-owner",
            "stage": PRECHECK_MANUAL,
            "submission_doc": {
                "manifest": _manifest_doc(adapter_id="adapter:legacy-source"),
                "source_metadata": {},
            },
        }

    monkeypatch.setattr(api, "select_one", legacy_row)

    with pytest.raises(api.HTTPException) as exc_info:
        await api.recheck_research_lab_source_adapter_provenance(
            "source_add_submission:" + "c" * 16,
            authorization="Bearer service-role-test",
        )

    assert exc_info.value.status_code == 400
    assert "submission metadata is incomplete or invalid" in str(exc_info.value.detail)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "admit_status,limit_type,limit,expects_retry_after",
    [
        ("hotkey_open_cap", "open_submissions", 3, False),
        ("hotkey_day_cap", "daily", 5, True),
        ("hotkey_30d_cap", "rolling_30d", 10, False),
    ],
)
async def test_hotkey_cap_429_names_the_limit_it_hit(
    monkeypatch, admit_status, limit_type, limit, expects_retry_after
):
    """Each per-hotkey cap must say which limit it hit, not just "limit reached".

    A bare refusal gives a miner no retry horizon, so clients retry in a tight
    loop against a cap that will not move for hours.
    """

    monkeypatch.setattr(
        api.ResearchLabGatewayConfig,
        "from_env",
        staticmethod(
            lambda: SimpleNamespace(
                api_enabled=True,
                production_writes_enabled=True,
                miner_submissions_enabled=True,
                source_add_enabled=True,
                source_add_max_concurrent_per_hotkey=3,
                source_add_max_per_day_per_hotkey=5,
                source_add_max_per_30d_per_hotkey=10,
            )
        ),
    )
    monkeypatch.setattr(api, "_verify_signed_miner", lambda _payload: _async_none())
    monkeypatch.setattr(
        api,
        "source_add_control_state",
        lambda *a, **k: _async_value({"paused": False, "status": "active"}),
    )
    monkeypatch.setattr(
        source_add_catalog,
        "source_add_api_is_current_builtin_sync",
        lambda *_args, **_kwargs: False,
    )

    async def cap_rpc(name, _params):
        assert name == "research_lab_source_add_admit_v3"
        return {"status": admit_status}

    monkeypatch.setattr(api, "_source_add_rpc", cap_rpc)
    payload = ResearchLabSourceAdapterSubmissionRequest(
        miner_hotkey="miner-hotkey-value",
        signature="signature-value-123",
        timestamp=int(time.time()),
        idempotency_key="source-submit-%s-1" % admit_status,
        manifest=_manifest_doc(),
        source_metadata=_source_metadata_doc(),
    )

    with pytest.raises(api.HTTPException) as exc_info:
        await api.submit_research_lab_source_adapter(payload)

    assert exc_info.value.status_code == 429
    detail = exc_info.value.detail
    assert detail["code"] == "research_lab_rate_limited"
    assert detail["route"] == "source_adapters"
    assert detail["stats"]["limit_type"] == limit_type
    assert detail["stats"]["limit"] == limit
    assert str(limit) in detail["message"]
    if expects_retry_after:
        retry_after = detail["stats"]["retry_after_seconds"]
        assert 1 <= retry_after <= 86400
    else:
        assert "retry_after_seconds" not in detail["stats"]


def test_daily_cap_retry_after_counts_to_the_next_utc_midnight():
    from datetime import datetime, timezone

    at_2359 = datetime(2026, 9, 3, 23, 59, 0, tzinfo=timezone.utc)
    assert api._source_add_seconds_until_utc_midnight(at_2359) == 60
    at_midnight = datetime(2026, 9, 3, 0, 0, 0, tzinfo=timezone.utc)
    assert api._source_add_seconds_until_utc_midnight(at_midnight) == 86400
