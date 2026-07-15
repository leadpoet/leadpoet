import base64
import time
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from gateway.research_lab import api, source_add_catalog
from gateway.research_lab.models import (
    AttestedCredentialCiphertextV2,
    ResearchLabSourceAdapterProbeConfigureRequest,
    ResearchLabSourceAdapterProvisionRequest,
    ResearchLabSourceAdapterSubmissionRequest,
    ResearchLabSourceAddCredentialRecipientRequest,
)
from gateway.research_lab.source_add_catalog import (
    PROVISION_STATUS_APPROVED_PENDING,
    PROVISION_STATUS_ELIGIBLE,
    provider_registry_entries_from_provisioned_rows,
    probe_endpoints_from_provisioned_rows,
    source_add_row_credential_ready,
)
from gateway.research_lab.source_add_llm_judge import _parse_verdict
from gateway.research_lab.source_add_provenance import PRECHECK_MANUAL, PRECHECK_PASSED
from research_lab.source_add_execution import SourceAddRejectionReason, intake_source_add_submission
from research_lab.source_add_identity import (
    normalize_source_add_domain,
    normalize_source_add_url,
    source_documentation_identity_hash,
    source_identity_alias_hashes_from_metadata,
    source_identity_hash,
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
    with pytest.raises(ValidationError, match="miners must not submit"):
        ResearchLabSourceAdapterSubmissionRequest(
            **common,
            adapter_credential="plaintext-secret-value",
        )
    encrypted = AttestedCredentialCiphertextV2(
        request_id="sha256:" + "8" * 64,
        ciphertext_b64=base64.b64encode(b"x" * 384).decode(),
    )
    with pytest.raises(ValidationError, match="miners must not submit"):
        ResearchLabSourceAdapterSubmissionRequest(
            **common,
            adapter_credential_v2=encrypted,
        )


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
        "_enforce_research_lab_submission_rate_limit",
        lambda *_args, **_kwargs: _async_none(),
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
    assert observed["name"] == "research_lab_source_add_admit"
    assert observed["params"]["p_max_open"] == 3
    assert observed["params"]["p_max_day"] == 5
    assert observed["params"]["p_max_30d"] == 10
    assert observed["params"]["p_documentation_identity_hash"].startswith(
        "sha256:"
    )
    assert observed["params"]["p_record_doc"]["manifest"]["credential_policy"] == "no_credentials"
    assert observed["params"]["p_record_doc"]["credential_envelope"] == {}


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
        "_enforce_research_lab_submission_rate_limit",
        lambda *_args, **_kwargs: _async_none(),
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
    assert exc_info.value.detail == "Already submitted"


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
        assert name == "research_lab_source_add_configure_probe"
        work_ids.append(params["p_work_id"])
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

    assert work_ids[0] == work_ids[1]
    assert first.queue_status == "queued"
    assert second.queue_status == "already_configured"
    assert second.stage == "functional_probe_passed"


@pytest.mark.asyncio
async def test_owner_provision_requires_exact_functional_pass_and_finalizes_atomically(monkeypatch):
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
                "stage": "provenance_precheck_passed",
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
        assert name == "research_lab_source_add_finalize_provision"
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
            provision_status=PROVISION_STATUS_APPROVED_PENDING,
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
    assert finalized["p_provision_row"]["provision_doc"]["provider_registry_entry"]["id"] == "test_source"
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

    async def fake_rpc(name, params):
        rpc_calls.append((name, params))
        if name == "research_lab_source_add_finalize_provision":
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
        assert name == "research_lab_source_add_enqueue_provision_smoke"
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
        "research_lab_source_add_finalize_provision",
        "research_lab_source_add_enqueue_provision_smoke",
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
        assert name == "research_lab_source_add_requeue_provenance"
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
