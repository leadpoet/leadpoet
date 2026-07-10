from types import SimpleNamespace

import pytest

from gateway.research_lab import api
from gateway.research_lab.models import ResearchLabSourceAdapterProvisionRequest
from gateway.research_lab.source_add_catalog import (
    PROVISION_STATUS_ELIGIBLE,
    provider_registry_entries_from_provisioned_rows,
    probe_endpoints_from_provisioned_rows,
)
from gateway.research_lab.source_add_llm_judge import _parse_verdict
from research_lab.source_add_execution import SourceAddRejectionReason, intake_source_add_submission
from research_lab.source_add_identity import source_identity_hash


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
    }
    doc.update(overrides)
    return doc


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


@pytest.mark.asyncio
async def test_owner_provision_endpoint_appends_catalog_and_provisioning(monkeypatch):
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "service-role-test")
    monkeypatch.setattr(
        api.ResearchLabGatewayConfig,
        "from_env",
        staticmethod(
            lambda: SimpleNamespace(
                api_enabled=True,
                production_writes_enabled=True,
                source_add_enabled=True,
                source_add_credential_kms_key_id="",
                openrouter_key_kms_key_id="",
            )
        ),
    )
    submission_doc = {
        "adapter_id": "adapter:test-source",
        "miner_hotkey": "hk-owner",
        "manifest": _manifest_doc(),
        "source_metadata": {
            "api_base_url": "https://api.test-source.example",
            "documentation_url": "https://docs.test-source.example",
        },
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
                "source_identity_hash": "sha256:" + "1" * 64,
            }
        if table == "research_lab_source_catalog":
            return None
        return None

    writes = []

    async def fake_insert_row(table, row):
        writes.append((table, dict(row)))
        return dict(row)

    async def fake_next_event_seq(*_args, **_kwargs):
        return 0

    monkeypatch.setattr(api, "select_one", fake_select_one)
    monkeypatch.setattr(api, "insert_row", fake_insert_row)
    monkeypatch.setattr(api, "next_event_seq", fake_next_event_seq)

    response = await api.provision_research_lab_source_adapter(
        "source_add_submission:" + "a" * 16,
        ResearchLabSourceAdapterProvisionRequest(
            registry_provider_id="test_source",
            provision_status=PROVISION_STATUS_ELIGIBLE,
            auth_kind="none",
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
    assert response.provision_status == PROVISION_STATUS_ELIGIBLE
    assert [table for table, _row in writes] == [
        "research_lab_source_catalog",
        "research_lab_source_add_provisioning_events",
    ]
    assert writes[1][1]["provision_doc"]["provider_registry_entry"]["id"] == "test_source"
    assert "api_credential" not in str(writes)


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
