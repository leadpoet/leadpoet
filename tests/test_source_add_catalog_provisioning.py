from datetime import datetime, timedelta, timezone
import base64
from types import SimpleNamespace

import pytest

from gateway.research_lab import api, source_add_catalog
from gateway.research_lab.models import ResearchLabSourceAdapterProvisionRequest
from gateway.research_lab.source_add_catalog import (
    PROVISION_STATUS_ELIGIBLE,
    provider_registry_entries_from_provisioned_rows,
    probe_endpoints_from_provisioned_rows,
    source_add_row_credential_ready,
)
from gateway.research_lab.source_add_llm_judge import _parse_verdict
from gateway.research_lab.source_add_provenance import PRECHECK_MANUAL, PRECHECK_PASSED, SourceAddProvenanceResult
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
async def test_intake_context_releases_rejected_dedupe_and_keeps_active_global_dedupe(monkeypatch):
    now = datetime.now(timezone.utc)
    rejected_hash = "sha256:" + "1" * 64
    rejected_precheck_hash = "sha256:" + "2" * 64
    active_other_miner_hash = "sha256:" + "3" * 64
    active_own_hash = "sha256:" + "4" * 64
    catalog_hash = "sha256:" + "5" * 64
    provisioned_hash = "sha256:" + "6" * 64
    miner_rows = [
        {
            "stage": "rejected",
            "created_at": now.isoformat(),
            "submission_doc": {"submitted_at": (now - timedelta(days=2)).isoformat()},
            "source_identity_hash": rejected_hash,
        },
        {
            "stage": "needs_manual_review",
            "created_at": now.isoformat(),
            "submission_doc": {"submitted_at": now.isoformat()},
            "source_identity_hash": active_own_hash,
        },
    ]
    global_rows = [
        {"stage": "rejected", "source_identity_hash": rejected_hash, "submission_doc": {}},
        {
            "stage": "rejected_precheck",
            "source_identity_hash": rejected_precheck_hash,
            "submission_doc": {},
        },
        {
            "stage": "needs_manual_review",
            "source_identity_hash": active_other_miner_hash,
            "submission_doc": {"manifest": {"declared_base_domains": ["pending.example"]}},
        },
        {
            "stage": "provenance_precheck_passed",
            "source_identity_hash": active_own_hash,
            "submission_doc": {},
        },
    ]

    async def fake_select_all(table, *, filters=(), **_kwargs):
        if table == "research_lab_source_add_submission_current":
            return miner_rows if filters else global_rows
        if table == "research_lab_source_catalog":
            return [{"declared_base_domains": ["approved.example"], "source_identity_hash": catalog_hash}]
        if table == "research_lab_source_add_provisioning_current":
            return [{"source_identity_hash": provisioned_hash}]
        raise AssertionError(f"unexpected table: {table}")

    monkeypatch.setattr(api, "select_all", fake_select_all)

    open_count, day_count, month_count, domains, identity_hashes = await api._source_add_intake_context(
        "hk-owner"
    )

    assert open_count == 1
    assert day_count == 1
    assert month_count == 2
    assert domains == ["pending.example", "approved.example"]
    assert rejected_hash not in identity_hashes
    assert rejected_precheck_hash not in identity_hashes
    assert identity_hashes == sorted(
        {active_other_miner_hash, active_own_hash, catalog_hash, provisioned_hash}
    )


@pytest.mark.asyncio
async def test_intake_context_fails_closed_when_global_duplicate_read_fails(monkeypatch):
    async def fake_select_all(table, *, filters=(), **_kwargs):
        if table == "research_lab_source_add_submission_current" and filters:
            return []
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(api, "select_all", fake_select_all)

    with pytest.raises(api.HTTPException) as exc_info:
        await api._source_add_intake_context("hk-owner")

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == "SOURCE_ADD intake temporarily unavailable"


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
    monkeypatch.setattr(api, "reserved_builtin_provider_ids_sync", lambda: set())

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


@pytest.mark.asyncio
async def test_owner_cannot_mark_authenticated_source_eligible_with_unresolved_env_ref(monkeypatch):
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "service-role-test")
    monkeypatch.delenv("SYNTHETIC_SOURCE_CREDENTIAL", raising=False)
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
        "source_metadata": {"api_base_url": "https://api.test-source.example"},
    }

    async def fake_select_one(table, **_kwargs):
        assert table == "research_lab_source_add_submission_current"
        return {
            "submission_id": "source_add_submission:" + "c" * 16,
            "adapter_id": "adapter:test-source",
            "miner_hotkey": "hk-owner",
            "stage": "provenance_precheck_passed",
            "submission_doc": submission_doc,
            "source_identity_hash": "sha256:" + "2" * 64,
        }

    monkeypatch.setattr(api, "select_one", fake_select_one)

    with pytest.raises(api.HTTPException) as exc_info:
        await api.provision_research_lab_source_adapter(
            "source_add_submission:" + "c" * 16,
            ResearchLabSourceAdapterProvisionRequest(
                registry_provider_id="test_source_auth",
                provision_status=PROVISION_STATUS_ELIGIBLE,
                auth_kind="header",
                auth_name="x-synthetic-key",
                credential_env_refs=["SYNTHETIC_SOURCE_CREDENTIAL"],
                probe_endpoints=[
                    {
                        "endpoint_id": "test_source_auth.search",
                        "provider_id": "test_source_auth",
                        "method": "GET",
                        "path": "/search",
                        "params": [],
                    }
                ],
            ),
            authorization="Bearer service-role-test",
        )

    assert exc_info.value.status_code == 400
    assert "cannot become provisioned_autoresearch_eligible" in exc_info.value.detail


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


@pytest.mark.asyncio
async def test_owner_recheck_advances_manual_submission_and_creates_leg1(monkeypatch):
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "service-role-test")
    monkeypatch.setattr(
        api.ResearchLabGatewayConfig,
        "from_env",
        staticmethod(
            lambda: SimpleNamespace(
                api_enabled=True,
                production_writes_enabled=True,
                source_add_enabled=True,
                source_add_rewards_enabled=True,
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
        "source_metadata": {
            "api_base_url": "https://api.test-source.example",
            "documentation_url": "https://docs.test-source.example",
            "auth_type": "none",
            "endpoint_examples": [{"method": "GET", "path": "/search"}],
            "rate_limit_notes": "Use conservative request pacing.",
        },
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

    persisted = []

    async def fake_persist(record_doc):
        persisted.append(dict(record_doc))

    async def fake_reward(**kwargs):
        assert kwargs["precheck_status"] == PRECHECK_PASSED
        return {
            "source_add_leg1_reward_status": "created",
            "reward_ref": "source_add_reward:" + "2" * 16,
            "start_epoch": 701,
        }

    monkeypatch.setattr(api, "select_one", fake_select_one)
    monkeypatch.setattr(
        api,
        "evaluate_source_add_provenance",
        lambda **_kwargs: SourceAddProvenanceResult(
            PRECHECK_PASSED,
            ("provenance_reference_backed",),
            {"docs_completeness": {"score": 5}},
        ),
    )
    monkeypatch.setattr(api, "persist_source_add_submission", fake_persist)
    monkeypatch.setattr(api, "_maybe_create_source_add_leg1_reward_for_precheck", fake_reward)

    response = await api.recheck_research_lab_source_adapter_provenance(
        submission_id,
        authorization="Bearer service-role-test",
    )

    assert response.precheck_status == PRECHECK_PASSED
    assert response.stage == PRECHECK_PASSED
    assert response.leg1_reward_status == "created"
    assert len(persisted) == 1
    assert persisted[0]["stage_history"][-1] == PRECHECK_PASSED
    assert persisted[0]["source_metadata"] == submission_doc["source_metadata"]


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
    assert "structured SOURCE_ADD fields" in str(exc_info.value.detail)
