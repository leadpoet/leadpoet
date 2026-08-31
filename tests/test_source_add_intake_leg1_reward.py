from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from gateway.research_lab import allocations
from gateway.research_lab import attested_v2_store
from gateway.research_lab import source_add_workflow as workflow
from gateway.research_lab import v2_authority
from gateway.research_lab.source_add_provenance import (
    PRECHECK_MANUAL,
    PRECHECK_PASSED,
    SourceAddProvenanceResult,
)
from leadpoet_canonical.attested_v2 import sha256_json


SUBMISSION_ID = "source_add_submission:abc123abc123abcd"
ADAPTER_ID = "adapter:credible-api"
MINER_HOTKEY = "5MinerHotkey111111111111111111111111111111111"


@pytest.fixture(autouse=True)
def _provider_execution_fence(monkeypatch):
    async def passthrough(work):
        return dict(work)

    monkeypatch.setattr(workflow, "_begin_provider_execution", passthrough)


def _config(**overrides):
    values = {
        "source_add_rewards_enabled": True,
        "source_add_functional_rewards_enabled": True,
        "source_add_functional_probes_enabled": True,
        "source_add_leg1_alpha_percent": 1.0,
        "source_add_leg1_max_per_utc_day": 10,
        "source_add_probe_timeout_seconds": 45,
        "source_add_probe_max_attempts": 5,
        "source_add_work_lease_seconds": 180,
        "lab_reward_epochs": 20,
        "evaluation_epoch": 700,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _metadata(**overrides):
    value = {
        "api_base_url": "https://api.credible.example/v1",
        "documentation_url": "https://credible.example/docs",
        "auth_type": "none",
        "endpoint_examples": [
            {
                "method": "GET",
                "path": "/records",
                "purpose": "Return current company records",
                "example_query": "limit=1",
            }
        ],
        "rate_limit_notes": "Use conservative request pacing.",
        "data_provenance_notes": "Official registry records.",
        "third_party_refs": [],
    }
    value.update(overrides)
    return value


def _submission_row(**overrides):
    row = {
        "submission_id": SUBMISSION_ID,
        "adapter_id": ADAPTER_ID,
        "miner_hotkey": MINER_HOTKEY,
        "stage": "provenance_queued",
        "precheck_status": "",
        "precheck_doc": {},
        "submission_doc": {
            "submission_id": SUBMISSION_ID,
            "adapter_id": ADAPTER_ID,
            "miner_hotkey": MINER_HOTKEY,
            "manifest": {
                "source_name": "Credible API",
                "source_kind": "registry",
                "declared_base_domains": ["credible.example"],
            },
            "source_metadata": _metadata(),
        },
    }
    row.update(overrides)
    return row


def _leased_work(kind: str, **overrides):
    value = {
        "work_id": workflow.source_add_work_id(SUBMISSION_ID, kind),
        "submission_id": SUBMISSION_ID,
        "adapter_id": ADAPTER_ID,
        "work_kind": kind,
        "attempt_count": 1,
        "lease_token": "11111111-1111-1111-1111-111111111111",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "job_doc": {},
    }
    value.update(overrides)
    return value


@pytest.mark.asyncio
async def test_provenance_pass_only_queues_functional_probe(monkeypatch):
    finished = {}
    observed = {}

    async def fake_provenance(**kwargs):
        observed.update(kwargs)
        return (
            SourceAddProvenanceResult(
                PRECHECK_PASSED,
                ("provenance_reference_backed",),
                {"docs_completeness": {"score": 5}},
            ),
            {"receipt": {"receipt_hash": "sha256:" + "9" * 64}},
        )

    async def fake_finish(_work, **kwargs):
        finished.update(kwargs)
        return {"status": "completed"}

    monkeypatch.setattr(workflow, "_load_submission", lambda _sid: _async_value(_submission_row()))
    monkeypatch.setattr(workflow, "evaluate_source_add_provenance_v2", fake_provenance)
    monkeypatch.setattr(workflow, "_finish_work", fake_finish)

    result = await workflow._process_provenance(_leased_work("provenance"), config=_config())

    assert result == {"status": "completed"}
    assert finished["stage"] == "functional_probe_queued"
    assert "reward_intent" not in finished
    assert finished["next_work"]["work_kind"] == "functional_probe"
    assert finished["probe_config"]["credential_envelope"] == {}
    assert finished["precheck_status"] == PRECHECK_PASSED
    assert observed["sequence"] == 1


@pytest.mark.asyncio
async def test_provenance_authority_uses_retry_sequence(monkeypatch):
    receipt_hash = "sha256:" + "7" * 64
    result = {
        "schema_version": v2_authority.SOURCE_ADD_PROVENANCE_RESULT_SCHEMA_VERSION,
        "submission_id": SUBMISSION_ID,
        "precheck_status": PRECHECK_MANUAL,
        "reasons": ["archive_provider_error"],
        "precheck_doc": {
            "precheck_status": PRECHECK_MANUAL,
            "reasons": ["archive_provider_error"],
        },
    }
    receipt = {
        "receipt_hash": receipt_hash,
        "output_root": sha256_json(result),
    }
    observed = {}

    async def execute(**kwargs):
        observed.update(kwargs)
        return {
            "result": result,
            "receipt": receipt,
            "receipt_graph": {
                "root_receipt_hash": receipt_hash,
                "receipts": [receipt],
                "edges": [],
            },
        }

    async def persist_links(**_kwargs):
        return {"status": "persisted"}

    monkeypatch.setattr(v2_authority, "legacy_v1_enabled", lambda: False)
    provenance, _outcome = await v2_authority.evaluate_source_add_provenance_v2(
        submission_id=SUBMISSION_ID,
        source_name="Credible API",
        source_kind="registry",
        declared_base_domains=["credible.example"],
        source_metadata=_metadata(),
        epoch_id=700,
        sequence=4,
        execute=execute,
        persist_links=persist_links,
    )

    assert provenance.precheck_status == PRECHECK_MANUAL
    assert observed["sequence"] == 4


@pytest.mark.asyncio
async def test_provenance_retry_reuses_identical_existing_authority(monkeypatch):
    current_hash = "sha256:" + "7" * 64
    existing_hash = "sha256:" + "8" * 64
    result = {
        "schema_version": v2_authority.SOURCE_ADD_PROVENANCE_RESULT_SCHEMA_VERSION,
        "submission_id": SUBMISSION_ID,
        "precheck_status": PRECHECK_MANUAL,
        "reasons": ["archive_provider_error"],
        "precheck_doc": {
            "precheck_status": PRECHECK_MANUAL,
            "reasons": ["archive_provider_error"],
        },
    }
    output_root = sha256_json(result)
    current_receipt = {
        "receipt_hash": current_hash,
        "output_root": output_root,
    }
    existing_receipt = {
        "receipt_hash": existing_hash,
        "role": "gateway_coordinator",
        "purpose": "research_lab.source_add_provenance.v2",
        "status": "succeeded",
        "output_root": output_root,
    }
    existing_graph = {
        "root_receipt_hash": existing_hash,
        "receipts": [existing_receipt],
        "edges": [],
    }
    observed = {}

    async def execute(**_kwargs):
        return {
            "result": result,
            "receipt": current_receipt,
            "receipt_graph": {
                "root_receipt_hash": current_hash,
                "receipts": [current_receipt],
                "edges": [],
            },
        }

    async def persist_links(**_kwargs):
        raise attested_v2_store.AttestedV2StoreError(
            "research_lab_attested_business_artifact_links_v2 "
            "stored row conflicts at receipt_hash"
        )

    async def load_existing(**kwargs):
        observed.update(kwargs)
        return existing_graph

    monkeypatch.setattr(v2_authority, "legacy_v1_enabled", lambda: False)
    monkeypatch.setattr(
        v2_authority,
        "validate_receipt_graph",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        attested_v2_store,
        "load_business_artifact_graph_v2",
        load_existing,
    )

    provenance, outcome = await v2_authority.evaluate_source_add_provenance_v2(
        submission_id=SUBMISSION_ID,
        source_name="Credible API",
        source_kind="registry",
        declared_base_domains=["credible.example"],
        source_metadata=_metadata(),
        epoch_id=700,
        sequence=2,
        execute=execute,
        persist_links=persist_links,
    )

    assert provenance.precheck_status == PRECHECK_MANUAL
    assert outcome["receipt"]["receipt_hash"] == existing_hash
    assert outcome["receipt_graph"] == existing_graph
    assert outcome["artifact_link_status"] == {
        "status": "reused_existing_authority",
        "receipt_hash": existing_hash,
    }
    assert observed == {
        "artifact_kind": "source_add_provenance",
        "artifact_ref": SUBMISSION_ID,
        "artifact_hash": output_root,
    }


@pytest.mark.asyncio
async def test_provenance_retry_rejects_different_existing_authority(monkeypatch):
    current_hash = "sha256:" + "7" * 64
    existing_hash = "sha256:" + "8" * 64
    result = {
        "schema_version": v2_authority.SOURCE_ADD_PROVENANCE_RESULT_SCHEMA_VERSION,
        "submission_id": SUBMISSION_ID,
        "precheck_status": PRECHECK_MANUAL,
        "reasons": ["archive_provider_error"],
        "precheck_doc": {
            "precheck_status": PRECHECK_MANUAL,
            "reasons": ["archive_provider_error"],
        },
    }
    output_root = sha256_json(result)
    current_receipt = {
        "receipt_hash": current_hash,
        "output_root": output_root,
    }

    async def execute(**_kwargs):
        return {
            "result": result,
            "receipt": current_receipt,
            "receipt_graph": {
                "root_receipt_hash": current_hash,
                "receipts": [current_receipt],
                "edges": [],
            },
        }

    async def persist_links(**_kwargs):
        raise attested_v2_store.AttestedV2StoreError(
            "research_lab_attested_business_artifact_links_v2 "
            "stored row conflicts at receipt_hash"
        )

    async def load_existing(**_kwargs):
        return {
            "root_receipt_hash": existing_hash,
            "receipts": [
                {
                    "receipt_hash": existing_hash,
                    "role": "gateway_coordinator",
                    "purpose": "research_lab.source_add_provenance.v2",
                    "status": "succeeded",
                    "output_root": "sha256:" + "9" * 64,
                }
            ],
            "edges": [],
        }

    monkeypatch.setattr(v2_authority, "legacy_v1_enabled", lambda: False)
    monkeypatch.setattr(
        v2_authority,
        "validate_receipt_graph",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        attested_v2_store,
        "load_business_artifact_graph_v2",
        load_existing,
    )

    with pytest.raises(
        v2_authority.ResearchLabV2AuthorityError,
        match="existing SOURCE_ADD provenance authority differs",
    ):
        await v2_authority.evaluate_source_add_provenance_v2(
            submission_id=SUBMISSION_ID,
            source_name="Credible API",
            source_kind="registry",
            declared_base_domains=["credible.example"],
            source_metadata=_metadata(),
            epoch_id=700,
            sequence=2,
            execute=execute,
            persist_links=persist_links,
        )


@pytest.mark.asyncio
async def test_manual_provenance_never_queues_probe_or_reward(monkeypatch):
    finished = {}

    async def fake_provenance(**_kwargs):
        return (
            SourceAddProvenanceResult(
                PRECHECK_MANUAL,
                ("documentation_provider_error",),
                {},
            ),
            {"receipt": {"receipt_hash": "sha256:" + "8" * 64}},
        )

    async def fake_finish(_work, **kwargs):
        finished.update(kwargs)
        return {"status": "retry_wait"}

    monkeypatch.setattr(workflow, "_load_submission", lambda _sid: _async_value(_submission_row()))
    monkeypatch.setattr(workflow, "evaluate_source_add_provenance_v2", fake_provenance)
    monkeypatch.setattr(workflow, "_finish_work", fake_finish)

    await workflow._process_provenance(_leased_work("provenance"), config=_config())

    assert finished["disposition"] == "retry"
    assert finished["stage"] == PRECHECK_MANUAL
    assert "next_work" not in finished
    assert "reward_intent" not in finished


@pytest.mark.parametrize(
    ("http_status", "expected_disposition"),
    (
        (400, "retry"),
        (408, "retry"),
        (429, "retry"),
        (503, "retry"),
        (404, "complete"),
        (410, "complete"),
    ),
)
@pytest.mark.asyncio
async def test_documentation_fetch_retries_only_transient_statuses(
    monkeypatch, http_status, expected_disposition
):
    finished = {}

    async def fake_provenance(**_kwargs):
        return (
            SourceAddProvenanceResult(
                PRECHECK_MANUAL,
                ("documentation_fetch_failed",),
                {"docs_fetch": {"provider_status": "ok", "status": http_status}},
            ),
            {"receipt": {"receipt_hash": "sha256:" + "8" * 64}},
        )

    async def fake_finish(_work, **kwargs):
        finished.update(kwargs)
        return {
            "status": (
                "retry_wait"
                if kwargs["disposition"] == "retry"
                else "completed"
            )
        }

    monkeypatch.setattr(
        workflow, "_load_submission", lambda _sid: _async_value(_submission_row())
    )
    monkeypatch.setattr(
        workflow, "evaluate_source_add_provenance_v2", fake_provenance
    )
    monkeypatch.setattr(workflow, "_finish_work", fake_finish)

    await workflow._process_provenance(
        _leased_work("provenance"), config=_config()
    )

    assert finished["disposition"] == expected_disposition
    assert finished["stage"] == PRECHECK_MANUAL
    if expected_disposition == "retry":
        assert "next_work" not in finished
        assert "reward_intent" not in finished
    else:
        assert finished["next_work"] == {}
        assert "reward_intent" not in finished


@pytest.mark.asyncio
async def test_exact_functional_pass_records_proof_without_early_leg1(monkeypatch):
    config_ref = "source_add_probe_config:0123456789abcdef"
    result_doc = {
        "schema_version": "leadpoet.source_add_functional_probe_result.v2",
        "evaluator_version": "leadpoet.source_add_functional_probe_evaluator.v2.1",
        "submission_id": SUBMISSION_ID,
        "adapter_id": ADAPTER_ID,
        "config_ref": config_ref,
        "evaluation_mode": "functional_probe",
        "result_status": "passed",
        "route_hash": "sha256:" + "2" * 64,
        "selected_probe_index": 0,
        "response_hash": "sha256:" + "3" * 64,
        "status_class": "2xx",
        "content_type": "application/json",
        "byte_count": 128,
        "duration_ms": 15,
        "retry_after_seconds": 0,
        "reason_codes": ["bounded_json_data_response"],
        "probe_summaries": [],
    }
    receipt_hash = "sha256:" + "4" * 64
    finished = {}

    async def fake_select_one(table, **_kwargs):
        assert table == "research_lab_source_add_probe_config_current"
        return {
            "config_ref": config_ref,
            "probe_doc": {
                "base_url": "https://api.credible.example/v1",
                "auth_kind": "none",
            },
            "credential_envelope": {},
        }

    async def fake_probe(**_kwargs):
        return result_doc, {
            "receipt": {
                "receipt_hash": receipt_hash,
                "output_root": sha256_json(result_doc),
            }
        }

    async def fake_finish(_work, **kwargs):
        finished.update(kwargs)
        return {"status": "completed"}

    monkeypatch.setattr(workflow, "_load_submission", lambda _sid: _async_value(_submission_row(precheck_status=PRECHECK_PASSED)))
    monkeypatch.setattr(workflow, "select_one", fake_select_one)
    monkeypatch.setattr(workflow, "evaluate_source_add_functional_probe_v2", fake_probe)
    monkeypatch.setattr(workflow, "_finish_work", fake_finish)

    await workflow._process_functional_probe(_leased_work("functional_probe"), config=_config())

    assert finished["stage"] == "functional_probe_passed"
    assert finished["functional_attempt"]["receipt_hash"] == receipt_hash
    assert "reward_intent" not in finished
    assert "next_work" not in finished


@pytest.mark.asyncio
async def test_generic_dns_failure_remains_retryable_without_claiming_nxdomain(monkeypatch):
    result_doc = {
        "schema_version": "leadpoet.source_add_functional_probe_result.v2",
        "evaluator_version": "leadpoet.source_add_functional_probe_evaluator.v2.1",
        "submission_id": SUBMISSION_ID,
        "adapter_id": ADAPTER_ID,
        "config_ref": "source_add_probe_config:0123456789abcdef",
        "evaluation_mode": "functional_probe",
        "result_status": "retryable",
        "route_hash": "sha256:" + "2" * 64,
        "selected_probe_index": -1,
        "response_hash": "sha256:" + "3" * 64,
        "status_class": "network_error",
        "content_type": "",
        "byte_count": 0,
        "duration_ms": 5,
        "retry_after_seconds": 0,
        "reason_codes": ["dns_failure"],
        "probe_summaries": [],
    }
    finished = {}

    async def fake_select_one(_table, **_kwargs):
        return {
            "config_ref": result_doc["config_ref"],
            "probe_doc": {"base_url": "https://api.credible.example/v1", "auth_kind": "none"},
            "credential_envelope": {},
        }

    async def fake_probe(**_kwargs):
        return result_doc, {
            "receipt": {
                "receipt_hash": "sha256:" + "4" * 64,
                "output_root": sha256_json(result_doc),
            }
        }

    async def fake_finish(_work, **kwargs):
        finished.update(kwargs)
        return {"status": "completed"}

    monkeypatch.setattr(workflow, "_load_submission", lambda _sid: _async_value(_submission_row(precheck_status=PRECHECK_PASSED)))
    monkeypatch.setattr(workflow, "select_one", fake_select_one)
    monkeypatch.setattr(workflow, "evaluate_source_add_functional_probe_v2", fake_probe)
    monkeypatch.setattr(workflow, "_finish_work", fake_finish)

    await workflow._process_functional_probe(_leased_work("functional_probe"), config=_config())

    assert finished["stage"] == "functional_probe_retryable"
    assert finished["disposition"] == "retry"
    assert finished["functional_attempt"]["result_doc"] == result_doc
    assert finished["functional_attempt"]["business_artifact_hash"] == sha256_json(result_doc)
    assert "workflow_result_status" not in finished["functional_attempt"]["result_doc"]
    assert finished["result_doc"] == result_doc


@pytest.mark.asyncio
async def test_generic_dns_failure_exhaustion_requires_manual_review(monkeypatch):
    result_doc = {
        "schema_version": "leadpoet.source_add_functional_probe_result.v2",
        "evaluator_version": "leadpoet.source_add_functional_probe_evaluator.v2.1",
        "submission_id": SUBMISSION_ID,
        "adapter_id": ADAPTER_ID,
        "config_ref": "source_add_probe_config:0123456789abcdef",
        "evaluation_mode": "functional_probe",
        "result_status": "retryable",
        "route_hash": "sha256:" + "2" * 64,
        "selected_probe_index": -1,
        "response_hash": "",
        "status_class": "network_error",
        "content_type": "",
        "byte_count": 0,
        "duration_ms": 5,
        "retry_after_seconds": 0,
        "reason_codes": ["dns_failure"],
        "probe_summaries": [],
    }
    finished = {}

    async def fake_select_one(_table, **_kwargs):
        return {
            "config_ref": result_doc["config_ref"],
            "probe_doc": {
                "base_url": "https://api.credible.example/v1",
                "auth_kind": "none",
            },
            "credential_envelope": {},
        }

    async def fake_probe(**_kwargs):
        return result_doc, {
            "receipt": {
                "receipt_hash": "sha256:" + "4" * 64,
                "output_root": sha256_json(result_doc),
            }
        }

    async def fake_finish(_work, **kwargs):
        finished.update(kwargs)
        return {"status": "completed"}

    monkeypatch.setattr(
        workflow,
        "_load_submission",
        lambda _sid: _async_value(
            _submission_row(precheck_status=PRECHECK_PASSED)
        ),
    )
    monkeypatch.setattr(workflow, "select_one", fake_select_one)
    monkeypatch.setattr(
        workflow, "evaluate_source_add_functional_probe_v2", fake_probe
    )
    monkeypatch.setattr(workflow, "_finish_work", fake_finish)

    await workflow._process_functional_probe(
        _leased_work("functional_probe", attempt_count=5),
        config=_config(source_add_probe_max_attempts=5),
    )

    assert finished["disposition"] == "complete"
    assert finished["stage"] == "needs_manual_review"
    assert finished["release_identity"] is False
    assert finished["result_doc"] == result_doc


@pytest.mark.asyncio
async def test_disabled_functional_rewards_remain_retryable(monkeypatch):
    finished = {}

    async def fake_finish(_work, **kwargs):
        finished.update(kwargs)
        return {"status": "retry_wait"}

    monkeypatch.setattr(workflow, "_finish_work", fake_finish)

    result = await workflow._process_leg1_reward(
        _leased_work("leg1_reward"),
        config=_config(source_add_functional_rewards_enabled=False),
    )

    assert result == {"status": "retry_wait"}
    assert finished["disposition"] == "retry"
    assert finished["result_doc"]["status"] == "functional_rewards_disabled"


@pytest.mark.asyncio
async def test_reward_worker_exception_never_dead_letters(monkeypatch):
    finished = {}

    async def fake_finish(_work, **kwargs):
        finished.update(kwargs)
        return {"status": "retry_wait"}

    async def fake_select_one(_table, **_kwargs):
        return None

    monkeypatch.setattr(workflow, "_finish_work", fake_finish)
    monkeypatch.setattr(workflow, "select_one", fake_select_one)

    await workflow._recover_failed_claim(
        _leased_work("leg1_reward", attempt_count=50, created_at="2000-01-01T00:00:00+00:00"),
        config=_config(source_add_probe_max_attempts=1),
    )

    assert finished["disposition"] == "retry"
    assert finished["result_doc"]["status"] == "reward_worker_exception_retry"
    assert "stage" not in finished


@pytest.mark.asyncio
async def test_leg1_finalization_binds_reward_decision_receipt(monkeypatch):
    config_ref = "source_add_probe_config:0123456789abcdef"
    functional_result = {
        "schema_version": "leadpoet.source_add_functional_probe_result.v2",
        "evaluator_version": "leadpoet.source_add_functional_probe_evaluator.v2.1",
        "submission_id": SUBMISSION_ID,
        "adapter_id": ADAPTER_ID,
        "config_ref": config_ref,
        "evaluation_mode": "functional_probe",
        "result_status": "passed",
        "route_hash": "sha256:" + "2" * 64,
    }
    functional_hash = sha256_json(functional_result)
    functional_receipt = "sha256:" + "3" * 64
    decision_receipt = "sha256:" + "4" * 64
    decision_artifact = "sha256:" + "5" * 64
    smoke_receipt = "sha256:" + "6" * 64
    smoke_result = {
        **functional_result,
        "evaluation_mode": "provisioning_smoke",
        "route_hash": "sha256:" + "7" * 64,
    }
    smoke_hash = sha256_json(smoke_result)
    catalog_id = "source_catalog:0123456789abcdef"
    provision_ref = "source_add_provision:0123456789abcdef"
    registry_provider_id = "sourceadd_credible_api"
    finalized_payload = {}
    authorized_payload = {}

    async def fake_select_one(table, **_kwargs):
        if table == "research_lab_source_add_reward_intents":
            return {
                "intent_id": "source_add_reward_intent:0123456789abcdef",
                "functional_receipt_hash": functional_receipt,
                "business_artifact_hash": functional_hash,
            }
        if table == "research_lab_source_add_functional_probe_current":
            return {
                "submission_id": SUBMISSION_ID,
                "adapter_id": ADAPTER_ID,
                "attempt_ref": "source_add_probe_attempt:0123456789abcdef",
                "config_ref": config_ref,
                "result_status": "passed",
                "receipt_hash": functional_receipt,
                "business_artifact_hash": functional_hash,
                "result_doc": functional_result,
            }
        if table == "research_lab_source_add_provisioning_current":
            return {
                "provision_ref": provision_ref,
                "catalog_id": catalog_id,
                "submission_id": SUBMISSION_ID,
                "adapter_id": ADAPTER_ID,
                "miner_hotkey": MINER_HOTKEY,
                "registry_provider_id": registry_provider_id,
                "provision_status": "provisioned_autoresearch_eligible",
            }
        if table == "research_lab_source_add_provisioning_smoke_current":
            return {
                "attempt_ref": "source_add_probe_attempt:fedcba9876543210",
                "submission_id": SUBMISSION_ID,
                "adapter_id": ADAPTER_ID,
                "config_ref": config_ref,
                "evaluation_mode": "provisioning_smoke",
                "result_status": "passed",
                "receipt_hash": smoke_receipt,
                "business_artifact_hash": smoke_hash,
                "result_doc": smoke_result,
            }
        if table == "research_lab_source_catalog":
            return {
                "catalog_id": catalog_id,
                "adapter_id": ADAPTER_ID,
                "miner_ref": MINER_HOTKEY,
                "registry_provider_id": registry_provider_id,
            }
        if table == "research_lab_source_add_reward_current":
            return None
        raise AssertionError(table)

    async def fake_select_many(table, **_kwargs):
        assert table == "research_lab_source_add_submissions"
        return [
            _submission_row(
                stage="accepted",
                precheck_status="provenance_precheck_passed",
            )
        ]

    async def fake_rpc(name, params):
        if name == "research_lab_source_add_reserve_leg1_slot":
            return {
                "status": "reserved",
                "slot_lease_token": "22222222-2222-2222-2222-222222222222",
            }
        if name == "research_lab_source_add_finalize_leg1":
            finalized_payload.update(params)
            return {
                "status": "created",
                "reward_ref": params["p_reward"]["reward_ref"],
            }
        raise AssertionError(name)

    async def fake_authorize(**kwargs):
        authorized_payload.update(kwargs)
        return {
            "status": "matched",
            "execution_receipt": {
                "receipt_hash": decision_receipt,
                "output_root": decision_artifact,
            },
        }

    monkeypatch.setattr(
        workflow,
        "_load_submission",
        lambda _sid: _async_value(_submission_row(stage="leg1_queued")),
    )
    monkeypatch.setattr(workflow, "select_one", fake_select_one)
    monkeypatch.setattr(workflow, "select_many", fake_select_many)
    monkeypatch.setattr(workflow, "_rpc", fake_rpc)
    monkeypatch.setattr(workflow, "authorize_reward_decision_v2", fake_authorize)
    monkeypatch.setattr(
        workflow,
        "resolve_research_lab_evaluation_epoch",
        lambda _epoch: _async_value((700, 0, "test")),
    )
    monkeypatch.setattr(
        "gateway.research_lab.attested_v2_store.load_business_artifact_graph_v2",
        lambda **kwargs: _async_value(
            {
                "root_receipt_hash": (
                    functional_receipt
                    if kwargs["artifact_kind"] == "source_add_functional_probe"
                    else smoke_receipt
                )
            }
        ),
    )

    result = await workflow._process_leg1_reward(
        _leased_work(
            "leg1_reward",
            job_doc={"intent_id": "source_add_reward_intent:0123456789abcdef"},
        ),
        config=_config(),
    )

    assert result["status"] == "created"
    assert finalized_payload["p_reward"]["decision_receipt_hash"] == decision_receipt
    assert finalized_payload["p_reward"]["decision_artifact_hash"] == decision_artifact
    assert finalized_payload["p_reward"]["start_epoch"] == 701
    trigger = finalized_payload["p_reward"]["trigger_evidence_doc"]
    assert trigger["final_acceptance_stage"] == "accepted"
    assert trigger["catalog_id"] == catalog_id
    assert trigger["provisioning_smoke_receipt_hash"] == smoke_receipt
    assert tuple(
        graph["root_receipt_hash"] for graph in authorized_payload["parent_graphs"]
    ) == (functional_receipt, smoke_receipt)


@pytest.mark.asyncio
async def test_allocation_reads_active_source_add_reward_without_catalog(monkeypatch):
    async def fake_select_all(table, *, filters=(), **_kwargs):
        assert table == "research_lab_source_add_reward_current"
        if ("current_reward_status", "active") not in filters:
            return []
        return [
            {
                "reward_ref": "source_add_reward:" + "1" * 16,
                "adapter_id": ADAPTER_ID,
                "catalog_id": None,
                "miner_hotkey": MINER_HOTKEY,
                "leg": 1,
                "reward_kind": "source_acceptance",
                "current_reward_status": "active",
                "desired_alpha_percent": 1.0,
                "start_epoch": 700,
                "epoch_count": 20,
            }
        ]

    monkeypatch.setattr(allocations, "select_all", fake_select_all)

    rows = await allocations._active_source_add_reward_rows(701)

    assert len(rows) == 1
    assert rows[0]["miner_hotkey"] == MINER_HOTKEY
    assert rows[0]["desired_alpha_percent"] == pytest.approx(1.0)
    assert rows[0]["reward_kind"] == "source_acceptance"


async def _async_value(value):
    return value
