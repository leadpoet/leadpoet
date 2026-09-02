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
        "source_add_leg1_alpha_percent": 0.2,
        "source_add_leg1_max_per_utc_day": 50,
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
    ("http_status", "attempt_count", "expected_disposition"),
    (
        (400, 1, "retry"),
        (400, 2, "retry"),
        (400, 3, "complete"),
        (408, 3, "retry"),
        (429, 3, "retry"),
        (503, 3, "retry"),
        (404, 1, "complete"),
        (410, 1, "complete"),
    ),
)
@pytest.mark.asyncio
async def test_documentation_fetch_retries_only_transient_statuses(
    monkeypatch, http_status, attempt_count, expected_disposition
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

    work = _leased_work("provenance")
    work["attempt_count"] = attempt_count
    await workflow._process_provenance(work, config=_config())

    assert finished["disposition"] == expected_disposition
    assert finished["stage"] == PRECHECK_MANUAL
    if expected_disposition == "retry":
        assert "next_work" not in finished
        assert "reward_intent" not in finished
    else:
        assert finished["next_work"] == {}
        assert "reward_intent" not in finished


@pytest.mark.parametrize("error_type", ("response_too_large", "tls_failure"))
@pytest.mark.asyncio
async def test_terminal_documentation_provider_errors_do_not_retry(
    monkeypatch, error_type
):
    finished = {}

    async def fake_provenance(**_kwargs):
        return (
            SourceAddProvenanceResult(
                PRECHECK_MANUAL,
                ("documentation_provider_error",),
                {
                    "docs_fetch": {
                        "provider_status": "error",
                        "status": 0,
                        "error_type": error_type,
                    }
                },
            ),
            {"receipt": {"receipt_hash": "sha256:" + "8" * 64}},
        )

    async def fake_finish(_work, **kwargs):
        finished.update(kwargs)
        return {"status": "completed"}

    monkeypatch.setattr(
        workflow, "_load_submission", lambda _sid: _async_value(_submission_row())
    )
    monkeypatch.setattr(
        workflow, "evaluate_source_add_provenance_v2", fake_provenance
    )
    monkeypatch.setattr(workflow, "_finish_work", fake_finish)

    result = await workflow._process_provenance(
        _leased_work("provenance"), config=_config()
    )

    assert result == {"status": "completed"}
    assert finished["disposition"] == "complete"
    assert finished["stage"] == PRECHECK_MANUAL
    assert finished["next_work"] == {}
    assert "available_at" not in finished
    assert "reward_intent" not in finished


@pytest.mark.asyncio
async def test_exact_functional_pass_waits_for_final_acceptance(monkeypatch):
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
async def test_leg1_fifo_wait_is_returned_without_reward_side_effects(monkeypatch):
    work = _leased_work(
        "leg1_reward",
        job_doc={"intent_id": "source_add_reward_intent:0123456789abcdef"},
    )

    async def fake_select_one(table, **_kwargs):
        assert table == "research_lab_source_add_reward_intents"
        return {"intent_id": work["job_doc"]["intent_id"]}

    async def fake_rpc(name, params):
        assert name == "research_lab_source_add_reserve_leg1_slot_v3"
        assert params["p_work_id"] == work["work_id"]
        assert params["p_work_lease_token"] == work["lease_token"]
        return {"status": "fifo_wait", "available_at": "2026-08-31T00:00:00Z"}

    async def fail_load(*_args, **_kwargs):
        raise AssertionError("fifo wait must not build or finalize a reward")

    monkeypatch.setattr(workflow, "select_one", fake_select_one)
    monkeypatch.setattr(workflow, "_rpc", fake_rpc)
    monkeypatch.setattr(workflow, "_load_submission", fail_load)

    result = await workflow._process_leg1_reward(work, config=_config())

    assert result == {
        "status": "fifo_wait",
        "available_at": "2026-08-31T00:00:00Z",
    }


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
async def test_leg1_finalization_binds_and_recovers_exact_decision_receipt(
    monkeypatch,
):
    functional_result = {
        "schema_version": "leadpoet.source_add_functional_probe_result.v2",
        "evaluator_version": "leadpoet.source_add_functional_probe_evaluator.v2.1",
        "result_status": "passed",
        "route_hash": "sha256:" + "2" * 64,
    }
    functional_hash = sha256_json(functional_result)
    functional_receipt = "sha256:" + "3" * 64
    smoke_result = {
        **functional_result,
        "evaluation_mode": "provisioning_smoke",
    }
    smoke_hash = sha256_json(smoke_result)
    smoke_receipt = "sha256:" + "6" * 64
    decision_receipt = "sha256:" + "4" * 64
    decision_artifacts = []
    existing_decision = {"available": False}
    finalize_status = {"value": "created"}
    finalized_payload = {}
    authorize_payload = {}

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
                "config_ref": "source_add_probe_config:0123456789abcdef",
                "result_status": "passed",
                "receipt_hash": functional_receipt,
                "business_artifact_hash": functional_hash,
                "result_doc": functional_result,
            }
        if table == "research_lab_source_add_provisioning_smoke_current":
            return {
                "submission_id": SUBMISSION_ID,
                "adapter_id": ADAPTER_ID,
                "attempt_ref": "source_add_probe_attempt:fedcba9876543210",
                "evaluation_mode": "provisioning_smoke",
                "config_ref": "source_add_probe_config:0123456789abcdef",
                "result_status": "passed",
                "receipt_hash": smoke_receipt,
                "business_artifact_hash": smoke_hash,
                "result_doc": smoke_result,
            }
        if table == "research_lab_source_add_provisioning_current":
            return {
                "submission_id": SUBMISSION_ID,
                "adapter_id": ADAPTER_ID,
                "miner_hotkey": MINER_HOTKEY,
                "provision_ref": "source_add_provision:0123456789abcdef",
                "catalog_id": "source_catalog:0123456789abcdef",
                "registry_provider_id": "credible_api",
                "provision_status": "provisioned_autoresearch_eligible",
            }
        if table == "research_lab_source_add_reward_current":
            return None
        if table == "research_lab_attested_business_artifact_links_v2":
            if existing_decision["available"]:
                return {
                    "receipt_hash": decision_receipt,
                    "artifact_hash": decision_artifacts[0],
                }
            return None
        raise AssertionError(table)

    async def fake_rpc(name, params):
        if name == "research_lab_source_add_reserve_leg1_slot_v3":
            return {
                "status": "reserved",
                "slot_lease_token": "22222222-2222-2222-2222-222222222222",
            }
        if name == "research_lab_source_add_finalize_leg1_v3":
            finalized_payload.update(params)
            return {
                "status": finalize_status["value"],
                "reward_ref": params["p_reward"]["reward_ref"],
            }
        raise AssertionError(name)

    async def fake_authorize(**kwargs):
        authorize_payload.update(kwargs)
        decision_artifact = sha256_json(
            workflow.source_add_reward_row_projection_v2(
                "source_add_leg1",
                kwargs["expected_result"]["reward"],
            )
        )
        decision_artifacts.append(decision_artifact)
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
        lambda _sid: _async_value(_submission_row(stage="accepted")),
    )
    monkeypatch.setattr(workflow, "select_one", fake_select_one)
    monkeypatch.setattr(workflow, "_rpc", fake_rpc)
    monkeypatch.setattr(workflow, "authorize_reward_decision_v2", fake_authorize)
    monkeypatch.setattr(
        workflow,
        "resolve_research_lab_evaluation_epoch",
        lambda _epoch: _async_value((700, 0, "test")),
    )
    async def load_graph(**kwargs):
        if kwargs["artifact_kind"] == "source_add_functional_probe":
            return {"root_receipt_hash": functional_receipt}
        if kwargs["artifact_kind"] == "source_add_provisioning_smoke":
            return {"root_receipt_hash": smoke_receipt}
        assert kwargs == {
            "artifact_kind": "source_add_reward_decision",
            "artifact_ref": finalized_payload["p_reward"]["reward_ref"],
            "artifact_hash": decision_artifacts[0],
        }
        return {
            "root_receipt_hash": decision_receipt,
            "receipts": [
                {
                    "receipt_hash": decision_receipt,
                    "purpose": "research_lab.reward_decision.v2",
                    "output_root": decision_artifacts[0],
                    "parent_receipt_hashes": sorted(
                        (functional_receipt, smoke_receipt)
                    ),
                }
            ],
        }

    monkeypatch.setattr(
        "gateway.research_lab.attested_v2_store.load_business_artifact_graph_v2",
        load_graph,
    )

    result = await workflow._process_leg1_reward(
        _leased_work(
            "leg1_reward",
            job_doc={"intent_id": "source_add_reward_intent:0123456789abcdef"},
        ),
        config=_config(),
    )
    existing_decision["available"] = True
    recovered = await workflow._process_leg1_reward(
        _leased_work(
            "leg1_reward",
            job_doc={"intent_id": "source_add_reward_intent:0123456789abcdef"},
        ),
        config=_config(),
    )
    finalize_status["value"] = "fifo_wait"
    waited = await workflow._process_leg1_reward(
        _leased_work(
            "leg1_reward",
            job_doc={"intent_id": "source_add_reward_intent:0123456789abcdef"},
        ),
        config=_config(),
    )

    assert result["status"] == "created"
    assert recovered["status"] == "created"
    assert waited["status"] == "fifo_wait"
    assert len(decision_artifacts) == 1
    assert finalized_payload["p_reward"]["decision_receipt_hash"] == decision_receipt
    assert finalized_payload["p_reward"]["decision_artifact_hash"] == decision_artifacts[0]
    assert finalized_payload["p_reward"]["start_epoch"] == 701
    assert finalized_payload["p_reward"]["alpha_percent"] == pytest.approx(0.2)
    assert finalized_payload["p_daily_cap"] == 50
    assert tuple(
        graph["root_receipt_hash"] for graph in authorize_payload["parent_graphs"]
    ) == (functional_receipt, smoke_receipt)
    assert authorize_payload["decision_payload"]["provisioning_smoke_result"] == (
        smoke_result
    )
    assert finalized_payload["p_reward"]["trigger_evidence_doc"] == {
        "functional_probe_passed": True,
        "attempt_ref": "source_add_probe_attempt:0123456789abcdef",
        "functional_probe_receipt_hash": functional_receipt,
        "business_artifact_hash": functional_hash,
        "functional_probe_result_hash": functional_hash,
        "evaluator_version": functional_result["evaluator_version"],
        "route_hash": functional_result["route_hash"],
        "provisioning_smoke_passed": True,
        "provisioning_smoke_attempt_ref": "source_add_probe_attempt:fedcba9876543210",
        "provisioning_smoke_receipt_hash": smoke_receipt,
        "provisioning_smoke_business_artifact_hash": smoke_hash,
        "provisioning_smoke_result_hash": smoke_hash,
        "submission_id": SUBMISSION_ID,
        "final_acceptance_stage": "accepted",
        "provision_ref": "source_add_provision:0123456789abcdef",
        "catalog_id": "source_catalog:0123456789abcdef",
        "registry_provider_id": "credible_api",
        "provision_status": "provisioned_autoresearch_eligible",
    }


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
