from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import logging
from types import SimpleNamespace

import pytest

from gateway.research_lab import source_add_workflow
from leadpoet_canonical.attested_v2 import sha256_json


def test_automatic_probe_selection_rejects_encoded_or_whitespace_paths():
    config, reason = source_add_workflow.build_automatic_probe_config(
        submission_id="source_add_submission:" + "a" * 16,
        adapter_id="adapter:test-source",
        source_metadata={
            "api_base_url": "https://api.example.com/v1",
            "auth_type": "none",
            "endpoint_examples": [
                {
                    "method": "GET",
                    "path": "/records%2Fadmin",
                    "purpose": "encoded route",
                    "example_query": "q=test",
                },
                {
                    "method": "GET",
                    "path": "/record search",
                    "purpose": "space route",
                    "example_query": "q=test",
                },
            ],
        },
    )

    assert config is None
    assert reason == "operator_probe_configuration_required"


def test_automatic_probe_selection_rejects_encoded_api_base_path():
    config, reason = source_add_workflow.build_automatic_probe_config(
        submission_id="source_add_submission:" + "a" * 16,
        adapter_id="adapter:test-source",
        source_metadata={
            "api_base_url": "https://api.example.com/v1%2Fadmin",
            "auth_type": "none",
            "endpoint_examples": [
                {
                    "method": "GET",
                    "path": "/records",
                    "purpose": "records",
                    "example_query": "q=test",
                }
            ],
        },
    )

    assert config is None
    assert reason == "https_api_base_url_required"


@pytest.mark.asyncio
async def test_dispatcher_survives_config_read_failure_and_fails_closed(
    monkeypatch, caplog
):
    caplog.set_level(logging.WARNING, logger=source_add_workflow.__name__)
    supplier_calls = 0
    sleep_calls: list[float] = []

    def config_supplier():
        nonlocal supplier_calls
        supplier_calls += 1
        if supplier_calls == 1:
            raise RuntimeError("transient config failure")
        return SimpleNamespace(
            source_add_enabled=False,
            source_add_dispatcher_enabled=False,
            source_add_dispatcher_poll_seconds=0.25,
        )

    async def fake_sleep(seconds: float) -> None:
        sleep_calls.append(float(seconds))
        if len(sleep_calls) == 2:
            raise asyncio.CancelledError

    monkeypatch.setattr(source_add_workflow.asyncio, "sleep", fake_sleep)

    with pytest.raises(asyncio.CancelledError):
        await source_add_workflow.run_source_add_dispatcher(
            config_supplier=config_supplier
        )

    assert supplier_calls == 2
    assert sleep_calls == [2.0, 0.25]
    assert "SOURCE_ADD_DISPATCHER_LOOP_FAILED type=RuntimeError" in caplog.text


def _smoke_result(status: str) -> dict:
    return {
        "schema_version": "leadpoet.source_add_functional_probe_result.v2",
        "submission_id": "source_add_submission:" + "a" * 16,
        "adapter_id": "adapter:test-source",
        "config_ref": "source_add_probe_config:0123456789abcdef",
        "evaluation_mode": "provisioning_smoke",
        "evaluator_version": "source-add-functional-probe-v2",
        "result_status": status,
        "route_hash": "sha256:" + "1" * 64,
        "response_hash": "sha256:" + "2" * 64,
        "status_class": "2xx" if status == "passed" else "5xx",
        "content_type": "application/json",
        "byte_count": 42,
        "duration_ms": 25,
        "retry_after_seconds": 0,
        "reason_codes": ["functional_json_passed"] if status == "passed" else ["upstream_5xx"],
    }


def _smoke_work() -> dict:
    return {
        "work_id": "source_add_work:" + "b" * 16,
        "submission_id": "source_add_submission:" + "a" * 16,
        "adapter_id": "adapter:test-source",
        "work_kind": "provisioning_smoke",
        "attempt_count": 1,
        "lease_token": "00000000-0000-0000-0000-000000000001",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "job_doc": {
            "config_ref": "source_add_probe_config:0123456789abcdef",
            "host_hash": "sha256:" + "3" * 64,
            "catalog_row": {"adapter_id": "adapter:test-source"},
            "provision_row": {
                "adapter_id": "adapter:test-source",
                "provision_status": "provisioned_autoresearch_eligible",
            },
        },
    }


def _workflow_config() -> SimpleNamespace:
    return SimpleNamespace(
        source_add_functional_probes_enabled=True,
        source_add_probe_timeout_seconds=45,
        source_add_probe_max_attempts=5,
        evaluation_epoch=10,
    )


@pytest.mark.asyncio
async def test_provisioning_smoke_pass_finalizes_with_exact_work_lease(monkeypatch):
    work = _smoke_work()
    result = _smoke_result("passed")

    async def fake_load(_submission_id):
        return {
            "submission_id": work["submission_id"],
            "adapter_id": work["adapter_id"],
            "miner_hotkey": "hk-owner",
            "submission_doc": {"manifest": {}, "source_metadata": {}},
            "precheck_status": "provenance_precheck_passed",
            "precheck_doc": {},
        }

    async def fake_evaluate(**kwargs):
        assert kwargs["evaluation_mode"] == "provisioning_smoke"
        assert kwargs["sequence"] == 1
        return result, {
            "receipt": {
                "receipt_hash": "sha256:" + "4" * 64,
                "output_root": sha256_json(result),
            }
        }

    observed = {}

    async def fake_rpc(name, params):
        observed["name"] = name
        observed["params"] = params
        return {"status": "provisioned", "catalog_id": "source_catalog:" + "c" * 16}

    monkeypatch.setattr(source_add_workflow, "_load_submission", fake_load)
    monkeypatch.setattr(
        source_add_workflow,
        "_begin_provider_execution",
        lambda value: asyncio.sleep(0, result=dict(value)),
    )
    monkeypatch.setattr(
        source_add_workflow, "evaluate_source_add_functional_probe_v2", fake_evaluate
    )
    monkeypatch.setattr(source_add_workflow, "_rpc", fake_rpc)

    response = await source_add_workflow.process_source_add_work_item(
        work, config=_workflow_config()
    )

    assert response["status"] == "provisioned"
    assert observed["name"] == "research_lab_source_add_finalize_provision_smoke"
    assert observed["params"]["p_work_id"] == work["work_id"]
    assert observed["params"]["p_lease_token"] == work["lease_token"]
    smoke = observed["params"]["p_smoke_attempt"]
    assert smoke["work_id"] == work["work_id"]
    assert smoke["attempt_number"] == 1
    assert smoke["evaluation_mode"] == "provisioning_smoke"
    assert smoke["business_artifact_hash"] == sha256_json(result)


@pytest.mark.asyncio
async def test_provisioning_smoke_transient_failure_persists_retry(monkeypatch):
    work = _smoke_work()
    result = _smoke_result("retryable")

    async def fake_load(_submission_id):
        return {
            "submission_id": work["submission_id"],
            "adapter_id": work["adapter_id"],
            "miner_hotkey": "hk-owner",
            "submission_doc": {"manifest": {}, "source_metadata": {}},
            "precheck_status": "provenance_precheck_passed",
            "precheck_doc": {},
        }

    async def fake_evaluate(**_kwargs):
        return result, {
            "receipt": {
                "receipt_hash": "sha256:" + "4" * 64,
                "output_root": sha256_json(result),
            }
        }

    observed = {}

    async def fake_finish(_work, **kwargs):
        observed.update(kwargs)
        return {"status": "retry_wait"}

    monkeypatch.setattr(source_add_workflow, "_load_submission", fake_load)
    monkeypatch.setattr(
        source_add_workflow,
        "_begin_provider_execution",
        lambda value: asyncio.sleep(0, result=dict(value)),
    )
    monkeypatch.setattr(
        source_add_workflow, "evaluate_source_add_functional_probe_v2", fake_evaluate
    )
    monkeypatch.setattr(source_add_workflow, "_finish_work", fake_finish)

    response = await source_add_workflow.process_source_add_work_item(
        work, config=_workflow_config()
    )

    assert response["status"] == "retry_wait"
    assert observed["disposition"] == "retry"
    assert observed["available_at"] is not None
    assert observed["functional_attempt"]["evaluation_mode"] == "provisioning_smoke"
    assert observed["functional_attempt"]["work_id"] == work["work_id"]


@pytest.mark.asyncio
async def test_uncertain_expired_provider_execution_never_calls_provider(monkeypatch):
    work = _smoke_work()
    work["job_doc"] = {
        **work["job_doc"],
        "provider_execution_state": "started",
        "provider_execution_attempt": 1,
        "provider_execution_recovery": "uncertain_after_lease_expiry",
    }

    async def fail_evaluate(**_kwargs):
        raise AssertionError("uncertain recovery must not call the provider")

    async def fake_load(_submission_id):
        return {
            "submission_id": work["submission_id"],
            "adapter_id": work["adapter_id"],
            "miner_hotkey": "hk-owner",
            "submission_doc": {"manifest": {}, "source_metadata": {}},
            "precheck_status": "provenance_precheck_passed",
            "precheck_doc": {},
        }

    observed = {}

    async def fake_finish(_work, **kwargs):
        observed.update(kwargs)
        return {"status": "completed"}

    monkeypatch.setattr(source_add_workflow, "_load_submission", fake_load)
    monkeypatch.setattr(
        source_add_workflow, "evaluate_source_add_functional_probe_v2", fail_evaluate
    )
    monkeypatch.setattr(source_add_workflow, "_finish_work", fake_finish)

    response = await source_add_workflow.process_source_add_work_item(
        work, config=_workflow_config()
    )

    assert response == {"status": "completed"}
    assert observed["disposition"] == "complete"
    assert observed["stage"] == ""
    assert observed["result_doc"]["status"] == (
        "provider_execution_outcome_unknown_after_worker_loss"
    )


@pytest.mark.asyncio
async def test_worker_exception_after_execution_fence_fails_closed(monkeypatch):
    work = _smoke_work()
    persisted = {
        **work,
        "job_doc": {
            **work["job_doc"],
            "provider_execution_state": "started",
            "provider_execution_attempt": 1,
        },
    }
    observed = {}

    async def fake_select(_table, **_kwargs):
        return persisted

    async def fake_uncertain(value):
        observed["work"] = value
        return {"status": "completed"}

    monkeypatch.setattr(source_add_workflow, "select_one", fake_select)
    monkeypatch.setattr(
        source_add_workflow, "_finish_uncertain_provider_outcome", fake_uncertain
    )

    await source_add_workflow._recover_failed_claim(
        work, config=_workflow_config()
    )

    assert observed["work"]["work_id"] == work["work_id"]
