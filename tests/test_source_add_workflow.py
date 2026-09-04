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
async def test_source_add_control_state_reads_durable_pause(monkeypatch):
    async def fake_select_one(*_args, **_kwargs):
        return {
            "paused": True,
            "reason": "operator rollout hold",
            "updated_at": "2026-08-08T00:00:00+00:00",
        }

    monkeypatch.setattr(source_add_workflow, "select_one", fake_select_one)

    state = await source_add_workflow.source_add_control_state()

    assert state == {
        "paused": True,
        "status": "paused",
        "reason": "operator rollout hold",
        "updated_at": "2026-08-08T00:00:00+00:00",
        "unavailable": False,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_mode", ["missing", "read_error"])
async def test_source_add_control_state_fails_closed(
    monkeypatch, caplog, failure_mode
):
    caplog.set_level(logging.WARNING, logger=source_add_workflow.__name__)

    async def fake_select_one(*_args, **_kwargs):
        if failure_mode == "read_error":
            raise RuntimeError("storage unavailable")
        return None

    monkeypatch.setattr(source_add_workflow, "select_one", fake_select_one)

    state = await source_add_workflow.source_add_control_state()

    assert state == {
        "paused": True,
        "status": "unavailable_fail_closed",
        "unavailable": True,
    }
    assert "SOURCE_ADD_CONTROL_" in caplog.text


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


@pytest.mark.asyncio
async def test_dispatcher_logs_durable_pause_once_and_resume(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger=source_add_workflow.__name__)
    claims = iter(
        [
            {"status": "paused"},
            {"status": "paused"},
            {"status": "empty"},
        ]
    )
    sleep_calls = 0

    async def fake_rpc(name, _params):
        if name == "research_lab_source_add_reconcile_provenance_leg1_v1":
            return {"status": "reconciled"}
        assert name == "research_lab_source_add_claim_work"
        return next(claims)

    async def fake_sleep(_seconds: float) -> None:
        nonlocal sleep_calls
        sleep_calls += 1
        if sleep_calls == 3:
            raise asyncio.CancelledError

    monkeypatch.setattr(source_add_workflow, "_rpc", fake_rpc)
    monkeypatch.setattr(source_add_workflow.asyncio, "sleep", fake_sleep)
    config = SimpleNamespace(
        source_add_enabled=True,
        source_add_dispatcher_enabled=True,
        source_add_dispatcher_poll_seconds=0.25,
        source_add_work_lease_seconds=120,
    )

    with pytest.raises(asyncio.CancelledError):
        await source_add_workflow.run_source_add_dispatcher(
            config_supplier=lambda: config
        )

    assert caplog.text.count("SOURCE_ADD_DISPATCHER_PAUSED") == 1
    assert caplog.text.count("SOURCE_ADD_DISPATCHER_RESUMED") == 1


@pytest.mark.asyncio
async def test_dispatcher_claims_work_when_leg1_reconciliation_times_out(
    monkeypatch, caplog
):
    caplog.set_level(logging.WARNING, logger=source_add_workflow.__name__)
    rpc_calls: list[str] = []
    processed_work: list[dict] = []
    claimed_work = {
        "work_id": "source_add_work:" + "a" * 16,
        "work_kind": "leg1_reward",
    }
    claims = iter(
        [
            {"status": "empty"},
            {"status": "claimed", "work": claimed_work},
            {"status": "empty"},
        ]
    )

    async def fake_rpc(name, _params):
        rpc_calls.append(name)
        if name == "research_lab_source_add_reconcile_provenance_leg1_v1":
            raise TimeoutError("reconciliation timed out")
        assert name == "research_lab_source_add_claim_work"
        return next(claims)

    async def fake_process(work, *, config):
        assert config is dispatcher_config
        processed_work.append(dict(work))

    sleep_calls = 0

    async def fake_sleep(_seconds: float) -> None:
        nonlocal sleep_calls
        sleep_calls += 1
        if sleep_calls == 2:
            raise asyncio.CancelledError

    monkeypatch.setattr(source_add_workflow, "_rpc", fake_rpc)
    monkeypatch.setattr(
        source_add_workflow, "process_source_add_work_item", fake_process
    )
    monkeypatch.setattr(source_add_workflow.asyncio, "sleep", fake_sleep)
    dispatcher_config = SimpleNamespace(
        source_add_enabled=True,
        source_add_dispatcher_enabled=True,
        source_add_dispatcher_poll_seconds=0.25,
        source_add_work_lease_seconds=120,
    )

    with pytest.raises(asyncio.CancelledError):
        await source_add_workflow.run_source_add_dispatcher(
            config_supplier=lambda: dispatcher_config
        )

    assert rpc_calls == [
        "research_lab_source_add_claim_work",
        "research_lab_source_add_reconcile_provenance_leg1_v1",
        "research_lab_source_add_claim_work",
        "research_lab_source_add_claim_work",
    ]
    assert processed_work == [claimed_work]
    assert "SOURCE_ADD_LEG1_RECONCILIATION_FAILED type=TimeoutError" in caplog.text


def test_historical_leg1_rebuild_removes_only_host_routing_reason():
    row = {
        "submission_id": "source_add_submission:" + "a" * 16,
        "precheck_status": "provenance_precheck_passed",
        "precheck_doc": {
            "precheck_status": "provenance_precheck_passed",
            "reasons": [
                "provenance_reference_backed",
                "automatic_safe_get_selected",
            ],
            "docs_completeness": {"score": 5},
        },
        "submission_doc": {},
    }

    result = source_add_workflow._provenance_result_from_submission(row)

    assert result == {
        "schema_version": "leadpoet.source_add_provenance_result.v2",
        "submission_id": row["submission_id"],
        "precheck_status": "provenance_precheck_passed",
        "reasons": ["provenance_reference_backed"],
        "precheck_doc": {
            "precheck_status": "provenance_precheck_passed",
            "reasons": ["provenance_reference_backed"],
            "docs_completeness": {"score": 5},
        },
    }


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
                "provision_doc": {
                    "provider_registry_entry": {
                        "base_url": "https://api.test-source.example/v1"
                    }
                },
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


def _functional_proof(work: dict) -> dict:
    return {
        "attempt_ref": "source_add_probe_attempt:1111111111111111",
        "submission_id": work["submission_id"],
        "adapter_id": work["adapter_id"],
        "result_status": "passed",
        "receipt_hash": "sha256:" + "6" * 64,
        "business_artifact_hash": "sha256:" + "7" * 64,
    }


@pytest.mark.asyncio
async def test_provisioning_smoke_pass_finalizes_with_exact_work_lease(monkeypatch):
    work = _smoke_work()
    result = _smoke_result("passed")

    async def fake_load(_submission_id):
        return {
            "submission_id": work["submission_id"],
            "adapter_id": work["adapter_id"],
            "miner_hotkey": "hk-owner",
            "submission_doc": {
                "manifest": {},
                "source_metadata": {
                    "api_base_url": "https://api.test-source.example/v1"
                },
            },
            "precheck_status": "provenance_precheck_passed",
            "precheck_doc": {},
        }

    async def fake_select_one(table, **_kwargs):
        assert table == "research_lab_source_add_functional_probe_current"
        return _functional_proof(work)

    async def fake_evaluate(**kwargs):
        assert kwargs["evaluation_mode"] == "provisioning_smoke"
        assert kwargs["sequence"] == 1
        return result, {
            "receipt": {
                "receipt_hash": "sha256:" + "5" * 64,
                "output_root": "sha256:" + "5" * 64,
            },
            "execution_receipt": {
                "receipt_hash": "sha256:" + "4" * 64,
                "output_root": sha256_json(result),
            },
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
    monkeypatch.setattr(source_add_workflow, "select_one", fake_select_one)
    monkeypatch.setattr(source_add_workflow, "_rpc", fake_rpc)
    # In-memory test replacement only: production exposes no request/env bypass.
    monkeypatch.setattr(
        source_add_workflow.source_add_catalog_contract,
        "source_add_api_is_current_builtin_sync",
        lambda *_args, **_kwargs: False,
    )

    response = await source_add_workflow.process_source_add_work_item(
        work, config=_workflow_config()
    )

    assert response["status"] == "provisioned"
    assert observed["name"] == "research_lab_source_add_finalize_provision_smoke_v3"
    assert observed["params"]["p_work_id"] == work["work_id"]
    assert observed["params"]["p_lease_token"] == work["lease_token"]
    smoke = observed["params"]["p_smoke_attempt"]
    assert smoke["work_id"] == work["work_id"]
    assert smoke["attempt_number"] == 1
    assert smoke["evaluation_mode"] == "provisioning_smoke"
    assert smoke["receipt_hash"] == "sha256:" + "4" * 64
    assert smoke["business_artifact_hash"] == sha256_json(result)
    assert set(observed["params"]) == {
        "p_work_id",
        "p_lease_token",
        "p_submission_id",
        "p_catalog_row",
        "p_provision_row",
        "p_smoke_attempt",
    }
    assert "reward" not in str(observed["params"])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("rpc_status", "should_fail_closed"),
    (("not_eligible", False), ("completed", True)),
)
async def test_current_model_domain_added_after_admission_blocks_final_acceptance(
    monkeypatch,
    rpc_status,
    should_fail_closed,
):
    from gateway.research_lab import provider_evidence_proxy, source_add_catalog

    work = _smoke_work()
    work["job_doc"]["catalog_row"] = {
        "catalog_id": "source_catalog:0123456789abcdef",
        "adapter_id": work["adapter_id"],
    }
    work["job_doc"]["provision_row"].update(
        {
            "provision_ref": "source_add_provision:0123456789abcdef",
            "submission_id": work["submission_id"],
            "miner_hotkey": "hk-owner",
            "registry_provider_id": "test_source",
            "credential_envelope": {
                "credential_ref": "encrypted_ref:source_add:synthetic"
            },
        }
    )
    work["job_doc"]["provision_row"]["provision_doc"].update(
        {
            "secret_note": "synthetic-redacted-value",
            "provider_registry_entry": {
                "id": "test_source",
                "base_url": "https://api.test-source.example/v1",
                "active": True,
            },
        }
    )
    result = _smoke_result("passed")
    current_domains: set[str] = set()
    monkeypatch.setattr(
        provider_evidence_proxy,
        "reserved_builtin_provider_domains_sync",
        lambda: set(current_domains),
    )
    assert source_add_catalog.source_add_api_is_current_builtin_sync(
        "https://api.test-source.example/v1"
    ) is False
    current_domains.add("api.test-source.example")

    async def fake_load(_submission_id):
        return {
            "submission_id": work["submission_id"],
            "adapter_id": work["adapter_id"],
            "miner_hotkey": "hk-owner",
            "submission_doc": {
                "manifest": {},
                "source_metadata": {
                    "api_base_url": "https://api.test-source.example/v1"
                },
            },
            "precheck_status": "provenance_precheck_passed",
            "precheck_doc": {},
        }

    async def fake_evaluate(**_kwargs):
        return result, {
            "execution_receipt": {
                "receipt_hash": "sha256:" + "4" * 64,
                "output_root": sha256_json(result),
            }
        }

    observed = {}

    async def fail_finish(*_args, **_kwargs):
        raise AssertionError("current-model provider rejection must be atomic")

    async def fake_rpc(name, params):
        observed["name"] = name
        observed["params"] = params
        return {"status": rpc_status}

    async def fail_select(*_args, **_kwargs):
        raise AssertionError("current-model provider must not load reward proof")

    monkeypatch.setattr(source_add_workflow, "_load_submission", fake_load)
    monkeypatch.setattr(
        source_add_workflow,
        "_begin_provider_execution",
        lambda value: asyncio.sleep(0, result=dict(value)),
    )
    monkeypatch.setattr(
        source_add_workflow, "evaluate_source_add_functional_probe_v2", fake_evaluate
    )
    monkeypatch.setattr(source_add_workflow, "_finish_work", fail_finish)
    monkeypatch.setattr(source_add_workflow, "_rpc", fake_rpc)
    monkeypatch.setattr(source_add_workflow, "select_one", fail_select)

    if should_fail_closed:
        with pytest.raises(
            source_add_workflow.SourceAddWorkflowError,
            match="rejection did not finalize",
        ):
            await source_add_workflow.process_source_add_work_item(
                work, config=_workflow_config()
            )
    else:
        response = await source_add_workflow.process_source_add_work_item(
            work, config=_workflow_config()
        )
        assert response == {"status": "not_eligible"}

    assert observed["name"] == "research_lab_source_add_reject_current_builtin_v3"
    params = observed["params"]
    assert params["p_work_id"] == work["work_id"]
    assert params["p_lease_token"] == work["lease_token"]
    assert params["p_smoke_attempt"]["result_status"] == "passed"
    disabled = params["p_disabled_provision_row"]
    assert disabled["provision_status"] == "disabled"
    assert disabled["provision_ref"].startswith("source_add_provision:")
    assert disabled["provision_ref"] != "source_add_provision:0123456789abcdef"
    assert disabled["provision_doc"]["provider_registry_entry"]["active"] is False
    assert disabled["provision_doc"]["secret_note"] == "[redacted]"
    assert disabled["credential_envelope"] == {
        "credential_ref": "encrypted_ref:source_add:synthetic"
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("attempt_count", "finish_status", "expected_disposition"),
    ((1, "retry_wait", "retry"), (5, "completed", "complete")),
)
async def test_current_model_catalog_read_failure_blocks_final_acceptance(
    monkeypatch,
    caplog,
    attempt_count,
    finish_status,
    expected_disposition,
):
    caplog.set_level(logging.WARNING, logger=source_add_workflow.__name__)
    work = _smoke_work()
    work["attempt_count"] = attempt_count
    result = _smoke_result("passed")

    async def fake_load(_submission_id):
        return {
            "submission_id": work["submission_id"],
            "adapter_id": work["adapter_id"],
            "miner_hotkey": "hk-owner",
            "submission_doc": {
                "manifest": {},
                "source_metadata": {
                    "api_base_url": "https://api.test-source.example/v1"
                },
            },
            "precheck_status": "provenance_precheck_passed",
            "precheck_doc": {},
        }

    async def fake_evaluate(**_kwargs):
        return result, {
            "execution_receipt": {
                "receipt_hash": "sha256:" + "4" * 64,
                "output_root": sha256_json(result),
            }
        }

    observed = {}

    async def fake_finish(_work, **kwargs):
        observed.update(kwargs)
        return {"status": finish_status}

    async def fail_rpc(*_args, **_kwargs):
        raise AssertionError("unreadable catalog must not reach finalizer")

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
    monkeypatch.setattr(source_add_workflow, "_rpc", fail_rpc)
    monkeypatch.setattr(
        source_add_workflow.source_add_catalog_contract,
        "source_add_api_is_current_builtin_sync",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("sensitive catalog failure detail")
        ),
    )

    response = await source_add_workflow.process_source_add_work_item(
        work, config=_workflow_config()
    )

    assert response == {"status": finish_status}
    assert observed["disposition"] == expected_disposition
    assert observed["result_doc"] == {"status": "current_model_catalog_unavailable"}
    assert (observed["available_at"] is not None) is (
        expected_disposition == "retry"
    )
    assert "reward_intent" not in observed
    assert "next_work" not in observed
    assert "sensitive catalog failure detail" not in caplog.text
    assert "type=RuntimeError" in caplog.text


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
async def test_provisioning_smoke_terminal_failure_persists_requeue_authority(
    monkeypatch,
):
    work = _smoke_work()
    result = _smoke_result("failed")

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
        return {"status": "completed"}

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

    assert response["status"] == "completed"
    assert observed["disposition"] == "complete"
    assert observed["available_at"] is None
    assert observed["result_doc"] == result
    assert observed["functional_attempt"]["result_status"] == "failed"
    assert observed["functional_attempt"]["attempt_number"] == work["attempt_count"]


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
