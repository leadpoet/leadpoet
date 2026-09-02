"""Disposable-PostgreSQL coverage for automatic provenance-era Leg 1."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gateway.research_lab.source_add_provenance import (
    rebuild_attested_provenance_result_v2,
)
from gateway.research_lab.source_add_workflow import (
    source_add_probe_attempt_ref,
    source_add_reward_intent_id,
    source_add_work_id,
)
from gateway.tee.supabase_schema_preflight_v2 import (
    SOURCE_ADD_PROVENANCE_LEG1_FUNCTION_AUTHORITY_SHA256,
    SOURCE_ADD_PROVENANCE_LEG1_TRIGGER_AUTHORITY_SHA256,
    SOURCE_ADD_PROVENANCE_LEG1_VIEW_AUTHORITY_SHA256,
)
from gateway.tee.reward_executor_v2 import source_add_reward_row_projection_v2
from leadpoet_canonical.attested_v2 import sha256_json
from research_lab.source_add_identity import (
    normalize_source_add_provider_origin,
    source_provider_origin_hash,
)
from research_lab.source_add_rewards import create_leg1_reward
from tests.test_source_add_claim_control_postgres import _insert_work
from tests.test_source_add_end_to_end_postgres import (
    _create_seed_leg1_reward,
    _database_with_migrations,
    _json,
    _scalar,
    _seed_boot_identity,
    _seed_business_link,
    _seed_leased_smoke_case,
    _seed_receipt,
)


ROOT = Path(__file__).resolve().parents[1]
MIGRATION = "175-research-lab-source-add-provenance-leg1.sql"
PRE_MIGRATIONS = (
    "72-research-lab-source-experiments.sql",
    "74-research-lab-source-add-provenance-precheck.sql",
    "78-research-lab-source-add-catalog-provisioning.sql",
    "79-research-lab-source-add-llm-leg2-evidence.sql",
    "82-research-lab-source-add-llm-only-leg2.sql",
    "84-expand-source-add-source-kinds.sql",
    "86-research-lab-attested-v2-authority.sql",
    "96-research-lab-source-add-functional-workflow.sql",
    "145-research-lab-source-add-admission-control.sql",
    "169-research-lab-source-add-post-accept-leg1.sql",
    "170-research-lab-source-add-provider-origin-uniqueness.sql",
    "171-research-lab-source-add-duplicate-privacy.sql",
    "172-research-lab-source-add-claim-control.sql",
    "173-research-lab-source-add-leg1-release-policy.sql",
    "174-research-lab-source-add-restart-state-restore.sql",
)


@pytest.fixture(scope="module")
def pre_migration_database():
    yield from _database_with_migrations(PRE_MIGRATIONS)


def _set_paused(cursor, paused: bool, reason: str) -> None:
    _scalar(
        cursor,
        "SELECT public.research_lab_source_add_set_paused(%s,%s,%s)",
        (paused, reason, "operator:migration-175-test"),
    )


def _record(seed: int) -> dict:
    token = f"{seed:016x}"
    base_url = f"https://api-{token}.migration-175.test/v1"
    origin = normalize_source_add_provider_origin(base_url)
    return {
        "submission_id": "source_add_submission:" + token,
        "adapter_id": "adapter:migration-175-" + token,
        "miner_hotkey": "5Migration175" + token,
        "credential_envelope": {},
        "provider_origin_host": origin,
        "provider_origin_hash": source_provider_origin_hash(base_url),
        "manifest": {
            "credential_policy": "no_credentials",
            "credential_ref": "",
            "source_name": "Migration 175 " + token,
            "source_kind": "registry",
            "declared_base_domains": [origin],
        },
        "source_metadata": {
            "api_base_url": base_url,
            "documentation_url": f"https://docs-{token}.migration-175.test",
            "auth_type": "none",
            "endpoint_examples": [
                {
                    "method": "GET",
                    "path": "/records",
                    "purpose": "Return bounded records",
                    "example_query": "limit=1",
                }
            ],
            "rate_limit_notes": "bounded",
        },
    }


def _admit_and_claim(cursor, seed: int) -> tuple[dict, dict]:
    record = _record(seed)
    submission_id = record["submission_id"]
    work_id = "source_add_work:" + f"{seed + 10_000:016x}"
    admitted = _scalar(
        cursor,
        """
        SELECT public.research_lab_source_add_admit_v2(
            %s::JSONB,%s,%s,%s,%s,%s,3,5,10
        )
        """,
        (
            _json(record),
            sha256_json({"primary": submission_id}),
            sha256_json({"documentation": submission_id}),
            sha256_json({"legacy": submission_id}),
            record["provider_origin_hash"],
            work_id,
        ),
    )
    assert admitted["status"] == "admitted"
    cursor.execute(
        """
        UPDATE public.research_lab_source_add_work_items
        SET available_at=NOW() + INTERVAL '1 hour'
        WHERE work_kind='leg1_reward'
          AND work_status='queued'
        """
    )
    claimed = _scalar(
        cursor,
        "SELECT public.research_lab_source_add_claim_work(%s,180)",
        ("migration-175-provenance-" + f"{seed:016x}",),
    )["work"]
    assert claimed["work_id"] == work_id
    cursor.execute(
        """
        UPDATE public.research_lab_source_add_work_items
        SET available_at=NOW()
        WHERE work_kind='leg1_reward'
          AND work_status='queued'
        """
    )
    return record, claimed


def _finish_provenance(
    cursor,
    *,
    record: dict,
    work: dict,
    status: str = "provenance_precheck_passed",
    attested: bool = True,
    historical_routing_reason: str = "",
) -> dict:
    signed_reasons: list[str] = []
    signed_precheck = {"precheck_status": status, "reasons": signed_reasons}
    provenance = {
        "schema_version": "leadpoet.source_add_provenance_result.v2",
        "submission_id": record["submission_id"],
        "precheck_status": status,
        "reasons": signed_reasons,
        "precheck_doc": signed_precheck,
    }
    stored_precheck = dict(signed_precheck)
    if historical_routing_reason:
        stored_precheck["reasons"] = [historical_routing_reason]
    document = dict(record)
    receipt_hash = ""
    artifact_hash = ""
    if attested:
        receipt_hash = sha256_json(
            {"purpose": "provenance", "submission": record["submission_id"]}
        )
        artifact_hash = sha256_json(provenance)
        _seed_receipt(
            cursor,
            receipt_hash=receipt_hash,
            purpose="research_lab.source_add_provenance.v2",
            job_id="migration-175-provenance-" + record["submission_id"][-16:],
            output_root=artifact_hash,
            sequence=20_000 + int(record["submission_id"][-6:], 16),
        )
        _seed_business_link(
            cursor,
            receipt_hash=receipt_hash,
            artifact_kind="source_add_provenance",
            artifact_ref=record["submission_id"],
            artifact_hash=artifact_hash,
        )
        document.update(
            {
                "provenance_receipt_hash": receipt_hash,
            }
        )
        if not historical_routing_reason:
            document["provenance_artifact_hash"] = artifact_hash
            document["provenance_result"] = provenance
    result = _scalar(
        cursor,
        """
        SELECT public.research_lab_source_add_finish_work(
            %s,%s::UUID,'complete',%s,%s::JSONB,%s,%s::JSONB,%s::JSONB,
            '{}'::JSONB,'{}'::JSONB,'{}'::JSONB,'{}'::JSONB,
            NULL,FALSE
        )
        """,
        (
            work["work_id"],
            work["lease_token"],
            status,
            _json(document),
            status,
            _json(stored_precheck),
            _json({"status": status}),
        ),
    )
    assert result["status"] == "completed"
    return {
        "record": record,
        "document": document,
        "precheck_doc": stored_precheck,
        "provenance": provenance,
        "receipt_hash": receipt_hash,
        "artifact_hash": artifact_hash,
    }


def _seed_case(cursor, seed: int, **finish_options) -> dict:
    record, work = _admit_and_claim(cursor, seed)
    return _finish_provenance(
        cursor,
        record=record,
        work=work,
        **finish_options,
    )


def _append_attested_provenance(cursor, case: dict, seed: int) -> dict:
    record = dict(case["record_doc"])
    submission_id = case["submission_id"]
    precheck = {
        "precheck_status": "provenance_precheck_passed",
        "reasons": [],
    }
    provenance = {
        "schema_version": "leadpoet.source_add_provenance_result.v2",
        "submission_id": submission_id,
        "precheck_status": "provenance_precheck_passed",
        "reasons": [],
        "precheck_doc": precheck,
    }
    receipt_hash = sha256_json({"paid-provenance": submission_id})
    artifact_hash = sha256_json(provenance)
    source_identity_hash = _scalar(
        cursor,
        "SELECT source_identity_hash FROM public.research_lab_source_add_submission_current WHERE submission_id=%s",
        (submission_id,),
    )
    _seed_receipt(
        cursor,
        receipt_hash=receipt_hash,
        purpose="research_lab.source_add_provenance.v2",
        job_id="migration-175-paid-provenance",
        output_root=artifact_hash,
        sequence=30_000 + seed,
    )
    _seed_business_link(
        cursor,
        receipt_hash=receipt_hash,
        artifact_kind="source_add_provenance",
        artifact_ref=submission_id,
        artifact_hash=artifact_hash,
    )
    record.update(
        {
            "provenance_receipt_hash": receipt_hash,
            "provenance_artifact_hash": artifact_hash,
            "provenance_result": provenance,
            "stage": "provenance_precheck_passed",
        }
    )
    cursor.execute(
        """
        INSERT INTO public.research_lab_source_add_submissions (
            submission_id,adapter_id,miner_hotkey,stage,seq,submission_doc,
            precheck_status,precheck_doc,source_identity_hash,
            source_identity_version
        )
        SELECT %s,%s,%s,'provenance_precheck_passed',
               COALESCE(MAX(history.seq),-1)+1,%s::JSONB,
               'provenance_precheck_passed',%s::JSONB,%s,'v2'
        FROM public.research_lab_source_add_submissions history
        WHERE history.submission_id=%s
        """,
        (
            submission_id,
            case["adapter_id"],
            case["miner_hotkey"],
            _json(record),
            _json(precheck),
            source_identity_hash,
            submission_id,
        ),
    )
    return {
        "record": record,
        "receipt_hash": receipt_hash,
        "artifact_hash": artifact_hash,
    }


def _claim_reward(cursor) -> dict:
    claimed = _scalar(
        cursor,
        "SELECT public.research_lab_source_add_claim_work(%s,300)",
        ("migration-175-reward",),
    )["work"]
    assert claimed["work_kind"] == "leg1_reward"
    return claimed


def _finalize_reward(cursor, *, work: dict, case: dict, caller_cap: int) -> dict:
    intent_id = work["job_doc"]["intent_id"]
    slot = _scalar(
        cursor,
        """
        SELECT public.research_lab_source_add_reserve_leg1_slot_v4(
            %s,%s,%s::UUID,%s,300
        )
        """,
        (intent_id, work["work_id"], work["lease_token"], caller_cap),
    )
    if slot["status"] != "reserved":
        return slot
    trigger = {
        "provenance_precheck_passed": True,
        "submission_id": work["submission_id"],
        "precheck_status": "provenance_precheck_passed",
        "provenance_receipt_hash": case["receipt_hash"],
        "provenance_artifact_hash": case["artifact_hash"],
        "provenance_result_hash": case["artifact_hash"],
    }
    leg1 = create_leg1_reward(
        adapter_id=work["adapter_id"],
        miner_ref=case["record"]["miner_hotkey"],
        start_epoch=50_000,
        alpha_percent=0.2,
        reward_epochs=20,
        trigger_evidence=trigger,
    )
    assert leg1 is not None
    projection = source_add_reward_row_projection_v2(
        "source_add_leg1", leg1.to_dict()
    )
    decision_artifact = sha256_json(projection)
    decision_receipt = sha256_json(
        {"reward-decision": leg1.reward_ref, "artifact": decision_artifact}
    )
    _seed_receipt(
        cursor,
        receipt_hash=decision_receipt,
        purpose="research_lab.reward_decision.v2",
        job_id="migration-175-reward-" + leg1.reward_ref[-16:],
        output_root=decision_artifact,
        sequence=40_000 + int(leg1.reward_ref[-6:], 16),
        parent_receipt_hashes=(case["receipt_hash"],),
    )
    _seed_business_link(
        cursor,
        receipt_hash=decision_receipt,
        artifact_kind="source_add_reward_decision",
        artifact_ref=leg1.reward_ref,
        artifact_hash=decision_artifact,
    )
    payload = {
        "reward_ref": leg1.reward_ref,
        "reward_kind": leg1.reward_kind,
        "alpha_percent": leg1.alpha_percent,
        "reward_epochs": leg1.reward_epochs,
        "start_epoch": leg1.start_epoch,
        "state": leg1.state,
        "trigger_evidence_doc": trigger,
        "public_label": leg1.public_label,
        "decision_receipt_hash": decision_receipt,
        "decision_artifact_hash": decision_artifact,
    }
    return _scalar(
        cursor,
        """
        SELECT public.research_lab_source_add_finalize_leg1_v4(
            %s,%s,%s::UUID,%s::UUID,%s,%s::JSONB,%s::JSONB
        )
        """,
        (
            intent_id,
            work["work_id"],
            work["lease_token"],
            slot["slot_lease_token"],
            caller_cap,
            _json(payload),
            _json(case["record"]),
        ),
    )


def _finish_functional_attempt(
    cursor,
    *,
    work: dict,
    case: dict,
    config_ref: str,
    status: str,
    receipt_hash: str = "",
    artifact_hash: str = "",
) -> dict:
    current_document = _scalar(
        cursor,
        "SELECT submission_doc FROM public.research_lab_source_add_submission_current WHERE submission_id=%s",
        (work["submission_id"],),
    )
    attempt_ref = source_add_probe_attempt_ref(
        work["submission_id"], work["work_id"], int(work["attempt_count"])
    )
    result_doc = {
        "schema_version": "leadpoet.source_add_functional_probe_result.v2",
        "evaluator_version": "leadpoet.source_add_functional_probe_evaluator.v2.1",
        "submission_id": work["submission_id"],
        "adapter_id": work["adapter_id"],
        "config_ref": config_ref,
        "evaluation_mode": (
            "provisioning_smoke"
            if work["work_kind"] == "provisioning_smoke"
            else "functional_probe"
        ),
        "result_status": status,
        "route_hash": sha256_json({"route": attempt_ref}),
    }
    attempt = {
        "attempt_ref": attempt_ref,
        "work_id": work["work_id"],
        "attempt_number": int(work["attempt_count"]),
        "evaluation_mode": result_doc["evaluation_mode"],
        "config_ref": config_ref,
        "result_status": status,
        "route_hash": result_doc["route_hash"],
        "response_hash": sha256_json({"response": attempt_ref}),
        "status_class": "2xx" if status == "passed" else "5xx",
        "content_type": "application/json",
        "byte_count": 32,
        "duration_ms": 5,
        "retry_after_seconds": 0,
        "reason_codes": ["bounded_json_data_response"],
        "receipt_hash": receipt_hash,
        "business_artifact_hash": artifact_hash,
        "result_doc": result_doc,
    }
    result = _scalar(
        cursor,
        """
        SELECT public.research_lab_source_add_finish_work(
            %s,%s::UUID,'complete',%s,%s::JSONB,
            'provenance_precheck_passed',%s::JSONB,%s::JSONB,
            %s::JSONB,'{}'::JSONB,'{}'::JSONB,'{}'::JSONB,NULL,FALSE
        )
        """,
        (
            work["work_id"],
            work["lease_token"],
            "functional_probe_passed"
            if work["work_kind"] == "functional_probe" and status == "passed"
            else "",
            _json(current_document),
            _json(case["precheck_doc"]),
            _json({"result_status": status}),
            _json(attempt),
        ),
    )
    assert result["status"] == "completed"
    return attempt


def _provision_after_leg1(
    cursor,
    case: dict,
    *,
    reject_current_builtin: bool = False,
    allow_unrewarded: bool = False,
) -> tuple[str, tuple] | None:
    submission_id = case["record"]["submission_id"]
    adapter_id = case["record"]["adapter_id"]
    config_ref = "source_add_probe_config:" + submission_id[-16:]
    functional_work_id = "source_add_work:" + sha256_json(
        {"functional": submission_id}
    )[7:23]
    probe_doc = {
        "schema_version": "leadpoet.source_add_probe_config.v2",
        "provider_id": "sourceadd_" + submission_id[-16:],
        "base_url": case["record"]["source_metadata"]["api_base_url"],
        "auth_kind": "none",
        "auth_name": "",
        "request_headers": {},
        "probes": [
            {"method": "GET", "path": "/records", "query": {}, "body_json": None}
        ],
    }
    configured = _scalar(
        cursor,
        """
        SELECT public.research_lab_source_add_configure_probe_v3(
            %s,%s,%s::JSONB,'{}'::JSONB,%s,%s,%s
        )
        """,
        (
            submission_id,
            config_ref,
            _json(probe_doc),
            "operator:migration-175-test",
            functional_work_id,
            sha256_json({"host": submission_id}),
        ),
    )
    assert configured["status"] == "queued"
    cursor.execute(
        """
        UPDATE public.research_lab_source_add_work_items
        SET available_at=NOW() + INTERVAL '1 hour'
        WHERE work_kind='leg1_reward'
          AND work_status='queued'
        """
    )
    functional = _scalar(
        cursor,
        "SELECT public.research_lab_source_add_claim_work(%s,180)",
        ("migration-175-functional",),
    )["work"]
    assert functional["work_id"] == functional_work_id
    functional_result = {
        "schema_version": "leadpoet.source_add_functional_probe_result.v2",
        "evaluator_version": "leadpoet.source_add_functional_probe_evaluator.v2.1",
        "submission_id": submission_id,
        "adapter_id": adapter_id,
        "config_ref": config_ref,
        "evaluation_mode": "functional_probe",
        "result_status": "passed",
        "route_hash": sha256_json({"functional-route": submission_id}),
    }
    functional_receipt = sha256_json({"functional-receipt": submission_id})
    functional_artifact = sha256_json(functional_result)
    attempt_ref = source_add_probe_attempt_ref(
        submission_id, functional_work_id, int(functional["attempt_count"])
    )
    _seed_receipt(
        cursor,
        receipt_hash=functional_receipt,
        purpose="research_lab.source_add_functional_probe.v2",
        job_id="migration-175-functional-" + submission_id[-16:],
        output_root=functional_artifact,
        sequence=70_000 + int(submission_id[-6:], 16),
    )
    _seed_business_link(
        cursor,
        receipt_hash=functional_receipt,
        artifact_kind="source_add_functional_probe",
        artifact_ref=attempt_ref,
        artifact_hash=functional_artifact,
    )
    functional_attempt = {
        "attempt_ref": attempt_ref,
        "work_id": functional_work_id,
        "attempt_number": int(functional["attempt_count"]),
        "evaluation_mode": "functional_probe",
        "config_ref": config_ref,
        "result_status": "passed",
        "route_hash": functional_result["route_hash"],
        "response_hash": sha256_json({"functional-response": submission_id}),
        "status_class": "2xx",
        "content_type": "application/json",
        "byte_count": 32,
        "duration_ms": 5,
        "retry_after_seconds": 0,
        "reason_codes": ["bounded_json_data_response"],
        "receipt_hash": functional_receipt,
        "business_artifact_hash": functional_artifact,
        "result_doc": functional_result,
    }
    current_document = _scalar(
        cursor,
        "SELECT submission_doc FROM public.research_lab_source_add_submission_current WHERE submission_id=%s",
        (submission_id,),
    )
    finished = _scalar(
        cursor,
        """
        SELECT public.research_lab_source_add_finish_work(
            %s,%s::UUID,'complete','functional_probe_passed',%s::JSONB,
            'provenance_precheck_passed',%s::JSONB,'{}'::JSONB,
            %s::JSONB,'{}'::JSONB,'{}'::JSONB,'{}'::JSONB,NULL,FALSE
        )
        """,
        (
            functional_work_id,
            functional["lease_token"],
            _json(current_document),
            _json(case["precheck_doc"]),
            _json(functional_attempt),
        ),
    )
    assert finished["status"] == "completed"

    catalog_id = "source_catalog:" + submission_id[-16:]
    provider_id = "sourceadd_" + submission_id[-16:]
    catalog = {
        "catalog_id": catalog_id,
        "adapter_id": adapter_id,
        "miner_ref": case["record"]["miner_hotkey"],
        "source_name": case["record"]["manifest"]["source_name"],
        "source_kind": "registry",
        "declared_base_domains": case["record"]["manifest"]["declared_base_domains"],
        "registry_provider_id": provider_id,
        "catalog_doc": {"source": "migration-175-test"},
        "source_identity_hash": _scalar(
            cursor,
            "SELECT source_identity_hash FROM public.research_lab_source_add_submission_current WHERE submission_id=%s",
            (submission_id,),
        ),
    }
    provision_doc = {
        "provider_registry_entry": {
            "provider_id": provider_id,
            "base_url": probe_doc["base_url"],
            "auth_kind": "none",
            "auth_name": "",
            "active": True,
        },
        "request_headers": {},
        "probe_endpoints": [{"method": "GET", "path": "/records"}],
    }

    def provision_row(suffix: str, status: str) -> dict:
        return {
            "provision_ref": "source_add_provision:" + suffix,
            "submission_id": submission_id,
            "adapter_id": adapter_id,
            "miner_hotkey": case["record"]["miner_hotkey"],
            "source_identity_hash": catalog["source_identity_hash"],
            "registry_provider_id": provider_id,
            "provision_status": status,
            "provision_doc": provision_doc,
            "credential_envelope": {},
        }

    pending = provision_row(
        sha256_json({"pending": submission_id})[7:23],
        "approved_pending_provision",
    )
    assert _scalar(
        cursor,
        "SELECT public.research_lab_source_add_finalize_provision_v3(%s,%s::JSONB,%s::JSONB,'{}'::JSONB)",
        (submission_id, _json(catalog), _json(pending)),
    )["status"] == "provisioned"
    eligible = provision_row(
        sha256_json({"eligible": submission_id})[7:23],
        "provisioned_autoresearch_eligible",
    )
    smoke_work_id = "source_add_work:" + sha256_json({"smoke": submission_id})[7:23]
    enqueue_args = (
        smoke_work_id,
        submission_id,
        config_ref,
        sha256_json({"smoke-host": submission_id}),
        _json(catalog),
        _json(eligible),
    )
    assert _scalar(
        cursor,
        """
        SELECT public.research_lab_source_add_enqueue_provision_smoke_v2(
            %s,%s,%s,%s,%s::JSONB,%s::JSONB
        )
        """,
        enqueue_args,
    )["status"] == "queued"
    smoke = _scalar(
        cursor,
        "SELECT public.research_lab_source_add_claim_work(%s,180)",
        ("migration-175-smoke-failed",),
    )["work"]
    assert smoke["work_id"] == smoke_work_id
    _finish_functional_attempt(
        cursor,
        work=smoke,
        case=case,
        config_ref=config_ref,
        status="failed",
    )
    retried = _scalar(
        cursor,
        """
        SELECT public.research_lab_source_add_enqueue_provision_smoke_v2(
            %s,%s,%s,%s,%s::JSONB,%s::JSONB
        )
        """,
        enqueue_args,
    )
    assert retried == {
        "status": "queued",
        "work_id": smoke_work_id,
        "work_status": "queued",
        "requeued": True,
    }
    smoke = _scalar(
        cursor,
        "SELECT public.research_lab_source_add_claim_work(%s,180)",
        ("migration-175-smoke-passed",),
    )["work"]
    smoke_attempt_ref = source_add_probe_attempt_ref(
        submission_id, smoke_work_id, int(smoke["attempt_count"])
    )
    smoke_result = {
        **functional_result,
        "evaluation_mode": "provisioning_smoke",
        "route_hash": sha256_json({"smoke-route": smoke_attempt_ref}),
    }
    smoke_receipt = sha256_json({"smoke-receipt": smoke_attempt_ref})
    smoke_artifact = sha256_json(smoke_result)
    _seed_receipt(
        cursor,
        receipt_hash=smoke_receipt,
        purpose="research_lab.source_add_functional_probe.v2",
        job_id="migration-175-smoke-" + submission_id[-16:],
        output_root=smoke_artifact,
        sequence=80_000 + int(submission_id[-6:], 16),
    )
    _seed_business_link(
        cursor,
        receipt_hash=smoke_receipt,
        artifact_kind="source_add_provisioning_smoke",
        artifact_ref=smoke_attempt_ref,
        artifact_hash=smoke_artifact,
    )
    smoke_attempt = {
        "attempt_ref": smoke_attempt_ref,
        "work_id": smoke_work_id,
        "attempt_number": int(smoke["attempt_count"]),
        "evaluation_mode": "provisioning_smoke",
        "config_ref": config_ref,
        "result_status": "passed",
        "route_hash": smoke_result["route_hash"],
        "response_hash": sha256_json({"smoke-response": smoke_attempt_ref}),
        "status_class": "2xx",
        "content_type": "application/json",
        "byte_count": 32,
        "duration_ms": 5,
        "retry_after_seconds": 0,
        "reason_codes": ["bounded_json_data_response"],
        "receipt_hash": smoke_receipt,
        "business_artifact_hash": smoke_artifact,
        "result_doc": smoke_result,
    }
    if reject_current_builtin:
        disabled_doc = {
            **provision_doc,
            "provider_registry_entry": {
                **provision_doc["provider_registry_entry"],
                "active": False,
            },
        }
        disabled = {
            **eligible,
            "provision_ref": "source_add_provision:"
            + sha256_json({"disabled": submission_id})[7:23],
            "provision_status": "disabled",
            "provision_doc": disabled_doc,
        }
        current = _scalar(
            cursor,
            "SELECT to_jsonb(s) FROM public.research_lab_source_add_submission_current s WHERE submission_id=%s",
            (submission_id,),
        )
        intent_before = _scalar(
            cursor,
            "SELECT to_jsonb(i) FROM public.research_lab_source_add_reward_intents i WHERE submission_id=%s AND leg=1",
            (submission_id,),
        )
        reward_work_before = _scalar(
            cursor,
            "SELECT to_jsonb(w) FROM public.research_lab_source_add_work_items w WHERE submission_id=%s AND work_kind='leg1_reward'",
            (submission_id,),
        )
        rewards_before = _scalar(
            cursor,
            "SELECT COALESCE(jsonb_agg(to_jsonb(r) ORDER BY r.reward_ref),'[]'::JSONB) FROM public.research_lab_source_add_reward_obligations r WHERE adapter_id=%s",
            (adapter_id,),
        )
        reward_events_before = _scalar(
            cursor,
            """
            SELECT COALESCE(jsonb_agg(to_jsonb(e) ORDER BY e.reward_ref,e.seq),'[]'::JSONB)
            FROM public.research_lab_source_add_reward_events e
            JOIN public.research_lab_source_add_reward_obligations r
              ON r.reward_ref=e.reward_ref
            WHERE r.adapter_id=%s
            """,
            (adapter_id,),
        )
        rejection_args = (
            smoke_work_id,
            smoke["lease_token"],
            submission_id,
            _json(current["submission_doc"]),
            current["precheck_status"],
            _json(current["precheck_doc"]),
            _json(catalog),
            _json(disabled),
            _json(smoke_attempt),
        )
        rejection_sql = """
            SELECT public.research_lab_source_add_reject_current_builtin_v3(
                %s,%s::UUID,%s,%s::JSONB,%s,%s::JSONB,%s::JSONB,
                %s::JSONB,%s::JSONB
            )
        """
        assert _scalar(cursor, rejection_sql, rejection_args) == {
            "status": "not_eligible"
        }
        assert _scalar(cursor, rejection_sql, rejection_args) == {
            "status": "not_eligible"
        }
        assert _scalar(
            cursor,
            "SELECT to_jsonb(i) FROM public.research_lab_source_add_reward_intents i WHERE submission_id=%s AND leg=1",
            (submission_id,),
        ) == intent_before
        assert _scalar(
            cursor,
            "SELECT to_jsonb(w) FROM public.research_lab_source_add_work_items w WHERE submission_id=%s AND work_kind='leg1_reward'",
            (submission_id,),
        ) == reward_work_before
        assert _scalar(
            cursor,
            "SELECT COALESCE(jsonb_agg(to_jsonb(r) ORDER BY r.reward_ref),'[]'::JSONB) FROM public.research_lab_source_add_reward_obligations r WHERE adapter_id=%s",
            (adapter_id,),
        ) == rewards_before
        assert _scalar(
            cursor,
            """
            SELECT COALESCE(jsonb_agg(to_jsonb(e) ORDER BY e.reward_ref,e.seq),'[]'::JSONB)
            FROM public.research_lab_source_add_reward_events e
            JOIN public.research_lab_source_add_reward_obligations r
              ON r.reward_ref=e.reward_ref
            WHERE r.adapter_id=%s
            """,
            (adapter_id,),
        ) == reward_events_before
        assert _scalar(
            cursor,
            "SELECT provision_status FROM public.research_lab_source_add_provisioning_current WHERE submission_id=%s",
            (submission_id,),
        ) == "disabled"
        assert _scalar(
            cursor,
            "SELECT count(*) FROM public.research_lab_source_add_provisioning_events WHERE submission_id=%s AND provision_status='disabled'",
            (submission_id,),
        ) == 1
        assert _scalar(
            cursor,
            "SELECT stage FROM public.research_lab_source_add_submission_current WHERE submission_id=%s",
            (submission_id,),
        ) == "functional_probe_failed"
        cursor.execute(
            """
            UPDATE public.research_lab_source_add_work_items
            SET available_at=NOW()
            WHERE work_kind='leg1_reward'
              AND work_status='queued'
            """
        )
        return rejection_sql, rejection_args

    finalized = _scalar(
        cursor,
        """
        SELECT public.research_lab_source_add_finalize_provision_smoke_v3(
            %s,%s::UUID,%s,%s::JSONB,%s::JSONB,%s::JSONB
        )
        """,
        (
            smoke_work_id,
            smoke["lease_token"],
            submission_id,
            _json(catalog),
            _json(eligible),
            _json(smoke_attempt),
        ),
    )
    assert finalized["status"] == "provisioned"
    assert _scalar(
        cursor,
        """
        SELECT count(*) FROM public.research_lab_source_add_reward_obligations
        WHERE adapter_id=%s AND leg=1
        """,
        (adapter_id,),
    ) == (0 if allow_unrewarded else 1)
    cursor.execute(
        """
        UPDATE public.research_lab_source_add_work_items
        SET available_at=NOW()
        WHERE work_kind='leg1_reward'
          AND work_status='queued'
        """
    )
    return None


def test_migration_175_full_provenance_leg1_contract(pre_migration_database):
    psycopg2, dsn = pre_migration_database
    migration_sql = (ROOT / "scripts" / MIGRATION).read_text(encoding="utf-8")
    connection = psycopg2.connect(**dsn)
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            _set_paused(cursor, False, "active gate")
            with pytest.raises(
                psycopg2.Error,
                match="SOURCE_ADD must be paused before provenance Leg 1 migration",
            ):
                cursor.execute(migration_sql)
            cursor.execute("ROLLBACK")
            _set_paused(cursor, True, "leased gate")
            _insert_work(cursor, suffix="1750000000000000", status="leased")
            with pytest.raises(
                psycopg2.Error,
                match="SOURCE_ADD work is leased during provenance Leg 1 migration",
            ):
                cursor.execute(migration_sql)
            cursor.execute("ROLLBACK")
            cursor.execute(
                "DELETE FROM public.research_lab_source_add_work_items WHERE work_id=%s",
                ("source_add_work:1750000000000000",),
            )

            _set_paused(cursor, False, "seed")
            _seed_boot_identity(cursor)
            paid_case = _create_seed_leg1_reward(
                cursor,
                case=_seed_leased_smoke_case(
                    cursor,
                    seed=0x1750000000010000,
                    base_url="https://already-paid.migration-175.test/v1",
                ),
                seed=0x1750000000010000,
                alpha_percent=1.0,
                economics_rpc_version=2,
            )
            _append_attested_provenance(cursor, paid_case, 1)
            exact = _seed_case(cursor, 0x1750000000020000)
            finalized_builtin = _seed_case(
                cursor,
                0x1750000000025000,
                historical_routing_reason="operator_probe_configuration_required",
            )
            historical = _seed_case(
                cursor,
                0x1750000000030000,
                historical_routing_reason="operator_credential_required",
            )
            unattested = _seed_case(
                cursor,
                0x1750000000040000,
                attested=False,
            )
            failed = _seed_case(
                cursor,
                0x1750000000050000,
                status="rejected_precheck",
            )

            _set_paused(cursor, True, "apply")
            cursor.execute(migration_sql)
            cursor.execute(migration_sql)
            contract = _scalar(
                cursor,
                "SELECT public.research_lab_source_add_post_accept_leg1_contract_v3()",
            )
            assert contract["schema_version"] == (
                "leadpoet.source_add_post_accept_leg1_contract.v3"
            )
            assert contract["daily_cap"] == 50
            assert contract["leg1_alpha_percent"] == 0.2
            assert contract["leg1_reward_epochs"] == 20
            assert contract["function_authority_sha256"] == (
                SOURCE_ADD_PROVENANCE_LEG1_FUNCTION_AUTHORITY_SHA256
            )
            assert contract["trigger_authority_sha256"] == (
                SOURCE_ADD_PROVENANCE_LEG1_TRIGGER_AUTHORITY_SHA256
            )
            assert contract["view_authority_sha256"] == (
                SOURCE_ADD_PROVENANCE_LEG1_VIEW_AUTHORITY_SHA256
            )
            assert all(contract["functions"].values())
            assert all(contract["triggers"].values())
            assert all(contract["columns"].values())
            assert all(contract["permissions"].values())

            for case in (exact, finalized_builtin, historical):
                intent_id = source_add_reward_intent_id(
                    case["record"]["submission_id"],
                    case["record"]["adapter_id"],
                )
                reward_work_id = source_add_work_id(
                    case["record"]["submission_id"],
                    "leg1_reward",
                    intent_id,
                )
                assert _scalar(
                    cursor,
                    "SELECT count(*) FROM public.research_lab_source_add_reward_intents WHERE intent_id=%s AND approval_kind='provenance_precheck_passed'",
                    (intent_id,),
                ) == 1
                assert _scalar(
                    cursor,
                    "SELECT count(*) FROM public.research_lab_source_add_work_items WHERE work_id=%s AND work_status='queued'",
                    (reward_work_id,),
                ) == 1
            rebuilt = rebuild_attested_provenance_result_v2(
                submission_id=historical["record"]["submission_id"],
                precheck_status="provenance_precheck_passed",
                precheck_doc=historical["precheck_doc"],
                submission_doc=historical["document"],
            )
            assert sha256_json(rebuilt) == historical["artifact_hash"]
            for case in (unattested, failed):
                assert _scalar(
                    cursor,
                    "SELECT count(*) FROM public.research_lab_source_add_reward_intents WHERE submission_id=%s",
                    (case["record"]["submission_id"],),
                ) == 0
            assert _scalar(
                cursor,
                "SELECT count(*) FROM public.research_lab_source_add_reward_intents WHERE submission_id=%s AND approval_kind='provenance_precheck_passed'",
                (paid_case["submission_id"],),
            ) == 0
            reconciled = _scalar(
                cursor,
                "SELECT public.research_lab_source_add_reconcile_provenance_leg1_v1()",
            )
            assert reconciled["eligible_count"] == 3

            _set_paused(cursor, False, "automatic")
            automatic = _seed_case(cursor, 0x1750000000060000)
            automatic_intent = source_add_reward_intent_id(
                automatic["record"]["submission_id"],
                automatic["record"]["adapter_id"],
            )
            assert _scalar(
                cursor,
                "SELECT count(*) FROM public.research_lab_source_add_reward_intents WHERE intent_id=%s",
                (automatic_intent,),
            ) == 1
            queued_builtin = _seed_case(cursor, 0x1750000000065000)
            queued_rejection = _provision_after_leg1(
                cursor, queued_builtin, reject_current_builtin=True
            )
            assert queued_rejection is not None
            assert _scalar(
                cursor,
                "SELECT count(*) FROM public.research_lab_source_add_reward_obligations WHERE adapter_id=%s AND leg=1",
                (queued_builtin["record"]["adapter_id"],),
            ) == 0

            accepted_before_reward = _seed_case(cursor, 0x1750000000067500)
            _provision_after_leg1(
                cursor,
                accepted_before_reward,
                allow_unrewarded=True,
            )
            assert _scalar(
                cursor,
                "SELECT stage FROM public.research_lab_source_add_submission_current WHERE submission_id=%s",
                (accepted_before_reward["record"]["submission_id"],),
            ) == "accepted"
            cursor.execute(
                """
                UPDATE public.research_lab_source_add_work_items
                SET available_at=NOW() + INTERVAL '1 hour'
                WHERE work_kind='leg1_reward'
                  AND work_status='queued'
                  AND submission_id<>%s
                """,
                (accepted_before_reward["record"]["submission_id"],),
            )
            cursor.execute(
                """
                UPDATE public.research_lab_source_add_work_items
                SET available_at=NOW()
                WHERE work_kind='leg1_reward'
                  AND work_status='queued'
                  AND submission_id=%s
                """,
                (accepted_before_reward["record"]["submission_id"],),
            )
            accepted_reward_work = _claim_reward(cursor)
            assert accepted_reward_work["submission_id"] == (
                accepted_before_reward["record"]["submission_id"]
            )
            assert _finalize_reward(
                cursor,
                work=accepted_reward_work,
                case=accepted_before_reward,
                caller_cap=999,
            )["status"] == "created"
            assert _scalar(
                cursor,
                "SELECT stage FROM public.research_lab_source_add_submission_current WHERE submission_id=%s",
                (accepted_before_reward["record"]["submission_id"],),
            ) == "accepted"
            cursor.execute(
                """
                INSERT INTO public.research_lab_source_add_submissions (
                    submission_id,adapter_id,miner_hotkey,stage,seq,
                    submission_doc,precheck_status,precheck_doc,
                    source_identity_hash,source_identity_version
                )
                SELECT
                    current.submission_id,current.adapter_id,
                    current.miner_hotkey,'leg1_created',current.seq + 1,
                    current.submission_doc ||
                        jsonb_build_object('stage','leg1_created'),
                    current.precheck_status,current.precheck_doc,
                    current.source_identity_hash,current.source_identity_version
                FROM public.research_lab_source_add_submission_current current
                WHERE current.submission_id=%s
                """,
                (accepted_before_reward["record"]["submission_id"],),
            )
            assert _scalar(
                cursor,
                "SELECT stage FROM public.research_lab_source_add_submission_current WHERE submission_id=%s",
                (accepted_before_reward["record"]["submission_id"],),
            ) == "leg1_created"
            frozen_counts = _scalar(
                cursor,
                """
                SELECT jsonb_build_object(
                    'configs',(SELECT COUNT(*) FROM public.research_lab_source_add_probe_config_events WHERE submission_id=%s),
                    'works',(SELECT COUNT(*) FROM public.research_lab_source_add_work_items WHERE submission_id=%s),
                    'history',(SELECT COUNT(*) FROM public.research_lab_source_add_submissions WHERE submission_id=%s)
                )
                """,
                (
                    accepted_before_reward["record"]["submission_id"],
                    accepted_before_reward["record"]["submission_id"],
                    accepted_before_reward["record"]["submission_id"],
                ),
            )
            assert _scalar(
                cursor,
                """
                SELECT public.research_lab_source_add_configure_probe_v3(
                    %s,%s,'{}'::JSONB,'{}'::JSONB,%s,%s,%s
                )
                """,
                (
                    accepted_before_reward["record"]["submission_id"],
                    "source_add_probe_config:1750000000067500",
                    "operator:migration-175-freeze-test",
                    "source_add_work:1750000000067500",
                    sha256_json({"accepted-freeze": True}),
                ),
            ) == {"status": "final_approval_frozen"}
            assert _scalar(
                cursor,
                """
                SELECT jsonb_build_object(
                    'configs',(SELECT COUNT(*) FROM public.research_lab_source_add_probe_config_events WHERE submission_id=%s),
                    'works',(SELECT COUNT(*) FROM public.research_lab_source_add_work_items WHERE submission_id=%s),
                    'history',(SELECT COUNT(*) FROM public.research_lab_source_add_submissions WHERE submission_id=%s)
                )
                """,
                (
                    accepted_before_reward["record"]["submission_id"],
                    accepted_before_reward["record"]["submission_id"],
                    accepted_before_reward["record"]["submission_id"],
                ),
            ) == frozen_counts
            cursor.execute(
                """
                UPDATE public.research_lab_source_add_work_items
                SET available_at=NOW()
                WHERE work_kind='leg1_reward'
                  AND work_status='queued'
                """
            )

            first = _claim_reward(cursor)
            assert first["submission_id"] == exact["record"]["submission_id"]
            second = _claim_reward(cursor)
            assert second["submission_id"] == finalized_builtin["record"]["submission_id"]
            assert _scalar(
                cursor,
                "SELECT public.research_lab_source_add_reserve_leg1_slot_v4(%s,%s,%s::UUID,999,300)",
                (second["job_doc"]["intent_id"], second["work_id"], second["lease_token"]),
            )["status"] == "fifo_wait"
            created = _finalize_reward(
                cursor,
                work=first,
                case=exact,
                caller_cap=1,
            )
            assert created["status"] == "created"
            reward = _scalar(
                cursor,
                "SELECT to_jsonb(r) FROM public.research_lab_source_add_reward_obligations r WHERE reward_ref=%s",
                (created["reward_ref"],),
            )
            assert reward["catalog_id"] is None
            assert float(reward["alpha_percent"]) == pytest.approx(0.2)
            assert reward["reward_epochs"] == 20
            assert reward["trigger_evidence_doc"] == {
                "provenance_precheck_passed": True,
                "submission_id": exact["record"]["submission_id"],
                "precheck_status": "provenance_precheck_passed",
                "provenance_receipt_hash": exact["receipt_hash"],
                "provenance_artifact_hash": exact["artifact_hash"],
                "provenance_result_hash": exact["artifact_hash"],
            }

            _set_paused(cursor, True, "post reward reapply")
            cursor.execute(migration_sql)
            _set_paused(cursor, False, "post reward provision")
            _provision_after_leg1(cursor, exact)
            cursor.execute(
                """
                UPDATE public.research_lab_source_add_reward_intents
                SET available_at=NOW() - INTERVAL '1 day'
                WHERE submission_id=%s AND leg=1
                """,
                (finalized_builtin["record"]["submission_id"],),
            )
            cursor.execute(
                """
                UPDATE public.research_lab_source_add_work_items
                SET available_at=NOW() - INTERVAL '1 day'
                WHERE submission_id=%s AND work_kind='leg1_reward'
                """,
                (finalized_builtin["record"]["submission_id"],),
            )
            finalized_builtin_work = _claim_reward(cursor)
            assert finalized_builtin_work["submission_id"] == (
                finalized_builtin["record"]["submission_id"]
            )
            assert _finalize_reward(
                cursor,
                work=finalized_builtin_work,
                case=finalized_builtin,
                caller_cap=999,
            )["status"] == "created"
            _provision_after_leg1(
                cursor, finalized_builtin, reject_current_builtin=True
            )

            cases_by_submission = {
                historical["record"]["submission_id"]: historical,
                automatic["record"]["submission_id"]: automatic,
                queued_builtin["record"]["submission_id"]: queued_builtin,
            }
            for index in range(52):
                case = _seed_case(cursor, 0x1750000000100000 + index)
                cases_by_submission[case["record"]["submission_id"]] = case
            while _scalar(
                cursor,
                """
                SELECT count(*) FROM public.research_lab_source_add_reward_events
                WHERE reason IN (
                    'leg1_functional_probe_passed',
                    'leg1_provenance_precheck_passed'
                ) AND created_at >= date_trunc('day',NOW() AT TIME ZONE 'UTC') AT TIME ZONE 'UTC'
                """,
            ) < 50:
                work = _claim_reward(cursor)
                result = _finalize_reward(
                    cursor,
                    work=work,
                    case=cases_by_submission[work["submission_id"]],
                    caller_cap=1,
                )
                assert result["status"] == "created"
            assert _scalar(
                cursor,
                "SELECT count(*) FROM public.research_lab_source_add_reward_obligations WHERE adapter_id=%s AND leg=1",
                (queued_builtin["record"]["adapter_id"],),
            ) == 1
            queued_rejection_sql, queued_rejection_args = queued_rejection
            assert _scalar(
                cursor, queued_rejection_sql, queued_rejection_args
            ) == {"status": "not_eligible"}
            overflow = _claim_reward(cursor)
            assert _finalize_reward(
                cursor,
                work=overflow,
                case=cases_by_submission[overflow["submission_id"]],
                caller_cap=999,
            )["status"] == "daily_cap_fifo"
    finally:
        connection.close()
