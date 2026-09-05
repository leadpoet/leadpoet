from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from urllib.error import HTTPError

import pytest

from gateway.tee import restart_preflight_v2 as preflight
from gateway.tee import supabase_schema_preflight_v2 as schema_preflight
from gateway.tee.provider_broker_v2 import credential_reference_hash
from gateway.tee.artifact_persistence_v2 import ARTIFACT_POLICY_SCHEMA_VERSION
from gateway.tee.release_manifest_v2 import (
    BUILD_EVIDENCE_SCHEMA_VERSION,
    build_release_manifest,
)
from gateway.tee.topology import ROLE_SPECS, manifest_document, topology_hash
from gateway.utils.tee_kms_provision_v2 import PROVIDER_ENVELOPE_SCHEMA_VERSION
from leadpoet_canonical.attested_v2 import sha256_bytes, sha256_json


COMMIT = "1" * 40
POLICY = {
    "schema_version": ARTIFACT_POLICY_SCHEMA_VERSION,
    "bucket_host": "leadpoet-v2.s3.us-east-1.amazonaws.com",
    "key_prefix": "/attested-v2/",
    "minimum_retention_days": 365,
}
FILE_TO_SLOT = {
    "artifact_master_key.json": "artifact_master_key",
    "openrouter.json": "openrouter",
    "exa.json": "exa",
    "scrapingdog.json": "scrapingdog",
    "deepline.json": "deepline",
    "supabase_service_role.json": "supabase_service_role",
    "truelist.json": "truelist",
}


def _hash(character: str) -> str:
    return "sha256:" + character * 64


def _release(commit: str = COMMIT):
    evidence = []
    for role_index, (role, spec) in enumerate(sorted(ROLE_SPECS.items()), start=1):
        character = str(role_index)
        deterministic = {
            "commit_sha": commit,
            "pcr0": character * 96,
            "normalized_image_hash": _hash(character),
            "eif_hash": _hash(character),
            "source_manifest_hash": _hash("a"),
            "build_identity_hash": _hash(character),
            "execution_manifest_hash": _hash(character),
            "dependency_lock_hash": _hash("b"),
            "dockerfile_hash": _hash("c"),
            "topology_hash": topology_hash(),
        }
        for domain in ("gateway", "validator"):
            for ordinal in (1, 2, 3):
                evidence.append(
                    {
                        "schema_version": BUILD_EVIDENCE_SCHEMA_VERSION,
                        "builder_domain": domain,
                        "builder_id": "%s-parent" % domain,
                        "build_ordinal": ordinal,
                        "physical_role": role,
                        "service_role": spec["service_role"],
                        **deterministic,
                    }
                )
    return build_release_manifest(
        evidence, acceptance_signer_pubkey_hash="sha256:" + "f" * 64
    )


def _credential_envelopes(tmp_path: Path) -> list[Path]:
    paths = []
    for filename, slot in FILE_TO_SLOT.items():
        ciphertext = ("kms-ciphertext:" + slot).encode("ascii")
        context = {"service": "leadpoet-v2", "credential_slot": slot}
        document = {
            "schema_version": PROVIDER_ENVELOPE_SCHEMA_VERSION,
            "credential_slot": slot,
            "credential_ref_hash": sha256_json({"credential_slot": slot}),
            "ciphertext_blob_b64": base64.b64encode(ciphertext).decode("ascii"),
            "ciphertext_blob_hash": sha256_bytes(ciphertext),
            "kms_key_id_hash": sha256_json({"kms_key": "test"}),
            "encryption_context": context,
            "encryption_context_hash": sha256_json(context),
        }
        path = tmp_path / filename
        path.write_text(json.dumps(document), encoding="utf-8")
        paths.append(path)
    return paths


def _verify(tmp_path: Path, monkeypatch, **overrides):
    del monkeypatch
    credential_envelopes = overrides.pop(
        "credential_envelope_paths",
        None,
    ) or _credential_envelopes(tmp_path)
    parent_environment = {}
    if "parent_environment" in overrides:
        parent_environment = dict(overrides.pop("parent_environment"))
    values = {
        "deploy_commit": COMMIT,
        "release_manifest": _release(),
        "topology_manifest": manifest_document(),
        "artifact_policy": POLICY,
        "credential_envelope_paths": credential_envelopes,
        "topology_mode": "full",
        "instance_type": "r7i.4xlarge",
        "parent_vcpus": 16,
        "parent_memory_mib": 125000,
        "parent_environment": parent_environment,
    }
    values.update(overrides)
    return preflight.verify_gateway_restart_preflight_v2(**values)


def test_full_restart_preflight_accepts_complete_local_release(
    tmp_path: Path,
    monkeypatch,
) -> None:
    result = _verify(tmp_path, monkeypatch)
    assert result["status"] == "ready"
    assert result["deploy_commit"] == COMMIT
    assert result["instance_type"] == "r7i.4xlarge"
    assert result["role_count"] == len(ROLE_SPECS)
    assert result["boot_credential_slot_count"] == 7
    assert result["parent_plaintext_provider_slot_count"] == 0
    assert "worker_proxy_profile_count" not in result
    assert "acceptance_corpus_manifest_hash" not in result
    assert "official_baseline_custody" not in result


def test_capacity_detection_counts_cpus_reserved_by_nitro(monkeypatch) -> None:
    monkeypatch.setattr(os, "sysconf", lambda name: 16)
    monkeypatch.setattr(os, "cpu_count", lambda: 14)

    assert preflight._configured_processor_count() == 16


def test_full_restart_preflight_rejects_current_undersized_gateway(
    tmp_path: Path,
    monkeypatch,
) -> None:
    with pytest.raises(
        preflight.GatewayRestartPreflightV2Error,
        match="requires r7i.4xlarge",
    ):
        _verify(tmp_path, monkeypatch, instance_type="r7i.2xlarge", parent_vcpus=8)


def test_restart_preflight_rejects_release_for_another_commit(
    tmp_path: Path,
    monkeypatch,
) -> None:
    with pytest.raises(
        preflight.GatewayRestartPreflightV2Error,
        match="another commit",
    ):
        _verify(tmp_path, monkeypatch, deploy_commit="2" * 40)


def test_restart_preflight_rejects_incomplete_or_misnamed_boot_envelopes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    envelopes = _credential_envelopes(tmp_path)
    with pytest.raises(
        preflight.GatewayRestartPreflightV2Error,
        match="incomplete",
    ):
        _verify(
            tmp_path,
            monkeypatch,
            credential_envelope_paths=envelopes[:-1],
        )

    renamed = tmp_path / "unexpected.json"
    envelopes[0].rename(renamed)
    with pytest.raises(
        preflight.GatewayRestartPreflightV2Error,
        match="filenames",
    ):
        _verify(
            tmp_path,
            monkeypatch,
            credential_envelope_paths=[renamed, *envelopes[1:]],
        )


def test_component_preflight_keeps_release_and_secret_gates_without_resize(
    tmp_path: Path,
    monkeypatch,
) -> None:
    result = _verify(
        tmp_path,
        monkeypatch,
        topology_mode="component",
        instance_type="r7i.2xlarge",
        parent_vcpus=8,
        parent_memory_mib=64000,
    )
    assert result["status"] == "ready"
    assert result["role_count"] == 1


def test_restart_preflight_rejects_protected_provider_key_in_parent_env(
    tmp_path: Path,
    monkeypatch,
) -> None:
    secret = "protected-openrouter-value"
    envelopes = _credential_envelopes(tmp_path)
    openrouter_path = next(
        path for path in envelopes if path.name == "openrouter.json"
    )
    document = json.loads(openrouter_path.read_text(encoding="utf-8"))
    document["credential_ref_hash"] = credential_reference_hash(secret)
    openrouter_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(
        preflight.GatewayRestartPreflightV2Error,
        match="protected openrouter credential",
    ):
        _verify(
            tmp_path,
            monkeypatch,
            credential_envelope_paths=envelopes,
            parent_environment={"UNRELATED_ALIAS": secret},
        )


def test_parent_env_parser_does_not_execute_shell(tmp_path: Path) -> None:
    marker = tmp_path / "must-not-exist"
    env_file = tmp_path / "parent.env"
    env_file.write_text(
        "export NORMAL='quoted value'\n"
        "export PAYLOAD='$(touch %s)'\n" % marker,
        encoding="utf-8",
    )
    assert preflight.load_parent_environment(env_file) == {
        "NORMAL": "quoted value",
        "PAYLOAD": "$(touch %s)" % marker,
    }
    assert not marker.exists()


class _SchemaResponse:
    def __init__(self, status: int = 200, body: bytes = b"[") -> None:
        self.status = status
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        return None

    def getcode(self) -> int:
        return self.status

    def read(self, _size: int = -1) -> bytes:
        return self.body if _size < 0 else self.body[:_size]


def _chain_realized_activation_response() -> bytes:
    return json.dumps(
        [
            {
                "netuid": 71,
                "schema_version": (
                    "leadpoet.research_lab_chain_realized_settlement_activation.v1"
                ),
                "first_epoch_id": 24196,
                "source_bundle_hash": "sha256:" + "a" * 64,
                "source_bundle_epoch_id": 24196,
                "source_finalized_block": 8715224,
            }
        ]
    ).encode()


def _compact_weight_settlement_contract_response() -> bytes:
    return json.dumps(
        {
            "schema_version": (
                "leadpoet.research_lab_compact_weight_settlement_contract.v1"
            ),
            "max_authority_bytes": 8_388_608,
            "size_constraint_valid": True,
            "append_only_trigger_enabled": True,
            "identity_unique_constraint_enabled": True,
            "row_level_security_enabled": True,
            "finalized_stage_supported": True,
        }
    ).encode()


def _source_add_provider_origin_contract_response(**overrides) -> bytes:
    contract = {
        "schema_version": "leadpoet.source_add_provider_origin_contract.v1",
        "identity_version": "v1",
        "identity_scope": "normalized_exact_host",
        "admission_rpc": "research_lab_source_add_admit_v2",
        "recheck_rpc": "research_lab_source_add_requeue_provenance_v2",
        "owner_count": 2,
        "reserved_count": 2,
        "coverage_complete": True,
        "collision_free": True,
        "submission_trigger_enabled": True,
        "catalog_trigger_enabled": True,
        "provision_trigger_enabled": True,
        "terminal_release_trigger_enabled": True,
        "append_only_trigger_enabled": True,
        "row_level_security_enabled": True,
        "service_role_policy_enabled": True,
    }
    contract.update(overrides)
    return json.dumps(contract).encode()


def _source_add_duplicate_privacy_contract_response(**overrides) -> bytes:
    contract = {
        "schema_version": "leadpoet.source_add_duplicate_privacy_contract.v1",
        "admission_rpc": "research_lab_source_add_admit_v3",
        "admission_signature": (
            "jsonb,text,text,text,text,text,integer,integer,integer,integer"
        ),
        "compatibility_rpc": "research_lab_source_add_admit_v2",
        "compatibility_signature": (
            "jsonb,text,text,text,text,text,integer,integer,integer"
        ),
        "compatibility_cooldown_seconds": 20,
        "cooldown_parameter_min_seconds": 1,
        "cooldown_parameter_max_seconds": 3600,
        "cooldown_clock": "clock_timestamp_after_advisory_locks",
        "cooldown_source": "durable_miner_provenance_work",
        "duplicate_precedes_cooldown": True,
        "lock_order": [
            "provider_origin_or_identity",
            "hotkey",
            "submission_or_work",
        ],
        "function_authority_sha256": (
            schema_preflight.SOURCE_ADD_DUPLICATE_PRIVACY_FUNCTION_AUTHORITY_SHA256
        ),
        "functions": {
            "admit_v1": True,
            "admit_v2_compatibility": True,
            "admit_v3": True,
            "provider_origin_hash_v1": True,
            "provider_origin_host_v1": True,
        },
        "permissions": {
            "service_role_exists": True,
            "v3_service_role_callable": True,
            "v2_service_role_callable": True,
            "contract_service_role_callable": True,
            "anon_callable": False,
            "authenticated_callable": False,
        },
    }
    contract.update(overrides)
    return json.dumps(contract).encode()


def _source_add_claim_control_contract_response(**overrides) -> bytes:
    contract = {
        "schema_version": "leadpoet.source_add_claim_control_contract.v2",
        "control_lock": "source-add-control",
        "pause_rpc": "research_lab_source_add_set_paused",
        "pause_signature": "boolean,text,text",
        "claim_rpc": "research_lab_source_add_claim_work",
        "claim_signature": "text,integer",
        "acquire_guard_rpc": (
            "research_lab_source_add_acquire_restart_guard_v2"
        ),
        "acquire_guard_signature": "text,text,bigint,integer,text",
        "guard_state_rpc": (
            "research_lab_source_add_restart_guard_state_v2"
        ),
        "guard_state_signature": "",
        "release_guard_rpc": (
            "research_lab_source_add_release_restart_guard_v2"
        ),
        "release_guard_signature": "text,text,bigint,text",
        "guard_state_result_fields": [
            "schema_version",
            "paused",
            "guard_active",
            "guard_commitment",
            "owner_commitment",
            "guard_generation",
            "owner_generation_commitment",
            "guard_expires_at",
            "restore_paused",
        ],
        "acquire_guard_result_fields": [
            "schema_version",
            "paused",
            "guard_active",
            "guard_commitment",
            "owner_commitment",
            "guard_generation",
            "owner_generation_commitment",
            "guard_expires_at",
            "restore_paused",
        ],
        "release_guard_result_fields": [
            "schema_version",
            "released",
            "paused",
            "guard_active",
            "guard_generation",
            "owner_generation_commitment",
            "restored_pre_restart_state",
        ],
        "restart_quiescence_rpc": (
            "research_lab_source_add_restart_quiescence_v1"
        ),
        "restart_quiescence_signature": "text,text,bigint",
        "restore_state_column": "restart_guard_restore_paused",
        "acquire_captures_pre_restart_paused": True,
        "renewal_preserves_restore_state": True,
        "expired_takeover_preserves_restore_state": True,
        "operator_pause_wins": True,
        "release_restores_pre_restart_state": True,
        "failed_restart_keeps_paused": True,
        "rollback_v1_contract_schema_version": (
            "leadpoet.source_add_claim_control_contract.v1"
        ),
        "rollback_v1_contract_sha256": (
            schema_preflight.SOURCE_ADD_CLAIM_CONTROL_ROLLBACK_V1_CONTRACT_SHA256
        ),
        "migration_requires_paused": True,
        "migration_requires_zero_leased": True,
        "migration_requires_guard_clear": True,
        "function_authority_sha256": (
            schema_preflight.SOURCE_ADD_CLAIM_CONTROL_V2_FUNCTION_AUTHORITY_SHA256
        ),
        "functions": {
            "admission_guard": True,
            "acquire_restart_guard_v1": True,
            "acquire_restart_guard_v2": True,
            "claim_work": True,
            "pause": True,
            "release_restart_guard_v1": True,
            "release_restart_guard_v2": True,
            "restart_guard_state_v1": True,
            "restart_guard_state_v2": True,
            "restart_quiescence_v1": True,
            "restore_trigger_v2": True,
        },
        "permissions": {
            "service_role_exists": True,
            "service_role_callable": True,
            "anon_callable": False,
            "authenticated_callable": False,
        },
    }
    contract.update(overrides)
    return json.dumps(contract).encode()


def _source_add_post_accept_leg1_contract_response(**overrides) -> bytes:
    contract = {
        "schema_version": "leadpoet.source_add_post_accept_leg1_contract.v4",
        "required_migration": (
            "scripts/176-research-lab-source-add-provenance-origin-repair.sql"
        ),
        "daily_cap": 50,
        "leg1_alpha_percent": 0.2,
        "leg1_reward_epochs": 20,
        "approval_boundary": "provenance_precheck_passed",
        "backfill_policy": (
            "earliest_exact_attested_provenance_per_provider_origin"
        ),
        "provider_origin_scope": "normalized_exact_host",
        "provider_origin_winner_order": [
            "provenance_created_at",
            "submission_id",
        ],
        "cancelled_intents_are_authority": False,
        "public_trigger_fields": [
            "precheck_status",
            "provenance_artifact_hash",
            "provenance_precheck_passed",
            "provenance_receipt_hash",
            "provenance_result_hash",
            "submission_id",
        ],
        "authority_view": (
            "research_lab_source_add_provenance_leg1_authority_v1"
        ),
        "function_authority_sha256": (
            schema_preflight.SOURCE_ADD_PROVENANCE_LEG1_FUNCTION_AUTHORITY_SHA256
        ),
        "trigger_authority_sha256": (
            schema_preflight.SOURCE_ADD_PROVENANCE_LEG1_TRIGGER_AUTHORITY_SHA256
        ),
        "view_authority_sha256": (
            schema_preflight.SOURCE_ADD_PROVENANCE_ORIGIN_VIEW_AUTHORITY_SHA256
        ),
        "repair_function_authority_sha256": (
            schema_preflight.SOURCE_ADD_PROVENANCE_ORIGIN_REPAIR_FUNCTION_AUTHORITY_SHA256
        ),
        "functions": {
            "configure_probe_v3": True,
            "enqueue_leg1_after_provenance_v1": True,
            "enqueue_provision_smoke_v2": True,
            "finalize_leg1_v4": True,
            "finalize_provision_smoke_v3": True,
            "finalize_provision_v3": True,
            "reject_current_builtin_v3": True,
            "reconcile_provenance_leg1_v1": True,
            "reserve_leg1_slot_v4": True,
        },
        "triggers": {
            "automatic_enqueue": True,
            "eligible_v2": True,
            "eligible_v3": True,
            "leg1_initial_event_v3": True,
            "leg1_obligation_v3": True,
            "leg1_slot_v3": True,
            "leg1_work_v3": True,
        },
        "columns": {
            "intent_approval_kind": True,
            "intent_provenance_artifact_hash": True,
            "intent_provenance_receipt_hash": True,
            "slot_approval_kind": True,
        },
        "permissions": {
            "service_role_exists": True,
            "candidate_callable": True,
            "internal_not_callable": True,
            "rollback_v2_callable": True,
        },
    }
    contract.update(overrides)
    return json.dumps(contract).encode()


def _source_add_miner_status_contract_response(**overrides) -> bytes:
    contract = {
        "schema_version": "leadpoet.source_add_miner_status_contract.v1",
        "view_name": "research_lab_source_add_miner_status_v1",
        "page_rpc": "research_lab_source_add_miner_status_page_v1",
        "page_signature": "text,text,integer",
        "view_columns": [
            "schema_version",
            "submission_id",
            "miner_hotkey",
            "source_name",
            "submitted_at",
            "updated_at",
            "decision_status",
            "decision_reason_code",
            "decision_reason",
            "reward_status",
            "alpha_percent",
            "reward_epochs",
            "start_epoch",
            "end_epoch",
        ],
        "view_security_invoker": True,
        "view_security_barrier": True,
        "page_security_invoker": True,
        "page_stable": True,
        "view_authority_sha256": (
            schema_preflight.SOURCE_ADD_MINER_STATUS_VIEW_AUTHORITY_SHA256
        ),
        "page_authority_sha256": (
            schema_preflight.SOURCE_ADD_MINER_STATUS_PAGE_AUTHORITY_SHA256
        ),
        "contract_authority_sha256": (
            schema_preflight.SOURCE_ADD_MINER_STATUS_CONTRACT_AUTHORITY_SHA256
        ),
        "permissions": {
            "view_service_role_select": True,
            "view_anon_select": False,
            "view_authenticated_select": False,
            "view_public_select": False,
            "page_service_role_callable": True,
            "page_anon_callable": False,
            "page_authenticated_callable": False,
            "page_public_callable": False,
            "contract_service_role_callable": True,
            "contract_anon_callable": False,
            "contract_authenticated_callable": False,
        },
    }
    contract.update(overrides)
    return json.dumps(contract).encode()


def test_required_supabase_v2_schema_probes_tables_and_columns() -> None:
    requests = []

    def opener(request, *, timeout):
        requests.append((request, timeout))
        if request.full_url.endswith("/rest/v1/"):
            paths = {
                f"/rpc/{function_name}": {"post": {}}
                for _migration, function_name in schema_preflight.REQUIRED_SUPABASE_V2_RPCS
            }
            return _SchemaResponse(body=json.dumps({"paths": paths}).encode())
        if (
            "research_lab_chain_realized_settlement_activation_v1"
            in request.full_url
            and "limit=2" in request.full_url
        ):
            return _SchemaResponse(body=_chain_realized_activation_response())
        if request.full_url.endswith(
            "/rpc/research_lab_compact_weight_settlement_contract_v1"
        ):
            return _SchemaResponse(
                body=_compact_weight_settlement_contract_response()
            )
        if request.full_url.endswith(
            "/rpc/research_lab_source_add_provider_origin_contract_v1"
        ):
            return _SchemaResponse(
                body=_source_add_provider_origin_contract_response()
            )
        if request.full_url.endswith(
            "/rpc/research_lab_source_add_duplicate_privacy_contract_v1"
        ):
            return _SchemaResponse(
                body=_source_add_duplicate_privacy_contract_response()
            )
        if request.full_url.endswith(
            "/rpc/research_lab_source_add_post_accept_leg1_contract_v4"
        ):
            return _SchemaResponse(
                body=_source_add_post_accept_leg1_contract_response()
            )
        if request.full_url.endswith(
                "/rpc/research_lab_source_add_claim_control_contract_v2"
        ):
            return _SchemaResponse(
                body=_source_add_claim_control_contract_response()
            )
        if request.full_url.endswith(
            "/rpc/research_lab_source_add_miner_status_contract_v1"
        ):
            return _SchemaResponse(
                body=_source_add_miner_status_contract_response()
            )
        return _SchemaResponse()

    result = schema_preflight.verify_required_supabase_v2_schema(
        {
            "SUPABASE_URL": "https://project.supabase.co/",
            "SUPABASE_SERVICE_ROLE_KEY": "service-role-value",
        },
        opener=opener,
    )

    assert result["status"] == "ready"
    assert result["probe_count"] == len(
        schema_preflight.REQUIRED_SUPABASE_V2_SCHEMA
    ) + len(schema_preflight.REQUIRED_SUPABASE_V2_RPCS) + 6
    assert result["table_probe_count"] == len(
        schema_preflight.REQUIRED_SUPABASE_V2_SCHEMA
    )
    assert result["rpc_probe_count"] == len(
        schema_preflight.REQUIRED_SUPABASE_V2_RPCS
    )
    assert result["data_probe_count"] == 6
    assert result["schema_document_probe_count"] == 1
    assert result["chain_realized_settlement_activation_http_probe_count"] == 1
    assert result["chain_realized_settlement_activation_source"] == "postgrest"
    assert result["chain_realized_settlement_activation"] == {
        "netuid": 71,
        "first_epoch_id": 24196,
        "source_bundle_hash": "sha256:" + "a" * 64,
        "source_finalized_block": 8715224,
    }
    assert result["compact_weight_settlement_contract"] == json.loads(
        _compact_weight_settlement_contract_response()
    )
    assert result["source_add_provider_origin_contract"] == json.loads(
        _source_add_provider_origin_contract_response()
    )
    assert result["source_add_post_accept_leg1_contract"] == json.loads(
        _source_add_post_accept_leg1_contract_response()
    )
    assert result["source_add_claim_control_contract"] == json.loads(
        _source_add_claim_control_contract_response()
    )
    assert result["source_add_miner_status_contract"] == json.loads(
        _source_add_miner_status_contract_response()
    )
    assert result["source_add_leg1_release_policy"] == {
        "schema_version": "leadpoet.source_add_leg1_release_policy.v1",
        "leg1_alpha_percent": 0.2,
        "leg2_alpha_percent": 0.0,
        "reward_epochs": 20,
        "daily_cap": 50,
    }
    assert len(requests) == result["table_probe_count"] + 8
    assert all("/rest/v1/" in request.full_url for request, _timeout in requests)
    requested_urls = [request.full_url for request, _timeout in requests]
    assert any("/rest/v1/lab_arena_reward_basis_v1?" in url for url in requested_urls)
    assert any("/rest/v1/research_lab_source_add_miner_status_v1?" in url for url in requested_urls)
    assert all(
        marker not in url
        for url in requested_urls
        for marker in (
            "/rest/v1/lab_arena_rounds?",
            "/rest/v1/lab_arena_submissions?",
            "/rest/v1/lab_arena_runs?",
            "/rest/v1/lab_arena_ledger?",
            "/rpc/lab_arena_",
        )
    )
    table_requests = [
        request
        for request, _timeout in requests
        if not request.full_url.endswith("/rest/v1/")
    ]
    schema_requests = [
        request
        for request, _timeout in requests
        if request.full_url.endswith("/rest/v1/")
    ]
    activation_requests = [
        request
        for request in table_requests
        if "research_lab_chain_realized_settlement_activation_v1"
        in request.full_url
        and "limit=2" in request.full_url
    ]
    contract_requests = [
        request
        for request in table_requests
        if request.full_url.endswith(
            "/rpc/research_lab_compact_weight_settlement_contract_v1"
        )
    ]
    origin_contract_requests = [
        request
        for request in table_requests
        if request.full_url.endswith(
            "/rpc/research_lab_source_add_provider_origin_contract_v1"
        )
    ]
    privacy_contract_requests = [
        request
        for request in table_requests
        if request.full_url.endswith(
            "/rpc/research_lab_source_add_duplicate_privacy_contract_v1"
        )
    ]
    leg1_contract_requests = [
        request
        for request in table_requests
        if request.full_url.endswith(
            "/rpc/research_lab_source_add_post_accept_leg1_contract_v4"
        )
    ]
    claim_control_contract_requests = [
        request
        for request in table_requests
        if request.full_url.endswith(
                "/rpc/research_lab_source_add_claim_control_contract_v2"
        )
    ]
    miner_status_contract_requests = [
        request
        for request in table_requests
        if request.full_url.endswith(
            "/rpc/research_lab_source_add_miner_status_contract_v1"
        )
    ]
    schema_table_requests = [
        request
        for request in table_requests
        if request not in activation_requests
        and request not in contract_requests
        and request not in origin_contract_requests
        and request not in privacy_contract_requests
        and request not in leg1_contract_requests
        and request not in claim_control_contract_requests
        and request not in miner_status_contract_requests
    ]
    assert all(
        "limit=0" in request.full_url for request in schema_table_requests
    )
    assert len(activation_requests) == 1
    assert len(contract_requests) == 1
    assert len(origin_contract_requests) == 1
    assert len(privacy_contract_requests) == 1
    assert len(leg1_contract_requests) == 1
    assert len(claim_control_contract_requests) == 1
    assert len(miner_status_contract_requests) == 1
    assert len(schema_requests) == 1
    assert schema_requests[0].headers["Accept"] == "application/openapi+json"
    assert {
        "scripts/126-research-lab-chain-realized-settlement.sql",
        "scripts/127-research-lab-chain-unattributed-settlement.sql",
        "scripts/128-research-lab-chain-settlement-transport-purposes.sql",
        "scripts/129-research-lab-attested-local-transport.sql",
        "scripts/132-research-lab-champion-lifetime-credit.sql",
        "scripts/133-research-lab-provider-outcome-contention-status.sql",
        "scripts/134-research-lab-provider-outcome-head-contention.sql",
        "scripts/144-research-lab-provider-persistence-batches.sql",
        "scripts/145-research-lab-source-add-admission-control.sql",
        "scripts/149-research-lab-compact-weight-settlement-authority.sql",
        "scripts/155-research-lab-ancestry-disclosure-root-fast-path.sql",
        "scripts/156-production-parity-readonly-role.sql",
        "scripts/169-research-lab-source-add-post-accept-leg1.sql",
        "scripts/170-research-lab-source-add-provider-origin-uniqueness.sql",
        "scripts/171-research-lab-source-add-duplicate-privacy.sql",
        "scripts/172-research-lab-source-add-claim-control.sql",
        "scripts/173-research-lab-source-add-leg1-release-policy.sql",
        "scripts/178-research-lab-source-add-miner-status.sql",
        "scripts/183-lab-arena-miner-reward-basis.sql",
    }.issubset(set(result["migration_files"]))
    assert "service-role-value" not in str(result)


def test_schema_preflight_provided_activation_avoids_data_request() -> None:
    requests = []
    activation = json.loads(_chain_realized_activation_response())[0]

    def opener(request, *, timeout):
        requests.append((request, timeout))
        if (
            "research_lab_chain_realized_settlement_activation_v1"
            in request.full_url
            and "limit=2" in request.full_url
        ):
            raise AssertionError("provided activation must not query clone rows")
        if request.full_url.endswith("/rest/v1/"):
            paths = {
                f"/rpc/{function_name}": {"post": {}}
                for _migration, function_name in (
                    schema_preflight.REQUIRED_SUPABASE_V2_RPCS
                )
            }
            return _SchemaResponse(body=json.dumps({"paths": paths}).encode())
        if request.full_url.endswith(
            "/rpc/research_lab_compact_weight_settlement_contract_v1"
        ):
            return _SchemaResponse(
                body=_compact_weight_settlement_contract_response()
            )
        if request.full_url.endswith(
            "/rpc/research_lab_source_add_provider_origin_contract_v1"
        ):
            return _SchemaResponse(
                body=_source_add_provider_origin_contract_response()
            )
        if request.full_url.endswith(
            "/rpc/research_lab_source_add_duplicate_privacy_contract_v1"
        ):
            return _SchemaResponse(
                body=_source_add_duplicate_privacy_contract_response()
            )
        if request.full_url.endswith(
            "/rpc/research_lab_source_add_post_accept_leg1_contract_v4"
        ):
            return _SchemaResponse(
                body=_source_add_post_accept_leg1_contract_response()
            )
        if request.full_url.endswith(
                "/rpc/research_lab_source_add_claim_control_contract_v2"
        ):
            return _SchemaResponse(
                body=_source_add_claim_control_contract_response()
            )
        if request.full_url.endswith(
            "/rpc/research_lab_source_add_miner_status_contract_v1"
        ):
            return _SchemaResponse(
                body=_source_add_miner_status_contract_response()
            )
        return _SchemaResponse()

    result = schema_preflight.verify_required_supabase_v2_schema(
        {
            "SUPABASE_URL": "http://127.0.0.1:3000",
            "SUPABASE_SERVICE_ROLE_KEY": "service-role-value",
            "BITTENSOR_NETUID": "71",
        },
        opener=opener,
        chain_realized_activation_authority=activation,
    )

    assert result["status"] == "ready"
    assert result["data_probe_count"] == 6
    assert result["chain_realized_settlement_activation_http_probe_count"] == 0
    assert result["chain_realized_settlement_activation_source"] == (
        "provided-authority"
    )
    assert result["chain_realized_settlement_activation"] == {
        "netuid": 71,
        "first_epoch_id": 24196,
        "source_bundle_hash": "sha256:" + "a" * 64,
        "source_finalized_block": 8715224,
    }
    assert not any(
        "research_lab_chain_realized_settlement_activation_v1"
        in request.full_url
        and "limit=2" in request.full_url
        for request, _timeout in requests
    )
    assert len(requests) == len(schema_preflight.REQUIRED_SUPABASE_V2_SCHEMA) + 7


@pytest.mark.parametrize(
    "contract_field",
    (
        "coverage_complete",
        "collision_free",
        "submission_trigger_enabled",
        "catalog_trigger_enabled",
        "provision_trigger_enabled",
        "terminal_release_trigger_enabled",
        "append_only_trigger_enabled",
        "row_level_security_enabled",
        "service_role_policy_enabled",
    ),
)
def test_source_add_provider_origin_contract_rejects_safety_drift(
    contract_field,
) -> None:
    def opener(_request, *, timeout):
        assert timeout == 10.0
        return _SchemaResponse(
            body=_source_add_provider_origin_contract_response(
                **{contract_field: False}
            )
        )

    with pytest.raises(
        schema_preflight.SupabaseSchemaPreflightV2Error,
        match="SOURCE_ADD provider-origin contract differs",
    ):
        schema_preflight._verify_source_add_provider_origin_contract_v1(
            headers={},
            supabase_url="https://project.supabase.co",
            opener=opener,
            timeout_seconds=10.0,
        )


@pytest.mark.parametrize(
    "overrides",
    (
        {"function_authority_sha256": "sha256:" + "f" * 64},
        {"compatibility_cooldown_seconds": 19},
        {"cooldown_clock": "transaction_start"},
        {"duplicate_precedes_cooldown": False},
    ),
)
def test_source_add_duplicate_privacy_contract_rejects_drift(
    overrides,
) -> None:
    def opener(_request, *, timeout):
        assert timeout == 10.0
        return _SchemaResponse(
            body=_source_add_duplicate_privacy_contract_response(**overrides)
        )

    with pytest.raises(
        schema_preflight.SupabaseSchemaPreflightV2Error,
        match="SOURCE_ADD duplicate-privacy contract differs",
    ):
        schema_preflight._verify_source_add_duplicate_privacy_contract_v1(
            headers={},
            supabase_url="https://project.supabase.co",
            opener=opener,
            timeout_seconds=10.0,
        )


@pytest.mark.parametrize(
    ("section", "field"),
    (
        (None, "view_authority_sha256"),
        (None, "page_authority_sha256"),
        (None, "contract_authority_sha256"),
        (None, "view_security_invoker"),
        (None, "view_security_barrier"),
        (None, "page_security_invoker"),
        (None, "page_stable"),
        ("permissions", "view_service_role_select"),
        ("permissions", "view_anon_select"),
        ("permissions", "view_authenticated_select"),
        ("permissions", "view_public_select"),
        ("permissions", "page_service_role_callable"),
        ("permissions", "page_anon_callable"),
        ("permissions", "page_authenticated_callable"),
        ("permissions", "page_public_callable"),
        ("permissions", "contract_service_role_callable"),
        ("permissions", "contract_anon_callable"),
        ("permissions", "contract_authenticated_callable"),
    ),
)
def test_source_add_miner_status_contract_rejects_privacy_drift(
    section,
    field,
) -> None:
    contract = json.loads(_source_add_miner_status_contract_response())
    target = contract if section is None else contract[section]
    if field.endswith("_sha256"):
        target[field] = "sha256:" + "f" * 64
    else:
        target[field] = not target[field]

    def opener(_request, *, timeout):
        assert timeout == 10.0
        return _SchemaResponse(body=json.dumps(contract).encode())

    with pytest.raises(
        schema_preflight.SupabaseSchemaPreflightV2Error,
        match="SOURCE_ADD miner-status privacy contract differs",
    ):
        schema_preflight._verify_source_add_miner_status_contract_v1(
            headers={},
            supabase_url="https://project.supabase.co",
            opener=opener,
            timeout_seconds=10.0,
        )


@pytest.mark.parametrize(
    "overrides",
    (
        {"function_authority_sha256": "sha256:" + "f" * 64},
        {"control_lock": "source-add-other-control"},
        {"operator_pause_wins": False},
        {"release_restores_pre_restart_state": False},
        {"migration_requires_zero_leased": False},
    ),
)
def test_source_add_claim_control_contract_rejects_drift(overrides) -> None:
    def opener(_request, *, timeout):
        assert timeout == 10.0
        return _SchemaResponse(
            body=_source_add_claim_control_contract_response(**overrides)
        )

    with pytest.raises(
        schema_preflight.SupabaseSchemaPreflightV2Error,
        match="SOURCE_ADD restart-state contract differs",
    ):
        schema_preflight._verify_source_add_claim_control_contract_v2(
            headers={},
            supabase_url="https://project.supabase.co",
            opener=opener,
            timeout_seconds=10.0,
        )


@pytest.mark.parametrize(
    ("section", "field"),
    (
        ("functions", "configure_probe_v3"),
        ("functions", "enqueue_leg1_after_provenance_v1"),
        ("functions", "enqueue_provision_smoke_v2"),
        ("functions", "finalize_leg1_v4"),
        ("functions", "finalize_provision_smoke_v3"),
        ("functions", "finalize_provision_v3"),
        ("functions", "reject_current_builtin_v3"),
        ("functions", "reconcile_provenance_leg1_v1"),
        ("functions", "reserve_leg1_slot_v4"),
        ("triggers", "automatic_enqueue"),
        ("triggers", "eligible_v2"),
        ("triggers", "eligible_v3"),
        ("triggers", "leg1_initial_event_v3"),
        ("triggers", "leg1_obligation_v3"),
        ("triggers", "leg1_slot_v3"),
        ("triggers", "leg1_work_v3"),
        ("columns", "intent_approval_kind"),
        ("columns", "intent_provenance_artifact_hash"),
        ("columns", "intent_provenance_receipt_hash"),
        ("columns", "slot_approval_kind"),
        ("permissions", "service_role_exists"),
        ("permissions", "candidate_callable"),
        ("permissions", "internal_not_callable"),
        ("permissions", "rollback_v2_callable"),
    ),
)
def test_source_add_automatic_provenance_leg1_contract_rejects_safety_drift(
    section,
    field,
) -> None:
    contract = json.loads(_source_add_post_accept_leg1_contract_response())
    contract[section][field] = False

    def opener(_request, *, timeout):
        assert timeout == 10.0
        return _SchemaResponse(body=json.dumps(contract).encode())

    with pytest.raises(
        schema_preflight.SupabaseSchemaPreflightV2Error,
        match="SOURCE_ADD provenance-origin Leg 1 contract differs",
    ):
        schema_preflight._verify_source_add_post_accept_leg1_contract_v4(
            headers={},
            supabase_url="https://project.supabase.co",
            opener=opener,
            timeout_seconds=10.0,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("schema_version", "leadpoet.source_add_post_accept_leg1_contract.v2"),
        (
            "required_migration",
            "scripts/175-research-lab-source-add-provenance-leg1.sql",
        ),
        ("daily_cap", 49),
        ("leg1_alpha_percent", 0.3),
        ("leg1_reward_epochs", 19),
        ("approval_boundary", "post_accept_functional_probe"),
        ("backfill_policy", "none"),
        ("provider_origin_scope", "full_url"),
        ("provider_origin_winner_order", ["submission_id"]),
        ("cancelled_intents_are_authority", True),
        ("public_trigger_fields", ["submission_id"]),
        ("authority_view", "research_lab_source_add_submission_current"),
        ("function_authority_sha256", "sha256:" + "0" * 64),
        ("trigger_authority_sha256", "sha256:" + "0" * 64),
        ("view_authority_sha256", "sha256:" + "0" * 64),
        ("repair_function_authority_sha256", "sha256:" + "0" * 64),
    ),
)
def test_source_add_automatic_provenance_leg1_contract_rejects_policy_drift(
    field,
    value,
) -> None:
    contract = json.loads(_source_add_post_accept_leg1_contract_response())
    contract[field] = value

    def opener(_request, *, timeout):
        assert timeout == 10.0
        return _SchemaResponse(body=json.dumps(contract).encode())

    with pytest.raises(
        schema_preflight.SupabaseSchemaPreflightV2Error,
        match="SOURCE_ADD provenance-origin Leg 1 contract differs",
    ):
        schema_preflight._verify_source_add_post_accept_leg1_contract_v4(
            headers={},
            supabase_url="https://project.supabase.co",
            opener=opener,
            timeout_seconds=10.0,
        )


@pytest.mark.parametrize(
    ("name", "value"),
    (
        ("RESEARCH_LAB_SOURCE_ADD_LEG1_ALPHA_PERCENT", "0.5"),
        ("RESEARCH_LAB_SOURCE_ADD_LEG2_ALPHA_PERCENT", "5"),
        ("RESEARCH_LAB_REWARD_EPOCHS", "21"),
        ("RESEARCH_LAB_SOURCE_ADD_LEG1_MAX_PER_UTC_DAY", "100"),
        ("RESEARCH_LAB_SOURCE_ADD_LEG1_ALPHA_PERCENT", "nan"),
    ),
)
def test_source_add_leg1_release_environment_rejects_policy_drift(
    name,
    value,
) -> None:
    with pytest.raises(
        schema_preflight.SupabaseSchemaPreflightV2Error,
        match="SOURCE_ADD Leg 1 release environment differs",
    ):
        schema_preflight._source_add_leg1_release_environment_policy_v1(
            {name: value}
        )


def test_required_supabase_v2_schema_keeps_only_public_arena_reward_boundary() -> None:
    schema_contract = {
        (migration, relation)
        for migration, relation, _columns in (
            schema_preflight.REQUIRED_SUPABASE_V2_SCHEMA
        )
    }
    rpc_contract = set(schema_preflight.REQUIRED_SUPABASE_V2_RPCS)
    assert all(
        len(function_name.encode("utf-8"))
        <= schema_preflight.POSTGRES_IDENTIFIER_MAX_BYTES
        for _migration, function_name in rpc_contract
    )
    relation_columns = {
        relation: set(columns)
        for _migration, relation, columns in (
            schema_preflight.REQUIRED_SUPABASE_V2_SCHEMA
        )
    }

    assert (
        "scripts/178-research-lab-source-add-miner-status.sql",
        "research_lab_source_add_miner_status_v1",
    ) in schema_contract
    assert {
        "schema_version",
        "submission_id",
        "miner_hotkey",
        "source_name",
        "submitted_at",
        "updated_at",
        "decision_status",
        "decision_reason_code",
        "decision_reason",
        "reward_status",
        "alpha_percent",
        "reward_epochs",
        "start_epoch",
        "end_epoch",
    } == relation_columns["research_lab_source_add_miner_status_v1"]
    assert (
        "scripts/178-research-lab-source-add-miner-status.sql",
        "research_lab_source_add_miner_status_page_v1",
    ) in rpc_contract
    assert (
        "scripts/178-research-lab-source-add-miner-status.sql",
        "research_lab_source_add_miner_status_contract_v1",
    ) in rpc_contract

    private_arena_relations = {
        "lab_arena_rounds",
        "lab_arena_submissions",
        "lab_arena_runs",
        "lab_arena_ledger",
    }
    assert private_arena_relations.isdisjoint(relation_columns)
    private_arena_rpcs = {
        "lab_arena_current_daily_icp_set",
        "lab_arena_register_submission",
        "lab_arena_update_submission",
        "lab_arena_claim_assignment",
        "lab_arena_activate_reward",
        "lab_arena_schema_version_v1",
    }
    assert private_arena_rpcs.isdisjoint(
        function for _migration, function in rpc_contract
    )
    assert (
        "scripts/183-lab-arena-miner-reward-basis.sql",
        "lab_arena_reward_basis_v1",
    ) in schema_contract
    assert {
        "round_id",
        "effective_reward_epoch",
        "reward_basis_hash",
        "reward_basis_doc",
        "signing_key_doc",
        "king_outcome",
        "king_hotkey",
        "king_start_epoch",
        "published_at",
    } == relation_columns["lab_arena_reward_basis_v1"]

    retired_markers = (
        "autoresearch",
        "git-tree",
        "trajectory",
        "private-model",
        "private_benchmark",
        "active_model",
        "candidate_hybrid",
        "routing_",
        "claim_next_research_loop",
        "claim_next_research_lab_candidate",
    )
    required_names = (
        [
            migration
            for migration, _relation, _columns in (
                schema_preflight.REQUIRED_SUPABASE_V2_SCHEMA
            )
        ]
        + [
            relation
            for _migration, relation, _columns in (
                schema_preflight.REQUIRED_SUPABASE_V2_SCHEMA
            )
        ]
        + [
            migration
            for migration, _function in schema_preflight.REQUIRED_SUPABASE_V2_RPCS
        ]
        + [
            function
            for _migration, function in schema_preflight.REQUIRED_SUPABASE_V2_RPCS
        ]
    )
    assert not {
        value
        for value in required_names
        if any(marker in value for marker in retired_markers)
    }


def test_required_supabase_v2_schema_rejects_missing_source_add_miner_status_view() -> None:
    def opener(request, *, timeout):
        del timeout
        if "research_lab_source_add_miner_status_v1?" in request.full_url:
            raise HTTPError(
                request.full_url,
                404,
                "miner status view missing",
                {},
                None,
            )
        return _SchemaResponse()

    with pytest.raises(
        schema_preflight.SupabaseSchemaPreflightV2Error,
        match=(
            r"research_lab_source_add_miner_status_v1.*"
            r"178-research-lab-source-add-miner-status"
        ),
    ):
        schema_preflight.verify_required_supabase_v2_schema(
            {
                "SUPABASE_URL": "https://project.supabase.co",
                "SUPABASE_SERVICE_ROLE_KEY": "service-role-value",
            },
            opener=opener,
        )


def test_required_supabase_v2_schema_rejects_missing_source_add_miner_status_rpc() -> None:
    required_function = "research_lab_source_add_miner_status_page_v1"

    def opener(request, *, timeout):
        del timeout
        if request.full_url.endswith("/rest/v1/"):
            paths = {
                f"/rpc/{function_name}": {"post": {}}
                for _migration, function_name in (
                    schema_preflight.REQUIRED_SUPABASE_V2_RPCS
                )
                if function_name != required_function
            }
            return _SchemaResponse(body=json.dumps({"paths": paths}).encode())
        if (
            "research_lab_chain_realized_settlement_activation_v1"
            in request.full_url
            and "limit=2" in request.full_url
        ):
            return _SchemaResponse(body=_chain_realized_activation_response())
        return _SchemaResponse()

    with pytest.raises(
        schema_preflight.SupabaseSchemaPreflightV2Error,
        match=(
            r"research_lab_source_add_miner_status_page_v1.*"
            r"178-research-lab-source-add-miner-status"
        ),
    ):
        schema_preflight.verify_required_supabase_v2_schema(
            {
                "SUPABASE_URL": "https://project.supabase.co",
                "SUPABASE_SERVICE_ROLE_KEY": "service-role-value",
            },
            opener=opener,
        )


def test_required_supabase_v2_schema_names_missing_migration() -> None:
    def opener(_request, *, timeout):
        del timeout
        raise HTTPError(
            "https://project.supabase.co/rest/v1/missing",
            404,
            "Not Found",
            {},
            None,
        )

    with pytest.raises(
        schema_preflight.SupabaseSchemaPreflightV2Error,
        match=r"validator_sourcing_epoch_inputs_v2.*92-validator-sourcing",
    ):
        schema_preflight.verify_required_supabase_v2_schema(
            {
                "SUPABASE_URL": "https://project.supabase.co",
                "SUPABASE_SERVICE_ROLE_KEY": "service-role-value",
            },
            opener=opener,
        )


def test_required_supabase_v2_schema_names_missing_non_arena_rpc_migration() -> None:
    def opener(request, *, timeout):
        del timeout
        if request.full_url.endswith("/rest/v1/"):
            return _SchemaResponse(body=b'{"paths":{}}')
        if (
            "research_lab_chain_realized_settlement_activation_v1"
            in request.full_url
            and "limit=2" in request.full_url
        ):
            return _SchemaResponse(body=_chain_realized_activation_response())
        return _SchemaResponse()

    with pytest.raises(
        schema_preflight.SupabaseSchemaPreflightV2Error,
        match=(r"persist_research_lab_chain_realized_settlement_v1.*"
               r"126-research-lab-chain-realized-settlement"),
    ):
        schema_preflight.verify_required_supabase_v2_schema(
            {
                "SUPABASE_URL": "https://project.supabase.co",
                "SUPABASE_SERVICE_ROLE_KEY": "service-role-value",
            },
            opener=opener,
        )


def test_required_supabase_v2_schema_requires_transport_purpose_migration() -> None:
    required_function = (
        "research_lab_attested_transport_purpose_contract_v2"
    )

    def opener(request, *, timeout):
        del timeout
        if request.full_url.endswith("/rest/v1/"):
            paths = {
                f"/rpc/{function_name}": {"post": {}}
                for _migration, function_name in (
                    schema_preflight.REQUIRED_SUPABASE_V2_RPCS
                )
                if function_name != required_function
            }
            return _SchemaResponse(body=json.dumps({"paths": paths}).encode())
        if (
            "research_lab_chain_realized_settlement_activation_v1"
            in request.full_url
            and "limit=2" in request.full_url
        ):
            return _SchemaResponse(body=_chain_realized_activation_response())
        return _SchemaResponse()

    with pytest.raises(
        schema_preflight.SupabaseSchemaPreflightV2Error,
        match=(
            r"research_lab_attested_transport_purpose_contract_v2.*"
            r"128-research-lab-chain-settlement-transport-purposes"
        ),
    ):
        schema_preflight.verify_required_supabase_v2_schema(
            {
                "SUPABASE_URL": "https://project.supabase.co",
                "SUPABASE_SERVICE_ROLE_KEY": "service-role-value",
            },
            opener=opener,
        )


def test_required_supabase_v2_schema_requires_transport_terminal_migration() -> None:
    required_function = (
        "research_lab_attested_transport_terminal_contract_v2"
    )

    def opener(request, *, timeout):
        del timeout
        if request.full_url.endswith("/rest/v1/"):
            paths = {
                f"/rpc/{function_name}": {"post": {}}
                for _migration, function_name in (
                    schema_preflight.REQUIRED_SUPABASE_V2_RPCS
                )
                if function_name != required_function
            }
            return _SchemaResponse(body=json.dumps({"paths": paths}).encode())
        if (
            "research_lab_chain_realized_settlement_activation_v1"
            in request.full_url
            and "limit=2" in request.full_url
        ):
            return _SchemaResponse(body=_chain_realized_activation_response())
        return _SchemaResponse()

    with pytest.raises(
        schema_preflight.SupabaseSchemaPreflightV2Error,
        match=(
            r"research_lab_attested_transport_terminal_contract_v2.*"
            r"129-research-lab-attested-local-transport"
        ),
    ):
        schema_preflight.verify_required_supabase_v2_schema(
            {
                "SUPABASE_URL": "https://project.supabase.co",
                "SUPABASE_SERVICE_ROLE_KEY": "service-role-value",
            },
            opener=opener,
        )


def test_required_supabase_v2_schema_requires_provider_outcome_append_migration() -> None:
    required_function = "append_research_lab_provider_outcome_checkpoint_v2"

    def opener(request, *, timeout):
        del timeout
        if request.full_url.endswith("/rest/v1/"):
            paths = {
                f"/rpc/{function_name}": {"post": {}}
                for _migration, function_name in (
                    schema_preflight.REQUIRED_SUPABASE_V2_RPCS
                )
                if function_name != required_function
            }
            return _SchemaResponse(body=json.dumps({"paths": paths}).encode())
        if (
            "research_lab_chain_realized_settlement_activation_v1"
            in request.full_url
            and "limit=2" in request.full_url
        ):
            return _SchemaResponse(body=_chain_realized_activation_response())
        return _SchemaResponse()

    with pytest.raises(
        schema_preflight.SupabaseSchemaPreflightV2Error,
        match=(
            r"append_research_lab_provider_outcome_checkpoint_v2.*"
            r"133-research-lab-provider-outcome-contention-status"
        ),
    ):
        schema_preflight.verify_required_supabase_v2_schema(
            {
                "SUPABASE_URL": "https://project.supabase.co",
                "SUPABASE_SERVICE_ROLE_KEY": "service-role-value",
            },
            opener=opener,
        )


def test_required_supabase_v2_schema_requires_provider_outcome_head_contract() -> None:
    required_function = "research_lab_provider_outcome_contention_contract_v3"

    def opener(request, *, timeout):
        del timeout
        if request.full_url.endswith("/rest/v1/"):
            paths = {
                f"/rpc/{function_name}": {"post": {}}
                for _migration, function_name in (
                    schema_preflight.REQUIRED_SUPABASE_V2_RPCS
                )
                if function_name != required_function
            }
            return _SchemaResponse(body=json.dumps({"paths": paths}).encode())
        if (
            "research_lab_chain_realized_settlement_activation_v1"
            in request.full_url
            and "limit=2" in request.full_url
        ):
            return _SchemaResponse(body=_chain_realized_activation_response())
        return _SchemaResponse()

    with pytest.raises(
        schema_preflight.SupabaseSchemaPreflightV2Error,
        match=(
            r"research_lab_provider_outcome_contention_contract_v3.*"
            r"134-research-lab-provider-outcome-head-contention"
        ),
    ):
        schema_preflight.verify_required_supabase_v2_schema(
            {
                "SUPABASE_URL": "https://project.supabase.co",
                "SUPABASE_SERVICE_ROLE_KEY": "service-role-value",
            },
            opener=opener,
        )


@pytest.mark.parametrize(
    "required_function",
    (
        "put_research_lab_provider_evidence_cache_v2",
        "append_research_lab_provider_outcome_checkpoints_v2",
        "research_lab_provider_persistence_batch_contract_v1",
    ),
)
def test_required_supabase_v2_schema_requires_provider_persistence_batch_contract(
    required_function: str,
) -> None:
    def opener(request, *, timeout):
        del timeout
        if request.full_url.endswith("/rest/v1/"):
            paths = {
                f"/rpc/{function_name}": {"post": {}}
                for _migration, function_name in (
                    schema_preflight.REQUIRED_SUPABASE_V2_RPCS
                )
                if function_name != required_function
            }
            return _SchemaResponse(body=json.dumps({"paths": paths}).encode())
        if (
            "research_lab_chain_realized_settlement_activation_v1"
            in request.full_url
            and "limit=2" in request.full_url
        ):
            return _SchemaResponse(body=_chain_realized_activation_response())
        return _SchemaResponse()

    with pytest.raises(
        schema_preflight.SupabaseSchemaPreflightV2Error,
        match=(
            rf"{required_function}.*"
            r"144-research-lab-provider-persistence-batches"
        ),
    ):
        schema_preflight.verify_required_supabase_v2_schema(
            {
                "SUPABASE_URL": "https://project.supabase.co",
                "SUPABASE_SERVICE_ROLE_KEY": "service-role-value",
            },
            opener=opener,
        )


def test_required_supabase_v2_schema_requires_chain_realized_activation() -> None:
    def opener(request, *, timeout):
        del timeout
        if request.full_url.endswith("/rest/v1/"):
            paths = {
                f"/rpc/{function_name}": {"post": {}}
                for _migration, function_name in (
                    schema_preflight.REQUIRED_SUPABASE_V2_RPCS
                )
            }
            return _SchemaResponse(body=json.dumps({"paths": paths}).encode())
        return _SchemaResponse(body=b"[]")

    with pytest.raises(
        schema_preflight.SupabaseSchemaPreflightV2Error,
        match="activation is missing or ambiguous",
    ):
        schema_preflight.verify_required_supabase_v2_schema(
            {
                "SUPABASE_URL": "https://project.supabase.co",
                "SUPABASE_SERVICE_ROLE_KEY": "service-role-value",
            },
            opener=opener,
        )


def test_required_supabase_v2_schema_requires_credentials() -> None:
    with pytest.raises(
        schema_preflight.SupabaseSchemaPreflightV2Error,
        match="lacks Supabase V2 schema credentials",
    ):
        schema_preflight.verify_required_supabase_v2_schema({})
