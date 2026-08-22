from __future__ import annotations

import base64
import hashlib
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
    monkeypatch.setattr(
        preflight,
        "verify_required_worker_proxy_profiles_v2",
        lambda **_kwargs: {
            "schema_version": "leadpoet.worker_proxy_profile_set.v2",
            "status": "ready",
            "profile_count": 35,
            "worker_counts": {
                "gateway_autoresearch": 10,
                "gateway_scoring": 25,
            },
        },
    )
    monkeypatch.setattr(
        preflight,
        "load_and_validate_acceptance_corpus_v2",
        lambda *_args, **_kwargs: {
            "manifest_hash": "sha256:" + "e" * 64,
        },
    )
    credential_envelopes = overrides.pop(
        "credential_envelope_paths",
        None,
    ) or _credential_envelopes(tmp_path)
    worker_environment = {
        "RESEARCH_LAB_HOSTED_RUNS_ENABLED": "true",
        "RESEARCH_LAB_EVALUATION_BUNDLES_ENABLED": "true",
        "RESEARCH_LAB_HOSTED_WORKER_PROCESS_COUNT": "10",
        "RESEARCH_LAB_SCORING_WORKER_PROCESS_COUNT": "25",
    }
    values = {
        "deploy_commit": COMMIT,
        "release_manifest": _release(),
        "topology_manifest": manifest_document(),
        "artifact_policy": POLICY,
        "credential_envelope_paths": credential_envelopes,
        "config_dir": tmp_path,
        "topology_mode": "full",
        "instance_type": "r7i.4xlarge",
        "parent_vcpus": 16,
        "parent_memory_mib": 125000,
        "parent_environment": worker_environment,
        "acceptance_corpus_manifest_path": tmp_path / "acceptance.json",
        "acceptance_corpus_root": tmp_path / "acceptance",
    }
    values.update(overrides)
    return preflight.verify_gateway_restart_preflight_v2(**values)


def test_full_restart_preflight_accepts_only_complete_approved_release(
    tmp_path: Path,
    monkeypatch,
) -> None:
    result = _verify(tmp_path, monkeypatch)
    assert result["status"] == "ready"
    assert result["deploy_commit"] == COMMIT
    assert result["instance_type"] == "r7i.4xlarge"
    assert result["role_count"] == 3
    assert result["boot_credential_slot_count"] == 7
    assert result["parent_plaintext_provider_slot_count"] == 0
    assert result["worker_proxy_profile_count"] == 35
    assert result["worker_counts"] == {
        "gateway_autoresearch": 10,
        "gateway_scoring": 25,
    }
    assert result["deferred_worker_fleet_roles"] == []
    assert result["acceptance_corpus_manifest_hash"] == "sha256:" + "e" * 64


def test_full_restart_preflight_reports_explicit_worker_deferral(
    tmp_path: Path,
    monkeypatch,
) -> None:
    parent_environment = {
        "RESEARCH_LAB_HOSTED_RUNS_ENABLED": "true",
        "RESEARCH_LAB_EVALUATION_BUNDLES_ENABLED": "true",
        "RESEARCH_LAB_HOSTED_WORKER_PROCESS_COUNT": "10",
        "RESEARCH_LAB_SCORING_WORKER_PROCESS_COUNT": "25",
        "GATEWAY_V2_DEFER_WORKER_FLEETS": (
            "gateway_autoresearch,gateway_scoring"
        ),
    }

    result = _verify(
        tmp_path,
        monkeypatch,
        parent_environment=parent_environment,
    )

    assert result["status"] == "ready"
    assert result["worker_counts"] == {
        "gateway_autoresearch": 10,
        "gateway_scoring": 25,
    }
    assert result["deferred_worker_fleet_roles"] == [
        "gateway_autoresearch",
        "gateway_scoring",
    ]


def test_full_restart_preflight_requires_explicit_worker_process_counts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    # Worker/proxy decoupling must be explicit: a full restart without the
    # *_PROCESS_COUNT variables silently falls back to one-worker-per-proxy and
    # preserves the oversized fleet, so the preflight must reject it.
    base_env = {
        "RESEARCH_LAB_HOSTED_RUNS_ENABLED": "true",
        "RESEARCH_LAB_EVALUATION_BUNDLES_ENABLED": "true",
        "RESEARCH_LAB_HOSTED_WORKER_PROCESS_COUNT": "10",
        "RESEARCH_LAB_SCORING_WORKER_PROCESS_COUNT": "25",
    }

    # Missing entirely.
    missing = {k: v for k, v in base_env.items() if "PROCESS_COUNT" not in k}
    with pytest.raises(preflight.GatewayRestartPreflightV2Error) as exc:
        _verify(tmp_path, monkeypatch, parent_environment=missing)
    assert "PROCESS_COUNT" in str(exc.value)

    # Present but non-positive / non-integer are also rejected.
    for bad in ("0", "-1", "", "auto"):
        env = dict(base_env)
        env["RESEARCH_LAB_SCORING_WORKER_PROCESS_COUNT"] = bad
        with pytest.raises(preflight.GatewayRestartPreflightV2Error):
            _verify(tmp_path, monkeypatch, parent_environment=env)


def test_component_restart_preflight_does_not_require_process_counts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    # The mandatory-count gate only applies to full (worker-spawning) restarts.
    result = _verify(
        tmp_path,
        monkeypatch,
        topology_mode="component",
        parent_environment={
            "RESEARCH_LAB_HOSTED_RUNS_ENABLED": "true",
            "RESEARCH_LAB_EVALUATION_BUNDLES_ENABLED": "true",
        },
    )
    assert result["status"] == "ready"


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
    assert result["worker_proxy_profile_count"] == 0
    assert result["acceptance_corpus_manifest_hash"] == "component_only"


def test_full_restart_preflight_rejects_missing_acceptance_corpus(
    tmp_path: Path,
    monkeypatch,
) -> None:
    with pytest.raises(
        preflight.GatewayRestartPreflightV2Error,
        match="requires the signed acceptance corpus",
    ):
        _verify(
            tmp_path,
            monkeypatch,
            acceptance_corpus_manifest_path=None,
            acceptance_corpus_root=None,
        )


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


def _candidate_hybrid_constraint_definition() -> str:
    clauses = []
    for role, purposes in schema_preflight.ROLE_PURPOSES.items():
        encoded_purposes = ", ".join(
            "'%s'::text" % purpose for purpose in sorted(purposes)
        )
        clauses.append(
            "((role = '%s'::text) AND (purpose = ANY (ARRAY[%s])))"
            % (role, encoded_purposes)
        )
    return "CHECK (%s)" % " OR ".join(clauses)


def _candidate_hybrid_purpose_contract_response() -> bytes:
    return json.dumps(
        {
            "schema_version": (
                "leadpoet.research_lab_candidate_hybrid_purpose_contract.v1"
            ),
            "constraint_name": (
                "research_lab_attested_execution_receipts_v2_role_purpose_check"
            ),
            "constraint_valid": True,
            "constraint_definition": _candidate_hybrid_constraint_definition(),
        }
    ).encode()


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
            "/rpc/research_lab_candidate_hybrid_purpose_contract_v1"
        ):
            return _SchemaResponse(
                body=_candidate_hybrid_purpose_contract_response()
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
    ) + len(schema_preflight.REQUIRED_SUPABASE_V2_RPCS) + 3
    assert result["table_probe_count"] == len(
        schema_preflight.REQUIRED_SUPABASE_V2_SCHEMA
    )
    assert result["rpc_probe_count"] == len(
        schema_preflight.REQUIRED_SUPABASE_V2_RPCS
    )
    assert result["data_probe_count"] == 3
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
    assert result["candidate_hybrid_purpose_contract"] == {
        "schema_version": (
            "leadpoet.research_lab_candidate_hybrid_purpose_contract.v1"
        ),
        "constraint_name": (
            "research_lab_attested_execution_receipts_v2_role_purpose_check"
        ),
        "constraint_valid": True,
        "role_count": len(schema_preflight.ROLE_PURPOSES),
        "role_purpose_pair_count": sum(
            len(purposes)
            for purposes in schema_preflight.ROLE_PURPOSES.values()
        ),
        "constraint_definition_sha256": "sha256:"
        + hashlib.sha256(
            _candidate_hybrid_constraint_definition().encode("utf-8")
        ).hexdigest(),
    }
    assert len(requests) == result["table_probe_count"] + 4
    assert all("/rest/v1/" in request.full_url for request, _timeout in requests)
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
    hybrid_contract_requests = [
        request
        for request in table_requests
        if request.full_url.endswith(
            "/rpc/research_lab_candidate_hybrid_purpose_contract_v1"
        )
    ]
    schema_table_requests = [
        request
        for request in table_requests
        if request not in activation_requests
        and request not in contract_requests
        and request not in hybrid_contract_requests
    ]
    assert all(
        "limit=0" in request.full_url for request in schema_table_requests
    )
    assert len(activation_requests) == 1
    assert len(contract_requests) == 1
    assert len(hybrid_contract_requests) == 1
    assert len(schema_requests) == 1
    assert schema_requests[0].headers["Accept"] == "application/openapi+json"
    assert {
        "scripts/95-research-lab-git-tree-autoresearch.sql",
        "scripts/115-research-lab-git-tree-root-replacement.sql",
        "scripts/117-research-lab-trajectory-antijoin.sql",
        "scripts/118-research-lab-maintenance-lease.sql",
        "scripts/119-research-lab-provider-usage-batch-insert.sql",
        "scripts/120-research-lab-trajectory-delta.sql",
        "scripts/121-research-lab-atomic-candidate-claim.sql",
        "scripts/122-research-lab-atomic-run-claim.sql",
        "scripts/123-research-lab-corpus-completeness.sql",
        "scripts/126-research-lab-chain-realized-settlement.sql",
        "scripts/127-research-lab-chain-unattributed-settlement.sql",
        "scripts/128-research-lab-chain-settlement-transport-purposes.sql",
        "scripts/129-research-lab-attested-local-transport.sql",
        "scripts/132-research-lab-champion-lifetime-credit.sql",
        "scripts/133-research-lab-provider-outcome-contention-status.sql",
        "scripts/134-research-lab-provider-outcome-head-contention.sql",
        "scripts/144-research-lab-provider-persistence-batches.sql",
        "scripts/145-research-lab-source-add-admission-control.sql",
        "scripts/146-research-lab-private-benchmark-schema-v11.sql",
        "scripts/148-research-lab-atomic-credit-resume.sql",
        "scripts/149-research-lab-compact-weight-settlement-authority.sql",
        "scripts/154-research-lab-model-compatibility-purpose.sql",
        "scripts/155-research-lab-ancestry-disclosure-root-fast-path.sql",
        "scripts/156-production-parity-readonly-role.sql",
        "scripts/161-research-lab-exact-model-transitions.sql",
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
            "/rpc/research_lab_candidate_hybrid_purpose_contract_v1"
        ):
            return _SchemaResponse(
                body=_candidate_hybrid_purpose_contract_response()
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
    assert result["data_probe_count"] == 3
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
    assert len(requests) == len(schema_preflight.REQUIRED_SUPABASE_V2_SCHEMA) + 3


def test_candidate_hybrid_purpose_contract_rejects_scope_drift() -> None:
    contract = json.loads(_candidate_hybrid_purpose_contract_response())
    contract["constraint_definition"] = contract[
        "constraint_definition"
    ].replace(
        "'research_lab.candidate_hybrid_discovery.v2'::text, ",
        "",
    )

    def opener(_request, *, timeout):
        assert timeout == 10.0
        return _SchemaResponse(body=json.dumps(contract).encode())

    with pytest.raises(
        schema_preflight.SupabaseSchemaPreflightV2Error,
        match="differs from canonical roles",
    ):
        schema_preflight._verify_candidate_hybrid_purpose_contract_v1(
            headers={},
            supabase_url="https://project.supabase.co",
            opener=opener,
            timeout_seconds=10.0,
        )


def test_required_supabase_v2_schema_covers_git_tree_runtime_contract() -> None:
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
        "scripts/95-research-lab-git-tree-autoresearch.sql",
        "research_lab_autoresearch_tree_node_current",
    ) in schema_contract
    assert {
        "root_manifest_hash",
        "root_source_tree_hash",
        "root_git_commit",
        "root_image_digest",
        "policy_hash",
        "evaluator_commitment_hash",
        "tree_doc",
        "current_event_type",
        "current_event_doc",
        "current_event_hash",
        "current_round_index",
        "current_frontier_hash",
        "current_frontier_doc",
    }.issubset(relation_columns["research_lab_autoresearch_tree_current"])
    assert {
        "tree_generation",
        "replaces_tree_id",
        "root_manifest_hash",
        "policy_hash",
        "evaluator_commitment_hash",
        "tree_doc",
        "current_event_doc",
        "current_event_hash",
    }.issubset(
        relation_columns["research_lab_autoresearch_run_tree_current"]
    )
    assert {
        "event_type",
        "node_id",
        "previous_event_hash",
        "event_doc",
        "created_at",
    }.issubset(
        relation_columns["research_lab_autoresearch_tree_events"]
    )
    assert {
        "schema_version",
        "expected_previous_hash",
        "frontier_doc",
        "commitment_hash",
        "created_at",
    }.issubset(
        relation_columns["research_lab_autoresearch_frontier_commitments"]
    )
    assert {"run_id", "candidate_id"}.issubset(
        relation_columns["research_lab_autoresearch_tree_handoffs"]
    )
    assert (
        "scripts/95-research-lab-git-tree-autoresearch.sql",
        "research_lab_autoresearch_operation_current",
    ) in schema_contract
    assert (
        "scripts/95-research-lab-git-tree-autoresearch.sql",
        "research_lab_autoresearch_tree_current",
    ) in schema_contract
    assert (
        "scripts/115-research-lab-git-tree-root-replacement.sql",
        "research_lab_autoresearch_run_tree_current",
    ) in schema_contract

    assert {
        (
            "scripts/95-research-lab-git-tree-autoresearch.sql",
            "create_research_lab_autoresearch_tree",
        ),
        (
            "scripts/95-research-lab-git-tree-autoresearch.sql",
            "plan_research_lab_autoresearch_tree_node",
        ),
        (
            "scripts/95-research-lab-git-tree-autoresearch.sql",
            "append_research_lab_autoresearch_tree_event",
        ),
        (
            "scripts/95-research-lab-git-tree-autoresearch.sql",
            "transition_research_lab_autoresearch_operation",
        ),
        (
            "scripts/95-research-lab-git-tree-autoresearch.sql",
            "commit_research_lab_autoresearch_frontier",
        ),
        (
            "scripts/95-research-lab-git-tree-autoresearch.sql",
            "select_research_lab_autoresearch_tree_final",
        ),
        (
            "scripts/95-research-lab-git-tree-autoresearch.sql",
            "record_research_lab_autoresearch_tree_handoff",
        ),
        (
            "scripts/115-research-lab-git-tree-root-replacement.sql",
            "create_research_lab_git_tree_candidate_handoff",
        ),
        (
            "scripts/115-research-lab-git-tree-root-replacement.sql",
            "research_lab_autoresearch_run_evaluation_usage",
        ),
    }.issubset(rpc_contract)


def test_required_supabase_v2_schema_rejects_incomplete_git_tree_current_view() -> None:
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
        if (
            "research_lab_autoresearch_tree_current?" in request.full_url
        ):
            raise HTTPError(
                request.full_url,
                400,
                "current_event_doc column missing",
                {},
                None,
            )
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
            r"research_lab_autoresearch_tree_current.*"
            r"95-research-lab-git-tree-autoresearch"
        ),
    ):
        schema_preflight.verify_required_supabase_v2_schema(
            {
                "SUPABASE_URL": "https://project.supabase.co",
                "SUPABASE_SERVICE_ROLE_KEY": "service-role-value",
            },
            opener=opener,
        )


def test_required_supabase_v2_schema_rejects_incomplete_git_tree_event_table() -> None:
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
        if "research_lab_autoresearch_tree_events?" in request.full_url:
            raise HTTPError(
                request.full_url,
                400,
                "event_doc column missing",
                {},
                None,
            )
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
            r"research_lab_autoresearch_tree_events.*"
            r"95-research-lab-git-tree-autoresearch"
        ),
    ):
        schema_preflight.verify_required_supabase_v2_schema(
            {
                "SUPABASE_URL": "https://project.supabase.co",
                "SUPABASE_SERVICE_ROLE_KEY": "service-role-value",
            },
            opener=opener,
        )


def test_required_supabase_v2_schema_rejects_missing_git_tree_replacement_rpc() -> None:
    required_function = "research_lab_autoresearch_run_evaluation_usage"

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
            r"research_lab_autoresearch_run_evaluation_usage.*"
            r"115-research-lab-git-tree-root-replacement"
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


def test_required_supabase_v2_schema_names_missing_rpc_migration() -> None:
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
        match=(
            r"create_research_lab_autoresearch_tree.*"
            r"95-research-lab-git-tree-autoresearch"
        ),
    ):
        schema_preflight.verify_required_supabase_v2_schema(
            {
                "SUPABASE_URL": "https://project.supabase.co",
                "SUPABASE_SERVICE_ROLE_KEY": "service-role-value",
            },
            opener=opener,
        )


def test_required_supabase_v2_schema_requires_lineage_generation_rpc() -> None:
    required_function = "research_lab_private_model_lineage_generation"

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
            r"research_lab_private_model_lineage_generation.*"
            r"153-research-lab-private-model-lineage-generation"
        ),
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


def test_required_supabase_v2_schema_requires_private_benchmark_schema_contract() -> None:
    required_function = "research_lab_private_benchmark_schema_contract_v1"

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
            r"research_lab_private_benchmark_schema_contract_v1.*"
            r"146-research-lab-private-benchmark-schema-v11"
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
