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
    ) + len(schema_preflight.REQUIRED_SUPABASE_V2_RPCS) + 2
    assert result["table_probe_count"] == len(
        schema_preflight.REQUIRED_SUPABASE_V2_SCHEMA
    )
    assert result["rpc_probe_count"] == len(
        schema_preflight.REQUIRED_SUPABASE_V2_RPCS
    )
    assert result["data_probe_count"] == 2
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
    assert len(requests) == result["table_probe_count"] + 3
    assert all("/rest/v1/" in request.full_url for request, _timeout in requests)
    requested_urls = [request.full_url for request, _timeout in requests]
    assert any("/rest/v1/lab_arena_reward_basis_v1?" in url for url in requested_urls)
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
    schema_table_requests = [
        request
        for request in table_requests
        if request not in activation_requests
        and request not in contract_requests
    ]
    assert all(
        "limit=0" in request.full_url for request in schema_table_requests
    )
    assert len(activation_requests) == 1
    assert len(contract_requests) == 1
    assert len(schema_requests) == 1
    assert schema_requests[0].headers["Accept"] == "application/openapi+json"
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
    assert result["data_probe_count"] == 2
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
    assert len(requests) == len(schema_preflight.REQUIRED_SUPABASE_V2_SCHEMA) + 2
















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






@pytest.mark.parametrize(
    "required_function",
    (
        "put_research_lab_provider_evidence_cache_v2",
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
