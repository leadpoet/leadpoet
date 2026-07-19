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
    assert result["acceptance_corpus_manifest_hash"] == "sha256:" + "e" * 64


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
    def __init__(self, status: int = 200) -> None:
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        return None

    def getcode(self) -> int:
        return self.status

    def read(self, _size: int = -1) -> bytes:
        return b"["


def test_required_supabase_v2_schema_probes_tables_and_columns() -> None:
    requests = []

    def opener(request, *, timeout):
        requests.append((request, timeout))
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
    )
    assert len(requests) == result["probe_count"]
    assert all("/rest/v1/" in request.full_url for request, _timeout in requests)
    assert all("limit=0" in request.full_url for request, _timeout in requests)
    assert "service-role-value" not in str(result)


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


def test_required_supabase_v2_schema_requires_credentials() -> None:
    with pytest.raises(
        schema_preflight.SupabaseSchemaPreflightV2Error,
        match="lacks Supabase V2 schema credentials",
    ):
        schema_preflight.verify_required_supabase_v2_schema({})
