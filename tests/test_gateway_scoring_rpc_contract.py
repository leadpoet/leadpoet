import importlib
import base64
import json
import os
import sys
from pathlib import Path


def _tee_service(monkeypatch):
    tee_dir = Path(__file__).resolve().parents[1] / "gateway" / "tee"
    monkeypatch.syspath_prepend(str(tee_dir))
    monkeypatch.setenv("LEADPOET_ENCLAVE_ROLE", "gateway_scoring_a")
    return importlib.import_module("gateway.tee.tee_service")


def test_v1_scoring_rpc_is_not_authorized_for_v2_role(monkeypatch):
    service = _tee_service(monkeypatch)
    for method in (
        "scoring_configure_runtime",
        "scoring_health",
        "scoring_submit_job",
        "scoring_get_status",
        "scoring_get_result",
    ):
        response = service.handle_rpc(method, {})
        assert response == {
            "error": "RPC method is not authorized for enclave role gateway_scoring_a"
        }
    assert service.handle_rpc("unknown", {}) == {
        "error": "RPC method is not authorized for enclave role gateway_scoring_a"
    }


def test_scoring_attestation_binds_exact_job_purpose_and_inputs(monkeypatch):
    service = _tee_service(monkeypatch)
    captured = {}

    def fake_attestation(*, user_data_fields):
        captured.update(user_data_fields)
        return {"attestation_document": b"signed-document".hex()}

    monkeypatch.setattr(service, "get_attestation_document_with_pcrs", fake_attestation)
    manifest = {
        "purpose": "research_lab.candidate_score.v1",
        "epoch_id": 42,
        "job_id": "job-42",
        "config_hash": "sha256:" + "a" * 64,
        "payload_sha256": "sha256:" + "b" * 64,
    }
    encoded = service._scoring_attestation_document_b64(manifest)
    assert base64.b64decode(encoded) == b"signed-document"
    assert captured == {
        "purpose": manifest["purpose"],
        "epoch_id": 42,
        "job_id": "job-42",
        "config_hash": manifest["config_hash"],
        "input_root": manifest["payload_sha256"],
    }


def test_v1_scoring_runtime_configuration_is_inaccessible(monkeypatch):
    from gateway.tee.scoring_executor import (
        SCORING_RUNTIME_ENV_NAMES,
        configuration_hash,
    )

    service = _tee_service(monkeypatch)
    monkeypatch.setattr(service, "scoring_job_manager", None)
    monkeypatch.setattr(service, "scoring_runtime_configuration", None)
    for name in SCORING_RUNTIME_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    environment = {name: None for name in SCORING_RUNTIME_ENV_NAMES}
    environment["QUALIFICATION_OPENROUTER_API_KEY"] = "secret-value-never-returned"
    expected_hash = configuration_hash(environment)
    params = {
        "schema_version": "leadpoet.gateway_scoring_runtime.v1",
        "environment": environment,
        "configuration_hash": expected_hash,
    }

    response = service.handle_rpc("scoring_configure_runtime", params)
    assert response == {
        "error": "RPC method is not authorized for enclave role gateway_scoring_a"
    }
    assert "QUALIFICATION_OPENROUTER_API_KEY" not in os.environ
