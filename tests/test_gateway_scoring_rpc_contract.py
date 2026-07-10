import importlib
import base64
import json
import os
import sys
from pathlib import Path


class _FakeManager:
    def health(self):
        return {"mode": "shadow"}

    def status(self, job_id):
        return {"job_id": job_id, "state": "running"}


def _tee_service(monkeypatch):
    tee_dir = Path(__file__).resolve().parents[1] / "gateway" / "tee"
    monkeypatch.syspath_prepend(str(tee_dir))
    return importlib.import_module("gateway.tee.tee_service")


def test_new_scoring_rpc_does_not_replace_existing_methods(monkeypatch):
    service = _tee_service(monkeypatch)
    fake = _FakeManager()
    monkeypatch.setattr(service, "get_scoring_job_manager", lambda: fake)

    assert service.handle_rpc("scoring_health", {}) == {"result": {"mode": "shadow"}}
    assert service.handle_rpc("scoring_get_status", {"job_id": "job-1"}) == {
        "result": {"job_id": "job-1", "state": "running"}
    }
    assert isinstance(service.handle_rpc("get_buffer_size", {})["result"], int)
    assert service.handle_rpc("unknown", {})["error"] == "Unknown method: unknown"


def test_scoring_rpc_rejections_use_existing_client_error_shape(monkeypatch):
    service = _tee_service(monkeypatch)

    def fail():
        raise ValueError("invalid scoring request")

    monkeypatch.setattr(service, "get_scoring_job_manager", fail)
    response = service.handle_rpc("scoring_health", {})
    assert response == {"status": "error", "error": "invalid scoring request"}


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


def test_scoring_runtime_configuration_is_one_time_hashed_and_secret_safe(monkeypatch):
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

    first = service.handle_rpc("scoring_configure_runtime", params)
    second = service.handle_rpc("scoring_configure_runtime", params)

    assert first == second
    assert first["result"]["status"] == "configured"
    assert first["result"]["configuration_hash"] == expected_hash
    assert "secret-value-never-returned" not in json.dumps(first)
    assert os.environ["QUALIFICATION_OPENROUTER_API_KEY"] == "secret-value-never-returned"

    changed = dict(environment)
    changed["QUALIFICATION_OPENROUTER_API_KEY"] = "different-secret"
    rejected = service.handle_rpc(
        "scoring_configure_runtime",
        {
            **params,
            "environment": changed,
            "configuration_hash": configuration_hash(changed),
        },
    )
    assert rejected["status"] == "error"
    assert "immutable" in rejected["error"]
