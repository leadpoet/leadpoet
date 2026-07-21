import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from gateway.research_lab import key_vault
from gateway.research_lab.config import ResearchLabGatewayConfig
from gateway.research_lab.worker import ResearchLabHostedWorker, HostedResearchLabWorkerError
import gateway.research_lab.worker as worker_module


class _FakeResponse:
    def __init__(self, payload: dict[str, Any], *, status: int = 200):
        self.payload = payload
        self.status = status
        self.code = status

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def test_openrouter_workspace_privacy_verification_patches_and_proves(monkeypatch):
    runtime_key = "sk-or-v1-" + "a" * 48
    management_key = "sk-or-v1-" + "b" * 48
    calls: list[tuple[str, str, Any]] = []

    def fake_urlopen(req, timeout: int):
        body = json.loads((req.data or b"{}").decode("utf-8")) if getattr(req, "data", None) else None
        calls.append((req.get_method(), req.full_url, body))
        if req.full_url.endswith("/api/v1/key"):
            return _FakeResponse(
                {
                    "data": {
                        "hash": "runtime-key-hash",
                        "label": "research-lab-runtime",
                        "creator_user_id": "user-1",
                    }
                }
            )
        if req.full_url.endswith("/api/v1/workspaces"):
            return _FakeResponse({"data": [{"id": "workspace-1"}]})
        if "/api/v1/keys?" in req.full_url:
            return _FakeResponse(
                {"data": [{"label": "research-lab-runtime", "creator_user_id": "user-1"}]}
            )
        if req.full_url.endswith("/api/v1/workspaces/workspace-1") and req.get_method() == "PATCH":
            return _FakeResponse({"data": {"id": "workspace-1"}})
        if req.full_url.endswith("/api/v1/workspaces/workspace-1") and req.get_method() == "GET":
            return _FakeResponse(
                {
                    "data": {
                        "id": "workspace-1",
                        "is_observability_io_logging_enabled": False,
                        "is_data_discount_logging_enabled": False,
                        "is_observability_broadcast_enabled": False,
                        "io_logging_api_key_ids": None,
                    }
                }
            )
        raise AssertionError(f"unexpected OpenRouter URL: {req.get_method()} {req.full_url}")

    monkeypatch.setattr(key_vault.urlrequest, "urlopen", fake_urlopen)

    proof = key_vault.verify_openrouter_workspace_privacy(
        runtime_key=runtime_key,
        management_key=management_key,
        stage="unit_test",
    )

    patch_calls = [call for call in calls if call[0] == "PATCH"]
    assert patch_calls
    assert calls[-1][0] == "GET"
    assert patch_calls[-1][2] == {
        "is_observability_io_logging_enabled": False,
        "is_data_discount_logging_enabled": False,
        "is_observability_broadcast_enabled": False,
    }
    assert proof["logging_flags"]["is_observability_io_logging_enabled"] is False
    assert proof["request_policy"] == {
        "data_collection": "deny",
        "allow_fallbacks": False,
    }
    assert runtime_key not in json.dumps(proof)
    assert management_key not in json.dumps(proof)


def test_openrouter_provider_policy_is_routeable_privacy_policy():
    policy = key_vault.strict_openrouter_provider_policy()

    assert policy == {
        "data_collection": "deny",
        "allow_fallbacks": False,
    }
    assert "zdr" not in policy
    assert "require_parameters" not in policy


def test_openrouter_key_validation_normalizes_prefix_case_only():
    raw = "Sk-or-v1-" + "A" * 48

    normalized = key_vault.validate_openrouter_key_format(raw)

    assert normalized == "sk-or-v1-" + "A" * 48


def test_openrouter_workspace_privacy_rejects_unmatched_workspace(monkeypatch):
    runtime_key = "sk-or-v1-" + "a" * 48
    management_key = "sk-or-v1-" + "b" * 48

    def fake_urlopen(req, timeout: int):
        if req.full_url.endswith("/api/v1/key"):
            return _FakeResponse(
                {"data": {"hash": "runtime-key-hash", "label": "runtime", "creator_user_id": "user-1"}}
            )
        if req.full_url.endswith("/api/v1/workspaces"):
            return _FakeResponse({"data": [{"id": "workspace-1"}]})
        if "/api/v1/keys?" in req.full_url:
            return _FakeResponse({"data": [{"label": "other", "creator_user_id": "user-2"}]})
        raise AssertionError(f"unexpected OpenRouter URL: {req.get_method()} {req.full_url}")

    monkeypatch.setattr(key_vault.urlrequest, "urlopen", fake_urlopen)

    with pytest.raises(key_vault.OpenRouterKeyVaultError, match="does not control"):
        key_vault.verify_openrouter_workspace_privacy(
            runtime_key=runtime_key,
            management_key=management_key,
            stage="unit_test",
        )


def test_openrouter_workspace_privacy_rejects_when_get_still_reports_logging_enabled(monkeypatch):
    runtime_key = "sk-or-v1-" + "a" * 48
    management_key = "sk-or-v1-" + "b" * 48

    def fake_urlopen(req, timeout: int):
        if req.full_url.endswith("/api/v1/key"):
            return _FakeResponse(
                {"data": {"hash": "runtime-key-hash", "label": "runtime", "creator_user_id": "user-1"}}
            )
        if req.full_url.endswith("/api/v1/workspaces"):
            return _FakeResponse({"data": [{"id": "workspace-1"}]})
        if "/api/v1/keys?" in req.full_url:
            return _FakeResponse({"data": [{"label": "runtime", "creator_user_id": "user-1"}]})
        if req.full_url.endswith("/api/v1/workspaces/workspace-1") and req.get_method() == "PATCH":
            return _FakeResponse({"data": {"id": "workspace-1"}})
        if req.full_url.endswith("/api/v1/workspaces/workspace-1") and req.get_method() == "GET":
            return _FakeResponse(
                {
                    "data": {
                        "id": "workspace-1",
                        "is_observability_io_logging_enabled": True,
                        "is_data_discount_logging_enabled": False,
                        "is_observability_broadcast_enabled": False,
                    }
                }
            )
        raise AssertionError(f"unexpected OpenRouter URL: {req.get_method()} {req.full_url}")

    monkeypatch.setattr(key_vault.urlrequest, "urlopen", fake_urlopen)

    with pytest.raises(key_vault.OpenRouterKeyVaultError, match="could not be verified off"):
        key_vault.verify_openrouter_workspace_privacy(
            runtime_key=runtime_key,
            management_key=management_key,
            stage="unit_test",
        )


def test_hidden_openrouter_call_writes_privacy_proof_before_prompt(monkeypatch):
    events: list[str] = []
    bodies: list[dict[str, Any]] = []

    def fake_verify(**kwargs):
        events.append("proof")
        return {
            "source": "openrouter_workspace_privacy_guard",
            "stage": kwargs["stage"],
            "workspace_id_hash": "a" * 64,
            "runtime_key_hash": "runtime-key-hash",
            "management_key_hash": "b" * 64,
            "logging_flags": {
                "is_observability_io_logging_enabled": False,
                "is_data_discount_logging_enabled": False,
                "is_observability_broadcast_enabled": False,
            },
            "request_policy": kwargs["request_policy"],
            "verified_at": "2026-07-04T00:00:00+00:00",
            "proof_hash": "sha256:" + "c" * 64,
        }

    def fake_insert(**kwargs):
        events.append("insert")
        assert kwargs["proof_status"] == "passed"
        assert kwargs["proof_doc"]["request_policy"]["allow_fallbacks"] is False
        return {"event_id": "00000000-0000-0000-0000-000000000001"}

    def fake_urlopen(req, timeout: int):
        events.append("post")
        body = json.loads((req.data or b"{}").decode("utf-8"))
        bodies.append(body)
        return _FakeResponse(
            {"model": "test/model", "choices": [{"message": {"content": '{"ok": true}'}}]}
        )

    monkeypatch.setattr(worker_module, "verify_openrouter_workspace_privacy", fake_verify)
    monkeypatch.setattr(worker_module, "create_openrouter_privacy_proof_event_sync", fake_insert)
    monkeypatch.setattr(worker_module.urlrequest, "urlopen", fake_urlopen)

    worker = ResearchLabHostedWorker(ResearchLabGatewayConfig(auto_research_model="test/model"))
    result = asyncio.run(
        worker._call_openrouter(
            messages=[{"role": "user", "content": '{"task":"test"}'}],
            api_key="sk-or-v1-" + "a" * 48,
            model_id="test/model",
            max_tokens=8,
            timeout_seconds=7,
            capture_run_id="00000000-0000-0000-0000-000000000002",
            capture_stage="code_edit_draft",
            privacy_key_ref="encrypted_ref:openrouter:" + "d" * 32,
            privacy_miner_hotkey="5F" + "x" * 46,
            privacy_management_key="sk-or-v1-" + "b" * 48,
        )
    )

    assert result.content == '{"ok": true}'
    assert events[:3] == ["proof", "insert", "post"]
    assert bodies[0]["provider"] == {
        "data_collection": "deny",
        "allow_fallbacks": False,
    }
    assert "zdr" not in bodies[0]["provider"]
    assert "require_parameters" not in bodies[0]["provider"]
    assert "include_reasoning" not in bodies[0]
    assert "reasoning_effort" not in bodies[0]
    assert "reasoning" not in bodies[0]


def test_hidden_openrouter_call_preserves_explicit_reasoning_effort(monkeypatch):
    bodies: list[dict[str, Any]] = []

    def fake_verify(**kwargs):
        return {
            "source": "openrouter_workspace_privacy_guard",
            "stage": kwargs["stage"],
            "workspace_id_hash": "a" * 64,
            "runtime_key_hash": "runtime-key-hash",
            "management_key_hash": "b" * 64,
            "logging_flags": {
                "is_observability_io_logging_enabled": False,
                "is_data_discount_logging_enabled": False,
                "is_observability_broadcast_enabled": False,
            },
            "request_policy": kwargs["request_policy"],
            "verified_at": "2026-07-04T00:00:00+00:00",
            "proof_hash": "sha256:" + "c" * 64,
        }

    def fake_insert(**kwargs):
        assert kwargs["proof_status"] == "passed"
        return {"event_id": "00000000-0000-0000-0000-000000000011"}

    def fake_urlopen(req, timeout: int):
        body = json.loads((req.data or b"{}").decode("utf-8"))
        bodies.append(body)
        return _FakeResponse(
            {"model": "test/model", "choices": [{"message": {"content": '{"ok": true}'}}]}
        )

    monkeypatch.setattr(worker_module, "verify_openrouter_workspace_privacy", fake_verify)
    monkeypatch.setattr(worker_module, "create_openrouter_privacy_proof_event_sync", fake_insert)
    monkeypatch.setattr(worker_module.urlrequest, "urlopen", fake_urlopen)

    worker = ResearchLabHostedWorker(ResearchLabGatewayConfig(auto_research_model="test/model"))
    asyncio.run(
        worker._call_openrouter(
            messages=[{"role": "user", "content": '{"task":"test"}'}],
            api_key="sk-or-v1-" + "a" * 48,
            model_id="test/model",
            reasoning_effort="max",
            max_tokens=8,
            timeout_seconds=7,
            capture_run_id="00000000-0000-0000-0000-000000000012",
            capture_stage="alignment_judge",
            privacy_key_ref="encrypted_ref:openrouter:" + "d" * 32,
            privacy_miner_hotkey="5F" + "x" * 46,
            privacy_management_key="sk-or-v1-" + "b" * 48,
        )
    )

    assert bodies[0]["reasoning_effort"] == "max"
    assert bodies[0]["reasoning"] == {"effort": "max"}
    assert bodies[0]["include_reasoning"] is True


def test_hidden_openrouter_call_can_opt_into_bare_include_reasoning(monkeypatch):
    bodies: list[dict[str, Any]] = []

    def fake_verify(**kwargs):
        return {
            "source": "openrouter_workspace_privacy_guard",
            "stage": kwargs["stage"],
            "workspace_id_hash": "a" * 64,
            "runtime_key_hash": "runtime-key-hash",
            "management_key_hash": "b" * 64,
            "logging_flags": {
                "is_observability_io_logging_enabled": False,
                "is_data_discount_logging_enabled": False,
                "is_observability_broadcast_enabled": False,
            },
            "request_policy": kwargs["request_policy"],
            "verified_at": "2026-07-04T00:00:00+00:00",
            "proof_hash": "sha256:" + "c" * 64,
        }

    def fake_insert(**kwargs):
        assert kwargs["proof_status"] == "passed"
        return {"event_id": "00000000-0000-0000-0000-000000000013"}

    def fake_urlopen(req, timeout: int):
        body = json.loads((req.data or b"{}").decode("utf-8"))
        bodies.append(body)
        return _FakeResponse(
            {"model": "test/model", "choices": [{"message": {"content": '{"ok": true}'}}]}
        )

    monkeypatch.setenv("RESEARCH_LAB_LLM_INCLUDE_REASONING", "true")
    monkeypatch.setattr(worker_module, "verify_openrouter_workspace_privacy", fake_verify)
    monkeypatch.setattr(worker_module, "create_openrouter_privacy_proof_event_sync", fake_insert)
    monkeypatch.setattr(worker_module.urlrequest, "urlopen", fake_urlopen)

    worker = ResearchLabHostedWorker(ResearchLabGatewayConfig(auto_research_model="test/model"))
    asyncio.run(
        worker._call_openrouter(
            messages=[{"role": "user", "content": '{"task":"test"}'}],
            api_key="sk-or-v1-" + "a" * 48,
            model_id="test/model",
            max_tokens=8,
            timeout_seconds=7,
            capture_run_id="00000000-0000-0000-0000-000000000014",
            capture_stage="code_edit_draft",
            privacy_key_ref="encrypted_ref:openrouter:" + "d" * 32,
            privacy_miner_hotkey="5F" + "x" * 46,
            privacy_management_key="sk-or-v1-" + "b" * 48,
        )
    )

    assert bodies[0]["include_reasoning"] is True
    assert "reasoning_effort" not in bodies[0]
    assert "reasoning" not in bodies[0]


def test_hidden_openrouter_call_blocks_before_prompt_when_privacy_fails(monkeypatch):
    events: list[str] = []

    def fake_verify(**_kwargs):
        events.append("proof")
        raise key_vault.OpenRouterKeyVaultError("logging still enabled")

    def fake_insert(**kwargs):
        events.append(f"insert:{kwargs['proof_status']}")
        return {"event_id": "00000000-0000-0000-0000-000000000003"}

    def fake_urlopen(_req, timeout: int):
        events.append("post")
        raise AssertionError("OpenRouter prompt was sent without privacy proof")

    monkeypatch.setattr(worker_module, "verify_openrouter_workspace_privacy", fake_verify)
    monkeypatch.setattr(worker_module, "create_openrouter_privacy_proof_event_sync", fake_insert)
    monkeypatch.setattr(worker_module.urlrequest, "urlopen", fake_urlopen)

    worker = ResearchLabHostedWorker(ResearchLabGatewayConfig(auto_research_model="test/model"))
    with pytest.raises(HostedResearchLabWorkerError, match="privacy verification failed"):
        asyncio.run(
            worker._call_openrouter(
                messages=[{"role": "user", "content": '{"task":"test"}'}],
                api_key="sk-or-v1-" + "a" * 48,
                model_id="test/model",
                max_tokens=8,
                timeout_seconds=7,
                capture_run_id="00000000-0000-0000-0000-000000000004",
                capture_stage="code_edit_draft",
                privacy_key_ref="encrypted_ref:openrouter:" + "d" * 32,
                privacy_miner_hotkey="5F" + "x" * 46,
                privacy_management_key="sk-or-v1-" + "b" * 48,
            )
        )

    assert events == ["proof", "insert:failed"]


def test_hidden_openrouter_call_requires_privacy_context(monkeypatch):
    def fake_urlopen(_req, timeout: int):
        raise AssertionError("OpenRouter prompt was sent without privacy context")

    monkeypatch.setattr(worker_module.urlrequest, "urlopen", fake_urlopen)
    worker = ResearchLabHostedWorker(ResearchLabGatewayConfig(auto_research_model="test/model"))

    with pytest.raises(HostedResearchLabWorkerError, match="privacy proof context is required"):
        asyncio.run(
            worker._call_openrouter(
                messages=[{"role": "user", "content": '{"task":"test"}'}],
                api_key="sk-or-v1-" + "a" * 48,
                model_id="test/model",
                max_tokens=8,
                timeout_seconds=7,
            )
        )


def test_miner_cli_encrypts_both_keys_with_attested_recipients():
    source = (Path(__file__).resolve().parents[1] / "neurons" / "miner.py").read_text()
    runtime_prompt_index = source.index("OpenRouter API key (hidden; encrypted locally)")
    management_prompt_index = source.index(
        "OpenRouter management key (hidden; encrypted locally)"
    )
    recipient_index = source.index(
        '"/research-lab/openrouter-keys/credential-recipient"'
    )
    encryption_index = source.index(
        "verify_and_encrypt_openrouter_credential_v2("
    )
    payload_index = source.index(
        '"openrouter_management_key_v2": encrypted_management'
    )

    assert (
        runtime_prompt_index
        < management_prompt_index
        < recipient_index
        < encryption_index
        < payload_index
    )
    prompt_block = source[runtime_prompt_index:payload_index]
    assert "logging" not in prompt_block.lower()
    assert '"openrouter_api_key": raw_key' not in source
    assert '"openrouter_management_key": raw_management_key' not in source
