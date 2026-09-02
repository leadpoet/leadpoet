"""Production wiring fails closed without its environment and never prints secrets."""

from __future__ import annotations

import io
import json

import pytest

from lab_arena import benchmark, wiring
from lab_arena.service import ServiceError


def test_service_wiring_requires_every_environment_value(monkeypatch):
    for name in list(__import__("os").environ):
        if name.startswith("LAB_ARENA_"):
            monkeypatch.delenv(name, raising=False)
    with pytest.raises(ServiceError) as failure:
        wiring.build_service_from_environment("shadow")
    assert "LAB_ARENA_SUPABASE_URL" in str(failure.value)


def test_generation_provider_maps_failures_and_redacts_key():
    key = "sk-or-v1-" + "z" * 40

    class Response:
        def __init__(self, status, body):
            self.status = status
            self._body = body

        def read(self, _n):
            return self._body

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    captured = {}

    def urlopen(request, timeout):
        captured["auth"] = request.get_header("Authorization")
        captured["body"] = json.loads(request.data.decode())
        return Response(200, json.dumps({"choices": [{"message": {"content": "{}"}}]}).encode())

    provider = wiring.OpenRouterGenerationProvider(key, urlopen=urlopen)
    result = provider.chat(messages=[{"role": "user", "content": "hi"}], temperature=0.0, max_tokens=None, timeout_seconds=5)
    assert result["choices"] and captured["auth"] == "Bearer " + key and "max_tokens" not in captured["body"]
    assert key not in repr(provider)

    def failing(request, timeout):
        raise OSError("boom")

    with pytest.raises(benchmark.ProviderFailure):
        wiring.OpenRouterGenerationProvider(key, urlopen=failing).chat(messages=[], temperature=0.0, max_tokens=10, timeout_seconds=5)

    def bad_status(request, timeout):
        return Response(500, b"{}")

    with pytest.raises(benchmark.ProviderFailure):
        wiring.OpenRouterGenerationProvider(key, urlopen=bad_status).chat(messages=[], temperature=0.0, max_tokens=10, timeout_seconds=5)
    with pytest.raises(ServiceError):
        wiring.OpenRouterGenerationProvider("")


def test_banned_hotkeys_snapshot_must_be_a_json_list(tmp_path, monkeypatch):
    path = tmp_path / "banned.json"
    path.write_text(json.dumps({"not": "a list"}))
    monkeypatch.setenv("LAB_ARENA_BANNED_HOTKEYS_PATH", str(path))
    with pytest.raises(ServiceError):
        wiring.banned_hotkeys_from_environment()
    path.write_text(json.dumps(["5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"]))
    assert wiring.banned_hotkeys_from_environment() == ["5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"]
    monkeypatch.delenv("LAB_ARENA_BANNED_HOTKEYS_PATH")
    assert wiring.banned_hotkeys_from_environment() == []


def test_funding_confirmer_maps_rejections_and_outages_without_leaking():
    from datetime import datetime, timezone

    from lab_arena import chain as chain_module, funding, wiring

    class StubChain:
        def __init__(self, error):
            self.error = error

        def finalized_head(self):
            raise self.error

    config = funding.FundingConfig(recipient_wallet="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY", network_name="finney")
    confirm = wiring.funding_confirmer(chain=StubChain(chain_module.ArenaChainError("endpoint down")), config=config, store=None, price_source=None, clock=lambda: datetime(2026, 9, 2, tzinfo=timezone.utc))
    with pytest.raises(ServiceError, match="funding_unavailable"):
        confirm("5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY", {"block_hash": "0x" + "a" * 64, "extrinsic_index": 1})
    malformed = confirm("5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY", {"block_hash": "not-a-hash", "extrinsic_index": 1})
    assert malformed["credited"] is False and malformed["rule"] in ("reference_malformed", "finality") or malformed.get("rejected")


def test_credential_registrar_rejects_bad_envelopes_and_records_good_ones():
    import json

    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    from lab_arena import credentials, wiring

    private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    recipient = credentials.recipient_document(private.public_key().public_bytes(serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo))
    decryptor = credentials.LocalRsaDecryptor(private)
    raw_key = "sk-or-v1-" + "w" * 40

    def urlopen(request, timeout):
        class Response:
            def read(self):
                return json.dumps({"data": {"limit": None, "limit_remaining": None, "usage": 1.25, "disabled": False}}).encode()

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        return Response()

    register = wiring.credential_registrar(decryptor=decryptor, urlopen=urlopen)
    record = register(credentials.encrypt_runtime_key(recipient, raw_key))
    assert record["preflight_status"] == "ok" and record["limit_remaining_microusd"] is None and record["usage_microusd"] == 1_250_000
    assert raw_key not in json.dumps(record)
    with pytest.raises(ServiceError, match="envelope_invalid"):
        register({"schema_version": "x"})
    tampered = credentials.encrypt_runtime_key(recipient, raw_key)
    tampered["ciphertext_b64"] = tampered["ciphertext_b64"][:-4] + "AAAA"
    with pytest.raises(ServiceError):
        register(tampered)
