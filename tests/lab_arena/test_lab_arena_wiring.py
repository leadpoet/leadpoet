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
