"""Keep native CLI analytics separate from Arena execution and capture."""

import os
from pathlib import Path
import tomllib

import pytest

from lab_arena import lab_arena_codex as codex
from tests.lab_arena.codex_runtime_test import broker_socket


@pytest.mark.parametrize("web_search", ["disabled", "live"])
@pytest.mark.parametrize("reasoning_effort", ["medium", "high"])
def test_session_disables_optional_analytics_without_changing_execution(
    monkeypatch, web_search, reasoning_effort,
):
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:18080")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:18080")
    monkeypatch.setenv("NO_PROXY", "existing.example")
    monkeypatch.setenv("LAB_ARENA_TRAINING_EVENTS_PATH", "/output/events.jsonl")
    original_environment = dict(os.environ)

    with broker_socket(monkeypatch), codex.session(
        model="openai/gpt-4o-mini", reasoning_effort=reasoning_effort,
        web_search=web_search,
    ) as environment:
        home = Path(environment["CODEX_HOME"])
        config = tomllib.loads((home / "config.toml").read_text())
        # Other sections already contain enabled=false. It must apply to the
        # native analytics section as well, rather than disabling another tool.
        assert config["analytics"] == {"enabled": False}
        assert config["agents"] == {"enabled": False}
        assert "otel" not in config
        assert config["model"] == "openai/gpt-4o-mini"
        assert config["model_provider"] == "arena"
        assert config["model_reasoning_effort"] == reasoning_effort
        assert config["web_search"] == web_search
        assert config["model_auto_compact_token_limit"] == 64_000
        provider = config["model_providers"]["arena"]
        assert provider["base_url"] == environment["LAB_ARENA_CODEX_BASE_URL"]
        assert provider["env_key"] == "LAB_ARENA_CODEX_TOKEN"
        assert provider["wire_api"] == "responses"
        assert provider["requires_openai_auth"] is False
        assert provider["supports_websockets"] is False
        assert provider["supports_standalone_web_search"] is True
        assert provider["request_max_retries"] == provider["stream_max_retries"] == 0
        for name in ("HTTP_PROXY", "HTTPS_PROXY", "LAB_ARENA_TRAINING_EVENTS_PATH"):
            assert environment[name] == original_environment[name]
        assert environment["NO_PROXY"].split(",") == [
            "existing.example", "127.0.0.1", "localhost",
        ]

    assert not home.exists()
    # Session configuration must not mutate the validator's own telemetry.
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "LAB_ARENA_TRAINING_EVENTS_PATH"):
        assert os.environ[name] == original_environment[name]
