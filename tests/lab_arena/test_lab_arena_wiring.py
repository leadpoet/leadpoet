"""Production wiring fails closed without its environment and never prints secrets."""

from __future__ import annotations

import json

import pytest

from lab_arena import wiring
from lab_arena.service import ServiceConfig, ServiceError


def test_service_wiring_requires_every_environment_value(monkeypatch):
    for name in list(__import__("os").environ):
        if name.startswith("LAB_ARENA_"):
            monkeypatch.delenv(name, raising=False)
    with pytest.raises(ServiceError) as failure:
        wiring.build_service_from_environment("shadow")
    assert "LAB_ARENA_SUPABASE_URL" in str(failure.value)


def test_service_config_requires_a_distinct_anonymous_source_registry():
    destination = object()
    required = {
        "mode": "live",
        "store": object(),
        "object_store": object(),
        "signer": object(),
        "chain": object(),
        "verify_signature": lambda *_args: True,
        "daily_icp_source": lambda **_kwargs: {"status": "unavailable"},
        "banned_hotkeys_source": lambda: (),
        "broker_factory": lambda *_args: object(),
        "registry": destination,
    }
    with pytest.raises(ServiceError, match="anonymous_source_registry_required"):
        ServiceConfig(**required)
    with pytest.raises(ServiceError, match="anonymous_source_registry_required"):
        ServiceConfig(**required, source_registry=destination)
    assert ServiceConfig(**required, source_registry=object()).source_registry is not destination


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


@pytest.mark.parametrize("raw, expected", [("", None), ("0", 0), ("23", 23)])
def test_daily_cutoff_hour_is_read_from_the_environment(monkeypatch, raw, expected):
    from lab_arena import wiring

    monkeypatch.setenv("LAB_ARENA_DAILY_CUTOFF_UTC", raw)
    assert wiring._daily_cutoff_hour_from_environment() == expected


@pytest.mark.parametrize("raw", ["24", "-1", "noon"])
def test_daily_cutoff_hour_rejects_values_outside_the_day(monkeypatch, raw):
    from lab_arena import wiring
    from lab_arena.service import ServiceError

    monkeypatch.setenv("LAB_ARENA_DAILY_CUTOFF_UTC", raw)
    with pytest.raises(ServiceError):
        wiring._daily_cutoff_hour_from_environment()
