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


def test_service_config_has_no_miner_registry_or_image_admission_dependency():
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
        "baseline_source_fetcher": lambda _url, _limit: b"archive",
    }
    config = ServiceConfig(**required)
    assert config.baseline_source_fetcher is required["baseline_source_fetcher"]
    assert not hasattr(config, "registry") and not hasattr(config, "source_registry")


def test_public_baseline_download_is_https_and_byte_bounded(monkeypatch):
    class Response:
        status = 200
        headers = {"Content-Length": "7"}

        @staticmethod
        def geturl():
            return "https://downloads.example/baseline.tar.gz"

        @staticmethod
        def read(limit):
            assert limit == 11
            return b"archive"

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    monkeypatch.setattr(
        wiring, "_DIRECT_URLOPEN", lambda request, timeout: Response()
    )
    assert wiring.fetch_public_source_archive(
        "https://example.test/baseline.tar.gz", 10
    ) == b"archive"
    with pytest.raises(ServiceError, match="must use https"):
        wiring.fetch_public_source_archive("http://example.test/source.tar.gz", 10)


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


@pytest.mark.parametrize(
    "raw, expected", [("", 0), ("0", 0), ("23", 23), ("disabled", None)]
)
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


def test_daily_capacity_defaults_are_bounded(monkeypatch):
    monkeypatch.delenv("LAB_ARENA_MAX_CHALLENGERS", raising=False)
    assert wiring._max_challengers_from_environment() == 16
    assert wiring._max_challengers_from_environment() < wiring.contracts.MAX_CHALLENGERS


def test_stage_minutes_override_is_complete_and_testnet_shadow_only(monkeypatch):
    override = {
        "benchmark": 1,
        "stage_1": 30,
        "stage_1_scoring": 10,
        "stage_2": 30,
        "final_scoring": 30,
    }
    monkeypatch.setenv("LAB_ARENA_STAGE_MINUTES", json.dumps(override))
    assert wiring._stage_minutes_from_environment(
        mode="shadow", network_name="test", netuid=401, rewards_enabled=False
    ) == override
    for context in (
        {"mode": "live", "network_name": "test", "netuid": 401, "rewards_enabled": False},
        {"mode": "shadow", "network_name": "finney", "netuid": 71, "rewards_enabled": False},
        {"mode": "shadow", "network_name": "test", "netuid": 401, "rewards_enabled": True},
    ):
        with pytest.raises(ServiceError, match="only for reward-disabled shadow testnet 401"):
            wiring._stage_minutes_from_environment(**context)


def test_stage_minutes_without_an_override_keep_native_defaults(monkeypatch):
    monkeypatch.delenv("LAB_ARENA_STAGE_MINUTES", raising=False)
    assert wiring._stage_minutes_from_environment(
        mode="live", network_name="finney", netuid=71, rewards_enabled=True
    ) == wiring.DEFAULT_STAGE_MINUTES


@pytest.mark.parametrize(
    "raw",
    [
        "not-json",
        json.dumps({"benchmark": 1}),
        json.dumps(
            {
                "benchmark": 31,
                "stage_1": 30,
                "stage_1_scoring": 10,
                "stage_2": 30,
                "final_scoring": 30,
            }
        ),
        json.dumps(
            {
                "benchmark": True,
                "stage_1": 30,
                "stage_1_scoring": 10,
                "stage_2": 30,
                "final_scoring": 30,
            }
        ),
    ],
)
def test_stage_minutes_override_rejects_partial_or_invalid_values(monkeypatch, raw):
    monkeypatch.setenv("LAB_ARENA_STAGE_MINUTES", raw)
    with pytest.raises(ServiceError):
        wiring._stage_minutes_from_environment(
            mode="shadow", network_name="test", netuid=401, rewards_enabled=False
        )


def test_service_requires_at_least_one_runner_hotkey(monkeypatch):
    monkeypatch.delenv("LAB_ARENA_RUNNER_HOTKEYS", raising=False)
    with pytest.raises(ServiceError, match="LAB_ARENA_RUNNER_HOTKEYS"):
        wiring._runner_hotkeys_from_environment()
    monkeypatch.setenv("LAB_ARENA_RUNNER_HOTKEYS", " runner-one, runner-two ")
    assert wiring._runner_hotkeys_from_environment() == (
        "runner-one",
        "runner-two",
    )
