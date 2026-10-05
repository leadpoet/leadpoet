"""The Arena launcher reuses only the existing gateway telemetry destination."""

import pytest

from lab_arena import telemetry
from scripts import run_lab_arena_service as launcher


@pytest.fixture(autouse=True)
def isolated_telemetry(monkeypatch):
    for key in launcher.OTEL_ENVIRONMENT_KEYS:
        monkeypatch.delenv(key, raising=False)
    telemetry.install_recorder(None)
    yield
    telemetry.install_recorder(None)


def test_loads_only_existing_telemetry_keys(tmp_path, monkeypatch):
    import os

    path = tmp_path / "gateway.env"
    path.write_text(
        "GATEWAY_OTEL_ENABLED=1\n"
        "GATEWAY_OTEL_ENDPOINT=https://collector.invalid/v1/traces\n"
        "GATEWAY_OTEL_TOKEN=fixture-ingest-token\n"
        "OPENROUTER_API_KEY=fixture-provider-secret\n"
        "GATEWAY_OTEL_METRICS_ENDPOINT=https://metrics.invalid\n"
    )
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("GATEWAY_OTEL_METRICS_ENDPOINT", raising=False)
    launcher.load_otel_environment(path)
    assert os.environ["GATEWAY_OTEL_ENABLED"] == "1"
    assert os.environ["GATEWAY_OTEL_ENDPOINT"] == "https://collector.invalid/v1/traces"
    assert os.environ["GATEWAY_OTEL_TOKEN"] == "fixture-ingest-token"
    assert "OPENROUTER_API_KEY" not in os.environ
    assert "GATEWAY_OTEL_METRICS_ENDPOINT" not in os.environ


def test_explicit_disable_is_preserved(tmp_path, monkeypatch):
    import os

    monkeypatch.setenv("GATEWAY_OTEL_ENABLED", "0")
    path = tmp_path / "gateway.env"
    path.write_text("GATEWAY_OTEL_ENABLED=1\n")
    launcher.load_otel_environment(path)
    assert os.environ["GATEWAY_OTEL_ENABLED"] == "0"


def test_unreadable_telemetry_configuration_does_not_block_startup(tmp_path, monkeypatch, capsys):
    from gateway.observability import read_gateway_otel_env

    def unreadable(*args):
        raise OSError("fixture-private-config-detail")

    monkeypatch.setattr(read_gateway_otel_env, "parse_env_file", unreadable)
    launcher.load_otel_environment(tmp_path / "absent.env")
    log = capsys.readouterr().err
    assert "OSError" in log
    assert "fixture-private-config-detail" not in log


def test_failed_telemetry_installation_does_not_block_service(monkeypatch, capsys):
    from gateway.observability import otel_bootstrap

    def broken(*args, **kwargs):
        raise RuntimeError("fixture-private-exporter-detail")

    monkeypatch.setattr(otel_bootstrap, "configure_arena_otel", broken)
    launcher._install_arena_telemetry(object())
    # Use the same seam called by the live driver after initialization.
    with telemetry.stage("driver_tick"):
        pass
    log = capsys.readouterr().err
    assert "RuntimeError" in log
    assert "fixture-private-exporter-detail" not in log
