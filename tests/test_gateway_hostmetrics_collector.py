from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OBSERVABILITY = ROOT / "gateway" / "observability"
INSTALLER = OBSERVABILITY / "install_hostmetrics_collector.sh"
CONFIG = OBSERVABILITY / "otelcol-hostmetrics.yaml"
ENV_READER = OBSERVABILITY / "read_gateway_otel_env.py"


def _load_env_reader():
    spec = importlib.util.spec_from_file_location("read_gateway_otel_env", ENV_READER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gateway_restart_does_not_provision_hostmetrics() -> None:
    restart = (ROOT / "gw_restart.sh").read_text(encoding="utf-8")
    assert "ensure_hostmetrics_collector" not in restart
    assert "install_hostmetrics_collector.sh" not in restart


def test_env_reader_handles_gateway_format_without_executing_shell(
    tmp_path: Path,
    monkeypatch,
) -> None:
    marker = tmp_path / "must-not-exist"
    env_path = tmp_path / "gateway.env"
    env_path.write_bytes(
        b"# comment\n"
        b"GATEWAY_OTEL_ENDPOINT='https://example.onepatch.dev/v1/traces'\0"
        b"GATEWAY_OTEL_ENDPOINT=https://ignored.example/v1/traces\n"
        b"GATEWAY_OTEL_TOKEN=\"op_file_token\"\n"
        b"INVALID LINE $(touch "
        + str(marker).encode()
        + b")\n"
        b"GATEWAY_OTEL_METRICS_ENDPOINT=https://example.onepatch.dev/v1/metrics\n"
    )

    reader = _load_env_reader()
    values = reader.parse_env_file(env_path)

    assert values == {
        "GATEWAY_OTEL_ENDPOINT": "https://example.onepatch.dev/v1/traces",
        "GATEWAY_OTEL_TOKEN": "op_file_token",
        "GATEWAY_OTEL_METRICS_ENDPOINT": "https://example.onepatch.dev/v1/metrics",
    }
    assert not marker.exists()

    monkeypatch.setenv("GATEWAY_OTEL_TOKEN", "op_ambient_token")
    assert reader.resolve_value(env_path, "GATEWAY_OTEL_TOKEN") == "op_ambient_token"


def test_installer_verifies_artifact_and_bounds_host_mutations() -> None:
    installer = INSTALLER.read_text(encoding="utf-8")

    assert 'OTELCOL_VERSION="${OTELCOL_VERSION:-0.153.0}"' in installer
    assert "sha256sum --check" in installer
    assert "--connect-timeout 10" in installer
    assert "--max-time 180" in installer
    assert "--retry-all-errors" in installer
    assert "rpm -Uvh" not in installer
    assert 'source "$GATEWAY_ENV_FILE"' not in installer
    assert 'validate --config "$CONFIG_SRC"' in installer
    assert "systemctl is-active --quiet" in installer
    assert installer.count("timeout --signal=TERM --kill-after=5s 30s") >= 5


def test_service_is_unprivileged_and_resource_bounded() -> None:
    installer = INSTALLER.read_text(encoding="utf-8")

    for directive in (
        "User=$SERVICE_USER",
        "Group=$SERVICE_GROUP",
        "CapabilityBoundingSet=",
        "AmbientCapabilities=",
        "ProtectSystem=strict",
        "ProtectHome=true",
        "NoNewPrivileges=true",
        "MemoryMax=192M",
        "TasksMax=128",
    ):
        assert directive in installer


def test_collector_does_not_export_ec2_metadata_or_open_an_ingest_receiver() -> None:
    config = CONFIG.read_text(encoding="utf-8")

    assert "receivers:\n  hostmetrics:" in config
    assert "detectors: [system]" in config
    assert "detectors: [env, system, ec2]" not in config
    assert "service.name\n        value: leadpoet-gateway-host" in config
    assert "memory_limiter" in config
    assert "otlp:" not in config
