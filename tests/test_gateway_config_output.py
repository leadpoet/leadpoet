import os
from pathlib import Path
import subprocess
import sys

import pytest

from gateway import config


@pytest.mark.parametrize(
    ("ambient_value", "expected_value"),
    ((None, "canonical-cache"), ("ambient-process", "ambient-process")),
)
def test_gateway_config_precedence_is_process_then_cache_then_dotenv(
    tmp_path,
    ambient_value,
    expected_value,
):
    env_name = "LEADPOET_TEST_CONFIG_PRECEDENCE"
    gateway_env = tmp_path / "gateway.env"
    gateway_env.write_text(f"{env_name}=canonical-cache\n", encoding="utf-8")
    (tmp_path / ".env").write_text(f"{env_name}=stale-dotenv\n", encoding="utf-8")

    environment = os.environ.copy()
    environment["GATEWAY_ENV_FILE"] = str(gateway_env)
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    if ambient_value is None:
        environment.pop(env_name, None)
    else:
        environment[env_name] = ambient_value

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            f"import gateway.config, os; print('RESULT=' + os.environ[{env_name!r}])",
        ],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    assert f"RESULT={expected_value}" in result.stdout.splitlines()
    assert "stale-dotenv" not in result.stdout


def test_fallback_env_diagnostic_does_not_contaminate_stdout(
    tmp_path,
    monkeypatch,
    capsys,
):
    env_name = "LEADPOET_TEST_FALLBACK_STDERR"
    monkeypatch.delenv(env_name, raising=False)
    env_file = tmp_path / "gateway.env"
    env_file.write_text(f"{env_name}=loaded\n", encoding="utf-8")

    config._load_gateway_env_file(env_file)

    captured = capsys.readouterr()
    assert captured.out == ""
    assert f"Loaded 1 fallback env var(s) from {env_file}" in captured.err
    assert config.os.environ[env_name] == "loaded"


def test_gateway_env_loader_skips_static_aws_keys_for_instance_role(
    tmp_path,
    monkeypatch,
):
    env_file = tmp_path / "gateway.env"
    env_file.write_text(
        "AWS_ACCESS_KEY_ID=stale-access\n"
        "AWS_SECRET_ACCESS_KEY=stale-secret\n"
        "AWS_PROFILE=stale-profile\n"
        "LEADPOET_TEST_SAFE_VALUE=kept\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("LEADPOET_AWS_INSTANCE_ROLE_ONLY", "true")
    monkeypatch.delenv("LEADPOET_TEST_SAFE_VALUE", raising=False)
    for key in config._AWS_STATIC_CREDENTIAL_KEYS:
        monkeypatch.delenv(key, raising=False)

    config._load_gateway_env_file(env_file)

    assert config.os.environ["LEADPOET_TEST_SAFE_VALUE"] == "kept"
    assert not (config._AWS_STATIC_CREDENTIAL_KEYS & set(config.os.environ))


def test_gateway_config_accepts_instance_role_without_static_keys(monkeypatch):
    monkeypatch.setenv("LEADPOET_AWS_INSTANCE_ROLE_ONLY", "true")
    monkeypatch.setattr(config, "AWS_ACCESS_KEY_ID", None)
    monkeypatch.setattr(config, "AWS_SECRET_ACCESS_KEY", None)
    monkeypatch.setattr(config, "AWS_PROFILE", None)
    monkeypatch.setattr(config, "SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setattr(config, "SUPABASE_SERVICE_ROLE_KEY", "configured")

    assert config.validate_config() is True
