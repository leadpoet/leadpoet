from gateway import config


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
