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
