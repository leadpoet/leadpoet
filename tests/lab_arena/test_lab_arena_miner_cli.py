"""The miner CLI accepts local source and has no image workflow."""

from __future__ import annotations

from lab_arena import miner_cli


def test_console_entrypoint_defaults_to_interactive(monkeypatch):
    monkeypatch.setattr(miner_cli.sys, "argv", ["leadpoet"])
    monkeypatch.setattr(miner_cli, "interactive", lambda _args: 17)
    assert miner_cli.main() == 17


def test_submit_model_uses_source_wallet_and_environment_credentials():
    parser = miner_cli.build_parser()
    args = parser.parse_args(
        [
            "submit-model",
            "--source",
            "./agent",
            "--wallet-name",
            "miner",
            "--hotkey-name",
            "default",
            "--wallet-path",
            "/var/lib/miner-wallets",
        ]
    )
    assert args.source == "./agent"
    assert args.wallet_path == "/var/lib/miner-wallets"
    assert not hasattr(args, "image")
    assert not hasattr(args, "openrouter_api_key")
    assert not hasattr(args, "openrouter_management_key")
    assert not hasattr(args, "deepline_api_key")


def test_submit_source_remains_a_compatibility_alias():
    args = miner_cli.build_parser().parse_args(
        ["submit-source", "--source", "./agent"]
    )
    assert args.command == "submit-source"


def test_retired_image_and_manual_envelope_commands_are_absent():
    parser = miner_cli.build_parser()
    for command in ("submission-body", "sign"):
        try:
            parser.parse_args([command])
        except SystemExit as exc:
            assert exc.code == 2
        else:  # pragma: no cover
            raise AssertionError("retired command remained available")


def test_scripted_submission_formats_real_failure(monkeypatch, capsys):
    args = miner_cli.build_parser().parse_args(["submit-model", "--source", "./agent"])
    monkeypatch.setattr(miner_cli, "submission_credentials_from_environment", lambda: {
        "openrouter_api_key": "openrouter-execution-secret",
        "openrouter_management_key": "openrouter-management-secret",
        "deepline_api_key": "deepline-execution-secret",
    })
    monkeypatch.setattr(miner_cli, "_keypair", lambda _args: object())
    monkeypatch.setattr(miner_cli, "submit_agent_source", lambda **_kwargs: (_ for _ in ()).throw(
        miner_cli.MinerSubmissionError("source_upload_failed", "http_403\nopenrouter-execution-secret")
    ))
    assert miner_cli.submit_source(args) == 2
    rendered = capsys.readouterr().err
    assert "source_upload_failed (http_403\\x0a[REDACTED])" in rendered
    assert "TypeError" not in rendered


def test_default_and_interactive_read_published_terms_without_opening_wallet(monkeypatch, capsys):
    import json
    from types import SimpleNamespace

    def get(url, **kwargs):
        assert url.startswith("https://gateway.subnet71.com/arena/v1/collection/terms")
        return SimpleNamespace(status_code=200, json=lambda: {"mode": "pilot", "scope": "synthetic"})

    def no_wallet(_):
        raise AssertionError("Viewing collection terms must not open a wallet")

    monkeypatch.setattr(miner_cli.sys, "argv", ["leadpoet"])
    monkeypatch.setattr(miner_cli, "_keypair", no_wallet)
    monkeypatch.setattr(miner_cli.collection_cli.requests, "get", get)
    monkeypatch.delenv("LAB_ARENA_API_BASE_URL", raising=False)
    monkeypatch.delenv("GATEWAY_URL", raising=False)
    for argv in (None, [], ["interactive"], ["interactive", "--task-scope", "synthetic_corrections"]):
        assert miner_cli.main(argv) == 0
        output = capsys.readouterr()
        assert json.loads(output.out)["mode"] == "pilot"
        assert "collection-register" in output.err


def test_public_legacy_mutations_stop_before_wallet_or_source(monkeypatch, capsys):
    def no_secrets(*_):
        raise AssertionError("Retired public commands must not request credentials")

    monkeypatch.delenv("LAB_ARENA_API_BASE_URL", raising=False)
    monkeypatch.delenv("GATEWAY_URL", raising=False)
    monkeypatch.setattr(miner_cli, "submission_credentials_from_environment", no_secrets)
    monkeypatch.setattr(miner_cli, "_keypair", no_secrets)
    for argv in (
        ["submit-model", "--source", "absent"],
        ["submit-source", "--source", "absent", "--api-base-url", "https://GATEWAY.subnet71.com:443/"],
        ["retry-credit-failures", "--round-id", "old", "--submission-id", "old"],
    ):
        assert miner_cli.main(argv) == 2
        assert "collection-terms" in capsys.readouterr().err
