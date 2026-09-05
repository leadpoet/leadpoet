"""The miner CLI accepts local source and has no image workflow."""

from __future__ import annotations

import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MINER = runpy.run_path(
    str(ROOT / "scripts" / "lab_arena_miner.py"), run_name="lab_arena_miner_module"
)


def test_submit_model_uses_source_wallet_and_environment_credentials():
    parser = MINER["build_parser"]()
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
    args = MINER["build_parser"]().parse_args(
        ["submit-source", "--source", "./agent"]
    )
    assert args.command == "submit-source"


def test_retired_image_and_manual_envelope_commands_are_absent():
    parser = MINER["build_parser"]()
    for command in ("submission-body", "sign"):
        try:
            parser.parse_args([command])
        except SystemExit as exc:
            assert exc.code == 2
        else:  # pragma: no cover
            raise AssertionError("retired command remained available")
