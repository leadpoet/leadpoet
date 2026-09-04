"""The miner CLI accepts local source and has no image workflow."""

from __future__ import annotations

import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MINER = runpy.run_path(
    str(ROOT / "scripts" / "lab_arena_miner.py"), run_name="lab_arena_miner_module"
)


def test_submit_source_needs_only_source_and_wallet_inputs():
    parser = MINER["build_parser"]()
    args = parser.parse_args(
        [
            "submit-source",
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


def test_retired_image_and_manual_envelope_commands_are_absent():
    parser = MINER["build_parser"]()
    for command in ("submission-body", "sign"):
        try:
            parser.parse_args([command])
        except SystemExit as exc:
            assert exc.code == 2
        else:  # pragma: no cover
            raise AssertionError("retired command remained available")
