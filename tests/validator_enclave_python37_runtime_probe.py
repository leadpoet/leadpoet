#!/usr/bin/env python3
"""Import the measured Arena signer surface under its production interpreter."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from leadpoet_canonical.arena_weights import (
    ACCEPTED_WEIGHT_STATE_SCHEMA_VERSION,
    ArenaWeightError,
    derive_arena_weights,
    validate_accepted_weight_state,
)
from validator_tee.enclave.arena_hotkey import ArenaHotkeyAuthority
from validator_tee.enclave.arena_weight_signer import ArenaWeightSigner


def main():
    if ACCEPTED_WEIGHT_STATE_SCHEMA_VERSION != "leadpoet.arena.accepted_weight_state.v1":
        raise SystemExit("unexpected Arena state schema")
    if not all((ArenaWeightError, derive_arena_weights, validate_accepted_weight_state)):
        raise SystemExit("Arena canonical surface is incomplete")
    if not all((ArenaHotkeyAuthority, ArenaWeightSigner)):
        raise SystemExit("Arena protected signer surface is incomplete")
    print("PYTHON37_ARENA_SIGNER_PROBE_SUCCESS")


if __name__ == "__main__":
    main()
