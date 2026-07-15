"""Regression gates proving rollout-era V1 cannot be selected at runtime."""

from __future__ import annotations

from pathlib import Path

import pytest

from validator_tee.host.weight_protocol_v2 import (
    AUTHORITATIVE_V2_PROTOCOL,
    normalize_weight_protocol,
)


ROOT = Path(__file__).resolve().parents[1]


def test_validator_protocol_defaults_to_v2_and_rejects_v1():
    assert normalize_weight_protocol(None) == AUTHORITATIVE_V2_PROTOCOL
    assert normalize_weight_protocol("AUTHORITATIVE_V2") == AUTHORITATIVE_V2_PROTOCOL
    with pytest.raises(RuntimeError, match="V1 authority is retired"):
        normalize_weight_protocol("legacy_v1_compat")


def test_primary_validator_and_restart_have_no_v1_runtime_branch():
    validator = (ROOT / "neurons" / "validator.py").read_text(encoding="utf-8")
    restart = (ROOT / "validator_restart.sh").read_text(encoding="utf-8")

    assert "build_legacy_v1_submission" not in validator
    assert "_publish_legacy_v1_bundle" not in validator
    assert "_set_legacy_weights_until_epoch_end" not in validator
    assert "legacy_v1_compat" not in restart
    assert "verify_legacy_v1_enclave" not in restart
    assert not (ROOT / "validator_tee" / "host" / "legacy_v1_compat.py").exists()
