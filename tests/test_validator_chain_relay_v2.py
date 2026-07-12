from __future__ import annotations

import json

import pytest

from leadpoet_canonical.chain_source_v2 import (
    CHAIN_ENDPOINT_HOST,
    chain_source_policy_hash,
)
from validator_tee.host.chain_relay_v2 import (
    ValidatorChainRelayV2Error,
    _validate_control,
)


def _control(**overrides):
    value = {
        "schema_version": "leadpoet.validator_chain_relay.v2",
        "host": CHAIN_ENDPOINT_HOST,
        "port": 443,
        "policy_hash": chain_source_policy_hash(),
    }
    value.update(overrides)
    return value


def test_validator_chain_relay_accepts_only_fixed_measured_destination():
    _validate_control(_control())
    with pytest.raises(ValidatorChainRelayV2Error, match="measured chain"):
        _validate_control(_control(host="attacker.example"))
    with pytest.raises(ValidatorChainRelayV2Error, match="policy hash"):
        _validate_control(_control(policy_hash="sha256:" + "0" * 64))
    with pytest.raises(ValidatorChainRelayV2Error, match="fields"):
        _validate_control({**_control(), "extra": True})
