"""Temporary exact testnet runtime support; removed after the chain proof."""

import pytest

from leadpoet_canonical.chain_source_v2 import (
    ChainSourceV2Error,
    resolve_reveal_period_metadata_default_v2,
)


REVIEWED = {
    "genesis_hash": "8f9cf856bf558a14440e75569c9e58594757048d7b3a84b5d25f6bd978263105",
    "runtime_spec_version": 455,
    "runtime_transaction_version": 1,
    "metadata_hash": "sha256:74c4067de4bf2eba95156e8a46c793b52fcd9862dfeb28502632e46416979ec7",
}


def test_exact_testnet455_metadata_default():
    assert resolve_reveal_period_metadata_default_v2(**REVIEWED) == 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("genesis_hash", "0" * 64),
        ("runtime_spec_version", 454),
        ("runtime_transaction_version", 2),
        ("metadata_hash", "sha256:" + "0" * 64),
    ],
)
def test_unreviewed_testnet_metadata_fails_closed(field, value):
    with pytest.raises(ChainSourceV2Error, match="not reviewed"):
        resolve_reveal_period_metadata_default_v2(**{**REVIEWED, field: value})
