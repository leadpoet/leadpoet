from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from Leadpoet.utils.subnet_epoch import (
    OFFICIAL_BITTENSOR_ARCHIVE_ENDPOINT,
    SubnetEpochError,
)
from gateway.utils import subnet_epoch_archive as archive


def test_archive_client_is_pinned_and_cached(monkeypatch):
    created = []
    source = SimpleNamespace(
        chain_endpoint=OFFICIAL_BITTENSOR_ARCHIVE_ENDPOINT,
        substrate=SimpleNamespace(url=OFFICIAL_BITTENSOR_ARCHIVE_ENDPOINT),
    )

    def construct(*, network):
        created.append(network)
        return source

    monkeypatch.setattr(archive, "_archive_subtensor", None)
    monkeypatch.setitem(
        sys.modules,
        "bittensor",
        SimpleNamespace(Subtensor=construct),
    )

    assert archive.get_official_archive_subtensor() is source
    assert archive.get_official_archive_subtensor() is source
    assert created == [OFFICIAL_BITTENSOR_ARCHIVE_ENDPOINT]


def test_exact_historical_read_rejects_lite_source(monkeypatch):
    lite = SimpleNamespace(
        chain_endpoint="wss://entrypoint-finney.opentensor.ai:443",
        substrate=SimpleNamespace(
            url="wss://entrypoint-finney.opentensor.ai:443"
        ),
    )
    monkeypatch.setattr(
        archive,
        "read_subnet_epoch_snapshot",
        lambda *_args, **_kwargs: pytest.fail(
            "historical read reached the lite client"
        ),
    )

    with pytest.raises(SubnetEpochError, match="official Bittensor archive"):
        archive.read_exact_subnet_epoch_snapshot_from_archive(
            netuid=71,
            block_number=8_637_156,
            archive_subtensor=lite,
        )


def test_exact_historical_read_uses_explicit_archive_source(monkeypatch):
    source = SimpleNamespace(
        chain_endpoint=OFFICIAL_BITTENSOR_ARCHIVE_ENDPOINT,
        substrate=SimpleNamespace(url=OFFICIAL_BITTENSOR_ARCHIVE_ENDPOINT),
    )
    observed = []
    sentinel = object()

    def read(subtensor, **kwargs):
        observed.append((subtensor, kwargs))
        return sentinel

    monkeypatch.setattr(archive, "read_subnet_epoch_snapshot", read)

    assert (
        archive.read_exact_subnet_epoch_snapshot_from_archive(
            netuid=71,
            block_hash="0x" + "1" * 64,
            archive_subtensor=source,
        )
        is sentinel
    )
    assert observed == [
        (
            source,
            {
                "netuid": 71,
                "block_number": None,
                "block_hash": "0x" + "1" * 64,
            },
        )
    ]
