from __future__ import annotations

import pytest

from Leadpoet.utils.subnet_epoch import SubnetEpochCutover, SubnetEpochSnapshot
from gateway.research_lab import stateful_epoch_cutover_cli_v1 as cutover_cli


GENESIS = "0x" + "11" * 32
BOUNDARY_HASH = "0x" + "22" * 32


def _boundary(*, block: int = 36_396) -> SubnetEpochSnapshot:
    return SubnetEpochSnapshot(
        network_genesis_hash=GENESIS,
        netuid=71,
        head_kind="exact",
        block_hash=BOUNDARY_HASH,
        current_block=block,
        last_epoch_block=block,
        pending_epoch_at=0,
        subnet_epoch_index=23_928,
        tempo=360,
        blocks_since_last_step=0,
        observed_at="2026-07-16T12:00:00Z",
    )


def _cutover() -> SubnetEpochCutover:
    boundary = _boundary()
    return SubnetEpochCutover(
        network_genesis_hash=GENESIS,
        netuid=71,
        cutover_block=boundary.current_block,
        cutover_block_hash=boundary.block_hash,
        first_subnet_epoch_index=boundary.subnet_epoch_index,
        first_settlement_epoch_id=101,
        last_legacy_epoch_id=100,
    )


def _live(*, elapsed: int, next_epoch: bool = False) -> SubnetEpochSnapshot:
    boundary = _boundary()
    anchor = boundary.current_block + (360 if next_epoch else 0)
    index = boundary.subnet_epoch_index + (1 if next_epoch else 0)
    return SubnetEpochSnapshot(
        network_genesis_hash=GENESIS,
        netuid=71,
        head_kind="finalized",
        block_hash="0x" + f"{anchor + elapsed:064x}"[-64:],
        current_block=anchor + elapsed,
        last_epoch_block=anchor,
        pending_epoch_at=0,
        subnet_epoch_index=index,
        tempo=360,
        blocks_since_last_step=elapsed,
        observed_at="2026-07-16T13:00:00Z",
    )


@pytest.mark.asyncio
async def test_manifest_proposal_is_archive_first_and_candidate_independent():
    selected = []

    async def select_rows(table, **_kwargs):
        selected.append(table)
        if table != cutover_cli.CUTOVER_STATE_TABLE:
            raise AssertionError("manifest proposal read candidate state")
        return [
            {
                "singleton": True,
                "lifecycle_state": "cutover_fenced",
                "network_genesis_hash": GENESIS,
                "netuid": 71,
                "last_legacy_epoch_id": 100,
                "first_settlement_epoch_id": 101,
            }
        ]

    observed = []

    async def validate_anchor(cutover):
        observed.append(cutover)

    report = await cutover_cli.propose_subnet_epoch_cutover_manifest_v1(
        network_genesis_hash=GENESIS,
        netuid=71,
        last_legacy_epoch_id=100,
        select_rows=select_rows,
        load_boundary=lambda _netuid: _async_value(_boundary()),
        validate_anchor=validate_anchor,
    )

    manifest = SubnetEpochCutover.from_mapping(report["manifest"])
    assert selected == [cutover_cli.CUTOVER_STATE_TABLE]
    assert manifest.cutover_block == 36_396
    assert manifest.first_settlement_epoch_id == 101
    assert report["status"] == "manifest_proposed"
    assert report["boundary_snapshot_hash"].startswith("sha256:")
    assert observed == [manifest]


@pytest.mark.asyncio
async def test_manifest_proposal_uses_official_boundary_not_global_bucket():
    async def select_rows(_table, **_kwargs):
        return [
            {
                "lifecycle_state": "cutover_fenced",
                "network_genesis_hash": GENESIS,
                "netuid": 71,
                "last_legacy_epoch_id": 100,
                "first_settlement_epoch_id": 101,
            }
        ]

    report = await cutover_cli.propose_subnet_epoch_cutover_manifest_v1(
        network_genesis_hash=GENESIS,
        netuid=71,
        last_legacy_epoch_id=100,
        select_rows=select_rows,
        load_boundary=lambda _netuid: _async_value(_boundary(block=36_000)),
        validate_anchor=lambda _cutover: _async_value(None),
    )

    manifest = SubnetEpochCutover.from_mapping(report["manifest"])
    assert manifest.cutover_block == 36_000
    assert manifest.first_settlement_epoch_id == 101


@pytest.mark.asyncio
async def test_archive_boundary_discovery_rereads_exact_anchor(monkeypatch):
    current = SubnetEpochSnapshot(
        network_genesis_hash=GENESIS,
        netuid=71,
        head_kind="finalized",
        block_hash="0x" + "33" * 32,
        current_block=36_400,
        last_epoch_block=36_396,
        pending_epoch_at=0,
        subnet_epoch_index=23_928,
        tempo=360,
        blocks_since_last_step=4,
        observed_at="2026-07-16T12:00:48Z",
    )
    calls = []
    monkeypatch.setattr(
        cutover_cli,
        "read_finalized_subnet_epoch_snapshot_from_archive",
        lambda **kwargs: calls.append(("finalized", kwargs)) or current,
    )
    monkeypatch.setattr(
        cutover_cli,
        "read_exact_subnet_epoch_snapshot_from_archive",
        lambda **kwargs: calls.append(("exact", kwargs)) or _boundary(),
    )

    result = await cutover_cli._load_official_archive_boundary(71)

    assert result == _boundary()
    assert calls == [
        ("finalized", {"netuid": 71}),
        ("exact", {"netuid": 71, "block_number": 36_396}),
    ]


@pytest.mark.parametrize(
    ("live", "eligible"),
    (
        (_live(elapsed=299), True),
        (_live(elapsed=300), False),
        (_live(elapsed=0, next_epoch=True), False),
    ),
)
def test_first_epoch_restart_window_is_wide_but_still_pre_block_300(
    live,
    eligible,
):
    cutover = _cutover()
    status = cutover_cli._live_initialization_window_status(
        cutover=cutover,
        snapshot_doc=_boundary().to_dict(cutover=cutover),
        live=live,
    )

    assert status["eligible"] is eligible
    assert status["latest_safe_epoch_block"] == 299
    assert status["restart_reserve_blocks"] == 60


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ("prepare", "activate"))
async def test_activation_validates_archive_before_database_reads(operation):
    selected = False

    async def select_rows(*_args, **_kwargs):
        nonlocal selected
        selected = True
        raise AssertionError("database read occurred before archive validation")

    async def unavailable(_cutover):
        raise RuntimeError("archive unavailable")

    with pytest.raises(
        cutover_cli.StatefulEpochCutoverActivationError,
        match="official archive rejected",
    ):
        if operation == "prepare":
            await cutover_cli.activate_subnet_epoch_cutover_v1(
                cutover=_cutover(),
                select_rows=select_rows,
                validate_anchor=unavailable,
            )
        else:
            await cutover_cli.activate_staged_subnet_epoch_cutover_v1(
                cutover=_cutover(),
                confirmed_cutover_authority_hash="sha256:" + "a" * 64,
                select_rows=select_rows,
                validate_anchor=unavailable,
            )
    assert selected is False


async def _async_value(value):
    return value
