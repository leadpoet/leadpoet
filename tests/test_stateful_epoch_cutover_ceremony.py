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
        (_live(elapsed=300), True),
        (_live(elapsed=301), False),
        (_live(elapsed=0, next_epoch=True), False),
    ),
)
def test_first_epoch_restart_window_uses_the_block_300_deadline(
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
    assert status["latest_safe_epoch_block"] == 300
    assert status["restart_start_epoch_block"] == live.epoch_block
    assert status["restart_start_captured"] is False


def test_captured_start_before_300_remains_valid_after_300() -> None:
    cutover = _cutover()
    status = cutover_cli._live_initialization_window_status(
        cutover=cutover,
        snapshot_doc=_boundary().to_dict(cutover=cutover),
        restart_start=_live(elapsed=250),
        live=_live(elapsed=330),
    )

    assert status["eligible"] is True
    assert status["restart_start_epoch_block"] == 250
    assert status["live_epoch_block"] == 330
    assert status["deadline_reapplied"] is False


def test_captured_start_does_not_authorize_the_next_epoch() -> None:
    cutover = _cutover()
    status = cutover_cli._live_initialization_window_status(
        cutover=cutover,
        snapshot_doc=_boundary().to_dict(cutover=cutover),
        restart_start=_live(elapsed=250),
        live=_live(elapsed=0, next_epoch=True),
    )

    assert status["eligible"] is False


@pytest.mark.asyncio
async def test_staged_activation_reuses_the_captured_restart_start(monkeypatch):
    cutover = _cutover()
    authority_hash = "sha256:" + "a" * 64
    receipt_hash = "sha256:" + "b" * 64
    captured = _live(elapsed=250)
    observed = {}
    state_reads = iter(("stateful_staged", "stateful_active"))
    cutover_row = {
        "cutover_authority_hash": authority_hash,
        "cutover_receipt_hash": receipt_hash,
        "first_snapshot_doc": _boundary().to_dict(cutover=cutover),
    }
    initialization = {"nonce": "nonce", "payload_hash": "sha256:" + "c" * 64}

    async def select_exactly_one(*_args, **_kwargs):
        return dict(cutover_row)

    def window_status(*, restart_start, **_kwargs):
        observed["restart_start"] = restart_start
        return {"eligible": True}

    def assert_state(*_args, **_kwargs):
        return {"lifecycle_state": next(state_reads)}

    async def select_rows(*_args, **_kwargs):
        return [{}]

    async def activate_rpc(_name, _params):
        return [
            {
                "lifecycle_state": "stateful_active",
                "mapping_hash": cutover.mapping_hash,
                "cutover_authority_hash": authority_hash,
                "cutover_receipt_hash": receipt_hash,
                "initialization_nonce": initialization["nonce"],
                "initialization_payload_hash": initialization["payload_hash"],
            }
        ]

    monkeypatch.setattr(cutover_cli, "_select_exactly_one", select_exactly_one)
    monkeypatch.setattr(
        cutover_cli,
        "_assert_existing_cutover",
        lambda value, **_kwargs: value,
    )
    monkeypatch.setattr(
        cutover_cli,
        "validate_receipt_graph",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        cutover_cli,
        "_validate_initialization",
        lambda *_args, **_kwargs: dict(initialization),
    )
    monkeypatch.setattr(cutover_cli, "_assert_cutover_state", assert_state)
    monkeypatch.setattr(
        cutover_cli,
        "_configured_restart_start",
        lambda _cutover: captured,
    )
    monkeypatch.setattr(
        cutover_cli,
        "_live_initialization_window_status",
        window_status,
    )

    report = await cutover_cli.activate_staged_subnet_epoch_cutover_v1(
        cutover=cutover,
        confirmed_cutover_authority_hash=authority_hash,
        select_rows=select_rows,
        load_graph=lambda _root: _async_value({}),
        load_initialization=lambda _epoch: _async_value(initialization),
        load_live_snapshot=lambda _cutover: _async_value(_live(elapsed=330)),
        activate_rpc=activate_rpc,
        boot_verifier=lambda _identity: {"verified": True},
        validate_anchor=lambda _cutover: _async_value(None),
    )

    assert report["status"] == "stateful_active"
    assert observed["restart_start"] == captured


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


@pytest.mark.asyncio
async def test_offline_cutover_injects_gateway_chain_dependencies(monkeypatch):
    events = []
    chain_client = object()

    class FakeAsyncSubtensor:
        def __init__(self, *, network):
            events.append(("created", network))

        async def __aenter__(self):
            events.append(("entered", chain_client))
            return chain_client

        async def __aexit__(self, exc_type, exc, traceback):
            events.append(("exited", exc_type))

    monkeypatch.setattr("bittensor.AsyncSubtensor", FakeAsyncSubtensor)
    monkeypatch.setattr("gateway.config.BITTENSOR_NETWORK", "finney")
    monkeypatch.setattr(
        "gateway.utils.epoch.inject_async_subtensor",
        lambda value: events.append(("epoch", value)),
    )
    monkeypatch.setattr(
        "gateway.utils.registry.inject_async_subtensor",
        lambda value: events.append(("registry", value)),
    )

    async def operation(*, expected):
        events.append(("operation", expected))
        return {"status": "ok"}

    result = await cutover_cli._run_with_gateway_chain_dependencies(
        operation,
        expected="value",
    )

    assert result == {"status": "ok"}
    assert events == [
        ("created", "finney"),
        ("entered", chain_client),
        ("epoch", chain_client),
        ("registry", chain_client),
        ("operation", "value"),
        ("exited", None),
    ]
