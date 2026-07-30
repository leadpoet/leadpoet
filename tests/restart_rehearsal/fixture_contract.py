"""Validated access to the candidate-owned restart rehearsal fixture."""

from __future__ import annotations

import json
from pathlib import Path


FIXTURE_PATH = Path(
    "tests/restart_rehearsal/fixtures/production_shaped_v2.json"
)


def _load_rehearsal_fixture(
    source_root: Path,
) -> dict[str, object]:
    """Return the candidate-declared exact-launcher fixture."""

    fixture_path = Path(source_root) / FIXTURE_PATH
    try:
        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(
            "candidate rehearsal fixture is unavailable or invalid"
        ) from exc
    if not isinstance(fixture, dict) or fixture.get("schema_version") != (
        "leadpoet.restart_rehearsal_fixture.v1"
    ):
        raise ValueError("candidate rehearsal fixture schema differs")
    return fixture


def _load_rehearsal_metagraph(
    source_root: Path,
) -> dict[str, object]:
    """Return the candidate-declared exact-launcher metagraph contract."""

    fixture = _load_rehearsal_fixture(source_root)
    metagraph = fixture.get("metagraph")
    if not isinstance(metagraph, dict):
        raise ValueError("candidate rehearsal metagraph differs")
    return metagraph


def load_rehearsal_current_settlement_epoch_id(
    source_root: Path,
) -> int:
    """Resolve the fixture's current settlement epoch from candidate policy."""

    root = Path(source_root)
    fixture = _load_rehearsal_fixture(root)
    network = fixture.get("network")
    if not isinstance(network, dict):
        raise ValueError("candidate rehearsal network differs")
    try:
        cutover = json.loads(
            (
                root / "config" / "stateful-epoch-cutover-sn71.json"
            ).read_text(encoding="utf-8")
        )
        netuid = int(network["netuid"])
        subnet_epoch_index = int(network["subnet_epoch_index"])
        current_block = int(network["current_block"])
        first_subnet_epoch = int(cutover["first_subnet_epoch_index"])
        first_settlement_epoch = int(cutover["first_settlement_epoch_id"])
        cutover_block = int(cutover["cutover_block"])
        cutover_netuid = int(cutover["netuid"])
    except (KeyError, OSError, TypeError, ValueError) as exc:
        raise ValueError(
            "candidate rehearsal settlement policy differs"
        ) from exc
    if (
        netuid != cutover_netuid
        or subnet_epoch_index < first_subnet_epoch
        or current_block <= cutover_block
    ):
        raise ValueError(
            "candidate rehearsal settlement fixture predates cutover"
        )
    return first_settlement_epoch + (
        subnet_epoch_index - first_subnet_epoch
    )


def load_rehearsal_metagraph_hotkeys(
    source_root: Path,
) -> tuple[str, ...]:
    """Return the ordered runtime metagraph identities declared by the candidate."""

    metagraph = _load_rehearsal_metagraph(source_root)
    hotkeys = metagraph.get("runtime_hotkeys")
    if not isinstance(hotkeys, list):
        raise ValueError("candidate rehearsal metagraph hotkeys differ")
    normalized = tuple(
        value.strip() if isinstance(value, str) else ""
        for value in hotkeys
    )
    if (
        len(normalized) != 4
        or any(not value for value in normalized)
        or len(set(normalized)) != len(normalized)
    ):
        raise ValueError("candidate rehearsal metagraph hotkeys differ")
    return normalized


def load_rehearsal_metagraph_account_ids(
    source_root: Path,
) -> tuple[bytes, ...]:
    """Return account IDs used to encode the authenticated chain response."""

    metagraph = _load_rehearsal_metagraph(source_root)
    values = metagraph.get("runtime_account_ids_hex")
    if not isinstance(values, list):
        raise ValueError("candidate rehearsal metagraph account IDs differ")
    try:
        account_ids = tuple(
            bytes.fromhex(value) if isinstance(value, str) else b""
            for value in values
        )
    except ValueError as exc:
        raise ValueError(
            "candidate rehearsal metagraph account IDs differ"
        ) from exc
    if (
        len(account_ids) != 4
        or any(len(value) != 32 for value in account_ids)
        or len(set(account_ids)) != len(account_ids)
    ):
        raise ValueError("candidate rehearsal metagraph account IDs differ")
    return account_ids
