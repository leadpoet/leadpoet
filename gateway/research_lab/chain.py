"""Research Lab chain helpers used by gateway workers and reports."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import sys
from typing import Any, Iterable

logger = logging.getLogger(__name__)
_DIRECT_EPOCH_TIMEOUT_SECONDS_ENV = "RESEARCH_LAB_DIRECT_EPOCH_TIMEOUT_SECONDS"
_DEFAULT_DIRECT_EPOCH_TIMEOUT_SECONDS = 20.0
_DIRECT_EPOCH_RESULT_PREFIX = "LEADPOET_EPOCH_RESULT="


async def resolve_research_lab_evaluation_epoch(configured_epoch: int | str | None = None) -> tuple[int, int | None, str]:
    """Resolve the live Bittensor epoch without requiring an operator override."""
    from gateway.utils.epoch import (
        get_current_epoch_context_async,
    )

    try:
        configured = int(configured_epoch or 0)
    except (TypeError, ValueError):
        configured = 0
    if configured > 0:
        raise RuntimeError(
            "configured Research Lab epoch overrides are forbidden"
        )

    try:
        timeout_seconds = _direct_epoch_timeout_seconds()
        snapshot, epoch = await asyncio.wait_for(
            get_current_epoch_context_async(finalized=True),
            timeout=timeout_seconds,
        )
        block = snapshot.current_block
        source = "gateway_epoch_utils:finalized"
    except Exception as exc:
        logger.warning(
            "research_lab_epoch_gateway_utils_failed_direct_probe: %s",
            str(exc)[:200],
        )
        try:
            epoch, block, network = await asyncio.wait_for(
                asyncio.to_thread(_fetch_current_chain_epoch_direct),
                timeout=_direct_epoch_timeout_seconds(),
            )
            source = f"direct_subtensor_official:{network}"
        except Exception as direct_exc:
            raise RuntimeError(
                "Research Lab evaluation epoch could not be resolved from "
                "an exact-hash official SN71 snapshot"
            ) from direct_exc

    if epoch <= 0:
        raise RuntimeError("Research Lab evaluation epoch resolved to 0")
    return epoch, block, source


def _direct_epoch_timeout_seconds() -> float:
    try:
        return max(1.0, float(os.getenv(_DIRECT_EPOCH_TIMEOUT_SECONDS_ENV, _DEFAULT_DIRECT_EPOCH_TIMEOUT_SECONDS)))
    except (TypeError, ValueError):
        return float(_DEFAULT_DIRECT_EPOCH_TIMEOUT_SECONDS)


async def resolve_hotkey_uids(hotkeys: Iterable[str]) -> dict[str, int]:
    """Resolve registered hotkeys to current subnet UIDs using one metagraph read."""
    unique_hotkeys = {str(hotkey) for hotkey in hotkeys if str(hotkey or "").strip()}
    if not unique_hotkeys:
        return {}
    metagraph = await _get_metagraph()
    resolved: dict[str, int] = {}
    for uid, hotkey in enumerate(getattr(metagraph, "hotkeys", []) or []):
        if hotkey in unique_hotkeys:
            resolved[str(hotkey)] = int(uid)
    return resolved


async def _get_metagraph() -> Any:
    try:
        from gateway.utils.registry import get_metagraph_async

        return await get_metagraph_async()
    except Exception as exc:
        logger.warning("research_lab_metagraph_gateway_registry_failed_fallback_direct: %s", str(exc)[:200])
        return await asyncio.to_thread(_fetch_metagraph_direct)


def _fetch_current_chain_epoch_direct() -> tuple[int, int, str]:
    network = os.getenv("BITTENSOR_NETWORK", "finney")
    proxy_keys = ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy")
    child_env = {key: value for key, value in os.environ.items() if key not in proxy_keys}
    probe = """
import json
import os
import bittensor as bt
from Leadpoet.utils.subnet_epoch import (
    load_subnet_epoch_cutover,
    read_subnet_epoch_snapshot,
)
from gateway.utils.epoch import validate_stateful_cutover_authority
from gateway.utils.subnet_epoch_archive import (
    validate_cutover_anchor_from_archive,
)

network = os.getenv("BITTENSOR_NETWORK", "finney")
netuid = int(os.getenv("BITTENSOR_NETUID", "71"))
subtensor = bt.Subtensor(network=network)
try:
    snapshot = read_subnet_epoch_snapshot(
        subtensor,
        netuid=netuid,
        finalized=True,
    )
    cutover = load_subnet_epoch_cutover()
    epoch = snapshot.settlement_epoch_id(cutover)
    block = snapshot.current_block
    validate_stateful_cutover_authority(cutover)
    validate_cutover_anchor_from_archive(cutover)
    official = {
        "official_subnet_epoch_id": snapshot.subnet_epoch_index,
        "epoch_ref": snapshot.epoch_ref,
    }
finally:
    close = getattr(subtensor, "close", None)
    if callable(close):
        close()
result = {
    "epoch": epoch,
    "block": block,
    "network": network,
}
result.update(official)
print(%r + json.dumps(result, separators=(",", ":")))
""" % _DIRECT_EPOCH_RESULT_PREFIX
    timeout_seconds = max(1.0, _direct_epoch_timeout_seconds() - 1.0)
    try:
        completed = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            check=False,
            env=child_env,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"direct subtensor epoch probe timed out after {timeout_seconds:.1f}s"
        ) from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip().splitlines()
        raise RuntimeError(
            "direct subtensor epoch probe failed: "
            + (detail[-1][:200] if detail else f"exit {completed.returncode}")
        )
    result_line = next(
        (
            line[len(_DIRECT_EPOCH_RESULT_PREFIX) :]
            for line in reversed(completed.stdout.splitlines())
            if line.startswith(_DIRECT_EPOCH_RESULT_PREFIX)
        ),
        "",
    )
    try:
        result = json.loads(result_line)
        epoch = int(result["epoch"])
        block = int(result["block"])
        result_network = str(result["network"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError("direct subtensor epoch probe returned invalid output") from exc
    if block <= 0 or result_network != network:
        raise RuntimeError("direct subtensor epoch probe returned inconsistent output")
    if epoch <= 0:
        raise RuntimeError("direct subtensor epoch probe returned an invalid epoch")
    return epoch, block, network


def _fetch_metagraph_direct() -> Any:
    import bittensor as bt

    network = os.getenv("BITTENSOR_NETWORK", "finney")
    netuid = int(os.getenv("BITTENSOR_NETUID", "71"))
    proxy_keys = ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy")
    saved_proxy_env = {key: os.environ.pop(key) for key in proxy_keys if key in os.environ}
    try:
        subtensor = bt.Subtensor(network=network)
        try:
            return subtensor.metagraph(netuid=netuid)
        finally:
            close = getattr(subtensor, "close", None)
            if callable(close):
                close()
    finally:
        os.environ.update(saved_proxy_env)
