"""Research Lab chain helpers used by gateway workers and reports."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, Iterable


logger = logging.getLogger(__name__)
_EPOCH_CACHE: tuple[int, int | None, str, float] | None = None


async def resolve_research_lab_evaluation_epoch(configured_epoch: int | str | None = None) -> tuple[int, int | None, str]:
    """Resolve the live Bittensor epoch without requiring an operator override."""
    try:
        configured = int(configured_epoch or 0)
    except (TypeError, ValueError):
        configured = 0
    if configured > 0:
        return configured, None, "configured"

    global _EPOCH_CACHE
    now = time.monotonic()
    if _EPOCH_CACHE is not None:
        epoch, block, source, cached_at = _EPOCH_CACHE
        if now - cached_at <= 60.0 and epoch > 0:
            return epoch, block, source

    try:
        from gateway.utils.epoch import get_current_epoch_id_async

        epoch = int(await get_current_epoch_id_async())
        block = None
        source = "gateway_epoch_utils"
    except Exception as exc:
        logger.warning("research_lab_epoch_gateway_utils_failed_fallback_direct: %s", str(exc)[:200])
        epoch, block, network = await asyncio.to_thread(_fetch_current_chain_epoch_direct)
        source = f"direct_subtensor:{network}"

    if epoch <= 0:
        raise RuntimeError("Research Lab evaluation epoch resolved to 0")
    _EPOCH_CACHE = (epoch, block, source, now)
    return epoch, block, source


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
    import bittensor as bt

    network = os.getenv("BITTENSOR_NETWORK", "finney")
    proxy_keys = ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy")
    saved_proxy_env = {key: os.environ.pop(key) for key in proxy_keys if key in os.environ}
    try:
        subtensor = bt.subtensor(network=network)
        try:
            block = int(subtensor.block)
        finally:
            close = getattr(subtensor, "close", None)
            if callable(close):
                close()
    finally:
        os.environ.update(saved_proxy_env)
    return block // 360, block, network


def _fetch_metagraph_direct() -> Any:
    import bittensor as bt

    network = os.getenv("BITTENSOR_NETWORK", "finney")
    netuid = int(os.getenv("BITTENSOR_NETUID", "71"))
    proxy_keys = ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy")
    saved_proxy_env = {key: os.environ.pop(key) for key in proxy_keys if key in os.environ}
    try:
        subtensor = bt.subtensor(network=network)
        try:
            return subtensor.metagraph(netuid=netuid)
        finally:
            close = getattr(subtensor, "close", None)
            if callable(close):
                close()
    finally:
        os.environ.update(saved_proxy_env)
