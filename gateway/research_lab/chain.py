"""Research Lab chain helpers used by gateway workers and reports."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from typing import Any, Iterable


logger = logging.getLogger(__name__)
_EPOCH_CACHE: tuple[int, int | None, str, float] | None = None
_EPOCH_HINT_ENV = "RESEARCH_LAB_GATEWAY_EPOCH_HINT"
_EPOCH_BLOCK_HINT_ENV = "RESEARCH_LAB_GATEWAY_BLOCK_HINT"
_EPOCH_HINT_TS_ENV = "RESEARCH_LAB_GATEWAY_EPOCH_HINT_TS"
_EPOCH_HINT_MAX_AGE_SECONDS_ENV = "RESEARCH_LAB_GATEWAY_EPOCH_HINT_MAX_AGE_SECONDS"
_DIRECT_EPOCH_TIMEOUT_SECONDS_ENV = "RESEARCH_LAB_DIRECT_EPOCH_TIMEOUT_SECONDS"
_DEFAULT_EPOCH_HINT_MAX_AGE_SECONDS = 2 * 60 * 60
_DEFAULT_DIRECT_EPOCH_TIMEOUT_SECONDS = 20.0
_DIRECT_EPOCH_RESULT_PREFIX = "LEADPOET_EPOCH_RESULT="


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

        timeout_seconds = _direct_epoch_timeout_seconds()
        epoch = int(await asyncio.wait_for(get_current_epoch_id_async(), timeout=timeout_seconds))
        block = None
        source = "gateway_epoch_utils"
    except Exception as exc:
        hint = _read_gateway_epoch_hint(require_fresh=True)
        if hint is not None:
            epoch, block, source = hint
        else:
            logger.warning("research_lab_epoch_gateway_utils_failed_fallback_hint: %s", str(exc)[:200])
            try:
                epoch, block, network = await asyncio.wait_for(
                    asyncio.to_thread(_fetch_current_chain_epoch_direct),
                    timeout=_direct_epoch_timeout_seconds(),
                )
                source = f"direct_subtensor:{network}"
            except Exception as direct_exc:
                stale_hint = _read_gateway_epoch_hint(require_fresh=False)
                if stale_hint is not None:
                    logger.warning(
                        "research_lab_epoch_direct_failed_using_stale_gateway_hint: %s",
                        str(direct_exc)[:200],
                    )
                    epoch, block, source = stale_hint
                    source = f"{source}:stale_fallback"
                else:
                    raise RuntimeError(
                        "Research Lab evaluation epoch could not be resolved from gateway utils, "
                        "gateway epoch hint, or direct subtensor"
                    ) from direct_exc

    if epoch <= 0:
        raise RuntimeError("Research Lab evaluation epoch resolved to 0")
    _EPOCH_CACHE = (epoch, block, source, now)
    return epoch, block, source


def _read_gateway_epoch_hint(*, require_fresh: bool) -> tuple[int, int | None, str] | None:
    try:
        epoch = int(str(os.getenv(_EPOCH_HINT_ENV, "0")).strip() or "0")
    except (TypeError, ValueError):
        return None
    if epoch <= 0:
        return None
    block = _optional_int_env(_EPOCH_BLOCK_HINT_ENV)
    hinted_at = _optional_float_env(_EPOCH_HINT_TS_ENV)
    if require_fresh:
        if hinted_at is None:
            return None
        if time.time() - hinted_at > _epoch_hint_max_age_seconds():
            return None
    return epoch, block, "gateway_epoch_hint"


def _epoch_hint_max_age_seconds() -> float:
    try:
        return max(0.0, float(os.getenv(_EPOCH_HINT_MAX_AGE_SECONDS_ENV, _DEFAULT_EPOCH_HINT_MAX_AGE_SECONDS)))
    except (TypeError, ValueError):
        return float(_DEFAULT_EPOCH_HINT_MAX_AGE_SECONDS)


def _direct_epoch_timeout_seconds() -> float:
    try:
        return max(1.0, float(os.getenv(_DIRECT_EPOCH_TIMEOUT_SECONDS_ENV, _DEFAULT_DIRECT_EPOCH_TIMEOUT_SECONDS)))
    except (TypeError, ValueError):
        return float(_DEFAULT_DIRECT_EPOCH_TIMEOUT_SECONDS)


def _optional_int_env(name: str) -> int | None:
    try:
        value = str(os.getenv(name, "")).strip()
        return int(value) if value else None
    except (TypeError, ValueError):
        return None


def _optional_float_env(name: str) -> float | None:
    try:
        value = str(os.getenv(name, "")).strip()
        return float(value) if value else None
    except (TypeError, ValueError):
        return None


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

network = os.getenv("BITTENSOR_NETWORK", "finney")
subtensor = bt.Subtensor(network=network)
try:
    block = int(subtensor.block)
finally:
    close = getattr(subtensor, "close", None)
    if callable(close):
        close()
print(%r + json.dumps({"block": block, "network": network}, separators=(",", ":")))
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
        block = int(result["block"])
        result_network = str(result["network"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError("direct subtensor epoch probe returned invalid output") from exc
    if block <= 0 or result_network != network:
        raise RuntimeError("direct subtensor epoch probe returned inconsistent output")
    return block // 360, block, network


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
