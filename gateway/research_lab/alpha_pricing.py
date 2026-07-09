"""Research Lab live alpha/USD valuation for reimbursement allocation."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
import logging
from typing import Any

import aiohttp


logger = logging.getLogger(__name__)

VALUATION_QUANT = Decimal("0.000000001")
USD_PER_PERCENT_QUANT = Decimal("0.000001")
COINGECKO_TAO_USD_URL = "https://api.coingecko.com/api/v3/simple/price"


@dataclass(frozen=True)
class AlphaPriceValuation:
    """Epoch-scoped valuation used to convert USD reimbursements to alpha share."""

    network: str
    netuid: int
    epoch: int
    pricing_status: str
    tao_per_alpha: Decimal
    tao_usd: Decimal
    alpha_usd: Decimal
    miner_alpha_per_epoch: Decimal
    usd_per_0_1_percent_epoch: Decimal
    fetched_at: str
    sources: tuple[str, ...]
    fallback_reason: str = ""

    def policy_doc(self) -> dict[str, Any]:
        doc: dict[str, Any] = {
            "pricing_status": self.pricing_status,
            "network": self.network,
            "netuid": self.netuid,
            "epoch": self.epoch,
            "tao_per_alpha": _money_float(self.tao_per_alpha),
            "tao_usd": _money_float(self.tao_usd),
            "alpha_usd": _money_float(self.alpha_usd),
            "miner_alpha_per_epoch": _money_float(self.miner_alpha_per_epoch),
            "usd_per_0_1_percent_epoch": _money_float(self.usd_per_0_1_percent_epoch),
            "sources": list(self.sources),
            "fetched_at": self.fetched_at,
        }
        if self.fallback_reason:
            doc["fallback_reason"] = self.fallback_reason
        return doc


_CACHE: dict[tuple[str, int, int], AlphaPriceValuation] = {}
_CACHE_LOCK = asyncio.Lock()


def compute_alpha_price_valuation(
    *,
    network: str,
    netuid: int,
    epoch: int,
    tao_per_alpha: Any,
    tao_usd: Any,
    miner_alpha_per_epoch: Any,
    pricing_status: str = "live",
    fetched_at: str | None = None,
    sources: tuple[str, ...] = ("bittensor.subnet.price", "coingecko.simple.price"),
    fallback_reason: str = "",
) -> AlphaPriceValuation:
    """Build the deterministic valuation document from observed prices."""

    tao_per_alpha_dec = _decimal(tao_per_alpha)
    tao_usd_dec = _decimal(tao_usd)
    miner_alpha_dec = _decimal(miner_alpha_per_epoch)
    if tao_per_alpha_dec < 0 or tao_usd_dec < 0 or miner_alpha_dec <= 0:
        raise ValueError("alpha valuation inputs must be non-negative and miner_alpha_per_epoch must be positive")

    alpha_usd = (tao_per_alpha_dec * tao_usd_dec).quantize(VALUATION_QUANT, rounding=ROUND_HALF_UP)
    usd_per_epoch = (alpha_usd * miner_alpha_dec * Decimal("0.001")).quantize(
        USD_PER_PERCENT_QUANT,
        rounding=ROUND_HALF_UP,
    )
    return AlphaPriceValuation(
        network=str(network or "finney"),
        netuid=int(netuid),
        epoch=int(epoch),
        pricing_status=str(pricing_status or "live"),
        tao_per_alpha=tao_per_alpha_dec,
        tao_usd=tao_usd_dec,
        alpha_usd=alpha_usd,
        miner_alpha_per_epoch=miner_alpha_dec,
        usd_per_0_1_percent_epoch=usd_per_epoch,
        fetched_at=fetched_at or _utc_now_iso(),
        sources=tuple(sources),
        fallback_reason=str(fallback_reason or ""),
    )


def static_alpha_price_fallback(
    *,
    network: str,
    netuid: int,
    epoch: int,
    static_usd_per_0_1_percent_epoch: Any,
    miner_alpha_per_epoch: Any,
    reason: str,
) -> AlphaPriceValuation:
    """Return an auditable fallback valuation using the configured static conversion."""

    usd_per_epoch = _decimal(static_usd_per_0_1_percent_epoch).quantize(
        USD_PER_PERCENT_QUANT,
        rounding=ROUND_HALF_UP,
    )
    miner_alpha_dec = _decimal(miner_alpha_per_epoch)
    alpha_usd = Decimal("0")
    if miner_alpha_dec > 0:
        alpha_usd = (usd_per_epoch / (miner_alpha_dec * Decimal("0.001"))).quantize(
            VALUATION_QUANT,
            rounding=ROUND_HALF_UP,
        )
    return AlphaPriceValuation(
        network=str(network or "finney"),
        netuid=int(netuid),
        epoch=int(epoch),
        pricing_status="static_fallback",
        tao_per_alpha=Decimal("0"),
        tao_usd=Decimal("0"),
        alpha_usd=alpha_usd,
        miner_alpha_per_epoch=miner_alpha_dec,
        usd_per_0_1_percent_epoch=usd_per_epoch,
        fetched_at=_utc_now_iso(),
        sources=("config.RESEARCH_LAB_REIMBURSEMENT_USD_PER_0_1_PERCENT_EPOCH",),
        fallback_reason=str(reason or "live_price_unavailable"),
    )


async def resolve_epoch_alpha_price_valuation(
    *,
    network: str,
    netuid: int,
    epoch: int,
    enabled: bool,
    require_live: bool,
    miner_alpha_per_epoch: Any,
    static_usd_per_0_1_percent_epoch: Any,
    timeout_seconds: float = 8.0,
    max_attempts: int = 3,
) -> AlphaPriceValuation:
    """Fetch or reuse the epoch valuation for Research Lab reimbursements."""

    cache_key = (str(network or "finney"), int(netuid), int(epoch))
    if not enabled:
        return static_alpha_price_fallback(
            network=network,
            netuid=netuid,
            epoch=epoch,
            static_usd_per_0_1_percent_epoch=static_usd_per_0_1_percent_epoch,
            miner_alpha_per_epoch=miner_alpha_per_epoch,
            reason="dynamic_alpha_price_disabled",
        )

    cached = _CACHE.get(cache_key)
    if cached is not None:
        return _with_status(cached, "cache_hit")

    async with _CACHE_LOCK:
        cached = _CACHE.get(cache_key)
        if cached is not None:
            return _with_status(cached, "cache_hit")

        last_error = ""
        for attempt in range(1, max(1, int(max_attempts)) + 1):
            try:
                tao_per_alpha, tao_usd = await asyncio.gather(
                    _fetch_tao_per_alpha(
                        network=str(network or "finney"),
                        netuid=int(netuid),
                        timeout_seconds=timeout_seconds,
                    ),
                    _fetch_tao_usd(timeout_seconds=timeout_seconds),
                )
                valuation = compute_alpha_price_valuation(
                    network=network,
                    netuid=netuid,
                    epoch=epoch,
                    tao_per_alpha=tao_per_alpha,
                    tao_usd=tao_usd,
                    miner_alpha_per_epoch=miner_alpha_per_epoch,
                    pricing_status="live",
                )
                _CACHE[cache_key] = valuation
                return valuation
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                logger.warning(
                    "research_lab_alpha_price_fetch_failed",
                    extra={
                        "attempt": attempt,
                        "max_attempts": max_attempts,
                        "network": str(network or "finney"),
                        "netuid": int(netuid),
                        "epoch": int(epoch),
                        "error_type": type(exc).__name__,
                    },
                )
                if attempt < max(1, int(max_attempts)):
                    await asyncio.sleep(min(2.0, 0.25 * attempt))

        if require_live:
            raise RuntimeError(f"live alpha price unavailable: {last_error}")

        fallback = static_alpha_price_fallback(
            network=network,
            netuid=netuid,
            epoch=epoch,
            static_usd_per_0_1_percent_epoch=static_usd_per_0_1_percent_epoch,
            miner_alpha_per_epoch=miner_alpha_per_epoch,
            reason=last_error or "live_price_unavailable",
        )
        return fallback


def inject_alpha_price_valuation(policy: dict[str, Any], valuation: AlphaPriceValuation) -> dict[str, Any]:
    """Return a policy doc with the epoch valuation applied."""

    updated = dict(policy)
    updated["usd_per_0_1_percent_epoch"] = _money_float(valuation.usd_per_0_1_percent_epoch)
    updated["alpha_price_valuation"] = valuation.policy_doc()
    return updated


async def _fetch_tao_per_alpha(*, network: str, netuid: int, timeout_seconds: float) -> Decimal:
    async def _fetch() -> Decimal:
        import bittensor as bt

        async with bt.AsyncSubtensor(network) as subtensor:
            subnet_info = await subtensor.subnet(netuid)
            return _decimal(getattr(subnet_info, "price"))

    return await asyncio.wait_for(_fetch(), timeout=max(1.0, float(timeout_seconds)))


async def _fetch_tao_usd(*, timeout_seconds: float) -> Decimal:
    timeout = aiohttp.ClientTimeout(total=max(1.0, float(timeout_seconds)))
    params = {"ids": "bittensor", "vs_currencies": "usd"}
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(COINGECKO_TAO_USD_URL, params=params) as response:
            response.raise_for_status()
            data = await response.json()
    return _decimal(data["bittensor"]["usd"])


def _with_status(valuation: AlphaPriceValuation, status: str) -> AlphaPriceValuation:
    return AlphaPriceValuation(
        network=valuation.network,
        netuid=valuation.netuid,
        epoch=valuation.epoch,
        pricing_status=status,
        tao_per_alpha=valuation.tao_per_alpha,
        tao_usd=valuation.tao_usd,
        alpha_usd=valuation.alpha_usd,
        miner_alpha_per_epoch=valuation.miner_alpha_per_epoch,
        usd_per_0_1_percent_epoch=valuation.usd_per_0_1_percent_epoch,
        fetched_at=valuation.fetched_at,
        sources=valuation.sources,
        fallback_reason=valuation.fallback_reason,
    )


def _decimal(value: Any) -> Decimal:
    tao_value = getattr(value, "tao", None)
    if tao_value is not None:
        return Decimal(str(tao_value))
    try:
        return Decimal(str(value))
    except Exception:
        return Decimal(str(float(value)))


def _money_float(value: Decimal) -> float:
    return float(value.quantize(USD_PER_PERCENT_QUANT, rounding=ROUND_HALF_UP))


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
