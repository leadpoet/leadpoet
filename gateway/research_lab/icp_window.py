"""Research Lab rolling ICP window selection.

The gateway owns hidden ICP plaintext. This module builds a deterministic
10-day / 60-ICP scoring window while exposing only public refs and hashes.
"""

from __future__ import annotations

from dataclasses import dataclass
import asyncio
from typing import Any, Mapping, Sequence

from research_lab.canonical import sha256_json


@dataclass(frozen=True)
class ResearchLabRollingIcpWindow:
    window_hash: str
    benchmark_id: str
    split_ref: str
    public_doc: dict[str, Any]
    benchmark_items: tuple[dict[str, Any], ...]
    item_refs: tuple[str, ...]
    set_ids: tuple[int, ...]


class RollingIcpWindowUnavailable(RuntimeError):
    """Raised when the configured Research Lab ICP window cannot be built."""


def normalize_sha256_ref(value: Any, *, field_name: str = "hash") -> str:
    raw = str(value or "").strip()
    if raw.startswith("sha256:") and len(raw) == 71:
        return raw
    if len(raw) == 64 and all(ch in "0123456789abcdef" for ch in raw.lower()):
        return "sha256:" + raw.lower()
    raise RollingIcpWindowUnavailable(f"{field_name}_must_be_sha256")


def select_rolling_icp_window_from_sets(
    rows: Sequence[Mapping[str, Any]],
    *,
    days: int = 10,
    icps_per_day: int = 6,
    allow_partial: bool = False,
) -> ResearchLabRollingIcpWindow:
    if days <= 0 or icps_per_day <= 0:
        raise ValueError("days and icps_per_day must be positive")

    normalized_rows = sorted(
        (_normalize_set_row(row) for row in rows),
        key=lambda row: int(row["set_id"]),
        reverse=True,
    )
    selected_sets = normalized_rows[:days]
    if len(selected_sets) < days and not allow_partial:
        raise RollingIcpWindowUnavailable(
            f"rolling_icp_window_requires_{days}_sets_found_{len(selected_sets)}"
        )
    selected_sets = sorted(selected_sets, key=lambda row: int(row["set_id"]))

    public_sets: list[dict[str, Any]] = []
    benchmark_items: list[dict[str, Any]] = []
    for row in selected_sets:
        selected_icps = _select_icps_for_day(row["icps"], icps_per_day=icps_per_day)
        if len(selected_icps) < icps_per_day and not allow_partial:
            raise RollingIcpWindowUnavailable(
                f"set_{row['set_id']}_requires_{icps_per_day}_selectable_icps_found_{len(selected_icps)}"
            )
        public_items: list[dict[str, Any]] = []
        day_index = len(public_sets) + 1
        for rank, icp in enumerate(selected_icps, start=1):
            icp_id = str(icp.get("icp_id") or f"icp_{row['set_id']}_{rank:03d}")
            icp_hash = sha256_json({"icp": icp})
            icp_ref = f"qualification_private_icp_sets:{row['set_id']}:{icp_id}"
            signal_signature = intent_signal_signature(icp)
            public_item = {
                "rank": rank,
                "day_index": day_index,
                "icp_ref": icp_ref,
                "icp_id": icp_id,
                "icp_hash": icp_hash,
                "intent_signal_signature": signal_signature,
                "industry": str(icp.get("industry") or ""),
                "sub_industry": str(icp.get("sub_industry") or ""),
            }
            public_items.append(public_item)
            benchmark_items.append(
                {
                    "icp": dict(icp),
                    "icp_hash": icp_hash,
                    "icp_ref": icp_ref,
                    "set_id": int(row["set_id"]),
                    "day_index": day_index,
                    "day_rank": rank,
                    "intent_signal_signature": signal_signature,
                }
            )
        public_sets.append(
            {
                "set_id": int(row["set_id"]),
                "icp_set_hash": row["icp_set_hash"],
                "selected_icps": public_items,
            }
        )

    public_doc_without_hash = {
        "schema_version": "1.0",
        "window_type": "research_lab_rolling_icp_window",
        "required_days": int(days),
        "icps_per_day": int(icps_per_day),
        "selected_set_count": len(public_sets),
        "selected_icp_count": len(benchmark_items),
        "selection_policy": "diverse_intent_industry_stable_hash:v2",
        "sets": public_sets,
    }
    window_hash = sha256_json(public_doc_without_hash)
    public_doc = {**public_doc_without_hash, "rolling_window_hash": window_hash}
    benchmark_id = f"research_lab:rolling_icp_window:{window_hash}"
    split_ref = f"supabase:qualification_private_icp_sets:rolling:{window_hash}"
    return ResearchLabRollingIcpWindow(
        window_hash=window_hash,
        benchmark_id=benchmark_id,
        split_ref=split_ref,
        public_doc=public_doc,
        benchmark_items=tuple(benchmark_items),
        item_refs=tuple(str(item["icp_ref"]) for item in benchmark_items),
        set_ids=tuple(int(row["set_id"]) for row in selected_sets),
    )


async def fetch_rolling_icp_window(
    *,
    days: int = 10,
    icps_per_day: int = 6,
    allow_partial: bool = False,
) -> ResearchLabRollingIcpWindow:
    """Fetch private ICP sets through the gateway service-role client."""

    from gateway.db.client import get_write_client

    def _call() -> Any:
        return (
            get_write_client()
            .table("qualification_private_icp_sets")
            .select("set_id,icps,icp_set_hash,active_from,active_until,is_active")
            .order("set_id", desc=True)
            .limit(max(days * 3, days))
            .execute()
        )

    response = await asyncio.to_thread(_call)
    rows = getattr(response, "data", None) or []
    return select_rolling_icp_window_from_sets(
        rows,
        days=days,
        icps_per_day=icps_per_day,
        allow_partial=allow_partial,
    )


def intent_signal_signature(icp: Mapping[str, Any]) -> str:
    signals = icp.get("intent_signals") or []
    if isinstance(signals, str):
        signals = [signals]
    normalized = sorted(
        {
            " ".join(str(signal).strip().lower().split())
            for signal in signals
            if str(signal).strip()
        }
    )
    if normalized:
        return "|".join(normalized)
    fallback = [
        str(icp.get("industry") or "").strip().lower(),
        str(icp.get("sub_industry") or "").strip().lower(),
        str(icp.get("product_service") or "").strip().lower(),
    ]
    return "|".join(part for part in fallback if part) or "unknown"


def _normalize_set_row(row: Mapping[str, Any]) -> dict[str, Any]:
    set_id = int(row["set_id"])
    icps = row.get("icps") or []
    if not isinstance(icps, list):
        raise RollingIcpWindowUnavailable(f"set_{set_id}_icps_must_be_array")
    return {
        "set_id": set_id,
        "icps": [dict(icp) for icp in icps if isinstance(icp, Mapping)],
        "icp_set_hash": normalize_sha256_ref(row.get("icp_set_hash"), field_name="icp_set_hash"),
    }


def _select_icps_for_day(icps: Sequence[Mapping[str, Any]], *, icps_per_day: int) -> list[dict[str, Any]]:
    ranked = _dedupe_and_rank_icps(icps)
    selected: list[dict[str, Any]] = []
    seen_features: dict[str, set[str]] = {
        "intent": set(),
        "industry": set(),
        "sub_industry": set(),
        "product_service": set(),
        "geography": set(),
        "company_size": set(),
    }
    remaining = list(ranked)
    while remaining and len(selected) < icps_per_day:
        best = min(
            remaining,
            key=lambda icp: (
                -_novelty_score(_icp_features(icp), seen_features),
                _stable_icp_hash(icp),
            ),
        )
        selected.append(best)
        remaining.remove(best)
        for key, value in _icp_features(best).items():
            if value:
                seen_features[key].add(value)
    return selected


def _dedupe_and_rank_icps(icps: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    deduped: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(icps):
        icp = dict(item)
        identity = str(icp.get("icp_id") or _stable_icp_hash(icp) or index)
        current = deduped.get(identity)
        if current is None or _stable_icp_hash(icp) < _stable_icp_hash(current):
            deduped[identity] = icp
    return sorted(deduped.values(), key=_stable_icp_hash)


def _icp_features(icp: Mapping[str, Any]) -> dict[str, str]:
    return {
        "intent": intent_signal_signature(icp),
        "industry": _normalize_feature(icp.get("industry")),
        "sub_industry": _normalize_feature(icp.get("sub_industry")),
        "product_service": _normalize_feature(icp.get("product_service")),
        "geography": _normalize_feature(icp.get("geography") or icp.get("country") or icp.get("target_geography")),
        "company_size": _normalize_feature(icp.get("employee_count") or icp.get("company_size")),
    }


def _novelty_score(features: Mapping[str, str], seen_features: Mapping[str, set[str]]) -> int:
    weights = {
        "intent": 100,
        "industry": 60,
        "sub_industry": 40,
        "product_service": 25,
        "geography": 15,
        "company_size": 10,
    }
    score = 0
    for key, weight in weights.items():
        value = features.get(key) or ""
        if value and value not in seen_features.get(key, set()):
            score += weight
    return score


def _stable_icp_hash(icp: Mapping[str, Any]) -> str:
    return sha256_json(
        {
            "icp_id": icp.get("icp_id"),
            "industry": icp.get("industry"),
            "sub_industry": icp.get("sub_industry"),
            "product_service": icp.get("product_service"),
            "geography": icp.get("geography") or icp.get("country") or icp.get("target_geography"),
            "employee_count": icp.get("employee_count") or icp.get("company_size"),
            "intent_signals": icp.get("intent_signals"),
            "prompt": icp.get("prompt"),
        }
    )


def _normalize_feature(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())
