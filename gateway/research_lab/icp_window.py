"""Research Lab ICP benchmark window selection.

The gateway owns hidden ICP plaintext. This module builds deterministic scoring
windows while exposing only public refs and hashes. New production windows use a
hybrid fresh/retained policy; legacy rolling windows remain reconstructable.
"""

from __future__ import annotations

from dataclasses import dataclass
import asyncio
from typing import Any, Mapping, Sequence

from research_lab.canonical import sha256_json


WINDOW_MODE_LEGACY_ROLLING = "legacy_rolling"
WINDOW_MODE_HYBRID_FRESH_RETAINED = "hybrid_fresh_retained"
SELECTION_POLICY_LEGACY = "diverse_intent_industry_stable_hash:v2"
SELECTION_POLICY_HYBRID = "hybrid_fresh_retained_stable_diverse:v1"


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
    # Matches the reconciled prod default (config lab_champion_icps_per_day).
    icps_per_day: int = 2,
    window_mode: str = WINDOW_MODE_HYBRID_FRESH_RETAINED,
    fresh_icp_count: int = 10,
    retained_icp_count: int = 10,
    min_new_icp_count: int | None = None,
    allow_partial: bool = False,
) -> ResearchLabRollingIcpWindow:
    if days <= 0 or icps_per_day <= 0:
        raise ValueError("days and icps_per_day must be positive")
    mode = str(window_mode or WINDOW_MODE_HYBRID_FRESH_RETAINED).strip().lower()

    normalized_rows = sorted(
        (_normalize_set_row(row) for row in rows),
        key=lambda row: int(row["set_id"]),
        reverse=True,
    )
    if mode in {"legacy", "rolling", WINDOW_MODE_LEGACY_ROLLING}:
        return _select_legacy_rolling_icp_window(
            normalized_rows,
            days=days,
            icps_per_day=icps_per_day,
            allow_partial=allow_partial,
        )
    if mode != WINDOW_MODE_HYBRID_FRESH_RETAINED:
        raise ValueError(f"unsupported Research Lab ICP window mode: {window_mode}")
    return _select_hybrid_fresh_retained_icp_window(
        normalized_rows,
        days=days,
        icps_per_day=icps_per_day,
        fresh_icp_count=fresh_icp_count,
        retained_icp_count=retained_icp_count,
        min_new_icp_count=min_new_icp_count,
        allow_partial=allow_partial,
    )


def reconstruct_icp_window_from_doc(
    rows: Sequence[Mapping[str, Any]],
    public_doc: Mapping[str, Any],
) -> ResearchLabRollingIcpWindow:
    """Rebuild benchmark plaintext for an immutable stored window doc.

    This path is intentionally doc-driven so old rolling windows and new hybrid
    windows can both be reconstructed without re-running whichever selector was
    current when the row was written.
    """

    window_hash = str(public_doc.get("rolling_window_hash") or "")
    if not window_hash:
        raise RollingIcpWindowUnavailable("rolling_window_hash_missing")
    rows_by_set_id = {int(row["set_id"]): _normalize_set_row(row) for row in rows}
    benchmark_items: list[dict[str, Any]] = []
    item_refs: list[str] = []
    seen_refs: set[str] = set()
    set_ids: list[int] = []
    for day_index, set_doc in enumerate(public_doc.get("sets") or [], start=1):
        if not isinstance(set_doc, Mapping):
            continue
        set_id = int(set_doc.get("set_id") or 0)
        row = rows_by_set_id.get(set_id)
        if row is None:
            raise RollingIcpWindowUnavailable(f"set_{set_id}_missing_for_reconstruction")
        if set_id not in set_ids:
            set_ids.append(set_id)
        for selected in set_doc.get("selected_icps") or []:
            if not isinstance(selected, Mapping):
                continue
            icp_ref = str(selected.get("icp_ref") or "").strip()
            icp_hash = str(selected.get("icp_hash") or "").strip()
            if not icp_ref:
                raise RollingIcpWindowUnavailable("selected_icp_ref_missing_for_reconstruction")
            if icp_ref in seen_refs:
                raise RollingIcpWindowUnavailable(f"duplicate_selected_icp_ref:{icp_ref}")
            icp = _find_selected_icp(row, icp_ref=icp_ref, icp_hash=icp_hash)
            if icp is None:
                raise RollingIcpWindowUnavailable(f"selected_icp_missing_for_reconstruction:{icp_ref}")
            seen_refs.add(icp_ref)
            cohort = str(selected.get("cohort") or "").strip()
            item = {
                "icp": dict(icp),
                "icp_hash": icp_hash or sha256_json({"icp": icp}),
                "icp_ref": icp_ref,
                "set_id": set_id,
                "day_index": int(selected.get("day_index") or day_index),
                "day_rank": int(selected.get("rank") or len(benchmark_items) + 1),
                "intent_signal_signature": str(
                    selected.get("intent_signal_signature") or intent_signal_signature(icp)
                ),
            }
            if cohort:
                item["cohort"] = cohort
            benchmark_items.append(item)
            item_refs.append(icp_ref)
    if not benchmark_items:
        raise RollingIcpWindowUnavailable("stored_window_has_no_selected_icps")
    return ResearchLabRollingIcpWindow(
        window_hash=window_hash,
        benchmark_id=f"research_lab:rolling_icp_window:{window_hash}",
        split_ref=f"supabase:qualification_private_icp_sets:rolling:{window_hash}",
        public_doc=dict(public_doc),
        benchmark_items=tuple(benchmark_items),
        item_refs=tuple(item_refs),
        set_ids=tuple(set_ids),
    )


def _select_legacy_rolling_icp_window(
    normalized_rows: Sequence[Mapping[str, Any]],
    *,
    days: int,
    icps_per_day: int,
    allow_partial: bool,
) -> ResearchLabRollingIcpWindow:
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
        "selection_policy": SELECTION_POLICY_LEGACY,
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


def _select_hybrid_fresh_retained_icp_window(
    normalized_rows: Sequence[Mapping[str, Any]],
    *,
    days: int,
    icps_per_day: int,
    fresh_icp_count: int,
    retained_icp_count: int,
    min_new_icp_count: int | None,
    allow_partial: bool,
) -> ResearchLabRollingIcpWindow:
    if fresh_icp_count <= 0 or retained_icp_count <= 0:
        raise ValueError("fresh_icp_count and retained_icp_count must be positive")
    expected_total = int(days) * int(icps_per_day)
    if fresh_icp_count + retained_icp_count != expected_total:
        raise ValueError("fresh_icp_count + retained_icp_count must equal days * icps_per_day")
    selected_sets = list(normalized_rows[:days])
    if len(selected_sets) < 2 and not allow_partial:
        raise RollingIcpWindowUnavailable(
            f"hybrid_icp_window_requires_fresh_and_retained_sets_found_{len(selected_sets)}"
        )
    if not selected_sets:
        raise RollingIcpWindowUnavailable("hybrid_icp_window_requires_sets")

    fresh_set = selected_sets[0]
    retained_sets = selected_sets[1:]
    fresh_selected = _select_icps_for_day(fresh_set["icps"], icps_per_day=fresh_icp_count)
    if len(fresh_selected) < fresh_icp_count and not allow_partial:
        raise RollingIcpWindowUnavailable(
            f"set_{fresh_set['set_id']}_requires_{fresh_icp_count}_fresh_icps_found_{len(fresh_selected)}"
        )
    selected_records: list[dict[str, Any]] = [
        {"row": fresh_set, "icp": icp, "cohort": "fresh"}
        for icp in fresh_selected
    ]
    excluded_refs = {
        _icp_ref(fresh_set["set_id"], icp)
        for icp in fresh_selected
    }
    retained_selected = _select_icps_across_sets(
        retained_sets,
        icp_count=retained_icp_count,
        excluded_refs=excluded_refs,
        initial_seen_features=_seen_features_for_icps(fresh_selected),
    )
    if len(retained_selected) < retained_icp_count and not allow_partial:
        raise RollingIcpWindowUnavailable(
            f"hybrid_icp_window_requires_{retained_icp_count}_retained_icps_found_{len(retained_selected)}"
        )
    selected_records.extend(retained_selected)
    min_new = fresh_icp_count if min_new_icp_count is None else int(min_new_icp_count)
    if len(fresh_selected) < min_new and not allow_partial:
        raise RollingIcpWindowUnavailable(
            f"hybrid_icp_window_requires_{min_new}_new_icps_found_{len(fresh_selected)}"
        )

    return _build_window_from_selected_records(
        selected_records,
        required_days=days,
        icps_per_day=icps_per_day,
        schema_version="1.1",
        selection_policy=SELECTION_POLICY_HYBRID,
        extra_doc={
            "window_mode": WINDOW_MODE_HYBRID_FRESH_RETAINED,
            "fresh_set_id": int(fresh_set["set_id"]),
            "fresh_icp_count": len(fresh_selected),
            "retained_icp_count": len(retained_selected),
            "min_new_icp_count": min_new,
        },
    )


async def fetch_rolling_icp_window(
    *,
    days: int = 10,
    # Matches the reconciled prod default (config lab_champion_icps_per_day).
    icps_per_day: int = 2,
    window_mode: str = WINDOW_MODE_HYBRID_FRESH_RETAINED,
    fresh_icp_count: int = 10,
    retained_icp_count: int = 10,
    min_new_icp_count: int | None = None,
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
        window_mode=window_mode,
        fresh_icp_count=fresh_icp_count,
        retained_icp_count=retained_icp_count,
        min_new_icp_count=min_new_icp_count,
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


def _build_window_from_selected_records(
    selected_records: Sequence[Mapping[str, Any]],
    *,
    required_days: int,
    icps_per_day: int,
    schema_version: str,
    selection_policy: str,
    extra_doc: Mapping[str, Any] | None = None,
) -> ResearchLabRollingIcpWindow:
    grouped: dict[int, list[Mapping[str, Any]]] = {}
    row_by_set_id: dict[int, Mapping[str, Any]] = {}
    for record in selected_records:
        row = record.get("row") if isinstance(record.get("row"), Mapping) else {}
        set_id = int(row.get("set_id") or 0)
        if set_id <= 0:
            continue
        row_by_set_id[set_id] = row
        grouped.setdefault(set_id, []).append(record)

    public_sets: list[dict[str, Any]] = []
    benchmark_items: list[dict[str, Any]] = []
    for day_index, set_id in enumerate(sorted(grouped), start=1):
        row = row_by_set_id[set_id]
        public_items: list[dict[str, Any]] = []
        for rank, record in enumerate(grouped[set_id], start=1):
            icp = dict(record.get("icp") if isinstance(record.get("icp"), Mapping) else {})
            if not icp:
                continue
            cohort = str(record.get("cohort") or "").strip()
            icp_id = str(icp.get("icp_id") or f"icp_{set_id}_{rank:03d}")
            icp_hash = sha256_json({"icp": icp})
            icp_ref = f"qualification_private_icp_sets:{set_id}:{icp_id}"
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
            benchmark_item = {
                "icp": icp,
                "icp_hash": icp_hash,
                "icp_ref": icp_ref,
                "set_id": set_id,
                "day_index": day_index,
                "day_rank": rank,
                "intent_signal_signature": signal_signature,
            }
            if cohort:
                public_item["cohort"] = cohort
                benchmark_item["cohort"] = cohort
            public_items.append(public_item)
            benchmark_items.append(benchmark_item)
        public_sets.append(
            {
                "set_id": set_id,
                "icp_set_hash": row["icp_set_hash"],
                "selected_icps": public_items,
            }
        )

    public_doc_without_hash = {
        "schema_version": schema_version,
        "window_type": "research_lab_rolling_icp_window",
        "required_days": int(required_days),
        "icps_per_day": int(icps_per_day),
        "selected_set_count": len(public_sets),
        "selected_icp_count": len(benchmark_items),
        "selection_policy": selection_policy,
        **dict(extra_doc or {}),
        "sets": public_sets,
    }
    window_hash = sha256_json(public_doc_without_hash)
    public_doc = {**public_doc_without_hash, "rolling_window_hash": window_hash}
    return ResearchLabRollingIcpWindow(
        window_hash=window_hash,
        benchmark_id=f"research_lab:rolling_icp_window:{window_hash}",
        split_ref=f"supabase:qualification_private_icp_sets:rolling:{window_hash}",
        public_doc=public_doc,
        benchmark_items=tuple(benchmark_items),
        item_refs=tuple(str(item["icp_ref"]) for item in benchmark_items),
        set_ids=tuple(sorted(grouped)),
    )


def _select_icps_across_sets(
    rows: Sequence[Mapping[str, Any]],
    *,
    icp_count: int,
    excluded_refs: set[str],
    initial_seen_features: Mapping[str, set[str]] | None = None,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for row in rows:
        set_id = int(row["set_id"])
        for icp in _dedupe_and_rank_icps(row["icps"]):
            ref = _icp_ref(set_id, icp)
            if ref in excluded_refs:
                continue
            candidates.append({"row": row, "icp": icp, "cohort": "retained", "ref": ref})
    selected: list[dict[str, Any]] = []
    seen_refs: set[str] = set(excluded_refs)
    seen_features = _empty_seen_features()
    for key, values in (initial_seen_features or {}).items():
        seen_features.setdefault(key, set()).update(str(value) for value in values if str(value).strip())
    remaining = list(candidates)
    seen_set_ids: set[int] = set()
    while remaining and len(selected) < icp_count:
        best = min(
            remaining,
            key=lambda record: (
                -_retained_candidate_score(record, seen_features=seen_features, seen_set_ids=seen_set_ids),
                -int(record["row"]["set_id"]),
                _stable_icp_hash(record["icp"]),
            ),
        )
        ref = str(best.get("ref") or "")
        if ref in seen_refs:
            remaining.remove(best)
            continue
        selected.append(best)
        seen_refs.add(ref)
        seen_set_ids.add(int(best["row"]["set_id"]))
        remaining.remove(best)
        for key, value in _icp_features(best["icp"]).items():
            if value:
                seen_features[key].add(value)
    return selected


def _select_icps_for_day(icps: Sequence[Mapping[str, Any]], *, icps_per_day: int) -> list[dict[str, Any]]:
    ranked = _dedupe_and_rank_icps(icps)
    selected: list[dict[str, Any]] = []
    seen_features = _empty_seen_features()
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


def _empty_seen_features() -> dict[str, set[str]]:
    return {
        "intent": set(),
        "industry": set(),
        "sub_industry": set(),
        "product_service": set(),
        "geography": set(),
        "company_size": set(),
    }


def _seen_features_for_icps(icps: Sequence[Mapping[str, Any]]) -> dict[str, set[str]]:
    seen = _empty_seen_features()
    for icp in icps:
        for key, value in _icp_features(icp).items():
            if value:
                seen[key].add(value)
    return seen


def _find_selected_icp(
    row: Mapping[str, Any],
    *,
    icp_ref: str,
    icp_hash: str,
) -> dict[str, Any] | None:
    for icp in row.get("icps") or []:
        if not isinstance(icp, Mapping):
            continue
        candidate = dict(icp)
        candidate_ref = _icp_ref(int(row["set_id"]), candidate)
        candidate_hash = sha256_json({"icp": candidate})
        ref_matches = not icp_ref or candidate_ref == icp_ref
        hash_matches = not icp_hash or candidate_hash == icp_hash
        if ref_matches and hash_matches:
            return candidate
    return None


def _icp_ref(set_id: int, icp: Mapping[str, Any]) -> str:
    icp_id = str(icp.get("icp_id") or _stable_icp_hash(icp))
    return f"qualification_private_icp_sets:{int(set_id)}:{icp_id}"


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


def _retained_candidate_score(
    record: Mapping[str, Any],
    *,
    seen_features: Mapping[str, set[str]],
    seen_set_ids: set[int],
) -> int:
    set_id = int(record["row"]["set_id"])
    set_coverage_bonus = 10_000 if set_id not in seen_set_ids else 0
    return set_coverage_bonus + _novelty_score(_icp_features(record["icp"]), seen_features)


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
    if isinstance(value, (list, tuple)):
        value = "|".join(str(item) for item in value if str(item).strip())
    return " ".join(str(value or "").strip().lower().split())
