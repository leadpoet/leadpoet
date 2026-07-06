"""§9.4 meta-allocator cell-yield priors from the results ledger.

Reads ``research_lab_results_ledger`` (one row per drafted node with
keep/discard/crash/timeout status, written by the §9.1 trajectory projector),
aggregates it into (island x lane x targeted_metric) cells, and feeds
``CellYieldPriorRecord``-shaped inputs through the dormant
``research_lab.meta_allocator.build_meta_allocator_selection_record``
(deterministic seeded Thompson selection — the seed derives from the UTC day
plus the ledger-window hash so every worker computes the same ranking).

The output is a compact ``cell_yield_priors`` context doc injected into the
loop-direction planner prompt as an ordering/weight hint ONLY: it never funds,
merges, or rejects anything, and an epsilon floor (uniform 10% exploration
mass) guarantees no cell is ever zeroed out.

Flag-gated OFF by default (``RESEARCH_LAB_ALLOCATOR_PRIORS_ENABLED``, §8.3
staged-enable discipline); callers treat every failure as best-effort.
"""

from __future__ import annotations

import math
import os
import re
from typing import Any, Mapping, Sequence

from research_lab.canonical import sha256_json, utc_now_iso
from research_lab.meta_allocator import (
    CellYieldPriorRecord,
    CellYieldPriorState,
    build_meta_allocator_selection_record,
)

ALLOCATOR_PRIORS_ENABLED_ENV = "RESEARCH_LAB_ALLOCATOR_PRIORS_ENABLED"

RESULTS_LEDGER_TABLE = "research_lab_results_ledger"
ALLOCATOR_SELECTION_RECORDS_TABLE = "research_lab_allocator_selection_records"

DEFAULT_WINDOW_ROWS = 500
DEFAULT_TOP_CELLS = 8
# Uniform exploration floor: 10% of the total weight mass stays uniformly
# spread so no cell's weight can reach zero (degenerate-convergence guard).
EXPLORATION_FLOOR = 0.10
MAX_LEDGER_REFS_PER_CELL = 20
# Nodes whose spend was recorded as 0 still cost something to attempt.
MIN_EXPECTED_COST_CENTS = 1

# The §9.1 projector writes ledger descriptions as
# "CODE_EDIT on {lane} targeted {metric}; decision=...; delta=...".
_LANE_FROM_DESCRIPTION = re.compile(r"CODE_EDIT on (\S+) targeted")
_CELL_TOKEN_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")


def allocator_priors_enabled() -> bool:
    """§8.3 staged enable: allocator priors in prompts are OFF by default."""
    return os.environ.get(ALLOCATOR_PRIORS_ENABLED_ENV, "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


class GatewayLedgerStore:
    """Thin async facade over gateway.research_lab.store's select helpers."""

    async def select_many(
        self,
        table: str,
        *,
        columns: str = "*",
        filters: Any,
        order_by: Any = (),
        limit: int = 100,
    ):
        from gateway.research_lab import store

        return await store.select_many(
            table, columns=columns, filters=filters, order_by=order_by, limit=limit
        )

    async def insert_row(self, table: str, row: dict[str, Any]) -> dict[str, Any]:
        from gateway.research_lab import store

        return await store.insert_row(table, row)


def _cell_token(value: str, fallback: str) -> str:
    token = _CELL_TOKEN_SAFE.sub("-", str(value or "").strip())[:64].strip("-")
    return token or fallback


def _lane_from_ledger_row(row: Mapping[str, Any]) -> str:
    match = _LANE_FROM_DESCRIPTION.search(str(row.get("description") or ""))
    lane = match.group(1) if match else ""
    return _cell_token(lane, "code_edit")


def _f(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def selection_seed(day: str, window_hash: str) -> int:
    """Deterministic 32-bit seed from the UTC day + ledger-window hash.

    Every worker that sees the same ledger window on the same day derives the
    same seed, so the seeded Thompson draws (and therefore the ranking hint)
    agree fleet-wide without coordination.
    """
    digest = sha256_json({"day": str(day), "window_hash": str(window_hash)}).split(":", 1)[1]
    return int(digest[:8], 16)


def build_cell_yield_prior_records(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[CellYieldPriorRecord], dict[str, dict[str, Any]]]:
    """Aggregate ledger rows into measured CellYieldPriorRecords per cell.

    Cells are (island x lane x targeted_metric); lane is recovered from the
    projector's deterministic description format. The posterior is a mechanical
    Laplace keep-rate estimate: mean = (keeps + 1) / (attempts + 2) with the
    matching binomial standard deviation — deterministic, bounded, and defined
    even for cells that have never produced a keep.
    """
    cells: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        ledger_row_id = str(row.get("ledger_row_id") or "")
        status = str(row.get("status") or "")
        if not ledger_row_id or status not in {"keep", "discard", "crash", "timeout"}:
            continue
        island = _cell_token(str(row.get("island") or ""), "unknown_island")
        lane = _lane_from_ledger_row(row)
        metric = _cell_token(
            str(row.get("targeted_metric") or ""), "candidate_delta_vs_daily_baseline"
        )
        cell_ref = f"map_cell:{island}:{lane}:{metric}"
        stats = cells.setdefault(
            cell_ref,
            {
                "cell_ref": cell_ref,
                "island": island,
                "lane": lane,
                "targeted_metric": metric,
                "observed_attempts": 0,
                "keep": 0,
                "discard": 0,
                "crash": 0,
                "timeout": 0,
                "spend_usd": 0.0,
                "delta_sum": 0.0,
                "delta_count": 0,
                "kept_delta_sum": 0.0,
                "ledger_refs": [],
            },
        )
        stats["observed_attempts"] += 1
        stats[status] += 1
        stats["spend_usd"] += max(0.0, _f(row.get("cost_usd")))
        delta = row.get("delta_vs_parent")
        if delta is not None:
            stats["delta_sum"] += _f(delta)
            stats["delta_count"] += 1
            if status == "keep":
                stats["kept_delta_sum"] += _f(delta)
        if len(stats["ledger_refs"]) < MAX_LEDGER_REFS_PER_CELL:
            stats["ledger_refs"].append(f"results_ledger:{ledger_row_id}")

    priors: list[CellYieldPriorRecord] = []
    for cell_ref in sorted(cells):
        stats = cells[cell_ref]
        attempts = int(stats["observed_attempts"])
        keeps = int(stats["keep"])
        laplace_mean = (keeps + 1.0) / (attempts + 2.0)
        laplace_std = math.sqrt(laplace_mean * (1.0 - laplace_mean) / (attempts + 3.0))
        spend_cents = int(round(stats["spend_usd"] * 100.0))
        expected_cost_cents = max(
            MIN_EXPECTED_COST_CENTS,
            int(round(spend_cents / attempts)) if attempts else MIN_EXPECTED_COST_CENTS,
        )
        stats["mean_delta_vs_parent"] = (
            round(stats["delta_sum"] / stats["delta_count"], 6)
            if stats["delta_count"]
            else None
        )
        stats["expected_cost_cents"] = expected_cost_cents
        priors.append(
            CellYieldPriorRecord(
                prior_id="cell_yield_prior:"
                + sha256_json({"cell_ref": cell_ref}).split(":", 1)[1][:16],
                cell_ref=cell_ref,
                island=stats["island"],
                target_component=stats["lane"],
                patch_type="CODE_EDIT",
                observed_attempts=attempts,
                kept_patches=keeps,
                total_spend_cents=max(0, spend_cents),
                verified_delta_sum=round(float(stats["kept_delta_sum"]), 6),
                posterior_mean_delta=round(laplace_mean, 6),
                posterior_std_delta=round(laplace_std, 6),
                expected_cost_cents=expected_cost_cents,
                source_results_ledger_refs=tuple(stats["ledger_refs"]),
                data_state=CellYieldPriorState.RESULTS_LEDGER_MEASURED.value,
                uses_local_fixtures=False,
                results_ledger_input_ready=True,
                local_only=True,
            )
        )
    return priors, cells


def _floored_weights(
    ranked_values: Sequence[float], *, floor: float = EXPLORATION_FLOOR
) -> list[float]:
    """Normalize sampled values into weights with a uniform exploration floor."""
    count = len(ranked_values)
    if not count:
        return []
    clamped = [max(0.0, float(value)) for value in ranked_values]
    total = sum(clamped)
    if total <= 0.0:
        normalized = [1.0 / count] * count
    else:
        normalized = [value / total for value in clamped]
    floor = min(max(0.0, float(floor)), 1.0)
    return [round((1.0 - floor) * value + floor / count, 6) for value in normalized]


async def build_cell_yield_priors(
    *,
    store: Any | None = None,
    window_rows: int = DEFAULT_WINDOW_ROWS,
    top_cells: int = DEFAULT_TOP_CELLS,
    day: str | None = None,
) -> dict[str, Any] | None:
    """Ledger -> priors -> deterministic seeded Thompson ranking context doc.

    Returns ``None`` when the ledger has no usable rows. The returned doc is a
    prompt-context hint only (never a funding/promotion input); weights carry a
    uniform 10% exploration floor so no cell is ever zeroed out.
    """
    store = store or GatewayLedgerStore()
    rows = await store.select_many(
        RESULTS_LEDGER_TABLE,
        columns=(
            "ledger_row_id,island,targeted_metric,status,delta_vs_parent,"
            "cost_usd,description,created_at"
        ),
        filters=[],
        order_by=[("created_at", True)],
        limit=max(1, int(window_rows)),
    )
    priors, cell_stats = build_cell_yield_prior_records(rows or [])
    if not priors:
        return None

    day = str(day or utc_now_iso()[:10])
    window_hash = sha256_json(
        {"ledger_row_ids": sorted(str(row.get("ledger_row_id") or "") for row in rows)}
    )
    seed = selection_seed(day, window_hash)
    selection = build_meta_allocator_selection_record(
        selection_id=(
            "meta_allocator_selection:cell-yield:"
            f"{day}:{window_hash.split(':', 1)[1][:12]}"
        ),
        priors=priors,
        # Nominal accounting fields: the selection record is used only as a
        # deterministic ranking here and is never persisted or spent against.
        budget_cents=max(1, sum(prior.expected_cost_cents for prior in priors)),
        selection_count=len(priors),
        seed=seed,
    )
    weights = _floored_weights(
        [score.sampled_value_per_1000_cents for score in selection.sample_scores]
    )
    ranked_cells: list[dict[str, Any]] = []
    for score, weight in zip(selection.sample_scores, weights):
        stats = cell_stats.get(score.cell_ref) or {}
        ranked_cells.append(
            {
                "cell_ref": score.cell_ref,
                "island": stats.get("island"),
                "lane": stats.get("lane"),
                "targeted_metric": stats.get("targeted_metric"),
                "weight": weight,
                "observed_attempts": stats.get("observed_attempts", 0),
                "keep": stats.get("keep", 0),
                "discard": stats.get("discard", 0),
                "crash": stats.get("crash", 0),
                "timeout": stats.get("timeout", 0),
                "mean_delta_vs_parent": stats.get("mean_delta_vs_parent"),
                "expected_cost_cents": stats.get("expected_cost_cents"),
                "sampled_value_per_1000_cents": score.sampled_value_per_1000_cents,
            }
        )
    return {
        "schema_version": "1.0",
        "selection_id": selection.selection_id,
        "seed": seed,
        "day": day,
        "window": {
            "row_count": len(rows),
            "window_hash": window_hash,
            "cell_count": len(priors),
        },
        "formula": (
            "deterministic seeded Thompson draw over the Laplace keep-rate per "
            "island x lane x targeted_metric cell, value-per-cost ranked"
        ),
        "exploration_floor": EXPLORATION_FLOOR,
        "note": (
            "Historical yield per exploration cell from the results ledger. Use the "
            "weights as an ordering hint when choosing a direction: prefer "
            "higher-weight cells but keep exploring — every cell retains a uniform "
            "exploration floor and none is forbidden. This hint never makes funding "
            "or promotion decisions."
        ),
        "ranked_cells": ranked_cells[: max(1, int(top_cells))],
    }


async def load_persisted_selection_doc(
    *,
    store: Any | None = None,
    day: str | None = None,
) -> dict[str, Any] | None:
    """Newest persisted selection doc for ``day`` (UTC today by default).

    Returns ``None`` when no record exists for the day (missing table errors
    are the caller's to treat as best-effort) so callers can fall back to the
    on-demand computation.
    """
    store = store or GatewayLedgerStore()
    day = str(day or utc_now_iso()[:10])
    rows = await store.select_many(
        ALLOCATOR_SELECTION_RECORDS_TABLE,
        columns="selection_id,day,window_hash,selection_doc,created_at",
        filters=[("day", day)],
        order_by=[("created_at", True)],
        limit=1,
    )
    for row in rows or []:
        doc = row.get("selection_doc") if isinstance(row, Mapping) else None
        if isinstance(doc, Mapping) and doc.get("ranked_cells"):
            return dict(doc)
    return None


async def load_cell_yield_priors(
    *,
    store: Any | None = None,
    window_rows: int = DEFAULT_WINDOW_ROWS,
    top_cells: int = DEFAULT_TOP_CELLS,
    day: str | None = None,
) -> dict[str, Any] | None:
    """Engine read path: prefer the persisted nightly selection, else compute.

    The persisted record (written by ``refresh_allocator_priors``) gives every
    run on a given day the identical hint and skips the per-run ledger scan;
    the fallback keeps the flag usable before the first nightly pass lands.
    """
    try:
        persisted = await load_persisted_selection_doc(store=store, day=day)
    except Exception:
        # Missing table / transient store failure: fall back to computing.
        persisted = None
    if persisted is not None:
        return persisted
    return await build_cell_yield_priors(
        store=store, window_rows=window_rows, top_cells=top_cells, day=day
    )


async def refresh_allocator_priors(
    *,
    store: Any | None = None,
    window_rows: int = DEFAULT_WINDOW_ROWS,
    top_cells: int = DEFAULT_TOP_CELLS,
    day: str | None = None,
    created_by: str = "",
) -> dict[str, Any]:
    """Nightly job body: compute today's selection and persist it (idempotent).

    Same day + same ledger window => same deterministic selection; the table's
    ``UNIQUE (day, window_hash)`` key plus the pre-check below make re-runs
    no-ops. Returns a compact status doc for logging.
    """
    store = store or GatewayLedgerStore()
    day = str(day or utc_now_iso()[:10])
    doc = await build_cell_yield_priors(
        store=store, window_rows=window_rows, top_cells=top_cells, day=day
    )
    if doc is None:
        return {"status": "empty_ledger", "day": day}
    window = doc.get("window") if isinstance(doc.get("window"), Mapping) else {}
    window_hash = str(window.get("window_hash") or "")
    existing = await store.select_many(
        ALLOCATOR_SELECTION_RECORDS_TABLE,
        columns="selection_record_id",
        filters=[("day", day), ("window_hash", window_hash)],
        order_by=(),
        limit=1,
    )
    if existing:
        return {
            "status": "already_persisted",
            "day": day,
            "selection_id": str(doc.get("selection_id") or ""),
            "window_hash": window_hash,
        }
    await store.insert_row(
        ALLOCATOR_SELECTION_RECORDS_TABLE,
        {
            "schema_version": "1.0",
            "selection_id": str(doc.get("selection_id") or ""),
            "day": day,
            "seed": int(doc.get("seed") or 0),
            "window_hash": window_hash,
            "window_row_count": int(window.get("row_count") or 0),
            "cell_count": int(window.get("cell_count") or 0),
            "selection_doc": doc,
            "created_by": str(created_by or "")[:120],
        },
    )
    return {
        "status": "persisted",
        "day": day,
        "selection_id": str(doc.get("selection_id") or ""),
        "window_hash": window_hash,
        "cell_count": int(window.get("cell_count") or 0),
    }
