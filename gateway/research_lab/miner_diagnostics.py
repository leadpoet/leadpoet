"""Sanitized per-candidate diagnostics for the miner who OWNS a research loop.

Projects data the evaluation already computes (candidate score bundle +
candidate patch manifest) into a miner-facing diagnostics document, keyed to
the miner's own run. Mirrors the shape already published for the champion, so
it is releasable without exposing training trajectories.

Privacy invariant (the load-bearing one, established by adversarial review):
    No outcome is ever attached to any identifiable subset of the SEALED
    (private-holdout) ICPs smaller than the whole private pool.

Concretely:
  * PUBLIC-split ICPs (bodies are already published in the champion report):
    exact per-ICP score / base / delta are exposed.
  * PRIVATE (sealed) ICPs: only WHOLE-POOL aggregates (helped/hurt/flat counts,
    infra-excluded count). Never a per-ICP row, never a per-ICP delta, never a
    bucket/sector group — those would give a controlled candidate a per-sealed-
    ICP oracle to reverse-engineer the hidden filters.

Also never emitted: the unified diff, private-repo file paths, raw per-company
score vectors, or anything ``contains_secret_material`` flags. This is a pure
projection module (no I/O) so it is fully unit-testable.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Optional, Sequence

from gateway.research_lab.bundles import contains_secret_material


SCHEMA_VERSION = "1.0"

# Per-ICP |delta| at or below this counts as "flat" rather than signed noise —
# candidate scoring is LLM-judged and has real run-to-run variance.
DEFAULT_FLAT_BAND = 1.0

# If more than this fraction of a run's ICPs were infra-excluded (provider /
# runtime errors), the run is flagged rerun-credit eligible — that outcome is
# not attributable to the miner's patch.
DEFAULT_INFRA_RERUN_FRACTION = 0.2

# Per-ICP statuses / failure-reason markers that mean "our infrastructure
# failed", NOT "the miner's patch produced a bad result". Kept as substrings so
# appended detail is tolerated.
_INFRA_MARKERS = (
    "provider_error",
    "runtime_provider",
    "provider_http",
    "provider_excluded",
    "infra_excluded",
    "reference_model_runtime",
    "candidate_model_runtime_provider",
)


def visibility_map_from_benchmark_split(
    benchmark_score_summary_doc: Mapping[str, Any] | None,
) -> dict[str, str]:
    """Map ``icp_ref -> "public"|"private"`` from a daily benchmark's
    ``score_summary_doc.visibility_split.items``.

    Join key: a candidate bundle's ``icp_set_hash`` equals the daily
    benchmark's ``rolling_window_hash`` (same rolling window), so the caller
    looks up the benchmark bundle for that window and passes its summary doc
    here. Missing/short docs degrade to an empty map (everything then treated
    as private, i.e. maximally conservative).
    """
    out: dict[str, str] = {}
    if not isinstance(benchmark_score_summary_doc, Mapping):
        return out
    split = benchmark_score_summary_doc.get("visibility_split")
    items = split.get("items") if isinstance(split, Mapping) else None
    if not isinstance(items, Sequence):
        return out
    for item in items:
        if not isinstance(item, Mapping):
            continue
        ref = str(item.get("icp_ref") or "").strip()
        vis = str(item.get("visibility") or "").strip().lower()
        if ref and vis in {"public", "private"}:
            out[ref] = vis
    return out


def _is_infra_row(row: Mapping[str, Any], provider_excluded: set[str]) -> bool:
    ref = str(row.get("icp_ref") or "")
    if ref and ref in provider_excluded:
        return True
    blob = f"{row.get('status') or ''} {row.get('failure_reason') or ''}".lower()
    return any(marker in blob for marker in _INFRA_MARKERS)


def _delta_band(delta: float, flat_band: float) -> str:
    if delta > flat_band:
        return "helped"
    if delta < -flat_band:
        return "hurt"
    return "flat"


def _num(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


_WITHHELD = "[summary withheld: contained a redacted token]"


def _safe_text(value: Any) -> str:
    """Free-text passed to a miner. ``redacted_summary`` is LLM-authored, so a
    stray secret marker in prose must NOT hard-fail the whole diagnostics doc —
    withhold just this field and keep everything else.
    """
    text = str(value or "").strip()
    if text and contains_secret_material(text):
        return _WITHHELD
    return text


def _emitted_patch(patch_manifest: Mapping[str, Any] | None) -> dict[str, Any]:
    """Component + patch_type + lane + redacted summary ONLY.

    Never the unified diff, never ``target_files`` (private-repo paths), never
    any hash of the raw payload.
    """
    if not isinstance(patch_manifest, Mapping):
        return {}
    patch_doc = patch_manifest.get("patch_doc")
    lane = ""
    if isinstance(patch_doc, Mapping):
        lane = str(patch_doc.get("lane") or "").strip()
    return {
        "target_component": _safe_text(patch_manifest.get("target_component_id")),
        "patch_type": _safe_text(patch_manifest.get("patch_type")),
        "candidate_kind": _safe_text(patch_manifest.get("candidate_kind")),
        "lane": _safe_text(lane),
        "redacted_summary": _safe_text(patch_manifest.get("redacted_summary")),
    }


def build_candidate_diagnostics(
    *,
    candidate_id: str,
    bundle_doc: Mapping[str, Any] | None,
    patch_manifest: Mapping[str, Any] | None,
    visibility_by_ref: Mapping[str, str] | None,
    candidate_status: str = "",
    flat_band: float = DEFAULT_FLAT_BAND,
    infra_rerun_fraction: float = DEFAULT_INFRA_RERUN_FRACTION,
) -> dict[str, Any]:
    """Build the sanitized own-run diagnostics document.

    Args:
        candidate_id: the owning candidate's id (echoed back).
        bundle_doc: the candidate score bundle doc (``aggregates`` etc.). May be
            None/empty for a ``failed`` candidate that never produced a bundle —
            a reduced doc (patch summary + status) is returned in that case.
        patch_manifest: the candidate patch manifest (component / patch_type /
            lane / redacted_summary). Diff and file paths are never read.
        visibility_by_ref: ``icp_ref -> "public"|"private"`` from
            ``visibility_map_from_benchmark_split``. Unknown refs are treated as
            PRIVATE (conservative — never over-expose a sealed ICP).
        candidate_status: terminal status label (scored/rejected/failed).

    The output is validated against ``contains_secret_material`` before return;
    a positive hit raises (a capture bug must never leak secrets to a miner).
    """
    vis = {str(k): str(v).lower() for k, v in (visibility_by_ref or {}).items()}

    def _visibility(ref: str) -> str:
        # Default PRIVATE: an unclassified ICP must never be treated as public.
        return "public" if vis.get(ref) == "public" else "private"

    emitted_patch = _emitted_patch(patch_manifest)

    aggregates = bundle_doc.get("aggregates") if isinstance(bundle_doc, Mapping) else None
    per_icp = (
        aggregates.get("per_icp_results")
        if isinstance(aggregates, Mapping) and isinstance(aggregates.get("per_icp_results"), Sequence)
        else []
    )

    # --- Reduced doc: candidate produced no scored bundle (e.g. runtime death).
    if not per_icp:
        doc = {
            "schema_version": SCHEMA_VERSION,
            "candidate_id": str(candidate_id),
            "candidate_status": str(candidate_status or ""),
            "emitted_patch": emitted_patch,
            "aggregate": {},
            "public_icps": [],
            "private_pool": {},
            "scored_companies": {"count": 0, "avg_final_score": 0.0},
            "note": "No scored ICP results available for this run.",
        }
        _assert_clean(doc)
        return doc

    # bundle_doc is guaranteed a Mapping here (per_icp is non-empty above).
    raw_excluded = bundle_doc.get("provider_excluded_icp_ids")
    provider_excluded = {str(x) for x in raw_excluded} if isinstance(raw_excluded, Sequence) and not isinstance(raw_excluded, str) else set()

    public_icps: list[dict[str, Any]] = []
    priv_helped = priv_hurt = priv_flat = priv_infra = priv_total = 0
    pub_infra = 0
    scored_score_sum = 0.0
    scored_score_count = 0

    for row in per_icp:
        if not isinstance(row, Mapping):
            continue
        ref = str(row.get("icp_ref") or "")
        is_infra = _is_infra_row(row, provider_excluded)
        delta = _num(row.get("delta_vs_base"))
        cand_score = _num(row.get("candidate_per_icp_score"))
        base_score = _num(row.get("base_per_icp_score"))

        # scored-company stats (count + avg ONLY — never the raw vector)
        for s in row.get("candidate_company_scores") or []:
            sv = _num(s, default=-1.0)
            if sv > 0:
                scored_score_sum += sv
                scored_score_count += 1

        if _visibility(ref) == "public":
            entry = {
                "icp_ref": ref,
                "candidate_score": round(cand_score, 4),
                "base_score": round(base_score, 4),
                "delta": round(delta, 4),
            }
            if is_infra:
                entry["status"] = "infra_excluded"
                entry["note"] = "provider/runtime error — not attributable to your patch"
                pub_infra += 1
            elif row.get("status") and str(row.get("status")) != "completed":
                entry["status"] = str(row.get("status"))
            public_icps.append(entry)
        else:
            # PRIVATE: pool aggregates only. Never a per-ICP row.
            priv_total += 1
            if is_infra:
                priv_infra += 1
            else:
                band = _delta_band(delta, flat_band)
                if band == "helped":
                    priv_helped += 1
                elif band == "hurt":
                    priv_hurt += 1
                else:
                    priv_flat += 1

    total_icps = len(per_icp)
    infra_total = pub_infra + priv_infra
    infra_fraction = (infra_total / total_icps) if total_icps else 0.0

    gate = bundle_doc.get("private_holdout_gate") if isinstance(bundle_doc, Mapping) else None
    gate_out: dict[str, Any] = {}
    if isinstance(gate, Mapping):
        gate_out = {
            "type": str(gate.get("gate_type") or ""),
            "decision": str(gate.get("decision") or ""),
            "candidate_public_score": round(_num(gate.get("candidate_public_score")), 4),
            "baseline_public_score": round(_num(gate.get("baseline_public_score")), 4),
        }

    aggregate_out = {
        "candidate_score": round(_num(aggregates.get("candidate_score")), 4),
        "base_score": round(_num(aggregates.get("base_score")), 4),
        "mean_delta": round(_num(aggregates.get("mean_delta")), 4),
        "delta_lcb": round(_num(aggregates.get("delta_lcb")), 4),
        "icp_count": total_icps,
        "gate": gate_out,
        "infra": {
            "excluded_icps": infra_total,
            "rerun_credit_eligible": infra_fraction > infra_rerun_fraction,
            "note": "ICPs killed by provider/runtime errors, not by your patch",
        },
    }

    private_pool = {
        "icp_count": priv_total,
        "helped": priv_helped,
        "hurt": priv_hurt,
        "flat": priv_flat,
        "infra_excluded": priv_infra,
    }

    scored_companies = {
        "count": scored_score_count,
        "avg_final_score": round(scored_score_sum / scored_score_count, 4) if scored_score_count else 0.0,
    }

    doc = {
        "schema_version": SCHEMA_VERSION,
        "candidate_id": str(candidate_id),
        "candidate_status": str(candidate_status or ""),
        "emitted_patch": emitted_patch,
        "aggregate": aggregate_out,
        "public_icps": public_icps,
        "private_pool": private_pool,
        "scored_companies": scored_companies,
    }
    _assert_clean(doc)
    return doc


def _assert_clean(doc: Mapping[str, Any]) -> None:
    """Fail closed: a diagnostics doc must never carry secret material."""
    if contains_secret_material(doc):
        raise ValueError("miner diagnostics doc failed secret-material check")
