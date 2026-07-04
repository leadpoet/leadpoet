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

import math
from collections import Counter
from typing import Any, Mapping, Optional, Sequence

from gateway.research_lab.bundles import contains_secret_material, redact_secret_material


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
    if not _is_list_like(items):
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


_PROVIDER_DISPLAY = {"scrapingdog": "ScrapingDog", "exa": "Exa", "openrouter": "OpenRouter"}


def _provider_error_detail_from_row(row: Mapping[str, Any]) -> dict[str, Any] | None:
    """The evaluator's parsed provider error for this ICP, if present + safe.

    Carries no company/ICP content — only provider host, endpoint path, HTTP
    status, a kind label, and a coarse fault attribution — so it is safe to
    surface (including in whole-pool private aggregates)."""
    detail = row.get("provider_error_detail")
    if not isinstance(detail, Mapping):
        return None
    out: dict[str, Any] = {}
    provider = str(detail.get("provider") or "").strip().lower()
    if provider:
        out["provider"] = provider
    endpoint = str(detail.get("endpoint") or "").strip()
    if endpoint:
        out["endpoint"] = endpoint[:60]
    status = detail.get("status")
    if isinstance(status, (int, float)) and 100 <= int(status) <= 599:
        out["status"] = int(status)
    kind = str(detail.get("kind") or "").strip()
    if kind:
        out["kind"] = kind[:40]
    fault = str(detail.get("fault") or "").strip()
    if fault in {"provider", "request"}:
        out["fault"] = fault
    return out or None


def _provider_error_label(detail: Mapping[str, Any]) -> str:
    """Human string, e.g. 'ScrapingDog /google/ai_mode — HTTP 410 (timeout)'."""
    provider = _PROVIDER_DISPLAY.get(str(detail.get("provider") or ""), str(detail.get("provider") or "provider"))
    parts = [provider]
    if detail.get("endpoint"):
        parts.append(str(detail["endpoint"]))
    tail = ""
    if detail.get("status"):
        tail = f"HTTP {detail['status']}"
        if detail.get("kind"):
            tail += f" ({detail['kind']})"
    label = " ".join(parts)
    return f"{label} — {tail}" if tail else label


def _provider_error_key(detail: Mapping[str, Any]) -> str:
    return ":".join(
        str(detail.get(k))
        for k in ("provider", "status", "kind")
        if detail.get(k)
    ) or "unknown"


def _is_list_like(value: Any) -> bool:
    """A real sequence of items — a list/tuple, but NOT a str/bytes (which are
    Sequences and would otherwise iterate as characters)."""
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _num(value: Any, default: float = 0.0) -> float:
    """Coerce to a FINITE float. Non-finite (NaN/inf) → default, so the
    diagnostics doc is always strict-JSON-serializable (json.dumps otherwise
    emits bare NaN/Infinity tokens that non-Python clients reject)."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(result):
        return default
    return result


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
    out = {
        "target_component": _safe_text(patch_manifest.get("target_component_id")),
        "patch_type": _safe_text(patch_manifest.get("patch_type")),
        "candidate_kind": _safe_text(patch_manifest.get("candidate_kind")),
        "lane": _safe_text(lane),
        "redacted_summary": _safe_text(patch_manifest.get("redacted_summary")),
    }
    # An empty/absent manifest yields all-blank fields — omit the block entirely
    # rather than hand the miner a hollow emitted_patch.
    return out if any(out.values()) else {}


def build_candidate_diagnostics(
    *,
    candidate_id: str,
    bundle_doc: Mapping[str, Any] | None,
    patch_manifest: Mapping[str, Any] | None,
    visibility_by_ref: Mapping[str, str] | None,
    candidate_status: str = "",
    rejection_reason: str = "",
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

    The output is redacted through ``redact_secret_material`` before return, so
    a stray secret marker is scrubbed rather than raising (fail-safe, no 500).
    """
    vis = {str(k): str(v).lower() for k, v in (visibility_by_ref or {}).items()}

    def _visibility(ref: str) -> str:
        # Default PRIVATE: an unclassified ICP must never be treated as public.
        return "public" if vis.get(ref) == "public" else "private"

    emitted_patch = _emitted_patch(patch_manifest)

    aggregates = bundle_doc.get("aggregates") if isinstance(bundle_doc, Mapping) else None
    raw_per_icp = aggregates.get("per_icp_results") if isinstance(aggregates, Mapping) else None
    # NB: str/bytes are Sequences too — exclude them so a stray string can't be
    # iterated as characters and inflate icp_count.
    per_icp = raw_per_icp if _is_list_like(raw_per_icp) else []

    # --- Reduced doc: candidate produced no scored bundle (rejected before
    # scoring, or runtime death). rejection_reason is the key signal here — many
    # such rejections are lineage/infra ("stale_parent_*"), NOT a bad patch, so
    # the miner must see WHY before abandoning a direction.
    if not per_icp:
        doc = {
            "schema_version": SCHEMA_VERSION,
            "candidate_id": str(candidate_id),
            "candidate_status": str(candidate_status or ""),
            "rejection_reason": _safe_text(rejection_reason),
            "scored": False,
            "emitted_patch": emitted_patch,
            "aggregate": {},
            "public_icps": [],
            "private_pool": {},
            "scored_companies": {"count": 0, "avg_final_score": 0.0},
            "note": "This candidate was not scored (rejected before scoring or runtime failure); see rejection_reason.",
        }
        return _finalize(doc)

    # bundle_doc is guaranteed a Mapping here (per_icp is non-empty above).
    raw_excluded = bundle_doc.get("provider_excluded_icp_ids")
    provider_excluded = {str(x) for x in raw_excluded} if isinstance(raw_excluded, Sequence) and not isinstance(raw_excluded, str) else set()

    public_icps: list[dict[str, Any]] = []
    priv_helped = priv_hurt = priv_flat = priv_infra = priv_total = 0
    pub_infra = 0
    scored_score_sum = 0.0
    scored_score_count = 0
    # Whole-run provider-error tally (both visibilities). Aggregate only, so
    # naming the provider never identifies a sealed ICP.
    provider_error_counts: dict[str, int] = {}
    provider_error_labels: dict[str, str] = {}

    for row in per_icp:
        if not isinstance(row, Mapping):
            continue
        ref = str(row.get("icp_ref") or "")
        is_infra = _is_infra_row(row, provider_excluded)
        pe_detail = _provider_error_detail_from_row(row)
        if pe_detail:
            key = _provider_error_key(pe_detail)
            provider_error_counts[key] = provider_error_counts.get(key, 0) + 1
            provider_error_labels.setdefault(key, _provider_error_label(pe_detail))
        delta = _num(row.get("delta_vs_base"))
        cand_score = _num(row.get("candidate_per_icp_score"))
        base_score = _num(row.get("base_per_icp_score"))

        # scored-company stats (count + avg ONLY — never the raw vector)
        raw_scores = row.get("candidate_company_scores")
        for s in (raw_scores if _is_list_like(raw_scores) else []):
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
                if pe_detail:
                    # Name the actual provider error, e.g.
                    # "ScrapingDog /google/ai_mode — HTTP 410 (timeout)".
                    entry["provider_error"] = pe_detail
                    fault_note = (
                        "provider was down — not attributable to your patch"
                        if pe_detail.get("fault") == "provider"
                        else "the request your model sent failed (e.g. profile not found)"
                    )
                    entry["note"] = f"{_provider_error_label(pe_detail)} — {fault_note}"
                else:
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
            # Whole-run breakdown of which provider errors occurred, e.g.
            # [{"error":"ScrapingDog /google/ai_mode — HTTP 410 (timeout)",
            #   "count":6,"fault":"provider"}] — lets a miner see "the provider
            # timed out" vs "my model queried a bad slug (404)".
            "provider_errors": [
                {
                    "error": provider_error_labels.get(key, key),
                    "count": count,
                    "fault": "request" if ":404:" in key or ":401:" in key or ":403:" in key else "provider",
                }
                for key, count in sorted(provider_error_counts.items(), key=lambda kv: -kv[1])
            ],
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
        "avg_final_score": round(_num(scored_score_sum / scored_score_count), 4) if scored_score_count else 0.0,
    }

    doc = {
        "schema_version": SCHEMA_VERSION,
        "candidate_id": str(candidate_id),
        "candidate_status": str(candidate_status or ""),
        "rejection_reason": _safe_text(rejection_reason),
        "scored": True,
        "emitted_patch": emitted_patch,
        "aggregate": aggregate_out,
        "public_icps": public_icps,
        "private_pool": private_pool,
        "scored_companies": scored_companies,
    }
    return _finalize(doc)


def _finalize(doc: dict[str, Any]) -> dict[str, Any]:
    """Fail-SAFE backstop: redact any residual secret-marker strings so the doc
    is ALWAYS returnable to the miner — never raise (a stray marker in a benign
    field must not 500 the request), never leak. The redacted copy passes
    ``contains_secret_material`` by construction.
    """
    redacted, _summary = redact_secret_material(doc)
    return redacted
