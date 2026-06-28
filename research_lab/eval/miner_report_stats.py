"""Aggregate, company-anonymous statistics for the autoresearch benchmark.

Turns the per-company ``LeadScoreBreakdown`` list (and the raw model output
count) for ONE ICP into a counts-only ``dict`` that miners can be shown without
revealing (a) any company identity or (b) the sealed ICP criteria.

Design constraints baked in:
  * NO company names / domains / per-company rows ever leave this function —
    everything is reduced to counts, sums, and rates here.
  * Per-ICP output stays COARSE (funnel counts + score) so a single ICP's
    rejection mix can't be used to reverse-engineer its sealed filters. The
    rich breakdowns (reason histogram, evidence-type table, decay bands) are
    meant to be summed ACROSS all ICPs at the bundle level before display —
    see ``merge_icp_stats``.
  * "Returned 0 companies" (sourcing/provider crash) is tracked SEPARATELY from
    "returned companies that were rejected" — infra failure must never be shown
    as miner-attributable skill loss.

Inputs come straight from the existing eval path:
  * ``sourced_count``  = len(model output companies) BEFORE the scorer's
    employee-bucket pre-filter (evaluator.score_with_breakdowns drops some with
    ``continue``; pass the pre-filter count so the funnel's first stage is real).
  * ``breakdowns``     = list of LeadScoreBreakdown.model_dump() dicts for the
    companies that entered scoring.
  * Optional ``signal_details`` per breakdown (added later in lead_scorer):
    list of {evidence_type, raw, after_decay, date_status}. If absent, the
    evidence-type and decay panels are simply empty (graceful degrade).

The funnel stages, in order, are derived from failure_reason text because the
scorer fails closed at distinct points (pre-checks -> binary fit -> existence
verification -> intent fabrication/scoring).
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence, Optional


# --- failure_reason -> coarse category (keep this list in sync with the
#     strings produced in qualification/scoring/lead_scorer.py) -------------
# Matching is substring + case-insensitive so we tolerate the appended detail
# (e.g. "Employee count mismatch: '...' not in {...}").
_REASON_CATEGORIES: list[tuple[str, str]] = [
    ("employee count mismatch", "employee_count_mismatch"),
    ("missing employee_count", "employee_count_missing"),
    ("company stage mismatch", "company_stage_mismatch"),
    ("missing company_stage", "company_stage_missing"),
    ("company verification failed", "company_unverifiable"),
    ("company verification error", "company_unverifiable"),
    ("fabricat", "intent_fabricated"),          # "Intent fabrication detected ..."
    ("llm scoring error", "scoring_error"),
    ("pre-check", "failed_prechecks"),
    ("duplicate", "duplicate_company"),
]

# Decay multiplier -> human band (mirrors calculate_time_decay_multiplier:
# 1.0 = <=2mo, 0.5 = <=12mo or undated, 0.25 = >12mo).
_DECAY_BANDS = {1.0: "fresh_<=2mo", 0.5: "aging_2-12mo_or_undated", 0.25: "stale_>12mo"}


def _categorize_reason(reason: Optional[str]) -> str:
    r = (reason or "").strip().lower()
    if not r:
        return "other"
    for needle, cat in _REASON_CATEGORIES:
        if needle in r:
            return cat
    return "other"


def _decay_band(mult: Any) -> str:
    try:
        m = float(mult)
    except (TypeError, ValueError):
        return "unknown"
    # snap to nearest known tier
    best = min(_DECAY_BANDS, key=lambda k: abs(k - m))
    return _DECAY_BANDS[best] if abs(best - m) < 0.13 else "unknown"


def build_icp_stats(
    *,
    sourced_count: int,
    breakdowns: Sequence[Mapping[str, Any]],
    signal_details: Optional[Sequence[Sequence[Mapping[str, Any]]]] = None,
) -> dict[str, Any]:
    """Reduce one ICP's scoring result to a counts-only stats dict.

    ``signal_details[i]`` (optional) is the per-signal evidence list for
    ``breakdowns[i]``. When omitted, evidence-type / decay panels are empty.
    """
    n_scored_input = len(breakdowns)

    # --- Infra vs skill: 0 companies returned == sourcing/provider failure ---
    if sourced_count == 0 and n_scored_input == 0:
        return {
            "sourcing_failed": True,          # infra (e.g. ScrapingDog) — NOT miner skill
            "funnel": {"sourced": 0, "fit_pass": 0, "verified": 0,
                       "intent_valid": 0, "scored": 0},
            "rejection_reasons": {},
            "evidence_types": {},
            "decay_bands": {},
            "per_signal": {},
            "intent": {"avg_raw": 0.0, "avg_final": 0.0, "decay_loss": 0.0},
            "scores": {"count": 0, "avg": 0.0, "max": 0.0, "sum": 0.0},
        }

    # --- Funnel: derive each company's terminal stage from failure_reason ----
    # A company is "scored" if final_score > 0 and no failure_reason.
    # Companies that failed map to the stage their reason category implies.
    fit_fail = {"employee_count_mismatch", "employee_count_missing",
                "company_stage_mismatch", "company_stage_missing"}
    verify_fail = {"company_unverifiable"}
    intent_fail = {"intent_fabricated", "scoring_error"}
    precheck_fail = {"failed_prechecks", "duplicate_company"}

    reason_counter: Counter[str] = Counter()
    scored = verified = intent_valid = fit_pass = 0
    raws: list[float] = []
    finals: list[float] = []
    score_vals: list[float] = []

    for b in breakdowns:
        final = float(b.get("final_score") or 0.0)
        cat = _categorize_reason(b.get("failure_reason"))
        if final > 0 and not b.get("failure_reason"):
            scored += 1
            fit_pass += 1
            verified += 1
            intent_valid += 1
            raws.append(float(b.get("intent_signal_raw") or 0.0))
            finals.append(float(b.get("intent_signal_final") or 0.0))
            score_vals.append(final)
        else:
            reason_counter[cat] += 1
            # passed earlier stages up to where it died
            if cat in fit_fail or cat in precheck_fail:
                pass                                   # died at/ before fit
            elif cat in verify_fail:
                fit_pass += 1                          # passed fit, failed verify
            elif cat in intent_fail:
                fit_pass += 1
                verified += 1                          # passed verify, failed intent

    # The scorer pre-filters some sourced companies on employee bucket BEFORE
    # they reach `breakdowns` (evaluator.py `continue`). Those are real fit
    # failures — represent them as the sourced->entered gap.
    prefiltered = max(0, sourced_count - n_scored_input)
    if prefiltered:
        reason_counter["employee_count_mismatch"] += prefiltered

    funnel = {
        "sourced": sourced_count,
        "fit_pass": fit_pass,
        "verified": verified,
        "intent_valid": intent_valid,
        "scored": scored,
    }

    # --- Evidence-type table + decay bands (need per-signal details) ---------
    # Also build the PER-SIGNAL panel keyed by the ICP signal index each piece
    # of evidence was matched to (``matched_icp_signal``). This answers, for one
    # ICP: how many distinct companies covered each buyer-requested signal
    # ("discovered" = submitted any evidence; "passed" = had evidence that
    # survived verification + decay), and the score earned per signal.
    ev_counter: Counter[str] = Counter()
    ev_score_sum: Counter[str] = Counter()
    ev_fresh: Counter[str] = Counter()
    ev_company_pass: Counter[str] = Counter()
    decay_counter: Counter[str] = Counter()

    # Per-signal accumulators (key = matched_icp_signal index as int).
    sig_evidence_type: dict[int, str] = {}
    sig_submitted: Counter[int] = Counter()        # rows submitted to this signal
    sig_passed: Counter[int] = Counter()           # rows with after_decay > 0
    sig_score_sum: dict[int, float] = {}
    sig_score_max: dict[int, float] = {}
    sig_companies_submitted: dict[int, set] = {}
    sig_companies_passed: dict[int, set] = {}

    if signal_details:
        for company_idx, sigs in enumerate(signal_details):
            ev_seen_pass: set[str] = set()
            for s in (sigs or []):
                et = str(s.get("evidence_type") or "UNSPECIFIED").upper()
                after_decay = float(s.get("after_decay") or 0.0)
                ev_counter[et] += 1
                ev_score_sum[et] += after_decay
                band = _decay_band(s.get("decay"))
                decay_counter[band] += 1
                if band == "fresh_<=2mo":
                    ev_fresh[et] += 1
                if after_decay > 0 and et not in ev_seen_pass:
                    ev_company_pass[et] += 1
                    ev_seen_pass.add(et)

                # Per-signal-index bookkeeping.
                try:
                    idx = int(s.get("matched_icp_signal", -1))
                except (TypeError, ValueError):
                    idx = -1
                sig_submitted[idx] += 1
                sig_evidence_type.setdefault(idx, et)
                sig_score_sum[idx] = sig_score_sum.get(idx, 0.0) + after_decay
                sig_score_max[idx] = max(sig_score_max.get(idx, 0.0), after_decay)
                sig_companies_submitted.setdefault(idx, set()).add(company_idx)
                if after_decay > 0:
                    sig_passed[idx] += 1
                    sig_companies_passed.setdefault(idx, set()).add(company_idx)

    evidence_types = {
        et: {
            "signals": ev_counter[et],
            "companies_passed": ev_company_pass[et],
            "avg_after_decay": round(ev_score_sum[et] / ev_counter[et], 2) if ev_counter[et] else 0.0,
            "fresh_rate": round(ev_fresh[et] / ev_counter[et], 3) if ev_counter[et] else 0.0,
        }
        for et in ev_counter
    }

    per_signal = {
        str(idx): {
            "signal_index": idx,
            "evidence_type": sig_evidence_type.get(idx, "UNSPECIFIED"),
            "companies_submitted": len(sig_companies_submitted.get(idx, ())),
            "companies_passed": len(sig_companies_passed.get(idx, ())),
            "signals_submitted": sig_submitted[idx],
            "signals_passed": sig_passed[idx],
            "avg_score": round(sig_score_sum.get(idx, 0.0) / sig_passed[idx], 2) if sig_passed[idx] else 0.0,
            "sum_score": round(sig_score_sum.get(idx, 0.0), 2),
            "max_score": round(sig_score_max.get(idx, 0.0), 2),
        }
        for idx in sorted(sig_submitted)
    }

    avg_raw = round(sum(raws) / len(raws), 2) if raws else 0.0
    avg_final = round(sum(finals) / len(finals), 2) if finals else 0.0

    return {
        "sourcing_failed": False,
        "funnel": funnel,
        "rejection_reasons": dict(reason_counter),
        "evidence_types": evidence_types,
        "per_signal": per_signal,
        "decay_bands": dict(decay_counter),
        "intent": {
            "avg_raw": avg_raw,
            "avg_final": avg_final,
            "decay_loss": round(avg_raw - avg_final, 2),   # points lost to staleness
        },
        "scores": {
            "count": len(score_vals),
            "avg": round(sum(score_vals) / len(score_vals), 2) if score_vals else 0.0,
            "max": round(max(score_vals), 2) if score_vals else 0.0,
            "sum": round(sum(score_vals), 2),
        },
    }


def merge_icp_stats(per_icp_stats: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Sum per-ICP stats into the bundle-level miner report (counts only).

    Per-ICP rows stay coarse; the rich panels (reasons / evidence / decay) are
    only meaningful — and only safe from ICP-criteria leakage — once summed
    across ALL ICPs here.
    """
    funnel = Counter()
    reasons = Counter()
    decay = Counter()
    ev_signals: Counter[str] = Counter()
    ev_score_sum: Counter[str] = Counter()
    ev_fresh: Counter[str] = Counter()
    ev_company_pass: Counter[str] = Counter()
    raw_sum = final_sum = raw_n = 0.0
    score_sum = score_n = 0.0
    score_max = 0.0
    sourcing_failed_icps = 0
    n_icps = 0

    for st in per_icp_stats:
        n_icps += 1
        if st.get("sourcing_failed"):
            sourcing_failed_icps += 1
            continue
        for k, v in (st.get("funnel") or {}).items():
            funnel[k] += v
        reasons.update(st.get("rejection_reasons") or {})
        decay.update(st.get("decay_bands") or {})
        for et, d in (st.get("evidence_types") or {}).items():
            n = d.get("signals", 0)
            ev_signals[et] += n
            ev_score_sum[et] += d.get("avg_after_decay", 0.0) * n
            ev_fresh[et] += round(d.get("fresh_rate", 0.0) * n)
            ev_company_pass[et] += d.get("companies_passed", 0)
        intent = st.get("intent") or {}
        sc = st.get("scores") or {}
        if sc.get("count"):
            raw_sum += intent.get("avg_raw", 0.0) * sc["count"]
            final_sum += intent.get("avg_final", 0.0) * sc["count"]
            raw_n += sc["count"]
        score_sum += sc.get("sum", 0.0)
        score_n += sc.get("count", 0)
        score_max = max(score_max, sc.get("max", 0.0))

    evidence_types = {
        et: {
            "signals": ev_signals[et],
            "companies_passed": ev_company_pass[et],
            "avg_after_decay": round(ev_score_sum[et] / ev_signals[et], 2) if ev_signals[et] else 0.0,
            "fresh_rate": round(ev_fresh[et] / ev_signals[et], 3) if ev_signals[et] else 0.0,
        }
        for et in ev_signals
    }

    return {
        "icps_total": n_icps,
        "icps_sourcing_failed_infra": sourcing_failed_icps,   # exclude from skill view
        "funnel": dict(funnel),
        "rejection_reasons": dict(reasons.most_common()),
        "evidence_types": evidence_types,
        "decay_bands": dict(decay),
        "intent": {
            "avg_raw": round(raw_sum / raw_n, 2) if raw_n else 0.0,
            "avg_final": round(final_sum / raw_n, 2) if raw_n else 0.0,
            "decay_loss": round((raw_sum - final_sum) / raw_n, 2) if raw_n else 0.0,
        },
        "scores": {
            "scored_companies": int(score_n),
            "avg": round(score_sum / score_n, 2) if score_n else 0.0,
            "max": round(score_max, 2),
            "sum": round(score_sum, 2),
        },
    }
