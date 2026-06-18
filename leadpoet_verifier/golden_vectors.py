"""Golden-vector runner for the open verifier package."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import date, datetime

from .aggregation import (
    aggregate_set_score,
    apply_signal_time_decay,
    company_final_score,
    per_icp_normalized_score,
    source_adjusted_intent_score,
    u16_weights_from_scores,
)
from .attestation import (
    is_pcr0_allowed,
    load_pcr0_allowlist,
    validate_attestation_response_shape,
)
from .crowning import (
    attacker_expected_value,
    ceiling_utilization,
    check_grant_curve_shape,
    compound_reentry_probability,
    derive_delta_min,
    evaluate_crowning,
    lower_confidence_bound,
    operating_characteristic_curve,
    simulate_sequential_crown_probability,
)
from .l0 import run_l0_checks
from .static_checks import run_static_gaming_checks


DEFAULT_FIXTURE = Path(__file__).with_name("fixtures").joinpath("golden_vectors.json")


def _round_public(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, list):
        return [_round_public(item) for item in value]
    if isinstance(value, dict):
        return {key: _round_public(item) for key, item in value.items()}
    return value


def load_golden_vectors(path: Optional[str] = None) -> Dict[str, Any]:
    fixture_path = Path(path) if path else DEFAULT_FIXTURE
    with fixture_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def run_golden_vectors(
    *,
    fixture_path: Optional[str] = None,
    pcr0_allowlist_path: Optional[str] = None,
) -> List[str]:
    fixture = load_golden_vectors(fixture_path)
    errors: List[str] = []

    today = date.fromisoformat(fixture["clock"]["today"])
    now = datetime.fromisoformat(fixture["clock"]["now"].replace("Z", "+00:00"))

    for case in fixture.get("l0_cases", []):
        result = run_l0_checks(case["signal"], case["snapshot"], today=today, now=now)
        actual_ids = [finding.check_id for finding in result.findings if finding.severity == "fail"]
        expected_ids = case["expected"]["fail_check_ids"]
        if result.passed != case["expected"]["passed"]:
            errors.append(f"l0 {case['id']}: passed {result.passed} != {case['expected']['passed']}")
        if actual_ids != expected_ids:
            errors.append(f"l0 {case['id']}: fail ids {actual_ids} != {expected_ids}")
        for key, expected_value in case["expected"].get("metrics", {}).items():
            actual_value = result.metrics.get(key)
            if actual_value != expected_value:
                errors.append(f"l0 {case['id']}: metric {key} {actual_value!r} != {expected_value!r}")

    for case in fixture.get("aggregation_cases", []):
        if case["kind"] == "source_adjusted_intent_score":
            actual = round(source_adjusted_intent_score(case["raw_score"], case["source"]), 6)
        elif case["kind"] == "apply_signal_time_decay":
            decayed, decay = apply_signal_time_decay(
                case["raw_score"],
                case.get("signal_date"),
                case["date_status"],
                case["source"],
                case.get("content_found_date"),
                today=today,
                decay_50_pct_months=case.get("decay_50_pct_months", 2),
                decay_25_pct_months=case.get("decay_25_pct_months", 12),
            )
            actual = [round(decayed, 6), round(decay, 6)]
        elif case["kind"] == "company_final_score":
            actual = company_final_score(
                case["icp_fit"],
                case["intent_signal_final"],
                run_cost_usd=case["run_cost_usd"],
                cost_penalty_threshold=case["cost_penalty_threshold"],
                variability_penalty_points=case["variability_penalty_points"],
                is_reference_model=case.get("is_reference_model", False),
            )
            actual = {key: round(value, 6) for key, value in actual.items()}
        elif case["kind"] == "per_icp_normalized_score":
            actual = round(per_icp_normalized_score(case["lead_scores"]), 6)
        elif case["kind"] == "aggregate_set_score":
            actual = round(aggregate_set_score(case["per_icp_scores"]), 6)
        elif case["kind"] == "u16_weights_from_scores":
            actual = u16_weights_from_scores({int(k): v for k, v in case["scores_by_uid"].items()})
            actual = {str(k): v for k, v in actual.items()}
        else:
            errors.append(f"aggregation {case['id']}: unknown kind {case['kind']}")
            continue

        if actual != case["expected"]:
            errors.append(f"aggregation {case['id']}: {actual!r} != {case['expected']!r}")

    for case in fixture.get("static_cases", []):
        result = run_static_gaming_checks(case["code"])
        if result.passed != case["expected"]["passed"]:
            errors.append(f"static {case['id']}: passed {result.passed} != {case['expected']['passed']}")
        if result.confidence != case["expected"]["confidence"]:
            errors.append(
                f"static {case['id']}: confidence {result.confidence} != {case['expected']['confidence']}"
            )

    for case in fixture.get("crowning_cases", []):
        if case["kind"] == "derive_delta_min":
            actual = derive_delta_min(
                case["within_day_paired_deltas"],
                temporal_buffer_b=case.get("temporal_buffer_b", 0.0),
                paired_days_for_b=case.get("paired_days_for_b", 0),
                truth_supply_ready=case.get("truth_supply_ready", False),
            )
        elif case["kind"] == "lower_confidence_bound":
            actual = lower_confidence_bound(
                case["deltas"],
                use_alpha_spending=case.get("use_alpha_spending", True),
            )
        elif case["kind"] == "evaluate_crowning":
            actual = evaluate_crowning(
                case["challenger_scores"],
                case["champion_scores"],
                public_baseline_score=case["public_baseline_score"],
                delta_min=case["delta_min"],
                canary_fabrication_confirmed=case.get("canary_fabrication_confirmed", False),
                challenger_mean=case.get("challenger_mean"),
                use_alpha_spending=case.get("use_alpha_spending", True),
            )
        elif case["kind"] == "operating_characteristic_curve":
            actual = operating_characteristic_curve(
                case["true_deltas"],
                paired_sd=case["paired_sd"],
                delta_min=case["delta_min"],
                l2_entry_probability=case.get("l2_entry_probability", 1.0),
                reentry_attempts=case.get("reentry_attempts", 3),
                use_alpha_spending=case.get("use_alpha_spending", True),
            )
        elif case["kind"] == "compound_reentry_probability":
            actual = compound_reentry_probability(
                case["per_attempt_probability"],
                attempts=case.get("attempts", 3),
            )
        elif case["kind"] == "simulate_sequential_crown_probability":
            actual = simulate_sequential_crown_probability(
                case["true_delta"],
                paired_sd=case["paired_sd"],
                delta_min=case["delta_min"],
                iterations=case.get("iterations", 20000),
                seed=case.get("seed", 8675309),
                use_alpha_spending=case.get("use_alpha_spending", True),
            )
        elif case["kind"] == "attacker_expected_value":
            actual = attacker_expected_value(
                p_crown=case["p_crown"],
                grant_value=case["grant_value"],
                loops_consumed=case["loops_consumed"],
                loop_price_usd=case.get("loop_price_usd", 10.0),
                probation_cost_usd=case.get("probation_cost_usd", 0.0),
            )
        elif case["kind"] == "ceiling_utilization":
            actual = ceiling_utilization(
                aggregate_funded_loop_spend=case["aggregate_funded_loop_spend"],
                total_crown_ev=case["total_crown_ev"],
            )
        elif case["kind"] == "check_grant_curve_shape":
            actual = check_grant_curve_shape(
                case["points"],
                typical_winning_spend=case["typical_winning_spend"],
            )
        else:
            errors.append(f"crowning {case['id']}: unknown kind {case['kind']}")
            continue

        actual = _round_public(actual)
        expected = _round_public(case["expected"])
        if actual != expected:
            errors.append(f"crowning {case['id']}: {actual!r} != {expected!r}")

    allowlist = None
    if pcr0_allowlist_path:
        allowlist = load_pcr0_allowlist(pcr0_allowlist_path)
    for case in fixture.get("attestation_cases", []):
        shape = validate_attestation_response_shape(case["response"])
        if shape["passed"] != case["expected"]["shape_passed"]:
            errors.append(
                f"attestation {case['id']}: shape {shape['passed']} != {case['expected']['shape_passed']}"
            )
        if allowlist and "pcr0_allowed" in case["expected"]:
            actual = is_pcr0_allowed(case["response"].get("pcr0", ""), allowlist, role=case["role"])
            if actual != case["expected"]["pcr0_allowed"]:
                errors.append(
                    f"attestation {case['id']}: pcr0_allowed {actual} != {case['expected']['pcr0_allowed']}"
                )

    return errors
