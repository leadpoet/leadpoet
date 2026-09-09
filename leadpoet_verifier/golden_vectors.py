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
from .l0 import run_l0_checks


DEFAULT_FIXTURE = Path(__file__).with_name("fixtures").joinpath("golden_vectors.json")


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
            actual = u16_weights_from_scores(
                {int(k): v for k, v in case["scores_by_uid"].items()},
                total_weight=int(case.get("total_weight", 65535)),
            )
            actual = {str(k): v for k, v in actual.items()}
        else:
            errors.append(f"aggregation {case['id']}: unknown kind {case['kind']}")
            continue

        if actual != case["expected"]:
            errors.append(f"aggregation {case['id']}: {actual!r} != {case['expected']!r}")

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
