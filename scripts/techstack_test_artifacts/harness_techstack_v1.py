"""TECHSTACK validator test harness — Stage 0 (precheck) + Stage 1+3 LLM.

Reads test_cases_v1.jsonl and runs each case through:

  1. Classifier check (when expected_classifier set) — verifies the
     regex classifier picks the right type for the signal text
  2. URL precheck — verifies _check_intent_url_evidence_quality returns
     the expected verdict + reason
  3. LLM substance check (for llm_only cases) — calls verify_three_stage
     with evidence_type=TECHSTACK and verifies the final signal_status

Reports FP / FN counts at each stage.  Acceptance bar: 0 FP / 0 FN on
ALL non-llm_only cases (deterministic) and 0 FP on llm_only cases (Sonar
variance can produce limited FN — re-run to confirm).
"""
import os
import sys
import json
import asyncio
import traceback
from typing import Any, Dict, List, Optional

sys.path.insert(0, "/Users/tasnimul/Desktop/leadpoet")

from dotenv import load_dotenv

load_dotenv("/Users/tasnimul/Desktop/leadpoet/.env")

from qualification.scoring.intent_precheck import (  # noqa: E402
    _classify_target_type,
    _check_intent_url_evidence_quality,
)


CASES_PATH = (
    "/Users/tasnimul/Desktop/leadpoet/scripts/techstack_test_artifacts/test_cases_v1.jsonl"
)


def load_cases() -> List[Dict[str, Any]]:
    cases = []
    with open(CASES_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def run_classifier_check(case: Dict[str, Any]) -> Dict[str, Any]:
    """When case has expected_classifier, verify regex picks that type."""
    text = case.get("target_signal_text") or case.get("claim") or ""
    actual = _classify_target_type(text)
    expected = case["expected_classifier"]
    return {
        "stage": "classifier",
        "case_id": case["case_id"],
        "expected": expected,
        "actual": actual,
        "passed": actual == expected,
    }


def run_precheck(case: Dict[str, Any]) -> Dict[str, Any]:
    """Verify URL precheck returns expected verdict + reason substring."""
    expected_verdict = case["expected_precheck_verdict"]
    expected_reason = case.get("expected_precheck_reason_contains", "")
    text = case.get("target_signal_text") or ""
    # Force target_type to the classifier output (or fallback to spec's
    # classifier expectation if classifier_only test).
    target_type = (
        case.get("expected_classifier")
        or _classify_target_type(text)
        or "TECHSTACK"
    )
    verdict, reason = _check_intent_url_evidence_quality(
        case.get("url", ""), target_type,
    )
    reason_ok = (not expected_reason) or expected_reason in reason
    return {
        "stage": "precheck",
        "case_id": case["case_id"],
        "url": case.get("url", "")[:80],
        "target_type": target_type,
        "expected_verdict": expected_verdict,
        "actual_verdict": verdict,
        "actual_reason": reason,
        "passed": (verdict == expected_verdict) and reason_ok,
    }


async def run_llm_substance(case: Dict[str, Any]) -> Dict[str, Any]:
    """Call verify_three_stage for the case and check signal_status."""
    import httpx
    from qualification.scoring.intent_verification_three_stage import (
        verify_three_stage,
    )
    try:
        async with httpx.AsyncClient() as client:
            result = await verify_three_stage(
                client,
                company_name=case["lead_company"],
                company_linkedin="",
                company_website=case["lead_website"],
                source_url=case["url"],
                miner_claim=case.get("claim", ""),
                target_signal_text=case["target_signal_text"],
                evidence_type="TECHSTACK",
            )
        decision = result.get("decision")
        s1 = (result.get("stage1") or {}).get("status")
        s3 = (result.get("stage3") or {}).get("status")
        return {
            "stage": "llm_substance",
            "case_id": case["case_id"],
            "expected": case["expected_signal_status"],
            "actual_decision": decision,
            "actual_s1_status": s1,
            "actual_s3_status": s3,
            "rejection_reason": result.get("rejection_reason", ""),
            "passed": _llm_match(decision, s1, s3, case["expected_signal_status"]),
        }
    except Exception as e:
        return {
            "stage": "llm_substance",
            "case_id": case["case_id"],
            "expected": case["expected_signal_status"],
            "actual": "ERROR",
            "error": f"{type(e).__name__}: {e}",
            "passed": False,
        }


def _llm_match(decision: str, s1: str, s3: str, expected: str) -> bool:
    """Map the expected status from the test case to the three-stage
    pipeline's outputs.

    Three-stage outputs (from intent_verification_three_stage.py):
      decision ∈ {approve, reject, review}
      stageN.status ∈ {supported, contradicted, wrong_entity,
                       unable_to_verify, llm_error, ...}
    """
    # Final status: prefer stage3 status if stage3 ran; else stage1.
    final = (s3 or s1 or "").lower()
    expected_l = (expected or "").lower()
    if expected_l == "supported":
        return decision == "approve" or final == "supported"
    if expected_l == "contradicted":
        return final == "contradicted" or decision == "reject"
    if expected_l == "wrong_entity":
        return final == "wrong_entity"
    if expected_l == "unable_to_verify":
        return final == "unable_to_verify" or decision == "reject"
    return False


async def main():
    cases = load_cases()
    print(f"Loaded {len(cases)} cases from {CASES_PATH}")
    print()

    classifier_results = []
    precheck_results = []
    llm_results = []
    skipped = 0

    for case in cases:
        # Stage A — classifier check (only when expected_classifier set)
        if "expected_classifier" in case:
            r = run_classifier_check(case)
            classifier_results.append(r)
            if not r["passed"]:
                print(f"  [CLASSIFIER FAIL] {r}")

        # Stage B — URL precheck (always when expected_precheck_verdict set)
        if "expected_precheck_verdict" in case:
            r = run_precheck(case)
            precheck_results.append(r)
            if not r["passed"]:
                print(f"  [PRECHECK FAIL] {r}")

        # Stage C — LLM substance (only when llm_only=true)
        if case.get("llm_only"):
            print(f"  [LLM] running {case['case_id']}...", flush=True)
            r = await run_llm_substance(case)
            llm_results.append(r)
            if not r["passed"]:
                print(f"  [LLM FAIL] {r}")

    print()
    print("=" * 70)
    print(f"CLASSIFIER:  {sum(r['passed'] for r in classifier_results)}/{len(classifier_results)} passed")
    print(f"PRECHECK:    {sum(r['passed'] for r in precheck_results)}/{len(precheck_results)} passed")
    print(f"LLM:         {sum(r['passed'] for r in llm_results)}/{len(llm_results)} passed")
    print()

    # FP / FN at deterministic stages
    classifier_fp = sum(
        1 for r in classifier_results
        if not r["passed"] and r["expected"] != r["actual"]
    )
    precheck_fp = sum(
        1 for r in precheck_results
        if not r["passed"] and r["expected_verdict"] == "pass"
        and r["actual_verdict"] == "reject"
    )
    precheck_fn = sum(
        1 for r in precheck_results
        if not r["passed"] and r["expected_verdict"] == "reject"
        and r["actual_verdict"] == "pass"
    )
    llm_fp = sum(
        1 for r in llm_results
        if not r["passed"] and r["expected"] != "supported"
        and r.get("actual_decision") == "approve"
    )
    llm_fn = sum(
        1 for r in llm_results
        if not r["passed"] and r["expected"] == "supported"
        and r.get("actual_decision") != "approve"
    )

    print("--- FP / FN counts ---")
    print(f"  Classifier  FP: {classifier_fp}")
    print(f"  Precheck    FP: {precheck_fp}     FN: {precheck_fn}")
    print(f"  LLM         FP: {llm_fp}          FN: {llm_fn}")
    print()

    total_fp = classifier_fp + precheck_fp + llm_fp
    total_fn = precheck_fn + llm_fn
    print(f"TOTAL FP: {total_fp}     TOTAL FN: {total_fn}")
    if total_fp == 0 and total_fn == 0:
        print("\n*** 0 FP / 0 FN — PRODUCTION READY ***")
    else:
        print("\n*** FAILURES PRESENT — iterate before deploy ***")


if __name__ == "__main__":
    asyncio.run(main())
