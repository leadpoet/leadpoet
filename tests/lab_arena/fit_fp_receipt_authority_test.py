"""Future fit-FP policy must use the verdict, not its diagnostic wording."""

from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
from types import SimpleNamespace

import pytest

from lab_arena import company_judgments, contracts, scoring, verify
from lab_arena.service import ArenaService, RoundDefaults
from lab_arena.contracts import (
    ArenaContractError,
    FIT_FP_CLASSIFICATION_BINDING,
    FIT_RECEIPT_AUTHORITY,
    validate_scorer_policy,
)
from qualification.scoring.competition import count_penalizable_false_positives
from tests.lab_arena.test_benchmark_disclosure_policy import _RoundStore
from tests.lab_arena.company_only_lifecycle_test import _company


def _augmeta() -> dict:
    # Reduced from the accepted October 20 baseline ICP4 judgment.
    return {
        "final_score": 0.0,
        "company_qualified": False,
        "duplicate_company": False,
        "failure_reason": (
            "Company fit unavailable: unproven dimensions: "
            "industry, required_attribute"
        ),
        "intent_signals_detail": None,
        "verifier_gate_receipts": [{
            "gate": "company_fit",
            "decision": "unavailable",
            "failure_class": "insufficient_fit_evidence",
            "company_fit_dimensions": {
                "identity": "match",
                "employee_size": "match",
                "industry": "unavailable",
                "geography": "match",
                "stage": "match",
            },
            "required_attribute_decision": "unavailable",
        }],
    }


def _count(row: dict, *, future: bool) -> tuple[int, int]:
    return count_penalizable_false_positives(
        [row], icp_has_intent_signals=True,
        fit_receipt_authoritative=future,
    )


def test_exact_augmeta_and_all_unproven_fit_dimensions_ignore_reason_wording():
    row = _augmeta()
    assert _count(row, future=False) == (1, 0)
    assert _count(row, future=True) == (0, 0)

    for dimension in (
        "identity", "employee_size", "industry", "geography", "stage",
        "required_attribute",
    ):
        changed = deepcopy(row)
        changed["failure_reason"] = (
            f"Company fit unavailable: unproven dimensions: {dimension}"
        )
        assert _count(changed, future=True) == (0, 0)
    for wording in (
        "Missing required_attribute evidence",
        "missing industry; missing employee_count; missing country",
        "company verification failed to prove required_attribute",
        "country mismatch remains unproven",
    ):
        changed = deepcopy(row)
        changed["failure_reason"] = wording
        assert _count(changed, future=True) == (0, 0)


def test_typed_mismatch_and_legacy_fit_reason_still_count():
    mismatch = _augmeta()
    mismatch["failure_reason"] = "a verified contradiction"
    mismatch["verifier_gate_receipts"][0]["decision"] = "mismatch"
    assert _count(mismatch, future=True) == (1, 0)

    legacy = _augmeta()
    legacy["verifier_gate_receipts"] = []
    assert _count(legacy, future=True) == (1, 0)
    legacy["verifier_gate_receipts"] = [{
        "gate": "company_fit", "decision": "unknown",
    }]
    assert _count(legacy, future=True) == (1, 0)


@pytest.mark.parametrize("reason", (
    "matches the ICP exclusion list", "duplicate company", "data quality issue",
))
def test_independent_gate_reasons_remain_penalized(reason):
    row = _augmeta()
    row["failure_reason"] = reason
    assert _count(row, future=True) == (1, 0)


def test_actual_underscored_duplicate_remains_zero_score_without_new_fp():
    row = _augmeta()
    row.update(duplicate_company=True, failure_reason="duplicate_company_identity")
    assert _count(row, future=False) == (0, 0)
    assert _count(row, future=True) == (0, 0)


def test_provider_and_local_verification_failures_remain_unpenalized():
    provider = _augmeta()
    provider["failure_reason"] = "required_attribute provider timeout"
    assert _count(provider, future=True) == (0, 0)
    assert _count(provider, future=False) == (0, 0)

    local = _augmeta()
    local["verifier_gate_receipts"][0]["failure_class"] = (
        "company_verification_exhausted"
    )
    local["verifier_gate_receipts"][0]["reason"] = (
        "independent employee-size verification failed"
    )
    assert _count(local, future=False) == (0, 0)
    assert _count(local, future=True) == (0, 0)


def test_valid_and_invalid_primary_intent_keep_separate_penalty():
    row = _augmeta()
    row["failure_reason"] = "Company fit unavailable: required_attribute"
    row["intent_signals_detail"] = [{
        "matched_icp_signal": 0,
        "after_decay": 0,
        "judge_verdict": {"decision": "rejected"},
    }]
    assert _count(row, future=True) == (0, 1)
    row["intent_signals_detail"][0]["after_decay"] = 40
    assert _count(row, future=True) == (0, 0)
    row["intent_signals_detail"][0]["after_decay"] = 0
    row["intent_signals_detail"][0]["judge_verdict"] = {
        "decision": "rejected_verifier_error", "error_class": "ProviderTimeout",
    }
    assert _count(row, future=True) == (0, 0)


def test_policy_opt_in_preserves_frozen_arithmetic_and_mixed_positive_icp():
    old = scoring.build_scorer_policy(
        scoring_adapter_version="qualification_integrity_v2",
        normalize_intent_scale=True,
    )
    future = scoring.build_scorer_policy(
        scoring_adapter_version="qualification_integrity_v2",
        normalize_intent_scale=True,
        fit_fp_receipt_authority=True,
    )
    assert FIT_FP_CLASSIFICATION_BINDING not in old["env_bindings"]
    assert future["env_bindings"][FIT_FP_CLASSIFICATION_BINDING] == (
        FIT_RECEIPT_AUTHORITY
    )
    assert validate_scorer_policy(json.loads(json.dumps(future))) == future

    icp = {
        "icp_id": "icp_20261019_005",
        "industry": "Data and Analytics",
        "employee_count": ["2-10"],
        "max_companies": 5,
        "intent_signals": ["Funding round in the last year", "Product launch"],
    }
    rows = [
        {"final_score": 0.0}, {"final_score": 51.0},
        {"final_score": 0.0}, _augmeta(),
    ]
    historical = verify.per_icp_score(icp, rows, old)
    corrected = verify.per_icp_score(icp, rows, future)
    assert (historical["per_icp_score"], historical["fp_gate_count"]) == (
        10.25, 1,
    )
    assert (corrected["per_icp_score"], corrected["fp_gate_count"]) == (
        12.75, 0,
    )
    assert corrected["company_scores"] == historical["company_scores"] == (
        [0.0, 51.0, 0.0, 0.0]
    )


def test_policy_binding_rejects_unknown_value_or_old_adapter():
    policy = scoring.build_scorer_policy(
        scoring_adapter_version="qualification_integrity_v2",
        fit_fp_receipt_authority=True,
    )
    policy["env_bindings"][FIT_FP_CLASSIFICATION_BINDING] = "unknown"
    with pytest.raises(ArenaContractError, match="unsupported fit FP classification"):
        validate_scorer_policy(policy)
    with pytest.raises(ArenaContractError, match="requires integrity scoring"):
        scoring.build_scorer_policy(fit_fp_receipt_authority=True)


def test_new_round_freezes_binding_without_changing_existing_round_or_cache_key():
    store = _RoundStore()
    defaults = RoundDefaults(
        baseline_hotkey="5" + "B" * 47,
        scorer_image_digest="sha256:" + "1" * 64,
        scorer_image_reference="registry.example/scorer@sha256:" + "1" * 64,
        integrity_from="2026-10-01T00:00:00Z",
        company_judgment_from="2026-10-01T00:00:00Z",
        fit_fp_receipt_authority=False,
    )
    service = object.__new__(ArenaService)
    service._config = SimpleNamespace(
        defaults=defaults, mode="shadow", network_name="test", netuid=401,
        pinned_round_id=None,
    )
    service._store = store
    service._scorer_policy = scoring.build_scorer_policy()
    service.runner_settings = lambda: (["5" + "C" * 47], [])
    service._require_integrity_schema = lambda: None
    service._require_company_quality_schema = lambda: None
    cutoff = datetime(2026, 10, 20, tzinfo=timezone.utc)
    old = service.create_round(cutoff, round_id="arena-2026-10-20-test")
    old_hash = contracts.document_hash(old["scorer_policy"])
    assert FIT_FP_CLASSIFICATION_BINDING not in old["scorer_policy"]["env_bindings"]

    service._config.defaults = replace(defaults, fit_fp_receipt_authority=True)
    assert service.create_round(
        cutoff, round_id="arena-2026-10-20-test"
    ) == old
    assert contracts.document_hash(store.rows[old["round_id"]]["configuration_doc"]["scorer_policy"]) == old_hash
    future = service.create_round(
        cutoff + timedelta(days=1), round_id="arena-2026-10-21-test"
    )
    assert future["scorer_policy"]["env_bindings"][FIT_FP_CLASSIFICATION_BINDING] == FIT_RECEIPT_AUTHORITY
    assert contracts.document_hash(future["scorer_policy"]) != old_hash
    assert RoundDefaults().fit_fp_receipt_authority is True

    icp = {
        "icp_id": "icp-test", "industry": "Software",
        "employee_count": ["51-200"], "intent_signals": ["Product launch"],
        "max_companies": 5,
    }
    def cache_key(policy):
        inp = scoring.build_scoring_input(
            scored_run_id="execute-1", icp=icp, companies=[_company()],
            policy=policy, evaluation_date="2026-10-20",
        )
        return company_judgments.build_company_scopes(
            scoring_input=inp, round_id="arena-2026-10-20-test",
            network_name="finney", netuid=71,
            scorer_image_digest="sha256:" + "a" * 64,
            scorer_image_reference="registry.example/scorer@sha256:" + "a" * 64,
            integrity_policy="arena_integrity_v1",
            company_judgment_policy="company_judgment_v1",
        )[0]["cache_key"]
    assert cache_key(future["scorer_policy"]) != cache_key(old["scorer_policy"])
