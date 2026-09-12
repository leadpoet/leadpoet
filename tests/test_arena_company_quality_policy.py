"""Company-local failures, reusable judgments, and coverage-weighted ranking."""
from __future__ import annotations

import asyncio
from copy import deepcopy
from itertools import permutations
from urllib.parse import urlsplit

import pytest

from lab_arena import contracts, contact_policy, quality_policy, scoring, verify
from lab_arena.output import OutputInvalid, validate_output_document
from qualification.company_quality import canonical_company_linkedin, normalize_company_claim
from qualification.competition_models import CompetitionCompanyV3, CompetitionCompanyV4
from qualification.scoring import lead_scorer
from qualification.scoring.competition import (
    CompetitionCompanyScorer, apply_company_judgment_context,
    effective_competition_input, raw_company_judgment,
)
from tests.test_arena_score_integrity import _icp, _positive_breakdown as _legacy_positive_breakdown, _public_company


def _positive_breakdown(name, domain, slug):
    from qualification.scoring.company_fit_decision import evaluate_company_identity
    row = _legacy_positive_breakdown(name, domain, slug)
    receipt = evaluate_company_identity(submitted_name=name, submitted_website="https://" + domain, submitted_linkedin="https://linkedin.com/company/" + slug, observed_name=name, observed_website="https://" + domain, observed_linkedin="https://linkedin.com/company/" + slug, evidence_source="company_web_reverification", company_quality=True)
    row["verifier_gate_receipts"][0]["dimension_evidence"]["identity"]["web_identity_receipt"] = receipt
    return row


def company(name="Acme", domain="acme.com", **updates):
    row = _public_company(name, domain=domain, linkedin=f"https://linkedin.com/company/{name.lower()}")
    return {**row, "state": "CA", **updates}


@pytest.mark.parametrize(
    "state", ["CA", "ca", "California", " california ", "California (CA)"]
)
@pytest.mark.parametrize("country", ["US", "USA", "United States", "United States of America", "U.S.", "U.S.A", "America"])
def test_us_claims_normalize_without_rejecting_variants(state, country):
    row, errors = normalize_company_claim(company(state=state, country=country))
    assert not errors
    assert (row["country"], row["state"]) == ("United States", "California")


@pytest.mark.parametrize("state", ["dc", "D.C.", "District of Columbia", "Washington, DC", "Washington, D.C."])
def test_dc_is_accepted(state):
    row, errors = normalize_company_claim(company(state=state))
    assert not errors and row["state"] == "District of Columbia"


def test_state_name_and_code_must_agree():
    _row, errors = normalize_company_claim(company(state="California (NY)"))

    assert errors == ("us_headquarters_state_required_or_invalid",)


@pytest.mark.parametrize("url", [
    "https://www.linkedin.com/company/Acme/?trk=test#about",
    "http://linkedin.com/company/acme/about/",
    "https://uk.linkedin.com/company/acme/jobs/",
])
def test_company_page_variants_share_one_url(url):
    assert canonical_company_linkedin(url) == "https://linkedin.com/company/acme"


@pytest.mark.parametrize("url", [None, "", [], "https://linkedin.com/in/acme", "https://linkedin.com.example.com/company/acme", "https://linkedin.com:8443/company/acme"])
def test_bad_linkedin_rejects_one_row_without_discarding_document(url):
    rows = [company(company_linkedin=url), company("Beta", "beta.com")]
    parsed = validate_output_document({"schema_version": quality_policy.OUTPUT_SCHEMA, "companies": rows}, expected_schema_version=quality_policy.OUTPUT_SCHEMA)
    assert len(parsed["companies"]) == 2
    assert normalize_company_claim(parsed["companies"][0])[1]
    assert not normalize_company_claim(parsed["companies"][1])[1]


def test_models_roundtrip_raw_new_claims_and_contacts():
    for model in (CompetitionCompanyV3, CompetitionCompanyV4):
        row = model.model_validate(company(company_linkedin=None, state=[]))
        assert model.model_validate_json(row.model_dump_json()) == row
    with pytest.raises(OutputInvalid):
        validate_output_document({"schema_version": contracts.OUTPUT_DOCUMENT_SCHEMA_VERSION, "companies": [company()]}, expected_schema_version=quality_policy.OUTPUT_SCHEMA)
    assert contact_policy.output_schema({"company_quality_policy": quality_policy.POLICY}) == quality_policy.OUTPUT_SCHEMA
    assert contact_policy.output_schema({"company_quality_policy": quality_policy.POLICY, "contact_policy": contact_policy.POLICY}) == quality_policy.CONTACT_OUTPUT_SCHEMA


def test_invalid_required_fields_zero_only_the_affected_company(monkeypatch):
    calls = []
    async def judge(**kwargs):
        assert kwargs["company_quality"] is True
        calls.append(kwargs["company"].company_name)
        c = kwargs["company"]
        return _positive_breakdown(c.company_name, urlsplit(c.company_website).hostname, c.company_name.lower())
    monkeypatch.setattr(lead_scorer, "score_company_competition_intent", judge)
    rows = [company(company_linkedin=None), company("Beta", "beta.com", state=""), company("Gamma", "gamma.com"), company("Delta", "delta.com", country="Canada", state=None)]
    results = asyncio.run(CompetitionCompanyScorer(company_quality=True).score_with_breakdowns(rows, _icp(), False))
    assert [row["final_score"] for row in results] == [0, 0, 60, 60]
    assert calls == ["Gamma", "Delta"]


def test_verified_linkedin_dedup_rebinds_raw_judgment_without_poisoning_it():
    rows = [company(), company("Acme", "acme-other.com"), company("Child", "acme.com", company_linkedin="https://linkedin.com/company/acme-child")]
    raw = [_positive_breakdown("Acme", "acme.com", "acme"), _positive_breakdown("Acme", "acme-other.com", "acme"), _positive_breakdown("Child", "acme.com", "acme-child")]
    original = deepcopy(raw)
    result = apply_company_judgment_context(rows, raw)
    assert [row["final_score"] for row in result] == [60, 0, 60]
    assert result[1]["company_identity_key"] == result[0]["company_identity_key"]
    assert raw == original
    alone = apply_company_judgment_context([rows[1]], [raw[1]])
    assert alone[0]["final_score"] == 60 and alone[0]["company_qualified"]


def test_distinct_verified_linkedin_entities_survive_weak_alias_collision():
    rows = [
        company(
            "Shared Inc",
            "platform.example",
            company_linkedin="https://linkedin.com/company/shared-one",
        ),
        company(
            "Shared LLC",
            "platform.example",
            company_linkedin="https://linkedin.com/company/shared-two",
        ),
    ]
    raw = [
        _positive_breakdown("Shared Inc", "platform.example", "shared-one"),
        _positive_breakdown("Shared LLC", "platform.example", "shared-two"),
    ]

    result = apply_company_judgment_context(rows, raw)

    assert [row["final_score"] for row in result] == [60, 60]
    assert [row["duplicate_company"] for row in result] == [False, False]


def test_verified_regional_linkedin_aliases_for_same_company_get_one_credit():
    rows = [
        company(company_linkedin="https://linkedin.com/company/acme-us"),
        company(company_linkedin="https://linkedin.com/company/acme-west"),
    ]
    raw = [
        _positive_breakdown("Acme", "acme.com", "acme-us"),
        _positive_breakdown("Acme", "acme.com", "acme-west"),
    ]

    result = apply_company_judgment_context(rows, raw)

    assert [row["final_score"] for row in result] == [60, 0]
    assert [row["duplicate_company"] for row in result] == [False, True]
    assert result[1]["duplicate_of_index"] == 0


@pytest.mark.parametrize(
    ("country", "state"),
    [("Canada", "Ontario"), ("United States", "NY")],
)
def test_same_display_name_and_domain_at_distinct_verified_hqs_stay_distinct(
    country, state
):
    rows = [
        company(company_linkedin="https://linkedin.com/company/acme-us"),
        company(
            company_linkedin="https://linkedin.com/company/acme-ca",
            country=country,
            state=state,
        ),
    ]
    raw = [
        _positive_breakdown("Acme", "acme.com", "acme-us"),
        _positive_breakdown("Acme", "acme.com", "acme-ca"),
    ]

    result = apply_company_judgment_context(rows, raw)

    assert [row["final_score"] for row in result] == [60, 60]
    assert [row["duplicate_company"] for row in result] == [False, False]


def test_non_us_country_alias_and_optional_state_cannot_evade_alias_dedup():
    rows = [
        company(
            company_linkedin="https://linkedin.com/company/acme-ca",
            country="Canada",
            state="Ontario",
        ),
        company(
            company_linkedin="https://linkedin.com/company/acme-canada",
            country="CA",
            state="Quebec",
        ),
    ]
    raw = [
        _positive_breakdown("Acme", "acme.com", "acme-ca"),
        _positive_breakdown("Acme", "acme.com", "acme-canada"),
    ]

    result = apply_company_judgment_context(rows, raw)

    assert [row["final_score"] for row in result] == [60, 0]
    assert [row["duplicate_company"] for row in result] == [False, True]


def test_first_failed_intent_does_not_block_one_later_verified_duplicate():
    rows = [company(), company()]
    failed = _positive_breakdown("Acme", "acme.com", "acme")
    failed["intent_signals_detail"][0]["after_decay"] = 0
    failed["final_score"] = 0
    valid = _positive_breakdown("Acme", "acme.com", "acme")

    result = apply_company_judgment_context(rows, [failed, valid])

    assert [row["final_score"] for row in result] == [0, 60]
    assert [row["company_qualified"] for row in result] == [False, True]
    assert not any(row["duplicate_company"] for row in result)


def test_failed_intent_does_not_reserve_verified_regional_alias():
    rows = [
        company(company_linkedin="https://linkedin.com/company/acme-us"),
        company(company_linkedin="https://linkedin.com/company/acme-west"),
    ]
    failed = _positive_breakdown("Acme", "acme.com", "acme-us")
    failed["intent_signals_detail"][0]["after_decay"] = 0
    failed["final_score"] = 0
    valid = _positive_breakdown("Acme", "acme.com", "acme-west")

    result = apply_company_judgment_context(rows, [failed, valid])

    assert [row["final_score"] for row in result] == [0, 60]
    assert not any(row["duplicate_company"] for row in result)


def test_only_one_successful_exact_verified_linkedin_identity_gets_credit():
    rows = [company(), company(), company()]
    failed = _positive_breakdown("Acme", "acme.com", "acme")
    failed["intent_signals_detail"][0]["after_decay"] = 0
    failed["final_score"] = 0
    valid = _positive_breakdown("Acme", "acme.com", "acme")

    result = apply_company_judgment_context(rows, [failed, valid, deepcopy(valid)])

    assert [row["final_score"] for row in result] == [0, 60, 0]
    assert [row["duplicate_company"] for row in result] == [False, False, True]
    assert result[2]["duplicate_of_index"] == 1


def test_effective_quality_input_keeps_state_but_ignores_unused_prose():
    base = company()
    changed = deepcopy(base)
    changed["intent_signals"][0]["why_now"] = "different ignored prose"
    assert effective_competition_input([base], _icp(), company_quality=True) == effective_competition_input([changed], _icp(), company_quality=True)
    changed["state"] = "NY"
    assert effective_competition_input([base], _icp(), company_quality=True) != effective_competition_input([changed], _icp(), company_quality=True)
    base["country"] = changed["country"] = "Canada"
    assert effective_competition_input([base], _icp(), company_quality=True) == effective_competition_input([changed], _icp(), company_quality=True)


def test_request_completeness_rewards_five_decent_companies_over_two_excellent():
    legacy = scoring.build_scorer_policy(scoring_adapter_version="qualification_integrity_v2")
    quality = scoring.build_scorer_policy(scoring_adapter_version="qualification_integrity_v2", company_quality=True)
    icp = {**_icp(), "max_companies": 5}
    def rows(values):
        return [{"final_score": value, "company_qualified": True, "duplicate_company": False} for value in values]
    sparse, complete = rows([100, 100]), rows([35] * 5)
    assert verify.per_icp_score(icp, sparse, legacy)["per_icp_score"] == 40
    assert verify.per_icp_score(icp, complete, legacy)["per_icp_score"] == 35
    assert verify.per_icp_score(icp, sparse, quality)["per_icp_score"] == 16
    assert verify.per_icp_score(icp, complete, quality)["per_icp_score"] == 35
    for invalid in ({"final_score": 0, "company_qualified": False},
                    {"final_score": 0, "company_qualified": True, "duplicate_company": True}):
        assert verify.per_icp_score(icp, sparse + [invalid] * 3, quality)["per_icp_score"] <= 16
    assert verify.per_icp_score(icp, [], quality)["per_icp_score"] == 0
    assert verify.per_icp_score({**icp, "max_companies": 2}, sparse, quality)["per_icp_score"] == 100


def test_cross_request_aggregation_keeps_existing_arithmetic_mean():
    scores = [100, 100, 10, 0, 0]
    assert verify.stage_score(scores, 5) == 42
    assert len({verify.stage_score(list(p), 5) for p in permutations(scores)}) == 1
    main_scores = [100] * 10 + [0] * 10
    assert verify.stage_score(main_scores, 20) == 50
    assert (verify.stage_score(main_scores[:10], 10) + verify.stage_score(main_scores[10:], 10)) / 2 == 50


def test_quality_policy_cannot_cache_an_operator_capped_unjudged_company():
    policy = scoring.build_scorer_policy(scoring_adapter_version="qualification_integrity_v2", company_quality=True)
    assert policy["max_scored_companies"] == 0
    policy["max_scored_companies"] = 1
    policy["env_bindings"]["RESEARCH_LAB_EVAL_MAX_SCORED_COMPANIES"] = "1"
    with pytest.raises(contracts.ArenaContractError, match="all assigned company slots"):
        contracts.validate_scorer_policy(policy)
    policy.pop("company_quality_policy")
    assert contracts.validate_scorer_policy(policy)["max_scored_companies"] == 1


@pytest.mark.parametrize("scores", [[100, 100, float("nan"), 0, 0], [100, 100, float("inf"), 0, 0]])
def test_bad_scores_cannot_enter_stage_mean(scores):
    with pytest.raises(contracts.ArenaContractError):
        verify.stage_score(scores, 5)


def test_judgment_worker_reuses_identical_misses_and_rebinds_indexes():
    from lab_arena import company_judgments
    rows = [company(), company(), company("Beta", "beta.com")]
    calls = []
    def judge(companies, icp, _reference):
        c = companies[0]
        calls.append(c["company_name"])
        raw = _positive_breakdown(c["company_name"], urlsplit(c["company_website"]).hostname, c["company_name"].lower())
        return apply_company_judgment_context(companies, [raw])
    judge.company_quality = judge.integrity_policy = True
    judge.contacts_required = False
    hashes = ["a", "a", "b"]
    lease = {"schema_version": company_judgments.LEASE_SCHEMA_VERSION, "hits": [], "misses": [{"company_index": i, "cache_key": "sha256:" + key * 64, "company_input_hash": "sha256:" + key * 64, "authority_slot": 0} for i, key in enumerate(hashes)]}
    full, new = scoring.score_quality_work_item({"scored_run_id": "run"}, icp=_icp(), companies=rows, scorer=judge, cache_context=lease)
    assert calls == ["Acme", "Beta"]
    assert [row["company_index"] for row in new] == [0, 2]
    assert [row["final_score"] for row in full] == [60, 0, 60]
    assert all("company_index" not in row["raw_judgment"] for row in new)
    assert scoring.validate_scoring_output_document(scoring.build_scoring_output("run", full, company_judgments=new))["company_judgments"] == new


def test_validator_cannot_credit_missing_state_or_unverified_identity():
    row = company()
    breakdown = apply_company_judgment_context([row], [_positive_breakdown("Acme", "acme.com", "acme")])[0]
    bad_claim = {**row, "state": None}
    with pytest.raises(scoring.ScoringError):
        scoring.validate_breakdowns_for_item([breakdown], icp=_icp(), companies=[bad_claim], integrity_policy=True, company_quality=True)


@pytest.mark.parametrize("field,value", [("observed_name", ""), ("observed_domain", ""), ("observed_domain", "localhost"), ("observed_domain", "acme.com/path"), ("observed_domain", "https://acme.com"), ("observed_linkedin_slug", ""), ("observed_linkedin_slug", "acme/jobs")])
def test_incomplete_verified_identity_cannot_qualify(field, value):
    raw = _positive_breakdown("Acme", "acme.com", "acme")
    raw["verifier_gate_receipts"][0]["dimension_evidence"]["identity"]["web_identity_receipt"][field] = value
    result = apply_company_judgment_context([company()], [raw])
    assert result[0]["final_score"] == 0
    assert not result[0]["company_qualified"]


def test_other_company_receipt_cannot_qualify_this_claim():
    raw = _positive_breakdown("Contoso", "contoso.com", "contoso")
    result = apply_company_judgment_context([company()], [raw])
    assert result[0]["final_score"] == 0
    assert not result[0]["company_qualified"]


def test_context_clears_contact_credit_when_primary_signal_does_not_qualify():
    from tests.test_arena_contact_scoring import _contact, _contact_result
    from qualification.scoring.competition import _merge_contact_breakdown
    row = company(contact=_contact())
    raw = _positive_breakdown("Acme", "acme.com", "acme")
    raw["intent_signals_detail"][0]["after_decay"] = 0
    raw["final_score"] = 0
    _merge_contact_breakdown(raw, _contact_result(qualified=True, email_status="valid"))
    result = apply_company_judgment_context([row], [raw], contacts_required=True)[0]
    assert not result["company_qualified"]
    assert not result["contact_qualified"]
    assert result["contact_verification"]["decision"] == "not_evaluated"
    contact_policy.validate_contact_breakdown(result)


def test_verified_contact_binding_keeps_multiword_claim_name():
    from qualification.scoring.competition import _verified_company_for_contact
    from qualification.scoring.contact_verification import (
        _company_identifiers,
        _company_matches,
    )

    claim = company(
        "Acme Technologies",
        company_linkedin="https://linkedin.com/company/acme-technologies",
    )
    bound = _verified_company_for_contact(
        claim,
        {
            "observed_name": "acmetechnologies",
            "observed_domain": "acme.com",
            "observed_linkedin_slug": "acme-technologies",
        },
    )

    assert bound["company_name"] == "Acme Technologies"
    assert _company_matches(
        _company_identifiers(bound),
        {"name": "acme technologies", "domain": "", "linkedin_slug": ""},
    )


@pytest.mark.parametrize(
    ("first_slug", "second_slug"),
    [("acme", "acme"), ("acme-us", "acme-west")],
)
def test_first_failed_contact_does_not_block_one_later_verified_duplicate(
    first_slug, second_slug
):
    from tests.test_arena_contact_scoring import _contact, _contact_result
    from qualification.scoring.competition import _merge_contact_breakdown

    rows = [
        company(
            contact=_contact(),
            company_linkedin=f"https://linkedin.com/company/{first_slug}",
        ),
        company(
            contact=_contact(slug="grace"),
            company_linkedin=f"https://linkedin.com/company/{second_slug}",
        ),
    ]
    failed = _positive_breakdown("Acme", "acme.com", first_slug)
    valid = _positive_breakdown("Acme", "acme.com", second_slug)
    _merge_contact_breakdown(failed, _contact_result(qualified=False))
    _merge_contact_breakdown(
        valid, _contact_result(qualified=True, email_status="valid")
    )

    result = apply_company_judgment_context(
        rows, [failed, valid], contacts_required=True
    )

    assert [row["final_score"] for row in result] == [0, 60]
    assert [row["company_qualified"] for row in result] == [False, True]
    assert not any(row["duplicate_company"] for row in result)


@pytest.mark.parametrize(
    ("first_slug", "second_slug"),
    [("acme", "acme"), ("acme-us", "acme-west")],
)
def test_contact_duplicate_context_survives_generic_quality_revalidation(
    first_slug, second_slug
):
    from tests.test_arena_contact_scoring import _contact, _contact_result
    from qualification.scoring.competition import _merge_contact_breakdown

    rows = [
        company(
            contact=_contact(),
            company_linkedin=f"https://linkedin.com/company/{first_slug}",
        ),
        company(
            contact=_contact(slug="grace"),
            company_linkedin=f"https://linkedin.com/company/{second_slug}",
        ),
    ]
    raw = [
        _positive_breakdown("Acme", "acme.com", first_slug),
        _positive_breakdown("Acme", "acme.com", second_slug),
    ]
    for breakdown in raw:
        contact_result = _contact_result(qualified=True, email_status="valid")
        contact_result["contact_verification"]["subchecks"] = {
            key: {"status": "pass"}
            for key in (
                "claim", "identity", "source", "company", "role", "location",
                "email_attribution", "email_verification",
            )
        }
        _merge_contact_breakdown(
            breakdown, contact_result
        )

    contextual = apply_company_judgment_context(
        rows, raw, contacts_required=True
    )

    assert contextual[1]["duplicate_company"] is True
    assert contextual[1]["contact_qualified"] is False
    assert scoring.validate_breakdowns_for_item(
        contextual,
        icp=_icp(),
        companies=rows,
        integrity_policy=True,
        contacts_required=True,
        company_quality=True,
    ) == contextual
