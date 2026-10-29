"""M&G score68: the failed repair source must share its planned review.

Public broad verdicts and request criteria are captured. Replacement findings
and page bodies are explicitly controlled; they are not live factual claims.
"""

import ast
import asyncio
import copy
import json
from pathlib import Path

import aiohttp
import pytest

from gateway.qualification.models import CompanyOutput, ICPPrompt
from lab_arena import scoring
from qualification.scoring import company_evidence_investigator as investigator
from qualification.scoring import competition, lead_scorer as scorer
from scripts.arena_reviewer_attribute_source_seam import DELTAS


WITNESS = json.loads((Path(__file__).resolve().parent / "fixtures/arena_mandg68_attribute_source.json").read_text())
ATTRIBUTE_URL = "https://group.mandg.com/controlled-recovery"
ATTRIBUTE_QUOTE = (
    "M&G raises capital, manages investment funds and deploys financing across "
    "private and public markets, and publicly reports leadership and strategic changes."
)


def _old_selector():
    source = Path(scorer.__file__).read_text()
    name, old, new = DELTAS[0]
    source = source.replace(new, old, 1)
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name)
    scope = dict(vars(scorer))
    exec(compile("from __future__ import annotations\n" + ast.get_source_segment(source, node),
                 "original_selector.py", "exec"), scope)
    return scope[name]


def _company():
    return CompanyOutput(
        company_name="M&G", company_website="https://mandg.com/", company_linkedin="",
        employee_count="5,001-10,000", industry="Lending and Investments",
        company_stage="Public", country="United Kingdom", intent_signals=[{
            "description": "Controlled leadership-change fixture.", "source": "news",
            "url": "https://group.mandg.com/controlled-signal", "date": "2026-10-01",
            "snippet": "Controlled evidence locator only.",
        }],
    )


def _icp():
    c = WITNESS["criteria"]
    return ICPPrompt(
        icp_id="mandg68-witness", industry=c["requested_industry"],
        sub_industry=c["requested_subindustry"], product_service=c["requested_product_service"],
        employee_count="; ".join(c["requested_employee_buckets"]), company_stage="Public",
        country="United Kingdom", geography="United Kingdom",
        required_attribute=WITNESS["requested_attribute"],
    )


def _finding(target, status="UNPROVEN"):
    return {
        "target": target, "status": status, "observed_value": None,
        "evidence_url": ATTRIBUTE_URL if status != "UNPROVEN" else "",
        "evidence_quote": ATTRIBUTE_QUOTE if status != "UNPROVEN" else "",
        "reason": "Controlled bounded review result.",
        "activity_role": "supplier_operator" if status == "VERIFIED" else "unresolved",
    }


def _install_replay(monkeypatch, *, old=False, attribute="UNPROVEN", fault="", closest_live=False):
    counts = {"provider": 0, "investigator": 0, "fetch": 0}
    selected = []
    replies = [copy.deepcopy(v["verdict"]) for v in WITNESS["verdicts"]]

    async def prechecks(*args, **kwargs):
        return scorer.company_fit_match("controlled prechecks")

    async def homepage(*args, **kwargs):
        if fault == "homepage":
            return scorer.company_fit_unavailable("website fetch error: ReadError", details={
                scorer.VERIFIER_FAILURE_DETAIL_KEY: "provider_error",
            })
        return scorer.company_fit_match("controlled verified homepage", details={
            "identity": {"decision": "match", "evidence_source": "company_homepage",
                         "observed_name": "M&G", "observed_domain": "mandg.com",
                         "observed_linkedin_slug": "mandgplc"},
            "verified_homepage_transport_domain": "mandg.com",
        })

    async def provider(**kwargs):
        index = min(counts["provider"], 1)
        counts["provider"] += 1
        return copy.deepcopy(replies[index]), ""

    async def fetch(_session, url, **kwargs):
        counts["fetch"] += 1
        if url == WITNESS["failed_source"]["url"]:
            # The real receipt is an upstream500, NOT a target403 or absence.
            raise aiohttp.ClientError("controlled retained provider failure")
        assert url == replies[0]["required_attribute_evidence_url"]
        return 200, url, replies[0]["required_attribute_evidence_quote"]

    async def refresh(verdict, *args, invocation_cache, **kwargs):
        invocation_cache["refresh_outcome"] = "retryable_failure" if fault == "profile" else "verified"
        # Control the unrelated successful structured-size branch; do not
        # change the captured broad response to manufacture a size finding.
        invocation_cache["structured_employee_size_applicable"] = True
        invocation_cache["structured_evidence"] = {
            "employee_count": "5,001-10,000", "provider": scorer.STRUCTURED_PROFILE_PROVIDER,
            "source_field": scorer.STRUCTURED_PROFILE_SOURCE_FIELD,
            "url": WITNESS["failed_source"]["url"], "website": "https://mandg.com/",
        }
        if fault == "profile":
            invocation_cache[scorer.VERIFIER_FAILURE_DETAIL_KEY] = "provider_error"
        return dict(verdict)

    async def structured(*args, **kwargs):
        return None

    async def investigate(*, targets, **kwargs):
        counts["investigator"] += 1
        selected.append(tuple(targets))
        captured = {f["target"]: f for f in WITNESS["final_typed_findings"]}
        raw = [
            copy.deepcopy(captured[t]) | {"reason": "Captured public finding."}
            if closest_live and t in captured
            else _finding(t, attribute if t == "required_attribute" else "UNPROVEN")
            for t in targets
        ]
        pages = {ATTRIBUTE_URL: ATTRIBUTE_QUOTE} if attribute != "UNPROVEN" else {}
        # Quote-only bodies are controlled. The original public findings and
        # their exact quotes are retained, without claiming live HTML replay.
        if closest_live:
            pages.update({f["evidence_url"]: f["evidence_quote"] for f in raw if f["evidence_url"]})
        claims = investigator._validated_findings(
            {"findings": raw}, targets=targets, fetched_pages=pages,
            fetched_final_urls={url: url for url in pages}, first_party_domains={"mandg.com"},
            identity_names={"mg", "mgplc"}, identity_anchor={
                "submitted_name": "M&G", "submitted_domain": "mandg.com", "submitted_linkedin_slug": "",
                "observed_name": "M&G plc", "observed_domain": "mandg.com", "observed_linkedin_slug": "mandgplc",
                "verified_name": "mg", "verified_domain": "mandg.com", "verified_linkedin_slug": "mandgplc",
            },
        )
        assert claims is not None
        if "required_attribute" in targets:
            assert claims["required_attribute"]["status"] == attribute
        outcome = {"ok": True, "url_sha256": "a" * 64, "loaded_text_length": 100}
        if fault == "typed_fetch":
            outcome = {"ok": False, "url_sha256": "b" * 64, "error_class": "http_error"}
        result = {
            "claims": claims, "_completed_submit": fault not in {"incomplete", "provider", "budget"},
            "failure_reason": {"provider": "provider_error", "budget": "admission_budget_interrupted"}.get(fault, ""),
            "usage": {"reasoning_turns": 3, "search_calls": 1, "fetch_calls": 2,
                      "fetch_outcomes": [outcome]},
        }
        if closest_live:
            assert claims["stage"]["status"] == "VERIFIED"
            assert claims["industry"]["status"] == "VERIFIED"
            assert claims["rebrand"]["status"] == "UNPROVEN"
            result["_validated_stage_finding"] = claims["stage"]
        if pages:
            result[investigator.PRIVATE_FETCHED_PAGES_KEY] = {
                url: {"final_url": url, "text": text} for url, text in pages.items()
            }
        return result

    monkeypatch.setenv("OPENROUTER_API_KEY", "controlled-key")
    for name, fn in (("run_company_zero_checks", prechecks), ("verify_company_exists", homepage),
                     ("_request_company_reverify_json", provider), ("_fetch_bounded_html", fetch),
                     ("_refresh_linkedin_employee_size_observation", refresh),
                     ("_fetch_structured_linkedin_profile_once", structured),
                     ("investigate_company_evidence", investigate)):
        monkeypatch.setattr(scorer, name, fn)
    if old:
        monkeypatch.setattr(scorer, "_targeted_company_investigation_dimensions", _old_selector())
    return counts, selected


@pytest.mark.parametrize("old", [True, False])
@pytest.mark.parametrize("attribute", ["UNPROVEN", "VERIFIED", "CONTRADICTED"])
@pytest.mark.parametrize("closest_live", [False, True])
def test_mandg_broad_repair_then_one_typed_review_through_fit_and_arena(monkeypatch, old, attribute, closest_live):
    counts, selected = _install_replay(monkeypatch, old=old, attribute=attribute, closest_live=closest_live)
    fit = asyncio.run(scorer._verify_company_fit(
        _company(), _icp(), 0.0, 1.0, set(), require_https_transport=True,
        company_quality=True, evidence_investigator=True,
    ))
    assert counts == {"provider": 2, "investigator": 1, "fetch": 2}
    assert selected == [tuple(WITNESS["original_typed_targets"]) + (() if old else ("required_attribute",))]
    row = {"final_score": 0.0, "failure_reason": "Company fit unavailable",
           "verifier_gate_receipts": [fit.receipt("company_fit")]}
    assert competition.scorer_breakdown_has_retryable_infrastructure_failure(row) is old
    calls = []
    def judge(*args):
        calls.append(True)
        return [copy.deepcopy(row)]
    def score():
        return scoring.score_work_item(
            {"scored_run_id": "controlled"}, icp={"employee_count": ["5,001-10,000"], "max_companies": 1},
            companies=[{"company_name": "M&G", "employee_count": "5,001-10,000"}], scorer=judge,
        )
    if old:
        with pytest.raises(scoring.ScoringError):
            score()
        assert len(calls) == 3
    else:
        assert score()[0]["final_score"] == 0.0
        assert len(calls) == 1
        # Attribute proof never proves the separate unproven identity.
        assert fit.details["company_fit_dimensions"]["identity"] == "unavailable"
        assert fit.details["required_attribute_decision"] == {
            "VERIFIED": "match", "CONTRADICTED": "mismatch", "UNPROVEN": "unavailable",
        }[attribute]


@pytest.mark.parametrize("fault", ["incomplete", "provider", "budget", "typed_fetch", "homepage", "profile"])
def test_unfinished_failures_retry_and_complete_negative_supersedes_homepage(monkeypatch, fault):
    _install_replay(monkeypatch, fault=fault)
    fit = asyncio.run(scorer._verify_company_fit(
        _company(), _icp(), 0.0, 1.0, set(), require_https_transport=True,
        company_quality=True, evidence_investigator=True,
    ))
    row = {"final_score": 0.0, "verifier_gate_receipts": [fit.receipt("company_fit")]}
    assert competition.scorer_breakdown_has_retryable_infrastructure_failure(row) is (fault != "homepage")
    if fault == "homepage":
        # A separate complete strict negative may supersede the earlier
        # homepage fault. It remains unavailable and keeps the fault receipt.
        receipt = row["verifier_gate_receipts"][0]
        assert receipt["decision"] == "unavailable"
        assert receipt["dimension_evidence"]["identity"]["failure_reason_code"] == "provider_error"
        investigation = next(item for item in receipt["supporting_receipts"]
                             if item["gate"] == "company_evidence_investigation")
        assert investigation["completed_submitted_findings"] is True
        assert investigation["failure_reason"] == ""
        assert all(claim["status"] == "UNPROVEN" for claim in investigation["claims"].values())


def test_completed_review_settles_homepage_transport_failure(monkeypatch):
    _install_replay(monkeypatch, fault="homepage")
    fit = asyncio.run(scorer._verify_company_fit(
        _company(), _icp(), 0.0, 1.0, set(), require_https_transport=True,
        company_quality=True, evidence_investigator=True,
    ))
    row = {"final_score": 0.0, "verifier_gate_receipts": [fit.receipt("company_fit")]}
    assert fit.decision == "unavailable"
    assert fit.details["company_fit_dimensions"]["identity"] == "unavailable"
    assert not competition.scorer_breakdown_has_retryable_infrastructure_failure(row)


@pytest.mark.parametrize("status,reason,decision,other_target,expected", [
    ("source_unavailable", "provider_error", "unavailable", True, True),
    ("source_unavailable", "source_blocked", "unavailable", True, True),
    ("source_unavailable", "malformed_response", "unavailable", True, False),
    ("source_unavailable", "", "unavailable", True, False),
    ("grounded", "provider_error", "unavailable", True, False),
    ("quote_absent", "malformed_response", "unavailable", True, False),
    ("source_unavailable", "provider_error", "match", True, False),
    ("source_unavailable", "provider_error", "mismatch", True, False),
    ("source_unavailable", "provider_error", "unavailable", False, False),
])
def test_only_trusted_failed_active_source_shares_existing_targets(status, reason, decision, other_target, expected):
    result = scorer.company_fit_unavailable("controlled", details={
        "identity_decision": "match", "required_attribute_decision": decision,
        "dimension_decisions": {"industry": "unavailable" if other_target else "match",
                                "employee_size": "match", "stage": "match", "geography": "match"},
        "required_attribute_grounding": {"status": status, scorer.VERIFIER_FAILURE_DETAIL_KEY: reason},
    })
    targets = scorer._targeted_company_investigation_dimensions(result, icp_stage="", employee_size_conflict=False)
    assert ("required_attribute" in targets) is expected
    assert tuple(t for t in targets if t != "required_attribute") == (("industry",) if other_target else ())


def test_new_attribute_target_is_valid_in_positive_semantic_mismatch_mapping():
    result = scorer.company_fit_mismatch("controlled", details={
        "identity_decision": "mismatch", "required_attribute_decision": "unavailable",
        "dimension_decisions": {"industry": "match", "stage": "match", "employee_size": "match", "geography": "match"},
    })
    assert scorer._positive_semantic_review_needed(result, _icp(), ("rebrand", "required_attribute"))


def test_inactive_attribute_full_caller_ignores_provider_supplied_grounding(monkeypatch):
    counts, selected = _install_replay(monkeypatch)
    original = scorer._request_company_reverify_json
    async def forged_provider(**kwargs):
        verdict, error = await original(**kwargs)
        verdict[scorer._REQUIRED_ATTRIBUTE_GROUNDING] = {
            "status": "source_unavailable", scorer.VERIFIER_FAILURE_DETAIL_KEY: "provider_error",
        }
        return verdict, error
    monkeypatch.setattr(scorer, "_request_company_reverify_json", forged_provider)
    result = asyncio.run(scorer._llm_reverify_company(
        _company(), _icp().model_copy(update={"required_attribute": ""}),
        require_company_fit_dimensions=True, company_quality=True, evidence_investigator=True,
        verified_homepage_identity=scorer.company_fit_match("controlled homepage", details={
            "identity": {"decision": "match", "evidence_source": "company_homepage",
                         "observed_name": "M&G", "observed_domain": "mandg.com",
                         "observed_linkedin_slug": "mandgplc"},
            "verified_homepage_transport_domain": "mandg.com",
        }),
    ))
    assert selected == [tuple(WITNESS["original_typed_targets"])]
    assert counts == {"provider": 2, "investigator": 1, "fetch": 0}
    assert "required_attribute_grounding" not in result.details
    assert result.details["required_attribute_decision"] == "match"
