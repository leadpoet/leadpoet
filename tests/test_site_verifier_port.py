"""Tests for the site-verifier improvements ported into the lab verifier.

Covers the four ported surfaces and — most importantly — proves the port is
OUTPUT-NEUTRAL under default flags (shadow/disabled), so benchmark scores,
champion selection, and validator weights are unchanged until an operator
explicitly enables enforcement.
"""

from __future__ import annotations

import asyncio
import importlib
import inspect
import os
import typing

import pytest

import leadpoet_verifier.industry_taxonomy as taxonomy
from leadpoet_verifier.industry_fit import b2b_saas_evidence, industry_fit
import qualification.scoring.pre_checks as pre_checks
from gateway.qualification.models import (
    CompanyOutput,
    ICPPrompt,
    IntentSignal,
    IntentSignalSource,
)


# ---------------------------------------------------------------------------
# Production runtime compatibility
# ---------------------------------------------------------------------------


def test_new_verifier_runtime_annotations_evaluate_on_python39() -> None:
    modules = (
        "leadpoet_verifier.contracts",
        "leadpoet_verifier.deepline_repair",
        "leadpoet_verifier.industry_fit",
        "leadpoet_verifier.industry_taxonomy",
        "leadpoet_verifier.semantic_gates",
        "leadpoet_verifier.identity.binding",
        "leadpoet_verifier.identity.canonical",
        "leadpoet_verifier.identity.models",
        "leadpoet_verifier.identity.normalization",
        "leadpoet_verifier.identity.observation",
        "leadpoet_verifier.identity.policy",
    )

    for module_name in modules:
        module = importlib.import_module(module_name)
        for value in vars(module).values():
            if (
                (inspect.isfunction(value) or inspect.isclass(value))
                and value.__module__ == module_name
            ):
                typing.get_type_hints(value)


# ---------------------------------------------------------------------------
# Canonical taxonomy matcher (ported industry_taxonomy.py)
# ---------------------------------------------------------------------------


def test_taxonomy_exact_parent_and_subindustry_accepts() -> None:
    ok, detail = taxonomy.leadpoet_taxonomy_match("Software", "Software", "SaaS")
    assert ok is True
    assert detail["candidate_parent_consistent"] is True


def test_taxonomy_conflicting_parent_rejects() -> None:
    # A candidate labeled with a different canonical parent must be rejected,
    # not fuzzy-passed (the lab's old gate would have passed this through).
    ok, _ = taxonomy.leadpoet_taxonomy_match("Software", "Manufacturing", "")
    assert ok is False


def test_taxonomy_unknown_provider_label_delegates() -> None:
    # Provider-specific labels outside the taxonomy return None so bounded
    # semantic concepts (not string luck) decide.
    ok, _ = taxonomy.leadpoet_taxonomy_match(
        "Software", "Computer Software Vendors", ""
    )
    assert ok is None


def test_fuel_cell_concepts_recognized() -> None:
    # Commit 3bcfca53: fuel-cell / electrolyzer / hydrogen are
    # energy-infrastructure, so buyer requests match provider labels.
    assert "energy_infrastructure" in taxonomy.industry_concepts("fuel cell systems")
    assert "energy_infrastructure" in taxonomy.industry_concepts(
        "hydrogen electrolyzer manufacturer"
    )


def test_broad_concept_suppressed_as_modifier() -> None:
    # "logistics software" must not match a plain software vendor via the
    # broad 'software' concept — broad concepts count only standalone.
    concepts, suppressed = taxonomy.requested_industry_concepts("logistics software")
    assert "software" not in concepts
    assert "software" in suppressed


# ---------------------------------------------------------------------------
# industry_fit + B2B SaaS evidence (ported adapter matchers)
# ---------------------------------------------------------------------------


def test_industry_fit_taxonomy_authoritative() -> None:
    passed, detail = industry_fit("Software", "Software", "SaaS")
    assert passed is True
    assert detail["match_strategy"] == "leadpoet_taxonomy"


def test_industry_fit_canonical_concept_for_provider_labels() -> None:
    passed, detail = industry_fit(
        "Cybersecurity", "Computer & Network Security", ""
    )
    assert passed is True
    assert detail["match_strategy"] in {"canonical_concept", "leadpoet_taxonomy"}


def test_industry_fit_rejects_unrelated() -> None:
    passed, detail = industry_fit("Software", "Food & Beverages", "")
    assert passed is False


def test_b2b_saas_needs_buyer_and_product_evidence() -> None:
    # Product label alone is NOT proof of B2B: no customer signal -> fail.
    result = b2b_saas_evidence("Software Development", "", "", "")
    assert result["passed"] is False
    # Grounding requires label- or quote-level evidence: the frozen evidence
    # quote showing a SaaS product sold to enterprises passes.
    result = b2b_saas_evidence(
        "Software Development",
        "",
        "A platform for enterprises to manage payroll",
        "Acme provides a SaaS platform for enterprises to manage payroll",
    )
    assert result["passed"] is True
    assert "saas" in result["product_signals"]
    assert result["source_grounded"] is True
    # A description alone corroborates but cannot ground (deliberate: the
    # description can merely repeat the buyer's request).
    result = b2b_saas_evidence(
        "Software Development", "", "A SaaS platform for enterprises", ""
    )
    assert result["passed"] is False


def test_b2b_saas_service_only_rejected_without_owned_software() -> None:
    result = b2b_saas_evidence(
        "IT Services",
        "",
        "A consulting agency serving businesses",
        "",
    )
    assert result["passed"] is False
    assert result["service_only_signals"]


# ---------------------------------------------------------------------------
# Shadow-mode invariance: the reward-preservation proof
# ---------------------------------------------------------------------------


def _company(industry: str = "Food & Beverages", sub: str = "") -> CompanyOutput:
    return CompanyOutput(
        company_name="Acme Corp",
        company_website="https://acmecorp.io",
        industry=industry,
        sub_industry=sub,
        employee_count="51-200",
        country="United States",
        description="",
        intent_signals=[
            IntentSignal(
                source=IntentSignalSource.NEWS,
                description="Announced expansion of data engineering team",
                url="https://technews.io/acme-hiring",
                date="2026-07-01",
                snippet="Acme Corp announced it is expanding its data engineering team",
                matched_icp_signal=0,
            )
        ],
    )


def _icp(industry: str = "Software") -> ICPPrompt:
    return ICPPrompt(
        icp_id="icp-test-1",
        industry=industry,
        sub_industry="",
        employee_count="51-200",
        company_stage="",
        country="United States",
        geography="United States",
        product_service="B2B software tools",
        intent_signals=["hiring for data engineers"],
    )


def _run_zero_checks(company: CompanyOutput, icp: ICPPrompt):
    # asyncio.run creates a fresh loop each call, immune to loop teardown by
    # earlier async tests in the module.
    return asyncio.run(
        pre_checks.run_company_zero_checks(
            company, icp, run_cost_usd=0.0, run_time_seconds=1.0, seen_companies=set()
        )
    )


def test_shadow_mode_is_output_neutral(monkeypatch, caplog) -> None:
    # A canonical industry mismatch in SHADOW mode (the default) must pass
    # exactly as before the port — only a tagged warning is emitted.
    monkeypatch.setenv("RESEARCH_LAB_TAXONOMY_INDUSTRY_GATE", "shadow")
    company, icp = _company("Food & Beverages"), _icp("Software")
    with caplog.at_level("WARNING"):
        passed, reason = _run_zero_checks(company, icp)
    assert passed is True and reason is None  # outcome unchanged
    assert any(
        "taxonomy_industry_gate_shadow_mismatch" in rec.message for rec in caplog.records
    )


def test_disabled_mode_skips_and_is_output_neutral(monkeypatch, caplog) -> None:
    monkeypatch.setenv("RESEARCH_LAB_TAXONOMY_INDUSTRY_GATE", "disabled")
    with caplog.at_level("WARNING"):
        passed, reason = _run_zero_checks(_company("Food & Beverages"), _icp("Software"))
    assert passed is True and reason is None
    assert not any("taxonomy_industry_gate" in rec.message for rec in caplog.records)


def test_enforce_mode_zeroes_canonical_mismatch(monkeypatch) -> None:
    monkeypatch.setenv("RESEARCH_LAB_TAXONOMY_INDUSTRY_GATE", "enforce")
    passed, reason = _run_zero_checks(_company("Food & Beverages"), _icp("Software"))
    assert passed is False
    assert "canonical taxonomy" in (reason or "")


def test_enforce_mode_passes_true_match(monkeypatch) -> None:
    monkeypatch.setenv("RESEARCH_LAB_TAXONOMY_INDUSTRY_GATE", "enforce")
    passed, reason = _run_zero_checks(_company("Software", "SaaS"), _icp("Software"))
    assert passed is True and reason is None


def test_gate_fails_open_on_internal_error(monkeypatch) -> None:
    # Availability over strictness: a broken matcher must never take down
    # scoring, even in enforce mode (it logs a tagged warning instead).
    monkeypatch.setenv("RESEARCH_LAB_TAXONOMY_INDUSTRY_GATE", "enforce")

    def _boom(*args, **kwargs):
        raise RuntimeError("taxonomy unavailable")

    import leadpoet_verifier.industry_fit as fit_mod

    monkeypatch.setattr(fit_mod, "industry_fit", _boom)
    passed, reason = _run_zero_checks(_company("Food & Beverages"), _icp("Software"))
    assert passed is True and reason is None


def test_invalid_mode_falls_back_to_shadow(monkeypatch) -> None:
    monkeypatch.setenv("RESEARCH_LAB_TAXONOMY_INDUSTRY_GATE", "banana")
    assert pre_checks._taxonomy_industry_gate_mode() == "shadow"


# ---------------------------------------------------------------------------
# Semantic gates: default-off + SSRF guard + acronym safety (ported module)
# ---------------------------------------------------------------------------


def test_semantic_gates_default_disabled(monkeypatch) -> None:
    import leadpoet_verifier.semantic_gates as sg

    monkeypatch.delenv("VERIFIER_SEMANTIC_GATES_MODE", raising=False)
    assert sg.semantic_gate_mode() == "disabled"


def test_semantic_gates_ssrf_guard() -> None:
    import leadpoet_verifier.semantic_gates as sg

    assert sg.is_safe_public_url("https://example.com/about") is True
    for bad in (
        "http://169.254.169.254/latest/meta-data",
        "http://127.0.0.1:8000/",
        "http://10.0.0.5/",
        "https://user:pass@example.com/",
        "https://gateway.internal/",
        "ftp://example.com/",
    ):
        assert sg.is_safe_public_url(bad) is False, bad


def test_short_acronym_entity_matching() -> None:
    # Commit 0197d5f6: <4-char acronyms match only as whole uppercase tokens.
    import leadpoet_verifier.semantic_gates as sg

    assert sg._short_external_entity_match("abc", "ABC announces a new product")
    assert not sg._short_external_entity_match("abc", "abcdef industries update")
    assert not sg._short_external_entity_match("abc", "the fabric company")


# ---------------------------------------------------------------------------
# Identity resolution (ported identity/ package) — pure policy basics
# ---------------------------------------------------------------------------


def test_identity_psl_registrable_domain() -> None:
    from leadpoet_verifier.identity.normalization import normalize_host

    parts = normalize_host("app.acme.co.uk")
    # PSL-aware: registrable domain is acme.co.uk, NOT co.uk (the naive
    # rsplit('.') parsing elsewhere would get this wrong).
    assert parts.registrable_domain == "acme.co.uk"
    assert parts.public_suffix == "co.uk"


def test_identity_linkedin_url_canonicalization() -> None:
    from leadpoet_verifier.identity.normalization import (
        normalize_linkedin_company_url,
    )

    a = normalize_linkedin_company_url(
        "https://www.linkedin.com/company/acme-corp/?utm=x"
    )
    b = normalize_linkedin_company_url("https://linkedin.com/company/acme-corp")
    assert a == b


def test_identity_canonical_hash_forbids_floats() -> None:
    from leadpoet_verifier.identity.canonical import canonical_sha256

    with pytest.raises(Exception):
        canonical_sha256({"score": 1.5})
    # Deterministic over key order.
    assert canonical_sha256({"a": 1, "b": "x"}) == canonical_sha256({"b": "x", "a": 1})


# ---------------------------------------------------------------------------
# Bounded intent-corroboration rescue (ported into three-stage verification)
# ---------------------------------------------------------------------------


def test_corroboration_rescue_default_off(monkeypatch) -> None:
    import qualification.scoring.intent_verification_three_stage as t3

    monkeypatch.delenv("RESEARCH_LAB_INTENT_CORROBORATION_RESCUE", raising=False)
    # Default OFF: the call site short-circuits before eligibility, so the
    # three-stage pipeline behaves byte-for-byte as before the port.
    assert t3._corroboration_rescue_enabled() is False
    monkeypatch.setenv("RESEARCH_LAB_INTENT_CORROBORATION_RESCUE", "true")
    assert t3._corroboration_rescue_enabled() is True


def test_corroboration_eligibility_is_narrow() -> None:
    import qualification.scoring.intent_verification_three_stage as t3

    row = {"signal_date": "2026-07-01"}
    good = {
        "signal_status": "supported",
        "confidence": "medium",
        "same_entity_check": "pass",
        "claim_matches_miner_date": "supported",
    }
    assert t3._medium_corroboration_eligible(row, good, "review") is True
    # Every deviation disqualifies: wrong decision, low confidence, entity
    # failure, contradicted date, or a dateless claim.
    assert t3._medium_corroboration_eligible(row, good, "reject") is False
    assert t3._medium_corroboration_eligible(row, {**good, "confidence": "low"}, "review") is False
    assert t3._medium_corroboration_eligible(row, {**good, "same_entity_check": "fail"}, "review") is False
    assert t3._medium_corroboration_eligible(
        row, {**good, "claim_matches_miner_date": "contradicted"}, "review"
    ) is False
    assert t3._medium_corroboration_eligible({}, good, "review") is False


def test_independent_corroboration_filters_dependent_sources() -> None:
    import qualification.scoring.intent_verification_three_stage as t3

    filler = " ".join(f"tok{i}" for i in range(120))
    original_text = f"Acme announced its series B funding round {filler}"
    original = {"results": [{"url": "https://news.acmewire.com/a", "text": original_text}]}
    original_urls = [
        "https://news.acmewire.com/a",
        "https://www.globenewswire.com/release/acme",
    ]
    candidates = {
        "results": [
            {"url": "https://news.acmewire.com/b", "text": "same host different path"},
            {"url": "https://globenewswire.com/other/acme", "text": "any"},
            {"url": "https://independent-tech-daily.com/acme", "text": original_text},
            {
                "url": "https://reporter-desk.com/acme-analysis",
                "text": "An independent analysis of the announcement with original reporting",
            },
        ]
    }
    out = t3._independent_corroboration(original, candidates, original_urls)
    accepted_urls = [r["url"] for r in out["results"]]
    reasons = {e["url"]: e["reason"] for e in out["excluded"]}
    assert accepted_urls == ["https://reporter-desk.com/acme-analysis"]
    assert reasons["https://news.acmewire.com/b"] == "same_domain"
    # Canonical-host dedup catches the wire domain before family logic.
    assert reasons["https://globenewswire.com/other/acme"] == "same_domain"
    assert reasons["https://independent-tech-daily.com/acme"] == "near_duplicate"


def test_wire_family_and_syndication_detection() -> None:
    import qualification.scoring.intent_verification_three_stage as t3

    assert t3._wire_family("https://www.globenewswire.com/x") == "globenewswire"
    assert t3._wire_family("https://independent-tech-daily.com/x") is None
    # Mirrors carry the wire distribution marker even on independent hosts.
    assert t3._looks_like_wire_syndication(
        "NEW YORK (GLOBE NEWSWIRE) -- Acme Corp today announced", "globenewswire"
    ) is True
    assert t3._looks_like_wire_syndication(
        "An original analysis without wire markers", "globenewswire"
    ) is False


# ---------------------------------------------------------------------------
# lead_scorer hardening ports: fail-closed buckets + provider-outage fairness
# ---------------------------------------------------------------------------


def test_icp_buckets_fail_closed_on_malformed() -> None:
    from qualification.scoring.lead_scorer import _normalize_icp_employee_buckets

    # Real bands verify; malformed requirements are UNVERIFIED (cannot match)
    # instead of silently disabling the size gate (the old fail-open hole).
    assert _normalize_icp_employee_buckets(["11-50", "51-200"]) == (
        {"11-50", "51-200"},
        True,
    )
    assert _normalize_icp_employee_buckets("about 500") == (set(), False)
    assert _normalize_icp_employee_buckets(["11-50", "garbage"]) == (set(), False)
    # Proven against live data: 560 production ICPs across 37 daily sets all
    # parse verified, so this changes nothing on real benchmarks.


@pytest.mark.asyncio
async def test_provider_outage_records_verifier_error_not_content_reject(
    monkeypatch,
) -> None:
    # A three-stage result that failed for PROVIDER reasons (llm_error /
    # stage3_llm_error) must be recorded as rejected_verifier_error so the
    # evaluator's fail-open path (infrastructure failure != falsified intent)
    # actually triggers; content rejects keep rejected_three_stage.
    import qualification.scoring.lead_scorer as ls
    import qualification.scoring.intent_verification_three_stage as t3

    def stub_result(reason, s3_status, decision="reject"):
        async def fake_verify(client, **kwargs):
            return {
                "client_ready": False,
                "decision": decision,
                "rejection_reason": reason,
                "stage1": {"status": "review"},
                "stage3": {"status": s3_status},
                "scrape": {"statuses": [], "result_count": 1},
                "verdict": {},
            }
        return fake_verify

    signal = IntentSignal(
        source=IntentSignalSource.NEWS,
        description="Announced expansion of data engineering team",
        url="https://technews.io/acme-hiring",
        date="2026-07-01",
        snippet="Acme Corp announced it is expanding its data engineering team",
        matched_icp_signal=0,
    )
    icp = _icp("Software")

    async def run_with(reason, s3_status):
        verdicts: list = []
        monkeypatch.setattr(t3, "verify_three_stage", stub_result(reason, s3_status))
        score, *_ = await ls._score_single_intent_signal(
            signal,
            icp,
            None,
            "Acme Corp",
            company_website="https://acmecorp.io",
            api_key="test-key",
            llm_only_intent_gate=True,
            verdict_out=verdicts,
        )
        assert score == 0.0
        return verdicts[-1]["decision"]

    # Provider outage -> verifier_error (evaluator fails open).
    assert await run_with("stage3_llm_error:timeout", "llm_error") == (
        "rejected_verifier_error"
    )
    # Genuine content rejection -> unchanged classification.
    assert await run_with("stage3_contradicted", "contradicted") == (
        "rejected_three_stage"
    )


# ---------------------------------------------------------------------------
# Semantic-gate wiring: rescue ambiguity only, never canonical conflicts
# ---------------------------------------------------------------------------


class _FakeSemanticResult:
    def __init__(self, outcome: str) -> None:
        self.outcome = outcome


class _FakeEvaluator:
    def __init__(self, outcome: str, calls: list) -> None:
        self._outcome = outcome
        self._calls = calls

    async def evaluate_industry(self, **kwargs):
        self._calls.append(kwargs)
        return _FakeSemanticResult(self._outcome)


def _install_semantic(monkeypatch, mode: str, outcome: str):
    import leadpoet_verifier.semantic_gates as sg

    calls: list = []
    monkeypatch.setattr(sg, "semantic_gate_mode", lambda value=None: mode)
    monkeypatch.setattr(
        sg.SemanticGateEvaluator,
        "from_env",
        classmethod(lambda cls, **kw: _FakeEvaluator(outcome, calls)),
    )
    return calls


def test_semantic_disabled_never_constructs_evaluator(monkeypatch) -> None:
    import leadpoet_verifier.semantic_gates as sg

    monkeypatch.setenv("RESEARCH_LAB_TAXONOMY_INDUSTRY_GATE", "enforce")
    monkeypatch.delenv("VERIFIER_SEMANTIC_GATES_MODE", raising=False)

    def _boom(cls, **kw):
        raise AssertionError("evaluator must not be constructed when disabled")

    monkeypatch.setattr(sg.SemanticGateEvaluator, "from_env", classmethod(_boom))
    # Ambiguous mismatch: unknown provider label, no concepts -> enforce zeroes
    # WITHOUT ever touching the semantic evaluator.
    passed, reason = _run_zero_checks(
        _company("Bespoke Provider Label Xyz"), _icp("Software")
    )
    assert passed is False and "canonical taxonomy" in (reason or "")


def test_semantic_enforce_rescues_ambiguous_label(monkeypatch) -> None:
    monkeypatch.setenv("RESEARCH_LAB_TAXONOMY_INDUSTRY_GATE", "enforce")
    calls = _install_semantic(monkeypatch, "enforce", "passed")
    passed, reason = _run_zero_checks(
        _company("Bespoke Provider Label Xyz"), _icp("Software")
    )
    assert passed is True and reason is None  # semantic match rescued it
    assert len(calls) == 1
    assert calls[0]["requested_industry"] == "Software"


def test_semantic_never_rescues_canonical_conflict(monkeypatch) -> None:
    # Site-faithful invariant: a canonical taxonomy REJECTION is final — the
    # semantic judge is not even consulted, whatever it would say.
    monkeypatch.setenv("RESEARCH_LAB_TAXONOMY_INDUSTRY_GATE", "enforce")
    calls = _install_semantic(monkeypatch, "enforce", "passed")
    passed, reason = _run_zero_checks(_company("Manufacturing"), _icp("Software"))
    assert passed is False
    assert calls == []  # judge never consulted for canonical conflicts


def test_semantic_no_match_keeps_enforce_zero(monkeypatch) -> None:
    monkeypatch.setenv("RESEARCH_LAB_TAXONOMY_INDUSTRY_GATE", "enforce")
    calls = _install_semantic(monkeypatch, "enforce", "failed")
    passed, reason = _run_zero_checks(
        _company("Bespoke Provider Label Xyz"), _icp("Software")
    )
    assert passed is False
    assert len(calls) == 1


# ---------------------------------------------------------------------------
# PR-28 audit fix: the FULL taxonomy x semantic x provider decision matrix.
# Semantic shadow must NEVER change the outcome; semantic-enforce provider
# unavailability must fail OPEN (never a verdict); canonical conflicts are
# never rescued in any combination.
# ---------------------------------------------------------------------------


class _MatrixEvaluator:
    def __init__(self, behavior: str, calls: list) -> None:
        self._behavior = behavior
        self._calls = calls

    async def evaluate_industry(self, **kwargs):
        self._calls.append(kwargs)
        if self._behavior == "error":
            raise RuntimeError("provider outage")
        return _FakeSemanticResult("passed" if self._behavior == "match" else "failed")


def _install_matrix(monkeypatch, semantic_mode: str, behavior: str):
    import leadpoet_verifier.semantic_gates as sg

    calls: list = []
    monkeypatch.setattr(sg, "semantic_gate_mode", lambda value=None: semantic_mode)
    if behavior == "construct_error":
        def _boom(cls, **kw):
            raise RuntimeError("no api key")
        monkeypatch.setattr(sg.SemanticGateEvaluator, "from_env", classmethod(_boom))
    else:
        monkeypatch.setattr(
            sg.SemanticGateEvaluator,
            "from_env",
            classmethod(lambda cls, **kw: _MatrixEvaluator(behavior, calls)),
        )
    return calls


AMBIGUOUS = ("Bespoke Provider Label Xyz", "Software")   # taxonomy silent
CONFLICT = ("Manufacturing", "Software")                  # canonical conflict


def test_matrix_taxonomy_shadow_is_always_output_neutral(monkeypatch) -> None:
    monkeypatch.setenv("RESEARCH_LAB_TAXONOMY_INDUSTRY_GATE", "shadow")
    for semantic_mode in ("disabled", "shadow", "enforce"):
        for behavior in ("match", "no_match", "error", "construct_error"):
            _install_matrix(monkeypatch, semantic_mode, behavior)
            for cand, req in (AMBIGUOUS, CONFLICT):
                passed, reason = _run_zero_checks(_company(cand), _icp(req))
                assert passed is True and reason is None, (
                    f"taxonomy shadow must pass: semantic={semantic_mode} "
                    f"behavior={behavior} cand={cand}"
                )


def test_matrix_semantic_shadow_never_changes_enforce_outcome(monkeypatch) -> None:
    # THE PR-28 AUDIT BUG: semantic shadow used to rescue under taxonomy
    # enforce. It must behave exactly like semantic disabled.
    monkeypatch.setenv("RESEARCH_LAB_TAXONOMY_INDUSTRY_GATE", "enforce")
    for behavior in ("match", "no_match", "error", "construct_error"):
        _install_matrix(monkeypatch, "shadow", behavior)
        passed, reason = _run_zero_checks(_company(*AMBIGUOUS[:1]), _icp(AMBIGUOUS[1]))
        assert passed is False, (
            f"semantic SHADOW must not rescue (behavior={behavior})"
        )
        passed, _ = _run_zero_checks(_company(CONFLICT[0]), _icp(CONFLICT[1]))
        assert passed is False


def test_matrix_semantic_enforce_decides_ambiguous_only(monkeypatch) -> None:
    monkeypatch.setenv("RESEARCH_LAB_TAXONOMY_INDUSTRY_GATE", "enforce")
    # match -> rescued
    _install_matrix(monkeypatch, "enforce", "match")
    passed, _ = _run_zero_checks(_company(AMBIGUOUS[0]), _icp(AMBIGUOUS[1]))
    assert passed is True
    # no-match -> zero
    _install_matrix(monkeypatch, "enforce", "no_match")
    passed, _ = _run_zero_checks(_company(AMBIGUOUS[0]), _icp(AMBIGUOUS[1]))
    assert passed is False
    # canonical conflict -> zero even on match, judge never consulted
    calls = _install_matrix(monkeypatch, "enforce", "match")
    passed, _ = _run_zero_checks(_company(CONFLICT[0]), _icp(CONFLICT[1]))
    assert passed is False
    assert calls == []


def test_matrix_semantic_enforce_unavailability_fails_open(monkeypatch) -> None:
    # THE SECOND PR-28 AUDIT BUG: provider unavailability used to hard-zero.
    # Unavailability is never a verdict — ambiguous labels FAIL OPEN.
    monkeypatch.setenv("RESEARCH_LAB_TAXONOMY_INDUSTRY_GATE", "enforce")
    for behavior in ("error", "construct_error"):
        _install_matrix(monkeypatch, "enforce", behavior)
        passed, reason = _run_zero_checks(_company(AMBIGUOUS[0]), _icp(AMBIGUOUS[1]))
        assert passed is True and reason is None, f"must fail open on {behavior}"
        # ...but canonical conflicts still zero (unavailability changes nothing).
        passed, _ = _run_zero_checks(_company(CONFLICT[0]), _icp(CONFLICT[1]))
        assert passed is False


def test_gate_receipts_are_durable_and_complete(monkeypatch) -> None:
    # Audit fix #4: durable receipts carry modes, deterministic detail,
    # semantic judgment, and the final scoring effect.
    monkeypatch.setenv("RESEARCH_LAB_TAXONOMY_INDUSTRY_GATE", "enforce")
    _install_matrix(monkeypatch, "enforce", "match")
    receipts: list = []
    passed, reason = asyncio.run(
        pre_checks.run_company_zero_checks(
            _company(AMBIGUOUS[0]),
            _icp(AMBIGUOUS[1]),
            run_cost_usd=0.0,
            run_time_seconds=1.0,
            seen_companies=set(),
            gate_receipts=receipts,
        )
    )
    assert passed is True
    assert len(receipts) == 1
    r = receipts[0]
    assert r["gate"] == "taxonomy_industry"
    assert r["taxonomy_mode"] == "enforce"
    assert r["semantic_mode"] == "enforce"
    assert r["final_effect"] == "rescued_semantic_enforce"
    assert r["deterministic"]["passed"] is False
    assert r["semantic_verdict"] is True


# ---------------------------------------------------------------------------
# PR-28 audit fix: country alias + unknown-code handling
# ---------------------------------------------------------------------------


def test_country_the_prefixed_official_names_resolve() -> None:
    from qualification.scoring.pre_checks import check_country_match

    assert check_country_match("The Bahamas", "BS").passed is True
    assert check_country_match("BS", "The Bahamas").passed is True
    assert check_country_match("The Gambia", "GM").passed is True
    assert check_country_match("GM", "The Gambia").passed is True
    assert check_country_match("The Netherlands", "Netherlands").passed is True


def test_country_unknown_code_fails_closed_prose_still_defers() -> None:
    from qualification.scoring.pre_checks import check_country_match

    # An unknown code-shaped geography must NOT accept everything.
    assert check_country_match("Brazil", "ZZ").passed is False
    assert check_country_match("ZZ", "Brazil").passed is False
    # Multi-region prose keeps the documented deferral contract.
    assert check_country_match("Germany", "EMEA").passed is True
    assert check_country_match("Japan", "APAC").passed is True


def test_country_eu_shorthand_expands_to_europe() -> None:
    from qualification.scoring.pre_checks import check_country_match

    assert check_country_match("Germany", "EU").passed is True
    assert check_country_match("France", "EU").passed is True
    assert check_country_match("Brazil", "EU").passed is False


@pytest.mark.asyncio
async def test_gate_receipts_persist_into_breakdown_end_to_end(monkeypatch) -> None:
    # Durability proof: the receipt must survive all the way into the
    # LeadScoreBreakdown the evaluator persists — not just the out-param.
    from qualification.scoring.lead_scorer import (
        score_company_autoresearch_intent_v2,
    )

    monkeypatch.setenv("RESEARCH_LAB_TAXONOMY_INDUSTRY_GATE", "enforce")
    monkeypatch.delenv("VERIFIER_SEMANTIC_GATES_MODE", raising=False)
    # Canonical conflict zeroes deterministically at pre-checks: no network.
    breakdown = await score_company_autoresearch_intent_v2(
        _company("Manufacturing"),
        _icp("Software"),
        run_cost_usd=0.0,
        run_time_seconds=1.0,
        seen_companies=set(),
    )
    assert breakdown.final_score == 0
    assert "canonical taxonomy" in (breakdown.failure_reason or "")
    receipts = breakdown.verifier_gate_receipts
    assert receipts and receipts[0]["gate"] == "taxonomy_industry"
    assert receipts[0]["final_effect"] == "zeroed"
    assert receipts[0]["taxonomy_mode"] == "enforce"
    # Round-trips through the model layer (what the evaluator serializes).
    dumped = breakdown.model_dump()
    assert dumped["verifier_gate_receipts"][0]["final_effect"] == "zeroed"


@pytest.mark.asyncio
async def test_clean_pass_carries_no_receipt(monkeypatch) -> None:
    # Payload discipline: a trivial deterministic pass must NOT attach an
    # audit receipt — otherwise the default shadow mode would bloat every
    # persisted breakdown in every benchmark.
    from qualification.scoring.lead_scorer import _run_autoresearch_binary_fit_checks

    monkeypatch.setenv("RESEARCH_LAB_TAXONOMY_INDUSTRY_GATE", "shadow")
    receipts: list = []
    passed, reason = await pre_checks.run_company_zero_checks(
        _company("Software", "SaaS"),
        _icp("Software"),
        run_cost_usd=0.0,
        run_time_seconds=1.0,
        seen_companies=set(),
        gate_receipts=receipts,
    )
    assert passed is True and reason is None
    assert receipts == []  # nothing to audit on a clean deterministic pass
