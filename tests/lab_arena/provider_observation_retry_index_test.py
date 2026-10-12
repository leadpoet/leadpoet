"""Original execution indices survive private scoring retries and splits."""

import asyncio
from copy import deepcopy
from types import SimpleNamespace
from urllib.parse import urlsplit

import pytest

from lab_arena import company_judgments, provider_observations, scorer_entrypoint, scoring, shim
from qualification.scoring import competition, lead_scorer
from qualification.scoring.evaluation_clock import use_evaluation_date
from tests.test_arena_company_quality_policy import company, _positive_breakdown
from tests.test_arena_score_integrity import _icp


def _companies():
    return [company(f"Company{i}", f"company{i}.example") for i in range(3)]


def _observations():
    return [{
        "company_index": i, "company_domain": f"company{i}.example",
        "source_url": f"https://company{i}.example/news/launch",
        "first_observed_date": f"2026-09-0{i + 1}",
    } for i in range(3)]


def _install_judge(monkeypatch, *, failures=None, handoff=True):
    calls, attempts, scopes = [], {}, {}
    observations = _observations()

    async def judge(**kwargs):
        c = kwargs["company"]
        index = int(urlsplit(c.company_website).hostname.split(".")[0][7:])
        observed = kwargs.get("provider_observation")
        expected = {k: v for k, v in observations[index].items() if k != "company_index"}
        assert observed == (expected if handoff else None)
        calls.append(index)
        attempts[index] = attempts.get(index, 0) + 1
        scope = kwargs["required_attribute_retry_source_cache"]
        assert scope.setdefault("mock_company", index) == index
        assert scopes.setdefault(index, scope) is scope
        row = _positive_breakdown(c.company_name, f"company{index}.example", c.company_name.lower())
        if attempts[index] <= (failures or {}).get(index, 0):
            row.update(final_score=0, icp_fit=0,
                       failure_reason="Company fit unavailable: provider timeout")
            row["verifier_gate_receipts"] = [{
                "gate": "company_fit", "decision": "unavailable",
                "company_fit_dimensions": {"identity": "unavailable"},
                "failure_reason_code": "provider_error",
            }]
        return row

    monkeypatch.setattr(lead_scorer, "score_company_competition_intent", judge)
    monkeypatch.setattr(competition, "_ensure_provider_environment", lambda: None)
    monkeypatch.setattr(scoring, "apply_policy_to_environment", lambda *a, **k: None)
    monkeypatch.setenv(shim.TRUSTED_SCORER_ENV, "0")
    return calls, scopes


def _input(companies, *, handoff=True, quality=False):
    policy = scoring.build_scorer_policy(
        scoring_adapter_version="qualification_integrity_v2",
        intent_details=True, provider_observation_handoff=handoff,
        company_quality=quality,
    )
    return scoring.build_scoring_input(
        scored_run_id="mock-execute", icp=_icp(), companies=companies,
        policy=policy, evaluation_date="2026-10-12",
        provider_observations=_observations() if handoff else None,
    )


@pytest.mark.parametrize("handoff", [False, True])
@pytest.mark.parametrize("sparse", [False, True])
def test_actual_score_input_preserves_first_and_narrowed_retry_indices(monkeypatch, handoff, sparse):
    companies = _companies()
    if sparse:
        companies[0]["employee_count"] = "10,001+"
    calls, scopes = _install_judge(monkeypatch, failures={1: 1, 2: 2}, handoff=handoff)
    document = _input(companies, handoff=handoff)
    before = deepcopy(document)
    result = scorer_entrypoint.score_input(document)

    assert calls == ([1, 2, 1, 2, 2] if sparse else [0, 1, 2, 1, 2, 2])
    assert [row["company_index"] for row in result["breakdowns"]] == ([1, 2] if sparse else [0, 1, 2])
    assert len({id(scope) for scope in scopes.values()}) == len(scopes)
    assert document == before


def test_duplicate_company_keeps_original_observation_and_prepaid_skip(monkeypatch):
    companies = _companies()
    companies[1] = deepcopy(companies[0])
    calls, _ = _install_judge(monkeypatch)
    result = scorer_entrypoint.score_input(_input(companies))
    assert calls == [0, 2]
    assert [row["company_index"] for row in result["breakdowns"]] == [0, 1, 2]
    assert [row["duplicate_company"] for row in result["breakdowns"]] == [False, True, False]


def test_quality_cache_misses_and_their_retries_keep_original_indices(monkeypatch):
    calls, _ = _install_judge(monkeypatch, failures={1: 1, 2: 2})
    document = _input(_companies(), quality=True)
    document["company_judgment_cache"] = {
        "schema_version": company_judgments.LEASE_SCHEMA_VERSION,
        "hits": [], "misses": [{
            "company_index": i, "cache_key": "sha256:" + str(i) * 64,
            "company_input_hash": "sha256:" + str(i) * 64,
            "authority_slot": 0,
        } for i in range(3)],
    }
    result = scorer_entrypoint.score_input(document)
    assert calls == [0, 1, 1, 2, 2, 2]
    assert [row["company_index"] for row in result["breakdowns"]] == [0, 1, 2]
    assert [row["company_index"] for row in result["company_judgments"]] == [0, 1, 2]


def test_quality_adapter_internal_split_keeps_original_indices(monkeypatch):
    calls, _ = _install_judge(monkeypatch)
    adapter = competition.CompetitionCompanyScorer(
        company_quality=True, provider_observations=_observations(),
    )
    with use_evaluation_date("2026-10-12"):
        rows = asyncio.run(adapter.score_with_breakdowns(
            _companies(), _icp(), False, retry_evidence_scope={},
        ))
    assert calls == [0, 1, 2]
    assert [row["company_index"] for row in rows] == [0, 1, 2]


def test_wrong_company_observation_cannot_pass_domain_or_source_binding():
    c = SimpleNamespace(company_website="https://company1.example/")
    row = _positive_breakdown("Company1", "company1.example", "company1")
    fit = SimpleNamespace(receipt=lambda _gate: row["verifier_gate_receipts"][0])
    signals = [{
        "after_decay": 60, "matched_icp_signal": 0,
        "evidence_urls": ["https://company1.example/news/launch"],
        "judge_verdict": {"decision": "verified", "client_ready": True},
    }]
    observations = [{k: v for k, v in o.items() if k != "company_index"} for o in _observations()]
    with use_evaluation_date("2026-10-12"):
        assert lead_scorer._matched_provider_observation(c, fit, signals, observations[0]) is None
        assert lead_scorer._matched_provider_observation(c, fit, signals, observations[1])["first_observed_date"] == "2026-09-02"
        foreign_url = {**observations[1], "source_url": observations[0]["source_url"]}
        assert lead_scorer._matched_provider_observation(c, fit, signals, foreign_url) is None


def test_real_company_pipeline_retains_own_authenticated_date_through_intent_retries(monkeypatch):
    from qualification.scoring import intent_details
    from qualification.scoring.company_fit_decision import company_fit_match

    companies = _companies()[:2]
    for c in companies:
        c["intent_details"] = "The company announced a product launch."
        c.pop("fit_summary")
        c.pop("fit_evidence_urls")
        for signal in c["intent_signals"]:
            signal.pop("why_now")
            signal.pop("snippet")
    dates, attempts = [], {}

    async def fit(c, *_args, **_kwargs):
        domain = urlsplit(c.company_website).hostname
        row = _positive_breakdown(c.company_name, domain, c.company_name.lower())
        return company_fit_match(details=row["verifier_gate_receipts"][0])

    async def signals(c, *_args, **_kwargs):
        domain = urlsplit(c.company_website).hostname
        signal = _positive_breakdown(c.company_name, domain, c.company_name.lower())["intent_signals_detail"][0]
        signal["evidence_urls"] = [f"https://{domain}/news/launch"]
        signal["judge_verdict"]["client_ready"] = True
        return 60, 60, 1, 90, False, [signal]

    async def paragraph(c, *_args, authenticated_provider_observation=None, **_kwargs):
        domain = urlsplit(c.company_website).hostname
        index = int(domain.split(".")[0][7:])
        assert authenticated_provider_observation == {
            "matched_icp_signal": 0,
            "source_url": f"https://{domain}/news/launch",
            "first_observed_date": f"2026-09-0{index + 1}",
        }
        dates.append((index, authenticated_provider_observation["first_observed_date"]))
        attempts[index] = attempts.get(index, 0) + 1
        unavailable = index == 1 and attempts[index] < 3
        return {
            "gate": "intent_details", "decision": "unavailable" if unavailable else "match",
            **({"failure_class": "intent_details_citation_unavailable"} if unavailable else {}),
        }

    monkeypatch.setattr(lead_scorer, "_verify_company_fit", fit)
    monkeypatch.setattr(lead_scorer, "score_company_competition_intent_signal", signals)
    monkeypatch.setattr(intent_details, "review_intent_details", paragraph)
    monkeypatch.setattr(competition, "_ensure_provider_environment", lambda: None)
    monkeypatch.setattr(scoring, "apply_policy_to_environment", lambda *a, **k: None)
    monkeypatch.setenv(shim.TRUSTED_SCORER_ENV, "0")
    result = scorer_entrypoint.score_input(_input(companies))
    assert dates == [(0, "2026-09-01"), (1, "2026-09-02"), (1, "2026-09-02"), (1, "2026-09-02")]
    assert [row["company_index"] for row in result["breakdowns"]] == [0, 1]
