"""PE repair feedback explains native checks without changing their verdicts."""
import asyncio
import copy
import json
from types import SimpleNamespace

import pytest

from qualification.scoring import company_evidence_investigator as investigator
from tests.test_investigator_ownership_contract import (
    CLASSIFICATION, CLASSIFICATION_URL, CONTROL, CONTROL_URL, OWNERSHIP, finding,
)

KPMG_URL = "https://corporatefinance.kpmg.com/us/en/transactions/vytex-corporation.html"
PORTFOLIO_URL = "https://graycliffpartners.com/investments/vytex"
MAJORITY = 'Vytex Corporation, (“Vytex” or the “Company”) has received a majority investment from Graycliff Partners (“Graycliff”).'
PORTFOLIO = ("Vytex is a manufacturer and installer of vinyl windows exclusively focused on the residential repair and remodel market. "
             "Website www.vytexwindows.com Headquarters Laurel, MD Industry Manufacturing Investment Date September 2025 Status Active")
FUND_DESCRIPTION = ("Graycliff Partners is an investment firm focused on making lower middle market investments, typically in manufacturing, "
                    "business services and value-added distribution businesses. Through dedicated equity and credit funds, Graycliff provides "
                    "capital for acquisitions, management buyouts, recapitalizations, growth and expansion.")
GENERIC_REASON = "source quote did not prove current private-equity ownership"


def _validate(raw, *, company_names=None, pages=None):
    details = {}
    claim = investigator._validated_findings(
        {"findings": [raw]}, targets=("stage",),
        fetched_pages=pages or {CONTROL_URL: raw["evidence_quote"], CLASSIFICATION_URL: CLASSIFICATION},
        first_party_domains={"clear.example"},
        identity_names=company_names or {"clearcapital"},
        requested_stage="private equity", rejection_details=details,
    )["stage"]
    return claim, details


@pytest.mark.parametrize("portfolio_primary", [False, True])
def test_retained_public_635_quotes_keep_decision_and_explain_failed_links(portfolio_primary):
    # Exact public quote spans from the retained case; no run or miner data.
    raw = finding(
        evidence_url=PORTFOLIO_URL if portfolio_primary else KPMG_URL,
        evidence_quote=PORTFOLIO if portfolio_primary else MAJORITY,
        supporting_evidence_url_1=KPMG_URL if portfolio_primary else PORTFOLIO_URL,
        supporting_evidence_quote_1=MAJORITY if portfolio_primary else PORTFOLIO,
        supporting_evidence_url_2=KPMG_URL,
        supporting_evidence_quote_2=FUND_DESCRIPTION,
        ownership={**OWNERSHIP, "controlling_owner": "Graycliff Partners"},
    )
    pages = {KPMG_URL: MAJORITY + " " + FUND_DESCRIPTION, PORTFOLIO_URL: PORTFOLIO}
    before = copy.deepcopy(raw)
    claim, details = _validate(raw, company_names={"vytexwindows", "vytexcorporation"}, pages=pages)
    assert claim["status"] == "UNPROVEN"
    assert claim["reason"] == GENERIC_REASON
    assert "ownership_proof_failures" not in claim
    assert raw == before
    assert any("supporting_evidence_quote_1" in item for item in details["stage"])
    assert any("controlling owner" in item for item in details["stage"]) is portfolio_primary


def test_valid_contract_keeps_positive_without_rejection_details():
    claim, details = _validate(finding())
    assert claim["status"] == "VERIFIED"
    assert claim["ownership"] == OWNERSHIP
    assert details == {}


@pytest.mark.parametrize(("quote", "ownership", "expected"), [
    (CONTROL.replace("Clear Capital", "Other Company"), OWNERSHIP, "investigated company"),
    ("Pinecrest Capital controls Subsidiary Widgets.", OWNERSHIP, "investigated company"),
    ("Pinecrest Capital controls Parent Holdings.", OWNERSHIP, "investigated company"),
    (CONTROL, {**OWNERSHIP, "controlling_owner": "Clear Capital"}, "cannot be its own"),
    (CONTROL, {**OWNERSHIP, "control_status": "minority"}, "current_control"),
    (CONTROL, {**OWNERSHIP, "control_status": "pending"}, "current_control"),
    (CONTROL, {**OWNERSHIP, "owner_type": "other"}, "private_equity"),
    (CONTROL.replace("Pinecrest Capital", "Different Sponsor"), OWNERSHIP, "controlling owner"),
])
def test_wrong_entities_and_unproved_control_stay_rejected(quote, ownership, expected):
    claim, details = _validate(finding(evidence_quote=quote, ownership=ownership))
    assert claim["status"] == "UNPROVEN"
    if expected == "investigated company":
        # The earlier attribution guard remains authoritative. Do not obscure
        # it with detail from a later ownership check.
        assert claim["reason"] == "source quote did not identify the investigated company"
        assert details == {}
    else:
        assert claim["reason"] == GENERIC_REASON
        assert any(expected in item for item in details["stage"])


def test_quote_absence_keeps_quote_feedback_without_unrelated_contract_detail():
    raw = finding(evidence_quote="Invented quote.")
    claim, details = _validate(raw, pages={CONTROL_URL: CONTROL, CLASSIFICATION_URL: CLASSIFICATION})
    assert claim["reason"] == "submitted quote was not present in fetched source"
    assert details == {}


@pytest.mark.parametrize("repair_to_positive", [False, True])
def test_controller_receives_detail_and_can_repair_or_submit_honest_negative(monkeypatch, repair_to_positive):
    requests = []
    clock = SimpleNamespace(now=0.0)
    context_url = "https://clear.example/profile"
    raw = finding(
        supporting_evidence_url_1=context_url,
        supporting_evidence_quote_1="Clear Capital supplies financial services.",
        supporting_evidence_url_2=CLASSIFICATION_URL,
        supporting_evidence_quote_2=CLASSIFICATION,
    )
    pages = {CONTROL_URL: CONTROL, CLASSIFICATION_URL: CLASSIFICATION,
             context_url: raw["supporting_evidence_quote_1"]}
    _, expected_details = _validate(raw, pages=pages)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    monkeypatch.setenv("EXA_API_KEY", "test")
    monkeypatch.setattr(investigator, "time", SimpleNamespace(monotonic=lambda: clock.now))

    async def search(*args, **kwargs):
        return {"results": []}

    async def post(_session, _url, *, headers, payload):
        requests.append(copy.deepcopy(payload))
        assert len(requests) <= 2
        if len(requests) == 1:
            clock.now = investigator.ADMISSION_DEADLINE_SECONDS - investigator.JUDGMENT_ADMISSION_RESERVE_SECONDS
            submitted = raw
        else:
            feedback = json.loads(next(m["content"] for m in reversed(payload["messages"]) if m["role"] == "tool"))
            rejection = feedback["rejected_findings"][0]
            assert rejection["reason"] == GENERIC_REASON
            assert rejection["ownership_proof_failures"] == expected_details["stage"]
            if repair_to_positive:
                submitted = copy.deepcopy(raw)
                for field in ("url", "quote"):
                    submitted[f"supporting_evidence_{field}_1"], submitted[f"supporting_evidence_{field}_2"] = (
                        submitted[f"supporting_evidence_{field}_2"], submitted[f"supporting_evidence_{field}_1"]
                    )
            else:
                submitted = {**raw, "status": "UNPROVEN", "evidence_url": "", "evidence_quote": "", "ownership": None}
        return 200, {"choices": [{"message": {"tool_calls": [{
            "id": f"call-{len(requests)}", "type": "function",
            "function": {"name": "submit_findings", "arguments": json.dumps({"findings": [submitted]})},
        }]}}]}

    monkeypatch.setattr(investigator, "_post_json", post)
    monkeypatch.setattr(investigator, "_search_web", search)
    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={"name": "Clear Capital", "website": "https://clear.example"},
        targets=("stage",), requested_stage="Private Equity",
        prior_observations={"submitted_source_urls": list(pages)},
        prefetched_pages={url: {"text": text, "final_url": url} for url, text in pages.items()},
    ))
    assert result["_completed_submit"] is True
    assert result["failure_reason"] == ""
    assert result["claims"]["stage"]["status"] == ("VERIFIED" if repair_to_positive else "UNPROVEN")
