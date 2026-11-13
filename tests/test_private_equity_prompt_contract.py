"""Document the existing lexical gate without changing accepted ownership proof."""

from qualification.scoring import company_evidence_investigator as inv
from tests.test_investigator_ownership_contract import (
    CONTROL, CONTROL_URL, CLASSIFICATION_URL, OWNERSHIP, finding, validate,
)

FUNDS = (
    "Graycliff Partners is an investment firm focused on making lower middle market "
    "investments, typically in manufacturing, business services and value-added "
    "distribution businesses. Through dedicated equity and credit funds, Graycliff "
    "provides capital for acquisitions, management buyouts, recapitalizations, growth and expansion."
)


def test_system_and_schema_explain_conditional_literal_grounding():
    system = " ".join(inv._SYSTEM_PROMPT.split())
    assert 'literally say "private equity"' in system
    assert "supporting_evidence_quote_1" in system
    assert "Absent accepted direct/linked literal ownership proof" in system
    assert "Generic equity funds do not qualify" in system
    schema = inv._tools(("stage",))[-1]["parameters"]["properties"][
        "findings"]["items"]["properties"]["ownership"]["properties"]
    assert "literal words 'private equity'" in schema["owner_type"]["description"]
    assert "Existing literal private-markets firm proofs remain valid" in schema["owner_type"]["description"]


def test_actual_fund_description_rejection_explains_literal_requirement():
    failures = []
    assert not inv._semantic_ownership_proof(
        {**OWNERSHIP, "controlling_owner": "Graycliff Partners"},
        "Graycliff Partners holds 70 percent of all votes at Clear Capital.",
        {"clearcapital"}, classification_quote=FUNDS, failure_details=failures,
    )
    assert any("literal words 'private equity'" in failure for failure in failures)
    assert any("Investment firm or dedicated equity/credit funds alone" in failure for failure in failures)


def test_literal_private_markets_direct_and_linked_proofs_remain_valid():
    direct = "Clear Capital is controlled by Pinecrest Capital, a private-markets firm."
    assert validate(finding(evidence_quote=direct))["status"] == "VERIFIED"
    linked = "Clear Capital is controlled by Pinecrest Capital."
    classification = "Pinecrest Capital is a private-markets firm."
    assert validate(finding(evidence_quote=linked, supporting_evidence_quote_1=classification), pages={
        CONTROL_URL: linked, CLASSIFICATION_URL: classification,
    })["status"] == "VERIFIED"


def test_no_literal_route_still_requires_classification_and_current_control():
    classification = "Pinecrest Capital is an investment firm with dedicated equity funds."
    assert validate(finding(supporting_evidence_quote_1=classification), pages={
        CONTROL_URL: CONTROL, CLASSIFICATION_URL: classification,
    })["status"] == "UNPROVEN"
    assert validate(finding(ownership={**OWNERSHIP, "control_status": "minority"}))["status"] == "UNPROVEN"
