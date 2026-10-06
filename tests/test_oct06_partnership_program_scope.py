"""Check source and rule delivery; live semantic verdicts need provider replay."""

import pytest

from qualification.scoring import intent_details
from tests.test_sep24_intent_audit_regressions import _prompts, _row


@pytest.mark.parametrize("claim,source_text", [
    (
        "Acme and RetailCo announced an exclusive deployment partnership.",
        "Acme and RetailCo announced a partnership under an exclusive national "
        "agreement. Acme will supply its checkout platform and RetailCo will "
        "deploy it across its corporate and franchise stores.",
    ),
    (
        "Acme purchased software from VendorCo.",
        "Acme purchased an annual license for VendorCo's standard software.",
    ),
    (
        "Acme describes VendorCo as a strategic partner.",
        "Acme's customer list calls VendorCo a strategic partner. No agreement "
        "or collaborative commitments are described.",
    ),
])
def test_customer_supplier_scope_and_affirmative_proof_reach_both_judges(
    claim, source_text,
):
    row = _row(
        company="Acme",
        claim=claim,
        target="Announced a strategic partnership in the last 365 days.",
        source_url="https://acme.example/announcement",
        evidence_type="PARTNERSHIP",
    )
    stage_one, stage_three = _prompts(row, source_text)
    for prompt in (stage_one, stage_three, intent_details._SYSTEM):
        normalized = " ".join(prompt.split())
        assert (
            "These additional program-status requirements do not make joint "
            "go-to-market or co-development mandatory for every partnership."
        ) in normalized
        assert (
            "exact source evidence affirmatively establishes a partnership or "
            "comparable collaborative commercial arrangement with concrete "
            "commitments by both named organizations."
        ) in normalized
        assert (
            "Do not reject it solely because the parties have customer-supplier roles."
        ) in normalized
        assert (
            "An ordinary purchase or marketing partnership label alone is insufficient."
        ) in normalized
        assert "partner tier or certification" in normalized
        assert "marketplace participation" in normalized
    assert source_text in stage_three
