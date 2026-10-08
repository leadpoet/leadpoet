"""Model finding receipts must pass the unchanged host scoring contract."""

import json

import pytest

from lab_arena import scoring
from qualification.scoring.company_evidence_investigator import _validated_findings


def _output_from_finding(raw_finding, *, fetched_pages=None):
    claims = _validated_findings(
        {"findings": [raw_finding]},
        targets=("required_attribute",),
        fetched_pages=fetched_pages or {},
        first_party_domains={"acme.example"},
        identity_names={"acme"},
    )
    assert claims is not None
    output = scoring.build_scoring_output(
        "receipt-control-regression",
        [{
            "final_score": 0,
            "verifier_gate_receipts": [{
                "gate": "company_fit",
                "supporting_receipts": [{
                    "gate": "company_evidence_investigation",
                    "claims": claims,
                }],
            }],
        }],
    )
    return claims["required_attribute"], json.dumps(output).encode("utf-8")


def test_model_reason_controls_are_normalized_before_host_scoring_parse():
    raw_reason = "No\x1fproof\x7f found\ncheck\tthe source" + "x" * 300
    finding, output = _output_from_finding({
        "target": "required_attribute",
        "status": "UNPROVEN",
        "reason": raw_reason,
    })

    assert finding["reason"] == "No proof  found\ncheck\tthe source" + "x" * (
        300 - len("No proof  found\ncheck\tthe source")
    )
    assert len(finding["reason"]) == 300
    assert scoring.scoring_output_from_bytes(output)["breakdowns"][0][
        "verifier_gate_receipts"
    ][0]["supporting_receipts"][0]["claims"]["required_attribute"] == finding


@pytest.mark.parametrize("field", ("evidence_quote", "observed_value"))
def test_reason_normalization_does_not_mask_other_invalid_receipt_fields(field):
    quote = "Acme supplies a workflow platform."
    raw = {
        "target": "required_attribute",
        "status": "VERIFIED",
        "observed_value": "workflow platform",
        "activity_role": "supplier_operator",
        "evidence_url": "https://acme.example/platform",
        "evidence_quote": quote,
        "reason": "model\x1fprose",
    }
    raw[field] += "\x1f"
    finding, output = _output_from_finding(
        raw, fetched_pages={raw["evidence_url"]: raw["evidence_quote"]},
    )

    assert finding["status"] == "VERIFIED"
    assert finding["reason"] == "model prose"
    assert "\x1f" in finding[field]
    with pytest.raises(scoring.ScoringError, match="structural limits"):
        scoring.scoring_output_from_bytes(output)
