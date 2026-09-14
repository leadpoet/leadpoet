import json

import pytest

from lab_arena import contracts, intent_details_policy
from lab_arena.output import OutputInvalid, output_document_from_bytes, validate_output_document
from qualification.contact_models import validate_contact_claim


def _company(name: str = "Acme") -> dict:
    return {
        "company_name": name,
        "company_website": f"https://{name.casefold()}.example",
        "company_linkedin": "",
        "industry": "Software",
        "employee_count": "51-200",
        "company_stage": "Series A",
        "country": "United States",
        "state": "",
        "fit_summary": "Matches the target market.",
        "fit_evidence_urls": [f"https://{name.casefold()}.example/about"],
        "intent_signals": [
            {
                "matched_icp_signal": 0,
                "description": "Hiring platform engineers",
                "date": "2026-09-01",
                "why_now": "The team is growing.",
                "url": f"https://{name.casefold()}.example/jobs",
                "snippet": "Open platform engineering roles",
            }
        ],
        "required_attribute": None,
    }


def _contact() -> dict:
    return {
        "full_name": "  Ada   Lovelace ",
        "role": " VP Engineering ",
        "linkedin_url": "https://uk.linkedin.com/in/Ada-Lovelace?trk=campaign#bio",
        "location": {"country": "United States of America", "region": " New York "},
        "email": "ADA@EXAMPLE.COM",
        "email_source": {
            "provider": "HarvestAPI",
            "tool": "harvestapi_get_profile",
            "broker_call_id": "call-123",
        },
    }


def _v5_company(name: str = "Acme") -> dict:
    row = _company(name)
    row.pop("fit_summary")
    row.pop("fit_evidence_urls")
    signal = row["intent_signals"][0]
    signal.pop("why_now")
    signal.pop("snippet")
    row["intent_details"] = (
        "Acme is hiring platform engineers after announcing a new product, "
        "which makes its infrastructure team likely to evaluate tools now."
    )
    row["contact"] = _contact()
    return row


def test_contact_claim_normalizes_identity_location_and_source() -> None:
    normalized = validate_contact_claim(_contact())
    assert normalized == {
        "full_name": "Ada Lovelace",
        "role": "VP Engineering",
        "linkedin_url": "https://www.linkedin.com/in/Ada-Lovelace/",
        "location": {"country": "US", "region": "New York"},
        "email": "ada@example.com",
        "email_source": {
            "provider": "harvestapi",
            "tool": "harvestapi_get_profile",
            "broker_call_id": "call-123",
        },
    }


@pytest.mark.parametrize(
    "mutate",
    [
        lambda row: row.pop("email"),
        lambda row: row.update(linkedin_url="https://linkedin.com/company/acme"),
        lambda row: row["location"].update(country="not-a-country"),
        lambda row: row["email_source"].update(tool="arbitrary_lookup"),
        lambda row: row["email_source"].pop("broker_call_id"),
        lambda row: row.update(email="password=secret-value"),
        lambda row: row.update(email=".ada@example.com"),
        lambda row: row.update(email="ada.@example.com"),
        lambda row: row.update(email="ada..lovelace@example.com"),
        lambda row: row.update(email=f"{'a' * 65}@example.com"),
    ],
)
def test_contact_claim_rejects_missing_unsupported_or_unsafe_values(mutate) -> None:
    claim = _contact()
    mutate(claim)
    with pytest.raises(ValueError):
        validate_contact_claim(claim)


def test_contact_source_accepts_record_id_without_broker_call_id() -> None:
    claim = _contact()
    claim["email_source"] = {
        "provider": "harvestapi",
        "tool": "harvestapi_get_profile",
        "record_id": "profile:ada-1",
    }
    assert validate_contact_claim(claim)["email_source"]["record_id"] == "profile:ada-1"


def test_v2_preserves_invalid_contacts_for_row_level_rejection() -> None:
    valid = _company("Acme")
    valid["contact"] = _contact()
    invalid = _company("Beta")
    invalid["contact"] = {"full_name": "Missing everything else"}
    missing = _company("Gamma")
    document = output_document_from_bytes(
        json.dumps([valid, invalid, missing]).encode(),
        expected_schema_version=contracts.CONTACT_OUTPUT_DOCUMENT_SCHEMA_VERSION,
    )
    assert document["schema_version"] == contracts.CONTACT_OUTPUT_DOCUMENT_SCHEMA_VERSION
    assert document["companies"][0]["contact"] == _contact()
    assert document["companies"][1]["contact"] == invalid["contact"]
    assert document["companies"][2]["contact"] is None


def test_v1_default_still_rejects_contact_and_v2_rejects_extra_company_fields() -> None:
    company = _company()
    company["contact"] = _contact()
    with pytest.raises(OutputInvalid):
        output_document_from_bytes(json.dumps([company]).encode())

    company["unsupported"] = True
    with pytest.raises(OutputInvalid):
        output_document_from_bytes(
            json.dumps([company]).encode(),
            expected_schema_version=contracts.CONTACT_OUTPUT_DOCUMENT_SCHEMA_VERSION,
        )


def test_stored_read_infers_only_a_declared_known_version() -> None:
    company = _company()
    company["contact"] = None
    document = {
        "schema_version": contracts.CONTACT_OUTPUT_DOCUMENT_SCHEMA_VERSION,
        "companies": [company],
    }
    assert validate_output_document(document)["schema_version"] == document["schema_version"]
    with pytest.raises(OutputInvalid):
        validate_output_document(
            document,
            expected_schema_version=contracts.OUTPUT_DOCUMENT_SCHEMA_VERSION,
        )
    with pytest.raises(OutputInvalid):
        validate_output_document({"companies": [company]})


def test_v5_replaces_fit_and_signal_prose_with_one_intent_paragraph() -> None:
    company = _v5_company()
    document = output_document_from_bytes(
        json.dumps([company]).encode(),
        expected_schema_version=intent_details_policy.OUTPUT_SCHEMA,
    )

    assert document["schema_version"] == intent_details_policy.OUTPUT_SCHEMA
    assert document["companies"][0]["intent_details"] == company["intent_details"]
    assert set(document["companies"][0]["intent_signals"][0]) == {
        "matched_icp_signal", "description", "date", "url",
    }
    for removed in ("fit_summary", "fit_evidence_urls"):
        changed = _v5_company()
        changed[removed] = "old" if removed == "fit_summary" else []
        with pytest.raises(OutputInvalid):
            output_document_from_bytes(
                json.dumps([changed]).encode(),
                expected_schema_version=intent_details_policy.OUTPUT_SCHEMA,
            )
    for removed in ("why_now", "snippet"):
        changed = _v5_company()
        changed["intent_signals"][0][removed] = "old"
        with pytest.raises(OutputInvalid):
            output_document_from_bytes(
                json.dumps([changed]).encode(),
                expected_schema_version=intent_details_policy.OUTPUT_SCHEMA,
            )


@pytest.mark.parametrize(
    "value",
    ["", "First paragraph.\n\nSecond paragraph.", "- list item"],
)
def test_v5_requires_one_plain_intent_details_paragraph(value: str) -> None:
    company = _v5_company()
    company["intent_details"] = value
    with pytest.raises(OutputInvalid):
        output_document_from_bytes(
            json.dumps([company]).encode(),
            expected_schema_version=intent_details_policy.OUTPUT_SCHEMA,
        )


def test_v4_remains_strictly_unchanged_after_v5() -> None:
    company = _company()
    company["contact"] = _contact()
    document = output_document_from_bytes(
        json.dumps([company]).encode(),
        expected_schema_version=contracts.CONTACT_OUTPUT_DOCUMENT_SCHEMA_VERSION,
    )
    assert document["companies"][0]["fit_summary"] == company["fit_summary"]
    company["intent_details"] = "New prose must not enter a frozen v4 round."
    with pytest.raises(OutputInvalid):
        output_document_from_bytes(
            json.dumps([company]).encode(),
            expected_schema_version=contracts.CONTACT_OUTPUT_DOCUMENT_SCHEMA_VERSION,
        )
