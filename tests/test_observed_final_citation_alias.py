"""Final citations use only exact, unambiguous retained transport bindings."""
from copy import deepcopy

import pytest

from qualification.scoring import company_evidence_investigator as inv


REQUEST = "https://example.com/about/"
FINAL = "https://example.com/about"
QUOTE = "Example headquarters are in Singapore."
DEFAULT_FINALS = object()
ANCHOR = {"submitted_domain": "example.com", "observed_domain": "example.com"}


def geography(*, citation=FINAL, pages=None, finals=DEFAULT_FINALS, quote=QUOTE):
    return inv._validated_findings(
        {"findings": [{"target": "geography", "status": "VERIFIED",
                       "observed_country": "Singapore", "observed_state": "",
                       "evidence_url": citation, "evidence_quote": quote}]},
        targets=("geography",),
        fetched_pages={REQUEST: QUOTE} if pages is None else pages,
        fetched_final_urls={REQUEST: FINAL} if finals is DEFAULT_FINALS else finals,
        first_party_domains={"example.com"}, identity_names={"example"},
        identity_anchor=ANCHOR,
    )["geography"]


@pytest.mark.parametrize("requested,final", [
    (REQUEST, FINAL), (FINAL, REQUEST),
    ("https://example.com/old", "https://example.com/new"),
])
def test_exact_observed_final_alias_preserves_requested_source(requested, final):
    finding = geography(citation=final, pages={requested: QUOTE}, finals={requested: final})
    assert finding["status"] == "VERIFIED"
    assert finding["evidence_url"] == requested
    assert finding["evidence_quote"] == QUOTE


def test_direct_retained_request_does_not_need_an_alias():
    assert geography(citation=REQUEST)["status"] == "VERIFIED"


@pytest.mark.parametrize("finals,citation", [
    ({REQUEST: REQUEST}, FINAL),  # A slash difference alone is not evidence.
    ({}, FINAL), (None, FINAL),
    ({REQUEST: FINAL}, FINAL + "?unseen=1"),
    ({REQUEST: FINAL}, "https://other.example/about"),
    ({REQUEST: FINAL}, FINAL + "#unseen"),
    ({REQUEST: FINAL}, "https://user:password@example.com/about"),
])
def test_unobserved_or_unsafe_alias_is_not_loaded(finals, citation):
    finding = geography(finals=finals, citation=citation)
    assert finding["status"] == "UNPROVEN"


def test_ambiguous_final_binding_is_rejected_even_when_one_quote_matches():
    second = "https://example.com/other"
    finding = geography(pages={REQUEST: QUOTE, second: "Other text"},
                        finals={REQUEST: FINAL, second: FINAL})
    assert finding["status"] == "UNPROVEN"
    assert "not present" in finding["reason"]


def test_final_mapping_without_a_retained_requested_page_is_not_evidence():
    assert geography(pages={}, finals={REQUEST: FINAL})["status"] == "UNPROVEN"


def test_observed_final_alias_resolution_does_not_follow_transitive_mappings():
    intermediate = "https://example.com/intermediate"
    finding = geography(finals={REQUEST: intermediate, intermediate: FINAL})
    assert finding["status"] == "UNPROVEN"


def test_exact_requested_page_keeps_its_own_text_and_destination():
    third = "https://unrelated.example/about"
    finding = geography(pages={FINAL: "Wrong quote", third: QUOTE},
                        finals={FINAL: FINAL, third: FINAL})
    assert finding["status"] == "UNPROVEN"
    assert "not present" in finding["reason"]


def test_alias_does_not_borrow_first_party_authority_for_third_party_request():
    third = "https://unrelated.example/about"
    finding = geography(pages={third: QUOTE}, finals={third: FINAL})
    assert finding["status"] == "UNPROVEN"
    assert "independently bound" in finding["reason"]


def test_alias_does_not_erase_unrelated_final_destination():
    third = "https://unrelated.example/about"
    finding = geography(citation=third, finals={REQUEST: third})
    assert finding["status"] == "UNPROVEN"
    assert "independently bound" in finding["reason"]


def test_alias_does_not_supply_missing_quote_or_headquarters_semantics():
    assert geography(quote="Example headquarters are in Canada.")["status"] == "UNPROVEN"
    finding = geography(pages={REQUEST: "Example has a Singapore sales office."},
                        quote="Example has a Singapore sales office.")
    assert finding["status"] == "UNPROVEN"


def test_required_attribute_primary_and_support_use_original_source_bindings():
    second, second_final = "https://example.com/services/", "https://example.com/services"
    quote = "Example provides managed infrastructure services."
    arguments = {"findings": [{"target": "required_attribute", "status": "VERIFIED",
        "evidence_url": FINAL, "evidence_quote": quote, "activity_role": "supplier_operator",
        "supporting_evidence_url_1": second_final, "supporting_evidence_quote_1": quote}]}
    original = deepcopy(arguments)
    finding = inv._validated_findings(
        arguments, targets=("required_attribute",),
        fetched_pages={REQUEST: quote, second: quote},
        fetched_final_urls={REQUEST: FINAL, second: second_final},
        first_party_domains={"example.com"}, identity_names={"example"},
        identity_anchor=ANCHOR,
    )["required_attribute"]
    assert arguments == original
    assert finding["status"] == "VERIFIED"
    assert finding["evidence_url"] == REQUEST
    assert finding["supporting_evidence"] == [{"url": second, "quote": quote}]


def test_standalone_support_repair_uses_same_exact_binding():
    quote = "Example provides managed infrastructure services."
    assert inv._required_attribute_support_rejection(
        FINAL, quote, fetched_pages={REQUEST: quote},
        fetched_final_urls={REQUEST: FINAL}, first_party_domains={"example.com"},
        identity_names={"example"}, identity_anchor=ANCHOR,
    ) is None
