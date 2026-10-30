"""Focused contracts for company-bound completed-acquisition evidence."""

import pytest

from qualification.scoring.lead_scorer import (
    _acquired_stage_quote_supports_names,
)


SOLARWINDS_COMPLETED_ACQUISITION = (
    "SolarWinds Corporation (“SolarWinds” or the “Company”), a leading "
    "provider of simple, powerful, secure observability and IT management "
    "software, today announced the closing of its acquisition by Turn/River "
    "Capital. The transaction is valued at approximately $4.4 billion, with "
    "SolarWinds stockholders receiving $18.50 per share in cash. With the "
    "closing of the transaction, SolarWinds common stock has ceased trading, "
    "and the Company is no longer listed on the New York Stock Exchange."
)


def test_completed_acquisition_target_accepts_exact_solarwinds_wording():
    assert _acquired_stage_quote_supports_names(
        ("solarwinds",), SOLARWINDS_COMPLETED_ACQUISITION
    )


def test_completed_acquisition_target_accepts_descriptive_clause():
    quote = (
        "Acme Analytics, a provider of workflow software, today announced the "
        "completion of its acquisition by Parent Holdings."
    )

    assert _acquired_stage_quote_supports_names(("Acme Analytics",), quote)


@pytest.mark.parametrize("verb", ["completed", "closed"])
@pytest.mark.parametrize("alias", ["", ' (“Acme”)', ' ("Acme")'])
def test_closed_acquisition_with_target_alias_is_bound(verb, alias):
    from qualification.scoring.lead_scorer import _stage_quote_supports_observation

    quote = (
        f"Parent Holdings today announced that it has {verb} its acquisition "
        f"of Acme Analytics{alias} from the former owner following receipt "
        "of all required regulatory approvals."
    )
    assert _stage_quote_supports_observation("acquired", quote)
    assert _acquired_stage_quote_supports_names(("Acme Analytics",), quote)
    assert not _acquired_stage_quote_supports_names(("Parent Holdings",), quote)


@pytest.mark.parametrize("quote", [
    'Parent Holdings has not closed its acquisition of Acme Analytics (“Acme”).',
    'Parent Holdings expects to have closed its acquisition of Acme Analytics (“Acme”) in June.',
    'If Parent Holdings closed its acquisition of Acme Analytics (“Acme”), ownership would change.',
    'Parent Holdings closed its acquisition of Acme Analytics (“Acme”), but the deal was cancelled.',
    'Parent Holdings closed its acquisition of Acme Analytics minority shares (“Acme”).',
    'Parent Holdings closed its acquisition of Acme Analytics assets (“Acme”).',
    'Parent Holdings closed its acquisition of Other Company (a partner of Acme Analytics).',
    'Parent Holdings closed its acquisition of Acme Analytics Systems (“Acme”).',
    'Parent Holdings closed its acquisition of Acme Analytics (“Acme”), which later became independent.',
])
def test_closed_acquisition_alias_does_not_hide_wrong_subject_or_uncertainty(quote):
    assert not _acquired_stage_quote_supports_names(("Acme Analytics",), quote)


def test_real_closed_acquisition_survives_investigator_and_saved_stage_decision():
    import json
    from qualification.scoring import company_evidence_investigator as investigator
    from qualification.scoring import lead_scorer
    from tests.test_company_evidence_investigator import _company, _finding

    url = "https://omnimax.com/gibraltar-completes-acquisition-of-omnimax-international-for-1-35-billion/"
    quote = (
        "Gibraltar Industries, Inc. (Nasdaq: ROCK), a leading manufacturer and "
        "provider of products and services for the residential, agtech, and "
        "infrastructure markets, today announced that it has closed its "
        "acquisition of OmniMax International (“OmniMax”) from funds managed "
        "by Strategic Value Partners, LLC and its affiliates following receipt "
        "of all required regulatory approvals and satisfaction of customary "
        "closing conditions."
    )
    finding = investigator._validated_findings(
        {"findings": [_finding("stage", status="CONTRADICTED",
            observed_value="Acquired", evidence_url=url, evidence_quote=quote)]},
        targets=("stage",), fetched_pages={url: quote},
        fetched_final_urls={url: url}, first_party_domains={"omnimax.com"},
        identity_names={"omnimax"},
        requested_stage="private equity",
    )["stage"]
    assert finding["status"] == "CONTRADICTED"
    assert finding["evidence_quote"] == quote
    company = _company(name="OmniMax", website="https://omnimax.com")
    projected = lead_scorer._project_investigator_stage(
        {"observed_company_name": company.company_name}, finding,
        icp_stage="private equity")
    retained = json.loads(json.dumps(projected))
    assert retained["stage_evidence_quote"] == quote
    assert lead_scorer._decision_from_observed_stage(
        retained, "private equity", company=company, evidence_attributed=True,
    ) == "mismatch"
    # An acquisition does not transfer the buyer's public listing to the target.
    assert retained["observed_company_stage"] == "acquired"
    # The same bound proof reopens a stale venture-stage result for research.
    assert lead_scorer._first_party_acquisition_conflicts_with_venture_stage(
        company, {"observed_company_name": company.company_name,
                  "dimension_evidence": {"required_attribute": {"url": url, "quote": quote}}},
        "series b")


@pytest.mark.parametrize("subject", ["assets of Acme Analytics", "Acme Analytics assets",
    "a division of Acme Analytics", "a minority stake in Acme Analytics"])
def test_alias_cannot_turn_partial_assets_into_a_company_acquisition(subject):
    quote = f'Parent closed its acquisition of {subject} (“Acme Analytics”).'
    assert not _acquired_stage_quote_supports_names(("Acme Analytics",), quote)


def test_explicit_target_alias_binds_the_same_target_not_a_mentioned_parent():
    quote = 'Parent closed its acquisition of Example Legal Name (“Acme Analytics”).'
    assert _acquired_stage_quote_supports_names(("Acme Analytics",), quote)
    assert not _acquired_stage_quote_supports_names(("Parent",), quote)
    assert not _acquired_stage_quote_supports_names(("Acme",), quote)


def test_nested_reported_acquisition_does_not_bind_outer_company():
    quote = (
        "SolarWinds, in an article reporting that OtherCo today announced the "
        "closing of its acquisition by Parent Holdings."
    )

    assert not _acquired_stage_quote_supports_names(("SolarWinds",), quote)


def test_compact_multiword_identity_requires_exact_normalized_name():
    quote = "Parent Holdings completed the acquisition of CISO Global."

    assert _acquired_stage_quote_supports_names(("cisoglobal",), quote)
    assert _acquired_stage_quote_supports_names(("CISO Global",), quote)
    assert not _acquired_stage_quote_supports_names(("cisoglobalsystems",), quote)
    assert not _acquired_stage_quote_supports_names(("global",), quote)


@pytest.mark.parametrize(
    "quote",
    [
        (
            "SolarWinds Corporation, a software provider, today announced the "
            "planned closing of its acquisition by Turn/River Capital."
        ),
        (
            "SolarWinds Corporation, a software provider, today announced the "
            "expected closing of its acquisition by Turn/River Capital."
        ),
        (
            "SolarWinds Corporation, a software provider, today announced the "
            "closing of its acquisition by Turn/River Capital, but the "
            "transaction was cancelled."
        ),
    ],
)
def test_incomplete_acquisition_wording_is_rejected(quote):
    assert not _acquired_stage_quote_supports_names(("SolarWinds",), quote)


@pytest.mark.parametrize(
    "quote",
    [
        (
            "SolarWinds Corporation, a software provider, today announced the "
            "closing of its acquisition of Pingdom."
        ),
        (
            "Turn/River Capital, a software investor, today announced the "
            "closing of its acquisition of Pingdom."
        ),
        (
            "SolarWinds Corporation, a software provider, today announced the "
            "closing of its minority investment by Turn/River Capital."
        ),
    ],
)
def test_buyer_wrong_target_and_minority_wording_are_rejected(quote):
    assert not _acquired_stage_quote_supports_names(("SolarWinds",), quote)


@pytest.mark.parametrize(
    "supersession",
    [
        "SolarWinds later relisted on Nasdaq.",
        "SolarWinds was subsequently spun out and became independent.",
    ],
)
def test_later_public_or_independent_supersession_is_rejected(supersession):
    quote = SOLARWINDS_COMPLETED_ACQUISITION + " " + supersession

    assert not _acquired_stage_quote_supports_names(("SolarWinds",), quote)
