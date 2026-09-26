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
