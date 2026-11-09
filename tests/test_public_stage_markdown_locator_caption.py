"""Visible submitted captions rank reads; they never establish Public status."""

import json
from pathlib import Path

import pytest

from qualification.scoring import company_evidence_investigator as investigator


FIXTURE = json.loads((Path(__file__).parent / "fixtures" /
    "public_stage_markdown_caption_eos_20261107.json").read_text())


def _select(hint, *, disputes=(), fetched_new=False):
    old_url = FIXTURE["older_sec_url"]
    url = FIXTURE["submitted_stage_url"]
    pages = {old_url: FIXTURE["older_sec_text"]}
    if fetched_new:
        pages[url] = FIXTURE["plain_visible_stage_hint"]
    return investigator._public_stage_submitted_source_to_prefetch(
        submitted_source_urls=[old_url, url, *disputes],
        stage_dispute_urls=disputes,
        submitted_source_hints={url: [hint]},
        first_party_domains={FIXTURE["company_domain"]},
        identity_names=set(FIXTURE["identity_names"]),
        fetched_pages=pages,
    )


@pytest.mark.parametrize("field", ["submitted_stage_hint", "plain_visible_stage_hint"])
def test_captured_issuer_caption_outranks_older_loaded_public_source(field):
    assert _select(FIXTURE[field]) == FIXTURE["submitted_stage_url"]


@pytest.mark.parametrize("hint", [
    "[Other Holdings, Inc. (NASDAQ: OTHR)](https://example.com/results)",
    "[Eos Energy Enterprises, Inc.](https://example.com/NASDAQ:EOSE)",
    "[Results](https://example.com/Eos-Energy-Enterprises-NASDAQ-EOSE)",
    "[Results](https://example.com/?text=Eos%20Energy%20Enterprises%20NASDAQ%3AEOSE)",
    "Eos Energy Enterprises partners with [Other Corp (NASDAQ: OTHR)](https://example.com/)",
    "Eos Energy Enterprises' parent [Other Corp (NASDAQ: OTHR)](https://example.com/)",
    "Eos Energy Enterprises' subsidiary [Other Corp (NASDAQ: OTHR)](https://example.com/)",
    "[Eos Energy Enterprises plans an IPO](https://example.com/NASDAQ:EOSE)",
    "[Eos Energy Enterprises bonds are listed on Nasdaq](https://example.com/)",
    "[Eos Energy Enterprises, Inc. (NASDAQ: EOSE)](javascript:listing)",
    "[Eos Energy Enterprises, Inc. (NASDAQ: EOSE)](https://example.com/unclosed",
])
def test_caption_or_href_does_not_supply_missing_issuer_or_equity_fact(hint):
    assert _select(hint) == ""


def test_existing_take_private_dispute_precedes_linked_caption():
    dispute = "https://example.com/completed-take-private"
    assert _select(FIXTURE["submitted_stage_hint"], disputes=[dispute]) == dispute


def test_already_loaded_exact_source_is_not_fetched_again():
    assert _select(FIXTURE["submitted_stage_hint"], fetched_new=True) == ""


@pytest.mark.parametrize("fetched_text", [
    "Eos Energy Enterprises filed Form 25-NSE; the affected security is not stated.",
    "Eos Energy Enterprises common stock ceased trading on Nasdaq after a completed take-private.",
    "Other Holdings, Inc. (NASDAQ: OTHR) announced results.",
])
def test_selected_caption_cannot_replace_missing_or_conflicting_fetched_proof(fetched_text):
    url = FIXTURE["submitted_stage_url"]
    assert _select(FIXTURE["submitted_stage_hint"]) == url
    finding = investigator._validated_findings(
        {"findings": [{
            "target": "stage", "status": "VERIFIED", "observed_value": "Public",
            "evidence_url": url,
            "evidence_quote": FIXTURE["plain_visible_stage_hint"],
        }]},
        targets=("stage",),
        fetched_pages={url: fetched_text},
        first_party_domains={FIXTURE["company_domain"]},
        identity_names=set(FIXTURE["identity_names"]),
    )["stage"]
    assert finding["status"] == "UNPROVEN"
    assert finding["evidence_url"] == finding["evidence_quote"] == ""


def test_markdown_hint_remains_untrusted_in_exact_quote_validation():
    assert not investigator._quote_occurs(
        FIXTURE["plain_visible_stage_hint"], "Unrelated fetched page text."
    )
