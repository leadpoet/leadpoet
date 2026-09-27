from qualification.scoring import company_evidence_investigator as investigator


def _prompt() -> str:
    return " ".join(investigator._SYSTEM_PROMPT.split())


def test_current_profile_trading_row_is_not_made_historical_by_ipo_date():
    prompt = _prompt()

    assert (
        "a separate current trading-information row that names the exact "
        "company and a literal exchange plus ticker is current "
        "company-attributed exchange/ticker evidence"
    ) in prompt
    assert (
        "Do not treat that trading row as historical merely because the same "
        "profile also lists the company's historical IPO date"
    ) in prompt


def test_archived_ipo_profile_does_not_become_current_listing_proof():
    prompt = _prompt()

    assert "This does not make an archived IPO profile current" in prompt
    assert "historical IPO-completion announcement alone remains insufficient" in prompt


def test_later_adverse_listing_event_still_overrides_current_profile_ticker():
    prompt = _prompt()

    assert (
        "a later completed delisting, take-private, or controlling acquisition "
        "still overrides the ticker row"
    ) in prompt
    assert "server-required current-stage search" in prompt
