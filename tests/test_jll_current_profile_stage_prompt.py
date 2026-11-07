from qualification.scoring import company_evidence_investigator as investigator


def _prompt() -> str:
    return " ".join(investigator._SYSTEM_PROMPT.split())


def test_current_profile_trading_row_is_not_made_historical_by_ipo_date():
    prompt = _prompt()

    assert (
        "A current first-party issuer or investor-relations profile's separate "
        "current trading row naming that issuer and a literal exchange/ticker "
        "is current proof"
    ) in prompt
    assert "even beside its historical IPO date" in prompt
    assert (
        "issuer means the investigated company for direct proof or the exact "
        "proven parent for listed-parent proof"
    ) in prompt


def test_archived_ipo_profile_does_not_become_current_listing_proof():
    prompt = _prompt()

    assert "An archived IPO profile is not current" in prompt
    assert "A historical IPO-completion announcement alone remains insufficient" in prompt


def test_later_adverse_listing_event_still_overrides_current_profile_ticker():
    prompt = _prompt()

    assert (
        "completed delisting or take-private overrides the row"
    ) in prompt
    assert "server-required current-stage search" in prompt
    assert (
        "For Public, review the issuer's current listing and later completed "
        "take-private and delisting results before retaining an older ticker quote"
    ) in prompt
