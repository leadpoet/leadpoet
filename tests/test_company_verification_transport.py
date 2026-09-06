from gateway.qualification.models import CompanyOutput
from qualification.scoring.company_fit_decision import (
    COMPANY_FIT_MATCH,
    COMPANY_FIT_MISMATCH,
    COMPANY_FIT_UNAVAILABLE,
    evaluate_company_identity,
)
from qualification.scoring.company_verification import (
    _upgrade_plain_http_company_url,
    verify_company_exists,
)
from qualification.scoring.lead_scorer import _web_identity_receipt


class _Content:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    async def read(self, _limit: int) -> bytes:
        return self._payload


class _Response:
    def __init__(self, status: int, payload: bytes, url: str = "") -> None:
        self.status = status
        self.content = _Content(payload)
        if url:
            self.url = url

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False


class _Session:
    def __init__(self, response: _Response) -> None:
        self._response = response

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False

    def get(self, *_args, **_kwargs):
        return self._response


def test_default_http_company_url_upgrades_to_https():
    assert _upgrade_plain_http_company_url("http://Example.com/about?q=1") == (
        "https://example.com/about?q=1"
    )
    assert _upgrade_plain_http_company_url("http://example.com:80/") == (
        "https://example.com/"
    )


def test_unsafe_or_nonstandard_company_urls_are_not_rewritten():
    for value in (
        "http://example.com:8080/",
        "http://user:pass@example.com/",
        "ftp://example.com/",
        "not-a-url",
    ):
        assert _upgrade_plain_http_company_url(value) == value
    assert _upgrade_plain_http_company_url("https://example.com/") == (
        "https://example.com/"
    )


async def _verify_with_response(
    monkeypatch,
    status: int,
    payload: bytes,
    *,
    company_linkedin: str = "https://www.linkedin.com/company/example-company",
    final_url: str = "",
    require_https_transport: bool = False,
):
    response = _Response(status, payload, final_url)
    monkeypatch.setattr(
        "qualification.scoring.company_verification._registrable_domain",
        lambda _url: "example.co.uk",
    )
    monkeypatch.setattr(
        "qualification.scoring.company_verification.aiohttp.ClientSession",
        lambda **_kwargs: _Session(response),
    )
    return await verify_company_exists(
        "Example Company",
        "https://www.example.co.uk",
        company_linkedin=company_linkedin,
        require_https_transport=require_https_transport,
    )


def test_homepage_name_is_a_match(monkeypatch):
    import asyncio

    result = asyncio.run(
        _verify_with_response(
            monkeypatch,
            200,
            b'<title>Example Company</title><a href="https://www.linkedin.com/company/example-company">LinkedIn</a>',
        )
    )
    assert result.decision == COMPANY_FIT_MATCH


def test_homepage_colon_title_is_a_match(monkeypatch):
    import asyncio

    result = asyncio.run(
        _verify_with_response(
            monkeypatch,
            200,
            b'<title>Example Company: product tagline</title>'
            b'<a href="https://www.linkedin.com/company/example-company">LinkedIn</a>',
        )
    )
    assert result.decision == COMPANY_FIT_MATCH


def test_homepage_name_without_linkedin_binding_is_unavailable(monkeypatch):
    import asyncio

    result = asyncio.run(
        _verify_with_response(monkeypatch, 200, b"<title>Example Company</title>")
    )
    assert result.decision == COMPANY_FIT_UNAVAILABLE
    assert "LinkedIn company binding not found" in (result.reason or "")


def test_linkedin_text_inside_html_comment_is_not_identity_proof(monkeypatch):
    import asyncio

    result = asyncio.run(
        _verify_with_response(
            monkeypatch,
            200,
            b"<title>Example Company</title>"
            b"<!-- https://www.linkedin.com/company/example-company -->",
        )
    )
    assert result.decision == COMPANY_FIT_UNAVAILABLE
    assert "LinkedIn company binding not found" in (result.reason or "")


def test_organization_jsonld_same_as_is_identity_proof(monkeypatch):
    import asyncio

    result = asyncio.run(
        _verify_with_response(
            monkeypatch,
            200,
            b'<title>Example Company</title><script type="application/ld+json">'
            b'{"@type":"Organization","sameAs":'
            b'["https://www.linkedin.com/company/example-company"]}'
            b"</script>",
        )
    )
    assert result.decision == COMPANY_FIT_MATCH


def test_https_mode_rejects_final_http_redirect(monkeypatch):
    import asyncio

    result = asyncio.run(
        _verify_with_response(
            monkeypatch,
            200,
            b'<title>Example Company</title>'
            b'<a href="https://www.linkedin.com/company/example-company">LinkedIn</a>',
            final_url="http://www.example.co.uk/final",
            require_https_transport=True,
        )
    )
    assert result.decision == COMPANY_FIT_UNAVAILABLE
    assert "final URL is not HTTPS" in (result.reason or "")


def test_homepage_named_linkedin_alias_with_exact_name_and_domain_is_unavailable(
    monkeypatch,
):
    import asyncio

    result = asyncio.run(
        _verify_with_response(
            monkeypatch,
            200,
            b'<title>Example Company</title><a href="https://www.linkedin.com/company/different-company">LinkedIn</a>',
        )
    )
    assert result.decision == COMPANY_FIT_UNAVAILABLE
    assert result.details["identity"]["reason_code"] == (
        "identity_linkedin_alias_unresolved"
    )


def test_web_reverification_named_linkedin_mismatch_remains_mismatch():
    receipt = evaluate_company_identity(
        submitted_name="Base Power",
        submitted_website="https://basepowercompany.com",
        submitted_linkedin="https://linkedin.com/company/basepowercompany",
        observed_name="Base Power",
        observed_website="https://basepowercompany.com/about",
        observed_linkedin="https://linkedin.com/company/base-power-company",
        evidence_source="company_web_reverification",
    )

    assert receipt["decision"] == COMPANY_FIT_MISMATCH
    assert receipt["reason_code"] == "identity_mismatch"


def test_homepage_numeric_linkedin_id_vs_vanity_slug_is_unavailable(monkeypatch):
    import asyncio

    result = asyncio.run(
        _verify_with_response(
            monkeypatch,
            200,
            b'<title>Example Company</title>'
            b'<a href="https://www.linkedin.com/company/123456">LinkedIn</a>',
        )
    )
    assert result.decision == COMPANY_FIT_UNAVAILABLE
    assert result.details["identity"]["reason_code"] == (
        "identity_linkedin_alias_unresolved"
    )


def test_missing_submitted_linkedin_uses_exact_name_and_domain(monkeypatch):
    import asyncio

    result = asyncio.run(
        _verify_with_response(
            monkeypatch,
            200,
            b'<title>Example Company</title><a href="https://www.linkedin.com/company/example-company">LinkedIn</a>',
            company_linkedin="",
        )
    )
    assert result.decision == COMPANY_FIT_MATCH


def test_iag_parenthetical_alias_uses_homepage_linkedin_binding(monkeypatch):
    import asyncio

    response = _Response(
        200,
        b'<title>IAG Limited</title>'
        b'<a href="https://www.linkedin.com/company/iag/">LinkedIn</a>'
        b'<div><p>&copy; 2026 INSURANCE AUSTRALIA GROUP LIMITED '
        b'ABN 60 090 739 923</p></div>',
        "https://www.iag.com.au/",
    )
    monkeypatch.setattr(
        "qualification.scoring.company_verification._registrable_domain",
        lambda _url: "iag.com.au",
    )
    monkeypatch.setattr(
        "qualification.scoring.company_verification.aiohttp.ClientSession",
        lambda **_kwargs: _Session(response),
    )

    result = asyncio.run(
        verify_company_exists(
            "Insurance Australia Group Limited (IAG)",
            "https://www.iag.com.au/",
            company_linkedin="",
        )
    )

    assert result.decision == COMPANY_FIT_MATCH
    assert result.details["identity"]["observed_linkedin_slug"] == "iag"


def test_iag_web_reverification_without_homepage_anchor_stays_unavailable():
    receipt = evaluate_company_identity(
        submitted_name="Insurance Australia Group Limited (IAG)",
        submitted_website="https://www.iag.com.au/",
        submitted_linkedin="",
        observed_name="Insurance Australia Group Limited",
        observed_website="https://www.iag.com.au/about-us",
        observed_linkedin="https://www.linkedin.com/company/iag/",
        evidence_source="company_web_reverification",
    )

    assert receipt["decision"] == COMPANY_FIT_UNAVAILABLE


def test_iag_web_reverification_uses_verified_homepage_anchor():
    company = CompanyOutput(
        company_name="Insurance Australia Group Limited (IAG)",
        company_website="https://www.iag.com.au/",
        company_linkedin="",
        industry="Financial Services",
        employee_count="10001+",
        company_stage="Public",
        country="Australia",
        intent_signals=[
            {
                "description": "IAG announced a leadership transition.",
                "source": "news",
                "url": "https://www.iag.com.au/newsroom",
                "date": "2026-02-23",
                "snippet": "IAG announced a leadership team update.",
            }
        ],
    )
    verdict = {
        "observed_company_name": "Insurance Australia Group Limited",
        "observed_company_website": "https://www.iag.com.au/about-us",
        "observed_company_linkedin": "https://www.linkedin.com/company/iag/",
    }
    anchor = {
        "normalized_name": "iag",
        "registrable_dns_domain": "iag.com.au",
        "linkedin_company_slug": "iag",
    }

    receipt = _web_identity_receipt(
        company,
        verdict,
        verified_homepage_identity=anchor,
    )

    assert receipt["decision"] == COMPANY_FIT_MATCH
    assert receipt["submitted_linkedin_slug"] == "iag"

    wrong_entity = _web_identity_receipt(
        company,
        {**verdict, "observed_company_name": "Unrelated Insurance Limited"},
        verified_homepage_identity=anchor,
    )
    assert wrong_entity["decision"] == COMPANY_FIT_MISMATCH

    incomplete_anchor = _web_identity_receipt(
        company,
        verdict,
        verified_homepage_identity={
            "normalized_name": "",
            "registrable_dns_domain": "iag.com.au",
            "linkedin_company_slug": "iag",
        },
    )
    assert incomplete_anchor["decision"] == COMPANY_FIT_UNAVAILABLE


def test_iag_parenthetical_alias_rejects_wrong_submitted_linkedin():
    receipt = evaluate_company_identity(
        submitted_name="Insurance Australia Group Limited (IAG)",
        submitted_website="https://www.iag.com.au/",
        submitted_linkedin="https://www.linkedin.com/company/not-iag/",
        observed_name="IAG Limited",
        observed_website="https://www.iag.com.au/",
        observed_linkedin="https://www.linkedin.com/company/iag/",
        evidence_source="company_homepage",
    )

    assert receipt["decision"] == COMPANY_FIT_MISMATCH


def test_parenthetical_initialism_collision_does_not_bind_wrong_company():
    receipt = evaluate_company_identity(
        submitted_name="Imaginary Assets Group Limited (IAG)",
        submitted_website="https://www.iag.com.au/",
        submitted_linkedin="",
        observed_name="Insurance Australia Group Limited",
        observed_website="https://www.iag.com.au/",
        observed_linkedin="https://www.linkedin.com/company/iag/",
        evidence_source="company_homepage",
    )

    assert receipt["decision"] == COMPANY_FIT_UNAVAILABLE


def test_parenthetical_alias_with_only_acronym_title_stays_unavailable(monkeypatch):
    import asyncio

    response = _Response(
        200,
        b'<title>IAG Limited</title>'
        b'<a href="https://www.linkedin.com/company/iag/">LinkedIn</a>',
        "https://www.iag.com.au/",
    )
    monkeypatch.setattr(
        "qualification.scoring.company_verification._registrable_domain",
        lambda _url: "iag.com.au",
    )
    monkeypatch.setattr(
        "qualification.scoring.company_verification.aiohttp.ClientSession",
        lambda **_kwargs: _Session(response),
    )

    result = asyncio.run(
        verify_company_exists(
            "Insurance Australia Group Limited (IAG)",
            "https://www.iag.com.au/",
            company_linkedin="",
        )
    )

    assert result.decision == COMPANY_FIT_UNAVAILABLE


def test_script_copyright_text_does_not_prove_parenthetical_alias(monkeypatch):
    import asyncio

    response = _Response(
        200,
        b'<title>IAG Limited</title>'
        b'<script>const footer = "&copy; 2026 INSURANCE AUSTRALIA GROUP '
        b'LIMITED ABN 60 090 739 923";</script>'
        b'<a href="https://www.linkedin.com/company/iag/">LinkedIn</a>',
        "https://www.iag.com.au/",
    )
    monkeypatch.setattr(
        "qualification.scoring.company_verification._registrable_domain",
        lambda _url: "iag.com.au",
    )
    monkeypatch.setattr(
        "qualification.scoring.company_verification.aiohttp.ClientSession",
        lambda **_kwargs: _Session(response),
    )

    result = asyncio.run(
        verify_company_exists(
            "Insurance Australia Group Limited (IAG)",
            "https://www.iag.com.au/",
            company_linkedin="",
        )
    )

    assert result.decision == COMPANY_FIT_UNAVAILABLE


def test_parenthetical_alias_rejects_wrong_domain():
    receipt = evaluate_company_identity(
        submitted_name="Insurance Australia Group Limited (IAG)",
        submitted_website="https://www.iag.com.au/",
        submitted_linkedin="",
        observed_name="IAG Limited",
        observed_website="https://different.example/",
        observed_linkedin="https://www.linkedin.com/company/iag/",
        evidence_source="company_homepage",
    )

    assert receipt["decision"] == COMPANY_FIT_MISMATCH


def test_parenthetical_alias_without_independent_linkedin_stays_unavailable():
    receipt = evaluate_company_identity(
        submitted_name="Insurance Australia Group Limited (IAG)",
        submitted_website="https://www.iag.com.au/",
        submitted_linkedin="",
        observed_name="IAG Limited",
        observed_website="https://www.iag.com.au/",
        observed_linkedin="",
        evidence_source="company_homepage",
    )

    assert receipt["decision"] == COMPANY_FIT_UNAVAILABLE


def test_parenthetical_alias_without_independent_source_stays_unavailable():
    receipt = evaluate_company_identity(
        submitted_name="Insurance Australia Group Limited (IAG)",
        submitted_website="https://www.iag.com.au/",
        submitted_linkedin="",
        observed_name="IAG Limited",
        observed_website="https://www.iag.com.au/",
        observed_linkedin="https://www.linkedin.com/company/iag/",
        evidence_source="",
    )

    assert receipt["decision"] == COMPANY_FIT_UNAVAILABLE


def test_parenthetical_alias_with_different_homepage_linkedin_stays_unavailable():
    receipt = evaluate_company_identity(
        submitted_name="Insurance Australia Group Limited (IAG)",
        submitted_website="https://www.iag.com.au/",
        submitted_linkedin="",
        observed_name="IAG Limited",
        observed_website="https://www.iag.com.au/",
        observed_linkedin="https://www.linkedin.com/company/unrelated/",
        evidence_source="company_homepage",
    )

    assert receipt["decision"] == COMPANY_FIT_UNAVAILABLE


def test_matching_domain_without_linkedin_does_not_bind_a_name_alias():
    receipt = evaluate_company_identity(
        submitted_name="Modulr",
        submitted_website="https://modulrfinance.com",
        submitted_linkedin="",
        observed_name="Modulr Finance",
        observed_website="https://www.modulrfinance.com/about",
        observed_linkedin="https://www.linkedin.com/company/modulr-finance/",
        evidence_source="company_web_reverification",
    )

    assert receipt["decision"] == COMPANY_FIT_UNAVAILABLE


def test_matching_domain_and_linkedin_bind_common_name_alias():
    receipt = evaluate_company_identity(
        submitted_name="Modulr",
        submitted_website="https://modulrfinance.com",
        submitted_linkedin="https://www.linkedin.com/company/modulr-finance/",
        observed_name="Modulr Finance",
        observed_website="https://www.modulrfinance.com/about",
        observed_linkedin="https://www.linkedin.com/company/modulr-finance/",
        evidence_source="company_web_reverification",
    )

    assert receipt["decision"] == COMPANY_FIT_MATCH


def test_domain_name_alone_is_not_a_match(monkeypatch):
    import asyncio

    result = asyncio.run(
        _verify_with_response(monkeypatch, 403, b"Access denied")
    )
    assert result.decision == COMPANY_FIT_UNAVAILABLE
    assert result.passed is False


def test_missing_homepage_identity_is_unavailable(monkeypatch):
    import asyncio

    result = asyncio.run(
        _verify_with_response(monkeypatch, 200, b"<title>Welcome</title>")
    )
    assert result.decision == COMPANY_FIT_UNAVAILABLE


def test_parked_homepage_is_a_mismatch(monkeypatch):
    import asyncio

    result = asyncio.run(
        _verify_with_response(monkeypatch, 200, b"This domain is for sale")
    )
    assert result.decision == COMPANY_FIT_MISMATCH


def test_identity_normalizer_error_is_unavailable(monkeypatch):
    import asyncio

    def unavailable(_url):
        raise RuntimeError("pinned PSL unavailable")

    monkeypatch.setattr(
        "qualification.scoring.company_verification._registrable_domain",
        unavailable,
    )
    result = asyncio.run(
        verify_company_exists(
            "Example Company",
            "https://example.com",
            company_linkedin="https://linkedin.com/company/example-company",
        )
    )
    assert result.decision == COMPANY_FIT_UNAVAILABLE
    assert "normalization unavailable" in (result.reason or "")


def test_invalid_company_domain_is_a_mismatch():
    import asyncio

    result = asyncio.run(
        verify_company_exists("Example Company", "https://localhost")
    )
    assert result.decision == COMPANY_FIT_MISMATCH


def test_homepage_name_is_observed_not_echoed_from_submission(monkeypatch):
    import asyncio

    result = asyncio.run(
        _verify_with_response(
            monkeypatch,
            200,
            b'<meta property="og:site_name" content="Example Company">'
            b'<a href="https://linkedin.com/company/example-company">LinkedIn</a>',
        )
    )
    assert result.decision == COMPANY_FIT_MATCH
    assert result.details["identity"]["observed_name"] == "example"
    assert result.details["identity"]["evidence_source"] == "company_homepage"


def test_conflicting_observed_homepage_name_is_mismatch(monkeypatch):
    import asyncio

    result = asyncio.run(
        _verify_with_response(
            monkeypatch,
            200,
            b'<title>Different Business</title>'
            b'<a href="https://linkedin.com/company/example-company">LinkedIn</a>',
        )
    )
    assert result.decision == COMPANY_FIT_MISMATCH
    assert result.details["identity"]["observed_name"] == "differentbusiness"


def test_cross_registrable_domain_redirect_is_identity_conflict(monkeypatch):
    import asyncio
    from urllib.parse import urlsplit

    monkeypatch.setattr(
        "qualification.scoring.company_verification._registrable_domain",
        lambda url: str(urlsplit(url).hostname or "").removeprefix("www."),
    )
    monkeypatch.setattr(
        "qualification.scoring.company_verification.aiohttp.ClientSession",
        lambda **_kwargs: _Session(
            _Response(
                200,
                b'<title>Example Company</title>'
                b'<a href="https://linkedin.com/company/example-company">LinkedIn</a>',
                "https://attacker.example/final",
            )
        ),
    )
    result = asyncio.run(
        verify_company_exists(
            "Example Company",
            "https://example.co.uk",
            company_linkedin="https://linkedin.com/company/example-company",
        )
    )
    assert result.decision == COMPANY_FIT_MISMATCH
    assert result.details["actual_final_url"] == "https://attacker.example/final"
    assert "redirect changed registrable domain" in (result.reason or "")


def test_invalid_submitted_linkedin_suffix_spoof_is_mismatch():
    import asyncio

    result = asyncio.run(
        verify_company_exists(
            "Example Company",
            "https://example.com",
            company_linkedin=(
                "https://linkedin.com.evil.example/company/example-company"
            ),
        )
    )
    assert result.decision == COMPANY_FIT_MISMATCH
