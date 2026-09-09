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
from qualification.scoring.lead_scorer import (
    _verified_homepage_identity_anchor,
    _web_identity_receipt,
)


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


def test_direct_company_verification_keeps_five_second_deadline(monkeypatch):
    import asyncio

    observed_timeouts = []
    response = _Response(
        200,
        b'<title>Example Company</title>'
        b'<a href="https://linkedin.com/company/example-company">LinkedIn</a>',
    )

    def session(**kwargs):
        observed_timeouts.append(kwargs["timeout"])
        return _Session(response)

    monkeypatch.setattr(
        "qualification.scoring.company_verification._registrable_domain",
        lambda _url: "example.com",
    )
    monkeypatch.setattr(
        "qualification.scoring.company_verification.aiohttp.ClientSession",
        session,
    )

    result = asyncio.run(
        verify_company_exists(
            "Example Company",
            "https://example.com/",
            company_linkedin="https://linkedin.com/company/example-company",
        )
    )

    assert result.decision == COMPANY_FIT_MATCH
    assert len(observed_timeouts) == 1
    assert observed_timeouts[0].total == 5
    assert observed_timeouts[0].connect == 3


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


def test_goldman_homepage_legal_name_binds_exact_web_observation(monkeypatch):
    import asyncio

    payload = (
        b'<title>Goldman Sachs</title><script type="application/ld+json">'
        b'{"@type":"Organization","legalName":"The Goldman Sachs Group, Inc.",'
        b'"name":"Goldman Sachs","sameAs":['
        b'"https://www.linkedin.com/company/goldman-sachs",'
        b'"https://twitter.com/goldmansachs"],'
        b'"url":"https://www.goldmansachs.com"}</script>'
    )
    response = _Response(200, payload, "https://www.goldmansachs.com/")
    monkeypatch.setattr(
        "qualification.scoring.company_verification._registrable_domain",
        lambda _url: "goldmansachs.com",
    )
    monkeypatch.setattr(
        "qualification.scoring.company_verification.aiohttp.ClientSession",
        lambda **_kwargs: _Session(response),
    )
    homepage = asyncio.run(verify_company_exists(
        "Goldman Sachs",
        "https://www.goldmansachs.com",
        company_linkedin="https://www.linkedin.com/company/goldman-sachs",
    ))
    anchor = _verified_homepage_identity_anchor(homepage)
    company = CompanyOutput(
        company_name="Goldman Sachs",
        company_website="https://www.goldmansachs.com",
        company_linkedin="https://www.linkedin.com/company/goldman-sachs",
        industry="Financial Services",
        employee_count="10001+",
        company_stage="Public",
        country="United States",
        intent_signals=[{
            "description": "Goldman Sachs published its annual report.",
            "source": "company_website",
            "url": "https://www.goldmansachs.com/investor-relations/",
            "date": "2026-01-01",
            "snippet": "The annual report lists current company information.",
        }],
    )
    receipt = _web_identity_receipt(company, {
        "observed_company_name": "The Goldman Sachs Group, Inc.",
        "observed_company_website": "https://www.goldmansachs.com/about-us",
        "observed_company_linkedin": "https://www.linkedin.com/company/goldman-sachs",
    }, verified_homepage_identity=anchor)

    assert homepage.decision == COMPANY_FIT_MATCH
    assert anchor["verified_legal_name_aliases"] == [
        "The Goldman Sachs Group, Inc."
    ]
    assert receipt["decision"] == COMPANY_FIT_MATCH
    assert receipt["observed_name"] == "thegoldmansachs"

    for wrong_company in (
        company.model_copy(update={"company_name": "Unrelated Bank"}),
        company.model_copy(update={"company_website": "https://wrong.example"}),
        company.model_copy(
            update={
                "company_linkedin": "https://www.linkedin.com/company/wrong"
            }
        ),
    ):
        assert _web_identity_receipt(
            wrong_company,
            {
                "observed_company_name": "The Goldman Sachs Group, Inc.",
                "observed_company_website": "https://www.goldmansachs.com/about-us",
                "observed_company_linkedin": (
                    "https://www.linkedin.com/company/goldman-sachs"
                ),
            },
            verified_homepage_identity=anchor,
        )["decision"] != COMPANY_FIT_MATCH


def test_homepage_identity_can_follow_large_bounded_style_prefix(monkeypatch):
    import asyncio

    late_identity = (
        b"<style>" + (b"x" * 250_000) + b"</style>"
        b"<title>Example Company</title>"
        b'<script type="application/ld+json">'
        b'{"@type":"Organization","name":"Example Company",'
        b'"legalName":"Example Company Holdings, Inc.",'
        b'"url":"https://www.example.co.uk","sameAs":'
        b'"https://www.linkedin.com/company/example-company"}</script>'
    )

    result = asyncio.run(_verify_with_response(monkeypatch, 200, late_identity))

    assert result.decision == COMPANY_FIT_MATCH
    assert result.details["identity"]["verified_legal_name_aliases"] == [
        "Example Company Holdings, Inc."
    ]


def test_homepage_identity_after_body_cap_remains_unavailable(monkeypatch):
    import asyncio
    from qualification.scoring.company_verification import _MAX_BYTES

    over_cap_identity = (
        b"<style>" + (b"x" * (_MAX_BYTES + 1)) + b"</style>"
        b"<title>Example Company</title>"
        b'<a href="https://www.linkedin.com/company/example-company">LinkedIn</a>'
    )

    result = asyncio.run(_verify_with_response(monkeypatch, 200, over_cap_identity))

    assert result.decision == COMPANY_FIT_UNAVAILABLE
    assert "company name metadata not found" in (result.reason or "")


def test_homepage_legal_alias_requires_bound_root_organization(monkeypatch):
    import asyncio

    base = {
        "normalized_name": "goldmansachs",
        "registrable_dns_domain": "goldmansachs.com",
        "linkedin_company_slug": "goldman-sachs",
    }
    company = CompanyOutput(
        company_name="Goldman Sachs",
        company_website="https://www.goldmansachs.com",
        company_linkedin="https://www.linkedin.com/company/goldman-sachs",
        industry="Financial Services",
        employee_count="10001+",
        company_stage="Public",
        country="United States",
        intent_signals=[{
            "description": "Goldman Sachs published its annual report.",
            "source": "company_website",
            "url": "https://www.goldmansachs.com/investor-relations/",
            "date": "2026-01-01",
            "snippet": "The annual report lists current company information.",
        }],
    )
    verdict = {
        "observed_company_name": "The Goldman Sachs Group, Inc.",
        "observed_company_website": "https://www.goldmansachs.com/about-us",
        "observed_company_linkedin": "https://www.linkedin.com/company/goldman-sachs",
    }
    for organization in (
        # Wrong first-party URL, LinkedIn slug, and brand each fail closed.
        '{"@type":"Organization","name":"Goldman Sachs","legalName":"The Goldman Sachs Group, Inc.","url":"https://wrong.example","sameAs":"https://www.linkedin.com/company/goldman-sachs"}',
        '{"@type":"Organization","name":"Goldman Sachs","legalName":"The Goldman Sachs Group, Inc.","url":"https://www.goldmansachs.com","sameAs":"https://www.linkedin.com/company/wrong"}',
        '{"@type":"Organization","name":"Unrelated Bank","legalName":"The Goldman Sachs Group, Inc.","url":"https://www.goldmansachs.com","sameAs":"https://www.linkedin.com/company/goldman-sachs"}',
        # Nested partner/member Organizations are not homepage identity roots.
        '{"@type":"WebSite","member":{"@type":"Organization","name":"Goldman Sachs","legalName":"The Goldman Sachs Group, Inc.","url":"https://www.goldmansachs.com","sameAs":"https://www.linkedin.com/company/goldman-sachs"}}',
    ):
        response = _Response(
            200,
            (f'<title>Goldman Sachs</title><a href="https://www.linkedin.com/company/goldman-sachs">LinkedIn</a><script type="application/ld+json">{organization}</script>').encode(),
            "https://www.goldmansachs.com/",
        )
        monkeypatch.setattr(
            "qualification.scoring.company_verification._registrable_domain",
            lambda url: "wrong.example" if "wrong.example" in str(url) else "goldmansachs.com",
        )
        monkeypatch.setattr(
            "qualification.scoring.company_verification.aiohttp.ClientSession",
            lambda **_kwargs: _Session(response),
        )
        homepage = asyncio.run(verify_company_exists(
            "Goldman Sachs",
            "https://www.goldmansachs.com",
            company_linkedin="https://www.linkedin.com/company/goldman-sachs",
        ))
        anchor = _verified_homepage_identity_anchor(homepage)
        assert "verified_legal_name_aliases" not in anchor
        assert _web_identity_receipt(
            company, verdict, verified_homepage_identity=anchor
        )["decision"] == COMPANY_FIT_MISMATCH

    # An alias supplied directly by the model or caller is not trusted.
    assert _web_identity_receipt(
        company, verdict, verified_homepage_identity=base
    )["decision"] == COMPANY_FIT_MISMATCH


def test_nested_publisher_linkedin_is_retained_but_legal_alias_is_not(monkeypatch):
    import asyncio

    nested_publisher = (
        b"<title>Example Company</title>"
        b'<script type="application/ld+json">'
        b'{"@type":"WebSite","publisher":{"@type":"Organization",'
        b'"name":"Example Company","legalName":"Unbound Legal Alias, Inc.",'
        b'"url":"https://www.example.co.uk","sameAs":'
        b'"https://www.linkedin.com/company/example-company"}}</script>'
    )

    result = asyncio.run(
        _verify_with_response(monkeypatch, 200, nested_publisher)
    )
    anchor = _verified_homepage_identity_anchor(result)

    assert result.decision == COMPANY_FIT_MATCH
    assert result.details["identity"]["observed_linkedin_slug"] == (
        "example-company"
    )
    assert "verified_legal_name_aliases" not in anchor


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


def test_conflicting_homepage_title_needs_independent_identity_check(monkeypatch):
    import asyncio

    result = asyncio.run(
        _verify_with_response(
            monkeypatch,
            200,
            b'<title>Different Business</title>'
            b'<a href="https://linkedin.com/company/example-company">LinkedIn</a>',
        )
    )
    assert result.decision == COMPANY_FIT_UNAVAILABLE
    assert result.passed is False
    assert result.details["identity"]["observed_name"] == "differentbusiness"


def test_marketing_title_and_old_linkedin_link_do_not_prove_a_conflict(monkeypatch):
    import asyncio

    result = asyncio.run(
        _verify_with_response(
            monkeypatch,
            200,
            b'<title>Save money. Stay powered. Example Company Map pin</title>'
            b'<a href="https://linkedin.com/company/old-example-name">LinkedIn</a>',
        )
    )
    assert result.decision == COMPANY_FIT_UNAVAILABLE
    assert result.passed is False
    assert result.details["identity"]["observed_linkedin_slug"] == "old-example-name"


def test_independent_web_identity_conflict_is_still_a_mismatch():
    receipt = evaluate_company_identity(
        submitted_name="Example Company",
        submitted_website="https://example.co.uk",
        submitted_linkedin="https://linkedin.com/company/example-company",
        observed_name="Different Business",
        observed_website="https://example.co.uk",
        observed_linkedin="https://linkedin.com/company/different-business",
        evidence_source="company_web_reverification",
    )
    assert receipt["decision"] == COMPANY_FIT_MISMATCH


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
