from qualification.scoring.company_verification import (
    _upgrade_plain_http_company_url,
    verify_company_exists,
)
from qualification.scoring.company_fit_decision import (
    COMPANY_FIT_MATCH,
    COMPANY_FIT_MISMATCH,
    COMPANY_FIT_UNAVAILABLE,
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


def test_homepage_conflicting_linkedin_binding_is_mismatch(monkeypatch):
    import asyncio

    result = asyncio.run(
        _verify_with_response(
            monkeypatch,
            200,
            b'<title>Example Company</title><a href="https://www.linkedin.com/company/different-company">LinkedIn</a>',
        )
    )
    assert result.decision == COMPANY_FIT_MISMATCH
    assert "identity conflict" in (result.reason or "")


def test_missing_submitted_linkedin_identity_is_mismatch(monkeypatch):
    import asyncio

    result = asyncio.run(
        _verify_with_response(
            monkeypatch,
            200,
            b'<title>Example Company</title><a href="https://www.linkedin.com/company/example-company">LinkedIn</a>',
            company_linkedin="",
        )
    )
    assert result.decision == COMPANY_FIT_MISMATCH
    assert "submitted company identity" in (result.reason or "")


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
