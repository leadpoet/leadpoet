from qualification.scoring.company_verification import (
    _upgrade_plain_http_company_url,
)


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
