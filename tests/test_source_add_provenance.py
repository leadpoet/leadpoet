import pytest

from gateway.research_lab import source_add_provenance as provenance


def _metadata(api_url="https://openrouter.ai/api/v1", docs_url="https://openrouter.ai/docs/quickstart"):
    return {
        "api_base_url": api_url,
        "documentation_url": docs_url,
        "auth_type": "bearer",
        "endpoint_examples": [
            {
                "method": "POST",
                "path": "/chat/completions",
                "purpose": "Create a chat completion",
                "example_query": '{"model":"openai/gpt-4o-mini","messages":[]}',
            }
        ],
        "rate_limit_notes": "Model and account specific rate limits are documented by the provider.",
        "data_provenance_notes": "Routes requests to model providers and returns usage metadata.",
        "third_party_refs": ["https://github.com/OpenRouterTeam"],
    }


def test_openrouter_like_docs_pass_with_references(monkeypatch):
    def fake_scrape(*_args, **_kwargs):
        return {
            "provider_status": "ok",
            "status": 200,
            "content_type": "text/html",
            "body_text": (
                "<title>OpenRouter Quickstart</title> API reference quickstart endpoint "
                "authentication authorization rate limit curl http status code"
            ),
        }

    def fake_ai(*_args, **_kwargs):
        return {
            "provider_status": "ok",
            "status": 200,
            "json": {
                "markdown": "OpenRouter is a legitimate API gateway with official documentation.",
                "references": [
                    {"title": "OpenRouter Docs", "link": "https://openrouter.ai/docs/quickstart"},
                    {"title": "OpenRouter GitHub", "link": "https://github.com/OpenRouterTeam"},
                ],
            },
        }

    monkeypatch.setattr(provenance, "_scrapingdog_scrape", fake_scrape)
    monkeypatch.setattr(provenance, "_scrapingdog_ai_mode", fake_ai)

    result = provenance.evaluate_source_add_provenance(
        source_name="OpenRouter",
        source_kind="web",
        declared_base_domains=("openrouter.ai",),
        source_metadata=_metadata(),
        scrapingdog_api_key="test-key",
    )

    assert result.precheck_status == provenance.PRECHECK_PASSED
    assert "provenance_reference_backed" in result.reasons
    assert result.doc["ai_mode"]["reference_count"] == 2


def test_jsonplaceholder_fake_docs_reject(monkeypatch):
    monkeypatch.setattr(
        provenance,
        "_scrapingdog_scrape",
        lambda *_args, **_kwargs: {
            "provider_status": "ok",
            "status": 200,
            "content_type": "text/html",
            "body_text": "JSONPlaceholder is a free fake REST API for testing and prototyping.",
        },
    )
    monkeypatch.setattr(
        provenance,
        "_scrapingdog_ai_mode",
        lambda *_args, **_kwargs: {
            "provider_status": "ok",
            "status": 200,
            "json": {
                "markdown": "This is a fake REST API for testing.",
                "references": [{"title": "JSONPlaceholder", "link": "https://jsonplaceholder.typicode.com"}],
            },
        },
    )

    result = provenance.evaluate_source_add_provenance(
        source_name="JSONPlaceholder",
        source_kind="web",
        declared_base_domains=("jsonplaceholder.typicode.com",),
        source_metadata=_metadata(
            api_url="https://jsonplaceholder.typicode.com",
            docs_url="https://jsonplaceholder.typicode.com/guide",
        ),
        scrapingdog_api_key="test-key",
    )

    assert result.precheck_status == provenance.PRECHECK_REJECTED
    assert "documentation_contains_fake_or_test_markers" in result.reasons


def test_negated_fake_language_does_not_reject(monkeypatch):
    monkeypatch.setattr(
        provenance,
        "_scrapingdog_scrape",
        lambda *_args, **_kwargs: {
            "provider_status": "ok",
            "status": 200,
            "content_type": "text/html",
            "body_text": "API reference quickstart endpoint authentication rate limit curl http",
        },
    )
    monkeypatch.setattr(
        provenance,
        "_scrapingdog_ai_mode",
        lambda *_args, **_kwargs: {
            "provider_status": "ok",
            "status": 200,
            "json": {
                "markdown": "This is a legitimate provider, not a fake REST API and not a dummy API.",
                "references": [
                    {"title": "Official Docs", "link": "https://openrouter.ai/docs/quickstart"},
                    {"title": "Integration Docs", "link": "https://arize.com/docs/openrouter"},
                ],
            },
        },
    )

    result = provenance.evaluate_source_add_provenance(
        source_name="OpenRouter",
        source_kind="web",
        declared_base_domains=("openrouter.ai",),
        source_metadata=_metadata(),
        scrapingdog_api_key="test-key",
    )

    assert result.precheck_status == provenance.PRECHECK_PASSED
    assert "ai_mode_identified_fake_or_test_api" not in result.reasons


def test_example_docs_remain_manual_review_not_pass(monkeypatch):
    monkeypatch.setattr(
        provenance,
        "_scrapingdog_scrape",
        lambda *_args, **_kwargs: {
            "provider_status": "ok",
            "status": 404,
            "content_type": "text/html",
            "body_text": "<title>Example Domain</title>",
        },
    )
    monkeypatch.setattr(
        provenance,
        "_scrapingdog_ai_mode",
        lambda *_args, **_kwargs: {
            "provider_status": "ok",
            "status": 200,
            "json": {"markdown": "No clear API documentation found.", "references": []},
        },
    )

    result = provenance.evaluate_source_add_provenance(
        source_name="Example API",
        source_kind="web",
        declared_base_domains=("example.com",),
        source_metadata=_metadata(api_url="https://example.com", docs_url="https://example.com/docs"),
        scrapingdog_api_key="test-key",
    )

    assert result.precheck_status == provenance.PRECHECK_MANUAL
    assert "ai_mode_no_references" in result.reasons
    assert result.precheck_status != provenance.PRECHECK_PASSED


def test_provider_failure_is_manual_review_not_exception(monkeypatch):
    monkeypatch.setattr(
        provenance,
        "_scrapingdog_scrape",
        lambda *_args, **_kwargs: {"provider_status": "error", "error_type": "TimeoutError"},
    )
    monkeypatch.setattr(
        provenance,
        "_scrapingdog_ai_mode",
        lambda *_args, **_kwargs: {"provider_status": "error", "error_type": "TimeoutError"},
    )

    result = provenance.evaluate_source_add_provenance(
        source_name="Plausible API",
        source_kind="web",
        declared_base_domains=("api.vendor.example",),
        source_metadata=_metadata(api_url="https://api.vendor.example", docs_url="https://api.vendor.example/docs"),
        scrapingdog_api_key="test-key",
    )

    assert result.precheck_status == provenance.PRECHECK_MANUAL


def test_missing_required_metadata_rejects_obvious_bad_submission():
    result = provenance.evaluate_source_add_provenance(
        source_name="No Docs API",
        source_kind="web",
        declared_base_domains=(),
        source_metadata={},
        scrapingdog_api_key="test-key",
    )

    assert result.precheck_status == provenance.PRECHECK_REJECTED
    assert "missing_documentation_url" in result.reasons


@pytest.mark.parametrize(
    ("source_name", "source_kind", "api_url", "docs_url", "domains", "reference_urls"),
    [
        (
            "SAM.gov Opportunities",
            "procurement",
            "https://api.sam.gov/opportunities/v2",
            "https://open.gsa.gov/api/get-opportunities-public-api/",
            ("sam.gov", "api.sam.gov", "open.gsa.gov"),
            ("https://sam.gov/content/opportunities", "https://open.gsa.gov/api/get-opportunities-public-api/", "https://apify.com/sam-gov"),
        ),
        (
            "SEC EDGAR Full-Text Search",
            "filing",
            "https://efts.sec.gov/LATEST",
            "https://www.sec.gov/edgar/search/efts-faq.html",
            ("sec.gov", "efts.sec.gov"),
            ("https://www.sec.gov/edgar/search/efts-faq.html", "https://sec-api.io/docs"),
        ),
        (
            "GDELT DOC 2.0",
            "news",
            "https://api.gdeltproject.org/api/v2/doc",
            "https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/",
            ("gdeltproject.org", "api.gdeltproject.org"),
            ("https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/", "https://www.tandfonline.com/doi/full/10.1080/gdelt"),
        ),
        (
            "GLEIF LEI Records",
            "registry",
            "https://api.gleif.org/api/v1",
            "https://www.gleif.org/en/lei-data/gleif-api",
            ("gleif.org", "api.gleif.org"),
            ("https://www.gleif.org/en/lei-data/gleif-api", "https://www.identity.org/gleif"),
        ),
        (
            "openFDA Approvals",
            "filing",
            "https://api.fda.gov",
            "https://open.fda.gov/apis/",
            ("fda.gov", "api.fda.gov", "open.fda.gov"),
            ("https://open.fda.gov/apis/", "https://catalog.data.gov/dataset/openfda"),
        ),
        (
            "USPTO PatentsView",
            "filing",
            "https://search.patentsview.org/api/v1",
            "https://search.patentsview.org/docs/",
            ("patentsview.org", "search.patentsview.org"),
            ("https://search.patentsview.org/docs/", "https://www.uspto.gov/ip-policy/patent-policy/patentsview"),
        ),
    ],
)
def test_six_known_miner_sources_pass_complete_current_metadata(
    monkeypatch,
    source_name,
    source_kind,
    api_url,
    docs_url,
    domains,
    reference_urls,
):
    # Production pages put substantial CSS/navigation before the useful docs.
    # The old 1,200-byte raw-HTML parser incorrectly scored these as empty.
    body = (
        "<html><head><title>Official API Documentation</title><style>"
        + (".docs{display:block}" * 500)
        + "</style></head><body><main>API reference endpoint authentication "
        "authorization rate limit curl HTTP status code</main></body></html>"
    )
    monkeypatch.setattr(
        provenance,
        "_scrapingdog_scrape",
        lambda *_args, **_kwargs: {
            "provider_status": "ok",
            "status": 200,
            "content_type": "text/html",
            "body_text": body,
        },
    )
    monkeypatch.setattr(
        provenance,
        "_scrapingdog_ai_mode",
        lambda *_args, **_kwargs: {
            "provider_status": "ok",
            "status": 200,
            "json": {
                "markdown": f"{source_name} is an established API with official documentation.",
                "references": [
                    {"title": f"Reference {index}", "link": url}
                    for index, url in enumerate(reference_urls)
                ],
            },
        },
    )
    metadata = _metadata(api_url=api_url, docs_url=docs_url)
    metadata["auth_type"] = "header" if source_name == "USPTO PatentsView" else "none"

    result = provenance.evaluate_source_add_provenance(
        source_name=source_name,
        source_kind=source_kind,
        declared_base_domains=domains,
        source_metadata=metadata,
        scrapingdog_api_key="test-key",
    )

    assert result.precheck_status == provenance.PRECHECK_PASSED
    assert result.doc["docs_completeness"]["score"] >= 2
    assert result.doc["reference_evidence"]["aligned_reference_domains"]
    assert result.doc["reference_evidence"]["independent_reference_domains"]


def test_self_references_only_remain_manual_review(monkeypatch):
    monkeypatch.setattr(
        provenance,
        "_scrapingdog_scrape",
        lambda *_args, **_kwargs: {
            "provider_status": "ok",
            "status": 200,
            "content_type": "text/html",
            "body_text": "API reference endpoint authentication rate limit curl HTTP",
        },
    )
    monkeypatch.setattr(
        provenance,
        "_scrapingdog_ai_mode",
        lambda *_args, **_kwargs: {
            "provider_status": "ok",
            "status": 200,
            "json": {
                "markdown": "The submitted site claims to provide an API.",
                "references": [
                    {"title": "Docs", "link": "https://docs.miner-owned.example/api"},
                    {"title": "Home", "link": "https://miner-owned.example"},
                ],
            },
        },
    )

    result = provenance.evaluate_source_add_provenance(
        source_name="Miner Wrapper",
        source_kind="web",
        declared_base_domains=("miner-owned.example",),
        source_metadata=_metadata(
            api_url="https://api.miner-owned.example",
            docs_url="https://docs.miner-owned.example/api",
        ),
        scrapingdog_api_key="test-key",
    )

    assert result.precheck_status == provenance.PRECHECK_MANUAL
    assert "ai_mode_legitimacy_not_confirmed" in result.reasons


def test_generic_testing_word_does_not_mark_real_docs_fake(monkeypatch):
    monkeypatch.setattr(
        provenance,
        "_scrapingdog_scrape",
        lambda *_args, **_kwargs: {
            "provider_status": "ok",
            "status": 200,
            "content_type": "text/html",
            "body_text": "API reference endpoint authentication rate limit curl HTTP sandbox for testing clients",
        },
    )
    monkeypatch.setattr(
        provenance,
        "_scrapingdog_ai_mode",
        lambda *_args, **_kwargs: {
            "provider_status": "ok",
            "status": 200,
            "json": {
                "markdown": "OpenRouter is an established provider.",
                "references": [
                    {"title": "Docs", "link": "https://openrouter.ai/docs"},
                    {"title": "GitHub", "link": "https://github.com/OpenRouterTeam"},
                ],
            },
        },
    )

    result = provenance.evaluate_source_add_provenance(
        source_name="OpenRouter",
        source_kind="web",
        declared_base_domains=("openrouter.ai",),
        source_metadata=_metadata(),
        scrapingdog_api_key="test-key",
    )

    assert result.precheck_status == provenance.PRECHECK_PASSED
    assert not result.doc["fake_test_markers"]


def test_cross_government_docs_can_pass_with_gsa_and_external_refs(monkeypatch):
    monkeypatch.setattr(
        provenance,
        "_scrapingdog_scrape",
        lambda *_args, **_kwargs: {
            "provider_status": "ok",
            "status": 200,
            "content_type": "text/html",
            "body_text": "API reference endpoint authentication rate limit curl HTTP",
        },
    )
    monkeypatch.setattr(
        provenance,
        "_scrapingdog_ai_mode",
        lambda *_args, **_kwargs: {
            "provider_status": "ok",
            "status": 200,
            "json": {
                "markdown": "VERDICT: CREDIBLE. This is the official public API documentation.",
                "references": [
                    {"title": "GSA Docs", "link": "https://open.gsa.gov/api/get-opportunities-public-api/"},
                    {"title": "Independent", "link": "https://www.pogo.org/sam-gov-api"},
                ],
            },
        },
    )

    result = provenance.evaluate_source_add_provenance(
        source_name="SAM.gov Opportunities",
        source_kind="procurement",
        declared_base_domains=("sam.gov", "api.sam.gov", "open.gsa.gov"),
        source_metadata=_metadata(
            api_url="https://api.sam.gov/opportunities/v2",
            docs_url="https://open.gsa.gov/api/get-opportunities-public-api/",
        ),
        scrapingdog_api_key="test-key",
    )

    assert result.precheck_status == provenance.PRECHECK_PASSED
    assert result.doc["reference_evidence"]["trusted_government_cross_domain"] is True


def test_negated_miner_wrapper_wording_is_credible(monkeypatch):
    monkeypatch.setattr(
        provenance,
        "_scrapingdog_scrape",
        lambda *_args, **_kwargs: {
            "provider_status": "ok",
            "status": 200,
            "content_type": "text/html",
            "body_text": "API reference endpoint authentication rate limit curl HTTP",
        },
    )
    monkeypatch.setattr(
        provenance,
        "_scrapingdog_ai_mode",
        lambda *_args, **_kwargs: {
            "provider_status": "ok",
            "status": 200,
            "json": {
                "markdown": (
                    "VERDICT: CREDIBLE. GDELT is a legitimate public API and does not appear to be a "
                    "miner-owned wrapper."
                ),
                "references": [
                    {"title": "OSINT Review", "link": "https://osintnewsletter.osint-jobs.com/gdelt"},
                    {"title": "Research", "link": "https://www.sobigdata.eu/gdelt"},
                ],
            },
        },
    )

    result = provenance.evaluate_source_add_provenance(
        source_name="GDELT DOC 2.0",
        source_kind="news",
        declared_base_domains=("gdeltproject.org", "api.gdeltproject.org"),
        source_metadata=_metadata(
            api_url="https://api.gdeltproject.org/api/v2/doc",
            docs_url="https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/",
        ),
        scrapingdog_api_key="test-key",
    )

    assert result.precheck_status == provenance.PRECHECK_PASSED
    assert result.doc["ai_legitimacy"]["uncertain_markers"] == []
