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
