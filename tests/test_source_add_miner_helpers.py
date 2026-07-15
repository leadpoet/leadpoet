import re
from pathlib import Path

import pytest

from research_lab.source_add import SourceAddSourceKind, validate_source_add_adapter_manifest
from research_lab.source_add_miner import (
    SOURCE_ADD_SOURCE_KIND_DESCRIPTIONS,
    SOURCE_ADD_SOURCE_KINDS,
    build_source_add_submission_docs,
    build_source_add_metadata,
    parse_source_add_domains,
)


EXPECTED_SOURCE_KINDS = (
    "web",
    "filing",
    "news",
    "registry",
    "procurement",
    "social",
    "hiring",
    "tech_stack",
    "funding",
    "firmographic",
    "people",
    "intent",
    "reviews",
    "events",
)


def test_source_add_kind_taxonomy_has_one_authority_and_descriptions():
    assert SOURCE_ADD_SOURCE_KINDS == EXPECTED_SOURCE_KINDS
    assert SOURCE_ADD_SOURCE_KINDS == tuple(kind.value for kind in SourceAddSourceKind)
    assert set(SOURCE_ADD_SOURCE_KIND_DESCRIPTIONS) == set(SOURCE_ADD_SOURCE_KINDS)


def test_source_add_catalog_constraint_migration_matches_code_taxonomy():
    migration = (
        Path(__file__).resolve().parents[1] / "scripts" / "84-expand-source-add-source-kinds.sql"
    ).read_text(encoding="utf-8")
    match = re.search(r"source_kind IN \((.*?)\)\s*\) NOT VALID", migration, re.DOTALL)

    assert match is not None
    assert tuple(re.findall(r"'([^']+)'", match.group(1))) == SOURCE_ADD_SOURCE_KINDS


@pytest.mark.parametrize(
    "source_kind",
    ("hiring", "tech_stack", "funding", "firmographic", "people", "intent", "reviews", "events"),
)
def test_build_source_add_docs_accepts_expanded_gtm_source_kinds(source_kind):
    manifest, _brief, _key, _metadata = build_source_add_submission_docs(
        miner_hotkey="5MinerHotkey",
        source_name=f"Example {source_kind} API",
        source_kind=source_kind,
        declared_base_domains=("example.com",),
        endpoint_summary="GET /v1/search",
        claimed_output_type="evidence",
        credential_supplied=False,
    )

    assert manifest["source_kind"] == source_kind
    assert validate_source_add_adapter_manifest(manifest) == []


def test_parse_source_add_domains_normalizes_and_dedupes():
    assert parse_source_add_domains(
        "https://www.example.com/path, api.vendor.io www.example.com"
    ) == ("example.com", "api.vendor.io")


def test_build_source_add_docs_emit_valid_manifest_without_credential():
    manifest, source_brief, idempotency_key, metadata = build_source_add_submission_docs(
        miner_hotkey="5MinerHotkey",
        source_name="Example News API",
        source_kind="news",
        declared_base_domains=("example.com",),
        endpoint_summary="GET /v1/search returns article evidence refs and normalized metadata",
        claimed_output_type="intent evidence",
        credential_supplied=False,
    )

    assert validate_source_add_adapter_manifest(manifest) == []
    assert manifest["credential_policy"] == "no_credentials"
    assert "evidence_refs" in manifest["allowed_output_fields"]
    assert idempotency_key.startswith("research-source-add:5MinerHotkey:")
    assert "GET /v1/search" in source_brief
    assert metadata == {}
    assert "secret" not in str(manifest).lower()


def test_build_source_add_docs_rejects_miner_credentials():
    with pytest.raises(ValueError, match="miners must not submit"):
        build_source_add_submission_docs(
            miner_hotkey="5MinerHotkey",
            source_name="Registry API",
            source_kind="registry",
            declared_base_domains=("registry.example",),
            endpoint_summary="POST /lookup returns firmographic evidence refs",
            claimed_output_type="firmographic",
            credential_supplied=True,
        )


def test_build_source_add_docs_emit_structured_metadata_and_derived_domains():
    manifest, source_brief, _idempotency_key, metadata = build_source_add_submission_docs(
        miner_hotkey="5MinerHotkey",
        source_name="OpenRouter",
        source_kind="web",
        api_base_url="https://openrouter.ai/api/v1",
        documentation_url="https://openrouter.ai/docs/quickstart",
        auth_type="bearer",
        endpoint_examples=[
            {
                "method": "POST",
                "path": "/api/v1/chat/completions",
                "purpose": "Create a chat completion",
                "example_query": '{"model":"openai/gpt-4o-mini","messages":[]}',
            }
        ],
        rate_limit_notes="Provider documents model/account-specific rate limits.",
        data_provenance_notes="Routes to model providers and returns metadata.",
        third_party_refs=("https://github.com/OpenRouterTeam",),
        credential_supplied=False,
    )

    assert validate_source_add_adapter_manifest(manifest) == []
    assert manifest["declared_base_domains"] == ["openrouter.ai"]
    assert manifest["credential_policy"] == "no_credentials"
    assert metadata["documentation_url"] == "https://openrouter.ai/docs/quickstart"
    assert metadata["auth_type"] == "bearer"
    assert metadata["endpoint_examples"][0]["path"] == "/api/v1/chat/completions"
    assert "Documentation URL: https://openrouter.ai/docs/quickstart" in source_brief
    assert "Source API credential" not in source_brief
    assert "sk-" not in str(manifest).lower()


def test_source_add_metadata_rejects_missing_docs_bad_auth_and_bad_endpoint():
    try:
        build_source_add_metadata(
            api_base_url="https://api.example.com",
            documentation_url="",
            auth_type="custom",
            endpoint_examples=[{"method": "GET", "path": "https://bad.example/path", "purpose": "x", "example_query": "q"}],
            rate_limit_notes="unknown",
        )
    except ValueError as exc:
        assert "auth_type" in str(exc)
    else:
        raise AssertionError("bad auth type should fail")

    try:
        build_source_add_metadata(
            api_base_url="https://api.example.com",
            documentation_url="https://docs.example.com",
            auth_type="none",
            endpoint_examples=[],
            rate_limit_notes="unknown",
        )
    except ValueError as exc:
        assert "endpoint example" in str(exc)
    else:
        raise AssertionError("missing endpoint examples should fail")

    for unsafe_path in (
        "/v1/search?q=test",
        "/v1/{tenant}/search",
        "/v1/../admin",
        "/v1\\admin",
        "/v1/%2e%2e/admin",
        "/v1/search%2Fadmin",
        "/v1/search results",
    ):
        with pytest.raises(ValueError, match="relative API path"):
            build_source_add_metadata(
                api_base_url="https://api.example.com",
                documentation_url="https://docs.example.com",
                auth_type="none",
                endpoint_examples=[
                    {
                        "method": "GET",
                        "path": unsafe_path,
                        "purpose": "Search",
                        "example_query": "test",
                    }
                ],
                rate_limit_notes="unknown",
            )

    with pytest.raises(ValueError, match="api_base_url path"):
        build_source_add_metadata(
            api_base_url="https://api.example.com/v1%2Fadmin",
            documentation_url="https://docs.example.com/reference%20guide",
            auth_type="none",
            endpoint_examples=[
                {
                    "method": "GET",
                    "path": "/records",
                    "purpose": "Search",
                    "example_query": "q=test",
                }
            ],
            rate_limit_notes="unknown",
        )


def test_build_source_add_docs_reject_invalid_kind():
    try:
        build_source_add_submission_docs(
            miner_hotkey="5MinerHotkey",
            source_name="Bad API",
            source_kind="payments",
            declared_base_domains=("example.com",),
            endpoint_summary="GET /v1/search",
            claimed_output_type="intent",
            credential_supplied=False,
        )
    except ValueError as exc:
        assert "source_kind" in str(exc)
    else:
        raise AssertionError("invalid source kind should fail")
