from research_lab.source_add import validate_source_add_adapter_manifest
from research_lab.source_add_miner import (
    build_source_add_submission_docs,
    parse_source_add_domains,
)


def test_parse_source_add_domains_normalizes_and_dedupes():
    assert parse_source_add_domains(
        "https://www.example.com/path, api.vendor.io www.example.com"
    ) == ("example.com", "api.vendor.io")


def test_build_source_add_docs_emit_valid_manifest_without_credential():
    manifest, source_brief, idempotency_key = build_source_add_submission_docs(
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
    assert "secret" not in str(manifest).lower()


def test_build_source_add_docs_emit_valid_manifest_with_credential_ref_only():
    manifest, source_brief, _idempotency_key = build_source_add_submission_docs(
        miner_hotkey="5MinerHotkey",
        source_name="Registry API",
        source_kind="registry",
        declared_base_domains=("registry.example",),
        endpoint_summary="POST /lookup returns firmographic evidence refs",
        claimed_output_type="firmographic",
        credential_supplied=True,
    )

    assert validate_source_add_adapter_manifest(manifest) == []
    assert manifest["credential_policy"] == "credential_ref_only"
    assert "credential_ref" not in manifest
    assert "Auth material submitted separately: yes" in source_brief


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
