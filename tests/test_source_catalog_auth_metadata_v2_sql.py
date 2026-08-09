from pathlib import Path


SQL = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "147-research-lab-source-catalog-auth-metadata.sql"
).read_text(encoding="utf-8")


def test_source_catalog_auth_metadata_migration_is_narrow_and_validated():
    for value in (
        "source_add_catalog_snapshot_v2",
        "leadpoet.source_add_catalog_snapshot.v2",
        "research_lab_attested_execution_result_secret_free_v2",
        "provider_registry_entry",
        "runtime_catalog",
        "auth_name",
        "authorization",
        "proxy-authorization",
    ):
        assert value in SQL
    assert "NOT IN ('header', 'bearer')" in SQL
    assert "match_count <> 1" in SQL
    assert (
        "VALIDATE CONSTRAINT\n"
        "        research_lab_attested_execution_results_v2_result_doc_check"
    ) in SQL
    assert "WHEN OTHERS THEN\n        RETURN FALSE" in SQL


def test_source_catalog_auth_metadata_migration_preserves_generic_secret_bans():
    for value in (
        "sk-or-",
        "sb_secret",
        "service_role",
        "openrouter_api_key",
        "scrapingdog_api_key",
        "raw_secret",
        "provider_output",
        "request_body",
        "response_body",
    ):
        assert value in SQL
    assert "p_operation <> 'source_add_catalog_snapshot_v2'" in SQL
    assert "private_registry_rows" in SQL
    assert "private_registry_rows -" not in SQL
