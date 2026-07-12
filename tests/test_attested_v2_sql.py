from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SQL = (ROOT / "scripts" / "86-research-lab-attested-v2-authority.sql").read_text(
    encoding="utf-8"
)


def test_v2_schema_is_separate_from_applied_v1_history():
    assert "research_lab_attested_execution_receipts_v2" in SQL
    assert "research_lab_attested_weight_bundles_v2" in SQL
    assert "ALTER TABLE public.research_lab_attested_execution_receipts " not in SQL
    assert "verification_mode" not in SQL


def test_v2_schema_has_complete_append_only_authority_tables():
    for table in (
        "research_lab_provider_credential_envelopes_v2",
        "research_lab_attested_boot_identities_v2",
        "research_lab_attested_transport_attempts_v2",
        "research_lab_attested_execution_receipts_v2",
        "research_lab_attested_receipt_edges_v2",
        "research_lab_attested_receipt_transport_v2",
        "research_lab_attested_artifact_links_v2",
        "research_lab_signed_transition_commands_v2",
        "research_lab_attested_weight_bundles_v2",
        "research_lab_attested_publication_events_v2",
        "research_lab_attested_weight_finalizations_v2",
    ):
        assert f"public.{table}" in SQL
        assert table in SQL.split("FOREACH table_name IN ARRAY ARRAY[", 1)[1]
    assert "BEFORE UPDATE OR DELETE" in SQL
    assert "ENABLE ROW LEVEL SECURITY" in SQL
    assert "TO service_role" in SQL


def test_transport_schema_cannot_label_host_failure_as_provider_response():
    assert "terminal_status IN ('authenticated_response', 'transport_failure')" in SQL
    assert "terminal_status = 'authenticated_response'" in SQL
    assert "http_status IS NOT NULL" in SQL
    assert "tls_peer_chain_hash IS NOT NULL" in SQL
    assert "terminal_status = 'transport_failure'" in SQL
    assert "http_status IS NULL" in SQL
    assert "failure_code IS NOT NULL" in SQL


def test_v2_schema_rejects_secrets_and_requires_durable_publication():
    for marker in (
        "openrouter_api_key",
        "scrapingdog_api_key",
        "exa_api_key",
        "raw_secret",
        "provider_output",
        "request_body",
        "response_body",
        "proxy-authorization",
    ):
        assert marker in SQL.lower()
    assert "weight_submission_event_hash" in SQL
    assert "durable_readback_hash" in SQL
    assert "publication_receipt_hash" in SQL
    assert "validator.weights.finalized.v2" in SQL
    assert "state_transition_hash" in SQL


def test_v2_schema_accepts_signed_storage_receipts_and_requires_compliance_lock():
    assert "'leadpoet.artifact_persistence.v2'" in SQL
    assert "object_lock_mode = 'COMPLIANCE'" in SQL
    assert "'GOVERNANCE'" not in SQL
