from pathlib import Path


def test_enclave_sealed_openrouter_migration_is_additive_and_append_only():
    sql = Path(
        "scripts/91-research-lab-enclave-sealed-provider-credentials-v2.sql"
    ).read_text(encoding="utf-8")
    assert "leadpoet.provider_credential_envelope.v2" in sql
    assert "leadpoet.provider_credential_envelope.enclave.v2" in sql
    assert "DROP TABLE" not in sql.upper()
    assert "UPDATE " not in sql.upper()
    assert "DELETE " not in sql.upper()
    assert "research_lab_provider_credential_envelopes_v2" in sql
