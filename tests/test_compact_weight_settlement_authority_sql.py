from __future__ import annotations

from pathlib import Path
import re

from gateway.tee.supabase_schema_preflight_v2 import REQUIRED_SUPABASE_V2_RPCS


ROOT = Path(__file__).resolve().parents[1]
MIGRATION = "scripts/149-research-lab-compact-weight-settlement-authority.sql"
SQL = (ROOT / MIGRATION).read_text(encoding="utf-8")


def test_compact_weight_settlement_authority_is_bounded_and_additive() -> None:
    assert re.search(r"\bBEGIN\s*;", SQL)
    assert re.search(r"\bCOMMIT\s*;\s*$", SQL)
    assert "SET LOCAL lock_timeout = '5s'" in SQL
    assert "research_lab_compact_weight_authority_size_v2" in SQL
    assert "pg_catalog.octet_length(authority_doc::TEXT) <= 8388608" in SQL
    assert "NOT VALID" in SQL
    assert "VALIDATE CONSTRAINT research_lab_compact_weight_authority_size_v2" in SQL
    assert not re.search(r"^\s*UPDATE\s+", SQL, flags=re.MULTILINE)
    assert not re.search(r"^\s*DELETE\s+FROM\s+", SQL, flags=re.MULTILINE)
    assert "DROP TABLE" not in SQL.upper()


def test_compact_weight_settlement_contract_preserves_v2_guards() -> None:
    for marker in (
        "size_constraint_valid",
        "append_only_trigger_enabled",
        "identity_unique_constraint_enabled",
        "row_level_security_enabled",
        "finalized_stage_supported",
        "prevent_research_lab_bounded_v2_mutation",
        "c.contype = 'u'",
        "authority_stage%finalized",
    ):
        assert marker in SQL
    assert re.search(
        r"REVOKE\s+ALL\s+ON\s+FUNCTION[\s\S]+?FROM\s+PUBLIC,\s*anon,\s*authenticated",
        SQL,
        flags=re.IGNORECASE,
    )
    assert re.search(
        r"GRANT\s+EXECUTE\s+ON\s+FUNCTION[\s\S]+?TO\s+service_role",
        SQL,
        flags=re.IGNORECASE,
    )
    assert (
        MIGRATION,
        "research_lab_compact_weight_settlement_contract_v1",
    ) in REQUIRED_SUPABASE_V2_RPCS


def test_compact_weight_settlement_contract_is_fail_closed_on_drift() -> None:
    assert "differs from the canonical 8 MiB bound" in SQL
    assert "c.convalidated" in SQL
    assert "t.tgenabled IN ('O', 'A')" in SQL
    assert "cls.relrowsecurity" in SQL
    assert "NOTIFY pgrst, 'reload schema'" in SQL
