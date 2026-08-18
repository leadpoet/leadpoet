from pathlib import Path
import re

from leadpoet_canonical.attested_v2 import ROLE_PURPOSES


ROOT = Path(__file__).resolve().parents[1]
MIGRATION = (
    ROOT
    / "scripts"
    / "138-research-lab-ancestry-checkpoint-bootstrap-purpose.sql"
)
SQL = MIGRATION.read_text(encoding="utf-8")


def test_bootstrap_purpose_migration_is_additive_and_validated() -> None:
    assert SQL.lstrip().startswith("-- Admit the measured coordinator job")
    assert re.search(r"\bBEGIN\s*;", SQL)
    assert re.search(r"\bCOMMIT\s*;\s*$", SQL)
    assert "SET LOCAL lock_timeout = '5s'" in SQL
    assert "research_lab.ancestry_checkpoint_bootstrap.v2" in SQL
    assert "NOT VALID" in SQL
    assert (
        "VALIDATE CONSTRAINT\n"
        "        research_lab_attested_execution_receipts_v2_role_purpose_check"
        in SQL
    )
    assert not re.search(r"^\s*(?:UPDATE|DELETE\s+FROM|INSERT\s+INTO)\s+", SQL, re.M)
    assert "research_lab_ancestry_checkpoint_bootstrap_contract_v2" in SQL
    assert "leadpoet.ancestry_checkpoint_bootstrap_contract.v2" in SQL
    assert "constraint_meta.convalidated" in SQL
    assert "pg_catalog.pg_get_constraintdef" in SQL
    assert re.search(
        r"GRANT\s+EXECUTE\s+ON\s+FUNCTION[\s\S]+?"
        r"research_lab_ancestry_checkpoint_bootstrap_contract_v2\(\)"
        r"[\s\S]+?TO\s+service_role",
        SQL,
        re.IGNORECASE,
    )


def test_bootstrap_purpose_migration_matches_canonical_allowlist_exactly() -> None:
    later_purposes = {
        "gateway_coordinator": {
            "research_lab.allocation_settlement_frontier_bootstrap.v2"
        },
        "gateway_scoring": {
            "research_lab.candidate_hybrid_test.v2",
            "research_lab.candidate_hybrid_discovery.v2",
            "research_lab.model_compatibility.v2",
        },
    }
    for role, expected_purposes in ROLE_PURPOSES.items():
        match = re.search(
            rf"role = '{re.escape(role)}' AND purpose IN \((.*?)\n\s*\)\)",
            SQL,
            re.DOTALL,
        )
        assert match is not None, role
        migrated_purposes = set(re.findall(r"'([^']+)'", match.group(1)))
        historical_purposes = set(expected_purposes) - later_purposes.get(
            role, set()
        )
        assert migrated_purposes == historical_purposes, role
