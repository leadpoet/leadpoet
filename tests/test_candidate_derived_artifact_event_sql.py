from __future__ import annotations

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_MIGRATION = ROOT / "scripts/95-research-lab-git-tree-autoresearch.sql"
MIGRATION = ROOT / "scripts/165-research-lab-candidate-derived-artifact-event.sql"


def _event_types(path: Path) -> set[str]:
    sql = path.read_text(encoding="utf-8")
    match = re.search(
        r"ADD CONSTRAINT research_lab_auto_research_loop_events_event_type_check"
        r"\s+CHECK \(\s*event_type IN \((.*?)\)\s*\)",
        sql,
        flags=re.DOTALL,
    )
    assert match is not None
    return set(re.findall(r"'([a-z][a-z0-9_]*)'", match.group(1)))


def test_candidate_derived_artifact_failure_extends_historical_allowlist() -> None:
    historical = _event_types(HISTORICAL_MIGRATION)
    current = _event_types(MIGRATION)

    assert current == historical | {"candidate_derived_artifact_failed"}


def test_migration_is_transactional_and_rerunnable() -> None:
    sql = MIGRATION.read_text(encoding="utf-8")
    assert sql.startswith("BEGIN;\n")
    assert sql.rstrip().endswith("COMMIT;")
    assert "DROP CONSTRAINT IF EXISTS" in sql
    assert "ADD CONSTRAINT" in sql
    assert "NOT VALID" in sql
    assert "VALIDATE CONSTRAINT" in sql
