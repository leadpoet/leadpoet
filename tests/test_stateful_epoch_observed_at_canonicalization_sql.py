"""Migration 112: the staged cutover row keeps its canonical "Z" timestamp.

The staging RPCs verify the cutover row with an exact JSONB round-trip
(``to_jsonb(jsonb_populate_record(...)) - 'created_at' IS DISTINCT FROM
p_cutover_row``). ``first_observed_at`` arrives as the canonical
``YYYY-MM-DDTHH:MM:SSZ`` string that every snapshot document and receipt
hash commits to, so the column must store that exact string: a TIMESTAMPTZ
column re-serializes it as ``+00:00`` and every staged row fails closed
with 'stateful epoch V2 cutover row shape is invalid'.

Executable proof lives in scripts/rehearse_stateful_epoch_stage_v2.py,
which applies the real migrations to a disposable cluster and drives
stage_v2 both without 112 (expects the rejection) and with it (expects
stateful_staged).
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SQL = (
    REPO_ROOT / "scripts" / "112-canonicalize-cutover-observed-at.sql"
).read_text(encoding="utf-8")
HISTORICAL_PREDECESSOR_SQL = (
    REPO_ROOT
    / "scripts"
    / "105-stateful-subnet-epoch-historical-predecessor-v2.sql"
).read_text(encoding="utf-8")
AUTHORITY_SQL = (
    REPO_ROOT / "scripts" / "101-stateful-subnet-epoch-authority.sql"
).read_text(encoding="utf-8")
REHEARSAL = (
    REPO_ROOT / "scripts" / "rehearse_stateful_epoch_stage_v2.py"
).read_text(encoding="utf-8")
SUBNET_EPOCH = (
    REPO_ROOT / "Leadpoet" / "utils" / "subnet_epoch.py"
).read_text(encoding="utf-8")


def _trigger_body(sql: str) -> str:
    match = re.search(
        r"CREATE OR REPLACE FUNCTION\s*\n?"
        r"public\.validate_research_lab_stateful_epoch_cutover_v2\(\)"
        r"(.*?)\n\$\$;",
        sql,
        re.DOTALL,
    )
    assert match is not None, "cutover validation trigger function is missing"
    return match.group(1)


def test_migration_refuses_to_run_on_a_nonempty_cutover_ledger():
    assert (
        "FROM public.research_lab_stateful_subnet_epoch_cutovers_v1" in SQL
    )
    assert "requires an empty cutover table" in SQL


def test_migration_swaps_exactly_one_semantic_check_for_the_type_change():
    assert "pg_get_constraintdef(oid) LIKE '%first_observed_at%'" in SQL
    assert "expected exactly one first_observed_at CHECK" in SQL
    assert re.search(
        r"ALTER COLUMN first_observed_at TYPE TEXT\s*\n\s*USING to_char\(\s*\n"
        r"\s*first_observed_at AT TIME ZONE 'UTC',\s*\n"
        r"\s*'YYYY-MM-DD\"T\"HH24:MI:SS\"Z\"'",
        SQL,
    ), "type change must render the canonical UTC Z format"


def test_replacement_constraints_are_strictly_stronger():
    # The dropped CHECK compared instants; the replacements pin the exact
    # canonical string, its format, and its parseability.
    assert (
        "research_lab_stateful_epoch_cutover_observed_at_format_v1" in SQL
    )
    assert (
        r"first_observed_at ~ '^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$'" in SQL
    )
    assert "(first_observed_at::TIMESTAMPTZ) IS NOT NULL" in SQL
    assert (
        "research_lab_stateful_epoch_cutover_observed_at_doc_v1" in SQL
    )
    assert (
        "CHECK ((first_snapshot_doc->>'observed_at') = first_observed_at)"
        in SQL
    )


def test_staging_row_shape_equality_is_not_weakened():
    # 112 must fix the column, never relax the exact round-trip checks that
    # protect the staged row (105 keeps both of them).
    assert (
        HISTORICAL_PREDECESSOR_SQL.count("IS DISTINCT FROM p_cutover_row") == 2
    )
    executable_112 = "\n".join(
        line for line in SQL.splitlines() if not line.lstrip().startswith("--")
    )
    assert "IS DISTINCT FROM p_cutover_row" not in executable_112


def test_trigger_repin_changes_only_the_candidate_timestamp_comparison():
    before = _trigger_body(HISTORICAL_PREDECESSOR_SQL)
    after = _trigger_body(SQL)
    canonical_before = re.sub(r"\s+", " ", before).strip()
    canonical_after = re.sub(r"\s+", " ", after).strip()
    expected = canonical_before.replace(
        "candidate_row.observed_at IS DISTINCT FROM NEW.first_observed_at",
        "candidate_row.observed_at IS DISTINCT FROM"
        " NEW.first_observed_at::TIMESTAMPTZ",
    )
    assert canonical_after == expected, (
        "112 must re-pin migration 105's trigger verbatim except for the"
        " explicit TIMESTAMPTZ cast on the canonical string column"
    )
    assert "SECURITY DEFINER" in after
    assert "SET search_path = ''" in after


def test_candidate_and_boundary_ledgers_keep_timestamptz_semantics():
    # Only the staged cutover row stores the canonical string; the candidate
    # and boundary ledgers written through the attested store keep semantic
    # timestamps (their readbacks normalize in _row_value_equal).
    assert "ALTER TABLE public.research_lab_stateful_subnet_epoch_candidates_v1" not in SQL
    assert "boundaries_v1" not in SQL
    assert "snapshots_v1" not in SQL
    assert AUTHORITY_SQL.count(
        "CHECK ((snapshot_doc->>'observed_at')::TIMESTAMPTZ = observed_at)"
    ) >= 1


def test_migration_reloads_postgrest_schema_cache():
    assert "NOTIFY pgrst, 'reload schema';" in SQL


def test_rehearsal_covers_both_directions_with_the_real_migrations():
    for migration in (
        "101-stateful-subnet-epoch-authority.sql",
        "105-stateful-subnet-epoch-historical-predecessor-v2.sql",
        "106-repair-stateful-epoch-fence-trigger-coverage.sql",
        "110-qualify-stateful-epoch-v2-binding.sql",
        "111-refresh-unactivated-stateful-epoch-fence.sql",
        "112-canonicalize-cutover-observed-at.sql",
    ):
        assert migration in REHEARSAL
    assert "--skip-112" in REHEARSAL
    assert "cutover row shape is invalid" in REHEARSAL
    assert "stateful_staged" in REHEARSAL
    assert re.search(r'OBSERVED_AT = "\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z"', REHEARSAL)


def test_canonical_snapshot_timestamp_stays_z_suffixed_at_the_source():
    # b9d07c22 pinned the canonical format; the TEXT column depends on it.
    assert 'strftime("%Y-%m-%dT%H:%M:%SZ")' in SUBNET_EPOCH
