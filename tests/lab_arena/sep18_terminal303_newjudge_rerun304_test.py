"""Public boundaries for the separately proved, exact-row Sep18 recovery.

The literal SQL is also applied twice to a protected terminal303 snapshot in
disposable PostgreSQL, followed by the production service lifecycle. Private
round documents and provider responses must never enter this public fixture.
"""
from pathlib import Path
import hashlib
import re


SQL = Path(__file__).parents[2] / "scripts/304-arena-2026-09-18-terminal303-baseline-newjudge-rerun.sql"


def test_committed_migration_is_the_exact_complete_transition_proof():
    assert hashlib.sha256(SQL.read_bytes()).hexdigest() == (
        "de6138e35192e1e5ce7c8102928e6ac015c30b792726a02f07ae20f6127e5cd0"
    )


def test_public_migration_does_not_embed_private_results_or_erase_history():
    body = SQL.read_text()
    for forbidden in (
        '"final_ranking"', '"cost_summary"', '"qualification_doc":',
        '"company_name"', '"company_identity_key"', '"intent_details"',
        "linkedin.com/", "DELETE FROM", "TRUNCATE ",
        "INSERT INTO public.lab_arena_ledger",
    ):
        assert forbidden not in body
    assert re.search(r"__[A-Z0-9_]+__", body) is None


def test_publication_requires_complete_fresh_scores_and_settled_costs():
    body = SQL.read_text().split(
        "CREATE OR REPLACE FUNCTION public.lab_arena_sep18_newjudge_rerun304_publication_guard_v1()", 1
    )[1]
    assert "status NOT IN('accepted','failed')" in body
    assert "count(DISTINCT scored_run_id)" in body
    assert "status='accepted' AND terminal_cause='accepted'" in body
    assert "success_unresolved_calls" in body
    assert "OR unsettled<>0" in body
    assert "NEW.reward_basis_hash IS DISTINCT FROM" in body
