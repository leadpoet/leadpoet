"""Exercise the exact retained-score repair using the existing rerun295 lifecycle."""
from __future__ import annotations

import hashlib
import copy
import json
import re
from pathlib import Path

import pytest

from tests.lab_arena import sep18_published_rerun295_postgres_test as previous
from tests.lab_arena.parallel_twenty_icp_execution_postgres_test import (
    test_parallel_execution_preserves_standard_twenty_icp_scoring_and_costs as ordinary_lifecycle,
)

database = previous.database
ROOT = Path(__file__).parents[2]
MIGRATION = ROOT / "scripts/296-arena-2026-09-18-rerun295-retained-scoring.sql"
CONFIG_HASH = "a583f20ca7573721258e27b5f80fd9128511a01a4ab50203f234ea8c69db3fb9"
SOURCE_HASH = "1ba24d1abac3849b8900238cb3f8f2ced3aab41fbe8f30064c0dcbeefc9c3cd3"
SOURCE_COMMIT = "e21d29698edb60e6b5635f5f27e5a0058709210f"


def _fixture_sql(configuration):
    """Bind production predicates to the existing hermetic295 sealed fixture."""
    sql = MIGRATION.read_text()
    match = re.search(r"\$configuration\$(.*?)\$configuration\$", sql, re.S)
    assert match and hashlib.sha256(previous._compact(json.loads(match[1])).encode()).hexdigest() == CONFIG_HASH
    sql = sql.replace(match[1], previous._compact(configuration))
    assert sql.count(SOURCE_HASH) == sql.count(SOURCE_COMMIT) == sql.count("641923") == 1
    return sql.replace(SOURCE_HASH, previous.NEW_SOURCE_SHA).replace(
        SOURCE_COMMIT, previous.NEW_SOURCE_COMMIT).replace("641923", str(previous.NEW_SOURCE_SIZE))


def test_production_scope_matches_unchanged_sealed295():
    sql = MIGRATION.read_text()
    configuration = json.loads(re.search(r"\$configuration\$(.*?)\$configuration\$", sql, re.S)[1])
    assert hashlib.sha256(previous._compact(configuration).encode()).hexdigest() == CONFIG_HASH
    sealed = (ROOT / "scripts/295-arena-2026-09-18-published-baseline-rerun.sql").read_text()
    old = json.loads(re.search(r"\$terminal_round\$(.*?)\$terminal_round\$", sealed, re.S)[1])["configuration_doc"]
    old["schedule"] = json.loads(re.search(r"\$forward_schedule\$(.*?)\$forward_schedule\$", sealed, re.S)[1])
    old["baseline_source_url"] = configuration["baseline_source_url"]
    assert configuration == old
    assert SOURCE_HASH in sealed and SOURCE_COMMIT in sealed
    assert "cac6ba180ff5b386706885e4318f9d8d3563c53d8afde002ecfb5516cedbed05" in sql


def _reject_invalid_reuse(connection, stage, items):
    item = next(row for row in items if row["submission_id"] != previous.BASELINE)
    score_id = f"{previous.ROUND}:{item['submission_id']}:{stage}:{item['icp_position']}:score:1"
    mutations = [
        ("missing score", "DELETE FROM public.lab_arena_runs WHERE run_id=%s", (score_id,)),
        ("pending score", "UPDATE public.lab_arena_runs SET status='pending' WHERE run_id=%s", (score_id,)),
        ("assignment mismatch", "UPDATE public.lab_arena_runs SET assignment_id=assignment_id||':wrong' WHERE run_id=%s", (score_id,)),
        ("scored run mismatch", "UPDATE public.lab_arena_runs SET scored_run_id='wrong-run' WHERE run_id=%s", (score_id,)),
        ("missing judge output", "UPDATE public.lab_arena_runs SET output_ref=NULL WHERE run_id=%s", (score_id,)),
        ("missing cache hash", "UPDATE public.lab_arena_runs SET judgment_input_hash=NULL WHERE run_id=%s", (score_id,)),
        ("wrong source hash", "UPDATE public.lab_arena_submissions SET submission_doc=jsonb_set(submission_doc,'{source_sha256}','\"wrong\"') WHERE submission_id=%s", (previous.BASELINE,)),
        ("wrong source ref", "UPDATE public.lab_arena_submissions SET source_ref='arena/arena-2026-09-18/sources/baseline-2026-09-18-wrong.tar.gz' WHERE submission_id=%s", (previous.BASELINE,)),
        ("wrong config", "UPDATE public.lab_arena_rounds SET configuration_doc=jsonb_set(configuration_doc,'{companies_per_icp}','4') WHERE round_id=%s", (previous.ROUND,)),
        ("wrong archive", "UPDATE public.lab_arena_rounds SET cancel_reason='wrong-archive' WHERE round_id=%s", (previous.ARCHIVE,)),
        ("changed ledger seal", "UPDATE public.lab_arena_ledger SET amount_microusd=2 WHERE entry_id=(SELECT min(entry_id) FROM public.lab_arena_ledger WHERE submission_id=%s)", (item["submission_id"],)),
    ]
    with connection.cursor() as cursor:
        for label, mutation, parameters in mutations:
            cursor.execute("SAVEPOINT rejected_reuse")
            cursor.execute("SET LOCAL session_replication_role=replica")
            cursor.execute(mutation, parameters)
            cursor.execute("SET LOCAL session_replication_role=origin")
            with pytest.raises(Exception) as error:
                cursor.execute("SELECT public.lab_arena_open_scoring_v2(%s,%s::smallint,%s::jsonb)",
                    (previous.ROUND, stage, json.dumps(items)))
            assert getattr(error.value, "pgcode", None) == "22023", label
            cursor.execute("ROLLBACK TO SAVEPOINT rejected_reuse")
        for key in ("judgment_input_hash", "judgment_cache_key", "judgment_scope_doc"):
            invalid = copy.deepcopy(items)
            target = next(row for row in invalid if row["submission_id"] == item["submission_id"] and row["icp_position"] == item["icp_position"])
            if key == "judgment_scope_doc":
                target[key]["unexpected_scope"] = True
            else:
                target[key] = "sha256:" + "f" * 64
                if key == "judgment_input_hash":
                    target["judgment_scope_doc"]["scoring_input_hash"] = target[key]
                else:
                    target["judgment_scope_doc"]["cache_key"] = target[key]
                    target["reuse_cache_key"] = target[key]
            cursor.execute("SAVEPOINT rejected_reuse")
            with pytest.raises(Exception) as error:
                cursor.execute("SELECT public.lab_arena_open_scoring_v2(%s,%s::smallint,%s::jsonb)",
                    (previous.ROUND, stage, json.dumps(invalid)))
            assert getattr(error.value, "pgcode", None) == "22023", key
            cursor.execute("ROLLBACK TO SAVEPOINT rejected_reuse")
        cursor.execute("SELECT public.lab_arena_sep18_published_rerun295_nonbaseline_valid_v1()")
        assert cursor.fetchone()[0] is True
    connection.rollback()


def _instrument_seal_calls(cursor):
    """Local PG counters measure expensive seal checks without changing results."""
    for name, counter in (
        ("lab_arena_sep18_published_rerun295_nonbaseline_valid_v1", "repair296_miner_seal_calls"),
        ("lab_arena_sep18_rerun291_archive_valid295_v1", "repair296_archive_seal_calls"),
    ):
        cursor.execute(f"ALTER FUNCTION public.{name}() RENAME TO {name}_original")
        cursor.execute(f"CREATE SEQUENCE public.{counter}")
        cursor.execute(f"ALTER SEQUENCE public.{counter} OWNER TO lab_arena_owner")
        cursor.execute(f"""CREATE FUNCTION public.{name}() RETURNS boolean
            LANGUAGE plpgsql VOLATILE SECURITY DEFINER SET search_path=pg_catalog,public
            AS $counter$ BEGIN PERFORM nextval('public.{counter}');
            RETURN public.{name}_original(); END; $counter$""")
        cursor.execute(f"ALTER FUNCTION public.{name}() OWNER TO lab_arena_owner")


def test_canonical_retained_scores_survive_both_stages_and_publication(database, tmp_path, monkeypatch):
    monkeypatch.setattr(previous.Harness, "objects_key", lambda self: "repair296-" + tmp_path.name)
    seed = previous._seed_published_terminal
    prepare = previous._prepare
    monkeypatch.setattr(previous, "_seed_published_terminal",
        lambda connection, harness: seed(connection, harness, canonical_miner_ids=True))
    applied = []
    drive = previous._drive_cycle
    def checked_drive(service, objects, icps, runner_hotkey):
        open_scoring = service.store.open_scoring
        score_stage = service.score_stage
        def checked_score(round_id, stage):
            result = score_stage(round_id, stage)
            psycopg2, dsn = database
            with psycopg2.connect(**dsn) as connection:
                with connection.cursor() as cursor:
                    cursor.execute("SELECT public.lab_arena_sep18_published_rerun295_nonbaseline_valid_v1()")
                    assert cursor.fetchone()[0] is True, stage
            return result
        monkeypatch.setattr(service, "score_stage", checked_score)
        def checked_open(round_id, stage, items, **kwargs):
            for item in items:
                if item["submission_id"] == previous.BASELINE:
                    continue
                retained = service.store.get_run(f"{round_id}:{item['submission_id']}:{stage}:{item['icp_position']}:score:1")
                assert retained is not None
                for key in ("judgment_cache_key", "judgment_input_hash", "judgment_scope_doc"):
                    assert retained[key] == item[key], (stage, key)
            psycopg2, dsn = database
            with psycopg2.connect(**dsn) as connection:
                _reject_invalid_reuse(connection, stage, items)
                with connection.cursor() as cursor:
                    cursor.execute("ALTER SEQUENCE public.repair296_miner_seal_calls RESTART WITH 1")
                    cursor.execute("ALTER SEQUENCE public.repair296_archive_seal_calls RESTART WITH 1")
            result = open_scoring(round_id, stage, items, **kwargs)
            assert result["assignments"] == 50 and result["reused"] == 40
            baseline_scores = [row for row in service.store.list_runs(round_id, stage=stage, kind="score")
                               if row["submission_id"] == previous.BASELINE]
            assert len(baseline_scores) == 10
            assert all(row["assignment_id"].endswith(":score:rerun295") for row in baseline_scores)
            with psycopg2.connect(**dsn) as connection:
                with connection.cursor() as cursor:
                    for counter in ("repair296_miner_seal_calls", "repair296_archive_seal_calls"):
                        cursor.execute(f"SELECT last_value,is_called FROM public.{counter}")
                        assert cursor.fetchone() == (1, True), (stage, counter)
            return result
        monkeypatch.setattr(service.store, "open_scoring", checked_open)
        return drive(service, objects, icps, runner_hotkey)
    monkeypatch.setattr(previous, "_drive_cycle", checked_drive)
    def install_repair(cursor, schedule):
        result = prepare(cursor, schedule)
        if not applied:
            cursor.execute("SELECT pg_get_functiondef('public.lab_arena_open_scoring_v2(text,smallint,jsonb)'::regprocedure)")
            original = cursor.fetchone()[0]
            cursor.execute("SAVEPOINT wrong_definition")
            assert original.count("v_created INTEGER := 0;") == 1
            cursor.execute(original.replace("v_created INTEGER := 0;", "v_created INTEGER := 99;"))
            # Strip transaction markers only for this rejected local savepoint.
            with pytest.raises(Exception, match="scoring definition differs"):
                cursor.execute(MIGRATION.read_text().removeprefix("-- Reuse only sealed Sep18 rerun295 miner judgments; do not rewrite retained evidence.\nBEGIN;\n").removesuffix("COMMIT;\n"))
            cursor.execute("ROLLBACK TO SAVEPOINT wrong_definition")
            cursor.execute(MIGRATION.read_text())
            cursor.execute(MIGRATION.read_text())
            cursor.execute("SELECT encode(extensions.digest(pg_get_functiondef('public.lab_arena_open_scoring_v2(text,smallint,jsonb)'::regprocedure),'sha256'),'hex')")
            assert cursor.fetchone()[0] == "c61a984d82286e61948b321b5bf763459acbdffff1393ecff44a3c6d0cb0c174"
            cursor.execute(original)
        cursor.execute("SELECT configuration_doc FROM public.lab_arena_rounds WHERE round_id=%s", (previous.ROUND,))
        sql = _fixture_sql(cursor.fetchone()[0])
        cursor.execute(sql)
        cursor.execute(sql)
        if not applied:
            _instrument_seal_calls(cursor)
        cursor.execute("SELECT public.lab_arena_sep18_published_rerun295_nonbaseline_valid_v1()")
        assert cursor.fetchone()[0] is True
        applied.append(True)
        return result
    monkeypatch.setattr(previous, "_prepare", install_repair)
    previous.test_rerun295_full_published_transition_and_fail_closed_winner_guard(database, tmp_path)
    assert applied == [True, True, True]
    psycopg2, dsn = database
    with psycopg2.connect(**dsn) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT status,public.lab_arena_sep18_published_rerun295_nonbaseline_valid_v1() FROM public.lab_arena_rounds WHERE round_id=%s", (previous.ROUND,))
            assert cursor.fetchone() == ("published", True)
    ordinary = tmp_path / "ordinary"
    ordinary.mkdir()
    ordinary_lifecycle(lambda: psycopg2.connect(**dsn), ordinary)
