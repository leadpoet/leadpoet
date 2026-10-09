"""Publication deadline and unchanged guards on disposable PostgreSQL."""

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from tests.lab_arena import test_lab_arena_service_round as fixtures
from tests.lab_arena.baseline_scored_first_postgres_test import (
    _baseline_first_judge,
    _enable_verified_proxy_runtime,
)
from tests.lab_arena.lab_arena_pg_harness import (
    CURRENT_PROVIDER_SERVICE_MIGRATIONS,
    database_with_lab_arena_migration,
)
from tests.lab_arena.test_integrity_round import IntegrityHarness

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
MIGRATION = SCRIPTS / "437-lab-arena-publication-transition-timeout.sql"
SIGNATURE = "public.lab_arena_transition_round(text,text,text,jsonb)"


@pytest.fixture(scope="module")
def database():
    yield from database_with_lab_arena_migration(
        CURRENT_PROVIDER_SERVICE_MIGRATIONS + (
            "409-lab-arena-publication-cost-document-reuse.sql",
            "410-lab-arena-publication-transition-timeout.sql",
            "423-lab-arena-partial-publication.sql",
            "424-lab-arena-partial-baseline-stage-transition.sql",
        )
    )


def _functions(cur):
    cur.execute("""
        SELECT oid::regprocedure::text, prosrc, proowner, proacl::text,
               prosecdef, provolatile, proconfig, pg_get_functiondef(oid)
        FROM pg_proc WHERE pronamespace='public'::regnamespace ORDER BY oid
    """)
    return {row[0]: row[1:] for row in cur.fetchall()}


def test_migration_preserves_every_function_and_replays(database):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as c, c.cursor() as cur:
        before = _functions(cur)
        signature = SIGNATURE.removeprefix("public.")
        assert before[signature][-2] == ["search_path=pg_catalog, public", "statement_timeout=60s"]
        cur.execute(MIGRATION.read_text())
        after = _functions(cur)
        assert before.keys() == after.keys()
        for name in before:
            if name == signature:
                assert after[name][:-2] == before[name][:-2]
                assert after[name][-2] == ["search_path=pg_catalog, public", "statement_timeout=600s"]
                assert after[name][-1] == before[name][-1].replace("SET statement_timeout TO '60s'", "SET statement_timeout TO '600s'")
            else:
                assert after[name] == before[name]
        cur.execute(MIGRATION.read_text())
        assert _functions(cur) == after


@pytest.mark.parametrize("drift", ["source", "config", "grants", "owner", "definer"])
def test_migration_fails_closed_on_unexpected_preimage(database, drift):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as c, c.cursor() as cur:
        if drift == "source":
            cur.execute("SELECT pg_get_functiondef(%s::regprocedure)", (SIGNATURE,))
            cur.execute(cur.fetchone()[0].replace("DECLARE", "-- unexpected source\nDECLARE", 1))
        elif drift == "config":
            cur.execute("ALTER FUNCTION " + SIGNATURE + " SET statement_timeout='61s'")
        elif drift == "grants":
            cur.execute("REVOKE EXECUTE ON FUNCTION " + SIGNATURE + " FROM lab_arena_service")
        elif drift == "owner":
            cur.execute("ALTER FUNCTION " + SIGNATURE + " OWNER TO postgres")
        else:
            cur.execute("ALTER FUNCTION " + SIGNATURE + " SECURITY INVOKER")
        expected = "preimage_changed" if drift in ("source", "config") else "security_shape_changed"
        with pytest.raises(psycopg.Error, match="transition_timeout_437_" + expected):
            cur.execute(MIGRATION.read_text())
        c.rollback()


@pytest.fixture(scope="module")
def proven_round(database, tmp_path_factory):
    """Use the established signed service/broker fixture, not mocked guards."""
    psycopg, dsn = database
    connect = lambda: psycopg.connect(**dsn)
    h = IntegrityHarness(connect, tmp_path_factory.mktemp("publication437"),
                         challengers=["DeadlineMiner"], runners=["alpha"])
    h.service.config.defaults = replace(h.service.config.defaults,
        execution_sequence_from="2000-01-01T00:00:00Z", per_icp_cost_policy=True)
    _enable_verified_proxy_runtime(h)
    # Reuse the existing deterministic scorer fixture. Qualification documents
    # are produced and validated by the real completion and publication paths.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(fixtures, "deterministic_scorer", _baseline_first_judge)
        assert fixtures._start_round(h, day=28, epoch=62_428) == 2
        h.clock.advance_to(h.schedule()["stage_1_start"])
        h.advance_until("stage1_scored", runners=1)
        h.service.advance_round(h.round_id)
        h.advance_until("scored", runners=1)
        assert h.service.publish(h.round_id)["status"] == "ok"
    with connect() as c, c.cursor() as cur:
        cur.execute("SELECT to_jsonb(r) FROM public.lab_arena_rounds r WHERE round_id=%s", (h.round_id,))
        row = cur.fetchone()[0]
        assert row["status"] == "published"
        assert len(row["publication_doc"]["final_ranking"]) == 2
        tables = {}
        for table in ("lab_arena_submissions", "lab_arena_runs", "lab_arena_ledger"):
            cur.execute("SELECT to_jsonb(t) FROM public." + table + " t WHERE round_id=%s", (h.round_id,))
            tables[table] = [value[0] for value in cur.fetchall()]
    h.service.store.close()
    return row, tables


def _rewrite(value, replacements):
    text = json.dumps(value)
    for old, new in replacements:
        text = text.replace(old, new)
    return json.loads(text)


def _expand(cur, proven, count):
    """Expand proven local rows, keeping their score/cost evidence unchanged."""
    from psycopg2.extras import execute_values

    source, source_tables = proven
    round_id = "arena-2099-08-01-deadline" + str(count)
    parts = source["participants"]
    baseline = next(p for p in parts if p["is_king"])
    miner = next(p for p in parts if not p["is_king"])
    publication = deepcopy(source["publication_doc"])
    participants, final, stage1, rows = [], [], [], {table: [] for table in source_tables}
    cur.execute("SELECT COALESCE(max(entry_id),0) FROM public.lab_arena_ledger")
    entry_id = cur.fetchone()[0]
    for index in range(count):
        template = baseline if index == 0 else miner
        sid = "sub-publication437-%d-%03d" % (count, index)
        hotkey = fixtures.keypair("publication437-%d-%d" % (count, index)).ss58_address
        replacements = [(source["round_id"], round_id), (template["submission_id"], sid),
                        (template["miner_hotkey"], hotkey)]
        selected = {table: [r for r in values if r.get("submission_id") == template["submission_id"]]
                    for table, values in source_tables.items()}
        # Rewrite call identities and all embedded execution-key/local-ID forms.
        for identity in sorted({r["call_identity"] for r in selected["lab_arena_ledger"] if r.get("call_identity")}):
            digest = hashlib.sha256((sid + identity).encode()).hexdigest()
            replacements += [(identity[7:], digest), (identity[7:39], digest[:32])]
        for run in selected["lab_arena_runs"]:
            if run.get("claim_request_id"):
                replacements.append((run["claim_request_id"], sid + "-" + run["claim_request_id"]))
        participants.append(_rewrite(template, replacements))
        for ranking in publication["final_ranking"]:
            if ranking["submission_id"] == template["submission_id"]:
                final.append(_rewrite(ranking, replacements))
        for ranking in publication["stage1_ranking"]:
            if index and ranking["submission_id"] == template["submission_id"]:
                stage1.append(_rewrite(ranking, replacements))
        for table, values in selected.items():
            for value in values:
                cloned = _rewrite(value, replacements)
                if table == "lab_arena_ledger":
                    entry_id += 1
                    cloned["entry_id"] = entry_id
                rows[table].append(cloned)
    for rankings in (stage1, final):
        rankings.sort(key=lambda r: (-(r.get("final_score", r.get("stage1_score")) or 0), r["submission_id"]))
        for rank, value in enumerate(rankings, 1):
            value["rank"] = rank
    finalists = [p["submission_id"] for p in participants if not p["is_king"]]
    round_row = _rewrite(source, [(source["round_id"], round_id)])
    assert round_row["configuration_doc"]["max_challengers"] == 256
    round_row.update(status="scored", participants=participants, finalists=finalists,
                     publication_doc=None, published_at=None, king_outcome=None,
                     king_submission_id=None, king_hotkey=None, king_start_epoch=None,
                     effective_reward_epoch=None, reward_basis_hash=None)
    publication = _rewrite(publication, [(source["round_id"], round_id)])
    publication.update(participants=[dict(submission_id=p["submission_id"],
        miner_hotkey=p["miner_hotkey"], is_baseline=p["is_king"]) for p in participants],
        finalists=finalists, final_ranking=final, stage1_ranking=stage1)
    # Only local fixture insertion bypasses write-once/transition triggers.
    # Every trigger is restored before any tested transition is invoked.
    for table in ("lab_arena_rounds", *rows):
        cur.execute("ALTER TABLE public." + table + " DISABLE TRIGGER USER")
    for table, values in {"lab_arena_rounds": [round_row], **rows}.items():
        cur.execute("SELECT attname FROM pg_attribute WHERE attrelid=%s::regclass AND attnum>0 AND NOT attisdropped AND attgenerated='' ORDER BY attnum", ("public." + table,))
        columns = [item[0] for item in cur.fetchall()]
        execute_values(cur, "INSERT INTO public." + table + " (" + ",".join(columns) + ") SELECT " + ",".join("r." + name for name in columns) + " FROM (VALUES %s) v(doc) CROSS JOIN LATERAL jsonb_populate_record(NULL::public." + table + ",v.doc) r",
                       [(json.dumps(value),) for value in values], template="(%s::jsonb)", page_size=500)
    for table in ("lab_arena_rounds", *rows):
        cur.execute("ALTER TABLE public." + table + " ENABLE TRIGGER USER")
    cur.execute("SELECT count(*) FROM public.lab_arena_runs WHERE round_id=%s AND kind='execute' AND per_icp_score IS NOT NULL", (round_id,))
    assert cur.fetchone()[0] == count * 20
    return round_id, publication


@pytest.mark.parametrize("participant_count", [143, 257])
def test_large_real_publication_and_forgery_rejection(database, proven_round, participant_count):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as c, c.cursor() as cur:
        cur.execute(MIGRATION.read_text())
        cur.execute("BEGIN")
        round_id, publication = _expand(cur, proven_round, participant_count)
        # Put tampering on the first row so these controls fail without
        # repeating the full large-round validation.
        for drift in ("score", "cost", "identity"):
            forged = deepcopy(publication)
            ranking = forged["final_ranking"][0]
            if drift == "score":
                ranking["final_score"] += 1
            elif drift == "cost":
                ranking["cost_summary"]["competition_sourcing_microusd"] += 1
            else:
                ranking["submission_id"] = "sub-unrelated"
            cur.execute("SAVEPOINT publication_forgery")
            with pytest.raises(psycopg.Error):
                cur.execute("SELECT public.lab_arena_transition_round(%s,'scored','published',%s::jsonb)",
                    (round_id, json.dumps(dict(publication_doc=forged, published_at=forged["published_at"]))))
            cur.execute("ROLLBACK TO SAVEPOINT publication_forgery")
        cur.execute("SET ROLE lab_arena_service")
        cur.execute("SET LOCAL statement_timeout='600s'")
        cur.execute("SELECT public.lab_arena_transition_round(%s,'scored','published',%s::jsonb)",
            (round_id, json.dumps(dict(publication_doc=publication, published_at=publication["published_at"]))))
        assert cur.fetchone()[0]["status"] == "ok"
        cur.execute("RESET ROLE")
        cur.execute("SELECT status,publication_doc FROM public.lab_arena_rounds WHERE round_id=%s", (round_id,))
        status, actual = cur.fetchone()
        assert status == "published" and actual == publication
