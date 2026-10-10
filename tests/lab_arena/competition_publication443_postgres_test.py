"""Real database controls for the competition-only publication projection."""

from copy import deepcopy
import json
from pathlib import Path
import time

import pytest

from lab_arena import contracts, public_dashboard as dashboard
from lab_arena.store import ArenaStore, PsycopgTransport
from tests.lab_arena.competition_summary_projection_test import _row, _service
from tests.lab_arena.lab_arena_pg_harness import (
    CURRENT_SERVICE_MIGRATIONS, database_with_lab_arena_migration,
)
from tests.lab_arena.test_lab_arena_postgrest_route import (
    DOCKER, make_transport, stack,
)


MIGRATION = Path(__file__).resolve().parents[2] / "scripts/443-lab-arena-competition-publication-projection.sql"
SIGNATURE = "public.lab_arena_competition_publication_v1(public.lab_arena_rounds)"


def _security_state(cursor):
    cursor.execute("SELECT nspacl::text FROM pg_namespace WHERE nspname='public'")
    schema = cursor.fetchone()
    cursor.execute("SELECT c.relname,c.relkind,c.relacl::text,c.relrowsecurity,c.relforcerowsecurity "
                   "FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace "
                   "WHERE n.nspname='public' ORDER BY c.relname")
    relations = cursor.fetchall()
    cursor.execute("SELECT schemaname,tablename,policyname,permissive,roles,cmd,qual,with_check "
                   "FROM pg_policies WHERE schemaname='public' ORDER BY tablename,policyname")
    return schema, relations, cursor.fetchall()


@pytest.fixture(scope="module")
def database():
    yield from database_with_lab_arena_migration(CURRENT_SERVICE_MIGRATIONS)


@pytest.fixture(scope="module")
def projected_database(database):
    psycopg2, dsn = database
    connection = psycopg2.connect(**dsn)
    connection.autocommit = True
    with connection.cursor() as cursor:
        before = _security_state(cursor)
        cursor.execute(MIGRATION.read_text(encoding="utf-8"))
        assert _security_state(cursor) == before
    connection.close()
    return psycopg2, dsn, before


def _project(cursor, row):
    cursor.execute("SELECT public.lab_arena_competition_publication_v1("
                   "pg_catalog.jsonb_populate_record(NULL::public.lab_arena_rounds,%s::jsonb))",
                   (json.dumps(row),))
    return {**row, "publication_doc": cursor.fetchone()[0]}


def _cost(policy):
    keys = dashboard._COST_COUNTER_KEYS
    if policy in (contracts.SUCCESSFUL_CALLS_COST_POLICY, contracts.PER_ICP_SUCCESSFUL_CALLS_COST_POLICY):
        keys += dashboard._SUCCESSFUL_CALL_COST_COUNTER_KEYS
    bucket = {key: 0 for key in keys}
    bucket["providers"] = [{"provider": "openrouter", **{key: 0 for key in keys}}]
    summary = {
        "returned_company_count": 0, "qualified_company_count": 0,
        "execution_cap_microusd": 100, "cost_per_company_cap_microusd": 100,
        "eligibility_cap_microusd": 0, "execution": bucket, "judge": deepcopy(bucket),
        "private_unused_cost_detail": "retain this entire selected object",
    }
    if policy == contracts.PER_ICP_SUCCESSFUL_CALLS_COST_POLICY:
        summary.update(eligible_icp_count=2, competition_sourcing_microusd=0,
                       execution_icp_cap_microusd=100, per_icp=[{
                           "icp_position": position, "returned_company_count": 0,
                           "qualified_company_count": 0, "competition_sourcing_microusd": 0,
                           "eligibility_cap_microusd": 0, "eligible": True,
                           "eligibility_reason": "eligible",
                       } for position in range(2)])
    return summary


@pytest.mark.parametrize("policy", [None, contracts.SUCCESSFUL_CALLS_COST_POLICY,
                                   contracts.PER_ICP_SUCCESSFUL_CALLS_COST_POLICY])
@pytest.mark.parametrize("outcome", ["crowned", "defended", "retained_ineligible", "no_king"])
def test_canonical_projection_preserves_complete_summary_and_selected_costs(projected_database, policy, outcome):
    psycopg2, dsn, _ = projected_database
    row = _row(stage_1_icp_count=1, stage_2_icp_count=1,
               sourcing_cost_eligibility_policy=policy)
    publication = row["publication_doc"]
    publication.update(stage1_ranking=[{"private": "x" * 10000}], finalists=["private"], extra="unused")
    # Legacy boolean fallback, explicit false override, multiple baselines and
    # duplicate participant identities all keep the existing Python order.
    publication["participants"][2].pop("is_baseline")
    publication["participants"][2]["is_king"] = True
    publication["participants"][3].update(is_baseline=False, is_king=True)
    publication["participants"].append(deepcopy(publication["participants"][0]))
    publication["king_decision"] = {
        "outcome": outcome, "king_submission_id": "model-1",
        "winner_submission_id": "model-1" if outcome == "crowned" else None,
        "extra_decision_evidence": [1, None],
    }
    for ranking in publication["final_ranking"]:
        ranking.update(eligibility_reason="eligible", cost_summary=_cost(policy))
    # Last duplicate ranking wins; projection must keep both, in input order.
    publication["final_ranking"].insert(0, {**deepcopy(publication["final_ranking"][1]), "final_score": 97})
    expected = dashboard.round_summary(row)
    with psycopg2.connect(**dsn) as connection, connection.cursor() as cursor:
        cursor.execute("SET ROLE lab_arena_service")
        projected = _project(cursor, row)
    actual = projected["publication_doc"]
    assert dashboard.round_summary(projected) == expected
    assert actual["participants"] == publication["participants"]
    assert actual["king_decision"] == publication["king_decision"]
    ids = {"model-0", "model-2"} | ({"model-1"} if outcome != "no_king" else set())
    selected = [ranking for ranking in publication["final_ranking"] if ranking["submission_id"] in ids]
    assert actual["final_ranking"] == selected
    assert set(actual) == {"participants", "king_decision", "final_ranking"}
    assert expected["participant_count"] == 146
    assert len(json.dumps(actual)) < len(json.dumps(publication)) / 4


def test_legacy_shapes_null_and_missing_preserve_original_publication(projected_database):
    psycopg2, dsn, _ = projected_database
    rows = []
    for value in (None, {}, [], False, "legacy"):
        rows.append({**_row(), "publication_doc": value})
    for key in ("participants", "king_decision", "final_ranking"):
        for value in (None, {}, False, "legacy"):
            row = _row()
            row["publication_doc"][key] = value
            rows.append(row)
        row = _row()
        del row["publication_doc"][key]
        rows.append(row)
    for field, value in (("submission_id", 123), ("submission_id", None),
                         ("submission_id", ""), ("is_baseline", 1), ("is_baseline", None),
                         ("is_king", "false")):
        row = _row()
        row["publication_doc"]["participants"][7][field] = value
        rows.append(row)
    for field, value in (("submission_id", ["model-7"]), ("eligible", 1),
                         ("eligibility_reason", []), ("cost_summary", [])):
        row = _row()
        row["publication_doc"]["final_ranking"][7][field] = value
        rows.append(row)
    for field, value in (("outcome", "legacy"), ("outcome", None),
                         ("winner_submission_id", 7), ("winner_submission_id", None),
                         ("king_submission_id", [])):
        row = _row()
        row["publication_doc"]["king_decision"][field] = value
        rows.append(row)
    for outcome in ("crowned", "defended", "retained_ineligible"):
        row = _row()
        row["publication_doc"]["king_decision"] = {"outcome": outcome}
        rows.append(row)
    row = _row()
    row["publication_doc"]["participants"].append("legacy")
    rows.append(row)
    row = _row()
    row["publication_doc"]["final_ranking"].append("legacy")
    rows.append(row)
    for status in ("open", "cancelled", "stage2_scoring"):
        rows.append({**_row(), "status": status})
    with psycopg2.connect(**dsn) as connection, connection.cursor() as cursor:
        for row in rows:
            assert _project(cursor, row)["publication_doc"] == row["publication_doc"]
        cursor.execute("SELECT public.lab_arena_competition_publication_v1(NULL::public.lab_arena_rounds)")
        assert cursor.fetchone()[0] is None


@pytest.mark.parametrize("per_icp", [False, True])
def test_unselected_malformed_cost_reason_keeps_original_failure(projected_database, per_icp):
    psycopg2, dsn, _ = projected_database
    row = _row(stage_1_icp_count=1, stage_2_icp_count=1,
               sourcing_cost_eligibility_policy=contracts.PER_ICP_SUCCESSFUL_CALLS_COST_POLICY)
    ranking = row["publication_doc"]["final_ranking"][7]
    if per_icp:
        ranking.update(eligibility_reason="eligible", cost_summary=_cost(contracts.PER_ICP_SUCCESSFUL_CALLS_COST_POLICY))
        ranking["cost_summary"]["per_icp"][0]["eligibility_reason"] = []
    else:
        ranking["eligibility_reason"] = []
    with psycopg2.connect(**dsn) as connection, connection.cursor() as cursor:
        projected = _project(cursor, row)
    assert projected["publication_doc"] == row["publication_doc"]
    with pytest.raises(TypeError) as original:
        dashboard.round_summary(row)
    with pytest.raises(TypeError, match=str(original.value)):
        dashboard.round_summary(projected)


def test_projection_metadata_acl_and_idempotent_migration(projected_database):
    psycopg2, dsn, before = projected_database
    connection = psycopg2.connect(**dsn)
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            cursor.execute(MIGRATION.read_text(encoding="utf-8"))
            assert _security_state(cursor) == before
            cursor.execute("SELECT p.prosecdef,p.provolatile,p.proowner::regrole::text,p.proconfig "
                           "FROM pg_proc p WHERE p.oid=%s::regprocedure", (SIGNATURE,))
            assert cursor.fetchone() == (False, "i", "lab_arena_owner", ["search_path=pg_catalog, public"])
            for role in ("anon", "authenticated", "service_role", "lab_arena_service"):
                cursor.execute("SELECT has_function_privilege(%s,%s,'EXECUTE')", (role, SIGNATURE))
                assert cursor.fetchone()[0] is (role == "lab_arena_service")
            cursor.execute("SELECT count(*) FROM pg_proc p, aclexplode(p.proacl) acl "
                           "WHERE p.oid=%s::regprocedure AND acl.grantee=0", (SIGNATURE,))
            assert cursor.fetchone()[0] == 0
            # A pre-existing owner schema grant is preserved too.
            cursor.execute("GRANT CREATE ON SCHEMA public TO lab_arena_owner")
            granted = _security_state(cursor)
            cursor.execute(MIGRATION.read_text(encoding="utf-8"))
            assert _security_state(cursor) == granted
            cursor.execute("REVOKE CREATE ON SCHEMA public FROM lab_arena_owner")
            assert _security_state(cursor) == before
    finally:
        connection.close()


def _insert_rows(connection, rows):
    # Synthetic published fixtures bypass publication guards only in this
    # disposable database. Production migrations never touch table data.
    with connection.cursor() as cursor:
        cursor.execute("SET session_replication_role=replica")
        try:
            for row in rows:
                cursor.execute("INSERT INTO public.lab_arena_rounds "
                               "(round_id,status,configuration_doc,participants,publication_doc,created_at,"
                               "published_at,cancel_reason,promotion_required,baseline_promoted_at,"
                               "evaluation_date,icp_set_date) VALUES "
                               "(%s,%s,%s::jsonb,%s::jsonb,%s::jsonb,%s,%s,%s,%s,%s,%s,%s)",
                               (row["round_id"], row["status"], json.dumps(row["configuration_doc"]),
                                json.dumps(row["participants"]), json.dumps(row["publication_doc"]),
                                row["created_at"], row["published_at"], row["cancel_reason"],
                                row["promotion_required"], row["baseline_promoted_at"],
                                row["evaluation_date"], row["icp_set_date"]))
        finally:
            cursor.execute("SET session_replication_role=origin")


def _response_controls(store, rows, monkeypatch):
    service, _ = _service(rows)
    service._store = store
    columns = dashboard._COMPETITION_ROUND_COLUMNS
    for limit in (1, 2, 30):
        monkeypatch.setattr(dashboard, "_COMPETITION_ROUND_COLUMNS", dashboard._ROUND_COLUMNS)
        expected = dashboard.competition_snapshot(service, limit=limit)
        monkeypatch.setattr(dashboard, "_COMPETITION_ROUND_COLUMNS", columns)
        assert dashboard.competition_snapshot(service, limit=limit) == expected
    assert expected["rounds"][-1]["promotion_status"] == "superseded"
    # Generic reads, results, miner history and pinned reads retain full docs.
    full = store.get_round(rows[-1]["round_id"])
    assert len(full["publication_doc"]["final_ranking"]) == 145
    assert store.list_rounds(limit=30)[-1]["publication_doc"] == rows[-1]["publication_doc"]
    service._config.pinned_round_id = rows[-1]["round_id"]
    assert service.public_competition()["latest_round"]["round_id"] == rows[-1]["round_id"]


def _history_rows():
    archived = _row("2026-10-12")
    archived.update(status="cancelled", cancel_reason="authorized_oct09_evidence_archive441")
    opened = _row("2026-10-11")
    opened.update(status="open", publication_doc=None)
    return [archived, opened, _row("2026-10-10"), _row("2026-10-09")]


def test_actual_psycopg_query_and_full_competition_response(projected_database, monkeypatch):
    psycopg2, dsn, _ = projected_database
    rows = _history_rows()
    connection = psycopg2.connect(**dsn)
    connection.autocommit = True
    _insert_rows(connection, rows)
    connection.close()
    transport = PsycopgTransport(lambda: psycopg2.connect(**dsn))
    try:
        store = ArenaStore(transport)
        projected = store.list_rounds(limit=30, columns=dashboard._COMPETITION_ROUND_COLUMNS)
        assert len(projected[-1]["publication_doc"]["final_ranking"]) == 2
        assert len(json.dumps(projected)) < len(json.dumps(store.list_rounds(limit=30))) / 2
        _response_controls(store, rows, monkeypatch)
    finally:
        transport.close()


@pytest.mark.skipif(DOCKER is None, reason="Docker is unavailable")
def test_actual_postgrest_computed_selection_and_response(stack, monkeypatch):
    rows = _history_rows()
    with stack["connection"].cursor() as cursor:
        cursor.execute(MIGRATION.read_text(encoding="utf-8"))
    _insert_rows(stack["connection"], rows)
    transport = make_transport(stack, "lab_arena_service")
    original_get = transport._client.get
    transport._client.get = lambda url, **kwargs: original_get(url.replace("/rest/v1", ""), **kwargs)
    try:
        store = ArenaStore(transport)
        # NOTIFY reload is asynchronous; poll only this bounded schema read.
        from lab_arena.store import ArenaStoreError
        deadline = time.monotonic() + 10
        while True:
            try:
                projected = store.list_rounds(limit=30, columns=dashboard._COMPETITION_ROUND_COLUMNS)
                break
            except ArenaStoreError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.1)
        assert len(projected[-1]["publication_doc"]["final_ranking"]) == 2
        _response_controls(store, rows, monkeypatch)
    finally:
        transport.close()
