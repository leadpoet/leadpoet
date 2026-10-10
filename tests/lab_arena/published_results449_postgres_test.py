"""Real PostgreSQL proof for the published-results cost projection."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from lab_arena import contact_policy, integrity, intent_details_policy, quality_policy
from lab_arena.store import ArenaStore, PsycopgTransport
from tests.lab_arena.lab_arena_pg_harness import (
    DEFAULT_MIGRATIONS, database_with_lab_arena_migration,
)
from tests.lab_arena.scoring_validator_attribution_test import (
    MINER, ROUND_ID, SUBMISSION_ID, _service,
)


MIGRATION = Path(__file__).resolve().parents[2] / "scripts/449-lab-arena-published-results-rank-view.sql"
VIEW = "public.lab_arena_published_results_v1"


@pytest.fixture(scope="module")
def database():
    yield from database_with_lab_arena_migration(DEFAULT_MIGRATIONS)


@pytest.fixture(scope="module")
def projected_database(database):
    psycopg2, dsn = database
    connection = psycopg2.connect(**dsn)
    connection.autocommit = True
    with connection.cursor() as cursor:
        cursor.execute("SELECT nspacl::text FROM pg_namespace WHERE nspname='public'")
        schema_acl = cursor.fetchone()[0]
        cursor.execute("SELECT relacl::text,relrowsecurity,relforcerowsecurity "
                       "FROM pg_class WHERE oid='public.lab_arena_rounds'::regclass")
        base_security = cursor.fetchone()
        cursor.execute(MIGRATION.read_text(encoding="utf-8"))
        cursor.execute(MIGRATION.read_text(encoding="utf-8"))
        cursor.execute("SELECT nspacl::text FROM pg_namespace WHERE nspname='public'")
        assert cursor.fetchone()[0] == schema_acl
        cursor.execute("SELECT relacl::text,relrowsecurity,relforcerowsecurity "
                       "FROM pg_class WHERE oid='public.lab_arena_rounds'::regclass")
        assert cursor.fetchone() == base_security
    connection.close()
    return psycopg2, dsn


def _insert(connection, round_id, publication, *, status="published", configuration=None):
    configuration = configuration or {"mode": "live", "network_name": "finney", "netuid": 71,
                                  "scorer_policy": {}}
    participants = [{"submission_id": "baseline", "is_king": True}]
    with connection.cursor() as cursor:
        cursor.execute("SET session_replication_role=replica")
        try:
            cursor.execute(
                "INSERT INTO public.lab_arena_rounds "
                "(round_id,status,configuration_doc,participants,publication_doc) "
                "VALUES (%s,%s,%s::jsonb,%s::jsonb,%s::jsonb)",
                (round_id, status, json.dumps(configuration), json.dumps(participants),
                 json.dumps(publication)),
            )
        finally:
            cursor.execute("SET session_replication_role=origin")


def _publication():
    return {
        "participants": [{"submission_id": SUBMISSION_ID, "miner_hotkey": MINER}],
        "stage1_ranking": [{"submission_id": SUBMISSION_ID, "stage1_score": 12.5}],
        "final_ranking": [
            {"submission_id": SUBMISSION_ID, "final_score": 15.0,
             "cost_summary": {"large": "x" * 20_000}, "extra": [None, 1]},
            {"submission_id": SUBMISSION_ID, "final_score": 99.0,
             "cost_summary": None},
            {"submission_id": "other", "final_score": None},
        ],
        "king_decision": {"outcome": "no_king"},
        "extra_publication": {"nested": [None, False]},
    }


def _expected(publication):
    result = deepcopy(publication)
    if isinstance(result, dict) and isinstance(result.get("final_ranking"), list):
        result["final_ranking"] = [
            {key: value for key, value in item.items() if key != "cost_summary"}
            if isinstance(item, dict) else item
            for item in result["final_ranking"]
        ]
    return result


def test_view_permissions_and_published_only_scope(projected_database):
    psycopg2, dsn = projected_database
    with psycopg2.connect(**dsn) as connection, connection.cursor() as cursor:
        cursor.execute("SELECT relowner::regrole::text,reloptions,relacl::text "
                       "FROM pg_class WHERE oid=%s::regclass", (VIEW,))
        owner, options, acl = cursor.fetchone()
        assert owner == "lab_arena_owner"
        assert "security_invoker=true" in options
        assert "lab_arena_service=r/lab_arena_owner" in acl
        cursor.execute("SELECT count(*) FROM pg_class c, aclexplode(c.relacl) acl "
                       "WHERE c.oid=%s::regclass AND acl.grantee=0", (VIEW,))
        assert cursor.fetchone()[0] == 0
        for role in ("anon", "authenticated", "service_role", "lab_arena_service"):
            cursor.execute("SELECT has_table_privilege(%s,%s,'SELECT')", (role, VIEW))
            assert cursor.fetchone()[0] is (role == "lab_arena_service")
        _insert(connection, "arena-2026-10-10-v449open", _publication(), status="open")
        cursor.execute("SET ROLE lab_arena_service")
        cursor.execute("SELECT round_id FROM public.lab_arena_published_results_v1 "
                       "WHERE round_id='arena-2026-10-10-v449open'")
        assert cursor.fetchall() == []


def test_view_only_drops_nested_cost_and_preserves_order_and_malformed_values(projected_database):
    psycopg2, dsn = projected_database
    variants = [
        _publication(), None, False, [], {},
        {"final_ranking": None}, {"final_ranking": {}},
        {"final_ranking": []},
        {"final_ranking": [None, False, [], "legacy", {"cost_summary": [1]}]},
    ]
    with psycopg2.connect(**dsn) as connection:
        connection.autocommit = True
        for index, publication in enumerate(variants):
            _insert(connection, "arena-2026-10-10-v449%02d" % index, publication)
    transport = PsycopgTransport(lambda: psycopg2.connect(**dsn))
    try:
        store = ArenaStore(transport)
        for index, publication in enumerate(variants):
            round_id = "arena-2026-10-10-v449%02d" % index
            full = store.get_round(round_id)
            projected = store.get_published_results_round(round_id)
            assert full["publication_doc"] == publication
            assert projected["publication_doc"] == _expected(publication)
            assert projected["participants"] == full["participants"]
            assert projected["benchmark_ref"] == full["benchmark_ref"]
            assert projected["cfg_mode"] == json.dumps("live")
        assert len(json.dumps(store.get_published_results_round(
            "arena-2026-10-10-v44900"))) < len(json.dumps(store.get_round(
                "arena-2026-10-10-v44900"))) / 10
        assert store.get_published_results_round("arena-2026-10-10-v449open") is None
        assert store.get_round("arena-2026-10-10-v449open")["status"] == "open"
    finally:
        transport.close()


def test_long_published_ranking_matches_live_shape_without_stored_change(projected_database):
    psycopg2, dsn = projected_database
    ranking = [
        {"submission_id": "submission-%03d" % index, "final_score": index / 10,
         "rank": index + 1, "cost_summary": {"metering": "m" * 5_000},
         "other": {"cost_summary": "keep"}}
        for index in range(143)
    ]
    publication = {
        "participants": [{"submission_id": entry["submission_id"]} for entry in ranking],
        "stage1_ranking": [{"submission_id": entry["submission_id"], "stage1_score": 1.0}
                           for entry in ranking[:91]],
        "final_ranking": ranking,
        "cost_summary": {"top_level": "keep"},
    }
    round_id = "arena-2026-10-10-v449large"
    with psycopg2.connect(**dsn) as connection:
        connection.autocommit = True
        _insert(connection, round_id, publication)
    transport = PsycopgTransport(lambda: psycopg2.connect(**dsn))
    try:
        store = ArenaStore(transport)
        source = store.get_round(round_id)["publication_doc"]
        projected = store.get_published_results_round(round_id)["publication_doc"]
        assert source == publication
        assert projected == _expected(publication)
        assert [entry["submission_id"] for entry in projected["final_ranking"]] == [
            entry["submission_id"] for entry in ranking]
        assert len(json.dumps(projected)) < len(json.dumps(source)) / 10
    finally:
        transport.close()


@pytest.mark.parametrize("mode,policies", [
    ("live", {}), ("shadow", {}),
    ("live", {"integrity_policy": integrity.POLICY,
              "scorer_policy": {"scoring_adapter_version": integrity.SCORING_ADAPTER}}),
    ("live", {"integrity_policy": integrity.POLICY,
              "contact_policy": contact_policy.POLICY,
              "company_quality_policy": quality_policy.POLICY,
              "intent_details_policy": intent_details_policy.POLICY,
              "scorer_policy": {"scoring_adapter_version": contact_policy.SCORING_ADAPTER,
                                "company_quality_policy": quality_policy.POLICY,
                                "intent_details_policy": intent_details_policy.POLICY}}),
])
def test_actual_view_keeps_public_result_body(projected_database, mode, policies):
    psycopg2, dsn = projected_database
    suffix = "shadow" if mode == "shadow" else "policy%d" % len(policies)
    round_id = "arena-2026-10-10-v449" + suffix
    configuration = {"mode": mode, "network_name": "finney", "netuid": 71,
                     "scorer_policy": {}, **policies}
    with psycopg2.connect(**dsn) as connection:
        connection.autocommit = True
        _insert(connection, round_id, _publication(), configuration=configuration)
    transport = PsycopgTransport(lambda: psycopg2.connect(**dsn))
    try:
        store = ArenaStore(transport)
        full_row = store.get_round(round_id)
        expected = _service([], public_positions=None)
        candidate = _service([], public_positions=None)
        for service in (expected, candidate):
            service._config = SimpleNamespace(mode=mode, pinned_round_id=None,
                                              network_name="finney", netuid=71)
            service._chain_scope = lambda: ("finney", 71)
            service._round = lambda _round_id: full_row
            service._store.list_runs = lambda *_args, **_kwargs: []
        candidate._store.get_published_results_round = store.get_published_results_round
        candidate._round = lambda _round_id: pytest.fail("published result fetched full row")
        assert candidate.public_results(round_id, SUBMISSION_ID) == expected.public_results(
            round_id, SUBMISSION_ID)
    finally:
        transport.close()


@pytest.mark.parametrize("final_ranking", [
    [],
    [None],
    {"legacy": "malformed"},
])
def test_missing_and_malformed_rankings_keep_result_behavior(projected_database, final_ranking):
    psycopg2, dsn = projected_database
    round_id = "arena-2026-10-10-v449shape%d" % len(json.dumps(final_ranking))
    publication = _publication()
    publication["final_ranking"] = final_ranking
    with psycopg2.connect(**dsn) as connection:
        connection.autocommit = True
        _insert(connection, round_id, publication)
    transport = PsycopgTransport(lambda: psycopg2.connect(**dsn))
    try:
        store = ArenaStore(transport)
        full_row = store.get_round(round_id)
        outcomes = []
        for projected in (False, True):
            service = _service([], public_positions=None)
            service._config = SimpleNamespace(mode="live", pinned_round_id=None,
                                              network_name="finney", netuid=71)
            service._chain_scope = lambda: ("finney", 71)
            service._store.list_runs = lambda *_args, **_kwargs: []
            if projected:
                service._store.get_published_results_round = store.get_published_results_round
                service._round = lambda _round_id: pytest.fail("published result fetched full row")
            else:
                service._round = lambda _round_id: full_row
            try:
                outcomes.append(("result", service.public_results(round_id, SUBMISSION_ID)))
            except Exception as exc:
                outcomes.append((type(exc), str(exc)))
        assert outcomes[0] == outcomes[1]
    finally:
        transport.close()
