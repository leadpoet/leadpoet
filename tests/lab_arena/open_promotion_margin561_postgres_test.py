"""The one-time margin correction cannot change committed Arena history."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from tests.lab_arena.lab_arena_pg_harness import (
    CURRENT_SERVICE_MIGRATIONS,
    database_with_lab_arena_migration,
)
from tests.lab_arena.test_lab_arena_contracts import base_round_configuration
from tests.lab_arena.test_lab_arena_migration_postgres import hotkey


SQL = Path(__file__).resolve().parents[2] / "scripts/561-lab-arena-open-promotion-margin.sql"


@pytest.fixture(scope="module")
def database():
    yield from database_with_lab_arena_migration(CURRENT_SERVICE_MIGRATIONS)


def _configuration(round_id, *, margin=0.5, mode="live", network="finney", cutoff="2099-10-24T12:00:00Z"):
    config = base_round_configuration()
    config.update(
        round_id=round_id,
        mode=mode,
        rewards_enabled=mode == "live",
        network_name=network,
        netuid=71,
        promotion_margin=margin,
    )
    config["schedule"] = dict(config["schedule"], submission_cutoff=cutoff)
    return config


def _rounds(cursor):
    cursor.execute("SELECT round_id,to_jsonb(r) FROM public.lab_arena_rounds r ORDER BY round_id")
    return dict(cursor.fetchall())


def _trigger_state(cursor):
    cursor.execute(
        "SELECT tgname,tgenabled FROM pg_trigger "
        "WHERE tgrelid='public.lab_arena_rounds'::regclass AND NOT tgisinternal "
        "ORDER BY tgname"
    )
    return cursor.fetchall()


def test_open_live_round_margin_changes_without_touching_history_or_submission(database):
    psycopg2, dsn = database
    cases = {
        "eligible": ("open", {}),
        "eligibleempty": ("open", {"participants": []}),
        "committed": ("committed", {}),
        "published": ("published", {"publication_doc": {"final_ranking": []}}),
        "cutoff": ("open", {"cutoff": "2000-01-01T00:00:00Z"}),
        "shadow": ("open", {"mode": "shadow"}),
        "testnet": ("open", {"network": "testnet"}),
        "participants": ("open", {"participants": [{"submission_id": "already-selected"}]}),
        "execution": ("open", {}),
        "frozen": ("open", {}),
        "bankready": ("open", {"benchmark_ref": "benchmarks/already-prepared.json"}),
        "reward": ("open", {"reward_basis_hash": "sha256:" + "a" * 64}),
        "rewardactive": ("open", {"reward_activated_at": "2026-10-24T00:00:00Z"}),
        "promoted": ("open", {"baseline_promoted_at": "2026-10-24T00:00:00Z"}),
        "promotion": ("open", {"promotion_doc": {"started": True}}),
        "custom": ("open", {"margin": 1.25}),
    }
    ids = {name: "arena-2026-10-24-" + name for name in cases}
    with psycopg2.connect(**dsn) as connection, connection.cursor() as cursor:
        cursor.execute("SET LOCAL session_replication_role=replica")
        for name, (status, options) in cases.items():
            config = _configuration(
                ids[name],
                margin=options.get("margin", 0.5),
                mode=options.get("mode", "live"),
                network=options.get("network", "finney"),
                cutoff=options.get("cutoff", "2099-10-24T12:00:00Z"),
            )
            cursor.execute(
                "INSERT INTO public.lab_arena_rounds "
                "(round_id,status,configuration_doc,rewards_enabled,participants,publication_doc,reward_basis_hash) "
                "VALUES (%s,%s,%s::jsonb,%s,%s::jsonb,%s::jsonb,%s)",
                (
                    ids[name], status, json.dumps(config), config["rewards_enabled"],
                    (json.dumps(options["participants"])
                     if "participants" in options else None),
                    (json.dumps(options["publication_doc"])
                     if "publication_doc" in options else None),
                    options.get("reward_basis_hash"),
                ),
            )
            if "reward_activated_at" in options:
                cursor.execute(
                    "UPDATE public.lab_arena_rounds SET reward_activated_at=%s WHERE round_id=%s",
                    (options["reward_activated_at"], ids[name]),
                )
            if "baseline_promoted_at" in options:
                cursor.execute(
                    "UPDATE public.lab_arena_rounds SET baseline_promoted_at=%s WHERE round_id=%s",
                    (options["baseline_promoted_at"], ids[name]),
                )
            if "promotion_doc" in options:
                cursor.execute(
                    "UPDATE public.lab_arena_rounds SET promotion_doc=%s::jsonb WHERE round_id=%s",
                    (json.dumps(options["promotion_doc"]), ids[name]),
                )
            if "benchmark_ref" in options:
                cursor.execute(
                    "UPDATE public.lab_arena_rounds SET benchmark_ref=%s WHERE round_id=%s",
                    (options["benchmark_ref"], ids[name]),
                )
        for name in ("eligible", "execution"):
            cursor.execute(
                "INSERT INTO public.lab_arena_submissions "
                "(submission_id,round_id,miner_hotkey,status,submission_doc) "
                "VALUES (%s,%s,%s,'accepted','{\"preserve\":true}'::jsonb)",
                ("margin-" + name, ids[name], hotkey("margin-" + name)),
            )
        cursor.execute(
            "INSERT INTO public.lab_arena_submissions "
            "(submission_id,round_id,miner_hotkey,status) VALUES (%s,%s,%s,'frozen')",
            ("margin-frozen", ids["frozen"], hotkey("margin-frozen")),
        )
        cursor.execute(
            "INSERT INTO public.lab_arena_runs "
            "(run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,icp_position,attempt,status) "
            "VALUES (%s,%s,%s,%s,%s,1,0,1,'pending')",
            ("margin-run", "margin-assignment", ids["execution"], "margin-execution", hotkey("margin-execution")),
        )
        cursor.execute("SET LOCAL session_replication_role=origin")
        before = _rounds(cursor)
        trigger_before = _trigger_state(cursor)
        assert before[ids["eligible"]]["participants"] is None
        assert before[ids["eligible"]]["benchmark_ref"] is None
        assert before[ids["eligible"]]["promotion_doc"] is None
        assert before[ids["eligible"]]["reward_basis_doc"] is None
        assert before[ids["eligible"]]["promotion_required"] is True
        cursor.execute("SELECT configuration_doc -> 'promotion_margin' = '0.5'::jsonb, "
                       "(configuration_doc #>> '{schedule,submission_cutoff}')::timestamptz > clock_timestamp() "
                       "FROM public.lab_arena_rounds WHERE round_id=%s", (ids["eligible"],))
        assert cursor.fetchone() == (True, True)
        cursor.execute(
            "SELECT to_jsonb(s) FROM public.lab_arena_submissions s "
            "WHERE round_id=%s", (ids["eligible"],)
        )
        accepted_before = cursor.fetchone()[0]
        cursor.execute("SELECT to_jsonb(a) FROM public.lab_arena_runs a WHERE round_id=%s", (ids["execution"],))
        execution_before = cursor.fetchone()[0]

        for _ in range(2):
            cursor.execute(SQL.read_text(encoding="utf-8"))
            after = _rounds(cursor)
            expected = copy.deepcopy(before)
            for eligible in ("eligible", "eligibleempty"):
                expected[ids[eligible]]["configuration_doc"]["promotion_margin"] = 8
                # The database may refresh updated_at on authorized rows.
                after[ids[eligible]]["updated_at"] = expected[ids[eligible]]["updated_at"]
            assert after == expected
            assert _trigger_state(cursor) == trigger_before
            cursor.execute(
                "SELECT to_jsonb(s) FROM public.lab_arena_submissions s "
                "WHERE round_id=%s", (ids["eligible"],)
            )
            assert cursor.fetchone()[0] == accepted_before
            cursor.execute("SELECT to_jsonb(a) FROM public.lab_arena_runs a WHERE round_id=%s", (ids["execution"],))
            assert cursor.fetchone()[0] == execution_before
