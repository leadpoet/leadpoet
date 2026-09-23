"""Exact September 24 contact retirement, with frozen state preserved."""
from copy import deepcopy
import json
from pathlib import Path
import pytest
from lab_arena import contracts, scoring
from tests.lab_arena.lab_arena_pg_harness import CURRENT_SERVICE_MIGRATIONS, database_with_lab_arena_migration
from tests.lab_arena.test_lab_arena_contracts import base_round_configuration

MIGRATION = Path(__file__).resolve().parents[2] / "scripts/357-arena-2026-09-24-company-only.sql"

@pytest.fixture(scope="module")
def database():
    yield from database_with_lab_arena_migration(CURRENT_SERVICE_MIGRATIONS)

def seed(cursor, day="24", status="open"):
    cfg = base_round_configuration()
    cfg.update(round_id=f"arena-2026-09-{day}", mode="live", network_name="finney", netuid=71,
               integrity_policy="arena_integrity_v1", contact_policy="contacts_v1",
               intent_details_policy="intent_details_v1", cost_per_company_microusd=800000,
               stage_1_icp_count=5, stage_2_icp_count=5, promotion_margin=0.5)
    cfg["schedule"] = {k:v.replace("2026-09-01",f"2026-09-{int(day)-1:02d}").replace("2026-09-02",f"2026-09-{day}") for k,v in cfg["schedule"].items()}
    cfg["scorer_policy"] = scoring.build_scorer_policy(scoring_adapter_version="qualification_contacts_v3", intent_details=True)
    cfg = contracts.validate_round_configuration(cfg)
    cursor.execute("SET session_replication_role=replica")
    cursor.execute("INSERT INTO public.lab_arena_rounds (round_id,status,configuration_doc,rewards_enabled) VALUES (%s,%s,%s::jsonb,false)", (cfg["round_id"],status,json.dumps(cfg)))
    cursor.execute("SET session_replication_role=origin")
    return cfg

def row(cursor, day="24"):
    cursor.execute("SELECT to_jsonb(r) FROM public.lab_arena_rounds r WHERE round_id=%s",(f"arena-2026-09-{day}",))
    return cursor.fetchone()[0]

def test_exact_migration_idempotent_and_preserves_company_bank_and_history(database):
    psycopg2,dsn=database
    with psycopg2.connect(**dsn) as conn,conn.cursor()as cur:
        seed(cur);seed(cur,"23","published")
        cur.execute("INSERT INTO public.qualification_private_icp_sets (set_id,icps) VALUES (%s,%s::jsonb)", (20260923,json.dumps([{"prompt":"Company criteria","target_roles":["CFO"]}])))
        before=row(cur);history=row(cur,"23")
        cur.execute(MIGRATION.read_text())
        after=row(cur)
        expected=deepcopy(before["configuration_doc"]);expected.pop("contact_policy");expected["scorer_policy"]["scoring_adapter_version"]="qualification_integrity_v2"
        assert after["configuration_doc"]==expected
        assert contracts.validate_round_configuration(expected)==expected
        assert {k:v for k,v in after.items()if k not in {"configuration_doc","updated_at"}}=={k:v for k,v in before.items()if k not in {"configuration_doc","updated_at"}}
        assert row(cur,"23")==history
        cur.execute("SELECT icps FROM public.qualification_private_icp_sets WHERE set_id=20260923")
        assert cur.fetchone()[0]==[{"prompt":"Company criteria","target_roles":["CFO"]}]
        cur.execute(MIGRATION.read_text());assert row(cur)==after
        cur.execute("SELECT tgenabled FROM pg_trigger WHERE tgrelid='public.lab_arena_rounds'::regclass AND tgname='lab_arena_rounds_write_once'")
        assert cur.fetchone()[0]=="O"
        cur.execute("SET session_replication_role=replica")
        cur.execute("DELETE FROM public.lab_arena_rounds WHERE round_id IN ('arena-2026-09-23','arena-2026-09-24')")
        cur.execute("DELETE FROM public.qualification_private_icp_sets WHERE set_id=20260923")
        cur.execute("SET session_replication_role=origin")

@pytest.mark.parametrize("status",["committed","stage1","published"])
def test_migration_refuses_started_or_published_contact_round(database,status):
    psycopg2,dsn=database
    conn=psycopg2.connect(**dsn)
    try:
        with conn.cursor()as cur:
            seed(cur,status=status);conn.commit()
            before=row(cur)
            with pytest.raises(psycopg2.Error,match="open, unfrozen, unstarted"):
                cur.execute(MIGRATION.read_text())
            conn.rollback();assert row(cur)==before
            cur.execute("SET session_replication_role=replica")
            cur.execute("DELETE FROM public.lab_arena_rounds WHERE round_id='arena-2026-09-24'")
            cur.execute("SET session_replication_role=origin");conn.commit()
    finally:conn.close()
