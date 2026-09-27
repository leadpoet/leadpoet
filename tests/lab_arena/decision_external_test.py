"""Decision capture uses the same lease-only API for both validator identities."""
import json
import os
import secrets

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from gateway.api import arena_proxy
from lab_arena import trajectory
from lab_arena.api import create_app
from lab_arena.runner import HttpArenaApiClient
from lab_arena.service import ArenaService, ServiceConfig
from lab_arena.store import ArenaStore, PsycopgTransport, hash_lease_token
from tests.lab_arena.trajectory_postgres_test import database
from tests.lab_arena.test_lab_arena_service_round import keypair


def test_decisions_both_validator_identities_and_models_through_public_proxy(database, monkeypatch):
    psycopg2, dsn = database
    for name in list(os.environ):
        if "SUPABASE" in name or name in {"DATABASE_URL", "PGPASSWORD", "LAB_ARENA_SERVICE_KEY"} or (any(p in name for p in ("DEEPLINE", "OPENROUTER", "SCRAPINGDOG")) and any(s in name for s in ("KEY", "TOKEN", "SECRET"))):
            monkeypatch.delenv(name)
    monkeypatch.setenv("LAB_ARENA_MODE", "live")
    round_id = "arena-2099-01-02-parity"
    keys = [keypair("trajectory-primary-parity").ss58_address, keypair("trajectory-unlisted-external-parity").ss58_address]
    cells = []
    with psycopg2.connect(**dsn) as con, con.cursor() as c:
        config = {"schema_version":"leadpoet.lab_arena.round_configuration.v1","round_id":round_id,"mode":"live","schedule":{"submission_cutoff":"2099-01-02T00:00:00Z"},"runner_hotkeys":[keys[0]]}
        c.execute("INSERT INTO public.lab_arena_rounds(round_id,status,status_generation,stage_generation,configuration_doc) VALUES (%s,'stage1',1,1,%s::jsonb)",(round_id,json.dumps(config)))
        for role in ("baseline", "miner"):
            sub = "parity-" + role
            miner = keypair("trajectory-model-"+role).ss58_address
            c.execute("INSERT INTO public.lab_arena_submissions(submission_id,round_id,miner_hotkey,status,is_king) VALUES (%s,%s,%s,'frozen',%s)",(sub,round_id,miner,role=="baseline"))
            for index, hotkey in enumerate(keys):
                run_id = "parity-%s-%d" % (role,index)
                lease = secrets.token_hex(32)
                c.execute("INSERT INTO public.lab_arena_runs(run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,icp_position,attempt,kind,status,runner_hotkey,lease_token_hash,lease_generation,stage_generation,lease_expires_at) VALUES (%s,%s,%s,%s,%s,1,%s,1,'execute','leased',%s,%s,1,1,clock_timestamp()+interval '1 hour')",(run_id,run_id+'-assignment',round_id,sub,miner,index,hotkey,hash_lease_token(lease)))
                cells.append((role,index,hotkey,run_id,lease))
    store = ArenaStore(PsycopgTransport(lambda: psycopg2.connect(**dsn)))
    service = ArenaService(ServiceConfig(mode="live",store=store,object_store=None,signer=None,chain=None,verify_signature=lambda *args:False,daily_icp_source=lambda **kw:None,banned_hotkeys_source=lambda:[],broker_factory=lambda *args:None))
    sidecar = create_app(service)
    forwarded = []
    async def forward(method,path,*,query,body,headers):
        assert "authorization" not in headers and "apikey" not in headers
        forwarded.append(path)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=sidecar),base_url="http://sidecar") as client:
            return await client.request(method,"/arena/"+path,content=body,headers=headers)
    monkeypatch.setattr(arena_proxy,"_request_sidecar",forward)
    app = FastAPI(); app.include_router(arena_proxy.router)
    try:
        with TestClient(app,base_url="https://gateway.example") as client:
            api = HttpArenaApiClient("https://gateway.example",client=client)
            for role,index,hotkey,run_id,lease in cells:
                decision = {
                    "source": "model_reported", "sequence": 0,
                    "after_action_sequence": 2,
                    "objective": "Choose a supported candidate",
                    "evidence": ["https://example.com/source"],
                    "rationale": "The source supports the required signal.",
                    "next_action": "Save the candidate", "decision": "accept",
                }
                events = [
                    trajectory.event("runtime.started", {"status": "starting"}),
                    trajectory.event("runtime.decision", decision),
                    trajectory.event("runtime.decision_capture", {
                        "status": "provided", "recorded": 1, "omitted": 0,
                    }),
                    trajectory.event("runtime.finished", {"status": "accepted"}),
                ]
                assert api.trajectory(run_id,lease,events)["inserted"]==4
                assert api.trajectory(run_id,lease,events)["existing"]==4
                rows=store.list_trajectory_events(run_id)
                assert len(rows)==4
                assert next(row["content"] for row in rows if row["event_kind"] == "runtime.decision") == decision
                assert all(row["runner_hotkey"]==hotkey and row["model_role"]==role and row["round_id"]==round_id and row["icp_position"]==index for row in rows)
                route="/arena/v1/runs/"+run_id+"/trajectory"
                assert client.post(route,json={"events":events}).status_code==401
                wrong=cells[(cells.index((role,index,hotkey,run_id,lease))+1)%len(cells)][4]
                assert client.post(route,headers={"x-lab-arena-lease":wrong},json={"events":events}).status_code==409
                forged=trajectory.event("runtime.started",{})
                forged["runner_hotkey"]=keys[1-index]
                assert client.post(route,headers={"x-lab-arena-lease":lease},json={"events":[forged]}).status_code==400
                assert client.post(route,headers={"x-lab-arena-lease":lease},json={"events":[trajectory.event("provider.request",{})]}).status_code==400
                assert len(store.list_trajectory_events(run_id))==4
        assert len(forwarded)==24
    finally:
        store.close()
