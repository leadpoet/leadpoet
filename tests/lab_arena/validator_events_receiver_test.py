"""Real signed public HTTP to service/RPC proof for validator observations."""
from __future__ import annotations

import json
import time
import threading
import uuid
from pathlib import Path

import httpx
import pytest
from bittensor_wallet import Keypair
from fastapi import FastAPI
from fastapi.testclient import TestClient

from gateway.api import arena_proxy
from lab_arena import contracts, validator_events
from lab_arena.api import create_app
from lab_arena.service import ArenaService, ServiceConfig
from lab_arena.store import ArenaStore, ArenaStoreError, PsycopgTransport
from tests.lab_arena.lab_arena_pg_harness import CURRENT_SERVICE_MIGRATIONS, database_with_lab_arena_migration
from tests.lab_arena.test_lab_arena_service_round import FakeChain, wallet_verify

MIGRATION = '434-lab-arena-validator-events.sql'
PRIMARY = Keypair.create_from_uri('//events-primary')
EXTERNAL = Keypair.create_from_uri('//events-external')
OUTSIDER = Keypair.create_from_uri('//events-outsider')
ROUND = 'arena-2099-01-01-valevents'
RUN = 'validator-events-run'


@pytest.fixture(scope='module')
def database():
    yield from database_with_lab_arena_migration(CURRENT_SERVICE_MIGRATIONS + (MIGRATION,))


@pytest.fixture()
def setup(database, monkeypatch):
    psycopg2, dsn = database
    connect = lambda: psycopg2.connect(**dsn)
    with connect() as db, db.cursor() as cur:
        cur.execute('TRUNCATE public.lab_arena_validator_events')
        cur.execute("INSERT INTO public.lab_arena_rounds(round_id,status,configuration_doc) VALUES (%s,'stage1',%s::jsonb) ON CONFLICT DO NOTHING", (ROUND,json.dumps({'mode':'live'})))
        cur.execute("INSERT INTO public.lab_arena_submissions(submission_id,round_id,miner_hotkey,status,is_king) VALUES ('events-submission',%s,%s,'frozen',TRUE) ON CONFLICT DO NOTHING",(ROUND,PRIMARY.ss58_address))
        cur.execute("INSERT INTO public.lab_arena_runs(run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,icp_position,attempt,kind,status,runner_hotkey) VALUES (%s,'events-assignment',%s,'events-submission',%s,1,3,1,'execute','failed',%s) ON CONFLICT DO NOTHING",(RUN,ROUND,PRIMARY.ss58_address,EXTERNAL.ss58_address))
    store = ArenaStore(PsycopgTransport(connect))
    chain = FakeChain([PRIMARY.ss58_address,EXTERNAL.ss58_address])
    # A low stake must not suppress the diagnostic that explains claim denial.
    chain.stakes[EXTERNAL.ss58_address] = 0
    service = ArenaService(ServiceConfig(
        mode='live',store=store,object_store=None,signer=None,chain=chain,
        verify_signature=wallet_verify,daily_icp_source=lambda **kw: {},
        banned_hotkeys_source=lambda: [],broker_factory=lambda *a: None,
    ))
    sidecar = create_app(service)
    async def forward(method,path,*,query,body,headers,**kw):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=sidecar),base_url='http://sidecar') as client:
            return await client.request(method,'/arena/'+path,content=body,headers=headers)
    monkeypatch.setenv('LAB_ARENA_MODE','live')
    monkeypatch.setattr(arena_proxy,'_request_sidecar',forward)
    app = FastAPI()
    app.include_router(arena_proxy.router)
    with TestClient(app) as http:
        yield http,service,store,connect,chain


def signed(key, events, **body_fields):
    return contracts.build_signed_request(
        scope=contracts.SCOPE_VALIDATOR_EVENTS,round_id='validator-events',
        hotkey=key.ss58_address,body={'network':'finney','netuid':71,'events':events,**body_fields},
        timestamp=int(time.time()),sign_message=lambda message: key.sign(message.encode()).hex(),
    )


def post(http,key,events,**fields):
    return http.post('/arena/v1/validators/events',json=signed(key,events,**fields))


def rows(connect):
    with connect() as db, db.cursor() as cur:
        cur.execute('SELECT validator_hotkey,event_kind,run_id,round_id,submission_id,icp_position,content,gateway_source_commit FROM public.lab_arena_validator_events ORDER BY validator_event_id')
        return cur.fetchall()


def test_primary_and_external_preclaim_error_then_correlated_run_and_replay(setup):
    http,service,store,connect,chain = setup
    for key in (PRIMARY,EXTERNAL):
        document = validator_events.event('validator.error',{'phase':'claim','reason':'claim_denied','session_id':str(uuid.uuid4())})
        response = post(http,key,[document])
        assert response.status_code == 200,response.text
        assert response.json()['inserted'] == 1
        assert post(http,key,[document]).json()['existing'] == 1
    correlated = validator_events.event('validator.error',{'phase':'execution','error_class':'RuntimeLaunchError'},run_id=RUN)
    assert post(http,EXTERNAL,[correlated]).status_code == 200
    saved = rows(connect)
    assert saved[0][0] == PRIMARY.ss58_address
    assert saved[1][0] == EXTERNAL.ss58_address
    assert saved[0][2:6] == (None,None,None,None)
    assert saved[2][2:6] == (RUN,ROUND,'events-submission',3)
    assert saved[2][7]


def test_forged_identity_bad_signature_scope_and_run_ownership_denied(setup):
    http,_,_,connect,chain = setup
    event = validator_events.event('validator.startup',{'state':'starting'})
    forged = signed(EXTERNAL,[event]);forged['hotkey'] = PRIMARY.ss58_address
    assert http.post('/arena/v1/validators/events',json=forged).status_code == 401
    assert post(http,OUTSIDER,[event]).status_code == 403
    assert post(http,EXTERNAL,[event],netuid=401).status_code == 400
    assert post(http,EXTERNAL,[{**event,'validator_hotkey':PRIMARY.ss58_address}]).status_code == 400
    assert post(http,PRIMARY,[{**event,'run_id':RUN}]).status_code == 403
    assert post(http,EXTERNAL,[{**event,'run_id':RUN,'round_id':'forged-round'}]).status_code == 400
    assert post(http,EXTERNAL,[{**event,'round_id':'missing-round'}]).status_code == 400
    assert not rows(connect)


def test_bounds_privacy_and_best_effort_failure(setup,monkeypatch):
    http,service,_,connect,_ = setup
    document = validator_events.event('validator.error',{'launch_stderr':'startup'})
    document['content']['launch_stderr'] = 'authorization="multi word credential" https://user:password@host/p?q=secret sk-or-v1-abcdefghijklmnopqrst'
    assert post(http,EXTERNAL,[document]).status_code == 200
    content = rows(connect)[0][6]['launch_stderr']
    for secret in ('multi word credential','password','q=secret','sk-or-v1-abcdefghijklmnopqrst'):
        assert secret not in content
    for value in ({'api_key':'credential'},{'launch_stderr':'x'*2049},{'phase':{}},{'active_runs':True}):
        assert post(http,EXTERNAL,[{**document,'content':value}]).status_code == 400
    assert post(http,EXTERNAL,[document]*17).status_code == 400
    assert http.post('/arena/v1/validators/events',content=b'x'*32769).status_code == 413
    def fail(*a): raise ArenaStoreError('database unavailable')
    monkeypatch.setattr(service.store,'append_validator_events',fail)
    assert post(http,EXTERNAL,[validator_events.event('validator.state',{'state':'idle'})]).status_code == 503


def test_rate_limit_exact_retry_daily_cap_and_global_retention(setup):
    http,_,store,connect,_ = setup
    docs = [validator_events.event('validator.state',{'state':'idle'}) for _ in range(60)]
    for start in range(0,60,15):
        assert post(http,EXTERNAL,docs[start:start+15]).status_code == 200
    assert post(http,EXTERNAL,[docs[0]]).status_code == 200
    limited = post(http,EXTERNAL,[validator_events.event('validator.state',{'state':'idle'})])
    assert limited.status_code == 429
    assert limited.headers['retry-after'] == '60'
    assert post(http,PRIMARY,[validator_events.event('validator.ready',{'state':'ready'})]).status_code == 200
    with connect() as db, db.cursor() as cur:
        cur.execute("UPDATE public.lab_arena_validator_events SET created_at = clock_timestamp() - interval '15 days' WHERE validator_hotkey=%s",(EXTERNAL.ss58_address,))
    assert post(http,PRIMARY,[validator_events.event('validator.state',{'state':'idle'})]).status_code == 200
    assert all(row[0] == PRIMARY.ss58_address for row in rows(connect))
    with connect() as db, db.cursor() as cur:
        cur.execute("INSERT INTO public.lab_arena_validator_events (validator_hotkey,network,netuid,event_id,event_kind,occurred_at,content,gateway_source_commit,created_at) SELECT %s,'finney',71,md5(i::text)::uuid,'validator.state',clock_timestamp(),'{}'::jsonb,'unknown',clock_timestamp()-interval '2 hours' FROM generate_series(1,1000) i",(EXTERNAL.ss58_address,))
    assert post(http,EXTERNAL,[validator_events.event('validator.state',{'state':'idle'})]).status_code == 429


def test_migration_replay_privileges_and_sql_bounds(setup):
    _,_,store,connect,_ = setup
    migration = (Path(__file__).resolve().parents[2]/'scripts'/MIGRATION).read_text()
    with connect() as db, db.cursor() as cur:
        cur.execute(migration);cur.execute(migration)
        cur.execute("SELECT has_table_privilege('lab_arena_service','public.lab_arena_validator_events','INSERT'),has_table_privilege('anon','public.lab_arena_validator_events','SELECT'),has_function_privilege('anon','public.lab_arena_append_validator_events_v1(text,text,integer,jsonb,text)','EXECUTE')")
        assert cur.fetchone() == (False,False,False)
    document = validator_events.event('validator.error',{'phase':'claim'})
    for bad in ({**document,'content':{'api_key':'secret'}},{**document,'content':{'phase':{'nested':'no'}}},{**document,'content':{'launch_stderr':'x'*5000}},{**document,'event_id':None},{**document,'occurred_at':'infinity'}):
        with pytest.raises(ArenaStoreError):
            store.append_validator_events(EXTERNAL.ss58_address,'finney',71,[bad],'unknown')
    assert not rows(connect)


def test_real_sender_persists_scoring_setup_failure_while_weights_continue(setup):
    from lab_arena import validator, runtime_version
    from lab_arena.runtime_host import RuntimeHostError
    from lab_arena.validator_logging import ValidatorOperationalLogger

    http, _, _, connect, _ = setup
    delivered = threading.Event()
    responses = []
    weights = []

    def transport(envelope):
        response = http.post('/arena/v1/validators/events', json=envelope)
        responses.append(response.status_code)
        response.raise_for_status()
        if any(item['kind'] == 'validator.error' for item in envelope['body']['events']):
            delivered.set()

    log = ValidatorOperationalLogger(
        'https://public-gateway.invalid', keypair=EXTERNAL, network='finney',
        netuid=71, transport=transport,
    )

    class Orchestrator:
        def run_once(self, epoch):
            weights.append(epoch)
            return 'state_unavailable'

        def poll_prior_outcomes(self, epoch):
            pass

    def broken_runner():
        raise RuntimeHostError('password=must-not-persist', reason='runsc_missing')

    log.emit('validator.startup', {'phase': 'startup'})
    validator.run_validator_loops(
        orchestrator=Orchestrator(), runner_factory=broken_runner,
        epoch_supplier=lambda: 1, stop=threading.Event(), once=True,
        operational_logger=log,
    )
    log.emit('validator.state', log.snapshot())
    log.start()
    try:
        assert delivered.wait(3), responses
    finally:
        log.close()
    assert not log._thread.is_alive()
    assert weights == [1]
    saved = rows(connect)
    error = next(row for row in saved if row[1] == 'validator.error')
    assert error[0] == EXTERNAL.ss58_address
    assert error[2:6] == (None, None, None, None)
    assert error[6]['phase'] == 'scoring_setup'
    assert error[6]['reason'] == 'runsc_missing'
    assert error[6]['error_class'] == 'RuntimeHostError'
    assert error[6]['validator_source_commit'] == runtime_version.SOURCE_METADATA['validator_source_commit']
    assert error[6]['session_id'] == log.session_id
    assert any(row[1] == 'validator.state' and row[6]['weights_state'] == 'state_unavailable' for row in saved)
    assert 'must-not-persist' not in str(saved)
    assert all(code == 200 for code in responses)
