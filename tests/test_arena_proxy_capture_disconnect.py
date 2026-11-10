"""Public capture disconnects must reach the sidecar's bounded admission queue."""
import asyncio
from contextlib import asynccontextmanager
import json
import socket
import threading
import time

import anyio
from fastapi import FastAPI
import httpx
import pytest
import uvicorn

from gateway.api import arena_proxy as proxy
from gateway.middleware.body_size import BodySizeLimitMiddleware
from lab_arena.api import create_app
from lab_arena.service import LocalObjectStore
from tests.lab_arena.api_provider_capacity_test import wait_until
from tests.lab_arena.api_training_input_capacity_test import (
    CAPTURE_LIMIT, CLAIM_HASH, TOKEN, CaptureService, post_input, real_service,
)

KINDS=['training-input','training-observations','training-observation-segments','verify']


def gateway():
    app=FastAPI();app.include_router(proxy.router);app.include_router(proxy.testnet_router)
    app.add_middleware(BodySizeLimitMiddleware)
    return app


def raw_request(app,kind,*,testnet=False):
    messages=asyncio.Queue();messages.put_nowait({'type':'http.request','body':b'{}','more_body':False})
    sent=[];reads=[]
    suffix='training-observations'if kind=='verify'else kind
    path=('/testnet'if testnet else '')+'/arena/v1/runs/queued/'+suffix
    headers=[(b'x-lab-arena-lease',TOKEN.encode())]
    if kind=='verify':headers.append((b'if-match',b'"sha256:'+b'1'*64+b'"'))
    scope={'type':'http','asgi':{'version':'3.0'},'http_version':'1.1','method':'HEAD'if kind=='verify'else'POST',
        'scheme':'http','path':path,'raw_path':path.encode(),'query_string':b'','headers':headers,
        'server':('test',80),'client':('client',1234),'root_path':''}
    async def receive():
        reads.append(time.monotonic());return await messages.get()
    async def send(message):sent.append(message)
    return asyncio.create_task(app(scope,receive,send)),messages,sent,reads


@pytest.mark.parametrize('testnet',[False,True])
@pytest.mark.parametrize('kind',KINDS)
def test_real_asgi_capture_disconnect_cancels_upstream_without_polling(monkeypatch,kind,testnet):
    monkeypatch.setenv('LAB_ARENA_MODE','live');monkeypatch.setenv('LAB_ARENA_TESTNET_ENABLED','true')
    async def scenario():
        entered=asyncio.Event();cancelled=asyncio.Event()
        async def forward(method,path,**kwargs):
            if method=='HEAD'and 'if-match'not in kwargs['headers']:return httpx.Response(204)
            entered.set()
            try:await anyio.sleep_forever()
            finally:cancelled.set()
        monkeypatch.setattr(proxy,'_request_sidecar',forward)
        task,messages,sent,reads=raw_request(gateway(),kind,testnet=testnet)
        try:
            await asyncio.wait_for(entered.wait(),1)
            messages.put_nowait({'type':'http.disconnect'})
            done,_=await asyncio.wait({task},timeout=.2)
            assert done,'capture upstream remained attached after downstream http.disconnect'
            await task
            assert cancelled.is_set()
            assert len(reads)==2  # One body read and one blocking disconnect receive.
            assert next(m['status']for m in sent if m['type']=='http.response.start')==499
        finally:
            if not task.done():task.cancel()
            await asyncio.gather(task,return_exceptions=True)
    asyncio.run(scenario())


@asynccontextmanager
async def serve(app):
    with socket.socket() as listener:
        listener.bind(('127.0.0.1',0))
        server=uvicorn.Server(uvicorn.Config(app,log_level='critical',ws='none',lifespan='on'))
        serving=asyncio.create_task(server.serve(sockets=[listener]))
        try:
            await wait_until(lambda:server.started)
            yield 'http://127.0.0.1:%d'%listener.getsockname()[1]
        finally:
            server.should_exit=True
            await asyncio.wait_for(serving,5)


def observed_sidecar(app):
    finished={};watching={}
    async def observed(scope,receive,send):
        if scope.get('type')!='http':return await app(scope,receive,send)
        run=scope['path'].split('/')[-2];body_done=False
        async def tracked_receive():
            nonlocal body_done
            if body_done:watching.setdefault(run,asyncio.Event()).set()
            message=await receive()
            if message['type']=='http.request'and not message.get('more_body',False):body_done=True
            return message
        try:await app(scope,tracked_receive,send)
        finally:finished.setdefault(run,asyncio.Event()).set()
    return observed,finished,watching


@pytest.mark.parametrize('kind',KINDS)
def test_tcp_gateway_disconnect_removes_actual_sidecar_queue(monkeypatch,kind):
    monkeypatch.setenv('LAB_ARENA_MODE','live')
    async def scenario():
        from tests.lab_arena.api_training_observation_capacity_test import ObservationService,request
        from tests.lab_arena.api_provider_capacity_test import post_provider
        service=(CaptureService if kind=='training-input'else ObservationService)(anyio.to_thread.current_default_thread_limiter())
        service.release.set()
        app,finished,watching=observed_sidecar(create_app(service))
        def post(client,run):
            return post_input(client,run)if kind=='training-input'else request(client,{'verify':'verify','training-observations':'final','training-observation-segments':'segment'}[kind],run)
        blockers=[]
        try:
            async with serve(app)as sidecar:
                monkeypatch.setattr(proxy,'_SIDECAR_URL',sidecar)
                async with serve(gateway())as public:
                    async with httpx.AsyncClient(base_url=sidecar,timeout=3)as direct,httpx.AsyncClient(base_url=public,timeout=.25)as client:
                        blockers=[post(direct,'active-'+str(i))for i in range(CAPTURE_LIMIT)]
                        active=lambda:service.capture_active if kind=='training-input'else service.observation_active
                        await wait_until(lambda:active()==CAPTURE_LIMIT)
                        queued=post(client,'queued')
                        await wait_until(lambda:'queued'in watching and watching['queued'].is_set())
                        with pytest.raises(httpx.ReadTimeout):await queued
                        timed_out=time.monotonic()
                        await wait_until(lambda:'queued'in finished and finished['queued'].is_set())
                        elapsed=time.monotonic()-timed_out
                        assert active()==CAPTURE_LIMIT
                        calls=[r[0]for r in service.input_calls]if kind=='training-input'else service.observation_calls
                        assert 'queued'not in calls
                        # These routes retain their ordinary behavior while storage is full.
                        responses=await asyncio.gather(client.post('/arena/v1/runs/claim',json={}),
                            client.post('/arena/v1/runs/done/complete',json={'body':{'run_id':'done'}}),post_provider(client,'provider'))
                        assert all(r.status_code==200 for r in responses)
                        service.capture_release.set()
                        if hasattr(service,'observation_release'):service.observation_release.set()
                        assert all(r.status_code in {200,204}for r in await asyncio.gather(*blockers))
                        assert (await post(client,'healthy')).status_code in {200,204}
                        assert 'queued'not in calls
                        print(json.dumps({'capture':kind,'queued_disconnect_cleanup_seconds':round(elapsed,4),'queued_storage_calls':0}))
        finally:
            service.capture_release.set()
            if hasattr(service,'observation_release'):service.observation_release.set()
            await asyncio.gather(*blockers,return_exceptions=True)
    asyncio.run(scenario())


def test_tcp_disconnect_does_not_abandon_admitted_real_storage_write(tmp_path,monkeypatch):
    monkeypatch.setenv('LAB_ARENA_MODE','live')
    async def scenario():
        from lab_arena import training_input
        objects=LocalObjectStore(tmp_path/'private-objects');original=objects.put
        entered=threading.Event();release=threading.Event();writes=[]
        def blocked_put(ref,data):
            writes.append(ref);entered.set()
            assert release.wait(5),'test did not release admitted storage write'
            original(ref,data)
        objects.put=blocked_put
        app,finished,_=observed_sidecar(create_app(real_service(objects)))
        try:
            async with serve(app)as sidecar:
                monkeypatch.setattr(proxy,'_SIDECAR_URL',sidecar)
                async with serve(gateway())as public:
                    async with httpx.AsyncClient(base_url=public,timeout=.25)as client:
                        capture=post_input(client,'admitted',{'input':'private unchanged bytes'})
                        await wait_until(entered.is_set)
                        with pytest.raises(httpx.ReadTimeout):await capture
                        assert not finished.get('admitted',asyncio.Event()).is_set()
                        assert len(writes)==1
                        release.set()
                        await wait_until(lambda:'admitted'in finished and finished['admitted'].is_set())
                        saved=training_input.read_input(objects,'admitted','score',1,CLAIM_HASH)
                        expected,_=training_input.build_snapshot('admitted','score',{'input':'private unchanged bytes'},
                            lease_generation=1,claim_request_hash=CLAIM_HASH,secrets=(TOKEN,))
                        assert saved==expected
                        assert len(writes)==1
        finally:release.set()
    asyncio.run(scenario())



@pytest.mark.parametrize('kind',KINDS)
def test_capture_healthy_response_stops_disconnect_watcher(monkeypatch,kind):
    monkeypatch.setenv('LAB_ARENA_MODE','live')
    async def scenario():
        async def forward(method,path,**kwargs):
            await anyio.lowlevel.checkpoint()
            return httpx.Response(204 if method=='HEAD'else 200,content=b''if method=='HEAD'else b'private-result',
                headers={'cache-control':'no-store','etag':'"sha256:'+'1'*64+'"'})
        monkeypatch.setattr(proxy,'_request_sidecar',forward)
        task,_,sent,reads=raw_request(gateway(),kind)
        await asyncio.wait_for(task,1)
        assert len(reads)==2
        start=next(m for m in sent if m['type']=='http.response.start')
        assert start['status']==(204 if kind=='verify'else 200)
        if kind=='verify':assert (b'etag',b'"sha256:'+b'1'*64+b'"')in start['headers']
    asyncio.run(scenario())


@pytest.mark.parametrize('failure',[httpx.ConnectError('offline transport'),RuntimeError('storage adapter failure')])
def test_capture_errors_retain_original_handling(monkeypatch,failure):
    monkeypatch.setenv('LAB_ARENA_MODE','live')
    async def scenario():
        async def forward(*args,**kwargs):raise failure
        monkeypatch.setattr(proxy,'_request_sidecar',forward)
        task,_,sent,_=raw_request(gateway(),'training-input')
        if isinstance(failure,httpx.HTTPError):
            await asyncio.wait_for(task,1)
            assert next(m['status']for m in sent if m['type']=='http.response.start')==503
        else:
            with pytest.raises(RuntimeError)as caught:await asyncio.wait_for(task,1)
            assert caught.value is failure
    asyncio.run(scenario())


@pytest.mark.parametrize('route',['provider','complete','claim'])
def test_non_capture_callbacks_do_not_start_disconnect_watcher(monkeypatch,route):
    monkeypatch.setenv('LAB_ARENA_MODE','live')
    async def scenario():
        async def forward(*args,**kwargs):
            await anyio.lowlevel.checkpoint();return httpx.Response(200)
        monkeypatch.setattr(proxy,'_request_sidecar',forward)
        task,messages,sent,reads=raw_request(gateway(),route)
        messages.put_nowait({'type':'http.disconnect'})
        await asyncio.wait_for(task,1)
        assert len(reads)==1
        assert next(m['status']for m in sent if m['type']=='http.response.start')==200
    asyncio.run(scenario())
