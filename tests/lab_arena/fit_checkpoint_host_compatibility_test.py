"""Capability proof separates an old worker from a failed real fit claim."""
from __future__ import annotations

import asyncio
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from lab_arena import company_judgments as cj, contracts, runner, signal_cache_control as control
from qualification.scoring import lead_scorer
from qualification.scoring.signal_judgment_cache import install_exchange, SignalJudgmentCacheError
from tests.lab_arena.signal_cache_control_test import state
from tests.lab_arena.verified_fit_checkpoint_test import native_fit, snapshot
from tests.test_arena_intent_retry_cache import _company_model, _icp_model


def probe():
    return {"operation":control.FIT_CAPABILITY_OPERATION,"company_index":0,
            "signal_index":-1,"checkpoint_kind":cj.FIT_CHECKPOINT_KIND}


def old_validator():
    source=Path(__file__).parents[1]/"fixtures/arena_signal_control_before_fit.py"
    namespace=dict(control.__dict__)
    exec(compile(source.read_text(),str(source),"exec"),namespace)
    return namespace["validate_request"]


def test_capability_is_worker_local_guarded_and_has_no_provider_action(tmp_path):
    def forbidden(*_args):pytest.fail("capability cannot call a gateway or provider")
    for kind,enabled,expected in (("score",True,control.fit_checkpoint_capability()),("execute",True,{"status":"unavailable"}),("score",False,{"status":"unavailable"})):
        run=state(kind=kind,enabled=enabled)
        worker=runner.WorkerSocketServer(tmp_path/"unused",SimpleNamespace(signal_judgment=forbidden),run)
        response=json.loads(worker.handle_frame(contracts.canonical_json({"schema_version":control.SCHEMA,"request":probe()}).encode()))
        assert response==expected
        assert run.action_sequence==0 and run.calls==[] and run.refusals==0


@pytest.mark.parametrize("change",[{"signal_index":0},{"signal_index":True},{"company_index":True},{"checkpoint_kind":"unknown"},{"judgment":None},{"reference":{}},{"scored_run_id":"foreign"}])
def test_capability_request_has_exact_typed_fields(change):
    with pytest.raises(ValueError):control.validate_request({**probe(),**change})


@pytest.mark.parametrize("response",[{}, {"status":"supported"}, {**control.fit_checkpoint_capability(),"extra":True}, {"status":"unavailable","reason":"corrupt"}, {"status":"busy"}])
def test_malformed_capability_is_not_a_fallback(monkeypatch,response):
    monkeypatch.setattr(control,"_exchange_once",lambda _request:response)
    with pytest.raises(SignalJudgmentCacheError,match="capability is invalid"):
        asyncio.run(control.exchange(probe()))


def test_capability_transport_error_does_not_enable_fallback(monkeypatch):
    def unavailable(_request):raise OSError("test transport down")
    monkeypatch.setattr(control,"_exchange_once",unavailable)
    with pytest.raises(SignalJudgmentCacheError,match="transport unavailable"):
        asyncio.run(control.exchange(probe()))


def test_only_pure_unsupported_capability_enables_full_native_fit(monkeypatch):
    operations=[];fits=[]
    async def exchange(request):
        operations.append(request["operation"])
        assert request["operation"]==control.FIT_CAPABILITY_OPERATION
        return {"status":"unavailable"}
    async def verify(*_args,**_kwargs):fits.append(1);return native_fit()
    async def intent(*_args,**_kwargs):return 80,80,1,100,False,[]
    monkeypatch.setattr(lead_scorer,"_verify_company_fit",verify)
    monkeypatch.setattr(lead_scorer,"score_company_competition_intent_signal",intent)
    with install_exchange(exchange):
        result=asyncio.run(lead_scorer.score_company_competition_intent(_company_model(),_icp_model(),0,0,set(),integrity_policy=True))
    assert result.final_score==80 and fits==[1]
    assert operations==[control.FIT_CAPABILITY_OPERATION]


@pytest.mark.parametrize("bad_claim",[{"status":"unavailable"},{"status":"hit","reference":{},"judgment":{}},{"status":"hit","reference":{},"judgment":{**snapshot(),"company_fit_receipt":{}}}])
def test_supported_worker_claim_failure_or_malformed_hit_never_falls_back(monkeypatch,bad_claim):
    operations=[]
    async def exchange(request):
        operations.append(request["operation"])
        return control.fit_checkpoint_capability() if request["operation"]==control.FIT_CAPABILITY_OPERATION else bad_claim
    async def forbidden(*_args,**_kwargs):pytest.fail("failed accepted-cache authority cannot dispatch fit")
    monkeypatch.setattr(lead_scorer,"_verify_company_fit",forbidden)
    with install_exchange(exchange),pytest.raises(SignalJudgmentCacheError):
        asyncio.run(lead_scorer.score_company_competition_intent(_company_model(),_icp_model(),0,0,set(),integrity_policy=True))
    assert operations==[control.FIT_CAPABILITY_OPERATION,"claim"]


def test_supported_complete_failure_never_enters_intent(monkeypatch):
    operations=[];fits=[]
    async def exchange(request):
        operations.append(request["operation"])
        if request["operation"]==control.FIT_CAPABILITY_OPERATION:return control.fit_checkpoint_capability()
        if request["operation"]=="claim":return {"status":"miss","reference":{}}
        return {"status":"unavailable"}
    async def verify(*_args,**_kwargs):fits.append(1);return native_fit()
    async def forbidden(*_args,**_kwargs):pytest.fail("failed durable write cannot enter intent")
    monkeypatch.setattr(lead_scorer,"_verify_company_fit",verify)
    monkeypatch.setattr(lead_scorer,"score_company_competition_intent_signal",forbidden)
    with install_exchange(exchange),pytest.raises(SignalJudgmentCacheError,match="completion is invalid"):
        asyncio.run(lead_scorer.score_company_competition_intent(_company_model(),_icp_model(),0,0,set(),integrity_policy=True))
    assert fits==[1] and operations==[control.FIT_CAPABILITY_OPERATION,"claim","complete"]
