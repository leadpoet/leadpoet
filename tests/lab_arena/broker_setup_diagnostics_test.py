"""Setup diagnostics must be private, bounded, and neutral to dispatch/payer policy."""
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from gateway.observability.otel_bootstrap import configure_arena_otel
from lab_arena import broker as br, telemetry, trajectory
from lab_arena.credentials import CredentialError
from lab_arena.store import ArenaStoreError, ArenaStoreUnavailable, hash_lease_token
from lab_arena.submission_runtime import SubmissionProviderKeys
from tests.lab_arena.test_lab_arena_broker import CONTEXT, FakeTransport, make_broker
from tests.lab_arena.trajectory_test import _run, _service_for_trajectory
from tests.lab_arena.provider_trajectory_audit_test import _retrying_broker, FRAME as CHAT_FRAME

SECRET = 'secret-value-must-not-escape-https://private.invalid'
FRAME = dict(operation_id='deepline.execute', parameters={'tool': 'exa_search', 'payload': {'query': 'synthetic'}}, action_sequence=4, timeout_ms=60_000)

class SetupStore:
    def __init__(self, mode, funding='miner_key'):
        self.mode, self.funding = mode, funding
    def get_submission(self, submission_id):
        return {'submission_id': submission_id, 'miner_hotkey': CONTEXT.miner_hotkey, 'status': 'frozen'}
    def provider_funding(self, *_args):
        if self.mode == 'timeout':
            raise ArenaStoreUnavailable(SECRET, operation='rpc', resource='lab_arena_provider_funding', transport_error='ReadTimeout')
        if self.mode == 'http':
            raise ArenaStoreError(SECRET, operation='rpc', resource='lab_arena_provider_funding', http_status=503, code='PGRST002')
        if self.mode == 'unknown':
            class SecretError(Exception):
                store_error = {'operation':'rpc', 'resource':'private_secret', 'transport_error':SECRET}
            raise SecretError(SECRET)
        if self.mode == 'store_subclass':
            class UnapprovedStoreError(ArenaStoreUnavailable):
                pass
            raise UnapprovedStoreError(SECRET,operation='rpc',resource='lab_arena_provider_funding',transport_error='ReadTimeout')
        if self.mode == 'shape':
            return {'status': SECRET}
        return {'status': 'available', 'funding_source': self.funding, 'champion_funding': False, 'credential_submission_id': CONTEXT.submission_id if self.funding == 'miner_key' else None, 'credential_miner_hotkey': CONTEXT.miner_hotkey if self.funding == 'miner_key' else None, 'restart_required': False}
    def get_submission_credential(self, submission_id, miner_hotkey, provider):
        return {'submission_id': submission_id, 'miner_hotkey': miner_hotkey, 'provider': provider}

class Keys:
    def __init__(self, invalid=False): self.invalid = invalid
    def runtime_key(self, *_args):
        if self.invalid:
            raise CredentialError('miner_credentials_unavailable', retryable=self.invalid=='retryable')
        return 'synthetic-key'

def provider(mode, funding='miner_key', invalid_key=False):
    resolver = SubmissionProviderKeys(store=SetupStore(mode, funding), credentials=Keys(invalid_key), organizer_keys={'deepline':'synthetic-key'})
    return make_broker(provider_funding_source_for=resolver.provider_funding_source_for, credential_for=resolver.credential_for)

@pytest.mark.parametrize('mode,phase,error_class,metadata', [
    ('timeout','provider_funding','ArenaStoreUnavailable',{'operation':'rpc','resource':'lab_arena_provider_funding','transport_error':'ReadTimeout'}),
    ('http','provider_funding','ArenaStoreError',{'operation':'rpc','resource':'lab_arena_provider_funding','http_status':503,'code':'PGRST002'}),
    ('shape','provider_funding_validation','BrokerError',{}),
    ('unknown','provider_funding','Exception',{}),
    ('store_subclass','provider_funding','Exception',{}),
])
def test_setup_failure_retains_safe_cause_without_changing_worker_document(mode,phase,error_class,metadata):
    b, ledger, transport = provider(mode)
    result = b.execute(CONTEXT, **FRAME)
    expected = br._error_result('broker_unavailable', {'operation_id':'deepline.execute','provider':'deepline','funding_source':'host','provider_attempt':1})
    assert result.to_document() == expected.to_document()
    assert result.setup_diagnostics == dict(phase=phase,error_class=error_class,funding_resolved=False,**({'store_error':metadata} if metadata else {}))
    assert SECRET not in json.dumps(result.setup_diagnostics)
    assert not ledger.calls and not transport.sent

@pytest.mark.parametrize('mode', ['timeout','http','shape'])
def test_real_handle_provider_exports_existing_onepatch_fields_and_private_phase(mode):
    b, ledger, transport = provider(mode)
    run = dict(_run(), round_id='arena-2026-10-24', miner_hotkey=CONTEXT.miner_hotkey, lease_token_hash=hash_lease_token('lease-token'))
    # Use the normal service route/context, with the exact matching submitted identity.
    events = []
    class Store:
        get_run = staticmethod(lambda _run_id: run)
        @staticmethod
        def append_trajectory_events(_run_id,_lease_hash,batch):
            events.extend(batch); return {'status':'accepted','accepted':len(batch)}
    service = _service_for_trajectory(Store(),b)
    exporter = InMemorySpanExporter()
    telemetry.install_recorder(configure_arena_otel(None, span_exporter=exporter))
    try:
        returned = service.handle_provider('run-1','lease-token',FRAME)
    finally:
        telemetry.install_recorder(None)
    diagnostic = events[-1]['content']['setup_diagnostics']
    assert diagnostic['funding_resolved'] is False
    assert diagnostic['phase'] == ('provider_funding_validation' if mode=='shape' else 'provider_funding')
    assert set(returned) == {'status','headers','body_b64','call'}
    assert 'setup_diagnostics' not in json.dumps(returned)
    (span,) = exporter.get_finished_spans()
    assert span.attributes['arena.error_type'] == diagnostic['error_class']
    for key,value in diagnostic.get('store_error',{}).items():
        assert span.attributes['arena.store_'+key] == value
    assert 'arena.phase' not in span.attributes
    assert SECRET not in repr(events) + repr(dict(span.attributes))
    assert not ledger.calls and not transport.sent


def test_resolved_miner_invalid_key_keeps_credential_refusal_and_known_payer():
    b, ledger, transport = provider('ok', invalid_key=True)
    result = b.execute(CONTEXT, **FRAME)
    assert result.call['error_code'] == 'miner_credentials_unavailable'
    assert result.call['funding_source'] == 'miner_key'
    assert result.setup_diagnostics == {'phase':'credential_resolution','error_class':'CredentialError','funding_resolved':True}
    assert not ledger.calls and not transport.sent


def test_retryable_key_failure_remains_broker_error_with_resolved_miner_payer():
    b,ledger,transport=provider('ok',invalid_key='retryable')
    result=b.execute(CONTEXT,**FRAME)
    assert result.call['error_code']=='broker_unavailable'
    assert result.call['funding_source']=='miner_key'
    assert result.setup_diagnostics=={'phase':'credential_resolution','error_class':'CredentialError','funding_resolved':True}
    assert not ledger.calls and not transport.sent


def test_route_failure_retains_fixed_phase_and_resolved_payer(monkeypatch):
    b,ledger,transport=provider('ok')
    def unavailable_route(**_kwargs):raise br.deepline_catalog.CatalogError(SECRET)
    monkeypatch.setattr(br.scoring_provider_compat,'route_for',unavailable_route)
    result=b.execute(CONTEXT,**FRAME)
    assert result.to_document()==br._error_result('broker_unavailable',{'operation_id':'deepline.execute','provider':'deepline','funding_source':'miner_key','provider_attempt':1}).to_document()
    assert result.setup_diagnostics=={'phase':'route_resolution','error_class':'CatalogError','funding_resolved':True}
    assert SECRET not in repr(result.setup_diagnostics)
    assert not ledger.calls and not transport.sent

@pytest.mark.parametrize('funding',['host','miner_key'])
def test_successful_setup_keeps_normal_dispatch_and_no_setup_failure_metadata(funding):
    b,ledger,transport=provider('ok',funding)
    result=b.execute(CONTEXT,**FRAME)
    assert result.status==200 and result.call['funding_source']==funding
    assert result.setup_diagnostics is None
    assert len(transport.sent)==1 and len(ledger.calls)==1


def test_champion_retry_result_copy_preserves_private_setup_diagnostics(monkeypatch):
    b=_retrying_broker(monkeypatch)
    first=b._execute_once(None)
    diagnostic={'phase':'credential_resolution','error_class':'BrokerError','funding_resolved':True}
    second=replace(first,call=dict(first.call,error_code='broker_unavailable'),setup_diagnostics=diagnostic)
    replies=iter([first,second]); b._execute_once=lambda *_a,**_k:next(replies)
    result=b.execute(CONTEXT,operation_id=CHAT_FRAME['operation_id'],parameters=CHAT_FRAME['parameters'],action_sequence=12,timeout_ms=120000)
    assert len(result.attempt_trace)==2
    assert result.setup_diagnostics==diagnostic
    assert 'setup_diagnostics' not in json.dumps(result.to_document())


@pytest.mark.parametrize('mark',['absent','marked','raises','stale'])
def test_existing_four_attempt_credential_fallback_wrappers_preserve_private_facts(mark):
    b,ledger,transport=provider('ok',invalid_key=True)
    b._retry_miner_credential_for=lambda _context:True
    calls=[]
    def mark_fallback(*args):
        calls.append(args)
        if mark=='raises':raise br.BrokerError('broker_unavailable')
        return {'status':mark}
    b._mark_provider_fallback=None if mark=='absent' else mark_fallback
    result=b.execute(CONTEXT,**FRAME)
    assert len(result.attempt_trace)==4
    assert len(calls)==(0 if mark=='absent' else 1)
    assert result.setup_diagnostics=={'phase':'credential_resolution','error_class':'CredentialError','funding_resolved':True}
    assert 'setup_diagnostics'not in json.dumps(result.to_document())
    assert not ledger.calls and not transport.sent

@pytest.mark.parametrize('forged',[
    {'phase':SECRET,'error_class':SECRET,'store_error':{'operation':'rpc','resource':SECRET}},
    {'phase':'provider_funding','error_class':SECRET,'store_error':{'operation':'rpc','resource':'lab_arena_provider_funding','transport_error':SECRET,'raw':SECRET}},
])
def test_unknown_diagnostic_data_is_omitted_or_reduced_to_fixed_values(forged):
    safe=br._safe_setup_diagnostics(forged)
    assert SECRET not in json.dumps(safe)
    assert 'raw' not in json.dumps(safe)


def test_failed_retry_eligibility_keeps_original_refusal_and_private_identity_phase():
    store=SetupStore('ok')
    store.get_submission=lambda _submission_id:None
    resolver=SubmissionProviderKeys(store=store,credentials=None,organizer_keys={})
    b,ledger,transport=make_broker(retry_miner_credential_for=resolver.retry_miner_credential_for)
    result=b.execute(CONTEXT,**FRAME)
    assert result.to_document()==br._error_result('broker_unavailable',{'operation_id':'deepline.execute'}).to_document()
    assert result.setup_diagnostics=={'phase':'submission_identity','error_class':'BrokerError','funding_resolved':False}
    assert not ledger.calls and not transport.sent


def test_broken_private_projection_preserves_service_reply_and_events(monkeypatch):
    b,_,_=provider('timeout')
    expected=b.execute(CONTEXT,**FRAME)
    events=[]
    class Store:
        get_run=staticmethod(lambda _run_id:_run())
        @staticmethod
        def append_trajectory_events(_run_id,_lease_hash,batch):
            events.extend(batch);return {'status':'accepted','accepted':len(batch)}
    service=_service_for_trajectory(Store(),SimpleNamespace(execute=lambda *_a,**_k:expected))
    monkeypatch.setattr(br,'_safe_setup_diagnostics',lambda _value:(_ for _ in ()).throw(ValueError(SECRET)))
    assert service.handle_provider('run-1','lease-token',FRAME)==expected.to_document()
    assert [event['kind']for event in events]==['provider.request','provider.response']
    assert SECRET not in repr(events)


def test_optional_setup_metadata_at_cap_keeps_original_event_identity_and_content(monkeypatch):
    b,_,_=provider('timeout');expected=b.execute(CONTEXT,**FRAME)
    events=[];originals=[]
    class Store:
        get_run=staticmethod(lambda _run_id:_run())
        @staticmethod
        def append_trajectory_events(_run_id,_lease_hash,batch):
            events.extend(batch);return {'status':'accepted','accepted':len(batch)}
    service=_service_for_trajectory(Store(),SimpleNamespace(execute=lambda *_a,**_k:expected))
    original_event=trajectory.event
    def record_original(*args,**kwargs):
        event=original_event(*args,**kwargs)
        if event['kind']=='provider.response':originals.append(event)
        return event
    # A valid pre-existing response projection very close to its event cap.
    content={'one':'x'*4000,'two':'x'*3930}
    assert original_event('provider.response',dict(content,operation_id='deepline.execute',action_sequence=4))
    monkeypatch.setattr(trajectory,'provider_response_content',lambda *_a,**_k:dict(content))
    monkeypatch.setattr(trajectory,'event',record_original)
    assert service.handle_provider('run-1','lease-token',FRAME)==expected.to_document()
    assert len(originals)==1 and events[-1]==originals[0]
    assert 'setup_diagnostics' not in events[-1]['content']
