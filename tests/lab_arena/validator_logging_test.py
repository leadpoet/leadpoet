"""Operational reporting does not control validator work or expose raw errors."""
import threading
import time
from dataclasses import replace
from types import SimpleNamespace

import pytest

from lab_arena import contracts, validator, validator_events
from lab_arena.runtime_host import RuntimeHostError
from lab_arena.validator_logging import ValidatorOperationalLogger


class Keypair:
    ss58_address = '5' + 'a' * 47

    def sign(self, message):
        return b'bounded-signature'


def logger(**kwargs):
    return ValidatorOperationalLogger('https://gateway.invalid', keypair=Keypair(),
                                      network='finney', netuid=71, **kwargs)


def queued(log):
    with log._lock:
        return list(log._queue)


def test_error_suppression_redaction_and_recovery():
    clock = [100.0]
    log = logger(monotonic=lambda: clock[0])
    error = RuntimeHostError('password=DO-NOT-LOG', reason='sandbox_launch_failed',
                            launch_stderr=b'Authorization: Bearer secret\nhttps://user:password@host/?token=secret')
    log.error('scoring_setup', error)
    log.error('scoring_setup', error)
    assert len(queued(log)) == 1
    text = str(queued(log))
    assert 'DO-NOT-LOG' not in text and 'user:password' not in text and 'Bearer secret' not in text
    assert queued(log)[0]['content']['reason'] == 'sandbox_launch_failed'
    assert queued(log)[0]['content']['session_id'] == log.session_id
    clock[0] += log.ERROR_INTERVAL_SECONDS
    log.error('scoring_setup', error)
    log.recovered('scoring_setup')
    log.recovered('scoring_setup')
    assert [e['kind'] for e in queued(log)] == ['validator.error', 'validator.error', 'validator.recovered']


@pytest.mark.parametrize('cause_type', (TimeoutError, ConnectionError))
def test_wrapped_errors_keep_underlying_type_without_private_messages(cause_type):
    from urllib.error import URLError

    private = 'https://private.invalid/?api_key=DO-NOT-LOG'
    root = cause_type(private)
    transport = URLError(private)
    transport.__cause__ = root
    error = RuntimeError(private)
    error.__cause__ = transport
    log = logger()
    log.error('weights', error)
    log.error('weights', error)
    assert len(queued(log)) == 1
    content = queued(log)[0]['content']
    assert content['error_class'] == 'RuntimeError'
    assert content['reason'] == 'caused_by:' + cause_type.__name__
    assert 'private.invalid' not in str(queued(log))
    assert 'DO-NOT-LOG' not in str(queued(log))


def _http_cause(kind, status):
    from urllib.error import HTTPError
    from websockets.datastructures import Headers
    from websockets.exceptions import InvalidStatus
    from websockets.http11 import Response

    private = 'https://user:password@private.invalid/?api_key=DO-NOT-LOG'
    headers = Headers({'Authorization': 'Bearer DO-NOT-LOG'})
    if kind == 'HTTPError':
        return HTTPError(private, status, 'DO-NOT-LOG', headers, None)
    return InvalidStatus(Response(status, private, headers, body=b'DO-NOT-LOG'))


@pytest.mark.parametrize('cause_kind', ('InvalidStatus', 'HTTPError'))
@pytest.mark.parametrize('status', (401, 403, 429, 500, 502, 503, 504, 599))
def test_startup_error_retains_only_typed_http_status(cause_kind, status):
    from lab_arena.chain import ArenaChainError
    from lab_arena.chain_source import ValidatorChainSourceV2Error

    error_class = ArenaChainError if cause_kind == 'InvalidStatus' else ValidatorChainSourceV2Error
    cause = _http_cause(cause_kind, status)
    error = error_class('password=DO-NOT-LOG')
    error.__cause__ = cause
    log = logger()
    log.error('startup', error)
    content = queued(log)[0]['content']
    assert content['phase'] == 'startup'
    assert content['error_class'] == error_class.__name__
    assert content['reason'] == 'caused_by:%s:http_%d' % (cause_kind, status)
    assert 'http_status' not in content
    rendered = str(queued(log))
    for private in ('private.invalid', 'user:password', 'DO-NOT-LOG', 'Authorization', 'Bearer'):
        assert private not in rendered
    assert error.__cause__ is cause


@pytest.mark.parametrize('cause_kind', ('InvalidStatus', 'HTTPError'))
@pytest.mark.parametrize('status', (None, True, False, '429', 429.0, 99, 600, -1))
def test_startup_http_status_rejects_missing_or_malformed_values(cause_kind, status):
    error = RuntimeError('DO-NOT-LOG')
    error.__cause__ = _http_cause(cause_kind, status)
    log = logger()
    log.error('startup', error)
    assert queued(log)[0]['content']['reason'] == 'caused_by:' + cause_kind
    assert 'DO-NOT-LOG' not in str(queued(log))


@pytest.mark.parametrize('cause_kind', ('InvalidStatus', 'HTTPError'))
def test_startup_http_status_rejects_absent_status_attribute(cause_kind):
    cause = _http_cause(cause_kind, 429)
    if cause_kind == 'InvalidStatus':
        del cause.response.status_code
    else:
        del cause.code
    error = RuntimeError('DO-NOT-LOG')
    error.__cause__ = cause
    log = logger()
    log.error('startup', error)
    assert queued(log)[0]['content']['reason'] == 'caused_by:' + cause_kind


def test_startup_status_does_not_trust_lookalike_exceptions_or_responses():
    from websockets.exceptions import InvalidStatus

    class LookalikeError(RuntimeError):
        code = 429
        response = SimpleNamespace(status_code=429)

    for cause in (LookalikeError('HTTP 429 DO-NOT-LOG'),
                  InvalidStatus(SimpleNamespace(status_code=429)),
                  InvalidStatus(None)):
        error = RuntimeError('DO-NOT-LOG')
        error.__cause__ = cause
        log = logger()
        log.error('startup', error)
        assert queued(log)[0]['content']['reason'] == 'caused_by:' + type(cause).__name__
        assert 'DO-NOT-LOG' not in str(queued(log))


@pytest.mark.parametrize('cause_kind', ('InvalidStatus', 'HTTPError'))
def test_http_status_does_not_replace_explicit_failure_classification(cause_kind):
    error = RuntimeError('DO-NOT-LOG')
    error.__cause__ = _http_cause(cause_kind, 403)
    log = logger()
    log.error('startup', error, reason='configuration_invalid')
    assert queued(log)[0]['content']['reason'] == 'configuration_invalid'


def test_optional_status_diagnostic_failure_keeps_original_cause(monkeypatch):
    import lab_arena.validator_logging as module

    def fail_status(_cause):
        raise RuntimeError('DO-NOT-LOG')

    monkeypatch.setattr(module, '_cause_http_status', fail_status)
    error = RuntimeError('DO-NOT-LOG')
    error.__cause__ = _http_cause('InvalidStatus', 429)
    log = logger()
    log.error('startup', error)
    assert queued(log)[0]['content']['reason'] == 'caused_by:InvalidStatus'
    assert 'DO-NOT-LOG' not in str(queued(log))


def test_cause_logging_preserves_explicit_reason_and_suppressed_context():
    error = RuntimeError('private message')
    error.__context__ = TimeoutError('private URL')
    log = logger()
    log.error('weights', error, reason='prior_epoch_25770')
    assert queued(log)[0]['content']['reason'] == 'prior_epoch_25770'
    log.error('other', error)
    assert queued(log)[1]['content']['reason'] == 'caused_by:TimeoutError'
    error.__suppress_context__ = True
    log.error('suppressed', error)
    assert 'reason' not in queued(log)[2]['content']


def test_cause_logging_bounds_cycles_depth_and_type_length():
    from lab_arena.validator_logging import _cause_reason

    error = RuntimeError('private')
    cause = TimeoutError('private')
    error.__cause__ = cause
    cause.__cause__ = error
    assert _cause_reason(error) == 'caused_by:TimeoutError'
    deep = error = RuntimeError('private')
    for _ in range(8):
        deep.__cause__ = ValueError('private')
        deep = deep.__cause__
    deep.__cause__ = TimeoutError('private')
    assert _cause_reason(error) == 'caused_by:ValueError'
    error.__cause__ = type('Long' * 60, (Exception,), {})('private')
    assert len(_cause_reason(error)) == len('caused_by:') + 128


def test_changed_underlying_failure_is_not_hidden_by_error_deduplication():
    log = logger()
    for cause in (TimeoutError(), ConnectionError()):
        error = RuntimeError('private')
        error.__cause__ = cause
        log.error('weights', error)
    assert [item['content']['reason'] for item in queued(log)] == [
        'caused_by:TimeoutError', 'caused_by:ConnectionError',
    ]


def test_gateway_unavailable_keeps_queue_bounded_and_close_bounded():
    attempted = threading.Event()

    def unavailable(envelope):
        assert envelope['scope'] == contracts.SCOPE_VALIDATOR_EVENTS
        assert envelope['round_id'] == 'validator-events'
        assert envelope['body']['network'] == 'finney'
        attempted.set()
        raise OSError('sensitive upstream URL')

    log = logger(transport=unavailable)
    for index in range(200):
        log.emit('validator.state', {'attempt': index})
    assert len(queued(log)) == log.QUEUE_LIMIT
    log.start()
    assert attempted.wait(1)
    started = time.monotonic()
    log.close()
    assert time.monotonic() - started < 1
    assert not log._thread.is_alive()
    assert len(queued(log)) <= log.QUEUE_LIMIT


def test_active_work_is_not_a_stalled_poll():
    clock = [0.0]
    log = logger(monotonic=lambda: clock[0])
    log.state(scoring_state='claiming', progress=True)
    clock[0] = 301
    assert log.snapshot()['scoring_state'] == 'claiming'
    assert log.snapshot()['last_progress_at']
    log.activity(1)
    log.state(scoring_state='discovering')
    clock[0] = 7200
    assert log.snapshot()['scoring_state'] == 'active'
    assert log.snapshot()['active_runs'] == 1


def test_claim_observations_track_effective_capacity_and_bounded_result(tmp_path):
    from lab_arena import runner
    from tests.lab_arena.test_lab_arena_runner import FakeApi, make_config, lease

    log = logger()
    pool = SimpleNamespace(capacity=5, quarantined=0)
    api = FakeApi([{'status': 'no_free_slot'}, {'status': 'no_pending'},
                   {'status': 'paused'}, lease(),
                   {'status': 'secret=DO-NOT-LOG'}])
    config = replace(make_config(tmp_path, api, None, parallel=5),
                     proxy_worker_pool=pool, operational_logger=log)
    worker = runner.Runner(config)
    try:
        assert log.snapshot()['usable_slots'] == 5
        assert log.snapshot()['quarantined_slots'] == 0
        pool.quarantined = 4
        assert worker.claim_one()['status'] == 'no_free_slot'
        assert log.snapshot()['usable_slots'] == 1
        assert log.snapshot()['quarantined_slots'] == 4
        assert log.snapshot()['last_claim_status'] == 'no_free_slot'
        assert log.snapshot()['last_claim_round_id'] == config.round_id
        pool.quarantined = 3
        assert worker.claim_one()['status'] == 'no_pending'
        assert log.snapshot()['usable_slots'] == 2
        assert log.snapshot()['last_claim_status'] == 'no_pending'
        pool.quarantined = 0
        assert worker.claim_one()['status'] == 'paused'
        assert log.snapshot()['usable_slots'] == 5
        assert log.snapshot()['last_claim_status'] == 'paused'
        assert worker.claim_one()['status'] == 'leased'
        assert log.snapshot()['usable_slots'] == 5
        assert log.snapshot()['last_claim_status'] == 'leased'
        assert worker.claim_one()['status'] == 'secret=DO-NOT-LOG'
        assert log.snapshot()['last_claim_status'] == 'other'
        assert 'DO-NOT-LOG' not in str(log.snapshot())
        assert [item['body']['declared_parallelism'] for item in api.claims] == [1, 2, 5, 5, 5]
        pool.quarantined = 5
        assert worker.run_once() == 0
        assert len(api.claims) == 5
        assert log.snapshot()['usable_slots'] == 0
        assert log.snapshot()['quarantined_slots'] == 5
        assert log.snapshot()['last_claim_status'] == 'no_local_capacity'
        assert log.snapshot()['last_claim_round_id'] == config.round_id
        assert validator_events.event('validator.state', log.snapshot())['content']['usable_slots'] == 0
    finally:
        worker.close()


def test_claim_observation_rejects_unbounded_values():
    for content in ({'usable_slots': -1}, {'quarantined_slots': 252},
                    {'last_claim_status': 'secret=DO-NOT-LOG'},
                    {'last_claim_round_id': 'secret=DO-NOT-LOG'}):
        with pytest.raises(validator_events.ValidatorEventError):
            validator_events.event('validator.state', content)


def test_scoring_broken_weights_continue_with_error_state():
    log = logger()
    weights = []

    class Orchestrator:
        def run_once(self, epoch):
            weights.append(epoch)
            return 'state_unavailable'

        def poll_prior_outcomes(self, epoch):
            pass

    def broken():
        raise RuntimeHostError(reason='runsc_missing')

    validator.run_validator_loops(orchestrator=Orchestrator(), runner_factory=broken,
                                  epoch_supplier=lambda: 1, stop=threading.Event(), once=True,
                                  operational_logger=log)
    assert weights == [1]
    assert log.snapshot()['weights_state'] == 'state_unavailable'
    assert any(e['kind'] == 'validator.error' and e['content']['phase'] == 'scoring_setup'
               for e in queued(log))


def test_heartbeat_runs_while_scoring_run_once_blocks():
    heartbeat = threading.Event()
    entered = threading.Event()
    release = threading.Event()
    stop = threading.Event()

    def delivered(envelope):
        if any(e['kind'] == 'validator.state' for e in envelope['body']['events']):
            heartbeat.set()

    log = logger(transport=delivered)
    log.HEARTBEAT_SECONDS = .03
    log.start()

    class Orchestrator:
        def run_once(self, epoch):
            return 'state_unavailable'

        def poll_prior_outcomes(self, epoch):
            pass

    class Runner:
        def run_once(self, **kwargs):
            log.activity(1)
            entered.set()
            assert release.wait(2)
            log.activity(-1)
            return 1

        def close(self):
            pass

    thread = threading.Thread(target=validator.run_validator_loops, kwargs=dict(
        orchestrator=Orchestrator(), runner_factory=Runner, epoch_supplier=lambda: 1,
        stop=stop, once=True, operational_logger=log))
    try:
        thread.start()
        assert entered.wait(1)
        assert heartbeat.wait(1)
        assert log.snapshot()['active_runs'] == 1
        assert log.snapshot()['weights_state'] == 'state_unavailable'
    finally:
        release.set()
        thread.join(2)
        log.close()
    assert not thread.is_alive()


def test_startup_failure_after_identity_before_chain_is_delivered(monkeypatch):
    from lab_arena import chain, validator_startup

    delivered = []
    wallets = []
    monkeypatch.setattr(validator_startup, 'maybe_reexec_rootful', lambda *args: None)
    monkeypatch.setattr(validator, 'load_local_hotkey', lambda args: wallets.append(1) or Keypair())
    def fail_chain(config):
        raise OSError('password=NEVER-LOG-THIS')
    monkeypatch.setattr(chain, 'connect_substrate', fail_chain)
    real_logger = ValidatorOperationalLogger
    import lab_arena.validator_logging as module
    monkeypatch.setattr(module, 'ValidatorOperationalLogger', lambda *a, **kw:
                        real_logger(*a, **kw, transport=lambda envelope: delivered.extend(envelope['body']['events'])))
    with pytest.raises(OSError):
        validator.main([])
    assert wallets == [1]
    assert any(e['kind'] == 'validator.startup' for e in delivered)
    assert any(e['kind'] == 'validator.error' and e['content']['phase'] == 'startup' for e in delivered)
    assert 'NEVER-LOG-THIS' not in str(delivered)


@pytest.mark.parametrize('status', (401, 403, 429, 503))
def test_startup_connection_http_status_is_delivered_without_retry(monkeypatch, status):
    import sys
    from lab_arena import chain, validator_startup
    import lab_arena.validator_logging as module

    delivered = []
    connects = []
    cause = _http_cause('InvalidStatus', status)

    def fail_connect(**kwargs):
        connects.append(1)
        raise cause

    monkeypatch.setitem(sys.modules, 'async_substrate_interface',
                        SimpleNamespace(SubstrateInterface=fail_connect))
    monkeypatch.setattr(validator_startup, 'maybe_reexec_rootful', lambda *args: None)
    monkeypatch.setattr(validator, 'load_local_hotkey', lambda args: Keypair())
    real_logger = ValidatorOperationalLogger
    monkeypatch.setattr(module, 'ValidatorOperationalLogger', lambda *a, **kw:
                        real_logger(*a, **kw, transport=lambda envelope: delivered.extend(envelope['body']['events'])))
    with pytest.raises(chain.ArenaChainError) as caught:
        validator.main([])
    assert caught.value.__cause__ is cause
    assert connects == [1]
    assert not any(e['kind'] == 'validator.ready' for e in delivered)
    errors = [e['content'] for e in delivered if e['kind'] == 'validator.error']
    assert len(errors) == 1
    assert errors[0]['error_class'] == 'ArenaChainError'
    assert errors[0]['phase'] == 'startup'
    assert errors[0]['reason'] == 'caused_by:InvalidStatus:http_%d' % status
    for private in ('private.invalid', 'user:password', 'DO-NOT-LOG', 'Authorization', 'Bearer'):
        assert private not in str(delivered)


def test_claim_and_completion_errors_have_run_correlation_and_recover(tmp_path):
    from dataclasses import replace
    from lab_arena import runner
    from tests.lab_arena.test_lab_arena_runner import FakeApi, lease, make_config

    log = logger()
    claimed = lease('bounded-run')
    class Api(FakeApi):
        failures = 1
        completion_failures = 1

        def claim(self, envelope):
            if self.failures:
                self.failures -= 1
                raise runner.RunnerError('https://user:secret@private.invalid', http_status=503)
            return super().claim(envelope)

        def complete(self, envelope):
            if self.completion_failures:
                self.completion_failures -= 1
                raise runner.RunnerError('private-raw-error')
            return super().complete(envelope)

    api = Api([claimed])
    config = replace(make_config(tmp_path, api, None), operational_logger=log,
                     claim_retry_seconds=(0,), completion_retry_seconds=(0,))
    worker = runner.Runner(config)
    # The runtime trajectory is separately tested. Isolate operational delivery
    # around claim and completion so this test performs no sandbox or provider work.
    def execute(lease, token, icp, *, before_finished):
        document = contracts.build_signed_request(
            scope=contracts.SCOPE_COMPLETE, round_id=config.round_id,
            hotkey=config.identity.hotkey,
            body={'run_id': lease['run_id']}, timestamp=int(config.clock().timestamp()),
            sign_message=config.identity.sign)
        before_finished(document)
        return document
    worker._executor.execute = execute
    try:
        assert worker.run_once() == 1
    finally:
        worker.close()
    errors = [e for e in queued(log) if e['kind'] == 'validator.error']
    assert {e['content']['phase'] for e in errors} == {'claim', 'completion'}
    completion = next(e for e in errors if e['content']['phase'] == 'completion')
    assert completion['run_id'] == 'bounded-run'
    assert completion['round_id'] == config.round_id
    assert {e['content']['phase'] for e in queued(log) if e['kind'] == 'validator.recovered'} == {'claim', 'completion'}
    assert 'private-raw-error' not in str(queued(log)) and 'user:secret' not in str(queued(log))
    assert worker.abandoned == 0
    assert log.snapshot()['active_runs'] == 0
    assert log.snapshot()['last_completion_at']


def test_transport_uses_direct_origin_no_redirect_or_response_body(monkeypatch):
    log = logger()
    requests = []
    class Response:
        status = 200
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def read(self, *args):
            pytest.fail('event sender must not read response bodies')
    def opened(request, **kwargs):
        requests.append((request, kwargs))
        return Response()
    monkeypatch.setattr(log._opener, 'open', opened)
    log._post({'bounded': True})
    assert requests[0][0].full_url == 'https://gateway.invalid/arena/v1/validators/events'
    assert requests[0][1]['timeout'] == 3
    import urllib.request
    from lab_arena.validator_logging import _NoRedirect
    assert any(isinstance(handler, _NoRedirect) for handler in log._opener.handlers)
    assert not any(isinstance(handler, urllib.request.ProxyHandler) and handler.proxies for handler in log._opener.handlers)
    invalid = ValidatorOperationalLogger('http://remote.invalid', keypair=Keypair(), network='finney', netuid=71)
    with pytest.raises(OSError, match='origin invalid'):
        invalid._post({})


def test_repeated_claim_denial_does_not_fabricate_recovery(tmp_path):
    from dataclasses import replace
    from lab_arena import runner
    from tests.lab_arena.test_lab_arena_runner import FakeApi, make_config

    log = logger()
    class Api(FakeApi):
        def claim(self, envelope):
            return {'status': 'rejected', 'code': 'validator_participation_required'}
    config = replace(make_config(tmp_path, Api([]), None), operational_logger=log)
    worker = runner.Runner(config)
    try:
        assert worker.run_once() == 0
        assert worker.run_once() == 0
    finally:
        worker.close()
    assert [e['kind'] for e in queued(log)] == ['validator.error']
    assert log.snapshot()['scoring_state'] == 'unavailable'
    assert 'last_progress_at' not in log.snapshot()


def test_unavailable_sender_thread_cannot_prevent_work(monkeypatch):
    log = logger()
    monkeypatch.setattr(threading.Thread, 'start', lambda self: (_ for _ in ()).throw(RuntimeError('thread unavailable')))
    log.start()
    assert log._thread is None
    log.error('scoring_setup', ValueError('never log arbitrary message'))
    assert len(queued(log)) == 1
    log.close()


def test_shutdown_flushes_error_queued_during_inflight_startup_request():
    entered = threading.Event()
    release = threading.Event()
    delivered = []

    def transport(envelope):
        if not entered.is_set():
            entered.set()
            assert release.wait(1)
        delivered.extend(envelope['body']['events'])

    log = logger(transport=transport)
    log.emit('validator.startup', {'phase': 'startup'})
    log.start()
    assert entered.wait(1)
    log.error('startup', OSError('private error'))
    closer = threading.Thread(target=log.close)
    closer.start()
    assert log._stop.wait(1)
    release.set()
    closer.join(1)
    assert not closer.is_alive()
    assert not log._thread.is_alive()
    assert [e['kind'] for e in delivered] == ['validator.startup', 'validator.error', 'validator.stopping']


def test_concurrent_run_errors_have_independent_deduplication():
    from concurrent.futures import ThreadPoolExecutor

    log = logger(monotonic=lambda: 100.0)
    ready = threading.Barrier(2)
    def report(run_id):
        ready.wait(timeout=2)
        log.error('completion', RuntimeError('private'), run_id=run_id,
                  round_id='arena-2026-10-16')
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(report, ('run-a', 'run-b')))
    assert {event['run_id'] for event in queued(log)} == {'run-a', 'run-b'}
    assert all(event['kind'] == 'validator.error' for event in queued(log))


def test_one_run_recovery_does_not_clear_another_run_failure():
    log = logger(monotonic=lambda: 100.0)
    scope = {'round_id': 'arena-2026-10-16'}
    log.error('completion', RuntimeError('private'), run_id='run-a', **scope)
    log.error('completion', RuntimeError('private'), run_id='run-b', **scope)
    log.recovered('completion', run_id='run-b', **scope)
    log.error('completion', RuntimeError('private'), run_id='run-a', **scope)
    assert [(event['kind'], event['run_id']) for event in queued(log)] == [
        ('validator.error', 'run-a'), ('validator.error', 'run-b'),
        ('validator.recovered', 'run-b'),
    ]
    log.recovered('completion', run_id='run-a', **scope)
    log.recovered('completion', run_id='run-a', **scope)
    assert [(event['kind'], event['run_id']) for event in queued(log)][-1] == (
        'validator.recovered', 'run-a')
    assert len(queued(log)) == 4
    log.error('completion', RuntimeError('private'), run_id='run-a', **scope)
    assert len(queued(log)) == 5


def test_unrelated_success_cannot_claim_recovery_without_its_own_error():
    log = logger(monotonic=lambda: 100.0)
    log.error('completion', RuntimeError('private'), run_id='run-a', round_id='round-a')
    log.recovered('completion', run_id='run-b', round_id='round-a')
    log.error('completion', RuntimeError('private'), run_id='run-a', round_id='round-a')
    assert [event['kind'] for event in queued(log)] == ['validator.error']


def test_same_run_error_stays_throttled_until_interval_or_matching_recovery():
    clock = [100.0]
    log = logger(monotonic=lambda: clock[0])
    scope = {'run_id': 'run-a', 'round_id': 'round-a'}
    log.error('completion', RuntimeError('private'), **scope)
    clock[0] += log.ERROR_INTERVAL_SECONDS - 1
    log.error('completion', RuntimeError('private'), **scope)
    assert len(queued(log)) == 1
    clock[0] += 1
    log.error('completion', RuntimeError('private'), **scope)
    assert len(queued(log)) == 2
    log.recovered('completion', **scope)
    log.error('completion', RuntimeError('private'), **scope)
    assert [event['kind'] for event in queued(log)] == [
        'validator.error', 'validator.error', 'validator.recovered', 'validator.error',
    ]


def test_round_identity_is_part_of_error_and_recovery_scope():
    log = logger(monotonic=lambda: 100.0)
    for round_id in ('round-a', 'round-b'):
        log.error('claim', RuntimeError('private'), round_id=round_id)
    log.recovered('claim', round_id='round-b')
    log.error('claim', RuntimeError('private'), round_id='round-a')
    assert [(event['kind'], event['round_id']) for event in queued(log)] == [
        ('validator.error', 'round-a'), ('validator.error', 'round-b'),
        ('validator.recovered', 'round-b'),
    ]


def test_global_startup_recovery_only_clears_global_errors():
    log = logger(monotonic=lambda: 100.0)
    log.error('startup', RuntimeError('private'))
    log.error('startup', RuntimeError('private'))
    log.error('startup', RuntimeError('private'), run_id='run-a', round_id='round-a')
    log.recovered('startup')
    log.recovered('startup')
    log.error('startup', RuntimeError('private'), run_id='run-a', round_id='round-a')
    events = queued(log)
    assert [event['kind'] for event in events] == [
        'validator.error', 'validator.error', 'validator.recovered',
    ]
    assert 'run_id' not in events[0] and 'run_id' not in events[-1]
    assert events[1]['run_id'] == 'run-a'
    log.recovered('startup', run_id='run-a', round_id='round-a')
    assert queued(log)[-1]['run_id'] == 'run-a'


def test_many_run_scopes_keep_error_and_event_queues_bounded():
    log = logger(monotonic=lambda: 100.0)
    for index in range(200):
        log.error('completion', RuntimeError('private'), run_id='run-%d' % index,
                  round_id='round-a')
    assert len(log._errors) == log.QUEUE_LIMIT == 64
    assert len(queued(log)) == log.QUEUE_LIMIT
    assert {event['run_id'] for event in queued(log)} == {
        'run-%d' % index for index in range(136, 200)}
    log.recovered('completion', run_id='run-199', round_id='round-a')
    assert len(log._errors) == 63 and len(queued(log)) == 64


def test_failed_claim_then_successful_lease_recovers_the_round_scope(tmp_path, monkeypatch):
    from lab_arena import runner
    from tests.lab_arena.test_lab_arena_runner import FakeApi, lease, make_config

    log = logger()
    calls = []
    recovered = log.recovered
    def observe(phase, **scope):
        calls.append((phase, scope))
        recovered(phase, **scope)
    monkeypatch.setattr(log, 'recovered', observe)
    class Api(FakeApi):
        failures = 1
        def claim(self, envelope):
            if self.failures:
                self.failures -= 1
                raise runner.RunnerError('private error', http_status=503)
            return super().claim(envelope)
    api = Api([lease('picked-up-run')])
    config = replace(make_config(tmp_path, api, None), operational_logger=log,
                     claim_retry_seconds=())
    worker = runner.Runner(config)
    def execute(assigned, token, icp, *, before_finished):
        document = contracts.build_signed_request(
            scope=contracts.SCOPE_COMPLETE, round_id=config.round_id,
            hotkey=config.identity.hotkey, body={'run_id': assigned['run_id']},
            timestamp=int(config.clock().timestamp()), sign_message=config.identity.sign)
        before_finished(document)
        return document
    worker._executor.execute = execute
    try:
        assert worker.run_once(max_claims=1) == 0
        assert worker.run_once(max_claims=1) == 1
    finally:
        worker.close()
    claim_events = [event for event in queued(log) if event['content'].get('phase') == 'claim']
    assert [event['kind'] for event in claim_events] == ['validator.error', 'validator.recovered']
    assert all(event['round_id'] == config.round_id and 'run_id' not in event for event in claim_events)
    # Both the inner availability path and the outer successful pickup must
    # match the request's round scope, before a particular run was assigned.
    claim_calls = [scope for phase, scope in calls if phase == 'claim']
    assert len(claim_calls) == 2
    assert all(scope == {'round_id': config.round_id} for scope in claim_calls)
    assert worker.abandoned == 0
