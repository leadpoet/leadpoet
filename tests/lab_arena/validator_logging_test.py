"""Operational reporting does not control validator work or expose raw errors."""
import threading
import time

import pytest

from lab_arena import contracts, validator
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
    worker._executor.execute = lambda lease, token, icp: contracts.build_signed_request(
        scope=contracts.SCOPE_COMPLETE, round_id=config.round_id,
        hotkey=config.identity.hotkey,
        body={'run_id': lease['run_id']}, timestamp=int(config.clock().timestamp()),
        sign_message=config.identity.sign)
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
