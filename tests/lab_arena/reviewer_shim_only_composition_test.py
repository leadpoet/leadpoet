"""A transport-only image composition must not copy current grading code."""
import copy
import base64
import gzip
import hashlib
import http.server
import json
from pathlib import Path
import tempfile
import threading
import time
from types import ModuleType

import pytest

from scripts import build_arena_reviewer_runtime as builder
from scripts.arena_reviewer_shim_response_window_seam import (
    NEW, OLD, ShimResponseWindowSeamError, convert_shim_response_window,
)


FIXTURES = Path(__file__).resolve().parents[1] / 'fixtures' / 'reviewer_seams'
SHIM = 'lab_arena/shim.py'
OPERATIONS = 'lab_arena/operations.py'
SOURCE_HASHES = {
    SHIM: '00399bb3a8994c45f67b83abcfde781c61fdb28fe58153816be93b17c835c486',
    OPERATIONS: '69ec35c484ad7cd0017b5dae360bc2e58a55c2b38c490eec1bba6cf4dc1e0c7e',
}


def frozen_sources():
    files = {name: gzip.decompress((FIXTURES / ('shim-return-margin-' +
             Path(name).name + '.gz')).read_bytes()) for name in SOURCE_HASHES}
    assert {name: hashlib.sha256(data).hexdigest() for name, data in files.items()} == SOURCE_HASHES
    return files


def test_frozen_live_image_changes_only_reviewed_shim_anchor():
    sources = frozen_sources()
    corrected = convert_shim_response_window(sources[SHIM], sources[OPERATIONS])
    assert corrected.replace(NEW.encode(), OLD.encode(), 1) == sources[SHIM]
    # Exact same module bytes as the independently reviewed PR951 source fix.
    assert hashlib.sha256(corrected).hexdigest() == '9d95c64ca570038c8bf2acfc42807552a8e92457595e3730360f163d369d13db'
    assert convert_shim_response_window(corrected, sources[OPERATIONS]) == corrected
    assert hashlib.sha256(sources[OPERATIONS]).hexdigest() == SOURCE_HASHES[OPERATIONS]


@pytest.mark.parametrize('replacement', [b'14.0', b'15', b'True', b'"15.0"', b'15.0 + 0.0'])
def test_changed_api_transport_grace_fails_closed(replacement):
    sources = frozen_sources()
    operations = sources[OPERATIONS].replace(
        b'PROVIDER_API_TIMEOUT_GRACE_SECONDS = 15.0',
        b'PROVIDER_API_TIMEOUT_GRACE_SECONDS = ' + replacement, 1)
    with pytest.raises(ShimResponseWindowSeamError):
        convert_shim_response_window(sources[SHIM], operations)


@pytest.mark.parametrize('change', ['missing', 'duplicate', 'augmented', 'invalid', 'import', 'globals'])
def test_unknown_operations_source_fails_closed(change):
    sources = frozen_sources()
    operations = sources[OPERATIONS]
    if change == 'missing':
        operations = operations.replace(b'PROVIDER_API_TIMEOUT_GRACE_SECONDS = 15.0', b'OTHER_GRACE = 15.0')
    elif change == 'duplicate':
        operations += b'\nPROVIDER_API_TIMEOUT_GRACE_SECONDS = 15.0\n'
    elif change == 'augmented':
        operations += b'\nPROVIDER_API_TIMEOUT_GRACE_SECONDS += 1.0\n'
    elif change == 'import':
        operations += b'\nfrom external import PROVIDER_API_TIMEOUT_GRACE_SECONDS\n'
    elif change == 'globals':
        operations += b'\nglobals()["PROVIDER_API_TIMEOUT_GRACE_SECONDS"] = 30.0\n'
    else:
        operations = b'\xff'
    with pytest.raises(ShimResponseWindowSeamError):
        convert_shim_response_window(sources[SHIM], operations)


@pytest.mark.parametrize('change', ['budget', 'partial', 'duplicate_function', 'missing_function', 'newlines'])
def test_unknown_shim_source_fails_closed(change):
    sources = frozen_sources()
    shim = sources[SHIM]
    if change == 'budget':
        shim = shim.replace(b'+ bounded_timeout_ms / 1000.0', b'+ bounded_timeout_ms / 2000.0', 1)
    elif change == 'partial':
        shim = shim.replace(b'+ SOCKET_GRACE_SECONDS', b'+ operations.PROVIDER_API_TIMEOUT_GRACE_SECONDS\n            + SOCKET_GRACE_SECONDS', 1)
    elif change == 'duplicate_function':
        shim += b'\ndef _dispatch_with_response_url(): pass\n'
    elif change == 'missing_function':
        shim = shim.replace(b'def _dispatch_with_response_url(', b'def other_dispatch(')
    else:
        shim = shim.replace(b'\n', b'\r\n')
    with pytest.raises(ShimResponseWindowSeamError):
        convert_shim_response_window(shim, sources[OPERATIONS])


@pytest.mark.parametrize('flag', [
    '--complete-negative-finality', '--source-access-denied', '--source-blocked-alias',
    '--pending-final-correction', '--failed-attribute-source-recovery',
    '--completed-investigation-finality', '--identity-schema-repair',
    '--authenticated-page-refusal', '--single-tool-protocol-repair',
    '--rejected-final-packet', '--structured-homepage-numeric-alias',
    '--bound-homepage-legal-alias', '--numeric-profile-legal-self-name',
    '--archive-snapshot-https', '--greenhouse-api-source', '--auxiliary-source-name-binding',
    '--homepage-linkedin-conflict-repair', '--observed-linkedin-ampersand',
    '--verified-homepage-brand-alias',
])
def test_shim_only_rejects_every_grading_option_before_image_access(monkeypatch, flag):
    monkeypatch.setattr(builder.sys, 'argv', ['compose', '--grading-image',
        'registry.example/reviewer@sha256:' + 'a' * 64, '--shim-only', flag])
    monkeypatch.setattr(builder, 'image_inspect', lambda *_: pytest.fail('image access'))
    with pytest.raises(builder.CompositionError, match='cannot include grading'):
        builder.main()


def test_shim_only_extracts_only_two_existing_image_files(monkeypatch, tmp_path):
    monkeypatch.setattr(builder, 'checked_output', lambda *_: 'container')
    commands = []
    sources = frozen_sources()

    def run(command, **_kwargs):
        commands.append(command)
        if command[1] == 'cp':
            name = command[2].removeprefix('container:/model/')
            Path(command[3]).write_bytes(sources[name])

    monkeypatch.setattr(builder.subprocess, 'run', run)
    extracted = builder.extract_frozen_sources('pinned-image', tmp_path, shim_only=True)
    assert extracted == sources
    assert [command[2] for command in commands if command[1] == 'cp'] == [
        'container:/model/' + SHIM, 'container:/model/' + OPERATIONS]
    assert commands[-1] == ['docker', 'rm', 'container']


@pytest.mark.parametrize('verify', [False, True])
def test_shim_only_cli_builds_and_verifies_only_frozen_transport(monkeypatch, capsys, verify):
    base_reference = 'registry.example/reviewer@sha256:' + 'a' * 64
    runtime_reference = 'registry.example/reviewer@sha256:' + 'b' * 64
    argv = ['compose', '--grading-image', base_reference, '--shim-only']
    if verify:
        argv += ['--verify-image', runtime_reference]
    monkeypatch.setattr(builder.sys, 'argv', argv)
    monkeypatch.setattr(builder, 'checked_output', lambda *a, **k: 'c' * 40)
    base = {'Architecture': 'amd64', 'Os': 'linux', 'Config': {'User': '65534:65534'},
            'RootFS': {'Layers': ['base']}, 'Id': 'base-image-id'}
    candidate = copy.deepcopy(base)
    candidate['RootFS']['Layers'].append('transport-layer')
    monkeypatch.setattr(builder, 'image_inspect', lambda name: base if name == base_reference else candidate)
    extracted = []

    def extract(*args, **kwargs):
        extracted.append((args[0], kwargs))
        return frozen_sources()

    monkeypatch.setattr(builder, 'extract_frozen_sources', extract)
    # Any call would import current grader/runtime files into the candidate.
    monkeypatch.setattr(builder, 'expected_overlay', lambda *a, **k: pytest.fail('grading overlay used'))
    verified, builds = [], []
    monkeypatch.setattr(builder, 'verify_saved_overlay', lambda *a: verified.append(a))
    monkeypatch.setattr(builder.subprocess, 'run', lambda command, **kwargs: builds.append(command))
    assert builder.main() == 0
    evidence = json.loads(capsys.readouterr().out)
    assert extracted == [(base_reference, {'shim_only': True})]
    assert len(verified) == 1 and set(verified[0][2]) == {SHIM}
    assert evidence['composition_mode'] == 'shim_only_transport'
    assert evidence['base_image'] == base_reference
    assert evidence['base_image_id'] == 'base-image-id'
    assert 'grading_image' not in evidence
    assert evidence['grading_corrections'] == []
    assert evidence['transport_corrections'] == ['worker_socket_return_margin_v1']
    assert evidence['original_files'] == SOURCE_HASHES
    assert set(evidence['overlay_files']) == {SHIM}
    assert evidence['base_layers_unchanged'] and evidence['process_configuration_unchanged']
    assert len(builds) == (0 if verify else 1)
    if verify:
        assert evidence['runtime_image'] == runtime_reference
    else:
        assert '-shim-only' in evidence['runtime_image']


@pytest.mark.parametrize('corrected', [False, True])
@pytest.mark.parametrize('worker_delay,provider_delay', [(0.01, 0.02), (0.5, 1.1)])
def test_exact_frozen_transport_over_real_uds_and_http(
    monkeypatch, corrected, worker_delay, provider_delay,
):
    from lab_arena import contracts, operations, runner
    from tests.lab_arena.test_lab_arena_runner import lease

    sources = frozen_sources()
    source = (convert_shim_response_window(sources[SHIM], sources[OPERATIONS])
              if corrected else sources[SHIM])
    shim = ModuleType('_frozen_image_shim')
    exec(compile(source, 'frozen-image/lab_arena/shim.py', 'exec'), shim.__dict__)
    # Scaled transport budgets and controlled delays; actual UDS, HTTP client,
    # worker framing, observation buffer and socket writes remain in use.
    monkeypatch.setattr(operations, 'BUDGET_ADMISSION_MAX_SECONDS', 0.4)
    monkeypatch.setattr(operations, 'PROVIDER_BILLING_RECONCILIATION_SECONDS', 0.6)
    monkeypatch.setattr(operations, 'PROVIDER_API_TIMEOUT_GRACE_SECONDS', 0.3)
    monkeypatch.setattr(runner, 'PROVIDER_API_TIMEOUT_GRACE_SECONDS', 0.3)
    monkeypatch.setattr(runner, 'API_TIMEOUT_SECONDS', 0.01)
    monkeypatch.setattr(shim, 'SOCKET_GRACE_SECONDS', 0.3)
    body = b'{"choices":[]}'
    identity = contracts.hash_bytes(b'exact frozen shim transport regression')
    frames, upstream_seconds = [], []

    class LocalApi(http.server.BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            frame = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            frames.append(frame)
            started = time.monotonic()
            time.sleep(provider_delay)
            document = {'status': 200, 'headers': {}, 'body_b64': base64.b64encode(body).decode(),
                        'call': {'operation_id': 'openrouter.chat', 'provider': 'openrouter',
                                 'action_sequence': frame['action_sequence'], 'call_identity': identity,
                                 'outcome': 'settled', 'status': 200, 'provider_status': 200}}
            encoded = contracts.canonical_json(document).encode()
            self.send_response(200)
            self.send_header('Content-Length', str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)
            upstream_seconds.append(time.monotonic() - started)

    http_server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), LocalApi)
    thread = threading.Thread(target=http_server.serve_forever, daemon=True)
    thread.start()
    api = runner.HttpArenaApiClient('http://127.0.0.1:%d' % http_server.server_port)
    read_timeouts = []
    api._client.event_hooks['request'].append(
        lambda request: read_timeouts.append(request.extensions['timeout']['read']))
    original_provider = api.provider

    def delayed_provider(*args, **kwargs):
        time.sleep(worker_delay)
        return original_provider(*args, **kwargs)

    monkeypatch.setattr(api, 'provider', delayed_provider)
    state = runner.RunState(lease=lease(), lease_token='local-test-token')
    with tempfile.TemporaryDirectory(prefix='ar-frozen-shim-', dir='/tmp') as directory:
        server = runner.WorkerSocketServer(Path(directory) / 'worker.sock', api, state)
        monkeypatch.setenv(shim.WORKER_SOCKET_ENV, str(server._path))
        server.start()
        try:
            try:
                result = shim.dispatch('openrouter.chat', {
                    'model': 'openai/gpt-4o-mini',
                    'messages': [{'role': 'user', 'content': 'local response-window control'}],
                    'max_tokens': 8}, 200)
                error = None
            except shim.ShimTransportError as exc:
                result, error = None, str(exc)
            assert server.wait_for_handlers()
            assert len(frames) == len(state.calls) == state.action_sequence == 1
            assert frames[0]['timeout_ms'] == 200
            assert read_timeouts == [pytest.approx(1.5)]
            assert upstream_seconds[0] < read_timeouts[0]
            receipt = state.worker_observations.document()['response_receipts'][0]
            if corrected or worker_delay < 0.5:
                assert error is None and result[0] == 200 and result[2] == body
                assert result[1][shim.CALL_IDENTITY_HEADER] == identity
                assert receipt['write_state'] == 'socket_written'
                assert receipt['body_sha256'] == contracts.hash_bytes(body)
            else:
                # Real old-image failure: API succeeds in budget, client exits
                # first, and the worker records the failed socket write.
                assert error == 'lab arena: worker_unavailable' and result is None
                assert receipt['write_state'] == 'socket_write_failed'
        finally:
            server.stop()
            server.wait_for_handlers()
            api._client.close()
            http_server.shutdown()
            http_server.server_close()
            thread.join(2)
