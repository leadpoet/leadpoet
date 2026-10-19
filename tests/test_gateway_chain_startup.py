"""Chain-startup recovery stays bounded and never publishes an unready client."""

import ast
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from gateway import chain_startup


class Clock:
    def __init__(self):
        self.now = 0
        self.sleeps = []
        self.timeouts = []

    def time(self):
        return self.now

    async def sleep(self, delay):
        self.sleeps.append(delay)
        self.now += delay

    async def wait_for(self, operation, timeout):
        self.timeouts.append(timeout)
        start = self.now
        try:
            return await operation
        finally:
            if self.now - start > timeout:
                self.now = start + timeout
                raise asyncio.TimeoutError


class Client:
    def __init__(self, clock, error=None, duration=0, cleanup_duration=0, cleanup_error=None):
        self.clock = clock
        self.error = error
        self.duration = duration
        self.cleanup_duration = cleanup_duration
        self.cleanup_error = cleanup_error
        self.closed = False

    async def __aenter__(self):
        self.clock.now += self.duration
        if self.error:
            raise self.error
        return self

    async def close(self):
        self.clock.now += self.cleanup_duration
        if self.cleanup_error:
            raise self.cleanup_error
        self.closed = True


@pytest.fixture
def harness(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(chain_startup, "asyncio", SimpleNamespace(
        get_running_loop=lambda: clock,
        sleep=clock.sleep,
        wait_for=clock.wait_for,
        create_task=asyncio.create_task,
        shield=asyncio.shield,
        TimeoutError=asyncio.TimeoutError,
        CancelledError=asyncio.CancelledError,
    ))
    retries, failures, clients = [], [], []

    async def connect(outcomes):
        def factory(**kwargs):
            assert kwargs == {"network": "finney"}
            spec = outcomes[min(len(clients), len(outcomes) - 1)]
            if isinstance(spec, BaseException):
                raise spec
            client = Client(clock, **spec)
            clients.append(client)
            return client

        return await chain_startup.connect_async_subtensor(
            factory, network="finney", runtime_sha="fixture-sha",
            record_retry=lambda code, **fields: retries.append((code, fields)),
            capture_failure=lambda code, **fields: failures.append((code, fields)),
        )

    return SimpleNamespace(clock=clock, connect=connect, retries=retries,
                           failures=failures, clients=clients)


@pytest.mark.asyncio
async def test_first_attempt_returns_the_only_live_client(harness):
    client = await harness.connect([{}])
    assert harness.clients == [client]
    assert not client.closed
    assert not harness.retries and not harness.failures
    assert harness.clock.now == 0


@pytest.mark.asyncio
async def test_recovers_after_more_than_three_failures_and_closes_each(harness):
    client = await harness.connect([
        *[{"error": ConnectionError(), "duration": 10}] * 5, {},
    ])
    assert len(harness.clients) == 6
    assert all(failed.closed for failed in harness.clients[:-1])
    assert not client.closed
    assert harness.clock.sleeps == [5, 10, 20, 30, 30]
    assert len(harness.retries) == 5 and not harness.failures
    assert harness.clock.now == 145
    assert harness.retries[-1][1]["attempt"] == 5
    assert harness.retries[-1][1]["exception_class"] == "ConnectionError"


@pytest.mark.asyncio
async def test_persistent_network_failure_exhausts_one_fixed_deadline(harness):
    with pytest.raises(RuntimeError, match="within 480s"):
        await harness.connect([{"error": ConnectionError(), "duration": 10,
                                "cleanup_duration": 2}])
    assert harness.clock.now == 480
    assert all(client.closed for client in harness.clients)
    assert len(harness.failures) == 1
    code, failure = harness.failures[0]
    assert code == "authority.dependency_unreadable"
    assert failure["terminal"] and failure["retryable"] and failure["fail_closed"]
    assert failure["timed_out"] and failure["attempts"] == len(harness.clients)
    assert failure["stage"] == "chain_runtime_startup"
    assert failure["dependency"] == "finney" and failure["runtime_sha"] == "fixture-sha"


@pytest.mark.asyncio
async def test_per_attempt_timeout_recovers_with_cleanup(harness):
    await harness.connect([{"duration": 100}, {}])
    assert harness.clock.timeouts[:2] == [30, 5]
    assert harness.clock.now == 35
    assert harness.clients[0].closed
    assert harness.retries[0][1]["timed_out"]


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [ValueError("bad network config"), TypeError("bad API")])
@pytest.mark.parametrize("constructor", [True, False])
async def test_nonnetwork_errors_fail_promptly(harness, error, constructor):
    with pytest.raises(type(error), match=str(error)):
        await harness.connect([error if constructor else {"error": error}])
    assert harness.clock.now == 0 and not harness.retries
    assert not harness.failures[0][1]["retryable"]
    assert constructor or harness.clients[0].closed


@pytest.mark.asyncio
@pytest.mark.parametrize("cleanup", [{"cleanup_duration": 20},
                                     {"cleanup_error": RuntimeError("close failed")}])
async def test_failed_cleanup_stops_retries(harness, cleanup):
    with pytest.raises(RuntimeError, match="failed-client cleanup failed"):
        await harness.connect([{"error": ConnectionError(), **cleanup}])
    assert len(harness.clients) == 1 and not harness.retries
    assert harness.clock.now <= 5
    assert harness.failures[0][1]["cleanup_result"] == "failed"
    assert not harness.failures[0][1]["retryable"]


@pytest.mark.asyncio
async def test_real_attempt_timeout_cancels_initialization_and_closes_before_retry(monkeypatch):
    calls = []
    clients = []
    real_sleep = asyncio.sleep

    async def fast_sleep(delay):
        await real_sleep(0)

    monkeypatch.setattr(chain_startup, "ATTEMPT_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(chain_startup, "asyncio", SimpleNamespace(
        get_running_loop=asyncio.get_running_loop, wait_for=asyncio.wait_for,
        create_task=asyncio.create_task, shield=asyncio.shield, sleep=fast_sleep,
        TimeoutError=asyncio.TimeoutError, CancelledError=asyncio.CancelledError,
    ))
    pending = set(asyncio.all_tasks())

    class PendingClient:
        async def __aenter__(self):
            if len(clients) == 1:
                try:
                    await asyncio.Event().wait()
                finally:
                    calls.append("initialization_cancelled")
            calls.append("connected")
            return self

        async def close(self):
            calls.append("closed")

    def factory(**kwargs):
        client = PendingClient()
        clients.append(client)
        return client

    result = await chain_startup.connect_async_subtensor(
        factory, network="finney", runtime_sha="fixture-sha",
        record_retry=lambda *args, **kwargs: None,
        capture_failure=lambda *args, **kwargs: pytest.fail("unexpected terminal event"),
    )
    assert result is clients[1]
    assert calls == ["initialization_cancelled", "closed", "connected"]
    assert not (set(asyncio.all_tasks()) - pending)


@pytest.mark.asyncio
async def test_cancellation_closes_partially_initialized_client(monkeypatch):
    entered, closed = asyncio.Event(), asyncio.Event()
    pending = set(asyncio.all_tasks())

    class PendingClient:
        async def __aenter__(self):
            entered.set()
            await asyncio.Event().wait()

        async def close(self):
            await asyncio.sleep(0)
            closed.set()

    task = asyncio.create_task(chain_startup.connect_async_subtensor(
        lambda **kwargs: PendingClient(), network="finney", runtime_sha="fixture-sha",
        record_retry=lambda *args, **kwargs: pytest.fail("unexpected retry"),
        capture_failure=lambda *args, **kwargs: pytest.fail("unexpected terminal event"),
    ))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert closed.is_set()
    assert not (set(asyncio.all_tasks()) - pending)


@pytest.mark.asyncio
async def test_cancellation_during_failed_client_cleanup_finishes_close():
    closing, release, closed = asyncio.Event(), asyncio.Event(), asyncio.Event()
    pending = set(asyncio.all_tasks())

    class FailedClient:
        async def __aenter__(self):
            raise ConnectionError

        async def close(self):
            closing.set()
            await release.wait()
            closed.set()

    task = asyncio.create_task(chain_startup.connect_async_subtensor(
        lambda **kwargs: FailedClient(), network="finney", runtime_sha="fixture-sha",
        record_retry=lambda *args, **kwargs: pytest.fail("unexpected retry"),
        capture_failure=lambda *args, **kwargs: pytest.fail("unexpected terminal event"),
    ))
    await closing.wait()
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert closed.is_set()
    assert not (set(asyncio.all_tasks()) - pending)


@pytest.mark.asyncio
async def test_lifespan_waits_for_chain_before_crypto_background_work_and_readiness(monkeypatch):
    # Execute the actual lifespan body without importing production config,
    # routers, credentials or heavyweight SDKs.
    source = Path(__file__).resolve().parents[1] / "gateway/main.py"
    tree = ast.parse(source.read_text())
    lifespan = next(node for node in tree.body
                    if isinstance(node, ast.AsyncFunctionDef) and node.name == "lifespan")
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def initialize_identity():
        return {"enclave_pubkey": "fixture-key"}

    async def connect(*args, **kwargs):
        entered.set()
        await release.wait()
        calls.append("connected")
        return SimpleNamespace(chain_endpoint="fixture", __aexit__=close)

    async def close(*args):
        calls.append("closed")

    async def ensure_icps():
        calls.append("icps")

    async def rotate_icps():
        await asyncio.Event().wait()

    class Keypair:
        @staticmethod
        def create_from_uri(uri):
            calls.append("crypto")
            return SimpleNamespace(sign=lambda data: b"sig", verify=lambda *args: True)

    modules = {
        "gateway.utils.logger": SimpleNamespace(initialize_enclave_identity=initialize_identity),
        "gateway.utils.loop_watchdog": SimpleNamespace(start_loop_watchdog=lambda loop: None),
        "gateway.config": SimpleNamespace(BITTENSOR_NETWORK="finney"),
        "bittensor": SimpleNamespace(AsyncSubtensor=object),
        "bittensor_wallet": SimpleNamespace(Keypair=Keypair),
        "leadpoet_canonical.nitro": SimpleNamespace(verify_nitro_attestation_full=object()),
        "gateway.utils": SimpleNamespace(epoch=SimpleNamespace(
            inject_async_subtensor=lambda client: calls.append("injected"))),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(chain_startup, "connect_async_subtensor", connect)
    namespace = dict(asynccontextmanager=asynccontextmanager, FastAPI=object,
                     asyncio=asyncio, GITHUB_COMMIT="fixture-sha",
                     ensure_icp_set_exists=ensure_icps, icp_rotation_task=rotate_icps,
                     _record_sentry_retry=lambda *a, **kw: None,
                     _capture_sentry_failure=lambda *a, **kw: None)
    exec(compile(ast.Module(body=[lifespan], type_ignores=[]), str(source), "exec"), namespace)
    app = SimpleNamespace(state=SimpleNamespace())
    context = namespace["lifespan"](app)
    startup = asyncio.create_task(context.__aenter__())
    await entered.wait()
    assert not startup.done() and not calls
    assert not hasattr(app.state, "async_subtensor")
    release.set()
    await startup
    assert calls == ["connected", "crypto", "injected", "icps"]
    assert hasattr(app.state, "async_subtensor")
    await context.__aexit__(None, None, None)
    assert calls[-1] == "closed"
