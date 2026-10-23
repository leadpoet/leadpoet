"""Local sockets only: failed TLS must not poison the Arena proxy client pool."""
import asyncio
import builtins
import errno
import json
import socket
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anyio
import httpcore
import httpx
import pytest
from httpcore._sync.http_proxy import TunnelHTTPConnection
from httpcore._async.http_proxy import AsyncTunnelHTTPConnection

from lab_arena import agent_entrypoint as entry
from lab_arena.web_egress import WebEgressServer
from lab_arena.web_egress_bridge import LoopbackWebEgressBridge


@pytest.fixture(autouse=True)
def restore_constructors():
    originals = [cls.__init__ for cls in (TunnelHTTPConnection, AsyncTunnelHTTPConnection)]
    yield
    for cls, original in zip((TunnelHTTPConnection, AsyncTunnelHTTPConnection), originals):
        cls.__init__ = original


@pytest.fixture
def broken_stack():
    calls = []
    guard = threading.Lock()
    def resolver(host, port, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port))]
    def connect(host, port):
        with guard:
            calls.append((host, port))
            number = len(calls)
        if number <= 8:
            raise ConnectionRefusedError(errno.ECONNREFUSED, "simulated upstream unavailable")
        client, peer = socket.socketpair()
        def serve():
            try:
                peer.recv(65536)
                peer.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok")
            finally:
                peer.close()
        threading.Thread(target=serve, daemon=True).start()
        return client
    with tempfile.TemporaryDirectory(prefix="arena-tls-", dir="/tmp") as directory:
        server = WebEgressServer(Path(directory) / "web.sock", resolver=resolver,
                                 upstream_connector=connect, connect_timeout_seconds=1).start()
        bridge = LoopbackWebEgressBridge(server.path).start()
        try:
            yield bridge.proxy_url, server, calls
        finally:
            bridge.stop()
            server.stop()


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("concurrent", [False, True])
def test_real_arena_bridge_eight_failures_then_healthy_ninth(broken_stack, asynchronous, concurrent):
    entry._install_proxy_tls_cleanup()
    proxy, server, calls = broken_stack
    options = dict(proxy=proxy, trust_env=False, timeout=httpx.Timeout(1, pool=.1),
                   limits=httpx.Limits(max_connections=8, max_keepalive_connections=4))
    # Same origin tests competing connection setup, not only distinct hosts.
    if asynchronous:
        async def run():
            async with httpx.AsyncClient(**options) as client:
                async def fail():
                    with pytest.raises(httpx.ConnectError):
                        await client.get("https://public.example/")
                if concurrent:
                    await asyncio.gather(*(fail() for _ in range(8)))
                else:
                    for _ in range(8):
                        await fail()
                response = await client.get("http://healthy.example/")
                assert response.status_code == 200 and response.text == "ok"
        asyncio.run(run())
    else:
        with httpx.Client(**options) as client:
            def fail(_index):
                with pytest.raises(httpx.ConnectError):
                    client.get("https://public.example/")
            if concurrent:
                with ThreadPoolExecutor(max_workers=8) as pool:
                    list(pool.map(fail, range(8)))
            else:
                for i in range(8):
                    fail(i)
            response = client.get("http://healthy.example/")
            assert response.status_code == 200 and response.text == "ok"
    assert len(calls) == 9  # Exactly the caller's requests, no adapter retry.
    assert server.summary["pending_cleanup_count"] == 0


def test_old_behavior_reproduces_pool_timeout(broken_stack):
    proxy, _server, calls = broken_stack
    with httpx.Client(proxy=proxy, trust_env=False, timeout=httpx.Timeout(1, pool=.05),
                      limits=httpx.Limits(max_connections=8)) as client:
        for _ in range(8):
            with pytest.raises(httpx.ConnectError):
                client.get("https://public.example/")
        with pytest.raises(httpx.PoolTimeout):
            client.get("http://healthy.example/")
    assert len(calls) == 8


def _tunnel(asynchronous=False):
    cls = AsyncTunnelHTTPConnection if asynchronous else TunnelHTTPConnection
    return cls(proxy_origin=httpcore.Origin(b"http", b"proxy.example", 80),
               remote_origin=httpcore.Origin(b"https", b"public.example", 443))


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("cleanup_error", [False, True])
def test_original_error_is_preserved_and_cleanup_holds_lock(asynchronous, cleanup_error):
    entry._install_proxy_tls_cleanup()
    tunnel = _tunnel(asynchronous)
    original = httpcore.ConnectError("original TLS failure")
    closed = []
    class Stream:
        def start_tls(self, **kwargs):
            raise original
    class AsyncStream:
        async def start_tls(self, **kwargs):
            raise original
    class Connection:
        def handle_request(self, request):
            return httpcore.Response(200, extensions={"network_stream": Stream()})
        async def handle_async_request(self, request):
            return httpcore.Response(200, extensions={"network_stream": AsyncStream()})
        def close(self):
            assert tunnel._connect_lock.lock._lock.locked()
            closed.append(True)
            if cleanup_error:
                raise RuntimeError("cleanup failed")
        async def aclose(self):
            assert tunnel._connect_lock.lock._anyio_lock.locked()
            await anyio.sleep(0)
            closed.append(True)
            if cleanup_error:
                raise RuntimeError("cleanup failed")
    tunnel._connection = Connection()
    request = httpcore.Request("GET", "https://public.example/")
    if asynchronous:
        async def run():
            with pytest.raises(httpcore.ConnectError) as caught:
                await tunnel.handle_async_request(request)
            assert caught.value is original
        anyio.run(run)
    else:
        with pytest.raises(httpcore.ConnectError) as caught:
            tunnel.handle_request(request)
        assert caught.value is original
    assert closed == [True]


def test_async_cancellation_shields_cleanup_and_releases_lock():
    entry._install_proxy_tls_cleanup()
    async def run():
        tunnel = _tunnel(True)
        closed = []
        class Stream:
            async def start_tls(self, **kwargs):
                await anyio.sleep_forever()
        class Connection:
            async def handle_async_request(self, request):
                return httpcore.Response(200, extensions={"network_stream": Stream()})
            async def aclose(self):
                await anyio.sleep(0)  # Fails under an unshielded cancelled scope.
                closed.append(True)
        tunnel._connection = Connection()
        with anyio.move_on_after(.01) as scope:
            await tunnel.handle_async_request(httpcore.Request("GET", "https://public.example/"))
        assert scope.cancel_called and closed == [True]
        assert not tunnel._connect_lock.lock._anyio_lock.locked()
    anyio.run(run)


def test_install_is_idempotent_and_preserves_upstream_request_methods():
    methods = (TunnelHTTPConnection.handle_request, AsyncTunnelHTTPConnection.handle_async_request)
    entry._install_proxy_tls_cleanup()
    initializers = (TunnelHTTPConnection.__init__, AsyncTunnelHTTPConnection.__init__)
    entry._install_proxy_tls_cleanup()
    assert initializers == (TunnelHTTPConnection.__init__, AsyncTunnelHTTPConnection.__init__)
    assert methods == (TunnelHTTPConnection.handle_request, AsyncTunnelHTTPConnection.handle_async_request)


@pytest.mark.parametrize("incompatible", ["version", "source", "missing"])
def test_optional_unknown_or_missing_package_is_unchanged(monkeypatch, incompatible):
    originals = (TunnelHTTPConnection.__init__, AsyncTunnelHTTPConnection.__init__)
    if incompatible == "version":
        monkeypatch.setattr(httpcore, "__version__", "1.0.10")
    elif incompatible == "source":
        monkeypatch.setattr(entry.inspect, "getsource", lambda cls: "unrecognized implementation")
    else:
        import_module = builtins.__import__
        def unavailable(name, *args, **kwargs):
            if name == "httpcore":
                raise ModuleNotFoundError(name)
            return import_module(name, *args, **kwargs)
        monkeypatch.setattr(builtins, "__import__", unavailable)
    entry._install_proxy_tls_cleanup()
    assert originals == (TunnelHTTPConnection.__init__, AsyncTunnelHTTPConnection.__init__)


def test_effective_miner_import_paths_precede_install_and_harness(tmp_path, monkeypatch):
    source = tmp_path / "source"; source.mkdir()
    deps = tmp_path / "deps"; deps.mkdir()
    (source / "harness.py").write_text("def run_icp(icp):\n    return []\n")
    calls = []
    def install(source_dir, dependency_dir):
        assert source_dir == source and dependency_dir == deps
        assert str(source) in sys.path and str(deps) in sys.path
        calls.append("installed")
    monkeypatch.setattr(entry, "DEPENDENCY_DIR", deps)
    monkeypatch.setattr(entry, "_install_proxy_tls_cleanup", install)
    monkeypatch.setattr(sys, "path", list(sys.path))
    function = entry._load_run_icp(source)
    assert calls == ["installed"] and function({}) == []


@pytest.mark.parametrize("asynchronous", [False, True])
def test_successful_tls_and_requests_are_unchanged(tmp_path, asynchronous):
    import ssl
    from web_egress_test import _certificate, FakeUpstreamProxy, _stack
    entry._install_proxy_tls_cleanup()
    cert, key = _certificate(tmp_path)
    upstream = FakeUpstreamProxy(cert, key)
    with tempfile.TemporaryDirectory(prefix="arena-tls-ok-", dir="/tmp") as directory:
        server, bridge = _stack(Path(directory), upstream)
        try:
            options = dict(proxy=bridge.proxy_url, trust_env=False,
                           verify=ssl.create_default_context(cafile=str(cert)), timeout=2)
            if asynchronous:
                async def run():
                    async with httpx.AsyncClient(**options) as client:
                        response = await client.get("https://public.example/")
                        assert response.status_code == 200 and response.text == "secure hello"
                asyncio.run(run())
            else:
                with httpx.Client(**options) as client:
                    response = client.get("https://public.example/")
                    assert response.status_code == 200 and response.text == "secure hello"
            assert len(upstream.requests) == len(upstream.inner_requests) == 1
        finally:
            bridge.stop(); server.stop(); upstream.close()


def test_real_effective_dependency_version_is_seen_before_miner_import(tmp_path):
    import subprocess
    source = tmp_path / "source"; source.mkdir()
    deps = tmp_path / "deps"; deps.mkdir()
    (deps / "httpcore.py").write_text("__version__ = 'miner-compatible-other'\n")
    (source / "harness.py").write_text(
        "import httpcore\ndef run_icp(icp):\n    return [{'version': httpcore.__version__}]\n")
    program = ("import importlib.util,json; from pathlib import Path; "
               "s=importlib.util.spec_from_file_location('trusted_entry',%r); "
               "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
               "m.DEPENDENCY_DIR=Path(%r); "
               "print(json.dumps(m._load_run_icp(Path(%r))({})))") % (
                   str(Path(entry.__file__).resolve()), str(deps), str(source))
    result = subprocess.run([sys.executable, "-I", "-B", "-c", program],
                            capture_output=True, text=True, timeout=10, check=True)
    assert json.loads(result.stdout) == [{"version": "miner-compatible-other"}]


@pytest.mark.parametrize("harness_imports", [False, True])
def test_broken_optional_module_does_not_newly_fail_valid_harness(tmp_path, harness_imports):
    import subprocess
    source = tmp_path / "source"; source.mkdir()
    deps = tmp_path / "deps"; deps.mkdir()
    (deps / "httpcore.py").write_text("raise RuntimeError('custom optional module unavailable')\n")
    (source / "harness.py").write_text(
        ("import httpcore\n" if harness_imports else "") + "def run_icp(icp):\n    return []\n")
    input_path = tmp_path / "icp.json"
    input_path.write_text(json.dumps({"icp": {"id": "test"}}))
    output = tmp_path / "companies.json"
    program = """
import importlib.util,json
from pathlib import Path
s=importlib.util.spec_from_file_location('trusted_entry',%r)
m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
m.DEPENDENCY_DIR=Path(%r)
m.AUTOMATIC_CAPTURE_PATH=Path(%r)
try:
 m.run(source_dir=Path(%r),input_path=Path(%r),output_path=Path(%r))
 print('success')
except Exception as error:
 print(type(error).__name__)
""" % (str(Path(entry.__file__).resolve()), str(deps), str(tmp_path / "missing_capture.py"),
       str(source), str(input_path), str(output))
    result = subprocess.run([sys.executable, "-I", "-B", "-c", program],
                            capture_output=True, text=True, timeout=10, check=True)
    assert result.stdout.strip() == ("RuntimeError" if harness_imports else "success")


def test_custom_package_source_is_not_inspected(tmp_path, monkeypatch):
    from types import SimpleNamespace
    module = "_test_miner_httpcore"
    monkeypatch.setitem(sys.modules, module, SimpleNamespace(__file__=str(tmp_path / "httpcore.py")))
    monkeypatch.setattr(TunnelHTTPConnection, "__module__", module)
    def forbidden(_cls):
        raise AssertionError("must not inspect admitted source")
    monkeypatch.setattr(entry.inspect, "getsource", forbidden)
    originals = (TunnelHTTPConnection.__init__, AsyncTunnelHTTPConnection.__init__)
    entry._install_proxy_tls_cleanup(tmp_path, tmp_path / "deps")
    assert originals == (TunnelHTTPConnection.__init__, AsyncTunnelHTTPConnection.__init__)


def test_lock_adapter_does_not_retain_tunnel():
    import weakref
    entry._install_proxy_tls_cleanup()
    tunnel = _tunnel()
    reference = weakref.ref(tunnel)
    lock = tunnel._connect_lock
    del tunnel
    assert reference() is None and lock.owner() is None


@pytest.mark.parametrize("asynchronous", [False, True])
def test_lock_context_results_are_preserved(asynchronous):
    entry._install_proxy_tls_cleanup()
    tunnel = _tunnel(asynchronous)
    adapter = tunnel._connect_lock
    token = object()
    class Lock:
        def __enter__(self):
            return token
        def __exit__(self, *args):
            assert args == (None, None, None)
            return True
        async def __aenter__(self):
            return token
        async def __aexit__(self, *args):
            assert args == (None, None, None)
            return True
    adapter.lock = Lock()
    if asynchronous:
        async def run():
            assert await adapter.__aenter__() is token
            assert await adapter.__aexit__(None, None, None) is True
        asyncio.run(run())
    else:
        assert adapter.__enter__() is token
        assert adapter.__exit__(None, None, None) is True


def test_async_original_cancellation_object_is_preserved_if_close_fails():
    entry._install_proxy_tls_cleanup()
    async def run():
        tunnel = _tunnel(True)
        original = asyncio.CancelledError("original cancellation")
        class Stream:
            async def start_tls(self, **kwargs):
                raise original
        class Connection:
            async def handle_async_request(self, request):
                return httpcore.Response(200, extensions={"network_stream": Stream()})
            async def aclose(self):
                raise RuntimeError("close failed")
        tunnel._connection = Connection()
        with pytest.raises(asyncio.CancelledError) as caught:
            await tunnel.handle_async_request(httpcore.Request("GET", "https://public.example/"))
        assert caught.value is original
    asyncio.run(run())


def test_custom_optional_module_is_not_executed_by_discovery(tmp_path):
    import subprocess
    source = tmp_path / "source"; source.mkdir()
    deps = tmp_path / "deps"; deps.mkdir()
    marker = tmp_path / "imported"
    (deps / "httpcore.py").write_text(
        "from pathlib import Path\nPath(%r).write_text('imported')\nraise SystemExit(73)\n" % str(marker))
    (source / "harness.py").write_text("def run_icp(icp):\n    return []\n")
    program = ("import importlib.util; from pathlib import Path; "
               "s=importlib.util.spec_from_file_location('trusted_entry',%r); "
               "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
               "m.DEPENDENCY_DIR=Path(%r); "
               "assert m._load_run_icp(Path(%r))({}) == []") % (
                   str(Path(entry.__file__).resolve()), str(deps), str(source))
    subprocess.run([sys.executable, "-I", "-B", "-c", program],
                   capture_output=True, text=True, timeout=10, check=True)
    assert not marker.exists()
