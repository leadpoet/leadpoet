"""Host-owned entrypoint for one untrusted Arena source bundle."""

from __future__ import annotations

import functools
import hashlib
import importlib.util
import inspect
import json
import os
import sys
import tempfile
import weakref
from contextlib import contextmanager
from pathlib import Path
from typing import Any

SOURCE_DIR = Path("/agent/source")
DEPENDENCY_DIR = Path("/agent/deps")
INPUT_PATH = Path("/input/icp.json")
OUTPUT_PATH = Path("/output/companies.json")
WEB_BRIDGE_PATH = Path("/agent/web_egress_bridge.py")
AUTOMATIC_CAPTURE_PATH = Path("/agent/automatic_capture.py")
AUTOMATIC_CAPTURE_SUMMARY = ".arena-automatic-capture.json"


class AgentContractError(RuntimeError):
    """The submitted harness does not implement the public boundary."""


def _install_proxy_tls_cleanup(source_dir: Path = SOURCE_DIR, dependency_dir: Path = DEPENDENCY_DIR) -> None:
    """Repair httpcore 1.0.9's failed CONNECT/TLS setup, without retrying.

    Retire this adapter when an official httpcore release fixes upstream #921
    and the proxy failure/cancellation regressions pass against that release.
    Unknown or miner-supplied implementations keep their original behavior.
    """
    try:
        specification = importlib.util.find_spec("httpcore")
        if specification is None or not specification.origin:
            return
        origin = Path(specification.origin).resolve()
        if any(origin.is_relative_to(root.resolve()) for root in (source_dir, dependency_dir)):
            return
        import httpcore
        from httpcore._sync.http_proxy import TunnelHTTPConnection
        from httpcore._async.http_proxy import AsyncTunnelHTTPConnection
        from httpcore._synchronization import AsyncShieldCancellation
        if httpcore.__version__ != "1.0.9":
            return
        classes = (
            (TunnelHTTPConnection, "4a4a9a364dbcb188f7949db92f658695618be323348bd0b79539006b9650961f", False),
            (AsyncTunnelHTTPConnection, "7027fbe210cc8c4d384301f65b2ce94846fbeeef7db29f9edcfde69af0a48664", True),
        )
        for cls, expected, _asynchronous in classes:
            location = Path(sys.modules[cls.__module__].__file__).resolve()
            if any(location.is_relative_to(root.resolve()) for root in (source_dir, dependency_dir)):
                return
            if hashlib.sha256(inspect.getsource(cls).encode()).hexdigest() != expected:
                return
    except Exception:
        # Optional compatibility must not reject another dependency tree.
        return

    class CleanupLock:
        def __init__(self, owner, lock):
            self.owner, self.lock = weakref.ref(owner), lock

        def __enter__(self):
            return self.lock.__enter__()

        def __exit__(self, kind, error, traceback):
            try:
                owner = self.owner()
                if kind is not None and owner is not None and not owner._connected:
                    owner._connection.close()
            except BaseException:
                # Cleanup must not replace the original request error.
                pass
            return self.lock.__exit__(kind, error, traceback)

    class AsyncCleanupLock(CleanupLock):
        async def __aenter__(self):
            return await self.lock.__aenter__()

        async def __aexit__(self, kind, error, traceback):
            try:
                owner = self.owner()
                if kind is not None and owner is not None and not owner._connected:
                    with AsyncShieldCancellation():
                        await owner._connection.aclose()
            except BaseException:
                pass
            return await self.lock.__aexit__(kind, error, traceback)

    def adapt(original, lock_class):
        @functools.wraps(original)
        def initialize(self, *args, **kwargs):
            original(self, *args, **kwargs)
            self._connect_lock = lock_class(self, self._connect_lock)
        initialize._arena_proxy_tls_cleanup = True
        return initialize

    for cls, _expected, asynchronous in classes:
        if not getattr(cls.__init__, "_arena_proxy_tls_cleanup", False):
            cls.__init__ = adapt(cls.__init__, AsyncCleanupLock if asynchronous else CleanupLock)


@contextmanager
def _automatic_capture(output_dir: Path):
    """Install host-owned observation before submitted imports; never replace results."""
    recorder = None
    try:
        if AUTOMATIC_CAPTURE_PATH.is_file():
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            import lab_arena_checkpoint

            specification = importlib.util.spec_from_file_location(
                "_arena_automatic_capture", AUTOMATIC_CAPTURE_PATH
            )
            if specification is not None and specification.loader is not None:
                module = importlib.util.module_from_spec(specification)
                sys.modules[specification.name] = module
                specification.loader.exec_module(module)
                recorder = module.install(lab_arena_checkpoint, handoff_dir=output_dir)
    except Exception:
        # Missing observation is an evidence gap, not an agent execution error.
        recorder = None
    try:
        yield
    finally:
        summary = {"source": "runtime_http_v1", "complete": False, "failure": "recorder_unavailable"}
        if recorder is not None:
            try:
                summary = recorder.close()
            except Exception:
                summary["failure"] = "recorder_close_failed"
        # A tiny atomic sidecar survives interpreter exit even if the daemon
        # cannot finish its queue. The host folds it into the existing artifact.
        temporary = None
        try:
            encoded = json.dumps(summary, separators=(",", ":"), allow_nan=False).encode()
            if len(encoded) <= 16_384:
                descriptor, temporary = tempfile.mkstemp(prefix=".arena-capture-", dir=output_dir)
                with os.fdopen(descriptor, "wb") as handle:
                    handle.write(encoded)
                os.replace(temporary, output_dir / AUTOMATIC_CAPTURE_SUMMARY)
        except Exception:
            pass
        finally:
            if temporary is not None:
                try:
                    os.unlink(temporary)
                except OSError:
                    pass


@contextmanager
def _web_access():
    socket_path = os.environ.get("LAB_ARENA_WEB_EGRESS_SOCKET", "")
    if not socket_path:
        yield
        return
    specification = importlib.util.spec_from_file_location("_arena_web_bridge", WEB_BRIDGE_PATH)
    if specification is None or specification.loader is None:
        raise AgentContractError("trusted web bridge is unavailable")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    with module.LoopbackWebEgressBridge(socket_path) as bridge:
        previous = {name: os.environ.get(name) for name in bridge.proxy_environment}
        os.environ.update(bridge.proxy_environment)
        try:
            yield
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value


def _load_run_icp(source_dir: Path) -> Any:
    harness_path = source_dir / "harness.py"
    if not harness_path.is_file():
        raise AgentContractError("harness.py is missing")
    sys.path.insert(0, str(DEPENDENCY_DIR))
    sys.path.insert(0, str(source_dir))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    _install_proxy_tls_cleanup(source_dir, DEPENDENCY_DIR)
    specification = importlib.util.spec_from_file_location(
        "_lab_arena_submission_harness", harness_path
    )
    if specification is None or specification.loader is None:
        raise AgentContractError("harness.py cannot be loaded")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    function = getattr(module, "run_icp", None)
    if not callable(function) or inspect.iscoroutinefunction(function):
        raise AgentContractError("harness.run_icp must be synchronous")
    try:
        parameters = list(inspect.signature(function).parameters.values())
    except (TypeError, ValueError) as exc:
        raise AgentContractError("harness.run_icp signature is unavailable") from exc
    positional = [
        parameter
        for parameter in parameters
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    if (
        len(positional) != 1
        or any(parameter.kind == inspect.Parameter.KEYWORD_ONLY for parameter in parameters)
        or any(
            parameter.kind
            in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
            for parameter in parameters
        )
    ):
        raise AgentContractError("harness.run_icp must take exactly one positional input")
    return function


def run(
    *,
    source_dir: Path = SOURCE_DIR,
    input_path: Path = INPUT_PATH,
    output_path: Path = OUTPUT_PATH,
) -> None:
    """Call ``harness.run_icp(icp)`` and write the standard output document."""

    with _automatic_capture(output_path.parent):
        _run(source_dir=source_dir, input_path=input_path, output_path=output_path)


def _run(*, source_dir: Path, input_path: Path, output_path: Path) -> None:
    document = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or not isinstance(document.get("icp"), dict):
        raise AgentContractError("ICP input is invalid")
    company_limit = document.get("company_limit", 5)
    if (
        isinstance(company_limit, bool)
        or not isinstance(company_limit, int)
        or not 1 <= company_limit <= 5
    ):
        raise AgentContractError("company_limit must be an integer from 1 through 5")
    previous_limit = os.environ.get("LAB_ARENA_COMPANY_LIMIT")
    os.environ["LAB_ARENA_COMPANY_LIMIT"] = str(company_limit)
    try:
        result = _load_run_icp(source_dir)(dict(document["icp"]))
    finally:
        if previous_limit is None:
            os.environ.pop("LAB_ARENA_COMPANY_LIMIT", None)
        else:
            os.environ["LAB_ARENA_COMPANY_LIMIT"] = previous_limit
    if inspect.isawaitable(result):
        raise AgentContractError("harness.run_icp must be synchronous")
    if not isinstance(result, list) or any(not isinstance(item, dict) for item in result):
        raise AgentContractError("harness.run_icp must return a list of company objects")
    import lab_arena_checkpoint

    lab_arena_checkpoint.write(result, output_path=output_path)


def main() -> int:
    with _web_access():
        run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
