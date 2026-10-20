"""Host-owned entrypoint for one untrusted Arena source bundle."""

from __future__ import annotations

import importlib.util
import inspect
import json
import os
import sys
import tempfile
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
                recorder = module.install(lab_arena_checkpoint)
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
