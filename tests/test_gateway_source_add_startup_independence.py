"""SOURCE_ADD startup must not depend on scoring/autoresearch workers."""

from __future__ import annotations

import ast
import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
MAIN_PATH = ROOT / "gateway" / "main.py"


def _main_tree() -> ast.Module:
    return ast.parse(MAIN_PATH.read_text(encoding="utf-8"))


def _load_function(name: str, **namespace):
    node = next(
        candidate
        for candidate in _main_tree().body
        if isinstance(candidate, (ast.FunctionDef, ast.AsyncFunctionDef))
        and candidate.name == name
    )
    node.decorator_list = []
    compiled = compile(
        ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
        str(MAIN_PATH),
        "exec",
    )
    values = {"asyncio": asyncio, **namespace}
    exec(compiled, values)
    return values[name]


@pytest.mark.asyncio
async def test_dispatcher_starts_and_survives_worker_startup_failure() -> None:
    start_dispatcher = _load_function("_start_source_add_dispatcher_task")
    observe_workers = _load_function(
        "_observe_research_lab_worker_startup_task"
    )
    application = SimpleNamespace(state=SimpleNamespace())
    dispatcher_started = asyncio.Event()
    keep_dispatcher_running = asyncio.Event()

    async def dispatcher() -> None:
        dispatcher_started.set()
        await keep_dispatcher_running.wait()

    config = SimpleNamespace(
        source_add_enabled=True,
        source_add_dispatcher_enabled=True,
    )
    dispatcher_task = start_dispatcher(
        application,
        config_supplier=lambda: config,
        dispatcher=dispatcher,
    )
    assert dispatcher_task is application.state.source_add_dispatcher_task
    await asyncio.wait_for(dispatcher_started.wait(), timeout=1.0)

    async def fail_worker_startup() -> None:
        raise RuntimeError("scoring worker startup failed")

    worker_task = asyncio.create_task(fail_worker_startup())
    observe_workers(application, worker_task)
    for _attempt in range(10):
        await asyncio.sleep(0)
        if application.state.research_lab_worker_startup_failure is not None:
            break

    assert application.state.research_lab_worker_startup_failure == {
        "status": "failed",
        "exception_type": "RuntimeError",
    }
    assert worker_task.done()
    assert dispatcher_task is not None
    assert not dispatcher_task.done()

    dispatcher_task.cancel()
    await asyncio.gather(dispatcher_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_dispatcher_disabled_or_failed_state_is_explicit() -> None:
    start_dispatcher = _load_function("_start_source_add_dispatcher_task")
    application = SimpleNamespace(state=SimpleNamespace())
    dispatcher_called = False

    async def unexpected_dispatcher() -> None:
        nonlocal dispatcher_called
        dispatcher_called = True

    disabled = SimpleNamespace(
        source_add_enabled=True,
        source_add_dispatcher_enabled=False,
    )
    assert (
        start_dispatcher(
            application,
            config_supplier=lambda: disabled,
            dispatcher=unexpected_dispatcher,
        )
        is None
    )
    assert application.state.source_add_dispatcher_task is None
    assert application.state.source_add_dispatcher_failure is None
    assert dispatcher_called is False

    async def failed_dispatcher() -> None:
        raise ValueError("dispatcher startup failed")

    enabled = SimpleNamespace(
        source_add_enabled=True,
        source_add_dispatcher_enabled=True,
    )
    task = start_dispatcher(
        application,
        config_supplier=lambda: enabled,
        dispatcher=failed_dispatcher,
    )
    for _attempt in range(10):
        await asyncio.sleep(0)
        if application.state.source_add_dispatcher_failure is not None:
            break
    assert task is not None and task.done()
    assert application.state.source_add_dispatcher_failure == {
        "status": "failed",
        "exception_type": "ValueError",
    }


def test_lifespan_starts_and_cleans_source_add_outside_worker_startup() -> None:
    lifespan = next(
        node
        for node in _main_tree().body
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "lifespan"
    )
    worker_services = next(
        node
        for node in ast.walk(lifespan)
        if isinstance(node, ast.AsyncFunctionDef)
        and node.name == "_start_research_lab_worker_services"
    )
    source_starts = [
        node
        for node in ast.walk(lifespan)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_start_source_add_dispatcher_task"
    ]
    worker_task_starts = [
        node
        for node in ast.walk(lifespan)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "asyncio"
        and node.func.attr == "create_task"
        and node.args
        and isinstance(node.args[0], ast.Call)
        and isinstance(node.args[0].func, ast.Name)
        and node.args[0].func.id == "_start_research_lab_worker_services"
    ]

    assert len(source_starts) == 1
    assert len(worker_task_starts) == 1
    assert source_starts[0].lineno < worker_task_starts[0].lineno
    assert not any(
        isinstance(node, ast.Name)
        and node.id in {
            "_start_source_add_dispatcher_task",
            "run_source_add_dispatcher",
        }
        for node in ast.walk(worker_services)
    )
    worker_imports = [
        node
        for node in ast.walk(worker_services)
        if isinstance(node, ast.ImportFrom)
        and node.module == "gateway.research_lab.worker_autostart"
    ]
    worker_constructors = [
        node
        for node in ast.walk(worker_services)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "ResearchLabWorkerSupervisor"
    ]
    assert len(worker_imports) == 1
    assert len(worker_constructors) == 1
    assert worker_imports[0].lineno > worker_services.lineno
    assert worker_constructors[0].lineno > worker_services.lineno

    cleanup_lists = [
        node
        for node in ast.walk(lifespan)
        if isinstance(node, ast.List)
        and {
            item.id
            for item in node.elts
            if isinstance(item, ast.Name)
        }
        >= {
            "research_lab_worker_startup_task",
            "source_add_dispatcher_task",
        }
    ]
    assert len(cleanup_lists) == 1


@pytest.mark.asyncio
async def test_source_add_admin_and_allocation_bypass_exact_failed_worker_gate() -> None:
    source_ready = _load_function(
        "_gateway_source_add_dispatcher_ready",
        FastAPI=object,
    )

    class Response:
        def __init__(self, *, status_code, content):
            self.status_code = status_code
            self.content = content

    middleware = _load_function(
        "require_worker_authority_after_liveness",
        Request=object,
        _gateway_worker_startup_ready=lambda _application: False,
        _gateway_source_add_dispatcher_ready=source_ready,
        _WORKER_STARTUP_DIAGNOSTIC_PATHS=frozenset(
            {
                "/research-lab/status",
            }
        ),
        _SOURCE_ADD_INDEPENDENT_PATHS=frozenset(
            {
                "/research-lab/source-adapters",
            }
        ),
        JSONResponse=Response,
    )
    loop = asyncio.get_running_loop()
    dispatcher_task = loop.create_future()
    application = SimpleNamespace(
        state=SimpleNamespace(source_add_dispatcher_task=dispatcher_task)
    )
    calls: list[tuple[str, str]] = []

    async def call_next(request):
        calls.append((request.method, request.url.path))
        return "allowed"

    def request(method: str, path: str):
        return SimpleNamespace(
            method=method,
            url=SimpleNamespace(path=path),
            app=application,
        )

    allowed = [
        ("POST", "/research-lab/source-adapters"),
        (
            "POST",
            "/research-lab/admin/source-adapters/submission-1/credential-recipient",
        ),
        (
            "POST",
            "/research-lab/admin/source-adapters/submission-1/configure-test",
        ),
        (
            "POST",
            "/research-lab/admin/source-adapters/submission-1/provision",
        ),
        ("GET", "/research-lab/status"),
        ("GET", "/research-lab/allocations/attested/24124"),
    ]
    for method, path in allowed:
        assert await middleware(request(method, path), call_next) == "allowed"
    assert calls == allowed

    blocked = [
        ("GET", "/research-lab/source-adapters"),
        ("POST", "/research-lab/source-adapters/credential-recipient"),
        (
            "GET",
            "/research-lab/admin/source-adapters/submission-1/configure-test",
        ),
        (
            "POST",
            "/research-lab/admin/source-adapters/submission-1/recheck-provenance",
        ),
        (
            "POST",
            "/research-lab/admin/source-adapters/submission-1/future-action",
        ),
        (
            "POST",
            "/research-lab/admin/source-adapters/submission-1/provision/extra",
        ),
        ("GET", "/research-lab/allocations/live/24124"),
        ("POST", "/research-lab/allocations/attested/24124"),
        ("GET", "/research-lab/allocations/attested/-1"),
        ("GET", "/research-lab/allocations/attested/\u0661"),
        ("GET", "/research-lab/allocations/attested/24124/extra"),
        ("POST", "/validate"),
    ]
    for method, path in blocked:
        response = await middleware(request(method, path), call_next)
        assert response.status_code == 503, (method, path)
    assert calls == allowed

    dispatcher_task.set_result(None)
    for method, path in allowed[:4]:
        response = await middleware(request(method, path), call_next)
        assert response.status_code == 503, (method, path)

    independent_reads = allowed[4:]
    for method, path in independent_reads:
        assert await middleware(request(method, path), call_next) == "allowed"
    assert calls == allowed + independent_reads
