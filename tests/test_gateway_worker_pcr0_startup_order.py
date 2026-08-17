"""Gateway worker authority must precede background PCR0 cache warming."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_historical_pcr0_warming_starts_after_authoritative_workers() -> None:
    tree = ast.parse((ROOT / "gateway/main.py").read_text(encoding="utf-8"))
    lifespan = next(
        node
        for node in tree.body
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "lifespan"
    )
    worker_services = next(
        node
        for node in ast.walk(lifespan)
        if isinstance(node, ast.AsyncFunctionDef)
        and node.name == "_start_research_lab_worker_services"
    )

    supervisor_waits = [
        node
        for node in ast.walk(worker_services)
        if isinstance(node, ast.Await)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id
        == "start_worker_supervisor_without_blocking_event_loop"
    ]
    pcr0_starts = [
        node
        for node in ast.walk(lifespan)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "start_pcr0_builder"
    ]

    assert len(supervisor_waits) == 1
    assert len(pcr0_starts) == 1
    assert pcr0_starts[0].lineno > supervisor_waits[0].lineno
    assert any(pcr0_starts[0] is node for node in ast.walk(worker_services))
