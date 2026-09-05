"""Gateway startup keeps PCR0 verification independent of retired workers."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_gateway_lifespan_starts_pcr0_without_retired_research_workers() -> None:
    tree = ast.parse((ROOT / "gateway/main.py").read_text(encoding="utf-8"))
    lifespan = next(
        node
        for node in tree.body
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "lifespan"
    )
    called_names = {
        node.func.id
        for node in ast.walk(lifespan)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }

    assert "start_worker_supervisor_without_blocking_event_loop" not in called_names
    assert "start_pcr0_builder" in called_names
    assert any(
        isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "pcr0_builder_task_handle"
            for target in node.targets
        )
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "start_pcr0_builder"
        for node in ast.walk(lifespan)
    )
    assert not any(
        isinstance(node, ast.AsyncFunctionDef)
        and node.name == "_start_research_lab_worker_services"
        for node in ast.walk(lifespan)
    )
