"""Protect Arena-facing epoch reads while retiring the old lead pipeline."""
from __future__ import annotations

import ast
from pathlib import Path
import sys
from types import ModuleType

from fastapi import FastAPI
from fastapi.testclient import TestClient

from gateway.api import epoch

ROOT = Path(__file__).resolve().parents[1]


def test_chain_epoch_reads_survive_without_a_lead_database(monkeypatch):
    chain = ModuleType("gateway.utils.epoch")

    async def status():
        return {"epoch": 123, "ready": True}

    async def context():
        return object(), 123

    async def info(epoch_id):
        return {"epoch": epoch_id}

    chain.get_epoch_authority_status_async = status
    chain.get_current_epoch_context_async = context
    chain.get_current_epoch_info_from_snapshot = lambda value: {"epoch": 123}
    chain.get_epoch_info_async = info
    monkeypatch.setitem(sys.modules, "gateway.utils.epoch", chain)
    app = FastAPI()
    app.include_router(epoch.router)
    with TestClient(app) as client:
        state = client.get("/epoch/state")
        assert state.status_code == 200
        assert state.json() == {"epoch": 123, "ready": True}
        assert state.headers["cache-control"] == "private, no-store"
        assert client.get("/epoch/current").json()["current_epoch_id"] == 123
        assert client.get("/epoch/123/info").json() == {"epoch": 123}
        assert client.get("/epoch/123/leads").status_code == 404


def test_gateway_cannot_start_or_register_the_retired_lead_pipeline():
    # Inspect the real bootstrap without connecting to credentials or the chain.
    tree = ast.parse((ROOT / "gateway/main.py").read_text())
    imports = [node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert not any(node.module == "gateway.tasks.epoch_monitor" for node in imports)
    assert not any(node.module == "gateway.api" and any(alias.name in {"validate", "submit", "manifest"} for alias in node.names) for node in imports)
    routers = {
        ast.unparse(node.args[0])
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr == "include_router" and node.args
    }
    assert {"arena_proxy_router", "arena_testnet_proxy_router", "epoch.router"} <= routers
    assert "validate.router" not in routers
    assert "submit.router" not in routers
    assert "manifest.router" not in routers
    assert not any(isinstance(node, ast.Name) and node.id == "EpochMonitor" for node in ast.walk(tree))
