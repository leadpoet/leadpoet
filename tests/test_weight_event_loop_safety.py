import threading
from types import SimpleNamespace

import pytest


class _Query:
    def __init__(self): self.execute_thread = None
    def select(self, *_args, **_kwargs): return self
    def eq(self, *_args, **_kwargs): return self
    def range(self, *_args, **_kwargs): return self
    def execute(self):
        self.execute_thread = threading.get_ident()
        return SimpleNamespace(data=[])


class _Client:
    def __init__(self, query): self.query = query
    def table(self, _name): return self.query


@pytest.mark.asyncio
async def test_fulfillment_consensus_read_executes_off_event_loop(monkeypatch):
    from gateway.fulfillment import consensus
    query = _Query()
    monkeypatch.setattr(consensus, "_get_supabase", lambda: _Client(query))
    assert await consensus._fetch_request_scores("request") == []
    assert query.execute_thread != threading.get_ident()


def test_validation_nonce_check_uses_async_path():
    from pathlib import Path
    source = Path("gateway/api/validate.py").read_text(encoding="utf-8")
    assert "await check_and_store_nonce_async(" in source
    assert "check_and_store_nonce," not in source
