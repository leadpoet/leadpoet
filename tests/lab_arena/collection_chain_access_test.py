from types import SimpleNamespace

import pytest

from lab_arena import chain, collection_chain_access as access
from scripts import run_locked_collection as worker
from scripts.prepare_collection_experiment import store_private_json
from tests.lab_arena.collection_pool_worker_test import pool_raw, PROMPT
from lab_arena.collection_experiment import load_experiment


def test_one_shared_429_cooldown_and_reuse_after_retry(tmp_path, monkeypatch):
    access.close_lane_chain()
    now = [1000]
    monkeypatch.setattr(access.time, "time", lambda: now[0])
    attempts, closed = [], []

    class Limited(Exception):
        response = SimpleNamespace(status_code=429, headers={"Retry-After": "90"})

    def connect(_):
        attempts.append(now[0])
        if len(attempts) == 1:
            raise chain.ArenaChainError("could not connect") from Limited()
        return SimpleNamespace(close=lambda: closed.append(True))

    monkeypatch.setattr(chain, "connect_substrate", connect)
    monkeypatch.setattr(chain, "ArenaChain", lambda config, conn: conn)
    with pytest.raises(chain.ArenaChainError, match="connect"):
        with access.lane_chain(None, tmp_path):
            pytest.fail("lease before finality")
    with pytest.raises(chain.ArenaChainError, match="cooling"):
        with access.lane_chain(None, tmp_path):
            pytest.fail("second handshake during throttling")
    assert attempts == [1000]
    now[0] = 1091
    try:
        with access.lane_chain(None, tmp_path) as first:
            pass
        with access.lane_chain(None, tmp_path) as second:
            assert first is second
        assert attempts == [1000, 1091] and not closed
    finally:
        access.close_lane_chain()
    assert closed == [True]


def test_chain_outage_stops_program_rotation_before_lease(tmp_path):
    for name in ("one", "two"):
        raw = pool_raw(repeats=1)
        raw["id"] = name
        spec = load_experiment(raw, PROMPT)
        store_private_json(tmp_path / (spec.batch_id + ".json"), raw)
    db = SimpleNamespace(rpc=lambda *_: "open")
    seen = []

    def run(raw, *, before_claim):
        seen.append(raw["id"])
        raise chain.ArenaChainError("shared endpoint down")

    with pytest.raises(chain.ArenaChainError):
        worker.poll_pool_experiments(PROMPT, tmp_path, db, run, lane=0, fundable=100000)
    assert len(seen) == 1
