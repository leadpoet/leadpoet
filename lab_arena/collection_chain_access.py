"""Reuse one finalized-chain connection per spawned sandbox process and honor a shared 429."""

from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
import time

from lab_arena import chain
from lab_arena.collection_governor import retry_after_seconds

_client = None


def chain_cooling(directory):
    path = directory / ".chain-admission" / "retry-at.json"
    try:
        state = json.loads(path.read_bytes())
    except FileNotFoundError:
        return False
    return time.time() < state["until"]


def _retry_after(exc):
    while exc is not None:
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None) or getattr(exc, "code", None)
        if status == 429:
            headers = getattr(response, "headers", {}) or {}
            return retry_after_seconds(headers.get("Retry-After"))
        exc = exc.__cause__
    return None


def _hold(folder, seconds):
    previous = folder / "retry-at.json"
    now = time.time()
    try:
        state = json.loads(previous.read_bytes())
    except FileNotFoundError:
        state = {}
    if now < state.get("until", 0) and seconds <= state["until"] - now:
        return
    delay = (
        max(60, min(600, state.get("delay", 0) * 2))
        if now - state.get("until", 0) < 600
        else 60
    )
    delay = max(delay, seconds)
    with NamedTemporaryFile(mode="w", dir=folder, delete=False) as stream:
        json.dump({"until": now + delay, "delay": delay}, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(stream.name, previous)


def close_lane_chain():
    global _client
    if _client is not None:
        old, _client = _client, None
        old.close()


@contextmanager
def lane_chain(config, directory):
    from scripts.prepare_collection_experiment import lock_pool_directory

    global _client
    folder = Path(directory) / ".chain-admission"
    # Serialize opening sockets only. Finalized checks and in-flight inference
    # remain independent. A rejected handshake prevents other lanes from hammering it.
    if _client is None:
        with lock_pool_directory(folder):
            if chain_cooling(directory):
                raise chain.ArenaChainError("sandbox chain connection is cooling down")
            try:
                _client = chain.ArenaChain(config, chain.connect_substrate(config))
            except chain.ArenaChainError as exc:
                seconds = _retry_after(exc)
                if seconds is not None:
                    _hold(folder, seconds)
                raise
    try:
        yield _client
    except chain.ArenaChainError as exc:
        close_lane_chain()
        seconds = _retry_after(exc)
        if seconds is not None:
            with lock_pool_directory(folder):
                _hold(folder, seconds)
        raise
