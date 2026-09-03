"""Tests for ``lab_arena.chain`` (labarena.md sections 3.2, 7.2, 13.2;
labarenaaudit.md blocker 4).

A ``FakeSubstrate`` answers exactly the method names the production
``async_substrate_interface.SubstrateInterface`` exposes, with
production-shaped payloads: decoded extrinsics carry ``.value`` dicts with
``address`` and ``call`` (``call_module``/``call_function``/``call_args``),
events carry ``phase``/``extrinsic_idx``/``event``, storage reads return
``ScaleObj``-like ``.value`` wrappers.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import threading
import time
import types
from datetime import datetime, timezone
from pathlib import Path

import pytest
from bittensor_wallet import Keypair

from lab_arena import chain as arena_chain
from lab_arena.chain import (
    BANNED_SNAPSHOT_SCHEMA_VERSION,
    RUNNER_ELIGIBILITY_RULE,
    ArenaBlockNotFound,
    ArenaChain,
    ArenaChainConfig,
    ArenaChainConfigError,
    ArenaChainError,
    InvalidBlockHash,
    MetagraphSnapshot,
    banned_snapshot,
    bittensor_metagraph_source,
    coldkey_owns_hotkey,
    connect_substrate,
    current_settlement_epoch,
    hotkeys_owned_by_coldkey,
    is_registered,
    load_arena_cutover,
    metagraph_snapshot_from_object,
    normalize_block_hash,
    runner_allowlist,
    uid_for_hotkey,
    validate_banned_snapshot,
    verify_hotkey_signature,
    wallet_signature_verifier,
)
from lab_arena.contracts import RUNNER_ALLOWLIST_SCHEMA_VERSION, ArenaContractError, document_hash
from Leadpoet.utils.subnet_epoch import CUTOVER_JSON_ENV, CUTOVER_PATH_ENV, SubnetEpochCutover

REPO_ROOT = Path(__file__).resolve().parents[2]

ALICE = Keypair.create_from_uri("//Alice")
BOB = Keypair.create_from_uri("//Bob")
CHARLIE = Keypair.create_from_uri("//Charlie")
DAVE = Keypair.create_from_uri("//Dave")
EVE = Keypair.create_from_uri("//Eve")
FERDIE = Keypair.create_from_uri("//Ferdie")

NETUID = 71
ENDPOINT = "wss://arena-chain.example.invalid:443"
FINALIZED_NUMBER = 8_700_000
BEST_NUMBER = 8_700_005
DEPOSIT_NUMBER = 8_699_990
BLOCK_TIME = datetime(2026, 9, 2, 8, 0, 0, tzinfo=timezone.utc)
BLOCK_MILLIS = int(BLOCK_TIME.timestamp() * 1000)
STORAGE = {
    "Tempo": 360,
    "LastEpochBlock": 8_699_800,
    "PendingEpochAt": 0,
    "SubnetEpochIndex": 24_030,
    "BlocksSinceLastStep": 200,
}


def _digest(label: str) -> str:
    return "0x" + hashlib.sha256(label.encode("utf-8")).hexdigest()


GENESIS = _digest("genesis")
FORK_HASH = _digest("fork-of-%d" % DEPOSIT_NUMBER)


_CANONICAL_BY_NUMBER: dict = {}
_NUMBER_BY_CANONICAL: dict = {}


def canonical_hash(number: int) -> str:
    """Deterministic block hash for a number, memoized so the fake node answers
    reverse lookups without scanning millions of block numbers."""
    cached = _CANONICAL_BY_NUMBER.get(number)
    if cached is None:
        cached = GENESIS if number == 0 else _digest("canonical-%d" % number)
        _CANONICAL_BY_NUMBER[number] = cached
        _NUMBER_BY_CANONICAL[cached] = number
    return cached


class _Scale:
    """Stand-in for ``ScaleObj`` / ``GenericExtrinsic``: the payload sits on ``.value``."""

    def __init__(self, value):
        self.value = value


def transfer_extrinsic(sender, dest, amount, *, function="transfer_keep_alive", extrinsic_hash=None):
    args = [{"name": "dest", "type": "MultiAddress", "value": dest}]
    if amount is None:
        args.append({"name": "keep_alive", "type": "bool", "value": True})
    else:
        args.append({"name": "value", "type": "Compact<u128>", "value": amount})
    return _Scale(
        {
            "extrinsic_hash": extrinsic_hash or _digest("transfer-%s-%s-%s" % (sender, dest, amount)),
            "extrinsic_length": 140,
            "address": sender,
            "signature": {"Sr25519": "0x" + "11" * 64},
            "call": {
                "call_index": "0x0503",
                "call_module": "Balances",
                "call_function": function,
                "call_args": args,
                "call_hash": _digest("call"),
            },
            "nonce": 3,
            "era": "00",
            "tip": 0,
        }
    )


def timestamp_inherent(millis):
    return _Scale(
        {
            "extrinsic_hash": _digest("timestamp-%d" % millis),
            "extrinsic_length": 10,
            "call": {
                "call_index": "0x0300",
                "call_module": "Timestamp",
                "call_function": "set",
                "call_args": [{"name": "now", "type": "Compact<u64>", "value": millis}],
                "call_hash": _digest("timestamp-call"),
            },
        }
    )


def batch_extrinsic(sender):
    return _Scale(
        {
            "extrinsic_hash": _digest("batch"),
            "address": sender,
            "call": {
                "call_module": "Utility",
                "call_function": "batch",
                "call_args": [{"name": "calls", "type": "Vec<Call>", "value": []}],
            },
        }
    )


def status_event(index, event_id="ExtrinsicSuccess", module_id="System"):
    return {
        "phase": "ApplyExtrinsic",
        "extrinsic_idx": index,
        "event": {
            "module_id": module_id,
            "event_id": event_id,
            "attributes": {"dispatch_info": {"weight": {"ref_time": 1, "proof_size": 0}, "class": "Normal", "pays_fee": "Yes"}},
        },
        "topics": [],
    }


def make_block(number, block_hash, extrinsics):
    return {
        "header": {
            "number": number,
            "hash": block_hash,
            "parentHash": canonical_hash(number - 1),
            "stateRoot": _digest("state-%d" % number),
            "extrinsicsRoot": _digest("ext-%d" % number),
            "digest": {"logs": []},
        },
        "extrinsics": list(extrinsics),
    }


def _guarded(method):
    def wrapper(self, *args, **kwargs):
        self._enter(method.__name__)
        try:
            return method(self, *args, **kwargs)
        finally:
            self._exit()

    wrapper.__name__ = method.__name__
    return wrapper


class FakeSubstrate:
    """Answers the ``SubstrateClient`` surface; raises for anything it does not hold."""

    def __init__(self, *, finalized_number=FINALIZED_NUMBER, best_number=BEST_NUMBER, netuid=NETUID):
        self.finalized_number = finalized_number
        self.best_number = best_number
        self.netuid = netuid
        self.blocks = {}
        self.events = {}
        self.timestamps = {}
        self.storage = dict(STORAGE)
        self.fail = set()
        self.missing_numbers = set()
        self.calls = []
        self.delay = 0.0
        self.overlap = False
        self._active = 0

    def _enter(self, name):
        self.calls.append(name)
        if name in self.fail:
            raise RuntimeError("%s unavailable" % name)
        self._active += 1
        if self._active > 1:
            self.overlap = True
        if self.delay:
            time.sleep(self.delay)

    def _exit(self):
        self._active -= 1

    def add_block(self, number, extrinsics, *, block_hash=None, events=None, millis=BLOCK_MILLIS):
        block_hash = block_hash or canonical_hash(number)
        self.blocks[block_hash] = make_block(number, block_hash, extrinsics)
        if events is not None:
            self.events[block_hash] = list(events)
        if millis is not None:
            self.timestamps[block_hash] = millis
        return block_hash

    @_guarded
    def get_chain_finalised_head(self):
        return canonical_hash(self.finalized_number)

    @_guarded
    def get_block_hash(self, block_id):
        if block_id < 0 or block_id > self.best_number or block_id in self.missing_numbers:
            return None
        return canonical_hash(block_id)

    @_guarded
    def get_block_number(self, block_hash=None):
        block = self.blocks.get(block_hash)
        if block is not None:
            return block["header"]["number"]
        number = _NUMBER_BY_CANONICAL.get(block_hash)
        if number is not None and 0 <= number <= self.best_number:
            return number
        raise RuntimeError("Unable to determine block number for %s" % block_hash)

    @_guarded
    def get_block(self, block_hash=None, **kwargs):
        block = self.blocks.get(block_hash)
        if block is None:
            return None
        if not isinstance(block, dict) or "header" not in block or "extrinsics" not in block:
            return block  # a deliberately malformed payload is returned as-is
        return {"header": dict(block["header"]), "extrinsics": list(block["extrinsics"])}

    @_guarded
    def get_events(self, block_hash=None):
        return list(self.events.get(block_hash, []))

    @_guarded
    def query(self, module, storage_function, params=None, block_hash=None):
        if module == "Timestamp":
            assert storage_function == "Now" and params == []
            millis = self.timestamps.get(block_hash)
            return None if millis is None else _Scale(millis)
        assert module == "SubtensorModule"
        assert params == [self.netuid]
        if storage_function not in self.storage:
            raise RuntimeError("storage %s missing" % storage_function)
        return _Scale(self.storage[storage_function])


def make_config(**overrides):
    values = dict(endpoint=ENDPOINT, netuid=NETUID, network_name="finney", request_timeout_seconds=15, metagraph_ttl_seconds=60)
    values.update(overrides)
    return ArenaChainConfig(**values)


def make_snapshot(number, block_hash, *, permits=None):
    hotkeys = (ALICE.ss58_address, CHARLIE.ss58_address, EVE.ss58_address)
    coldkeys = (BOB.ss58_address, BOB.ss58_address, DAVE.ss58_address)
    return MetagraphSnapshot(
        netuid=NETUID,
        block_number=number,
        block_hash=block_hash,
        hotkeys=hotkeys,
        coldkeys=coldkeys,
        validator_permit=permits or (True, False, True),
        stake=(1000.0, 0.0, 250.5),
    )


class CountingSource:
    def __init__(self, snapshot_for=None):
        self.calls = []
        self.snapshot_for = snapshot_for or (lambda client, netuid, block_hash: make_snapshot(client.get_block_number(block_hash), block_hash))

    def __call__(self, client, netuid, block_hash):
        self.calls.append((netuid, block_hash))
        return self.snapshot_for(client, netuid, block_hash)


class FakeClock:
    def __init__(self, start=1000.0):
        self.now = start

    def __call__(self):
        return self.now


def deposit_chain(**kwargs):
    fake = FakeSubstrate()
    deposit_hash = fake.add_block(
        DEPOSIT_NUMBER,
        [timestamp_inherent(BLOCK_MILLIS), transfer_extrinsic(BOB.ss58_address, ALICE.ss58_address, 500_000_000)],
        events=[status_event(0), status_event(1)],
    )
    chain = ArenaChain(make_config(**kwargs), fake, metagraph_source=CountingSource(), clock=FakeClock())
    return fake, chain, deposit_hash


# ---------------------------------------------------------------------------
# Configuration and connection
# ---------------------------------------------------------------------------


def test_config_rejects_malformed_values():
    with pytest.raises(ArenaChainConfigError):
        make_config(endpoint="https://entrypoint-finney.opentensor.ai:443")
    with pytest.raises(ArenaChainConfigError):
        make_config(endpoint="wss://")
    with pytest.raises(ArenaChainConfigError):
        make_config(endpoint="wss://user:secret@host:443")
    with pytest.raises(ArenaChainConfigError):
        make_config(endpoint="finney")
    with pytest.raises(ArenaChainConfigError):
        make_config(netuid=0)
    with pytest.raises(ArenaChainConfigError):
        make_config(netuid=True)
    with pytest.raises(ArenaChainConfigError):
        make_config(network_name="Finney")
    with pytest.raises(ArenaChainConfigError):
        make_config(network_name="")
    with pytest.raises(ArenaChainConfigError):
        make_config(request_timeout_seconds=0)
    with pytest.raises(ArenaChainConfigError):
        make_config(request_timeout_seconds=3.5)
    with pytest.raises(ArenaChainConfigError):
        make_config(metagraph_ttl_seconds=-1)


def test_config_normalizes_whitespace_only():
    config = make_config(endpoint=" ws://127.0.0.1:9944 ", network_name=" test ")
    assert config.endpoint == "ws://127.0.0.1:9944"
    assert config.network_name == "test"


def test_connect_substrate_uses_only_the_configured_endpoint(monkeypatch):
    created = []

    class RecordingInterface:
        def __init__(self, **kwargs):
            created.append(kwargs)

    module = types.ModuleType("async_substrate_interface")
    module.SubstrateInterface = RecordingInterface
    monkeypatch.setitem(sys.modules, "async_substrate_interface", module)
    monkeypatch.setenv("BITTENSOR_NETWORK", "test")
    monkeypatch.setenv("BITTENSOR_NETUID", "999")

    client = connect_substrate(make_config(request_timeout_seconds=7))

    assert isinstance(client, RecordingInterface)
    assert created == [
        {
            "url": ENDPOINT,
            "ss58_format": 42,
            "type_registry_preset": "substrate-node-template",
            "max_retries": arena_chain.SUBSTRATE_MAX_RETRIES,
            "retry_timeout": 7.0,
        }
    ]

    monkeypatch.setitem(sys.modules, "async_substrate_interface", None)
    with pytest.raises(ArenaChainError, match="not installed"):
        connect_substrate(make_config())
    with pytest.raises(ArenaChainConfigError):
        connect_substrate("wss://not-a-config")


def test_import_closure_is_lazy_and_boundary_clean():
    code = (
        "import json, sys\n"
        "import lab_arena.chain\n"
        "loaded = sorted(m for m in sys.modules if m.startswith(('gateway.tee', 'gateway.db', "
        "'async_substrate_interface', 'bittensor', 'scalecodec')))\n"
        "print(json.dumps(loaded))\n"
    )
    env = dict(os.environ)
    env.setdefault("SUPABASE_URL", "https://test.invalid")
    env.setdefault("SUPABASE_SERVICE_ROLE_KEY", "test-only-service-role-placeholder")
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip().splitlines()[-1]) == []


# ---------------------------------------------------------------------------
# Finalized chain reads
# ---------------------------------------------------------------------------


def test_normalize_block_hash_variants():
    body = "AB" * 32
    expected = "0x" + "ab" * 32
    assert normalize_block_hash("0x" + body) == expected
    assert normalize_block_hash("0X" + body) == expected
    assert normalize_block_hash(" " + body.lower() + " ") == expected
    assert normalize_block_hash(bytes.fromhex(body)) == expected
    for bad in ("0x" + "ab" * 31, "zz" * 32, "", None, 12, "0x" + "ab" * 33):
        with pytest.raises(InvalidBlockHash):
            normalize_block_hash(bad)


def test_finalized_head_and_block_hash_rule():
    fake, chain, deposit_hash = deposit_chain()
    head = chain.finalized_head()
    assert head.number == FINALIZED_NUMBER
    assert head.hash == canonical_hash(FINALIZED_NUMBER)
    assert chain.finalized_head_hash() == head.hash
    assert chain.finalized_block_hash(DEPOSIT_NUMBER) == deposit_hash
    assert chain.finalized_block_hash(FINALIZED_NUMBER) == head.hash
    assert chain.block_number(deposit_hash) == DEPOSIT_NUMBER
    # Numbers above the finalized head are refused even though the node knows a best-chain hash.
    with pytest.raises(ArenaChainError, match="beyond the finalized head"):
        chain.finalized_block_hash(FINALIZED_NUMBER + 1)
    with pytest.raises(ArenaChainError):
        chain.finalized_block_hash(-1)
    with pytest.raises(ArenaChainError):
        chain.finalized_block_hash(True)
    # A number at or below the finalized head whose hash the endpoint cannot
    # produce is a not-found failure, never a silently substituted hash.
    fake.missing_numbers.add(FINALIZED_NUMBER - 50)
    with pytest.raises(ArenaBlockNotFound):
        chain.finalized_block_hash(FINALIZED_NUMBER - 50)


def test_is_finalized_detects_forks_and_unfinalized_blocks():
    fake, chain, deposit_hash = deposit_chain()
    assert chain.is_finalized(deposit_hash) is True
    fake.add_block(DEPOSIT_NUMBER, [timestamp_inherent(BLOCK_MILLIS)], block_hash=FORK_HASH)
    assert chain.is_finalized(FORK_HASH) is False
    fake.add_block(FINALIZED_NUMBER + 3, [timestamp_inherent(BLOCK_MILLIS)])
    assert chain.is_finalized(canonical_hash(FINALIZED_NUMBER + 3)) is False
    with pytest.raises(ArenaChainError, match="get_block_number failed"):
        chain.is_finalized(_digest("never-seen"))


def test_chain_failures_raise_typed_errors_with_cause():
    fake, chain, deposit_hash = deposit_chain()
    fake.fail.update({"get_chain_finalised_head", "get_block_number"})
    with pytest.raises(ArenaChainError, match="get_chain_finalised_head failed") as info:
        chain.finalized_head()
    assert isinstance(info.value.__cause__, RuntimeError)
    with pytest.raises(ArenaChainError, match="get_block_number failed"):
        chain.is_finalized(deposit_hash)

    class Missing:
        pass

    with pytest.raises(ArenaChainError, match="lacks get_chain_finalised_head"):
        ArenaChain(make_config(), Missing(), metagraph_source=CountingSource()).finalized_head()
    with pytest.raises(ArenaChainConfigError):
        ArenaChain(make_config(), None, metagraph_source=CountingSource())


def test_client_calls_are_serialized_across_threads():
    fake, chain, deposit_hash = deposit_chain()
    fake.delay = 0.002
    errors = []

    def worker():
        try:
            for _ in range(10):
                chain.finalized_head()
                chain.is_finalized(deposit_hash)
                chain.epoch_subtensor().substrate.get_block_hash(0)
        except Exception as exc:  # pragma: no cover - surfaced through the assertion below
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)
    assert not errors
    assert fake.overlap is False
    assert fake.calls.count("get_block_hash") == 120  # sixty direct calls plus one inside every is_finalized


# ---------------------------------------------------------------------------
# Finalized metagraph cache
# ---------------------------------------------------------------------------


def test_metagraph_cache_ttl_and_refresh():
    fake = FakeSubstrate()
    source = CountingSource()
    clock = FakeClock()
    chain = ArenaChain(make_config(metagraph_ttl_seconds=60), fake, metagraph_source=source, clock=clock)

    first = chain.metagraph(finalized=True)
    assert first.block_number == FINALIZED_NUMBER
    assert first.block_hash == canonical_hash(FINALIZED_NUMBER)
    assert source.calls == [(NETUID, canonical_hash(FINALIZED_NUMBER))]

    head_reads = fake.calls.count("get_chain_finalised_head")
    clock.now += 30
    assert chain.metagraph() is first
    assert fake.calls.count("get_chain_finalised_head") == head_reads  # served without a chain read

    clock.now += 31  # TTL expired, finalized head unchanged: no refetch, cache extended
    assert chain.metagraph() is first
    assert len(source.calls) == 1
    assert fake.calls.count("get_chain_finalised_head") == head_reads + 1

    fake.finalized_number += 1  # new finalized head after expiry: refetch
    clock.now += 61
    second = chain.metagraph()
    assert second is not first
    assert second.block_number == FINALIZED_NUMBER + 1
    assert len(source.calls) == 2

    third = chain.refresh_metagraph()  # explicit refresh inside TTL
    assert third is not second and len(source.calls) == 3

    fake.fail.add("get_chain_finalised_head")
    clock.now += 61
    with pytest.raises(ArenaChainError):
        chain.metagraph()  # expired cache is never served after a failure

    zero_ttl = ArenaChain(make_config(metagraph_ttl_seconds=0), FakeSubstrate(), metagraph_source=CountingSource(), clock=FakeClock())
    zero_ttl.metagraph()
    zero_ttl.metagraph()
    assert len(zero_ttl._metagraph_source.calls) == 1  # same head: extended, not refetched


def test_metagraph_source_results_are_validated():
    fake = FakeSubstrate()
    wrong_hash = CountingSource(lambda client, netuid, block_hash: make_snapshot(FINALIZED_NUMBER, canonical_hash(5)))
    with pytest.raises(ArenaChainError, match="not pinned"):
        ArenaChain(make_config(), fake, metagraph_source=wrong_hash).metagraph()
    wrong_number = CountingSource(lambda client, netuid, block_hash: make_snapshot(FINALIZED_NUMBER - 1, block_hash))
    with pytest.raises(ArenaChainError, match="not pinned"):
        ArenaChain(make_config(), fake, metagraph_source=wrong_number).metagraph()
    not_snapshot = CountingSource(lambda client, netuid, block_hash: {"hotkeys": []})
    with pytest.raises(ArenaChainError, match="unexpected object"):
        ArenaChain(make_config(), fake, metagraph_source=not_snapshot).metagraph()
    with pytest.raises(ArenaChainError, match="only the finalized"):
        ArenaChain(make_config(), fake, metagraph_source=CountingSource()).metagraph(finalized=False)


def test_metagraph_snapshot_helpers_and_validation():
    snapshot = make_snapshot(FINALIZED_NUMBER, canonical_hash(FINALIZED_NUMBER))
    assert snapshot.size == 3
    assert uid_for_hotkey(snapshot, CHARLIE.ss58_address) == 1
    assert uid_for_hotkey(snapshot, FERDIE.ss58_address) is None
    assert is_registered(snapshot, ALICE.ss58_address) is True
    assert is_registered(snapshot, FERDIE.ss58_address) is False
    assert coldkey_owns_hotkey(snapshot, BOB.ss58_address, CHARLIE.ss58_address) is True
    assert coldkey_owns_hotkey(snapshot, DAVE.ss58_address, CHARLIE.ss58_address) is False
    assert coldkey_owns_hotkey(snapshot, BOB.ss58_address, FERDIE.ss58_address) is False
    assert coldkey_owns_hotkey(snapshot, "", CHARLIE.ss58_address) is False
    assert hotkeys_owned_by_coldkey(snapshot, BOB.ss58_address) == [ALICE.ss58_address, CHARLIE.ss58_address]
    assert hotkeys_owned_by_coldkey(snapshot, FERDIE.ss58_address) == []

    with pytest.raises(ArenaChainError, match="differ in length"):
        MetagraphSnapshot(netuid=NETUID, block_number=1, block_hash=GENESIS, hotkeys=(ALICE.ss58_address,), coldkeys=(), validator_permit=(True,))
    with pytest.raises(ArenaChainError, match="not unique"):
        MetagraphSnapshot(netuid=NETUID, block_number=1, block_hash=GENESIS, hotkeys=(ALICE.ss58_address, ALICE.ss58_address), coldkeys=(BOB.ss58_address, BOB.ss58_address), validator_permit=(True, True))
    with pytest.raises(ArenaChainError, match="validator_permit"):
        MetagraphSnapshot(netuid=NETUID, block_number=1, block_hash=GENESIS, hotkeys=(ALICE.ss58_address,), coldkeys=(BOB.ss58_address,), validator_permit=(1,))
    with pytest.raises(ArenaChainError, match="stake"):
        MetagraphSnapshot(netuid=NETUID, block_number=1, block_hash=GENESIS, hotkeys=(ALICE.ss58_address,), coldkeys=(BOB.ss58_address,), validator_permit=(True,), stake=(float("nan"),))
    with pytest.raises(ArenaContractError):
        MetagraphSnapshot(netuid=NETUID, block_number=1, block_hash=GENESIS, hotkeys=("nope",), coldkeys=(BOB.ss58_address,), validator_permit=(True,))
    with pytest.raises(InvalidBlockHash):
        MetagraphSnapshot(netuid=NETUID, block_number=1, block_hash="0x12", hotkeys=(), coldkeys=(), validator_permit=())


class _Item:
    """numpy-scalar stand-in: exposes ``.item()``."""

    def __init__(self, value):
        self._value = value

    def item(self):
        return self._value


class _FakeMetagraph:
    def __init__(self, block):
        self.hotkeys = [ALICE.ss58_address, CHARLIE.ss58_address]
        self.coldkeys = [BOB.ss58_address, BOB.ss58_address]
        self.validator_permit = [_Item(True), _Item(False)]
        self.S = [_Item(12.5), _Item(0.0)]
        self.block = _Item(block)


def test_metagraph_snapshot_from_object():
    snapshot = metagraph_snapshot_from_object(_FakeMetagraph(FINALIZED_NUMBER), netuid=NETUID, block_number=FINALIZED_NUMBER, block_hash=canonical_hash(FINALIZED_NUMBER))
    assert snapshot.validator_permit == (True, False)
    assert snapshot.stake == (12.5, 0.0)
    assert snapshot.coldkeys == (BOB.ss58_address, BOB.ss58_address)
    with pytest.raises(ArenaChainError, match="differs from the finalized block"):
        metagraph_snapshot_from_object(_FakeMetagraph(FINALIZED_NUMBER - 1), netuid=NETUID, block_number=FINALIZED_NUMBER, block_hash=canonical_hash(FINALIZED_NUMBER))
    with pytest.raises(ArenaChainError, match="pinned fields"):
        metagraph_snapshot_from_object(object(), netuid=NETUID, block_number=FINALIZED_NUMBER, block_hash=canonical_hash(FINALIZED_NUMBER))


def test_bittensor_metagraph_source_uses_the_endpoint_and_pinned_api(monkeypatch):
    instances = []

    class FakeSubtensor:
        def __init__(self, network):
            self.network = network
            self.requested = None
            self.closed = False
            instances.append(self)

        def metagraph(self, netuid, block=None):
            self.requested = (netuid, block)
            return _FakeMetagraph(block)

        def close(self):
            self.closed = True

    module = types.ModuleType("bittensor")
    module.Subtensor = FakeSubtensor
    monkeypatch.setitem(sys.modules, "bittensor", module)
    monkeypatch.setenv("BITTENSOR_NETWORK", "test")

    fake = FakeSubstrate()
    chain = ArenaChain(make_config(), fake, metagraph_source=bittensor_metagraph_source(make_config()), clock=FakeClock())
    snapshot = chain.metagraph()
    assert snapshot.block_number == FINALIZED_NUMBER
    assert instances[0].network == ENDPOINT
    assert instances[0].requested == (NETUID, FINALIZED_NUMBER)
    assert instances[0].closed is True

    class ElevenShaped:
        def __init__(self, network):
            self.closed = False

        def close(self):
            self.closed = True

    module.Subtensor = ElevenShaped
    source = bittensor_metagraph_source(make_config())
    with pytest.raises(ArenaChainError, match="pinned Subtensor.metagraph API"):
        source(chain.client, NETUID, canonical_hash(FINALIZED_NUMBER))

    monkeypatch.setitem(sys.modules, "bittensor", None)
    with pytest.raises(ArenaChainError, match="not installed"):
        source(chain.client, NETUID, canonical_hash(FINALIZED_NUMBER))


# ---------------------------------------------------------------------------
# Banned snapshot and runner allowlist (section 3.2)
# ---------------------------------------------------------------------------


def test_banned_snapshot_is_sorted_unique_and_hashed():
    one = banned_snapshot([DAVE.ss58_address, ALICE.ss58_address, DAVE.ss58_address])
    two = banned_snapshot((ALICE.ss58_address, DAVE.ss58_address))
    assert one == two
    assert one["schema_version"] == BANNED_SNAPSHOT_SCHEMA_VERSION
    assert one["hotkeys"] == sorted([ALICE.ss58_address, DAVE.ss58_address])
    assert one["snapshot_hash"] == document_hash({"schema_version": BANNED_SNAPSHOT_SCHEMA_VERSION, "hotkeys": one["hotkeys"]})
    assert banned_snapshot([])["hotkeys"] == []
    assert validate_banned_snapshot(one) == one
    with pytest.raises(ArenaContractError):
        banned_snapshot(["not-a-hotkey"])
    tampered = dict(one, hotkeys=[ALICE.ss58_address])
    with pytest.raises(ArenaContractError, match="hash"):
        validate_banned_snapshot(tampered)
    unsorted = dict(one, hotkeys=list(reversed(one["hotkeys"])))
    with pytest.raises(ArenaContractError, match="sorted"):
        validate_banned_snapshot(unsorted)
    with pytest.raises(ArenaContractError):
        validate_banned_snapshot(dict(one, extra=1))
    with pytest.raises(ArenaContractError):
        validate_banned_snapshot("nope")


def test_runner_allowlist_rule():
    snapshot = make_snapshot(FINALIZED_NUMBER, canonical_hash(FINALIZED_NUMBER))  # permits: ALICE, EVE
    banned = banned_snapshot([EVE.ss58_address])
    allowlist = runner_allowlist(snapshot, banned=banned, floor_runner_hotkeys=[FERDIE.ss58_address, DAVE.ss58_address, FERDIE.ss58_address])
    assert allowlist["schema_version"] == RUNNER_ALLOWLIST_SCHEMA_VERSION
    assert allowlist["eligibility_rule"] == RUNNER_ELIGIBILITY_RULE
    assert allowlist["hotkeys"] == sorted([ALICE.ss58_address, DAVE.ss58_address, FERDIE.ss58_address])
    assert allowlist["floor_runner_hotkeys"] == sorted([DAVE.ss58_address, FERDIE.ss58_address])
    assert allowlist["banned_snapshot_hash"] == banned["snapshot_hash"]
    assert allowlist["block_number"] == FINALIZED_NUMBER and allowlist["block_hash"] == canonical_hash(FINALIZED_NUMBER)
    body = {key: value for key, value in allowlist.items() if key != "allowlist_hash"}
    assert allowlist["allowlist_hash"] == document_hash(body)
    assert set(allowlist["floor_runner_hotkeys"]) <= set(allowlist["hotkeys"])
    assert len(set(allowlist["hotkeys"])) == len(allowlist["hotkeys"])

    again = runner_allowlist(snapshot, banned=banned, floor_runner_hotkeys=[DAVE.ss58_address, FERDIE.ss58_address])
    assert again == allowlist  # deterministic regardless of input order

    # A floor runner that is also permitted and unbanned appears once.
    merged = runner_allowlist(snapshot, banned=banned, floor_runner_hotkeys=[ALICE.ss58_address])
    assert merged["hotkeys"] == [ALICE.ss58_address]

    with pytest.raises(ArenaContractError, match="banned"):
        runner_allowlist(snapshot, banned=banned, floor_runner_hotkeys=[EVE.ss58_address])
    with pytest.raises(ArenaContractError, match="at least one"):
        runner_allowlist(snapshot, banned=banned, floor_runner_hotkeys=[])
    with pytest.raises(ArenaContractError):
        runner_allowlist(snapshot, banned={"hotkeys": []}, floor_runner_hotkeys=[DAVE.ss58_address])
    with pytest.raises(ArenaContractError):
        runner_allowlist({"hotkeys": []}, banned=banned, floor_runner_hotkeys=[DAVE.ss58_address])


# ---------------------------------------------------------------------------
# Settlement epoch ordinals (section 13.2)
# ---------------------------------------------------------------------------


def synthetic_cutover(**overrides):
    values = dict(
        network_genesis_hash=GENESIS,
        netuid=NETUID,
        cutover_block=8_670_636,
        cutover_block_hash=canonical_hash(8_670_636),
        first_subnet_epoch_index=24_020,
        first_settlement_epoch_id=24_073,
        last_legacy_epoch_id=24_072,
    )
    values.update(overrides)
    return SubnetEpochCutover(**values)


def test_current_settlement_epoch_reads_the_finalized_head():
    fake, chain, _ = deposit_chain()
    fake.timestamps[canonical_hash(FINALIZED_NUMBER)] = BLOCK_MILLIS
    cutover = synthetic_cutover()
    assert current_settlement_epoch(chain, cutover) == 24_073 + (24_030 - 24_020)
    assert "get_chain_head" not in fake.calls
    assert fake.calls.count("get_chain_finalised_head") >= 1

    with pytest.raises(ArenaChainError, match="netuid differs"):
        current_settlement_epoch(chain, synthetic_cutover(netuid=72))
    with pytest.raises(ArenaChainError, match="does not map"):
        current_settlement_epoch(chain, synthetic_cutover(network_genesis_hash=_digest("other-chain")))
    fake.storage["SubnetEpochIndex"] = 24_000
    with pytest.raises(ArenaChainError, match="does not map"):
        current_settlement_epoch(chain, cutover)
    del fake.storage["SubnetEpochIndex"]
    with pytest.raises(ArenaChainError, match="unavailable"):
        current_settlement_epoch(chain, cutover)
    with pytest.raises(ArenaChainError):
        current_settlement_epoch(chain, {"netuid": 71})


def test_load_arena_cutover_defaults_to_the_repo_manifest_without_mutating_env(tmp_path):
    before = dict(os.environ)
    cutover = load_arena_cutover({})
    assert cutover.netuid == 71
    assert cutover.first_settlement_epoch_id == 24_073
    assert dict(os.environ) == before

    synthetic = synthetic_cutover()
    inline = load_arena_cutover({CUTOVER_JSON_ENV: json.dumps(synthetic.to_dict())}, default_manifest_path=tmp_path / "missing.json")
    assert inline == synthetic

    manifest = tmp_path / "cutover.json"
    manifest.write_text(json.dumps(synthetic.to_dict()), encoding="utf-8")
    assert load_arena_cutover({}, default_manifest_path=manifest) == synthetic
    assert load_arena_cutover({CUTOVER_PATH_ENV: str(manifest)}) == synthetic

    with pytest.raises(ArenaChainError, match="unavailable or invalid"):
        load_arena_cutover({CUTOVER_JSON_ENV: "{}", CUTOVER_PATH_ENV: str(manifest)})
    with pytest.raises(ArenaChainError, match="unavailable or invalid"):
        load_arena_cutover({}, default_manifest_path=tmp_path / "absent.json")


# ---------------------------------------------------------------------------
# Signature seam
# ---------------------------------------------------------------------------


def test_signature_seam():
    message = json.dumps({"scope": "lab_arena.claim.v1", "n": 1}, sort_keys=True)
    signature = CHARLIE.sign(message.encode("utf-8")).hex()
    assert wallet_signature_verifier(CHARLIE.ss58_address, signature, message) is True
    assert wallet_signature_verifier(CHARLIE.ss58_address, "0x" + signature, message) is True
    assert wallet_signature_verifier(CHARLIE.ss58_address, signature, message + " ") is False
    assert wallet_signature_verifier(DAVE.ss58_address, signature, message) is False
    assert wallet_signature_verifier(CHARLIE.ss58_address, "zz" + signature[2:], message) is False
    assert wallet_signature_verifier("not-an-address", signature, message) is False
    assert wallet_signature_verifier(CHARLIE.ss58_address, signature, 12) is False

    hex_message = "0x" + b"raw-bytes".hex()
    raw_signature = CHARLIE.sign(b"raw-bytes").hex()
    assert wallet_signature_verifier(CHARLIE.ss58_address, raw_signature, hex_message) is True

    production = verify_hotkey_signature(CHARLIE.ss58_address, signature, message)
    assert isinstance(production, bool)
    try:
        from bittensor import Keypair as _ProductionKeypair  # noqa: F401
    except ImportError:
        pass  # bittensor 11 locally: the Lab verifier cannot import Keypair and returns False
    else:
        assert production is True


def test_close_releases_the_client_once_and_never_raises():
    """The wiring registers close at exit; a client without close, or one that raises, is fine."""

    class Closable(FakeSubstrate):
        def __init__(self):
            super().__init__()
            self.closed = 0

        def close(self):
            self.closed += 1
            if self.closed > 1:
                raise RuntimeError("already closed")

    config = ArenaChainConfig(endpoint="wss://chain.example:443", netuid=71, network_name="finney", request_timeout_seconds=5, metagraph_ttl_seconds=5)
    client = Closable()
    arena = ArenaChain(config, client, metagraph_source=lambda *args: None)
    arena.close()
    arena.close()  # the second close raises inside the client and is swallowed
    assert client.closed == 2
    ArenaChain(config, object(), metagraph_source=lambda *args: None).close()  # no close method: nothing to do
