"""Tests for ``lab_arena.funding`` (labarena.md section 7.2 and the section
18.3 deposit bullet; labarenaaudit.md blocker 4).

The fake substrate mirrors the production client's method names and payload
shapes; the fake deposit store enforces reference uniqueness under a lock and
can be told to fail, standing in for the ``lab_arena_credit_deposit`` SQL
function.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_FLOOR

import pytest
from bittensor_wallet import Keypair

from lab_arena.chain import ArenaChain, ArenaChainConfig, ArenaChainError, MetagraphSnapshot
from lab_arena.contracts import ArenaContractError, verify_hashed_document
from lab_arena.funding import (
    COINGECKO_PRICE_URL,
    DEPOSIT_DOCUMENT_SCHEMA_VERSION,
    DEPOSIT_RULE_IDS,
    FUTURE_BLOCK_TOLERANCE_SECONDS,
    DepositRejected,
    FundingConfig,
    FundingError,
    FundingStoreError,
    MalformedReference,
    PriceUnavailable,
    coingecko_price_source,
    confirm_funding,
    normalize_extrinsic_index,
    normalize_payment_reference,
    parse_payment_reference,
    rao_to_microusd,
    verify_deposit,
)

ARENA_WALLET = Keypair.create_from_uri("//Alice")  # the dedicated recipient
OWNER_COLDKEY = Keypair.create_from_uri("//Bob")
MINER_HOTKEY = Keypair.create_from_uri("//Charlie")
STRANGER = Keypair.create_from_uri("//Dave")
OTHER_HOTKEY = Keypair.create_from_uri("//Eve")

NETUID = 71
NETWORK = "finney"
ENDPOINT = "wss://arena-chain.example.invalid:443"
FINALIZED_NUMBER = 8_700_000
BEST_NUMBER = 8_700_005
DEPOSIT_NUMBER = 8_699_990
DEPOSIT_INDEX = 1
AMOUNT_RAO = 500_000_000  # 0.5 TAO
PRICE = Decimal("412.37")
EXPECTED_MICROUSD = 206_185_000
BLOCK_TIME = datetime(2026, 9, 2, 8, 0, 0, tzinfo=timezone.utc)
BLOCK_MILLIS = int(BLOCK_TIME.timestamp() * 1000)
NOW = BLOCK_TIME + timedelta(hours=1)
STORAGE = {"Tempo": 360, "LastEpochBlock": 8_699_800, "PendingEpochAt": 0, "SubnetEpochIndex": 24_030, "BlocksSinceLastStep": 200}


def _digest(label: str) -> str:
    return "0x" + hashlib.sha256(label.encode("utf-8")).hexdigest()


GENESIS = _digest("genesis")


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


DEPOSIT_HASH = canonical_hash(DEPOSIT_NUMBER)
DEPOSIT_HEX = DEPOSIT_HASH[2:]
CANONICAL_REFERENCE = "%s:%s:%d" % (NETWORK, DEPOSIT_HASH, DEPOSIT_INDEX)


class _Scale:
    def __init__(self, value):
        self.value = value


def transfer_extrinsic(sender, dest, amount, *, function="transfer_keep_alive"):
    args = [{"name": "dest", "type": "MultiAddress", "value": dest}]
    if amount is None:
        args.append({"name": "keep_alive", "type": "bool", "value": True})
    else:
        args.append({"name": "value", "type": "Compact<u128>", "value": amount})
    return _Scale(
        {
            "extrinsic_hash": _digest("transfer-%s-%s-%s-%s" % (sender, dest, amount, function)),
            "address": sender,
            "signature": {"Sr25519": "0x" + "11" * 64},
            "call": {"call_module": "Balances", "call_function": function, "call_args": args},
            "nonce": 3,
            "era": "00",
            "tip": 0,
        }
    )


def timestamp_inherent(millis):
    return _Scale(
        {
            "extrinsic_hash": _digest("timestamp-%d" % millis),
            "call": {"call_module": "Timestamp", "call_function": "set", "call_args": [{"name": "now", "type": "Compact<u64>", "value": millis}]},
        }
    )


def status_event(index, event_id="ExtrinsicSuccess", module_id="System"):
    return {"phase": "ApplyExtrinsic", "extrinsic_idx": index, "event": {"module_id": module_id, "event_id": event_id, "attributes": {}}, "topics": []}


class FakeSubstrate:
    def __init__(self):
        self.finalized_number = FINALIZED_NUMBER
        self.best_number = BEST_NUMBER
        self.blocks = {}
        self.events = {}
        self.timestamps = {}
        self.storage = dict(STORAGE)
        self.fail = set()
        self.calls = []
        self.delay = 0.0

    def _enter(self, name):
        self.calls.append(name)
        if name in self.fail:
            raise RuntimeError("%s unavailable" % name)
        if self.delay:
            time.sleep(self.delay)

    def add_block(self, number, extrinsics, *, block_hash=None, events=None, millis=BLOCK_MILLIS):
        block_hash = block_hash or canonical_hash(number)
        self.blocks[block_hash] = {
            "header": {"number": number, "hash": block_hash, "parentHash": canonical_hash(number - 1), "digest": {"logs": []}},
            "extrinsics": list(extrinsics),
        }
        if events is not None:
            self.events[block_hash] = list(events)
        if millis is not None:
            self.timestamps[block_hash] = millis
        return block_hash

    def get_chain_finalised_head(self):
        self._enter("get_chain_finalised_head")
        return canonical_hash(self.finalized_number)

    def get_block_hash(self, block_id):
        self._enter("get_block_hash")
        if block_id < 0 or block_id > self.best_number:
            return None
        return canonical_hash(block_id)

    def get_block_number(self, block_hash=None):
        self._enter("get_block_number")
        block = self.blocks.get(block_hash)
        if block is not None:
            return block["header"]["number"]
        number = _NUMBER_BY_CANONICAL.get(block_hash)
        if number is not None and 0 <= number <= self.best_number:
            return number
        raise RuntimeError("Unable to determine block number for %s" % block_hash)

    def get_block(self, block_hash=None, **kwargs):
        self._enter("get_block")
        block = self.blocks.get(block_hash)
        if block is None:
            return None
        return {"header": dict(block["header"]), "extrinsics": list(block["extrinsics"])}

    def get_events(self, block_hash=None):
        self._enter("get_events")
        return list(self.events.get(block_hash, []))

    def query(self, module, storage_function, params=None, block_hash=None):
        self._enter("query")
        if module == "Timestamp":
            millis = self.timestamps.get(block_hash)
            return None if millis is None else _Scale(millis)
        return _Scale(self.storage[storage_function])


def metagraph_source(client, netuid, block_hash):
    return MetagraphSnapshot(
        netuid=netuid,
        block_number=client.get_block_number(block_hash),
        block_hash=block_hash,
        hotkeys=(MINER_HOTKEY.ss58_address, OTHER_HOTKEY.ss58_address),
        coldkeys=(OWNER_COLDKEY.ss58_address, STRANGER.ss58_address),
        validator_permit=(False, True),
    )


class FixedPrice:
    name = "fixed"

    def __init__(self, price=PRICE):
        self.price = price
        self.calls = 0

    def tao_price_usd(self):
        self.calls += 1
        if isinstance(self.price, Exception):
            raise self.price
        return self.price


class FakeDepositStore:
    """Reference-unique, lock-protected stand-in for ``lab_arena_credit_deposit``."""

    def __init__(self):
        self.lock = threading.Lock()
        self.rows = {}
        self.balances = {}
        self.calls = 0
        self.fail = None
        self.result_override = None

    def credit_deposit(self, *, miner_hotkey, payment_reference, amount_microusd, deposit_doc):
        with self.lock:
            self.calls += 1
            if self.fail is not None:
                raise self.fail
            if self.result_override is not None:
                return self.result_override
            if payment_reference in self.rows:
                return {"credited": False, "idempotent": True, "balance_microusd": self.balances.get(miner_hotkey, 0)}
            self.rows[payment_reference] = {"miner_hotkey": miner_hotkey, "amount_microusd": amount_microusd, "deposit_doc": deposit_doc}
            self.balances[miner_hotkey] = self.balances.get(miner_hotkey, 0) + amount_microusd
            return {"credited": True, "idempotent": False, "balance_microusd": self.balances[miner_hotkey]}


def make_chain(*, sender=OWNER_COLDKEY.ss58_address, dest=ARENA_WALLET.ss58_address, amount=AMOUNT_RAO, function="transfer_keep_alive", events=None, network=NETWORK):
    fake = FakeSubstrate()
    fake.add_block(
        DEPOSIT_NUMBER,
        [timestamp_inherent(BLOCK_MILLIS), transfer_extrinsic(sender, dest, amount, function=function)],
        events=[status_event(0), status_event(1)] if events is None else events,
    )
    config = ArenaChainConfig(endpoint=ENDPOINT, netuid=NETUID, network_name=network, request_timeout_seconds=15)
    chain = ArenaChain(config, fake, metagraph_source=metagraph_source, clock=lambda: 0.0)
    return fake, chain


def funding_config(**overrides):
    values = dict(recipient_wallet=ARENA_WALLET.ss58_address, network_name=NETWORK)
    values.update(overrides)
    return FundingConfig(**values)


def run_verify(chain, **overrides):
    values = dict(
        chain=chain,
        config=funding_config(),
        miner_hotkey=MINER_HOTKEY.ss58_address,
        block_hash=DEPOSIT_HASH,
        extrinsic_index=DEPOSIT_INDEX,
        now=NOW,
        price_source=FixedPrice(),
    )
    values.update(overrides)
    return verify_deposit(**values)


def run_confirm(chain, store, **overrides):
    values = dict(
        chain=chain,
        config=funding_config(),
        store=store,
        miner_hotkey=MINER_HOTKEY.ss58_address,
        block_hash=DEPOSIT_HASH,
        extrinsic_index=DEPOSIT_INDEX,
        now=NOW,
        price_source=FixedPrice(),
    )
    values.update(overrides)
    return confirm_funding(**values)


def rejected(chain, rule_id, **overrides):
    with pytest.raises(DepositRejected) as info:
        run_verify(chain, **overrides)
    assert info.value.rule_id == rule_id, str(info.value)
    return info.value


# Every accepted presentation of the same deposit.
REFERENCE_VARIANTS = [
    ("0x" + DEPOSIT_HEX, 1),
    ("0X" + DEPOSIT_HEX.upper(), "1"),
    (DEPOSIT_HEX, " 1 "),
    (DEPOSIT_HEX.upper(), "01"),
    ("  0x" + DEPOSIT_HEX.upper() + "\n", "\t1"),
    ("0x" + DEPOSIT_HEX, "0001"),
]


# ---------------------------------------------------------------------------
# Reference normalization and arithmetic
# ---------------------------------------------------------------------------


def test_every_accepted_reference_variant_normalizes_to_one_string():
    references = {normalize_payment_reference(NETWORK, block_hash, index) for block_hash, index in REFERENCE_VARIANTS}
    assert references == {CANONICAL_REFERENCE}
    assert normalize_payment_reference(" finney ", DEPOSIT_HASH, 1) == CANONICAL_REFERENCE
    assert parse_payment_reference(CANONICAL_REFERENCE) == (NETWORK, DEPOSIT_HASH, 1)
    assert normalize_payment_reference("test", DEPOSIT_HASH, 0).endswith(":0")


@pytest.mark.parametrize(
    "network, block_hash, index",
    [
        ("", DEPOSIT_HASH, 1),
        ("Finney", DEPOSIT_HASH, 1),
        ("finney:x", DEPOSIT_HASH, 1),
        (None, DEPOSIT_HASH, 1),
        (NETWORK, DEPOSIT_HEX[:-2], 1),
        (NETWORK, "0x" + "zz" * 32, 1),
        (NETWORK, "", 1),
        (NETWORK, None, 1),
        (NETWORK, 12345, 1),
        (NETWORK, DEPOSIT_HASH, -1),
        (NETWORK, DEPOSIT_HASH, True),
        (NETWORK, DEPOSIT_HASH, 1.0),
        (NETWORK, DEPOSIT_HASH, "1.0"),
        (NETWORK, DEPOSIT_HASH, "-1"),
        (NETWORK, DEPOSIT_HASH, "+1"),
        (NETWORK, DEPOSIT_HASH, "1 2"),
        (NETWORK, DEPOSIT_HASH, ""),
        (NETWORK, DEPOSIT_HASH, "1" * 13),
        (NETWORK, DEPOSIT_HASH, None),
    ],
)
def test_malformed_references_are_rejected(network, block_hash, index):
    with pytest.raises(MalformedReference):
        normalize_payment_reference(network, block_hash, index)


def test_parse_payment_reference_requires_canonical_form():
    for bad in ("finney:0X%s:1" % DEPOSIT_HEX, "finney:%s:1" % DEPOSIT_HEX, "finney:0x%s:01" % DEPOSIT_HEX, "Finney:0x%s:1" % DEPOSIT_HEX, 42, ""):
        with pytest.raises(MalformedReference):
            parse_payment_reference(bad)
    assert normalize_extrinsic_index("  42 ") == 42


def test_rao_to_microusd_rounds_down_exactly():
    assert rao_to_microusd(AMOUNT_RAO, PRICE) == EXPECTED_MICROUSD
    assert rao_to_microusd(1_000_000_000, Decimal("123.456789")) == 123_456_789
    assert rao_to_microusd(1, Decimal("500")) == 0
    assert rao_to_microusd(3, Decimal("333.333333")) == 0
    assert rao_to_microusd(0, PRICE) == 0
    amount = 1_234_567
    expected = int((Decimal(amount) * PRICE / 1000).to_integral_value(rounding=ROUND_FLOOR))
    assert rao_to_microusd(amount, PRICE) == expected
    assert rao_to_microusd(1_999, Decimal("1")) == 1  # 1.999 floors to 1
    # Large balances keep exact precision: 21M TAO at a 12-decimal price.
    whole = 21_000_000 * 1_000_000_000
    price = Decimal("1234.567890123456")
    assert rao_to_microusd(whole, price) == int((Decimal(whole) * price / 1000).to_integral_value(rounding=ROUND_FLOOR))
    for bad_amount in (-1, True, 1.5, "1"):
        with pytest.raises(FundingError):
            rao_to_microusd(bad_amount, PRICE)
    for bad_price in (Decimal("0"), Decimal("-1"), Decimal("NaN"), Decimal("Infinity"), 412.37, "412.37"):
        with pytest.raises(PriceUnavailable):
            rao_to_microusd(1, bad_price)


class _Response:
    def __init__(self, body, status=200):
        self._body = body
        self.status = status

    def read(self, limit=-1):
        return self._body if limit < 0 else self._body[:limit]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_coingecko_price_source_parses_and_fails_closed():
    seen = []

    def urlopen(request, timeout=None):
        seen.append((request.full_url, timeout, request.get_header("Accept")))
        return _Response(json.dumps({"bittensor": {"usd": 412.37}}).encode("utf-8"))

    source = coingecko_price_source(urlopen, timeout=4)
    assert source.tao_price_usd() == Decimal("412.37")
    assert source.name == "coingecko"
    assert seen == [(COINGECKO_PRICE_URL, 4, "application/json")]

    def failing(kind):
        def _urlopen(request, timeout=None):
            if kind == "raise":
                raise OSError("network down")
            if kind == "status":
                return _Response(b"{}", status=503)
            if kind == "json":
                return _Response(b"not json")
            if kind == "missing":
                return _Response(b'{"bittensor": {}}')
            if kind == "zero":
                return _Response(b'{"bittensor": {"usd": 0}}')
            if kind == "negative":
                return _Response(b'{"bittensor": {"usd": -3}}')
            if kind == "bool":
                return _Response(b'{"bittensor": {"usd": true}}')
            if kind == "huge":
                return _Response(b"[" + b"1," * 70_000 + b"1]")
            if kind == "list":
                return _Response(b"[1, 2]")
            raise AssertionError(kind)

        return _urlopen

    for kind in ("raise", "status", "json", "missing", "zero", "negative", "bool", "huge", "list"):
        with pytest.raises(PriceUnavailable):
            coingecko_price_source(failing(kind), timeout=4).tao_price_usd()
    with pytest.raises(FundingError):
        coingecko_price_source(urlopen, timeout=0)
    with pytest.raises(FundingError):
        coingecko_price_source("not callable", timeout=1)


# ---------------------------------------------------------------------------
# verify_deposit
# ---------------------------------------------------------------------------


def test_verify_deposit_accepts_a_finalized_owned_transfer():
    fake, chain = make_chain()
    price = FixedPrice()
    verified = run_verify(chain, price_source=price)
    assert verified.payment_reference == CANONICAL_REFERENCE
    assert verified.network == NETWORK
    assert verified.block_number == DEPOSIT_NUMBER
    assert verified.block_hash == DEPOSIT_HASH
    assert verified.extrinsic_index == DEPOSIT_INDEX
    assert verified.extrinsic_hash.startswith("0x") and len(verified.extrinsic_hash) == 66
    assert verified.sender == OWNER_COLDKEY.ss58_address
    assert verified.recipient == ARENA_WALLET.ss58_address
    assert verified.amount_rao == AMOUNT_RAO
    assert verified.block_timestamp == BLOCK_TIME
    assert verified.price_usd == PRICE
    assert verified.price_source == "fixed"
    assert verified.price_observed_at == NOW
    assert verified.amount_microusd == EXPECTED_MICROUSD
    assert price.calls == 1

    document = verified.deposit_doc()
    assert document["schema_version"] == DEPOSIT_DOCUMENT_SCHEMA_VERSION
    assert document["price_usd"] == "412.37"
    assert document["price_observed_at"] == "2026-09-02T09:00:00.000Z"
    assert document["block_timestamp"] == "2026-09-02T08:00:00.000Z"
    assert document["amount_microusd"] == EXPECTED_MICROUSD
    assert document["amount_rao"] == AMOUNT_RAO
    assert document["extrinsic_hash"] == verified.extrinsic_hash
    assert document["sender"] == OWNER_COLDKEY.ss58_address
    assert document["recipient"] == ARENA_WALLET.ss58_address
    assert document["block_number"] == DEPOSIT_NUMBER and document["block_hash"] == DEPOSIT_HASH
    assert document["extrinsic_index"] == DEPOSIT_INDEX
    assert document["payment_reference"] == CANONICAL_REFERENCE
    assert document["miner_hotkey"] == MINER_HOTKEY.ss58_address
    verify_hashed_document(document, "deposit_hash")
    json.dumps(document)

    for block_hash, index in REFERENCE_VARIANTS:
        assert run_verify(chain, block_hash=block_hash, extrinsic_index=index).payment_reference == CANONICAL_REFERENCE


def test_verify_deposit_rejects_each_rule_in_order():
    fake, chain = make_chain()
    assert set(DEPOSIT_RULE_IDS) == {"malformed_reference", "finality", "not_transfer", "transfer_failed", "wrong_destination", "sender_not_owner", "stale_block", "amount_invalid"}

    rejected(chain, "malformed_reference", block_hash="0x1234")
    rejected(chain, "malformed_reference", extrinsic_index="one")

    # finality: unknown block, fork block, best-chain block beyond the finalized head
    rejected(chain, "finality", block_hash=_digest("unknown"))
    fork = fake.add_block(DEPOSIT_NUMBER, [timestamp_inherent(BLOCK_MILLIS), transfer_extrinsic(OWNER_COLDKEY.ss58_address, ARENA_WALLET.ss58_address, AMOUNT_RAO)], block_hash=_digest("fork"), events=[status_event(0), status_event(1)])
    rejected(chain, "finality", block_hash=fork)
    unfinalized = fake.add_block(FINALIZED_NUMBER + 2, [timestamp_inherent(BLOCK_MILLIS), transfer_extrinsic(OWNER_COLDKEY.ss58_address, ARENA_WALLET.ss58_address, AMOUNT_RAO)], events=[status_event(0), status_event(1)])
    rejected(chain, "finality", block_hash=unfinalized)

    # not_transfer: the inherent, an out-of-range index
    rejected(chain, "not_transfer", extrinsic_index=0)
    rejected(chain, "not_transfer", extrinsic_index=2)

    # transfer_failed: absent success event, other pallet's event only, ExtrinsicFailed
    fake.events[DEPOSIT_HASH] = [status_event(0)]
    rejected(chain, "transfer_failed")
    fake.events[DEPOSIT_HASH] = [status_event(0), status_event(1, "Transfer", module_id="Balances")]
    rejected(chain, "transfer_failed")
    fake.events[DEPOSIT_HASH] = [status_event(0), status_event(1, "ExtrinsicFailed")]
    rejected(chain, "transfer_failed")
    fake.events[DEPOSIT_HASH] = [status_event(0), status_event(1)]

    # wrong_destination
    rejected(chain, "wrong_destination", config=funding_config(recipient_wallet=STRANGER.ss58_address))
    _, wrong_dest_chain = make_chain(dest=STRANGER.ss58_address)
    rejected(wrong_dest_chain, "wrong_destination")

    # sender_not_owner: stranger signs; unregistered miner hotkey; owner of a different hotkey
    _, stranger_chain = make_chain(sender=STRANGER.ss58_address)
    rejected(stranger_chain, "sender_not_owner")
    rejected(chain, "sender_not_owner", miner_hotkey=ARENA_WALLET.ss58_address)
    rejected(chain, "sender_not_owner", miner_hotkey=OTHER_HOTKEY.ss58_address)

    # stale_block: too old, and in the future
    rejected(chain, "stale_block", now=BLOCK_TIME + timedelta(seconds=86_401))
    rejected(chain, "stale_block", now=BLOCK_TIME - timedelta(seconds=FUTURE_BLOCK_TOLERANCE_SECONDS + 1))
    assert run_verify(chain, now=BLOCK_TIME + timedelta(seconds=86_400)).amount_microusd == EXPECTED_MICROUSD
    assert run_verify(chain, now=BLOCK_TIME - timedelta(seconds=FUTURE_BLOCK_TOLERANCE_SECONDS)).amount_microusd == EXPECTED_MICROUSD
    rejected(chain, "stale_block", config=funding_config(max_age_seconds=60), now=BLOCK_TIME + timedelta(seconds=61))

    # amount_invalid: floors to zero, transfer_all, zero amount
    rejected(chain, "amount_invalid", price_source=FixedPrice(Decimal("0.000000001")))
    _, transfer_all_chain = make_chain(amount=None, function="transfer_all")
    rejected(transfer_all_chain, "amount_invalid")
    _, zero_chain = make_chain(amount=0)
    rejected(zero_chain, "amount_invalid")


def test_verify_deposit_propagates_chain_price_and_input_failures():
    fake, chain = make_chain()
    fake.fail.add("get_block")
    with pytest.raises(ArenaChainError):
        run_verify(chain)
    fake.fail.clear()
    fake.fail.add("get_events")
    with pytest.raises(ArenaChainError):
        run_verify(chain)
    fake.fail.clear()
    del fake.timestamps[DEPOSIT_HASH]
    with pytest.raises(ArenaChainError):
        run_verify(chain)
    fake.timestamps[DEPOSIT_HASH] = BLOCK_MILLIS

    price = FixedPrice(PriceUnavailable("coingecko down"))
    with pytest.raises(PriceUnavailable):
        run_verify(chain, price_source=price)
    with pytest.raises(PriceUnavailable):
        run_verify(chain, price_source=FixedPrice(412.37))  # a float is refused, Decimal only

    with pytest.raises(FundingError, match="timezone-aware"):
        run_verify(chain, now=NOW.replace(tzinfo=None))
    with pytest.raises(ArenaContractError):
        run_verify(chain, miner_hotkey="nope")
    with pytest.raises(FundingError, match="does not match"):
        run_verify(chain, config=funding_config(network_name="test"))
    with pytest.raises(FundingError):
        run_verify("not a chain")

    # A rejected presentation never costs a price request.
    counting = FixedPrice()
    rejected(chain, "not_transfer", extrinsic_index=0, price_source=counting)
    assert counting.calls == 0


def test_price_is_fetched_after_chain_checks_and_snapshotted():
    fake, chain = make_chain()
    price = FixedPrice(Decimal("999.000"))
    verified = run_verify(chain, price_source=price)
    assert verified.deposit_doc()["price_usd"] == "999.000"
    assert verified.amount_microusd == 499_500_000
    assert fake.calls.index("get_block") < fake.calls.index("query")


# ---------------------------------------------------------------------------
# confirm_funding
# ---------------------------------------------------------------------------


def test_confirm_funding_credits_once_and_replays_idempotently():
    fake, chain = make_chain()
    store = FakeDepositStore()
    receipt = run_confirm(chain, store)
    assert receipt.credited is True and receipt.idempotent is False
    assert receipt.balance_microusd == EXPECTED_MICROUSD
    assert receipt.deposit_doc == receipt.deposit.deposit_doc()
    assert store.rows[CANONICAL_REFERENCE]["deposit_doc"]["price_usd"] == "412.37"
    assert store.rows[CANONICAL_REFERENCE]["amount_microusd"] == EXPECTED_MICROUSD

    replay = run_confirm(chain, store, block_hash=DEPOSIT_HEX.upper(), extrinsic_index=" 01 ")
    assert replay.credited is False and replay.idempotent is True
    assert replay.balance_microusd == EXPECTED_MICROUSD
    assert len(store.rows) == 1 and store.calls == 2


def test_confirm_funding_creates_no_credit_when_verification_or_store_fails():
    fake, chain = make_chain()
    store = FakeDepositStore()

    with pytest.raises(DepositRejected):
        run_confirm(chain, store, extrinsic_index=0)
    fake.fail.add("get_block")
    with pytest.raises(ArenaChainError):
        run_confirm(chain, store)
    fake.fail.clear()
    with pytest.raises(PriceUnavailable):
        run_confirm(chain, store, price_source=FixedPrice(PriceUnavailable("down")))
    assert store.calls == 0 and store.rows == {}

    store.fail = RuntimeError("database unavailable")
    with pytest.raises(FundingStoreError) as info:
        run_confirm(chain, store)
    assert isinstance(info.value.__cause__, RuntimeError)
    assert store.rows == {} and store.balances == {}

    store.fail = None
    for override in (
        {"credited": True, "idempotent": True, "balance_microusd": 1},
        {"credited": False, "idempotent": False, "balance_microusd": 1},
        {"credited": True, "idempotent": False, "balance_microusd": -1},
        {"credited": True, "idempotent": False, "balance_microusd": "1"},
        {"credited": "yes", "idempotent": False, "balance_microusd": 1},
        "ok",
    ):
        store.result_override = override
        with pytest.raises(FundingStoreError):
            run_confirm(chain, store)
    store.result_override = None
    assert run_confirm(chain, store).credited is True


def test_concurrent_presentations_in_every_format_credit_once():
    fake, chain = make_chain()
    fake.delay = 0.001
    store = FakeDepositStore()
    receipts = []
    errors = []
    barrier = threading.Barrier(len(REFERENCE_VARIANTS) * 2)

    def present(block_hash, index):
        try:
            barrier.wait(timeout=30)
            receipts.append(run_confirm(chain, store, block_hash=block_hash, extrinsic_index=index))
        except Exception as exc:  # pragma: no cover - surfaced through the assertion below
            errors.append(exc)

    threads = [threading.Thread(target=present, args=variant) for variant in REFERENCE_VARIANTS * 2]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=120)

    assert not errors
    assert len(receipts) == len(REFERENCE_VARIANTS) * 2
    assert sum(1 for receipt in receipts if receipt.credited) == 1
    assert sum(1 for receipt in receipts if receipt.idempotent) == len(receipts) - 1
    assert {receipt.deposit.payment_reference for receipt in receipts} == {CANONICAL_REFERENCE}
    assert list(store.rows) == [CANONICAL_REFERENCE]
    assert store.balances == {MINER_HOTKEY.ss58_address: EXPECTED_MICROUSD}
    assert all(receipt.balance_microusd == EXPECTED_MICROUSD for receipt in receipts)


def test_funding_config_validation():
    with pytest.raises(ArenaContractError):
        funding_config(recipient_wallet="nope")
    with pytest.raises(ValueError):
        funding_config(network_name="Finney")
    with pytest.raises(FundingError):
        funding_config(max_age_seconds=0)
    with pytest.raises(FundingError):
        funding_config(max_age_seconds=True)
    assert funding_config(network_name=" test ").network_name == "test"
    with pytest.raises(ValueError):
        DepositRejected("no_such_rule")
