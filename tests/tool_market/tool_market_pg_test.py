"""The tool market on a disposable PostgreSQL: order book, tabs, calls, money.

Runs scripts/770 on the Lab Arena base (179) through the harness the Arena
tests use. The vault, vendor and chain are fakes; every SQL rule is real.
Each test gets its own terms hash, and the book lists only keys whose
miner accepted it, so tests never see each other's supply.
"""

from __future__ import annotations

import dataclasses
import json
import random
import uuid

import pytest
from cryptography.hazmat.primitives.asymmetric import ec

from lab_arena.broker import ProviderResponse
from lab_arena.chain import ArenaChainError
from lab_arena.lead_orders.auth import BuyerIdentity
from lab_arena.lead_orders.gamma import GammaReading
from lab_arena.lead_orders.vault_client import VaultTransportError
from lab_arena.store import ArenaStoreError, PsycopgTransport
from lab_arena.tool_market import market as market_module
from lab_arena.tool_market.config import ToolMarketConfig
from lab_arena.tool_market.market import ATTESTATION, ToolMarket, ToolMarketError
from lab_arena.tool_market.store import ToolMarketStore
from lead_order_vault import envelope as vault_envelope
from tests.lab_arena.lab_arena_pg_harness import database_with_lab_arena_migration

MIGRATIONS = ("179-lab-arena-v1.sql", "770-tool-market.sql")
B58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
USD = 1_000_000
VAULT_KEY = vault_envelope.compressed(ec.generate_private_key(ec.SECP256K1()).public_key()).hex()
PROFILE = {"tool": "harvestapi", "operation": "linkedin_profile",
           "params": {"url": "https://www.linkedin.com/in/someone"}}
SEARCH = {"tool": "exa", "operation": "search", "params": {"query": "payroll"}}
VALIDATE = {"tool": "zerobounce", "operation": "validate", "params": {"email": "ada@acme.com"}}
FIND = {"tool": "hunter", "operation": "email_finder", "params": {"domain": "acme.com", "full_name": "Ada Lovelace"}}


def hotkey() -> str:
    return "5" + "".join(random.choice(B58) for _ in range(47))


HARVEST_PROFILE = {"status": 200, "element": {"id": "profile-id"}}


def ok(document=None, status=200, headers=None):
    return ProviderResponse(status, headers or {"content-type": "application/json"},
                            json.dumps(document if document is not None else HARVEST_PROFILE).encode())


@pytest.fixture(scope="module")
def database():
    yield from database_with_lab_arena_migration(MIGRATIONS)


@pytest.fixture
def admin(database):
    """An owner cursor on the scratch database, past the append-only triggers."""

    psycopg2, dsn = database
    connection = psycopg2.connect(**dsn)
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            cursor.execute("SET session_replication_role=replica")
            yield cursor
    finally:
        connection.close()


@pytest.fixture(scope="module")
def store(database):
    psycopg2, dsn = database

    def connect():
        connection = psycopg2.connect(**dsn)
        connection.autocommit = True
        return connection

    transport = PsycopgTransport(connect, role="lab_arena_service")
    bound = ToolMarketStore(transport)
    bound.bind_gamma_source(genesis="0x" + "00" * 32, netuid=71, service="5" * 48, registered_at=1)
    yield bound
    transport.close()


class FakeVault:
    """Opens the test's envelopes; ``status`` and ``balance`` per secret."""

    def __init__(self):
        self.secrets = {}
        self.answers = {}
        self.checks = 0

    def seal(self, tool, miner, secret):
        envelope = vault_envelope.seal(secret, vault_envelope.Header(tool, "runtime", miner, VAULT_KEY))
        self.secrets[envelope] = (tool, miner, secret)
        return envelope

    def check(self, *, provider, envelope, management_envelope):
        self.checks += 1
        tool, miner, secret = self.secrets[envelope]
        answer = {"provider": provider, "miner_hotkey": miner,
                  "key_fingerprint": vault_envelope.fingerprint(tool, secret), "status": "valid", "balance": 50}
        answer.update(self.answers.get(secret, {}))
        return answer


class FakeVendor:
    """Answers per sealed key, in order; records what each key was sent."""

    def __init__(self, vault):
        self.vault = vault
        self.script = {}
        self.sent = []

    def send(self, *, method, url, headers, body, timeout_seconds, max_response_bytes=None):
        secret = self.vault.secrets[headers["authorization"][len("Bearer "):]][2]
        self.sent.append((secret, method, url, body))
        answers = self.script.get(secret) or [ok()]
        item = answers.pop(0) if len(answers) > 1 else answers[0]
        if isinstance(item, Exception):
            raise item
        return item


class FakeGamma:
    def __init__(self):
        self.units = {}
        self.block = 100

    def read(self, hk, head=None):
        self.block += 1
        return GammaReading(hk, self.block, "0x" + "%064x" % self.block, self.units.get(hk, 0), True)


class World:
    def __init__(self, store):
        self.terms = uuid.uuid4().hex + uuid.uuid4().hex
        self.config = ToolMarketConfig(mode="live", tools=frozenset({"harvestapi", "exa", "zerobounce", "hunter"}),
                                       gamma_service="5" * 48, terms_version="v1", terms_sha256=self.terms,
                                       public_base_url="https://gw.example")
        self.vault = FakeVault()
        self.vendor = FakeVendor(self.vault)
        self.gamma = FakeGamma()
        self.store = store
        self.unregistered = set()
        self.market = ToolMarket(config=self.config, store=store, vault=self.vault, transport=self.vendor,
                                 gamma_reader=self.gamma, current_epoch=lambda: 7,
                                 is_registered=lambda hk: hk not in self.unregistered)

    def who(self, hk):
        return BuyerIdentity(hotkey=hk, signed_at=1, host="gw.example", signature="0x" + "ab" * 64,
                             body_sha256="0" * 64)

    def accept(self, hk, role):
        self.market.accept_terms(self.who(hk), {"role": role, "terms_version": "v1", "terms_sha256": self.terms})

    def miner_key(self, secret, ask, *, tool="harvestapi", cap=10 * USD, miner=None):
        miner = miner or hotkey()
        self.accept(miner, "miner")
        listed = self.register(miner, {"tool": tool, "envelope": self.vault.seal(tool, miner, secret),
                                       "daily_cap_microusd": cap})
        self.market.post_ask(self.who(miner), {"key_id": listed["key_id"], "ask_microusd": ask})
        return miner, listed["key_id"]

    def register(self, miner, body):
        """A register request as the gateway hands it over: parsed and as signed."""

        self.config = dataclasses.replace(
            self.market._config, approved_suppliers=self.market._config.approved_suppliers | {(body["tool"], miner)})
        self.market._config = self.config
        body = dict(body, attestation=ATTESTATION) if "attestation" not in body else body
        return self.market.register_key(self.who(miner), body, json.dumps(body).encode())

    def buyer(self, usd=1, budget=100_000, seconds=600):
        buyer = hotkey()
        self.gamma.units[buyer] = usd * 10**9
        self.accept(buyer, "buyer")
        return buyer, self.market.open_tab(self.who(buyer), {"budget_microusd": budget, "seconds": seconds})["token"]


@pytest.fixture
def world(store):
    return World(store)


def test_a_call_goes_to_the_cheapest_key_and_pays_its_ask_plus_ten_percent(world):
    dear, _ = world.miner_key("dear-key", 12_000)
    cheap, cheap_key = world.miner_key("cheap-key", 10_000)
    buyer, token = world.buyer(usd=1, budget=100_000)
    assert world.market.order_book("harvestapi")["asks"] == [
        {"ask_microusd": 10_000, "buyer_pays_microusd": 11_000, "private": False},
        {"ask_microusd": 12_000, "buyer_pays_microusd": 13_200, "private": False}]
    result = world.market.call(token, PROFILE)
    assert result["status"] == 200 and result["charge_microusd"] == 11_000 and result["body"] == HARVEST_PROFILE
    secret, method, url, _ = world.vendor.sent[0]
    assert secret == "cheap-key" and method == "GET"
    assert url == "https://api.harvestapi.io/linkedin/profile?query=https%3A%2F%2Fwww.linkedin.com%2Fin%2Fsomeone"
    assert world.market.tab(token)["committed_microusd"] == 11_000
    earned = world.market.statement(cheap)["keys"][0]
    assert earned["key_id"] == cheap_key and earned["owed_microusd"] == 10_000 and earned["calls"] == 1
    assert world.market.statement(dear)["keys"][0]["owed_microusd"] == 0
    # An open tab holds its whole budget; a closed one only what it spent.
    assert world.market.balance(buyer)["balance_microusd"] == USD - 100_000
    world.market.close_tab(token)
    assert world.market.balance(buyer)["balance_microusd"] == USD - 11_000
    with pytest.raises(ToolMarketError) as closed:
        world.market.call(token, PROFILE)
    assert closed.value.code == "tab_closed"


def test_a_key_that_ran_out_fails_over_and_goes_to_the_back(world):
    # HarvestAPI has no free check, so a 402 cannot be confirmed: the key
    # stays listed, and goes behind keys that answered once 3 buyers saw it.
    miner, _ = world.miner_key("empty-key", 9_000)
    world.miner_key("full-key", 10_000)
    world.vendor.script["empty-key"] = [ok({"error": "no credits"}, status=402)]
    for n in range(3):
        _, token = world.buyer()
        result = world.market.call(token, PROFILE)
        assert result["charge_microusd"] == 11_000
        assert world.market.tab(token)["committed_microusd"] == 11_000  # the failed try costs nothing
    assert [s[0] for s in world.vendor.sent] == ["empty-key", "full-key"] * 3
    assert world.market.statement(miner)["keys"][0]["health"]["status"] != "invalid"
    world.market.call(token, PROFILE)
    assert world.vendor.sent[-1][0] == "full-key" and len(world.vendor.sent) == 7


def test_unknown_usage_charges_the_buyer_nothing_and_the_miner_bears_it(world):
    miner, _ = world.miner_key("exa-key", 900_000, tool="exa")
    _, token = world.buyer()
    world.vendor.script["exa-key"] = [ok({"costDollars": {"total": 0.002}}), ok({"results": []}),
                                      ok({"costDollars": {"total": 1}}, status=500),
                                      VaultTransportError("provider_unreachable", forwarded=None)]
    assert world.market.call(token, SEARCH)["charge_microusd"] == 1_980  # $0.002 x 90% x 1.1
    assert world.market.call(token, SEARCH)["charge_microusd"] == 0  # 2xx without a cost
    assert world.market.call(token, SEARCH)["charge_microusd"] == 0  # vendor error
    with pytest.raises(ToolMarketError) as unreachable:
        world.market.call(token, SEARCH)
    assert unreachable.value.status == 502 and len(world.vendor.sent) == 4  # never retried on another key
    assert world.market.tab(token)["committed_microusd"] == 1_980
    key = world.market.statement(miner)["keys"][0]
    assert key["owed_microusd"] == 1_800 and key["uncertain_calls"] == 3
    assert json.loads(world.vendor.sent[0][3]) == {"query": "payroll"}


def test_a_request_the_vault_never_sent_moves_to_the_next_key(world):
    world.miner_key("down-key", 9_000)
    world.miner_key("up-key", 10_000)
    _, token = world.buyer()
    world.vendor.script["down-key"] = [VaultTransportError("vault_unreachable", forwarded=False)]
    # A 429 is the key's limit, with Retry-After too: the call moves on.
    world.vendor.script["up-key"] = [ok({"error": "slow down"}, status=429, headers={"retry-after": "2"})]
    with pytest.raises(ToolMarketError) as empty:
        world.market.call(token, PROFILE)
    assert empty.value.code == "no_supply" and world.market.tab(token)["committed_microusd"] == 0


def test_rate_limited_and_unreliable_keys_go_to_the_back_of_the_book(world):
    world.miner_key("flaky-key", 8_000)
    world.miner_key("limited-key", 9_000)
    world.miner_key("steady-key", 10_000)
    world.vendor.script["flaky-key"] = [ok(status=500)]
    for _ in range(3):  # unknown usage for 3 buyers, never retried elsewhere
        _, token = world.buyer()
        assert world.market.call(token, PROFILE)["status"] == 500
    world.vendor.script["limited-key"] = [ok(status=429, headers={})]
    for n in range(3):
        _, token = world.buyer()
        world.market.call(token, PROFILE)
        if n < 2:  # one or two buyers' 429s leave the key where it is
            assert world.market.order_book("harvestapi")["asks"][0]["ask_microusd"] == 9_000
    world.vendor.sent.clear()
    world.market.call(token, PROFILE)
    # The flaky key (3 buyers saw unknown usage this hour) and the limited
    # key (3 buyers saw a 429 this minute) are tried after the steady one.
    assert [s[0] for s in world.vendor.sent] == ["steady-key"]


def test_only_vendor_errors_push_a_key_back_not_what_a_buyer_can_cause(world):
    world.miner_key("pushback-cheap", 8_000)
    world.miner_key("pushback-other", 10_000)
    _, token = world.buyer()
    # A buyer can cause unknown usage (an oversized answer) or a 403 (a
    # shaped request): the miner bears it, but one buyer cannot push the
    # key back, however often.
    world.vendor.script["pushback-cheap"] = [VaultTransportError("response_too_large", forwarded=True)] * 3 \
        + [ok(status=403)] * 3 + [ok()]
    for _ in range(3):
        with pytest.raises(ToolMarketError):
            world.market.call(token, PROFILE)
    for _ in range(3):
        world.market.call(token, PROFILE)  # skipped, served by the other key
    world.vendor.sent.clear()
    world.market.call(token, PROFILE)
    assert [s[0] for s in world.vendor.sent] == ["pushback-cheap"]


def test_a_tab_short_for_one_key_still_buys_from_a_cheaper_one(world):
    world.miner_key("short-pricey", 200_000)
    world.miner_key("short-cheap", 10_000)
    world.vendor.script["short-cheap"] = [ok(status=429, headers={})] * 3 + [ok()]
    for _ in range(3):  # 429s for 3 buyers: the cheap key goes to the back
        _, other = world.buyer()
        with pytest.raises(ToolMarketError):
            world.market.call(other, PROFILE)
    _, token = world.buyer(budget=50_000)
    assert world.market.order_book("harvestapi")["asks"][0]["ask_microusd"] == 200_000
    assert world.market.call(token, PROFILE)["charge_microusd"] == 11_000
    assert world.vendor.sent[-1][0] == "short-cheap"  # the 220,000 reserve would not fit
    world.vendor.script["short-cheap"] = [ok(status=403)]
    with pytest.raises(ToolMarketError) as short:
        world.market.call(token, PROFILE)
    assert (short.value.status, short.value.code) == (402, "tab_budget")


def test_a_bad_token_reads_neither_the_chain_nor_the_book(world):
    _, closed = world.buyer()
    world.market.close_tab(closed)

    def chain_read():
        raise AssertionError("the chain was read")

    world.market._epoch = chain_read
    world.market._supply = lambda tool: chain_read()
    for token in ("tmt_nope", closed):
        with pytest.raises(ToolMarketError) as refused:
            world.market.call(token, PROFILE)
        assert refused.value.status == 401


def test_the_catalog_lists_each_tool_its_operations_and_best_price(world):
    world.miner_key("cat-cheap", 9_000)
    world.miner_key("cat-dear", 12_000)
    tools = {t["tool"]: t for t in world.market.tool_catalog()["tools"]}
    assert set(tools) == set(world.config.tools)
    harvest = tools["harvestapi"]
    assert (harvest["best_ask_microusd"], harvest["buyer_pays_microusd"], harvest["max_ask_microusd"]) \
        == (9_000, 9_900, 250_000)
    assert {op["operation"] for op in harvest["operations"]} == {"linkedin_profile", "linkedin_company"}
    assert tools["exa"]["best_ask_microusd"] is None  # nobody sells it yet


def test_polling_a_balance_adds_a_reading_only_when_it_changes(world, admin):
    buyer = hotkey()
    read = dict(hotkey=buyer, block_number=7, block_hash="0x" + "07" * 32, spent_units=5, available=True)
    for _ in range(3):
        world.store.record_gamma_reading(**read)
    world.store.record_gamma_reading(**dict(read, block_number=8, block_hash="0x" + "08" * 32))
    admin.execute("SELECT count(*) FROM public.tool_market_gamma_readings WHERE hotkey=%s", (buyer,))
    assert admin.fetchone()[0] == 2


def test_a_buyer_pages_their_own_finished_calls_by_watermark(world, admin):
    world.miner_key("usage-key", 10_000)
    buyer, token = world.buyer()
    other, other_token = world.buyer()
    world.vendor.script["usage-key"] = [ok(HARVEST_PROFILE)]
    for _ in range(3):
        world.market.call(token, PROFILE)
    world.market.call(other_token, PROFILE)
    # A call is listed only once it can no longer change.
    assert world.market.usage(world.who(buyer), 0, 100)["calls"] == []
    # Age the calls. A later sequence with an earlier opened_at (two calls
    # racing) waits for the older one, so paging cannot step past it.
    latest = ("(SELECT max(c.sequence) FROM public.tool_market_calls c JOIN public.tool_market_tabs t "
              "USING (tab_id) WHERE t.hotkey=%s)")
    age = "UPDATE public.tool_market_calls SET opened_at=opened_at-interval '11 minutes' WHERE sequence"
    admin.execute(age + "=" + latest, (buyer,))
    assert world.market.usage(world.who(buyer), 0, 100)["calls"] == []
    admin.execute(age + "<>" + latest, (buyer,))
    first = world.market.usage(world.who(buyer), 0, 2)
    assert [c["charge_microusd"] for c in first["calls"]] == [11_000, 11_000]
    assert set(first["calls"][0]) == {"call_id", "sequence", "tool", "operation", "opened_at", "status",
                                      "http_status", "units", "charge_microusd", "settled_at"}  # no answers
    rest = world.market.usage(world.who(buyer), first["watermark"], 100)
    assert len(rest["calls"]) == 1 and world.market.usage(world.who(buyer), rest["watermark"], 100)["calls"] == []
    assert len(world.market.usage(world.who(other), 0, 100)["calls"]) == 1  # only the signer's own
    for after, limit in ((-1, 10), (0, 0), (0, 501), ("1", 10)):
        with pytest.raises(ToolMarketError):
            world.market.usage(world.who(buyer), after, limit)


def test_a_private_call_goes_only_to_a_private_supplier(world):
    world.miner_key("open-key", 8_000)
    trusted, _ = world.miner_key("trusted-key", 10_000)
    world.market._config = dataclasses.replace(world.config, private_suppliers=frozenset({trusted}))
    _, token = world.buyer()
    listed = world.market.order_book("harvestapi")
    assert listed["request_privacy"] == "unverified" and [a["private"] for a in listed["asks"]] == [False, True]
    world.market.call(token, dict(PROFILE, private=True))
    assert [s[0] for s in world.vendor.sent] == ["trusted-key"]
    world.market.call(token, PROFILE)  # not private: the cheapest key
    assert world.vendor.sent[-1][0] == "open-key"
    with pytest.raises(ToolMarketError) as bad:
        world.market.call(token, dict(PROFILE, private="yes"))
    assert bad.value.code == "private_invalid"
    with pytest.raises(ToolMarketError) as odd:  # not a crash
        world.market.call(token, dict(PROFILE, tool=["harvestapi"]))
    assert odd.value.code == "tool_unavailable"


def test_tabs_need_terms_balance_and_budget(world):
    world.miner_key("budget-key", 10_000)
    broke = hotkey()
    world.gamma.units[broke] = 10**6  # $0.001
    with pytest.raises(ToolMarketError) as terms:
        world.market.open_tab(world.who(broke), {"budget_microusd": 1_000, "seconds": 600})
    assert terms.value.code == "terms_required"
    world.accept(broke, "buyer")
    with pytest.raises(ToolMarketError) as low:
        world.market.open_tab(world.who(broke), {"budget_microusd": 1_001, "seconds": 600})
    assert low.value.status == 402 and low.value.detail["balance"]["balance_microusd"] == 1_000

    def chain_down(hk, head=None):
        raise ArenaChainError("finalized head unavailable")

    read, world.gamma.read = world.gamma.read, chain_down
    with pytest.raises(ToolMarketError) as down:  # any chain error, not only a Gamma read
        world.market.open_tab(world.who(broke), {"budget_microusd": 1_000, "seconds": 600})
    assert (down.value.status, down.value.code) == (503, "gamma_unavailable")
    world.gamma.read = read
    # One call reserves $0.011; a $0.015 budget allows one, not two.
    _, token = world.buyer(budget=15_000)
    world.market.call(token, PROFILE)
    with pytest.raises(ToolMarketError) as budget:
        world.market.call(token, PROFILE)
    assert budget.value.code == "tab_budget"
    for bad in ({"budget_microusd": 0, "seconds": 600}, {"budget_microusd": 1, "seconds": 59}, {"budget_microusd": 1}):
        with pytest.raises(ToolMarketError):
            world.market.open_tab(world.who(broke), bad)
    with pytest.raises(ToolMarketError) as unknown:
        world.market.call("tmt_nope", PROFILE)
    assert unknown.value.code == "tab_unknown"


def test_listing_rules(world):
    miner, key_id = world.miner_key("listed-key", 10_000)
    other, third = hotkey(), hotkey()
    world.accept(other, "miner")
    world.accept(third, "miner")
    # The same vendor key, sealed again by anyone, is refused while listed.
    with pytest.raises(ToolMarketError) as twice:
        world.register(third, {"tool": "harvestapi", "daily_cap_microusd": USD,
                               "envelope": world.vault.seal("harvestapi", third, "listed-key")})
    assert twice.value.code == "key_already_listed"
    # An envelope sealed for another miner or tool is refused before the vault is asked.
    checks = world.vault.checks
    for envelope in (world.vault.seal("harvestapi", miner, "k2"), world.vault.seal("exa", other, "k3")):
        with pytest.raises(ToolMarketError):
            world.register(other, {"tool": "harvestapi", "envelope": envelope, "daily_cap_microusd": USD})
    assert world.vault.checks == checks
    # Only the owner may price, check or revoke a key.
    for act in (lambda: world.market.post_ask(world.who(other), {"key_id": key_id, "ask_microusd": 1}),
                lambda: world.market.revoke_key(world.who(other), key_id)):
        with pytest.raises(ToolMarketError):
            act()
    world.market.post_ask(world.who(miner), {"key_id": key_id, "ask_microusd": None})
    assert world.market.order_book("harvestapi")["asks"] == []
    world.market.post_ask(world.who(miner), {"key_id": key_id, "ask_microusd": 10_000})
    world.market.revoke_key(world.who(miner), key_id)
    assert world.market.order_book("harvestapi")["asks"] == []
    # Revoked, the key may be listed again.
    world.miner_key("listed-key", 10_000, miner=other)
    assert len(world.market.order_book("harvestapi")["asks"]) == 1


def test_a_key_stops_trading_at_its_daily_cap(world):
    miner, key_id = world.miner_key("capped-key", 10_000, cap=45_000)
    with pytest.raises(ToolMarketError) as dear:
        world.market.post_ask(world.who(miner), {"key_id": key_id, "ask_microusd": 11_250 + 1})
    assert dear.value.code == "ask_too_high_for_cap"  # the cap must hold 4 calls
    _, exa = world.miner_key("exa-cheap", 900_000, tool="exa", miner=miner)
    with pytest.raises(ToolMarketError) as cheap:  # its largest search would owe 0
        world.market.post_ask(world.who(miner), {"key_id": exa, "ask_microusd": 19})
    assert cheap.value.code == "ask_too_low"
    world.market.post_ask(world.who(miner), {"key_id": exa, "ask_microusd": 20})
    _, token = world.buyer()
    for _ in range(4):
        world.market.call(token, PROFILE)
    with pytest.raises(ToolMarketError) as capped:
        world.market.call(token, PROFILE)  # 40,000 earned + 10,000 more would pass 45,000
    assert capped.value.code == "no_supply"
    assert world.market.order_book("harvestapi")["asks"] == []  # 5,000 left fits no call


def test_unknown_usage_counts_toward_a_keys_daily_cap_a_quarter_per_buyer(world):
    # Whatever buyers cause, a miner never bears more than their cap a day,
    # and no one buyer can fill it for free.
    world.miner_key("uncertain-capped", 10_000, cap=40_000)
    world.vendor.script["uncertain-capped"] = [ok(status=500)]
    _, token = world.buyer()
    world.market.call(token, PROFILE)  # 10,000 at risk: a quarter of the cap
    with pytest.raises(ToolMarketError) as share:
        world.market.call(token, PROFILE)
    assert share.value.code == "no_supply" and len(world.vendor.sent) == 1
    for _ in range(3):
        _, token = world.buyer()
        world.market.call(token, PROFILE)
    _, token = world.buyer()
    with pytest.raises(ToolMarketError) as capped:
        world.market.call(token, PROFILE)  # 40,000 at risk: the cap holds for everyone
    assert capped.value.code == "no_supply" and len(world.vendor.sent) == 4


def test_a_buyers_calls_in_flight_count_toward_their_quarter(world, admin):
    world.miner_key("parallel-key", 10_000, cap=40_000)
    world.miner_key("parallel-spare", 12_000)
    _, token = world.buyer()
    tab = market_module.token_hash(token)
    cheap = min(world.store.book(tool="harvestapi", terms_sha256=world.terms), key=lambda r: r["ask_microusd"])

    def reserve():
        return world.store.reserve_call(call_id=str(uuid.uuid4()), token_hash=tab, tool="harvestapi",
                                        operation="linkedin_profile", key_id=cheap["key_id"], ask_microusd=10_000,
                                        reserve_microusd=11_000, owed_cap_microusd=10_000)

    assert [reserve(), reserve()] == ["reserved", "key_cap"]  # the first is still in flight
    # A restart cut the first off: it never settles, and still counts.
    admin.execute("UPDATE public.tool_market_calls SET opened_at=opened_at-interval '11 minutes' WHERE key_id=%s",
                  (cheap["key_id"],))
    assert reserve() == "key_cap"


def test_keys_without_room_for_a_call_do_not_use_up_its_attempts(world):
    for n in range(market_module.MAX_ASKS_PER_CALL + 2):
        # Keys whose cap has less room left than a call (the store, as the
        # market refuses such an ask for a fresh key).
        miner = hotkey()
        world.accept(miner, "miner")
        key_id = world.register(miner, {"tool": "harvestapi", "daily_cap_microusd": 1,
                                        "envelope": world.vault.seal("harvestapi", miner, f"tiny-cap-{n}")})["key_id"]
        world.store.post_ask(key_id=key_id, hotkey=miner, ask_microusd=1_000)
    world.miner_key("roomy-key", 10_000)
    _, token = world.buyer()
    assert world.market.call(token, PROFILE)["charge_microusd"] == 11_000
    assert [s[0] for s in world.vendor.sent] == ["roomy-key"]


def test_an_answer_the_gateway_cannot_send_back_is_free(world):
    world.miner_key("odd-answer", 900_000, tool="exa")
    cost = b'{"costDollars":{"total":0.005},"results":'
    world.vendor.script["odd-answer"] = [
        ProviderResponse(200, {"content-type": "application/json"}, cost + b'[{"score":NaN}]}'),
        ProviderResponse(200, {"content-type": "application/json"}, cost + b"[" * 100 + b"]" * 100 + b"}")]
    _, token = world.buyer()
    for _ in range(2):
        result = world.market.call(token, SEARCH)
        assert result["body"] is None and result["charge_microusd"] == 0  # unknown usage: the miner bears it


def test_a_tab_stops_buying_when_its_buyer_leaves_the_allowlist(world):
    world.miner_key("allow-key", 10_000)
    _, token = world.buyer()
    world.market._config = dataclasses.replace(world.market._config, buyer_hotkeys=frozenset({hotkey()}))
    with pytest.raises(ToolMarketError) as gone:
        world.market.call(token, PROFILE)
    assert (gone.value.status, gone.value.code) == (403, "buyer_not_allowed") and world.vendor.sent == []


def test_a_month_long_tab_stops_buying_when_the_terms_change(world):
    world.miner_key("terms-key", 10_000)
    _, token = world.buyer()
    world.market._config = dataclasses.replace(world.market._config, terms_sha256="ab" * 32)
    with pytest.raises(ToolMarketError) as stale:
        world.market.call(token, PROFILE)
    assert (stale.value.status, stale.value.code) == (403, "terms_required") and world.vendor.sent == []


def test_a_miner_checks_one_key_at_a_time(world):
    miner, key_id = world.miner_key("one-check", 10_000, tool="zerobounce")
    world.market._in_flight["check:" + miner] = 1  # a registration or check still waiting on the vault
    with pytest.raises(ToolMarketError) as busy:
        world.market.check_key(world.who(miner), key_id)
    assert (busy.value.status, busy.value.code) == (429, "check_in_flight")


def test_a_buyer_who_keeps_causing_unknown_usage_is_stopped(world):
    miner, _ = world.miner_key("drained-key", 10_000)
    _, token = world.buyer()
    world.vendor.script["drained-key"] = [VaultTransportError("response_too_large", forwarded=True)]
    for _ in range(20):
        with pytest.raises(ToolMarketError) as unknown:
            world.market.call(token, PROFILE)
        assert unknown.value.code == "vendor_unreachable"
    with pytest.raises(ToolMarketError) as stopped:
        world.market.call(token, PROFILE)
    assert (stopped.value.status, stopped.value.code) == (429, "uncertain_limit") and len(world.vendor.sent) == 20
    world.miner_key("other-tool-key", 900_000, tool="exa")
    assert world.market.call(token, SEARCH)["status"] == 200  # other tools are not stopped
    _, other = world.buyer()
    with pytest.raises(ToolMarketError) as fine:
        world.market.call(other, PROFILE)  # nor are other buyers
    assert fine.value.code == "vendor_unreachable"


def test_calls_in_flight_count_toward_the_unknown_usage_limit(world):
    world.miner_key("busy-key", 10)
    _, token = world.buyer()
    tab = market_module.token_hash(token)
    key_id = world.store.book(tool="harvestapi", terms_sha256=world.terms)[0]["key_id"]

    def reserve():
        return world.store.reserve_call(call_id=str(uuid.uuid4()), token_hash=tab, tool="harvestapi",
                                        operation="linkedin_profile", key_id=key_id, ask_microusd=10,
                                        reserve_microusd=11, owed_cap_microusd=10)

    assert [reserve() for _ in range(20)] == ["reserved"] * 20  # none has settled yet
    assert reserve() == "uncertain_limit"


def test_a_month_long_api_key_still_counts_toward_the_unknown_usage_limit(world, database):
    world.miner_key("long-key", 10)
    month = 30 * 24 * 3600
    _, token = world.buyer(seconds=month)
    with pytest.raises(ToolMarketError) as too_long:
        world.buyer(seconds=month + 1)
    assert too_long.value.code == "seconds_invalid"
    tab = market_module.token_hash(token)
    psycopg2, dsn = database
    connection = psycopg2.connect(**dsn)
    connection.autocommit = True
    with connection.cursor() as cursor:  # the key was created two days ago, past the append-only trigger
        cursor.execute("SET session_replication_role=replica")
        cursor.execute("UPDATE public.tool_market_tabs SET opened_at=opened_at-interval '2 days' "
                       "WHERE token_hash=%s", (tab,))
    connection.close()
    key_id = world.store.book(tool="harvestapi", terms_sha256=world.terms)[0]["key_id"]

    def reserve():
        return world.store.reserve_call(call_id=str(uuid.uuid4()), token_hash=tab, tool="harvestapi",
                                        operation="linkedin_profile", key_id=key_id, ask_microusd=10,
                                        reserve_microusd=11, owed_cap_microusd=10)

    assert [reserve() for _ in range(20)] == ["reserved"] * 20
    assert reserve() == "uncertain_limit"


def test_a_settlement_cannot_charge_more_than_its_reserve_or_settle_twice(world):
    world.miner_key("rule-key", 10_000)
    _, token = world.buyer()
    call_id = world.market.call(token, PROFILE)["call_id"]
    assert world.store.settle_call(call_id=call_id, status="settled", http_status=200, units=1, owed_microusd=10_000,
                                   charge_microusd=11_000, epoch=7) == "exists"
    reserve = world.store.book(tool="harvestapi", terms_sha256=world.terms)[0]
    other = str(uuid.uuid4())
    assert world.store.reserve_call(call_id=other, token_hash=market_module.token_hash(token), tool="harvestapi",
                                    operation="linkedin_profile", key_id=reserve["key_id"], ask_microusd=10_000,
                                    reserve_microusd=11_000, owed_cap_microusd=10_000) == "reserved"
    with pytest.raises(ArenaStoreError):
        world.store.settle_call(call_id=other, status="settled", http_status=200, units=1, owed_microusd=10_000,
                                charge_microusd=11_001, epoch=7)
    assert world.store.reserve_call(call_id=str(uuid.uuid4()), token_hash=market_module.token_hash(token),
                                    tool="harvestapi", operation="linkedin_profile", key_id=reserve["key_id"],
                                    ask_microusd=9_999, reserve_microusd=11_000, owed_cap_microusd=10_000) \
        == "ask_changed"


def test_a_key_is_listed_only_with_its_miners_signed_attestation(world, admin):
    miner = hotkey()
    world.accept(miner, "miner")
    body = {"tool": "harvestapi", "envelope": world.vault.seal("harvestapi", miner, "attested-key"),
            "daily_cap_microusd": USD}
    for attestation in (None, "yes", ATTESTATION + " "):
        with pytest.raises(ToolMarketError) as refused:
            world.register(miner, dict(body, attestation=attestation))
        assert refused.value.code in ("attestation_required", "request_invalid")
    assert world.vault.checks == 0  # refused before any vendor check
    key_id = world.register(miner, body)["key_id"]
    admin.execute("SELECT signed_body,signature,host,signed_at FROM public.tool_market_keys WHERE key_id=%s", (key_id,))
    signed_body, signature, host, signed_at = admin.fetchone()
    assert json.loads(signed_body)["attestation"] == ATTESTATION
    assert (signature, host, signed_at) == ("ab" * 64, "gw.example", 1)


def test_zerobounce_charges_a_check_but_not_an_unknown_and_a_dead_key_fails_over(world):
    miner, _ = world.miner_key("zb-dead", 4_000, tool="zerobounce")
    world.miner_key("zb-live", 5_000, tool="zerobounce")
    _, token = world.buyer()
    # A dead key and an empty account both answer 200 with only an error.
    world.vendor.script["zb-dead"] = [ok({"error": "Invalid API Key or your account ran out of credits"})]
    # The answer alone fails no key: the vault's free check confirms it.
    world.vault.answers["zb-dead"] = {"status": "invalid", "reason": "no_credit"}
    world.vendor.script["zb-live"] = [ok({"address": "ada@acme.com", "status": "valid", "sub_status": ""}),
                                      ok({"address": "ada@acme.com", "status": "unknown", "sub_status": "timeout_exceeded"})]
    assert world.market.call(token, VALIDATE)["charge_microusd"] == 5_500
    assert [s[0] for s in world.vendor.sent] == ["zb-dead", "zb-live"]
    assert world.market.statement(miner)["keys"][0]["health"]["reason"] == "no_credit"
    assert world.market.call(token, VALIDATE)["charge_microusd"] == 0  # ZeroBounce does not charge unknowns
    assert world.vendor.sent[-1][0] == "zb-live"  # the dead key is out of the book
    _, method, url, _ = world.vendor.sent[1]
    # The vault, not the market, puts the key in the query.
    assert method == "GET" and url == "https://api.zerobounce.net/v2/validate?email=ada%40acme.com"


def test_hunter_rate_limits_skip_a_key_and_an_empty_account_fails_it(world):
    slow, _ = world.miner_key("hunter-slow", 8_000, tool="hunter")
    empty, _ = world.miner_key("hunter-empty", 9_000, tool="hunter")
    world.miner_key("hunter-ok", 10_000, tool="hunter")
    _, token = world.buyer()
    world.vendor.script["hunter-slow"] = [ok({"errors": [{"id": "too_many_requests"}]}, status=403)]
    world.vendor.script["hunter-empty"] = [ok({"errors": [{"id": "too_many_requests"}]}, status=429)]
    # Hunter's check proves the key but reads no balance, so its 402 fails it.
    world.vault.answers["hunter-empty"] = {"balance": None}
    world.vendor.script["hunter-ok"] = [ok({"data": {"email": "ada@acme.com", "score": 97}}),
                                        ok({"data": {"email": None, "score": None}})]
    assert world.market.call(token, FIND)["charge_microusd"] == 11_000
    assert [s[0] for s in world.vendor.sent] == ["hunter-slow", "hunter-empty", "hunter-ok"]
    # Hunter's 403 is a rate limit: skipped for this call only.
    assert world.market.statement(slow)["keys"][0]["health"]["status"] != "invalid"
    assert world.market.statement(empty)["keys"][0]["health"]["status"] == "invalid"
    world.vendor.script["hunter-slow"] = [ok({"data": {"email": None}})]
    assert world.market.call(token, FIND)["charge_microusd"] == 0  # no email found: free


def test_a_buyer_cannot_take_working_keys_out_of_the_book(world):
    world.miner_key("zb-a", 4_000, tool="zerobounce")
    world.miner_key("exa-a", 900_000, tool="exa")
    _, token = world.buyer()
    checks = world.vault.checks
    # A 403 only skips the key, with no check.
    world.vendor.script["zb-a"] = [ok({}, status=403)]
    with pytest.raises(ToolMarketError):
        world.market.call(token, VALIDATE)
    assert world.vault.checks == checks and len(world.market.order_book("zerobounce")["asks"]) == 1
    # Out of credit, the answer says; the check still sees a balance. Within
    # the minute after it, no check runs again and the key stays.
    world.vendor.script["zb-a"] = [ok({"error": "your account ran out of credits"})]
    for _ in range(3):
        with pytest.raises(ToolMarketError):
            world.market.call(token, VALIDATE)
    assert world.vault.checks == checks + 1 and len(world.market.order_book("zerobounce")["asks"]) == 1
    # Exa has no free check: a 402 only skips it; a 401 fails it.
    world.vendor.script["exa-a"] = [ok({}, status=402), ok({}, status=401)]
    with pytest.raises(ToolMarketError):
        world.market.call(token, SEARCH)
    assert len(world.market.order_book("exa")["asks"]) == 1
    with pytest.raises(ToolMarketError):
        world.market.call(token, SEARCH)
    assert world.market.order_book("exa")["asks"] == []


def test_a_miner_rechecks_a_key_only_where_a_free_check_proves_it(world):
    miner, exa = world.miner_key("exa-dead", 900_000, tool="exa")
    _, zb = world.miner_key("zb-topped-up", 4_000, tool="zerobounce", miner=miner)
    who = world.who(miner)
    with pytest.raises(ToolMarketError) as refused:
        world.market.check_key(who, exa)  # "unchecked" would put a dead key back
    assert (refused.value.status, refused.value.code) == (409, "key_cannot_be_checked")
    assert world.market.check_key(who, zb)["tradeable"] is True
    with pytest.raises(ToolMarketError) as soon:
        world.market.check_key(who, zb)
    assert (soon.value.status, soon.value.code) == (429, "check_too_soon")


def test_a_buyer_has_a_few_calls_in_flight_at_most(world):
    world.miner_key("slow-key", 10_000)
    buyer, token = world.buyer()
    world.market._in_flight[buyer] = market_module.MAX_CALLS_IN_FLIGHT_PER_BUYER
    with pytest.raises(ToolMarketError) as busy:
        world.market.call(token, PROFILE)
    assert (busy.value.status, busy.value.code) == (429, "too_many_calls_in_flight") and world.vendor.sent == []
    world.market._in_flight[buyer] -= 1
    world.market.call(token, PROFILE)
    assert world.market._in_flight[buyer] == market_module.MAX_CALLS_IN_FLIGHT_PER_BUYER - 1


def test_a_miners_tab_never_buys_from_their_own_key(world):
    miner, _ = world.miner_key("own-key", 10_000)
    world.gamma.units[miner] = 10**9
    world.accept(miner, "buyer")
    # A budget below the reserve: the own-key rule still decides.
    token = world.market.open_tab(world.who(miner), {"budget_microusd": 5_000, "seconds": 600})["token"]
    with pytest.raises(ToolMarketError) as own:
        world.market.call(token, PROFILE)
    assert own.value.code == "no_supply" and world.vendor.sent == []


def test_only_a_registered_hotkey_lists_keys(world):
    miner = hotkey()
    world.accept(miner, "miner")
    world.unregistered.add(miner)
    body = {"tool": "exa", "daily_cap_microusd": USD, "envelope": world.vault.seal("exa", miner, "unregistered")}
    with pytest.raises(ToolMarketError) as refused:
        world.register(miner, body)
    assert (refused.value.status, refused.value.code) == (403, "miner_not_registered")
    world.unregistered.clear()
    world.register(miner, body)


def test_a_hotkey_registers_one_key_per_tool_every_30_days(world):
    miner, typo_id = world.miner_key("typo-key", 10_000)
    world.market.revoke_key(world.who(miner), typo_id)
    # A key that never served a call does not start the 30 days.
    key_id = world.register(miner, {"tool": "harvestapi", "daily_cap_microusd": USD,
                                    "envelope": world.vault.seal("harvestapi", miner, "first-key")})["key_id"]
    world.market.post_ask(world.who(miner), {"key_id": key_id, "ask_microusd": 10_000})
    world.vendor.script["first-key"] = [ok(status=500)]  # even a call of unknown usage
    _, token = world.buyer()
    world.market.call(token, PROFILE)
    world.market.revoke_key(world.who(miner), key_id)
    with pytest.raises(ToolMarketError) as soon:
        world.register(miner, {"tool": "harvestapi", "daily_cap_microusd": USD,
                               "envelope": world.vault.seal("harvestapi", miner, "second-key")})
    assert soon.value.status == 429 and soon.value.code == "key_registration_too_soon"
    # Listed keys count even before they serve, so keys cannot be stacked.
    world.register(miner, {"tool": "exa", "daily_cap_microusd": USD,
                           "envelope": world.vault.seal("exa", miner, "exa-stack-1")})
    with pytest.raises(ToolMarketError) as stacked:
        world.register(miner, {"tool": "exa", "daily_cap_microusd": USD,
                               "envelope": world.vault.seal("exa", miner, "exa-stack-2")})
    assert stacked.value.code == "key_registration_too_soon"
    checks = world.vault.checks
    with pytest.raises(ToolMarketError) as again:  # a refused listing re-posted is not checked again
        world.register(miner, {"tool": "exa", "daily_cap_microusd": USD,
                               "envelope": world.vault.seal("exa", miner, "exa-stack-2")})
    assert again.value.code == "check_too_soon" and world.vault.checks == checks
    # A key revoked before any call is a one-off retry, not a loop.
    for tag in ("zb-loop-1", "zb-loop-2"):
        looped = world.register(miner, {"tool": "zerobounce", "daily_cap_microusd": USD,
                                        "envelope": world.vault.seal("zerobounce", miner, tag)})
        world.market.revoke_key(world.who(miner), looped["key_id"])
    world.market._rechecked.clear()  # past the minute between refused probes
    with pytest.raises(ToolMarketError) as loop:
        world.register(miner, {"tool": "zerobounce", "daily_cap_microusd": USD,
                               "envelope": world.vault.seal("zerobounce", miner, "zb-loop-3")})
    assert loop.value.code == "key_registration_too_soon"
    # Another tool is its own 30 days.
    world.register(miner, {"tool": "hunter", "daily_cap_microusd": USD,
                           "envelope": world.vault.seal("hunter", miner, "hu-thirty-day-key")})


def test_asks_have_a_ceiling_and_a_buyer_can_cap_what_they_pay(world):
    miner, key_id = world.miner_key("pricey-key", 200_000)
    for ask in (250_001, 10_000_000):
        with pytest.raises(ToolMarketError) as high:
            world.market.post_ask(world.who(miner), {"key_id": key_id, "ask_microusd": ask})
        assert high.value.code == "ask_invalid"
    _, token = world.buyer(budget=500_000)
    with pytest.raises(ToolMarketError) as capped:
        world.market.call(token, dict(PROFILE, max_ask_microusd=199_999))
    assert capped.value.code == "no_supply" and world.vendor.sent == []
    assert world.market.call(token, dict(PROFILE, max_ask_microusd=200_000))["charge_microusd"] == 220_000
    with pytest.raises(ToolMarketError) as bad:
        world.market.call(token, dict(PROFILE, max_ask_microusd=0))
    assert bad.value.code == "max_ask_invalid"


def test_the_epoch_is_read_before_any_vendor_request(world):
    world.miner_key("epoch-key", 10_000)
    _, token = world.buyer()

    def chain_down():
        raise ArenaChainError("chain unavailable")

    world.market._epoch = chain_down
    with pytest.raises(ToolMarketError) as down:
        world.market.call(token, PROFILE)
    assert (down.value.status, down.value.code) == (503, "chain_unavailable")
    assert world.vendor.sent == [] and world.market.tab(token)["calls"] == 0


def test_tables_are_append_only_and_closed_to_the_service_role(database):
    psycopg2, dsn = database
    connection = psycopg2.connect(**dsn)
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            cursor.execute("SET ROLE lab_arena_service")
            with pytest.raises(psycopg2.Error):
                cursor.execute("SELECT * FROM public.tool_market_keys")
            cursor.execute("RESET ROLE")
            with pytest.raises(psycopg2.Error):
                cursor.execute("DELETE FROM public.tool_market_asks")
            with pytest.raises(psycopg2.Error):
                cursor.execute("UPDATE public.tool_market_call_settlements SET charge_microusd=0")
            cursor.execute("SET ROLE lab_arena_service")
            with pytest.raises(psycopg2.Error):
                cursor.execute("SELECT public.tool_market__committed_v1(gen_random_uuid())")
    finally:
        connection.close()
