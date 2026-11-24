"""Lead orders on a disposable PostgreSQL: SQL rules, service, broker, receipts.

Runs scripts/707 and 730 on top of the Lab Arena default chain through the same harness
the Arena tests use (Docker in CI, ``LAB_ARENA_PG_LOCAL=1`` locally). Every
order a test places is claimed in the same test, because the claim queue is
shared by the module's database.
"""

from __future__ import annotations

import hashlib
import json
import random
import uuid
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from time import monotonic, sleep

import pytest

from lab_arena import signing
from lab_arena.broker import ProviderResponse, ProviderTransportError, price_table_from_models_response
from lab_arena.credentials import CredentialError
from lab_arena.lead_orders import money, receipts, rewards, suppressions
from lab_arena.lead_orders.auth import BuyerIdentity, order_token_hash
from lab_arena.lead_orders.broker import BrokerRequest, LeadOrderBroker, LeadOrderBrokerError
from lab_arena.lead_orders.config import LeadOrdersConfig
from lab_arena.lead_orders.gamma import GammaReading
from lab_arena.lead_orders.keys import MinerKeyRegistrar, ProviderKey
from lab_arena.lead_orders.policy import DEFAULT_POLICY_BODY, sign_policy, load_policy, unsigned_policy
from lab_arena.lead_orders.service import LeadOrderService, LeadOrderServiceError
from lab_arena.lead_orders.store import LeadOrderStore
from lab_arena.store import ArenaStoreError, PsycopgTransport
from tests.lab_arena.lab_arena_pg_harness import DEFAULT_MIGRATIONS, database_with_lab_arena_migration

# 730 guards the accepted weight states (202, in the default chain) and collection jobs (720, 721).
MIGRATIONS = DEFAULT_MIGRATIONS + (
    "720-locked-collection-intake.sql", "721-locked-collection-reward-epoch.sql",
    "707-lead-orders.sql", "730-lead-order-weight-state.sql",
)
USD = 1_000_000
B58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
MODEL = "openai/gpt-test"
DEAR = "openai/gpt-dear"  # 10,000 output tokens reserve $1.00: room for the costs the reward tests report
SECRET = "x" * 40
SIGNER = signing.LocalSigner.generate()
# A 30% lane cap, so the payout tests show a capped epoch and its carry-forward.
POLICY_DOC = sign_policy(SIGNER, dict(DEFAULT_POLICY_BODY, lane_emission_cap="0.30"))
POLICY = load_policy(POLICY_DOC, public_key_der=SIGNER.public_key_der, expected_public_key_hash=SIGNER.public_key_hash)
PRICE_TABLE = price_table_from_models_response(
    {"data": [{"id": MODEL, "pricing": {"prompt": "0.000001", "completion": "0.000002"}},
              {"id": DEAR, "pricing": {"prompt": "0.000001", "completion": "0.000100"}}]},
    fetched_at="2026-11-19T00:00:00Z",
)


ORDER = {
    "seller": {"product": "Payroll software", "summary": "Payroll for small US companies"},
    "search": {"signal": "hiring", "terms": ["payroll manager"], "buyer_titles": ["CFO"],
               "market": {"countries": ["US"], "employee_ranges": ["11-50"], "industries": ["software"]},
               "excluded_companies": ["example.com"]},
}


def claim(service):
    claimed = service.claim()
    return claimed["broker"]["token"], claimed["order_id"]


def hotkey() -> str:
    return "5" + "".join(random.choice(B58) for _ in range(47))


@pytest.fixture(scope="module")
def database():
    yield from database_with_lab_arena_migration(MIGRATIONS)


@pytest.fixture(scope="module")
def transport(database):
    psycopg2, dsn = database

    def connect():
        connection = psycopg2.connect(**dsn)
        connection.autocommit = True
        return connection

    t = PsycopgTransport(connect, role="lab_arena_service")
    yield t
    t.close()


@pytest.fixture(scope="module")
def store(transport):
    return LeadOrderStore(transport)


@pytest.fixture
def owner(database):
    psycopg2, dsn = database
    connection = psycopg2.connect(**dsn)
    connection.autocommit = True
    yield connection
    connection.close()


class Epoch:
    def __init__(self, value=1000):
        self.value = value

    def __call__(self):
        return self.value


class FakeGamma:
    def __init__(self):
        self.units = {}
        self.available = True
        self.block = 100

    def read(self, hk, head=None):
        self.block += 1
        return GammaReading(hk, self.block, "0x" + "%064x" % self.block, self.units.get(hk, 0), self.available)


class FakeTransport:
    def __init__(self, responses):
        self.responses = list(responses)
        self.sent = []

    def send(self, *, method, url, headers, body, timeout_seconds, max_response_bytes=None):
        self.sent.append({"method": method, "url": url, "headers": dict(headers), "body": body})
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class FakeRegistrar:
    def __init__(self, secrets):
        self.secrets = secrets

    def decrypt(self, key):
        return self.secrets[key.key_id]

    def register(self, *, miner_hotkey, provider, credentials):
        key_id = "k" + uuid.uuid4().hex[:20]
        return {
            "key_id": key_id, "provider": provider, "miner_hotkey": miner_hotkey,
            "submission_id": "lead-order:" + key_id, "ciphertext_b64": "Y2lwaGVy",
            "internal_fingerprint": "sha256:" + uuid.uuid4().hex + uuid.uuid4().hex,
        }


def openrouter_ok(cost_usd="0.010000"):
    return ProviderResponse(200, {"content-type": "application/json", "x-secret": "no"},
                            json.dumps({"choices": [], "usage": {"cost": cost_usd}}).encode())


def deepline_ok(credits):
    return ProviderResponse(200, {"content-type": "application/json"},
                            json.dumps({"billing": {"credits_charged": credits}}).encode())


def deepline(operation="harvestapi_get_job", body=b"{}"):
    return BrokerRequest("deepline", "POST", "api/v2/integrations/%s/execute" % operation, {}, body, "application/json")


def chat(max_tokens=100, model=MODEL):
    return BrokerRequest("openrouter", "POST", "api/v1/chat/completions", {},
                         json.dumps({"model": model, "messages": [{"role": "user", "content": "hi"}],
                                     "max_tokens": max_tokens}).encode(), "application/json")


def dear_chat():
    """A call whose reservation ($1.00 and a little prompt) covers the costs
    the reward tests report; rewards stop at the reservation."""
    return chat(max_tokens=10_000, model=DEAR)


def make(store, mode, *, staff=(), gamma=None, registrar=None, epoch=None):
    config = LeadOrdersConfig(
        mode=mode, staff_hotkeys=frozenset(staff), runner_secret="r" * 40, policy_path="/x",
        gamma_service="7", terms_version="t1", public_base_url="https://gw.example",
    )
    epoch = epoch or Epoch()
    service = LeadOrderService(config=config, policy=POLICY, policy_document=POLICY_DOC, store=store,
                               signer=SIGNER, current_epoch=epoch, gamma_reader=gamma, registrar=registrar)
    return config, service, epoch


def broker_for(store, transport, *, registrar=None, epoch=None):
    return LeadOrderBroker(
        store=store, policy=POLICY, transport=transport, registrar=registrar,
        leadpoet_secrets={"openrouter": SECRET, "deepline": SECRET},
        price_table=lambda: PRICE_TABLE, current_epoch=epoch or Epoch(),
    )


def confirm(owner, hk):
    """Mark a buyer confirmed the way the confirmation link would."""
    with owner.cursor() as cur:
        cur.execute("SELECT token_hash FROM lead_order_buyers WHERE hotkey=%s", (hk,))
        token_hash = cur.fetchone()[0]
        cur.execute("INSERT INTO lead_order_buyer_confirmations(token_hash) VALUES(%s)", (token_hash,))


# -- schema ----------------------------------------------------------------------


def test_tables_are_append_only_rls_on_and_not_granted(owner, store):
    store.record_policy(POLICY_DOC)  # row triggers need a row to fire on
    with owner.cursor() as cur:
        cur.execute("""SELECT c.relname, c.relrowsecurity FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace
                       WHERE n.nspname='public' AND c.relkind='r' AND c.relname LIKE 'lead_order%'""")
        tables = dict(cur.fetchall())
        assert len(tables) == 24 and all(tables.values())
        for role in ("anon", "authenticated", "service_role", "lab_arena_service"):
            cur.execute("""SELECT count(*) FROM information_schema.role_table_grants
                           WHERE grantee=%s AND table_name LIKE 'lead_order%%'""", (role,))
            assert cur.fetchone()[0] == 0
        cur.execute("SELECT has_function_privilege('anon','public.lead_order_place_v1(uuid,text,text,integer,integer,jsonb)','EXECUTE')")
        assert cur.fetchone()[0] is False
        cur.execute("SELECT has_function_privilege('lab_arena_service','public.lead_order__settled_microusd_v1(uuid)','EXECUTE')")
        assert cur.fetchone()[0] is False
        for statement in ("UPDATE lead_order_policies SET created_at=now()", "DELETE FROM lead_order_policies",
                          "TRUNCATE lead_order_policies CASCADE"):
            with pytest.raises(Exception, match="append-only"):
                cur.execute(statement)


def test_policy_is_versioned_and_a_changed_document_is_refused(store):
    assert store.record_policy(POLICY_DOC)["status"] in ("recorded", "exists")
    assert store.record_policy(POLICY_DOC)["status"] == "exists"
    other = sign_policy(SIGNER, dict(DEFAULT_POLICY_BODY, price_per_lead_usd="0.30"))
    with pytest.raises(ArenaStoreError):
        store.record_policy(other)


# -- staff mode, worked example -------------------------------------------------


def test_staff_order_worked_example_charges_ten_leads_and_signs_receipt(store):
    staff = hotkey()
    config, service, epoch = make(store, "staff", staff=[staff])
    with pytest.raises(LeadOrderServiceError) as refused:
        service.place_order(hotkey(), {"leads": 10, **ORDER})
    assert refused.value.code == "hotkey_not_allowed"
    with pytest.raises(LeadOrderServiceError) as small:
        service.place_order(staff, {"leads": 9, **ORDER})
    assert small.value.code == "below_minimum"
    with pytest.raises(LeadOrderServiceError) as old_shape:
        service.place_order(staff, {"leads": 10, "icp": {}})
    assert old_shape.value.code == "order_invalid"
    with pytest.raises(LeadOrderServiceError) as unclaimed:
        service.heartbeat(str(uuid.uuid4()))
    assert unclaimed.value.code == "order_unknown"

    placed = service.place_order(staff, {"leads": 10, **ORDER})
    assert placed["status"] == "queued" and placed["hold_microusd"] == 2 * USD
    assert placed["spend_ceiling_microusd"] == 5 * USD

    assert placed["hold_usd"] == "2.00"

    claimed = service.claim()
    order_id, token = claimed["order_id"], claimed["broker"]["token"]
    assert order_id == placed["order_id"] and claimed["deadline"].endswith("Z")
    assert claimed["spend_ceiling_usd"] == "5.00" and claimed["leads_requested"] == 10
    assert claimed["seller"] == ORDER["seller"] and claimed["search"] == ORDER["search"]
    assert claimed["broker"]["openrouter_base_url"] == "https://gw.example/lead-orders/broker/openrouter/api/v1"
    assert claimed["broker"]["deepline_base_url"] == "https://gw.example/lead-orders/broker/deepline"
    assert service.heartbeat(order_id) == {"continue": True, "spent_usd": "0.00", "deadline": claimed["deadline"]}
    transport = FakeTransport([openrouter_ok("0.400000"), deepline_ok(6)])
    broker = broker_for(store, transport, epoch=epoch)
    response = broker.call(token, chat())
    assert response.status == 200 and set(response.headers) == {"content-type"}
    sent = transport.sent[0]
    assert sent["url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert sent["headers"]["authorization"] == "Bearer " + SECRET
    assert json.loads(sent["body"])["usage"] == {"include": True}
    assert broker.call(token, deepline()).status == 200

    assert service.heartbeat(order_id)["spent_usd"] == "1.00"
    done = service.finish(order_id, {"status": "delivered", "leads_delivered": 10, "delivery_url": "https://files.example/o.csv"})
    assert done["charged_usd"] == "2.00"
    assert service.heartbeat(order_id)["continue"] is False
    # spend $1.00; charge = min($2 hold, max(10 x $0.20, ceil(0.4 x $1))) = $2.00
    assert done["charge_microusd"] == 2 * USD
    assert done["charge_microusd"] == money.charge_microusd(POLICY, hold_microusd=2 * USD, delivered=10, spend_microusd=USD)
    public = service.public_receipt(placed["order_id"])
    receipt = public["receipt"]
    assert receipt["spend_microusd"] == USD and receipt["rewards_microusd"] == {}  # staff: Leadpoet keys, no rewards
    receipts.verify_receipt(receipt, calls=public["calls"], public_key_der=SIGNER.public_key_der,
                            expected_public_key_hash=SIGNER.public_key_hash)
    assert all("authorization" not in json.dumps(row) for row in public["calls"])
    # Finishing again changes nothing, and the token no longer brokers calls.
    assert service.finish(order_id, {"status": "failed"})["receipt_hash"] == receipt["receipt_hash"]
    with pytest.raises(LeadOrderBrokerError) as closed:
        broker.call(token, chat())
    assert closed.value.code == "order_finished"


def test_partial_delivery_charges_per_lead_and_failure_charges_minimum(store):
    staff = hotkey()
    _, service, epoch = make(store, "staff", staff=[staff])
    for delivered, status, cost, expected in ((6, "delivered", "1.000000", 1_200_000), (0, "failed", "2.000001", 800_001)):
        order = service.place_order(staff, {"leads": 10, **ORDER})
        token, order_id = claim(service)
        broker_for(store, FakeTransport([openrouter_ok(cost)]), epoch=epoch).call(token, chat())
        payload = {"status": status, "leads_delivered": delivered}
        if status == "delivered":
            payload["delivery_url"] = "https://files.example/x"
        assert service.finish(order_id, payload)["charge_microusd"] == expected
        assert expected == money.charge_microusd(
            POLICY, hold_microusd=2 * USD, delivered=delivered if status == "delivered" else 0,
            spend_microusd=int(float(cost) * USD + 0.5) if cost != "2.000001" else 2_000_001,
        )
        assert service.order_view(staff, order["order_id"])["status"] == status


def test_ceiling_refuses_the_next_call_with_402_and_uncertain_cost_counts(store):
    staff = hotkey()
    _, service, epoch = make(store, "staff", staff=[staff])
    order = service.place_order(staff, {"leads": 10, **ORDER})
    token, order_id = claim(service)
    # Deepline reserves $0.50 a call. Ten calls with no billing object are
    # uncertain: they keep their $0.50 against the $5 ceiling, and charge 0.
    no_billing = ProviderResponse(200, {"content-type": "application/json"}, b"{}")
    transport = FakeTransport([no_billing] * 10)
    broker = broker_for(store, transport, epoch=epoch)
    request = deepline("harvestapi_search_jobs")
    for _ in range(10):
        assert broker.call(token, request).status == 200
    refused = broker.call(token, request)
    assert refused.status == 402 and json.loads(refused.body) == {"error": "spend_ceiling"}
    assert len(transport.sent) == 10  # the refused call was never sent
    assert transport.sent[0]["url"] == "https://code.deepline.com/api/v2/integrations/harvestapi_search_jobs/execute"
    calls = store.calls(order["order_id"])
    assert len(calls) == 10 and {c["status"] for c in calls} == {"uncertain"}
    assert store.by_token(order_token_hash(token))["committed_microusd"] == 5 * USD
    assert service.heartbeat(order_id) == {"continue": False, "spent_usd": "5.00", "deadline": store.view(order_id)["deadline"]}
    finished = service.finish(order_id, {"status": "failed", "failure_reason": "no leads"})
    assert finished["charge_microusd"] == 0


def test_transport_failure_settles_uncertain_and_routes_are_closed(store):
    staff = hotkey()
    _, service, epoch = make(store, "staff", staff=[staff])
    order = service.place_order(staff, {"leads": 10, **ORDER})
    token, order_id = claim(service)
    broker = broker_for(store, FakeTransport([ProviderTransportError("down")]), epoch=epoch)
    assert broker.call(token, chat()).status == 502
    assert [c["status"] for c in store.calls(order["order_id"])] == ["uncertain"]
    def body(**extra):
        return json.dumps(dict({"model": MODEL, "messages": [{"role": "user", "content": "hi"}]}, **extra)).encode()

    def chat_with(**extra):
        return BrokerRequest("openrouter", "POST", "api/v1/chat/completions", {}, body(**extra), None)

    for request, code in (
        (BrokerRequest("openrouter", "POST", "api/v1/models", {}, b"{}", None), "route_not_allowed"),
        (BrokerRequest("deepline", "GET", "api/v2/billing/balance", {}, b"", None), "route_not_allowed"),
        (BrokerRequest("deepline", "GET", "api/v2/jobs/abc", {}, b"", None), "route_not_allowed"),
        (deepline("monitors_create"), "operation_not_allowed"),
        (BrokerRequest("deepline", "POST", "api/v2/integrations/harvestapi_get_job/execute", {"api_key": "s"}, b"{}", None),
         "query_not_allowed"),
        (BrokerRequest("scrapingdog", "GET", "x", {}, b"", None), "provider_not_allowed"),
        (BrokerRequest("openrouter", "POST", "api/v1/chat/completions", {},
                       json.dumps({"model": "other/model", "messages": []}).encode(), None), "model_not_allowed"),
        # Cost the reservation does not price is refused before anything is sent.
        (chat_with(models=[MODEL, "other/model"]), "model_not_allowed"),
        (chat_with(plugins=[{"id": "web"}]), "field_not_allowed:plugins"),
        (chat_with(provider={"order": ["x"]}), "provider_field_not_allowed:order"),
        (chat_with(reasoning={"budget": 9999}), "reasoning_invalid"),
        (chat_with(tools=[{"type": "openrouter:web_search"}]), "tool_not_allowed"),
        (BrokerRequest("openrouter", "POST", "api/v1/chat/completions", {},
                       json.dumps({"model": MODEL + ":online", "messages": []}).encode(), None), "model_variant_not_allowed"),
        (BrokerRequest("openrouter", "POST", "api/v1/chat/completions", {},
                       json.dumps({"model": MODEL, "messages": [{"role": "user", "content": [
                           {"type": "file", "file": {"filename": "a.pdf"}}]}]}).encode(), None), "content_not_allowed"),
    ):
        with pytest.raises(LeadOrderBrokerError) as refused:
            broker.call(token, request)
        assert refused.value.code == code and refused.value.status in (400, 404)
    assert len(store.calls(order["order_id"])) == 1  # no refused request reserved or sent anything
    with pytest.raises(LeadOrderBrokerError) as bad:
        broker.call("not-a-token", chat())
    assert bad.value.code == "order_token_invalid"
    service.finish(order_id, {"status": "failed"})


def test_overdue_orders_expire_as_failed_at_the_next_claim(store, owner):
    staff = hotkey()
    _, service, epoch = make(store, "staff", staff=[staff])
    order = service.place_order(staff, {"leads": 10, **ORDER})
    token, order_id = claim(service)
    with owner.cursor() as cur:  # simulate time passing past deadline and grace
        cur.execute("ALTER TABLE lead_order_claims DISABLE TRIGGER lead_order_claims_append_only")
        cur.execute("UPDATE lead_order_claims SET deadline=now()-interval '6 minutes' WHERE order_id=%s", (order["order_id"],))
        cur.execute("ALTER TABLE lead_order_claims ENABLE TRIGGER lead_order_claims_append_only")
    with pytest.raises(LeadOrderBrokerError) as late:
        broker_for(store, FakeTransport([]), epoch=epoch).call(token, chat())
    assert late.value.code == "order_expired"
    assert service.claim() is None
    view = service.order_view(staff, order["order_id"])
    assert view["status"] == "failed" and view["failure_reason"] == "deadline_passed" and view["receipt"]


def test_daily_cap_is_a_rolling_sum_per_hotkey(store):
    staff = hotkey()
    _, service, _ = make(store, "staff", staff=[staff])
    service.place_order(staff, {"leads": 400, **ORDER})
    service.claim()
    with pytest.raises(LeadOrderServiceError) as capped:
        service.place_order(staff, {"leads": 101, **ORDER})
    assert capped.value.code == "daily_cap"
    service.place_order(staff, {"leads": 100, **ORDER})
    service.claim()


# -- live mode: Gamma balance ----------------------------------------------------


def test_live_orders_need_confirmed_buyer_gamma_and_balance(store, owner):
    gamma = FakeGamma()
    buyer = hotkey()
    config, service, epoch = make(store, "live", gamma=gamma)
    with pytest.raises(LeadOrderServiceError) as freemail:
        service.register_buyer(buyer, {"email": "a@gmail.com", "company": "Acme", "country": "US", "terms_version": "t1"})
    assert freemail.value.code == "company_email_required"
    with pytest.raises(LeadOrderServiceError) as nomail:
        service.register_buyer(buyer, {"email": "a@acme.io", "company": "Acme", "country": "us", "terms_version": "t1"})
    assert nomail.value.code == "confirmation_email_unavailable"  # stored, but unconfirmed
    assert store.buyer(buyer) == {"registered": True, "confirmed": False, "terms_version": "t1", "country": "US"}
    with pytest.raises(LeadOrderServiceError) as unconfirmed:
        service.place_order(buyer, {"leads": 10, **ORDER})
    assert unconfirmed.value.code == "buyer_not_confirmed"
    confirm(owner, buyer)

    gamma.available = False  # storage item missing: balance 0, refused
    with pytest.raises(LeadOrderServiceError) as missing:
        service.place_order(buyer, {"leads": 10, **ORDER})
    assert missing.value.code == "gamma_unavailable"
    gamma.available = True
    gamma.units[buyer] = 3 * 10**9  # $3 of Gamma spent to Leadpoet
    with pytest.raises(LeadOrderServiceError) as lowbal:
        service.place_order(buyer, {"leads": 20, **ORDER})
    assert lowbal.value.code == "insufficient_balance"
    first = service.place_order(buyer, {"leads": 10, **ORDER})
    assert store.balance(buyer)["balance_microusd"] == USD  # $3 - $2 hold
    with pytest.raises(LeadOrderServiceError):
        service.place_order(buyer, {"leads": 10, **ORDER})  # the hold blocks a second $2 order
    token, order_id = claim(service)
    broker_for(store, FakeTransport([openrouter_ok("0.100000")]), epoch=epoch).call(token, chat())
    service.finish(order_id, {"status": "delivered", "leads_delivered": 3, "delivery_url": "https://f.example/x"})
    assert store.balance(buyer)["balance_microusd"] == 3 * USD - 600_000
    gamma.units[buyer] = 2 * 10**9  # the counter going backwards is refused
    with pytest.raises(LeadOrderServiceError) as backwards:
        service.balance(buyer)
    assert backwards.value.status == 422 and backwards.value.code == "gamma_counter_backwards"
    assert first["mode"] == "live"


# -- shadow mode: miner keys, failover, rewards, payouts ------------------------


def test_live_ranks_miner_keys_fails_over_and_owes_rewards(store, owner):
    service, registrar, buyer, epoch, _ = live(store, owner, epoch_value=2000)
    miner_a, miner_b = hotkey(), hotkey()
    accept_terms(service, miner_a)
    accept_terms(service, miner_b)
    key_a = service.register_key(miner_a, KEY)
    key_b = service.register_key(miner_b, KEY)
    with pytest.raises(LeadOrderServiceError) as twice:
        service.register_key(miner_a, KEY)
    assert twice.value.code == "active_key_exists"
    registrar.secrets = {key_a["key_id"]: "a" * 40, key_b["key_id"]: "b" * 40}

    order = service.place_order(buyer, {"leads": 10, **ORDER})
    token, order_id = claim(service)
    spent = ProviderResponse(402, {"content-type": "application/json"},
                             json.dumps({"error": {"code": 402, "message": "Insufficient credits"}}).encode())
    transport = FakeTransport([spent, openrouter_ok("0.800000"), openrouter_ok("0.200000")])
    broker = broker_for(store, transport, registrar=registrar, epoch=epoch)
    assert broker.call(token, dear_chat()).status == 200
    # First key is A (registered first, no spend yet); its 402 moves the
    # order to B; B serves the rest of the order.
    assert [s["headers"]["authorization"][-1] for s in transport.sent] == ["a", "b"]
    assert broker.call(token, dear_chat()).status == 200
    assert transport.sent[-1]["headers"]["authorization"].endswith("b")
    calls = store.calls(order["order_id"])
    assert [c["cost_microusd"] for c in calls] == [0, 800_000, 200_000]
    assert store.exhausted_keys(order["order_id"]) == [key_a["key_id"]]
    service.finish(order_id, {"status": "delivered", "leads_delivered": 10, "delivery_url": "https://f.example/x"})
    receipt = service.public_receipt(order["order_id"])["receipt"]
    assert receipt["rewards_microusd"] == {miner_b: 1_250_000}  # $1.00 x 1.25
    assert "what_if_rewards_microusd" not in receipt

    # B now ranks first by settled spend in the window.
    service.place_order(buyer, {"leads": 10, **ORDER})
    token2, order_id2 = claim(service)
    transport2 = FakeTransport([openrouter_ok("0.000100")])
    broker_for(store, transport2, registrar=registrar, epoch=epoch).call(token2, chat())
    assert transport2.sent[0]["headers"]["authorization"].endswith("b")
    service.finish(order_id2, {"status": "failed"})
    revoke_all_keys(store)


def test_shadow_runs_on_leadpoet_keys_and_publishes_what_if_rewards_only(store):
    # Old behaviour: a shadow order spent registered miner keys and owed
    # rewards, which moved the live ranking. Now it never touches them.
    revoke_all_keys(store)
    staff, miner = hotkey(), hotkey()
    registrar = FakeRegistrar({})
    epoch = Epoch(2100)
    _, service, _ = make(store, "shadow", staff=[staff], registrar=registrar, epoch=epoch)
    accept_terms(service, miner)
    key = service.register_key(miner, KEY)  # registration works in shadow
    registrar.secrets = {key["key_id"]: "a" * 40}
    order = service.place_order(staff, {"leads": 10, **ORDER})
    token, order_id = claim(service)
    transport = FakeTransport([openrouter_ok("0.800000")])
    broker_for(store, transport, registrar=registrar, epoch=epoch).call(token, dear_chat())
    assert transport.sent[0]["headers"]["authorization"] == "Bearer " + SECRET
    assert {c["miner_hotkey"] for c in store.calls(order_id)} == {None}
    assert key["key_id"] not in store.key_spend(since_epoch=0)
    service.finish(order_id, {"status": "delivered", "leads_delivered": 10, "delivery_url": "https://f.example/x"})
    receipt = service.public_receipt(order["order_id"])["receipt"]
    assert receipt["rewards_microusd"] == {} and receipt["what_if_rewards_microusd"] == 1_000_000  # $0.80 x 1.25
    doc = service.run_epoch(2100)
    assert doc["mode"] == "shadow" and doc["payments"] == []
    assert not [r for r in store.owed(through_epoch=10**9, mode="shadow") if r["order_id"] == order["order_id"]]
    assert service.statement(miner, "0")["calls"] == []
    with pytest.raises(ArenaStoreError, match="only live"):
        store.record_payout(epoch=4_000_000, mode="live", document={"payout_hash": "sha256:" + "c" * 64},
                            payments=[{"order_id": order["order_id"], "miner_hotkey": miner, "paid_microusd": 1}])
    revoke_all_keys(store)


class FixedPrice:
    def __init__(self, emission_microusd):
        self.emission = emission_microusd

    def valuation(self, epoch):
        if self.emission is None:
            raise rewards.EmissionPriceUnavailable("no quote")
        return {"emission_microusd": self.emission, "block": 50, "tao_price_source": "test"}


def test_live_epoch_pays_owed_rewards_from_a_published_price_and_owes_them_without_one(store, owner):
    order, miner = live_order_with_miner_reward(store, owner, epoch_value=7000)
    config, _, epoch = make(store, "live", gamma=FakeGamma(), epoch=Epoch(7000))

    def service(price):
        return LeadOrderService(config=config, policy=POLICY, policy_document=POLICY_DOC, store=store, signer=SIGNER,
                                current_epoch=epoch, gamma_reader=FakeGamma(), emission_price=price)

    def remaining():
        return [(r["owed_microusd"], r["paid_microusd"]) for r in store.owed(through_epoch=10**9, mode="live")
                if r["order_id"] == order["order_id"]]

    # No price: nothing is paid or recorded, and the reward stays owed.
    assert service(FixedPrice(None)).run_epoch(7001, registered_hotkeys=[miner]) is None
    assert store.reward_epoch(7001) is None and remaining() == [(500_000, 0)]
    # $1 of miner emission, 30% cap: $0.30 of the $0.50 is paid now, $0.20 carries forward.
    doc = service(FixedPrice(1_000_000)).run_epoch(7001, registered_hotkeys=[miner])
    assert doc["emission_valuation"] == {"emission_microusd": 1_000_000, "block": 50, "tao_price_source": "test"}
    assert doc["payments"] == [{"order_id": order["order_id"], "miner_hotkey": miner, "paid_microusd": 300_000,
                                "call_id": None}]
    assert doc["carry_forward_microusd"] == 200_000 and remaining() == [(500_000, 300_000)]
    # The stored document is the signed one anyone can fetch.
    assert store.reward_epoch(7001)["payout_hash"] == doc["payout_hash"]
    # The next epoch pays the rest at that epoch's own price.
    later = service(FixedPrice(10_000_000)).run_epoch(7002, registered_hotkeys=[miner])
    assert [p["paid_microusd"] for p in later["payments"]] == [200_000] and remaining() == [(500_000, 500_000)]


def test_weight_state_and_live_payout_must_agree_or_the_database_refuses(store, owner):
    """Paid in the ledger means signed in the epoch's weight state, and the reverse."""
    from lab_arena.lead_orders.weights import LeadOrderLane

    order, miner = live_order_with_miner_reward(store, owner, epoch_value=8000)
    config, _, epoch = make(store, "live", gamma=FakeGamma(), epoch=Epoch(8000))
    service = LeadOrderService(config=config, policy=POLICY, policy_document=POLICY_DOC, store=store, signer=SIGNER,
                               current_epoch=epoch, gamma_reader=FakeGamma(), emission_price=FixedPrice(None))
    burn, reads, metagraph_reads = hotkey(), [], []

    def registered():
        metagraph_reads.append(1)
        return [miner, burn]

    lane = LeadOrderLane(service, registered_hotkeys=registered)

    def valuation():
        reads.append(1)
        return {"block": 50, "block_hash": "0x" + "c" * 64, "miner_alpha_rao": 10**9, "tao_rao_per_alpha": 10**9,
                "tao_price_microusd": 1_000_000, "price_updated_at": 200, "observed_at": 201,
                "emission_microusd": 1_000_000}

    def settle(epoch_id, collection_microusd=0):
        return lane.settle(epoch_id, collection_microusd=collection_microusd, burn_hotkey=burn, valuation=valuation)

    def publish(epoch_id, lead_orders):
        # The fields the table requires. These guards read only lead_orders; the signed
        # collection state itself is covered in tests/lab_arena/collection_rewards_test.py.
        doc = {"schema_version": "leadpoet.arena.accepted_weight_state.v1", "network": "finney",
               "genesis_hash": "2f" * 32, "netuid": 71, "epoch": epoch_id, "valid_from_block": 50,
               "valid_until_block": 409, "reward_basis": {}, "burn_hotkey": burn, "issued_at": "2026-11-23T00:01:00Z",
               "state_hash": "sha256:" + hashlib.sha256(b"%d" % epoch_id).hexdigest(), "signature": {},
               **({"lead_orders": lead_orders} if lead_orders else {})}
        return sql(owner, "SELECT public.lab_arena_publish_weight_state_v1('finney',71,%s,%s,%s)",
                   (epoch_id, doc["state_hash"], json.dumps(doc)))

    def owed():
        return [(r["owed_microusd"], r["paid_microusd"]) for r in store.owed(through_epoch=10**9, mode="live")
                if r["order_id"] == order["order_id"]]

    # The payout was capped against a collection total the database no longer agrees with: nothing is paid.
    with pytest.raises(ArenaStoreError, match="collection total changed"):
        settle(8001, collection_microusd=900_000)
    assert store.reward_epoch(8001) is None and owed() == [(500_000, 0)]
    # $1 of emission, no collection work: the policy's 30% lane cap pays $0.30 of the $0.50.
    settled = settle(8001)
    assert settled["lead_orders"]["paid_microusd"] == {miner: 300_000} and settled["valuation"] == valuation()
    recorded = store.reward_epoch(8001)
    assert recorded["collection_microusd"] == 0 and recorded["min_hotkey_microusd"] == 16
    assert owed() == [(500_000, 300_000)]
    # A retry returns the recorded payout and its valuation: no new price or metagraph read.
    reads.clear(), metagraph_reads.clear()
    assert settle(8001) == settled and reads == [] and metagraph_reads == []
    # The epoch's state must carry exactly that payout.
    for wrong in (None, {"payout_hash": "sha256:" + "0" * 64, "paid_microusd": {miner: 300_000}},
                  dict(settled["lead_orders"], paid_microusd={miner: 300_001})):
        with pytest.raises(Exception, match="lead order payout"):
            publish(8001, wrong)
    publish(8001, settled["lead_orders"])
    # A state may not claim a payout that was never recorded.
    with pytest.raises(Exception, match="lead order payout"):
        publish(8002, settled["lead_orders"])
    # Once an epoch's weights are signed, no payout is recorded for it: the rest stays owed.
    publish(8002, None)
    with pytest.raises(ArenaStoreError, match="already signed"):
        settle(8002)
    assert store.reward_epoch(8002) is None and owed() == [(500_000, 300_000)]
    # The burn hotkey is never paid, even when it is owed: its reward waits.
    burn_lane = LeadOrderLane(service, registered_hotkeys=lambda: [burn])
    assert burn_lane.settle(8003, collection_microusd=0, burn_hotkey=miner, valuation=valuation) is None
    assert store.reward_epoch(8003) is None
    # The next epoch pays the rest.
    later = settle(8003)
    assert later["lead_orders"]["paid_microusd"] == {miner: 200_000} and owed() == [(500_000, 500_000)]
    publish(8003, later["lead_orders"])
    # Nothing owed: nothing recorded, no price read.
    reads.clear()
    assert settle(8004) is None and reads == [] and store.reward_epoch(8004) is None


def sql(owner, query, params):
    with owner.cursor() as cur:
        cur.execute(query, params)
        return [tuple(row) for row in cur.fetchall()] if cur.description else []


TERMS_BODY = json.dumps({"terms_sha256": DEFAULT_POLICY_BODY["miner_terms_sha256"],
                         "terms_version": DEFAULT_POLICY_BODY["miner_terms_version"]},
                        sort_keys=True, separators=(",", ":")).encode()


def signed_identity(miner, body=TERMS_BODY):
    return BuyerIdentity(hotkey=miner, signed_at=1_763_500_000, host="gw.example", signature="ab" * 64,
                         body_sha256=hashlib.sha256(body).hexdigest())


def accept_terms(service, miner):
    return service.accept_miner_terms(signed_identity(miner), {"terms_version": POLICY.miner_terms_version,
                                                               "terms_sha256": POLICY.miner_terms_sha256})


def revoke_all_keys(store):
    for provider in ("openrouter", "deepline"):
        for row in store.keys(provider):
            store.revoke_key(row["key_id"], reason="test reset")


KEY = {"provider": "openrouter", "credentials": {}, "daily_cap_usd": "50"}


def live(store, owner, *, epoch_value, registrar=None):
    """A live service with one confirmed, funded buyer and no active keys."""
    revoke_all_keys(store)
    gamma, buyer = FakeGamma(), hotkey()
    registrar = registrar or FakeRegistrar({})
    _, service, epoch = make(store, "live", gamma=gamma, registrar=registrar, epoch=Epoch(epoch_value))
    with pytest.raises(LeadOrderServiceError):
        service.register_buyer(buyer, {"email": "b@acme.io", "company": "Acme", "country": "US", "terms_version": "t1"})
    confirm(owner, buyer)
    gamma.units[buyer] = 100 * 10**9
    return service, registrar, buyer, epoch, gamma


def live_order_with_miner_reward(store, owner, *, epoch_value=3000):
    """A finished live order whose spend went through a miner's key."""
    service, registrar, buyer, epoch, _ = live(store, owner, epoch_value=epoch_value)
    miner = hotkey()
    accept_terms(service, miner)
    key = service.register_key(miner, KEY)
    registrar.secrets = {key["key_id"]: "m" * 40}
    service.place_order(buyer, {"leads": 10, **ORDER})
    token, order_id = claim(service)
    transport = FakeTransport([openrouter_ok("0.400000")])
    broker_for(store, transport, registrar=registrar, epoch=epoch).call(token, dear_chat())
    assert transport.sent[0]["headers"]["authorization"] == "Bearer " + "m" * 40
    service.finish(order_id, {"status": "delivered", "leads_delivered": 10, "delivery_url": "https://f.example/x"})
    revoke_all_keys(store)
    return {"order_id": order_id}, miner


def test_live_payout_pays_oldest_first_under_cap_and_never_overpays(store, owner):
    order, miner = live_order_with_miner_reward(store, owner)
    entries = [row for row in store.owed(through_epoch=10**9, mode="live") if row["order_id"] == order["order_id"]]
    assert entries == [dict(entries[0], miner_hotkey=miner, owed_microusd=500_000)]
    epoch = 5_000_000
    payments = [{"order_id": entries[0]["order_id"], "miner_hotkey": entries[0]["miner_hotkey"],
                 "paid_microusd": entries[0]["owed_microusd"] + 1}]
    doc = {"payout_hash": "sha256:" + "a" * 64, "epoch": epoch}
    with pytest.raises(ArenaStoreError, match="exceeds"):
        store.record_payout(epoch=epoch, mode="live", document=doc, payments=payments)
    payments[0]["paid_microusd"] = 1
    assert store.record_payout(epoch=epoch, mode="live", document=doc, payments=payments)["status"] == "recorded"
    assert store.record_payout(epoch=epoch, mode="live", document=doc, payments=payments)["status"] == "exists"
    paid = [r for r in store.owed(through_epoch=10**9, mode="live") if r["order_id"] == entries[0]["order_id"]
            and r["miner_hotkey"] == entries[0]["miner_hotkey"]]
    assert paid[0]["paid_microusd"] == 1


def test_sql_charge_matches_money_module_for_many_cases(store):
    staff = hotkey()
    _, service, epoch = make(store, "staff", staff=[staff])
    rng = random.Random(7)
    for _ in range(6):
        leads = rng.randint(10, 40)
        cost = rng.randint(0, leads * 500_000)
        delivered = rng.randint(0, leads)
        service.place_order(staff, {"leads": leads, **ORDER})
        token, order_id = claim(service)
        if cost:
            broker_for(store, FakeTransport([openrouter_ok("%.6f" % (cost / USD))]), epoch=epoch).call(
                token, chat(max_tokens=1))
        result = service.finish(order_id, {"status": "delivered", "leads_delivered": delivered,
                                        "delivery_url": "https://f.example/x"})
        assert result["charge_microusd"] == money.charge_microusd(
            POLICY, hold_microusd=leads * 200_000, delivered=delivered, spend_microusd=cost)


# -- review fixes: policy version, per-order keys, open calls, reconcile ----------


def test_rewards_and_receipt_use_the_orders_policy_version_not_the_current_one(store, owner):
    # v1: ratio 0.40, premium 0.25. v2 raises the premium to 0.45. An order
    # placed under v1 and settled after the switch must still pay x1.25.
    service_v1, registrar, buyer, epoch, gamma = live(store, owner, epoch_value=6000)
    miner = hotkey()
    accept_terms(service_v1, miner)
    key = service_v1.register_key(miner, KEY)
    registrar.secrets = {key["key_id"]: "v" * 40}
    order = service_v1.place_order(buyer, {"leads": 10, **ORDER})
    token, order_id = claim(service_v1)
    broker_for(store, FakeTransport([openrouter_ok("0.400000")]), registrar=registrar, epoch=epoch).call(
        token, dear_chat())

    v2_doc = sign_policy(SIGNER, dict(DEFAULT_POLICY_BODY, version=2, price_per_lead_usd="0.30",
                                      min_charge_ratio="0.60", miner_premium="0.45"))
    v2 = load_policy(v2_doc, public_key_der=SIGNER.public_key_der, expected_public_key_hash=SIGNER.public_key_hash)
    config = LeadOrdersConfig(mode="live", runner_secret="r" * 40, policy_path="/x", gamma_service="7",
                              terms_version="t1", public_base_url="https://gw.example")
    service_v2 = LeadOrderService(config=config, policy=v2, policy_document=v2_doc, store=store, signer=SIGNER,
                                  current_epoch=epoch, gamma_reader=gamma, registrar=registrar)
    service_v2.finish(order_id, {"status": "delivered", "leads_delivered": 10, "delivery_url": "https://f.example/x"})
    receipt = service_v2.public_receipt(order["order_id"])["receipt"]
    assert receipt["rewards_microusd"] == {miner: 500_000}  # $0.40 x 1.25, not x 1.45
    assert receipt["policy_version"] == 1 and receipt["policy_hash"] == POLICY.policy_hash
    assert receipt["charge_microusd"] == 2 * USD  # v1 price, from the order's stored terms
    # The policy each receipt names is public, by version.
    assert service_v2.public_policy(1) == POLICY_DOC
    with pytest.raises(LeadOrderServiceError) as unknown:
        service_v2.public_policy("99")
    assert (unknown.value.status, unknown.value.code) == (404, "policy_unknown")
    revoke_all_keys(store)


def test_staff_order_spends_only_leadpoet_keys_after_the_gateway_moves_to_shadow(store):
    revoke_all_keys(store)
    staff, miner = hotkey(), hotkey()
    registrar = FakeRegistrar({})
    _, staff_service, epoch = make(store, "staff", staff=[staff])
    staff_service.place_order(staff, {"leads": 10, **ORDER})
    token, order_id = claim(staff_service)
    # The gateway switches to shadow mid-order and a miner key is registered.
    _, shadow_service, _ = make(store, "shadow", staff=[staff], registrar=registrar, epoch=epoch)
    accept_terms(shadow_service, miner)
    key = shadow_service.register_key(miner, {"provider": "openrouter", "credentials": {}, "daily_cap_usd": "50"})
    registrar.secrets = {key["key_id"]: "z" * 40}
    transport = FakeTransport([openrouter_ok("0.100000")])
    broker_for(store, transport, registrar=registrar, epoch=epoch).call(token, chat())
    assert transport.sent[0]["headers"]["authorization"] == "Bearer " + SECRET  # Leadpoet's key
    assert {c["miner_hotkey"] for c in store.calls(order_id)} == {None}
    shadow_service.finish(order_id, {"status": "failed"})
    revoke_all_keys(store)


def test_finish_refuses_open_calls_and_only_the_overdue_sweep_settles_them_uncertain(store, owner):
    staff = hotkey()
    _, service, epoch = make(store, "staff", staff=[staff])
    order = service.place_order(staff, {"leads": 10, **ORDER})
    token, order_id = claim(service)
    broker_for(store, FakeTransport([openrouter_ok("0.000100")]), epoch=epoch).call(token, chat())
    leadpoet = ProviderKey(key_id="leadpoet-openrouter", provider="openrouter", miner_hotkey=None,
                           registered_sequence=0, internal_fingerprint="sha256:" + "e" * 64)
    in_flight = str(uuid.uuid4())  # reserved, provider has not answered yet
    assert store.reserve_call(call_id=in_flight, order_id=order_id, provider="openrouter", key_id=leadpoet.key_id,
                              miner_hotkey=None, public_fingerprint=leadpoet.public_fingerprint,
                              reserve_microusd=300_000)["status"] == "reserved"
    with pytest.raises(LeadOrderServiceError) as too_many:
        service.finish(order_id, {"status": "delivered", "leads_delivered": 11, "delivery_url": "https://f.example/x"})
    assert too_many.value.status == 422 and too_many.value.code == "leads_delivered_invalid"
    # Old behaviour: finish closed the open call uncertain, so its cost was
    # never repaid. Now the runner must wait for its own calls.
    with pytest.raises(LeadOrderServiceError) as still_open:
        service.finish(order_id, {"status": "delivered", "leads_delivered": 1, "delivery_url": "https://f.example/x"})
    assert (still_open.value.status, still_open.value.code) == (409, "calls_open")
    assert store.view(order_id)["status"] == "running"

    # A stuck call: past deadline and grace, the sweep closes it uncertain.
    with owner.cursor() as cur:
        cur.execute("ALTER TABLE lead_order_claims DISABLE TRIGGER lead_order_claims_append_only")
        cur.execute("UPDATE lead_order_claims SET deadline=now()-interval '6 minutes' WHERE order_id=%s", (order_id,))
        cur.execute("ALTER TABLE lead_order_claims ENABLE TRIGGER lead_order_claims_append_only")
    assert service.claim() is None
    statuses = {c["call_id"]: c["status"] for c in store.calls(order_id)}
    assert statuses[in_flight] == "uncertain" and "open" not in statuses.values()
    public = service.public_receipt(order["order_id"])
    assert public["receipt"]["charge_microusd"] == money.charge_microusd(
        POLICY, hold_microusd=2 * USD, delivered=0, spend_microusd=100)  # the open call is never charged

    # The provider answers after the finish: kept aside, nothing published moves.
    assert store.settle_call(call_id=in_flight, cost_microusd=250_000, http_status=200, epoch=epoch())["status"] == "late"
    assert {c["call_id"]: c["status"] for c in store.calls(order_id)} == statuses
    assert service.public_receipt(order["order_id"]) == public
    receipts.verify_receipt(public["receipt"], calls=public["calls"], public_key_der=SIGNER.public_key_der,
                            expected_public_key_hash=SIGNER.public_key_hash)
    assert store.reserve_call(call_id=str(uuid.uuid4()), order_id=order_id, provider="openrouter",
                              key_id=leadpoet.key_id, miner_hotkey=None, public_fingerprint=leadpoet.public_fingerprint,
                              reserve_microusd=1)["status"] == "closed"


class BrokenSigner(signing.ArenaSigner):
    def __init__(self, public_key_der):
        self.public_key_der = public_key_der

    def sign(self, message):
        raise RuntimeError("kms unavailable")


def test_reconcile_settles_a_finished_order_whose_receipt_failed(store):
    staff = hotkey()
    config, healthy, epoch = make(store, "staff", staff=[staff])
    broken = LeadOrderService(config=config, policy=POLICY, policy_document=POLICY_DOC, store=store,
                              signer=BrokenSigner(public_key_der=SIGNER.public_key_der), current_epoch=epoch)
    order = broken.place_order(staff, {"leads": 10, **ORDER})
    token, order_id = claim(broken)
    with pytest.raises(RuntimeError):
        broken.finish(order_id, {"status": "failed"})
    assert store.view(order_id)["status"] == "failed" and not store.view(order_id)["receipt"]
    assert order_id in store.unsettled(limit=500)
    assert healthy.claim() is None  # the next claim pass reconciles
    receipt = healthy.public_receipt(order["order_id"])["receipt"]
    assert receipt["status"] == "failed" and order_id not in store.unsettled(limit=500)
    assert healthy.reconcile() == []  # idempotent


def test_claim_race_takes_the_next_order_instead_of_failing(store, database):
    import threading
    import time as _time

    staff = hotkey()
    _, service, _ = make(store, "staff", staff=[staff])
    first = service.place_order(staff, {"leads": 10, **ORDER})["order_id"]
    second = service.place_order(staff, {"leads": 10, **ORDER})["order_id"]
    psycopg2, dsn = database
    rival = psycopg2.connect(**dsn)  # another claimer, mid-transaction on the first order
    try:
        with rival.cursor() as cur:
            cur.execute("INSERT INTO lead_order_claims(order_id,token_hash,deadline) VALUES(%s,%s,now()+interval '1 hour')",
                        (first, order_token_hash("rival")))
        result = {}

        def claim_now():
            try:
                result["claimed"] = service.claim()
            except Exception as exc:  # pragma: no cover - the failure this test guards
                result["error"] = exc

        worker = threading.Thread(target=claim_now)
        worker.start()
        _time.sleep(0.5)  # the worker is now waiting on the rival's claim row
        rival.commit()
        worker.join(10)
    finally:
        rival.close()
    assert "error" not in result and result["claimed"]["order_id"] == second
    service.finish(first, {"status": "failed"})
    service.finish(second, {"status": "failed"})


def test_order_retry_with_the_same_id_and_a_different_body_is_409(store):
    staff = hotkey()
    _, service, _ = make(store, "staff", staff=[staff])
    order_id = str(uuid.uuid4())
    placed = service.place_order(staff, {"order_id": order_id, "leads": 10, **ORDER})
    assert service.place_order(staff, {"order_id": order_id, "leads": 10, **ORDER})["order_id"] == placed["order_id"]
    for changed in ({"leads": 12, **ORDER}, {"leads": 10, "seller": ORDER["seller"], "search": dict(ORDER["search"], signal="funding")}):
        with pytest.raises(LeadOrderServiceError) as taken:
            service.place_order(staff, dict(changed, order_id=order_id))
        assert taken.value.status == 409 and taken.value.code == "order_id_taken"
    token, claimed = claim(service)
    service.finish(claimed, {"status": "failed"})


RAW_V2 = {"x-deepline-execute-response-contract": "raw-v2", "x-deepline-execute-response-intent": "raw",
          "x-deepline-tool-error-schema": "1"}


def test_deepline_raw_v2_headers_are_sent_and_never_reach_receipts_or_the_ledger(store):
    staff = hotkey()
    _, service, epoch = make(store, "staff", staff=[staff])
    placed = service.place_order(staff, {"leads": 10, **ORDER})
    token, order_id = claim(service)
    # A raw-v2 execute answer, as Icypeas's verification returns it.
    raw = ProviderResponse(200, {"content-type": "application/json"}, json.dumps({
        "status": "completed", "toolResponse": {"success": True, "item": {"_id": "v1", "status": "NONE"}},
        "billing": {"credits_charged": 0.02}}).encode())
    transport = FakeTransport([raw])
    broker = broker_for(store, transport, epoch=epoch)
    request = BrokerRequest("deepline", "POST", "api/v2/integrations/icypeas_email_verification/execute", {},
                            b'{"payload":{"email":"ada.one@northwind.com"}}', "application/json",
                            headers=dict(RAW_V2, **{"x-other": "dropped", "authorization": "Bearer runner"}))
    assert broker.call(token, request).status == 200
    sent = transport.sent[0]["headers"]
    assert {name: sent[name] for name in RAW_V2} == RAW_V2
    assert "x-other" not in sent and sent["authorization"] == "Bearer " + SECRET
    for name, value in RAW_V2.items():
        with pytest.raises(LeadOrderBrokerError) as refused:
            broker.call(token, BrokerRequest("deepline", "POST", "api/v2/integrations/icypeas_read_results/execute",
                                             {}, b"{}", "application/json", headers={name: value + "x"}))
        assert (refused.value.status, refused.value.code) == (400, "header_not_allowed:" + name)
    assert len(transport.sent) == 1  # refused calls are never sent
    service.finish(order_id, {"status": "delivered", "leads_delivered": 10, "delivery_url": "https://files.example/o.csv"})
    public = service.public_receipt(placed["order_id"])
    assert public["receipt"]["spend_microusd"] == 2_000  # 0.02 credits x $0.10, read from the raw-v2 billing
    stored = json.dumps([public, store.view(order_id)], default=str)
    assert "raw-v2" not in stored and "x-deepline" not in stored


# -- miner data terms and the suppression feed -----------------------------------


def test_keys_need_signed_current_miner_terms_and_outdated_acceptances_are_skipped(store, owner):
    live_service, registrar, buyer, epoch, _ = live(store, owner, epoch_value=7000)
    staff, miner, late = hotkey(), hotkey(), hotkey()
    config, service, _ = make(store, "shadow", staff=[staff], registrar=registrar, epoch=epoch)
    with pytest.raises(LeadOrderServiceError) as unsigned:
        service.register_key(miner, {"provider": "openrouter", "credentials": {}, "daily_cap_usd": "50"})
    assert (unsigned.value.status, unsigned.value.code) == (403, "miner_terms_not_accepted")
    with pytest.raises(LeadOrderServiceError) as old:
        service.accept_miner_terms(signed_identity(miner), {"terms_version": "miner-terms-0",
                                                            "terms_sha256": POLICY.miner_terms_sha256})
    assert (old.value.status, old.value.code) == (422, "terms_version_not_current")
    with pytest.raises(LeadOrderServiceError) as other_text:  # current version, different text
        service.accept_miner_terms(signed_identity(miner), {"terms_version": POLICY.miner_terms_version,
                                                            "terms_sha256": "0" * 64})
    assert (other_text.value.status, other_text.value.code) == (422, "terms_version_not_current")
    with pytest.raises(LeadOrderServiceError) as no_hash:
        service.accept_miner_terms(signed_identity(miner), {"terms_version": POLICY.miner_terms_version})
    assert (no_hash.value.status, no_hash.value.code) == (422, "terms_invalid")
    served = service.miner_terms()
    assert served["terms_sha256"] == hashlib.sha256(served["text"].encode()).hexdigest() == POLICY.miner_terms_sha256
    stale = LeadOrderService(config=config, policy=unsigned_policy(dict(DEFAULT_POLICY_BODY, miner_terms_sha256="0" * 64)),
                             policy_document=POLICY_DOC, store=store, signer=SIGNER, current_epoch=lambda: 1)
    with pytest.raises(LeadOrderServiceError) as unavailable:
        stale.miner_terms()
    assert (unavailable.value.status, unavailable.value.code) == (503, "terms_unavailable")
    loose = json.dumps({"terms_version": POLICY.miner_terms_version, "terms_sha256": POLICY.miner_terms_sha256}).encode()
    with pytest.raises(LeadOrderServiceError) as not_canonical:  # same values, other bytes
        service.accept_miner_terms(signed_identity(miner, loose), {"terms_version": POLICY.miner_terms_version,
                                                                   "terms_sha256": POLICY.miner_terms_sha256})
    assert (not_canonical.value.status, not_canonical.value.code) == (422, "terms_body_not_canonical")
    first = accept_terms(service, miner)
    assert first["status"] == "accepted" and first["terms_version"] == POLICY.miner_terms_version
    assert accept_terms(service, miner)["status"] == "exists"
    stored = sql(owner, "SELECT signed_method,signed_host,signed_path,signed_at,body_sha256,signature "
                        "FROM public.lead_order_miner_terms WHERE miner_hotkey=%s", (miner,))
    assert stored == [("POST", "gw.example", "/lead-orders/miner-terms", 1_763_500_000,
                       hashlib.sha256(TERMS_BODY).hexdigest(), "ab" * 64)]
    key = service.register_key(miner, {"provider": "openrouter", "credentials": {}, "daily_cap_usd": "50"})
    registrar.secrets = {key["key_id"]: "t" * 40}
    # The text changes but the version label does not: the key stays
    # registered but is skipped until its miner signs the new text, so the
    # order runs on Leadpoet's key.
    v2_doc = sign_policy(SIGNER, dict(DEFAULT_POLICY_BODY, version=3, miner_terms_sha256="1" * 64))
    v2 = load_policy(v2_doc, public_key_der=SIGNER.public_key_der, expected_public_key_hash=SIGNER.public_key_hash)
    live_service.place_order(buyer, {"leads": 10, **ORDER})
    token, order_id = claim(live_service)
    transport = FakeTransport([openrouter_ok("0.000100"), openrouter_ok("0.000100")])
    broker_v2 = LeadOrderBroker(store=store, policy=v2, transport=transport, registrar=registrar,
                                leadpoet_secrets={"openrouter": SECRET, "deepline": SECRET},
                                price_table=lambda: PRICE_TABLE, current_epoch=epoch)
    broker_v2.call(token, chat())
    assert transport.sent[0]["headers"]["authorization"] == "Bearer " + SECRET
    broker_for(store, transport, registrar=registrar, epoch=epoch).call(token, chat())
    assert transport.sent[1]["headers"]["authorization"] == "Bearer " + "t" * 40  # current terms: ranked
    live_service.finish(order_id, {"status": "failed"})
    with pytest.raises(Exception):
        sql(owner, "UPDATE public.lead_order_miner_terms SET terms_sha256=%s WHERE miner_hotkey=%s", ("2" * 64, miner))
    revoke_all_keys(store)


def test_suppression_push_stores_hashes_only_and_the_feed_is_for_buyers_and_signed_miners(store, owner):
    staff, miner, stranger, buyer = hotkey(), hotkey(), hotkey(), hotkey()
    _, service, _ = make(store, "staff", staff=[staff])
    email = suppressions.email_hash("  Ada.One@Northwind.com ")
    linkedin = suppressions.linkedin_hash("https://de.linkedin.com/in/Ada-One/")
    assert email == hashlib.sha256(b"ada.one@northwind.com").hexdigest()
    assert linkedin == hashlib.sha256(b"linkedin.com/in/ada-one").hexdigest()
    pushed = service.push_suppressions({"suppressions": [
        {"email_sha256": email, "linkedin_sha256": linkedin, "added_at": "2026-11-19T10:00:00Z"},
        {"email_sha256": "0" * 64, "added_at": "2026-11-19T10:01:00.123Z"}]})
    assert pushed == {"received": 2, "added": 3}
    assert service.push_suppressions({"suppressions": [
        {"email_sha256": email, "added_at": "2026-11-19T11:00:00Z"}]}) == {"received": 1, "added": 0}
    good = {"email_sha256": "1" * 64, "added_at": "2026-11-19T10:00:00Z"}
    for bad, code in (
        ({"email_sha256": "ada.one@northwind.com", "added_at": "2026-11-19T10:00:00Z"}, "hash_invalid"),
        ({"linkedin_sha256": "linkedin.com/in/ada-one", "added_at": "2026-11-19T10:00:00Z"}, "hash_invalid"),
        ({"email_sha256": "A" * 64, "added_at": "2026-11-19T10:00:00Z"}, "hash_invalid"),
        ({"email": "ada.one@northwind.com", "added_at": "2026-11-19T10:00:00Z"}, "field_not_allowed"),
        ({"linkedin_url": "https://linkedin.com/in/x", "added_at": "2026-11-19T10:00:00Z"}, "field_not_allowed"),
        ({"added_at": "2026-11-19T10:00:00Z"}, "hash_required"),
        ({"email_sha256": "2" * 64, "added_at": "2026-11-19"}, "added_at_invalid"),
        ({"email_sha256": "2" * 64, "added_at": "2999-01-01T00:00:00Z"}, "added_at_invalid"),
    ):
        with pytest.raises(LeadOrderServiceError) as refused:
            service.push_suppressions({"suppressions": [good, bad]})
        assert (refused.value.status, refused.value.code) == (422, code)
        assert "northwind" not in refused.value.code
    with pytest.raises(LeadOrderServiceError):
        service.push_suppressions({"suppressions": [good] * 501})
    rows = sql(owner, "SELECT kind,hash FROM public.lead_order_suppressions", ())
    assert all(len(h) == 64 for _, h in rows) and ("email_sha256", "1" * 64) not in rows  # whole push refused
    assert not any("northwind" in h or "linkedin" in h for _, h in rows)

    with pytest.raises(LeadOrderServiceError) as unauthorised:
        service.read_suppressions(stranger, "0")
    assert (unauthorised.value.status, unauthorised.value.code) == (403, "suppressions_not_allowed")
    accept_terms(service, miner)
    feed = service.read_suppressions(miner, "")
    mine = [r for r in feed["suppressions"] if r["hash"] in (email, linkedin, "0" * 64)]
    assert [(r["kind"], r["added_at"]) for r in mine] == [
        ("email_sha256", "2026-11-19T10:00:00Z"), ("linkedin_sha256", "2026-11-19T10:00:00Z"),
        ("email_sha256", "2026-11-19T10:01:00Z")]
    assert feed["next_since"] == feed["suppressions"][-1]["sequence"] and feed["more"] is False
    later = service.read_suppressions(miner, str(mine[0]["sequence"]))
    assert [r["hash"] for r in later["suppressions"] if r["hash"] in (email, linkedin, "0" * 64)] == [linkedin, "0" * 64]
    with pytest.raises(LeadOrderServiceError) as bad_since:
        service.read_suppressions(miner, "-1")
    assert bad_since.value.code == "since_invalid"
    # A confirmed buyer reads it too; a registered but unconfirmed one does not.
    _, live, _ = make(store, "live", gamma=FakeGamma())
    with pytest.raises(LeadOrderServiceError):
        live.register_buyer(buyer, {"email": "b@acme.io", "company": "Acme", "country": "US", "terms_version": "t1"})
    with pytest.raises(LeadOrderServiceError) as unconfirmed:
        service.read_suppressions(buyer, "0")
    assert unconfirmed.value.code == "suppressions_not_allowed"
    confirm(owner, buyer)
    assert service.read_suppressions(buyer, "0")["suppressions"]


def test_suppression_cursor_keeps_both_concurrent_pushes(database, owner, store):
    psycopg2, dsn = database
    cursor = sql(owner, "SELECT coalesce(max(sequence),0) FROM lead_order_suppressions", ())[0][0]
    hashes = [hashlib.sha256(uuid.uuid4().bytes).hexdigest() for _ in range(2)]
    push = "SELECT public.lead_order_add_suppressions_v1(%s::jsonb)"

    def batch(digest):
        return (json.dumps([{"kind": "email_sha256", "hash": digest, "added_at": "2026-11-20T00:00:00Z"}]),)

    first = psycopg2.connect(**dsn)
    second = psycopg2.connect(**dsn)
    second.autocommit = True
    try:
        sql(first, push, batch(hashes[0]))
        sql(second, "SET statement_timeout='5s'", ())
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(sql, second, push, batch(hashes[1]))
            try:
                deadline = monotonic() + 3
                while not pending.done():
                    if sql(owner, "SELECT %s=ANY(pg_blocking_pids(%s))",
                           (first.get_backend_pid(), second.get_backend_pid()))[0][0]:
                        break
                    assert monotonic() < deadline, "second push did not start"
                    sleep(0.01)
                # Poll the feed while the first push still holds its transaction.
                assert store.suppressions(since=cursor, limit=1) == []
            finally:
                first.commit()
            assert pending.result(timeout=5)[0][0]["added"] == 1
        for expected in hashes:
            row, = store.suppressions(since=cursor, limit=1)
            assert row["hash"] == expected
            cursor = row["sequence"]
        assert store.suppressions(since=cursor, limit=1) == []
    finally:
        first.close()
        second.close()


def test_orders_the_runner_would_refuse_are_refused_at_placement_with_no_hold(store, owner):
    staff = hotkey()
    _, service, _ = make(store, "staff", staff=[staff])
    seller, search = ORDER["seller"], ORDER["search"]
    market = search["market"]
    cases = (
        ({"leads": 1_001, **ORDER}, "leads_invalid"),
        ({"leads": 10, "seller": {"summary": "x"}, "search": search}, "seller_invalid:product"),
        ({"leads": 10, "seller": {"product": "   ", "summary": "x"}, "search": search}, "seller_invalid:product"),
        ({"leads": 10, "seller": {"product": "p" * 201, "summary": ""}, "search": search}, "seller_invalid:product"),
        ({"leads": 10, "seller": {"product": "Payroll"}, "search": search}, "seller_invalid:summary"),
        ({"leads": 10, "seller": dict(seller, summary="s" * 1_001), "search": search}, "seller_invalid:summary"),
        ({"leads": 10, "seller": seller, "search": dict(search, signal="")}, "search_invalid:signal"),
        ({"leads": 10, "seller": seller, "search": {"signal": "hiring"}}, "search_invalid:market"),
        ({"leads": 10, "seller": seller, "search": dict(search, terms=["t"] * 9)}, "search_invalid:terms"),
        ({"leads": 10, "seller": seller, "search": dict(search, terms=None)}, "search_invalid:terms"),
        ({"leads": 10, "seller": seller, "search": dict(search, buyer_titles=[""])}, "search_invalid:buyer_titles"),
        ({"leads": 10, "seller": seller, "search": dict(search, excluded_companies=["d" * 254])},
         "search_invalid:excluded_companies"),
        ({"leads": 10, "seller": seller, "search": dict(search, market=dict(market, countries=["US"] * 21))},
         "search_invalid:market.countries"),
        ({"leads": 10, "seller": seller, "search": dict(search, market=dict(market, industries="software"))},
         "search_invalid:market.industries"),
    )
    for body, code in cases:
        with pytest.raises(LeadOrderServiceError) as refused:
            service.place_order(staff, body)
        assert (refused.value.status, refused.value.code) == (422, code)
    assert sql(owner, "SELECT count(*) FROM public.lead_orders WHERE hotkey=%s", (staff,)) == [(0,)]
    # What the runner allows: an empty summary, missing lists, an empty market, extra fields.
    minimal = {"leads": 10, "seller": {"product": "Payroll", "summary": "", "site": "acme.io"},
               "search": {"signal": "hiring", "market": {}, "note": "kept"}}
    placed = service.place_order(staff, minimal)
    assert placed["status"] == "queued"
    claimed = service.claim()
    assert claimed["seller"] == minimal["seller"] and claimed["search"] == minimal["search"]
    service.finish(claimed["order_id"], {"status": "failed"})


# -- key-safety review: daily caps, overshoot, repayment, statement ----------------


@pytest.mark.parametrize("wrong_binding", ["provider", "public_fingerprint", "miner_hotkey", "key_id"])
def test_reserve_requires_the_registered_miner_key_binding(store, owner, wrong_binding):
    service, _, buyer, _, _ = live(store, owner, epoch_value=8900)
    miner = hotkey()
    accept_terms(service, miner)
    key = service.register_key(miner, KEY)
    service.place_order(buyer, {"leads": 10, **ORDER})
    _, order_id = claim(service)
    binding = dict(provider="openrouter", key_id=key["key_id"], miner_hotkey=miner,
                   public_fingerprint=key["public_fingerprint"])
    binding[wrong_binding] = {
        "provider": "deepline", "public_fingerprint": "sha256:" + "0" * 32,
        "miner_hotkey": hotkey(), "key_id": "missing-key",
    }[wrong_binding]
    assert store.reserve_call(call_id=str(uuid.uuid4()), order_id=order_id,
                              reserve_microusd=100_000, **binding)["status"] == "key_unknown"
    assert store.calls(order_id) == []
    service.finish(order_id, {"status": "failed"})


def test_concurrent_key_registration_keeps_one_active_key_per_miner_and_provider(database, owner):
    psycopg2, dsn = database
    miner = hotkey()
    register = "SELECT public.lead_order_register_key_v1(%s,%s,%s,%s,%s,%s,%s,%s)"

    def key(provider):
        key_id = uuid.uuid4().hex
        return (key_id, provider, miner, "lead-order:" + key_id, "Y2lwaGVy",
                "sha256:" + hashlib.sha256(key_id.encode()).hexdigest(), "sha256:" + key_id, USD)

    first = psycopg2.connect(**dsn)
    try:
        sql(first, "SET ROLE lab_arena_service", ())
        assert sql(first, register, key("openrouter"))[0][0]["status"] == "registered"
        # The second session can register another provider immediately. For the
        # same provider it must wait until the first registration is visible.
        sql(owner, "SET lock_timeout='300ms'", ())
        assert sql(owner, register, key("deepline"))[0][0]["status"] == "registered"
        second_key = key("openrouter")
        with pytest.raises(psycopg2.errors.LockNotAvailable):
            sql(owner, register, second_key)
        first.commit()
        assert sql(owner, register, second_key)[0][0]["status"] == "active_key_exists"
    finally:
        first.close()


def test_revocation_stops_new_reservations_but_preserves_inflight_settlement(store, owner):
    service, _, buyer, epoch, _ = live(store, owner, epoch_value=8901)
    miner = hotkey()
    accept_terms(service, miner)
    key = service.register_key(miner, KEY)
    service.place_order(buyer, {"leads": 10, **ORDER})
    _, order_id = claim(service)
    reservation = dict(order_id=order_id, provider="openrouter", key_id=key["key_id"],
                       miner_hotkey=miner, public_fingerprint=key["public_fingerprint"],
                       reserve_microusd=100_000)
    in_flight = str(uuid.uuid4())
    assert store.reserve_call(call_id=in_flight, **reservation)["status"] == "reserved"
    assert service.revoke_key(miner, key["key_id"])["status"] == "revoked"
    assert store.reserve_call(call_id=str(uuid.uuid4()), **reservation)["status"] == "key_unknown"
    assert store.settle_call(call_id=in_flight, cost_microusd=80_000, http_status=200,
                             epoch=epoch())["status"] == "settled"
    service.finish(order_id, {"status": "failed"})
    receipt = service.public_receipt(order_id)["receipt"]
    assert receipt["rewards_microusd"] == {miner: 100_000}
    row, = store.calls(order_id)
    assert row["call_id"] == in_flight and row["cost_microusd"] == 80_000


@pytest.mark.parametrize("revoke_first", [True, False], ids=["revoke-first", "reserve-first"])
def test_reserve_and_revoke_serialize_on_the_miner_key(database, store, owner, revoke_first):
    service, _, buyer, epoch, _ = live(store, owner, epoch_value=8902)
    miner = hotkey()
    accept_terms(service, miner)
    key = service.register_key(miner, KEY)
    service.place_order(buyer, {"leads": 10, **ORDER})
    _, order_id = claim(service)
    call_id = str(uuid.uuid4())
    reserve = ("SELECT public.lead_order_reserve_call_v1(%s,%s,%s,%s,%s,%s,%s)",
               (call_id, order_id, "openrouter", key["key_id"], miner, key["public_fingerprint"], 100_000))
    revoke = ("SELECT public.lead_order_revoke_key_v1(%s,%s)", (key["key_id"], "test revoke"))
    first, second = (revoke, reserve) if revoke_first else (reserve, revoke)
    psycopg2, dsn = database
    blocker = psycopg2.connect(**dsn)
    contender = psycopg2.connect(**dsn)
    contender.autocommit = True
    try:
        for connection in (blocker, contender):
            sql(connection, "SET statement_timeout='5s'", ())
            sql(connection, "SET ROLE lab_arena_service", ())
        assert sql(blocker, *first)[0][0]["status"] == ("revoked" if revoke_first else "reserved")
        with ThreadPoolExecutor(max_workers=1) as executor:
            pending = executor.submit(sql, contender, *second)
            try:
                deadline = monotonic() + 3
                while monotonic() < deadline:
                    waiting = sql(owner, "SELECT %s=ANY(pg_blocking_pids(%s))",
                                  (blocker.get_backend_pid(), contender.get_backend_pid()))[0][0]
                    if waiting or pending.done():
                        break
                    sleep(0.01)
                assert waiting, "The competing operation did not wait for the miner-key lock"
                blocker.commit()
                result = pending.result(timeout=5)[0][0]
                assert result["status"] == ("key_unknown" if revoke_first else "revoked")
            finally:
                blocker.rollback()
    finally:
        blocker.close()
        contender.close()
    if revoke_first:
        assert store.calls(order_id) == []
    else:
        assert store.settle_call(call_id=call_id, cost_microusd=80_000, http_status=200,
                                 epoch=epoch())["status"] == "settled"
    assert store.reserve_call(call_id=str(uuid.uuid4()), order_id=order_id, provider="openrouter",
                              key_id=key["key_id"], miner_hotkey=miner,
                              public_fingerprint=key["public_fingerprint"],
                              reserve_microusd=1)["status"] == "key_unknown"
    service.finish(order_id, {"status": "failed"})


def reward(cost, reserved):
    """Rewards stop at the reservation: min(cost, reserve) x 1.25."""
    return min(cost, reserved) * 1_250_000 // 1_000_000


def test_daily_cap_is_required_bounded_and_moves_the_order_to_the_next_key(store, owner):
    service, registrar, buyer, epoch, _ = live(store, owner, epoch_value=9000)
    miner_a, miner_b = hotkey(), hotkey()
    accept_terms(service, miner_a)
    accept_terms(service, miner_b)
    for payload, code in (({"provider": "openrouter", "credentials": {}}, "key_invalid"),
                          (dict(KEY, daily_cap_usd="0.99"), "daily_cap_invalid"),
                          (dict(KEY, daily_cap_usd="10000.01"), "daily_cap_invalid"),
                          (dict(KEY, daily_cap_usd="1.0000001"), "daily_cap_invalid"),
                          (dict(KEY, daily_cap_usd=True), "daily_cap_invalid"),
                          (dict(KEY, daily_cap_usd="NaN"), "daily_cap_invalid")):
        with pytest.raises(LeadOrderServiceError) as refused:
            service.register_key(miner_a, payload)
        assert (refused.value.status, refused.value.code) == (422, code)
    key_a = service.register_key(miner_a, dict(KEY, daily_cap_usd="1"))
    assert key_a["daily_cap_usd"] == "1.00"
    key_b = service.register_key(miner_b, KEY)
    registrar.secrets = {key_a["key_id"]: "a" * 40, key_b["key_id"]: "b" * 40}
    order = service.place_order(buyer, {"leads": 10, **ORDER})
    token, order_id = claim(service)
    # A's first call reports far more than its reservation; the next
    # reservation would take A over its $1.00 day, so the order moves to B.
    transport = FakeTransport([openrouter_ok("0.999999"), openrouter_ok("0.000100")])
    broker = broker_for(store, transport, registrar=registrar, epoch=epoch)
    assert broker.call(token, chat()).status == 200
    assert broker.call(token, chat()).status == 200
    assert [s["headers"]["authorization"][-1] for s in transport.sent] == ["a", "b"]
    first, second = store.calls(order_id)
    assert (first["miner_hotkey"], second["miner_hotkey"]) == (miner_a, miner_b)
    assert store.reserve_call(call_id=str(uuid.uuid4()), order_id=order_id, provider="openrouter",
                              key_id=key_a["key_id"], miner_hotkey=miner_a, public_fingerprint=key_a["public_fingerprint"],
                              reserve_microusd=2)["status"] == "key_capped"

    # Overshoot: the buyer's charge counts the full cost, the miner's
    # reward stops at the reservation, and the statement shows the excess.
    service.finish(order_id, {"status": "delivered", "leads_delivered": 10, "delivery_url": "https://f.example/x"})
    receipt = service.public_receipt(order["order_id"])["receipt"]
    assert receipt["rewards_microusd"] == {miner_a: reward(999_999, first["reserved_microusd"]),
                                           miner_b: reward(100, second["reserved_microusd"])}
    statement = service.statement(miner_a, "")
    row, = statement["calls"]
    assert row["cost_microusd"] == 999_999 and row["excess_microusd"] == 999_999 - first["reserved_microusd"]
    assert statement["keys"][0]["excess_microusd"] == row["excess_microusd"]
    revoke_all_keys(store)


@pytest.mark.parametrize("source", ["late_response", "readback"])
@pytest.mark.parametrize("actual_cost", [250_000, 750_000])
def test_daily_key_cap_uses_actual_late_cost_across_orders(store, owner, source, actual_cost):
    service, _, buyer, epoch, _ = live(store, owner, epoch_value=9001)
    miner = hotkey()
    accept_terms(service, miner)
    key = service.register_key(miner, dict(KEY, daily_cap_usd="1"))
    binding = dict(provider="openrouter", key_id=key["key_id"], miner_hotkey=miner,
                   public_fingerprint=key["public_fingerprint"])
    service.place_order(buyer, {"leads": 10, **ORDER})
    _, first = claim(service)
    call_id = str(uuid.uuid4())
    assert store.reserve_call(order_id=first, call_id=call_id, reserve_microusd=500_000,
                              **binding)["status"] == "reserved"
    if source == "readback":
        store.settle_call(call_id=call_id, cost_microusd=None, http_status=None, epoch=epoch(),
                          provider_request_id="gen-test-late-cost")
    else:
        backdate(owner, "lead_order_calls", "created_at", call_id, 301)
    service.finish(first, {"status": "failed"})

    service.place_order(buyer, {"leads": 10, **ORDER})
    _, second = claim(service)
    new_call = dict(order_id=second, call_id=str(uuid.uuid4()), **binding)
    assert store.reserve_call(**new_call, reserve_microusd=500_001)["status"] == "key_capped"
    if source == "readback":
        assert store.resolve_call(call_id=call_id, cost_microusd=actual_cost, epoch=epoch())["status"] == "resolved"
    else:
        assert store.settle_call(call_id=call_id, cost_microusd=actual_cost,
                                 http_status=200, epoch=epoch())["status"] == "late"
    remaining = USD - actual_cost
    assert store.reserve_call(**new_call, reserve_microusd=remaining + 1)["status"] == "key_capped"
    assert store.reserve_call(**new_call, reserve_microusd=remaining)["status"] == "reserved"
    store.settle_call(call_id=new_call["call_id"], cost_microusd=0, http_status=200, epoch=epoch())
    service.finish(second, {"status": "failed"})
    store.reward_late(epoch=epoch(), limit=10)  # Drain this case from the shared reward queue.


def test_timeout_is_read_back_on_the_miner_key_and_repaid_next_epoch_never_charged(store, owner):
    service, registrar, buyer, epoch, _ = live(store, owner, epoch_value=9100)
    miner, other = hotkey(), hotkey()
    accept_terms(service, miner)
    key = service.register_key(miner, KEY)
    registrar.secrets = {key["key_id"]: "m" * 40}
    order = service.place_order(buyer, {"leads": 10, **ORDER})
    token, order_id = claim(service)
    transport = FakeTransport([openrouter_ok("0.400000"),
                               ProviderTransportError("timed out", openrouter_generation_id="gen-late-1")])
    broker = broker_for(store, transport, registrar=registrar, epoch=epoch)
    assert broker.call(token, dear_chat()).status == 200
    assert broker.call(token, dear_chat()).status == 502
    paid, timed_out = store.calls(order_id)
    assert timed_out["status"] == "uncertain" and timed_out["cost_microusd"] is None
    service.finish(order_id, {"status": "delivered", "leads_delivered": 10, "delivery_url": "https://f.example/x"})
    public = service.public_receipt(order["order_id"])
    assert public["receipt"]["charge_microusd"] == 2 * USD
    assert public["receipt"]["rewards_microusd"] == {miner: reward(400_000, paid["reserved_microusd"])}
    assert "gen-late-1" not in json.dumps(public)  # the request id is for the miner's statement only

    # Reconcile reads the generation back once, on the same miner key, free.
    transport.responses.append(ProviderResponse(200, {"content-type": "application/json"},
                                                json.dumps({"data": {"id": "gen-late-1", "total_cost": 0.3}}).encode()))
    assert broker.resolve_uncertain() == 1
    lookup = transport.sent[-1]
    assert lookup["method"] == "GET" and lookup["url"].endswith("gen-late-1") and lookup["body"] == b""
    assert lookup["headers"]["authorization"] == "Bearer " + "m" * 40
    assert broker.resolve_uncertain() == 0  # resolved once; no second lookup
    assert len(transport.sent) == 3
    assert store.reward_late(epoch=9101, limit=10) == 1
    assert store.reward_late(epoch=9102, limit=10) == 0

    # Old behaviour: the cost was lost. Now it is owed in the next payout,
    # while the buyer's charge and the signed receipt stay as they were.
    late = reward(300_000, timed_out["reserved_microusd"])
    assert service.public_receipt(order["order_id"]) == public
    owed = [r for r in store.owed(through_epoch=10**9, mode="live") if r["order_id"] == order_id]
    assert {r["call_id"]: r["owed_microusd"] for r in owed} == {
        None: reward(400_000, paid["reserved_microusd"]), timed_out["call_id"]: late}
    statement = service.statement(miner, "0")
    rows = {r["call_id"]: r for r in statement["calls"]}
    assert rows[timed_out["call_id"]]["provider_request_id"] == "gen-late-1"
    assert rows[timed_out["call_id"]]["late_cost_microusd"] == 300_000
    assert rows[timed_out["call_id"]]["late_reward_microusd"] == late
    assert rows[timed_out["call_id"]]["receipt_hash"] == public["receipt"]["receipt_hash"]
    assert statement["rewards_owed_microusd"] == reward(400_000, paid["reserved_microusd"]) + late
    assert service.statement(other, "0")["calls"] == []

    doc = {"payout_hash": "sha256:" + "d" * 64, "epoch": 6_000_000}
    too_much = [{"order_id": order_id, "miner_hotkey": miner, "paid_microusd": late + 1, "call_id": timed_out["call_id"]}]
    with pytest.raises(ArenaStoreError, match="exceeds"):
        store.record_payout(epoch=6_000_000, mode="live", document=doc, payments=too_much)
    exact = [dict(too_much[0], paid_microusd=late)]
    assert store.record_payout(epoch=6_000_000, mode="live", document=doc, payments=exact)["status"] == "recorded"
    assert service.statement(miner, "0")["rewards_paid_microusd"] == late
    revoke_all_keys(store)


def test_late_cost_on_a_failed_order_is_repaid_only_within_its_charge(store, owner):
    service, registrar, buyer, epoch, _ = live(store, owner, epoch_value=9200)
    miner = hotkey()
    accept_terms(service, miner)
    key = service.register_key(miner, KEY)
    registrar.secrets = {key["key_id"]: "m" * 40}
    order = service.place_order(buyer, {"leads": 10, **ORDER})
    token, order_id = claim(service)
    broker_for(store, FakeTransport([openrouter_ok("0.400000")]), registrar=registrar, epoch=epoch).call(
        token, dear_chat())
    stuck = str(uuid.uuid4())
    assert store.reserve_call(call_id=stuck, order_id=order_id, provider="openrouter", key_id=key["key_id"],
                              miner_hotkey=miner, public_fingerprint=key["public_fingerprint"],
                              reserve_microusd=300_000)["status"] == "reserved"
    with owner.cursor() as cur:
        cur.execute("ALTER TABLE lead_order_claims DISABLE TRIGGER lead_order_claims_append_only")
        cur.execute("UPDATE lead_order_claims SET deadline=now()-interval '6 minutes' WHERE order_id=%s", (order_id,))
        cur.execute("ALTER TABLE lead_order_claims ENABLE TRIGGER lead_order_claims_append_only")
    assert service.claim() is None  # expired as failed, the stuck call closed uncertain
    assert store.settle_call(call_id=stuck, cost_microusd=250_000, http_status=200, epoch=epoch(),
                             provider_request_id="gen-stuck-1")["status"] == "late"
    assert store.reward_late(epoch=9201, limit=10) == 1
    receipt = service.public_receipt(order_id)["receipt"]
    charge, order_reward = receipt["charge_microusd"], receipt["rewards_microusd"][miner]
    # Charge >= 0.40 x rewarded spend must hold after the late reward too.
    room = charge * 1_250_000 // 400_000 - order_reward
    expected = max(0, min(reward(250_000, 300_000), room))
    assert expected < reward(250_000, 300_000)
    owed = {r["call_id"]: r["owed_microusd"] for r in store.owed(through_epoch=10**9, mode="live")
            if r["order_id"] == order_id}
    assert owed.get(stuck, 0) == expected  # a zero late reward is not listed as owed
    row = {r["call_id"]: r for r in service.statement(miner, "0")["calls"]}[stuck]
    assert row["late_cost_microusd"] == 250_000 and row["provider_request_id"] == "gen-stuck-1"
    assert row["late_reward_microusd"] == expected
    revoke_all_keys(store)


def test_statement_pages_by_call_sequence(store, owner):
    service, registrar, buyer, epoch, _ = live(store, owner, epoch_value=9300)
    miner = hotkey()
    accept_terms(service, miner)
    key = service.register_key(miner, KEY)
    registrar.secrets = {key["key_id"]: "m" * 40}
    service.place_order(buyer, {"leads": 10, **ORDER})
    token, order_id = claim(service)
    broker = broker_for(store, FakeTransport([openrouter_ok("0.000100")] * 3), registrar=registrar, epoch=epoch)
    for _ in range(3):
        broker.call(token, chat())
    service.finish(order_id, {"status": "failed"})
    first = store.statement(miner, since=0, limit=2)
    assert len(first["calls"]) == 2 and first["more"] is True
    rest = store.statement(miner, since=first["next_since"], limit=2)
    assert len(rest["calls"]) == 1 and rest["more"] is False
    assert first["keys"] == rest["keys"] and first["keys"][0]["calls"] == 3
    with pytest.raises(LeadOrderServiceError) as bad:
        service.statement(miner, "-1")
    assert bad.value.code == "since_invalid"
    revoke_all_keys(store)


def test_deepline_reserve_is_the_known_fixed_price_when_there_is_one(store):
    staff = hotkey()
    _, service, epoch = make(store, "staff", staff=[staff])
    service.place_order(staff, {"leads": 10, **ORDER})
    token, order_id = claim(service)
    broker = broker_for(store, FakeTransport([deepline_ok(0.01), deepline_ok(1)]), epoch=epoch)
    broker.call(token, deepline("harvestapi_get_job"))  # 0.01 credits = $0.001
    broker.call(token, deepline("harvestapi_search_jobs"))  # no fixed price: the flat reserve
    fixed, dynamic = store.calls(order_id)
    assert fixed["reserved_microusd"] == 1_000
    assert dynamic["reserved_microusd"] == POLICY.deepline_reserve_per_call_microusd
    service.finish(order_id, {"status": "failed"})


# -- second review: claim stays fast, read-back backs off, orphaned calls ---------


def backdate(owner, table, column, call_id, seconds):
    with owner.cursor() as cur:
        cur.execute("ALTER TABLE %s DISABLE TRIGGER %s_append_only" % (table, table))
        cur.execute("UPDATE %s SET %s=%s-make_interval(secs=>%%s) WHERE call_id=%%s" % (table, column, column),
                    (seconds, call_id))
        cur.execute("ALTER TABLE %s ENABLE TRIGGER %s_append_only" % (table, table))


def test_claim_never_runs_read_back_and_the_epoch_pass_does(store):
    staff = hotkey()
    config, _, epoch = make(store, "shadow", staff=[staff])
    passes = []
    service = LeadOrderService(config=config, policy=POLICY, policy_document=POLICY_DOC, store=store, signer=SIGNER,
                               current_epoch=epoch, resolver=lambda: passes.append("read-back"))
    service.place_order(staff, {"leads": 10, **ORDER})
    token, order_id = claim(service)
    assert passes == []  # old behaviour: claim ran a read-back pass of up to 15 s
    service.finish(order_id, {"status": "failed"})
    service.run_epoch(3_500_000)
    assert passes == ["read-back"]


def test_finish_closes_orphaned_open_calls_and_waits_only_for_young_ones(store, owner):
    staff = hotkey()
    _, service, epoch = make(store, "staff", staff=[staff])
    service.place_order(staff, {"leads": 10, **ORDER})
    token, order_id = claim(service)
    leadpoet = ProviderKey(key_id="leadpoet-openrouter", provider="openrouter", miner_hotkey=None,
                           registered_sequence=0, internal_fingerprint="sha256:" + "e" * 64)
    orphan = str(uuid.uuid4())  # reserved, then the broker restarted before settling
    assert store.reserve_call(call_id=orphan, order_id=order_id, provider="openrouter", key_id=leadpoet.key_id,
                              miner_hotkey=None, public_fingerprint=leadpoet.public_fingerprint,
                              reserve_microusd=300_000)["status"] == "reserved"
    delivered = {"status": "delivered", "leads_delivered": 5, "delivery_url": "https://f.example/x"}
    with pytest.raises(LeadOrderServiceError) as young:
        service.finish(order_id, delivered)
    assert young.value.code == "calls_open"
    backdate(owner, "lead_order_calls", "created_at", orphan, 301)
    # Old behaviour: 409 forever, then the sweep failed the order with nothing delivered.
    done = service.finish(order_id, delivered)
    assert done["charge_microusd"] == 5 * 200_000
    assert {c["call_id"]: c["status"] for c in store.calls(order_id)}[orphan] == "uncertain"


def test_broker_withholds_credential_echo_and_settles_reported_cost(store, owner):
    service, registrar, buyer, epoch, _ = live(store, owner, epoch_value=9401)
    miner = hotkey()
    accept_terms(service, miner)
    key = service.register_key(miner, KEY)
    fixture_key = "test-miner-credential"
    registrar.secrets[key["key_id"]] = fixture_key
    service.place_order(buyer, {"leads": 10, **ORDER})
    token, order_id = claim(service)
    response = ProviderResponse(200, {"content-type": "application/json"},
                                json.dumps({"text": fixture_key, "id": "gen-" + fixture_key,
                                            "usage": {"cost": "0.01"}}).encode())
    broker = broker_for(store, FakeTransport([response]), registrar=registrar, epoch=epoch)
    answer = broker.call(token, dear_chat())
    assert answer.status == 502 and fixture_key.encode() not in answer.body
    row, = store.statement(miner, since=0, limit=1)["calls"]
    assert row["cost_microusd"] == 10_000 and row["provider_request_id"] is None
    service.finish(order_id, {"status": "failed"})


def test_openrouter_settlement_uses_paid_result_after_catalog_expires(store):
    staff = hotkey()
    _, service, epoch = make(store, "staff", staff=[staff])
    service.place_order(staff, {"leads": 10, **ORDER})
    token, order_id = claim(service)
    reads = []

    def price_table():
        if reads:
            raise RuntimeError("catalog temporarily unavailable")
        reads.append(True)
        return PRICE_TABLE

    broker = LeadOrderBroker(store=store, policy=POLICY, transport=FakeTransport([openrouter_ok("0.01")]),
                             registrar=None, leadpoet_secrets={"openrouter": SECRET},
                             price_table=price_table, current_epoch=epoch)
    assert broker.call(token, dear_chat()).status == 200
    row, = store.calls(order_id)
    assert row["status"] == "settled" and row["cost_microusd"] == 10_000
    service.finish(order_id, {"status": "failed"})


@pytest.mark.parametrize("phase", ["dispatch", "readback"])
def test_transient_kms_failure_keeps_miner_key_eligible(store, owner, phase):
    service, registrar, buyer, epoch, _ = live(store, owner, epoch_value=9402)
    miner = hotkey()
    accept_terms(service, miner)
    key = service.register_key(miner, KEY)
    fixture_key = "test-runtime-key"
    registrar.secrets[key["key_id"]] = fixture_key
    service.place_order(buyer, {"leads": 10, **ORDER})
    token, order_id = claim(service)
    generation = "gen-transient-1"
    transport = FakeTransport([ProviderTransportError("timeout", openrouter_generation_id=generation)])
    broker = broker_for(store, transport, registrar=registrar, epoch=epoch)
    if phase == "readback":
        broker.call(token, dear_chat())
        service.finish(order_id, {"status": "failed"})

    class RecoveringKms:
        available = False

        def runtime_key(self, row, provider):
            if not self.available:
                raise CredentialError("credential_kms_unavailable", retryable=True)
            return fixture_key

    kms = RecoveringKms()
    broker = broker_for(store, transport, registrar=MinerKeyRegistrar(kms), epoch=epoch)
    if phase == "dispatch":
        with pytest.raises(LeadOrderBrokerError) as unavailable:
            broker.call(token, dear_chat())
        assert unavailable.value.status == 503
        assert store.exhausted_keys(order_id) == [] and not transport.sent
        kms.available = True
        transport.responses = [openrouter_ok("0.01")]
        assert broker.call(token, dear_chat()).status == 200
        assert store.calls(order_id)[0]["key_id"] == key["key_id"]
        service.finish(order_id, {"status": "failed"})
    else:
        assert broker.resolve_uncertain() == 0
        row, = store.calls(order_id)
        assert sql(owner, "SELECT outcome FROM lead_order_call_readbacks WHERE call_id=%s",
                   (row["call_id"],)) == [("retry",)]
        backdate(owner, "lead_order_call_readbacks", "attempted_at", row["call_id"], 601)
        kms.available = True
        record = {"data": {"id": generation, "total_cost": "0.01"}}
        transport.responses = [ProviderResponse(200, {}, json.dumps(record).encode())]
        assert broker.resolve_uncertain() == 1


def test_read_back_records_attempts_skips_terminal_and_backs_off(store, owner):
    service, registrar, buyer, epoch, _ = live(store, owner, epoch_value=9400)
    miner = hotkey()
    accept_terms(service, miner)
    key = service.register_key(miner, KEY)
    registrar.secrets = {key["key_id"]: "m" * 40}
    service.place_order(buyer, {"leads": 10, **ORDER})
    token, order_id = claim(service)
    transport = FakeTransport([ProviderTransportError("timed out", openrouter_generation_id="gen-old-1"),
                               ProviderTransportError("timed out", openrouter_generation_id="gen-new-1")])
    broker = broker_for(store, transport, registrar=registrar, epoch=epoch)
    broker.call(token, dear_chat())
    broker.call(token, dear_chat())
    old, new = store.calls(order_id)
    service.finish(order_id, {"status": "failed"})
    ids = lambda: [r["call_id"] for r in store.uncertain_calls(limit=10) if r["order_id"] == order_id]
    assert ids() == [old["call_id"], new["call_id"]]

    # The oldest call's key is refused: terminal, never tried again. The newer
    # one is not found yet: retry, then it waits out its back-off.
    unauthorized = ProviderResponse(401, {"content-type": "application/json"}, b'{"error":"no"}')
    not_yet = ProviderResponse(404, {"content-type": "application/json"}, b'{"error":"not found"}')
    transport.responses += [unauthorized, not_yet]
    broker.resolve_uncertain(limit=2)
    assert ids() == []  # old behaviour: the same oldest calls on every pass
    outcomes = sql(owner, "SELECT call_id::text,attempt,outcome FROM lead_order_call_readbacks WHERE call_id IN (%s,%s) "
                          "ORDER BY call_id,attempt", (old["call_id"], new["call_id"]))
    assert sorted(outcomes) == sorted([(old["call_id"], 1, "terminal"), (new["call_id"], 1, "retry")])
    backdate(owner, "lead_order_call_readbacks", "attempted_at", new["call_id"], 601)
    assert ids() == [new["call_id"]]
    # Six tries at most, however long it waits.
    for attempt in range(2, 7):
        assert store.record_readback(call_id=new["call_id"], outcome="retry")["attempt"] == attempt
    backdate(owner, "lead_order_call_readbacks", "attempted_at", new["call_id"], 10**6)
    assert ids() == []
    assert store.record_readback(call_id=str(uuid.uuid4()), outcome="retry")["status"] == "not_recordable"
    revoke_all_keys(store)
