"""Lead orders on a disposable PostgreSQL: SQL rules, service, broker, receipts.

Runs scripts/703 on top of the Lab Arena base (179) through the same harness
the Arena tests use (Docker in CI, ``LAB_ARENA_PG_LOCAL=1`` locally). Every
order a test places is claimed in the same test, because the claim queue is
shared by the module's database.
"""

from __future__ import annotations

import hashlib
import json
import random
import uuid
from fractions import Fraction

import pytest

from lab_arena import signing
from lab_arena.broker import ProviderResponse, ProviderTransportError, price_table_from_models_response
from lab_arena.lead_orders import money, receipts, suppressions
from lab_arena.lead_orders.auth import BuyerIdentity, order_token_hash
from lab_arena.lead_orders.broker import BrokerRequest, LeadOrderBroker, LeadOrderBrokerError
from lab_arena.lead_orders.config import LeadOrdersConfig
from lab_arena.lead_orders.gamma import GammaReading
from lab_arena.lead_orders.keys import ProviderKey
from lab_arena.lead_orders.policy import DEFAULT_POLICY_BODY, sign_policy, load_policy
from lab_arena.lead_orders.service import LeadOrderService, LeadOrderServiceError
from lab_arena.lead_orders.store import LeadOrderStore
from lab_arena.store import ArenaStoreError, PsycopgTransport
from tests.lab_arena.lab_arena_pg_harness import database_with_lab_arena_migration

MIGRATIONS = ("179-lab-arena-v1.sql", "703-lead-orders.sql")
USD = 1_000_000
B58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
MODEL = "openai/gpt-test"
SECRET = "x" * 40
SIGNER = signing.LocalSigner.generate()
POLICY_DOC = sign_policy(SIGNER, DEFAULT_POLICY_BODY)
POLICY = load_policy(POLICY_DOC, public_key_der=SIGNER.public_key_der, expected_public_key_hash=SIGNER.public_key_hash)
PRICE_TABLE = price_table_from_models_response(
    {"data": [{"id": MODEL, "pricing": {"prompt": "0.000001", "completion": "0.000002"}}]},
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


def chat(max_tokens=100):
    return BrokerRequest("openrouter", "POST", "api/v1/chat/completions", {},
                         json.dumps({"model": MODEL, "messages": [{"role": "user", "content": "hi"}],
                                     "max_tokens": max_tokens}).encode(), "application/json")


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
        assert len(tables) == 20 and all(tables.values())
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


def test_shadow_ranks_miner_keys_fails_over_and_owes_rewards(store):
    staff, miner_a, miner_b = hotkey(), hotkey(), hotkey()
    registrar = FakeRegistrar({})
    epoch = Epoch(2000)
    _, service, _ = make(store, "shadow", staff=[staff], registrar=registrar, epoch=epoch)
    accept_terms(service, miner_a)
    accept_terms(service, miner_b)
    key_a = service.register_key(miner_a, {"provider": "openrouter", "credentials": {}})
    key_b = service.register_key(miner_b, {"provider": "openrouter", "credentials": {}})
    with pytest.raises(LeadOrderServiceError) as twice:
        service.register_key(miner_a, {"provider": "openrouter", "credentials": {}})
    assert twice.value.code == "active_key_exists"
    registrar.secrets = {key_a["key_id"]: "a" * 40, key_b["key_id"]: "b" * 40}

    order = service.place_order(staff, {"leads": 10, **ORDER})
    token, order_id = claim(service)
    spent = ProviderResponse(402, {"content-type": "application/json"},
                             json.dumps({"error": {"code": 402, "message": "Insufficient credits"}}).encode())
    transport = FakeTransport([spent, openrouter_ok("0.800000"), openrouter_ok("0.200000")])
    broker = broker_for(store, transport, registrar=registrar, epoch=epoch)
    assert broker.call(token, chat()).status == 200
    # First key is A (registered first, no spend yet); its 402 moves the
    # order to B; B serves the rest of the order.
    assert [s["headers"]["authorization"][-1] for s in transport.sent] == ["a", "b"]
    assert broker.call(token, chat()).status == 200
    assert transport.sent[-1]["headers"]["authorization"].endswith("b")
    calls = store.calls(order["order_id"])
    assert [c["cost_microusd"] for c in calls] == [0, 800_000, 200_000]
    assert store.exhausted_keys(order["order_id"]) == [key_a["key_id"]]
    service.finish(order_id, {"status": "delivered", "leads_delivered": 10, "delivery_url": "https://f.example/x"})
    receipt = service.public_receipt(order["order_id"])["receipt"]
    assert receipt["rewards_microusd"] == {miner_b: 1_250_000}  # $1.00 x 1.25
    assert receipt["charge_microusd"] >= money.minimum_charge_microusd(POLICY, 1_000_000)

    # B now ranks first by settled spend in the window.
    order2 = service.place_order(staff, {"leads": 10, **ORDER})
    token2, order_id2 = claim(service)
    transport2 = FakeTransport([openrouter_ok("0.010000")])
    broker_for(store, transport2, registrar=registrar, epoch=epoch).call(token2, chat())
    assert transport2.sent[0]["headers"]["authorization"].endswith("b")
    service.finish(order_id2, {"status": "failed"})

    # Shadow payout: published, never recorded as paid, and with no emission
    # price it shows what is owed with no payments.
    doc = service.run_epoch(2000)
    assert doc["mode"] == "shadow" and doc["emission_price_available"] is False and doc["payments"] == []
    assert doc["owed_microusd"] >= 1_250_000
    assert service.public_epoch(2000)["payout_hash"] == doc["payout_hash"]
    shadow_owed = store.owed(through_epoch=2000, mode="shadow")
    assert any(row["order_id"] == order["order_id"] for row in shadow_owed)
    assert all(row["paid_microusd"] == 0 for row in shadow_owed)
    # Live never sees, and so never pays, a shadow order's reward.
    assert not [row for row in store.owed(through_epoch=10**9, mode="live") if row["order_id"] == order["order_id"]]
    with pytest.raises(ArenaStoreError, match="only live"):
        store.record_payout(epoch=4_000_000, mode="live", document={"payout_hash": "sha256:" + "c" * 64},
                            payments=[{"order_id": order["order_id"], "miner_hotkey": miner_b, "paid_microusd": 1}])
    assert order2["status"] == "queued"


def sql(owner, query, params):
    with owner.cursor() as cur:
        cur.execute(query, params)
        return [tuple(row) for row in cur.fetchall()] if cur.description else []


def signed_identity(miner, body=b'{"terms_version":"miner-terms-2026-11-20"}'):
    return BuyerIdentity(hotkey=miner, signed_at=1_763_500_000, signature="ab" * 64,
                         body_sha256=hashlib.sha256(body).hexdigest())


def accept_terms(service, miner):
    return service.accept_miner_terms(signed_identity(miner), {"terms_version": POLICY.miner_terms_version})


def revoke_all_keys(store):
    for provider in ("openrouter", "deepline"):
        for row in store.keys(provider):
            store.revoke_key(row["key_id"], reason="test reset")


def live_order_with_miner_reward(store, owner, *, epoch_value=3000):
    """A finished live order whose spend went through a miner's key."""
    revoke_all_keys(store)
    gamma, buyer, miner = FakeGamma(), hotkey(), hotkey()
    registrar = FakeRegistrar({})
    _, service, epoch = make(store, "live", gamma=gamma, registrar=registrar, epoch=Epoch(epoch_value))
    with pytest.raises(LeadOrderServiceError):
        service.register_buyer(buyer, {"email": "b@acme.io", "company": "Acme", "country": "US", "terms_version": "t1"})
    confirm(owner, buyer)
    gamma.units[buyer] = 10 * 10**9
    accept_terms(service, miner)
    key = service.register_key(miner, {"provider": "openrouter", "credentials": {}})
    registrar.secrets = {key["key_id"]: "m" * 40}
    order = service.place_order(buyer, {"leads": 10, **ORDER})
    token, order_id = claim(service)
    transport = FakeTransport([openrouter_ok("0.400000")])
    broker_for(store, transport, registrar=registrar, epoch=epoch).call(token, chat())
    assert transport.sent[0]["headers"]["authorization"] == "Bearer " + "m" * 40
    service.finish(order_id, {"status": "delivered", "leads_delivered": 10, "delivery_url": "https://f.example/x"})
    revoke_all_keys(store)
    return order, miner


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


def test_rewards_and_receipt_use_the_orders_policy_version_not_the_current_one(store):
    # v1: ratio 0.40, premium 0.25. v2 raises the premium to 0.45. An order
    # placed under v1 and settled after the switch must still pay x1.25.
    revoke_all_keys(store)
    staff, miner = hotkey(), hotkey()
    registrar = FakeRegistrar({})
    epoch = Epoch(6000)
    _, service_v1, _ = make(store, "shadow", staff=[staff], registrar=registrar, epoch=epoch)
    accept_terms(service_v1, miner)
    key = service_v1.register_key(miner, {"provider": "openrouter", "credentials": {}})
    registrar.secrets = {key["key_id"]: "v" * 40}
    order = service_v1.place_order(staff, {"leads": 10, **ORDER})
    token, order_id = claim(service_v1)
    broker_for(store, FakeTransport([openrouter_ok("0.400000")]), registrar=registrar, epoch=epoch).call(token, chat())

    v2_doc = sign_policy(SIGNER, dict(DEFAULT_POLICY_BODY, version=2, price_per_lead_usd="0.30",
                                      min_charge_ratio="0.60", miner_premium="0.45"))
    v2 = load_policy(v2_doc, public_key_der=SIGNER.public_key_der, expected_public_key_hash=SIGNER.public_key_hash)
    config = LeadOrdersConfig(mode="shadow", staff_hotkeys=frozenset([staff]), runner_secret="r" * 40,
                              policy_path="/x", terms_version="t1", public_base_url="https://gw.example")
    service_v2 = LeadOrderService(config=config, policy=v2, policy_document=v2_doc, store=store, signer=SIGNER,
                                  current_epoch=epoch, registrar=registrar)
    service_v2.finish(order_id, {"status": "delivered", "leads_delivered": 10, "delivery_url": "https://f.example/x"})
    receipt = service_v2.public_receipt(order["order_id"])["receipt"]
    assert receipt["rewards_microusd"] == {miner: 500_000}  # $0.40 x 1.25, not x 1.45
    assert receipt["policy_version"] == 1 and receipt["policy_hash"] == POLICY.policy_hash
    assert receipt["charge_microusd"] == 2 * USD  # v1 price, from the order's stored terms
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
    key = shadow_service.register_key(miner, {"provider": "openrouter", "credentials": {}})
    registrar.secrets = {key["key_id"]: "z" * 40}
    transport = FakeTransport([openrouter_ok("0.100000")])
    broker_for(store, transport, registrar=registrar, epoch=epoch).call(token, chat())
    assert transport.sent[0]["headers"]["authorization"] == "Bearer " + SECRET  # Leadpoet's key
    assert {c["miner_hotkey"] for c in store.calls(order_id)} == {None}
    shadow_service.finish(order_id, {"status": "failed"})
    revoke_all_keys(store)


def test_open_calls_settle_uncertain_at_finish_and_late_answers_never_change_the_receipt(store):
    staff = hotkey()
    _, service, epoch = make(store, "staff", staff=[staff])
    order = service.place_order(staff, {"leads": 10, **ORDER})
    token, order_id = claim(service)
    broker_for(store, FakeTransport([openrouter_ok("0.100000")]), epoch=epoch).call(token, chat())
    leadpoet = ProviderKey(key_id="leadpoet-openrouter", provider="openrouter", miner_hotkey=None,
                           registered_sequence=0, internal_fingerprint="sha256:" + "e" * 64)
    in_flight = str(uuid.uuid4())  # reserved, provider has not answered yet
    assert store.reserve_call(call_id=in_flight, order_id=order_id, provider="openrouter", key_id=leadpoet.key_id,
                              miner_hotkey=None, public_fingerprint=leadpoet.public_fingerprint,
                              reserve_microusd=300_000)["status"] == "reserved"
    with pytest.raises(LeadOrderServiceError) as too_many:
        service.finish(order_id, {"status": "delivered", "leads_delivered": 11, "delivery_url": "https://f.example/x"})
    assert too_many.value.status == 422 and too_many.value.code == "leads_delivered_invalid"

    done = service.finish(order_id, {"status": "delivered", "leads_delivered": 1, "delivery_url": "https://f.example/x"})
    assert done["charge_microusd"] == 200_000  # 1 lead; the open call is never charged
    statuses = {c["call_id"]: c["status"] for c in store.calls(order_id)}
    assert statuses[in_flight] == "uncertain" and "open" not in statuses.values()
    public = service.public_receipt(order["order_id"])

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
    revoke_all_keys(store)
    staff, miner, late = hotkey(), hotkey(), hotkey()
    registrar = FakeRegistrar({})
    epoch = Epoch(7000)
    _, service, _ = make(store, "shadow", staff=[staff], registrar=registrar, epoch=epoch)
    with pytest.raises(LeadOrderServiceError) as unsigned:
        service.register_key(miner, {"provider": "openrouter", "credentials": {}})
    assert (unsigned.value.status, unsigned.value.code) == (403, "miner_terms_not_accepted")
    with pytest.raises(LeadOrderServiceError) as old:
        service.accept_miner_terms(signed_identity(miner), {"terms_version": "miner-terms-0"})
    assert (old.value.status, old.value.code) == (422, "terms_version_not_current")
    first = accept_terms(service, miner)
    assert first["status"] == "accepted" and first["terms_version"] == POLICY.miner_terms_version
    assert accept_terms(service, miner)["status"] == "exists"
    stored = sql(owner, "SELECT signed_method,signed_path,signed_at,body_sha256,signature "
                        "FROM public.lead_order_miner_terms WHERE miner_hotkey=%s", (miner,))
    assert stored == [("POST", "/lead-orders/miner-terms", 1_763_500_000,
                       hashlib.sha256(b'{"terms_version":"miner-terms-2026-11-20"}').hexdigest(), "ab" * 64)]
    key = service.register_key(miner, {"provider": "openrouter", "credentials": {}})
    registrar.secrets = {key["key_id"]: "t" * 40}
    # The terms move to version 2: the key stays registered but is skipped
    # until its miner signs again, so the order runs on Leadpoet's key.
    v2_doc = sign_policy(SIGNER, dict(DEFAULT_POLICY_BODY, version=3, miner_terms_version="miner-terms-2"))
    v2 = load_policy(v2_doc, public_key_der=SIGNER.public_key_der, expected_public_key_hash=SIGNER.public_key_hash)
    service.place_order(staff, {"leads": 10, **ORDER})
    token, order_id = claim(service)
    transport = FakeTransport([openrouter_ok("0.100000"), openrouter_ok("0.100000")])
    broker_v2 = LeadOrderBroker(store=store, policy=v2, transport=transport, registrar=registrar,
                                leadpoet_secrets={"openrouter": SECRET, "deepline": SECRET},
                                price_table=lambda: PRICE_TABLE, current_epoch=epoch)
    broker_v2.call(token, chat())
    assert transport.sent[0]["headers"]["authorization"] == "Bearer " + SECRET
    broker_for(store, transport, registrar=registrar, epoch=epoch).call(token, chat())
    assert transport.sent[1]["headers"]["authorization"] == "Bearer " + "t" * 40  # current terms: ranked
    service.finish(order_id, {"status": "failed"})
    with pytest.raises(Exception):
        sql(owner, "UPDATE public.lead_order_miner_terms SET terms_version='x' WHERE miner_hotkey=%s", (miner,))
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
