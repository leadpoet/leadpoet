"""Config, Gamma reader, keys, auth, receipts and payout documents."""

from __future__ import annotations

import json
from fractions import Fraction

import pytest

from lab_arena import signing
from lab_arena.broker import ProviderResponse
from lab_arena.chain import BlockRef
from lab_arena.credentials import CredentialError
from lab_arena.lead_orders import auth, keys, money, receipts, rewards
from lab_arena.lead_orders.config import LeadOrdersConfigError, load_config
from lab_arena.lead_orders.gamma import GammaBalanceReader, GammaReadError
from lab_arena.lead_orders.policy import DEFAULT_POLICY_BODY, unsigned_policy

POLICY = unsigned_policy()
ALICE = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
BOB = "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"
# A valid format-40 AccountId32, a mistyped format-42 checksum, and numeric text.
WRONG_NETWORK = "4zVPe3pyrYm2nenze7ns4xVoGycNdggAAbEmCszvXGDw2CKz"
INVALID_ADDRESSES = [ALICE[:-1] + "Z", WRONG_NETWORK, "1" * 46]
BASE_ENV = {
    "LEADPOET_LEAD_ORDERS_RUNNER_SECRET": "s" * 32,
    "LEADPOET_LEAD_ORDERS_POLICY_PATH": "/etc/policy.json",
    "LEADPOET_LEAD_ORDERS_TERMS_VERSION": "2026-11",
    "LEADPOET_LEAD_ORDERS_PUBLIC_BASE_URL": "https://gw.example/",
    "LEADPOET_LEAD_ORDERS_STAFF_HOTKEYS": ALICE,
}


# -- config ---------------------------------------------------------------------


def test_config_is_off_by_default_and_ignores_other_settings():
    for env in ({}, {"LEADPOET_LEAD_ORDERS_MODE": "off", "LEADPOET_LEAD_ORDERS_RUNNER_SECRET": "x",
                     "LEADPOET_LEAD_ORDERS_GAMMA_SERVICE": "7"}):
        config = load_config(env)
        assert config.mode == "off" and not config.enabled and not config.weights_rewards
        assert not config.may_order(ALICE) and not config.runner_secret_matches("x")


def test_config_requires_every_setting_outside_off():
    assert load_config(dict(BASE_ENV, LEADPOET_LEAD_ORDERS_MODE="staff")).public_base_url == "https://gw.example"
    for name in ("LEADPOET_LEAD_ORDERS_RUNNER_SECRET", "LEADPOET_LEAD_ORDERS_POLICY_PATH",
                 "LEADPOET_LEAD_ORDERS_TERMS_VERSION", "LEADPOET_LEAD_ORDERS_PUBLIC_BASE_URL",
                 "LEADPOET_LEAD_ORDERS_STAFF_HOTKEYS"):
        env = dict(BASE_ENV, LEADPOET_LEAD_ORDERS_MODE="shadow")
        env.pop(name)
        with pytest.raises(LeadOrdersConfigError):
            load_config(env)
    with pytest.raises(LeadOrdersConfigError):
        load_config(dict(BASE_ENV, LEADPOET_LEAD_ORDERS_MODE="live"))  # no Gamma service id
    with pytest.raises(LeadOrdersConfigError):
        load_config(dict(BASE_ENV, LEADPOET_LEAD_ORDERS_MODE="loud"))
    live = load_config(dict(BASE_ENV, LEADPOET_LEAD_ORDERS_MODE="live", LEADPOET_LEAD_ORDERS_GAMMA_SERVICE=ALICE,
                            LEADPOET_LEAD_ORDERS_BLOCKED_HOTKEYS=BOB))
    assert live.may_order(ALICE) is True and live.may_order(BOB) is False and live.weights_rewards
    staff = load_config(dict(BASE_ENV, LEADPOET_LEAD_ORDERS_MODE="staff"))
    assert staff.may_order(ALICE) and not staff.may_order(BOB) and not staff.uses_miner_keys
    assert staff.runner_secret_matches("s" * 32) and not staff.runner_secret_matches("s" * 31)
    assert "s" * 32 not in repr(staff)


# -- Gamma reader -----------------------------------------------------------------


@pytest.mark.parametrize("service_id", ["0", "7", "4294967295", *INVALID_ADDRESSES])
def test_gamma_service_config_rejects_invalid_accounts(service_id):
    with pytest.raises(LeadOrdersConfigError, match="SS58"):
        load_config(dict(BASE_ENV, LEADPOET_LEAD_ORDERS_MODE="live",
                         LEADPOET_LEAD_ORDERS_GAMMA_SERVICE=service_id))


@pytest.mark.parametrize("service_id", ["0", "7", "4294967295", *INVALID_ADDRESSES])
def test_gamma_service_reader_rejects_invalid_accounts_before_chain_access(service_id):
    client = FakeClient()
    with pytest.raises(GammaReadError, match="SS58"):
        GammaBalanceReader(FakeChain(client), service_id=service_id)
    assert client.calls == []


@pytest.mark.parametrize("setting", ["LEADPOET_LEAD_ORDERS_STAFF_HOTKEYS", "LEADPOET_LEAD_ORDERS_BLOCKED_HOTKEYS"])
@pytest.mark.parametrize("address", INVALID_ADDRESSES)
def test_config_validates_each_configured_hotkey(setting, address):
    env = dict(BASE_ENV, LEADPOET_LEAD_ORDERS_MODE="staff")
    env[setting] = ALICE + "," + address
    with pytest.raises(LeadOrdersConfigError, match="invalid hotkey"):
        load_config(env)


@pytest.mark.parametrize("address", INVALID_ADDRESSES)
def test_gamma_reader_rejects_invalid_buyer_before_chain_access(address):
    client = FakeClient()
    reader = GammaBalanceReader(FakeChain(client), service_id=BOB)
    with pytest.raises(GammaReadError, match="hotkey is invalid"):
        reader.read(address)
    assert client.calls == []


class StorageFunctionNotFound(Exception):
    pass


class FakeClient:
    def __init__(self, value=None, error=None):
        self.value, self.error, self.calls = value, error, []

    def query(self, module, storage, params, block_hash=None):
        self.calls.append((module, storage, list(params), block_hash))
        if self.error:
            raise self.error
        return self.value


class FakeChain:
    def __init__(self, client):
        self.client = client
        self.config = type("C", (), {"netuid": 71})()

    def finalized_head(self):
        return BlockRef(500, "0x" + "ab" * 32)


def test_gamma_reader_reads_finalized_spent_counter():
    client = FakeClient(value=type("V", (), {"value": 2_500_000_000})())
    reading = GammaBalanceReader(FakeChain(client), service_id=BOB).read(ALICE)
    assert reading.available and reading.spent_units == 2_500_000_000 and reading.block_number == 500
    assert client.calls == [("SubtensorModule", "GammaSpent", [71, BOB, ALICE], "0x" + "ab" * 32)]
    assert money.gamma_units_to_microusd(reading.spent_units) == 2_500_000
    assert GammaBalanceReader(FakeChain(FakeClient(value=None)), service_id=BOB).read(ALICE).spent_units == 0


def test_gamma_missing_storage_reads_zero_unavailable_and_other_errors_raise():
    missing = GammaBalanceReader(FakeChain(FakeClient(error=StorageFunctionNotFound("GammaSpent"))), service_id=BOB)
    reading = missing.read(ALICE)
    assert reading.available is False and reading.spent_units == 0
    wrapped = RuntimeError("query failed")
    wrapped.__cause__ = ValueError("Storage function SubtensorModule.GammaSpent not found")
    assert GammaBalanceReader(FakeChain(FakeClient(error=wrapped)), service_id=BOB).read(ALICE).available is False
    with pytest.raises(GammaReadError):
        GammaBalanceReader(FakeChain(FakeClient(error=TimeoutError("slow"))), service_id=BOB).read(ALICE)
    for bad in (-1, 1 << 128, "12", True):
        with pytest.raises(GammaReadError):
            GammaBalanceReader(FakeChain(FakeClient(value=bad)), service_id=BOB).read(ALICE)
    with pytest.raises(GammaReadError):
        GammaBalanceReader(FakeChain(FakeClient()), service_id="not a service")


# -- keys -----------------------------------------------------------------------


def key(key_id, hotkey, seq):
    return keys.ProviderKey(key_id, "openrouter", hotkey, seq, "sha256:" + key_id.ljust(64, "0")[:64])


def test_rank_keys_by_settled_spend_with_leadpoet_last_and_exhausted_skipped():
    leadpoet = keys.leadpoet_keys("o" * 40, "")["openrouter"]
    miners = [key("aa", ALICE, 1), key("bb", BOB, 2), key("cc", BOB, 3)]
    ranked = keys.rank_keys("openrouter", miners, leadpoet, settled_spend_by_key={"bb": 50, "cc": 50, "aa": 10},
                            use_miner_keys=True)
    assert [k.key_id for k in ranked] == ["bb", "cc", "aa", "leadpoet-openrouter"]
    ranked = keys.rank_keys("openrouter", miners, leadpoet, settled_spend_by_key={}, use_miner_keys=True,
                            exhausted=["aa", "leadpoet-openrouter"])
    assert [k.key_id for k in ranked] == ["bb", "cc"]
    assert keys.rank_keys("openrouter", miners, leadpoet, settled_spend_by_key={}, use_miner_keys=False) == [leadpoet]
    assert keys.rank_keys("deepline", miners, leadpoet, settled_spend_by_key={}, use_miner_keys=True) == []
    assert leadpoet.public_fingerprint != leadpoet.internal_fingerprint and len(leadpoet.public_fingerprint) == 39


@pytest.mark.parametrize("status,headers,body,reason", [
    (401, {}, b"", "unauthorized"),
    (403, {}, b"", "forbidden"),
    (402, {}, b"{}", None),  # set below: spent or payment_required, both fail over
    (429, {}, b"", "rate_limited_no_retry"),
    (429, {"Retry-After": "3"}, b"", None),
    (500, {}, b"", None),
    (200, {}, b"", None),
])
def test_failover_reasons(status, headers, body, reason):
    got = keys.failover_reason("openrouter", ProviderResponse(status, headers, body))
    if status == 402:
        assert got in ("spent_limit", "payment_required")
    else:
        assert got == reason


class FakeManager:
    def __init__(self):
        self.encrypted = []

    def _validate_openrouter(self, runtime, management):
        if runtime.endswith("bad"):
            raise CredentialError("openrouter_key_rejected")

    def _validate_deepline(self, secret):
        pass

    def _encrypt(self, secret, *, submission_id, miner_hotkey, provider):
        self.encrypted.append((submission_id, miner_hotkey, provider))
        return "Y2lwaGVy"

    def runtime_key(self, row, provider):
        assert row["submission_id"].startswith("lead-order:")
        return "decrypted"


def test_registrar_binds_ciphertext_to_a_lead_order_submission_id():
    manager = FakeManager()
    registrar = keys.MinerKeyRegistrar(manager, new_key_id=lambda: "k1")
    row = registrar.register(miner_hotkey=ALICE, provider="deepline", credentials={"deepline_api_key": "d" * 40})
    assert row["submission_id"] == "lead-order:k1" and manager.encrypted == [("lead-order:k1", ALICE, "deepline")]
    assert "d" * 40 not in json.dumps(row)
    stored = keys.ProviderKey("k1", "deepline", ALICE, 1, row["internal_fingerprint"], row["ciphertext_b64"])
    assert registrar.decrypt(stored) == "decrypted"
    with pytest.raises(keys.LeadOrderKeyError):
        registrar.register(miner_hotkey=ALICE, provider="scrapingdog", credentials={})
    with pytest.raises(keys.LeadOrderKeyError):
        registrar.register(miner_hotkey=ALICE, provider="deepline", credentials={"other": "x"})


# -- auth -----------------------------------------------------------------------


@pytest.mark.parametrize("address", INVALID_ADDRESSES)
def test_invalid_buyer_hotkey_is_rejected_before_signature_verification(address):
    with pytest.raises(auth.AuthError, match="hotkey_invalid"):
        auth.verify_buyer_request(
            method="GET", host="gw.example", path="/lead-orders/balance", body=b"", hotkey=address,
            timestamp="1000", signature="00" * 64, verifier=lambda *_: pytest.fail("unexpected verifier call"),
            use_nonce=lambda *_: pytest.fail("unexpected nonce use"), now=lambda: 1000,
        )


def test_buyer_signature_window_and_replay():
    from bittensor_wallet import Keypair
    from lab_arena.chain import wallet_signature_verifier

    pair = Keypair.create_from_uri("//Alice")
    assert pair.ss58_address == ALICE
    used = set()
    nonce = lambda h, s, t: (h, s) not in used and not used.add((h, s))
    body = b'{"leads":10}'
    message = auth.signed_message("POST", "gw.example", "/lead-orders", "1000", body)
    signature = "0x" + pair.sign(message.encode()).hex()
    kwargs = dict(method="POST", host="gw.example", path="/lead-orders", body=body, hotkey=ALICE, timestamp="1000",
                  signature=signature, verifier=wallet_signature_verifier, use_nonce=nonce, now=lambda: 1030)
    assert auth.verify_buyer_request(**kwargs).hotkey == ALICE
    with pytest.raises(auth.AuthError, match="signature_replayed"):
        auth.verify_buyer_request(**kwargs)
    for change, code in (({"now": lambda: 1061}, "timestamp_outside_window"),
                         ({"body": b'{"leads":11}'}, "signature_invalid"),
                         ({"path": "/lead-orders/keys"}, "signature_invalid"),
                         ({"host": "other.example"}, "signature_invalid"),
                         ({"host": "gw.example:8443"}, "signature_invalid"),
                         ({"host": ""}, "host_unconfigured"),
                         ({"hotkey": BOB}, "signature_invalid"),
                         ({"signature": "zz"}, "signature_invalid"),
                         ({"timestamp": "abc"}, "timestamp_invalid"),
                         ({"hotkey": "nope"}, "hotkey_invalid")):
        with pytest.raises(auth.AuthError) as refused:
            auth.verify_buyer_request(**dict(kwargs, **change))
        assert refused.value.code == code and refused.value.status == 401
    assert len(used) == 1  # failed attempts never consume a nonce


# -- receipts and payouts -------------------------------------------------------


def rows(n):
    return [{"sequence": i, "call_id": "c%d" % i, "provider": "openrouter", "miner_hotkey": ALICE,
             "public_fingerprint": "sha256:" + "0" * 32, "reserved_microusd": 10, "status": "settled",
             "cost_microusd": i, "key_id": "secret-ish-not-published"} for i in range(1, n + 1)]


def test_merkle_root_is_order_independent_input_and_detects_any_change():
    roots = {receipts.merkle_root([receipts.call_leaf(r) for r in rows(n)]) for n in range(0, 6)}
    assert len(roots) == 6
    base = [receipts.call_leaf(r) for r in rows(5)]
    changed = [dict(leaf) for leaf in base]
    changed[3]["cost_microusd"] += 1
    assert receipts.merkle_root(base) != receipts.merkle_root(changed)
    assert "key_id" not in base[0]


def test_receipt_signature_and_rows_verify_and_tampering_fails():
    signer = signing.LocalSigner.generate()
    order = {"order_id": "o", "hotkey": ALICE, "mode": "shadow", "policy_version": 1, "status": "delivered",
             "leads_requested": 10, "leads_delivered": 10, "hold_microusd": 2_000_000,
             "spend_ceiling_microusd": 5_000_000, "spend_microusd": 15, "charge_microusd": 2_000_000,
             "settled_epoch": 9}
    receipt = receipts.sign_receipt(signer, receipts.build_receipt(order=order, calls=rows(5), rewards={ALICE: 18},
                                                                   policy_hash="sha256:" + "1" * 64))
    check = dict(public_key_der=signer.public_key_der, expected_public_key_hash=signer.public_key_hash)
    assert receipts.verify_receipt(receipt, calls=list(reversed(rows(5))), **check) == receipt["receipt_hash"]
    # Per-key totals are published; call order and internal key ids are not.
    assert receipt["keys"] == [{"provider": "openrouter", "miner_hotkey": ALICE, "public_fingerprint": "sha256:" + "0" * 32,
                                "calls": 5, "settled_microusd": 15, "uncertain_calls": 0}]
    assert "sequence" not in json.dumps(receipt) and "key_id" not in json.dumps(receipt)
    with pytest.raises(receipts.ReceiptError):
        receipts.verify_receipt(receipt, calls=rows(4), **check)
    tampered = dict(receipt, charge_microusd=1)
    with pytest.raises(Exception):
        receipts.verify_receipt(tampered, **check)
    with pytest.raises(receipts.ReceiptError):
        receipts.build_receipt(order=dict(order, status="running"), calls=[], rewards={}, policy_hash="x")


def test_payout_document_shadow_without_price_and_live_with_price():
    entries = [money.OwedEntry("o1", ALICE, 1, 1, 600_000), money.OwedEntry("o2", BOB, 2, 2, 900_000)]
    shadow = rewards.epoch_payout_document(POLICY, epoch=3, mode="shadow", entries=entries, valuation=None)
    assert shadow["payments"] == [] and shadow["owed_microusd"] == 1_500_000 and rewards.lane_fractions(shadow) == {}
    with pytest.raises(rewards.EmissionPriceUnavailable):
        rewards.epoch_payout_document(POLICY, epoch=3, mode="live", entries=entries, valuation=None)
    with pytest.raises(rewards.EmissionPriceUnavailable):
        rewards.UnavailableEmissionPrice().valuation(3)
    # $4 of miner emission under a 30% policy cap = $1.20: o1 in full, $0.60 of o2, $0.30 carried.
    valuation = {"emission_microusd": 4_000_000, "block": 50}
    capped = unsigned_policy(dict(DEFAULT_POLICY_BODY, lane_emission_cap="0.30"))
    live = rewards.epoch_payout_document(capped, epoch=3, mode="live", entries=entries, valuation=valuation)
    # The default policy leaves the cap to the split: the whole epoch.
    assert rewards.epoch_payout_document(POLICY, epoch=3, mode="live", entries=entries,
                                         valuation=valuation)["carry_forward_microusd"] == 0
    assert live["cap_microusd"] == 1_200_000 and live["carry_forward_microusd"] == 300_000
    assert [p["paid_microusd"] for p in live["payments"]] == [600_000, 600_000]
    assert rewards.lane_fractions(live) == {ALICE: Fraction(3, 20), BOB: Fraction(3, 20)}
    # The price inputs are published, so the cap can be recomputed from the document alone.
    assert live["emission_valuation"] == valuation and live["miner_emission_microusd"] == 4_000_000
    lowered = rewards.epoch_payout_document(POLICY, epoch=3, mode="live", entries=entries,
                                            valuation=valuation, payable_cap_microusd=100)
    assert lowered["paid_microusd"] == 100


def chain_price(*, epoch=7, alpha_price=1_500_000, tao_usd=lambda now: (200 * 10**6, now - 10)):
    """The real valuation over a fake chain: 3 alpha to miners at block 50."""
    from types import SimpleNamespace as NS

    from lab_arena import collection_rewards

    block_hash = "0x" + "b" * 64
    emitted = {"event": {"module_id": "SubtensorModule", "event_id": "IncentiveAlphaEmittedToMiners",
                         "attributes": {"netuid": 71, "emissions": [10**9, 2 * 10**9]}}}
    chain = NS(
        config=NS(netuid=71, network_name="test"), finalized_block_hash=lambda number: block_hash,
        client=NS(get_events=lambda _hash: [emitted], runtime_call=lambda **_: alpha_price),
    )
    return rewards.ChainEmissionPrice(
        scope=lambda: {"epoch": epoch, "valid_from_block": 50},
        valuation=lambda block: collection_rewards.emission_valuation(chain, block, now=lambda: 500, price_source=tao_usd),
        tao_price_source="test:tao/usd",
    )


def test_chain_emission_price_is_alpha_times_alpha_price_times_tao_price_at_the_epoch_block():
    # 3 alpha x 0.0015 TAO per alpha x $200 per TAO = $0.90 of miner emission.
    assert chain_price().valuation(7) == {
        "block": 50, "block_hash": "0x" + "b" * 64, "miner_alpha_rao": 3 * 10**9, "tao_rao_per_alpha": 1_500_000,
        "tao_price_microusd": 200 * 10**6, "price_updated_at": 490, "observed_at": 500,
        "emission_microusd": 900_000, "tao_price_source": "test:tao/usd",
    }


def test_chain_emission_price_fails_closed():
    def refused(now):
        raise ValueError("stale TAO/USD price")

    for source, epoch in (
        (chain_price(), 6),                 # a past epoch: today's TAO price says nothing about it
        (chain_price(tao_usd=refused), 7),  # no fresh TAO/USD quote
        (chain_price(alpha_price=0), 7),    # the chain gives no alpha price
        (chain_price(tao_usd=lambda now: (0, now)), 7),  # a zero price would make the lane cap zero
    ):
        with pytest.raises(rewards.EmissionPriceUnavailable):
            source.valuation(epoch)


def test_signed_host_drops_default_ports_on_both_sides():
    from lab_arena.lead_orders.config import LeadOrdersConfig

    same = {auth.signed_message("GET", h, "/p", "1", b"") for h in ("gw.example", "GW.Example:443", "gw.example:80")}
    assert len(same) == 1
    assert auth.signed_message("GET", "gw.example:8443", "/p", "1", b"") not in same
    for url, host in (("https://GW.example:443/x", "gw.example"), ("http://gw.example:80", "gw.example"),
                      ("https://gw.example:8443", "gw.example:8443")):
        assert LeadOrdersConfig(mode="off", public_base_url=url).public_host == host
