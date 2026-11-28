from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

import pytest
from bittensor_wallet import Keypair

from lab_arena import collection_cli, collection_intake as intake, contracts, miner_cli
from lab_arena.collection_pool import POOL_SCOPE, POOL_PAYMENT_RULE, POOL_TERMS, POOL_TERMS_HASH
from lab_arena.service import ArenaService, ServiceError
from lab_arena.chain import wallet_signature_verifier


@pytest.mark.parametrize("priority", [None, 0, 1])
@pytest.mark.parametrize("reward_rate_bps", [12500, 15000])
def test_registration_authorizes_a_total_and_reuse_without_per_run_selection(monkeypatch, priority, reward_rate_bps):
    key = Keypair.create_from_uri("//StandingSandbox")
    for name in ("OPENROUTER_API_KEY", "OPENROUTER_MANAGEMENT_KEY"):
        monkeypatch.setenv(name, "sk-or-v1-" + "m" * 40)
    published = intake.terms("live", POOL_SCOPE, reward_rate_bps=reward_rate_bps)
    args = ["collection-register", "--spend-limit-usd", "1.23", "--accept-terms", published["terms_sha256"]]
    if priority is not None:
        args += ["--priority", str(priority)]
    parsed = miner_cli.build_parser().parse_args(args)
    assert published["terms"] == POOL_TERMS.replace("1.25 times", f"{reward_rate_bps / 10000:.2f} times")
    def read(url, **kwargs):
        assert url.endswith("/terms?task_scope=sales_sandbox")
        return SimpleNamespace(status_code=200, json=lambda: published)
    def register(url, *, json, **kwargs):
        auth = json["authorization"]
        sig = contracts.validate_signed_request(auth, expected_scope=intake.SCOPE["register"],
            expected_round_id=intake.scope_id("live"), now=auth["timestamp"], verify_signature=wallet_signature_verifier)
        body = sig["body"]
        assert set(body) == {"spend_limit_microusd","terms_sha256","task_scope","priority","credentials_sha256"}
        assert intake.standing_limit(body) == 1230000
        assert body["priority"] == (1 if priority is None else priority)
        rpc = Mock(return_value={"registration_id": str(uuid4())})
        cm = Mock(); cm.validate_openrouter_funding.return_value = "encrypted"
        result = intake.CollectionIntake(SimpleNamespace(rpc=rpc),cm,"live").register(sig,"owner",json["credentials"],published=published)
        assert rpc.call_args.args[1]["p_cap"] == body["spend_limit_microusd"]
        assert rpc.call_args.args[1]["p_consent"]["body"] == body
        return SimpleNamespace(status_code=200, json=lambda: result)
    assert collection_cli.run(parsed, lambda _: key, session=SimpleNamespace(get=read,post=register)) == 0


def test_live_terms_follow_activation_and_remain_bound_to_scope():
    rpc = Mock(side_effect=lambda name, _: True if name == "collection_pool_active_v1" else 15000);svc=SimpleNamespace(collection_mode=lambda:"live",_store=SimpleNamespace(_transport=SimpleNamespace(rpc=rpc)))
    assert ArenaService.collection_terms(svc) == ArenaService.collection_terms(svc,POOL_SCOPE) == intake.terms("live",POOL_SCOPE,reward_rate_bps=15000)
    assert contracts.hash_bytes(POOL_TERMS.encode()) == POOL_TERMS_HASH
    rpc.assert_called_with("collection_pool_reward_rate_bps_v1",{})
    rpc.side_effect = None
    with pytest.raises(intake.IntakeError):
        intake.terms("pilot",POOL_SCOPE)
    rpc.return_value=False
    with pytest.raises(ServiceError, match="collection_inactive"):
        ArenaService.collection_terms(svc,POOL_SCOPE)


@pytest.mark.parametrize("extra", [{"priority":True},{"priority":2},{"offer_id":"dv-"+"a"*64},{"max_slots_per_utc_day":1},{"spend_limit_microusd":0},{"terms_sha256":intake.PAID_WORKFLOW_TERMS_HASH}])
def test_modified_pool_signature_is_refused(extra):
    body=dict(spend_limit_microusd=1000000, priority=0, task_scope=POOL_SCOPE, terms_sha256=POOL_TERMS_HASH, credentials_sha256="sha256:"+"a"*64)
    with pytest.raises(intake.IntakeError):
        intake.standing_limit(body | extra)


def test_pool_does_not_send_keys_to_a_gateway_with_legacy_terms():
    args=miner_cli.build_parser().parse_args(["collection-register","--spend-limit-usd","1","--accept-terms",POOL_TERMS_HASH])
    remote=SimpleNamespace(status_code=200,json=lambda:intake.terms("live",intake.TASK_SCOPE))
    session=SimpleNamespace(get=lambda *a,**k:remote,post=lambda *a,**k:pytest.fail("sent credentials"))
    assert collection_cli.run(args,lambda _:pytest.fail("opened wallet"),session=session)==2


@pytest.mark.parametrize("label",["accepted","code_rejected"])
def test_both_native_labels_can_be_verified_for_payment(label):
    intake._validate_quality(dict(payment_rule=POOL_PAYMENT_RULE,training_status=label))
    with pytest.raises(intake.IntakeError):
        intake._validate_quality(dict(payment_rule=POOL_PAYMENT_RULE,training_status="cannot_judge"))

@pytest.mark.parametrize("code", ["42883", "PGRST202", "08006", None])
def test_only_absent_rate_rpc_can_advertise_the_legacy_terms(code):
    exc = intake.store.ArenaStoreError("RPC unavailable", operation="rpc",
                                     resource="collection_pool_reward_rate_bps_v1", code=code)
    transport = SimpleNamespace(rpc=Mock(side_effect=exc))
    if code in ("42883", "PGRST202"):
        assert intake.active_pool_reward_rate(transport) == 12500
    else:
        with pytest.raises(intake.store.ArenaStoreError):
            intake.active_pool_reward_rate(transport)


def test_unrelated_missing_rpc_is_not_a_rate_signal():
    error = intake.store.ArenaStoreError("RPC unavailable", operation="rpc",
                                         resource="collection_pool_active_v1", code="PGRST202")
    with pytest.raises(intake.store.ArenaStoreError):
        intake.active_pool_reward_rate(SimpleNamespace(rpc=Mock(side_effect=error)))


@pytest.mark.parametrize("rate", [None, True, 1.50, "15000", 12501, 15001])
def test_unknown_active_rates_cannot_publish_terms(rate):
    with pytest.raises(intake.IntakeError, match="collection_pool_rate_unavailable"):
        intake.active_pool_reward_rate(SimpleNamespace(rpc=Mock(return_value=rate)))


def test_cli_refuses_mismatched_multiplier_before_reading_funding_keys():
    args = miner_cli.build_parser().parse_args(["collection-register", "--spend-limit-usd", "1"])
    document = intake.terms("live", POOL_SCOPE)
    document["reward_multiplier"] = "1.50"
    session = SimpleNamespace(get=lambda *_a, **_kw: SimpleNamespace(status_code=200, json=lambda: document),
                              post=lambda *_a, **_kw: pytest.fail("unexpected enrollment"))
    assert collection_cli.run(args, lambda _: pytest.fail("unexpected wallet access"), session=session) == 2


def test_service_resumes_only_the_exact_pre_cutover_registration(monkeypatch):
    from copy import deepcopy
    from datetime import datetime, timezone
    from uuid import UUID
    from lab_arena import service as gateway

    key = Keypair.create_from_uri("//CutoverRetry")
    creds = {field: "private-fixture-" + field for field in collection_cli.OPENROUTER_FUNDING_FIELDS}
    def authorization(cap):
        return contracts.build_signed_request(scope=intake.SCOPE["register"], round_id=intake.scope_id("live"),
            hotkey=key.ss58_address, body={"task_scope":POOL_SCOPE,"terms_sha256":POOL_TERMS_HASH,
                "spend_limit_microusd":cap,"priority":0,"credentials_sha256":contracts.document_hash(creds)},
            sign_message=lambda m:key.sign(m).hex(), timestamp=100)
    signed = authorization(500)
    accepted = {}
    rate = 12500
    def rpc(name, args):
        if name == "collection_pool_active_v1":
            return True
        if name == "collection_pool_reward_rate_bps_v1":
            return rate
        assert name == "collection_register_v1"
        envelope = args["p_consent"]
        if args["p_id"] in accepted:
            assert args["p_openrouter"] is None and accepted[args["p_id"]] == envelope
        elif rate == 15000:
            assert args["p_openrouter"] is None
            return {"error":"collection_consent_invalid"}
        else:
            assert args["p_openrouter"]["fingerprint"] == contracts.hash_bytes(creds["openrouter_api_key"].encode())
            accepted[args["p_id"]] = deepcopy(envelope)
        return {"status":"enrolled","registration_id":args["p_id"]}
    svc = object.__new__(ArenaService)
    manager = Mock()
    manager.validate_openrouter_funding.return_value = "test-ciphertext"
    svc._clock = lambda:datetime.fromtimestamp(100,timezone.utc)
    svc._store = SimpleNamespace(_transport=SimpleNamespace(rpc=rpc))
    svc._config = SimpleNamespace(chain=object(),verify_signature=wallet_signature_verifier,credential_manager=manager,
        defaults=SimpleNamespace(baseline_hotkey="baseline"),accepted_burn_hotkey="burn",banned_hotkeys_source=lambda:())
    svc.collection_mode = lambda:"live"
    svc._enforce_submission_request_limit = lambda _:None
    monkeypatch.setattr(gateway,"resolve_finalized_owner", lambda *_:SimpleNamespace(coldkey=key.ss58_address))
    req = {"authorization":signed,"credentials":creds}
    result = svc.handle_collection_request("register", req)
    assert result["registration_id"] == str(UUID(hex=signed["request_id"]))
    rate = 15000
    assert svc.handle_collection_request("register",req) == result
    with pytest.raises(ServiceError,match="collection_consent_invalid"):
        svc.handle_collection_request("register",{"authorization":authorization(499),"credentials":creds})
    with pytest.raises(ServiceError,match="collection_consent_invalid"):
        svc.handle_collection_request("register",{"authorization":signed,"credentials":{**creds,"openrouter_api_key":"changed"}})
    assert manager.validate_openrouter_funding.call_count == 1 and len(accepted)==1
