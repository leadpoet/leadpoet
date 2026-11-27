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
def test_registration_authorizes_a_total_and_reuse_without_per_run_selection(monkeypatch, priority):
    key = Keypair.create_from_uri("//StandingSandbox")
    for name in ("OPENROUTER_API_KEY", "OPENROUTER_MANAGEMENT_KEY"):
        monkeypatch.setenv(name, "sk-or-v1-" + "m" * 40)
    args = ["collection-register", "--spend-limit-usd", "1.23", "--accept-terms", POOL_TERMS_HASH]
    if priority is not None:
        args += ["--priority", str(priority)]
    parsed = miner_cli.build_parser().parse_args(args)
    published = intake.terms("live", POOL_SCOPE)
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
    rpc = Mock(return_value=True);svc=SimpleNamespace(collection_mode=lambda:"live",_store=SimpleNamespace(_transport=SimpleNamespace(rpc=rpc)))
    assert ArenaService.collection_terms(svc) == ArenaService.collection_terms(svc,POOL_SCOPE) == intake.terms("live",POOL_SCOPE)
    assert contracts.hash_bytes(POOL_TERMS.encode()) == POOL_TERMS_HASH
    rpc.assert_called_with("collection_pool_active_v1",{})
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
