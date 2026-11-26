"""Exercise the collection commands miners use without submitting real keys."""

import uuid
from types import SimpleNamespace

import pytest
from bittensor_wallet import Keypair

from lab_arena import collection_cli, collection_intake as intake, contracts, miner_cli
from lab_arena.chain import wallet_signature_verifier


def register_args(scope=intake.TASK_SCOPE, extra=()):
    terms = intake.terms("pilot", scope)
    return miner_cli.build_parser().parse_args([
        "collection-register", "--mode", "pilot", "--task-scope", scope,
        "--wallet-name", "miner", "--hotkey-name", "default",
        "--daily-cap-usd", "0.30", "--accept-terms", terms["terms_sha256"], *extra,
    ])


def test_wallet_read_error_reports_how_to_select_the_wallet(capsys, tmp_path):
    assert miner_cli.main([
        "collection-statement", "--mode", "pilot", "--wallet-path", str(tmp_path),
        "--wallet-name", "miner",
    ]) == 2
    assert "--wallet-name" in capsys.readouterr().err


@pytest.mark.parametrize("scope", [intake.TASK_SCOPE, intake.CORRECTION_SCOPE])
@pytest.mark.parametrize("slots", [None, 1, 6])
def test_register_from_terminal_prompts_only_for_funding_keys(scope, slots, monkeypatch):
    from lab_arena import miner_submit
    for name in ("OPENROUTER_API_KEY", "OPENROUTER_MANAGEMENT_KEY"):
        monkeypatch.delenv(name, raising=False)
    prompts = []
    def masked(prompt):
        prompts.append(prompt)
        return "sk-or-v1-" + ("m" if "management" in prompt.lower() else "g") * 32
    monkeypatch.setattr(miner_submit.getpass, "getpass", masked)
    monkeypatch.setattr(collection_cli.sys.stdin, "isatty", lambda: True)
    key = Keypair.create_from_uri("//CollectionOnboarding")
    def post(url, *, json, **kwargs):
        assert set(json["credentials"]) == {"openrouter_api_key", "openrouter_management_key"}
        signed = json["authorization"]
        assert signed["body"]["daily_cap_microusd"] == 300_000
        assert signed["body"]["task_scope"] == scope
        assert signed["body"].get("max_slots_per_utc_day") == slots
        checked = contracts.validate_signed_request(
            signed,
            expected_scope=intake.SCOPE["register"],
            expected_round_id=intake.scope_id("pilot"),
            now=signed["timestamp"],
            verify_signature=wallet_signature_verifier,
        )
        assert checked["body"] == signed["body"]
        assert signed["body"]["credentials_sha256"] == contracts.document_hash(
            json["credentials"]
        )
        return SimpleNamespace(
            status_code=200, json=lambda: {"registration_id": str(uuid.uuid4())}
        )
    session = SimpleNamespace(
        get=lambda *a, **k: SimpleNamespace(
            status_code=200, json=lambda: intake.terms("pilot", scope)
        ),
        post=post,
    )
    extra = [] if slots is None else ["--max-slots-per-day", str(slots)]
    assert collection_cli.run(register_args(scope,extra), lambda _: key, session=session) == 0
    assert len(prompts) == 2


@pytest.mark.parametrize(("code", "guidance"), [
    ("hotkey_unregistered", "registered"),
    ("revoke_current_registration_first", "collection-statement"),
    ("openrouter_management_key_invalid", "management key"),
    ("sk-or-v1-do-not-echo", "HTTP 422"),
    (["sk-or-v1-do-not-echo"], "HTTP 422"),
])
def test_enrollment_errors_only_show_known_guidance(code, guidance, monkeypatch, capsys):
    for name in ("OPENROUTER_API_KEY", "OPENROUTER_MANAGEMENT_KEY"):
        monkeypatch.setenv(name, "sk-or-v1-" + "m" * 32)
    key = Keypair.create_from_uri("//CollectionOnboarding")
    response = SimpleNamespace(status_code=422, json=lambda: {"code": code, "detail": "sk-or-v1-do-not-echo"})
    session = SimpleNamespace(post=lambda *a, **k: response,
        get=lambda *a, **k: SimpleNamespace(status_code=200, json=lambda: intake.terms("pilot")))
    assert collection_cli.run(register_args(), lambda _: key, session=session) == 2
    message = capsys.readouterr().err
    assert guidance in message
    assert "sk-or" not in message


def test_comparison_rechecks_published_mode_before_using_funding_keys():
    remote = SimpleNamespace(status_code=200, json=lambda: intake.terms("live"))
    session = SimpleNamespace(get=lambda *a, **kw: remote,
                              post=lambda *a, **kw: pytest.fail("credentials were sent"))
    assert collection_cli.run(register_args(), lambda _: pytest.fail("wallet was opened"), session=session) == 2


@pytest.mark.parametrize("slots",[0,7,False,"3",2.5,None])
def test_gateway_rejects_an_invalid_signed_count(slots):
    with pytest.raises(intake.IntakeError,match="collection_consent_invalid"):
        intake.signed_daily_slots(dict.fromkeys(intake.REGISTRATION_FIELDS)|{"max_slots_per_utc_day":slots})


def test_explicit_slots_on_an_older_gateway_fails_before_wallet_or_credentials(monkeypatch):
    old_terms = dict(intake.terms("pilot"))
    old_terms.pop("max_selectable_slots_per_utc_day")
    session = SimpleNamespace(
        get=lambda *a, **kw: SimpleNamespace(status_code=200, json=lambda: old_terms),
        post=lambda *a, **kw: pytest.fail("credentials were sent"),
    )
    monkeypatch.setattr(collection_cli, "read_masked_credential", lambda _: pytest.fail("requested a key"))
    assert collection_cli.run(
        register_args(extra=["--max-slots-per-day", "3"]),
        lambda _: pytest.fail("wallet was opened"),
        session=session,
    ) == 2


OFFER_ID = 'dv-' + 'c' * 64
MODEL = intake.MODELS[2]


def paid_schedule():
    return {'pilot_evidence': {'schema': 'leadpoet.collection.verified_p7_prices.v1',
            'live_batch_id': OFFER_ID, 'prices': {'P': {MODEL: 2822}},
            'budgets': {'P': {MODEL: 30000}}}}


def test_paid_registration_displays_offer_and_signs_only_after_consent(monkeypatch, capsys):
    from lab_arena.service import ArenaService
    service = SimpleNamespace(collection_mode=lambda: 'live', _store=SimpleNamespace(
        _transport=SimpleNamespace(rpc=lambda *args: paid_schedule())))
    offer = ArenaService.collection_terms(service)
    assert offer['offer_id'] == OFFER_ID
    assert offer['fixed_quotes'] == [{'model': MODEL, 'task_class': 'P', 'quote_microusd': 2822, 'cap_microusd': 30000}]
    assert offer['terms_sha256'] != intake.TERMS_HASH
    args = miner_cli.build_parser().parse_args(['collection-register', '--mode', 'live', '--daily-cap-usd', '0.30'])
    monkeypatch.setattr(collection_cli.sys.stdin, 'isatty', lambda: True)
    got = []
    key = Keypair.create_from_uri('//PaidOfferFixture')
    def post(url, *, json, **kwargs):
        signed = contracts.validate_signed_request(json['authorization'], expected_scope=intake.SCOPE['register'],
                expected_round_id=intake.scope_id('live'), now=json['authorization']['timestamp'], verify_signature=wallet_signature_verifier)
        got.append(signed['body'])
        return SimpleNamespace(status_code=200, json=lambda: {'registration_id': str(uuid.uuid4())})
    session = SimpleNamespace(get=lambda *a, **kw: SimpleNamespace(status_code=200, json=lambda: offer), post=post)
    monkeypatch.setattr('builtins.input', lambda prompt: 'no')
    assert collection_cli.run(args, lambda _: pytest.fail('wallet accessed before consent'), session=session) == 2
    assert not got
    monkeypatch.setattr('builtins.input', lambda prompt: 'yes')
    for field in ('OPENROUTER_API_KEY', 'OPENROUTER_MANAGEMENT_KEY'):
        monkeypatch.setenv(field, 'sk-or-v1-' + 'g'*40)
    assert collection_cli.run(args, lambda _: key, session=session) == 0
    assert got[0]['offer_id'] == OFFER_ID and got[0]['terms_sha256'] == offer['terms_sha256']
    printed = capsys.readouterr()
    assert '$0.002822' in printed.err and MODEL in printed.err and '$0.30' in printed.err


@pytest.mark.parametrize('action', ['statement', 'revoke'])
def test_pilot_funder_can_manage_signed_pilot_registration_after_live_cutover(action):
    from datetime import datetime, timezone
    from lab_arena.service import ArenaService
    from lab_arena.miner_submit import _signed_request
    from time import time
    key = Keypair.create_from_uri('//PilotAccessAfterCutover')
    reg = str(uuid.uuid4())
    body = {'registration_id': reg} if action == 'revoke' else {}
    signed = _signed_request(scope=intake.SCOPE[action], round_id=intake.scope_id('pilot'), body=body, keypair=key, now=time)
    service = object.__new__(ArenaService)
    service._config = SimpleNamespace(verify_signature=wallet_signature_verifier, credential_manager=None)
    service._clock = lambda: datetime.now(timezone.utc)
    service.collection_mode = lambda: 'live'
    calls = []
    service._store = SimpleNamespace(_transport=SimpleNamespace(rpc=lambda name, args: calls.append((name,args)) or {}))
    assert service.handle_collection_request(action, signed) == {}
    assert calls[0][1]['p_mode'] == 'pilot' and calls[0][1]['p_hotkey'] == key.ss58_address and len(calls) == 1


@pytest.mark.parametrize('change', ['no_schedule', 'offer_changed', 'old_terms', 'wrong_scope'])
def test_gateway_denies_inactive_or_stale_paid_offer_before_key_validation(change):
    from tests.lab_arena.collection_intake_test import registration
    signed, creds = registration('X', 'live')
    signed['body'].update(offer_id=OFFER_ID, terms_sha256=intake.PAID_WORKFLOW_TERMS_HASH)
    schedule = paid_schedule()
    if change == 'no_schedule':
        schedule = {}
    elif change == 'offer_changed':
        schedule['pilot_evidence']['live_batch_id'] = 'dv-' + 'd'*64
    elif change == 'old_terms':
        signed['body']['terms_sha256'] = intake.TERMS_HASH
        signed['body'].pop('offer_id')
    else:
        signed['body']['task_scope'] = intake.CORRECTION_SCOPE
    transport = SimpleNamespace(rpc=lambda name, _: schedule if name == 'collection_reward_schedule_v1' else pytest.fail(name))
    offer = {**intake.terms('live'), **intake.comparison_offer(transport)}
    service = intake.CollectionIntake(transport, SimpleNamespace(
        validate_openrouter_funding=lambda *a, **kw: pytest.fail('contacted provider before verifying consent')), 'live')
    with pytest.raises(intake.IntakeError):
        service.register(signed, '5' + 'p'*47, creds, published=offer)


def test_paid_allocator_uses_only_the_signed_p7_model_roster():
    calls = []
    db = SimpleNamespace(rpc=lambda name, args: calls.append(args) or None)
    service = intake.CollectionIntake(db, None, 'live')
    def claim(prices, caps, scenarios=('P7-s40401',), digest=intake.PAID_WORKFLOW_TERMS_HASH):
        return service.claim(OFFER_ID, scenarios, 'a'*40, prices, caps, terms_hash=digest, expected_model=None)
    claim({MODEL: 2822}, {MODEL: 30000})
    assert calls[-1]['p_models'] == [MODEL] and calls[-1]['p_prices'] == {MODEL: 2822}
    for fees, caps, cases, digest in (
        ({MODEL: 2822}, {}, ('P7-s40401',), intake.PAID_WORKFLOW_TERMS_HASH),
        ({'no-such-model': 1}, {'no-such-model': 5}, ('P7-s40401',), intake.PAID_WORKFLOW_TERMS_HASH),
        ({MODEL: 2822}, {MODEL: 30000}, ('P6-s40401',), intake.PAID_WORKFLOW_TERMS_HASH),
        ({MODEL: 2822}, {MODEL: 30000}, ('P7-s40401',), intake.TERMS_HASH),
    ):
        with pytest.raises(intake.IntakeError):
            claim(fees, caps, cases, digest)
    assert len(calls) == 1


def test_paid_registration_requires_unattended_offer_pin(monkeypatch):
    from lab_arena.service import ArenaService
    service = SimpleNamespace(collection_mode=lambda: 'live', _store=SimpleNamespace(
        _transport=SimpleNamespace(rpc=lambda *args: paid_schedule())))
    offer = ArenaService.collection_terms(service)
    args = miner_cli.build_parser().parse_args(['collection-register', '--mode', 'live',
            '--daily-cap-usd', '0.30', '--accept-terms', offer['terms_sha256']])
    session = SimpleNamespace(get=lambda *a, **kw: SimpleNamespace(status_code=200, json=lambda: offer),
                              post=lambda *a, **kw: pytest.fail('posted credentials'))
    monkeypatch.setattr(collection_cli.sys.stdin, 'isatty', lambda: False)
    assert collection_cli.run(args, lambda _: pytest.fail('wallet opened'), session=session) == 2
    args.accept_offer = 'dv-' + '0'*64
    assert collection_cli.run(args, lambda _: pytest.fail('wallet opened'), session=session) == 2
