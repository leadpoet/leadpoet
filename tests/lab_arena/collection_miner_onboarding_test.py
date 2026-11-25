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
