import base64
import hashlib
from urllib.error import URLError

import pytest

from lab_arena import contracts, credentials as credentials_module, miner_submit
from lab_arena.credentials import CredentialError, CredentialManager
from lab_arena.service import ArenaService, ServiceError


HOTKEY = "5" * 48
SUBMISSION_ID = "submission-1"
RUNTIME_KEY = "sk-or-v1-" + "a" * 24
MANAGEMENT_KEY = "sk-or-v1-" + "b" * 24
DEEPLINE_KEY = "deepline-key-0001"
SCRAPINGDOG_KEY = "scrapingdog-key-0001"


def _submitted_credentials(*, scrapingdog=True):
    result = {
        "openrouter_api_key": RUNTIME_KEY,
        "openrouter_management_key": MANAGEMENT_KEY,
        "deepline_api_key": DEEPLINE_KEY,
    }
    if scrapingdog:
        result["scrapingdog_api_key"] = SCRAPINGDOG_KEY
    return result


class _Kms:
    def __init__(self):
        self.encrypted = []
        self.decrypted = []

    def encrypt(self, **kwargs):
        self.encrypted.append(kwargs)
        return {"CiphertextBlob": b"cipher:" + kwargs["Plaintext"]}

    def decrypt(self, **kwargs):
        self.decrypted.append(kwargs)
        assert kwargs["CiphertextBlob"].startswith(b"cipher:")
        return {"Plaintext": kwargs["CiphertextBlob"][7:]}


def _http_get(url, secret, _timeout):
    if url == credentials_module.OPENROUTER_CURRENT_KEY_URL:
        return {
            "data": {
                "disabled": False,
                "is_management_key": False,
                "limit_remaining": 10,
            }
        }
    if url.startswith("https://openrouter.ai/api/v1/keys/"):
        return {
            "data": {
                "hash": hashlib.sha256(RUNTIME_KEY.encode()).hexdigest(),
                "disabled": False,
            }
        }
    if url == credentials_module.DEEPLINE_BALANCE_URL:
        return {"balance": 10}
    if url == credentials_module.SCRAPINGDOG_ACCOUNT_URL:
        return {
            "requestLimit": 100,
            "requestUsed": 5,
            "apiKey": secret,
            "email": "must-not-be-retained@example.test",
        }
    raise AssertionError("unexpected validation URL")


def test_optional_scrapingdog_key_validates_encrypts_and_decrypts_separately():
    kms = _Kms()
    manager = CredentialManager(kms_key_id="test-key", kms_client=kms, http_get=_http_get)

    encrypted = manager.validate_and_encrypt(
        _submitted_credentials(), submission_id=SUBMISSION_ID, miner_hotkey=HOTKEY
    )

    assert set(encrypted) == {"openrouter", "deepline", "scrapingdog"}
    assert [call["EncryptionContext"]["credential_kind"] for call in kms.encrypted] == [
        "openrouter_runtime",
        "deepline",
        "scrapingdog",
    ]
    row = {
        "submission_id": SUBMISSION_ID,
        "miner_hotkey": HOTKEY,
        "provider": "scrapingdog",
        "ciphertext_b64": encrypted["scrapingdog"],
    }
    assert manager.runtime_key(row, "scrapingdog") == SCRAPINGDOG_KEY
    assert kms.decrypted[0]["EncryptionContext"]["credential_kind"] == "scrapingdog"


def test_absent_scrapingdog_key_preserves_required_two_key_admission():
    manager = CredentialManager(kms_key_id="test-key", kms_client=_Kms(), http_get=_http_get)

    encrypted = manager.validate_and_encrypt(
        _submitted_credentials(scrapingdog=False),
        submission_id=SUBMISSION_ID,
        miner_hotkey=HOTKEY,
    )

    assert set(encrypted) == {"openrouter", "deepline"}


@pytest.mark.parametrize(
    "patch,code",
    [
        ({"scrapingdog_api_key": ""}, "scrapingdog_api_key_invalid"),
        ({"unknown_api_key": "x" * 16}, "submission_credentials_invalid"),
    ],
)
def test_provided_optional_or_unknown_fields_fail_closed(patch, code):
    manager = CredentialManager(kms_key_id="test-key", kms_client=_Kms(), http_get=_http_get)
    submitted = _submitted_credentials(scrapingdog=False)
    submitted.update(patch)

    with pytest.raises(CredentialError, match="^%s$" % code):
        manager.validate_and_encrypt(
            submitted, submission_id=SUBMISSION_ID, miner_hotkey=HOTKEY
        )


def test_scrapingdog_account_document_must_echo_key_and_counters_without_leaking():
    def malformed_account(url, secret, timeout):
        if url == credentials_module.SCRAPINGDOG_ACCOUNT_URL:
            return {"apiKey": "different", "requestLimit": 100, "requestUsed": 5}
        return _http_get(url, secret, timeout)

    manager = CredentialManager(
        kms_key_id="test-key", kms_client=_Kms(), http_get=malformed_account
    )
    with pytest.raises(CredentialError) as captured:
        manager.validate_and_encrypt(
            _submitted_credentials(), submission_id=SUBMISSION_ID, miner_hotkey=HOTKEY
        )
    assert captured.value.code == "scrapingdog_api_key_invalid"
    assert SCRAPINGDOG_KEY not in str(captured.value)


def test_default_scrapingdog_probe_uses_only_fixed_account_query(monkeypatch):
    observed = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _size):
            return (
                b'{"requestLimit":100,"requestUsed":5,"apiKey":"'
                + SCRAPINGDOG_KEY.encode()
                + b'"}'
            )

    def opener(request, timeout):
        observed["url"] = request.full_url
        observed["authorization"] = request.get_header("Authorization")
        observed["timeout"] = timeout
        return Response()

    monkeypatch.setattr(credentials_module, "_DIRECT_URLOPEN", opener)
    document = credentials_module._default_http_get(
        credentials_module.SCRAPINGDOG_ACCOUNT_URL, SCRAPINGDOG_KEY, 7
    )
    assert observed == {
        "url": credentials_module.SCRAPINGDOG_ACCOUNT_URL
        + "?api_key=scrapingdog-key-0001",
        "authorization": None,
        "timeout": 7,
    }
    assert document["apiKey"] == SCRAPINGDOG_KEY


def test_default_probe_drops_secret_bearing_transport_exception(monkeypatch):
    def opener(_request, timeout):
        raise URLError("failed " + SCRAPINGDOG_KEY)

    monkeypatch.setattr(credentials_module, "_DIRECT_URLOPEN", opener)
    with pytest.raises(credentials_module._ProbeError) as captured:
        credentials_module._default_http_get(
            credentials_module.SCRAPINGDOG_ACCOUNT_URL, SCRAPINGDOG_KEY, 7
        )
    assert SCRAPINGDOG_KEY not in str(captured.value)


def test_environment_and_prompt_omit_blank_optional_key():
    environment = {
        "OPENROUTER_API_KEY": RUNTIME_KEY,
        "OPENROUTER_MANAGEMENT_KEY": MANAGEMENT_KEY,
        "DEEPLINE_API_KEY": DEEPLINE_KEY,
    }
    assert miner_submit.submission_credentials_from_environment(environment) == (
        _submitted_credentials(scrapingdog=False)
    )
    prompted = miner_submit.prompt_submission_credentials(
        environ=environment, getpass_fn=lambda prompt: "" if "Scrapingdog" in prompt else None
    )
    assert prompted == _submitted_credentials(scrapingdog=False)


def test_finalize_contract_accepts_omission_and_rejects_malformed_optional_field():
    body = {
        "submission_id": SUBMISSION_ID,
        "source_ref": "arena/arena-2026-09-11/sources/submission-1.tar.gz",
        "source_size_bytes": 100,
        "credentials": _submitted_credentials(scrapingdog=False),
    }
    assert "scrapingdog_api_key" not in contracts.validate_submission_finalize_body(body)[
        "credentials"
    ]
    for value in (None, ""):
        body["credentials"]["scrapingdog_api_key"] = value
        with pytest.raises(contracts.ArenaContractError):
            contracts.validate_submission_finalize_body(body)


class _AcceptedStore:
    def __init__(self, *, scrapingdog):
        self.scrapingdog = scrapingdog

    def get_submission(self, _submission_id):
        return {
            "submission_id": SUBMISSION_ID,
            "round_id": "arena-2026-09-11",
            "miner_hotkey": HOTKEY,
            "status": "accepted",
            "source_ref": "arena/arena-2026-09-11/sources/submission-1.tar.gz",
            "source_size_bytes": 100,
        }

    def get_submission_credential(self, submission_id, miner_hotkey, provider):
        if provider == "scrapingdog" and not self.scrapingdog:
            return None
        return {
            "submission_id": submission_id,
            "miner_hotkey": miner_hotkey,
            "provider": provider,
            "ciphertext_b64": base64.b64encode(provider.encode()).decode(),
        }


def _accepted_retry_service(*, scrapingdog):
    service = object.__new__(ArenaService)
    service._store = _AcceptedStore(scrapingdog=scrapingdog)
    service._request_round = lambda _envelope, scope: (
        {"hotkey": HOTKEY, "body": _accepted_retry_body(scrapingdog=False)},
        {"round_id": "arena-2026-09-11"},
    )
    service._require_submission_window = lambda _round: None
    return service


def _accepted_retry_body(*, scrapingdog):
    return {
        "submission_id": SUBMISSION_ID,
        "source_ref": "arena/arena-2026-09-11/sources/submission-1.tar.gz",
        "source_size_bytes": 100,
        "credentials": _submitted_credentials(scrapingdog=scrapingdog),
    }


def test_accepted_two_key_retry_stays_valid_but_cannot_add_optional_key():
    service = _accepted_retry_service(scrapingdog=False)
    assert service.handle_submission_finalize(SUBMISSION_ID, object())["status"] == "accepted"
    service._request_round = lambda _envelope, scope: (
        {"hotkey": HOTKEY, "body": _accepted_retry_body(scrapingdog=True)},
        {"round_id": "arena-2026-09-11"},
    )
    with pytest.raises(ServiceError) as captured:
        service.handle_submission_finalize(SUBMISSION_ID, object())
    assert captured.value.code == "submission_credentials_immutable"
    assert captured.value.status == 409
