"""Tests for lab_arena.credentials (labarena.md sections 7.3 and 18.5).

Every network call goes through an injected fake ``urlopen``; every decrypt
goes through a locally generated RSA key or a fake KMS client. The canary
test at the end proves the runtime key never appears in any returned
document, any ``repr``, or any exception message.
"""

from __future__ import annotations

import copy
import json
import pickle
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from urllib.error import HTTPError, URLError

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec, rsa

from lab_arena import credentials as creds
from lab_arena.contracts import ArenaContractError, RECIPIENT_DOCUMENT_SCHEMA_VERSION, SHA256_RE

CANARY_KEY = "sk-or-v1-" + "a1b2c3d4e5f6g7h8i9j0" * 2
assert len(CANARY_KEY) == len("sk-or-v1-") + 40
OTHER_KEY = "sk-or-v1-" + "z9y8x7w6v5u4t3s2r1q0" * 2
FIXED_NOW = datetime(2026, 9, 2, 12, 30, 45, 123456, tzinfo=timezone.utc)


class FakeResponse:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *exc: Any) -> bool:
        return False


class FakeUrlopen:
    """Records requests and returns canned key metadata or raises."""

    def __init__(self, data: Optional[Dict[str, Any]] = None, *, body: Optional[bytes] = None, error: Optional[BaseException] = None) -> None:
        self.data = data
        self.body = body
        self.error = error
        self.calls: List[Any] = []

    def __call__(self, request: Any, timeout: Optional[int] = None) -> FakeResponse:
        self.calls.append((request, timeout))
        if self.error is not None:
            raise self.error
        if self.body is not None:
            return FakeResponse(self.body)
        return FakeResponse(json.dumps({"data": self.data}).encode("utf-8"))


def key_metadata(**overrides: Any) -> Dict[str, Any]:
    data = {
        "label": "arena-runtime",
        "usage": 0.25,
        "limit": 10.0,
        "limit_remaining": 9.75,
        "limit_reset": None,
        "is_free_tier": False,
        "is_management_key": False,
        "disabled": False,
        "expires_at": None,
        "hash": "b" * 64,
    }
    data.update(overrides)
    return data


def public_der(private_key: Any) -> bytes:
    return private_key.public_key().public_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )


@pytest.fixture(scope="module")
def rsa_2048() -> rsa.RSAPrivateKey:
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


@pytest.fixture(scope="module")
def rsa_4096() -> rsa.RSAPrivateKey:
    return rsa.generate_private_key(public_exponent=65537, key_size=4096)


@pytest.fixture
def decryptor(rsa_2048: rsa.RSAPrivateKey) -> creds.LocalRsaDecryptor:
    return creds.LocalRsaDecryptor(rsa_2048)


@pytest.fixture
def recipient(rsa_2048: rsa.RSAPrivateKey) -> Dict[str, Any]:
    return creds.recipient_document(public_der(rsa_2048))


# ---------------------------------------------------------------------------
# Copied key-vault helpers
# ---------------------------------------------------------------------------


def test_key_format_accepts_and_strips_whitespace() -> None:
    assert creds.validate_openrouter_key_format("  " + CANARY_KEY + "\n") == CANARY_KEY


def test_key_format_repairs_prefix_case_only() -> None:
    upper = "SK-OR-V1-" + CANARY_KEY[len("sk-or-v1-") :]
    assert creds.validate_openrouter_key_format(upper) == CANARY_KEY
    mixed = "Sk-Or-v1-" + CANARY_KEY[len("sk-or-v1-") :]
    assert creds.validate_openrouter_key_format(mixed) == CANARY_KEY


@pytest.mark.parametrize(
    "raw",
    [
        "",
        None,
        "sk-or-v2-" + "a" * 40,
        "sk-or-v1-" + "a" * 23,
        "sk-or-v1-" + "a" * 30 + "!",
        "sk-" + "a" * 40,
        "sk-or-v1-" + "a" * 40 + " trailing",
    ],
)
def test_key_format_rejects_invalid_keys(raw: Any) -> None:
    with pytest.raises(creds.OpenRouterKeyError) as info:
        creds.validate_openrouter_key_format(raw)
    assert isinstance(info.value, ArenaContractError)
    assert "sk-or-v1-" in str(info.value)


def test_strict_policy_copy_is_independent() -> None:
    policy = creds.strict_openrouter_provider_policy()
    assert policy == {"data_collection": "deny", "allow_fallbacks": False}
    policy["allow_fallbacks"] = True
    assert creds.STRICT_OPENROUTER_PROVIDER_POLICY["allow_fallbacks"] is False


def test_preflight_success_uses_observed_hash_and_sends_bearer() -> None:
    fake = FakeUrlopen(key_metadata())
    record = creds.preflight_openrouter_key(CANARY_KEY, urlopen=fake, timeout_seconds=7)
    assert len(fake.calls) == 1
    request, timeout = fake.calls[0]
    assert timeout == 7
    assert request.full_url == creds.OPENROUTER_KEY_INFO_URL == "https://openrouter.ai/api/v1/key"
    assert request.get_method() == "GET"
    assert request.get_header("Authorization") == "Bearer " + CANARY_KEY
    assert request.get_header("Accept") == "application/json"
    assert record["key_hash"] == "b" * 64
    assert record["key_label_hash"] == creds._local_key_hash("arena-runtime")
    assert record["creator_user_id_hash"] is None
    assert record["limit"] == 10.0 and record["limit_remaining"] == 9.75 and record["usage"] == 0.25
    assert record["is_free_tier"] is False
    assert set(record) == {
        "key_hash",
        "key_label_hash",
        "creator_user_id_hash",
        "limit",
        "limit_remaining",
        "limit_reset",
        "usage",
        "is_free_tier",
        "is_management_key",
        "expires_at",
    }


def test_preflight_falls_back_to_local_hash_without_observed_hash() -> None:
    record = creds.preflight_openrouter_key(CANARY_KEY, urlopen=FakeUrlopen(key_metadata(hash=None)))
    assert record["key_hash"] == creds._local_key_hash(CANARY_KEY)


def test_preflight_expected_hash_paths() -> None:
    expected = "c" * 64
    with pytest.raises(creds.OpenRouterKeyError, match="differs from the expected"):
        creds.preflight_openrouter_key(CANARY_KEY, urlopen=FakeUrlopen(key_metadata(hash="b" * 64)), expected_key_hash=expected)
    with pytest.raises(creds.OpenRouterKeyError, match="expected OpenRouter runtime key hash is invalid"):
        creds.preflight_openrouter_key(CANARY_KEY, urlopen=FakeUrlopen(key_metadata()), expected_key_hash="not-hex")
    record = creds.preflight_openrouter_key(CANARY_KEY, urlopen=FakeUrlopen(key_metadata(hash="")), expected_key_hash=expected)
    assert record["key_hash"] == expected
    record = creds.preflight_openrouter_key(CANARY_KEY, urlopen=FakeUrlopen(key_metadata(hash=expected)), expected_key_hash=expected)
    assert record["key_hash"] == expected


@pytest.mark.parametrize("code", [401, 403])
def test_preflight_unauthorized(code: int) -> None:
    error = HTTPError(creds.OPENROUTER_KEY_INFO_URL, code, "denied", {}, None)
    with pytest.raises(creds.OpenRouterKeyError, match="invalid or unauthorized"):
        creds.preflight_openrouter_key(CANARY_KEY, urlopen=FakeUrlopen(error=error))


def test_preflight_other_http_error() -> None:
    error = HTTPError(creds.OPENROUTER_KEY_INFO_URL, 503, "unavailable", {}, None)
    with pytest.raises(creds.OpenRouterKeyError, match="HTTP 503"):
        creds.preflight_openrouter_key(CANARY_KEY, urlopen=FakeUrlopen(error=error))


def test_preflight_url_error_and_timeout() -> None:
    with pytest.raises(creds.OpenRouterKeyError, match="preflight failed: name resolution"):
        creds.preflight_openrouter_key(CANARY_KEY, urlopen=FakeUrlopen(error=URLError("name resolution")))
    with pytest.raises(creds.OpenRouterKeyError, match="transport error"):
        creds.preflight_openrouter_key(CANARY_KEY, urlopen=FakeUrlopen(error=TimeoutError("read timed out")))


def test_preflight_invalid_json() -> None:
    with pytest.raises(creds.OpenRouterKeyError, match="invalid JSON"):
        creds.preflight_openrouter_key(CANARY_KEY, urlopen=FakeUrlopen(body=b"{not json"))


@pytest.mark.parametrize("body", [b"{}", b'{"data": "text"}', b"[1, 2]", b'{"data": null}'])
def test_preflight_no_data(body: bytes) -> None:
    with pytest.raises(creds.OpenRouterKeyError, match="no key metadata"):
        creds.preflight_openrouter_key(CANARY_KEY, urlopen=FakeUrlopen(body=body))


def test_preflight_disabled_key() -> None:
    with pytest.raises(creds.OpenRouterKeyError, match="disabled"):
        creds.preflight_openrouter_key(CANARY_KEY, urlopen=FakeUrlopen(key_metadata(disabled=True)))


def test_preflight_rejects_bad_key_before_any_network() -> None:
    fake = FakeUrlopen(key_metadata())
    with pytest.raises(creds.OpenRouterKeyError):
        creds.preflight_openrouter_key("not-a-key", urlopen=fake)
    assert fake.calls == []


# ---------------------------------------------------------------------------
# Recipient document
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fixture_name,expected_spec", [("rsa_2048", "RSA_2048"), ("rsa_4096", "RSA_4096")])
def test_recipient_document_roundtrip(request: Any, fixture_name: str, expected_spec: str) -> None:
    private_key = request.getfixturevalue(fixture_name)
    der = public_der(private_key)
    document = creds.recipient_document(der)
    assert document["schema_version"] == RECIPIENT_DOCUMENT_SCHEMA_VERSION
    assert document["algorithm"] == "RSAES_OAEP_SHA_256"
    assert document["key_spec"] == expected_spec
    assert SHA256_RE.match(document["public_key_hash"])
    assert document["public_key_hash"] == creds.recipient_key_hash(der)
    assert set(document) == {"schema_version", "algorithm", "key_spec", "public_key_der_b64", "public_key_hash"}
    assert creds.validate_recipient_document(document) == der


def _mutate(document: Dict[str, Any], key: str, value: Any) -> Dict[str, Any]:
    out = dict(document)
    out[key] = value
    return out


@pytest.mark.parametrize(
    "mutation,message",
    [
        (lambda d: _mutate(d, "schema_version", "leadpoet.lab_arena.openrouter_recipient.v2"), "must be one of"),
        (lambda d: _mutate(d, "algorithm", "RSAES_OAEP_SHA_1"), "must be one of"),
        (lambda d: _mutate(d, "key_spec", "RSA_4096"), "key_spec does not match"),
        (lambda d: _mutate(d, "public_key_hash", "sha256:" + "0" * 64), "hash mismatch"),
        (lambda d: _mutate(d, "public_key_der_b64", "!!!!"), "not base64"),
        (lambda d: {**d, "extra": 1}, "unknown fields"),
        (lambda d: {k: v for k, v in d.items() if k != "public_key_hash"}, "missing field"),
        (lambda d: _mutate(d, "public_key_der_b64", "AAAA"), "not valid DER"),
    ],
)
def test_validate_recipient_document_fails_closed(recipient: Dict[str, Any], mutation: Any, message: str) -> None:
    with pytest.raises(ArenaContractError, match=message):
        creds.validate_recipient_document(mutation(recipient))


def test_recipient_document_rejects_non_rsa_keys() -> None:
    ec_key = ec.generate_private_key(ec.SECP256R1())
    with pytest.raises(ArenaContractError, match="must be an RSA key"):
        creds.recipient_document(public_der(ec_key))


# ---------------------------------------------------------------------------
# Envelope: miner side and broker side
# ---------------------------------------------------------------------------


def test_envelope_roundtrip(recipient: Dict[str, Any], decryptor: creds.LocalRsaDecryptor) -> None:
    envelope = creds.encrypt_runtime_key(recipient, CANARY_KEY)
    assert set(envelope) == {"schema_version", "algorithm", "recipient_key_hash", "key_hash", "ciphertext_b64"}
    assert envelope["schema_version"] == "leadpoet.lab_arena.openrouter_envelope.v1"
    assert envelope["algorithm"] == "RSAES_OAEP_SHA_256"
    assert envelope["recipient_key_hash"] == recipient["public_key_hash"] == decryptor.recipient_key_hash
    assert envelope["key_hash"] == creds._local_key_hash(CANARY_KEY)
    validated = creds.validate_envelope_shape(envelope, recipient["public_key_hash"], modulus_bytes=256)
    assert len(validated["ciphertext"]) == 256
    handle = creds.decrypt_runtime_key(envelope, decryptor)
    assert handle.key_hash == envelope["key_hash"]
    assert handle.bearer_header() == {"Authorization": "Bearer " + CANARY_KEY}
    # A second encryption of the same key produces a different ciphertext
    # (OAEP is randomized) but the same commitment.
    again = creds.encrypt_runtime_key(recipient, CANARY_KEY)
    assert again["ciphertext_b64"] != envelope["ciphertext_b64"]
    assert again["key_hash"] == envelope["key_hash"]


def test_envelope_wrong_recipient_hash(recipient: Dict[str, Any], decryptor: creds.LocalRsaDecryptor) -> None:
    envelope = creds.encrypt_runtime_key(recipient, CANARY_KEY)
    with pytest.raises(ArenaContractError, match="different recipient key"):
        creds.validate_envelope_shape(envelope, "sha256:" + "1" * 64)
    with pytest.raises(ArenaContractError, match="does not hold the expected recipient key"):
        creds.decrypt_runtime_key(envelope, decryptor, expected_recipient_key_hash="sha256:" + "1" * 64)
    other = creds.LocalRsaDecryptor(rsa.generate_private_key(public_exponent=65537, key_size=2048))
    with pytest.raises(ArenaContractError, match="different recipient key"):
        creds.decrypt_runtime_key(envelope, other)


def test_envelope_tampered_ciphertext(recipient: Dict[str, Any], decryptor: creds.LocalRsaDecryptor) -> None:
    import base64

    envelope = creds.encrypt_runtime_key(recipient, CANARY_KEY)
    raw = bytearray(base64.b64decode(envelope["ciphertext_b64"]))
    raw[10] ^= 0xFF
    tampered = dict(envelope, ciphertext_b64=base64.b64encode(bytes(raw)).decode("ascii"))
    with pytest.raises(creds.OpenRouterKeyError, match="decryption failed"):
        creds.decrypt_runtime_key(tampered, decryptor)


def test_envelope_bad_base64_and_lengths(recipient: Dict[str, Any], decryptor: creds.LocalRsaDecryptor) -> None:
    import base64

    envelope = creds.encrypt_runtime_key(recipient, CANARY_KEY)
    with pytest.raises(ArenaContractError, match="not base64"):
        creds.validate_envelope_shape(dict(envelope, ciphertext_b64="@@@@"), recipient["public_key_hash"])
    short = base64.b64encode(b"\x01" * 100).decode("ascii")
    with pytest.raises(ArenaContractError, match="not an RSA modulus size"):
        creds.validate_envelope_shape(dict(envelope, ciphertext_b64=short), recipient["public_key_hash"])
    wrong_modulus = base64.b64encode(b"\x01" * 512).decode("ascii")
    with pytest.raises(ArenaContractError, match="does not match the recipient key"):
        creds.validate_envelope_shape(dict(envelope, ciphertext_b64=wrong_modulus), recipient["public_key_hash"], modulus_bytes=256)
    with pytest.raises(ArenaContractError, match="unsupported recipient modulus"):
        creds.validate_envelope_shape(envelope, recipient["public_key_hash"], modulus_bytes=100)


@pytest.mark.parametrize(
    "mutation,message",
    [
        (lambda e: {**e, "note": "x"}, "unknown fields"),
        (lambda e: {k: v for k, v in e.items() if k != "key_hash"}, "missing field"),
        (lambda e: dict(e, key_hash="G" * 64), "64 lowercase hex"),
        (lambda e: dict(e, key_hash="ab"), "too short"),
        (lambda e: dict(e, schema_version="leadpoet.lab_arena.openrouter_envelope.v0"), "must be one of"),
        (lambda e: dict(e, algorithm="RSAES_PKCS1_V1_5"), "must be one of"),
        (lambda e: dict(e, recipient_key_hash="abc"), "must be a sha256 hash"),
        (lambda e: "not an object", "must be an object"),
    ],
)
def test_envelope_shape_fails_closed(recipient: Dict[str, Any], mutation: Any, message: str) -> None:
    envelope = creds.encrypt_runtime_key(recipient, CANARY_KEY)
    with pytest.raises(ArenaContractError, match=message):
        creds.validate_envelope_shape(mutation(envelope), recipient["public_key_hash"])


def test_envelope_key_hash_must_match_plaintext(recipient: Dict[str, Any], decryptor: creds.LocalRsaDecryptor) -> None:
    envelope = creds.encrypt_runtime_key(recipient, CANARY_KEY)
    forged = dict(envelope, key_hash=creds._local_key_hash(OTHER_KEY))
    with pytest.raises(creds.OpenRouterKeyError, match="key_hash does not match the decrypted key"):
        creds.decrypt_runtime_key(forged, decryptor)


def test_decrypted_plaintext_must_be_a_valid_key(recipient: Dict[str, Any], rsa_2048: rsa.RSAPrivateKey) -> None:
    import base64

    public_key = rsa_2048.public_key()
    ciphertext = public_key.encrypt(b"\xff\xfe not utf8", creds._OAEP_PADDING)
    envelope = {
        "schema_version": creds.ENVELOPE_SCHEMA_VERSION,
        "algorithm": creds.RECIPIENT_ALGORITHM,
        "recipient_key_hash": recipient["public_key_hash"],
        "key_hash": "0" * 64,
        "ciphertext_b64": base64.b64encode(ciphertext).decode("ascii"),
    }
    with pytest.raises(creds.OpenRouterKeyError, match="not UTF-8"):
        creds.decrypt_runtime_key(envelope, creds.LocalRsaDecryptor(rsa_2048))
    ciphertext = public_key.encrypt(b"sk-live-not-openrouter-key-material", creds._OAEP_PADDING)
    envelope["ciphertext_b64"] = base64.b64encode(ciphertext).decode("ascii")
    with pytest.raises(creds.OpenRouterKeyError, match="must start with sk-or-v1-"):
        creds.decrypt_runtime_key(envelope, creds.LocalRsaDecryptor(rsa_2048))


def test_runtime_key_handle_redacts_everywhere() -> None:
    handle = creds.RuntimeKeyHandle(CANARY_KEY)
    assert CANARY_KEY not in repr(handle)
    assert CANARY_KEY not in str(handle)
    assert handle.key_hash in repr(handle)
    assert not hasattr(handle, "__dict__")
    with pytest.raises(TypeError):
        vars(handle)
    with pytest.raises(TypeError):
        pickle.dumps(handle)
    with pytest.raises(TypeError):
        copy.copy(handle)
    with pytest.raises(TypeError):
        copy.deepcopy(handle)
    assert handle.bearer_header()["Authorization"].endswith(CANARY_KEY)
    handle.revoke()
    with pytest.raises(creds.OpenRouterKeyError, match="revoked"):
        handle.bearer_header()
    with pytest.raises(creds.OpenRouterKeyError):
        creds.RuntimeKeyHandle("garbage")


# ---------------------------------------------------------------------------
# Registration record
# ---------------------------------------------------------------------------


def test_register_returns_non_secret_record(recipient: Dict[str, Any], decryptor: creds.LocalRsaDecryptor) -> None:
    envelope = creds.encrypt_runtime_key(recipient, CANARY_KEY)
    fake = FakeUrlopen(key_metadata())
    record = creds.register_openrouter_key(
        envelope,
        decryptor=decryptor,
        urlopen=fake,
        expected_recipient_key_hash=recipient["public_key_hash"],
        timeout_seconds=5,
        now=lambda: FIXED_NOW,
    )
    assert tuple(record) == creds.PREFLIGHT_RECORD_FIELDS
    assert record == {
        "key_hash": creds._local_key_hash(CANARY_KEY),
        "limit_microusd": 10_000_000,
        "limit_remaining_microusd": 9_750_000,
        "usage_microusd": 250_000,
        "observed_at": "2026-09-02T12:30:45Z",
        "preflight_status": "ok",
    }
    assert fake.calls[0][1] == 5
    assert fake.calls[0][0].get_header("Authorization") == "Bearer " + CANARY_KEY


def test_register_rounds_micro_usd_down(recipient: Dict[str, Any], decryptor: creds.LocalRsaDecryptor) -> None:
    envelope = creds.encrypt_runtime_key(recipient, CANARY_KEY)
    fake = FakeUrlopen(key_metadata(limit=10.0000009, limit_remaining="9.8765433", usage=0.1234567))
    record = creds.register_openrouter_key(envelope, decryptor=decryptor, urlopen=fake, expected_recipient_key_hash=recipient["public_key_hash"])
    assert record["limit_microusd"] == 10_000_000
    assert record["limit_remaining_microusd"] == 9_876_543
    assert record["usage_microusd"] == 123_456


def test_register_unlimited_key(recipient: Dict[str, Any], decryptor: creds.LocalRsaDecryptor) -> None:
    envelope = creds.encrypt_runtime_key(recipient, CANARY_KEY)
    fake = FakeUrlopen(key_metadata(limit=None, limit_remaining=None, usage=3))
    record = creds.register_openrouter_key(envelope, decryptor=decryptor, urlopen=fake, expected_recipient_key_hash=recipient["public_key_hash"])
    assert record["limit_microusd"] is None
    assert record["limit_remaining_microusd"] is None
    assert record["usage_microusd"] == 3_000_000


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"usage": None}, "no usage"),
        ({"usage": "lots"}, "non-numeric usage"),
        ({"usage": True}, "non-numeric usage"),
        ({"limit": 5.0, "limit_remaining": None}, "without a remaining balance"),
        ({"limit": "NaN"}, "non-finite limit"),
        ({"limit": [1]}, "non-numeric limit"),
    ],
)
def test_register_fails_closed_on_bad_amounts(recipient: Dict[str, Any], decryptor: creds.LocalRsaDecryptor, overrides: Dict[str, Any], message: str) -> None:
    envelope = creds.encrypt_runtime_key(recipient, CANARY_KEY)
    with pytest.raises(creds.OpenRouterKeyError, match=message):
        creds.register_openrouter_key(envelope, decryptor=decryptor, urlopen=FakeUrlopen(key_metadata(**overrides)), expected_recipient_key_hash=recipient["public_key_hash"])


def test_usd_to_microusd_edges() -> None:
    assert creds.usd_to_microusd(None, "limit") is None
    assert creds.usd_to_microusd(0, "limit") == 0
    assert creds.usd_to_microusd("0.000001", "limit") == 1
    assert creds.usd_to_microusd(0.0000019, "limit") == 1
    assert creds.usd_to_microusd(-0.0000015, "limit") == -2
    with pytest.raises(creds.OpenRouterKeyError):
        creds.usd_to_microusd(float("inf"), "limit")


def test_register_preflight_failure_produces_no_record(recipient: Dict[str, Any], decryptor: creds.LocalRsaDecryptor) -> None:
    envelope = creds.encrypt_runtime_key(recipient, CANARY_KEY)
    error = HTTPError(creds.OPENROUTER_KEY_INFO_URL, 401, "denied", {}, None)
    with pytest.raises(creds.OpenRouterKeyError, match="invalid or unauthorized"):
        creds.register_openrouter_key(envelope, decryptor=decryptor, urlopen=FakeUrlopen(error=error), expected_recipient_key_hash=recipient["public_key_hash"])


def test_register_bad_envelope_never_decrypts(recipient: Dict[str, Any], decryptor: creds.LocalRsaDecryptor) -> None:
    class CountingDecryptor:
        recipient_key_hash = decryptor.recipient_key_hash
        modulus_bytes = decryptor.modulus_bytes
        calls = 0

        def decrypt(self, ciphertext: bytes) -> bytes:
            type(self).calls += 1
            return decryptor.decrypt(ciphertext)

    envelope = creds.encrypt_runtime_key(recipient, CANARY_KEY)
    fake = FakeUrlopen(key_metadata())
    counting = CountingDecryptor()
    with pytest.raises(ArenaContractError):
        creds.register_openrouter_key({**envelope, "extra": True}, decryptor=counting, urlopen=fake, expected_recipient_key_hash=recipient["public_key_hash"])
    assert CountingDecryptor.calls == 0
    assert fake.calls == []
    with pytest.raises(ArenaContractError):
        creds.register_openrouter_key(envelope, decryptor=counting, urlopen=fake, expected_recipient_key_hash="not-a-hash")
    assert CountingDecryptor.calls == 0
    creds.register_openrouter_key(envelope, decryptor=counting, urlopen=fake, expected_recipient_key_hash=recipient["public_key_hash"])
    assert CountingDecryptor.calls == 1


# ---------------------------------------------------------------------------
# KMS decryptor with a fake client
# ---------------------------------------------------------------------------


class FakeKmsClient:
    def __init__(self, private_key: rsa.RSAPrivateKey, *, key_spec: str = "RSA_4096", usage: str = "ENCRYPT_DECRYPT", algorithms: Any = ("RSAES_OAEP_SHA_1", "RSAES_OAEP_SHA_256")) -> None:
        self._private_key = private_key
        self.key_spec = key_spec
        self.usage = usage
        self.algorithms = algorithms
        self.decrypt_calls: List[Dict[str, Any]] = []

    def get_public_key(self, KeyId: str) -> Dict[str, Any]:
        return {
            "KeyId": KeyId,
            "PublicKey": public_der(self._private_key),
            "KeySpec": self.key_spec,
            "KeyUsage": self.usage,
            "EncryptionAlgorithms": list(self.algorithms),
        }

    def decrypt(self, **kwargs: Any) -> Dict[str, Any]:
        self.decrypt_calls.append(kwargs)
        return {"Plaintext": self._private_key.decrypt(kwargs["CiphertextBlob"], creds._OAEP_PADDING)}


def test_kms_decryptor_roundtrip(rsa_4096: rsa.RSAPrivateKey) -> None:
    client = FakeKmsClient(rsa_4096)
    kms = creds.KmsDecryptor("alias/lab-arena-openrouter", client=client)
    assert kms.modulus_bytes == 512
    assert kms.recipient_key_hash == creds.recipient_key_hash(public_der(rsa_4096))
    recipient = creds.recipient_document(kms.public_key_der)
    assert recipient["key_spec"] == "RSA_4096"
    envelope = creds.encrypt_runtime_key(recipient, CANARY_KEY)
    handle = creds.decrypt_runtime_key(envelope, kms)
    assert handle.key_hash == envelope["key_hash"]
    assert client.decrypt_calls == [
        {
            "KeyId": "alias/lab-arena-openrouter",
            "CiphertextBlob": creds.validate_envelope_shape(envelope, kms.recipient_key_hash)["ciphertext"],
            "EncryptionAlgorithm": "RSAES_OAEP_SHA_256",
        }
    ]
    assert CANARY_KEY not in repr(kms)


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"usage": "SIGN_VERIFY"}, "ENCRYPT_DECRYPT"),
        ({"key_spec": "RSA_2048"}, "must be RSA_4096"),
        ({"algorithms": ("RSAES_OAEP_SHA_1",)}, "does not support RSAES_OAEP_SHA_256"),
    ],
)
def test_kms_decryptor_pins_key_properties(rsa_4096: rsa.RSAPrivateKey, kwargs: Dict[str, Any], message: str) -> None:
    with pytest.raises(creds.OpenRouterKeyError, match=message):
        creds.KmsDecryptor("alias/x", client=FakeKmsClient(rsa_4096, **kwargs))


def test_kms_decryptor_rejects_key_size_mismatch(rsa_2048: rsa.RSAPrivateKey) -> None:
    with pytest.raises(creds.OpenRouterKeyError, match="does not match its key spec"):
        creds.KmsDecryptor("alias/x", client=FakeKmsClient(rsa_2048, key_spec="RSA_4096"))


# ---------------------------------------------------------------------------
# Canary (section 18.5)
# ---------------------------------------------------------------------------


def _exception_texts(exc: BaseException) -> List[str]:
    texts: List[str] = []
    seen: set = set()
    current: Optional[BaseException] = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        texts.append(str(current))
        texts.append(repr(current))
        texts.extend(str(arg) for arg in current.args)
        texts.append(repr(getattr(current, "__dict__", {})))
        current = current.__cause__ or current.__context__
    return texts


def test_canary_key_never_leaks(recipient: Dict[str, Any], decryptor: creds.LocalRsaDecryptor, rsa_4096: rsa.RSAPrivateKey) -> None:
    observed: List[str] = []

    def capture(*values: Any) -> None:
        for value in values:
            observed.append(repr(value))
            observed.append(str(value))
            if isinstance(value, (dict, list)):
                observed.append(json.dumps(value, default=str))

    def capture_failure(action: Any) -> None:
        with pytest.raises(Exception) as info:
            action()
        observed.extend(_exception_texts(info.value))

    capture(creds.validate_openrouter_key_format(CANARY_KEY))
    observed.pop()  # the validated key is the key itself; everything else must be clean
    observed.pop()
    capture(recipient, creds.validate_recipient_document(recipient))
    envelope = creds.encrypt_runtime_key(recipient, CANARY_KEY)
    capture(envelope, creds.validate_envelope_shape(envelope, recipient["public_key_hash"]))
    handle = creds.decrypt_runtime_key(envelope, decryptor)
    capture(handle, decryptor, creds.KmsDecryptor("alias/x", client=FakeKmsClient(rsa_4096)))
    capture(creds.preflight_openrouter_key(CANARY_KEY, urlopen=FakeUrlopen(key_metadata())))
    capture(creds.preflight_record(handle, urlopen=FakeUrlopen(key_metadata()), now=lambda: FIXED_NOW))
    capture(
        creds.register_openrouter_key(
            envelope, decryptor=decryptor, urlopen=FakeUrlopen(key_metadata()), expected_recipient_key_hash=recipient["public_key_hash"]
        )
    )
    unauthorized = HTTPError(creds.OPENROUTER_KEY_INFO_URL, 401, "denied", {}, None)
    capture_failure(lambda: creds.preflight_openrouter_key(CANARY_KEY, urlopen=FakeUrlopen(error=unauthorized)))
    capture_failure(lambda: creds.preflight_openrouter_key(CANARY_KEY, urlopen=FakeUrlopen(error=URLError("down"))))
    capture_failure(lambda: creds.preflight_openrouter_key(CANARY_KEY, urlopen=FakeUrlopen(body=b"nope")))
    capture_failure(lambda: creds.preflight_openrouter_key(CANARY_KEY, urlopen=FakeUrlopen(key_metadata(disabled=True))))
    capture_failure(lambda: creds.preflight_openrouter_key(CANARY_KEY, urlopen=FakeUrlopen(key_metadata()), expected_key_hash="c" * 64))
    capture_failure(lambda: creds.decrypt_runtime_key(dict(envelope, key_hash="0" * 64), decryptor))
    capture_failure(lambda: creds.register_openrouter_key(envelope, decryptor=decryptor, urlopen=FakeUrlopen(key_metadata(usage=None)), expected_recipient_key_hash=recipient["public_key_hash"]))
    capture_failure(lambda: creds.encrypt_runtime_key(recipient, CANARY_KEY + " x"))
    capture_failure(lambda: creds.validate_openrouter_key_format(CANARY_KEY.replace("sk-or-v1-", "sk-or-v9-")))
    capture_failure(lambda: creds.RuntimeKeyHandle(CANARY_KEY[:-30]))
    capture_failure(lambda: pickle.dumps(handle))

    assert observed, "canary captured nothing"
    leaks = [text for text in observed if CANARY_KEY in text]
    assert leaks == []
    # The only sanctioned exit is the bearer header for the broker's outbound request.
    assert handle.bearer_header() == {"Authorization": "Bearer " + CANARY_KEY}
