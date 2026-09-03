"""Arena OpenRouter runtime-key handling (labarena.md sections 7.3 and 18.5).

This module owns four things:

1. The Lab helpers copied verbatim from ``gateway/research_lab/key_vault.py``
   (never imported: that module's import closure reaches ``gateway.db``): the
   OpenRouter key format, the strict provider policy, and the bounded key
   preflight. The preflight takes an injectable ``urlopen`` so tests never
   touch the network.
2. The Arena recipient document served by ``GET /recipient``: the RSA public
   key of the Arena-owned KMS ``RSAES_OAEP_SHA_256`` encryption key.
3. The miner-side envelope helper: validate the recipient document locally
   and RSA-OAEP encrypt the runtime key to it. Only the ciphertext and the
   key hash ever leave the miner's machine.
4. The broker-side path: strict envelope validation, one decrypt through an
   injectable ``Decryptor`` (KMS in production, a local RSA key in tests), a
   memory-only ``RuntimeKeyHandle`` that redacts itself everywhere, and
   ``register_openrouter_key`` which returns a non-secret preflight record.

No credential value may ever appear in a returned document, a ``repr``, a
log line, or an exception message. Every failure raises; nothing here
returns a partial record.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_FLOOR
from typing import Any, Callable, Dict, Mapping, Optional, Protocol
from urllib import request as urlrequest
from urllib.parse import quote as urlquote
from urllib.error import HTTPError, URLError

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from lab_arena.contracts import (
    MINER_KEY_PROVIDERS,
    ArenaContractError,
    F,
    RECIPIENT_DOCUMENT_SCHEMA_VERSION,
    REQUEST_LIMITS,
    check_strict_document,
    hash_bytes,
    require_sha256,
    validate_document,
)

# ---------------------------------------------------------------------------
# Copied from gateway/research_lab/key_vault.py (section 3.1: copied, not
# imported). Keep these byte-for-byte in sync with the Lab except for the
# error type and the injectable transport.
# ---------------------------------------------------------------------------

OPENROUTER_KEY_RE = re.compile(r"^sk-or-v1-[A-Za-z0-9_-]{24,}$")
# Scrapingdog and Deepline keys have no published shape beyond "an API key";
# the bounds below only refuse whitespace, control characters, and absurd lengths.
SCRAPINGDOG_KEY_RE = re.compile(r"^[A-Za-z0-9]{16,128}$")
DEEPLINE_KEY_RE = re.compile(r"^[A-Za-z0-9_.\-]{16,256}$")
PROVIDER_KEY_PATTERNS = {"openrouter": OPENROUTER_KEY_RE, "scrapingdog": SCRAPINGDOG_KEY_RE, "deepline": DEEPLINE_KEY_RE}
SCRAPINGDOG_ACCOUNT_URL = "https://api.scrapingdog.com/account"
# Read-only, authenticated, and free: the tool's schema, not an execution.
DEEPLINE_PROBE_URL = "https://code.deepline.com/api/v2/integrations/exa_search/get"
OPENROUTER_API_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_KEY_INFO_URL = f"{OPENROUTER_API_BASE_URL}/key"
_OPENROUTER_KEY_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
# The broker injects this policy into every chat-completion body (section
# 7.3): deny provider data collection and provider fallbacks.
STRICT_OPENROUTER_PROVIDER_POLICY: Dict[str, Any] = {
    "data_collection": "deny",
    "allow_fallbacks": False,
}


class ProviderKeyError(ArenaContractError):
    """Raised when a provider key's validation, decryption, or preflight fails."""


OpenRouterKeyError = ProviderKeyError  # historical name


def strict_openrouter_provider_policy() -> Dict[str, Any]:
    """Return a fresh copy of the strict provider policy for one request body."""

    return dict(STRICT_OPENROUTER_PROVIDER_POLICY)


def validate_provider_key_format(provider: str, raw_key: str) -> str:
    """Per-provider shape check; returns the normalized key or raises."""

    if provider not in PROVIDER_KEY_PATTERNS:
        raise ProviderKeyError("unknown provider")
    if provider == "openrouter":
        return validate_openrouter_key_format(raw_key)
    value = (raw_key or "").strip() if isinstance(raw_key, str) else ""
    if not PROVIDER_KEY_PATTERNS[provider].match(value):
        raise ProviderKeyError("%s key does not look like a valid API key" % provider)
    return value


def _fetch_json_object(request: "urlrequest.Request", provider: str, *, timeout_seconds: int, urlopen: Callable[..., Any]) -> Dict[str, Any]:
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            body = response.read().decode("utf-8")
    except HTTPError as exc:
        if exc.code in (401, 403):
            raise ProviderKeyError("%s key preflight failed: key is invalid or unauthorized" % provider) from exc
        raise ProviderKeyError("%s key preflight failed: HTTP %d" % (provider, exc.code)) from exc
    except URLError as exc:
        raise ProviderKeyError("%s key preflight failed: %s" % (provider, exc.reason)) from exc
    except OSError as exc:
        raise ProviderKeyError("%s key preflight failed: transport error" % provider) from exc
    try:
        decoded = json.loads(body)
    except json.JSONDecodeError as exc:
        raise ProviderKeyError("%s key preflight returned invalid JSON" % provider) from exc
    if not isinstance(decoded, Mapping):
        raise ProviderKeyError("%s key preflight returned no key metadata" % provider)
    return dict(decoded)


def _optional_int(value: Any) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value)


def preflight_provider_key(
    provider: str,
    raw_key: str,
    *,
    timeout_seconds: int = 12,
    urlopen: Callable[..., Any] = urlrequest.urlopen,
) -> Dict[str, Any]:
    """Authenticated, read-only probe of a provider key; returns non-secret metadata only.

    OpenRouter reports limits and usage. Scrapingdog's account endpoint
    reports request counts. Deepline has no usage endpoint, so the probe
    reads one tool schema with the key, which costs nothing and proves the
    key is accepted by the workspace.
    """

    if provider == "openrouter":
        return preflight_openrouter_key(raw_key, timeout_seconds=timeout_seconds, urlopen=urlopen)
    key = validate_provider_key_format(provider, raw_key)
    if provider == "scrapingdog":
        request = urlrequest.Request(SCRAPINGDOG_ACCOUNT_URL + "?api_key=" + urlquote(key, safe=""), headers={"Accept": "application/json"}, method="GET")
        decoded = _fetch_json_object(request, provider, timeout_seconds=timeout_seconds, urlopen=urlopen)
        return {"request_limit": _optional_int(decoded.get("requestLimit")), "request_used": _optional_int(decoded.get("requestUsed"))}
    if provider == "deepline":
        request = urlrequest.Request(DEEPLINE_PROBE_URL, headers={"Authorization": "Bearer " + key, "Accept": "application/json"}, method="GET")
        decoded = _fetch_json_object(request, provider, timeout_seconds=timeout_seconds, urlopen=urlopen)
        return {"tool": "exa_search", "has_input_schema": isinstance(decoded.get("inputSchema"), Mapping)}
    raise ProviderKeyError("unknown provider")


def validate_openrouter_key_format(raw_key: str) -> str:
    value = (raw_key or "").strip()
    prefix = "sk-or-v1-"
    if value[: len(prefix)].lower() == prefix and value[: len(prefix)] != prefix:
        value = prefix + value[len(prefix) :]
    if not OPENROUTER_KEY_RE.match(value):
        raise ProviderKeyError("OpenRouter key must start with sk-or-v1- and look like a valid API key")
    return value


def preflight_openrouter_key(
    raw_key: str,
    *,
    timeout_seconds: int = 12,
    expected_key_hash: Optional[str] = None,
    urlopen: Callable[..., Any] = urlrequest.urlopen,
) -> Dict[str, Any]:
    """Verify a raw OpenRouter key and return only non-secret metadata.

    ``urlopen(request, timeout=...)`` must behave like
    ``urllib.request.urlopen``; tests inject a fake. ``expected_key_hash`` is
    OpenRouter's own key hash (``data.hash``), not the local sha256 commitment.
    """

    key = validate_openrouter_key_format(raw_key)
    req = urlrequest.Request(
        OPENROUTER_KEY_INFO_URL,
        headers={
            "Authorization": f"Bearer {key}",
            "Accept": "application/json",
        },
        method="GET",
    )
    try:
        with urlopen(req, timeout=timeout_seconds) as response:
            body = response.read().decode("utf-8")
    except HTTPError as exc:
        if exc.code in (401, 403):
            raise ProviderKeyError("OpenRouter key preflight failed: key is invalid or unauthorized") from exc
        raise ProviderKeyError(f"OpenRouter key preflight failed: HTTP {exc.code}") from exc
    except URLError as exc:
        raise ProviderKeyError(f"OpenRouter key preflight failed: {exc.reason}") from exc
    except OSError as exc:
        # Read timeouts surface as TimeoutError rather than URLError; the
        # message deliberately carries no transport detail.
        raise ProviderKeyError("OpenRouter key preflight failed: transport error") from exc

    try:
        decoded = json.loads(body)
    except json.JSONDecodeError as exc:
        raise ProviderKeyError("OpenRouter key preflight returned invalid JSON") from exc
    data = decoded.get("data") if isinstance(decoded, Mapping) else None
    if not isinstance(data, Mapping):
        raise ProviderKeyError("OpenRouter key preflight returned no key metadata")
    if data.get("disabled") is True:
        raise ProviderKeyError("OpenRouter key is disabled")
    observed_key_hash = str(data.get("hash") or "").strip()
    if expected_key_hash is not None:
        if not _OPENROUTER_KEY_HASH_RE.fullmatch(expected_key_hash):
            raise ProviderKeyError("expected OpenRouter runtime key hash is invalid")
        if observed_key_hash and observed_key_hash != expected_key_hash:
            raise ProviderKeyError("OpenRouter key preflight differs from the expected runtime key")
        key_hash = expected_key_hash
    else:
        key_hash = observed_key_hash or _local_key_hash(key)
    return {
        "key_hash": key_hash,
        "key_label_hash": _optional_hash(data.get("label")),
        "creator_user_id_hash": _optional_hash(data.get("creator_user_id")),
        "limit": data.get("limit"),
        "limit_remaining": data.get("limit_remaining"),
        "limit_reset": data.get("limit_reset"),
        "usage": data.get("usage"),
        "is_free_tier": data.get("is_free_tier"),
        "is_management_key": data.get("is_management_key"),
        "expires_at": data.get("expires_at"),
    }


def _local_key_hash(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def _optional_hash(value: Any) -> Optional[str]:
    text = "" if value is None else str(value).strip()
    return _local_key_hash(text) if text else None


# ---------------------------------------------------------------------------
# Arena recipient document (section 7.3, served by GET /recipient)
# ---------------------------------------------------------------------------

RECIPIENT_ALGORITHM = "RSAES_OAEP_SHA_256"
RECIPIENT_KEY_SPEC = "RSA_4096"
RECIPIENT_KEY_SPEC_BY_BITS = {2048: "RSA_2048", 3072: "RSA_3072", 4096: "RSA_4096"}
RECIPIENT_MODULUS_BYTES = frozenset(bits // 8 for bits in RECIPIENT_KEY_SPEC_BY_BITS)
ENVELOPE_SCHEMA_VERSION = "leadpoet.lab_arena.provider_key_envelope.v1"

RECIPIENT_DOCUMENT_FIELDS = (
    F("schema_version", "str", choices=(RECIPIENT_DOCUMENT_SCHEMA_VERSION,)),
    F("algorithm", "str", choices=(RECIPIENT_ALGORITHM,)),
    F("key_spec", "str", choices=tuple(RECIPIENT_KEY_SPEC_BY_BITS.values())),
    F("public_key_der_b64", "str", minimum=4, maximum=4096),
    F("public_key_hash", "sha256"),
)

ENVELOPE_FIELDS = (
    F("schema_version", "str", choices=(ENVELOPE_SCHEMA_VERSION,)),
    F("algorithm", "str", choices=(RECIPIENT_ALGORITHM,)),
    F("provider", "str", choices=MINER_KEY_PROVIDERS),
    F("recipient_key_hash", "sha256"),
    F("key_hash", "str", minimum=64, maximum=64),
    F("ciphertext_b64", "str", minimum=4, maximum=1024),
)

_OAEP_PADDING = padding.OAEP(
    mgf=padding.MGF1(algorithm=hashes.SHA256()),
    algorithm=hashes.SHA256(),
    label=None,
)


def recipient_key_hash(public_key_der: bytes) -> str:
    """``sha256:<hex>`` of the DER SubjectPublicKeyInfo bytes."""

    return hash_bytes(bytes(public_key_der))


def _load_rsa_public_key(public_key_der: bytes) -> rsa.RSAPublicKey:
    try:
        key = serialization.load_der_public_key(bytes(public_key_der))
    except (ValueError, TypeError) as exc:
        raise ArenaContractError("recipient public key is not valid DER") from exc
    if not isinstance(key, rsa.RSAPublicKey):
        raise ArenaContractError("recipient public key must be an RSA key")
    if key.key_size not in RECIPIENT_KEY_SPEC_BY_BITS:
        raise ArenaContractError("recipient RSA key size is not supported")
    return key


def recipient_document(public_key_der: bytes) -> Dict[str, Any]:
    """The public document served by ``GET /recipient``.

    ``key_spec`` is derived from the key size; production keys are
    ``RSA_4096`` (``KmsDecryptor`` pins that), tests may use ``RSA_2048``.
    """

    der = bytes(public_key_der)
    key = _load_rsa_public_key(der)
    return {
        "schema_version": RECIPIENT_DOCUMENT_SCHEMA_VERSION,
        "algorithm": RECIPIENT_ALGORITHM,
        "key_spec": RECIPIENT_KEY_SPEC_BY_BITS[key.key_size],
        "public_key_der_b64": base64.b64encode(der).decode("ascii"),
        "public_key_hash": recipient_key_hash(der),
    }


def validate_recipient_document(document: Any) -> bytes:
    """Validate a recipient document and return the DER public key bytes.

    Fails closed on unknown fields, a different schema or algorithm, a
    ``key_spec`` that does not match the decoded key, bad base64, a non-RSA
    key, or a hash that does not match the key bytes.
    """

    validated = validate_document(document, RECIPIENT_DOCUMENT_FIELDS)
    try:
        der = base64.b64decode(validated["public_key_der_b64"], validate=True)
    except (ValueError, TypeError) as exc:
        raise ArenaContractError("recipient public key is not base64") from exc
    key = _load_rsa_public_key(der)
    if RECIPIENT_KEY_SPEC_BY_BITS[key.key_size] != validated["key_spec"]:
        raise ArenaContractError("recipient key_spec does not match the public key")
    if recipient_key_hash(der) != validated["public_key_hash"]:
        raise ArenaContractError("recipient public key hash mismatch")
    return der


# ---------------------------------------------------------------------------
# Miner-side envelope helper (no network)
# ---------------------------------------------------------------------------


def encrypt_runtime_key(recipient: Mapping[str, Any], raw_key: str, *, provider: str = "openrouter") -> Dict[str, Any]:
    """Encrypt a provider runtime key to the Arena recipient; returns the envelope.

    The envelope carries the provider name, the ciphertext, the recipient key
    hash, and the local sha256 commitment of the key. It is what the miner
    submits, once per provider.
    """

    der = validate_recipient_document(recipient)
    key = validate_provider_key_format(provider, raw_key)
    public_key = _load_rsa_public_key(der)
    modulus_bytes = public_key.key_size // 8
    plaintext = key.encode("utf-8")
    if len(plaintext) > modulus_bytes - 2 * hashes.SHA256.digest_size - 2:
        raise ProviderKeyError("runtime key is too long for the recipient key")
    ciphertext = public_key.encrypt(plaintext, _OAEP_PADDING)
    return {
        "schema_version": ENVELOPE_SCHEMA_VERSION,
        "algorithm": RECIPIENT_ALGORITHM,
        "provider": provider,
        "recipient_key_hash": recipient_key_hash(der),
        "key_hash": _local_key_hash(key),
        "ciphertext_b64": base64.b64encode(ciphertext).decode("ascii"),
    }


def validate_envelope_shape(
    envelope: Any,
    expected_recipient_key_hash: str,
    *,
    modulus_bytes: Optional[int] = None,
) -> Dict[str, Any]:
    """Strictly validate an envelope without decrypting anything.

    Returns the validated envelope with ``ciphertext`` (bytes) added. The
    ciphertext length must equal the RSA modulus size: exactly
    ``modulus_bytes`` when the caller knows the recipient key, otherwise one
    of the supported modulus sizes.
    """

    if not isinstance(envelope, Mapping):
        raise ArenaContractError("envelope must be an object")
    check_strict_document(envelope, REQUEST_LIMITS)
    validated = validate_document(envelope, ENVELOPE_FIELDS)
    require_sha256(expected_recipient_key_hash, "expected_recipient_key_hash")
    if validated["recipient_key_hash"] != expected_recipient_key_hash:
        raise ArenaContractError("envelope is encrypted to a different recipient key")
    if not _OPENROUTER_KEY_HASH_RE.fullmatch(validated["key_hash"]):
        raise ArenaContractError("envelope key_hash must be 64 lowercase hex characters")
    try:
        ciphertext = base64.b64decode(validated["ciphertext_b64"], validate=True)
    except (ValueError, TypeError) as exc:
        raise ArenaContractError("envelope ciphertext is not base64") from exc
    if modulus_bytes is not None:
        if modulus_bytes not in RECIPIENT_MODULUS_BYTES:
            raise ArenaContractError("unsupported recipient modulus size")
        if len(ciphertext) != modulus_bytes:
            raise ArenaContractError("envelope ciphertext length does not match the recipient key")
    elif len(ciphertext) not in RECIPIENT_MODULUS_BYTES:
        raise ArenaContractError("envelope ciphertext length is not an RSA modulus size")
    out = dict(validated)
    out["ciphertext"] = ciphertext
    return out


# ---------------------------------------------------------------------------
# Broker-side decryption
# ---------------------------------------------------------------------------


class Decryptor(Protocol):
    """Decrypts one RSA-OAEP(SHA-256) ciphertext. Only the broker holds one."""

    recipient_key_hash: str
    modulus_bytes: int

    def decrypt(self, ciphertext: bytes) -> bytes:  # pragma: no cover - interface
        ...


class LocalRsaDecryptor:
    """Test-only decryptor around an in-memory RSA private key."""

    def __init__(self, private_key: rsa.RSAPrivateKey) -> None:
        if not isinstance(private_key, rsa.RSAPrivateKey):
            raise ArenaContractError("LocalRsaDecryptor needs an RSA private key")
        self._private_key = private_key
        self.public_key_der = private_key.public_key().public_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        _load_rsa_public_key(self.public_key_der)
        self.modulus_bytes = private_key.key_size // 8
        self.recipient_key_hash = recipient_key_hash(self.public_key_der)

    def decrypt(self, ciphertext: bytes) -> bytes:
        return self._private_key.decrypt(bytes(ciphertext), _OAEP_PADDING)

    def __repr__(self) -> str:
        return "LocalRsaDecryptor(recipient_key_hash=%s)" % self.recipient_key_hash


class KmsDecryptor:
    """Production decryptor: AWS KMS ``Decrypt`` with ``RSAES_OAEP_SHA_256``.

    Only the broker container's IAM identity may call decrypt on this key
    (section 7.3). ``boto3`` is imported lazily so no other path needs AWS.
    The key must be an ``RSA_4096`` ``ENCRYPT_DECRYPT`` key.
    """

    def __init__(self, key_id: str, *, client: Any = None, region_name: Optional[str] = None) -> None:
        if client is None:
            import boto3  # noqa: WPS433 - lazy production import

            client = boto3.client("kms", region_name=region_name)
        self._client = client
        self.key_id = str(key_id)
        described = self._client.get_public_key(KeyId=self.key_id)
        if described.get("KeyUsage") != "ENCRYPT_DECRYPT":
            raise ProviderKeyError("Arena recipient key must be an ENCRYPT_DECRYPT key")
        if described.get("KeySpec") != RECIPIENT_KEY_SPEC:
            raise ProviderKeyError("Arena recipient key must be %s" % RECIPIENT_KEY_SPEC)
        if RECIPIENT_ALGORITHM not in tuple(described.get("EncryptionAlgorithms") or ()):
            raise ProviderKeyError("Arena recipient key does not support %s" % RECIPIENT_ALGORITHM)
        der = bytes(described["PublicKey"])
        key = _load_rsa_public_key(der)
        if RECIPIENT_KEY_SPEC_BY_BITS[key.key_size] != RECIPIENT_KEY_SPEC:
            raise ProviderKeyError("Arena recipient public key size does not match its key spec")
        self.public_key_der = der
        self.modulus_bytes = key.key_size // 8
        self.recipient_key_hash = recipient_key_hash(der)

    def decrypt(self, ciphertext: bytes) -> bytes:
        response = self._client.decrypt(
            KeyId=self.key_id,
            CiphertextBlob=bytes(ciphertext),
            EncryptionAlgorithm=RECIPIENT_ALGORITHM,
        )
        return bytes(response["Plaintext"])

    def __repr__(self) -> str:
        return "KmsDecryptor(key_id=%s, recipient_key_hash=%s)" % (self.key_id, self.recipient_key_hash)


class RuntimeKeyHandle:
    """A decrypted runtime key that lives only in broker process memory.

    The plaintext is captured in a closure: it is not an attribute, there is
    no ``__dict__``, ``repr``/``str`` show only the key hash, and the handle
    refuses pickling and copying. ``bearer_header()`` is the only way out and
    is meant solely for the broker's outbound OpenRouter request. ``revoke``
    drops the closure at the end of a run.
    """

    __slots__ = ("_reveal", "key_hash", "provider")

    def __init__(self, raw_key: str, provider: str = "openrouter") -> None:
        validated = validate_provider_key_format(provider, raw_key)
        self.key_hash = _local_key_hash(validated)
        self.provider = provider

        def reveal() -> str:
            return validated

        self._reveal = reveal

    def bearer_header(self) -> Dict[str, str]:
        return {"Authorization": "Bearer " + self._reveal()}

    def secret(self) -> str:
        """The raw key for the broker's outbound request only; never store or log it."""

        return self._reveal()

    def revoke(self) -> None:
        def revoked() -> str:
            raise ProviderKeyError("runtime key handle has been revoked")

        self._reveal = revoked

    def __repr__(self) -> str:
        return "RuntimeKeyHandle(key_hash=%s)" % self.key_hash

    __str__ = __repr__

    def __reduce__(self) -> Any:
        raise TypeError("RuntimeKeyHandle cannot be pickled or copied")

    def __copy__(self) -> Any:
        raise TypeError("RuntimeKeyHandle cannot be copied")

    def __deepcopy__(self, memo: Any) -> Any:
        raise TypeError("RuntimeKeyHandle cannot be copied")


def decrypt_runtime_key(
    envelope: Mapping[str, Any],
    decryptor: Decryptor,
    *,
    expected_recipient_key_hash: Optional[str] = None,
) -> RuntimeKeyHandle:
    """Validate the envelope, decrypt once, and wrap the key in a handle.

    The decrypted key must match the envelope's declared ``key_hash`` so a
    stored ciphertext can never be paired with a different commitment.
    """

    expected = expected_recipient_key_hash or decryptor.recipient_key_hash
    if expected != decryptor.recipient_key_hash:
        raise ArenaContractError("decryptor does not hold the expected recipient key")
    validated = validate_envelope_shape(envelope, expected, modulus_bytes=decryptor.modulus_bytes)
    try:
        plaintext = decryptor.decrypt(validated["ciphertext"])
    except ArenaContractError:
        raise
    except Exception as exc:  # noqa: BLE001 - every decrypt failure is closed
        raise ProviderKeyError("envelope decryption failed: %s" % type(exc).__name__) from None
    try:
        raw_key = bytes(plaintext).decode("utf-8")
    except UnicodeDecodeError:
        raise ProviderKeyError("decrypted runtime key is not UTF-8 text") from None
    handle = RuntimeKeyHandle(raw_key, validated["provider"])
    if handle.key_hash != validated["key_hash"]:
        raise ProviderKeyError("envelope key_hash does not match the decrypted key")
    return handle


# ---------------------------------------------------------------------------
# Registration: shape -> decrypt once -> format -> preflight -> record
# ---------------------------------------------------------------------------

PREFLIGHT_RECORD_FIELDS = (
    "key_hash",
    "provider",
    "limit_microusd",
    "limit_remaining_microusd",
    "usage_microusd",
    "observed_at",
    "preflight_status",
    "probe",
)
_MICROUSD = Decimal(1_000_000)


def usd_to_microusd(value: Any, field_name: str) -> Optional[int]:
    """Convert an OpenRouter USD amount to integer micro-USD, rounded down.

    ``None`` stays ``None``. Booleans, non-numeric strings, NaN and infinity
    fail closed. ``ROUND_FLOOR`` is used for every field so a remaining limit
    is never over-stated by rounding.
    """

    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float, str, Decimal)):
        raise ProviderKeyError("OpenRouter key preflight returned a non-numeric %s" % field_name)
    try:
        amount = Decimal(str(value).strip())
    except (InvalidOperation, ValueError):
        raise ProviderKeyError("OpenRouter key preflight returned a non-numeric %s" % field_name) from None
    if not amount.is_finite():
        raise ProviderKeyError("OpenRouter key preflight returned a non-finite %s" % field_name)
    return int((amount * _MICROUSD).to_integral_value(rounding=ROUND_FLOOR))


def _utc_now_iso(now: Optional[Callable[[], datetime]]) -> str:
    moment = now() if now is not None else datetime.now(timezone.utc)
    if moment.tzinfo is None:
        raise ArenaContractError("clock must return a timezone-aware datetime")
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def preflight_record(
    handle: RuntimeKeyHandle,
    *,
    urlopen: Callable[..., Any],
    timeout_seconds: int = 12,
    now: Optional[Callable[[], datetime]] = None,
) -> Dict[str, Any]:
    """Run the bounded preflight for a handle and return the non-secret record.

    The broker reruns this at each stage start (section 7.3). ``key_hash`` is
    the local sha256 commitment that the account row stores; a ``limit`` of
    ``None`` is an unlimited key and yields ``None`` for the remaining limit.
    """

    if handle.provider != "openrouter":
        probe = preflight_provider_key(handle.provider, handle._reveal(), timeout_seconds=timeout_seconds, urlopen=urlopen)
        record = {
            "key_hash": handle.key_hash,
            "provider": handle.provider,
            "limit_microusd": None,
            "limit_remaining_microusd": None,
            "usage_microusd": None,
            "observed_at": _utc_now_iso(now),
            "preflight_status": "ok",
            "probe": probe,
        }
        check_strict_document(record, REQUEST_LIMITS)
        return record
    metadata = preflight_openrouter_key(
        handle._reveal(),
        timeout_seconds=timeout_seconds,
        urlopen=urlopen,
    )
    limit_microusd = usd_to_microusd(metadata.get("limit"), "limit")
    if limit_microusd is None:
        limit_remaining_microusd: Optional[int] = None
    else:
        remaining = metadata.get("limit_remaining")
        if remaining is None:
            raise ProviderKeyError("OpenRouter key preflight returned a limit without a remaining balance")
        limit_remaining_microusd = usd_to_microusd(remaining, "limit_remaining")
    usage = metadata.get("usage")
    if usage is None:
        raise ProviderKeyError("OpenRouter key preflight returned no usage")
    usage_microusd = usd_to_microusd(usage, "usage")
    record = {
        "key_hash": handle.key_hash,
        "provider": "openrouter",
        "limit_microusd": limit_microusd,
        "limit_remaining_microusd": limit_remaining_microusd,
        "usage_microusd": usage_microusd,
        "observed_at": _utc_now_iso(now),
        "preflight_status": "ok",
        "probe": {"is_free_tier": metadata.get("is_free_tier")},
    }
    check_strict_document(record, REQUEST_LIMITS)
    return record


def register_provider_key(
    envelope: Mapping[str, Any],
    *,
    decryptor: Decryptor,
    urlopen: Callable[..., Any],
    expected_recipient_key_hash: str,
    timeout_seconds: int = 12,
    now: Optional[Callable[[], datetime]] = None,
) -> Dict[str, Any]:
    """Register a miner's envelope: validate, decrypt once, preflight, record.

    Any failure raises and no partial record exists. The returned record
    contains only ``PREFLIGHT_RECORD_FIELDS``; the plaintext is dropped when
    this function returns.
    """

    require_sha256(expected_recipient_key_hash, "expected_recipient_key_hash")
    handle = decrypt_runtime_key(envelope, decryptor, expected_recipient_key_hash=expected_recipient_key_hash)
    try:
        return preflight_record(handle, urlopen=urlopen, timeout_seconds=timeout_seconds, now=now)
    finally:
        handle.revoke()


register_openrouter_key = register_provider_key  # historical name


__all__ = [
    "Decryptor",
    "ENVELOPE_FIELDS",
    "ENVELOPE_SCHEMA_VERSION",
    "KmsDecryptor",
    "LocalRsaDecryptor",
    "OPENROUTER_KEY_INFO_URL",
    "OPENROUTER_KEY_RE",
    "ProviderKeyError",
    "PREFLIGHT_RECORD_FIELDS",
    "RECIPIENT_ALGORITHM",
    "RECIPIENT_DOCUMENT_FIELDS",
    "RECIPIENT_KEY_SPEC",
    "RuntimeKeyHandle",
    "STRICT_OPENROUTER_PROVIDER_POLICY",
    "decrypt_runtime_key",
    "encrypt_runtime_key",
    "preflight_openrouter_key",
    "preflight_record",
    "recipient_document",
    "recipient_key_hash",
    "register_openrouter_key",
    "register_provider_key",
    "preflight_provider_key",
    "validate_provider_key_format",
    "ProviderKeyError",
    "PROVIDER_KEY_PATTERNS",
    "strict_openrouter_provider_policy",
    "usd_to_microusd",
    "validate_envelope_shape",
    "validate_openrouter_key_format",
    "validate_recipient_document",
]
