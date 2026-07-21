"""Client verification and encryption for attested credential recipients."""

from __future__ import annotations

import base64
from datetime import datetime, timezone
import hashlib
import json
import re
from typing import Any, Mapping
from urllib.parse import parse_qs, unquote, urlsplit
from urllib.request import Request, build_opener, HTTPRedirectHandler

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from leadpoet_canonical.attested_v2 import canonical_json, sha256_json
from leadpoet_canonical.nitro import verify_nitro_attestation_full


OPENROUTER_INGRESS_RECIPIENT_SCHEMA_VERSION = (
    "leadpoet.openrouter_ingress_recipient.v2"
)
OPENROUTER_INGRESS_RECIPIENT_PURPOSE = (
    "leadpoet.openrouter_credential_ingress.v2"
)
KMS_KEY_ENCRYPTION_ALGORITHM = "RSAES_OAEP_SHA_256"
OPENROUTER_RELEASE_EVIDENCE_SCHEMA_VERSION = (
    "leadpoet.openrouter_release_evidence.v2"
)
_RELEASE_BUCKET = "leadpoet-attested-v2-artifacts-493765492819"
_RELEASE_PREFIX = "attested-v2/releases"
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_MAX_RELEASE_CHANNEL_BYTES = 2 * 1024 * 1024

_REQUIRED_FIELDS = {
    "schema_version",
    "purpose",
    "request_id",
    "boot_identity_hash",
    "miner_hotkey_hash",
    "credential_kind",
    "credential_slot",
    "recipient_public_key_hash",
    "request_nonce",
    "recipient_public_key_der_b64",
    "attestation_document_b64",
    "key_encryption_algorithm",
}
_CLAIM_FIELDS = (
    "schema_version",
    "purpose",
    "boot_identity_hash",
    "miner_hotkey_hash",
    "credential_kind",
    "credential_slot",
    "recipient_public_key_hash",
    "request_nonce",
)


class CredentialRecipientV2Error(ValueError):
    """An attested recipient is invalid or cannot safely carry a credential."""


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _release_url(
    value: str,
    *,
    commit: str,
    version_id: str,
) -> str:
    parsed = urlsplit(str(value or ""))
    allowed_hosts = {
        f"{_RELEASE_BUCKET}.s3.amazonaws.com",
        f"{_RELEASE_BUCKET}.s3.us-east-1.amazonaws.com",
    }
    expected_path = f"/{_RELEASE_PREFIX}/{commit}/release-channel-v2.json"
    if (
        parsed.scheme != "https"
        or parsed.hostname not in allowed_hosts
        or parsed.port not in (None, 443)
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
        or unquote(parsed.path) != expected_path
    ):
        raise CredentialRecipientV2Error(
            "OpenRouter release verification URL violates policy"
        )
    query = {key.lower(): values for key, values in parse_qs(parsed.query).items()}
    required = {
        "x-amz-algorithm",
        "x-amz-credential",
        "x-amz-date",
        "x-amz-expires",
        "x-amz-signedheaders",
        "x-amz-signature",
        "versionid",
    }
    allowed = required | {"x-amz-security-token"}
    if set(query) - allowed or not required.issubset(query):
        raise CredentialRecipientV2Error(
            "OpenRouter release verification URL is not an exact S3 signature"
        )
    if (
        query["x-amz-algorithm"] != ["AWS4-HMAC-SHA256"]
        or query["versionid"] != [version_id]
        or "host" not in str(query["x-amz-signedheaders"][0]).split(";")
    ):
        raise CredentialRecipientV2Error(
            "OpenRouter release verification URL signature policy differs"
        )
    try:
        expires = int(query["x-amz-expires"][0])
    except (TypeError, ValueError, IndexError) as exc:
        raise CredentialRecipientV2Error(
            "OpenRouter release verification URL expiry is invalid"
        ) from exc
    if not 1 <= expires <= 300:
        raise CredentialRecipientV2Error(
            "OpenRouter release verification URL expiry exceeds policy"
        )
    return parsed.geturl()


def _open_exact_url(url: str, *, method: str):
    opener = build_opener(_NoRedirect())
    return opener.open(Request(url, method=method), timeout=30)


def verify_openrouter_credential_release_v2(
    evidence: Mapping[str, Any],
    *,
    http_open: Any = None,
) -> dict[str, Any]:
    """Verify the coordinator against its exact immutable V2 release."""

    fields = {
        "schema_version",
        "coordinator_boot_identity",
        "release_channel_version_id",
        "release_channel_get_url",
        "release_channel_head_url",
    }
    if not isinstance(evidence, Mapping) or set(evidence) != fields:
        raise CredentialRecipientV2Error(
            "OpenRouter release evidence fields are invalid"
        )
    if evidence.get("schema_version") != OPENROUTER_RELEASE_EVIDENCE_SCHEMA_VERSION:
        raise CredentialRecipientV2Error("OpenRouter release evidence policy differs")
    boot = evidence.get("coordinator_boot_identity")
    if not isinstance(boot, Mapping):
        raise CredentialRecipientV2Error("OpenRouter coordinator boot identity is invalid")
    commit = str(boot.get("commit_sha") or "").lower()
    if not _COMMIT_RE.fullmatch(commit):
        raise CredentialRecipientV2Error("OpenRouter coordinator commit is invalid")
    version_id = str(evidence.get("release_channel_version_id") or "")
    if not version_id or len(version_id) > 1024:
        raise CredentialRecipientV2Error("OpenRouter release version is invalid")
    get_url = _release_url(
        str(evidence["release_channel_get_url"]),
        commit=commit,
        version_id=version_id,
    )
    head_url = _release_url(
        str(evidence["release_channel_head_url"]),
        commit=commit,
        version_id=version_id,
    )
    open_url = http_open or _open_exact_url
    try:
        with open_url(head_url, method="HEAD") as response:
            headers = {key.lower(): value for key, value in response.headers.items()}
            if response.geturl() != head_url:
                raise CredentialRecipientV2Error(
                    "OpenRouter release verification redirected"
                )
        with open_url(get_url, method="GET") as response:
            get_headers = {
                key.lower(): value for key, value in response.headers.items()
            }
            if response.geturl() != get_url:
                raise CredentialRecipientV2Error(
                    "OpenRouter release verification redirected"
                )
            payload = response.read(_MAX_RELEASE_CHANNEL_BYTES + 1)
    except CredentialRecipientV2Error:
        raise
    except Exception as exc:
        raise CredentialRecipientV2Error(
            "OpenRouter immutable release channel is unavailable"
        ) from exc
    if (
        headers.get("x-amz-object-lock-mode", "").upper() != "COMPLIANCE"
        or headers.get("x-amz-version-id") != version_id
        or get_headers.get("x-amz-version-id") != version_id
    ):
        raise CredentialRecipientV2Error(
            "OpenRouter release channel is not the Object-Locked version"
        )
    try:
        retain_until = datetime.fromisoformat(
            headers["x-amz-object-lock-retain-until-date"].replace("Z", "+00:00")
        )
    except (KeyError, ValueError) as exc:
        raise CredentialRecipientV2Error(
            "OpenRouter release retention is invalid"
        ) from exc
    if retain_until <= datetime.now(timezone.utc):
        raise CredentialRecipientV2Error("OpenRouter release retention has expired")
    if not payload or len(payload) > _MAX_RELEASE_CHANNEL_BYTES:
        raise CredentialRecipientV2Error("OpenRouter release channel size is invalid")
    try:
        channel_value = json.loads(payload)
        from gateway.tee.release_channel_v2 import validate_release_channel_v2
        from gateway.tee.release_manifest_v2 import role_expectation

        channel = validate_release_channel_v2(
            channel_value,
            expected_commit=commit,
        )
        expectation = role_expectation(
            channel["gateway_release_manifest"],
            "gateway_coordinator",
        )
        from leadpoet_canonical.attested_v2 import (
            validate_boot_identity,
            verify_boot_identity_nitro,
        )

        validate_boot_identity(boot)
        for field in (
            "physical_role",
            "commit_sha",
            "pcr0",
            "build_manifest_hash",
            "dependency_lock_hash",
        ):
            if boot.get(field) != expectation[field]:
                raise CredentialRecipientV2Error(
                    f"OpenRouter coordinator boot {field} differs from release"
                )
        verify_boot_identity_nitro(
            boot,
            expected_pcr0=expectation["pcr0"],
            certificate_validity_at_attestation_time=True,
        )
    except CredentialRecipientV2Error:
        raise
    except Exception as exc:
        raise CredentialRecipientV2Error(
            "OpenRouter coordinator release verification failed"
        ) from exc
    return dict(boot)


def verify_and_encrypt_openrouter_credential_v2(
    recipient: Mapping[str, Any],
    credential: str,
    *,
    miner_hotkey: str,
    credential_kind: str,
    verified_coordinator_boot_identity: Mapping[str, Any],
) -> dict[str, str]:
    """Verify a one-use Nitro recipient and encrypt one credential locally."""

    expected_slot = {
        "runtime": "openrouter",
        "management": "openrouter_management",
    }.get(credential_kind)
    if expected_slot is None:
        raise CredentialRecipientV2Error("OpenRouter credential kind is invalid")
    if not isinstance(recipient, Mapping) or set(recipient) != _REQUIRED_FIELDS:
        raise CredentialRecipientV2Error("OpenRouter recipient fields are invalid")
    if (
        recipient.get("schema_version")
        != OPENROUTER_INGRESS_RECIPIENT_SCHEMA_VERSION
        or recipient.get("purpose") != OPENROUTER_INGRESS_RECIPIENT_PURPOSE
        or recipient.get("key_encryption_algorithm")
        != KMS_KEY_ENCRYPTION_ALGORITHM
        or recipient.get("credential_kind") != credential_kind
        or recipient.get("credential_slot") != expected_slot
    ):
        raise CredentialRecipientV2Error("OpenRouter recipient policy is invalid")

    expected_miner_hash = "sha256:" + hashlib.sha256(
        miner_hotkey.encode("utf-8")
    ).hexdigest()
    if recipient.get("miner_hotkey_hash") != expected_miner_hash:
        raise CredentialRecipientV2Error("OpenRouter recipient miner binding differs")
    if (
        not isinstance(verified_coordinator_boot_identity, Mapping)
        or recipient.get("boot_identity_hash")
        != verified_coordinator_boot_identity.get("boot_identity_hash")
    ):
        raise CredentialRecipientV2Error(
            "OpenRouter recipient boot identity differs from verified release"
        )

    encoded_credential = credential.encode("utf-8")
    if not encoded_credential or b"\x00" in encoded_credential:
        raise CredentialRecipientV2Error("OpenRouter credential is empty or invalid")
    try:
        public_der = base64.b64decode(
            str(recipient["recipient_public_key_der_b64"]), validate=True
        )
    except Exception as exc:
        raise CredentialRecipientV2Error(
            "OpenRouter recipient public key is invalid"
        ) from exc
    public_hash = "sha256:" + hashlib.sha256(public_der).hexdigest()
    if public_hash != recipient.get("recipient_public_key_hash"):
        raise CredentialRecipientV2Error(
            "OpenRouter recipient public key hash differs"
        )

    claim = {name: recipient[name] for name in _CLAIM_FIELDS}
    request_id = sha256_json(claim)
    if request_id != recipient.get("request_id"):
        raise CredentialRecipientV2Error("OpenRouter recipient claim hash differs")

    valid, attestation = verify_nitro_attestation_full(
        attestation_b64=str(recipient["attestation_document_b64"]),
        expected_pcr0=str(verified_coordinator_boot_identity.get("pcr0") or ""),
        expected_pubkey=None,
        expected_purpose=OPENROUTER_INGRESS_RECIPIENT_PURPOSE,
        role="gateway",
    )
    if not valid:
        raise CredentialRecipientV2Error(
            "OpenRouter recipient Nitro attestation verification failed"
        )
    expected_user_data = {
        "schema_version": OPENROUTER_INGRESS_RECIPIENT_SCHEMA_VERSION,
        "purpose": OPENROUTER_INGRESS_RECIPIENT_PURPOSE,
        "claim_hash": request_id,
    }
    if (
        attestation.get("attestation_public_key") != public_der.hex()
        or attestation.get("user_data") != expected_user_data
        or canonical_json(attestation.get("user_data"))
        != canonical_json(expected_user_data)
    ):
        raise CredentialRecipientV2Error(
            "OpenRouter attestation is not bound to the recipient claim"
        )

    try:
        public_key = serialization.load_der_public_key(public_der)
    except Exception as exc:
        raise CredentialRecipientV2Error(
            "OpenRouter recipient RSA key is invalid"
        ) from exc
    if not isinstance(public_key, rsa.RSAPublicKey) or public_key.key_size < 2048:
        raise CredentialRecipientV2Error(
            "OpenRouter recipient key policy is invalid"
        )
    max_plaintext = (
        (public_key.key_size // 8) - (2 * hashes.SHA256.digest_size) - 2
    )
    if len(encoded_credential) > max_plaintext:
        raise CredentialRecipientV2Error(
            "OpenRouter credential exceeds the attested RSA-OAEP limit"
        )
    ciphertext = public_key.encrypt(
        encoded_credential,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )
    return {
        "request_id": request_id,
        "ciphertext_b64": base64.b64encode(ciphertext).decode("ascii"),
    }
