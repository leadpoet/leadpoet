"""Ephemeral TLS 1.3 identities bound to attested enclave boot keys."""

from __future__ import annotations

import hashlib
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
import ssl
from typing import Any, Dict, Mapping, Sequence, Union

from cryptography import x509
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.x509.oid import NameOID

from leadpoet_canonical.attested_v2 import ROLE_PURPOSES, validate_boot_identity


class MutualAttestationError(ValueError):
    """A TLS certificate is not the key attested for the expected enclave."""


def _public_key_hex(public_key: Ed25519PublicKey) -> str:
    return public_key.public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    ).hex()


def generate_ephemeral_tls_identity(*, service_role: str) -> Dict[str, Any]:
    if service_role not in ROLE_PURPOSES:
        raise MutualAttestationError("unsupported attested TLS service role")
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key()
    now = datetime.utcnow()
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, service_role)])
    certificate = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(subject)
        .public_key(public_key)
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=1))
        .not_valid_after(now + timedelta(hours=24))
        .add_extension(
            x509.BasicConstraints(ca=True, path_length=0),
            critical=True,
        )
        .sign(private_key, algorithm=None)
    )
    certificate_der = certificate.public_bytes(serialization.Encoding.DER)
    return {
        "service_role": service_role,
        "transport_pubkey": _public_key_hex(public_key),
        "certificate_sha256": "sha256:" + hashlib.sha256(certificate_der).hexdigest(),
        "certificate_pem": certificate.public_bytes(serialization.Encoding.PEM),
        "private_key_pem": private_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        ),
    }


def verify_peer_certificate_binding(
    *,
    boot_identity: Mapping[str, Any],
    certificate_der: bytes,
    expected_service_role: str,
) -> Dict[str, str]:
    validate_boot_identity(boot_identity)
    if boot_identity.get("role") != expected_service_role:
        raise MutualAttestationError("peer boot identity role mismatch")
    try:
        certificate = x509.load_der_x509_certificate(bytes(certificate_der))
        public_key = certificate.public_key()
    except Exception as exc:
        raise MutualAttestationError("peer TLS certificate is invalid") from exc
    if not isinstance(public_key, Ed25519PublicKey):
        raise MutualAttestationError("peer TLS certificate key type is invalid")
    if _public_key_hex(public_key) != boot_identity.get("transport_pubkey"):
        raise MutualAttestationError("peer TLS key differs from attested transport key")
    certificate_hash = "sha256:" + hashlib.sha256(bytes(certificate_der)).hexdigest()
    if certificate_hash != boot_identity.get("transport_certificate_hash"):
        raise MutualAttestationError(
            "peer TLS certificate differs from attested certificate"
        )
    common_names = certificate.subject.get_attributes_for_oid(NameOID.COMMON_NAME)
    if len(common_names) != 1 or common_names[0].value != expected_service_role:
        raise MutualAttestationError("peer TLS certificate role mismatch")
    if certificate.issuer != certificate.subject:
        raise MutualAttestationError("peer TLS certificate issuer is invalid")
    try:
        public_key.verify(certificate.signature, certificate.tbs_certificate_bytes)
    except Exception as exc:
        raise MutualAttestationError("peer TLS certificate self-signature is invalid") from exc
    now = datetime.now(timezone.utc)
    not_before = getattr(certificate, "not_valid_before_utc", None)
    not_after = getattr(certificate, "not_valid_after_utc", None)
    if not_before is None:
        not_before = certificate.not_valid_before.replace(tzinfo=timezone.utc)
    if not_after is None:
        not_after = certificate.not_valid_after.replace(tzinfo=timezone.utc)
    if now < not_before or now > not_after:
        raise MutualAttestationError("peer TLS certificate is outside validity window")
    return {
        "service_role": expected_service_role,
        "transport_pubkey": _public_key_hex(public_key),
        "certificate_sha256": "sha256:"
        + hashlib.sha256(bytes(certificate_der)).hexdigest(),
    }


def write_identity_to_tmpfs(
    identity: Mapping[str, Any], *, directory: Path
) -> Dict[str, Path]:
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        directory.chmod(0o700)
    except OSError:
        pass
    certificate_path = directory / "tls-certificate.pem"
    private_key_path = directory / "tls-private-key.pem"
    certificate_path.write_bytes(bytes(identity["certificate_pem"]))
    private_key_path.write_bytes(bytes(identity["private_key_pem"]))
    certificate_path.chmod(0o600)
    private_key_path.chmod(0o600)
    return {"certificate": certificate_path, "private_key": private_key_path}


def create_mutual_tls_context(
    *,
    identity_paths: Mapping[str, Path],
    trusted_peer_certificate_pem: Union[bytes, Sequence[bytes]],
    server_side: bool,
) -> ssl.SSLContext:
    if not getattr(ssl, "HAS_TLSv1_3", False):
        raise MutualAttestationError("TLS 1.3 is unavailable in this runtime")
    purpose = ssl.Purpose.CLIENT_AUTH if server_side else ssl.Purpose.SERVER_AUTH
    context = ssl.create_default_context(purpose=purpose)
    try:
        context.minimum_version = ssl.TLSVersion.TLSv1_3
        context.maximum_version = ssl.TLSVersion.TLSv1_3
    except ValueError as exc:
        raise MutualAttestationError("TLS 1.3 is unavailable in this runtime") from exc
    context.check_hostname = False
    context.verify_mode = ssl.CERT_REQUIRED
    context.load_cert_chain(
        certfile=os.fspath(identity_paths["certificate"]),
        keyfile=os.fspath(identity_paths["private_key"]),
    )
    certificates = (
        [trusted_peer_certificate_pem]
        if isinstance(trusted_peer_certificate_pem, bytes)
        else list(trusted_peer_certificate_pem)
    )
    if not certificates:
        raise MutualAttestationError("at least one attested peer certificate is required")
    context.load_verify_locations(
        cadata="\n".join(bytes(item).decode("ascii") for item in certificates)
    )
    return context
