import base64
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import hashlib
from pathlib import Path
import ssl

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from gateway.tee.mtls_identity import (
    ATTESTED_TLS_CERTIFICATE_LIFETIME,
    MutualAttestationError,
    create_mutual_tls_context,
    generate_ephemeral_tls_identity,
    verify_peer_certificate_binding,
    write_identity_to_tmpfs,
)
from leadpoet_canonical.attested_v2 import (
    COORDINATOR_ROLE,
    SCORING_ROLE,
    build_boot_identity_body,
    create_boot_identity,
)


HASH = "sha256:" + "a" * 64


def _boot(role, transport_pubkey, certificate_hash):
    signing_key = Ed25519PrivateKey.generate()
    signing_pubkey = signing_key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    ).hex()
    return create_boot_identity(
        body=build_boot_identity_body(
            role=role,
            physical_role=(
                "gateway_scoring"
                if role == SCORING_ROLE
                else "gateway_coordinator"
            ),
            commit_sha="b" * 40,
            pcr0="c" * 96,
            build_manifest_hash=HASH,
            dependency_lock_hash=HASH,
            config_hash=HASH,
            boot_nonce="d" * 32,
            signing_pubkey=signing_pubkey,
            transport_pubkey=transport_pubkey,
            transport_certificate_hash=certificate_hash,
            attestation_user_data_hash=HASH,
            issued_at="2026-07-10T20:00:00Z",
        ),
        attestation_document_b64=base64.b64encode(b"nitro").decode("ascii"),
    )


def _certificate_der(identity):
    return x509.load_pem_x509_certificate(identity["certificate_pem"]).public_bytes(
        serialization.Encoding.DER
    )


def _certificate_time(certificate, field):
    aware = getattr(certificate, field + "_utc", None)
    if aware is not None:
        return aware
    return getattr(certificate, field).replace(tzinfo=timezone.utc)


def test_tls_certificate_is_bound_to_attested_transport_key_and_role():
    identity = generate_ephemeral_tls_identity(service_role=SCORING_ROLE)
    boot = _boot(
        SCORING_ROLE,
        identity["transport_pubkey"],
        identity["certificate_sha256"],
    )
    verified = verify_peer_certificate_binding(
        boot_identity=boot,
        certificate_der=_certificate_der(identity),
        expected_service_role=SCORING_ROLE,
    )
    assert verified["transport_pubkey"] == identity["transport_pubkey"]
    assert verified["certificate_sha256"] == identity["certificate_sha256"]


def test_attested_tls_certificate_outlives_long_running_enclave_boot():
    identity = generate_ephemeral_tls_identity(service_role=SCORING_ROLE)
    certificate = x509.load_pem_x509_certificate(identity["certificate_pem"])
    not_before = _certificate_time(certificate, "not_valid_before")
    not_after = _certificate_time(certificate, "not_valid_after")

    assert not_after - not_before >= ATTESTED_TLS_CERTIFICATE_LIFETIME
    assert not_after > datetime.now(timezone.utc) + timedelta(days=3650) - timedelta(
        minutes=2
    )


def test_expired_attested_tls_certificate_still_fails_closed():
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key()
    now = datetime.now(timezone.utc)
    subject = x509.Name([x509.NameAttribute(x509.oid.NameOID.COMMON_NAME, SCORING_ROLE)])
    certificate = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(subject)
        .public_key(public_key)
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(days=2))
        .not_valid_after(now - timedelta(days=1))
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .sign(private_key, algorithm=None)
    )
    certificate_der = certificate.public_bytes(serialization.Encoding.DER)
    certificate_hash = "sha256:" + hashlib.sha256(certificate_der).hexdigest()
    public_key_hex = public_key.public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    ).hex()
    boot = _boot(SCORING_ROLE, public_key_hex, certificate_hash)

    with pytest.raises(MutualAttestationError, match="outside validity window"):
        verify_peer_certificate_binding(
            boot_identity=boot,
            certificate_der=certificate_der,
            expected_service_role=SCORING_ROLE,
        )


def test_parent_cannot_substitute_a_different_valid_certificate():
    expected = generate_ephemeral_tls_identity(service_role=SCORING_ROLE)
    substituted = generate_ephemeral_tls_identity(service_role=SCORING_ROLE)
    boot = _boot(
        SCORING_ROLE,
        expected["transport_pubkey"],
        expected["certificate_sha256"],
    )
    with pytest.raises(MutualAttestationError, match="differs"):
        verify_peer_certificate_binding(
            boot_identity=boot,
            certificate_der=_certificate_der(substituted),
            expected_service_role=SCORING_ROLE,
        )


def test_certificate_cannot_be_reused_for_another_enclave_role():
    identity = generate_ephemeral_tls_identity(service_role=SCORING_ROLE)
    boot = _boot(
        SCORING_ROLE,
        identity["transport_pubkey"],
        identity["certificate_sha256"],
    )
    with pytest.raises(MutualAttestationError, match="boot identity role"):
        verify_peer_certificate_binding(
            boot_identity=boot,
            certificate_der=_certificate_der(identity),
            expected_service_role=COORDINATOR_ROLE,
        )


def test_mutual_tls_context_requires_exact_peer_and_tls13(tmp_path: Path):
    coordinator = generate_ephemeral_tls_identity(service_role=COORDINATOR_ROLE)
    scoring = generate_ephemeral_tls_identity(service_role=SCORING_ROLE)
    paths = write_identity_to_tmpfs(coordinator, directory=tmp_path / "coordinator")
    kwargs = {
        "identity_paths": paths,
        "trusted_peer_certificate_pem": scoring["certificate_pem"],
        "server_side": False,
    }
    if not ssl.HAS_TLSv1_3:
        with pytest.raises(MutualAttestationError, match="TLS 1.3"):
            create_mutual_tls_context(**kwargs)
        return
    context = create_mutual_tls_context(**kwargs)
    assert context.minimum_version == ssl.TLSVersion.TLSv1_3
    assert context.maximum_version == ssl.TLSVersion.TLSv1_3
    assert context.verify_mode == ssl.CERT_REQUIRED
    assert context.check_hostname is False
    assert (tmp_path / "coordinator" / "tls-private-key.pem").stat().st_mode & 0o777 == 0o600


def test_identity_materialization_is_atomic_and_idempotent(tmp_path, monkeypatch):
    identity = generate_ephemeral_tls_identity(service_role=COORDINATOR_ROLE)
    directory = tmp_path / "coordinator"

    def reject_direct_overwrite(_path, _payload):
        raise AssertionError("identity files must not be overwritten in place")

    monkeypatch.setattr(Path, "write_bytes", reject_direct_overwrite)

    def materialize(_index):
        paths = write_identity_to_tmpfs(identity, directory=directory)
        return (
            paths["certificate"].read_bytes(),
            paths["private_key"].read_bytes(),
        )

    with ThreadPoolExecutor(max_workers=16) as executor:
        observed = list(executor.map(materialize, range(64)))

    assert set(observed) == {
        (identity["certificate_pem"], identity["private_key_pem"])
    }
    assert not list(directory.glob(".tls-*.pem.*"))
    assert (directory / "tls-certificate.pem").stat().st_mode & 0o777 == 0o600
    assert (directory / "tls-private-key.pem").stat().st_mode & 0o777 == 0o600
