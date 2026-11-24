"""Decide whether the live vault can be trusted, and talk only to it.

Used by the miner CLI and by Leadpoet's gateway alike, so both trust exactly
the same thing: a vault whose keys chain to Phala's KMS root for the app in
``deployment.json``, reached over a TLS session that ends inside it.

The app id and the KMS root key in ``deployment.json`` can be compared with
the on-chain records on Base, so the check needs nothing from Leadpoet.
"""

from __future__ import annotations

import http.client
import json
import ssl
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple
from urllib.parse import urlsplit

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec

from . import chain
from .vault import ENCRYPTION_DOMAIN, KEY_ALGORITHM, RECEIPTS_DOMAIN, TLS_SCHEMA, canonical, spki_sha256

DEPLOYMENT_FILE = Path(__file__).with_name("deployment.json")
_DEPLOYMENT_FIELDS = ("url", "app_id", "kms_root_public_key", "kms_contract", "chain")


class TrustError(Exception):
    """The vault, or something it signed, cannot be trusted."""


def load_deployment(path: Path = DEPLOYMENT_FILE) -> Mapping:
    deployment = json.loads(path.read_text())
    missing = [name for name in _DEPLOYMENT_FIELDS if not deployment.get(name)]
    if missing:
        raise TrustError("The vault is not deployed yet (deployment.json is missing %s)." % ", ".join(missing))
    parts = urlsplit(str(deployment["url"]))
    if parts.scheme != "https" or not parts.hostname or parts.path not in ("", "/") or parts.query:
        raise TrustError("deployment.json url must be https with no path")
    return deployment


def tls_context(certificate_pem: Optional[str]) -> ssl.SSLContext:
    """Accept only ``certificate_pem``, or, with None, any certificate, which
    ``connect`` then checks against the vault's signed identity."""

    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.minimum_version = ssl.TLSVersion.TLSv1_3
    # The vault's certificate is self-signed and names no host; it is trusted
    # only as the exact key its identity signed.
    context.check_hostname = False
    if certificate_pem is None:
        context.verify_mode = ssl.CERT_NONE
    else:
        context.verify_flags |= ssl.VERIFY_X509_PARTIAL_CHAIN
        context.load_verify_locations(cadata=certificate_pem)
    return context


def connect(deployment: Mapping) -> Tuple[Mapping, str]:
    """The verified identity and the vault's certificate, from one TLS session.

    The identity is read over the same session whose certificate it must
    sign, so the session provably ends inside the vault.
    """

    parts = urlsplit(str(deployment["url"]))
    connection = http.client.HTTPSConnection(parts.hostname, parts.port or 443, timeout=30,
                                             context=tls_context(None))
    try:
        connection.request("GET", "/v1/identity")
        certificate = connection.sock.getpeercert(binary_form=True) if connection.sock else None
        if not certificate:
            raise TrustError("The vault sent no certificate.")
        response = connection.getresponse()
        if response.status != 200:
            raise TrustError("The vault's identity answered %d." % response.status)
        identity = json.loads(response.read(65536))
    except http.client.HTTPException:
        raise TrustError("The vault's identity answer is not HTTP.") from None
    finally:
        connection.close()
    identity = verified_identity(deployment, identity)
    tls = verify_signed(identity, identity.get("tls"))
    if tls != {"schema": TLS_SCHEMA, "boot_id": identity.get("boot_id"), "spki_sha256": spki_sha256(certificate)}:
        raise TrustError("The TLS session does not end in the vault (a gateway or proxy ended it).")
    return identity, ssl.DER_cert_to_PEM_cert(certificate)


def verified_identity(deployment: Mapping, identity: Any) -> Mapping:
    """Return the identity only if both vault keys chain to the KMS root for our app."""

    if not isinstance(identity, Mapping):
        raise TrustError("The vault's identity is not an object.")
    app_id = str(deployment["app_id"]).lower().removeprefix("0x")
    if str(identity.get("app_id", "")).lower().removeprefix("0x") != app_id:
        raise TrustError("The vault reports a different app id than deployment.json.")
    for name, domain in (("encryption", ENCRYPTION_DOMAIN), ("receipts", RECEIPTS_DOMAIN)):
        key = identity.get(name)
        if not isinstance(key, Mapping) or key.get("domain") != domain or key.get("algorithm") != KEY_ALGORITHM:
            raise TrustError("The vault's %s key is not the expected kind." % name)
        try:
            chain.verify_key(algorithm=KEY_ALGORITHM, domain=domain, public_key_hex=str(key.get("public_key", "")),
                             signature_chain=list(key.get("signature_chain") or []), app_id_hex=app_id,
                             kms_root_public_key_hex=str(deployment["kms_root_public_key"]))
        except chain.ChainError as error:
            raise TrustError("The vault's %s key does not chain to the KMS root (%s)." % (name, error)) from None
    return identity


def verify_signed(identity: Mapping, signed: Any) -> Mapping:
    """The document of ``signed`` if the verified identity's receipts key signed it.

    Never trust the ``public_key`` a signed document carries itself.
    """

    if not isinstance(signed, Mapping) or not isinstance(signed.get("document"), Mapping):
        raise TrustError("A signed vault document is malformed.")
    try:
        key = ec.EllipticCurvePublicKey.from_encoded_point(ec.SECP256K1(),
                                                           bytes.fromhex(identity["receipts"]["public_key"]))
        key.verify(bytes.fromhex(str(signed.get("signature"))), canonical(signed["document"]),
                   ec.ECDSA(hashes.SHA256()))
    except (InvalidSignature, ValueError):
        raise TrustError("A vault document is not signed by the vault's receipts key.") from None
    return signed["document"]
