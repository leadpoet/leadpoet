"""Miner command: seal a provider key to the lead-order key vault.

    python -m lead_order_vault.cli verify
    python -m lead_order_vault.cli seal --provider openrouter --hotkey <ss58>
    python -m lead_order_vault.cli receipts receipts.json

``seal`` reads the key from the terminal (never from arguments, so it stays
out of shell history), checks that the vault's encryption key was issued by
Phala's KMS to the app named in ``deployment.json``, and prints an envelope.
Submit the envelope to ``POST /lead-orders/keys``. Only the vault can open it.

``receipts`` reads every page of ``GET /lead-orders/vault-receipts`` from
``since=0`` (saved files, or ``-`` for stdin), checks every receipt's signature against the
vault's receipts key, and reports any gap in each key's sequence numbers and
any key that must be sealed again because the vault's key changed.

The check needs nothing from Leadpoet: the app id and the KMS root key in
``deployment.json`` can be compared with the on-chain records on Base. It
also refuses any TLS session the vault does not end itself, so Phala's
gateway or any proxy in between sees only ciphertext.
"""

from __future__ import annotations

import argparse
import getpass
import http.client
import json
import ssl
import sys
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Tuple
from urllib.parse import urlsplit

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec

from . import chain
from . import envelope as env
from .vault import (
    ENCRYPTION_DOMAIN, KEY_ALGORITHM, RECEIPT_SCHEMA, RECEIPTS_DOMAIN, TLS_SCHEMA, canonical, spki_sha256,
)

DEPLOYMENT_FILE = Path(__file__).with_name("deployment.json")
_DEPLOYMENT_FIELDS = ("url", "app_id", "kms_root_public_key", "kms_contract", "chain")


class CliError(Exception):
    pass


def load_deployment(path: Path = DEPLOYMENT_FILE) -> Mapping:
    deployment = json.loads(path.read_text())
    missing = [name for name in _DEPLOYMENT_FIELDS if not deployment.get(name)]
    if missing:
        raise CliError("The vault is not deployed yet (deployment.json is missing %s)." % ", ".join(missing))
    if not str(deployment["url"]).startswith("https://"):
        raise CliError("deployment.json url must be https")
    return deployment


def _tls_context(certificate_pem: Optional[str]) -> ssl.SSLContext:
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


def pinned_context(certificate_pem: str) -> ssl.SSLContext:
    """A TLS context that accepts only the certificate ``connect`` verified."""

    return _tls_context(certificate_pem)


def connect(deployment: Mapping) -> Tuple[Mapping, str]:
    """The verified identity and the vault's certificate, from one TLS session.

    The identity is read over the same session whose certificate it must
    sign, so the session provably ends inside the vault.
    """

    parts = urlsplit(str(deployment["url"]))
    if parts.scheme != "https" or not parts.hostname or parts.path not in ("", "/") or parts.query:
        raise CliError("deployment.json url must be https with no path")
    connection = http.client.HTTPSConnection(parts.hostname, parts.port or 443, timeout=30,
                                             context=_tls_context(None))
    try:
        connection.request("GET", "/v1/identity")
        certificate = connection.sock.getpeercert(binary_form=True) if connection.sock else None
        if not certificate:
            raise CliError("The vault sent no certificate.")
        response = connection.getresponse()
        if response.status != 200:
            raise CliError("The vault's identity answered %d." % response.status)
        identity = json.loads(response.read(65536))
    except http.client.HTTPException:
        raise CliError("The vault's identity answer is not HTTP.") from None
    finally:
        connection.close()
    identity = verified_identity(deployment, identity)
    if not isinstance(identity.get("tls"), Mapping) or \
            identity["tls"].get("document") != {"schema": TLS_SCHEMA, "boot_id": identity.get("boot_id"),
                                                "spki_sha256": spki_sha256(certificate)}:
        raise CliError("The TLS session does not end in the vault (a gateway or proxy ended it).")
    receipts_key = ec.EllipticCurvePublicKey.from_encoded_point(
        ec.SECP256K1(), bytes.fromhex(identity["receipts"]["public_key"]))
    try:
        receipts_key.verify(bytes.fromhex(str(identity["tls"].get("signature"))),
                            canonical(identity["tls"]["document"]), ec.ECDSA(hashes.SHA256()))
    except (InvalidSignature, ValueError):
        raise CliError("The vault's TLS key is not signed by its receipts key.") from None
    return identity, ssl.DER_cert_to_PEM_cert(certificate)


def verified_identity(deployment: Mapping, identity: Mapping) -> Mapping:
    """Return the identity only if both vault keys chain to the KMS root for our app."""

    app_id = str(deployment["app_id"]).lower().removeprefix("0x")
    if str(identity.get("app_id", "")).lower().removeprefix("0x") != app_id:
        raise CliError("The vault reports a different app id than deployment.json.")
    for name, domain in (("encryption", ENCRYPTION_DOMAIN), ("receipts", RECEIPTS_DOMAIN)):
        key = identity.get(name)
        if not isinstance(key, Mapping) or key.get("domain") != domain or key.get("algorithm") != KEY_ALGORITHM:
            raise CliError("The vault's %s key is not the expected kind." % name)
        try:
            chain.verify_key(algorithm=KEY_ALGORITHM, domain=domain, public_key_hex=str(key.get("public_key", "")),
                             signature_chain=list(key.get("signature_chain") or []), app_id_hex=app_id,
                             kms_root_public_key_hex=str(deployment["kms_root_public_key"]))
        except chain.ChainError as error:
            raise CliError("The vault's %s key does not chain to the KMS root (%s)." % (name, error)) from None
    return identity


def seal_key(*, provider: str, hotkey: str, identity: Mapping, read_secret: Callable[[str], str]) -> Mapping:
    vault_key = identity["encryption"]["public_key"]
    runtime = read_secret("%s API key: " % provider)
    sealed = {"provider": provider,
              "envelope": env.seal(runtime, env.Header(provider, "runtime", hotkey, vault_key)),
              "key_fingerprint": env.fingerprint(provider, runtime)}
    if provider == "openrouter":
        management = read_secret("OpenRouter management key (used once to prove ownership): ")
        sealed["management_envelope"] = env.seal(management, env.Header(provider, "management", hotkey, vault_key))
    return sealed


def audit_receipts(page: Mapping, identity: Mapping) -> Mapping:
    """Check signatures and per-key sequences in one vault-receipts answer.

    Only the receipts key from the verified identity is trusted, never the
    ``public_key`` a receipt carries. Numbers restart with each vault boot,
    so a sequence is checked per key and boot.
    """

    receipts_key = ec.EllipticCurvePublicKey.from_encoded_point(
        ec.SECP256K1(), bytes.fromhex(identity["receipts"]["public_key"]))
    bad: List[int] = []
    seen: Dict[Tuple[str, str], List[int]] = {}
    for row in page.get("receipts") or []:
        signed = row.get("receipt") if isinstance(row, Mapping) else None
        document = signed.get("document") if isinstance(signed, Mapping) else None
        try:
            if not isinstance(document, Mapping) or document.get("schema") != RECEIPT_SCHEMA:
                raise InvalidSignature
            receipts_key.verify(bytes.fromhex(str(signed.get("signature"))), canonical(document),
                                ec.ECDSA(hashes.SHA256()))
        except (InvalidSignature, ValueError, TypeError):
            bad.append(row.get("sequence") if isinstance(row, Mapping) else None)
            continue
        seen.setdefault((document["key_fingerprint"], document["boot_id"]), []).append(int(document["key_sequence"]))
    sequences = []
    for (fingerprint, boot_id), numbers in sorted(seen.items()):
        numbers.sort()
        have = set(numbers)
        missing = [n for n in range(1, numbers[-1] + 1) if n not in have]
        sequences.append({"key_fingerprint": fingerprint, "boot_id": boot_id, "receipts": len(numbers),
                          "highest": numbers[-1], "missing": missing[:100], "missing_count": len(missing)})
    current = identity["encryption"]["public_key"]
    reseal = [key.get("key_id") for key in page.get("keys") or []
              if isinstance(key, Mapping) and key.get("sealed_to") != current]
    return {"ok": not bad and not reseal and not any(s["missing_count"] for s in sequences),
            "bad_signatures": bad, "sequences": sequences, "keys_to_seal_again": reseal}


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(prog="lead_order_vault")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("verify", help="check the live vault against deployment.json")
    seal = commands.add_parser("seal", help="encrypt a provider key to the vault")
    seal.add_argument("--provider", choices=env.PROVIDERS, required=True)
    seal.add_argument("--hotkey", required=True)
    receipts = commands.add_parser("receipts", help="check the receipts Leadpoet reports for your keys")
    receipts.add_argument("files", nargs="+",
                          help="every page of GET /lead-orders/vault-receipts from since=0, or - for stdin")
    args = parser.parse_args(argv)
    try:
        deployment = load_deployment()
        identity, _ = connect(deployment)
        if args.command == "verify":
            print(json.dumps({"verified": True, "app_id": identity["app_id"], "compose_hash": identity.get("compose_hash"),
                              "encryption_public_key": identity["encryption"]["public_key"],
                              "receipts_public_key": identity["receipts"]["public_key"],
                              "tls_spki_sha256": identity["tls"]["document"]["spki_sha256"]}, indent=2))
            return 0
        if args.command == "receipts":
            pages = [json.loads(sys.stdin.read() if name == "-" else Path(name).read_text()) for name in args.files]
            merged = {"receipts": [row for page in pages for row in page.get("receipts") or []],
                      "keys": pages[-1].get("keys") or []}
            report = audit_receipts(merged, identity)
            print(json.dumps(report, indent=2))
            return 0 if report["ok"] else 2
        sealed = seal_key(provider=args.provider, hotkey=args.hotkey, identity=identity,
                          read_secret=lambda prompt: getpass.getpass(prompt).strip())
    except (CliError, env.EnvelopeError, OSError, ValueError) as error:
        print("error: %s" % error, file=sys.stderr)
        return 1
    print(json.dumps(sealed, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
