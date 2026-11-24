"""Miner command: seal a provider key to the lead-order key vault.

    python -m lead_order_vault.cli verify
    python -m lead_order_vault.cli seal --provider openrouter --hotkey <ss58>

``seal`` reads the key from the terminal (never from arguments, so it stays
out of shell history), checks that the vault's encryption key was issued by
Phala's KMS to the app named in ``deployment.json``, and prints an envelope.
Submit the envelope to ``POST /lead-orders/keys``. Only the vault can open it.

The check needs nothing from Leadpoet: the app id and the KMS root key in
``deployment.json`` can be compared with the on-chain records on Base.
"""

from __future__ import annotations

import argparse
import getpass
import json
import sys
import urllib.request
from pathlib import Path
from typing import Callable, Mapping, Optional

from . import chain
from . import envelope as env
from .vault import ENCRYPTION_DOMAIN, KEY_ALGORITHM, RECEIPTS_DOMAIN

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


def fetch_identity(url: str) -> Mapping:
    with urllib.request.urlopen(url.rstrip("/") + "/v1/identity", timeout=30) as response:  # noqa: S310 (https only)
        return json.loads(response.read(65536))


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


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(prog="lead_order_vault")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("verify", help="check the live vault against deployment.json")
    seal = commands.add_parser("seal", help="encrypt a provider key to the vault")
    seal.add_argument("--provider", choices=env.PROVIDERS, required=True)
    seal.add_argument("--hotkey", required=True)
    args = parser.parse_args(argv)
    try:
        deployment = load_deployment()
        identity = verified_identity(deployment, fetch_identity(deployment["url"]))
        if args.command == "verify":
            print(json.dumps({"verified": True, "app_id": identity["app_id"], "compose_hash": identity.get("compose_hash"),
                              "encryption_public_key": identity["encryption"]["public_key"],
                              "receipts_public_key": identity["receipts"]["public_key"]}, indent=2))
            return 0
        sealed = seal_key(provider=args.provider, hotkey=args.hotkey, identity=identity,
                          read_secret=lambda prompt: getpass.getpass(prompt).strip())
    except (CliError, env.EnvelopeError) as error:
        print("error: %s" % error, file=sys.stderr)
        return 1
    print(json.dumps(sealed, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
