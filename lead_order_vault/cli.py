"""Miner command: seal a provider key to the lead-order key vault.

    python -m lead_order_vault.cli verify
    python -m lead_order_vault.cli seal --provider openrouter --hotkey <ss58>

Every command first checks the live vault with ``trust.connect`` and then
reads Base with ``trust.on_chain`` (``--rpc``, Base's public endpoint by
default).

``seal`` reads the key from the terminal (never from arguments, so it stays
out of shell history) and prints an envelope. Submit the envelope to ``POST /lead-orders/keys``. Only
the vault can open it.
"""

from __future__ import annotations

import argparse
import getpass
import json
import sys
from typing import Callable, Mapping, Optional

from . import envelope as env
from . import trust


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
    parser.add_argument("--rpc", default=BASE_RPC, help="a Base JSON-RPC endpoint you trust")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("verify", help="check the live vault against deployment.json")
    seal = commands.add_parser("seal", help="encrypt a provider key to the vault")
    seal.add_argument("--provider", choices=env.PROVIDERS, required=True)
    seal.add_argument("--hotkey", required=True)
    args = parser.parse_args(argv)
    try:
        deployment = trust.load_deployment()
        identity, _ = trust.connect(deployment)
        owner = trust.on_chain(deployment, identity, args.rpc)
        if args.command == "verify":
            print(json.dumps({"verified": True, "app_id": identity["app_id"], "compose_hash": identity.get("compose_hash"),
                              "kms_contract": deployment["kms_contract"], "app_owner": owner,
                              "encryption_public_key": identity["encryption"]["public_key"],
                              "receipts_public_key": identity["receipts"]["public_key"],
                              "tls_spki_sha256": identity["tls"]["document"]["spki_sha256"]}, indent=2))
            return 0
        sealed = seal_key(provider=args.provider, hotkey=args.hotkey, identity=identity,
                          read_secret=lambda prompt: getpass.getpass(prompt).strip())
    except (trust.TrustError, env.EnvelopeError, OSError, ValueError) as error:
        print("error: %s" % error, file=sys.stderr)
        return 1
    print(json.dumps(sealed, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
