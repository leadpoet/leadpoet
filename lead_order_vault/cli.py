"""Miner command: seal a provider key to the lead-order key vault.

    python -m lead_order_vault.cli verify
    python -m lead_order_vault.cli seal --provider openrouter --hotkey <ss58>
    python -m lead_order_vault.cli receipts receipts.json

Every command first checks the live vault with ``trust.connect`` and then
reads Base with ``trust.on_chain`` (``--rpc``, Base's public endpoint by
default).

``seal`` reads the key from the terminal (never from arguments, so it stays
out of shell history) and prints an envelope. Submit the envelope to ``POST /lead-orders/keys``. Only
the vault can open it.

``receipts`` reads every page of ``GET /lead-orders/vault-receipts`` from
``since=0`` (saved files, or ``-`` for stdin), checks every receipt's
signature against the vault's receipts key, and reports any gap in each
key's sequence numbers and any key that must be sealed again because the
vault's key changed.
"""

from __future__ import annotations

import argparse
import getpass
import json
import sys
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Tuple

from . import envelope as env
from . import trust
from .vault import RECEIPT_SCHEMA


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

    bad: List[int] = []
    seen: Dict[Tuple[str, str], List[int]] = {}
    for row in page.get("receipts") or []:
        try:
            document = trust.verify_signed(identity, row.get("receipt") if isinstance(row, Mapping) else None)
            if document.get("schema") != RECEIPT_SCHEMA:
                raise trust.TrustError("not a receipt")
        except trust.TrustError:
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
    parser.add_argument("--rpc", default=BASE_RPC, help="a Base JSON-RPC endpoint you trust")
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
        if args.command == "receipts":
            pages = [json.loads(sys.stdin.read() if name == "-" else Path(name).read_text()) for name in args.files]
            merged = {"receipts": [row for page in pages for row in page.get("receipts") or []],
                      "keys": pages[-1].get("keys") or []}
            report = audit_receipts(merged, identity)
            print(json.dumps(report, indent=2))
            return 0 if report["ok"] else 2
        sealed = seal_key(provider=args.provider, hotkey=args.hotkey, identity=identity,
                          read_secret=lambda prompt: getpass.getpass(prompt).strip())
    except (trust.TrustError, env.EnvelopeError, OSError, ValueError) as error:
        print("error: %s" % error, file=sys.stderr)
        return 1
    print(json.dumps(sealed, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
