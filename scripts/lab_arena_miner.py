#!/usr/bin/env python3
"""Lab Arena miner helper (labarena.md sections 6.2, 7.2, 7.3, 14.2).

    python3 scripts/lab_arena_miner.py encrypt-key --recipient recipient.json --out envelope.json
    python3 scripts/lab_arena_miner.py package --source-dir ./model --entry-point model/main.py \\
        --lock requirements.lock --out package.tar.gz
    python3 scripts/lab_arena_miner.py sign --scope submission --round-id arena-2026-09-02 \\
        --body body.json --out envelope.json [--wallet-name W --hotkey-name H | --hotkey-uri //Alice]

The runtime key is read from ``LAB_ARENA_OPENROUTER_RUNTIME_KEY`` and never
written anywhere but the encrypted envelope. Hotkey signing uses the local
Bittensor wallet; ``--hotkey-uri`` exists for development only.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lab_arena import build, contracts, credentials  # noqa: E402

KEY_ENV = "LAB_ARENA_OPENROUTER_RUNTIME_KEY"
SCOPES = {
    "submission": contracts.SCOPE_SUBMISSION,
    "funding": contracts.SCOPE_FUNDING,
    "credential": contracts.SCOPE_CREDENTIAL,
    "submission-status": contracts.SCOPE_SUBMISSION_STATUS,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Leadpoet Lab Arena miner helper")
    commands = parser.add_subparsers(dest="command", required=True)
    encrypt = commands.add_parser("encrypt-key", help="encrypt the OpenRouter runtime key to the Arena recipient key")
    encrypt.add_argument("--recipient", required=True, help="path to the GET /recipient document")
    encrypt.add_argument("--out", required=True)
    package = commands.add_parser("package", help="build a deterministic signed-package tarball")
    package.add_argument("--source-dir", required=True)
    package.add_argument("--entry-point", required=True)
    package.add_argument("--lock", required=True, help="requirements lock file with name==version lines")
    package.add_argument("--out", required=True)
    sign = commands.add_parser("sign", help="sign a canonical Arena request document")
    sign.add_argument("--scope", required=True, choices=sorted(SCOPES))
    sign.add_argument("--round-id", required=True)
    sign.add_argument("--body", required=True, help="path to the JSON request body")
    sign.add_argument("--out", required=True)
    sign.add_argument("--wallet-name", default=None)
    sign.add_argument("--hotkey-name", default=None)
    sign.add_argument("--hotkey-uri", default=None, help="development only: derive the hotkey from a URI")
    return parser


def encrypt_key(args) -> int:
    raw_key = os.environ.get(KEY_ENV, "")
    if not raw_key:
        print("%s is not set" % KEY_ENV, file=sys.stderr)
        return 2
    recipient = json.loads(Path(args.recipient).read_text(encoding="utf-8"))
    credentials.validate_recipient_document(recipient)
    envelope = credentials.encrypt_runtime_key(recipient, credentials.validate_openrouter_key_format(raw_key))
    Path(args.out).write_text(json.dumps(envelope, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"key_hash": envelope["key_hash"], "recipient_key_hash": envelope["recipient_key_hash"], "out": args.out}))
    return 0


def build_package_bytes(source_dir: Path, *, entry_point: str, lock_lines, consent=None) -> bytes:
    manifest = {
        "schema_version": contracts.SUBMISSION_PACKAGE_SCHEMA_VERSION,
        "entry_point": entry_point,
        "dependency_lock": [line.strip() for line in lock_lines if line.strip() and not line.startswith("#")],
        "consent": consent or {"source_publication": True, "public_rerun": True},
    }
    files = {}
    for path in sorted(source_dir.rglob("*")):
        if path.is_file() and not path.is_symlink():
            relative = path.relative_to(source_dir).as_posix()
            if relative == build.MANIFEST_PATH or relative == build.REQUIREMENTS_LOCK_PATH:
                continue
            files[relative] = path.read_bytes()
    files[build.MANIFEST_PATH] = json.dumps(manifest, sort_keys=True).encode("utf-8")
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz", compresslevel=6) as archive:
        for name in sorted(files):
            info = tarfile.TarInfo(name)
            data = files[name]
            info.size = len(data)
            info.mode = 0o644
            info.mtime = 0
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            archive.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def package(args) -> int:
    source_dir = Path(args.source_dir)
    if not source_dir.is_dir():
        print("source directory not found", file=sys.stderr)
        return 2
    archive = build_package_bytes(source_dir, entry_point=args.entry_point, lock_lines=Path(args.lock).read_text(encoding="utf-8").splitlines())
    inspection = build.inspect_package(archive)
    build.scan_source_archive_raise(inspection.files)
    Path(args.out).write_bytes(archive)
    print(json.dumps({"package_hash": contracts.hash_bytes(archive), "source_tree_hash": inspection.source_tree_hash, "bytes": len(archive), "out": args.out}))
    return 0


def _keypair(args):
    if args.hotkey_uri:
        from bittensor_wallet import Keypair

        return Keypair.create_from_uri(args.hotkey_uri)
    from bittensor_wallet import Wallet

    return Wallet(name=args.wallet_name or "default", hotkey=args.hotkey_name or "default").hotkey


def sign(args) -> int:
    body = json.loads(Path(args.body).read_text(encoding="utf-8"))
    keypair = _keypair(args)
    envelope = contracts.build_signed_request(
        scope=SCOPES[args.scope], round_id=args.round_id, hotkey=keypair.ss58_address, body=body,
        timestamp=int(time.time()), sign_message=lambda message: keypair.sign(message.encode("utf-8")).hex(),
    )
    Path(args.out).write_text(json.dumps(envelope, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"hotkey": envelope["hotkey"], "scope": envelope["scope"], "request_id": envelope["request_id"], "out": args.out}))
    return 0


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "encrypt-key":
        return encrypt_key(args)
    if args.command == "package":
        return package(args)
    if args.command == "sign":
        return sign(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
