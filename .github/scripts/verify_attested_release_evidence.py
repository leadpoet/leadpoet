#!/usr/bin/env python3
"""Bind each untrusted CI artifact to its expected build and source identity.

Run only trusted main-branch code. Artifacts are data, never executable source.
The legacy gateway/validator evidence domains are stable schema identifiers;
they do not grant authority or identify a production host.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gateway.tee.release_manifest_v2 import (  # noqa: E402
    BUILDS_PER_DOMAIN,
    ReleaseManifestV2Error,
    build_release_manifest,
    normalize_build_evidence,
)
from gateway.tee.topology import ROLE_SPECS  # noqa: E402


MAX_EVIDENCE_BYTES = 1024 * 1024


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ReleaseManifestV2Error("artifact contains duplicate JSON keys")
        value[key] = item
    return value


def verify_artifact(
    directory: Path, *, domain: str, builder_id: str, commit: str
) -> list[dict]:
    expected_name = f"gateway-{domain}-evidence.json"
    entries = list(directory.iterdir())
    if len(entries) != 1 or entries[0].name != expected_name:
        raise ReleaseManifestV2Error("artifact contains unexpected files")
    path = entries[0]
    if path.is_symlink() or not path.is_file():
        raise ReleaseManifestV2Error("artifact evidence is not a regular file")
    if path.stat().st_size > MAX_EVIDENCE_BYTES:
        raise ReleaseManifestV2Error("artifact evidence exceeds size limit")
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    if not isinstance(value, list) or len(value) != len(ROLE_SPECS) * BUILDS_PER_DOMAIN:
        raise ReleaseManifestV2Error("artifact build evidence is incomplete")
    rows = [normalize_build_evidence(row) for row in value]
    expected_identity = {(domain, builder_id, commit)}
    if {
        (row["builder_domain"], row["builder_id"], row["commit_sha"])
        for row in rows
    } != expected_identity:
        raise ReleaseManifestV2Error("artifact source or builder identity differs")
    if {(row["physical_role"], row["build_ordinal"]) for row in rows} != {
        (role, ordinal)
        for role in ROLE_SPECS
        for ordinal in range(1, BUILDS_PER_DOMAIN + 1)
    }:
        raise ReleaseManifestV2Error("artifact role or ordinal coverage differs")
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gateway-artifact", required=True, type=Path)
    parser.add_argument("--validator-artifact", required=True, type=Path)
    parser.add_argument("--gateway-builder-id", required=True)
    parser.add_argument("--validator-builder-id", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--acceptance-signer-pubkey-hash", required=True)
    parser.add_argument("--expected-pcr0")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[0-9a-f]{40}(?:[0-9a-f]{24})?", args.commit):
        raise ReleaseManifestV2Error("expected commit is invalid")
    if args.gateway_builder_id == args.validator_builder_id:
        raise ReleaseManifestV2Error("independent builder identities must differ")
    rows = verify_artifact(
        args.gateway_artifact,
        domain="gateway",
        builder_id=args.gateway_builder_id,
        commit=args.commit,
    )
    rows += verify_artifact(
        args.validator_artifact,
        domain="validator",
        builder_id=args.validator_builder_id,
        commit=args.commit,
    )
    release = build_release_manifest(
        rows, acceptance_signer_pubkey_hash=args.acceptance_signer_pubkey_hash
    )
    if release["commit_sha"] != args.commit:
        raise ReleaseManifestV2Error("assembled release source differs")
    if args.expected_pcr0 is not None:
        if not re.fullmatch(r"[0-9a-f]{96}", args.expected_pcr0):
            raise ReleaseManifestV2Error("expected PCR0 is invalid")
        if any(row["pcr0"] != args.expected_pcr0 for row in rows):
            raise ReleaseManifestV2Error("reproduced PCR0 differs from expected measurement")
    print("Independent exact-source build artifacts verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
