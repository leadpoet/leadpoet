#!/usr/bin/env python3
"""Verify a Research Lab epoch audit event inside an Arweave checkpoint."""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
import sys
from typing import Any, Mapping, Sequence

import requests


def canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)


def sha256_json(data: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(data).encode("utf-8")).hexdigest()


def event_hash(event: Mapping[str, Any]) -> bytes:
    return hashlib.sha256(canonical_json(event).encode("utf-8")).digest()


def merkle_root(events: Sequence[Mapping[str, Any]]) -> str:
    if not events:
        return "0" * 64
    level = [event_hash(event) for event in events]
    while len(level) > 1:
        next_level = []
        for index in range(0, len(level), 2):
            left = level[index]
            right = level[index + 1] if index + 1 < len(level) else left
            next_level.append(hashlib.sha256(left + right).digest())
        level = next_level
    return level[0].hex()


def download_checkpoint(tx_id: str) -> dict[str, Any]:
    response = requests.get(f"https://arweave.net/{tx_id}", timeout=45)
    response.raise_for_status()
    return response.json()


def decompress_events(checkpoint: Mapping[str, Any]) -> list[dict[str, Any]]:
    encoded = checkpoint.get("events_compressed")
    if not isinstance(encoded, str) or not encoded:
        raise ValueError("checkpoint is missing events_compressed")
    return json.loads(gzip.decompress(base64.b64decode(encoded)))


def find_research_lab_event(
    events: Sequence[Mapping[str, Any]],
    *,
    event_hash_hex: str | None,
    tee_sequence: int | None,
) -> Mapping[str, Any]:
    matches = []
    for event in events:
        if event.get("event_type") != "RESEARCH_LAB_EPOCH_AUDIT":
            continue
        log_entry = event.get("signed_log_entry") if isinstance(event.get("signed_log_entry"), Mapping) else {}
        if event_hash_hex and log_entry.get("event_hash") != event_hash_hex:
            continue
        if tee_sequence is not None and int(event.get("sequence", -1)) != int(tee_sequence):
            continue
        matches.append(event)
    if not matches:
        raise ValueError("Research Lab epoch audit event not found in checkpoint")
    if len(matches) > 1:
        raise ValueError("multiple matching Research Lab epoch audit events found")
    return matches[0]


def verify_signed_event(event: Mapping[str, Any]) -> dict[str, Any]:
    log_entry = event.get("signed_log_entry")
    if not isinstance(log_entry, Mapping):
        raise ValueError("event is missing signed_log_entry")
    signed_event = log_entry.get("signed_event")
    if not isinstance(signed_event, Mapping):
        raise ValueError("signed_log_entry is missing signed_event")
    expected_event_hash = hashlib.sha256(canonical_json(signed_event).encode("utf-8")).hexdigest()
    if log_entry.get("event_hash") != expected_event_hash:
        raise ValueError("signed event hash mismatch")
    payload = signed_event.get("payload")
    if not isinstance(payload, Mapping):
        raise ValueError("signed event payload is not an object")
    expected_payload_hash = sha256_json({key: value for key, value in payload.items() if key != "payload_hash"})
    if payload.get("payload_hash") != expected_payload_hash:
        raise ValueError("Research Lab payload_hash mismatch")
    return dict(payload)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tx-id", help="Arweave checkpoint transaction id")
    parser.add_argument("--event-hash", help="Expected transparency event hash")
    parser.add_argument("--tee-sequence", type=int, help="Expected TEE checkpoint sequence")
    parser.add_argument("--epoch", type=int, help="Expected Research Lab epoch")
    parser.add_argument("--netuid", type=int, help="Expected subnet netuid")
    args = parser.parse_args()
    if not args.tx_id:
        print("Research Lab Arweave audit verifier ready; pass --tx-id for live checkpoint verification")
        return 0

    checkpoint = download_checkpoint(args.tx_id)
    events = decompress_events(checkpoint)
    header = checkpoint.get("header") if isinstance(checkpoint.get("header"), Mapping) else {}
    computed_root = merkle_root(events)
    expected_root = str(header.get("merkle_root") or "")
    if computed_root != expected_root:
        raise ValueError(f"checkpoint Merkle root mismatch: {computed_root} != {expected_root}")

    event = find_research_lab_event(
        events,
        event_hash_hex=args.event_hash,
        tee_sequence=args.tee_sequence,
    )
    payload = verify_signed_event(event)
    if args.epoch is not None and int(payload.get("epoch", -1)) != args.epoch:
        raise ValueError("epoch mismatch")
    if args.netuid is not None and int(payload.get("netuid", -1)) != args.netuid:
        raise ValueError("netuid mismatch")

    print("Research Lab Arweave audit verified")
    print(f"  tx_id: {args.tx_id}")
    print(f"  checkpoint_number: {header.get('checkpoint_number')}")
    print(f"  merkle_root: {computed_root}")
    print(f"  event_hash: {event.get('signed_log_entry', {}).get('event_hash')}")
    print(f"  tee_sequence: {event.get('sequence')}")
    print(f"  epoch: {payload.get('epoch')}")
    print(f"  netuid: {payload.get('netuid')}")
    print(f"  audit_kind: {payload.get('audit_kind')}")
    print(f"  payload_hash: {payload.get('payload_hash')}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
