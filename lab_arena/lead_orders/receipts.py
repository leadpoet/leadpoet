"""Signed order receipts with a Merkle root over the per-call ledger rows.

A receipt lets the buyer, any miner and any validator check one order: what
was held, what was spent, what was charged, who earned what, and that the
published call rows are exactly the rows the money was computed from. It
never carries a prompt, a tool name, a model, a key, lead data or call order.
The call leaves carry only the fields needed to recompute the money and are
sorted by their random call id, so the published rows reveal no sequence.

Merkle construction (RFC 6962 style): leaf = sha256(0x00 || canonical JSON),
node = sha256(0x01 || left || right), an odd last node is promoted unchanged,
and the empty tree is sha256(b"").
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Mapping, Optional, Sequence

from lab_arena import signing
from lab_arena.contracts import ArenaContractError, canonical_json, document_hash

RECEIPT_SCHEMA_VERSION = "leadpoet.lead_orders.receipt.v1"
RECEIPT_HASH_FIELD = "receipt_hash"
LEAF_FIELDS = (
    "call_id", "provider", "miner_hotkey", "public_fingerprint",
    "reserved_microusd", "status", "cost_microusd",
)


class ReceiptError(ArenaContractError):
    pass


def call_leaf(row: Mapping[str, Any]) -> Dict[str, Any]:
    """The published, money-relevant part of one ledger row."""

    missing = [name for name in LEAF_FIELDS if name not in row]
    if missing:
        raise ReceiptError("call row lacks %s" % ", ".join(missing))
    return {name: row[name] for name in LEAF_FIELDS}


def call_leaves(calls: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    return sorted((call_leaf(row) for row in calls), key=lambda leaf: str(leaf["call_id"]))


def key_totals(leaves: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Per key: provider, miner hotkey (null for Leadpoet), calls, settled USD."""

    totals: Dict[tuple, Dict[str, Any]] = {}
    for leaf in leaves:
        group = (leaf["provider"], leaf["public_fingerprint"])
        row = totals.setdefault(group, {
            "provider": leaf["provider"], "miner_hotkey": leaf["miner_hotkey"],
            "public_fingerprint": leaf["public_fingerprint"], "calls": 0, "settled_microusd": 0,
            "uncertain_calls": 0,
        })
        row["calls"] += 1
        if leaf["status"] == "settled":
            row["settled_microusd"] += int(leaf["cost_microusd"])
        else:
            row["uncertain_calls"] += 1
    return [totals[k] for k in sorted(totals, key=lambda k: (str(k[0]), str(k[1])))]


def _leaf_hash(leaf: Mapping[str, Any]) -> bytes:
    return hashlib.sha256(b"\x00" + canonical_json(dict(leaf)).encode("utf-8")).digest()


def merkle_root(leaves: Sequence[Mapping[str, Any]]) -> str:
    level: List[bytes] = [_leaf_hash(leaf) for leaf in leaves]
    if not level:
        return "sha256:" + hashlib.sha256(b"").hexdigest()
    while len(level) > 1:
        nxt: List[bytes] = []
        for i in range(0, len(level), 2):
            if i + 1 < len(level):
                nxt.append(hashlib.sha256(b"\x01" + level[i] + level[i + 1]).digest())
            else:
                nxt.append(level[i])
        level = nxt
    return "sha256:" + level[0].hex()


def build_receipt(
    *,
    order: Mapping[str, Any],
    calls: Sequence[Mapping[str, Any]],
    rewards: Mapping[str, int],
    policy_hash: str,
) -> Dict[str, Any]:
    """Receipt body plus its hash, from the stored order view and call rows."""

    if order.get("status") not in ("delivered", "failed"):
        raise ReceiptError("only a finished order has a receipt")
    leaves = call_leaves(calls)
    body = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "order_id": str(order["order_id"]),
        "hotkey": order["hotkey"],
        "mode": order["mode"],
        "policy_version": int(order["policy_version"]),
        "policy_hash": policy_hash,
        "status": order["status"],
        "leads_requested": int(order["leads_requested"]),
        "leads_delivered": int(order["leads_delivered"] or 0),
        "hold_microusd": int(order["hold_microusd"]),
        "spend_ceiling_microusd": int(order["spend_ceiling_microusd"]),
        "spend_microusd": int(order["spend_microusd"]),
        "charge_microusd": int(order["charge_microusd"]),
        "settled_epoch": int(order["settled_epoch"]),
        "keys": key_totals(leaves),
        "call_count": len(leaves),
        "calls_merkle_root": merkle_root(leaves),
        "rewards_microusd": {k: int(v) for k, v in sorted(rewards.items())},
    }
    return dict(body, **{RECEIPT_HASH_FIELD: document_hash(body)})


def sign_receipt(signer: signing.ArenaSigner, receipt: Mapping[str, Any]) -> Dict[str, Any]:
    return signing.sign_document(signer, receipt, hash_field=RECEIPT_HASH_FIELD)


def verify_receipt(
    receipt: Mapping[str, Any],
    *,
    calls: Optional[Sequence[Mapping[str, Any]]] = None,
    public_key_der: bytes,
    expected_public_key_hash: str,
) -> str:
    """Check the signature and, when call rows are given, the Merkle root."""

    digest = signing.verify_document_signature(
        receipt, hash_field=RECEIPT_HASH_FIELD,
        public_key_der=public_key_der, expected_public_key_hash=expected_public_key_hash,
    )
    if calls is not None:
        leaves = call_leaves(calls)
        if len(leaves) != receipt.get("call_count") or merkle_root(leaves) != receipt.get("calls_merkle_root"):
            raise ReceiptError("call rows do not match the receipt")
    return digest


__all__ = [
    "RECEIPT_HASH_FIELD",
    "RECEIPT_SCHEMA_VERSION",
    "ReceiptError",
    "build_receipt",
    "call_leaf",
    "call_leaves",
    "key_totals",
    "merkle_root",
    "sign_receipt",
    "verify_receipt",
]
