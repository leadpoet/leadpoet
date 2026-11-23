"""Verified reuse of independent, already-recorded Poet claim-reader votes."""

from __future__ import annotations

import base64
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from types import FunctionType
import re
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.serialization import (
    load_pem_private_key,
    load_pem_public_key,
)
from lab_arena import collection_judge as judge
from lab_arena.collection_workflow import POET_REV

MODEL = "typesafe/jev-1.14"
PACK_SCHEMA = "collection.precomputed-claims.v1"


def reproduce_record(reader, unit, saved):
    """Run the pinned reader's check() with its network boundary replaced by the original votes.

    Historical Poet runs saved each sentence's three votes. Rebuild the exact questions
    and vote aggregation offline. The importer signing key attests to the source folder.
    No minimum number of successful unit matches or provider-receipt provenance is inferred.
    """
    if reader.MODEL != MODEL or reader.CALLS != 3 or saved.get("id") != unit["id"]:
        raise ValueError("claim reader source differs")
    rows = saved.get("sentences")
    sentences = reader.split(unit["text"])
    if (
        not isinstance(rows, list)
        or not sentences
        or [r.get("text") for r in rows] != sentences
    ):
        raise ValueError("claim sentences differ")
    if any(
        type(r.get("votes")) is not list
        or len(r["votes"]) != 3
        or any(v not in reader.CRITERIA for v in r["votes"])
        for r in rows
    ):
        raise ValueError("independent votes incomplete")
    request_hashes = []

    def ask(_secret, request, _label):
        doc = json.loads(request)
        turn = len(request_hashes)
        if (
            turn >= 3
            or doc.get("model") != MODEL
            or len(doc.get("questions", {})) != len(rows)
        ):
            raise ValueError("reader request differs")
        request_hashes.append(hashlib.sha256(request).hexdigest())
        return {
            "answers": {
                f"s{i+1}": {"choice": r["votes"][turn]} for i, r in enumerate(rows)
            }
        }

    # The supplied function comes only from the immutable POET_REV archive. Copy its
    # namespace so another thread or process can never observe a changed ask()/key().
    check = FunctionType(reader.check.__code__, dict(reader.check.__globals__, ask=ask))
    computed = check(unit, "", "offline")
    if (
        judge._json(computed) != judge._json(saved)
        or len(request_hashes) != 3
        or len(set(request_hashes)) != 1
    ):
        raise ValueError("saved claim-reader record cannot be reproduced")
    return computed, request_hashes[0]


def ingest_run(builders, reader, source_id, spec, out, meta, saved, claims_file_sha256):
    """Construct entries only from a source's recorded task, actual output and original claims.json."""
    if (
        not judge._text(source_id, 256)
        or meta.get("finish") != "answered"
        or not re.fullmatch("[a-f0-9]{64}", claims_file_sha256)
    ):
        raise ValueError("source run evidence incomplete")
    task = builders.build(
        spec["task"], spec["seed"], spec.get("shape"), spec.get("product")
    )
    if builders.plain(task)["guard"]["token"] != (spec.get("guard") or {}).get("token"):
        raise ValueError("stored scenario changed")
    units = [u for u in builders.units(task, out) if u.get("text")]
    by_id = {u["id"]: u for u in units}
    if (
        len(by_id) != len(units)
        or not isinstance(saved, list)
        or len(saved) != len({r["id"] for r in saved})
        or any(r["id"] not in by_id for r in saved)
    ):
        raise ValueError("source unit identities differ")
    entries = []
    for r in saved:
        if (
            r.get("unread")
            or r.get("missing")
            or r.get("error")
            or any(s.get("unread") for s in r.get("sentences", []))
        ):
            continue
        unit = by_id[r["id"]]
        report, wire_hash = reproduce_record(reader, unit, r)
        entries.append(
            {
                "source_run_id": source_id,
                "source_claims_sha256": claims_file_sha256,
                "source_guard": builders.plain(task)["guard"]["token"],
                "source_output_sha256": judge.digest(out),
                "unit_sha256": judge.digest(unit),
                "request_sha256": wire_hash,
                "record": report,
            }
        )
    return entries


def sign_pack(entries, private_key_pem):
    # Do not silently choose a more favorable result if an identical unit was
    # measured more than once. Conflicting records stay unresolved in the index.
    if not entries:
        raise ValueError("no complete recorded claim checks")
    body = {
        "schema": PACK_SCHEMA,
        "reader_model": MODEL,
        "reader_commit": POET_REV,
        "entries": sorted(
            entries, key=lambda r: (r["unit_sha256"], r["source_run_id"])
        ),
    }
    key = load_pem_private_key(private_key_pem, password=None)
    if not isinstance(key, Ed25519PrivateKey):
        raise ValueError("expected archive signer key")
    return {
        **body,
        "signature": base64.b64encode(key.sign(judge._json(body).encode())).decode(),
    }


@dataclass(frozen=True)
class VerifiedPack:
    digest: str
    # The caller gets a new packet for each binding. A dict of canonical bytes
    # prevents mutable collector inputs from rewriting approved reports.
    records: tuple


def verify_pack(document, pinned_public_key_pem, expected_pack_sha256):
    judge._fields(
        document, ("schema", "reader_model", "reader_commit", "entries", "signature")
    )
    if not (
        judge.digest(document) == expected_pack_sha256
        and document["schema"] == PACK_SCHEMA
        and document["reader_model"] == MODEL
        and document["reader_commit"] == POET_REV
    ):
        raise ValueError("claim pack changed")
    key = load_pem_public_key(pinned_public_key_pem)
    if not isinstance(key, Ed25519PublicKey):
        raise ValueError("expected archive verification key")
    body = {k: v for k, v in document.items() if k != "signature"}
    try:
        key.verify(
            base64.b64decode(document["signature"], validate=True),
            judge._json(body).encode(),
        )
    except (InvalidSignature, TypeError, ValueError) as exc:
        raise ValueError("claim archive signature invalid") from exc
    entries = document["entries"]
    if not isinstance(entries, list) or not 0 < len(entries) <= 100000:
        raise ValueError("invalid claim archive")
    index = {}
    seen = set()
    for e in entries:
        judge._fields(
            e,
            (
                "source_run_id",
                "source_claims_sha256",
                "source_guard",
                "source_output_sha256",
                "unit_sha256",
                "request_sha256",
                "record",
            ),
        )
        if not judge._text(e["source_run_id"], 256) or not all(
            type(e[k]) is str and re.fullmatch("[a-f0-9]{64}", e[k])
            for k in (
                "source_claims_sha256",
                "source_output_sha256",
                "unit_sha256",
                "request_sha256",
            )
        ):
            raise ValueError("invalid source binding")
        identity = e["source_run_id"], e["unit_sha256"]
        if identity in seen:
            raise ValueError("duplicate original claim check")
        seen.add(identity)
        prior = index.get(e["unit_sha256"])
        encoded = judge._json(e)
        if prior is None:
            index[e["unit_sha256"]] = encoded
        elif prior and json.loads(prior)["record"] != e["record"]:
            index[e["unit_sha256"]] = ""
    return VerifiedPack(expected_pack_sha256, tuple(index.items()))


def apply_pack(packet, *, batch_id, pack):
    """Look up exact unit IDs, written text and authoritative sources before judging.

    Identical text with different sources cannot reuse a record. The retained packet
    separately records both the measured origin and the exact current run binding.
    """
    if (
        not all(packet.get("checks", {}).values())
        or pack is None
        or "claim_reader" not in packet
    ):
        return packet
    if not isinstance(pack, VerifiedPack):
        raise ValueError("verified claim pack required")
    reader = packet["claim_reader"]
    judge._fields(reader, ("units", "results"))
    if reader["results"]:
        raise ValueError("refusing to replace original claim evidence")
    records = dict(pack.records)
    results = []
    bindings = {}
    for u in reader["units"]:
        sha = judge.digest(u)
        raw = records.get(sha)
        if not raw:
            continue
        e = json.loads(raw)
        if e["record"]["id"] != u["id"]:
            raise ValueError("claim unit identity mismatch")
        results.append({"unit_sha256": sha, "record": e["record"]})
        bindings[sha] = {
            "batch_id": batch_id,
            "run_id": packet["run_id"],
            "source_run_id": e["source_run_id"],
            "source_claims_sha256": e["source_claims_sha256"],
            "request_sha256": e["request_sha256"],
        }
    return {
        **packet,
        "claim_reader": {"units": reader["units"], "results": results},
        "claim_source_proof": {"pack_sha256": pack.digest, "bindings": bindings},
    }
