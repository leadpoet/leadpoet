"""Verify a gateway-created student against its original private capture and billing account."""

from __future__ import annotations

from decimal import Decimal
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from urllib.parse import urlencode

from lab_arena import collection_judge as judge
from lab_arena.collection_checkpoint import (
    FILES,
    prepare_student_source,
    correction_scenario,
)
from lab_arena.collection_quality import admit_built_split
from lab_arena.provider_costs import openrouter_generation_cost
from lab_arena.collection_workflow import store_artifact

ORIGIN = "owner-students"
NATIVE = "collection.gateway-native.v1"
RECEIPT = "collection.gateway-student-sources.v1"
NAME = re.compile(r"[a-zA-Z0-9_-]{1,64}")
DIGEST = re.compile(r"sha256:[a-f0-9]{64}")
GENERATION = re.compile(r"gen-[0-9A-Za-z-]{1,124}")


def object_store():
    from lab_arena.collection_intake import live_finney_arena
    from lab_arena.service import S3ObjectStore

    if not live_finney_arena():
        raise ValueError(
            "gateway provenance requires the production Arena evidence store"
        )
    return S3ObjectStore(
        os.environ["LAB_ARENA_BUCKET"], prefix=os.getenv("LAB_ARENA_OBJECT_PREFIX", "")
    )


def _read(objects, run_id, kind, digest, limit):
    if not NAME.fullmatch(run_id) or not DIGEST.fullmatch(digest):
        raise ValueError("invalid private gateway source reference")
    key = f"collection/{ORIGIN}/{run_id}/{kind}-{digest[7:]}.json"
    data = objects.get_bounded(key, limit)
    if hashlib.sha256(data).hexdigest() != digest[7:]:
        raise ValueError("private gateway source checksum changed")
    return json.loads(data)


def _match_charge(original, response, normalized):
    model, revision, provider = (original[k] for k in ("model", "revision", "provider"))
    usage = response["usage"]
    expected_cost = Decimal(str(usage["cost"]))
    if (
        normalized["id"] != response["id"]
        or normalized["model"] not in {model, revision}
        or normalized["provider"] != provider
        or not expected_cost.is_finite()
        or Decimal(normalized["cost_usd"]) != expected_cost
        or normalized["cancelled"] is not False
    ):
        raise ValueError(
            "owner generation billing identity or cost differs from capture"
        )


def read_owner_charge(transport, owner_key, response, provenance):
    generation = response["id"]
    if not owner_key or not GENERATION.fullmatch(generation):
        raise ValueError("owner generation or credential unavailable")
    reply = transport.send(
        method="GET",
        url="https://openrouter.ai/api/v1/generation?" + urlencode({"id": generation}),
        headers={"Authorization": "Bearer " + owner_key},
        body=b"",
        timeout_seconds=20,
        max_response_bytes=256 * 1024,
    )
    if reply.status != 200:
        raise ValueError(f"owner generation lookup returned HTTP {reply.status}")
    doc = json.loads(reply.body)
    cost = openrouter_generation_cost(doc, generation_id=generation)
    if cost is None:
        raise ValueError("owner generation has no confirmed charge")
    data = doc["data"]
    normalized = {
        "id": data["id"],
        "model": data.get("model"),
        "provider": data.get("provider_name"),
        "cost_usd": str(cost.units),
        "cancelled": data.get("cancelled"),
    }
    _match_charge(provenance, response, normalized)
    return normalized


def native_scenario(objects, run_id, digest):
    native = _read(objects, run_id, "native", digest, 64 * 1024 * 1024)
    trace = json.loads(native["original"]["trace.json"])
    return f"{trace['task']}-s{trace['seed']}"


def stage_gateway_source(
    objects, sources, *, run_id, native_sha256, split, transport, owner_key
):
    """Read only the operator's immutable origin. The owner API key is used for GETs only."""
    native = _read(objects, run_id, "native", native_sha256, 64 * 1024 * 1024)
    if (
        set(native) != {"schema", "original", "provenance"}
        or native["schema"] != NATIVE
        or set(native["original"]) != set(FILES)
    ):
        raise ValueError("expected the gateway's original student capture")
    provenance = native["provenance"]
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp)
        for name in FILES:
            (path / name).write_text(native["original"][name], encoding="utf-8")
        source = prepare_student_source(
            *sources, path, model=provenance["model"], provenance=provenance
        )
    if source["rejected_fatal"] is not True:
        raise ValueError("a correction needs a genuine replayable first failure")
    scenario = correction_scenario(sources[1], source)
    admit_built_split(
        sources[1],
        scenario,
        json.loads(native["original"]["trace.json"]).get("shape"),
        split,
    )
    charges = [
        read_owner_charge(transport, owner_key, c["response"], provenance)
        for c in provenance["calls"]
    ]
    if not charges or not any(Decimal(x["cost_usd"]) > 0 for x in charges):
        raise ValueError("no confirmed positive owner-funded work")
    source_hash = "sha256:" + judge.digest(source)
    split_hash = "sha256:" + judge.digest(split)
    settlement = {
        "schema": "collection.gateway-generation-costs.v1",
        "native_sha256": native_sha256,
        "source_sha256": source_hash,
        "split_sha256": split_hash,
        "charges": charges,
    }
    saved = store_artifact(objects, ORIGIN, run_id, "verified", settlement)
    receipt = {
        "schema": RECEIPT,
        "producer": "gateway",
        "split_sha256": split_hash,
        "sources": {
            source_hash: {
                "source_run_id": run_id,
                "model": source["student_model"],
                "model_revision": provenance["revision"],
                "role": "untuned_student",
                "rights": "owned_synthetic_training_and_review",
                "split": "training",
                "native_sha256": native_sha256,
                "verification_sha256": saved["sha256"],
            }
        },
    }
    return source, receipt


def verify_gateway_source(spec, receipt, sources, objects):
    if (
        set(receipt) != {"schema", "producer", "split_sha256", "sources"}
        or receipt["producer"] != "gateway"
        or receipt["split_sha256"] != spec.raw["split_sha256"]
        or not isinstance(receipt["sources"], dict)
        or set(receipt["sources"])
        != {case["source_sha256"] for case in spec.raw["cases"]}
        or "sha256:" + judge.digest(receipt) != spec.raw["provenance_sha256"]
    ):
        raise ValueError("gateway source receipt differs from frozen experiment")
    for source_hash, source in sources.items():
        item = receipt["sources"][source_hash]
        if set(item) != {
            "source_run_id",
            "model",
            "model_revision",
            "role",
            "rights",
            "split",
            "native_sha256",
            "verification_sha256",
        }:
            raise ValueError("invalid gateway provenance fields")
        run = item["source_run_id"]
        original = _read(
            objects, run, "native", item["native_sha256"], 64 * 1024 * 1024
        )
        settled = _read(
            objects, run, "verified", item["verification_sha256"], 512 * 1024
        )
        if (
            original.get("schema") != NATIVE
            or original.get("original") != source["original"]
            or source.get("provenance") is None
            or original.get("provenance") != source["provenance"]
            or "sha256:" + judge.digest(source) != source_hash
            or source["student_model"] != item["model"]
            or source["provenance"]["revision"] != item["model_revision"]
            or item["role"] != "untuned_student"
            or item["rights"] != "owned_synthetic_training_and_review"
            or item["split"] != "training"
            or settled.get("schema") != "collection.gateway-generation-costs.v1"
            or settled.get("source_sha256") != source_hash
            or settled.get("native_sha256") != item["native_sha256"]
            or settled.get("split_sha256") != receipt["split_sha256"]
        ):
            raise ValueError(
                "gateway source differs from the privately verified original"
            )
        calls = source["provenance"]["calls"]
        charges = settled["charges"]
        if (
            not calls
            or len(calls) != len(charges)
            or not any(Decimal(x["cost_usd"]) > 0 for x in charges)
        ):
            raise ValueError("gateway source has incomplete billing evidence")
        for call, charge in zip(calls, charges, strict=True):
            _match_charge(source["provenance"], call["response"], charge)
    return receipt
