"""Derive a measured System.Events profile from a pinned public archive fixture.

This offline release helper needs scalecodec. The validator does not import it.
The resulting profile still needs an independent exact-block reveal test before
registration in subtensor_events_v2.DEFAULT_PROFILE_PATHS.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from scalecodec.base import RuntimeConfigurationObject, ScaleBytes
from scalecodec.type_registry import load_type_registry_preset

from leadpoet_canonical.subtensor_events_v2 import (
    PROFILE_SCHEMA_VERSION, RUNTIME_CODE_STORAGE_KEY,
    SYSTEM_EVENT_COUNT_STORAGE_KEY, SYSTEM_EVENTS_STORAGE_KEY,
    _validate_profile_structure,
)


def _one(values, name):
    matches = [value for value in values if value["name"] == name]
    if len(matches) != 1:
        raise ValueError("metadata has no unique %s" % name)
    return matches[0]


def _field(value):
    return {"name": value["name"], "type": value["type"]}


def _simple_type(value):
    kind, body = next(iter(value["def"].items()))
    if kind == "composite":
        body = {"fields": [_field(field) for field in body["fields"]]}
    elif kind == "variant":
        body = {"variants": [
            {"name": variant["name"], "index": variant["index"],
             "fields": [_field(field) for field in variant["fields"]]}
            for variant in body["variants"]
        ]}
    elif kind in ("sequence", "compact"):
        body = {"type": body["type"]}
    elif kind == "array":
        body = {"len": body["len"], "type": body["type"]}
    elif kind == "tuple":
        body = list(body)
    elif kind == "bitSequence":
        body = {"bitStoreType": body["bitStoreType"],
                "bitOrderType": body["bitOrderType"]}
    elif kind != "primitive":
        raise ValueError("unsupported portable type %s" % kind)
    return {"path": list(value["path"]), "def": {kind: body}}


def _references(value):
    kind, body = next(iter(value["def"].items()))
    if kind == "composite":
        return [field["type"] for field in body["fields"]]
    if kind == "variant":
        return [field["type"] for variant in body["variants"]
                for field in variant["fields"]]
    if kind in ("sequence", "compact", "array"):
        return [body["type"]]
    if kind == "tuple":
        return body
    if kind == "bitSequence":
        return [body["bitStoreType"], body["bitOrderType"]]
    return []


def _reachable_types(types, root):
    result = {}
    pending = [root]
    while pending:
        type_id = pending.pop()
        if type_id in result:
            continue
        value = _simple_type(types[type_id])
        result[type_id] = value
        pending.extend(_references(value))
    return {str(type_id): result[type_id] for type_id in sorted(result)}


def _variant(types, type_id, name):
    variants = types[type_id]["def"]["variant"]["variants"]
    return _one(variants, name)


def profile_from_fixture(fixture):
    metadata_hex = fixture["metadata_hex"]
    if not isinstance(metadata_hex, str) or not metadata_hex.startswith("0x"):
        raise ValueError("fixture metadata bytes are invalid")
    metadata_raw = bytes.fromhex(metadata_hex[2:])
    if not metadata_raw.startswith(b"meta\x0e"):
        raise ValueError("fixture is not SCALE metadata V14")
    configuration = RuntimeConfigurationObject()
    configuration.update_type_registry(load_type_registry_preset("core"))
    scale = configuration.create_scale_object(
        "MetadataVersioned", data=ScaleBytes(metadata_hex)
    )
    decoded = scale.decode()
    if scale.data.offset != len(metadata_raw):
        raise ValueError("metadata has trailing bytes")
    if decoded[0] != "0x6d657461" or set(decoded[1]) != {"V14"}:
        raise ValueError("metadata version is not V14")
    metadata = decoded[1]["V14"]
    types = {item["id"]: item["type"] for item in metadata["types"]["types"]}
    system = _one(metadata["pallets"], "System")
    if system["storage"]["prefix"] != "System":
        raise ValueError("System storage prefix differs")
    entries = system["storage"]["entries"]
    events = _one(entries, "Events")
    count = _one(entries, "EventCount")
    if set(events["type"]) != {"Plain"} or set(count["type"]) != {"Plain"}:
        raise ValueError("System event storage is not plain")
    events_type = events["type"]["Plain"]
    record_type = types[events_type]["def"]["sequence"]["type"]
    record_fields = types[record_type]["def"]["composite"]["fields"]
    if [field["name"] for field in record_fields] != ["phase", "event", "topics"]:
        raise ValueError("EventRecord fields differ")
    phase_type, runtime_type, topics_type = [field["type"] for field in record_fields]
    initialization = _variant(types, phase_type, "Initialization")
    subtensor = _variant(types, runtime_type, "SubtensorModule")
    if len(subtensor["fields"]) != 1:
        raise ValueError("SubtensorModule event fields differ")
    subtensor_type = subtensor["fields"][0]["type"]
    weights = _variant(types, subtensor_type, "WeightsSet")
    reveal = _variant(types, subtensor_type, "TimelockedWeightsRevealed")
    if len(weights["fields"]) != 2 or len(reveal["fields"]) != 2:
        raise ValueError("target event arity differs")
    netuid_type = weights["fields"][0]["type"]
    if reveal["fields"][0]["type"] != netuid_type:
        raise ValueError("target event netuid types differ")
    events_raw = bytes.fromhex(fixture["system_events"].removeprefix("0x"))
    count_raw = bytes.fromhex(fixture["system_event_count"].removeprefix("0x"))
    profile = {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "network": "finney",
        "genesis_hash": fixture["genesis_hash"],
        "spec_version": fixture["runtime_version"]["specVersion"],
        "transaction_version": fixture["runtime_version"]["transactionVersion"],
        "metadata_version": 14,
        "metadata_raw_sha256": hashlib.sha256(metadata_raw).hexdigest(),
        "runtime_code_storage_key": RUNTIME_CODE_STORAGE_KEY,
        "runtime_code_storage_hash": fixture["runtime_code_storage_hash"],
        "storage": {
            "events": {"pallet": "System", "name": "Events",
                       "key": SYSTEM_EVENTS_STORAGE_KEY, "modifier": events["modifier"],
                       "type_id": events_type},
            "event_count": {"pallet": "System", "name": "EventCount",
                            "key": SYSTEM_EVENT_COUNT_STORAGE_KEY, "modifier": count["modifier"],
                            "type_id": count["type"]["Plain"]},
        },
        "event_layout": {
            "event_record_type_id": record_type,
            "phase_type_id": phase_type,
            "runtime_event_type_id": runtime_type,
            "topics_type_id": topics_type,
            "subtensor_event_type_id": subtensor_type,
            "initialization_phase_index": initialization["index"],
            "subtensor_runtime_event_index": subtensor["index"],
            "weights_set_event_index": weights["index"],
            "timelocked_weights_revealed_event_index": reveal["index"],
            "netuid_type_id": netuid_type,
            "uid_type_id": weights["fields"][1]["type"],
            "account_id_type_id": reveal["fields"][1]["type"],
        },
        "types": _reachable_types(types, events_type),
        "measurement": {
            "archive_host": fixture["archive_host"],
            "block_number": fixture["block_number"],
            "block_hash": fixture["block_hash"],
            "parent_hash": "0x" + fixture["parent_hash"].removeprefix("0x"),
            "metadata_raw_bytes": len(metadata_raw),
            "system_events_bytes": len(events_raw),
            "system_events_sha256": hashlib.sha256(events_raw).hexdigest(),
            "system_event_count_raw_sha256": hashlib.sha256(count_raw).hexdigest(),
            "system_event_count": int.from_bytes(count_raw, "little"),
        },
    }
    return _validate_profile_structure(profile)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    profile = profile_from_fixture(json.loads(args.fixture.read_text()))
    args.output.write_text(json.dumps(profile, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
