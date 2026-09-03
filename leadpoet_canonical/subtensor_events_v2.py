"""Dependency-free, metadata-bound decoder for Subtensor reveal events.

The decoder accepts only a measured portable-metadata profile. It consumes the
complete ``System.Events`` value and each record's topics before it can return
a TimelockedWeightsRevealed witness.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


PROFILE_SCHEMA_VERSION = "leadpoet.subtensor_events_profile.v2"
PROOF_SCHEMA_VERSION = "leadpoet.timelocked_weights_reveal_proof.v2"
DEFAULT_PROFILE_PATH = Path(__file__).with_name("subtensor_events_profile_v2.json")
SYSTEM_EVENTS_STORAGE_KEY = (
    "0x26aa394eea5630e07c48ae0c9558cef7" "80d41e5e16056765bc8461851072c9d7"
)
SYSTEM_EVENT_COUNT_STORAGE_KEY = (
    "0x26aa394eea5630e07c48ae0c9558cef7" "0a98fdbe9ce6c55837576c60c7af3850"
)
RUNTIME_CODE_STORAGE_KEY = "0x3a636f6465"
MAX_PROFILE_BYTES = 2 * 1024 * 1024
MAX_EVENTS_BYTES = 8 * 1024 * 1024
MAX_EVENT_RECORDS = 15_000
MAX_COLLECTION_ITEMS = 15_000
MAX_ARRAY_ITEMS = 65_536
MAX_TYPE_COUNT = 1_024
MAX_DECODE_DEPTH = 96
MAX_DECODE_NODES = 250_000
MAX_JSON_DEPTH = 96
MAX_JSON_NODES = 250_000
_RAW_HASH_RE = re.compile(r"^(?:0x)?[0-9a-f]{64}$")
_SHA256_RE = re.compile(r"^(?:sha256:)?[0-9a-f]{64}$")
_PRIMITIVES = frozenset(
    {
        "bool",
        "char",
        "str",
        "u8",
        "u16",
        "u32",
        "u64",
        "u128",
        "u256",
        "i8",
        "i16",
        "i32",
        "i64",
        "i128",
        "i256",
    }
)


class SubtensorEventsV2Error(ValueError):
    """The profile or SCALE event value is not an exact measured value."""


def _fail(message: str) -> None:
    raise SubtensorEventsV2Error(message)


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, RecursionError) as exc:
        raise SubtensorEventsV2Error("event profile is not canonical JSON") from exc


def _sha256_json(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json(value)).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _strict_json(payload: bytes) -> Dict[str, Any]:
    if type(payload) is not bytes or not payload or len(payload) > MAX_PROFILE_BYTES:
        _fail("event profile size is invalid")

    def pairs(values):
        result = {}
        for key, value in values:
            if key in result:
                _fail("event profile contains a duplicate JSON key")
            result[key] = value
        return result

    def constant(_value):
        _fail("event profile contains a non-finite number")

    try:
        result = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=constant,
        )
    except SubtensorEventsV2Error:
        raise
    except (UnicodeDecodeError, ValueError, RecursionError) as exc:
        raise SubtensorEventsV2Error("event profile JSON is invalid") from exc
    if not isinstance(result, Mapping):
        _fail("event profile root is invalid")
    stack = [(result, 0)]
    nodes = 0
    while stack:
        current, depth = stack.pop()
        nodes += 1
        if nodes > MAX_JSON_NODES or depth > MAX_JSON_DEPTH:
            _fail("event profile JSON exceeds policy")
        if isinstance(current, Mapping):
            stack.extend((value, depth + 1) for value in current.values())
        elif isinstance(current, list):
            stack.extend((value, depth + 1) for value in current)
    return dict(result)


def _exact_fields(value: Any, fields: Sequence[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(fields):
        _fail("%s fields are invalid" % label)
    return value


def _integer(value: Any, label: str, *, maximum: int = (1 << 32) - 1) -> int:
    if type(value) is not int or not 0 <= value <= maximum:
        _fail("%s is invalid" % label)
    return int(value)


def _text(value: Any, label: str, *, allow_empty: bool = False) -> str:
    if type(value) is not str:
        _fail("%s is invalid" % label)
    try:
        encoded = value.encode("ascii")
    except UnicodeEncodeError as exc:
        raise SubtensorEventsV2Error("%s is not ASCII" % label) from exc
    if (not encoded and not allow_empty) or len(encoded) > 256:
        _fail("%s length is invalid" % label)
    return value


def _hash(value: Any, label: str, *, prefix: str) -> str:
    pattern = _RAW_HASH_RE if prefix == "0x" else _SHA256_RE
    if type(value) is not str or not pattern.fullmatch(value):
        _fail("%s is invalid" % label)
    raw = value
    if prefix == "" and raw.startswith("sha256:"):
        raw = raw[7:]
    elif prefix == "0x" and raw.startswith("0x"):
        raw = raw[2:]
    return prefix + raw


def _type_reference(value: Any, label: str) -> int:
    return _integer(value, label)


def _validate_field(value: Any, label: str) -> Dict[str, Any]:
    value = _exact_fields(value, ("name", "type"), label)
    name = value["name"]
    if name is not None:
        name = _text(name, "%s name" % label)
    return {"name": name, "type": _type_reference(value["type"], "%s type" % label)}


def _validate_type(type_id: int, value: Any) -> Dict[str, Any]:
    label = "portable type %d" % type_id
    value = _exact_fields(value, ("path", "def"), label)
    path = value["path"]
    if not isinstance(path, list) or len(path) > 32:
        _fail("%s path is invalid" % label)
    clean_path = [
        _text(item, "%s path item" % label, allow_empty=False) for item in path
    ]
    definition = value["def"]
    if not isinstance(definition, Mapping) or len(definition) != 1:
        _fail("%s definition is invalid" % label)
    kind, body = next(iter(definition.items()))
    if kind == "primitive":
        if body not in _PRIMITIVES:
            _fail("%s primitive is unsupported" % label)
        clean_body = body
    elif kind in ("sequence", "compact"):
        body = _exact_fields(body, ("type",), "%s %s" % (label, kind))
        clean_body = {"type": _type_reference(body["type"], "%s item" % label)}
    elif kind == "array":
        body = _exact_fields(body, ("len", "type"), "%s array" % label)
        clean_body = {
            "len": _integer(
                body["len"], "%s array length" % label, maximum=MAX_ARRAY_ITEMS
            ),
            "type": _type_reference(body["type"], "%s array item" % label),
        }
    elif kind == "tuple":
        if not isinstance(body, list) or len(body) > 256:
            _fail("%s tuple is invalid" % label)
        clean_body = [_type_reference(item, "%s tuple item" % label) for item in body]
    elif kind == "composite":
        body = _exact_fields(body, ("fields",), "%s composite" % label)
        if not isinstance(body["fields"], list) or len(body["fields"]) > 256:
            _fail("%s composite fields are invalid" % label)
        clean_body = {
            "fields": [
                _validate_field(item, "%s field" % label) for item in body["fields"]
            ]
        }
    elif kind == "variant":
        body = _exact_fields(body, ("variants",), "%s variant" % label)
        if not isinstance(body["variants"], list) or len(body["variants"]) > 256:
            _fail("%s variants are invalid" % label)
        variants = []
        indexes = set()
        names = set()
        for entry in body["variants"]:
            entry = _exact_fields(
                entry, ("name", "index", "fields"), "%s variant entry" % label
            )
            name = _text(entry["name"], "%s variant name" % label)
            index = _integer(entry["index"], "%s variant index" % label, maximum=255)
            fields = entry["fields"]
            if (
                index in indexes
                or name in names
                or not isinstance(fields, list)
                or len(fields) > 256
            ):
                _fail("%s variant entries are ambiguous" % label)
            indexes.add(index)
            names.add(name)
            variants.append(
                {
                    "name": name,
                    "index": index,
                    "fields": [
                        _validate_field(item, "%s variant field" % label)
                        for item in fields
                    ],
                }
            )
        clean_body = {"variants": variants}
    elif kind == "bitSequence":
        body = _exact_fields(
            body, ("bitStoreType", "bitOrderType"), "%s bit sequence" % label
        )
        clean_body = {
            "bitStoreType": _type_reference(
                body["bitStoreType"], "%s bit store" % label
            ),
            "bitOrderType": _type_reference(
                body["bitOrderType"], "%s bit order" % label
            ),
        }
    else:
        _fail("%s definition kind is unsupported" % label)
    return {"path": clean_path, "def": {kind: clean_body}}


def _type_references(value: Mapping[str, Any]) -> Sequence[int]:
    kind, body = next(iter(value["def"].items()))
    if kind == "composite":
        return [item["type"] for item in body["fields"]]
    if kind == "variant":
        return [
            field["type"] for variant in body["variants"] for field in variant["fields"]
        ]
    if kind in ("sequence", "array", "compact"):
        return [body["type"]]
    if kind == "tuple":
        return body
    if kind == "bitSequence":
        return [body["bitStoreType"], body["bitOrderType"]]
    return []


def _definition(types: Mapping[int, Mapping[str, Any]], type_id: int, kind: str):
    value = types.get(type_id)
    if value is None or set(value["def"]) != {kind}:
        _fail("portable type %d is not %s" % (type_id, kind))
    return value["def"][kind]


def _fixed_u8_array(
    types: Mapping[int, Mapping[str, Any]], type_id: int, seen: Optional[set] = None
) -> Optional[int]:
    visited = set() if seen is None else seen
    if type_id in visited:
        return None
    visited.add(type_id)
    definition = types[type_id]["def"]
    kind, body = next(iter(definition.items()))
    if kind == "array" and _definition(types, body["type"], "primitive") == "u8":
        return body["len"]
    if kind == "composite" and len(body["fields"]) == 1:
        field = body["fields"][0]
        if field["name"] is None:
            return _fixed_u8_array(types, field["type"], visited)
    return None


def _named_variant(
    types: Mapping[int, Mapping[str, Any]], type_id: int, name: str, index: int
) -> Mapping[str, Any]:
    variants = _definition(types, type_id, "variant")["variants"]
    matches = [
        value for value in variants if value["name"] == name and value["index"] == index
    ]
    if len(matches) != 1:
        _fail("portable event variant %s differs from measured layout" % name)
    return matches[0]


def _validate_profile_structure(profile: Any) -> Dict[str, Any]:
    top_fields = (
        "schema_version",
        "network",
        "genesis_hash",
        "spec_version",
        "transaction_version",
        "metadata_version",
        "metadata_raw_sha256",
        "runtime_code_storage_key",
        "runtime_code_storage_hash",
        "storage",
        "event_layout",
        "types",
        "measurement",
    )
    profile = _exact_fields(profile, top_fields, "event profile")
    if profile["schema_version"] != PROFILE_SCHEMA_VERSION:
        _fail("event profile schema version is unsupported")
    if profile["network"] != "finney":
        _fail("event profile network is unsupported")
    normalized: Dict[str, Any] = {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "network": "finney",
        "genesis_hash": _hash(profile["genesis_hash"], "genesis hash", prefix="0x"),
        "spec_version": _integer(profile["spec_version"], "spec version"),
        "transaction_version": _integer(
            profile["transaction_version"], "transaction version"
        ),
        "metadata_version": _integer(
            profile["metadata_version"], "metadata version", maximum=255
        ),
        "metadata_raw_sha256": _hash(
            profile["metadata_raw_sha256"], "metadata hash", prefix=""
        ),
        "runtime_code_storage_key": _text(
            profile["runtime_code_storage_key"], "runtime code storage key"
        ),
        "runtime_code_storage_hash": _hash(
            profile["runtime_code_storage_hash"],
            "runtime code storage hash",
            prefix="0x",
        ),
    }
    if (
        normalized["metadata_version"] != 14
        or normalized["runtime_code_storage_key"] != RUNTIME_CODE_STORAGE_KEY
    ):
        _fail("event profile runtime binding is unsupported")

    storage = _exact_fields(profile["storage"], ("events", "event_count"), "storage")
    clean_storage = {}
    storage_expectations = {
        "events": ("Events", SYSTEM_EVENTS_STORAGE_KEY),
        "event_count": ("EventCount", SYSTEM_EVENT_COUNT_STORAGE_KEY),
    }
    for label, (name, key) in storage_expectations.items():
        entry = _exact_fields(
            storage[label],
            ("pallet", "name", "key", "modifier", "type_id"),
            "%s storage" % label,
        )
        clean = {
            "pallet": _text(entry["pallet"], "%s pallet" % label),
            "name": _text(entry["name"], "%s name" % label),
            "key": _text(entry["key"], "%s key" % label),
            "modifier": _text(entry["modifier"], "%s modifier" % label),
            "type_id": _type_reference(entry["type_id"], "%s type" % label),
        }
        if (
            clean["pallet"] != "System"
            or clean["name"] != name
            or clean["key"] != key
            or clean["modifier"] != "Default"
        ):
            _fail("%s storage layout differs from System metadata" % label)
        clean_storage[label] = clean
    normalized["storage"] = clean_storage

    layout_fields = (
        "event_record_type_id",
        "phase_type_id",
        "runtime_event_type_id",
        "topics_type_id",
        "subtensor_event_type_id",
        "initialization_phase_index",
        "subtensor_runtime_event_index",
        "weights_set_event_index",
        "timelocked_weights_revealed_event_index",
        "netuid_type_id",
        "uid_type_id",
        "account_id_type_id",
    )
    layout = _exact_fields(profile["event_layout"], layout_fields, "event layout")
    clean_layout = {
        field: _integer(
            layout[field],
            "event layout %s" % field,
            maximum=255 if field.endswith("_index") else (1 << 32) - 1,
        )
        for field in layout_fields
    }
    normalized["event_layout"] = clean_layout

    raw_types = profile["types"]
    if (
        not isinstance(raw_types, Mapping)
        or not raw_types
        or len(raw_types) > MAX_TYPE_COUNT
    ):
        _fail("portable type graph size is invalid")
    types: Dict[int, Dict[str, Any]] = {}
    for raw_id, value in raw_types.items():
        if type(raw_id) is not str or not raw_id.isdigit():
            _fail("portable type identifier is invalid")
        type_id = int(raw_id)
        if str(type_id) != raw_id or type_id > (1 << 32) - 1 or type_id in types:
            _fail("portable type identifier is not canonical")
        types[type_id] = _validate_type(type_id, value)
    for type_id, value in types.items():
        for reference in _type_references(value):
            if reference not in types:
                _fail("portable type %d has an absent reference" % type_id)

    events_type = clean_storage["events"]["type_id"]
    record_type = clean_layout["event_record_type_id"]
    if _definition(types, events_type, "sequence")["type"] != record_type:
        _fail("System.Events root type differs from measured layout")
    record_fields = _definition(types, record_type, "composite")["fields"]
    expected_record_fields = [
        {"name": "phase", "type": clean_layout["phase_type_id"]},
        {"name": "event", "type": clean_layout["runtime_event_type_id"]},
        {"name": "topics", "type": clean_layout["topics_type_id"]},
    ]
    if record_fields != expected_record_fields:
        _fail("EventRecord fields differ from measured layout")

    initialization = _named_variant(
        types,
        clean_layout["phase_type_id"],
        "Initialization",
        clean_layout["initialization_phase_index"],
    )
    if initialization["fields"]:
        _fail("Initialization phase fields differ from measured layout")
    outer = _named_variant(
        types,
        clean_layout["runtime_event_type_id"],
        "SubtensorModule",
        clean_layout["subtensor_runtime_event_index"],
    )
    if outer["fields"] != [
        {"name": None, "type": clean_layout["subtensor_event_type_id"]}
    ]:
        _fail("Subtensor runtime event fields differ from measured layout")
    weights = _named_variant(
        types,
        clean_layout["subtensor_event_type_id"],
        "WeightsSet",
        clean_layout["weights_set_event_index"],
    )
    if weights["fields"] != [
        {"name": None, "type": clean_layout["netuid_type_id"]},
        {"name": None, "type": clean_layout["uid_type_id"]},
    ]:
        _fail("WeightsSet fields differ from measured layout")
    reveal = _named_variant(
        types,
        clean_layout["subtensor_event_type_id"],
        "TimelockedWeightsRevealed",
        clean_layout["timelocked_weights_revealed_event_index"],
    )
    if reveal["fields"] != [
        {"name": None, "type": clean_layout["netuid_type_id"]},
        {"name": None, "type": clean_layout["account_id_type_id"]},
    ]:
        _fail("TimelockedWeightsRevealed fields differ from measured layout")

    uid_type = clean_layout["uid_type_id"]
    if _definition(types, uid_type, "primitive") != "u16":
        _fail("UID type differs from u16")
    netuid_fields = _definition(types, clean_layout["netuid_type_id"], "composite")[
        "fields"
    ]
    if netuid_fields != [{"name": None, "type": uid_type}]:
        _fail("netuid type differs from its measured u16 wrapper")
    if _fixed_u8_array(types, clean_layout["account_id_type_id"]) != 32:
        _fail("AccountId32 bytes differ from measured layout")
    topics = _definition(types, clean_layout["topics_type_id"], "sequence")
    if _fixed_u8_array(types, topics["type"]) != 32:
        _fail("event topic type differs from H256")
    if (
        _definition(types, clean_storage["event_count"]["type_id"], "primitive")
        != "u32"
    ):
        _fail("System.EventCount type differs from u32")

    reachable = set()
    pending = [events_type]
    while pending:
        type_id = pending.pop()
        if type_id in reachable:
            continue
        reachable.add(type_id)
        pending.extend(_type_references(types[type_id]))
    if reachable != set(types):
        _fail("portable type graph contains unreachable types")
    normalized["types"] = {str(type_id): types[type_id] for type_id in sorted(types)}

    measurement_fields = (
        "archive_host",
        "block_number",
        "block_hash",
        "parent_hash",
        "metadata_raw_bytes",
        "system_events_bytes",
        "system_events_sha256",
        "system_event_count_raw_sha256",
        "system_event_count",
    )
    measurement = _exact_fields(
        profile["measurement"], measurement_fields, "measurement"
    )
    clean_measurement = {
        "archive_host": _text(measurement["archive_host"], "archive host"),
        "block_number": _integer(
            measurement["block_number"],
            "measurement block number",
            maximum=(1 << 64) - 1,
        ),
        "block_hash": _hash(
            measurement["block_hash"], "measurement block hash", prefix="0x"
        ),
        "parent_hash": _hash(
            measurement["parent_hash"], "measurement parent hash", prefix="0x"
        ),
        "metadata_raw_bytes": _integer(
            measurement["metadata_raw_bytes"],
            "metadata byte count",
            maximum=MAX_PROFILE_BYTES,
        ),
        "system_events_bytes": _integer(
            measurement["system_events_bytes"],
            "event byte count",
            maximum=MAX_EVENTS_BYTES,
        ),
        "system_events_sha256": _hash(
            measurement["system_events_sha256"], "measured events hash", prefix=""
        ),
        "system_event_count_raw_sha256": _hash(
            measurement["system_event_count_raw_sha256"],
            "measured event count hash",
            prefix="",
        ),
        "system_event_count": _integer(
            measurement["system_event_count"],
            "measured event count",
            maximum=MAX_EVENT_RECORDS,
        ),
    }
    if clean_measurement["archive_host"] != "archive.chain.opentensor.ai":
        _fail("event profile archive host is unsupported")
    normalized["measurement"] = clean_measurement
    return normalized


def load_subtensor_events_profile_v2(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load and structurally validate the measured event profile."""

    profile_path = Path(path) if path is not None else DEFAULT_PROFILE_PATH
    try:
        payload = profile_path.read_bytes()
    except OSError as exc:
        raise SubtensorEventsV2Error("event profile cannot be read") from exc
    return _validate_profile_structure(_strict_json(payload))


def validate_subtensor_events_profile_v2(
    profile: Mapping[str, Any],
    *,
    genesis_hash: str,
    spec_version: int,
    transaction_version: int,
    metadata_raw: Optional[bytes] = None,
    metadata_sha256: Optional[str] = None,
    runtime_code_hash: str,
) -> Dict[str, Any]:
    """Bind a profile to the exact live/archive runtime observation."""

    normalized = _validate_profile_structure(profile)
    if (
        _hash(genesis_hash, "observed genesis hash", prefix="0x")
        != normalized["genesis_hash"]
    ):
        _fail("observed genesis hash differs from event profile")
    if _integer(spec_version, "observed spec version") != normalized["spec_version"]:
        _fail("observed spec version differs from event profile")
    if (
        _integer(transaction_version, "observed transaction version")
        != normalized["transaction_version"]
    ):
        _fail("observed transaction version differs from event profile")
    if metadata_raw is None and metadata_sha256 is None:
        _fail("observed metadata binding is absent")
    observed_metadata_hash = None
    if metadata_raw is not None:
        if (
            type(metadata_raw) is not bytes
            or not metadata_raw
            or len(metadata_raw) > MAX_PROFILE_BYTES
        ):
            _fail("observed metadata bytes are invalid")
        if not metadata_raw.startswith(b"meta\x0e"):
            _fail("observed metadata is not SCALE metadata V14")
        observed_metadata_hash = hashlib.sha256(metadata_raw).hexdigest()
    if metadata_sha256 is not None:
        supplied_hash = _hash(metadata_sha256, "observed metadata hash", prefix="")
        if (
            observed_metadata_hash is not None
            and supplied_hash != observed_metadata_hash
        ):
            _fail("observed metadata bytes and hash differ")
        observed_metadata_hash = supplied_hash
    if observed_metadata_hash != normalized["metadata_raw_sha256"]:
        _fail("observed metadata differs from event profile")
    if (
        _hash(runtime_code_hash, "observed runtime code hash", prefix="0x")
        != normalized["runtime_code_storage_hash"]
    ):
        _fail("observed runtime code differs from event profile")
    return normalized


class _ScaleReader:
    def __init__(self, data: bytes):
        if type(data) is not bytes or not data or len(data) > MAX_EVENTS_BYTES:
            _fail("System.Events bytes are invalid")
        self.data = data
        self.offset = 0
        self.nodes = 0

    def read(self, size: int) -> bytes:
        if type(size) is not int or size < 0:
            _fail("SCALE read size is invalid")
        end = self.offset + size
        if end > len(self.data):
            _fail("SCALE value is truncated")
        value = self.data[self.offset : end]
        self.offset = end
        return value

    def compact(self, *, maximum: int) -> int:
        first = self.read(1)[0]
        mode = first & 3
        if mode == 0:
            value = first >> 2
        elif mode == 1:
            raw = bytes((first,)) + self.read(1)
            value = int.from_bytes(raw, "little") >> 2
            if value < 1 << 6:
                _fail("SCALE compact integer is not canonical")
        elif mode == 2:
            raw = bytes((first,)) + self.read(3)
            value = int.from_bytes(raw, "little") >> 2
            if value < 1 << 14:
                _fail("SCALE compact integer is not canonical")
        else:
            length = (first >> 2) + 4
            if length > 32:
                _fail("SCALE compact integer exceeds policy")
            raw = self.read(length)
            value = int.from_bytes(raw, "little")
            minimum_length = max(4, (value.bit_length() + 7) // 8)
            if value < 1 << 30 or raw[-1] == 0 or length != minimum_length:
                _fail("SCALE compact integer is not canonical")
        if value > maximum:
            _fail("SCALE compact integer exceeds type or policy")
        return value


def _primitive_width(name: str) -> Optional[int]:
    if len(name) >= 2 and name[0] in ("u", "i") and name[1:].isdigit():
        bits = int(name[1:])
        if bits in (8, 16, 32, 64, 128, 256):
            return bits // 8
    return None


def _underlying_primitive(
    types: Mapping[int, Mapping[str, Any]], type_id: int, seen: Optional[set] = None
) -> Optional[str]:
    visited = set() if seen is None else seen
    if type_id in visited:
        return None
    visited.add(type_id)
    value = types[type_id]
    kind, body = next(iter(value["def"].items()))
    if kind == "primitive":
        return body
    if kind == "composite" and len(body["fields"]) == 1:
        return _underlying_primitive(types, body["fields"][0]["type"], visited)
    return None


def _decode_type(
    reader: _ScaleReader,
    types: Mapping[int, Mapping[str, Any]],
    type_id: int,
    depth: int,
) -> Dict[str, Any]:
    if depth > MAX_DECODE_DEPTH:
        _fail("SCALE value exceeds maximum type depth")
    reader.nodes += 1
    if reader.nodes > MAX_DECODE_NODES:
        _fail("SCALE value exceeds maximum node count")
    value = types.get(type_id)
    if value is None:
        _fail("SCALE value references an unknown type")
    kind, body = next(iter(value["def"].items()))

    if kind == "primitive":
        if body == "bool":
            raw = reader.read(1)[0]
            if raw not in (0, 1):
                _fail("SCALE bool is invalid")
            decoded: Any = bool(raw)
        elif body == "char":
            number = int.from_bytes(reader.read(4), "little")
            if number > 0x10FFFF or 0xD800 <= number <= 0xDFFF:
                _fail("SCALE char is invalid")
            decoded = chr(number)
        elif body == "str":
            length = reader.compact(maximum=MAX_EVENTS_BYTES)
            try:
                decoded = reader.read(length).decode("utf-8")
            except UnicodeDecodeError as exc:
                raise SubtensorEventsV2Error("SCALE string is invalid UTF-8") from exc
        else:
            width = _primitive_width(body)
            if width is None:
                _fail("SCALE primitive is unsupported")
            decoded = int.from_bytes(
                reader.read(width), "little", signed=body.startswith("i")
            )
        return {"type_id": type_id, "kind": kind, "value": decoded}

    if kind in ("sequence", "array"):
        item_type = body["type"]
        if kind == "sequence":
            count = reader.compact(maximum=MAX_COLLECTION_ITEMS)
        else:
            count = body["len"]
        if _underlying_primitive(types, item_type) == "u8" and set(
            types[item_type]["def"]
        ) == {"primitive"}:
            return {
                "type_id": type_id,
                "kind": kind,
                "value_hex": reader.read(count).hex(),
                "length": count,
            }
        items = [
            _decode_type(reader, types, item_type, depth + 1) for _index in range(count)
        ]
        return {"type_id": type_id, "kind": kind, "items": items, "length": count}

    if kind == "tuple":
        items = [
            _decode_type(reader, types, item_type, depth + 1) for item_type in body
        ]
        return {"type_id": type_id, "kind": kind, "items": items}

    if kind == "composite":
        fields = [
            {
                "name": field["name"],
                "type_id": field["type"],
                "value": _decode_type(reader, types, field["type"], depth + 1),
            }
            for field in body["fields"]
        ]
        return {"type_id": type_id, "kind": kind, "fields": fields}

    if kind == "variant":
        index = reader.read(1)[0]
        matches = [item for item in body["variants"] if item["index"] == index]
        if len(matches) != 1:
            _fail("SCALE variant index is unknown")
        variant = matches[0]
        fields = [
            {
                "name": field["name"],
                "type_id": field["type"],
                "value": _decode_type(reader, types, field["type"], depth + 1),
            }
            for field in variant["fields"]
        ]
        return {
            "type_id": type_id,
            "kind": kind,
            "name": variant["name"],
            "index": index,
            "fields": fields,
        }

    if kind == "compact":
        primitive = _underlying_primitive(types, body["type"])
        width = _primitive_width(primitive or "")
        if primitive is None or width is None or not primitive.startswith("u"):
            _fail("SCALE compact type is not an unsigned integer")
        decoded = reader.compact(maximum=(1 << (width * 8)) - 1)
        return {"type_id": type_id, "kind": kind, "value": decoded}

    if kind == "bitSequence":
        bit_count = reader.compact(maximum=MAX_COLLECTION_ITEMS * 64)
        primitive = _underlying_primitive(types, body["bitStoreType"])
        width = _primitive_width(primitive or "")
        order_path = types[body["bitOrderType"]]["path"]
        if (
            primitive not in ("u8", "u16", "u32", "u64")
            or width is None
            or not order_path
            or order_path[-1] not in ("Lsb0", "Msb0")
        ):
            _fail("SCALE bit sequence layout is unsupported")
        store_bits = width * 8
        store_count = (bit_count + store_bits - 1) // store_bits
        raw = reader.read(store_count * width)
        return {
            "type_id": type_id,
            "kind": kind,
            "bit_length": bit_count,
            "value_hex": raw.hex(),
            "bit_order": order_path[-1],
        }

    _fail("SCALE type kind is unsupported")


def _public_value(value: Mapping[str, Any]) -> Any:
    kind = value["kind"]
    if kind in ("primitive", "compact"):
        return value["value"]
    if kind in ("sequence", "array"):
        if "value_hex" in value:
            return "0x" + value["value_hex"]
        return [_public_value(item) for item in value["items"]]
    if kind == "tuple":
        return [_public_value(item) for item in value["items"]]
    if kind == "composite":
        fields = value["fields"]
        if len(fields) == 1 and fields[0]["name"] is None:
            return _public_value(fields[0]["value"])
        if fields and all(field["name"] is not None for field in fields):
            names = [field["name"] for field in fields]
            if len(set(names)) == len(names):
                return {
                    field["name"]: _public_value(field["value"]) for field in fields
                }
        return [_public_value(field["value"]) for field in fields]
    if kind == "variant":
        return {
            "name": value["name"],
            "index": value["index"],
            "fields": [_public_value(field["value"]) for field in value["fields"]],
        }
    if kind == "bitSequence":
        return {
            "bit_length": value["bit_length"],
            "value_hex": "0x" + value["value_hex"],
            "bit_order": value["bit_order"],
        }
    _fail("decoded SCALE value kind is unsupported")


def _composite_field(value: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    if value.get("kind") != "composite":
        _fail("decoded EventRecord is not composite")
    matches = [field["value"] for field in value["fields"] if field["name"] == name]
    if len(matches) != 1:
        _fail("decoded EventRecord field is absent or ambiguous")
    return matches[0]


def _decode_event_count(event_count_raw: bytes) -> int:
    if type(event_count_raw) is not bytes or len(event_count_raw) != 4:
        _fail("System.EventCount bytes are invalid")
    count = int.from_bytes(event_count_raw, "little")
    if count > MAX_EVENT_RECORDS:
        _fail("System.EventCount exceeds policy")
    return count


def decode_system_events_v2(
    events_raw: bytes,
    *,
    profile: Mapping[str, Any],
    event_count_raw: bytes,
) -> Sequence[Dict[str, Any]]:
    """Decode all EventRecords and reject any unconsumed or unknown bytes."""

    normalized = _validate_profile_structure(profile)
    if (
        type(events_raw) is not bytes
        or not events_raw
        or len(events_raw) > MAX_EVENTS_BYTES
    ):
        _fail("System.Events bytes are invalid")
    reader = _ScaleReader(events_raw)
    count = reader.compact(maximum=MAX_EVENT_RECORDS)
    if count != _decode_event_count(event_count_raw):
        _fail("System.Events length differs from System.EventCount")
    types = {int(type_id): value for type_id, value in normalized["types"].items()}
    record_type = normalized["event_layout"]["event_record_type_id"]
    records: List[Dict[str, Any]] = []
    for record_index in range(count):
        start = reader.offset
        decoded = _decode_type(reader, types, record_type, 0)
        end = reader.offset
        phase = _composite_field(decoded, "phase")
        runtime_event = _composite_field(decoded, "event")
        topics = _composite_field(decoded, "topics")
        if phase.get("kind") != "variant" or runtime_event.get("kind") != "variant":
            _fail("decoded event discriminant is invalid")
        if topics.get("kind") != "sequence" or "items" not in topics:
            _fail("decoded event topics are invalid")
        topic_values = [_public_value(item) for item in topics["items"]]
        if any(
            type(item) is not str or not re.fullmatch(r"0x[0-9a-f]{64}", item)
            for item in topic_values
        ):
            _fail("decoded event topic is not H256")

        pallet_event_name = None
        pallet_event_index = None
        event_fields: Sequence[Any] = ()
        outer_fields = runtime_event["fields"]
        if len(outer_fields) == 1:
            inner = outer_fields[0]["value"]
            if inner.get("kind") != "variant":
                _fail("decoded pallet event is not a variant")
            pallet_event_name = inner["name"]
            pallet_event_index = inner["index"]
            event_fields = [_public_value(field["value"]) for field in inner["fields"]]
        elif outer_fields:
            _fail("decoded runtime event fields are ambiguous")
        record_raw = events_raw[start:end]
        records.append(
            {
                "record_index": record_index,
                "record_sha256": _sha256_bytes(record_raw),
                "phase": phase["name"],
                "phase_index": phase["index"],
                "runtime_event": runtime_event["name"],
                "runtime_event_index": runtime_event["index"],
                "pallet_event": pallet_event_name,
                "pallet_event_index": pallet_event_index,
                "fields": list(event_fields),
                "topics": topic_values,
            }
        )
    if reader.offset != len(events_raw):
        _fail("System.Events contains trailing bytes")
    return tuple(records)


def _account_id_hex(value: Any, label: str) -> str:
    if type(value) is not str or not re.fullmatch(r"(?:0x)?[0-9a-fA-F]{64}", value):
        _fail("%s is invalid" % label)
    return value[2:].lower() if value.startswith("0x") else value.lower()


def prove_timelocked_weights_reveal_v2(
    events_raw: bytes,
    *,
    profile: Mapping[str, Any],
    event_count_raw: bytes,
    expected_netuid: int,
    expected_uid: int,
    expected_account_id_hex: str,
) -> Dict[str, Any]:
    """Prove one exact adjacent initialization reveal pair in a full block."""

    normalized = _validate_profile_structure(profile)
    netuid = _integer(expected_netuid, "expected netuid", maximum=0xFFFF)
    uid = _integer(expected_uid, "expected UID", maximum=0xFFFF)
    account_id = _account_id_hex(expected_account_id_hex, "expected account ID")
    records = decode_system_events_v2(
        events_raw, profile=normalized, event_count_raw=event_count_raw
    )
    layout = normalized["event_layout"]
    matches: List[Tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    for index in range(len(records) - 1):
        weights, reveal = records[index], records[index + 1]
        if (
            weights["phase"] == "Initialization"
            and weights["phase_index"] == layout["initialization_phase_index"]
            and weights["runtime_event"] == "SubtensorModule"
            and weights["runtime_event_index"]
            == layout["subtensor_runtime_event_index"]
            and weights["pallet_event"] == "WeightsSet"
            and weights["pallet_event_index"] == layout["weights_set_event_index"]
            and weights["fields"] == [netuid, uid]
            and reveal["phase"] == "Initialization"
            and reveal["phase_index"] == layout["initialization_phase_index"]
            and reveal["runtime_event"] == "SubtensorModule"
            and reveal["runtime_event_index"] == layout["subtensor_runtime_event_index"]
            and reveal["pallet_event"] == "TimelockedWeightsRevealed"
            and reveal["pallet_event_index"]
            == layout["timelocked_weights_revealed_event_index"]
            and reveal["fields"] == [netuid, "0x" + account_id]
        ):
            matches.append((weights, reveal))
    if len(matches) != 1:
        _fail("matching timelocked reveal event pair is absent or ambiguous")
    weights, reveal = matches[0]
    return {
        "schema_version": PROOF_SCHEMA_VERSION,
        "profile_sha256": _sha256_json(normalized),
        "events_sha256": _sha256_bytes(events_raw),
        "event_count": len(records),
        "weights_set_record_index": weights["record_index"],
        "weights_set_record_sha256": weights["record_sha256"],
        "reveal_record_index": reveal["record_index"],
        "reveal_record_sha256": reveal["record_sha256"],
        "netuid": netuid,
        "uid": uid,
        "account_id_hex": account_id,
        "phase": "Initialization",
        "runtime_event_index": layout["subtensor_runtime_event_index"],
        "weights_set_event_index": layout["weights_set_event_index"],
        "timelocked_weights_revealed_event_index": layout[
            "timelocked_weights_revealed_event_index"
        ],
    }
