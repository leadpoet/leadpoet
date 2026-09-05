"""Lab Arena V1 contracts: limits, request authentication, and stable I/O.

This module owns the Arena's canonical JSON and SHA-256 helpers. It performs
no I/O and has no dependency on validator attestation code.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import re
import secrets
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Public V1 constants (labarena.md section 1)
# ---------------------------------------------------------------------------

# Every participant runs the first ten ICPs. The ten best challengers then run
# the remaining ten ICPs; the incumbent baseline always runs both stages.
# These are the twenty ICPs in the organizer's current daily qualification set.
STAGE_1_ICP_COUNT = 10
STAGE_2_ICP_COUNT = 10
BENCHMARK_ICP_COUNT = STAGE_1_ICP_COUNT + STAGE_2_ICP_COUNT
FINALIST_COUNT = 10
MAX_CHALLENGERS = 256  # one entry per registered miner; each round pins its own admitted ceiling at or below this
DEFAULT_MAX_CHALLENGERS = 16  # fits one daily round on the default eight-slot runner while still making a stage-1 cut
RUNNER_SLOT_CEILING = 8
MAX_ATTEMPTS_PER_ASSIGNMENT = 2
LAB_ARENA_POOL_PERCENT = 25  # default share of total emissions for the king's pool; LAB_ARENA_POOL_PERCENT overrides it per round
# The pool is a share of total emissions, not of what remains after the other
# allocations (owner decision, 2026-09-03).
LAB_ARENA_POOL_BASIS = "total_emissions"
KING_POOL_SHARE_PERCENT_BY_WEEK = (100, 80, 60, 40, 20)
EPOCHS_PER_REWARD_WEEK = 140
ELIGIBILITY_MAX_EPOCHS = 45
# The host supplies these provider keys. Every bundle receives the same
# externally enforced call quota per provider and ICP attempt.
PROVIDERS = ("scrapingdog", "deepline", "openrouter")
CALL_QUOTAS_PER_ICP = {"scrapingdog": 30, "deepline": 30, "openrouter": 60}
# Judge calls made while scoring one work item (one output on one ICP), using
# the same organizer-supplied provider accounts as bundle execution.
# Sized from the real judge through the shim (tests/lab_arena/test_lab_arena_real_judge.py):
# per company about 6 Scrapingdog fetches (homepage, Wayback check, evidence
# pages), 3 OpenRouter calls, and 1 Deepline contents call, with retry headroom
# for five companies. Infrastructure and account failures never become a
# miner score of zero.
SCORING_CALL_QUOTAS_PER_WORK_ITEM = {"scrapingdog": 150, "deepline": 40, "openrouter": 120}
# Assignment kinds: a validator either executes a miner's model on one ICP or
# scores one output on one ICP with the Arena judge.
ASSIGNMENT_KINDS = ("execute", "score")
ICP_WALL_CLOCK_SECONDS = 300
# A judge run reads pages and calls several models per company against live
# providers; it gets its own wall clock, longer than a model's, under the same
# lease; provider calls refresh the lease while the judge is working.
SCORING_WALL_CLOCK_SECONDS = 900
# One lease covers the longest sandbox run, first-use source setup, and the
# small completion retry window. Provider calls renew it while work continues.
LEASE_TTL_SECONDS = 1200


def stage_positions(stage: int) -> Tuple[int, ...]:
    """Return the fixed benchmark positions for one execution stage."""

    if stage == 1:
        return tuple(range(STAGE_1_ICP_COUNT))
    if stage == 2:
        return tuple(range(STAGE_1_ICP_COUNT, BENCHMARK_ICP_COUNT))
    raise ArenaContractError("stage must be 1 or 2")

# Signed request timestamp window (section 9.1).
REQUEST_TIMESTAMP_WINDOW_SECONDS = 300

# ---------------------------------------------------------------------------
# Schema versions and scopes
# ---------------------------------------------------------------------------

ARENA_CONTRACT_VERSION = "lab_arena.v1"
SIGNED_REQUEST_SCHEMA_VERSION = "leadpoet.lab_arena.signed_request.v1"
SIGNED_DOCUMENT_SCHEMA_VERSION = "leadpoet.lab_arena.signed_document.v1"
ROUND_CONFIGURATION_SCHEMA_VERSION = "leadpoet.lab_arena.round_configuration.v1"
SCORER_POLICY_SCHEMA_VERSION = "leadpoet.lab_arena.scorer_policy.v1"
SCORING_PLAN_SCHEMA_VERSION = "leadpoet.lab_arena.scoring_plan.v1"
RUN_RESULT_SCHEMA_VERSION = "leadpoet.lab_arena.run_result.v1"
PUBLICATION_SCHEMA_VERSION = "leadpoet.lab_arena.publication.v1"
REWARD_BASIS_SCHEMA_VERSION = "leadpoet.lab_arena.reward_basis.v1"
OUTPUT_DOCUMENT_SCHEMA_VERSION = "leadpoet.lab_arena.output.v1"
SUBMISSION_SCHEMA_VERSION = "leadpoet.lab_arena.submission.v1"
PROVIDER_CALL_SCHEMA_VERSION = "leadpoet.lab_arena.provider_call.v1"
SIGNING_KEY_DOCUMENT_SCHEMA_VERSION = "leadpoet.lab_arena.signing_key.v1"

SCOPE_CLAIM = "lab_arena.claim.v1"
SCOPE_COMPLETE = "lab_arena.complete.v1"
SCOPE_SUBMISSION_PRESIGN = "lab_arena.submission.presign.v1"
SCOPE_SUBMISSION_FINALIZE = "lab_arena.submission.finalize.v1"
REQUEST_SCOPES = frozenset(
    {
        SCOPE_CLAIM,
        SCOPE_COMPLETE,
        SCOPE_SUBMISSION_PRESIGN,
        SCOPE_SUBMISSION_FINALIZE,
    }
)

# ---------------------------------------------------------------------------
# Closed state vocabularies (section 11)
# ---------------------------------------------------------------------------

# Validators execute every stage and then score it: ``stageN_closed`` commits
# the scoring plan, ``stageN_scoring`` is the window in which validators claim
# scoring assignments, ``stageN_judged`` is that window closed with every
# assignment terminal, and the scored state is reached once the Arena has
# verified the breakdowns and recorded the per-run scores.
ROUND_STATUSES = (
    "open",
    "committed",
    "stage1",
    "stage1_closed",
    "stage1_scoring",
    "stage1_judged",
    "stage1_scored",
    "stage2",
    "stage2_closed",
    "stage2_scoring",
    "stage2_judged",
    "scored",
    "published",
    "cancelled",
)
ROUND_TRANSITIONS = {
    "open": ("committed", "cancelled"),
    "committed": ("stage1", "cancelled"),
    "stage1": ("stage1_closed", "cancelled"),
    "stage1_closed": ("stage1_scoring", "cancelled"),
    "stage1_scoring": ("stage1_judged", "cancelled"),
    "stage1_judged": ("stage1_scored", "cancelled"),
    "stage1_scored": ("stage2", "cancelled"),
    "stage2": ("stage2_closed", "cancelled"),
    "stage2_closed": ("stage2_scoring", "cancelled"),
    "stage2_scoring": ("stage2_judged", "cancelled"),
    "stage2_judged": ("scored", "cancelled"),
    "scored": ("published", "cancelled"),
    "published": (),
    "cancelled": (),
}
ATTEMPT_STATUSES = ("pending", "leased", "submitted", "accepted", "failed")
SUBMISSION_STATUSES = ("uploading", "accepted", "rejected", "frozen")
KING_OUTCOMES = ("crowned", "defended", "retained_ineligible", "no_king")
TERMINAL_CAUSES = (
    "accepted",
    "model_timeout",
    "invalid_output",
    "budget_exhausted",
    "credential_error",
    "model_error",
    "lease_expired",
    "worker_lost",
    "result_rejected",
    "provider_error",
    "stage_closed",
    # Scoring assignments: the judge failed or timed out (infrastructure).
    "judge_error",
    "judge_timeout",
)
# Causes that the miner caused. A crash, timeout, or invalid output still
# gets one confirmation attempt (another validator when there is one); a
# quota exhaustion does not, and a second failure stands as the zero.
MODEL_CAUSED_TERMINAL_CAUSES = frozenset(
    {"model_timeout", "invalid_output", "budget_exhausted", "credential_error", "model_error"}
)
SCORE_TERMINAL_CAUSES = ("accepted", "credential_error", "judge_error", "judge_timeout")
# Causes that infrastructure caused: a second attempt with a fresh per-ICP cap.
INFRASTRUCTURE_TERMINAL_CAUSES = frozenset({"lease_expired", "worker_lost", "result_rejected", "provider_error", "judge_error", "judge_timeout"})

PROVIDER_CALL_STATES = ("reserved", "dispatched", "settled", "uncertain", "recovered", "refused")
LEDGER_ENTRY_KINDS = (
    "reservation",
    "dispatch",
    "settlement",
    "uncertain",
    "recovery",
    "refusal",
)

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ArenaContractError(ValueError):
    """A document violates an Arena contract. Always fail closed on it."""


class ArenaSignatureError(ArenaContractError):
    """A signature or signing-key binding failed verification."""


def canonical_json(value: Any) -> str:
    """Return stable JSON for Arena requests and results."""

    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ArenaContractError("value is not canonical JSON: %s" % exc) from exc


def sha256_bytes(value: bytes) -> str:
    if not isinstance(value, bytes):
        raise ArenaContractError("sha256 input must be bytes")
    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


# ---------------------------------------------------------------------------
# Strict document limits (sections 6.1, 9.1, 14)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StrictLimits:
    max_depth: int = 12
    max_list_items: int = 512
    max_object_keys: int = 256
    max_string_bytes: int = 16_384
    max_total_bytes: int = 1_048_576
    max_abs_int: int = 2 ** 53


REQUEST_LIMITS = StrictLimits(
    max_depth=10,
    max_list_items=256,
    max_object_keys=128,
    max_string_bytes=8_192,
    max_total_bytes=262_144,
)
# A provider frame carries one operation's parameters; the operation table
# allows chat contents of 32,000 characters (the judge sends page content),
# so the frame's structural limits must not undercut the table's own.
PROVIDER_FRAME_LIMITS = StrictLimits(
    max_depth=14,
    max_list_items=256,
    max_object_keys=256,
    max_string_bytes=131_072,
    max_total_bytes=1_100_000,
)
OUTPUT_LIMITS = StrictLimits(
    max_depth=8,
    max_list_items=200,
    max_object_keys=64,
    max_string_bytes=4_096,
    max_total_bytes=524_288,
)
PUBLICATION_LIMITS = StrictLimits(
    max_depth=16,
    max_list_items=20_000,
    max_object_keys=512,
    max_string_bytes=65_536,
    max_total_bytes=64 * 1_048_576,
)
# A completion can contain the judge's accepted 2 MiB scoring output. Keep one
# small allowance for the signed request fields that wrap that output.
COMPLETION_REQUEST_LIMITS = StrictLimits(
    max_depth=16,
    max_list_items=20_000,
    max_object_keys=512,
    max_string_bytes=65_536,
    max_total_bytes=(2 * 1_048_576) + 65_536,
)

_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_SURROGATE_RE = re.compile(r"[\ud800-\udfff]")


def _check_string(value: str, limits: StrictLimits, path: str) -> None:
    if len(value.encode("utf-8", errors="surrogatepass")) > limits.max_string_bytes:
        raise ArenaContractError("string too long at %s" % path)
    if _CONTROL_CHAR_RE.search(value):
        raise ArenaContractError("control character at %s" % path)
    if _SURROGATE_RE.search(value):
        raise ArenaContractError("unpaired surrogate at %s" % path)


def _check_value(value: Any, limits: StrictLimits, depth: int, path: str) -> None:
    if depth > limits.max_depth:
        raise ArenaContractError("document nesting exceeds %d at %s" % (limits.max_depth, path))
    if value is None or isinstance(value, bool):
        return
    if isinstance(value, int):
        if abs(value) > limits.max_abs_int:
            raise ArenaContractError("integer out of range at %s" % path)
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ArenaContractError("non-finite number at %s" % path)
        return
    if isinstance(value, str):
        _check_string(value, limits, path)
        return
    if isinstance(value, (list, tuple)):
        if len(value) > limits.max_list_items:
            raise ArenaContractError("list too long at %s" % path)
        for index, item in enumerate(value):
            _check_value(item, limits, depth + 1, "%s[%d]" % (path, index))
        return
    if isinstance(value, Mapping):
        if len(value) > limits.max_object_keys:
            raise ArenaContractError("object has too many keys at %s" % path)
        for key, item in value.items():
            if not isinstance(key, str):
                raise ArenaContractError("non-string key at %s" % path)
            _check_string(key, limits, path + "." + key)
            _check_value(item, limits, depth + 1, path + "." + key)
        return
    raise ArenaContractError("unsupported value type %s at %s" % (type(value).__name__, path))


def check_strict_document(value: Any, limits: StrictLimits = REQUEST_LIMITS) -> None:
    """Reject documents that exceed depth, size, list, string, or numeric limits.

    NaN, Infinity, control characters, unpaired surrogates, non-string keys and
    unsupported Python types all fail. The total canonical byte size is checked
    last so a huge document fails on structure before it is re-encoded.
    """

    _check_value(value, limits, 0, "$")
    encoded = canonical_json(value).encode("utf-8")
    if len(encoded) > limits.max_total_bytes:
        raise ArenaContractError("document exceeds %d bytes" % limits.max_total_bytes)


def require_only_keys(document: Mapping[str, Any], allowed: Iterable[str], *, path: str = "$") -> None:
    unknown = sorted(set(document) - set(allowed))
    if unknown:
        raise ArenaContractError("unknown fields at %s: %s" % (path, ", ".join(unknown)))


def require_keys(document: Mapping[str, Any], required: Iterable[str], *, path: str = "$") -> None:
    missing = sorted(set(required) - set(document))
    if missing:
        raise ArenaContractError("missing fields at %s: %s" % (path, ", ".join(missing)))


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
HEX32_RE = re.compile(r"^[0-9a-f]{32}$")
SS58_RE = re.compile(r"^[1-9A-HJ-NP-Za-km-z]{46,48}$")
ROUND_ID_RE = re.compile(r"^arena-[0-9]{4}-[0-9]{2}-[0-9]{2}(?:-[a-z0-9]{1,16})?$")
SUBMISSION_ID_RE = re.compile(r"^[A-Za-z0-9._:-]{1,64}$")


def document_hash(document: Any) -> str:
    """Return ``sha256:<hex>`` over the canonical JSON encoding."""

    return sha256_json(document)


def hash_bytes(data: bytes) -> str:
    return sha256_bytes(data)


def require_sha256(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.match(value):
        raise ArenaContractError("%s must be a sha256 hash" % field_name)
    return value


def require_hotkey(value: Any, field_name: str = "hotkey") -> str:
    if not isinstance(value, str) or not SS58_RE.match(value):
        raise ArenaContractError("%s must be an SS58 address" % field_name)
    return value


def new_request_id() -> str:
    return secrets.token_hex(16)


# ---------------------------------------------------------------------------
# Signed hotkey request envelope (section 9.1)
# ---------------------------------------------------------------------------

SIGNED_REQUEST_FIELDS = (
    "schema_version",
    "scope",
    "round_id",
    "hotkey",
    "timestamp",
    "request_id",
    "body",
)


def signed_request_message(envelope: Mapping[str, Any]) -> str:
    """Canonical bytes a hotkey signs: the envelope without ``signature``."""

    payload = {key: envelope[key] for key in SIGNED_REQUEST_FIELDS}
    return canonical_json(payload)


def build_signed_request(
    *,
    scope: str,
    round_id: str,
    hotkey: str,
    body: Mapping[str, Any],
    timestamp: int,
    request_id: Optional[str] = None,
    sign_message: Any,
) -> Dict[str, Any]:
    """Build one signed request. ``sign_message(message: str) -> hex signature``."""

    if scope not in REQUEST_SCOPES:
        raise ArenaContractError("unknown request scope")
    envelope: Dict[str, Any] = {
        "schema_version": SIGNED_REQUEST_SCHEMA_VERSION,
        "scope": scope,
        "round_id": str(round_id),
        "hotkey": require_hotkey(hotkey),
        "timestamp": int(timestamp),
        "request_id": request_id or new_request_id(),
        "body": dict(body),
    }
    signature = sign_message(signed_request_message(envelope))
    if not isinstance(signature, str) or not signature:
        raise ArenaContractError("signer returned no signature")
    envelope["signature"] = signature if signature.startswith("0x") else "0x" + signature
    return envelope


def validate_signed_request(
    envelope: Any,
    *,
    expected_scope: str,
    now: int,
    verify_signature: Any,
    expected_round_id: Optional[str] = None,
    limits: Optional[StrictLimits] = None,
    window_seconds: int = REQUEST_TIMESTAMP_WINDOW_SECONDS,
) -> Dict[str, Any]:
    """Validate scope, window, shape, and signature. Returns the envelope.

    ``verify_signature(hotkey, signature_hex, message) -> bool`` is injected so
    this module performs no chain or crypto import of its own.
    """

    if limits is None:
        limits = COMPLETION_REQUEST_LIMITS if expected_scope == SCOPE_COMPLETE else REQUEST_LIMITS
    if not isinstance(envelope, Mapping):
        raise ArenaContractError("signed request must be an object")
    check_strict_document(envelope, limits)
    require_only_keys(envelope, SIGNED_REQUEST_FIELDS + ("signature",))
    require_keys(envelope, SIGNED_REQUEST_FIELDS + ("signature",))
    if envelope["schema_version"] != SIGNED_REQUEST_SCHEMA_VERSION:
        raise ArenaContractError("unsupported signed request schema")
    if envelope["scope"] != expected_scope or expected_scope not in REQUEST_SCOPES:
        raise ArenaContractError("request scope mismatch")
    if expected_round_id is not None and envelope["round_id"] != expected_round_id:
        raise ArenaContractError("request round mismatch")
    if not isinstance(envelope["round_id"], str) or not envelope["round_id"]:
        raise ArenaContractError("request round_id required")
    require_hotkey(envelope["hotkey"])
    timestamp = envelope["timestamp"]
    if isinstance(timestamp, bool) or not isinstance(timestamp, int):
        raise ArenaContractError("request timestamp must be an integer")
    if abs(int(now) - timestamp) > window_seconds:
        raise ArenaContractError("request timestamp outside window")
    request_id = envelope["request_id"]
    if not isinstance(request_id, str) or not HEX32_RE.match(request_id):
        raise ArenaContractError("request_id must be 32 hex characters")
    if not isinstance(envelope["body"], Mapping):
        raise ArenaContractError("request body must be an object")
    signature = envelope["signature"]
    if not isinstance(signature, str) or not re.match(r"^(0x)?[0-9a-f]{128}$", signature):
        raise ArenaContractError("signature must be 64 hex bytes")
    if not verify_signature(envelope["hotkey"], signature, signed_request_message(envelope)):
        raise ArenaSignatureError("request signature invalid")
    return dict(envelope)


def request_bytes_hash(envelope: Mapping[str, Any]) -> str:
    """Hash of the full signed request, used to detect request-id reuse."""

    return document_hash(envelope)


# ---------------------------------------------------------------------------
# Minimal declarative schema engine for Arena documents
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class F:
    """One field specification: ``kind`` is a closed vocabulary below."""

    name: str
    kind: str
    required: bool = True
    choices: Optional[Tuple[Any, ...]] = None
    minimum: Optional[int] = None
    maximum: Optional[int] = None
    fields: Optional[Tuple["F", ...]] = None  # for object / list[object]


_SCALAR_KINDS = {"str", "int", "bool", "float", "sha256", "hotkey", "hex", "iso8601", "any"}
_ISO8601_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z$")


def _check_scalar(spec: F, value: Any, path: str) -> Any:
    kind = spec.kind
    if kind == "any":
        return value
    if kind == "bool":
        if not isinstance(value, bool):
            raise ArenaContractError("%s must be a boolean" % path)
        return value
    if kind == "int":
        if isinstance(value, bool) or not isinstance(value, int):
            raise ArenaContractError("%s must be an integer" % path)
        if spec.minimum is not None and value < spec.minimum:
            raise ArenaContractError("%s below minimum %d" % (path, spec.minimum))
        if spec.maximum is not None and value > spec.maximum:
            raise ArenaContractError("%s above maximum %d" % (path, spec.maximum))
        return value
    if kind == "float":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ArenaContractError("%s must be a number" % path)
        number = float(value)
        if not math.isfinite(number):
            raise ArenaContractError("%s must be finite" % path)
        if spec.minimum is not None and number < spec.minimum:
            raise ArenaContractError("%s below minimum" % path)
        if spec.maximum is not None and number > spec.maximum:
            raise ArenaContractError("%s above maximum" % path)
        return number
    if not isinstance(value, str):
        raise ArenaContractError("%s must be a string" % path)
    if kind == "sha256":
        require_sha256(value, path)
    elif kind == "hotkey":
        require_hotkey(value, path)
    elif kind == "hex":
        if not re.match(r"^[0-9a-f]+$", value) or len(value) % 2:
            raise ArenaContractError("%s must be lowercase hex" % path)
    elif kind == "iso8601":
        if not _ISO8601_RE.match(value):
            raise ArenaContractError("%s must be an ISO-8601 UTC timestamp" % path)
    elif kind == "str":
        if spec.minimum is not None and len(value) < spec.minimum:
            raise ArenaContractError("%s too short" % path)
        if spec.maximum is not None and len(value) > spec.maximum:
            raise ArenaContractError("%s too long" % path)
    else:
        raise ArenaContractError("unknown scalar kind %s" % kind)
    if spec.choices is not None and value not in spec.choices:
        raise ArenaContractError("%s must be one of %s" % (path, ", ".join(map(str, spec.choices))))
    return value


def _check_field(spec: F, value: Any, path: str) -> Any:
    kind = spec.kind
    if kind in _SCALAR_KINDS:
        return _check_scalar(spec, value, path)
    if kind == "object":
        if not isinstance(value, Mapping):
            raise ArenaContractError("%s must be an object" % path)
        if spec.fields is None:
            return dict(value)
        return validate_document(value, spec.fields, path=path)
    if kind.startswith("list["):
        inner_kind = kind[5:-1]
        if not isinstance(value, (list, tuple)):
            raise ArenaContractError("%s must be a list" % path)
        if spec.minimum is not None and len(value) < spec.minimum:
            raise ArenaContractError("%s needs at least %d items" % (path, spec.minimum))
        if spec.maximum is not None and len(value) > spec.maximum:
            raise ArenaContractError("%s has too many items" % path)
        inner = F(spec.name, inner_kind, fields=spec.fields, choices=spec.choices)
        return [_check_field(inner, item, "%s[%d]" % (path, index)) for index, item in enumerate(value)]
    if kind == "map[int]":
        if not isinstance(value, Mapping):
            raise ArenaContractError("%s must be an object" % path)
        out: Dict[str, int] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ArenaContractError("%s keys must be strings" % path)
            out[key] = _check_scalar(F(key, "int", minimum=spec.minimum), item, path + "." + key)
        return out
    raise ArenaContractError("unknown field kind %s" % kind)


def validate_document(document: Any, fields: Sequence[F], *, path: str = "$") -> Dict[str, Any]:
    """Validate ``document`` against ``fields`` with unknown-field rejection."""

    if not isinstance(document, Mapping):
        raise ArenaContractError("%s must be an object" % path)
    names = [spec.name for spec in fields]
    require_only_keys(document, names, path=path)
    out: Dict[str, Any] = {}
    for spec in fields:
        if spec.name not in document:
            if spec.required:
                raise ArenaContractError("missing field %s.%s" % (path, spec.name))
            continue
        value = document[spec.name]
        if value is None and not spec.required:
            out[spec.name] = None
            continue
        out[spec.name] = _check_field(spec, value, path + "." + spec.name)
    return out


def hashed_document(document: Mapping[str, Any], hash_field: str) -> Dict[str, Any]:
    """Return a copy with ``hash_field`` set to the hash of the other fields."""

    body = {key: value for key, value in document.items() if key not in (hash_field, "signature")}
    out = dict(body)
    out[hash_field] = document_hash(body)
    return out


def verify_hashed_document(document: Mapping[str, Any], hash_field: str) -> str:
    body = {key: value for key, value in document.items() if key not in (hash_field, "signature")}
    expected = document_hash(body)
    if document.get(hash_field) != expected:
        raise ArenaContractError("%s does not match document contents" % hash_field)
    return expected


# ---------------------------------------------------------------------------
# Round configuration (section 5.1)
# ---------------------------------------------------------------------------

STAGE_SCHEDULE_FIELDS = (
    F("submission_open", "iso8601"),
    F("submission_cutoff", "iso8601"),
    F("benchmark_deadline", "iso8601"),
    F("stage_1_start", "iso8601"),
    F("stage_1_close", "iso8601"),
    F("stage_1_scoring_close", "iso8601"),
    F("stage_2_start", "iso8601"),
    F("stage_2_close", "iso8601"),
    F("final_scoring_close", "iso8601"),
    F("publication_deadline", "iso8601"),
)

REWARD_CONSTANTS_FIELDS = (
    F("pool_percent", "int", minimum=0, maximum=100),
    F("pool_basis", "str", choices=(LAB_ARENA_POOL_BASIS,)),
    F("king_pool_share_percent_by_week", "list[int]", minimum=5, maximum=5),
    F("epochs_per_reward_week", "int", minimum=1),
    F("eligibility_max_epochs", "int", minimum=1),
)

ROUND_CONFIGURATION_FIELDS = (
    F("schema_version", "str", choices=(ROUND_CONFIGURATION_SCHEMA_VERSION,)),
    F("round_id", "str", minimum=6, maximum=64),
    F("mode", "str", choices=("shadow", "live")),
    F("rewards_enabled", "bool"),
    F("schedule", "object", fields=STAGE_SCHEDULE_FIELDS),
    F("stage_1_icp_count", "int", minimum=1),
    F("stage_2_icp_count", "int", minimum=1),
    F("finalist_count", "int", minimum=1),
    F("max_challengers", "int", minimum=1),
    F("runner_slot_ceiling", "int", minimum=1),
    F("max_attempts_per_assignment", "int", minimum=1, maximum=2),
    F("lease_ttl_seconds", "int", minimum=60),
    F("companies_per_icp", "int", minimum=1, maximum=50),
    F("providers", "list[str]", minimum=1, maximum=8),
    F("call_quotas", "object", fields=tuple(F(provider, "int", minimum=1) for provider in PROVIDERS)),
    F("scoring_call_quotas", "object", fields=tuple(F(provider, "int", minimum=1) for provider in PROVIDERS)),
    F("icp_wall_clock_seconds", "int", minimum=30),
    F("scoring_wall_clock_seconds", "int", minimum=30),
    F("scorer_policy", "object"),
    F("execution_cap_microusd", "int", minimum=1),
    F("scoring_cap_microusd", "int", minimum=1),
    F("scorer_image_digest", "sha256"),
    F("scorer_image_reference", "str", minimum=1, maximum=512),
    F("baseline_hotkey", "hotkey"),
    F("baseline_source_url", "str", minimum=8, maximum=2048),
    F("runner_hotkeys", "list[hotkey]", minimum=1, maximum=64),
    F("banned_hotkeys", "list[hotkey]", minimum=0, maximum=4096),
    F("reward_constants", "object", fields=REWARD_CONSTANTS_FIELDS),
)


def validate_round_configuration(document: Any) -> Dict[str, Any]:
    config = validate_document(document, ROUND_CONFIGURATION_FIELDS)
    if not ROUND_ID_RE.match(config["round_id"]):
        raise ArenaContractError("round_id has an invalid shape")
    if config["stage_1_icp_count"] != STAGE_1_ICP_COUNT or config["stage_2_icp_count"] != STAGE_2_ICP_COUNT:
        raise ArenaContractError("stage ICP counts are fixed public constants")
    if config["finalist_count"] != FINALIST_COUNT:
        raise ArenaContractError("the finalist count is a fixed public constant")
    if config["max_attempts_per_assignment"] != MAX_ATTEMPTS_PER_ASSIGNMENT:
        raise ArenaContractError("attempt limit is a fixed public constant")
    if config["runner_slot_ceiling"] > RUNNER_SLOT_CEILING:
        raise ArenaContractError("runner slot ceiling exceeds the public constant")
    if config["max_challengers"] > MAX_CHALLENGERS:
        raise ArenaContractError("max challengers exceeds the public constant")
    if tuple(config["providers"]) != PROVIDERS or dict(config["call_quotas"]) != dict(CALL_QUOTAS_PER_ICP):
        raise ArenaContractError("providers and call quotas are fixed public constants")
    if dict(config["scoring_call_quotas"]) != dict(SCORING_CALL_QUOTAS_PER_WORK_ITEM):
        raise ArenaContractError("scoring call quotas are fixed public constants")
    if not config["scorer_image_reference"].endswith("@" + config["scorer_image_digest"]):
        raise ArenaContractError("scorer image reference must pin the scorer image digest")
    if not config["baseline_source_url"].startswith("https://"):
        raise ArenaContractError("baseline source URL must use https")
    rewards = config["reward_constants"]
    # ``pool_percent`` is the one adjustable reward setting (LAB_ARENA_POOL_PERCENT,
    # 0..100 by the field bounds); the decay, week length, and window are fixed.
    if (
        rewards["pool_basis"] != LAB_ARENA_POOL_BASIS
        or tuple(rewards["king_pool_share_percent_by_week"]) != KING_POOL_SHARE_PERCENT_BY_WEEK
        or rewards["epochs_per_reward_week"] != EPOCHS_PER_REWARD_WEEK
        or rewards["eligibility_max_epochs"] != ELIGIBILITY_MAX_EPOCHS
    ):
        raise ArenaContractError("reward decay constants are fixed public constants")
    if len(set(config["runner_hotkeys"])) != len(config["runner_hotkeys"]):
        raise ArenaContractError("runner hotkeys have duplicates")
    if len(set(config["banned_hotkeys"])) != len(config["banned_hotkeys"]):
        raise ArenaContractError("banned hotkeys have duplicates")
    if set(config["runner_hotkeys"]) & set(config["banned_hotkeys"]):
        raise ArenaContractError("a banned hotkey cannot run Arena work")
    schedule = config["schedule"]
    ordered = [schedule[spec.name] for spec in STAGE_SCHEDULE_FIELDS]
    if ordered != sorted(ordered) or len(set(ordered)) != len(ordered):
        raise ArenaContractError("stage schedule must be strictly increasing")
    return config


# ---------------------------------------------------------------------------
# Benchmark generation attempt log
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Submission body
# ---------------------------------------------------------------------------

SUBMISSION_CONSENT_FIELDS = (
    F("public_rerun", "bool"),
)
SUBMISSION_PRESIGN_BODY_FIELDS = (
    F("source_size_bytes", "int", minimum=1, maximum=10 * 1024 * 1024),
    F("source_content_md5", "str", required=False, minimum=24, maximum=24),
    F("consent", "object", fields=SUBMISSION_CONSENT_FIELDS),
)
SUBMISSION_FINALIZE_BODY_FIELDS = (
    F("submission_id", "str", minimum=1, maximum=64),
    F("source_ref", "str", minimum=1, maximum=512),
    F("source_size_bytes", "int", minimum=1, maximum=10 * 1024 * 1024),
    F("credentials", "object", fields=(
        F("openrouter_api_key", "str", minimum=16, maximum=4096),
        F("openrouter_management_key", "str", minimum=16, maximum=4096),
        F("deepline_api_key", "str", minimum=16, maximum=4096),
    )),
)


def validate_source_content_md5(value: Any) -> str:
    """Return one canonical base64-encoded 128-bit transport checksum."""

    if not isinstance(value, str):
        raise ArenaContractError("source_content_md5 must be base64 MD5")
    try:
        decoded = base64.b64decode(value, validate=True)
    except (ValueError, TypeError):
        raise ArenaContractError("source_content_md5 must be base64 MD5") from None
    if len(decoded) != 16 or base64.b64encode(decoded).decode("ascii") != value:
        raise ArenaContractError("source_content_md5 must be base64 MD5")
    return value


def validate_submission_presign_body(body: Any) -> Dict[str, Any]:
    """Validate transport facts before the Arena permits one source upload."""

    document = validate_document(body, SUBMISSION_PRESIGN_BODY_FIELDS)
    checksum = document.get("source_content_md5")
    if checksum is not None:
        validate_source_content_md5(checksum)
    consent = document["consent"]
    if consent.get("public_rerun") is not True:
        raise ArenaContractError("public_rerun consent must be true")
    return document


def validate_submission_finalize_body(body: Any) -> Dict[str, Any]:
    """Validate the facts repeated after the source upload."""

    document = validate_document(body, SUBMISSION_FINALIZE_BODY_FIELDS)
    if not SUBMISSION_ID_RE.match(document["submission_id"]):
        raise ArenaContractError("submission_id has an invalid shape")
    if not re.match(r"^arena/[A-Za-z0-9._:-]{1,64}/sources/[A-Za-z0-9._:-]{1,64}\.tar\.gz$", document["source_ref"]):
        raise ArenaContractError("source_ref has an invalid shape")
    return document


# ---------------------------------------------------------------------------
# Scorer policy and scoring plan (section 12.1)
# ---------------------------------------------------------------------------

SCORER_POLICY_FIELDS = (
    F("schema_version", "str", choices=(SCORER_POLICY_SCHEMA_VERSION,)),
    F("scoring_adapter_version", "str", minimum=1, maximum=64),
    F("fp_penalty_points", "float", minimum=0, maximum=10),
    F("fp_unverified_primary_penalty_points", "float", minimum=0, maximum=10),
    F("fp_penalty_icp_floor", "float", minimum=-100, maximum=0),
    F("company_cap_rule", "str", choices=("icp_max_companies",)),
    F("max_scored_companies", "int", minimum=0, maximum=50),
    F("judge_models", "object"),
    F("provider_profile", "str", minimum=1, maximum=64),
    F("pre_slice_rule", "str", choices=("first_n_model_order",)),
    F("employee_bucket_rule", "str", choices=("lab_relaxed_buckets",)),
    F("env_bindings", "object"),
)


def validate_scorer_policy(document: Any) -> Dict[str, Any]:
    policy = validate_document(document, SCORER_POLICY_FIELDS)
    for key, value in policy["env_bindings"].items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ArenaContractError("env_bindings must map strings to strings")
    return policy


# One work item per accepted run. Plain row identifiers link it to the output.
SCORING_WORK_ITEM_FIELDS = (
    F("scored_run_id", "str", minimum=1, maximum=128),
    F("icp_position", "int", minimum=0, maximum=BENCHMARK_ICP_COUNT - 1),
    F("submission_id", "str", minimum=1, maximum=64),
    F("output_ref", "str", minimum=1, maximum=512),
)

SCORING_PLAN_FIELDS = (
    F("schema_version", "str", choices=(SCORING_PLAN_SCHEMA_VERSION,)),
    F("round_id", "str", minimum=6, maximum=64),
    F("stage", "int", minimum=1, maximum=2),
    F("work_items", "list[object]", fields=SCORING_WORK_ITEM_FIELDS, minimum=0, maximum=BENCHMARK_ICP_COUNT * (MAX_CHALLENGERS + 1)),
    F("zero_rows", "list[object]", fields=(
        F("submission_id", "str", minimum=1, maximum=64),
        F("icp_position", "int", minimum=0, maximum=BENCHMARK_ICP_COUNT - 1),
        F("cause", "str", choices=TERMINAL_CAUSES),
    ), minimum=0, maximum=BENCHMARK_ICP_COUNT * (MAX_CHALLENGERS + 1)),
)


def validate_scoring_plan(document: Any) -> Dict[str, Any]:
    plan = validate_document(document, SCORING_PLAN_FIELDS)
    positions = set(stage_positions(int(plan["stage"])))
    seen_runs: set = set()
    seen_positions: set = set()
    for item in plan["work_items"]:
        position_key = (item["submission_id"], int(item["icp_position"]))
        if int(item["icp_position"]) not in positions:
            raise ArenaContractError("work item is outside its stage")
        if item["scored_run_id"] in seen_runs or position_key in seen_positions:
            raise ArenaContractError("duplicate work item")
        seen_runs.add(item["scored_run_id"])
        seen_positions.add(position_key)
    for item in plan["zero_rows"]:
        position_key = (item["submission_id"], int(item["icp_position"]))
        if int(item["icp_position"]) not in positions or position_key in seen_positions:
            raise ArenaContractError("zero row is duplicate or outside its stage")
        seen_positions.add(position_key)
    return plan


# ---------------------------------------------------------------------------
# Run result
# ---------------------------------------------------------------------------

RUN_RESULT_FIELDS = (
    F("schema_version", "str", choices=(RUN_RESULT_SCHEMA_VERSION,)),
    F(
        "resource_summary",
        "object",
        fields=(
            F("wall_seconds", "float", minimum=0),
            F("cpu_seconds", "float", minimum=0),
            F("max_rss_bytes", "int", minimum=0),
            F("stdout_bytes", "int", minimum=0),
            F("stderr_bytes", "int", minimum=0),
            F("provider_call_count", "int", minimum=0),
        ),
    ),
    F("started_at", "iso8601"),
    F("finished_at", "iso8601"),
    F("terminal_status", "str", choices=("accepted", "model_timeout", "invalid_output", "budget_exhausted", "credential_error", "model_error", "provider_error", "judge_error", "judge_timeout")),
)


def validate_run_result(document: Any) -> Dict[str, Any]:
    """Validate the small worker result carried by an authenticated completion."""

    return validate_document(document, RUN_RESULT_FIELDS)


# ---------------------------------------------------------------------------
# Reward basis (section 13.4)
# ---------------------------------------------------------------------------

REWARD_BASIS_FIELDS = (
    F("schema_version", "str", choices=(REWARD_BASIS_SCHEMA_VERSION,)),
    F("round_id", "str", minimum=6, maximum=64),
    F("published_at", "iso8601"),
    F("effective_reward_epoch", "int", minimum=0),
    F("king_hotkey", "str", maximum=48),
    F("king_outcome", "str", choices=KING_OUTCOMES),
    F("king_start_epoch", "int", minimum=0),
    F("reward_constants", "object", fields=REWARD_CONSTANTS_FIELDS),
    F("reward_basis_hash", "sha256", required=False),
    F("signature", "object", required=False),
)


def validate_reward_basis(document: Any) -> Dict[str, Any]:
    basis = validate_document(document, REWARD_BASIS_FIELDS)
    if basis["king_outcome"] == "no_king":
        if basis["king_hotkey"] != "":
            raise ArenaContractError("no_king outcome cannot name a king")
    else:
        require_hotkey(basis["king_hotkey"], "king_hotkey")
    if "reward_basis_hash" in basis:
        verify_hashed_document(basis, "reward_basis_hash")
    return basis


def finalize_reward_basis(document: Mapping[str, Any]) -> Dict[str, Any]:
    unsigned = {k: v for k, v in document.items() if k not in ("reward_basis_hash", "signature")}
    validate_reward_basis(unsigned)
    return hashed_document(unsigned, "reward_basis_hash")


# ---------------------------------------------------------------------------
# Provider call identity
# ---------------------------------------------------------------------------

def provider_call_identity(
    *,
    assignment_id: str,
    attempt: int,
    icp_position: int,
    action_sequence: int,
    operation_id: str,
    request_hash: str,
) -> str:
    """Worker-owned call identity; the model supplies no identity field.

    The identity is per attempt: a retried attempt is its own run with its own
    lease, quota, and ledger lineage, so its calls never collide with the
    entries of the attempt it replaces.
    """

    return document_hash(
        {
            "call": PROVIDER_CALL_SCHEMA_VERSION,
            "assignment_id": str(assignment_id),
            "attempt": int(attempt),
            "icp_position": int(icp_position),
            "action_sequence": int(action_sequence),
            "operation_id": str(operation_id),
            "request_hash": require_sha256(request_hash, "request_hash"),
        }
    )
