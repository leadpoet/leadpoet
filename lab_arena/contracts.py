"""Lab Arena V1 contracts: constants, strict document limits, canonical hashing,
signed request envelopes, and the document schemas every Arena module shares.

Canonical encoding is ``leadpoet_canonical.attested_v2.canonical_json`` and
``sha256_json``; nothing here performs I/O. This module is the vocabulary of
the Arena and is imported by every other ``lab_arena`` module.
"""

from __future__ import annotations

import math
import re
import secrets
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from leadpoet_canonical.attested_v2 import canonical_json, sha256_bytes, sha256_json

# ---------------------------------------------------------------------------
# Public V1 constants (labarena.md section 1)
# ---------------------------------------------------------------------------

# One stage: every participant runs the same 30 ICPs and there is no finalist
# cut (owner decision, 2026-09-03). The stage is numbered 1 in every document.
STAGE_1_ICP_COUNT = 30
BENCHMARK_ICP_COUNT = STAGE_1_ICP_COUNT
MAX_CHALLENGERS = 256  # one entry per registered miner; each round pins its own admitted ceiling at or below this
RUNNER_SLOT_CEILING = 8
MAX_ATTEMPTS_PER_ASSIGNMENT = 2
LAB_ARENA_POOL_PERCENT = 25  # default share of total emissions for the king's pool; LAB_ARENA_POOL_PERCENT overrides it per round
# The pool is a share of total emissions, not of what remains after the other
# allocations (owner decision, 2026-09-03).
LAB_ARENA_POOL_BASIS = "total_emissions"
KING_POOL_SHARE_PERCENT_BY_WEEK = (100, 80, 60, 40, 20)
EPOCHS_PER_REWARD_WEEK = 140
ELIGIBILITY_MAX_EPOCHS = 45
# Miners bring their own provider keys (section 7 as revised on 2026-09-02):
# every participant registers one key per provider below, and fairness is a
# fixed call quota per provider, per ICP attempt, instead of a money ceiling.
MINER_KEY_PROVIDERS = ("scrapingdog", "deepline", "openrouter")
CALL_QUOTAS_PER_ICP = {"scrapingdog": 30, "deepline": 30, "openrouter": 60}
# Judge calls made while scoring one work item (one output on one ICP), also
# on the scored miner's keys: verification scrapes, corroboration searches,
# and the three-stage judge for up to five companies.
# Sized from the real judge through the shim (tests/lab_arena/test_lab_arena_real_judge.py):
# per company about 6 Scrapingdog fetches (homepage, Wayback check, evidence
# pages), 3 OpenRouter calls, and 1 Deepline contents call, with retry headroom
# for five companies. A judge that exhausts a quota is the scored miner's zero.
SCORING_CALL_QUOTAS_PER_WORK_ITEM = {"scrapingdog": 150, "deepline": 40, "openrouter": 120}
CALL_QUOTA_SCHEMA_VERSION = "lab_arena.call_quotas.v2"
# Assignment kinds: a validator either executes a miner's model on one ICP or
# scores one output on one ICP with the Arena judge.
ASSIGNMENT_KINDS = ("execute", "score")
ICP_WALL_CLOCK_SECONDS = 300
# A judge run reads pages and calls several models per company against live
# providers; it gets its own wall clock, longer than a model's, under the same
# lease (provider calls refresh the lease) and inside the replay timeout.
SCORING_WALL_CLOCK_SECONDS = 900
LEASE_TTL_SECONDS = 420

# Generation requests are fixed (section 8): 20 + 10 across the ordered
# industry list; the second request covers the first ten industries.
GENERATION_BATCH_SIZES = (20, 10)
MAX_GENERATION_ATTEMPTS = 12

# Signed request timestamp window (section 9.1).
REQUEST_TIMESTAMP_WINDOW_SECONDS = 300

# ---------------------------------------------------------------------------
# Schema versions and scopes
# ---------------------------------------------------------------------------

ARENA_CONTRACT_VERSION = "lab_arena.v1"
SIGNED_REQUEST_SCHEMA_VERSION = "leadpoet.lab_arena.signed_request.v1"
SIGNED_DOCUMENT_SCHEMA_VERSION = "leadpoet.lab_arena.signed_document.v1"
ROUND_CONFIGURATION_SCHEMA_VERSION = "leadpoet.lab_arena.round_configuration.v1"
BENCHMARK_COMMITMENT_SCHEMA_VERSION = "leadpoet.lab_arena.benchmark_commitment.v1"
GENERATION_JOURNAL_SCHEMA_VERSION = "leadpoet.lab_arena.generation_journal.v1"
SCORER_POLICY_SCHEMA_VERSION = "leadpoet.lab_arena.scorer_policy.v1"
SCORING_PLAN_SCHEMA_VERSION = "leadpoet.lab_arena.scoring_plan.v1"
ICP_RECEIPT_SCHEMA_VERSION = "leadpoet.lab_arena.icp_receipt.v1"
SCORE_BUNDLE_SCHEMA_VERSION = "leadpoet.lab_arena.score_bundle.v1"
PUBLICATION_SCHEMA_VERSION = "leadpoet.lab_arena.publication.v1"
REWARD_BASIS_SCHEMA_VERSION = "leadpoet.lab_arena.reward_basis.v1"
OUTPUT_DOCUMENT_SCHEMA_VERSION = "leadpoet.lab_arena.output.v1"
SUBMISSION_SCHEMA_VERSION = "leadpoet.lab_arena.submission.v1"
PROVIDER_CALL_SCHEMA_VERSION = "leadpoet.lab_arena.provider_call.v1"
PRIVATE_EVENT_SCHEMA_VERSION = "leadpoet.lab_arena.private_event.v1"
RECIPIENT_DOCUMENT_SCHEMA_VERSION = "leadpoet.lab_arena.openrouter_recipient.v1"
SIGNING_KEY_DOCUMENT_SCHEMA_VERSION = "leadpoet.lab_arena.signing_key.v1"
PARTICIPANT_SET_SCHEMA_VERSION = "leadpoet.lab_arena.participant_set.v1"
RUNNER_ALLOWLIST_SCHEMA_VERSION = "leadpoet.lab_arena.runner_allowlist.v1"

SCOPE_CLAIM = "lab_arena.claim.v1"
SCOPE_COMPLETE = "lab_arena.complete.v1"
SCOPE_SUBMISSION = "lab_arena.submission.v1"
SCOPE_CREDENTIAL = "lab_arena.credential.v1"
SCOPE_SUBMISSION_STATUS = "lab_arena.submission_status.v1"
REQUEST_SCOPES = frozenset(
    {
        SCOPE_CLAIM,
        SCOPE_COMPLETE,
        SCOPE_SUBMISSION,
        SCOPE_CREDENTIAL,
        SCOPE_SUBMISSION_STATUS,
    }
)

# ---------------------------------------------------------------------------
# Closed state vocabularies (section 11)
# ---------------------------------------------------------------------------

# Validators execute every stage and then score it: ``stageN_closed`` commits
# the scoring plan, ``stageN_scoring`` is the window in which validators claim
# scoring assignments, ``stageN_judged`` is that window closed with every
# assignment terminal, and the scored state is reached once the Arena has
# verified the breakdowns and built the bundle.
ROUND_STATUSES = (
    "open",
    "committed",
    "stage1",
    "stage1_closed",
    "stage1_scoring",
    "stage1_judged",
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
    "stage1_judged": ("scored", "cancelled"),
    "scored": ("published", "cancelled"),
    "published": (),
    "cancelled": (),
}
ATTEMPT_STATUSES = ("pending", "leased", "submitted", "accepted", "failed")
SUBMISSION_STATUSES = ("uploaded", "accepted", "rejected", "frozen")
KING_OUTCOMES = ("crowned", "defended", "retained_ineligible", "no_king")
TERMINAL_CAUSES = (
    "accepted",
    "model_timeout",
    "invalid_output",
    "budget_exhausted",
    "model_error",
    "lease_expired",
    "worker_lost",
    "receipt_rejected",
    "preflight_failed",
    "stage_closed",
    # Scoring assignments: the judge failed or timed out (infrastructure), or
    # the scored miner's own key refused the judge's calls (miner-caused).
    "judge_error",
    "judge_timeout",
    "judge_key_refused",
)
# Causes that the miner caused. A crash, timeout, or invalid output still
# gets one confirmation attempt (another validator when there is one); a
# quota exhaustion does not, and a second failure stands as the zero.
MODEL_CAUSED_TERMINAL_CAUSES = frozenset(
    {"model_timeout", "invalid_output", "budget_exhausted", "model_error", "judge_key_refused"}
)
SCORE_TERMINAL_CAUSES = ("accepted", "judge_error", "judge_timeout", "judge_key_refused")
# Causes that infrastructure caused: a second attempt with a fresh per-ICP cap.
INFRASTRUCTURE_TERMINAL_CAUSES = frozenset({"lease_expired", "worker_lost", "receipt_rejected", "judge_error", "judge_timeout"})

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
EVENT_BATCH_LIMITS = StrictLimits(
    max_depth=8,
    max_list_items=256,
    max_object_keys=64,
    max_string_bytes=16_384,
    max_total_bytes=524_288,
)
PUBLICATION_LIMITS = StrictLimits(
    max_depth=16,
    max_list_items=20_000,
    max_object_keys=512,
    max_string_bytes=65_536,
    max_total_bytes=64 * 1_048_576,
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


def ordered_root(leaves: Sequence[str]) -> str:
    """Ordered commitment over leaf hashes: a hash chain in position order.

    Position is part of every leaf, so reordering two leaves changes the
    root. An empty sequence commits to the empty-list hash.
    """

    running = document_hash({"root": ARENA_CONTRACT_VERSION, "count": len(leaves)})
    for position, leaf in enumerate(leaves):
        require_sha256(leaf, "leaf")
        running = document_hash({"prev": running, "position": position, "leaf": leaf})
    return running


def chain_hash(previous_hash: Optional[str], entry: Mapping[str, Any]) -> str:
    return document_hash({"prev": previous_hash or "", "entry": entry})


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
    limits: StrictLimits = REQUEST_LIMITS,
    window_seconds: int = REQUEST_TIMESTAMP_WINDOW_SECONDS,
) -> Dict[str, Any]:
    """Validate scope, window, shape, and signature. Returns the envelope.

    ``verify_signature(hotkey, signature_hex, message) -> bool`` is injected so
    this module performs no chain or crypto import of its own.
    """

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
    F("schedule", "object", fields=STAGE_SCHEDULE_FIELDS),
    F(
        "generator",
        "object",
        fields=(
            F("prompt_hash", "sha256"),
            F("exclusion_prompt_hash", "sha256"),
            F("model", "str", minimum=1, maximum=128),
            F("settings", "object"),
            F("journal_schema_version", "str", choices=(GENERATION_JOURNAL_SCHEMA_VERSION,)),
            F("batch_sizes", "list[int]", minimum=2, maximum=2),
            F("max_generation_attempts", "int", minimum=3, maximum=64),
        ),
    ),
    F("tie_break_rule", "str", choices=("finalized_block_after_cutoff.v1",)),
    F("stage_1_icp_count", "int", minimum=1),
    F("max_challengers", "int", minimum=1),
    F("runner_slot_ceiling", "int", minimum=1),
    F("max_attempts_per_assignment", "int", minimum=1, maximum=2),
    F("lease_ttl_seconds", "int", minimum=60),
    F("companies_per_icp", "int", minimum=1, maximum=50),
    F(
        "release",
        "object",
        fields=(
            F("repository_commit", "hex"),
            F("runsc_lock_hash", "sha256"),
            F("worker_release_hash", "sha256"),
            F("shim_hash", "sha256"),
            # The Arena-built judge image every validator runs for scoring
            # assignments: its pinned single-platform digest, the reference
            # runners pull, and the entry command pinned from its config.
            F("scorer_image_digest", "sha256"),
            F("scorer_image_reference", "str", minimum=0, maximum=512),
            F("scorer_entry_command", "list[str]", minimum=1, maximum=64),
        ),
    ),
    F("operation_table_hash", "sha256"),
    F("openrouter_price_table_hash", "sha256"),
    F("openrouter_allowed_models", "list[str]", minimum=1, maximum=64),
    F("miner_key_providers", "list[str]", minimum=1, maximum=8),
    F("call_quotas", "object", fields=tuple(F(provider, "int", minimum=1) for provider in MINER_KEY_PROVIDERS)),
    F("scoring_call_quotas", "object", fields=tuple(F(provider, "int", minimum=1) for provider in MINER_KEY_PROVIDERS)),
    F("call_quota_hash", "sha256"),
    F("icp_wall_clock_seconds", "int", minimum=30),
    F("scoring_wall_clock_seconds", "int", minimum=30),
    F("scorer_policy_hash", "sha256"),
    F("scoring_cap_microusd", "int", minimum=0),
    F("runner_allowlist", "list[hotkey]", minimum=1, maximum=4096),
    F("floor_runner_hotkeys", "list[hotkey]", minimum=1, maximum=64),
    F("banned_hotkeys_snapshot_hash", "sha256"),
    F("signing_public_key_hash", "sha256"),
    # Image by digest: the public limits every submitted image must meet and
    # the Arena repository every accepted image is mirrored into.
    F(
        "image_rules",
        "object",
        fields=(
            F("schema_version", "str", minimum=1, maximum=64),
            F("max_image_bytes", "int", minimum=1),
            F("max_layers", "int", minimum=1),
            F("max_rootfs_bytes", "int", minimum=1),
            F("platform", "object", fields=(F("os", "str", minimum=1, maximum=32), F("architecture", "str", minimum=1, maximum=32))),
            F("layer_media_types", "list[str]", minimum=1, maximum=16),
        ),
    ),
    F("registry_repository", "str", minimum=0, maximum=512),
    F("publication_terms_hash", "sha256"),
    F("reward_constants", "object", fields=REWARD_CONSTANTS_FIELDS),
    F("configuration_hash", "sha256", required=False),
    F("signature", "object", required=False),
)


def call_quota_document() -> Dict[str, Any]:
    """The public quota policy hashed into every round configuration.

    A quota is per provider and per ICP attempt; the stage quota is the
    per-ICP quota times the stage's ICP count times the attempt limit, so a
    retry after an infrastructure failure is never starved.
    """

    return {
        "schema_version": CALL_QUOTA_SCHEMA_VERSION,
        "providers": list(MINER_KEY_PROVIDERS),
        "per_icp_attempt": {provider: int(CALL_QUOTAS_PER_ICP[provider]) for provider in MINER_KEY_PROVIDERS},
        "per_scoring_work_item": {provider: int(SCORING_CALL_QUOTAS_PER_WORK_ITEM[provider]) for provider in MINER_KEY_PROVIDERS},
        "stage_multiplier": MAX_ATTEMPTS_PER_ASSIGNMENT,
    }


def validate_round_configuration(document: Any) -> Dict[str, Any]:
    config = validate_document(document, ROUND_CONFIGURATION_FIELDS)
    if not ROUND_ID_RE.match(config["round_id"]):
        raise ArenaContractError("round_id has an invalid shape")
    if config["stage_1_icp_count"] != STAGE_1_ICP_COUNT:
        raise ArenaContractError("the stage ICP count is a fixed public constant")
    if config["max_attempts_per_assignment"] != MAX_ATTEMPTS_PER_ASSIGNMENT:
        raise ArenaContractError("attempt limit is a fixed public constant")
    if config["runner_slot_ceiling"] > RUNNER_SLOT_CEILING:
        raise ArenaContractError("runner slot ceiling exceeds the public constant")
    if config["max_challengers"] > MAX_CHALLENGERS:
        raise ArenaContractError("max challengers exceeds the public constant")
    if tuple(config["miner_key_providers"]) != MINER_KEY_PROVIDERS or dict(config["call_quotas"]) != dict(CALL_QUOTAS_PER_ICP):
        raise ArenaContractError("miner key providers and call quotas are fixed public constants")
    if dict(config["scoring_call_quotas"]) != dict(SCORING_CALL_QUOTAS_PER_WORK_ITEM):
        raise ArenaContractError("scoring call quotas are fixed public constants")
    if config["call_quota_hash"] != document_hash(call_quota_document()):
        raise ArenaContractError("call quota hash does not match the public quota document")
    if tuple(config["generator"]["batch_sizes"]) != GENERATION_BATCH_SIZES:
        raise ArenaContractError("generation batch sizes are fixed public constants")
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
    if not set(config["floor_runner_hotkeys"]) <= set(config["runner_allowlist"]):
        raise ArenaContractError("floor runners must be on the runner allowlist")
    if len(set(config["runner_allowlist"])) != len(config["runner_allowlist"]):
        raise ArenaContractError("runner allowlist has duplicates")
    schedule = config["schedule"]
    ordered = [schedule[spec.name] for spec in STAGE_SCHEDULE_FIELDS]
    if ordered != sorted(ordered) or len(set(ordered)) != len(ordered):
        raise ArenaContractError("stage schedule must be strictly increasing")
    if "configuration_hash" in config:
        verify_hashed_document(config, "configuration_hash")
    return config


def finalize_round_configuration(document: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate and stamp ``configuration_hash`` (signature is added separately)."""

    unsigned = {k: v for k, v in document.items() if k not in ("configuration_hash", "signature")}
    validate_round_configuration(unsigned)
    return hashed_document(unsigned, "configuration_hash")


# ---------------------------------------------------------------------------
# Benchmark commitment and generation journal (sections 5.2 and 8)
# ---------------------------------------------------------------------------

JOURNAL_ENTRY_KINDS = ("request", "response", "unknown", "rejection", "acceptance", "exclusion")

GENERATION_JOURNAL_ENTRY_FIELDS = (
    F("schema_version", "str", choices=(GENERATION_JOURNAL_SCHEMA_VERSION,)),
    F("sequence", "int", minimum=0),
    F("kind", "str", choices=JOURNAL_ENTRY_KINDS),
    F("batch_id", "str", minimum=1, maximum=64),
    F("attempt", "int", minimum=1),
    F("slots", "list[int]", minimum=0, maximum=BENCHMARK_ICP_COUNT),
    F("industries", "list[str]", minimum=0, maximum=BENCHMARK_ICP_COUNT),
    F("request_hash", "sha256", required=False),
    F("response_hash", "sha256", required=False),
    F("response_ref", "str", required=False, maximum=512),
    F("slot", "int", required=False, minimum=0),
    F("icp_hash", "sha256", required=False),
    F("content_hash", "sha256", required=False),
    F("rejection_rule", "str", required=False, maximum=128),
    F("timestamp", "iso8601"),
    F("prev_hash", "str", maximum=71),
    F("entry_hash", "sha256", required=False),
)


def finalize_journal_entry(entry: Mapping[str, Any]) -> Dict[str, Any]:
    body = {k: v for k, v in entry.items() if k != "entry_hash"}
    validated = validate_document(body, [f for f in GENERATION_JOURNAL_ENTRY_FIELDS if f.name != "entry_hash"])
    validated["entry_hash"] = chain_hash(validated["prev_hash"] or None, {k: v for k, v in validated.items() if k != "prev_hash"})
    return validated


def verify_journal_chain(entries: Sequence[Mapping[str, Any]]) -> str:
    """Verify the hash chain and return the head hash (empty for no entries)."""

    previous = ""
    for index, entry in enumerate(entries):
        validated = validate_document(entry, GENERATION_JOURNAL_ENTRY_FIELDS)
        if validated["sequence"] != index:
            raise ArenaContractError("journal sequence gap at %d" % index)
        if validated["prev_hash"] != previous:
            raise ArenaContractError("journal chain broken at %d" % index)
        expected = chain_hash(previous or None, {k: v for k, v in validated.items() if k not in ("prev_hash", "entry_hash")})
        if validated.get("entry_hash") != expected:
            raise ArenaContractError("journal entry hash mismatch at %d" % index)
        previous = expected
    return previous


BENCHMARK_COMMITMENT_FIELDS = (
    F("schema_version", "str", choices=(BENCHMARK_COMMITMENT_SCHEMA_VERSION,)),
    F("round_id", "str", minimum=6, maximum=64),
    F("configuration_hash", "sha256"),
    F("participant_set_hash", "sha256"),
    F("tie_break_block_number", "int", minimum=0),
    F("tie_break_block_hash", "str", minimum=66, maximum=66),
    F("journal_head_hash", "sha256"),
    F("journal_length", "int", minimum=1),
    F("evaluation_date", "str", minimum=10, maximum=10),
    F("benchmark_root", "sha256"),
    F("icp_leaf_hashes", "list[sha256]", minimum=BENCHMARK_ICP_COUNT, maximum=BENCHMARK_ICP_COUNT),
    F("generation_started_at", "iso8601"),
    F("generation_finished_at", "iso8601"),
    F("commitment_hash", "sha256", required=False),
    F("signature", "object", required=False),
)


def icp_leaf_hash(position: int, icp_hash: str) -> str:
    """Leaf binding slot identity separately from ICP content (section 8)."""

    return document_hash({"leaf": "lab_arena.icp_leaf.v1", "position": int(position), "icp_hash": require_sha256(icp_hash, "icp_hash")})


def benchmark_roots(icp_hashes: Sequence[str]) -> Dict[str, Any]:
    if len(icp_hashes) != BENCHMARK_ICP_COUNT:
        raise ArenaContractError("benchmark requires exactly %d ICPs" % BENCHMARK_ICP_COUNT)
    leaves = [icp_leaf_hash(position, value) for position, value in enumerate(icp_hashes)]
    return {
        "icp_leaf_hashes": leaves,
        "benchmark_root": ordered_root(leaves),
    }


def validate_benchmark_commitment(document: Any) -> Dict[str, Any]:
    commitment = validate_document(document, BENCHMARK_COMMITMENT_FIELDS)
    leaves = commitment["icp_leaf_hashes"]
    if commitment["benchmark_root"] != ordered_root(leaves):
        raise ArenaContractError("benchmark root does not match leaves")
    if not re.match(r"^0x[0-9a-f]{64}$", commitment["tie_break_block_hash"]):
        raise ArenaContractError("tie-break block hash must be 0x-prefixed hex")
    if "commitment_hash" in commitment:
        verify_hashed_document(commitment, "commitment_hash")
    return commitment


def finalize_benchmark_commitment(document: Mapping[str, Any]) -> Dict[str, Any]:
    unsigned = {k: v for k, v in document.items() if k not in ("commitment_hash", "signature")}
    validate_benchmark_commitment(unsigned)
    return hashed_document(unsigned, "commitment_hash")


def participant_set_hash(participants: Sequence[Mapping[str, Any]]) -> str:
    """Hash of the frozen participant set, sorted by submission id."""

    rows = sorted(
        (
            {
                "submission_id": str(item["submission_id"]),
                "miner_hotkey": require_hotkey(item["miner_hotkey"], "miner_hotkey"),
                "image_digest": require_sha256(item["image_digest"], "image_digest"),
                "image_reference": str(item.get("image_reference") or ""),
                "is_king": bool(item.get("is_king", False)),
            }
            for item in participants
        ),
        key=lambda row: row["submission_id"],
    )
    return document_hash({"schema_version": PARTICIPANT_SET_SCHEMA_VERSION, "participants": rows})


# ---------------------------------------------------------------------------
# Submission body (image by digest)
# ---------------------------------------------------------------------------

SUBMISSION_CONSENT_FIELDS = (
    F("public_rerun", "bool"),
    F("image_publication", "bool"),
)
SUBMISSION_BODY_FIELDS = (
    F("image_reference", "str", minimum=1, maximum=512),
    F("consent", "object", fields=SUBMISSION_CONSENT_FIELDS),
)


def validate_submission_body(body: Any) -> Dict[str, Any]:
    """A miner's signed submission body: one image reference and both consents."""

    document = validate_document(body, SUBMISSION_BODY_FIELDS)
    consent = document["consent"]
    if consent.get("public_rerun") is not True or consent.get("image_publication") is not True:
        raise ArenaContractError("public_rerun and image_publication consent must both be true")
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
    F("cache_version", "str", minimum=1, maximum=64),
    F("provider_profile", "str", minimum=1, maximum=64),
    F("pre_slice_rule", "str", choices=("first_n_model_order",)),
    F("employee_bucket_rule", "str", choices=("lab_relaxed_buckets",)),
    F("env_bindings", "object"),
    F("policy_hash", "sha256", required=False),
)


def validate_scorer_policy(document: Any) -> Dict[str, Any]:
    policy = validate_document(document, SCORER_POLICY_FIELDS)
    for key, value in policy["env_bindings"].items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ArenaContractError("env_bindings must map strings to strings")
    if "policy_hash" in policy:
        verify_hashed_document(policy, "policy_hash")
    return policy


def finalize_scorer_policy(document: Mapping[str, Any]) -> Dict[str, Any]:
    unsigned = {k: v for k, v in document.items() if k != "policy_hash"}
    validate_scorer_policy(unsigned)
    return hashed_document(unsigned, "policy_hash")


# One work item per accepted assignment: a miner's output on one ICP is judged
# on that miner's own keys and never shared with another miner, even when the
# outputs are byte-identical (results are not cached across miners).
SCORING_WORK_ITEM_FIELDS = (
    F("work_item_id", "sha256"),
    F("icp_position", "int", minimum=0, maximum=BENCHMARK_ICP_COUNT - 1),
    F("icp_hash", "sha256"),
    F("output_hash", "sha256"),
    F("submission_id", "str", minimum=1, maximum=64),
)

SCORING_PLAN_FIELDS = (
    F("schema_version", "str", choices=(SCORING_PLAN_SCHEMA_VERSION,)),
    F("round_id", "str", minimum=6, maximum=64),
    F("stage", "int", minimum=1, maximum=1),
    F("configuration_hash", "sha256"),
    F("commitment_hash", "sha256"),
    F("scorer_policy_hash", "sha256"),
    F("work_items", "list[object]", fields=SCORING_WORK_ITEM_FIELDS, minimum=0, maximum=BENCHMARK_ICP_COUNT * (MAX_CHALLENGERS + 1)),
    F("zero_rows", "list[object]", fields=(
        F("submission_id", "str", minimum=1, maximum=64),
        F("icp_position", "int", minimum=0, maximum=BENCHMARK_ICP_COUNT - 1),
        F("cause", "str", choices=TERMINAL_CAUSES),
    ), minimum=0, maximum=BENCHMARK_ICP_COUNT * (MAX_CHALLENGERS + 1)),
    F("plan_hash", "sha256", required=False),
    F("signature", "object", required=False),
)


def work_item_id(icp_hash: str, submission_id: str, output_hash: str) -> str:
    return document_hash({"work_item": "lab_arena.scoring_work_item.v2", "icp_hash": icp_hash, "submission_id": submission_id, "output_hash": output_hash})


def validate_scoring_plan(document: Any) -> Dict[str, Any]:
    plan = validate_document(document, SCORING_PLAN_FIELDS)
    seen: set = set()
    for item in plan["work_items"]:
        expected = work_item_id(item["icp_hash"], item["submission_id"], item["output_hash"])
        if item["work_item_id"] != expected:
            raise ArenaContractError("work item id does not bind its ICP, submission, and output")
        if item["work_item_id"] in seen:
            raise ArenaContractError("duplicate work item")
        seen.add(item["work_item_id"])
    if "plan_hash" in plan:
        verify_hashed_document(plan, "plan_hash")
    return plan


def finalize_scoring_plan(document: Mapping[str, Any]) -> Dict[str, Any]:
    unsigned = {k: v for k, v in document.items() if k not in ("plan_hash", "signature")}
    validate_scoring_plan(unsigned)
    return hashed_document(unsigned, "plan_hash")


# ---------------------------------------------------------------------------
# ICP receipt (section 9.4)
# ---------------------------------------------------------------------------

ICP_RECEIPT_FIELDS = (
    F("schema_version", "str", choices=(ICP_RECEIPT_SCHEMA_VERSION,)),
    F("round_id", "str", minimum=6, maximum=64),
    F("submission_id", "str", minimum=1, maximum=64),
    F("assignment_id", "str", minimum=1, maximum=64),
    F("attempt", "int", minimum=1, maximum=MAX_ATTEMPTS_PER_ASSIGNMENT),
    F("stage", "int", minimum=1, maximum=1),
    F("icp_position", "int", minimum=0, maximum=BENCHMARK_ICP_COUNT - 1),
    F("lease_generation", "int", minimum=1),
    F("runner_hotkey", "hotkey"),
    F("miner_hotkey", "hotkey"),
    F("worker_release_hash", "sha256"),
    F("image_digest", "str", minimum=1, maximum=200),
    F("icp_hash", "sha256"),
    F("provider_call_root", "sha256"),
    F("private_event_root", "sha256"),
    F("output_hash", "sha256"),
    F("cost_root", "sha256"),
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
    F("terminal_status", "str", choices=("accepted", "model_timeout", "invalid_output", "budget_exhausted", "model_error", "judge_error", "judge_timeout", "judge_key_refused")),
    F("kind", "str", choices=ASSIGNMENT_KINDS, required=False),
    F("receipt_hash", "sha256", required=False),
    F("runner_signature", "str", required=False, minimum=128, maximum=130),
)


def finalize_icp_receipt(document: Mapping[str, Any]) -> Dict[str, Any]:
    unsigned = {k: v for k, v in document.items() if k not in ("receipt_hash", "runner_signature")}
    validated = validate_document(unsigned, [f for f in ICP_RECEIPT_FIELDS if f.name not in ("receipt_hash", "runner_signature")])
    validated["receipt_hash"] = document_hash(validated)
    return validated


def validate_icp_receipt(document: Any, *, verify_signature: Any) -> Dict[str, Any]:
    receipt = validate_document(document, ICP_RECEIPT_FIELDS)
    if "receipt_hash" not in receipt or "runner_signature" not in receipt:
        raise ArenaContractError("receipt must carry its hash and runner signature")
    body = {k: v for k, v in receipt.items() if k not in ("receipt_hash", "runner_signature")}
    if document_hash(body) != receipt["receipt_hash"]:
        raise ArenaContractError("receipt hash mismatch")
    signature = receipt["runner_signature"]
    if not re.match(r"^(0x)?[0-9a-f]{128}$", signature):
        raise ArenaContractError("runner signature must be 64 hex bytes")
    if not verify_signature(receipt["runner_hotkey"], signature, receipt["receipt_hash"]):
        raise ArenaSignatureError("runner receipt signature invalid")
    return receipt


# ---------------------------------------------------------------------------
# Reward basis (section 13.4)
# ---------------------------------------------------------------------------

REWARD_BASIS_FIELDS = (
    F("schema_version", "str", choices=(REWARD_BASIS_SCHEMA_VERSION,)),
    F("round_id", "str", minimum=6, maximum=64),
    F("configuration_hash", "sha256"),
    F("commitment_hash", "sha256"),
    F("result_bundle_hash", "sha256"),
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
# Provider call identity and private events (sections 7.5 and 10)
# ---------------------------------------------------------------------------

EVENT_TYPES = (
    "process_started",
    "process_finished",
    "provider_call",
    "output_validated",
    "output_rejected",
    "stdout",
    "stderr",
    "trajectory",
    "lease_expired",
    "attempt_failed",
)


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


PRIVATE_EVENT_FIELDS = (
    F("event_type", "str", choices=EVENT_TYPES),
    F("sequence", "int", minimum=0),
    F("timestamp", "iso8601"),
    F("payload", "object"),
)


def validate_private_event(document: Any) -> Dict[str, Any]:
    event = validate_document(document, PRIVATE_EVENT_FIELDS)
    check_strict_document(event, EVENT_BATCH_LIMITS)
    return event


def event_root(event_hashes: Sequence[str]) -> str:
    return ordered_root(list(event_hashes))


# ---------------------------------------------------------------------------
# Private event chain helpers shared by the broker, the runner, and the service
# ---------------------------------------------------------------------------


def build_private_event(
    *,
    event_type: str,
    sequence: int,
    prev_hash: str,
    timestamp: str,
    payload: Mapping[str, Any],
) -> Dict[str, Any]:
    """Build one hash-chained private event (section 10).

    ``event_hash`` covers every field except ``prev_hash`` and is chained from
    the previous event's hash, which is what the database checks on append
    and what the service re-verifies at completion.
    """

    body = validate_private_event(
        {"event_type": event_type, "sequence": int(sequence), "timestamp": timestamp, "payload": dict(payload)}
    )
    if not isinstance(prev_hash, str) or (prev_hash and not SHA256_RE.match(prev_hash)):
        raise ArenaContractError("prev_hash must be empty or a sha256 hash")
    body["prev_hash"] = prev_hash
    body["event_hash"] = chain_hash(prev_hash or None, {k: v for k, v in body.items() if k not in ("prev_hash", "event_hash")})
    return body


def verify_event_chain(events: Sequence[Mapping[str, Any]]) -> str:
    """Verify sequence, linkage, and hashes; return the head hash."""

    previous = ""
    for index, event in enumerate(events):
        if not isinstance(event, Mapping):
            raise ArenaContractError("event %d is not an object" % index)
        body = {k: v for k, v in event.items() if k not in ("prev_hash", "event_hash")}
        validate_private_event(body)
        if event.get("sequence") != index:
            raise ArenaContractError("event sequence gap at %d" % index)
        if event.get("prev_hash") != previous:
            raise ArenaContractError("event chain broken at %d" % index)
        expected = chain_hash(previous or None, body)
        if event.get("event_hash") != expected:
            raise ArenaContractError("event hash mismatch at %d" % index)
        previous = expected
    return previous


def private_event_root(events: Sequence[Mapping[str, Any]]) -> str:
    """The receipt's private-event root: the ordered root over event hashes."""

    verify_event_chain(events)
    return ordered_root([str(event["event_hash"]) for event in events])
