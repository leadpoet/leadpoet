"""Lab Arena reward kernel: the champion triple from a signed reward basis.

This is the one kernel every side of the weight path runs: the validator host
proposes with it, the gateway coordinator re-derives with it against the
measured database row, and the canonical weight computation checks with it
that a snapshot's champion triple is exactly what its basis implies. It is
pure: no I/O, no environment, no chain state. ``verify_reward_basis_signature``
is the only function that touches a dependency (``cryptography``), imported
lazily, so importing this module needs the standard library alone.

Every constant the arithmetic needs comes from the signed basis itself
(``reward_constants``): the pool percent of total emissions, the weekly king
shares, the epochs per reward week, and the eligibility window. The Arena
fixes them per round and signs them, so a change (5%, 50%) is one Arena
setting and reaches validators through the next published basis; this
module only bounds them.

Written in Python 3.7 syntax (``typing`` generics, ``# type:`` comments, no
walrus, no PEP 604 unions) because the validator enclave image copies it.
"""

from __future__ import annotations

import base64
import hashlib
import math
from fractions import Fraction
from typing import Any, Dict, List, Mapping, Optional, Sequence

from leadpoet_canonical.attested_v2 import canonical_json, sha256_json

REWARD_BASIS_SCHEMA_VERSION = "leadpoet.lab_arena.reward_basis.v1"
SIGNING_KEY_DOCUMENT_SCHEMA_VERSION = "leadpoet.lab_arena.signing_key.v1"
SIGNING_ALGORITHM = "ECDSA_SHA_256"
SIGNING_KEY_SPEC = "ECC_NIST_P256"
POOL_BASIS_TOTAL_EMISSIONS = "total_emissions"
KING_OUTCOMES = ("crowned", "defended", "retained_ineligible", "no_king")
# Outcomes that pay the king (labarena.md 13.3). Every other outcome returns the
# whole Arena amount to fulfillment.
PAYING_KING_OUTCOMES = ("crowned", "defended")
# The validator and the coordinator pin the Arena signing key by this variable.
SIGNING_KEY_HASH_ENV = "LAB_ARENA_SIGNING_PUBLIC_KEY_HASH"
REWARDS_ENABLED_ENV = "LAB_ARENA_REWARDS_ENABLED"
MAX_WEEK_SHARES = 12

_HASH_PREFIX = "sha256:"
_BASIS_BODY_FIELDS = (
    "schema_version",
    "round_id",
    "configuration_hash",
    "commitment_hash",
    "result_bundle_hash",
    "published_at",
    "effective_reward_epoch",
    "king_hotkey",
    "king_outcome",
    "king_start_epoch",
    "reward_constants",
)
_CONSTANT_FIELDS = (
    "pool_percent",
    "pool_basis",
    "king_pool_share_percent_by_week",
    "epochs_per_reward_week",
    "eligibility_max_epochs",
)


class LabArenaRewardError(ValueError):
    """A reward basis, its signature, or a derived value is invalid."""


# ---------------------------------------------------------------------------
# Input guards (every reader of a reward basis fails closed)
# ---------------------------------------------------------------------------


def _require_epoch(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise LabArenaRewardError("%s must be an integer epoch ordinal" % name)
    if value < 0:
        raise LabArenaRewardError("%s must not be negative" % name)
    return value


def _require_int(value: Any, name: str, minimum: int, maximum: Optional[int] = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise LabArenaRewardError("%s must be an integer" % name)
    if value < minimum or (maximum is not None and value > maximum):
        raise LabArenaRewardError("%s is outside its bounds" % name)
    return value


def _require_hash(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != len(_HASH_PREFIX) + 64 or not value.startswith(_HASH_PREFIX):
        raise LabArenaRewardError("%s must be a sha256 hash" % name)
    digest = value[len(_HASH_PREFIX):]
    if any(char not in "0123456789abcdef" for char in digest):
        raise LabArenaRewardError("%s must be a sha256 hash" % name)
    return value


def _exact(value: float) -> Fraction:
    """Exact rational of a float's shortest decimal representation."""

    return Fraction(repr(float(value)))


def require_king_outcome(value: Any) -> str:
    """Return ``value`` when it is one of the four closed outcomes, else raise."""

    if not isinstance(value, str) or value not in KING_OUTCOMES:
        raise LabArenaRewardError("king_outcome must be one of %s" % ", ".join(KING_OUTCOMES))
    return value


def validate_reward_constants(constants: Any) -> Dict[str, Any]:
    """Bound the signed constants: the Arena chooses them, every side checks them."""

    if not isinstance(constants, Mapping) or set(constants) != set(_CONSTANT_FIELDS):
        raise LabArenaRewardError("reward_constants fields are invalid")
    pool_percent = _require_int(constants["pool_percent"], "pool_percent", 0, 100)
    if constants["pool_basis"] != POOL_BASIS_TOTAL_EMISSIONS:
        raise LabArenaRewardError("pool_basis must be %s" % POOL_BASIS_TOTAL_EMISSIONS)
    weeks = constants["king_pool_share_percent_by_week"]
    if not isinstance(weeks, list) or not weeks or len(weeks) > MAX_WEEK_SHARES:
        raise LabArenaRewardError("king_pool_share_percent_by_week must list 1..%d percents" % MAX_WEEK_SHARES)
    shares = [_require_int(item, "king_pool_share_percent_by_week[]", 0, 100) for item in weeks]
    return {
        "pool_percent": pool_percent,
        "pool_basis": POOL_BASIS_TOTAL_EMISSIONS,
        "king_pool_share_percent_by_week": shares,
        "epochs_per_reward_week": _require_int(constants["epochs_per_reward_week"], "epochs_per_reward_week", 1),
        "eligibility_max_epochs": _require_int(constants["eligibility_max_epochs"], "eligibility_max_epochs", 0),
    }


def _basis_fields(basis: Any) -> Dict[str, Any]:
    """Read the reward-basis fields the kernel needs, failing closed on shape."""

    if not isinstance(basis, Mapping):
        raise LabArenaRewardError("reward basis must be an object")
    outcome = require_king_outcome(basis.get("king_outcome"))
    effective = _require_epoch(basis.get("effective_reward_epoch"), "effective_reward_epoch")
    start = _require_epoch(basis.get("king_start_epoch"), "king_start_epoch")
    if start > effective:
        raise LabArenaRewardError("king_start_epoch cannot follow effective_reward_epoch")
    hotkey = basis.get("king_hotkey")
    if not isinstance(hotkey, str):
        raise LabArenaRewardError("king_hotkey must be a string")
    if outcome == "no_king":
        if hotkey != "":
            raise LabArenaRewardError("no_king outcome cannot name a king")
    elif not hotkey:
        raise LabArenaRewardError("%s outcome requires a king hotkey" % outcome)
    return {
        "king_outcome": outcome,
        "effective_reward_epoch": effective,
        "king_start_epoch": start,
        "king_hotkey": hotkey,
        "reward_constants": validate_reward_constants(basis.get("reward_constants")),
    }


def validate_reward_basis(document: Any) -> Dict[str, Any]:
    """Shape-check a published reward basis and recompute its hash.

    Returns the document unchanged when every field is present and typed,
    ``reward_basis_hash`` equals the hash of the body, and the signature block,
    when present, has the Arena shape. The signature itself is checked by
    :func:`verify_reward_basis_signature` against a pinned key.
    """

    if not isinstance(document, Mapping):
        raise LabArenaRewardError("reward basis must be an object")
    allowed = set(_BASIS_BODY_FIELDS) | {"reward_basis_hash", "signature"}
    if set(document) - allowed or not set(_BASIS_BODY_FIELDS) <= set(document):
        raise LabArenaRewardError("reward basis fields are invalid")
    if document["schema_version"] != REWARD_BASIS_SCHEMA_VERSION:
        raise LabArenaRewardError("unsupported reward basis schema")
    for name in ("round_id", "published_at"):
        if not isinstance(document[name], str) or not document[name]:
            raise LabArenaRewardError("%s must be a non-empty string" % name)
    for name in ("configuration_hash", "commitment_hash", "result_bundle_hash"):
        _require_hash(document[name], name)
    _basis_fields(document)
    body = {key: document[key] for key in _BASIS_BODY_FIELDS}
    if "reward_basis_hash" in document and document["reward_basis_hash"] != sha256_json(body):
        raise LabArenaRewardError("reward_basis_hash does not match the reward basis body")
    if "signature" in document:
        signature = document["signature"]
        if (
            not isinstance(signature, Mapping)
            or set(signature) != {"algorithm", "public_key_hash", "signature_b64"}
            or signature["algorithm"] != SIGNING_ALGORITHM
        ):
            raise LabArenaRewardError("reward basis signature block is invalid")
        _require_hash(signature["public_key_hash"], "signature.public_key_hash")
    return dict(document)


# ---------------------------------------------------------------------------
# Signature (the Arena key, pinned by hash on the validator and the coordinator)
# ---------------------------------------------------------------------------


def public_key_hash(public_key_der: bytes) -> str:
    return _HASH_PREFIX + hashlib.sha256(bytes(public_key_der)).hexdigest()


def signing_key_from_document(document: Any, expected_public_key_hash: str) -> bytes:
    """The DER public key of an Arena signing-key document that hashes to the pinned value."""

    _require_hash(expected_public_key_hash, "expected_public_key_hash")
    if (
        not isinstance(document, Mapping)
        or document.get("schema_version") != SIGNING_KEY_DOCUMENT_SCHEMA_VERSION
        or document.get("algorithm") != SIGNING_ALGORITHM
        or document.get("key_spec") != SIGNING_KEY_SPEC
    ):
        raise LabArenaRewardError("Arena signing key document is invalid")
    try:
        der = base64.b64decode(str(document.get("public_key_der_b64") or ""), validate=True)
    except (ValueError, TypeError) as exc:
        raise LabArenaRewardError("Arena signing key is not base64") from exc
    if not der or public_key_hash(der) != expected_public_key_hash or document.get("public_key_hash") != expected_public_key_hash:
        raise LabArenaRewardError("Arena signing key does not match the pinned key hash")
    return der


def verify_reward_basis_signature(basis: Any, *, public_key_der: bytes, expected_public_key_hash: str) -> str:
    """Verify the Arena signature on a reward basis; returns ``reward_basis_hash``.

    The signed bytes are ``reward_basis_hash:<hash>`` (the field name binds the
    document type). Raises ``LabArenaRewardError`` on any mismatch: shape,
    hash, key hash, key bytes, or signature.
    """

    document = validate_reward_basis(basis)
    if "reward_basis_hash" not in document or "signature" not in document:
        raise LabArenaRewardError("reward basis is unsigned")
    _require_hash(expected_public_key_hash, "expected_public_key_hash")
    if public_key_hash(public_key_der) != expected_public_key_hash:
        raise LabArenaRewardError("public key does not match the pinned key hash")
    signature = document["signature"]
    if signature["public_key_hash"] != expected_public_key_hash:
        raise LabArenaRewardError("reward basis was signed by a different key")
    try:
        raw_signature = base64.b64decode(str(signature["signature_b64"] or ""), validate=True)
    except (ValueError, TypeError) as exc:
        raise LabArenaRewardError("reward basis signature is not base64") from exc
    digest = str(document["reward_basis_hash"])
    message = ("reward_basis_hash:%s" % digest).encode("utf-8")
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import ec

        public_key = serialization.load_der_public_key(bytes(public_key_der))
        public_key.verify(raw_signature, message, ec.ECDSA(hashes.SHA256()))
    except Exception as exc:  # InvalidSignature, ValueError, TypeError, or an unsupported key
        raise LabArenaRewardError("Arena reward basis signature invalid") from exc
    return digest


def signing_key_hash_from_environment(environment: Mapping[str, str]) -> str:
    """The pinned Arena signing-key hash, required whenever rewards are enabled."""

    value = str(environment.get(SIGNING_KEY_HASH_ENV) or "").strip().lower()
    return _require_hash(value, SIGNING_KEY_HASH_ENV)


def rewards_enabled_from_environment(environment: Mapping[str, str]) -> bool:
    value = str(environment.get(REWARDS_ENABLED_ENV) or "").strip().lower()
    return value in ("1", "true", "yes", "on")


# ---------------------------------------------------------------------------
# Weekly decay (13.2), governing row and eligibility (13.3)
# ---------------------------------------------------------------------------


def reward_week_index(epoch_id: int, king_start_epoch: int, constants: Mapping[str, Any]) -> int:
    """``min(floor((epoch_id - king_start_epoch) / epochs_per_reward_week), last week)``."""

    epoch_id = _require_epoch(epoch_id, "epoch_id")
    king_start_epoch = _require_epoch(king_start_epoch, "king_start_epoch")
    fields = validate_reward_constants(constants)
    if epoch_id < king_start_epoch:
        raise LabArenaRewardError("epoch_id precedes king_start_epoch")
    last = len(fields["king_pool_share_percent_by_week"]) - 1
    return min((epoch_id - king_start_epoch) // fields["epochs_per_reward_week"], last)


def champion_share_for_week(week_index: int, constants: Mapping[str, Any]) -> float:
    """``pool_percent / 100 * week_share / 100`` of total emissions, exactly.

    The pool basis is total emissions: the king's share never shrinks when the
    Research Lab or leaderboard allocations grow. The product is evaluated
    with ``fractions.Fraction`` and converted once, so the weekly values are
    exactly the floats every side compares (0.25, 0.2, 0.15, 0.1, 0.05 at 25%).
    """

    fields = validate_reward_constants(constants)
    shares = fields["king_pool_share_percent_by_week"]
    week_index = _require_int(week_index, "week_index", 0, len(shares) - 1)
    return float(Fraction(fields["pool_percent"], 100) * Fraction(shares[week_index], 100))


def governing_reward_basis(rows: Sequence[Any], epoch_id: int) -> Optional[Dict[str, Any]]:
    """The published basis with the greatest ``effective_reward_epoch <= epoch_id``.

    Every row is shape-checked. Two rows sharing one effective epoch violate
    the write-once publication rule and raise, whether or not either governs.
    """

    epoch_id = _require_epoch(epoch_id, "epoch_id")
    seen = set()  # type: set
    governing = None  # type: Optional[Dict[str, Any]]
    for row in rows:
        basis = validate_reward_basis(row)
        effective = int(basis["effective_reward_epoch"])
        if effective in seen:
            raise LabArenaRewardError("duplicate effective_reward_epoch %d among reward bases" % effective)
        seen.add(effective)
        if effective > epoch_id:
            continue
        if governing is None or effective > int(governing["effective_reward_epoch"]):
            governing = basis
    return governing


def epoch_eligible(basis: Any, epoch_id: int) -> bool:
    """Eligible when the row is at most ``eligibility_max_epochs`` old and pays.

    Raises when the basis is not yet effective for ``epoch_id``: only the
    governing row may be passed here. An outcome outside the closed vocabulary
    raises as well, so an unknown outcome can never pay.
    """

    fields = _basis_fields(basis)
    epoch_id = _require_epoch(epoch_id, "epoch_id")
    effective = fields["effective_reward_epoch"]
    if epoch_id < effective:
        raise LabArenaRewardError("reward basis is not effective at epoch %d" % epoch_id)
    if epoch_id - effective > fields["reward_constants"]["eligibility_max_epochs"]:
        return False
    return fields["king_outcome"] in PAYING_KING_OUTCOMES


# ---------------------------------------------------------------------------
# Hotkey binding and the champion triple (13.1)
# ---------------------------------------------------------------------------


def _require_hotkeys(metagraph_hotkeys: Any) -> List[str]:
    if isinstance(metagraph_hotkeys, (str, bytes)) or not isinstance(metagraph_hotkeys, Sequence):
        raise LabArenaRewardError("metagraph_hotkeys must be a sequence of strings")
    out = []  # type: List[str]
    for item in metagraph_hotkeys:
        if not isinstance(item, str):
            raise LabArenaRewardError("metagraph_hotkeys must be a sequence of strings")
        out.append(item)
    return out


def champion_uid_for_hotkey(metagraph_hotkeys: Sequence[str], king_hotkey: str) -> Optional[int]:
    """UID whose metagraph hotkey equals the king, or ``None`` when unregistered."""

    hotkeys = _require_hotkeys(metagraph_hotkeys)
    if not isinstance(king_hotkey, str) or not king_hotkey:
        return None
    matches = [uid for uid, hotkey in enumerate(hotkeys) if hotkey == king_hotkey]
    if len(matches) > 1:
        raise LabArenaRewardError("king hotkey is registered at more than one UID")
    if not matches:
        return None
    return matches[0]


def champion_uid_matches(metagraph_hotkeys: Sequence[str], champion_uid: Any, king_hotkey: str) -> bool:
    """True only when ``metagraph_hotkeys[champion_uid]`` is the king hotkey."""

    hotkeys = _require_hotkeys(metagraph_hotkeys)
    if isinstance(champion_uid, bool) or not isinstance(champion_uid, int):
        return False
    if champion_uid < 0 or champion_uid >= len(hotkeys):
        return False
    return bool(king_hotkey) and hotkeys[champion_uid] == king_hotkey


def champion_values(basis: Any, epoch_id: int, metagraph_hotkeys: Sequence[str]) -> Dict[str, Any]:
    """The champion triple for one weight epoch from the governing basis.

    ``champion_share`` is the week's share of total emissions.
    ``champion_share`` and ``effective_champion_share`` are always equal
    (13.1: the Arena never burns a gap). Both are ``0.0`` and ``champion_uid``
    is ``None`` whenever the epoch is ineligible or the king hotkey is not
    registered on the supplied finalized metagraph. ``reward_week_index`` is
    reported whenever a king exists so the decay clock is visible even on an
    ineligible epoch; it is ``None`` for ``no_king``.
    """

    fields = _basis_fields(basis)
    epoch_id = _require_epoch(epoch_id, "epoch_id")
    hotkeys = _require_hotkeys(metagraph_hotkeys)
    eligible = epoch_eligible(fields, epoch_id)
    week_index = None  # type: Optional[int]
    if fields["king_outcome"] != "no_king":
        week_index = reward_week_index(epoch_id, fields["king_start_epoch"], fields["reward_constants"])
    uid = None  # type: Optional[int]
    share = 0.0
    if eligible:
        uid = champion_uid_for_hotkey(hotkeys, fields["king_hotkey"])
        if uid is not None:
            if week_index is None:
                raise LabArenaRewardError("eligible basis without a king start epoch")
            share = champion_share_for_week(week_index, fields["reward_constants"])
    return {
        "champion_share": share,
        "effective_champion_share": share,
        "champion_uid": uid,
        "reward_week_index": week_index,
        "eligible": eligible,
    }


def check_snapshot_champion_triple(snapshot: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """Require a snapshot's champion triple to be exactly what its basis implies.

    A snapshot without ``lab_arena_reward_basis`` is untouched (the legacy
    slot keeps its meaning). One that carries a basis must carry the triple
    :func:`champion_values` derives from it for the snapshot's epoch and
    metagraph, else the snapshot is rejected on every side of the weight path.
    Returns the derived values, or ``None`` when no basis is present.
    """

    if "lab_arena_reward_basis" not in snapshot:
        return None
    basis = validate_reward_basis(snapshot["lab_arena_reward_basis"])
    values = champion_values(basis, int(snapshot["epoch_id"]), list(snapshot["metagraph_hotkeys"] or []))
    proposed = (snapshot.get("champion_share"), snapshot.get("effective_champion_share"), snapshot.get("champion_uid"))
    derived = (values["champion_share"], values["effective_champion_share"], values["champion_uid"])
    if canonical_json(proposed) != canonical_json(derived):
        raise LabArenaRewardError("champion triple differs from the reward basis it names")
    return values
