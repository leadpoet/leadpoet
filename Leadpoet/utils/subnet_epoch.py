"""Authoritative Bittensor subnet epoch identity and cutover helpers.

An epoch is identified by ``SubnetEpochIndex`` and anchored by
``LastEpochBlock``. All storage fields in this module are read at one exact
block hash so callers never combine state from different chain heads.

Existing LeadPoet reward schedules use a monotonic integer epoch ordinal. A
signed and attested mapping projects the official subnet index onto that
ordinal so historical rows are never reused when the counters differ. The
official identity remains available as ``epoch_ref`` and
``subnet_epoch_index``; the compatibility ordinal must not be presented as the
Bittensor epoch ID.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Optional


EPOCH_SCHEME = "bittensor.subnet_epoch_index.v1"
CUTOVER_SCHEMA_VERSION = "leadpoet.subnet_epoch_cutover.v1"
SNAPSHOT_SCHEMA_VERSION = "leadpoet.subnet_epoch_snapshot.v1"
CUTOVER_JSON_ENV = "LEADPOET_SUBNET_EPOCH_CUTOVER_JSON"
CUTOVER_PATH_ENV = "LEADPOET_SUBNET_EPOCH_CUTOVER_PATH"
OFFICIAL_BITTENSOR_ARCHIVE_ENDPOINT = (
    "wss://archive.chain.opentensor.ai:443"
)

_STORAGE_FIELDS = {
    "tempo": "Tempo",
    "last_epoch_block": "LastEpochBlock",
    "pending_epoch_at": "PendingEpochAt",
    "subnet_epoch_index": "SubnetEpochIndex",
    "blocks_since_last_step": "BlocksSinceLastStep",
}
SUBTENSOR_MAX_TEMPO = 50_400


class SubnetEpochError(RuntimeError):
    """The chain epoch state or cutover mapping is missing or inconsistent."""


def assert_official_archive_subtensor(subtensor: Any) -> None:
    """Require a client pinned to Bittensor's official archive endpoint.

    Stateful cutover proof reads immutable storage older than lite-node
    retention.  Accepting a normal Finney client here would work briefly and
    then fail permanently once that state is pruned, so the historical
    authority is explicit and fail closed.
    """

    endpoint = getattr(subtensor, "chain_endpoint", None)
    if endpoint is None:
        endpoint = getattr(getattr(subtensor, "substrate", None), "url", None)
    normalized = str(endpoint or "").strip().lower().rstrip("/")
    if normalized != OFFICIAL_BITTENSOR_ARCHIVE_ENDPOINT:
        raise SubnetEpochError(
            "historical epoch authority must use the official Bittensor archive"
        )


def load_subnet_epoch_cutover(
    environ: Optional[Mapping[str, str]] = None,
) -> "SubnetEpochCutover":
    """Load and validate the configured settlement mapping.

    Runtime epoch authority is intentionally unusable without this mapping.
    Supplying both environment forms is rejected so two processes cannot
    silently choose different authorities.
    """

    source = os.environ if environ is None else environ
    raw = str(source.get(CUTOVER_JSON_ENV, "") or "").strip()
    path = str(source.get(CUTOVER_PATH_ENV, "") or "").strip()
    if raw and path:
        raise SubnetEpochError(
            f"set only one of {CUTOVER_JSON_ENV} or {CUTOVER_PATH_ENV}"
        )
    if path:
        try:
            raw = Path(path).expanduser().read_text(encoding="utf-8")
        except OSError as exc:
            raise SubnetEpochError(
                "failed to read subnet epoch cutover manifest"
            ) from exc
    if not raw:
        raise SubnetEpochError("official epoch authority requires a cutover manifest")
    try:
        document = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SubnetEpochError(
            "subnet epoch cutover manifest is not valid JSON"
        ) from exc
    if not isinstance(document, dict):
        raise SubnetEpochError("subnet epoch cutover manifest must be an object")
    return SubnetEpochCutover.from_mapping(document)


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def _sha256_json(value: Mapping[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json(value)).hexdigest()


def _hash(value: Any, field: str) -> str:
    normalized = str(value or "").strip().lower()
    if normalized.startswith("0x"):
        normalized = normalized[2:]
    if len(normalized) != 64 or any(c not in "0123456789abcdef" for c in normalized):
        raise SubnetEpochError(f"{field} must be a 32-byte lowercase hex hash")
    return "0x" + normalized


def _integer(value: Any, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise SubnetEpochError(f"{field} must be an integer")
    try:
        normalized = int(value)
    except (TypeError, ValueError) as exc:
        raise SubnetEpochError(f"{field} must be an integer") from exc
    if normalized < minimum:
        raise SubnetEpochError(f"{field} must be >= {minimum}")
    return normalized


def _scale_value(value: Any, field: str) -> int:
    decoded = getattr(value, "value", value)
    return _integer(decoded, field)


@dataclass(frozen=True)
class SubnetEpochCutover:
    """Monotonic settlement-ordinal bridge for one official epoch lineage."""

    network_genesis_hash: str
    netuid: int
    cutover_block: int
    cutover_block_hash: str
    first_subnet_epoch_index: int
    first_settlement_epoch_id: int
    last_legacy_epoch_id: int
    schema_version: str = CUTOVER_SCHEMA_VERSION
    epoch_scheme: str = EPOCH_SCHEME
    mapping_hash: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "network_genesis_hash",
            _hash(self.network_genesis_hash, "network_genesis_hash"),
        )
        object.__setattr__(
            self,
            "cutover_block_hash",
            _hash(self.cutover_block_hash, "cutover_block_hash"),
        )
        for field in (
            "netuid",
            "cutover_block",
            "first_subnet_epoch_index",
            "first_settlement_epoch_id",
            "last_legacy_epoch_id",
        ):
            object.__setattr__(self, field, _integer(getattr(self, field), field))
        if self.netuid <= 0:
            raise SubnetEpochError("netuid must be positive")
        if self.schema_version != CUTOVER_SCHEMA_VERSION:
            raise SubnetEpochError("unsupported epoch cutover schema")
        if self.epoch_scheme != EPOCH_SCHEME:
            raise SubnetEpochError("unsupported epoch scheme")
        if self.first_settlement_epoch_id != self.last_legacy_epoch_id + 1:
            raise SubnetEpochError(
                "first settlement epoch must immediately follow the last legacy epoch"
            )
        expected = _sha256_json(self.body())
        if self.mapping_hash is None:
            object.__setattr__(self, "mapping_hash", expected)
        elif str(self.mapping_hash).lower() != expected:
            raise SubnetEpochError("epoch cutover mapping hash mismatch")

    def body(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "epoch_scheme": self.epoch_scheme,
            "network_genesis_hash": self.network_genesis_hash,
            "netuid": self.netuid,
            "cutover_block": self.cutover_block,
            "cutover_block_hash": self.cutover_block_hash,
            "first_subnet_epoch_index": self.first_subnet_epoch_index,
            "first_settlement_epoch_id": self.first_settlement_epoch_id,
            "last_legacy_epoch_id": self.last_legacy_epoch_id,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.body(), "mapping_hash": self.mapping_hash}

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SubnetEpochCutover":
        return cls(**dict(value))

    def settlement_epoch_id(self, subnet_epoch_index: int) -> int:
        index = _integer(subnet_epoch_index, "subnet_epoch_index")
        if index < self.first_subnet_epoch_index:
            raise SubnetEpochError("subnet epoch predates the configured cutover")
        return self.first_settlement_epoch_id + (
            index - self.first_subnet_epoch_index
        )


@dataclass(frozen=True)
class SubnetEpochSnapshot:
    """One exact-hash observation of the official Subtensor epoch scheduler."""

    network_genesis_hash: str
    netuid: int
    head_kind: str
    block_hash: str
    current_block: int
    last_epoch_block: int
    pending_epoch_at: int
    subnet_epoch_index: int
    tempo: int
    blocks_since_last_step: int
    observed_at: str
    schema_version: str = SNAPSHOT_SCHEMA_VERSION
    epoch_scheme: str = EPOCH_SCHEME

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "network_genesis_hash",
            _hash(self.network_genesis_hash, "network_genesis_hash"),
        )
        object.__setattr__(self, "block_hash", _hash(self.block_hash, "block_hash"))
        for field in (
            "netuid",
            "current_block",
            "last_epoch_block",
            "pending_epoch_at",
            "subnet_epoch_index",
            "tempo",
            "blocks_since_last_step",
        ):
            object.__setattr__(self, field, _integer(getattr(self, field), field))
        if self.netuid <= 0:
            raise SubnetEpochError("netuid must be positive")
        if self.tempo <= 0:
            raise SubnetEpochError("tempo must be positive")
        if self.last_epoch_block > self.current_block:
            raise SubnetEpochError("last epoch block cannot exceed the observed block")
        if self.head_kind not in {"best", "finalized", "exact"}:
            raise SubnetEpochError("head_kind must be best, finalized, or exact")
        if self.schema_version != SNAPSHOT_SCHEMA_VERSION:
            raise SubnetEpochError("unsupported epoch snapshot schema")
        if self.epoch_scheme != EPOCH_SCHEME:
            raise SubnetEpochError("unsupported epoch scheme")
        try:
            parsed = datetime.fromisoformat(self.observed_at.replace("Z", "+00:00"))
        except (TypeError, ValueError) as exc:
            raise SubnetEpochError("observed_at must be an ISO-8601 timestamp") from exc
        if parsed.tzinfo is None:
            raise SubnetEpochError("observed_at must include a timezone")

    @property
    def epoch_block(self) -> int:
        return self.current_block - self.last_epoch_block

    @property
    def next_epoch_block(self) -> int:
        # Subtensor's scheduler has a safety valve which makes an epoch due
        # once BlocksSinceLastStep exceeds MAX_TEMPO, even when the current
        # schedule anchor would place the automatic boundary later. The stored
        # counter is the post-block value: run_coinbase increments it before
        # testing the next block, so 50_400 means the safety valve fires one
        # block later and 50_399 means it fires two blocks later.
        if self.blocks_since_last_step > SUBTENSOR_MAX_TEMPO:
            return self.current_block
        automatic = self.last_epoch_block + self.tempo
        safety = self.current_block + (
            SUBTENSOR_MAX_TEMPO + 1 - self.blocks_since_last_step
        )
        if self.pending_epoch_at > 0:
            return min(automatic, self.pending_epoch_at, safety)
        return min(automatic, safety)

    @property
    def blocks_remaining(self) -> int:
        return max(0, self.next_epoch_block - self.current_block)

    @property
    def epoch_ref(self) -> str:
        return _sha256_json(
            {
                "epoch_scheme": self.epoch_scheme,
                "network_genesis_hash": self.network_genesis_hash,
                "netuid": self.netuid,
                "subnet_epoch_index": self.subnet_epoch_index,
            }
        )

    def settlement_epoch_id(self, cutover: SubnetEpochCutover) -> int:
        if self.network_genesis_hash != cutover.network_genesis_hash:
            raise SubnetEpochError("snapshot and cutover genesis hashes differ")
        if self.netuid != cutover.netuid:
            raise SubnetEpochError("snapshot and cutover netuids differ")
        return cutover.settlement_epoch_id(self.subnet_epoch_index)

    def to_dict(
        self, *, cutover: Optional[SubnetEpochCutover] = None
    ) -> dict[str, Any]:
        result = asdict(self)
        result.update(
            {
                "epoch_id": self.subnet_epoch_index,
                "epoch_ref": self.epoch_ref,
                "epoch_block": self.epoch_block,
                "next_epoch_block": self.next_epoch_block,
                "blocks_remaining": self.blocks_remaining,
            }
        )
        if cutover is not None:
            result["settlement_epoch_id"] = self.settlement_epoch_id(cutover)
            result["cutover_mapping_hash"] = cutover.mapping_hash
        return result

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SubnetEpochSnapshot":
        allowed = {
            "schema_version",
            "epoch_scheme",
            "network_genesis_hash",
            "netuid",
            "head_kind",
            "block_hash",
            "current_block",
            "last_epoch_block",
            "pending_epoch_at",
            "subnet_epoch_index",
            "tempo",
            "blocks_since_last_step",
            "observed_at",
        }
        return cls(**{key: value[key] for key in allowed if key in value})


def read_subnet_epoch_snapshot(
    subtensor: Any,
    *,
    netuid: int,
    finalized: bool = False,
    block_number: Optional[int] = None,
    block_hash: Optional[str] = None,
    observed_at: Optional[str] = None,
) -> SubnetEpochSnapshot:
    """Read scheduler fields at one exact best, finalized, or requested hash.

    Raw substrate storage queries are used instead of SDK convenience methods
    so this works with both Bittensor 9.x and 10.x and all fields are pinned to
    the same hash.
    """

    normalized_netuid = _integer(netuid, "netuid", minimum=1)
    if block_number is not None and block_hash is not None:
        raise SubnetEpochError("set only one of block_number or block_hash")
    if finalized and (block_number is not None or block_hash is not None):
        raise SubnetEpochError(
            "finalized cannot be combined with an exact block request"
        )
    substrate = getattr(subtensor, "substrate", None)
    if substrate is None:
        raise SubnetEpochError("subtensor substrate client is unavailable")
    try:
        expected_block = (
            None
            if block_number is None
            else _integer(block_number, "block_number")
        )
        if expected_block is not None:
            reference_hash = substrate.get_block_hash(expected_block)
            head_kind = "exact"
        elif block_hash is not None:
            reference_hash = _hash(block_hash, "block_hash")
            head_kind = "exact"
        elif finalized:
            reference_hash = substrate.get_chain_finalised_head()
            head_kind = "finalized"
        else:
            reference_hash = substrate.get_chain_head()
            head_kind = "best"
        reference_hash = _hash(reference_hash, "block_hash")
        current_block = substrate.get_block_number(reference_hash)
        genesis_hash = substrate.get_block_hash(0)
    except Exception as exc:
        raise SubnetEpochError("failed to resolve the epoch reference block") from exc
    if current_block is None:
        raise SubnetEpochError("reference block number is unavailable")
    current_block = _integer(current_block, "current_block")
    if expected_block is not None and current_block != expected_block:
        raise SubnetEpochError("resolved block hash and block number differ")

    decoded: dict[str, int] = {}
    for field, storage_name in _STORAGE_FIELDS.items():
        try:
            value = substrate.query(
                module="SubtensorModule",
                storage_function=storage_name,
                params=[normalized_netuid],
                block_hash=reference_hash,
            )
        except Exception as exc:
            raise SubnetEpochError(
                f"failed to read {storage_name} at the reference block"
            ) from exc
        decoded[field] = _scale_value(value, storage_name)

    if observed_at is None:
        try:
            chain_moment = substrate.query(
                module="Timestamp",
                storage_function="Now",
                params=[],
                block_hash=reference_hash,
            )
            chain_millis = _scale_value(chain_moment, "Timestamp.Now")
            observed_at = datetime.fromtimestamp(
                chain_millis / 1000, tz=timezone.utc
            ).isoformat().replace("+00:00", "Z")
        except Exception as exc:
            raise SubnetEpochError(
                "failed to read Timestamp.Now at the reference block"
            ) from exc

    return SubnetEpochSnapshot(
        network_genesis_hash=genesis_hash,
        netuid=normalized_netuid,
        head_kind=head_kind,
        block_hash=reference_hash,
        current_block=current_block,
        observed_at=observed_at,
        **decoded,
    )


def validate_subnet_epoch_cutover_anchor(
    subtensor: Any,
    cutover: SubnetEpochCutover,
) -> None:
    """Prove a cutover exclusively against Bittensor's official archive."""

    assert_official_archive_subtensor(subtensor)
    substrate = getattr(subtensor, "substrate", None)
    if substrate is None:
        raise SubnetEpochError("subtensor substrate client is unavailable")
    try:
        genesis_hash = _hash(
            substrate.get_block_hash(0), "network_genesis_hash"
        )
        cutover_hash = _hash(
            substrate.get_block_hash(cutover.cutover_block),
            "cutover_block_hash",
        )
    except Exception as exc:
        raise SubnetEpochError("failed to resolve the cutover chain anchor") from exc
    if genesis_hash != cutover.network_genesis_hash:
        raise SubnetEpochError("cutover manifest targets a different chain")
    if cutover_hash != cutover.cutover_block_hash:
        raise SubnetEpochError("cutover block hash differs from the live chain")
    if cutover.cutover_block <= 0:
        raise SubnetEpochError("cutover block must have a predecessor")
    boundary = read_subnet_epoch_snapshot(
        subtensor,
        netuid=cutover.netuid,
        block_hash=cutover.cutover_block_hash,
    )
    predecessor = read_subnet_epoch_snapshot(
        subtensor,
        netuid=cutover.netuid,
        block_number=cutover.cutover_block - 1,
    )
    if boundary.current_block != cutover.cutover_block:
        raise SubnetEpochError("cutover hash does not resolve to the cutover block")
    if boundary.subnet_epoch_index != cutover.first_subnet_epoch_index:
        raise SubnetEpochError("cutover official epoch index differs from chain state")
    if boundary.last_epoch_block != cutover.cutover_block:
        raise SubnetEpochError("cutover block is not the scheduler epoch anchor")
    if predecessor.subnet_epoch_index + 1 != boundary.subnet_epoch_index:
        raise SubnetEpochError("cutover block is not an official epoch transition")
