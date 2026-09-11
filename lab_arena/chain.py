"""Arena chain access: one explicit endpoint, fail-closed finalized reads, a
finalized metagraph cache, settlement
epoch ordinals, and the hotkey signature seam.

labarena.md sections 3.2 (trust model and allowlist), 7.2 (funding reads),
13.2 (epoch ordinals), and 15.1 (module row); labarenaaudit.md blocker 4.

What this module deliberately does not do:

- It never chooses a network from ``BITTENSOR_NETWORK``. ``ArenaChainConfig``
  carries one explicit ``wss://`` or ``ws://`` endpoint and
  ``connect_substrate`` opens exactly that endpoint. The Lab helper
  ``gateway.qualification.utils.chain`` keeps a module-global connection that
  is selected by environment and swallows read errors to ``None``; only its
  two pure functions, ``get_transfer_details`` and
  ``verify_sr25519_signature``, are imported here.
- It never returns ``None`` for a failed read and never synthesizes
  ``"Unknown"`` extrinsics. Every failure raises ``ArenaChainError`` or one of
  its subclasses so callers fail closed.
- It never reads a Lab table. The banned set is an explicit input that the
  service obtains from an operator-configured public read; the Arena holds no
  Lab credential and never queries ``banned_hotkeys`` itself.
- Network operations are bounded by the timeout configured on the substrate
  client at construction; this module adds no unbounded waits.

The substrate client is not thread-safe. ``ArenaChain`` serializes every call
on it, including the calls the epoch helper and the metagraph source make
through the client handle they are given.
"""

from __future__ import annotations

import os
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple
from urllib.parse import urlsplit

from gateway.qualification.utils.chain import verify_sr25519_signature
from Leadpoet.utils.subnet_epoch import (
    CUTOVER_JSON_ENV,
    CUTOVER_PATH_ENV,
    DEFAULT_SN71_CUTOVER_MANIFEST_PATH,
    SubnetEpochCutover,
    SubnetEpochError,
    SubnetEpochSnapshot,
    load_subnet_epoch_cutover,
    read_subnet_epoch_snapshot,
)
from lab_arena.contracts import (
    SS58_RE,
    ArenaContractError,
    require_hotkey,
    require_keys,
    require_only_keys,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SS58_FORMAT = 42
TYPE_REGISTRY_PRESET = "substrate-node-template"
# One retry per RPC: a call waits at most ``request_timeout_seconds`` per
# attempt, so the worst case per RPC is ``2 * request_timeout_seconds``.
SUBSTRATE_MAX_RETRIES = 2
DEFAULT_METAGRAPH_TTL_SECONDS = 60
MAX_REQUEST_TIMEOUT_SECONDS = 600

NETWORK_NAME_RE = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")
_HEX64_RE = re.compile(r"^[0-9a-fA-F]{64}$")

FAILED_EVENT = "ExtrinsicFailed"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ArenaChainError(RuntimeError):
    """A chain read failed or returned something that cannot be trusted."""


class ArenaChainConfigError(ArenaChainError, ValueError):
    """The chain configuration is malformed."""


class InvalidBlockHash(ArenaChainError, ValueError):
    """A block hash is not 32 bytes of hex."""


class ArenaBlockNotFound(ArenaChainError):
    """The endpoint does not know the requested block."""


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------


def normalize_block_hash(value: Any) -> str:
    """Return the ``0x``-prefixed lowercase 64-hex form or raise ``InvalidBlockHash``."""

    if isinstance(value, (bytes, bytearray)):
        if len(value) != 32:
            raise InvalidBlockHash("block hash must be 32 bytes")
        return "0x" + bytes(value).hex()
    if not isinstance(value, str):
        raise InvalidBlockHash("block hash must be a string")
    text = value.strip()
    if text[:2] in ("0x", "0X"):
        text = text[2:]
    if not _HEX64_RE.match(text):
        raise InvalidBlockHash("block hash must be 32 bytes of hex")
    return "0x" + text.lower()


def normalize_network_name(value: Any) -> str:
    if not isinstance(value, str):
        raise ArenaChainConfigError("network name must be a string")
    text = value.strip()
    if not NETWORK_NAME_RE.match(text):
        raise ArenaChainConfigError("network name must match %s" % NETWORK_NAME_RE.pattern)
    return text


def _require_int(value: Any, field_name: str, *, minimum: Optional[int] = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ArenaChainError("%s must be an integer" % field_name)
    if minimum is not None and value < minimum:
        raise ArenaChainError("%s must be >= %d" % (field_name, minimum))
    return value


def _unwrap(value: Any) -> Any:
    """Unwrap ``ScaleObj`` / ``GenericExtrinsic`` style objects to plain values."""

    if isinstance(value, (Mapping, list, tuple, str, bytes, int, float)) or value is None:
        return value
    return getattr(value, "value", value)


def _chain_int(value: Any, field_name: str) -> int:
    """Decode an integer the chain may return as int, ``ScaleObj``, or hex string."""

    decoded = _unwrap(value)
    if isinstance(decoded, bool):
        raise ArenaChainError("%s must be an integer" % field_name)
    if isinstance(decoded, int):
        return decoded
    if isinstance(decoded, str):
        text = decoded.strip()
        try:
            if text[:2] in ("0x", "0X"):
                return int(text, 16)
            return int(text, 10)
        except ValueError as exc:
            raise ArenaChainError("%s is not an integer" % field_name) from exc
    raise ArenaChainError("%s is not an integer" % field_name)


# ---------------------------------------------------------------------------
# Configuration and client protocol
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArenaChainConfig:
    """One explicit chain endpoint. Nothing here is read from the environment."""

    endpoint: str
    netuid: int
    network_name: str
    request_timeout_seconds: int
    metagraph_ttl_seconds: int = DEFAULT_METAGRAPH_TTL_SECONDS

    def __post_init__(self) -> None:
        if not isinstance(self.endpoint, str):
            raise ArenaChainConfigError("endpoint must be a string")
        endpoint = self.endpoint.strip()
        parts = urlsplit(endpoint)
        if parts.scheme not in ("wss", "ws") or not parts.hostname:
            raise ArenaChainConfigError("endpoint must be an explicit ws:// or wss:// URL with a host")
        if parts.username or parts.password:
            raise ArenaChainConfigError("endpoint must not embed credentials")
        object.__setattr__(self, "endpoint", endpoint)
        if isinstance(self.netuid, bool) or not isinstance(self.netuid, int) or self.netuid <= 0:
            raise ArenaChainConfigError("netuid must be a positive integer")
        object.__setattr__(self, "network_name", normalize_network_name(self.network_name))
        timeout = self.request_timeout_seconds
        if isinstance(timeout, bool) or not isinstance(timeout, int) or not 1 <= timeout <= MAX_REQUEST_TIMEOUT_SECONDS:
            raise ArenaChainConfigError("request_timeout_seconds must be an integer between 1 and %d" % MAX_REQUEST_TIMEOUT_SECONDS)
        ttl = self.metagraph_ttl_seconds
        if isinstance(ttl, bool) or not isinstance(ttl, int) or ttl < 0:
            raise ArenaChainConfigError("metagraph_ttl_seconds must be a non-negative integer")


class SubstrateClient(Protocol):
    """The narrow surface ``ArenaChain`` uses.

    ``async_substrate_interface.SubstrateInterface`` (the synchronous facade
    Bittensor 10 ships) fulfils it with exactly these method names; tests use
    a fake. ``get_block`` returns the flat block dict (``header`` and
    ``extrinsics``) or ``None`` for an unknown hash; ``get_events`` returns the
    decoded event list; ``query`` returns a ``ScaleObj`` or plain value.
    """

    def get_chain_finalised_head(self) -> Any: ...

    def get_block_hash(self, block_id: int) -> Any: ...

    def get_block_number(self, block_hash: Any = None) -> Any: ...

    def get_block(self, block_hash: Any = None, **kwargs: Any) -> Any: ...

    def get_events(self, block_hash: Any = None) -> Any: ...

    def query(self, module: str, storage_function: str, params: Any = None, block_hash: Any = None) -> Any: ...

    def runtime_call(self, runtime_api: str, method: str, params: Any = None, block_hash: Any = None) -> Any: ...


def connect_substrate(config: ArenaChainConfig) -> Any:
    """Open one ``SubstrateInterface`` on ``config.endpoint``.

    The library is imported lazily so importing this module never loads it.
    The endpoint comes only from ``config``; ``BITTENSOR_NETWORK`` is never
    consulted. ``retry_timeout`` bounds every RPC wait and
    ``SUBSTRATE_MAX_RETRIES`` bounds retries, so a single call waits at most
    ``2 * request_timeout_seconds``. Connection failure raises.
    """

    if not isinstance(config, ArenaChainConfig):
        raise ArenaChainConfigError("config must be an ArenaChainConfig")
    try:
        from async_substrate_interface import SubstrateInterface
    except ImportError as exc:
        raise ArenaChainError("async_substrate_interface is not installed") from exc
    try:
        return SubstrateInterface(
            url=config.endpoint,
            ss58_format=SS58_FORMAT,
            type_registry_preset=TYPE_REGISTRY_PRESET,
            max_retries=SUBSTRATE_MAX_RETRIES,
            retry_timeout=float(config.request_timeout_seconds),
        )
    except ArenaChainError:
        raise
    except Exception as exc:
        raise ArenaChainError("failed to open the Arena chain endpoint") from exc


# ---------------------------------------------------------------------------
# Finalized metagraph snapshot
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BlockRef:
    number: int
    hash: str


@dataclass(frozen=True)
class MetagraphSnapshot:
    """The finalized metagraph fields the Arena needs, pinned to one block."""

    netuid: int
    block_number: int
    block_hash: str
    hotkeys: Tuple[str, ...]
    coldkeys: Tuple[str, ...]
    validator_permit: Tuple[bool, ...]
    stake: Optional[Tuple[float, ...]] = None
    active: Optional[Tuple[bool, ...]] = None

    def __post_init__(self) -> None:
        _require_int(self.netuid, "netuid", minimum=1)
        _require_int(self.block_number, "block_number", minimum=0)
        object.__setattr__(self, "block_hash", normalize_block_hash(self.block_hash))
        hotkeys = tuple(require_hotkey(item, "hotkey") for item in self.hotkeys)
        coldkeys = tuple(require_hotkey(item, "coldkey") for item in self.coldkeys)
        if len(hotkeys) != len(coldkeys):
            raise ArenaChainError("hotkeys and coldkeys differ in length")
        if len(set(hotkeys)) != len(hotkeys):
            raise ArenaChainError("metagraph hotkeys are not unique")
        permits = tuple(self.validator_permit)
        if len(permits) != len(hotkeys) or any(not isinstance(item, bool) for item in permits):
            raise ArenaChainError("validator_permit must hold one boolean per uid")
        stake: Optional[Tuple[float, ...]] = None
        if self.stake is not None:
            stake = tuple(float(item) for item in self.stake)
            if len(stake) != len(hotkeys) or any(item != item or item in (float("inf"), float("-inf")) for item in stake):
                raise ArenaChainError("stake must hold one finite number per uid")
        active: Optional[Tuple[bool, ...]] = None
        if self.active is not None:
            active = tuple(self.active)
            if len(active) != len(hotkeys) or any(not isinstance(item, bool) for item in active):
                raise ArenaChainError("active must hold one boolean per uid")
        object.__setattr__(self, "hotkeys", hotkeys)
        object.__setattr__(self, "coldkeys", coldkeys)
        object.__setattr__(self, "active", active)
        object.__setattr__(self, "validator_permit", permits)
        object.__setattr__(self, "stake", stake)

    @property
    def size(self) -> int:
        return len(self.hotkeys)


def uid_for_hotkey(snapshot: MetagraphSnapshot, hotkey: str) -> Optional[int]:
    for uid, candidate in enumerate(snapshot.hotkeys):
        if candidate == hotkey:
            return uid
    return None


def is_registered(snapshot: MetagraphSnapshot, hotkey: str) -> bool:
    return uid_for_hotkey(snapshot, hotkey) is not None


def coldkey_owns_hotkey(snapshot: MetagraphSnapshot, coldkey: str, hotkey: str) -> bool:
    """True only when ``hotkey`` is registered and its owner coldkey is ``coldkey``."""

    uid = uid_for_hotkey(snapshot, hotkey)
    if uid is None or not isinstance(coldkey, str) or not coldkey:
        return False
    return snapshot.coldkeys[uid] == coldkey


def hotkeys_owned_by_coldkey(snapshot: MetagraphSnapshot, coldkey: str) -> List[str]:
    """Hotkeys whose owner coldkey is ``coldkey``, in uid order."""

    return [hotkey for hotkey, owner in zip(snapshot.hotkeys, snapshot.coldkeys) if owner == coldkey]


def _scalar(value: Any) -> Any:
    item = getattr(value, "item", None)
    return item() if callable(item) else value


def metagraph_snapshot_from_object(
    metagraph: Any,
    *,
    netuid: int,
    block_number: int,
    block_hash: str,
) -> MetagraphSnapshot:
    """Map a bittensor ``Metagraph`` (or any object with the same fields).

    Requires ``hotkeys``, ``coldkeys``, ``active``, ``validator_permit`` and
    ``S`` (or ``stake``). When the object reports a ``block`` it must
    equal ``block_number``: a metagraph that is not pinned to the finalized
    block is refused rather than trusted.
    """

    try:
        hotkeys = [str(item) for item in metagraph.hotkeys]
        coldkeys = [str(item) for item in metagraph.coldkeys]
        active = [bool(_scalar(item)) for item in metagraph.active]
        permits = [bool(_scalar(item)) for item in metagraph.validator_permit]
    except (AttributeError, TypeError, ValueError) as exc:
        raise ArenaChainError("metagraph object lacks the pinned fields") from exc
    stake_source = getattr(metagraph, "S", None)
    if stake_source is None:
        stake_source = getattr(metagraph, "stake", None)
    if stake_source is None:
        raise ArenaChainError("metagraph object lacks the pinned fields")
    try:
        stake = [float(_scalar(item)) for item in stake_source]
    except (TypeError, ValueError) as exc:
        raise ArenaChainError("metagraph stake is not numeric") from exc
    reported_block = getattr(metagraph, "block", None)
    if reported_block is not None:
        observed = _chain_int(_scalar(reported_block), "metagraph block")
        if observed != block_number:
            raise ArenaChainError("metagraph block differs from the finalized block")
    return MetagraphSnapshot(
        netuid=netuid,
        block_number=block_number,
        block_hash=block_hash,
        hotkeys=tuple(hotkeys),
        coldkeys=tuple(coldkeys),
        active=tuple(active),
        validator_permit=tuple(permits),
        stake=tuple(stake),
    )


MetagraphSource = Callable[[Any, int, str], MetagraphSnapshot]


def bittensor_metagraph_source(config: ArenaChainConfig) -> MetagraphSource:
    """Read the SDK's metagraph role fields at the finalized hash.

    Bittensor 10.5 populates ``metagraph.active`` and overwrites ``metagraph.S``
    from ``SubnetInfoRuntimeApi.get_subnet_state``. Read that same response on
    Arena's existing serialized client, with the same ``Balance.from_rao``
    conversion, rather than creating another Subtensor connection. Bittensor
    9.12 passes a scalar ``u16`` to some metadata that describes the parameter
    as a one-field composite, so retry only that known encoding mismatch.
    """

    if not isinstance(config, ArenaChainConfig):
        raise ArenaChainConfigError("config must be an ArenaChainConfig")

    def source(client: Any, netuid: int, block_hash: str) -> MetagraphSnapshot:
        normalized = normalize_block_hash(block_hash)
        try:
            block_number = _chain_int(client.get_block_number(normalized), "block number")
        except ArenaChainError:
            raise
        except Exception as exc:
            raise ArenaChainError("chain call get_block_number failed") from exc
        try:
            reader = getattr(client, "runtime_call")
            try:
                response = reader("SubnetInfoRuntimeApi", "get_subnet_state", [netuid], normalized)
            except ValueError as exc:
                message = str(exc)
                if "type_def: Composite" not in message or 'type_name: Some("u16")' not in message:
                    raise
                response = reader("SubnetInfoRuntimeApi", "get_subnet_state", [[netuid]], normalized)
        except Exception as exc:
            raise ArenaChainError("finalized metagraph runtime read failed") from exc
        state = _unwrap(response)
        if not isinstance(state, Mapping):
            raise ArenaChainError("finalized metagraph runtime result is invalid")
        if _chain_int(_runtime_scalar(state.get("netuid")), "subnet state netuid") != netuid:
            raise ArenaChainError("finalized subnet state netuid differs")
        try:
            hotkeys = tuple(
                _runtime_account(item, "neuron hotkey")
                for item in state["hotkeys"]
            )
            coldkeys = tuple(
                _runtime_account(item, "neuron coldkey")
                for item in state["coldkeys"]
            )
            active = tuple(
                _runtime_bool(item, "neuron active") for item in state["active"]
            )
            permits = tuple(
                _runtime_bool(item, "validator permit")
                for item in state["validator_permit"]
            )
            total_stake = tuple(
                _runtime_stake_tao(item) for item in state["total_stake"]
            )
        except (KeyError, TypeError) as exc:
            raise ArenaChainError("finalized subnet state role fields are invalid") from exc
        return MetagraphSnapshot(
            netuid=netuid, block_number=block_number, block_hash=normalized,
            hotkeys=hotkeys,
            coldkeys=coldkeys,
            active=active,
            validator_permit=permits,
            stake=total_stake,
        )

    return source


def _runtime_scalar(value: Any) -> Any:
    value = _unwrap(value)
    while isinstance(value, (tuple, list)) and len(value) == 1:
        value = _unwrap(value[0])
    return value


def _runtime_account(value: Any, field_name: str) -> str:
    value = _runtime_scalar(value)
    if isinstance(value, str) and SS58_RE.fullmatch(value):
        return value
    if isinstance(value, (tuple, list)) and len(value) == 32:
        try:
            from leadpoet_canonical.chain_source_v2 import ss58_encode_account_id
            return ss58_encode_account_id(bytes(value))
        except (TypeError, ValueError):
            pass
    raise ArenaChainError("%s is not a chain account" % field_name)


def _runtime_bool(value: Any, field_name: str) -> bool:
    value = _runtime_scalar(value)
    if not isinstance(value, bool):
        raise ArenaChainError("%s is not boolean" % field_name)
    return value


def _runtime_stake_tao(value: Any) -> float:
    """Decode SubnetState.total_stake exactly as Bittensor Metagraph.S does."""

    rao = _chain_int(_runtime_scalar(value), "neuron total stake")
    if rao < 0:
        raise ArenaChainError("neuron total stake must be non-negative")
    try:
        from bittensor.utils.balance import Balance

        return float(Balance.from_rao(rao).tao)
    except (ImportError, TypeError, ValueError, AttributeError) as exc:
        raise ArenaChainError("neuron total stake is invalid") from exc


# ---------------------------------------------------------------------------
# Serialized client handle
# ---------------------------------------------------------------------------


class _SerializedClient:
    """Every method call on the wrapped client runs under one lock."""

    def __init__(self, client: Any, lock: threading.RLock) -> None:
        self._client = client
        self._lock = lock

    def __getattr__(self, name: str) -> Any:
        target = getattr(self._client, name)
        if not callable(target):
            return target

        def call(*args: Any, **kwargs: Any) -> Any:
            with self._lock:
                return target(*args, **kwargs)

        return call


@dataclass(frozen=True)
class _EpochSubtensor:
    """The minimal object ``read_subnet_epoch_snapshot`` accepts."""

    substrate: Any
    chain_endpoint: str


# ---------------------------------------------------------------------------
# ArenaChain
# ---------------------------------------------------------------------------


class ArenaChain:
    """Fail-closed reads on one explicit endpoint plus a finalized metagraph cache."""

    def __init__(
        self,
        config: ArenaChainConfig,
        client: Any,
        *,
        metagraph_source: Optional[MetagraphSource] = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not isinstance(config, ArenaChainConfig):
            raise ArenaChainConfigError("config must be an ArenaChainConfig")
        if client is None:
            raise ArenaChainConfigError("client is required")
        self._config = config
        self._client = client
        self._lock = threading.RLock()
        self._serialized = _SerializedClient(client, self._lock)
        self._metagraph_source: MetagraphSource = metagraph_source or bittensor_metagraph_source(config)
        self._clock = clock
        self._metagraph_lock = threading.Lock()
        self._cached_metagraph: Optional[Tuple[MetagraphSnapshot, float]] = None

    def close(self) -> None:
        """Release the substrate client's connection and threads; idempotent.

        A process that read the chain and never closed the client can hang at
        exit on its websocket threads, so the wiring registers this at exit.
        """

        client = self._client
        closer = getattr(client, "close", None)
        if callable(closer):
            try:
                closer()
            except Exception:  # closing is best effort at shutdown
                return

    @property
    def config(self) -> ArenaChainConfig:
        return self._config

    @property
    def client(self) -> Any:
        """The serialized client handle (safe to hand to helpers)."""

        return self._serialized

    def epoch_subtensor(self) -> _EpochSubtensor:
        """Holder exposing ``.substrate`` for ``read_subnet_epoch_snapshot``."""

        return _EpochSubtensor(substrate=self._serialized, chain_endpoint=self._config.endpoint)

    # -- raw call boundary ---------------------------------------------------

    def _call(self, name: str, *args: Any, **kwargs: Any) -> Any:
        method = getattr(self._client, name, None)
        if not callable(method):
            raise ArenaChainError("chain client lacks %s" % name)
        try:
            with self._lock:
                return method(*args, **kwargs)
        except ArenaChainError:
            raise
        except Exception as exc:
            raise ArenaChainError("chain call %s failed" % name) from exc

    # -- finalized chain -------------------------------------------------------

    def finalized_head(self) -> BlockRef:
        raw_hash = self._call("get_chain_finalised_head")
        if raw_hash is None:
            raise ArenaChainError("finalized head is unavailable")
        block_hash = normalize_block_hash(raw_hash)
        number = self._call("get_block_number", block_hash)
        if number is None:
            raise ArenaChainError("finalized head number is unavailable")
        return BlockRef(number=_chain_int(number, "finalized head number"), hash=block_hash)

    def finalized_head_hash(self) -> str:
        return self.finalized_head().hash

    def block_number(self, block_hash: Any) -> int:
        normalized = normalize_block_hash(block_hash)
        number = self._call("get_block_number", normalized)
        if number is None:
            raise ArenaBlockNotFound("block %s is unknown to the endpoint" % normalized)
        return _chain_int(number, "block number")

    def finalized_block_hash(self, number: int, *, head: Optional[BlockRef] = None) -> str:
        """Hash of block ``number`` on the finalized chain.

        Rule: ``chain_getBlockHash(n)`` returns the hash of block ``n`` on the
        node's best chain. Finality is prefix-closed, so for every ``n`` at or
        below the finalized head number the best chain and the finalized chain
        coincide and that hash is the finalized hash. Above the finalized head
        the best-chain hash can still be reorganized, so this method refuses
        to answer rather than returning a hash that may change.
        """

        number = _require_int(number, "block number", minimum=0)
        head = head if head is not None else self.finalized_head()
        if number > head.number:
            raise ArenaChainError("block %d is beyond the finalized head %d" % (number, head.number))
        raw_hash = self._call("get_block_hash", number)
        if raw_hash is None:
            raise ArenaBlockNotFound("block %d has no hash on the endpoint" % number)
        return normalize_block_hash(raw_hash)

    def is_finalized(self, block_hash: Any) -> bool:
        """True only when ``block_hash`` is the finalized chain's hash for its number."""

        normalized = normalize_block_hash(block_hash)
        number = self.block_number(normalized)
        head = self.finalized_head()
        if number > head.number:
            return False
        return self.finalized_block_hash(number, head=head) == normalized
    # -- finalized metagraph cache ------------------------------------------------

    def metagraph(self, finalized: bool = True, *, refresh: bool = False) -> MetagraphSnapshot:
        """The metagraph at the finalized head, cached by block hash with a TTL.

        Within ``metagraph_ttl_seconds`` the cached snapshot is served without
        a chain read. After the TTL the finalized head is re-read; an unchanged
        head keeps the snapshot, a new head refetches through the injected
        source. ``refresh=True`` forces a fetch. A source failure raises and
        never serves an expired snapshot. Only ``finalized=True`` is supported.
        """

        if finalized is not True:
            raise ArenaChainError("the Arena reads only the finalized metagraph")
        ttl = self._config.metagraph_ttl_seconds
        with self._metagraph_lock:
            now = float(self._clock())
            cached = self._cached_metagraph
            if cached is not None and not refresh and now - cached[1] < ttl:
                return cached[0]
            head = self.finalized_head()
            if cached is not None and not refresh and cached[0].block_hash == head.hash:
                self._cached_metagraph = (cached[0], now)
                return cached[0]
            snapshot = self._metagraph_source(self._serialized, self._config.netuid, head.hash)
            if not isinstance(snapshot, MetagraphSnapshot):
                raise ArenaChainError("metagraph source returned an unexpected object")
            if (
                snapshot.block_hash != head.hash
                or snapshot.block_number != head.number
                or snapshot.netuid != self._config.netuid
            ):
                raise ArenaChainError("metagraph snapshot is not pinned to the finalized head")
            self._cached_metagraph = (snapshot, now)
            return snapshot

    def refresh_metagraph(self) -> MetagraphSnapshot:
        return self.metagraph(finalized=True, refresh=True)

    def finalized_weight_submission_context(
        self, validator_hotkey: str
    ) -> Tuple[BlockRef, MetagraphSnapshot, bool]:
        """Read one finalized metagraph and the matching weight-rate limit."""

        hotkey = require_hotkey(validator_hotkey, "validator_hotkey")
        head = self.finalized_head()
        snapshot = self._metagraph_source(
            self._serialized, self._config.netuid, head.hash
        )
        if (
            not isinstance(snapshot, MetagraphSnapshot)
            or snapshot.block_hash != head.hash
            or snapshot.block_number != head.number
            or snapshot.netuid != self._config.netuid
        ):
            raise ArenaChainError(
                "weight submission metagraph is not pinned to the finalized head"
            )
        uid = uid_for_hotkey(snapshot, hotkey)
        if uid is None:
            raise ArenaChainError("validator hotkey is absent from finalized metagraph")
        rate_limit = _chain_int(
            self._call(
                "query",
                "SubtensorModule",
                "WeightsSetRateLimit",
                [self._config.netuid],
                block_hash=head.hash,
            ),
            "weights set rate limit",
        )
        updates = _unwrap(
            self._call(
                "query",
                "SubtensorModule",
                "LastUpdate",
                [self._config.netuid],
                block_hash=head.hash,
            )
        )
        if (
            rate_limit < 0
            or not isinstance(updates, (list, tuple))
            or uid >= len(updates)
        ):
            raise ArenaChainError("finalized weight rate-limit state is invalid")
        last_update = _chain_int(updates[uid], "validator last update")
        if last_update < 0 or last_update > head.number:
            raise ArenaChainError("finalized validator last update is invalid")
        ready = last_update == 0 or head.number - last_update >= rate_limit
        return head, snapshot, ready


# ---------------------------------------------------------------------------
# Settlement epoch ordinals (section 13.2)
# ---------------------------------------------------------------------------


def load_arena_cutover(
    environ: Optional[Mapping[str, str]] = None,
    *,
    default_manifest_path: Any = DEFAULT_SN71_CUTOVER_MANIFEST_PATH,
) -> SubnetEpochCutover:
    """Load the settlement cutover manifest without mutating the process environment.

    Explicit ``LEADPOET_SUBNET_EPOCH_CUTOVER_JSON`` / ``_PATH`` values win;
    when neither is set the repository's public SN71 manifest is used. The
    manifest is fully validated by the repository helper; failures raise.
    """

    source: Dict[str, str] = dict(os.environ if environ is None else environ)
    has_json = bool(str(source.get(CUTOVER_JSON_ENV, "") or "").strip())
    has_path = bool(str(source.get(CUTOVER_PATH_ENV, "") or "").strip())
    if not has_json and not has_path:
        source[CUTOVER_PATH_ENV] = str(default_manifest_path)
    try:
        return load_subnet_epoch_cutover(source)
    except SubnetEpochError as exc:
        raise ArenaChainError("subnet epoch cutover manifest is unavailable or invalid") from exc


def finalized_epoch_snapshot(chain: ArenaChain) -> SubnetEpochSnapshot:
    """Official scheduler state at the finalized head through the Arena's client."""

    if not isinstance(chain, ArenaChain):
        raise ArenaChainError("chain must be an ArenaChain")
    try:
        return read_subnet_epoch_snapshot(chain.epoch_subtensor(), netuid=chain.config.netuid, finalized=True)
    except SubnetEpochError as exc:
        raise ArenaChainError("finalized subnet epoch state is unavailable") from exc


def current_settlement_epoch(chain: ArenaChain, cutover: SubnetEpochCutover) -> int:
    """The settlement epoch ordinal at the finalized head.

    Reads ``SubnetEpochIndex`` and its sibling fields at the finalized hash via
    ``read_subnet_epoch_snapshot`` and maps it through
    ``SubnetEpochCutover.settlement_epoch_id``; the snapshot method also proves
    the genesis hash and netuid match the manifest.
    """

    if not isinstance(cutover, SubnetEpochCutover):
        raise ArenaChainError("cutover must be a SubnetEpochCutover")
    if not isinstance(chain, ArenaChain):
        raise ArenaChainError("chain must be an ArenaChain")
    if cutover.netuid != chain.config.netuid:
        raise ArenaChainError("cutover manifest netuid differs from the Arena chain configuration")
    snapshot = finalized_epoch_snapshot(chain)
    try:
        return int(snapshot.settlement_epoch_id(cutover))
    except SubnetEpochError as exc:
        raise ArenaChainError("finalized epoch does not map onto the configured cutover") from exc


# ---------------------------------------------------------------------------
# Signature verification seam
# ---------------------------------------------------------------------------


def verify_hotkey_signature(hotkey: str, signature_hex: str, message: str) -> bool:
    """Production verifier: the Lab's pure ``verify_sr25519_signature``.

    That function uses ``bittensor.Keypair`` (present in the pinned 10.5.0
    build). Under bittensor 11 that import fails and it returns ``False``, so
    local tests inject ``wallet_signature_verifier`` instead.
    """

    return bool(verify_sr25519_signature(hotkey, signature_hex, message))


def wallet_signature_verifier(hotkey: str, signature_hex: str, message: str) -> bool:
    """Local-environment verifier built on ``bittensor_wallet.Keypair``.

    Mirrors the Lab verifier's message handling: a ``0x``-prefixed message is
    hex-decoded, anything else is UTF-8 encoded. A malformed hotkey or
    signature is a well-formed "does not verify" (``False``); any other error
    propagates.
    """

    try:
        from bittensor_wallet import Keypair
    except ImportError as exc:
        raise ArenaChainError("bittensor_wallet is not installed") from exc
    if not isinstance(hotkey, str) or not isinstance(signature_hex, str) or not isinstance(message, str):
        return False
    signature = signature_hex[2:] if signature_hex.startswith("0x") else signature_hex
    try:
        data = bytes.fromhex(message[2:]) if message.startswith("0x") else message.encode("utf-8")
        keypair = Keypair(ss58_address=hotkey)
        return bool(keypair.verify(data, bytes.fromhex(signature)))
    except ValueError:
        return False


__all__ = [
    "ArenaBlockNotFound",
    "ArenaChain",
    "ArenaChainConfig",
    "ArenaChainConfigError",
    "ArenaChainError",
    "BlockRef",
    "InvalidBlockHash",
    "MetagraphSnapshot",
    "SubstrateClient",
    "bittensor_metagraph_source",
    "coldkey_owns_hotkey",
    "connect_substrate",
    "current_settlement_epoch",
    "finalized_epoch_snapshot",
    "hotkeys_owned_by_coldkey",
    "is_registered",
    "load_arena_cutover",
    "metagraph_snapshot_from_object",
    "normalize_block_hash",
    "normalize_network_name",
    "uid_for_hotkey",
    "verify_hotkey_signature",
    "wallet_signature_verifier",
]
