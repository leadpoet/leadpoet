"""Dedicated-wallet TAO deposit verifier (labarena.md section 7.2, the
section 18.3 deposit bullet, and labarenaaudit.md blocker 4).

The Arena publishes one TAO recipient wallet that no other Leadpoet payment
flow accepts. A miner presents ``block_hash`` and ``extrinsic_index``; this
module verifies the deposit through ``lab_arena.chain.ArenaChain`` (one
explicit endpoint, every read raising on failure) and then asks the deposit
store to claim the normalized payment reference exactly once.

Verification order. Each step raises ``DepositRejected(rule_id)``:

    malformed_reference  the presented reference cannot be normalized
    finality             the block hash is not the finalized chain's hash for
                         its number (unknown block, fork, or beyond the head)
    not_transfer         the extrinsic is not a direct ``Balances`` transfer
    transfer_failed      no explicit ``System.ExtrinsicSuccess`` event, or an
                         ``ExtrinsicFailed`` event
    wrong_destination    the destination is not the dedicated Arena wallet
    sender_not_owner     the signer coldkey does not own the miner hotkey on
                         the finalized metagraph cache
    stale_block          the block is older than ``max_age_seconds`` or more
                         than ``FUTURE_BLOCK_TOLERANCE_SECONDS`` in the future
    amount_invalid       no explicit positive amount (``transfer_all``), or an
                         amount that floors to zero micro-USD

A chain failure raises ``ArenaChainError``, a price failure raises
``PriceUnavailable``, and a store failure raises ``FundingStoreError``; in
every case no credit exists. The reference is normalized to
``network:0x<lowercase 64 hex>:<int>`` before any uniqueness check so a
formatting variant can never create a second credit.

The real ``DepositStore`` is the SQL function ``lab_arena_credit_deposit``
(migration 178), which claims the reference and credits the miner's single
account under a row lock; tests use an in-memory fake.
"""

from __future__ import annotations

import json
import re
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_FLOOR, localcontext
from typing import Any, Callable, Dict, Mapping, Optional, Protocol, Tuple

from lab_arena.chain import (
    ArenaBlockNotFound,
    ArenaChain,
    ArenaChainConfigError,
    ArenaExtrinsicStatusUnknown,
    ArenaNotTransfer,
    ArenaTransferAmountUnknown,
    InvalidBlockHash,
    coldkey_owns_hotkey,
    normalize_block_hash,
    normalize_network_name,
)
from lab_arena.contracts import check_strict_document, hashed_document, require_hotkey

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEPOSIT_DOCUMENT_SCHEMA_VERSION = "leadpoet.lab_arena.deposit.v1"
DEPOSIT_RULE_IDS = (
    "malformed_reference",
    "finality",
    "not_transfer",
    "transfer_failed",
    "wrong_destination",
    "sender_not_owner",
    "stale_block",
    "amount_invalid",
)
RAO_PER_TAO = 1_000_000_000
MICROUSD_PER_USD = 1_000_000
DEFAULT_MAX_DEPOSIT_AGE_SECONDS = 86_400
FUTURE_BLOCK_TOLERANCE_SECONDS = 300
MAX_EXTRINSIC_INDEX_DIGITS = 12

COINGECKO_PRICE_URL = "https://api.coingecko.com/api/v3/simple/price?ids=bittensor&vs_currencies=usd"
PRICE_SOURCE_NAME = "coingecko"
PRICE_USER_AGENT = "leadpoet-lab-arena/1"
MAX_PRICE_RESPONSE_BYTES = 65_536
DEFAULT_PRICE_TIMEOUT_SECONDS = 10

_DIGITS_RE = re.compile(r"^[0-9]{1,%d}$" % MAX_EXTRINSIC_INDEX_DIGITS)
_REFERENCE_RE = re.compile(r"^([a-z][a-z0-9_-]{0,31}):(0x[0-9a-f]{64}):(0|[1-9][0-9]{0,%d})$" % (MAX_EXTRINSIC_INDEX_DIGITS - 1))


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class FundingError(RuntimeError):
    """A funding operation cannot proceed. Never creates a credit."""


class MalformedReference(FundingError, ValueError):
    """A payment reference component cannot be normalized."""


class PriceUnavailable(FundingError):
    """The TAO price could not be obtained or is not a positive number."""


class FundingStoreError(FundingError):
    """The deposit store failed or returned an unusable result."""


class DepositRejected(FundingError):
    """One verification rule rejected the deposit. ``rule_id`` names it."""

    def __init__(self, rule_id: str, detail: str = "") -> None:
        if rule_id not in DEPOSIT_RULE_IDS:
            raise ValueError("unknown deposit rule id %r" % rule_id)
        self.rule_id = rule_id
        self.detail = detail
        super().__init__("%s: %s" % (rule_id, detail) if detail else rule_id)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FundingConfig:
    """The dedicated recipient wallet and the network it lives on."""

    recipient_wallet: str
    network_name: str
    max_age_seconds: int = DEFAULT_MAX_DEPOSIT_AGE_SECONDS

    def __post_init__(self) -> None:
        require_hotkey(self.recipient_wallet, "recipient_wallet")
        object.__setattr__(self, "network_name", normalize_network_name(self.network_name))
        age = self.max_age_seconds
        if isinstance(age, bool) or not isinstance(age, int) or age <= 0:
            raise FundingError("max_age_seconds must be a positive integer")


# ---------------------------------------------------------------------------
# Payment reference normalization
# ---------------------------------------------------------------------------


def normalize_extrinsic_index(value: Any) -> int:
    """Accept a non-negative int or a digit string (surrounding whitespace allowed)."""

    if isinstance(value, bool):
        raise MalformedReference("extrinsic index must not be a boolean")
    if isinstance(value, int):
        if value < 0:
            raise MalformedReference("extrinsic index must be non-negative")
        return value
    if isinstance(value, str):
        text = value.strip()
        if not _DIGITS_RE.match(text):
            raise MalformedReference("extrinsic index must be a non-negative integer")
        return int(text, 10)
    raise MalformedReference("extrinsic index must be an integer")


def normalize_payment_reference(network_name: Any, block_hash: Any, extrinsic_index: Any) -> str:
    """``network:0x<lowercase 64 hex>:<int>`` for every accepted input variant.

    Accepted hash variants: with ``0x`` or ``0X`` or no prefix, any hex case,
    surrounding whitespace. Accepted index variants: int, or a digit string
    with surrounding whitespace. Everything else raises ``MalformedReference``.
    """

    try:
        network = normalize_network_name(network_name)
    except ArenaChainConfigError as exc:
        raise MalformedReference("network name is malformed") from exc
    try:
        normalized_hash = normalize_block_hash(block_hash)
    except InvalidBlockHash as exc:
        raise MalformedReference("block hash is malformed") from exc
    index = normalize_extrinsic_index(extrinsic_index)
    return "%s:%s:%d" % (network, normalized_hash, index)


def parse_payment_reference(reference: Any) -> Tuple[str, str, int]:
    """Split a canonical reference; anything non-canonical raises."""

    if not isinstance(reference, str):
        raise MalformedReference("payment reference must be a string")
    match = _REFERENCE_RE.match(reference)
    if not match:
        raise MalformedReference("payment reference is not canonical")
    return match.group(1), match.group(2), int(match.group(3), 10)


# ---------------------------------------------------------------------------
# Price source and micro-USD conversion
# ---------------------------------------------------------------------------


class PriceSource(Protocol):
    def tao_price_usd(self) -> Decimal: ...


def _require_price(value: Any) -> Decimal:
    if not isinstance(value, Decimal):
        raise PriceUnavailable("TAO price must be a Decimal")
    if not value.is_finite() or value <= 0:
        raise PriceUnavailable("TAO price must be a positive finite number")
    return value


class CoinGeckoPriceSource:
    """``PriceSource`` over CoinGecko's simple-price endpoint.

    ``urlopen`` is injected (``urllib.request.urlopen`` in production); it is
    always called with an explicit ``timeout``. Any transport, status, size,
    JSON, or value problem raises ``PriceUnavailable``.
    """

    def __init__(self, urlopen: Callable[..., Any], timeout: float) -> None:
        if not callable(urlopen):
            raise FundingError("urlopen must be callable")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
            raise FundingError("timeout must be a positive number of seconds")
        self._urlopen = urlopen
        self._timeout = timeout

    @property
    def name(self) -> str:
        return PRICE_SOURCE_NAME

    def tao_price_usd(self) -> Decimal:
        request = urllib.request.Request(
            COINGECKO_PRICE_URL,
            headers={"Accept": "application/json", "User-Agent": PRICE_USER_AGENT},
        )
        try:
            with self._urlopen(request, timeout=self._timeout) as response:
                status = getattr(response, "status", None)
                if status != 200:
                    raise PriceUnavailable("price endpoint returned status %r" % (status,))
                raw = response.read(MAX_PRICE_RESPONSE_BYTES + 1)
        except PriceUnavailable:
            raise
        except Exception as exc:
            raise PriceUnavailable("TAO price request failed") from exc
        if not isinstance(raw, (bytes, bytearray)) or len(raw) > MAX_PRICE_RESPONSE_BYTES:
            raise PriceUnavailable("price response is missing or too large")
        try:
            payload = json.loads(bytes(raw).decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise PriceUnavailable("price response is not JSON") from exc
        entry = payload.get("bittensor") if isinstance(payload, Mapping) else None
        value = entry.get("usd") if isinstance(entry, Mapping) else None
        if value is None or isinstance(value, bool) or not isinstance(value, (int, float, str)):
            raise PriceUnavailable("price response lacks bittensor.usd")
        try:
            price = Decimal(str(value))
        except InvalidOperation as exc:
            raise PriceUnavailable("price response value is not numeric") from exc
        return _require_price(price)


def coingecko_price_source(
    urlopen: Callable[..., Any] = urllib.request.urlopen,
    timeout: float = DEFAULT_PRICE_TIMEOUT_SECONDS,
) -> CoinGeckoPriceSource:
    return CoinGeckoPriceSource(urlopen, timeout)


def rao_to_microusd(amount_rao: int, price_usd: Decimal) -> int:
    """Integer micro-USD for ``amount_rao`` at ``price_usd`` per TAO, rounded down.

    1 TAO = 1e9 rao and 1 USD = 1e6 micro-USD, so the value is
    ``floor(amount_rao * price_usd / 1000)`` computed exactly in ``Decimal``.
    """

    if isinstance(amount_rao, bool) or not isinstance(amount_rao, int) or amount_rao < 0:
        raise FundingError("amount_rao must be a non-negative integer")
    price = _require_price(price_usd)
    with localcontext() as context:
        context.prec = 60
        micro = (Decimal(amount_rao) * price * MICROUSD_PER_USD) / RAO_PER_TAO
        return int(micro.to_integral_value(rounding=ROUND_FLOOR))


# ---------------------------------------------------------------------------
# Deposit store protocol and verified deposit
# ---------------------------------------------------------------------------


class DepositStore(Protocol):
    """Claims one normalized reference exactly once and credits the miner.

    Returns ``{"credited": bool, "idempotent": bool, "balance_microusd": int}``
    where exactly one of ``credited`` (this call created the credit) and
    ``idempotent`` (the reference was already credited) is true. Any failure
    must raise; the caller then creates no credit.
    """

    def credit_deposit(
        self,
        *,
        miner_hotkey: str,
        payment_reference: str,
        amount_microusd: int,
        deposit_doc: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...


def _iso(moment: datetime) -> str:
    utc = moment.astimezone(timezone.utc)
    return utc.strftime("%Y-%m-%dT%H:%M:%S.") + "%03dZ" % (utc.microsecond // 1000)


def _require_aware(value: Any, field_name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise FundingError("%s must be a timezone-aware datetime" % field_name)
    return value


@dataclass(frozen=True)
class VerifiedDeposit:
    """Everything the ledger row records for one verified deposit."""

    miner_hotkey: str
    payment_reference: str
    network: str
    block_number: int
    block_hash: str
    extrinsic_index: int
    extrinsic_hash: Optional[str]
    sender: str
    recipient: str
    amount_rao: int
    block_timestamp: datetime
    price_usd: Decimal
    price_source: str
    price_observed_at: datetime
    amount_microusd: int

    def deposit_doc(self) -> Dict[str, Any]:
        body = {
            "schema_version": DEPOSIT_DOCUMENT_SCHEMA_VERSION,
            "network": self.network,
            "payment_reference": self.payment_reference,
            "miner_hotkey": self.miner_hotkey,
            "block_number": self.block_number,
            "block_hash": self.block_hash,
            "extrinsic_index": self.extrinsic_index,
            "extrinsic_hash": self.extrinsic_hash,
            "sender": self.sender,
            "recipient": self.recipient,
            "amount_rao": self.amount_rao,
            "block_timestamp": _iso(self.block_timestamp),
            "price_usd": format(self.price_usd, "f"),
            "price_source": self.price_source,
            "price_observed_at": _iso(self.price_observed_at),
            "amount_microusd": self.amount_microusd,
        }
        document = hashed_document(body, "deposit_hash")
        check_strict_document(document)
        return document


@dataclass(frozen=True)
class FundingReceipt:
    deposit: VerifiedDeposit
    deposit_doc: Dict[str, Any]
    credited: bool
    idempotent: bool
    balance_microusd: int


# ---------------------------------------------------------------------------
# Verification and confirmation
# ---------------------------------------------------------------------------


def verify_deposit(
    *,
    chain: ArenaChain,
    config: FundingConfig,
    miner_hotkey: str,
    block_hash: Any,
    extrinsic_index: Any,
    now: datetime,
    price_source: PriceSource,
) -> VerifiedDeposit:
    """Run the section 7.2 checks in order and return the verified deposit.

    Performs no write. The price is fetched only after every chain check has
    passed so an invalid presentation never costs a price request.
    """

    if not isinstance(chain, ArenaChain):
        raise FundingError("chain must be an ArenaChain")
    if not isinstance(config, FundingConfig):
        raise FundingError("config must be a FundingConfig")
    if config.network_name != chain.config.network_name:
        raise FundingError("funding network does not match the chain connection")
    observed_at = _require_aware(now, "now")
    hotkey = require_hotkey(miner_hotkey, "miner_hotkey")

    try:
        reference = normalize_payment_reference(config.network_name, block_hash, extrinsic_index)
        normalized_hash = normalize_block_hash(block_hash)
        index = normalize_extrinsic_index(extrinsic_index)
    except MalformedReference as exc:
        raise DepositRejected("malformed_reference", str(exc)) from exc

    # 1. Finality: the presented hash must be the finalized chain's hash for
    #    its number. An unknown block, a fork block, or a block beyond the
    #    finalized head all fail here.
    try:
        block = chain.block(normalized_hash)
    except ArenaBlockNotFound as exc:
        raise DepositRejected("finality", "block is unknown to the endpoint") from exc
    head = chain.finalized_head()
    number = block["block_number"]
    if number > head.number:
        raise DepositRejected("finality", "block %d is beyond the finalized head %d" % (number, head.number))
    if chain.finalized_block_hash(number, head=head) != normalized_hash:
        raise DepositRejected("finality", "block hash is not on the finalized chain")

    # 2. A direct balance transfer with an explicit amount.
    try:
        transfer = chain.transfer_details(block, index)
    except ArenaTransferAmountUnknown as exc:
        raise DepositRejected("amount_invalid", str(exc)) from exc
    except ArenaNotTransfer as exc:
        raise DepositRejected("not_transfer", str(exc)) from exc

    # 3. An explicit success event for that extrinsic.
    try:
        succeeded = chain.extrinsic_succeeded(normalized_hash, index)
    except ArenaExtrinsicStatusUnknown as exc:
        raise DepositRejected("transfer_failed", "no explicit success event") from exc
    if not succeeded:
        raise DepositRejected("transfer_failed", "extrinsic failed on chain")

    # 4. The dedicated Arena wallet.
    if transfer["destination"] != config.recipient_wallet:
        raise DepositRejected("wrong_destination", "destination is not the Arena wallet")

    # 5. The signer coldkey owns the miner hotkey on the finalized metagraph.
    snapshot = chain.metagraph(finalized=True)
    if not coldkey_owns_hotkey(snapshot, transfer["sender"], hotkey):
        raise DepositRejected("sender_not_owner", "signer coldkey does not own the miner hotkey")

    # 6. The block is inside the deposit window.
    block_time = chain.block_timestamp(normalized_hash)
    age_seconds = (observed_at - block_time).total_seconds()
    if age_seconds > config.max_age_seconds:
        raise DepositRejected("stale_block", "block is older than %d seconds" % config.max_age_seconds)
    if age_seconds < -FUTURE_BLOCK_TOLERANCE_SECONDS:
        raise DepositRejected("stale_block", "block timestamp is in the future")

    # 7. Integer micro-USD, rounded down, at a price fetched now.
    amount_rao = transfer["amount_rao"]
    price = _require_price(price_source.tao_price_usd())
    amount_microusd = rao_to_microusd(amount_rao, price)
    if amount_microusd <= 0:
        raise DepositRejected("amount_invalid", "amount rounds down to zero micro-USD")

    return VerifiedDeposit(
        miner_hotkey=hotkey,
        payment_reference=reference,
        network=config.network_name,
        block_number=number,
        block_hash=normalized_hash,
        extrinsic_index=index,
        extrinsic_hash=transfer.get("extrinsic_hash"),
        sender=transfer["sender"],
        recipient=transfer["destination"],
        amount_rao=amount_rao,
        block_timestamp=block_time,
        price_usd=price,
        price_source=str(getattr(price_source, "name", type(price_source).__name__)),
        price_observed_at=observed_at,
        amount_microusd=amount_microusd,
    )


def _require_result_bool(result: Mapping[str, Any], key: str) -> bool:
    value = result.get(key)
    if not isinstance(value, bool):
        raise FundingStoreError("deposit store result field %s must be a boolean" % key)
    return value


def confirm_funding(
    *,
    chain: ArenaChain,
    config: FundingConfig,
    store: DepositStore,
    miner_hotkey: str,
    block_hash: Any,
    extrinsic_index: Any,
    now: datetime,
    price_source: PriceSource,
) -> FundingReceipt:
    """Verify, then credit through the store exactly once.

    Verification errors propagate before any store call. A store failure or an
    unusable store result raises ``FundingStoreError``; the caller must treat
    that as "no credit exists" and let the miner retry with the same reference.
    """

    verified = verify_deposit(
        chain=chain,
        config=config,
        miner_hotkey=miner_hotkey,
        block_hash=block_hash,
        extrinsic_index=extrinsic_index,
        now=now,
        price_source=price_source,
    )
    document = verified.deposit_doc()
    try:
        result = store.credit_deposit(
            miner_hotkey=verified.miner_hotkey,
            payment_reference=verified.payment_reference,
            amount_microusd=verified.amount_microusd,
            deposit_doc=document,
        )
    except FundingError:
        raise
    except Exception as exc:
        raise FundingStoreError("deposit store is unavailable") from exc
    if not isinstance(result, Mapping):
        raise FundingStoreError("deposit store returned a non-object result")
    credited = _require_result_bool(result, "credited")
    idempotent = _require_result_bool(result, "idempotent")
    if credited == idempotent:
        raise FundingStoreError("deposit store must report exactly one of credited or idempotent")
    balance = result.get("balance_microusd")
    if isinstance(balance, bool) or not isinstance(balance, int) or balance < 0:
        raise FundingStoreError("deposit store balance must be a non-negative integer")
    return FundingReceipt(
        deposit=verified,
        deposit_doc=document,
        credited=credited,
        idempotent=idempotent,
        balance_microusd=balance,
    )


__all__ = [
    "COINGECKO_PRICE_URL",
    "CoinGeckoPriceSource",
    "DEFAULT_MAX_DEPOSIT_AGE_SECONDS",
    "DEPOSIT_DOCUMENT_SCHEMA_VERSION",
    "DEPOSIT_RULE_IDS",
    "DepositRejected",
    "DepositStore",
    "FUTURE_BLOCK_TOLERANCE_SECONDS",
    "FundingConfig",
    "FundingError",
    "FundingReceipt",
    "FundingStoreError",
    "MICROUSD_PER_USD",
    "MalformedReference",
    "PriceSource",
    "PriceUnavailable",
    "RAO_PER_TAO",
    "VerifiedDeposit",
    "coingecko_price_source",
    "confirm_funding",
    "normalize_extrinsic_index",
    "normalize_payment_reference",
    "parse_payment_reference",
    "rao_to_microusd",
    "verify_deposit",
]
