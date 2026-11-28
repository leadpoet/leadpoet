"""Project public contributions and verified native payments from the existing ledgers."""

from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path

from lab_arena.store import ArenaStoreUnavailable
from scripts.collection_review import read_file

PAGE_SIZE = 25
READ_ERRORS = (OSError, ValueError, TypeError, KeyError)


@dataclass(frozen=True)
class _Payment:
    amounts: dict
    pool_rao: int | None
    closed_at: str | None
    digest: str


@lru_cache(maxsize=128)
def _confirmed_payment(path: Path, epoch: int) -> _Payment:
    # The settlement watcher publishes each epoch file exactly once by atomic link.
    data = read_file(path, maximum=4 << 20)
    receipt = json.loads(data)
    if (receipt["schema"] != "leadpoet.collection.revealed_payment.v1"
            or receipt["result"] != "collection_recipients_observed"
            or receipt["quote_epoch"] != epoch or receipt["distribution_epoch"] != epoch + 1):
        raise ValueError("unconfirmed collection settlement")
    result = {}
    for hotkey, quote in receipt["quotes_microusd"].items():
        amount = receipt["amounts"][hotkey]
        paid = amount["actual_rao"]
        if hotkey in receipt.get("supplier_targets_microusd", {}) and paid is None:
            continue  # Combined UID earnings cannot be reported as lab income.
        if (hotkey == receipt["burn_hotkey"] or amount["quoted_microusd"] != quote
                or type(paid) is not int or paid < 0
                or receipt["miner_emissions_rao"].get(hotkey, 0) != paid):
            raise ValueError("settlement amount mismatch")
        result[hotkey] = (quote, paid)
    pool = receipt.get("total_miner_rao")
    emissions = receipt["miner_emissions_rao"]
    if (type(pool) is not int or pool <= 0
            or any(type(value) is not int or value < 0 for value in emissions.values())
            or sum(emissions.values()) != pool):
        pool = None
    return _Payment(result, pool, receipt.get("closed_at"), sha256(data).hexdigest())


def _latest_payout(progress, directory):
    """Compare all recipients of the same payment, including burn in the denominator only."""
    try:
        through = progress["through"]
        if type(through) is not int:
            raise ValueError("invalid settlement cursor")
        if "last_paid" not in progress:
            return {"status": "none"}
        last = progress["last_paid"]
        epoch = last["quote_epoch"]
        if (type(epoch) is not int or not 0 <= epoch <= through
                or last["distribution_epoch"] != epoch + 1
                or last["result"] != "collection_recipients_observed"):
            raise ValueError("unconfirmed payout")
        payment = _confirmed_payment(directory / "settlements" / f"{epoch}.json", epoch)
        if (payment.digest != last["receipt_sha256"] or payment.pool_rao is None
                or datetime.fromisoformat(payment.closed_at).utcoffset() is None):
            raise ValueError("invalid payout receipt")
        return {
            "status": "confirmed", "epoch": epoch + 1, "closed_at": payment.closed_at,
            "shares": {key: paid * 100 / payment.pool_rao for key, (_, paid) in payment.amounts.items()},
        }
    except READ_ERRORS:
        return {"status": "unavailable"}


def public_page(rows, directory: Path):
    if not isinstance(rows, list) or len(rows) > PAGE_SIZE + 1:
        raise ArenaStoreUnavailable("unavailable collection contributors")

    try:
        progress = json.loads(read_file(directory / "status.json", maximum=4 << 20))["settlement"]
        through = progress["through"]
    except READ_ERRORS:
        progress, through = {}, None  # Quotes remain available during a watcher outage.

    miners = []
    for row in rows[:PAGE_SIZE]:
        hotkey = row["hotkey"]
        paid = 0
        status = "confirmed"
        for key, quote in row["epochs"].items():
            epoch = int(key)
            if type(through) is not int:
                status = "unavailable"
                break
            if epoch > through:
                status = "pending"
                continue
            try:
                observed, amount = _confirmed_payment(directory / "settlements" / f"{epoch}.json", epoch).amounts[hotkey]
                if observed != quote:
                    raise ValueError("settled quote mismatch")
                paid += amount
            except READ_ERRORS:
                status = "unavailable"
                break
        miners.append({
            "hotkey": hotkey, "joined_at": row["joined_at"], "last_funded_at": row["last_funded_at"],
            "funded_runs": row["funded_runs"],
            "earned_quote_microusd": str(sum(row["epochs"].values())),
            "paid_alpha_rao": None if status == "unavailable" else str(paid), "payout_status": status,
        })
    return {
        "miners": miners,
        "next": miners[-1]["hotkey"] if len(rows) > PAGE_SIZE else None,
        "latest_payout": _latest_payout(progress, directory),
    }
