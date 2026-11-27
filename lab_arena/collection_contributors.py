"""Project public contributions and verified native payments from the existing ledgers."""

from functools import lru_cache
import json
from pathlib import Path

from lab_arena.store import ArenaStoreUnavailable
from scripts.collection_review import read_file

PAGE_SIZE = 25
READ_ERRORS = (OSError, ValueError, TypeError, KeyError)


@lru_cache(maxsize=128)
def _confirmed_amounts(path: Path, epoch: int):
    # The settlement watcher publishes each epoch file exactly once by atomic link.
    receipt = json.loads(read_file(path, maximum=4 << 20))
    if (receipt["schema"] != "leadpoet.collection.revealed_payment.v1"
            or receipt["result"] != "collection_recipients_observed"
            or receipt["quote_epoch"] != epoch or receipt["distribution_epoch"] != epoch + 1):
        raise ValueError("unconfirmed collection settlement")
    result = {}
    for hotkey, quote in receipt["quotes_microusd"].items():
        amount = receipt["amounts"][hotkey]
        paid = amount["actual_rao"]
        if (hotkey == receipt["burn_hotkey"] or amount["quoted_microusd"] != quote
                or type(paid) is not int or paid < 0
                or receipt["miner_emissions_rao"].get(hotkey, 0) != paid):
            raise ValueError("settlement amount mismatch")
        result[hotkey] = (quote, paid)
    return result


def public_page(rows, directory: Path):
    if not isinstance(rows, list) or len(rows) > PAGE_SIZE + 1:
        raise ArenaStoreUnavailable("unavailable collection contributors")

    through = None
    if any(row["epochs"] for row in rows[:PAGE_SIZE]):
        try:
            snapshot = json.loads(read_file(directory / "status.json", maximum=4 << 20))
            through = snapshot["settlement"]["through"]
        except READ_ERRORS:
            pass  # Quotes remain available during a watcher outage.

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
                observed, amount = _confirmed_amounts(directory / "settlements" / f"{epoch}.json", epoch)[hotkey]
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
    return {"miners": miners, "next": miners[-1]["hotkey"] if len(rows) > PAGE_SIZE else None}
