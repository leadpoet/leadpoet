"""One UTC evaluation date shared by baseline and miner scoring."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import date, datetime, time, timezone
import os
from typing import Iterator


EVALUATION_DATE_ENV = "LEADPOET_COMPETITION_EVALUATION_DATE"


def evaluation_date() -> date:
    value = str(os.getenv(EVALUATION_DATE_ENV) or "").strip()
    if value:
        try:
            return date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(f"{EVALUATION_DATE_ENV} must use YYYY-MM-DD") from exc
    return datetime.now(timezone.utc).date()


def evaluation_datetime() -> datetime:
    return datetime.combine(evaluation_date(), time.min, tzinfo=timezone.utc)


@contextmanager
def use_evaluation_date(value: str) -> Iterator[None]:
    try:
        normalized = date.fromisoformat(str(value)).isoformat()
    except ValueError as exc:
        raise ValueError("evaluation date must use YYYY-MM-DD") from exc
    previous = os.environ.get(EVALUATION_DATE_ENV)
    os.environ[EVALUATION_DATE_ENV] = normalized
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(EVALUATION_DATE_ENV, None)
        else:
            os.environ[EVALUATION_DATE_ENV] = previous
