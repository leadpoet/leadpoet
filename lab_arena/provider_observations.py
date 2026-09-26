"""Project authenticated provider observations from one accepted execution."""

from __future__ import annotations

import base64
import binascii
import json
from datetime import date, datetime, timezone
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit

from lab_arena import contracts


TOOL = "predictleads_company_job_openings"


def enabled(policy: Mapping[str, Any]) -> bool:
    """Return whether a frozen scorer policy supports this input field."""

    bindings = policy.get("env_bindings") if isinstance(policy, Mapping) else None
    return isinstance(bindings, Mapping) and bindings.get(
        contracts.PROVIDER_OBSERVATION_HANDOFF_BINDING
    ) == contracts.AUTHENTICATED_PROVIDER_OBSERVATION_HANDOFF


_MAX_RESPONSE_BYTES = 1024 * 1024
_LEDGER_SCAN_LIMIT = (
    2 * max(profile["deepline"] for profile in contracts.EXECUTION_CALL_QUOTA_PROFILES)
    + 1
)
_MAX_COMPANIES = 5


def _company_host(company: Mapping[str, Any]) -> str:
    try:
        return (urlsplit(str(company.get("company_website") or "")).hostname or "").lower().removeprefix("www.")
    except ValueError:
        return ""


def _same_company_host(source_url: str, company_host: str) -> bool:
    try:
        source_host = (urlsplit(source_url).hostname or "").lower().removeprefix("www.")
    except ValueError:
        return False
    return bool(
        company_host
        and source_host
        and (
            source_host == company_host
            or source_host.endswith("." + company_host)
        )
    )


def _decode_success(row: Mapping[str, Any]) -> Any:
    terminal = row.get("terminal_response")
    if not isinstance(terminal, Mapping) or terminal.get("call_succeeded") is not True:
        return None
    try:
        if not 200 <= int(terminal["status"]) < 300:
            return None
        raw = base64.b64decode(terminal["body_b64"], validate=True)
        if len(raw) > _MAX_RESPONSE_BYTES:
            return None
        return json.loads(raw)
    except (KeyError, TypeError, ValueError, binascii.Error):
        return None


def _trusted_response(
    rows: Sequence[Mapping[str, Any]], *, run_id: str, call_identity: str
) -> Any:
    reservations = [row for row in rows if row.get("entry_kind") == "reservation"]
    settlements = [row for row in rows if row.get("entry_kind") == "settlement"]
    if len(reservations) != 1 or len(settlements) != 1:
        return None
    reservation, settlement = reservations[0], settlements[0]
    if any(
        row.get("run_id") != run_id
        or row.get("call_identity") != call_identity
        or row.get("provider") != "deepline"
        or row.get("operation_id") != "deepline.execute"
        for row in (reservation, settlement)
    ):
        return None
    entry_doc = reservation.get("entry_doc")
    request_hash = entry_doc.get("request_hash") if isinstance(entry_doc, Mapping) else None
    if (
        not isinstance(entry_doc, Mapping)
        or entry_doc.get("tool") != TOOL
        or not isinstance(request_hash, str)
        or len(request_hash) != 71
        or not request_hash.startswith("sha256:")
        or any(character not in "0123456789abcdef" for character in request_hash[7:])
    ):
        return None
    # The append-only ledger stores the request hash on its unique reservation.
    # A settlement has no separate request-hash column: its unique call identity
    # is copied from and foreign-bound to that exact reservation by the RPC.
    return _decode_success(settlement)


def _first_observed_date(value: Any, *, evaluated_on: date) -> str:
    if not isinstance(value, str) or not value or len(value) > 40:
        return ""
    try:
        observed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return ""
    if observed.tzinfo is None or observed.utcoffset() != timezone.utc.utcoffset(observed):
        return ""
    observed_date = observed.date()
    return observed_date.isoformat() if observed_date <= evaluated_on else ""


def _records(response: Any, *, evaluated_on: date) -> list[tuple[str, str]]:
    if not isinstance(response, Mapping) or response.get("status") != "completed":
        return []
    result = response.get("result")
    data = result.get("data") if isinstance(result, Mapping) else None
    if not isinstance(data, list) or len(data) > 500:
        return []
    records: list[tuple[str, str]] = []
    for item in data:
        attributes = item.get("attributes") if isinstance(item, Mapping) else None
        if not isinstance(attributes, Mapping):
            continue
        source_url = attributes.get("url")
        observed_date = _first_observed_date(
            attributes.get("first_seen_at"), evaluated_on=evaluated_on
        )
        if (
            isinstance(source_url, str)
            and source_url == source_url.strip()
            and 8 <= len(source_url) <= 2_000
            and source_url.startswith("https://")
            and observed_date
        ):
            records.append((source_url, observed_date))
    return records


def resolve_observations(
    store: Any,
    run: Mapping[str, Any],
    companies: Sequence[Mapping[str, Any]],
    evaluation_date: str,
) -> list[dict[str, Any]]:
    """Return only unique, company-bound observations for an accepted run."""

    if run.get("status") != "accepted":
        return []
    run_id = str(run.get("run_id") or "")
    try:
        evaluated_on = date.fromisoformat(str(evaluation_date))
    except ValueError:
        return []
    submitted: dict[str, list[tuple[int, str]]] = {}
    for company_index, company in enumerate(list(companies)[:_MAX_COMPANIES]):
        company_host = _company_host(company)
        signals = company.get("intent_signals")
        if not company_host or not isinstance(signals, list):
            continue
        for signal in signals:
            source_url = signal.get("url") if isinstance(signal, Mapping) else None
            if (
                isinstance(source_url, str)
                and _same_company_host(source_url, company_host)
            ):
                submitted.setdefault(source_url, []).append((company_index, company_host))

    ledger_rows = store.list_ledger(
        run_id=run_id,
        provider="deepline",
        limit=_LEDGER_SCAN_LIMIT,
    )
    if len(ledger_rows) >= _LEDGER_SCAN_LIMIT:
        return []
    candidates: dict[tuple[int, str], list[dict[str, Any]]] = {}
    rows_by_call: dict[str, list[Mapping[str, Any]]] = {}
    for row in ledger_rows:
        call_identity = str(row.get("call_identity") or "")
        if call_identity:
            rows_by_call.setdefault(call_identity, []).append(row)
    for call_identity, rows in rows_by_call.items():
        response = _trusted_response(
            rows, run_id=run_id, call_identity=call_identity
        )
        for source_url, observed_date in _records(
            response, evaluated_on=evaluated_on
        ):
            bindings = submitted.get(source_url, [])
            if len(bindings) != 1:
                continue
            company_index, company_host = bindings[0]
            candidate = {
                "company_index": company_index,
                "company_domain": company_host,
                "source_url": source_url,
                "first_observed_date": observed_date,
            }
            candidates.setdefault((company_index, source_url), []).append(candidate)

    unique_by_company: dict[
        int, dict[tuple[str, str, str], dict[str, Any]]
    ] = {}
    for (company_index, _source_url), rows in sorted(candidates.items()):
        for row in rows:
            identity = (
                row["company_domain"], row["source_url"],
                row["first_observed_date"],
            )
            unique_by_company.setdefault(company_index, {})[identity] = row
    # Admit one unambiguous listing observation per company. A company with
    # conflicting dates or several candidate listings gets no provider date.
    # Repeated identical authenticated observations are safe corroboration.
    return [
        next(iter(rows.values()))
        for _company_index, rows in sorted(unique_by_company.items())
        if len(rows) == 1
    ]


def company_observation(
    observations: Any, company_index: int
) -> dict[str, Any] | None:
    """Validate and select one bounded scorer projection for a company."""

    if not isinstance(observations, list) or len(observations) > _MAX_COMPANIES:
        return None
    matches = []
    for raw in observations:
        if (
            not isinstance(raw, Mapping)
            or set(raw) != {
                "company_index", "company_domain", "source_url",
                "first_observed_date",
            }
            or type(raw.get("company_index")) is not int
            or not isinstance(raw.get("company_domain"), str)
            or not isinstance(raw.get("source_url"), str)
            or not isinstance(raw.get("first_observed_date"), str)
        ):
            continue
        try:
            parsed_date = date.fromisoformat(raw["first_observed_date"])
            parsed_url = urlsplit(raw["source_url"])
        except ValueError:
            continue
        if (
            not 0 <= raw["company_index"] < _MAX_COMPANIES
            or not raw["company_domain"]
            or len(raw["company_domain"]) > 253
            or parsed_url.scheme != "https"
            or not parsed_url.hostname
            or parsed_date.isoformat() != raw["first_observed_date"]
        ):
            continue
        if raw["company_index"] == company_index:
            matches.append(dict(raw))
    return matches[0] if len(matches) == 1 else None
