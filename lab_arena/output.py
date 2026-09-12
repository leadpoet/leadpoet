"""Public sourcing-competition output contract.

A model writes ``/output/companies.json`` with the same company structure used
by the daily public baseline. The validated document is stored by its
service-owned run.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Sequence

from lab_arena import contracts
from lab_arena.contracts import ArenaContractError
from qualification.competition_models import validate_companies as validate_public_companies

MAX_OUTPUT_BYTES = 512 * 1024
MAX_COMPANIES = 5
SUPPORTED_OUTPUT_SCHEMA_VERSIONS = frozenset(
    {
        contracts.OUTPUT_DOCUMENT_SCHEMA_VERSION,
        contracts.CONTACT_OUTPUT_DOCUMENT_SCHEMA_VERSION,
        "leadpoet.lab_arena.output.v3",
        "leadpoet.lab_arena.output.v4",
    }
)


class OutputInvalid(ArenaContractError):
    """The model output violates the contract; the ICP scores zero."""


def _reject_constant(value: str) -> Any:
    raise ValueError("non-finite JSON constant %s" % value)


def parse_output_bytes(data: bytes) -> Any:
    if not isinstance(data, (bytes, bytearray)) or len(data) > MAX_OUTPUT_BYTES:
        raise OutputInvalid("output is missing or exceeds %d bytes" % MAX_OUTPUT_BYTES)
    try:
        return json.loads(bytes(data).decode("utf-8"), parse_constant=_reject_constant)
    except (UnicodeDecodeError, ValueError) as exc:
        raise OutputInvalid("output is not valid JSON") from exc


def validate_companies(
    companies: Any,
    *,
    require_intent_dates: bool = False,
    schema_version: str = contracts.OUTPUT_DOCUMENT_SCHEMA_VERSION,
) -> List[Dict[str, Any]]:
    """Validate companies with the shared public competition model."""

    try:
        rows = validate_public_companies(
            companies,
            max_companies=MAX_COMPANIES,
            schema_version=schema_version,
        )
    except (TypeError, ValueError) as exc:
        raise OutputInvalid("companies fail the public output contract") from exc
    if require_intent_dates and any(
        signal.get("date") is None
        for company in rows
        for signal in company["intent_signals"]
    ):
        raise OutputInvalid("intent signal date is required")
    return rows


def output_document_from_bytes(
    data: bytes,
    *,
    require_intent_dates: bool = False,
    expected_schema_version: str = contracts.OUTPUT_DOCUMENT_SCHEMA_VERSION,
) -> Dict[str, Any]:
    """Parse and validate the model's ``companies.json`` into the output document.

    Accepted shapes: a bare list of companies, or an object whose only keys
    are ``companies`` and optionally ``schema_version``.
    """

    if expected_schema_version not in SUPPORTED_OUTPUT_SCHEMA_VERSIONS:
        raise OutputInvalid("unsupported output schema version")
    parsed = parse_output_bytes(data)
    try:
        contracts.check_strict_document(parsed, contracts.OUTPUT_LIMITS)
    except ArenaContractError as exc:
        raise OutputInvalid(str(exc)) from exc
    if isinstance(parsed, list):
        companies = parsed
    elif isinstance(parsed, Mapping):
        try:
            contracts.require_only_keys(parsed, ("schema_version", "companies"))
        except ArenaContractError as exc:
            raise OutputInvalid("output contains unsupported fields") from exc
        if (
            "schema_version" in parsed
            and parsed["schema_version"] != expected_schema_version
        ):
            raise OutputInvalid("unsupported output schema version")
        companies = parsed.get("companies")
    else:
        raise OutputInvalid("output must be a list or an object")
    validated = validate_companies(
        companies,
        require_intent_dates=require_intent_dates,
        schema_version=expected_schema_version,
    )
    return {"schema_version": expected_schema_version, "companies": validated}


def validate_output_document(
    document: Any,
    *,
    require_intent_dates: bool = False,
    expected_schema_version: str | None = None,
) -> Dict[str, Any]:
    """Validate a stored output, optionally requiring its assigned schema.

    ``None`` infers only from the document's declared, known version. Model
    output parsing uses :func:`output_document_from_bytes`, whose default stays
    pinned to v1 for backwards compatibility.
    """

    if not isinstance(document, Mapping):
        raise OutputInvalid("output document must be an object")
    try:
        contracts.require_only_keys(document, ("schema_version", "companies"))
    except ArenaContractError as exc:
        raise OutputInvalid("output contains unsupported fields") from exc
    declared_schema_version = document.get("schema_version")
    if declared_schema_version not in SUPPORTED_OUTPUT_SCHEMA_VERSIONS:
        raise OutputInvalid("unsupported output schema version")
    if (
        expected_schema_version is not None
        and declared_schema_version != expected_schema_version
    ):
        raise OutputInvalid("unsupported output schema version")
    try:
        contracts.check_strict_document(document, contracts.OUTPUT_LIMITS)
    except ArenaContractError as exc:
        raise OutputInvalid(str(exc)) from exc
    return {
        "schema_version": declared_schema_version,
        "companies": validate_companies(
            document.get("companies"),
            require_intent_dates=require_intent_dates,
            schema_version=declared_schema_version,
        ),
    }
