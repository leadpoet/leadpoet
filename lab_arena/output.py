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


def validate_companies(companies: Any) -> List[Dict[str, Any]]:
    """Validate companies with the shared public competition model."""

    try:
        return validate_public_companies(companies, max_companies=MAX_COMPANIES)
    except (TypeError, ValueError) as exc:
        raise OutputInvalid("companies fail the public output contract") from exc


def output_document_from_bytes(data: bytes) -> Dict[str, Any]:
    """Parse and validate the model's ``companies.json`` into the output document.

    Accepted shapes: a bare list of companies, or an object whose only keys
    are ``companies`` and optionally ``schema_version``.
    """

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
        if "schema_version" in parsed and parsed["schema_version"] != contracts.OUTPUT_DOCUMENT_SCHEMA_VERSION:
            raise OutputInvalid("unsupported output schema version")
        companies = parsed.get("companies")
    else:
        raise OutputInvalid("output must be a list or an object")
    validated = validate_companies(companies)
    return {"schema_version": contracts.OUTPUT_DOCUMENT_SCHEMA_VERSION, "companies": validated}


def validate_output_document(document: Any) -> Dict[str, Any]:
    """Validate an already-parsed output document (the service side of completion)."""

    if not isinstance(document, Mapping):
        raise OutputInvalid("output document must be an object")
    try:
        contracts.require_only_keys(document, ("schema_version", "companies"))
    except ArenaContractError as exc:
        raise OutputInvalid("output contains unsupported fields") from exc
    if document.get("schema_version") != contracts.OUTPUT_DOCUMENT_SCHEMA_VERSION:
        raise OutputInvalid("unsupported output schema version")
    try:
        contracts.check_strict_document(document, contracts.OUTPUT_LIMITS)
    except ArenaContractError as exc:
        raise OutputInvalid(str(exc)) from exc
    return {"schema_version": contracts.OUTPUT_DOCUMENT_SCHEMA_VERSION, "companies": validate_companies(document.get("companies"))}
