"""Model output contract (labarena.md section 6.1).

A model writes ``/output/companies.json`` holding only the current
``CompanyOutput`` structure. The Arena validates the bytes against its strict
limits and every company against ``CompanyOutput`` (unknown fields rejected by
the model itself); the resulting output document is what receipts, scoring
plans, and public bundles hash.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Sequence

from lab_arena import contracts
from lab_arena.contracts import ArenaContractError

MAX_OUTPUT_BYTES = 512 * 1024
MAX_COMPANIES = 50


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
    """Validate each company with the current ``CompanyOutput`` model."""

    from gateway.qualification.models import CompanyOutput

    if not isinstance(companies, list):
        raise OutputInvalid("companies must be a list")
    if len(companies) > MAX_COMPANIES:
        raise OutputInvalid("too many companies")
    validated: List[Dict[str, Any]] = []
    for index, item in enumerate(companies):
        if not isinstance(item, Mapping):
            raise OutputInvalid("company %d is not an object" % index)
        try:
            model = CompanyOutput(**dict(item))
        except Exception as exc:  # pydantic ValidationError or the models' ValueError guards
            raise OutputInvalid("company %d fails the CompanyOutput contract: %s" % (index, type(exc).__name__)) from exc
        validated.append(model.model_dump(mode="json"))
    return validated


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
        contracts.require_only_keys(parsed, ("schema_version", "companies"))
        if "schema_version" in parsed and parsed["schema_version"] != contracts.OUTPUT_DOCUMENT_SCHEMA_VERSION:
            raise OutputInvalid("unsupported output schema version")
        companies = parsed.get("companies")
    else:
        raise OutputInvalid("output must be a list or an object")
    validated = validate_companies(companies)
    return {"schema_version": contracts.OUTPUT_DOCUMENT_SCHEMA_VERSION, "companies": validated}


def output_document_hash(document: Mapping[str, Any]) -> str:
    return contracts.document_hash(dict(document))


def validate_output_document(document: Any) -> Dict[str, Any]:
    """Validate an already-parsed output document (the service side of completion)."""

    if not isinstance(document, Mapping):
        raise OutputInvalid("output document must be an object")
    contracts.require_only_keys(document, ("schema_version", "companies"))
    if document.get("schema_version") != contracts.OUTPUT_DOCUMENT_SCHEMA_VERSION:
        raise OutputInvalid("unsupported output schema version")
    try:
        contracts.check_strict_document(document, contracts.OUTPUT_LIMITS)
    except ArenaContractError as exc:
        raise OutputInvalid(str(exc)) from exc
    return {"schema_version": contracts.OUTPUT_DOCUMENT_SCHEMA_VERSION, "companies": validate_companies(document.get("companies"))}
