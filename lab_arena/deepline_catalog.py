"""Host-owned, round-frozen Deepline discovery and company research policy.

Catalog text is untrusted provider data. Only the deterministic policy below
can grant access. This module reads no credentials and executes no tools.
"""
from __future__ import annotations

import hashlib
import ipaddress
import json
import re
from typing import Any, Callable, Mapping
from urllib.parse import parse_qsl, urlsplit

from jsonschema import Draft202012Validator

SCHEMA_VERSION = "leadpoet.lab_arena.deepline_catalog.v1"
POLICY_VERSION = "leadpoet.lab_arena.deepline_company_research.v1"
CATALOG_URL = "https://code.deepline.com/api/v2/tools?compact=false"
_ID = re.compile(r"^[a-z][a-z0-9_]{0,79}$")
_CATEGORY = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_COMPANY_CATEGORIES = frozenset({"company_search", "company_enrich", "research", "smb"})
_PEOPLE_CATEGORIES = frozenset({"people_search", "people_enrich", "email_finder", "email_verify", "phone_finder", "phone_verify", "reverse_lookup"})
_DENIED_CATEGORIES = frozenset({"outbound_tools", "admin", "automation"})
_WRITE_WORDS = frozenset({"create", "update", "delete", "remove", "send", "publish", "upload", "insert", "upsert", "subscribe", "unsubscribe", "schedule", "deploy", "connect", "disconnect", "invite", "campaign", "sequence", "webhook"})
_PRIVATE_PROVIDERS = frozenset({"affinity", "attio", "hubspot", "salesforce", "pipedrive", "zoho_crm", "customer_db", "snowflake", "postgres", "bigquery", "clickhouse", "databricks", "redshift", "gong", "fireflies", "grain", "attention", "intercom", "outreach", "slack", "google_workspace", "google_ads_audiences", "linkedin_ads_audiences", "meta_audiences"})
_SECRET_FIELDS = frozenset({"authorization", "auth", "api_key", "apikey", "access_token", "token", "secret", "password", "cookie", "cookies", "credentials", "headers", "proxy", "proxies", "webhook", "callback_url", "script", "code", "command", "sql", "workflow", "actions", "tools", "actor_id", "actor_input", "dataset_id"})
_PERSON_FIELDS = frozenset({"person", "people", "persons", "contact", "contacts", "email", "emails", "phone", "phones", "phone_number", "first_name", "last_name", "full_name", "person_id", "contact_id", "profile_id", "public_identifier", "linkedin_handle"})
_PERSON_FLAGS = frozenset({"find_email", "include_emails", "include_email", "include_phones", "include_phone", "include_contacts", "include_people", "enrich_people", "enrich_contacts"})


class CatalogError(ValueError):
    """Bounded error; provider metadata and request values are never echoed."""


def _copy(value: Any) -> Any:
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
        if len(encoded) > 16_777_216:
            raise CatalogError("catalog_too_large")
        return json.loads(encoded)
    except (TypeError, ValueError, RecursionError) as exc:
        raise CatalogError("invalid_catalog") from exc


def _name(name: str) -> str:
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name).lower().replace("-", "_")


def _words(name: str) -> set[str]:
    return set(_name(name).split("_"))


def _schema(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise CatalogError("missing_schema")
    if isinstance(raw.get("jsonSchema"), dict):
        schema = raw["jsonSchema"]
    elif isinstance(raw.get("fields"), list):
        properties = {}
        required = []
        for field in raw["fields"]:
            if not isinstance(field, dict) or not isinstance(field.get("name"), str):
                raise CatalogError("invalid_schema")
            name = field["name"]
            if name in properties:
                raise CatalogError("invalid_schema")
            properties[name] = {k: v for k, v in field.items() if k not in {"name", "required"}}
            if field.get("required") is True:
                required.append(name)
        schema = {"type": "object", "properties": properties, "required": required}
    else:
        schema = raw
    schema = _copy(schema)
    if schema.get("type") != "object" or not isinstance(schema.get("properties"), dict):
        raise CatalogError("invalid_schema")
    # Never permit a remote resolver to turn validation into network access.
    def inspect(node: Any) -> None:
        if isinstance(node, list):
            for child in node:
                inspect(child)
        elif isinstance(node, dict):
            for key, child in node.items():
                if key in {"$dynamicRef", "$recursiveRef"} or key == "$ref" and (not isinstance(child, str) or not child.startswith("#/")):
                    raise CatalogError("unsupported_schema_reference")
                if key == "$id" and isinstance(child, str) and ":" in child:
                    raise CatalogError("unsupported_schema_reference")
                inspect(child)
    inspect(schema)
    try:
        Draft202012Validator.check_schema(schema)
    except Exception as exc:
        raise CatalogError("invalid_schema") from exc
    return schema


def _rate(value: Any) -> Any:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1_000_000:
        return None
    return value


def _pricing(row: Mapping[str, Any]) -> dict[str, Any]:
    pricing = row.get("pricing") if isinstance(row.get("pricing"), dict) else {}
    return {
        "unit": pricing.get("unit") if pricing.get("unit") in {"call", "request", "result", "page", "usage"} else None,
        "usd_per_unit": _rate(pricing.get("usdPerUnit", pricing.get("usd_per_unit", row.get("deeplineUsdPerPricingUnit")))),
        "credits_per_unit": _rate(pricing.get("creditsPerUnit", pricing.get("credits_per_unit", row.get("deeplineCreditsPerPricingUnit")))),
        "currency": pricing.get("currency", "USD"),
        "billing_source": row.get("billingSource", row.get("billing_source")),
    }


def _eligible(row: Mapping[str, Any], allow_people: bool) -> bool:
    categories = row.get("categories")
    tool = row.get("toolId", row.get("id"))
    provider = row.get("provider")
    if not isinstance(tool, str) or not _ID.fullmatch(tool) or not isinstance(provider, str) or not _ID.fullmatch(provider):
        return False
    if not isinstance(categories, list) or not categories or any(not isinstance(c, str) or not _CATEGORY.fullmatch(c) for c in categories):
        return False
    category_set = set(categories)
    if category_set & _DENIED_CATEGORIES or not allow_people and category_set & _PEOPLE_CATEGORIES:
        return False
    if not category_set & (_COMPANY_CATEGORIES | (_PEOPLE_CATEGORIES if allow_people else frozenset())):
        return False
    operation_names = [tool, row.get("operation"), row.get("operationId")]
    execution = row.get("executionMetadata", row.get("execution_metadata"))
    if isinstance(execution, dict):
        if execution.get("provider", provider) != provider:
            return False
        operation_names.extend([execution.get("sourceOperation"), execution.get("effectiveOperation")])
    if provider in _PRIVATE_PROVIDERS or any(isinstance(name, str) and _words(name) & _WRITE_WORDS for name in operation_names):
        return False
    if row.get("callable") is False or row.get("connected") is False or row.get("deprecated") is True or row.get("requiresOwnCredential") is True or row.get("credentialStatus") in {"requires_connection", "deprecated"}:
        return False
    if row.get("playReference") or row.get("playExpansion") or row.get("asyncFlow") or row.get("asyncGetAction"):
        return False
    return True


def freeze_catalog(document: Mapping[str, Any], allow_people: bool = False) -> dict[str, Any]:
    """Freeze all available tools approved by the versioned round policy."""
    document = _copy(document)
    rows = document.get("tools")
    if not isinstance(rows, list) or not rows or len(rows) > 4096 or type(allow_people) is not bool:
        raise CatalogError("invalid_catalog")
    seen = set()
    tools = []
    for row in rows:
        if not isinstance(row, dict):
            raise CatalogError("invalid_catalog")
        tool_id = row.get("toolId", row.get("id"))
        if not isinstance(tool_id, str) or tool_id in seen:
            raise CatalogError("duplicate_catalog_tool")
        seen.add(tool_id)
        if not _eligible(row, allow_people):
            continue
        try:
            schema = _schema(row.get("inputSchema", row.get("input_schema")))
        except CatalogError:
            continue
        required = schema.get("required", [])
        if any(_name(name) in _SECRET_FIELDS or not allow_people and _name(name) in _PERSON_FIELDS for name in required):
            continue
        pricing = _pricing(row)
        if pricing["currency"] != "USD":
            continue
        aliases = row.get("operationAliases", [])
        if not isinstance(aliases, list) or any(not isinstance(a, str) or not _ID.fullmatch(a) for a in aliases):
            continue
        tools.append({
            "tool_id": tool_id, "provider": row["provider"],
            "operation_aliases": sorted(set(aliases)),
            "categories": sorted(set(row["categories"])),
            "description": str(row.get("description", ""))[:16_384],
            "input_schema": schema, "pricing": pricing,
        })
    snapshot = {"schema_version": SCHEMA_VERSION, "policy_version": POLICY_VERSION, "allow_people": allow_people, "tools": sorted(tools, key=lambda row: row["tool_id"])}
    if not tools:
        raise CatalogError("empty_approved_catalog")
    snapshot["catalog_hash"] = _hash(snapshot)
    return snapshot


def _hash(document: Mapping[str, Any]) -> str:
    value = {k: v for k, v in document.items() if k != "catalog_hash"}
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def validate_catalog(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """Verify hash and rebuild authority; a valid hash cannot grant new policy."""
    snapshot = _copy(snapshot)
    if set(snapshot) != {"schema_version", "policy_version", "allow_people", "tools", "catalog_hash"} or snapshot.get("schema_version") != SCHEMA_VERSION or snapshot.get("policy_version") != POLICY_VERSION or snapshot.get("catalog_hash") != _hash(snapshot):
        raise CatalogError("invalid_catalog_snapshot")
    try:
        rows = []
        for row in snapshot.get("tools", []):
            rows.append({"toolId": row["tool_id"], "provider": row["provider"], "operationAliases": row["operation_aliases"], "categories": row["categories"], "description": row["description"], "inputSchema": row["input_schema"], "pricing": {"unit": row["pricing"]["unit"], "usdPerUnit": row["pricing"]["usd_per_unit"], "creditsPerUnit": row["pricing"]["credits_per_unit"], "currency": row["pricing"]["currency"]}, "billingSource": row["pricing"]["billing_source"]})
        rebuilt = freeze_catalog({"tools": rows}, snapshot["allow_people"])
    except (CatalogError, KeyError, TypeError) as exc:
        raise CatalogError("invalid_catalog_snapshot") from exc
    if rebuilt != snapshot:
        raise CatalogError("invalid_catalog_snapshot")
    return snapshot


def tool_entry(snapshot: Mapping[str, Any], tool_id: str) -> dict[str, Any]:
    snapshot = validate_catalog(snapshot)
    for row in snapshot["tools"]:
        if row["tool_id"] == tool_id:
            return row
    raise CatalogError("tool_not_approved")


def allowed_tool_ids(snapshot: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(row["tool_id"] for row in validate_catalog(snapshot)["tools"])


def _public_url(value: Any) -> None:
    if not isinstance(value, str) or not value.isascii() or len(value) > 2000 or any(c.isspace() for c in value) or "\\" in value:
        raise CatalogError("invalid_public_url")
    try:
        parts = urlsplit(value)
        host = parts.hostname
        if parts.scheme != "https" or not host or parts.username or parts.password or parts.fragment or parts.port not in {None, 443}:
            raise CatalogError("invalid_public_url")
        try:
            ipaddress.ip_address(host)
        except ValueError:
            pass
        else:
            raise CatalogError("invalid_public_url")
        if "." not in host or host.endswith((".localhost", ".local", ".internal", ".test", ".invalid")) or host in {"localhost", "metadata.google.internal"} or not re.fullmatch(r"[a-zA-Z0-9.-]+", host) or host.rsplit(".", 1)[-1].isdigit():
            raise CatalogError("invalid_public_url")
        if any(_name(key) in _SECRET_FIELDS for key, _ in parse_qsl(parts.query)):
            raise CatalogError("invalid_public_url")
    except ValueError as exc:
        raise CatalogError("invalid_public_url") from exc


def validate_payload(snapshot: Mapping[str, Any], tool_id: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a request against its frozen schema and company-only policy."""
    entry = tool_entry(snapshot, tool_id)
    payload = _copy(payload)
    if not isinstance(payload, dict):
        raise CatalogError("invalid_payload")
    schema = entry["input_schema"]
    if set(payload) - set(schema["properties"]):
        raise CatalogError("unknown_payload_field")
    try:
        if not Draft202012Validator(schema).is_valid(payload):
            raise CatalogError("invalid_payload_schema")
    except CatalogError:
        raise
    except Exception as exc:
        raise CatalogError("invalid_payload_schema") from exc
    def inspect(value: Any) -> None:
        if isinstance(value, list):
            for child in value:
                inspect(child)
        elif isinstance(value, dict):
            for key, child in value.items():
                name = _name(key)
                if name in _SECRET_FIELDS or _words(name) & _WRITE_WORDS:
                    raise CatalogError("forbidden_payload_field")
                inactive_flag = child is False or child is None or isinstance(child, str) and child in {"false", "False"}
                if not snapshot["allow_people"] and (name in _PERSON_FIELDS or name in _PERSON_FLAGS and not inactive_flag):
                    raise CatalogError("people_payload_forbidden")
                if name in {"method", "http_method"} and child != "GET":
                    raise CatalogError("write_payload_forbidden")
                if name in {"body", "data", "json"} and entry["provider"] == "generic_http":
                    raise CatalogError("write_payload_forbidden")
                if name in {"category", "entity_type", "type", "target_type"} and isinstance(child, str) and _name(child) in {"person", "people", "contact", "contacts", "email", "phone"} and not snapshot["allow_people"]:
                    raise CatalogError("people_payload_forbidden")
                if isinstance(child, str) and child.startswith(("http://", "https://")):
                    _public_url(child)
                    if not snapshot["allow_people"] and urlsplit(child).hostname in {"linkedin.com", "www.linkedin.com"} and urlsplit(child).path.startswith("/in/"):
                        raise CatalogError("people_payload_forbidden")
                inspect(child)
    inspect(payload)
    return payload


def fetch_catalog(*, read_json: Callable[[str, Mapping[str, str]], Mapping[str, Any]], allow_people: bool = False) -> dict[str, Any]:
    """Read only official catalog routes with a caller-owned trusted transport."""
    document = _copy(read_json(CATALOG_URL, {}))
    rows = document.get("tools")
    if not isinstance(rows, list) or len(rows) > 4096:
        raise CatalogError("invalid_catalog")
    detailed = []
    for row in rows:
        if not isinstance(row, dict):
            raise CatalogError("invalid_catalog")
        if _eligible(row, allow_people):
            tool_id = row.get("toolId", row.get("id"))
            detail = _copy(read_json("https://code.deepline.com/api/v2/integrations/%s/get" % tool_id, {"x-deepline-tool-meta-only": "1"}))
            if detail.get("toolId", detail.get("id")) != tool_id or detail.get("provider") != row.get("provider"):
                raise CatalogError("catalog_identity_changed")
            row = detail
        detailed.append(row)
    return freeze_catalog({"tools": detailed}, allow_people)
