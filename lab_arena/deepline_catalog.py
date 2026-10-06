"""Host-owned, round-frozen Deepline discovery and company research policy.

Catalog text is untrusted provider data. Only the deterministic policy below
can grant access. This module reads no credentials and executes no tools.
"""
from __future__ import annotations

import hashlib
import ipaddress
import json
import re
from functools import lru_cache
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
_DENIED_CATEGORIES = frozenset({"outbound_tools", "admin"})
_MODIFIER_CATEGORIES = frozenset({"autocomplete", "identity_resolution", "enrichment", "batch", "premium", "free"})
_WRITE_WORDS = frozenset({"create", "update", "delete", "remove", "send", "publish", "upload", "insert", "upsert", "subscribe", "unsubscribe", "schedule", "deploy", "connect", "disconnect", "invite", "campaign", "sequence", "webhook"})
_PRIVATE_PROVIDERS = frozenset({"affinity", "attio", "hubspot", "salesforce", "pipedrive", "zoho_crm", "customer_db", "snowflake", "postgres", "bigquery", "clickhouse", "databricks", "redshift", "gong", "fireflies", "grain", "attention", "intercom", "outreach", "slack", "google_workspace", "google_ads_audiences", "linkedin_ads_audiences", "meta_audiences", "lemlist", "instantly", "emailbison", "heyreach", "smartlead", "salesloft", "salesforge", "kernel", "browserbase", "clay", "apify"})
_SECRET_FIELDS = frozenset({"authorization", "auth", "api_key", "apikey", "access_token", "token", "secret", "password", "cookie", "cookies", "credentials", "headers", "proxy", "proxies", "webhook", "callback_url", "script", "code", "command", "workflow", "actions", "tools", "actor_id", "actor_input", "dataset_id"})
_PERSON_FIELDS = frozenset({"person", "people", "persons", "contact", "contacts", "email", "emails", "phone", "phones", "phone_number", "first_name", "last_name", "full_name", "person_id", "contact_id", "profile", "profiles", "profile_id", "profile_url", "public_identifier", "linkedin_handle", "person_url", "member_id", "mentioning_member", "user_id", "user_ids", "username", "screen_name"})
_PERSON_FLAGS = frozenset({"find_email", "include_emails", "include_email", "include_phones", "include_phone", "include_contacts", "include_people", "enrich_people", "enrich_contacts"})
# Provider categories can label a people roster as company research. Judge the
# operation target too, including aliases; aggregate headcount remains useful.
_PEOPLE_TARGET_WORDS = frozenset({"person", "people", "profile", "contact", "contacts", "email", "phone", "employee", "employees", "alumni", "alumnis", "reposter", "reposters", "reactor", "reactors", "follower", "followers", "following", "connection", "connections", "member", "members", "officer", "officers", "director", "directors", "shareholder", "shareholders", "founder", "founders", "user", "users"})
_PERSON_INPUT_FIELDS = (_PERSON_FIELDS - {"profile", "profiles", "profile_url", "public_identifier"}) | frozenset({"linkedin_profile_url", "linkedin_profile_id", "linkedin_profile_handle", "sales_navigator_profile_url", "sales_navigator_profile_id"})
_COMPANY_SQL_FUNCTIONS = frozenset({"COUNT", "SUM", "MIN", "MAX", "AVG", "LOWER", "UPPER", "LENGTH", "CHAR_LENGTH", "TRIM", "LTRIM", "RTRIM", "COALESCE", "NULLIF", "ROUND", "ABS", "CEIL", "CEILING", "FLOOR", "SUBSTRING", "SUBSTR", "REPLACE", "CONCAT", "CAST"})


class CatalogError(ValueError):
    """Bounded error; provider metadata and request values are never echoed."""


def _encoded(value: Any) -> str:
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
        if len(encoded) > 67_108_864:
            raise CatalogError("catalog_too_large")
        return encoded
    except CatalogError:
        raise
    except (TypeError, ValueError, RecursionError) as exc:
        raise CatalogError("invalid_catalog") from exc


def _copy(value: Any) -> Any:
    return json.loads(_encoded(value))


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
    # Variable-rate tools describe their rate in text. Freeze these hints too;
    # they guide research but are never treated as confirmed billing amounts.
    hints = {}
    for key in ("displayText", "summary"):
        if key in pricing:
            value = pricing[key]
            if value is not None and (not isinstance(value, str) or len(value) > 16_384):
                raise CatalogError("invalid_pricing_hint")
            hints[key] = value
    if "details" in pricing:
        details = pricing["details"]
        if not isinstance(details, list) or len(details) > 64 or any(not isinstance(value, str) or len(value) > 16_384 for value in details):
            raise CatalogError("invalid_pricing_hint")
        hints["details"] = _copy(details)
    return {
        "unit": pricing.get("unit") if pricing.get("unit") in {"call", "request", "result", "page", "usage"} else None,
        "usd_per_unit": _rate(pricing.get("usdPerUnit", pricing.get("usd_per_unit", row.get("deeplineUsdPerPricingUnit")))),
        "credits_per_unit": _rate(pricing.get("creditsPerUnit", pricing.get("credits_per_unit", row.get("deeplineCreditsPerPricingUnit")))),
        "currency": pricing.get("currency", "USD"),
        "billing_source": row.get("billingSource", row.get("billing_source")),
        **hints,
    }


def _aggregate_company_output(row: Mapping[str, Any]) -> bool:
    """An aggregate name is insufficient if its result contains people."""
    try:
        schema = _schema(row.get("outputSchema", row.get("output_schema")))
    except CatalogError:
        return False
    names = set()
    def inspect(node: Any) -> None:
        if isinstance(node, list):
            for child in node:
                inspect(child)
        elif isinstance(node, dict):
            properties = node.get("properties", {})
            if isinstance(properties, dict):
                names.update(_name(name) for name in properties)
            for child in node.values():
                inspect(child)
    inspect(schema)
    return bool(names & {"count", "employee_count", "headcount"}) and not names & _PERSON_INPUT_FIELDS


def _eligible(row: Mapping[str, Any], allow_people: bool, *, owned_poll: bool = False) -> bool:
    categories = row.get("categories")
    tool = row.get("toolId", row.get("id"))
    provider = row.get("provider")
    if not isinstance(tool, str) or not _ID.fullmatch(tool) or not isinstance(provider, str) or not _ID.fullmatch(provider):
        return False
    if not isinstance(categories, list) or not categories or any(not isinstance(c, str) or not _CATEGORY.fullmatch(c) for c in categories):
        return False
    category_set = set(categories)
    # A generic HTTP operation is narrowed locally to public HTTPS GET. The
    # provider's admin label describes its broader native request surface.
    public_http = provider == "generic_http" and _words(tool) >= {"http", "request"}
    if category_set - (_COMPANY_CATEGORIES | _PEOPLE_CATEGORIES | _DENIED_CATEGORIES | _MODIFIER_CATEGORIES | {"automation"}):
        return False
    if category_set & _DENIED_CATEGORIES and not public_http and not owned_poll or not allow_people and category_set & _PEOPLE_CATEGORIES:
        return False
    if not public_http and not owned_poll and not category_set & (_COMPANY_CATEGORIES | (_PEOPLE_CATEGORIES if allow_people else frozenset())):
        # Automation also describes ordinary public page readers. Their
        # required URL and read-only operation identify the narrower surface.
        raw_schema = row.get("inputSchema", {})
        raw_schema = raw_schema.get("jsonSchema", raw_schema) if isinstance(raw_schema, dict) else {}
        required = raw_schema.get("required", [])
        if not isinstance(required, list) or any(not isinstance(name, str) for name in required) or "automation" not in category_set or not set(required) & {"url", "urls"} or not _words(tool) & {"scrape", "fetch", "extract", "read", "get", "crawl", "map"}:
            return False
    operation_names = [tool, row.get("operation"), row.get("operationId")]
    aliases = row.get("operationAliases", [])
    if isinstance(aliases, list):
        operation_names.extend(aliases)
    execution = row.get("executionMetadata", row.get("execution_metadata"))
    if isinstance(execution, dict):
        if execution.get("provider", provider) != provider:
            return False
        operation_names.extend([execution.get("sourceOperation"), execution.get("effectiveOperation")])
    if provider in _PRIVATE_PROVIDERS or any(isinstance(name, str) and _words(name) & _WRITE_WORDS for name in operation_names):
        return False
    if not allow_people:
        for name in operation_names:
            if not isinstance(name, str):
                continue
            words = _words(name)
            targets = words & _PEOPLE_TARGET_WORDS
            aggregate_headcount = (targets <= {"employee", "employees"}
                                   and "company" in words
                                   and bool(words & {"count", "counts", "headcount", "insights", "metrics", "statistics", "distribution", "distributions"})
                                   and _aggregate_company_output(row))
            if targets and not aggregate_headcount:
                return False
    if row.get("callable") is False or row.get("connected") is False or row.get("deprecated") is True or row.get("requiresOwnCredential") is True or row.get("credentialStatus") in {"requires_connection", "deprecated"}:
        return False
    if row.get("playReference") or row.get("playExpansion"):
        return False
    return True


def _requires_person_input(schema: Any) -> bool:
    """Do not advertise tools whose required input identifies a person."""
    if isinstance(schema, list):
        return any(_requires_person_input(child) for child in schema)
    if not isinstance(schema, dict):
        return False
    required = schema.get("required", [])
    return (isinstance(required, list) and any(isinstance(name, str) and _name(name) in _PERSON_INPUT_FIELDS for name in required)
            or any(_requires_person_input(child) for child in schema.values()))


def _company_sql(value: Any) -> None:
    """Restrict public corpus queries to a single SELECT over companies."""
    if not isinstance(value, str) or len(value) > 32_000 or any(s in value for s in (";", "--", "/*", "*/", "\\", "$")):
        raise CatalogError("invalid_public_company_sql")
    # Ignore string literals when checking SQL keywords and relation names.
    statement = re.sub(r"'(?:[^']|'')*'", "''", value)
    if "'" in statement.replace("''", "") or '"' in statement or not re.match(r"^\s*SELECT\b", statement, re.I):
        raise CatalogError("invalid_public_company_sql")
    words = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", statement.upper()))
    if words & {"INSERT", "UPDATE", "DELETE", "MERGE", "DROP", "ALTER", "CREATE", "COPY", "CALL", "EXECUTE", "GRANT", "REVOKE", "INTO", "UNION", "JOIN", "WITH", "EXPLAIN", "INFORMATION_SCHEMA"}:
        raise CatalogError("invalid_public_company_sql")
    tables = re.findall(r"\bFROM\s+([\w.]+)", statement, re.I)
    if not tables or any(table.lower() != "companies" for table in tables) or re.search(r"\bFROM\s+companies\b(?!\s*(?:WHERE\b|GROUP\b|HAVING\b|ORDER\b|LIMIT\b|OFFSET\b|\)|$))", statement, re.I):
        raise CatalogError("invalid_public_company_sql")
    # SELECT is not itself a read-only boundary: SQL functions can read local
    # files, reach other databases, or perform writes. Permit only common
    # scalar/aggregate company filters, with no schema-qualified functions.
    functions = re.findall(r"\b([A-Za-z_][A-Za-z0-9_.]*)\s*\(", statement)
    if any(name.upper() not in _COMPANY_SQL_FUNCTIONS | {"IN"} for name in functions):
        raise CatalogError("invalid_public_company_sql")
    limits = re.findall(r"\bLIMIT\s+(\d+)\b", statement, re.I)
    if not limits or any(not 1 <= int(limit) <= 100_000 for limit in limits):
        raise CatalogError("invalid_public_company_sql")


def freeze_catalog(document: Mapping[str, Any], allow_people: bool = False) -> dict[str, Any]:
    """Freeze all available tools approved by the versioned round policy."""
    document = _copy(document)
    if not isinstance(document, dict):
        raise CatalogError("invalid_catalog")
    rows = document.get("tools")
    if not isinstance(rows, list) or not rows or len(rows) > 4096 or type(allow_people) is not bool:
        raise CatalogError("invalid_catalog")
    seen = set()
    tools = []
    by_id = {row.get("toolId", row.get("id")): row for row in rows if isinstance(row, dict) and isinstance(row.get("toolId", row.get("id")), str)}
    flows = {}
    polls = {}
    for row in rows:
        if not isinstance(row, dict) or not _eligible(row, allow_people) or not row.get("asyncFlow"):
            continue
        flow = row["asyncFlow"]
        operation = row.get("asyncOperation")
        parent = row.get("toolId", row.get("id"))
        if not isinstance(flow, dict) or flow.get("startAction") != parent or flow.get("finishAction") or not isinstance(operation, dict):
            continue
        job = operation.get("job")
        ids = job.get("idPaths") if isinstance(job, dict) else None
        actions = flow.get("pollActions")
        if not isinstance(ids, list) or not ids or any(not isinstance(path, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*){0,5}", path) for path in ids) or not isinstance(actions, list) or len(actions) != 1 or not isinstance(actions[0], str) or not _ID.fullmatch(actions[0]):
            continue
        poll = by_id.get(actions[0])
        if not poll or poll.get("provider") != row.get("provider") or not _eligible(poll, allow_people, owned_poll=True) or not _words(actions[0]) & {"get", "status", "result", "results"}:
            continue
        try:
            required = _schema(poll.get("inputSchema", poll.get("input_schema"))).get("required", [])
        except CatalogError:
            continue
        if len(required) != 1 or _name(required[0]) not in {"id", "job_id", "run_id", "task_id", "request_id"} or actions[0] in polls:
            continue
        flows[parent] = {"start_action": parent, "poll_actions": actions, "job_id_paths": ids, "poll_input": required[0]}
        polls[actions[0]] = parent
    for row in rows:
        if not isinstance(row, dict):
            raise CatalogError("invalid_catalog")
        tool_id = row.get("toolId", row.get("id"))
        if not isinstance(tool_id, str) or tool_id in seen:
            raise CatalogError("duplicate_catalog_tool")
        seen.add(tool_id)
        parent = polls.get(tool_id)
        if not _eligible(row, allow_people, owned_poll=parent is not None):
            continue
        if (row.get("asyncFlow") or row.get("asyncGetAction")) and tool_id not in flows:
            continue
        try:
            schema = _schema(row.get("inputSchema", row.get("input_schema")))
        except CatalogError:
            continue
        required = schema.get("required", [])
        if not allow_people and _requires_person_input(schema):
            continue
        if not parent and any(_name(name) in {"job_id", "run_id", "task_id", "request_id"} for name in required):
            continue
        if not parent and "get" in _words(tool_id) and _words(tool_id) & {"run", "job", "status", "result", "results"}:
            continue
        if any(_name(name) in _SECRET_FIELDS or not allow_people and _name(name) in _PERSON_FIELDS for name in required):
            continue
        if "sql" in schema["properties"] and not (row["provider"] == "deepline_native" and "company_search" in row["categories"]):
            continue
        try:
            pricing = _pricing(row)
        except CatalogError:
            continue
        if pricing["currency"] != "USD":
            continue
        aliases = row.get("operationAliases", [])
        if not isinstance(aliases, list) or any(not isinstance(a, str) or not _ID.fullmatch(a) for a in aliases):
            continue
        output = row.get("outputSchema", row.get("output_schema"))
        try:
            output = _schema(output) if isinstance(output, dict) else None
        except CatalogError:
            output = None
        tools.append({
            "tool_id": tool_id, "provider": row["provider"],
            "operation_aliases": sorted(set(aliases)),
            "categories": sorted(set(row["categories"])),
            "description": str(row.get("description", ""))[:16_384],
            "input_schema": schema, "pricing": pricing,
            **({"output_schema": output} if output is not None else {}),
            **({"async_flow": flows[tool_id]} if tool_id in flows else {}),
            **({"async_parent": parent} if parent is not None else {}),
        })
    approved_ids = {tool["tool_id"] for tool in tools}
    tools = [tool for tool in tools if (not tool.get("async_parent") or tool["async_parent"] in approved_ids) and (not tool.get("async_flow") or set(tool["async_flow"]["poll_actions"]) <= approved_ids)]
    snapshot = {"schema_version": SCHEMA_VERSION, "policy_version": POLICY_VERSION, "allow_people": allow_people, "tools": sorted(tools, key=lambda row: row["tool_id"])}
    if not tools:
        raise CatalogError("empty_approved_catalog")
    snapshot["catalog_hash"] = _hash(snapshot)
    return snapshot


def _hash(document: Mapping[str, Any]) -> str:
    value = {k: v for k, v in document.items() if k != "catalog_hash"}
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()).hexdigest()


@lru_cache(maxsize=4)
def _validated(encoded: str) -> dict[str, Any]:
    """Cache only exact immutable bytes; never expose the cached object."""
    snapshot = json.loads(encoded)
    if not isinstance(snapshot, dict):
        raise CatalogError("invalid_catalog_snapshot")
    if set(snapshot) != {"schema_version", "policy_version", "allow_people", "tools", "catalog_hash"} or snapshot.get("schema_version") != SCHEMA_VERSION or snapshot.get("policy_version") != POLICY_VERSION or snapshot.get("catalog_hash") != _hash(snapshot):
        raise CatalogError("invalid_catalog_snapshot")
    try:
        rows = []
        for row in snapshot.get("tools", []):
            rows.append({"toolId": row["tool_id"], "provider": row["provider"], "operationAliases": row["operation_aliases"], "categories": row["categories"], "description": row["description"], "inputSchema": row["input_schema"], "pricing": _public_pricing(row["pricing"]), "billingSource": row["pricing"]["billing_source"]})
            if "async_flow" in row:
                flow = row["async_flow"]
                rows[-1]["asyncFlow"] = {"startAction": flow["start_action"], "pollActions": flow["poll_actions"], "finishAction": None}
                rows[-1]["asyncOperation"] = {"job": {"idPaths": flow["job_id_paths"]}}
            if "output_schema" in row:
                rows[-1]["outputSchema"] = row["output_schema"]
        rebuilt = freeze_catalog({"tools": rows}, snapshot["allow_people"])
    except (CatalogError, KeyError, TypeError) as exc:
        raise CatalogError("invalid_catalog_snapshot") from exc
    if rebuilt != snapshot:
        raise CatalogError("invalid_catalog_snapshot")
    return snapshot


def validate_catalog(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """Verify hash and authority once per exact snapshot; return a fresh copy."""
    return _copy(_validated(_encoded(snapshot)))


def tool_entry(snapshot: Mapping[str, Any], tool_id: str) -> dict[str, Any]:
    snapshot = _validated(_encoded(snapshot))
    for row in snapshot["tools"]:
        if row["tool_id"] == tool_id:
            return _copy(row)
    raise CatalogError("tool_not_approved")


def allowed_tool_ids(snapshot: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(row["tool_id"] for row in _validated(_encoded(snapshot))["tools"])


def _public_pricing(pricing: Mapping[str, Any]) -> dict[str, Any]:
    return {"unit": pricing["unit"], "usdPerUnit": pricing["usd_per_unit"], "creditsPerUnit": pricing["credits_per_unit"], "currency": pricing["currency"], **{key: _copy(pricing[key]) for key in ("displayText", "summary", "details") if key in pricing}}


def public_tool_definition(entry: Mapping[str, Any], *, compact: bool = False) -> dict[str, Any]:
    """Return the official CLI's public discovery shape, from frozen data."""
    entry = _copy(entry)
    result = {
        "toolId": entry["tool_id"], "provider": entry["provider"],
        "description": entry["description"], "categories": entry["categories"],
        "operationAliases": entry["operation_aliases"],
        "inputSchema": {"jsonSchema": entry["input_schema"]},
        "pricing": _public_pricing(entry["pricing"]),
        "billingSource": entry["pricing"]["billing_source"],
        "callable": True, "connected": True, "credentialStatus": "managed",
    }
    if entry.get("async_flow"):
        flow = entry["async_flow"]
        result["asyncFlow"] = {"startAction": flow["start_action"], "pollActions": flow["poll_actions"], "finishAction": None}
        result["asyncOperation"] = {"job": {"idPaths": flow["job_id_paths"]}}
    if "output_schema" in entry:
        result["outputSchema"] = {"jsonSchema": entry["output_schema"]}
    if compact:
        result.pop("inputSchema", None)
        result.pop("outputSchema", None)
        result["hasInputSchema"] = True
        result["hasOutputSchema"] = "output_schema" in entry
    return result


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
    if entry.get("async_parent") and "next" in payload:
        raise CatalogError("async_paging_forbidden")
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
                if name in {"job_id", "run_id", "task_id", "request_id"} and not entry.get("async_parent"):
                    raise CatalogError("unowned_job_payload")
                if name == "sql":
                    if entry["provider"] != "deepline_native" or "company_search" not in entry["categories"]:
                        raise CatalogError("forbidden_payload_field")
                    _company_sql(child)
                inactive_flag = child is False or child is None or isinstance(child, str) and child in {"false", "False"}
                if not snapshot["allow_people"] and (name in _PERSON_FIELDS or name in _PERSON_FLAGS and not inactive_flag):
                    raise CatalogError("people_payload_forbidden")
                if name in {"method", "http_method"} and child != "GET":
                    raise CatalogError("write_payload_forbidden")
                if entry["provider"] == "generic_http" and (name in {"body", "data", "json"} or name.startswith("body_")):
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
    if not isinstance(document, dict):
        raise CatalogError("invalid_catalog")
    rows = document.get("tools")
    if not isinstance(rows, list) or len(rows) > 4096:
        raise CatalogError("invalid_catalog")
    detailed = {}
    for row in rows:
        if not isinstance(row, dict):
            raise CatalogError("invalid_catalog")
        schema = row.get("inputSchema", row.get("input_schema"))
        hint = (str(row.get("toolId", "")).replace("_", " ") + " " + str(schema.get("description", "") if isinstance(schema, dict) else ""))
        needs_detail = not isinstance(schema, dict) or row.get("asyncFlow") or re.search(r"\b(?:async|asynchronous|batch|crawl|agent|run)\b", hint, re.I)
        if _eligible(row, allow_people) and needs_detail:
            tool_id = row.get("toolId", row.get("id"))
            detail = _copy(read_json("https://code.deepline.com/api/v2/integrations/%s/get" % tool_id, {"x-deepline-tool-meta-only": "1"}))
            if not isinstance(detail, dict) or detail.get("toolId", detail.get("id")) != tool_id or detail.get("provider") != row.get("provider"):
                raise CatalogError("catalog_identity_changed")
            row = detail
        detailed[row.get("toolId", row.get("id"))] = row
    # Declared polls are metadata reads too. Their authority is granted only
    # by the safe parent descriptor, never by their broad admin category.
    for row in list(detailed.values()):
        flow = row.get("asyncFlow")
        if not isinstance(flow, dict) or not isinstance(flow.get("pollActions"), list):
            continue
        for tool_id in flow["pollActions"]:
            if not isinstance(tool_id, str) or not _ID.fullmatch(tool_id):
                raise CatalogError("invalid_async_descriptor")
            detail = _copy(read_json("https://code.deepline.com/api/v2/integrations/%s/get" % tool_id, {"x-deepline-tool-meta-only": "1"}))
            if not isinstance(detail, dict) or detail.get("toolId", detail.get("id")) != tool_id or detail.get("provider") != row.get("provider"):
                raise CatalogError("catalog_identity_changed")
            detailed[tool_id] = detail
    return freeze_catalog({"tools": list(detailed.values())}, allow_people)
