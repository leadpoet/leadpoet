"""Typed provider-probe catalog for the loop's `probe_provider` operation (W4).

Loops never send free-form URLs to providers: a probe names a catalog entry
(`endpoint_id`) plus schema-validated params, and the gateway resolves it
through the evidence proxy (baseline tape → day cache → live). The catalog is
the whole reachable surface — anything not listed here cannot be probed.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from research_lab.canonical import sha256_json

PROBE_CATALOG_PATH_ENV = "RESEARCH_LAB_PROBE_CATALOG_PATH"

_VALID_METHODS = ("GET", "POST")
_VALID_PARAM_TYPES = ("string", "integer", "number", "boolean")
_MAX_PARAMS_PER_ENDPOINT = 12
_MAX_PARAM_VALUE_CHARS = 400


@dataclass(frozen=True)
class ProbeParamSpec:
    name: str
    type: str = "string"
    required: bool = False
    location: str = "query"  # query | body
    max_length: int = _MAX_PARAM_VALUE_CHARS

    @classmethod
    def from_mapping(cls, name: str, data: Mapping[str, Any]) -> "ProbeParamSpec":
        return cls(
            name=str(name),
            type=str(data.get("type") or "string"),
            required=bool(data.get("required", False)),
            location=str(data.get("location") or "query"),
            max_length=int(data.get("max_length") or _MAX_PARAM_VALUE_CHARS),
        )


@dataclass(frozen=True)
class ProviderProbeEndpoint:
    endpoint_id: str  # e.g. "exa.search"
    provider_id: str  # evidence-proxy registry id
    method: str
    path: str  # fixed upstream path — no template slots, no free-form URLs
    params: tuple[ProbeParamSpec, ...] = ()
    est_cost_microusd: int = 0
    description: str = ""

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ProviderProbeEndpoint":
        raw_params = data.get("params") or {}
        if isinstance(raw_params, Mapping):
            params = tuple(ProbeParamSpec.from_mapping(name, spec if isinstance(spec, Mapping) else {}) for name, spec in raw_params.items())
        else:
            params = tuple(
                ProbeParamSpec.from_mapping(str(item.get("name") or ""), item)
                for item in raw_params
                if isinstance(item, Mapping)
            )
        return cls(
            endpoint_id=str(data.get("endpoint_id") or ""),
            provider_id=str(data.get("provider_id") or ""),
            method=str(data.get("method") or "GET").upper(),
            path=str(data.get("path") or ""),
            params=params,
            est_cost_microusd=int(data.get("est_cost_microusd") or 0),
            description=str(data.get("description") or ""),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["params"] = [asdict(param) for param in self.params]
        return data

    def prompt_summary(self) -> dict[str, Any]:
        """The shape shown to the model: ids, params, cost — nothing routable."""

        return {
            "endpoint_id": self.endpoint_id,
            "provider": self.provider_id,
            "params": {
                param.name: ("required" if param.required else "optional") + f" {param.type}"
                for param in self.params
            },
            "est_cost_microusd": self.est_cost_microusd,
            "description": self.description[:200],
        }


def validate_probe_catalog(endpoints: Sequence[ProviderProbeEndpoint]) -> list[str]:
    errors: list[str] = []
    seen: set[str] = set()
    for endpoint in endpoints:
        label = endpoint.endpoint_id or "<missing endpoint_id>"
        if not endpoint.endpoint_id:
            errors.append(f"{label}: endpoint_id is required")
        if endpoint.endpoint_id in seen:
            errors.append(f"{label}: duplicate endpoint_id")
        seen.add(endpoint.endpoint_id)
        if not endpoint.provider_id:
            errors.append(f"{label}: provider_id is required")
        if endpoint.method not in _VALID_METHODS:
            errors.append(f"{label}: method must be one of {_VALID_METHODS}")
        if not endpoint.path.startswith("/") or "{" in endpoint.path or "?" in endpoint.path:
            errors.append(f"{label}: path must be a fixed absolute path with no templates or query")
        if len(endpoint.params) > _MAX_PARAMS_PER_ENDPOINT:
            errors.append(f"{label}: too many params")
        for param in endpoint.params:
            if not param.name:
                errors.append(f"{label}: param name required")
            if param.type not in _VALID_PARAM_TYPES:
                errors.append(f"{label}: param {param.name} has unknown type {param.type}")
            if param.location not in ("query", "body"):
                errors.append(f"{label}: param {param.name} has unknown location {param.location}")
            if param.max_length <= 0 or param.max_length > _MAX_PARAM_VALUE_CHARS:
                errors.append(f"{label}: param {param.name} max_length out of bounds")
        if endpoint.est_cost_microusd < 0:
            errors.append(f"{label}: est_cost_microusd must be >= 0")
    return errors


def probe_catalog_hash(endpoints: Sequence[ProviderProbeEndpoint]) -> str:
    return sha256_json({"catalog": [e.to_dict() for e in sorted(endpoints, key=lambda e: e.endpoint_id)]})


def default_probe_catalog() -> list[ProviderProbeEndpoint]:
    """Launch catalog: one diagnostic-grade endpoint per seed provider."""

    return [
        ProviderProbeEndpoint(
            endpoint_id="exa.search",
            provider_id="exa",
            method="POST",
            path="/search",
            params=(
                ProbeParamSpec(name="query", type="string", required=True, location="body", max_length=300),
                ProbeParamSpec(name="numResults", type="integer", location="body"),
                ProbeParamSpec(name="category", type="string", location="body"),
            ),
            est_cost_microusd=5_000,
            description="Exa semantic search: verify result shape/coverage for a query form.",
        ),
        ProviderProbeEndpoint(
            endpoint_id="sd.scrape",
            provider_id="sd",
            method="GET",
            path="/scrape",
            params=(
                ProbeParamSpec(name="url", type="string", required=True, location="query"),
                ProbeParamSpec(name="dynamic", type="boolean", location="query"),
            ),
            est_cost_microusd=1_000,
            description="ScrapingDog fetch: verify a page renders/parses for an evidence pattern.",
        ),
    ]


def load_probe_catalog(path: str = "") -> list[ProviderProbeEndpoint]:
    """Load the catalog file; fall back to the built-in defaults.

    A present-but-invalid catalog raises: a typo'd catalog must fail loudly,
    not silently shrink the probe surface.
    """

    resolved = str(path or os.getenv(PROBE_CATALOG_PATH_ENV) or "").strip()
    if not resolved:
        return default_probe_catalog()
    with Path(resolved).open("r", encoding="utf-8") as handle:
        doc = json.load(handle)
    raw = doc.get("endpoints") if isinstance(doc, Mapping) else doc
    if not isinstance(raw, list) or not raw:
        raise ValueError("probe catalog file must contain a non-empty endpoints list")
    endpoints = [ProviderProbeEndpoint.from_mapping(item) for item in raw]
    errors = validate_probe_catalog(endpoints)
    if errors:
        raise ValueError("invalid probe catalog: " + "; ".join(errors))
    return endpoints


def find_probe_endpoint(
    endpoints: Sequence[ProviderProbeEndpoint], endpoint_id: str
) -> ProviderProbeEndpoint | None:
    wanted = str(endpoint_id or "").strip()
    for endpoint in endpoints:
        if endpoint.endpoint_id == wanted:
            return endpoint
    return None


def validate_probe_params(
    endpoint: ProviderProbeEndpoint, params: Mapping[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    """Validate + coerce request params against the endpoint schema.

    Returns (normalized_params, errors). Unknown keys are errors — the catalog
    schema is the whole allowed surface, so nothing unvetted reaches a query
    string or body.
    """

    errors: list[str] = []
    normalized: dict[str, Any] = {}
    specs = {spec.name: spec for spec in endpoint.params}
    for key in params:
        if str(key) not in specs:
            errors.append(f"unknown param: {key}")
    for spec in endpoint.params:
        if spec.name not in params:
            if spec.required:
                errors.append(f"missing required param: {spec.name}")
            continue
        value = params[spec.name]
        if spec.type == "string":
            if not isinstance(value, str):
                value = str(value)
            if len(value) > spec.max_length:
                errors.append(f"param {spec.name} exceeds max_length {spec.max_length}")
                continue
            normalized[spec.name] = value
        elif spec.type == "integer":
            try:
                normalized[spec.name] = int(value)
            except (TypeError, ValueError):
                errors.append(f"param {spec.name} must be an integer")
        elif spec.type == "number":
            try:
                normalized[spec.name] = float(value)
            except (TypeError, ValueError):
                errors.append(f"param {spec.name} must be a number")
        elif spec.type == "boolean":
            if isinstance(value, bool):
                normalized[spec.name] = value
            elif str(value).strip().lower() in {"true", "false", "1", "0"}:
                normalized[spec.name] = str(value).strip().lower() in {"true", "1"}
            else:
                errors.append(f"param {spec.name} must be a boolean")
    return normalized, errors
