"""Bounded product-requirement accounting over already admitted evidence.

The model still interprets natural-language scope and entailment. The host
checks lossless source spans, explicit connector consistency, stable repair
structure, exact clause IDs and citations, then evaluates the declared tree.
This is not a general natural-language parser or a second research workflow.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping

MAX_NODES = 24
MAX_DEPTH = 6
MAX_BINDINGS = 3
SCHEMA = "product_coverage_v1"
PRIVATE_KEY = "_validated_product_coverage"


class CoverageError(ValueError):
    pass


class ValidatedProductCoverage(dict):
    """Host-created proof; JSON round trips deliberately lose this authority."""


def response_schema(*, description: str = "") -> dict:
    """Encode the bounded matrix within the existing broker schema depth.

    A nested tool schema exceeds the broker's depth limit. JSON text is only
    the wire representation; the host decodes and validates every node/claim.
    """
    return {"type": ["string", "null"], "maxLength": 24000,
            "description": description or "Compact JSON text with nodes and claims as instructed."}


def _decode(raw: object) -> Mapping:
    if isinstance(raw, str):
        if len(raw) > 24000:
            raise CoverageError("product coverage exceeds bounded matrix size")
        try:
            raw = json.loads(raw)
        except (ValueError, RecursionError):
            raise CoverageError("product coverage is not valid JSON") from None
    if not isinstance(raw, Mapping):
        raise CoverageError("missing structured product coverage")
    return raw


def criterion_hash(criterion: str) -> str:
    return hashlib.sha256(criterion.strip().encode("utf-8")).hexdigest()


def _plan(criterion: str, nodes: object) -> list[dict]:
    if not isinstance(nodes, list) or not 1 <= len(nodes) <= MAX_NODES:
        raise CoverageError("product coverage requires a bounded requirement tree")
    indexed = {}
    for raw in nodes:
        if (not isinstance(raw, Mapping) or set(raw) != {"id", "span", "op", "children"}
                or not isinstance(raw.get("id"), str)
                or not re.fullmatch(r"p(?:0|[1-9]\d*)", raw["id"])
                or raw["id"] in indexed
                or not isinstance(raw.get("op"), str)
                or raw.get("op") not in {"all", "any", "leaf", "example"}
                or not isinstance(raw.get("span"), str) or not raw["span"].strip()
                or not isinstance(raw.get("children"), list)
                or any(not isinstance(v, str) for v in raw["children"])
                or len(set(raw["children"])) != len(raw["children"])):
            raise CoverageError("invalid product requirement node or exact ID")
        span = raw["span"]
        # Exact text, not model-supplied offsets or rewritten requirement text.
        starts = [m.start() for m in re.finditer(re.escape(span), criterion)]
        if len(starts) != 1:
            raise CoverageError("product requirement span is absent or ambiguous")
        start = starts[0]
        indexed[raw["id"]] = {**dict(raw), "start": start, "end": start + len(span)}
    if set(indexed) != {f"p{i}" for i in range(len(nodes))}:
        raise CoverageError("product requirement IDs must be unique and dense")
    if indexed["p0"]["span"] != criterion:
        raise CoverageError("product tree root must retain the complete original criterion")
    if indexed["p0"]["op"] == "leaf" and re.search(
        r"\b(?:and|or|both|either|such as|for example)\b", criterion, re.I,
    ):
        raise CoverageError("compound product criterion requires explicit requirement structure")
    seen = set()

    def visit(key: str, depth: int) -> None:
        if key not in indexed or key in seen or depth > MAX_DEPTH:
            raise CoverageError(f"missing, repeated, cyclic or over-depth product node {key[:32]}")
        seen.add(key)
        node = indexed[key]
        children = node["children"]
        if node["op"] in {"leaf", "example"}:
            if children:
                raise CoverageError("product leaf cannot hide child requirements")
            if node["op"] == "example":
                prefix = criterion[max(0, node["start"] - 40):node["start"]]
                if (not (re.search(r"(?:such as|for example|e\.g\.)\s*[:,(]?\s*$", prefix, re.I)
                         or re.match(r"(?:such as|for example|e\.g\.)(?:\s|[:,(])", node["span"], re.I))
                        or re.search(r"\b(?:must|required|requires|mandatory|only|that|which|who|whose|where|through)\b", node["span"], re.I)):
                    raise CoverageError("unlicensed product example or hidden mandatory qualifier")
            return
        if len(children) < 2:
            raise CoverageError("product operator requires at least two children")
        cursor = node["start"]
        gaps = []
        previous_child = None
        for child_id in children:
            if child_id not in indexed:
                raise CoverageError(f"product node {key} references missing child {child_id[:32]}")
            child = indexed[child_id]
            if child["start"] < cursor or child["end"] > node["end"]:
                raise CoverageError(
                    f"product node {key}: child {child_id} span [{child['start']},{child['end']}) "
                    f"overlaps/out of order after {previous_child or 'parent start'} at {cursor}, "
                    f"or exceeds parent [{node['start']},{node['end']})"
                )
            gaps.append(criterion[cursor:child["start"]])
            cursor = child["end"]
            previous_child = child_id
            visit(child_id, depth + 1)
        gaps.append(criterion[cursor:node["end"]])
        connector = " ".join(gaps)
        residual = re.sub(r"\b(?:and|or|both|either|all|of|such as|for example)\b|e\.g\.", "", connector, flags=re.I)
        if re.search(r"\w", residual):
            gap = next((gap for gap in gaps if re.search(r"\w", re.sub(
                r"\b(?:and|or|both|either|all|of|such as|for example)\b|e\.g\.",
                "", gap, flags=re.I,
            ))), connector)
            raise CoverageError(
                f"product node {key} omitted criterion gap {gap[:120]!r}; "
                "retain the complete gap in an adjacent child span without changing scope"
            )
        if node["op"] == "any":
            if not re.search(r"\bor\b|\beither\b|/", connector, re.I) or re.search(r"\band\b|\bboth\b|\ball\b", connector, re.I):
                raise CoverageError("product OR conflicts with original connectors")
        elif re.search(r"\bor\b|\beither\b|/", connector, re.I):
            raise CoverageError("product AND conflicts with original alternatives")

    visit("p0", 0)
    if seen != set(indexed):
        raise CoverageError("unreachable product requirement nodes")
    return [indexed[f"p{i}"] for i in range(len(nodes))]


def compile_plan(criterion: str, raw: object) -> list[dict]:
    """Validate spans before claims so citation repairs cannot rewrite scope."""
    raw = _decode(raw)
    return _plan(criterion.strip(), raw.get("nodes"))


def validate_coverage(
    raw: object, *, criterion: str,
    citation_rejection: Callable[[str, str], str | None],
    locked_nodes: list[dict] | None = None,
) -> ValidatedProductCoverage:
    criterion = criterion.strip()
    raw = _decode(raw)
    if not criterion or set(raw) != {"nodes", "claims"}:
        raise CoverageError("missing structured product coverage")
    nodes = _plan(criterion, raw["nodes"])
    if locked_nodes is not None and nodes != locked_nodes:
        raise CoverageError("product requirement tree changed during repair")
    expected = {n["id"] for n in nodes if n["op"] == "leaf"}
    claims = raw["claims"]
    if not isinstance(claims, list) or len(claims) != len(expected):
        raise CoverageError("incomplete product clause findings")
    statuses, validated, issues = {}, [], {}
    for claim in claims:
        if (not isinstance(claim, Mapping) or set(claim) != {"id", "status", "activity_role", "evidence"}
                or not isinstance(claim.get("id"), str)
                or not isinstance(claim.get("status"), str)
                or not isinstance(claim.get("activity_role"), str)
                or claim.get("id") not in expected or claim["id"] in statuses
                or claim.get("status") not in {"VERIFIED", "CONTRADICTED", "UNPROVEN"}
                or claim.get("activity_role") not in {"supplier_operator", "customer_user", "internal_function", "third_party", "unresolved"}
                or not isinstance(claim.get("evidence"), list) or len(claim["evidence"]) > MAX_BINDINGS):
            raise CoverageError("invalid or forged product clause finding")
        key, status = claim["id"], claim["status"]
        evidence, reasons = [], []
        for binding in claim["evidence"]:
            if (not isinstance(binding, Mapping) or set(binding) != {"url", "quote"}
                    or not isinstance(binding["url"], str) or not isinstance(binding["quote"], str)):
                raise CoverageError("invalid product clause evidence")
            rejection = citation_rejection(binding["url"], binding["quote"])
            if rejection:
                reasons.append(rejection)
            else:
                evidence.append(dict(binding))
        if status in {"VERIFIED", "CONTRADICTED"} and not evidence:
            reasons.append("product clause lacks bound evidence")
        if status == "VERIFIED" and claim["activity_role"] != "supplier_operator":
            reasons.append("product clause does not establish supplier/operator role")
        if reasons:
            status = "UNPROVEN"
            issues[key] = sorted(set(reasons))
        statuses[key] = status
        validated.append({**dict(claim), "status": status, "evidence": evidence})
    if set(statuses) != expected:
        raise CoverageError("missing product clause IDs")
    by_id = {n["id"]: n for n in nodes}

    def evaluate(key: str) -> tuple[str, list[str]]:
        node = by_id[key]
        if node["op"] == "example":
            return "VERIFIED", []
        if node["op"] == "leaf":
            return statuses[key], ([] if statuses[key] == "VERIFIED" else [key])
        children = [evaluate(child) for child in node["children"] if by_id[child]["op"] != "example"]
        if not children:
            raise CoverageError("product operator contains only examples")
        values = [v for v, _ in children]
        if node["op"] == "any":
            status = "VERIFIED" if "VERIFIED" in values else ("CONTRADICTED" if all(v == "CONTRADICTED" for v in values) else "UNPROVEN")
        else:
            status = "CONTRADICTED" if "CONTRADICTED" in values else ("VERIFIED" if all(v == "VERIFIED" for v in values) else "UNPROVEN")
        return status, ([] if status == "VERIFIED" else [v for _, keys in children for v in keys])

    status, unresolved = evaluate("p0")
    return ValidatedProductCoverage(
        schema=SCHEMA, criterion_sha256=criterion_hash(criterion), nodes=nodes,
        claims=validated, status=status, unresolved_ids=unresolved, citation_issues=issues,
        tree_sha256=hashlib.sha256(json.dumps(nodes, sort_keys=True).encode()).hexdigest(),
    )


def coverage_matches(value: object, criterion: str) -> bool:
    return (isinstance(value, ValidatedProductCoverage)
            and value.get("criterion_sha256") == criterion_hash(criterion)
            and value.get("status") == "VERIFIED")
