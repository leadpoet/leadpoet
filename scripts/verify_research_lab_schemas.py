#!/usr/bin/env python3
"""Verify Leadpoet Research Lab Phase 0 schema fixtures.

The project declares jsonschema as a dependency, but local development shells do
not always have the full dependency set installed. This script uses jsonschema
when available and falls back to a small validator for the Draft 2020-12 subset
used by the Phase 0 schemas.
"""

from __future__ import annotations

import json
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_DIR = ROOT / "schemas"
FIXTURE_DIR = SCHEMA_DIR / "fixtures"


CASES = [
    ("research_trajectory.schema.json", "research_trajectory.valid.json", True),
    ("execution_trace.schema.json", "execution_trace.valid.json", True),
    ("evidence_bundle.schema.json", "evidence_bundle.valid.json", True),
    ("results_ledger_row.schema.json", "results_ledger_row.valid.json", True),
    (
        "research_trajectory.schema.json",
        "research_trajectory.invalid_missing_event_hash.json",
        False,
    ),
    ("research_trajectory.schema.json", "research_trajectory.invalid_raw_prompt.json", False),
    (
        "research_trajectory.schema.json",
        "research_trajectory.invalid_champion_code.json",
        False,
    ),
    ("execution_trace.schema.json", "execution_trace.invalid_judge_prompt.json", False),
    ("evidence_bundle.schema.json", "evidence_bundle.invalid_page_content.json", False),
]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


class MiniSchemaValidator:
    """A strict-enough validator for the schema subset in this directory."""

    def __init__(self, root_schema: dict[str, Any]):
        self.root_schema = root_schema

    def validate(self, schema: dict[str, Any], value: Any, path: str = "$") -> list[str]:
        errors: list[str] = []

        if "$ref" in schema:
            ref_schema = self._resolve_ref(schema["$ref"])
            errors.extend(self.validate(ref_schema, value, path))
            return errors

        if "allOf" in schema:
            for idx, sub_schema in enumerate(schema["allOf"]):
                errors.extend(self.validate(sub_schema, value, f"{path}.allOf[{idx}]"))

        if "oneOf" in schema:
            matches = [
                sub_schema
                for sub_schema in schema["oneOf"]
                if not self.validate(sub_schema, value, path)
            ]
            if len(matches) != 1:
                errors.append(f"{path}: expected exactly one oneOf match, found {len(matches)}")
            return errors

        if "if" in schema:
            if not self.validate(schema["if"], value, path):
                errors.extend(self.validate(schema.get("then", {}), value, path))

        if "const" in schema and value != schema["const"]:
            errors.append(f"{path}: expected const {schema['const']!r}, got {value!r}")

        if "enum" in schema and value not in schema["enum"]:
            errors.append(f"{path}: expected one of {schema['enum']!r}, got {value!r}")

        if "type" in schema:
            expected = schema["type"]
            if isinstance(expected, str):
                expected = [expected]
            if not any(self._matches_type(value, t) for t in expected):
                errors.append(f"{path}: expected type {expected!r}, got {type(value).__name__}")
                return errors

        if isinstance(value, dict):
            required = schema.get("required", [])
            for key in required:
                if key not in value:
                    errors.append(f"{path}: missing required property {key!r}")

            properties = schema.get("properties", {})
            for key, sub_schema in properties.items():
                if key in value:
                    errors.extend(self.validate(sub_schema, value[key], f"{path}.{key}"))

            additional = schema.get("additionalProperties", True)
            if additional is False:
                allowed = set(properties)
                for key in value:
                    if key not in allowed:
                        errors.append(f"{path}: unexpected property {key!r}")
            elif isinstance(additional, dict):
                for key in value:
                    if key not in properties:
                        errors.extend(self.validate(additional, value[key], f"{path}.{key}"))

        if isinstance(value, list):
            if "minItems" in schema and len(value) < schema["minItems"]:
                errors.append(f"{path}: expected at least {schema['minItems']} items")
            if "maxItems" in schema and len(value) > schema["maxItems"]:
                errors.append(f"{path}: expected at most {schema['maxItems']} items")
            if "items" in schema:
                for idx, item in enumerate(value):
                    errors.extend(self.validate(schema["items"], item, f"{path}[{idx}]"))

        if isinstance(value, str):
            if "minLength" in schema and len(value) < schema["minLength"]:
                errors.append(f"{path}: expected minLength {schema['minLength']}")
            if "maxLength" in schema and len(value) > schema["maxLength"]:
                errors.append(f"{path}: expected maxLength {schema['maxLength']}")
            if "format" in schema:
                fmt_error = self._format_error(value, schema["format"], path)
                if fmt_error:
                    errors.append(fmt_error)

        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if "minimum" in schema and value < schema["minimum"]:
                errors.append(f"{path}: expected minimum {schema['minimum']}, got {value}")
            if "maximum" in schema and value > schema["maximum"]:
                errors.append(f"{path}: expected maximum {schema['maximum']}, got {value}")

        return errors

    def _resolve_ref(self, ref: str) -> dict[str, Any]:
        if not ref.startswith("#/"):
            raise ValueError(f"Only local refs are supported by the fallback validator: {ref}")
        cur: Any = self.root_schema
        for part in ref[2:].split("/"):
            cur = cur[part]
        return cur

    @staticmethod
    def _matches_type(value: Any, expected: str) -> bool:
        if expected == "null":
            return value is None
        if expected == "object":
            return isinstance(value, dict)
        if expected == "array":
            return isinstance(value, list)
        if expected == "string":
            return isinstance(value, str)
        if expected == "boolean":
            return isinstance(value, bool)
        if expected == "integer":
            return isinstance(value, int) and not isinstance(value, bool)
        if expected == "number":
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        raise ValueError(f"Unsupported schema type {expected!r}")

    @staticmethod
    def _format_error(value: str, fmt: str, path: str) -> str | None:
        if fmt == "uuid":
            try:
                uuid.UUID(value)
            except ValueError:
                return f"{path}: expected uuid format"
        elif fmt == "date-time":
            try:
                datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                return f"{path}: expected date-time format"
        elif fmt == "uri":
            parsed = urlparse(value)
            if not parsed.scheme or not parsed.netloc:
                return f"{path}: expected uri format"
        return None


@dataclass
class ValidationBackend:
    name: str

    def validate(self, schema: dict[str, Any], data: Any) -> list[str]:
        raise NotImplementedError


class JsonschemaBackend(ValidationBackend):
    def __init__(self) -> None:
        super().__init__("jsonschema")
        import jsonschema

        self._jsonschema = jsonschema

    def validate(self, schema: dict[str, Any], data: Any) -> list[str]:
        validator = self._jsonschema.Draft202012Validator(
            schema,
            format_checker=self._jsonschema.FormatChecker(),
        )
        return [error.message for error in sorted(validator.iter_errors(data), key=str)]


class FallbackBackend(ValidationBackend):
    def __init__(self) -> None:
        super().__init__("built-in fallback")

    def validate(self, schema: dict[str, Any], data: Any) -> list[str]:
        return MiniSchemaValidator(schema).validate(schema, data)


def get_backend() -> ValidationBackend:
    try:
        return JsonschemaBackend()
    except Exception:
        return FallbackBackend()


def assert_schema_invariants(schemas: dict[str, dict[str, Any]]) -> list[str]:
    errors: list[str] = []

    trajectory_event = schemas["research_trajectory.schema.json"]["$defs"]["event"]
    for field in ["seq", "ts", "type", "cost_usd", "anchored_hash"]:
        if field not in trajectory_event.get("required", []):
            errors.append(f"research_trajectory event missing required base field {field}")

    trace_call = schemas["execution_trace.schema.json"]["$defs"]["call"]
    emitter_enum = trace_call["properties"]["call_emitter"].get("enum")
    if emitter_enum != ["model", "code"]:
        errors.append("execution_trace call_emitter must be exactly ['model', 'code']")
    if "teacher_model_flag" not in trace_call.get("required", []):
        errors.append("execution_trace calls must require teacher_model_flag")

    evidence_snapshot = schemas["evidence_bundle.schema.json"]["$defs"]["snapshot"]
    for field in ["content_hash", "normalized_text_hash", "snapshot_ref", "signature", "l0_verification_state"]:
        if field not in evidence_snapshot.get("required", []):
            errors.append(f"evidence_bundle snapshot missing required field {field}")

    ledger_status = schemas["results_ledger_row.schema.json"]["properties"]["status"].get("enum")
    if ledger_status != ["keep", "discard", "crash", "timeout"]:
        errors.append("results ledger status enum drifted from the plan contract")

    return errors


def main() -> int:
    schema_paths = sorted(SCHEMA_DIR.glob("*.schema.json"))
    schemas = {path.name: load_json(path) for path in schema_paths}

    missing = {
        "research_trajectory.schema.json",
        "execution_trace.schema.json",
        "evidence_bundle.schema.json",
        "results_ledger_row.schema.json",
    } - set(schemas)
    if missing:
        print(f"Missing schema files: {sorted(missing)}", file=sys.stderr)
        return 1

    invariant_errors = assert_schema_invariants(schemas)
    if invariant_errors:
        for error in invariant_errors:
            print(f"INVARIANT FAIL: {error}", file=sys.stderr)
        return 1

    backend = get_backend()
    failures: list[str] = []
    for schema_name, fixture_name, should_pass in CASES:
        schema = schemas[schema_name]
        fixture = load_json(FIXTURE_DIR / fixture_name)
        errors = backend.validate(schema, fixture)
        passed = not errors
        if passed != should_pass:
            status = "passed" if passed else "failed"
            expected = "pass" if should_pass else "fail"
            detail = "; ".join(errors[:5])
            failures.append(f"{fixture_name}: expected {expected}, {status}. {detail}")

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        return 1

    print(f"Research Lab schema fixtures verified with {backend.name}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
