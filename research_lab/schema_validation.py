"""Schema validation helpers for Research Lab emitter records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_DIR = ROOT / "schemas"


def validate_schema_record(schema_name: str, record: dict[str, Any]) -> list[str]:
    """Return validation errors for a Research Lab schema record."""
    schema = _load_schema(schema_name)
    try:
        import jsonschema

        validator = jsonschema.Draft202012Validator(
            schema,
            format_checker=jsonschema.FormatChecker(),
        )
        return [error.message for error in sorted(validator.iter_errors(record), key=str)]
    except Exception:
        return _MiniValidator(schema).validate(schema, record)


def assert_schema_record(schema_name: str, record: dict[str, Any]) -> None:
    errors = validate_schema_record(schema_name, record)
    if errors:
        raise ValueError(f"{schema_name} validation failed: {'; '.join(errors[:5])}")


def _load_schema(schema_name: str) -> dict[str, Any]:
    path = SCHEMA_DIR / schema_name
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


class _MiniValidator:
    def __init__(self, root_schema: dict[str, Any]) -> None:
        self.root_schema = root_schema

    def validate(self, schema: dict[str, Any], value: Any, path: str = "$") -> list[str]:
        errors: list[str] = []

        if "$ref" in schema:
            return self.validate(self._resolve_ref(schema["$ref"]), value, path)

        if "const" in schema and value != schema["const"]:
            errors.append(f"{path}: expected const {schema['const']!r}, got {value!r}")
        if "enum" in schema and value not in schema["enum"]:
            errors.append(f"{path}: expected one of {schema['enum']!r}, got {value!r}")

        expected_type = schema.get("type")
        if expected_type is not None:
            types = expected_type if isinstance(expected_type, list) else [expected_type]
            if not any(self._matches_type(value, t) for t in types):
                errors.append(f"{path}: expected type {types!r}, got {type(value).__name__}")
                return errors

        if isinstance(value, dict):
            for key in schema.get("required", []):
                if key not in value:
                    errors.append(f"{path}: missing required property {key!r}")

            properties = schema.get("properties", {})
            for key, sub_schema in properties.items():
                if key in value:
                    errors.extend(self.validate(sub_schema, value[key], f"{path}.{key}"))

            additional = schema.get("additionalProperties", True)
            if additional is False:
                for key in value:
                    if key not in properties:
                        errors.append(f"{path}: unexpected property {key!r}")
            elif isinstance(additional, dict):
                for key in value:
                    if key not in properties:
                        errors.extend(self.validate(additional, value[key], f"{path}.{key}"))

        if isinstance(value, list):
            if len(value) < schema.get("minItems", 0):
                errors.append(f"{path}: expected at least {schema['minItems']} items")
            if "items" in schema:
                for idx, item in enumerate(value):
                    errors.extend(self.validate(schema["items"], item, f"{path}[{idx}]"))

        if isinstance(value, str):
            if "minLength" in schema and len(value) < schema["minLength"]:
                errors.append(f"{path}: expected minLength {schema['minLength']}")
            if "maxLength" in schema and len(value) > schema["maxLength"]:
                errors.append(f"{path}: expected maxLength {schema['maxLength']}")
            if schema.get("format") == "uuid":
                import uuid

                try:
                    uuid.UUID(value)
                except ValueError:
                    errors.append(f"{path}: expected uuid format")
            elif schema.get("format") == "date-time":
                from datetime import datetime

                try:
                    datetime.fromisoformat(value.replace("Z", "+00:00"))
                except ValueError:
                    errors.append(f"{path}: expected date-time format")
            elif schema.get("format") == "uri":
                from urllib.parse import urlparse

                parsed = urlparse(value)
                if not parsed.scheme or not parsed.netloc:
                    errors.append(f"{path}: expected uri format")

        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if "minimum" in schema and value < schema["minimum"]:
                errors.append(f"{path}: expected minimum {schema['minimum']}, got {value}")

        return errors

    def _resolve_ref(self, ref: str) -> dict[str, Any]:
        if not ref.startswith("#/"):
            raise ValueError(f"Only local refs are supported: {ref}")
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
