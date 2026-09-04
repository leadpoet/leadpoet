"""Shared Research Lab hashing and secret-material checks."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping


SECRET_MARKERS = (
    "sk-or-",
    "openrouter_api_key",
    "raw_openrouter_key",
    "raw_secret",
    "service_role",
    "hidden_icp",
    "icp_plaintext",
    ".dkr.ecr.",
    "image_digest",
    "private_repo",
    "judge_prompt",
)
SECRET_KEY_MARKERS = (
    "api_key",
    "raw_secret",
    "raw_openrouter",
    "credential",
    "private_model_manifest_doc",
    "candidate_patch_manifest",
    "image_digest",
    "proxy_url",
)
SECRET_TOKEN_KEY_MARKERS = (
    "access_token",
    "api_token",
    "auth_token",
    "bearer_token",
    "refresh_token",
    "session_token",
    "token_key",
    "token_secret",
    "token_value",
)


def canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def sha256_json(data: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(data).encode("utf-8")).hexdigest()


def contains_secret_material(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, item in value.items():
            lowered_key = str(key).lower()
            if _looks_like_secret_key(lowered_key):
                return True
            if contains_secret_material(item):
                return True
    elif isinstance(value, list):
        return any(contains_secret_material(item) for item in value)
    elif isinstance(value, str):
        lowered = value.lower()
        return any(marker in lowered for marker in SECRET_MARKERS)
    return False


def _looks_like_secret_key(lowered_key: str) -> bool:
    return _first_secret_key_marker(lowered_key) is not None


def _first_secret_key_marker(lowered_key: str) -> str | None:
    for marker in SECRET_KEY_MARKERS:
        if marker in lowered_key:
            return marker
    for marker in SECRET_TOKEN_KEY_MARKERS:
        if marker in lowered_key:
            return marker
    return None
