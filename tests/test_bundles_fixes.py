"""Tests for the retained Research Lab bundle helpers."""

from gateway.research_lab.bundles import contains_secret_material, sha256_json


def test_sha256_json_is_canonical():
    assert sha256_json({"b": 2, "a": [1, 3]}) == sha256_json({"a": [1, 3], "b": 2})


def test_contains_secret_material_rejects_nested_marker_values():
    assert contains_secret_material(
        {"result": [{"error": "provider returned a service_role error"}]}
    )
    assert contains_secret_material(["prefix sk-or-secret suffix"])


def test_contains_secret_material_rejects_secret_looking_keys():
    for key in ("api_key", "credential_ref", "access_token", "session_token"):
        assert contains_secret_material({key: "redacted"}), key


def test_contains_secret_material_accepts_clean_current_state():
    assert not contains_secret_material(
        {
            "bundle_id": "allocation:123",
            "source_state": {
                "arena_submission_id": "submission-1",
                "provider": "openrouter",
                "score": 0.75,
            },
            "u16_weights": {"1": 32768, "2": 32767},
        }
    )
