from __future__ import annotations

import pytest

from scripts import verify_arena_ddgs_client_hello as probe


def _hello(host: str) -> dict[str, str]:
    return {"authority": host + ":443", "server_name": host}


def test_profile_and_backend_connection_counts_are_independent():
    labels = ["profile-%d" % index for index in range(probe.EXPECTED_PROFILE_COUNT)]
    profile_hellos = [_hello(probe.DESTINATION_HOST), _hello(probe.DESTINATION_HOST)]
    backend_hellos = [
        _hello("html.duckduckgo.com"),
        _hello("links.duckduckgo.com"),
    ]

    assert probe._validated_profile_inventory(labels) == labels
    assert probe._validated_profile_connection_count(labels[0], profile_hellos) == 2
    assert probe._validated_backend_count(backend_hellos) == 2


def test_backend_requires_matching_public_duckduckgo_sni():
    with pytest.raises(RuntimeError, match="backend failed Arena SNI inspection"):
        probe._validated_backend_count(
            [{"authority": "html.duckduckgo.com:443", "server_name": "other.example"}]
        )

    with pytest.raises(RuntimeError, match="did not use its explicit proxy"):
        probe._validated_backend_count([])


def test_linux_profile_support_classification_is_exact():
    assert probe._is_unsupported_linux_profile(
        "chrome_100",
        {
            "client_error_type": "BuilderError",
            "client_error_detail": 'Invalid impersonate: "chrome_100"',
        },
    )
    assert not probe._is_unsupported_linux_profile(
        "chrome_100",
        {
            "client_error_type": "BuilderError",
            "client_error_detail": "proxy connection failed",
        },
    )
