import httpx
import pytest

from lab_arena.store import ArenaStoreError, PostgrestTransport


def test_scoped_service_key_is_only_apikey_header():
    observed = {}

    def handler(request):
        observed.update(request.headers)
        return httpx.Response(200, json={"ok": True})

    client = httpx.Client(transport=httpx.MockTransport(handler), trust_env=False)
    transport = PostgrestTransport(
        "https://project.example",
        anon_key="anon-must-not-be-used",
        service_key="sb_secret_scoped-value",
        http_client=client,
    )
    transport.rpc("lab_arena_whoami", {})
    assert observed["apikey"] == "sb_secret_scoped-value"
    assert "authorization" not in observed


def test_legacy_jwt_header_path_remains_available():
    transport = PostgrestTransport(
        "https://project.example", anon_key="anon", service_jwt="a.b.c"
    )
    assert transport._headers["apikey"] == "anon"
    assert transport._headers["Authorization"] == "Bearer a.b.c"
    transport.close()


def test_broad_service_role_key_is_rejected():
    with pytest.raises(ArenaStoreError, match="invalid shape"):
        PostgrestTransport(
            "https://project.example", anon_key="anon", service_key="service-role-key"
        )


def test_exactly_one_service_credential_is_required():
    with pytest.raises(ArenaStoreError, match="exactly one"):
        PostgrestTransport("https://project.example", anon_key="anon")
    with pytest.raises(ArenaStoreError, match="exactly one"):
        PostgrestTransport(
            "https://project.example",
            anon_key="anon",
            service_key="sb_secret_scoped-value",
            service_jwt="a.b.c",
        )
