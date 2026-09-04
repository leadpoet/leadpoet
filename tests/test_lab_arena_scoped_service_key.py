import httpx
import pytest

from lab_arena.store import ArenaRoleError, ArenaStore, PostgrestTransport


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


def test_scoped_service_key_still_rejects_wrong_database_role():
    class WrongRoleTransport:
        def rpc(self, function, params):
            assert function == "lab_arena_whoami"
            return {
                "schema_version": "leadpoet.lab_arena.whoami.v1",
                "current_user": "service_role",
                "rolsuper": False,
                "rolbypassrls": False,
                "rolcanlogin": False,
            }

    with pytest.raises(ArenaRoleError, match="database role is not lab_arena_service"):
        ArenaStore(WrongRoleTransport()).require_service_role()


def test_legacy_jwt_header_path_remains_available():
    transport = PostgrestTransport(
        "https://project.example", anon_key="anon", service_jwt="a.b.c"
    )
    assert transport._headers["apikey"] == "anon"
    assert transport._headers["Authorization"] == "Bearer a.b.c"
    transport.close()


def test_full_service_role_key_is_not_accepted_as_scoped_key():
    with pytest.raises(Exception, match="invalid shape"):
        PostgrestTransport(
            "https://project.example", anon_key="anon", service_key="service-role-key"
        )
