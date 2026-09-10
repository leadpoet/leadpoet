"""Arena-only gateway schema preflight contract."""
import json
from urllib.error import HTTPError
from urllib.parse import urlparse

import pytest

from gateway.tee.supabase_schema_preflight_v2 import (
    REQUIRED_SUPABASE_V2_RPCS,
    REQUIRED_SUPABASE_V2_SCHEMA,
    SupabaseSchemaPreflightV2Error,
    verify_required_supabase_v2_schema,
)


class _Response:
    def __init__(self, body=b"[]", status=200): self.body, self.status = body, status
    def __enter__(self): return self
    def __exit__(self, *_): return False
    def getcode(self): return self.status
    def read(self, size=-1): return self.body if size < 0 else self.body[:size]


def _opener(*, missing=None, bad_capability=None):
    paths = {f"/rpc/{name}": {} for _, name in REQUIRED_SUPABASE_V2_RPCS}
    capabilities = {
        "lab_arena_schema_version_v1": {"schema_version": "leadpoet.lab_arena.schema_version.v1", "version": 197},
        "lab_arena_weight_state_schema_v1": {"schema_version": "leadpoet.lab_arena.weight_state_schema.v1", "version": 202},
        "lab_arena_incentive_retirement_schema_v1": {"schema_version": "leadpoet.lab_arena.incentive_retirement_schema.v1", "version": 203},
    }
    def open_(request, timeout):
        path = urlparse(request.full_url).path
        if missing and missing in path:
            raise HTTPError(request.full_url, 404, "missing", {}, None)
        if path == "/rest/v1/":
            return _Response(json.dumps({"paths": paths}).encode())
        if "/rpc/" in path:
            name = path.rsplit("/", 1)[-1]
            value = capabilities[name]
            if bad_capability == name: value = dict(value, version=0)
            return _Response(json.dumps(value).encode())
        return _Response()
    return open_


def test_preflight_proves_arena_203_and_generic_scoring_schema_only():
    result = verify_required_supabase_v2_schema(
        {"SUPABASE_URL": "https://db.example", "SUPABASE_SERVICE_ROLE_KEY": "secret"},
        opener=_opener(),
    )
    assert result["status"] == "ready"
    assert result["schema_capabilities"]["lab_arena_incentive_retirement_schema_v1"]["version"] == 203
    names = {name for _, name, _ in REQUIRED_SUPABASE_V2_SCHEMA}
    assert "lab_arena_accepted_weight_states" in names
    assert "research_lab_chain_realized_epoch_settlements_v1" not in names
    assert all("compact_weight" not in name and "allocation" not in name for _, name in REQUIRED_SUPABASE_V2_RPCS)


def test_preflight_fails_closed_for_missing_arena_table_and_wrong_retirement_capability():
    env = {"SUPABASE_URL": "https://db.example", "SUPABASE_SERVICE_ROLE_KEY": "secret"}
    with pytest.raises(SupabaseSchemaPreflightV2Error, match="202-arena"):
        verify_required_supabase_v2_schema(env, opener=_opener(missing="lab_arena_accepted_weight_states"))
    with pytest.raises(SupabaseSchemaPreflightV2Error, match="capability differs"):
        verify_required_supabase_v2_schema(env, opener=_opener(bad_capability="lab_arena_incentive_retirement_schema_v1"))


def test_preflight_requires_credentials():
    with pytest.raises(SupabaseSchemaPreflightV2Error, match="credentials"):
        verify_required_supabase_v2_schema({})
