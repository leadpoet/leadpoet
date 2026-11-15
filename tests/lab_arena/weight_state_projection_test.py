"""Signed weight reads omit catalogs without changing reward authority."""

from copy import deepcopy
import json
from types import SimpleNamespace

import httpx
import pytest

from lab_arena import weight_state
from lab_arena.service import ArenaService, ServiceError
from lab_arena.store import ArenaStore, PUBLIC_CONFIGURATION_COLUMN, PostgrestTransport, PsycopgTransport
from tests.lab_arena.public_current_projection_test import ALICE, BASELINE, _reward, _service


def _weight_service(rows, signer, state, *, legacy=False):
    service, transport = _service(rows)
    select = transport.select

    def with_existing_state(table, **query):
        if table == "lab_arena_accepted_weight_states":
            assert query["filters"] == {"network": "finney", "netuid": 71, "epoch": 101}
            return [{"state_doc": deepcopy(state)}]
        return select(table, **query)

    transport.select = with_existing_state
    service._reward_signer = lambda: signer
    if legacy:
        service.public_reward_basis = lambda epoch, **_kwargs: ArenaService.public_reward_basis(
            service, epoch, public_only=False,
        )
    return service, transport


def _accepted(signer, basis):
    return weight_state.build_accepted_weight_state(
        signer, network="finney", genesis_hash="1" * 64, netuid=71,
        epoch=101, valid_from_block=1000, valid_until_block=1359,
        reward_basis=basis, burn_hotkey=BASELINE, issued_at=basis["published_at"],
    )


def test_weight_state_90_rounds_matches_legacy_signed_bytes_without_catalogs():
    rows = []
    for index in range(90):
        row, signer = _reward(epoch=11 + index)
        row["configuration_doc"]["deepline_catalog"] = {"private-catalog": "x" * 1_000_000}
        # Preserve activation filtering, including activated replayed rounds.
        if index < 32:
            row[("reward_activated_at", "signing_key_doc")[index % 2]] = None
        if index == 89:
            row["status"] = "stage1"
        rows.append(row)
    latest = rows[-1]
    state = _accepted(signer, latest["reward_basis_doc"])
    original_rows = deepcopy(rows)
    old, old_transport = _weight_service(rows, signer, state, legacy=True)
    new, new_transport = _weight_service(rows, signer, state)

    assert new.public_weight_state(101) == old.public_weight_state(101) == {
        "state": state, "lookup_ok": True,
    }
    assert rows == original_rows
    assert len(old_transport.returned) == len(new_transport.returned) == 90
    assert len(json.dumps(old_transport.returned)) > 90_000_000
    assert len(json.dumps(new_transport.returned)) < 1_000_000
    assert "private-catalog" not in json.dumps(new_transport.returned)
    assert new_transport.calls[0]["columns"].split(",").count(PUBLIC_CONFIGURATION_COLUMN) == 1
    assert "configuration_doc" not in new_transport.calls[0]["columns"].split(",")
    assert {k: v for k, v in new_transport.calls[0].items() if k != "columns"} == {
        k: v for k, v in old_transport.calls[0].items() if k != "columns"
    }
    assert all({k: v for k, v in new_row.items() if k != "configuration_doc"} == {
        k: v for k, v in old_row.items() if k != "configuration_doc"
    } for new_row, old_row in zip(new_transport.returned, old_transport.returned))


@pytest.mark.parametrize("baseline", ["missing", None, "", ALICE, BASELINE])
@pytest.mark.parametrize("outcome", ["crowned", "no_king"])
def test_weight_projection_preserves_missing_baseline_and_no_king_barrier(baseline, outcome):
    row, signer = _reward(baseline=baseline, outcome=outcome)
    if baseline == "missing":
        row["configuration_doc"].pop("baseline_hotkey")
    state = _accepted(signer, row["reward_basis_doc"])
    old, _ = _weight_service([row], signer, state, legacy=True)
    new, _ = _weight_service([row], signer, state)
    assert new.public_weight_state(101) == old.public_weight_state(101)


@pytest.mark.parametrize("field", ["reward_activated_at", "reward_basis_doc", "signing_key_doc", "rewards_enabled"])
@pytest.mark.parametrize("missing", [False, True])
def test_weight_projection_keeps_activation_and_missing_field_filtering(field, missing):
    row, signer = _reward()
    state = _accepted(signer, row["reward_basis_doc"])
    if missing:
        row.pop(field)
    else:
        row[field] = None
    old, _ = _weight_service([row], signer, state, legacy=True)
    new, _ = _weight_service([row], signer, state)
    assert new.public_weight_state(101) == old.public_weight_state(101) == {
        "state": None, "lookup_ok": True,
    }


def test_weight_projection_retains_invalid_reward_history_failure():
    row, signer = _reward()
    state = _accepted(signer, row["reward_basis_doc"])
    row["reward_basis_doc"]["king_hotkey"] = BASELINE
    for legacy in (False, True):
        service, _ = _weight_service([row], signer, state, legacy=legacy)
        with pytest.raises(ServiceError, match="reward_history_invalid"):
            service.public_weight_state(101)


def test_weight_projection_retains_invalid_accepted_signature_failure():
    row, signer = _reward()
    state = _accepted(signer, row["reward_basis_doc"])
    state["signature"]["signature_b64"] = "AAAA"
    for legacy in (False, True):
        service, _ = _weight_service([row], signer, state, legacy=legacy)
        with pytest.raises(ServiceError, match="accepted_weight_state_invalid"):
            service.public_weight_state(101)


def test_weight_state_postgrest_and_psycopg_query_contracts():
    requests = []
    with httpx.Client(transport=httpx.MockTransport(
        lambda request: requests.append(request) or httpx.Response(200, json=[])
    )) as http:
        service, _ = _service([])
        service._store = ArenaStore(PostgrestTransport(
            "https://example.test", service_key="sb_secret_test", http_client=http,
        ))
        assert service.public_weight_state(101) == {"state": None, "lookup_ok": True}
    assert len(requests) == 2
    assert requests[1].url.path == '/rest/v1/rpc/lab_arena_correction_reward_history_v1'
    assert json.loads(requests[1].content) == {
        'p_mode': 'live', 'p_network_name': 'finney', 'p_netuid': 71,
        'p_limit': 200, 'p_offset': 0, 'p_public_only': True,
    }
    params = requests[0].url.params
    assert requests[0].url.path == "/rest/v1/lab_arena_rounds"
    assert params["select"] == (
        "round_id,status,arena_network_name,arena_netuid,rewards_enabled,effective_reward_epoch,"
        "king_outcome,king_hotkey,king_start_epoch,reward_basis_hash,reward_basis_doc,"
        "signing_key_doc,reward_activated_at,published_at," + PUBLIC_CONFIGURATION_COLUMN
    )
    assert params["configuration_doc->>mode"] == "eq.live"
    assert params["arena_network_name"] == "eq.finney" and params["arena_netuid"] == "eq.71"
    assert params["order"] == "effective_reward_epoch.desc"
    assert params["limit"] == "200" and params["offset"] == "0"
    queries = []

    class Cursor:
        def __enter__(self): return self
        def __exit__(self, *_args): pass
        def execute(self, sql, values): queries.append((sql, values))
        def fetchall(self): return []
        def fetchone(self): return ([],)

    direct = object.__new__(PsycopgTransport)
    direct._acquire = lambda: SimpleNamespace(cursor=lambda: Cursor())
    direct._release = lambda _connection: None
    service._store = ArenaStore(direct)
    assert service.public_weight_state(101) == {"state": None, "lookup_ok": True}
    assert len(queries) == 2
    assert 'public.lab_arena_correction_reward_history_v1(' in queries[1][0]
    assert queries[1][1] == ['live', 'finney', 71, 200, 0, True]
    sql, values = queries[0]
    assert "public.lab_arena_public_configuration_v1(lab_arena_rounds) AS configuration_doc" in sql
    assert "\"configuration_doc\"," not in sql
    assert "finney" in values and 71 in values and "live" in values
    assert sql.endswith(" ORDER BY effective_reward_epoch DESC LIMIT 200 OFFSET 0) t")
