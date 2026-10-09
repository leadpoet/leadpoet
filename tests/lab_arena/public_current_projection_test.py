"""Current polling uses compact reads; driver and signed weight reads do not."""

from copy import deepcopy
import json
from types import SimpleNamespace

import httpx
import pytest

from lab_arena import contact_policy, contracts, integrity, intent_details_policy, quality_policy, rewards, signing
from lab_arena.service import ArenaService, ServiceError
from lab_arena.store import ArenaStore, CURRENT_CONFIGURATION_FIELDS, PostgrestTransport, PsycopgTransport
from leadpoet_canonical.lab_arena_rewards import verify_reward_basis_signature
from tests.lab_arena.competition_summary_projection_test import _project


ALICE, BASELINE = ["5" + letter * 47 for letter in "AB"]


class CurrentTransport:
    def __init__(self, rows, *, full=False):
        self.rows, self.full, self.calls, self.returned = rows, full, [], []

    def select(self, table, **query):
        assert table == "lab_arena_rounds"
        self.calls.append(query)
        rows = [row for row in self.rows if all(
            (row["configuration_doc"].get("mode") if key == "configuration_doc->>mode"
             else row.get(key)) == value for key, value in (query.get("filters") or {}).items()
        )]
        if query.get("status_in"):
            rows = [row for row in rows if row["status"] in query["status_in"]]
        if query.get("order"):
            rows.sort(key=lambda row: row.get(query["order"]) or -1, reverse=query.get("descending", False))
        start = query.get("offset") or 0
        result = [_project(row, "*" if self.full else query.get("columns", "*"))
                  for row in rows[start:start + query["limit"]]]
        self.returned.extend(result)
        return result


def _active(index=0, **configuration):
    return {
        "round_id": "arena-active-%02d" % index, "status": "open" if index == 0 else "stage1",
        "created_at": "2026-10-09T%02d:00:00Z" % index,
        "arena_network_name": "finney", "arena_netuid": 71,
        "configuration_doc": {
            "mode": "live", "schedule": {"submission_cutoff": "2026-10-10T00:00:00Z"},
            "deepline_catalog": {"private-catalog": "x" * 100_000}, **configuration,
        },
    }


def _reward(*, epoch=100, outcome="crowned", baseline=BASELINE, legacy=False, slots=False):
    signer = signing.LocalSigner.generate()
    basis = rewards.reward_basis_document(
        round_id="arena-reward-%d" % epoch, published_at="2026-10-09T00:00:00Z",
        finalized_epoch=epoch - 1, king_outcome=outcome,
        king_hotkey="" if outcome == "no_king" else ALICE,
        **({"slot_policy": rewards.reward_slot_policy_document(), "reward_slots": [None, None, None]} if slots else {}),
    )
    if legacy:
        basis.pop("champion_reward_factor_ppm")
        basis.pop("reward_basis_hash")
        basis["reward_basis_hash"] = contracts.document_hash(basis)
    signed = signing.sign_document(signer, basis, hash_field="reward_basis_hash")
    row = _active()
    row.update(round_id=basis["round_id"], status="published", published_at=basis["published_at"],
               rewards_enabled=True, reward_activated_at="2026-10-09T00:00:01Z",
               effective_reward_epoch=epoch, king_outcome=outcome, king_hotkey=basis["king_hotkey"],
               king_start_epoch=basis["king_start_epoch"], reward_basis_hash=signed["reward_basis_hash"],
               reward_basis_doc=signed, signing_key_doc=signing.signing_key_document(signer.public_key_der))
    row["configuration_doc"]["baseline_hotkey"] = baseline
    return row, signer


def _service(rows, *, full=False, epoch=101, mode="live", pinned=None):
    service = object.__new__(ArenaService)
    transport = CurrentTransport(rows, full=full)
    service._store = ArenaStore(transport)

    def current_epoch():
        if epoch is None:
            raise RuntimeError("epoch unavailable")
        return epoch

    service._config = SimpleNamespace(mode=mode, pinned_round_id=pinned,
                                      chain=SimpleNamespace(current_settlement_epoch=current_epoch))
    return service, transport


@pytest.mark.parametrize("epoch", [99, 101, None])
@pytest.mark.parametrize("legacy, slots", [(False, False), (True, False), (False, True)])
def test_current_response_matches_full_rows_and_preserves_signed_governing_basis(epoch, legacy, slots):
    published, signer = _reward(legacy=legacy, slots=slots)
    active = _active(stage_1_icp_count=5, stage_2_icp_count=5, promotion_margin=1.5,
                     integrity_policy=integrity.POLICY, contact_policy=contact_policy.POLICY,
                     company_quality_policy=quality_policy.POLICY,
                     intent_details_policy=intent_details_policy.POLICY)
    rows = [active, published]
    before = deepcopy(rows)
    original, _ = _service(rows, full=True, epoch=epoch)
    candidate, transport = _service(rows, epoch=epoch)
    expected = original.public_current()
    assert candidate.public_current() == expected
    assert rows == before
    assert expected["round"]["benchmark_icp_count"] == 10
    assert expected["round"]["output_schema_version"] == intent_details_policy.OUTPUT_SCHEMA
    assert "private-catalog" not in json.dumps(transport.returned)
    assert all("configuration_doc" not in row for row in transport.returned)
    assert all("configuration_doc" not in call["columns"].split(",") for call in transport.calls)
    if epoch is not None:
        full_basis = original.public_reward_basis(epoch)
        compact_basis = candidate.public_reward_basis(epoch, public_only=True)
        assert compact_basis == full_basis
        if compact_basis:
            assert verify_reward_basis_signature(compact_basis, public_key_der=signer.public_key_der,
                expected_public_key_hash=signer.public_key_hash) == published["reward_basis_hash"]
    reward_rows = candidate._store.published_reward_bases(mode="live", network_name="finney", netuid=71, public_only=True)
    assert reward_rows[0]["reward_basis_doc"] == published["reward_basis_doc"]
    assert reward_rows[0]["signing_key_doc"] == published["signing_key_doc"]


@pytest.mark.parametrize("baseline", [None, "", ALICE])
@pytest.mark.parametrize("outcome", ["crowned", "no_king"])
def test_current_preserves_organizer_rejection_and_no_king_continuity(baseline, outcome):
    prior, _ = _reward(epoch=99)
    latest, _ = _reward(epoch=100, baseline=baseline, outcome=outcome)
    original, _ = _service([prior, latest], full=True)
    candidate, _ = _service([prior, latest])
    assert candidate.public_current() == original.public_current()
    assert candidate.public_reward_basis(101, public_only=True) == original.public_reward_basis(101)
    if outcome == "no_king":
        assert candidate.public_current()["king"]["outcome"] == "no_king"


def test_missing_baseline_hotkey_matches_full_rejection():
    row, _ = _reward()
    row["configuration_doc"].pop("baseline_hotkey")
    full, _ = _service([row], full=True)
    compact, _ = _service([row])
    assert compact.public_current() == full.public_current()
    assert compact.public_current()["king"] is None


@pytest.mark.parametrize("key", CURRENT_CONFIGURATION_FIELDS + ("baseline_hotkey",))
@pytest.mark.parametrize("value", [None, False, 0, 1.5, "null", [], {"nested": [1, None]}])
def test_current_configuration_preserves_all_json_types_and_absent_keys(key, value):
    columns = "cfg_%s:configuration_doc->%s::text" % (key, key)
    projected = _project({"configuration_doc": {key: value}}, columns)
    assert ArenaService._public_current_round(projected)["configuration_doc"] == {key: value}
    assert ArenaService._public_current_round(_project({"configuration_doc": {}}, columns))["configuration_doc"] == {}


@pytest.mark.parametrize("key", ["stage_1_icp_count", "promotion_margin", "integrity_policy", "contact_policy", "company_quality_policy", "intent_details_policy"])
def test_current_explicit_null_configuration_keeps_failure(key):
    rows = [_active(**{key: None})]
    full, _ = _service(rows, full=True)
    compact, _ = _service(rows)
    with pytest.raises(ValueError) as expected:
        full.public_current()
    with pytest.raises(type(expected.value), match=str(expected.value)):
        compact.public_current()


def test_current_invalid_basis_still_fails_closed():
    row, _ = _reward()
    row["reward_basis_doc"]["king_hotkey"] = BASELINE
    for full in (True, False):
        service, _ = _service([row], full=full)
        with pytest.raises(ServiceError, match="reward_history_invalid"):
            service.public_current()


@pytest.mark.parametrize("status", ["open", "stage1", "published", "cancelled"])
def test_current_pinned_round_preserves_full_read_and_selection(status):
    row, _ = _reward()
    row["status"] = status
    full, _ = _service([row], full=True, pinned=row["round_id"], mode="live")
    compact, transport = _service([row], pinned=row["round_id"], mode="live")
    assert compact.public_current() == full.public_current()
    pinned_reads = [query for query in transport.calls if query.get("filters", {}).get("round_id")]
    assert len(pinned_reads) == 2
    assert all(query.get("columns", "*") == "*" for query in pinned_reads)


def test_current_legacy_empty_shadow_and_scoped_paging():
    rows = [_active(index) for index in range(21)]
    wrong_mode = _active(22, mode="shadow")
    wrong_chain = _active(23)
    wrong_chain["arena_netuid"] = 72
    full, _ = _service(rows + [wrong_mode, wrong_chain], full=True, epoch=None)
    compact, transport = _service(rows + [wrong_mode, wrong_chain], epoch=None)
    assert compact.public_current() == full.public_current()
    assert compact.active_rounds(public_only=True) == full.active_rounds()
    active_reads = [query for query in transport.calls if query.get("status_in")]
    assert [query["offset"] for query in active_reads] == [0, 20, 0, 20]
    assert all(query["filters"] == {"configuration_doc->>mode": "live", "arena_network_name": "finney", "arena_netuid": 71} for query in active_reads)
    shadow, _ = _service([wrong_mode], mode="shadow", epoch=None)
    assert shadow.public_current()["king"] is None
    empty, _ = _service([])
    assert empty.public_current()["published_round"] is None


def test_default_driver_reward_and_weight_reads_keep_full_configuration():
    row, _ = _reward()
    service, transport = _service([_active(), row])
    service.active_rounds()
    assert transport.calls[-1]["columns"] == "round_id,status,created_at,configuration_doc"
    service.latest_published_round()
    assert transport.calls[-1]["columns"] == "round_id,status,published_at,configuration_doc,king_outcome,king_hotkey,effective_reward_epoch,reward_basis_hash"
    assert service.public_reward_basis(101) == row["reward_basis_doc"]
    assert transport.calls[-1]["columns"].endswith(",configuration_doc")
    assert service.public_weight_state(99) == {"state": None, "lookup_ok": True}
    assert transport.calls[-1]["columns"].endswith(",configuration_doc")
    captured = []

    def full_store_only(**kwargs):
        assert "public_only" not in kwargs
        captured.append(kwargs)
        return [row]

    service._store = SimpleNamespace(published_reward_bases=full_store_only)
    assert service.public_reward_basis(101) == row["reward_basis_doc"]
    assert len(captured) == 1


def test_reward_paging_filters_and_signed_metadata_match_full_read():
    first, _ = _reward(epoch=100)
    second, _ = _reward(epoch=99)
    incomplete, _ = _reward(epoch=101)
    incomplete["signing_key_doc"] = None
    rows = [first, second, incomplete]
    service, transport = _service(rows)
    full = service._store.published_reward_bases(mode="live", network_name="finney", netuid=71, limit=2)
    full_queries = deepcopy(transport.calls)
    transport.calls.clear()
    compact = service._store.published_reward_bases(mode="live", network_name="finney", netuid=71, limit=2, public_only=True)
    assert [row["round_id"] for row in compact] == [row["round_id"] for row in full]
    assert [query["offset"] for query in transport.calls] == [0, 2]
    for original, candidate in zip(full, compact):
        assert {key: value for key, value in candidate.items() if not key.startswith("cfg_")} == {key: value for key, value in original.items() if key != "configuration_doc"}
    for original, candidate in zip(full_queries, transport.calls):
        assert {key: value for key, value in candidate.items() if key != "columns"} == {key: value for key, value in original.items() if key != "columns"}


def test_compact_postgrest_select_and_fixed_psycopg_aliases():
    requests = []
    with httpx.Client(transport=httpx.MockTransport(lambda request: requests.append(request) or httpx.Response(200, json=[]))) as http:
        service, _ = _service([])
        service._store = ArenaStore(PostgrestTransport("https://example.test", service_key="sb_secret_test", http_client=http))
        service.public_current()
    assert len(requests) == 3
    expected_fields = [CURRENT_CONFIGURATION_FIELDS, ("mode",), ("mode", "baseline_hotkey")]
    for request, fields in zip(requests, expected_fields):
        params = request.url.params
        assert "configuration_doc" not in params["select"].split(",")
        assert {column for column in params["select"].split(",") if ":configuration_doc" in column} == {"cfg_%s:configuration_doc->%s::text" % (key, key) for key in fields}
        assert params["configuration_doc->>mode"] == "eq.live"
        assert params["arena_network_name"] == "eq.finney" and params["arena_netuid"] == "eq.71"
    assert requests[0].url.params["limit"] == "20"
    assert requests[1].url.params["limit"] == "1"
    assert requests[2].url.params["limit"] == "200"
    queries = []

    class Cursor:
        def __enter__(self): return self
        def __exit__(self, *_args): pass
        def execute(self, sql, values): queries.append((sql, values))
        def fetchall(self): return []

    direct = object.__new__(PsycopgTransport)
    direct._acquire = lambda: SimpleNamespace(cursor=lambda: Cursor())
    direct._release = lambda _connection: None
    service._store = ArenaStore(direct)
    service.public_current()
    for (sql, values), fields in zip(queries, expected_fields):
        for key in fields:
            assert "(configuration_doc -> '%s')::text AS cfg_%s" % (key, key) in sql
        assert ":configuration_doc" not in sql
        assert "finney" in values and 71 in values and "live" in values
