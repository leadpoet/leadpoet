"""Store-level behavior that does not need a database: batched score recording."""

from __future__ import annotations

import httpx
import pytest

from lab_arena.store import ArenaStore, ArenaStoreError, FUNCTION_SIGNATURES, PostgrestTransport, PsycopgTransport, SCORE_BATCH_SIZE, TABLES, create_http1_client
from lab_arena.store import ArenaStoreUnavailable


@pytest.mark.parametrize("error_type", [httpx.ReadTimeout, httpx.ReadError])
@pytest.mark.parametrize("recovers", [True, False])
def test_select_retries_one_read_failure_without_changing_query(error_type, recovers):
    requests = []
    rows = [{"round_id": "arena-2026-09-10", "status": "published"}]

    def handler(request):
        requests.append(request)
        if len(requests) == 1 or not recovers:
            raise error_type("private transport diagnostic", request=request)
        return httpx.Response(200, json=rows)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        transport = PostgrestTransport("https://project.example", anon_key="anon", service_jwt="a.b.c", http_client=client)
        kwargs = dict(filters={"status": "published"}, order="created_at", descending=True, limit=5, offset=20, columns="round_id,status")
        if recovers:
            assert transport.select("lab_arena_rounds", **kwargs) == rows
        else:
            with pytest.raises(ArenaStoreUnavailable) as caught:
                transport.select("lab_arena_rounds", **kwargs)
            assert isinstance(caught.value.__cause__, error_type)
            assert "private transport diagnostic" not in str(caught.value)
    assert len(requests) == 2
    assert requests[0].method == requests[1].method == "GET"
    assert requests[0].url == requests[1].url
    assert requests[0].headers == requests[1].headers


@pytest.mark.parametrize("response", [
    httpx.Response(403, json={"message": "permission denied"}),
    httpx.Response(200, json={"unexpected": "object"}),
    httpx.Response(200, content=b"invalid JSON"),
])
def test_select_does_not_retry_authorization_or_contract_errors(response):
    requests = []

    def handler(request):
        requests.append(request)
        return response

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        transport = PostgrestTransport("https://project.example", anon_key="anon", service_jwt="a.b.c", http_client=client)
        with pytest.raises((ArenaStoreError, ValueError)) as caught:
            transport.select("lab_arena_rounds")
        assert not isinstance(caught.value, ArenaStoreUnavailable)
    assert len(requests) == 1


@pytest.mark.parametrize("error_type", [httpx.ReadTimeout, httpx.ReadError])
def test_rpc_does_not_retry_ambiguous_read_failure(error_type):
    requests = []

    def handler(request):
        requests.append(request)
        raise error_type("private transport diagnostic", request=request)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        transport = PostgrestTransport("https://project.example", anon_key="anon", service_jwt="a.b.c", http_client=client)
        with pytest.raises(ArenaStoreError) as caught:
            transport.rpc("lab_arena_cancel_round", {"p_round_id": "arena-2026-09-10", "p_reason": "test"})
        assert not isinstance(caught.value, ArenaStoreUnavailable)
    assert len(requests) == 1 and requests[0].method == "POST"


class RecordingTransport:
    def __init__(self, *, stale_at_batch=None):
        self.calls = []
        self.stale_at_batch = stale_at_batch

    def rpc(self, function, params):
        self.calls.append((function, params))
        if self.stale_at_batch is not None and len(self.calls) == self.stale_at_batch:
            return {"status": "stale", "round_status": "scored"}
        count = len(params["p_scores"])
        return {"status": "ok", "recorded": max(0, count - 1), "existing": 1 if count else 0}

    def select(self, *args, **kwargs):  # pragma: no cover - unused
        raise AssertionError("no selects expected")

    def close(self):
        pass


def scores(count):
    return [{"run_id": "run-%05d" % index, "per_icp_score": 50.0} for index in range(count)]


def test_scores_are_written_in_bounded_idempotent_batches():
    transport = RecordingTransport()
    store = ArenaStore(transport)
    total = store.record_run_scores("arena-2026-09-02", 1, scores(12_850))
    assert total["batches"] == 26 and total["recorded"] == 12_850 - 26 and total["existing"] == 26
    sizes = [len(params["p_scores"]) for _function, params in transport.calls]
    assert max(sizes) == SCORE_BATCH_SIZE and sum(sizes) == 12_850 and sizes[-1] == 350
    assert all(function == "lab_arena_record_run_scores" and params["p_stage"] == 1 for function, params in transport.calls)
    # Order is preserved across batches so a partial write is resumable.
    assert transport.calls[0][1]["p_scores"][0]["run_id"] == "run-00000" and transport.calls[-1][1]["p_scores"][-1]["run_id"] == "run-12849"


def test_an_empty_stage_still_makes_one_status_checked_call():
    transport = RecordingTransport()
    total = ArenaStore(transport).record_run_scores("arena-2026-09-02", 2, [])
    assert total == {"status": "ok", "recorded": 0, "existing": 0, "batches": 1} and len(transport.calls) == 1


def test_a_stale_round_stops_the_batches_and_surfaces_the_status():
    transport = RecordingTransport(stale_at_batch=2)
    result = ArenaStore(transport).record_run_scores("arena-2026-09-02", 1, scores(1_200))
    assert result["status"] == "stale" and len(transport.calls) == 2


def test_batch_size_must_be_positive():
    with pytest.raises(ArenaStoreError):
        ArenaStore(RecordingTransport()).record_run_scores("arena-2026-09-02", 1, scores(3), batch_size=0)


class ShapeTransport:
    def __init__(self):
        self.calls = []

    def rpc(self, function, params):
        self.calls.append((function, params))
        return {"status": "ok"}

    def select(self, *args, **kwargs):  # pragma: no cover - unused
        raise AssertionError("no selects expected")

    def close(self):
        pass


def test_simple_stage_and_completion_rpc_shapes():
    transport = ShapeTransport()
    store = ArenaStore(transport)
    store.open_stage("arena-2026-09-02", 1, [{"submission_id": "s1", "miner_hotkey": "h"}], [0, 1])
    store.complete_attempt(
        run_id="run-1",
        lease_token_hash="sha256:" + "1" * 64,
        result={"terminal_status": "accepted"},
        terminal_cause="accepted",
        output_ref="arena/round/run-1.json",
    )
    assert transport.calls == [
        (
            "lab_arena_open_stage",
            {
                "p_round_id": "arena-2026-09-02",
                "p_stage": 1,
                "p_participants": [{"submission_id": "s1", "miner_hotkey": "h"}],
                "p_icp_positions": [0, 1],
            },
        ),
        (
            "lab_arena_complete_attempt",
            {
                "p_run_id": "run-1",
                "p_lease_token_hash": "sha256:" + "1" * 64,
                "p_result": {"terminal_status": "accepted"},
                "p_terminal_cause": "accepted",
                "p_output_ref": "arena/round/run-1.json",
            },
        ),
    ]


def test_removed_durable_state_has_no_store_boundary():
    assert TABLES == (
        "lab_arena_rounds",
        "lab_arena_submissions",
        "lab_arena_runs",
        "lab_arena_ledger",
    )
    for removed in (
        "lab_arena_append_generation_attempt",
        "lab_arena_upsert_account_credential",
        "lab_arena_record_preflight",
    ):
        assert removed not in FUNCTION_SIGNATURES


def test_postgrest_rejects_http_loopback_lookalikes():
    for url in (
        "http://localhost.evil.example",
        "http://127.0.0.1.evil.example",
        "http://[::2]",
        "https://user:password@example.com",
    ):
        with pytest.raises(ArenaStoreError, match="base URL"):
            PostgrestTransport(url, anon_key="anon", service_jwt="a.b.c")


def test_production_http_client_ignores_proxy_environment_and_redirects(monkeypatch):
    captured = {}
    marker = object()

    def build(**kwargs):
        captured.update(kwargs)
        return marker

    monkeypatch.setattr(httpx, "Client", build)
    assert create_http1_client(8) is marker
    assert captured["follow_redirects"] is False
    assert captured["trust_env"] is False


def test_postgrest_does_not_follow_a_cross_origin_redirect():
    contacted = []

    def handler(request):
        contacted.append(str(request.url))
        return httpx.Response(307, headers={"location": "https://attacker.example/collect"})

    client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False, trust_env=False)
    transport = PostgrestTransport("https://project.example", anon_key="anon", service_jwt="a.b.c", http_client=client)
    with pytest.raises(ArenaStoreError, match="HTTP 307"):
        transport.rpc("lab_arena_whoami", {})
    assert contacted == ["https://project.example/rest/v1/rpc/lab_arena_whoami"]
    transport.close()


def test_postgrest_filters_round_mode_and_status_before_limit_with_pagination():
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=[])

    client = httpx.Client(
        transport=httpx.MockTransport(handler), follow_redirects=False, trust_env=False
    )
    transport = PostgrestTransport(
        "https://project.example",
        anon_key="anon",
        service_jwt="a.b.c",
        http_client=client,
    )
    rows = transport.select(
        "lab_arena_rounds",
        filters={
            "configuration_doc->>mode": "live",
            "publication_doc->king_decision->>outcome": "crowned",
            "arena_network_name": "test",
            "arena_netuid": 401,
        },
        status_in=("open", "committed"),
        order="created_at",
        descending=True,
        limit=20,
        offset=40,
    )
    assert rows == []
    params = list(requests[0].url.params.multi_items())
    assert ("configuration_doc->>mode", "eq.live") in params
    assert ("publication_doc->king_decision->>outcome", "eq.crowned") in params
    assert ("arena_network_name", "eq.test") in params
    assert ("arena_netuid", "eq.401") in params
    assert ("status", "in.(open,committed)") in params
    assert ("order", "created_at.desc") in params
    assert ("limit", "20") in params and ("offset", "40") in params
    transport.close()


@pytest.mark.parametrize(
    "filters,statuses",
    [
        ({"configuration_doc->>mode": "live,or(status.eq.open)"}, ("open",)),
        ({"configuration_doc->>mode": "live"}, ("open,published",)),
    ],
)
def test_postgrest_round_filters_reject_reserved_query_syntax(filters, statuses):
    client = httpx.Client(
        transport=httpx.MockTransport(lambda _request: httpx.Response(200, json=[])),
        follow_redirects=False,
        trust_env=False,
    )
    transport = PostgrestTransport(
        "https://project.example",
        anon_key="anon",
        service_jwt="a.b.c",
        http_client=client,
    )
    with pytest.raises(ArenaStoreError, match="reserved characters"):
        transport.select(
            "lab_arena_rounds", filters=filters, status_in=statuses, limit=20
        )
    transport.close()


def test_psycopg_parameterizes_round_mode_and_status_before_limit():
    calls = []

    class Cursor:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def execute(self, sql, values):
            calls.append((sql, values))

        def fetchall(self):
            return []

    class Connection:
        closed = 0
        autocommit = False

        @staticmethod
        def cursor():
            return Cursor()

        @staticmethod
        def close():
            return None

    transport = PsycopgTransport(lambda: Connection(), role=None)
    assert transport.select(
        "lab_arena_rounds",
        filters={
            "configuration_doc->>mode": "live",
            "publication_doc->king_decision->>outcome": "crowned",
            "arena_network_name": "test",
            "arena_netuid": 401,
        },
        status_in=("open", "committed"),
        order="created_at",
        descending=True,
        limit=20,
        offset=40,
    ) == []
    sql, values = calls[0]
    assert "configuration_doc ->> 'mode' = %s" in sql
    assert "publication_doc #>> '{king_decision,outcome}' = %s" in sql
    assert "arena_network_name = %s" in sql
    assert "arena_netuid = %s" in sql
    assert "status = ANY(%s)" in sql
    assert sql.index(" WHERE ") < sql.index(" ORDER BY ") < sql.index(" LIMIT 20")
    assert sql.endswith(" OFFSET 40) t")
    assert values == ["live", "crowned", "test", 401, ["open", "committed"]]
    assert "live" not in sql and "crowned" not in sql and "committed" not in sql
    transport.close()


def test_round_store_pushes_mode_and_status_filters_into_the_bounded_read():
    calls = []

    class Transport:
        @staticmethod
        def select(table, **kwargs):
            calls.append((table, kwargs))
            return []

        @staticmethod
        def close():
            return None

    store = ArenaStore(Transport())
    store.list_rounds(
        statuses=("open", "committed"), mode="live",
        network_name="test", netuid=401, limit=20, offset=20
    )
    store.published_reward_bases(
        mode="live", network_name="test", netuid=401, limit=200
    )
    active = calls[0][1]
    rewards = calls[1][1]
    assert active["filters"] == {
        "configuration_doc->>mode": "live",
        "arena_network_name": "test",
        "arena_netuid": 401,
    }
    assert active["status_in"] == ("open", "committed")
    assert active["limit"] == 20 and active["offset"] == 20
    assert rewards["filters"] == {
        "status": "published",
        "configuration_doc->>mode": "live",
        "arena_network_name": "test",
        "arena_netuid": 401,
    }
    assert rewards["limit"] == 200


def test_round_store_requires_the_network_filter_pair():
    class Transport:
        @staticmethod
        def select(*_args, **_kwargs):
            raise AssertionError("invalid filters reached the transport")

    store = ArenaStore(Transport())
    with pytest.raises(ArenaStoreError, match="supplied together"):
        store.list_rounds(mode="live", network_name="test")
    with pytest.raises(ArenaStoreError, match="supplied together"):
        store.published_reward_bases(mode="live", network_name="test")
    with pytest.raises(ArenaStoreError, match="supplied together"):
        store.list_rounds(mode="live", netuid=401)


def test_pending_promotions_pushes_durable_filters_and_oldest_order():
    calls = []

    class Transport:
        @staticmethod
        def select(table, **kwargs):
            calls.append((table, kwargs))
            return [
                {"round_id": "winner", "publication_doc": {"king_decision": {"outcome": "crowned"}}},
            ]

    rows = ArenaStore(Transport()).pending_promotions(
        pinned_round_id="winner", network_name="test", netuid=401, limit=3
    )
    assert [row["round_id"] for row in rows] == ["winner"]
    assert calls == [
        (
            "lab_arena_rounds",
            {
                "filters": {
                    "status": "published",
                    "configuration_doc->>mode": "live",
                    "promotion_required": True,
                    "baseline_promoted_at": None,
                    "publication_doc->king_decision->>outcome": "crowned",
                    "round_id": "winner",
                    "arena_network_name": "test",
                    "arena_netuid": 401,
                },
                "order": "created_at",
                "descending": False,
                "limit": 3,
                "columns": "round_id,publication_doc,promotion_doc,published_at,created_at",
            },
        )
    ]
