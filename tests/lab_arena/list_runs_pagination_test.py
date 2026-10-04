"""A full Arena round stays visible through a capped PostgREST read."""

import httpx

from lab_arena.store import ArenaStore, PostgrestTransport


def test_full_round_runs_survive_postgrest_row_cap():
    runs = [
        {
            "round_id": "arena-full-day",
            "run_id": f"run-{kind}-{stage}-{index:04d}",
            "stage": stage,
            "kind": kind,
            "status": "accepted" if index % 2 == 0 else "pending",
            "submission_id": f"sub-{index // 10:03d}",
        }
        for kind in ("execute", "score")
        for stage in (1, 2)
        for index in range(2570)
    ]
    requests = []

    def respond(request):
        params = request.url.params
        requests.append(params)
        selected = [
            row for row in runs
            if all(
                row[key] > value[3:] if key == "run_id" else
                row[key] == (int(value[3:]) if key == "stage" else value[3:])
                for key, value in params.multi_items()
                if key in ("round_id", "stage", "kind", "status", "submission_id", "run_id")
            )
        ]
        selected.sort(key=lambda row: row["run_id"])
        limit = min(int(params.get("limit", "1000")), 1000)
        return httpx.Response(200, json=selected[:limit])

    client = httpx.Client(transport=httpx.MockTransport(respond))
    store = ArenaStore(PostgrestTransport(
        "http://127.0.0.1:3000", service_key="sb_secret_test",
        http_client=client,
    ))
    try:
        stage_runs = store.list_runs("arena-full-day", stage=1, kind="execute")
        assert len(stage_runs) == 2570
        assert stage_runs == sorted(stage_runs, key=lambda row: row["run_id"])
        assert len(store.list_runs("arena-full-day", kind="score")) == 5140
        assert len(store.list_runs("arena-full-day", stage=2, status="accepted", kind="execute")) == 1285
        assert len(store.list_runs("arena-full-day")) == 10280
        assert all(int(params["limit"]) <= 500 for params in requests)
    finally:
        store.close()


def test_run_cursor_does_not_skip_later_rows_after_earlier_rows_disappear():
    rows = [{"round_id": "arena-changing", "run_id": f"run-{index:04d}"} for index in range(1001)]
    calls = 0

    def respond(request):
        nonlocal calls
        calls += 1
        if calls == 2:
            del rows[:100]
            rows.append({"round_id": "arena-changing", "run_id": "run-0500a"})
        cursor = request.url.params.get("run_id", "")
        after = cursor[3:] if cursor else ""
        selected = sorted(
            (row for row in rows if row["run_id"] > after),
            key=lambda row: row["run_id"],
        )
        return httpx.Response(200, json=selected[:500])

    client = httpx.Client(transport=httpx.MockTransport(respond))
    store = ArenaStore(PostgrestTransport(
        "http://127.0.0.1:3000", service_key="sb_secret_test",
        http_client=client,
    ))
    try:
        ids = [row["run_id"] for row in store.list_runs("arena-changing")]
        assert len(ids) == 1002
        assert "run-0500a" in ids
        assert ids == sorted(set(ids))
    finally:
        store.close()
