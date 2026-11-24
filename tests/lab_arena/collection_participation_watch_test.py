import json
from scripts.collection_participation_watch import notices, save, worker_snapshot


def test_idle_sampling_and_chain_head_updates_do_not_repeat_notifications(tmp_path):
    snapshot = {"edge": {}, "worker": {"result": "success", "timer_enabled_active": True,
                           "worker": "inactive", "recent_arms": {"comparison": "idle", "correction": "idle"},
                           "unit_failed_recently": False},
                "collection": {}, "weights": {"signed_state": True, "block": 1, "champion_ppb": 325000000}}
    def latest():
        return json.loads((tmp_path / "status.json").read_text())["event"]
    save(tmp_path, snapshot)
    before = latest()
    snapshot["weights"]["block"] += 1
    save(tmp_path, snapshot)
    assert latest() == before
    snapshot["collection"]["billing"] = [{"mode": "pilot", "arm": "correction", "funded_runs": 1, "unresolved": 0}]
    save(tmp_path, snapshot)
    event = latest()
    assert event["number"] == before["number"] + 1
    assert event["changes"] == {"pilot correction first billed execution": True}
    del snapshot["collection"]
    save(tmp_path, snapshot)
    assert latest()["changes"] == {"collection read unavailable": 1}
    save(tmp_path, snapshot)
    assert latest()["number"] == event["number"] + 1


def test_recent_failed_unit_still_actionable_after_idle_success(monkeypatch):
    def read(command, **kwargs):
        if command[0] == "journalctl":
            entries = [{"UNIT_RESULT": "exit-code", "MESSAGE": "Failed"},
                       {"MESSAGE": json.dumps({"status": "correction_held", "reason": "private details"})},
                       {"MESSAGE": json.dumps({"batch": "aa", "status": "idle"})}]
            return "\n".join(json.dumps(e) for e in entries)
        if command[2].endswith(".timer"):
            return "UnitFileState=enabled\nActiveState=active"
        return "Result=success\nActiveState=inactive"
    monkeypatch.setattr("subprocess.check_output", read)
    observed = worker_snapshot()
    signals = notices({"worker": observed, "weights": {"signed_state": False}, "collection": {}})
    assert signals["inspect pilot unit and timer"]
    assert signals["inspect correction admission hold"]
    assert signals["current signed state pending; inspect validator publication"]
    assert "private details" not in str(observed)


def test_desktop_retries_failed_delivery_and_only_notifies_on_change(tmp_path):
    import pytest
    from scripts.collection_watch_desktop import deliver

    calls = []
    snapshot = {"event": {"number": 1, "changes": {}}}
    def unavailable():
        raise OSError("no readback")
    def send_failure(_):
        raise OSError("not delivered")
    with pytest.raises(OSError):
        deliver(tmp_path, lambda: snapshot, send_failure)
    assert not (tmp_path / "latest.json").exists()
    for result in (lambda: snapshot, lambda: snapshot, unavailable, unavailable, lambda: snapshot):
        deliver(tmp_path, result, calls.append)
    assert len(calls) == 3
    assert "unavailable" in calls[1]
    assert "restored" in calls[2]


def test_zero_champion_share_requires_collection_job_basis_and_all_three_vectors():
    weights = {"epoch": 100, "signed_state": True, "basis_schema": "legacy",
               "champion_ppb": 0, "quoted_jobs_match": True, "unweighted_funders": 0,
               "validators": {"primary": "matches", "rizzo": "matches", "yuma": "matches"}}
    data = {"collection": {"first_epoch": 100}, "weights": weights}
    verified = "post-cutover collection basis and zero champion component verified on all three validators"
    assert not notices(data).get(verified)
    weights["basis_schema"] = "leadpoet.collection.fixed_run_quotes.v1"
    assert notices(data)[verified]
    weights["validators"]["rizzo"] = "differs"
    assert notices(data)["inspect post-cutover reward evidence"] and not notices(data).get(verified)


def test_edge_counts_only_allowlisted_new_complete_responses_and_detects_rotation(tmp_path):
    from scripts.collection_participation_watch import edge_snapshot
    path = tmp_path / "arena-edge-slot0.log"
    path.write_bytes(b'old POST /arena/v1/collection/register HTTP/1.1" 400\n')
    prior = edge_snapshot(tmp_path, {})
    assert prior["http"] == {}
    with path.open("ab") as f:
        f.write(b'private-peer "GET /arena/v1/collection/terms?private-value HTTP/1.1" 200 OK\n')
        f.write(b'"POST /arena/v1/collection/register HTTP/1.1"')
    first = edge_snapshot(tmp_path, prior)
    assert first["http"] == {"GET terms 200": 1}
    with path.open("ab") as f:
        f.write(b' 400 bad\n"PUT /arena/v1/collection/terms HTTP/1.1" 501\n')
    second = edge_snapshot(tmp_path, first)
    assert second["http"] == {"GET terms 200": 1, "POST register 400": 1}
    assert edge_snapshot(tmp_path, second) == second and "private" not in str(second)
    path.rename(tmp_path / "old")
    path.write_bytes(b'"POST /arena/v1/collection/register HTTP/1.1" 200 OK\n')
    third = edge_snapshot(tmp_path, second)
    assert third["windows_with_gap"] == 1 and third["http"]["POST register 200"] == 1
