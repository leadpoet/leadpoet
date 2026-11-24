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


def test_readback_uses_the_private_rpc_through_normal_scoped_transport():
    import httpx
    from lab_arena.store import PostgrestTransport

    def respond(request):
        assert request.method == "POST"
        assert request.url.path == "/rest/v1/rpc/collection_participation_v1"
        assert json.loads(request.content) == {}
        return httpx.Response(200, json={"enrollments": []})

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        db = PostgrestTransport("https://arena.example", service_key="sb_secret_test", http_client=client)
        assert db.rpc("collection_participation_v1", {}) == {"enrollments": []}


def test_retirement_before_paid_collection_requires_the_actual_burn_vectors(tmp_path):
    weights = {"epoch": 99, "weight_start_epoch": 100, "signed_state": True,
               "basis_schema": "leadpoet.collection.fixed_run_quotes.v1",
               "champion_ppb": 0, "quoted_jobs_match": True, "unweighted_funders": 0,
               "signed_burn_only": True, "validators": dict.fromkeys(("primary", "rizzo", "yuma"), "matches")}
    data = {"weights": weights, "collection": {"first_epoch": None}}
    proved = "retired champion weights verified at the designated burn hotkey on all three validators"
    inspect = "inspect burn-only retirement weights"
    assert not notices(data).get(proved)
    weights["epoch"] = 100
    assert notices(data).get(proved)
    save(tmp_path, data)
    first = json.loads((tmp_path / "status.json").read_text())["event"]
    weights["epoch"] = 101
    save(tmp_path, data)
    assert json.loads((tmp_path / "status.json").read_text())["event"] == first
    for field, wrong in (("signed_burn_only", False), ("basis_schema", "legacy"),
                         ("champion_ppb", 325000000), ("quoted_jobs_match", False)):
        altered = dict(weights, **{field: wrong})
        signals = notices(dict(data, weights=altered))
        assert signals.get(inspect) and not signals.get(proved)
    weights["validators"]["rizzo"] = "differs"
    assert notices(data).get(inspect) and not notices(data).get(proved)
    weights["validators"]["rizzo"] = "matches"
    data["collection"]["first_epoch"] = 102
    assert notices(data).get(proved)
    weights["epoch"] = 102
    weights["signed_burn_only"] = False
    signals = notices(data)
    assert signals["post-cutover collection basis and zero champion component verified on all three validators"]
    assert not signals.get(proved) and not signals.get(inspect)


def test_validator_snapshot_authenticates_original_burn_hotkey(monkeypatch):
    from hashlib import sha256
    from types import SimpleNamespace
    from lab_arena import chain, signing, store, weight_state, local_weight_signer
    from leadpoet_canonical import chain_source_v2, arena_weights
    from scripts.collection_participation_watch import VALIDATORS, validator_snapshot

    signer = signing.LocalSigner.generate()
    hotkeys = list(VALIDATORS.values())
    epoch = 100

    def signed(epoch_id, burn):
        return weight_state.build_accepted_weight_state(
            signer, network="finney", netuid=71, genesis_hash="1" * 64, epoch=epoch_id,
            valid_from_block=1000, valid_until_block=1359, burn_hotkey=burn,
            reward_basis={"schema_version": arena_weights.COLLECTION_BASIS_SCHEMA,
                          "source_epoch": epoch_id - 1, "job_count": 0,
                          "job_set_sha256": sha256(b"").hexdigest(), "quotes_microusd": {}, "valuation": None},
            slot_eligibility=None, issued_at="2026-11-25T00:00:00Z")

    states = {epoch: signed(epoch, hotkeys[0]), epoch-1: signed(epoch-1, hotkeys[0])}
    db = SimpleNamespace(rpc=lambda name, params: {k: v for k, v in states[epoch]["reward_basis"].items()
                                                   if k != "valuation"})
    snapshot = SimpleNamespace(current_block=1002, block_hash="0x" + "b" * 64, network_genesis_hash="0x" + "1" * 64,
                               settlement_epoch_id=lambda _: epoch)
    rpc = SimpleNamespace(call=lambda method, **kwargs: {"netuid": 71, "block": 1002, "hotkeys": hotkeys}
                          if method == "state_call" else [(0, 65535)])
    monkeypatch.setattr(chain, "connect_substrate", lambda _:None)
    monkeypatch.setattr(chain, "ArenaChain", lambda *_:SimpleNamespace(close=lambda:None))
    monkeypatch.setattr(chain, "finalized_epoch_snapshot", lambda _:snapshot)
    monkeypatch.setattr(chain, "load_arena_cutover", lambda _:None)
    monkeypatch.setattr(local_weight_signer, "HttpsJsonRpcTransport", lambda *a, **k:rpc)
    monkeypatch.setattr(chain_source_v2, "decode_selective_metagraph_result", lambda x:x)
    monkeypatch.setattr(chain_source_v2, "decode_weights_storage", lambda x:x)
    monkeypatch.setattr(signing, "KmsSigner", lambda *a, **k:signer)
    monkeypatch.setattr(store, "ArenaStore", lambda _:SimpleNamespace(collection_reward_start_epoch=lambda:epoch,
                        get_weight_state=lambda network, netuid, idx:{"state_doc":states[idx]}))
    env = {"LAB_ARENA_CHAIN_ENDPOINT": "wss://example.invalid", "LAB_ARENA_SIGNING_KEY_ID": "fixture"}
    healthy = validator_snapshot(env, db)
    assert healthy["signed_burn_only"] and healthy["quoted_jobs_match"]
    assert healthy["weight_start_epoch"] == epoch and set(healthy["validators"].values()) == {"matches"}
    states[epoch-1] = signed(epoch-1, hotkeys[1])
    assert not validator_snapshot(env, db)["signed_burn_only"]
