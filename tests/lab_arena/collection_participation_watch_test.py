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
    snapshot = SimpleNamespace(current_block=1002, last_epoch_block=1000, block_hash="0x" + "b" * 64, network_genesis_hash="0x" + "1" * 64,
                               settlement_epoch_id=lambda _: epoch)
    def reply(method, **kwargs):
        if method == "chain_getFinalizedHead":
            return snapshot.block_hash
        if method == "state_call":
            return {"netuid": 71, "block": 1002, "hotkeys": hotkeys}
        return [(0, 65535)]
    rpc = SimpleNamespace(call=reply, close=lambda: None)
    monkeypatch.setattr(chain, "connect_substrate", lambda _:None)
    monkeypatch.setattr(chain, "ArenaChain", lambda *_:SimpleNamespace(close=lambda:None))
    monkeypatch.setattr(chain, "finalized_epoch_snapshot", lambda _:snapshot)
    monkeypatch.setattr(chain, "load_arena_cutover", lambda _:None)
    monkeypatch.setattr(local_weight_signer, "HttpsJsonRpcTransport", lambda *a, **k:rpc)
    monkeypatch.setattr("lab_arena.collection_distributions.decode_selective_metagraph_result", lambda x:x)
    monkeypatch.setattr("lab_arena.collection_distributions.decode_weights_storage", lambda x:x)
    monkeypatch.setattr(signing, "KmsSigner", lambda *a, **k:signer)
    monkeypatch.setattr(store, "ArenaStore", lambda _:SimpleNamespace(collection_reward_start_epoch=lambda:epoch,
                        get_weight_state=lambda network, netuid, idx:{"state_doc":states[idx]}))
    env = {"LAB_ARENA_CHAIN_ENDPOINT": "wss://example.invalid", "LAB_ARENA_SIGNING_KEY_ID": "fixture"}
    healthy = validator_snapshot(env, db)
    assert healthy["signed_burn_only"] and healthy["quoted_jobs_match"]
    assert healthy["weight_start_epoch"] == epoch and set(healthy["validators"].values()) == {"matches"}
    states[epoch-1] = signed(epoch-1, hotkeys[1])
    assert not validator_snapshot(env, db)["signed_burn_only"]


def test_desktop_keeps_failed_attempts_private_and_reports_work_after_recovery(tmp_path):
    import subprocess
    from scripts.collection_watch_desktop import deliver

    def snapshot(number, enrollments, billed):
        return {"event": {"number": number, "changes": {"current epoch": number}}, "collection": {
            "enrollments": [{"registrations": enrollments}], "billing": [{"funded_runs": billed}]}}

    def disconnected():
        raise subprocess.CalledProcessError(255, ['aws'], stderr='Could not connect to the endpoint URL: private-detail')

    sent = []
    first = snapshot(8, 1, 2)
    deliver(tmp_path, lambda: first, sent.append)
    deliver(tmp_path, disconnected, sent.append)
    deliver(tmp_path, disconnected, sent.append)
    record = json.loads((tmp_path / 'latest.json').read_text())
    assert record['delivery'] == 'offline' and record['snapshot'] == first
    assert record['readback_failure']['attempts'] == 2
    assert record['readback_failure']['reason'] == 'operator_connection_unavailable'
    assert record['readback_failure']['checked_at']
    assert 'private-detail' not in str((record, sent)) and len(sent) == 2
    deliver(tmp_path, lambda: snapshot(12, 2, 3), sent.append)
    assert len(sent) == 3 and 'restored' in sent[-1] and '1 new enrollment' in sent[-1] and '1 newly billed run' in sent[-1]
    assert 'readback_failure' not in json.loads((tmp_path / 'latest.json').read_text())


def test_desktop_retries_poll_timeout_using_the_same_read_command(monkeypatch):
    from datetime import datetime, timezone
    from types import SimpleNamespace
    import subprocess
    from scripts import collection_watch_desktop as watcher

    data = {'observed_at': datetime.now(timezone.utc).isoformat()}
    calls = []
    def check_output(args, **kwargs):
        calls.append('submit'); return 'example-read-command'
    def run(args, **kwargs):
        assert args[args.index('--command-id') + 1] == 'example-read-command'
        calls.append('poll')
        if calls.count('poll') == 1:
            raise subprocess.TimeoutExpired(args, 20)
        return SimpleNamespace(returncode=0, stdout=json.dumps({'Status':'Success','StandardOutputContent':json.dumps(data)}))
    monkeypatch.setattr(watcher.subprocess, 'check_output', check_output)
    monkeypatch.setattr(watcher.subprocess, 'run', run)
    monkeypatch.setattr(watcher.time, 'sleep', lambda _:None)
    assert watcher.fetch() == data and calls == ['submit','poll','poll']


def test_desktop_retries_a_disconnected_submit_and_preserves_credential_errors(monkeypatch):
    from datetime import datetime, timezone
    from types import SimpleNamespace
    import subprocess
    import pytest
    from scripts import collection_watch_desktop as watcher

    data = {'observed_at': datetime.now(timezone.utc).isoformat()}
    attempts = []
    def submit(*args, **kwargs):
        attempts.append(True)
        if len(attempts) == 1:
            raise subprocess.CalledProcessError(255, 'aws', stderr='Could not connect to the endpoint URL: hidden')
        return 'one-accepted-command'
    monkeypatch.setattr(watcher.subprocess, 'check_output', submit)
    monkeypatch.setattr(watcher.time, 'sleep', lambda _: None)
    monkeypatch.setattr(watcher.subprocess, 'run', lambda *a, **kw: SimpleNamespace(
        returncode=0, stdout=json.dumps({'Status': 'Success', 'StandardOutputContent': json.dumps(data)})))
    assert watcher.fetch() == data and len(attempts) == 2
    monkeypatch.setattr(watcher.subprocess, 'run', lambda *a, **kw: SimpleNamespace(
        returncode=254, stderr='An error occurred (AccessDeniedException): hidden'))
    with pytest.raises(subprocess.CalledProcessError) as refused:
        watcher.fetch()
    assert watcher.failure_reason(refused.value) == 'operator_access_denied' and len(attempts) == 3


def test_current_commit_on_prior_vector_is_pending_without_claiming_payment():
    weights = {
        "epoch": 101, "weight_start_epoch": 100, "signed_state": True,
        "basis_schema": "leadpoet.collection.fixed_run_quotes.v1",
        "quoted_jobs_match": True, "unweighted_funders": 0, "champion_ppb": 0,
        "validators": {"primary": "pending_reveal", "rizzo": "matches", "yuma": "pending_reveal"},
    }
    data = {"collection": {"first_epoch": 100}, "weights": weights}
    observed = notices(data)
    assert observed.get("validator commitments pending reveal")
    assert not observed.get("inspect post-cutover reward evidence")
    assert not any("verified on all three validators" in key for key in observed)
    weights["validators"]["primary"] = "differs"
    assert notices(data)["primary finalized vector differs"]
    assert notices(data)["inspect post-cutover reward evidence"]
    weights["validators"]["primary"] = "pending_reveal"
    weights["quoted_jobs_match"] = False
    assert notices(data)["inspect post-cutover reward evidence"]


def test_worker_reports_real_standing_claims_and_idle_polls(monkeypatch):
    def read(command, **kwargs):
        if command[0] == "journalctl":
            rows = [{"batch": "pool-"+"a"*64, "job": "ordinary", "status": "accepted"},
                    {"batch": "pool-"+"b"*64, "status": "idle"}]
            return "\n".join(json.dumps({"MESSAGE":json.dumps(row)}) for row in rows)
        return "UnitFileState=enabled\nActiveState=active" if command[2].endswith(".timer") else "Result=success\nActiveState=inactive"
    monkeypatch.setattr("subprocess.check_output", read)
    result = worker_snapshot()
    assert result["recent_arms"] == {"pool":"idle"}
    assert "no queue readback in 20 minutes" not in notices({"worker":result})


def test_pending_reveal_needs_registered_recipient_signed_prior_vector_and_unique_live_commit(monkeypatch):
    from types import SimpleNamespace
    import pytest
    from lab_arena import collection_distributions, signing, weight_state
    from leadpoet_canonical import arena_weights, chain_source_v2
    from hashlib import sha256
    from scripts.collection_participation_watch import VALIDATORS, pending_reveals

    signer = signing.LocalSigner.generate()
    keys = list(VALIDATORS.values())

    def state(epoch, start, burn):
        return weight_state.build_accepted_weight_state(
            signer, network="finney", netuid=71, genesis_hash="1" * 64, epoch=epoch,
            valid_from_block=start, valid_until_block=start+359, burn_hotkey=burn,
            reward_basis={"schema_version": arena_weights.COLLECTION_BASIS_SCHEMA, "source_epoch": epoch-1,
                          "job_count": 0, "job_set_sha256": sha256(b"").hexdigest(), "quotes_microusd": {},
                          "valuation": None}, slot_eligibility=None, issued_at="2026-11-25T00:00:00Z")

    previous, current = state(100, 1000, keys[0]), state(101, 1360, keys[0])
    old_hash = arena_weights.derive_arena_weights(previous, keys)["weights_hash"]
    matches = dict.fromkeys(VALIDATORS, "differs")
    meta = SimpleNamespace(current_block=1400, block_hash="block")
    db = SimpleNamespace(get_weight_state=lambda *_: {"state_doc": previous})
    cut = SimpleNamespace(subnet_epoch_index=lambda _: 100)
    queue = [{"submitted_at": 1368, "hotkey_public_key": (bytes([i])*32).hex()} for i in range(len(keys))]
    monkeypatch.setattr(chain_source_v2, "ss58_encode_account_id", lambda value: keys[value[0]])
    monkeypatch.setattr(chain_source_v2, "decode_timelocked_weight_commits", lambda data: data)
    rpc = SimpleNamespace(call=lambda method, **kw: queue if method=="state_getStorage" else "parent")
    monkeypatch.setattr(collection_distributions, "registry_at", lambda *a: keys)

    def vectors(_, block, registry, epoch, expected, validators):
        assert epoch == 100 and expected == old_hash and block == meta.block_hash
        return {"primary":"matches", "rizzo":"matches", "yuma":"differs"}

    monkeypatch.setattr(collection_distributions, "validator_vectors", vectors)
    def sample(registry=None):
        return pending_reveals(rpc, db, signer, cut, meta, current, registry or keys, matches)

    assert sample() == {"primary":"pending_reveal", "rizzo":"pending_reveal", "yuma":"differs"}
    queue.append(queue[0])
    queue[1]["submitted_at"] = 1359
    assert sample() == matches
    queue.pop()
    assert sample(registry=list(reversed(keys))) == matches
    previous["burn_hotkey"] = keys[2]
    with pytest.raises(ValueError):
        sample()


def test_monitor_uses_an_exact_block_finalized_by_both_sources(monkeypatch):
    from types import SimpleNamespace
    import pytest
    from lab_arena import chain
    from Leadpoet.utils import subnet_epoch
    from scripts import collection_participation_watch as watch

    live_hash, older_hash, other_hash = ("0x" + s * 64 for s in ("1", "2", "3"))
    live = SimpleNamespace(current_block=120, block_hash=live_hash)
    earlier = SimpleNamespace(current_block=99, block_hash=older_hash)
    reader = SimpleNamespace(epoch_subtensor=lambda: "live", finalized_block_hash=lambda n: older_hash if n == 99 else live_hash)
    monkeypatch.setattr(chain, "finalized_epoch_snapshot", lambda r: live)
    def exact(reader, *, netuid, block_hash):
        assert reader == "live" and netuid == 71 and block_hash == older_hash
        return earlier
    monkeypatch.setattr(subnet_epoch, "read_subnet_epoch_snapshot", exact)

    def rpc(height, digest, live_at_height=live_hash):
        def call(*, method, params, request_id):
            if method == "chain_getFinalizedHead":
                return digest
            if method == "chain_getHeader":
                assert params == [digest]
                return {"number":hex(height)}
            assert method == "chain_getBlockHash" and params == [120]
            return live_at_height
        return SimpleNamespace(call=call)
    assert watch.shared_finalized_snapshot(reader, rpc(99, older_hash)) == (earlier, live, 99)
    assert watch.shared_finalized_snapshot(reader, rpc(120, live_hash)) == (live, live, 120)
    assert watch.shared_finalized_snapshot(reader, rpc(121, other_hash)) == (live, live, 121)
    for archive in (rpc(99, other_hash), rpc(120, other_hash), rpc(121, other_hash, other_hash)):
        with pytest.raises(ValueError, match="finalized hash"):
            watch.shared_finalized_snapshot(reader, archive)
