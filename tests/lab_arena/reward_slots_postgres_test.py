"""Literal-migration proofs for reward slots after existing crown/promotion."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import itertools
import hashlib
import json
from pathlib import Path
import threading
import time

import pytest

from lab_arena import rewards, signing
from lab_arena.store import ArenaStore, ArenaStoreError, PsycopgTransport
from tests.lab_arena.lab_arena_pg_harness import (
    CURRENT_SERVICE_MIGRATIONS, database_with_lab_arena_migration,
)

MIGRATION = Path(__file__).parents[2] / "scripts/428-lab-arena-reward-slots.sql"
MIGRATIONS = CURRENT_SERVICE_MIGRATIONS + (
    "342-lab-arena-reward-predecessor-barrier.sql",
    "377-lab-arena-monotonic-day-authority.sql", MIGRATION.name,
)
BASELINE = "5" + "A" * 47
MINER = "5" + "B" * 47
CHAIN_IDS = itertools.count(800)
POLICY = {
    "assignment_mode": "highest_only",
    "tiers": [
        {"minimum_improvement": 10, "allocation_percent": 50},
        {"minimum_improvement": 5, "allocation_percent": 30},
        {"minimum_improvement": 1, "allocation_percent": 20},
    ],
}


@pytest.fixture(scope="module")
def database():
    yield from database_with_lab_arena_migration(MIGRATIONS)


@pytest.fixture()
def state(database):
    psycopg, dsn = database
    control = psycopg.connect(**dsn)
    control.autocommit = True
    store = ArenaStore(PsycopgTransport(lambda: psycopg.connect(**dsn)))
    try:
        yield store, control, next(CHAIN_IDS)
    finally:
        store.close()
        control.close()


def _publication(round_id, delta=1, *, winner=MINER, baseline_score=40):
    return {
        "schema_version": "leadpoet.lab_arena.publication.v1",
        "round_id": round_id,
        "published_at": "2026-10-08T12:00:00Z",
        "king_decision": {
            "outcome": "crowned", "king_hotkey": winner,
            "king_submission_id": round_id + "-winner",
            "winner_submission_id": round_id + "-winner",
        },
        "participants": [
            {"submission_id": round_id + "-winner", "miner_hotkey": winner, "is_baseline": False},
            {"submission_id": round_id + "-baseline", "miner_hotkey": BASELINE, "is_baseline": True},
        ],
        "final_ranking": [
            {"submission_id": round_id + "-winner", "final_score": baseline_score + delta,
             "eligible": True, "is_baseline": False},
            {"submission_id": round_id + "-baseline", "final_score": baseline_score,
             "eligible": True, "is_baseline": True},
        ],
    }


def _insert(control, netuid, name, *, day="2026-10-08", delta=1,
            activated=True, promoted=True, status="published", mode="live",
            enabled=True, publication=None, created="00:00:00", epoch=None,
            baseline_hotkey=BASELINE, promotion_required=True):
    round_id = f"arena-{day}-{netuid}{name}"
    pub = publication or _publication(round_id, delta)
    config = {
        "mode": mode, "network_name": "finney", "netuid": netuid,
        "rewards_enabled": enabled, "baseline_hotkey": baseline_hotkey,
        "reward_constants": rewards.reward_constants_document(),
    }
    with control.cursor() as cursor:
        if activated and epoch is None:
            cursor.execute("SELECT COALESCE(max(effective_reward_epoch),9)+1 "
                           "FROM public.lab_arena_rounds WHERE arena_network_name='finney' AND arena_netuid=%s", (netuid,))
            epoch = cursor.fetchone()[0]
        cursor.execute(
            "INSERT INTO public.lab_arena_rounds "
            "(round_id,status,configuration_doc,rewards_enabled,evaluation_date,"
            "publication_doc,king_outcome,king_hotkey,published_at,promotion_required,"
            "baseline_promoted_at,created_at,effective_reward_epoch,reward_basis_doc,"
            "signing_key_doc,reward_activated_at,reward_basis_hash,king_start_epoch) "
            "VALUES (%s,%s,%s::jsonb,%s,%s,%s::jsonb,%s,%s,%s::timestamptz,%s,"
            "%s::timestamptz,%s::timestamptz,%s,%s::jsonb,%s::jsonb,%s::timestamptz,%s,%s)",
            (round_id,status,json.dumps(config),enabled,day,json.dumps(pub),
             pub["king_decision"]["outcome"],pub["king_decision"].get("king_hotkey") or None,
             pub["published_at"], promotion_required,
             f"{day}T12:00:00Z" if promoted else None,
             f"{day}T{created}Z",epoch if activated else None,
             json.dumps({"king_outcome": "no_king"}) if activated else None,
             json.dumps({"public_key_hash": "sha256:" + "a" * 64}) if activated else None,
             f"{day}T13:00:00Z" if activated else None,
             "sha256:" + hashlib.sha256(round_id.encode()).hexdigest() if activated else None,
             0 if activated else None),
        )
    return round_id


def _basis(round_id, slots, *, epoch=100, policy=None, version="v2", hotkey=MINER):
    signer = signing.LocalSigner.generate()
    doc = {
        "schema_version": "leadpoet.lab_arena.reward_basis." + version,
        "round_id": round_id, "published_at": "2026-10-08T12:00:00Z",
        "effective_reward_epoch": epoch, "king_start_epoch": epoch,
        "king_outcome": "crowned", "king_hotkey": hotkey,
        "reward_constants": rewards.reward_constants_document(),
        "champion_reward_factor_ppm": 1000000,
    }
    if version == "v2":
        doc.update(slot_policy=deepcopy(policy or POLICY), reward_slots=slots)
    doc["reward_basis_hash"] = "sha256:" + hashlib.sha256(
        json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return signing.sign_document(signer, doc, hash_field="reward_basis_hash"), signing.signing_key_document(signer.public_key_der)


def test_literal_migration_replays_and_preserves_existing_activation_guards(state):
    store, control, chain = state
    with control.cursor() as cursor:
        cursor.execute(MIGRATION.read_text())
        cursor.execute(MIGRATION.read_text())
        cursor.execute("SELECT pg_catalog.pg_get_functiondef('public.lab_arena_activate_reward(text,jsonb,jsonb)'::regprocedure)")
        definition = cursor.fetchone()[0]
    assert "377 monotonic day authority" in definition
    assert "lab_arena_champion_reward_factor_mismatch" in definition
    assert "lab_arena_reward_activation_mismatch" in definition
    assert "pg_advisory_xact_lock" in definition
    assert "epoch_conflict" in definition
    round_id = _insert(control, chain, "legacy", activated=False, delta=10)
    basis, key = _basis(round_id, [], version="v1")
    assert store.activate_reward(round_id, basis, key)["status"] == "activated"
    assert store.activate_reward(round_id, basis, key)["status"] == "existing"
    assert store.get_round(round_id)["reward_basis_doc"] == basis
    with pytest.raises(ArenaStoreError, match="activation_mismatch"):
        store.activate_reward(round_id, dict(basis, slot_policy=POLICY), key)


def test_thresholds_latest_replacement_and_same_round_baseline(state):
    store, control, chain = state
    high = _insert(control, chain, "high", day="2026-10-05", delta=10)
    middle = _insert(control, chain, "middle", day="2026-10-06", delta=5)
    low = _insert(control, chain, "low", day="2026-10-07", delta=1)
    target = _insert(control, chain, "target", delta=0.925, activated=False)
    assert [x["round_id"] for x in store.reward_slot_snapshot(target, POLICY)] == [high, middle, low]
    latest = _insert(control, chain, "latest", delta=5.5, created="01:00:00")
    assert [x["round_id"] for x in store.reward_slot_snapshot(target, POLICY)] == [high, latest, low]
    all_policy = dict(POLICY, assignment_mode="all_qualifying")
    assert [x["round_id"] for x in store.reward_slot_snapshot(target, all_policy)] == [high, latest, latest]
    # Earlier +1 against that day's 40 baseline must not become a +5 against
    # a different round's baseline. Every returned score stays on its event.
    assert store.reward_slot_snapshot(target, POLICY)[2]["baseline_score"] == 40


@pytest.mark.parametrize("delta,index", [(10,0),(9.999999,1),(5,1),(4.999999,2),(1,2),(0.999999,None)])
def test_exact_numeric_boundaries_and_empty_slots(state, delta, index):
    store, control, chain = state
    target = _insert(control, chain, "boundary", delta=delta, activated=False)
    slots = store.reward_slot_snapshot(target, POLICY)
    assert [i for i, slot in enumerate(slots) if slot is not None] == ([] if index is None else [index])


@pytest.mark.parametrize("mutation", [
    "unpromoted", "unactivated", "cross_chain", "cancelled", "test", "disabled",
    "ineligible_winner", "ineligible_baseline", "missing_eligibility", "duplicate_winner",
    "duplicate_baseline", "wrong_participant_hotkey", "wrong_baseline_hotkey",
    "organizer_winner", "wrong_decision", "invalid_score", "null_score", "missing_score",
    "out_of_range", "future_day", "duplicate_participant", "no_king",
    "wrong_publication_round", "wrong_publication_schema", "duplicate_baseline_id",
    "second_baseline_participant",
])
def test_invalid_history_cannot_seed_slots(state, mutation):
    store, control, chain = state
    name = "bad"
    kwargs = {"delta": 10}
    if mutation in ("unpromoted", "unactivated"):
        kwargs["promoted" if mutation == "unpromoted" else "activated"] = False
    elif mutation == "cancelled": kwargs["status"] = "cancelled"
    elif mutation == "test": kwargs["mode"] = "test"
    elif mutation == "disabled": kwargs["enabled"] = False
    elif mutation == "future_day": kwargs["day"] = "2026-10-09"
    candidate_chain = chain + 10000 if mutation == "cross_chain" else chain
    candidate_id = f"arena-{kwargs.get('day','2026-10-08')}-{candidate_chain}{name}"
    pub = _publication(candidate_id, 10)
    if mutation == "ineligible_winner": pub["final_ranking"][0]["eligible"] = False
    elif mutation == "ineligible_baseline": pub["final_ranking"][1]["eligible"] = False
    elif mutation == "missing_eligibility": del pub["final_ranking"][0]["eligible"]
    elif mutation == "duplicate_winner": pub["final_ranking"].append(deepcopy(pub["final_ranking"][0]))
    elif mutation == "duplicate_baseline": pub["final_ranking"].append(deepcopy(pub["final_ranking"][1]))
    elif mutation == "wrong_participant_hotkey": pub["participants"][0]["miner_hotkey"] = BASELINE
    elif mutation == "wrong_baseline_hotkey": pub["participants"][1]["miner_hotkey"] = MINER
    elif mutation == "organizer_winner": pub["king_decision"]["king_hotkey"] = BASELINE
    elif mutation == "wrong_decision": pub["king_decision"]["king_submission_id"] = "other"
    elif mutation == "invalid_score": pub["final_ranking"][0]["final_score"] = "NaN"
    elif mutation == "null_score": pub["final_ranking"][0]["final_score"] = None
    elif mutation == "missing_score": del pub["final_ranking"][0]["final_score"]
    elif mutation == "out_of_range": pub["final_ranking"][0]["final_score"] = 101
    elif mutation == "duplicate_participant": pub["participants"].append(deepcopy(pub["participants"][0]))
    elif mutation == "no_king": pub["king_decision"]["outcome"] = "no_king"
    elif mutation == "wrong_publication_round": pub["round_id"] = "other"
    elif mutation == "wrong_publication_schema": pub["schema_version"] = "other"
    elif mutation == "duplicate_baseline_id":
        pub["final_ranking"].append(dict(pub["final_ranking"][1],is_baseline=False))
    elif mutation == "second_baseline_participant":
        pub["participants"].append(dict(pub["participants"][1],submission_id="extra"))
    _insert(control, candidate_chain, name, publication=pub, **kwargs)
    target = _insert(control, chain, "target", delta=0.5, activated=False, created="01:00:00")
    assert store.reward_slot_snapshot(target, POLICY) == [None,None,None]


@pytest.mark.parametrize("field", ["miner_hotkey","round_id","submission_id","baseline_score","winner_score"])
def test_activation_rejects_fabricated_or_stale_slots(state, field):
    store, control, chain = state
    target = _insert(control, chain, "tamper", activated=False, delta=10)
    slots = store.reward_slot_snapshot(target, POLICY)
    slots[0][field] = "forged" if field not in ("baseline_score","winner_score") else 12
    basis, key = _basis(target, slots)
    with pytest.raises(ArenaStoreError, match="reward_slots_mismatch"):
        store.activate_reward(target, basis, key)
    assert store.get_round(target)["reward_basis_doc"] is None


def test_v2_activates_once_and_existing_v1_rows_are_immutable(state):
    store, control, chain = state
    old = _insert(control, chain, "old", day="2026-10-07", delta=1)
    old_doc = store.get_round(old)["reward_basis_doc"]
    current = _insert(control, chain, "current", activated=False, delta=10)
    slots = store.reward_slot_snapshot(current, POLICY)
    basis, key = _basis(current, slots)
    assert store.activate_reward(current, basis, key)["status"] == "activated"
    assert store.activate_reward(current, basis, key)["status"] == "existing"
    assert store.get_round(old)["reward_basis_doc"] == old_doc
    assert store.get_round(current)["reward_basis_doc"]["reward_slots"] == slots
    with control.cursor() as cursor:
        with pytest.raises(Exception, match="immutable"):
            cursor.execute("UPDATE public.lab_arena_rounds SET reward_basis_doc='{}'::jsonb WHERE round_id=%s", (old,))


@pytest.mark.parametrize("change", ["mode","count","order","shares","float","zero","extra","too_large"])
def test_policy_validation_rejects_invalid_signed_policy(state, change):
    store, control, chain = state
    target = _insert(control, chain, "policy", activated=False)
    policy = deepcopy(POLICY)
    if change == "mode": policy["assignment_mode"] = "other"
    elif change == "count": policy["tiers"].pop()
    elif change == "order": policy["tiers"].reverse()
    elif change == "shares": policy["tiers"][0]["allocation_percent"] = 49
    elif change == "float": policy["tiers"][0]["minimum_improvement"] = 10.0
    elif change == "zero": policy["tiers"][2]["minimum_improvement"] = 0
    elif change == "extra": policy["extra"] = True
    elif change == "too_large": policy["tiers"][0]["minimum_improvement"] = 101
    with pytest.raises(ArenaStoreError, match="slot_policy_invalid"):
        store.reward_slot_snapshot(target, policy)


def test_generic_policy_does_not_duplicate_live_python_constants(state):
    store, control, chain = state
    target = _insert(control, chain, "alternate", delta=7, activated=False)
    policy = {"assignment_mode":"highest_only", "tiers":[
        {"minimum_improvement":8,"allocation_percent":40},
        {"minimum_improvement":4,"allocation_percent":35},
        {"minimum_improvement":2,"allocation_percent":25},
    ]}
    assert [i for i,s in enumerate(store.reward_slot_snapshot(target, policy)) if s] == [1]


def test_histories_beyond_legacy_200_basis_window_still_seed_slots(state):
    store, control, chain = state
    oldest = _insert(control, chain, "achievement", day="2026-10-01", delta=10)
    for index in range(205):
        _insert(control, chain, f"miss{index}", day="2026-10-07", delta=0.5, epoch=20+index)
    target = _insert(control, chain, "target", delta=0.5, activated=False)
    assert store.reward_slot_snapshot(target, POLICY)[0]["round_id"] == oldest


def test_same_day_concurrent_activations_recheck_slots_after_advisory_lock(database, state):
    psycopg, dsn = database
    store, control, chain = state
    first_id = f"arena-2026-10-08-{chain}first"
    first_hotkey = "5" + "C" * 47
    first = _insert(control, chain, "first", activated=False,
                    publication=_publication(first_id, 1, winner=first_hotkey))
    second = _insert(control, chain, "second", delta=10, activated=False, created="01:00:00")
    first_basis, first_key = _basis(first, store.reward_slot_snapshot(first, POLICY), hotkey=first_hotkey)
    second_basis, second_key = _basis(second, store.reward_slot_snapshot(second, POLICY))
    # Hold the first activation transaction open. The second signed snapshot
    # was read before that transaction; its low slot must be rechecked after
    # waiting on the existing activation lock, even though the outer SELECT
    # began before the first activation committed.
    control.autocommit = False
    with control.cursor() as cursor:
        cursor.execute("SELECT public.lab_arena_activate_reward(%s,%s::jsonb,%s::jsonb)",
                       (first,json.dumps(first_basis),json.dumps(first_key)))
        assert cursor.fetchone()[0]["status"] == "activated"
    started = threading.Event()
    def activate():
        other = ArenaStore(PsycopgTransport(lambda: psycopg.connect(**dsn)))
        try:
            started.set()
            return other.activate_reward(second, second_basis, second_key)
        finally:
            other.close()
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(activate)
        assert started.wait(timeout=5)
        try:
            with psycopg.connect(**dsn) as observer:
                with observer.cursor() as cursor:
                    deadline = time.monotonic()+5
                    while time.monotonic() < deadline:
                        cursor.execute("SELECT count(*) FROM pg_catalog.pg_locks WHERE locktype='advisory' AND NOT granted")
                        if cursor.fetchone()[0]:
                            break
                        time.sleep(0.01)
                    else:
                        pytest.fail("second activation did not wait on advisory lock")
        finally:
            control.commit()
            control.autocommit = True
        with pytest.raises(ArenaStoreError, match="reward_slots_mismatch"):
            future.result(timeout=10)
    assert store.get_round(first)["reward_basis_doc"] == first_basis
    assert store.get_round(second)["reward_basis_doc"] is None
    refreshed, key = _basis(second, store.reward_slot_snapshot(second, POLICY))
    assert store.activate_reward(second, refreshed, key)["status"] == "epoch_conflict"


def test_first_v2_basis_blocks_new_v1_activation_but_preserves_accepted_replay(state):
    store, control, chain = state
    first = _insert(control, chain, "firstv2", activated=False, delta=10)
    accepted, accepted_key = _basis(first, store.reward_slot_snapshot(first, POLICY))
    assert store.activate_reward(first, accepted, accepted_key)["status"] == "activated"
    second_id = f"arena-2026-10-08-{chain}oldgateway"
    other_hotkey = "5" + "C" * 47
    second = _insert(control, chain, "oldgateway", activated=False, created="01:00:00",
                     publication=_publication(second_id,5,winner=other_hotkey))
    legacy, key = _basis(second, [], version="v1", epoch=101, hotkey=other_hotkey)
    with pytest.raises(ArenaStoreError, match="reward_basis_downgrade"):
        store.activate_reward(second, legacy, key)
    assert store.get_round(second)["reward_basis_doc"] is None
    assert store.activate_reward(first, accepted, accepted_key)["status"] == "existing"
    # A different chain remains on v1 until its own first v2 activation.
    other = _insert(control, chain+10000, "legacy", activated=False, delta=10)
    legacy, key = _basis(other, [], version="v1")
    assert store.activate_reward(other, legacy, key)["status"] == "activated"


@pytest.mark.parametrize("malformation", ["identity","score","duplicate_baseline"])
def test_malformed_current_promoted_crown_cannot_activate_empty_slots(state, malformation):
    store, control, chain = state
    target_id = f"arena-2026-10-08-{chain}malformed"
    pub = _publication(target_id,10)
    if malformation == "identity": pub["participants"][0]["miner_hotkey"] = BASELINE
    elif malformation == "score": pub["final_ranking"][0]["final_score"] = None
    else: pub["final_ranking"].append(deepcopy(pub["final_ranking"][1]))
    target = _insert(control, chain, "malformed", publication=pub, activated=False)
    with pytest.raises(ArenaStoreError, match="current_achievement_invalid"):
        store.reward_slot_snapshot(target, POLICY)
    basis, key = _basis(target, [None,None,None])
    with pytest.raises(ArenaStoreError, match="current_achievement_invalid"):
        store.activate_reward(target, basis, key)
    assert store.get_round(target)["reward_basis_doc"] is None


def test_migration_guard_rejects_mutated_activation_authority(state):
    _, control, _ = state
    with control.cursor() as cursor:
        cursor.execute("BEGIN")
        cursor.execute("SELECT pg_catalog.pg_get_functiondef('public.lab_arena_activate_reward(text,jsonb,jsonb)'::regprocedure)")
        definition = cursor.fetchone()[0]
        cursor.execute(definition.replace("-- 428 signed reward slots:","-- unexpected change\n  -- 428 signed reward slots:"))
        with pytest.raises(Exception, match="428 reward activation definition differs"):
            cursor.execute(MIGRATION.read_text())
        cursor.execute("ROLLBACK")


def test_rpc_acl_is_service_only_and_read_only(state):
    store, control, chain = state
    target = _insert(control,chain,"acl",activated=False,delta=5)
    before = store.get_round(target)
    with control.cursor() as cursor:
        for role in ("anon","authenticated"):
            cursor.execute("SELECT has_function_privilege(%s,'public.lab_arena_reward_slot_snapshot(text,jsonb)','EXECUTE')",(role,))
            assert cursor.fetchone()[0] is False
        for role in ("lab_arena_service","service_role"):
            cursor.execute("SELECT has_function_privilege(%s,'public.lab_arena_reward_slot_snapshot(text,jsonb)','EXECUTE')",(role,))
            assert cursor.fetchone()[0] is True
        cursor.execute("SET ROLE lab_arena_service")
        try:
            cursor.execute("SELECT public.lab_arena_reward_slot_snapshot(%s,%s::jsonb)",(target,json.dumps(POLICY)))
            assert cursor.fetchone()[0]["reward_slots"][1]["round_id"] == target
        finally:
            cursor.execute("RESET ROLE")
    assert store.get_round(target) == before


def test_new_v2_crown_waits_for_actual_promotion_when_legacy_flag_is_false(state):
    store, control, chain = state
    target = _insert(control,chain,"legacyflag",activated=False,promoted=False,
                     promotion_required=False,delta=10)
    basis, key = _basis(target,[None,None,None])
    assert store.activate_reward(target,basis,key) == {"status":"waiting_for_promotion"}
    row = store.get_round(target)
    assert row["reward_basis_doc"] is None
    assert row["reward_activated_at"] is None
    assert row["baseline_promoted_at"] is None
    # The v2 guard does not change the legacy v1 activation contract.
    legacy, key = _basis(target,[],version="v1")
    assert store.activate_reward(target,legacy,key)["status"] == "activated"
    assert store.activate_reward(target,legacy,key)["status"] == "existing"
