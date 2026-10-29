"""Hermetic service regression tests for the Git/DB promotion boundary.

These tests use persisted scoring fixtures and do not claim live ICP execution.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
import subprocess

import pytest

from lab_arena import contracts, rewards, service as svc, signing, source_bundle
from lab_arena.promotion import GitPromoter, PromotionError
from lab_arena.store import ArenaStore, ArenaStoreError, PsycopgTransport
from tests.lab_arena.test_lab_arena_promotion_git import git, repository, source
from tests.lab_arena.test_lab_arena_promotion_migration_postgres import (
    _plan,
    connections,
)
from tests.lab_arena.lab_arena_pg_harness import current_service_database
from tests.lab_arena.test_lab_arena_reward_migration_postgres import (
    BASELINE,
    MINER_A,
    _basis,
)


@pytest.fixture()
def database():
    yield from current_service_database()


class _Chain:
    def current_settlement_epoch(self):
        return 900


def _git_archive(remote: Path) -> bytes:
    return subprocess.run(
        (
            "git", "archive", "--format=tar.gz", "--prefix=pydantic-harness-lab/",
            f"--remote={remote}", "lab",
        ),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
    ).stdout


def _service(
    store: ArenaStore,
    objects: svc.LocalObjectStore,
    remote: Path,
    cache: Path,
    signer: signing.LocalSigner,
) -> svc.ArenaService:
    def fetch(url: str, limit: int) -> bytes:
        assert url == svc.DEFAULT_BASELINE_SOURCE_URL
        payload = _git_archive(remote)
        assert len(payload) <= limit
        return payload

    return svc.ArenaService(
        svc.ServiceConfig(
            mode="live",
            clock=lambda: datetime(2026, 9, 9, tzinfo=timezone.utc),
            store=store,
            object_store=objects,
            signer=signer,
            chain=_Chain(),
            verify_signature=lambda *_args: True,
            daily_icp_source=lambda **_kwargs: {"status": "unavailable"},
            banned_hotkeys_source=lambda: (),
            broker_factory=lambda *_args: None,
            defaults=svc.RoundDefaults(
                baseline_hotkey=BASELINE,
                baseline_source_url=svc.DEFAULT_BASELINE_SOURCE_URL,
                rewards_enabled=True,
            ),
            baseline_source_fetcher=fetch,
            baseline_promoter_factory=lambda: GitPromoter(str(remote), cache),
        )
    )


def _winner(store, control, objects, round_id: str, payload: bytes) -> str:
    _publish(
        store,
        control,
        round_id,
        miner=MINER_A,
        baseline_score=50,
        miner_score=60,
        crowned=True,
        evaluation_date="2026-09-07",
        published_at="2026-09-08T00:00:00Z",
    )
    submission_id = round_id + "-miner"
    source_ref = f"arena/{round_id}/sources/{submission_id}.tar.gz"
    objects.put(source_ref, payload)
    with control.cursor() as cursor:
        cursor.execute(
            "INSERT INTO public.lab_arena_submissions "
            "(submission_id, round_id, miner_hotkey, status, is_king, source_ref, "
            "source_size_bytes, consent) VALUES (%s, %s, %s, 'frozen', FALSE, %s, %s, "
            "'{\"public_rerun\":true}'::jsonb)",
            (submission_id, round_id, MINER_A, source_ref, len(payload)),
        )
    return submission_id


def _new_store(database, request) -> ArenaStore:
    psycopg2, dsn = database
    store = ArenaStore(PsycopgTransport(lambda: psycopg2.connect(**dsn)))
    request.addfinalizer(store.close)
    return store


def _publish(
    store, control, round_id: str, *, miner, baseline_score, miner_score,
    crowned, rewards_enabled=True, evaluation_date=None, published_at=None,
) -> str:
    """Persist complete authority before using the real publication RPC."""
    evaluation_date = evaluation_date or round_id[6:16]
    published_at = published_at or evaluation_date + "T01:00:00Z"
    constants = rewards.reward_constants_document()
    assert store.create_round(
        round_id,
        {
            "mode": "live",
            "rewards_enabled": rewards_enabled,
            "baseline_hotkey": BASELINE,
            "reward_constants": constants,
            "stage_1_icp_count": 1,
            "stage_2_icp_count": 1,
        },
    )["status"] == "created"
    generation_floor = store.get_round(round_id)["scoring_generation_floor"]
    baseline_id = round_id + "-baseline"
    winner_id = round_id + "-miner"
    participants = [
        {"submission_id": baseline_id, "miner_hotkey": BASELINE, "is_king": True},
        {"submission_id": winner_id, "miner_hotkey": miner, "is_king": False},
    ]
    with control, control.cursor() as cursor:
        cursor.execute("BEGIN")
        # Only fixture construction bypasses input/accepted-row immutability
        # triggers. Restore them before publication, promotion and rewards.
        cursor.execute("SET LOCAL session_replication_role=replica")
        cursor.execute(
            "UPDATE public.lab_arena_rounds SET status = 'scored', participants = %s::jsonb, "
            "finalists = '[]'::jsonb, evaluation_date=%s WHERE round_id = %s",
            (json.dumps(participants), evaluation_date, round_id),
        )
        cursor.execute(
            "INSERT INTO public.lab_arena_submissions "
            "(submission_id, round_id, miner_hotkey, status, is_king) "
            "VALUES (%s, %s, %s, 'frozen', TRUE)",
            (baseline_id, round_id, BASELINE),
        )
        for stage, position in ((1, 0), (2, 1)):
            execution = f"{baseline_id}-execute-{position}"
            for kind in ("execute", "score"):
                run_id = f"{baseline_id}-{kind}-{position}"
                cursor.execute(
                    "INSERT INTO public.lab_arena_runs "
                    "(run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,"
                    "icp_position,attempt,kind,scored_run_id,status,terminal_cause,"
                    "per_icp_score,result_doc,output_ref,scoring_generation_floor) VALUES "
                    "(%s,%s,%s,%s,%s,%s,%s,1,%s,%s,'accepted','accepted',%s,%s::jsonb,%s,%s)",
                    (run_id, run_id + "-assignment", round_id, baseline_id, BASELINE,
                     stage, position, kind, execution if kind == "score" else None,
                     baseline_score, json.dumps({"terminal_status": "accepted",
                         "schema_version": "cached" if kind == "score" else "execution"}),
                     run_id + "-output", generation_floor),
                )
    with control.cursor() as cursor:
        cursor.execute("SHOW session_replication_role")
        assert cursor.fetchone() == ("origin",)
        cursor.execute("SELECT public.lab_arena__baseline_final_ready_v1(%s)", (round_id,))
        assert cursor.fetchone() == (True,)
    publication = {
        "schema_version": contracts.PUBLICATION_SCHEMA_VERSION,
        "round_id": round_id,
        "participants": [
            {"submission_id": baseline_id, "miner_hotkey": BASELINE, "is_baseline": True},
            {"submission_id": winner_id, "miner_hotkey": miner, "is_baseline": False},
        ],
        "stage1_ranking": [],
        "finalists": [],
        "final_ranking": [
            {"rank": 1 if crowned else 2, "submission_id": winner_id,
             "final_score": miner_score, "is_baseline": False, "eligible": True},
            {"rank": 2 if crowned else 1, "submission_id": baseline_id,
             "final_score": baseline_score, "is_baseline": True, "eligible": True},
        ],
        "king_decision": {
            "outcome": "crowned" if crowned else "no_king",
            "king_submission_id": winner_id if crowned else None,
            "king_hotkey": miner if crowned else "",
            "winner_submission_id": winner_id if crowned else None,
        },
        "published_at": published_at,
    }
    publication["final_ranking"].sort(key=lambda row: row["rank"])
    assert store.transition_round(
        round_id,
        "scored",
        "published",
        {"publication_doc": publication, "published_at": published_at},
    )["status"] == "ok"
    return published_at


def _reward_disabled_winner(store, control, round_id: str, *,
                            published_at="2026-09-06T00:00:00Z") -> str:
    published_at = _publish(
        store, control, round_id, miner=MINER_A, baseline_score=50,
        miner_score=60, crowned=True, rewards_enabled=False,
        published_at=published_at,
    )
    with control.cursor() as cursor:
        cursor.execute(
            "INSERT INTO public.lab_arena_submissions "
            "(submission_id,round_id,miner_hotkey,status,is_king) "
            "VALUES (%s,%s,%s,'frozen',FALSE)",
            (round_id + "-miner", round_id, MINER_A),
        )
    return published_at


def _assert_promoted_baseline(
    service: svc.ArenaService,
    store: ArenaStore,
    objects: svc.LocalObjectStore,
    tmp_path: Path,
) -> None:
    next_round = "arena-2026-09-08-next"
    assert store.create_round(
        next_round,
        {
            "mode": "live",
            "rewards_enabled": True,
            "baseline_hotkey": BASELINE,
            "reward_constants": rewards.reward_constants_document(),
            "stage_1_icp_count": 1,
            "stage_2_icp_count": 1,
            "schedule": {
                "submission_open": "2026-09-08T00:00:00Z",
                "submission_cutoff": "2026-09-08T12:00:00Z",
            },
        },
    )["status"] == "created"
    baseline = service._initial_baseline(store.get_round(next_round))
    frozen = objects.get(baseline["source_ref"])
    extracted = tmp_path / "promoted-source"
    extracted.mkdir()
    source_bundle.extract_source_archive(frozen, extracted)
    assert (extracted / "harness.py").read_bytes() == b"def run_icp(icp):\n    return []\n"
    assert (extracted / "bin" / "run").read_bytes() == b"#!/bin/sh\nexit 0\n"


def test_lost_db_completion_reconciles_git_then_activates_miner_once(
    connections, database, tmp_path, monkeypatch, request,
):
    store, control = connections
    remote, _, _ = repository(tmp_path)
    objects = svc.LocalObjectStore(tmp_path / "objects")
    signer = signing.LocalSigner.generate()
    round_id = "arena-2026-09-07-servicea"
    payload = source()
    _winner(store, control, objects, round_id, payload)
    service = _service(store, objects, remote, tmp_path / "git-cache-one", signer)

    original_complete = store.complete_promotion
    monkeypatch.setattr(
        store,
        "complete_promotion",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("injected DB outage")),
    )
    with pytest.raises(RuntimeError, match="injected DB outage"):
        service.promote_baseline(round_id)
    row = store.get_round(round_id)
    commit = row["promotion_doc"]["commit"]
    assert row["baseline_promoted_at"] is None
    assert git(remote, "rev-parse", "main") == git(remote, "rev-parse", "lab") == commit
    monkeypatch.setattr(store, "complete_promotion", original_complete)

    fresh_store = _new_store(database, request)
    fresh = _service(
        fresh_store, objects, remote, tmp_path / "git-cache-empty", signer
    )
    assert fresh.promote_pending_baselines() == {"status": "ok", "promoted": 1}
    activated = fresh.activate_reward(round_id)
    assert activated["status"] == "activated"
    first_row = fresh_store.get_round(round_id)
    assert first_row["reward_basis_doc"]["king_hotkey"] == MINER_A
    assert first_row["reward_basis_doc"]["king_hotkey"] != BASELINE
    # This links activation to the production reward calculation, not to a
    # chain write. On-chain settlement still needs its separate live proof.
    from leadpoet_canonical.lab_arena_rewards import slot_allocations

    allocations = slot_allocations(
        first_row["reward_basis_doc"],
        first_row["effective_reward_epoch"],
        [BASELINE, MINER_A],
    )
    assert allocations == {MINER_A: Fraction(3, 10)}
    assert fresh.activate_reward(round_id)["status"] == "existing"
    assert fresh_store.get_round(round_id)["reward_activated_at"] == first_row["reward_activated_at"]
    _assert_promoted_baseline(fresh, fresh_store, objects, tmp_path)


def test_rejected_atomic_push_blocks_reward_and_next_baseline_until_retry(
    connections, tmp_path
):
    store, control = connections
    remote, main, lab = repository(tmp_path)
    hook = remote / "hooks" / "pre-receive"
    hook.write_text("#!/bin/sh\nexit 1\n")
    hook.chmod(0o755)
    objects = svc.LocalObjectStore(tmp_path / "objects")
    signer = signing.LocalSigner.generate()
    round_id = "arena-2026-09-07-serviceb"
    _winner(store, control, objects, round_id, source())
    service = _service(store, objects, remote, tmp_path / "git-cache", signer)

    with pytest.raises(PromotionError, match="promotion_git_failed"):
        service.promote_baseline(round_id)
    assert git(remote, "rev-parse", "main") == main
    assert git(remote, "rev-parse", "lab") == lab
    assert store.get_round(round_id)["baseline_promoted_at"] is None
    assert service.activate_reward(round_id) == {"status": "waiting_for_promotion"}
    with pytest.raises(svc.ServiceError, match="baseline_promotion_pending"):
        service._initial_baseline(
            {
                "round_id": "arena-2026-09-08-blocked",
                "configuration_doc": {"mode": "live", "baseline_hotkey": BASELINE},
            }
        )

    hook.unlink()
    assert service.promote_baseline(round_id)["status"] == "promoted"
    assert service.activate_reward(round_id)["status"] == "activated"


def test_missing_baseline_score_blocks_promotion_before_git(
    connections, tmp_path, monkeypatch,
):
    store, control = connections
    remote, main, lab = repository(tmp_path)
    objects = svc.LocalObjectStore(tmp_path / "objects")
    round_id = "arena-2026-09-07-missingscore"
    _winner(store, control, objects, round_id, source())
    service = _service(
        store, objects, remote, tmp_path / "git-cache",
        signing.LocalSigner.generate(),
    )
    with control, control.cursor() as cursor:
        cursor.execute("BEGIN")
        cursor.execute("SET LOCAL session_replication_role=replica")
        cursor.execute(
            "UPDATE public.lab_arena_runs SET per_icp_score=NULL "
            "WHERE run_id=%s", (round_id + "-baseline-execute-0",),
        )
    with control.cursor() as cursor:
        cursor.execute("SELECT public.lab_arena__baseline_final_ready_v1(%s)", (round_id,))
        assert cursor.fetchone() == (False,)

    def unexpected_git():
        pytest.fail("an incomplete baseline must not create a Git promoter")

    monkeypatch.setattr(service.config, "baseline_promoter_factory", unexpected_git)
    assert service.promote_baseline(round_id) == {"status": "waiting_for_baseline"}
    assert service.activate_reward(round_id) == {"status": "waiting_for_baseline"}
    assert git(remote, "rev-parse", "main") == main
    assert git(remote, "rev-parse", "lab") == lab
    row = store.get_round(round_id)
    assert row["promotion_doc"] is None
    assert row["baseline_promoted_at"] is None
    assert row["reward_basis_doc"] is None
    assert row["reward_activated_at"] is None


def test_earlier_reward_disabled_promotion_blocks_same_day_no_winner_reward(
    connections, tmp_path
):
    store, control = connections
    older = "arena-2026-09-06-globalgateearly"
    _reward_disabled_winner(store, control, older)
    later = "arena-2026-09-06-globalgate"
    later_at = _publish(
        store,
        control,
        later,
        miner=MINER_A,
        baseline_score=60,
        miner_score=50,
        crowned=False,
    )
    signer = signing.LocalSigner.generate()
    basis = _basis(signer, later, later_at, 901, "no_king", "")
    key = signing.signing_key_document(signer.public_key_der)
    with pytest.raises(ArenaStoreError, match="reward_waiting_for_promotion"):
        store.activate_reward(later, basis, key)

    remote, _, _ = repository(tmp_path)
    service = _service(
        store,
        svc.LocalObjectStore(tmp_path / "objects-global"),
        remote,
        tmp_path / "git-cache-global",
        signer,
    )
    assert service.activate_reward(later) == {"status": "waiting_for_promotion"}

    plan = _plan("e")
    assert store.prepare_promotion(older, plan)["status"] == "prepared"
    assert store.complete_promotion(older, plan)["status"] == "promoted"
    assert service.activate_reward(later)["status"] == "activated"


def test_incomplete_baseline_cannot_promote_or_activate_rewards(connections, tmp_path):
    store, control = connections
    remote, main, lab = repository(tmp_path)
    objects = svc.LocalObjectStore(tmp_path / "objects")
    round_id = "arena-2026-09-07-incomplete"
    _winner(store, control, objects, round_id, source())
    # Disposable published authority counterexample: remove one accepted judge,
    # then restore trigger enforcement before invoking either service path.
    with control, control.cursor() as cursor:
        cursor.execute("BEGIN")
        cursor.execute("SET LOCAL session_replication_role=replica")
        cursor.execute("DELETE FROM public.lab_arena_runs WHERE run_id=%s",
                       (round_id + "-baseline-score-0",))
    service = _service(store, objects, remote, tmp_path / "git-cache", signing.LocalSigner.generate())
    assert not service._baseline_final_ready(store.get_round(round_id))
    assert service.promote_baseline(round_id) == {"status": "waiting_for_baseline"}
    assert service.activate_reward(round_id) == {"status": "waiting_for_baseline"}
    assert git(remote, "rev-parse", "main") == main
    assert git(remote, "rev-parse", "lab") == lab
    row = store.get_round(round_id)
    assert row["baseline_promoted_at"] is None and row["reward_activated_at"] is None


def test_historical_db_barrier_is_explicit_while_service_supersedes_older_promotion(
    connections, tmp_path
):
    store, control = connections
    older = "arena-2026-09-05-globalgate"
    _reward_disabled_winner(store, control, older, published_at="2026-09-05T00:00:00Z")
    later = "arena-2026-09-06-globalgate"
    later_at = _publish(
        store,
        control,
        later,
        miner=MINER_A,
        baseline_score=60,
        miner_score=50,
        crowned=False,
        evaluation_date="2026-09-06",
    )
    signer = signing.LocalSigner.generate()
    basis = _basis(signer, later, later_at, 901, "no_king", "")
    key = signing.signing_key_document(signer.public_key_der)
    with pytest.raises(ArenaStoreError, match="reward_waiting_for_promotion"):
        store.activate_reward(later, basis, key)

    remote, _, _ = repository(tmp_path)
    service = _service(
        store,
        svc.LocalObjectStore(tmp_path / "objects-global"),
        remote,
        tmp_path / "git-cache-global",
        signer,
    )
    assert service.promote_baseline(older) == {"status": "superseded"}
    assert service._pending_promotion_blocks() is False

    plan = _plan("e")
    assert store.prepare_promotion(older, plan) == {"status": "superseded", "newer_round_id": later}
    assert store.get_round(older)["baseline_promoted_at"] is None
    assert service.activate_reward(later)["status"] == "activated"
