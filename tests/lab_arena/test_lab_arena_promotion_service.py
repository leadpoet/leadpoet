"""Hermetic service regression tests for the Git/DB promotion boundary.

These tests use persisted scoring fixtures and do not claim live ICP execution.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
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
    database,
)
from tests.lab_arena.test_lab_arena_reward_migration_postgres import (
    BASELINE,
    MINER_A,
    _basis,
    _publish,
)


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


def _new_store(database) -> ArenaStore:
    psycopg2, dsn = database
    return ArenaStore(PsycopgTransport(lambda: psycopg2.connect(**dsn)))


def _reward_disabled_winner(store, control, round_id: str) -> str:
    constants = rewards.reward_constants_document()
    assert store.create_round(
        round_id,
        {
            "mode": "live",
            "rewards_enabled": False,
            "baseline_hotkey": BASELINE,
            "reward_constants": constants,
        },
    )["status"] == "created"
    baseline_id = round_id + "-baseline"
    winner_id = round_id + "-miner"
    participants = [
        {"submission_id": baseline_id, "miner_hotkey": BASELINE, "is_king": True},
        {"submission_id": winner_id, "miner_hotkey": MINER_A, "is_king": False},
    ]
    with control.cursor() as cursor:
        cursor.execute(
            "UPDATE public.lab_arena_rounds SET status = 'scored', participants = %s::jsonb, "
            "finalists = '[]'::jsonb WHERE round_id = %s",
            (json.dumps(participants), round_id),
        )
        cursor.execute(
            "INSERT INTO public.lab_arena_submissions "
            "(submission_id, round_id, miner_hotkey, status, is_king) "
            "VALUES (%s, %s, %s, 'frozen', FALSE)",
            (winner_id, round_id, MINER_A),
        )
    published_at = "2026-09-05T00:00:00Z"
    publication = {
        "schema_version": contracts.PUBLICATION_SCHEMA_VERSION,
        "round_id": round_id,
        "participants": [
            {"submission_id": baseline_id, "miner_hotkey": BASELINE, "is_baseline": True},
            {"submission_id": winner_id, "miner_hotkey": MINER_A, "is_baseline": False},
        ],
        "stage1_ranking": [],
        "finalists": [],
        "final_ranking": [
            {"rank": 1, "submission_id": winner_id, "final_score": 60, "is_baseline": False},
            {"rank": 2, "submission_id": baseline_id, "final_score": 50, "is_baseline": True},
        ],
        "king_decision": {
            "outcome": "crowned",
            "king_submission_id": winner_id,
            "king_hotkey": MINER_A,
            "winner_submission_id": winner_id,
        },
        "published_at": published_at,
    }
    assert store.transition_round(
        round_id,
        "scored",
        "published",
        {"publication_doc": publication, "published_at": published_at},
    )["status"] == "ok"
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
    connections, database, tmp_path, monkeypatch
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

    fresh_store = _new_store(database)
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
    from leadpoet_canonical.lab_arena_rewards import champion_values

    values = champion_values(
        first_row["reward_basis_doc"],
        first_row["effective_reward_epoch"],
        [BASELINE, MINER_A],
    )
    assert values["champion_uid"] == 1
    assert values["champion_share"] == values["effective_champion_share"] == 0.25
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


def test_older_reward_disabled_promotion_blocks_later_no_winner_reward(
    connections, tmp_path
):
    store, control = connections
    older = "arena-2026-09-05-globalgate"
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
