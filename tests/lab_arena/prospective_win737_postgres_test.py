"""Exercise the normal publish-to-reward authorities with a later, immutable no-win."""

import json
import re
from pathlib import Path

import pytest

from lab_arena import rewards, signing
from tests.lab_arena import correction_reward_history696_postgres_test as history
from tests.lab_arena.reward_slots_postgres_test import _publication, BASELINE, MINER
from tests.lab_arena.total_emissions_reward_slots681_postgres_test import POLICY

ROOT = Path(__file__).parents[2] / "scripts"
OLD, LATER = "arena-2026-11-23", "arena-2026-11-24"
BASIS = "sha256:94f99614eaed7e18b7e8114dc9e69b1a5c03fa2dddbe436d390216b94939abe7"
SQL = ROOT / "737-lab-arena-nov23-prospective-win.sql"


@pytest.fixture(scope="module")
def database():
    db = history.database.__wrapped__()
    pg, dsn = next(db)
    try:
        with pg.connect(**dsn) as c, c.cursor() as q:
            q.execute(next(ROOT.glob("698-*.sql")).read_text())
            # Install only the literal freshness migrations used by this reward fixture.
            src = next(ROOT.glob("627-*.sql")).read_text()
            q.execute(
                src[
                    src.index("CREATE OR REPLACE FUNCTION") : src.index(
                        "CREATE TEMP TABLE arena627_patches"
                    )
                ]
            )
            q.execute(next(ROOT.glob("640-*.sql")).read_text())
            src = next(ROOT.glob("678-*.sql")).read_text()
            q.execute(
                src[
                    src.index("CREATE OR REPLACE FUNCTION") : src.index(
                        "CREATE OR REPLACE FUNCTION public.lab_arena__postpublication_judge_v1"
                    )
                ]
            )
            patch = re.search(r"DO \$freshness678\$.*?\$freshness678\$;", src, re.S)[0]
            patch = re.sub(
                r",\n \('public.lab_arena_freeze_full_thinking_baseline_v1.*?\n \)f\(",
                "\n )f(",
                patch,
                flags=re.S,
            )
            q.execute(patch)
            q.execute(SQL.read_text())
            q.execute(SQL.read_text())
        yield pg, dsn
    finally:
        db.close()


def seeded(q, *, won):
    # Seed only prior facts. Preparation, completion and activation use normal RPCs.
    q.execute("SET LOCAL session_replication_role=replica")

    def row(rid, day, publication):
        config = dict(
            mode="live",
            rewards_enabled=True,
            baseline_hotkey=BASELINE,
            network_name="finney",
            netuid=71,
            stage_1_icp_count=1,
            stage_2_icp_count=1,
            scorer_policy={},
            scorer_image_digest="sha256:" + "a" * 64,
            scorer_image_reference="test/scorer",
            reward_constants=rewards.reward_constants_document(),
        )
        q.execute(
            """INSERT INTO lab_arena_rounds(round_id,evaluation_date,status,rewards_enabled,
        configuration_doc,publication_doc,king_outcome,king_hotkey,published_at,promotion_required,participants)
        VALUES(%s,%s,'published',true,%s,%s,%s,%s,%s,true,%s)""",
            (
                rid,
                day,
                json.dumps(config),
                json.dumps(publication),
                publication["king_decision"]["outcome"],
                publication["king_decision"].get("king_hotkey") or None,
                publication["published_at"],
                json.dumps(
                    [
                        dict(
                            submission_id=rid + "-baseline",
                            miner_hotkey=BASELINE,
                            is_king=True,
                        )
                    ]
                ),
            ),
        )

    old = _publication(OLD, 8)
    now = "2026-11-24T08:30:00Z"
    old["published_at"] = now
    if not won:
        old["king_decision"] = dict(outcome="no_king")
    row(OLD, "2026-11-23", old)
    later = _publication(LATER, 0)
    later["king_decision"] = dict(outcome="no_king")
    row(LATER, "2026-11-24", later)
    previous = "arena-2026-11-18"
    prior = _publication(previous, 20)
    row(previous, "2026-11-18", prior)
    # Nov24's ruling basis retains a seat which the earlier basis did not represent.
    seat = dict(
        round_id=previous,
        submission_id=previous + "-winner",
        miner_hotkey=MINER,
        baseline_submission_id=previous + "-baseline",
        baseline_score=40,
        winner_score=60,
        start_epoch=50,
    )
    q.execute(
        """UPDATE lab_arena_rounds SET signing_key_doc='{}', effective_reward_epoch=50,
                baseline_promoted_at=clock_timestamp(),reward_activated_at=clock_timestamp(),
                reward_basis_hash=%s,reward_basis_doc=%s WHERE round_id=%s""",
        (
            "sha256:" + "a" * 64,
            json.dumps(
                dict(
                    round_id=previous,
                    effective_reward_epoch=50,
                    slot_policy=POLICY,
                    reward_slots=[None] * 3,
                    schema_version="leadpoet.lab_arena.reward_basis.v2",
                    king_outcome="no_king",
                )
            ),
            previous,
        ),
    )
    carry = dict(
        schema_version="leadpoet.lab_arena.reward_basis.v2",
        round_id=LATER,
        effective_reward_epoch=80,
        slot_policy=POLICY,
        reward_slots=[seat, None, None],
        reward_basis_hash=BASIS,
        king_outcome="no_king",
    )
    q.execute(
        """UPDATE lab_arena_rounds SET reward_basis_hash=%s,signing_key_doc='{}',reward_basis_doc=%s,
                 reward_activated_at=clock_timestamp(),effective_reward_epoch=80 WHERE round_id=%s""",
        (BASIS, json.dumps(carry), LATER),
    )
    for sid, hk, king in [
        (OLD + "-winner", MINER, False),
        (OLD + "-baseline", BASELINE, True),
    ]:
        q.execute(
            "INSERT INTO lab_arena_submissions(submission_id,round_id,miner_hotkey,status,is_king)VALUES(%s,%s,%s,'frozen',%s)",
            (sid, OLD, hk, king),
        )
    for pos in range(2):
        execution = f"{OLD}-execute-{pos}"
        for kind in ("execute", "score"):
            q.execute(
                """INSERT INTO lab_arena_runs(run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,icp_position,attempt,kind,status,terminal_cause,per_icp_score,output_ref,result_doc,scored_run_id,scoring_generation_floor)VALUES(%s,%s,%s,%s,%s,%s,%s,1,%s,'accepted','accepted',0,'fixture-output','{"terminal_status":"accepted"}',%s,0)""",
                (
                    execution if kind == "execute" else execution + "-score",
                    execution if kind == "execute" else execution + ":score",
                    OLD,
                    OLD + "-baseline",
                    BASELINE,
                    pos + 1,
                    pos,
                    kind,
                    None if kind == "execute" else execution,
                ),
            )
    q.execute(
        """UPDATE lab_arena_runs SET judgment_scope_doc=%s WHERE round_id=%s AND kind='score'""",
        (
            json.dumps(
                dict(
                    scorer_image_digest="sha256:" + "a" * 64,
                    scorer_image_reference="test/scorer",
                )
            ),
            OLD,
        ),
    )
    q.execute("SET LOCAL session_replication_role=origin")
    return now


def authorize(q):
    doc = dict(
        operation="round_operational_update_v1",
        reason="nov23_dispatch_after_predecessor",
        later_round_id=LATER,
        later_reward_basis_hash=BASIS,
    )
    q.execute(
        """INSERT INTO lab_arena_scoring_supersessions(round_id,generation_floor,request_id,request_hash,
              actor_ref,transaction_id,backend_pid,old_round_doc,new_round_doc,old_score_projections,recovery_doc)
              VALUES(%s,1,'nov23-restore-unclaimed-736-v1','sha256:'||encode(extensions.digest(%s::jsonb::text,'sha256'),'hex'),
              'authorized-nov23-dispatch-recovery',txid_current(),pg_backend_pid(),
              '{"status":"cancelled","cancel_reason":"scoring_incomplete"}','{"status":"committed"}','{}',%s)""",
        (OLD, json.dumps(doc), json.dumps(doc)),
    )


def scalar(q, sql, params=()):
    q.execute(sql, params)
    return q.fetchone()[0]


@pytest.mark.parametrize("won", [True, False])
def test_only_authorized_win_can_earn_future_rewards(database, won):
    pg, dsn = database
    c = pg.connect(**dsn)
    try:
        with c.cursor() as q:
            now = seeded(q, won=won)

            def snapshot():
                return scalar(
                    q,
                    "SELECT to_jsonb(r) FROM lab_arena_rounds r WHERE round_id=%s",
                    (LATER,),
                )

            before = snapshot()
            assert (
                scalar(q, "SELECT lab_arena_prospective_win_737(%s)", (OLD,)) is False
            )
            authorize(q)
            assert scalar(q, "SELECT lab_arena_prospective_win_737(%s)", (OLD,)) is True
            if not won:
                assert (
                    scalar(q, "SELECT lab_arena_activate_reward(%s,'{}','{}')", (OLD,))[
                        "status"
                    ]
                    == "superseded"
                )
            else:
                plan = json.dumps(
                    dict(
                        commit="c" * 40,
                        main_before="a" * 40,
                        lab_before="b" * 40,
                        timestamp=now,
                    )
                )
                for rpc, status in [("prepare", "prepared"), ("complete", "promoted")]:
                    assert (
                        scalar(
                            q, f"SELECT lab_arena_{rpc}_promotion(%s,%s)", (OLD, plan)
                        )["status"]
                        == status
                    )
                slots = scalar(
                    q,
                    "SELECT lab_arena_reward_slot_snapshot(%s,%s)",
                    (OLD, json.dumps(POLICY)),
                )["reward_slots"]
                assert slots[0] == before["reward_basis_doc"]["reward_slots"][0]
                assert (
                    slots[1] is None
                    and slots[2]["round_id"] == OLD
                    and slots[2]["start_epoch"] is None
                )
                signer = signing.LocalSigner.generate()
                doc = rewards.reward_basis_document(
                    round_id=OLD,
                    published_at=now,
                    finalized_epoch=99,
                    king_hotkey=MINER,
                    king_outcome="crowned",
                    slot_policy=POLICY,
                    reward_slots=slots,
                )
                basis = json.dumps(
                    signing.sign_document(signer, doc, hash_field="reward_basis_hash")
                )
                key = json.dumps(signing.signing_key_document(signer.public_key_der))
                assert (
                    scalar(
                        q,
                        "SELECT lab_arena_activate_reward(%s,%s,%s)",
                        (OLD, basis, key),
                    )["status"]
                    == "activated"
                )
                assert (
                    scalar(
                        q,
                        "SELECT lab_arena_activate_reward(%s,%s,%s)",
                        (OLD, basis, key),
                    )["status"]
                    == "existing"
                )
            assert snapshot() == before
    finally:
        c.rollback()
        c.close()


@pytest.mark.parametrize(
    "fault", ["later_frozen", "later_winner", "wrong_signed_basis"]
)
def test_wrong_later_authority_remains_superseded(database, fault):
    pg, dsn = database
    conn = pg.connect(**dsn)
    try:
        with conn.cursor() as q:
            seeded(q, won=True)
            authorize(q)
            q.execute("SET LOCAL session_replication_role=replica")
            if fault == "later_frozen":
                q.execute(
                    """INSERT INTO lab_arena_rounds(round_id,status,evaluation_date,rewards_enabled,configuration_doc,champion_funding_frozen)
                  SELECT 'arena-2026-11-25','committed','2026-11-25',true,configuration_doc,true FROM lab_arena_rounds WHERE round_id=%s""",
                    (LATER,),
                )
            elif fault == "later_winner":
                q.execute(
                    "UPDATE lab_arena_rounds SET king_outcome='crowned' WHERE round_id=%s",
                    (LATER,),
                )
            else:
                q.execute(
                    "UPDATE lab_arena_rounds SET reward_basis_hash=%s WHERE round_id=%s",
                    ("sha256:" + "e" * 64, LATER),
                )
            q.execute("SET LOCAL session_replication_role=origin")
            assert not scalar(q, "SELECT lab_arena_prospective_win_737(%s)", (OLD,))
            assert (
                scalar(q, "SELECT lab_arena_prepare_promotion(%s,'{}')", (OLD,))[
                    "status"
                ]
                == "superseded"
            )
    finally:
        conn.rollback()
        conn.close()


def test_next_full_thinking_baseline_waits_for_approved_crown(database):
    pg, dsn = database
    conn = pg.connect(**dsn)
    try:
        with conn.cursor() as q:
            seeded(q, won=True)
            authorize(q)
            q.execute(
                """INSERT INTO lab_arena_rounds(round_id,status,evaluation_date,rewards_enabled,configuration_doc)
             SELECT 'arena-2026-11-25','open','2026-11-25',true,
             configuration_doc || '{"reasoning_capture_policy":"full_generated_thinking_v1"}'::jsonb
             FROM lab_arena_rounds WHERE round_id=%s""",
                (LATER,),
            )
            assert not scalar(
                q,
                "SELECT lab_arena__baseline_source_current_v1(r) FROM lab_arena_rounds r WHERE round_id='arena-2026-11-25'",
            )
    finally:
        conn.rollback()
        conn.close()


def test_python_driver_requires_the_same_db_authority_for_a_genuine_historical_win():
    from tests.lab_arena.monotonic_day_authority_test import _row, _service

    old, later = _row("2026-11-23"), _row("2026-11-24")
    service = _service([old, later])
    service._store.prospective_win_ready = lambda rid: rid == OLD
    assert not service._superseded_by_published_day(old)
    old["publication_doc"]["king_decision"]["outcome"] = "no_king"
    assert service._superseded_by_published_day(old)
    old["publication_doc"]["king_decision"]["outcome"] = "crowned"
    service._store.prospective_win_ready = lambda rid: False
    assert service._superseded_by_published_day(old)
