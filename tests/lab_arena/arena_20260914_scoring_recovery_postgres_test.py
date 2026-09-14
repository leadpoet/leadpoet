"""Executable PostgreSQL proof for the September 14 scoring recovery."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from lab_arena import contracts, scoring
from lab_arena.owner_admission import OwnerAdmission
from lab_arena.store import ArenaStore, PsycopgTransport, hash_lease_token
from tests.lab_arena.arena_20260912_recovery_postgres_test import _row_hash
from tests.lab_arena.lab_arena_pg_harness import (
    CURRENT_SERVICE_MIGRATIONS,
    database_with_lab_arena_migration,
)
from tests.lab_arena.test_lab_arena_migration_postgres import (
    claim,
    complete,
    encrypted_runtime_credentials,
    hotkey,
    pass_code_review,
    round_config,
    sha,
    source_submission_doc,
)


ROOT = Path(__file__).resolve().parents[2]
MIGRATION = ROOT / "scripts" / "244-recover-arena-2026-09-14-scoring.sql"
ROUND_ID = "arena-2026-09-14"
UNRELATED_ROUND_ID = "arena-2026-09-15"
TARGET_SUBMISSION = "sub-09f530ce94b6f63221ce4d69962e3988"
TARGET_RUN = f"{ROUND_ID}:{TARGET_SUBMISSION}:1:0:score:1"
TARGET_IDENTITY = (
    "sha256:975bc7fbac081e7f081bdd6bcfc19698"
    "cbdb44c5e4b95b05e5c921fda0f7e6a9"
)
REQUEST_HASH = (
    "sha256:7b87f4d4738cd42dbe01fc94bb38b5e7"
    "e386883b04a2cf26343e6947d143d952"
)
OLD_DIGEST = (
    "sha256:412beed7799ecfcceb9c07c44f36644f"
    "8214c33513a117d603dede98c0b918e9"
)
OLD_REFERENCE = (
    "493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/"
    f"sourcing-model@{OLD_DIGEST}"
)
BASELINE_HOTKEY = "5FNVgRnrxMibhcBGEAaajGrYjsaCn441a5HuGUBUNnxEBLo9"
JUDGE_ONE = hotkey("arena-2026-09-14-judge-one")
JUDGE_TWO = hotkey("arena-2026-09-14-judge-two")
PARTICIPANTS = (
    ("baseline-2026-09-14", BASELINE_HOTKEY, True),
    (TARGET_SUBMISSION, "5CPuRiA715x8PbZetpxfvamsAB4h69Q7Em2sQAUqFK2wfBxe", False),
    ("sub-0bee4ffefcb6a792ca70d16d722fa1ac", "5CDbB7NHrppEHM4QRtoYfRpzKURVKD2mc1F1qqTYU7TTXv4M", False),
    ("sub-287f56f3e3718776a6aa10e4c130e154", "5HbMWHGoAqJphoVejGUqC31dR8rAVQVJjr9iqZBRBxFXnwuN", False),
    ("sub-2968d4fba9582f9d8881bfd998290cbe", "5GsukxFS2cpNr913xjqkAADDAFNC3FhyQBpSqF3yxkjzkSok", False),
    ("sub-4c5b60cfe00763f2d022031ecbd40039", "5C7bgX2NJzH1xWrvatxw88PKnY7d622mZmpBvj7ZUAmWowoy", False),
    ("sub-4e87858d429552dd34271afa3be03a9b", "5H1mHutaCVUVqzF8yZ1wfkEXYWCZmhvLpBeS8Wmfyt6mCMpD", False),
    ("sub-804245a7d75aa38aa4c74a59e07b51d2", "5EvjcxLMtDMuAHv9gFYXtqZ9Qj69km7N7HuhheUEvLjchRa6", False),
    ("sub-8ec81a864da4e10204d803a3fb7c3b58", "5C7ScKx6qjBha2W9AgxucTv87pKy2BRzxmBTKnE3qgKg64YP", False),
    ("sub-946ff9115aec053aeb36a02476175b82", "5H956XR9zjVxgzgPAbamfNrWboXTMp9iLDu7LKrBW4VYKF8r", False),
    ("sub-a7a50fba0fe795bc6f87a433fb6682f3", "5Hp6fhYwgD1HEZJbL2CVbD4GQ8Y3SncFWWLDNQVShxaZNjWk", False),
    ("sub-dae52a6a6cca8db5914a259116bc1b56", "5FtFPnXCEnc1VQuRz3NCsAjcE3LMe3p63Wpxtr27JFgx78Fh", False),
    ("sub-dd76cafd44732cb230a476ab031b24ad", "5FqcBT8JJ2Sr4KHWkpJG4nza9QtwGMSWXucQXeotVUEeM7VX", False),
)
PARTICIPANT_IDS = tuple(sorted(item[0] for item in PARTICIPANTS))


@pytest.fixture(scope="module")
def database():
    yield from database_with_lab_arena_migration(CURRENT_SERVICE_MIGRATIONS)


def _connect(database):
    psycopg2, dsn = database
    return psycopg2.connect(**dsn)


def _scope(position: int) -> dict[str, object]:
    body: dict[str, object] = {
        "schema_version": "leadpoet.lab_arena.judgment_cache_scope.v1",
        "integrity_policy": "arena_integrity_v1",
        "round_id": ROUND_ID,
        "network_name": "finney",
        "netuid": 71,
        "scorer_image_digest": OLD_DIGEST,
        "scorer_image_reference": OLD_REFERENCE,
        "evaluation_date": "2026-09-14",
        "scoring_input_hash": sha(f"sep14-input-{position}"),
    }
    return {**body, "cache_key": contracts.document_hash(body)}


def _prepare(database, *, reservation_amount: int = 49_945_650):
    connection = _connect(database)
    connection.autocommit = True
    transport = PsycopgTransport(lambda: _connect(database))
    store = ArenaStore(transport, lease_ttl_seconds=120)
    with connection.cursor() as cursor:
        cursor.execute(
            "TRUNCATE public.lab_arena_ledger,public.lab_arena_rounds "
            "RESTART IDENTITY CASCADE"
        )
    assert store.create_round(
        UNRELATED_ROUND_ID, round_config(UNRELATED_ROUND_ID, [JUDGE_ONE])
    )["status"] == "created"
    config = round_config(
        ROUND_ID,
        [JUDGE_ONE, JUDGE_TWO],
        stage_1_icps=10,
        max_attempts=2,
        execution_cap_microusd=80_000_000,
        scoring_cap_microusd=50_000_000,
        rewards_enabled=True,
        cost_per_company_microusd=800_000,
    )
    config.update(
        {
            "integrity_policy": "arena_integrity_v1",
            "scorer_policy": scoring.build_scorer_policy(
                scoring_adapter_version="qualification_integrity_v2"
            ),
            "scorer_image_digest": OLD_DIGEST,
            "scorer_image_reference": OLD_REFERENCE,
            "baseline_hotkey": BASELINE_HOTKEY,
            "sourcing_cost_eligibility_policy": "successful_calls_v1",
        }
    )
    assert store.create_round(ROUND_ID, config)["status"] == "created"
    assert store.prepare_confirmation_bank(
        ROUND_ID,
        f"arena/{ROUND_ID}/confirmation/{'c' * 64}.json",
        "sha256:" + "c" * 64,
    )["status"] == "ok"
    participant_docs = []
    for submission_id, miner, is_king in PARTICIPANTS:
        owner = None if is_king else OwnerAdmission(
            hotkey("owner-" + submission_id), 123, "0x" + "d" * 64
        )
        assert store.register_submission(
            ROUND_ID,
            submission_id,
            miner,
            source_submission_doc(ROUND_ID, submission_id, is_king=is_king),
            owner_admission=owner,
        )["status"] == "registered"
        if is_king:
            assert store.update_submission(
                ROUND_ID, submission_id, "uploading", "accepted", {"is_king": True}
            )["status"] == "ok"
        else:
            assert store.accept_submission_with_credentials(
                ROUND_ID,
                submission_id,
                miner,
                encrypted_runtime_credentials(submission_id),
            )["status"] == "ok"
            pass_code_review(store, submission_id, miner)
        assert store.update_submission(
            ROUND_ID, submission_id, "accepted", "frozen", {"is_king": is_king}
        )["status"] == "ok"
        participant_docs.append(
            {"submission_id": submission_id, "miner_hotkey": miner, "is_king": is_king}
        )
    baseline_credentials = encrypted_runtime_credentials("baseline-2026-09-14")
    with connection.cursor() as cursor:
        for provider, ciphertext_b64 in baseline_credentials.items():
            cursor.execute(
                "INSERT INTO public.lab_arena_submission_credentials "
                "(submission_id,miner_hotkey,provider,ciphertext) VALUES "
                "('baseline-2026-09-14',%s,%s,decode(%s,'base64'))",
                (BASELINE_HOTKEY, provider, ciphertext_b64),
            )
    assert store.transition_round(
        ROUND_ID,
        "open",
        "committed",
        {
            "participants": participant_docs,
            "benchmark_ref": f"arena/{ROUND_ID}/benchmark.json",
            "evaluation_date": "2026-09-14",
        },
    )["status"] == "ok"
    assert store.open_stage(ROUND_ID, 1, participant_docs, range(10))["assignments"] == 130

    miners = {submission_id: miner for submission_id, miner, _ in PARTICIPANTS}
    work_items = []
    with connection.cursor() as cursor:
        cursor.execute("ALTER TABLE public.lab_arena_runs DISABLE TRIGGER lab_arena_runs_terminal")
        for submission_id, _miner, _is_king in PARTICIPANTS:
            for position in range(10):
                assignment = f"{ROUND_ID}:{submission_id}:1:{position}"
                run_id = assignment + ":1"
                output_ref = f"arena/{ROUND_ID}/outputs/{run_id}.json"
                cursor.execute(
                    "UPDATE public.lab_arena_runs SET status='accepted',runner_hotkey=%s,"
                    "lease_generation=1,result_doc=%s::jsonb,output_ref=%s,"
                    "terminal_cause='accepted' WHERE run_id=%s",
                    (
                        JUDGE_ONE,
                        json.dumps({"terminal_status": "accepted"}),
                        output_ref,
                        run_id,
                    ),
                )
                work_items.append(
                    {
                        "output_ref": output_ref,
                        "icp_position": position,
                        "scored_run_id": run_id,
                        "submission_id": submission_id,
                    }
                )
        accepted_keys = {
            (item["submission_id"], item["icp_position"])
            for item in work_items
            if item["submission_id"] != TARGET_SUBMISSION
        }
        accepted_keys = set(sorted(accepted_keys)[:37])
        failed_keys = [
            (item["submission_id"], item["icp_position"])
            for item in work_items
            if (item["submission_id"], item["icp_position"]) not in accepted_keys
        ]
        exhausted = set(sorted(failed_keys)[:2])
        for item in work_items:
            submission_id = str(item["submission_id"])
            position = int(item["icp_position"])
            key = (submission_id, position)
            assignment = f"{ROUND_ID}:{submission_id}:1:{position}:score"
            scope = _scope(position)
            causes = ("accepted",) if key in accepted_keys else (
                ("judge_error", "stage_closed") if key in exhausted else ("stage_closed",)
            )
            for attempt, cause in enumerate(causes, start=1):
                cursor.execute(
                    "INSERT INTO public.lab_arena_runs (run_id,assignment_id,round_id,"
                    "submission_id,miner_hotkey,stage,icp_position,attempt,kind,"
                    "scored_run_id,status,runner_hotkey,previous_runner_hotkey,"
                    "stage_generation,lease_generation,result_doc,output_ref,"
                    "terminal_cause,terminal_doc,judgment_cache_key,"
                    "judgment_input_hash,judgment_scope_doc,judgment_group_leader,"
                    "judgment_group_miner_hotkeys) VALUES "
                    "(%s,%s,%s,%s,%s,1,%s,%s,'score',%s,%s,%s,%s,4,%s,%s::jsonb,"
                    "%s,%s,%s::jsonb,%s,%s,%s::jsonb,%s,%s)",
                    (
                        assignment + f":{attempt}",
                        assignment,
                        ROUND_ID,
                        submission_id,
                        miners[submission_id],
                        position,
                        attempt,
                        item["scored_run_id"],
                        "accepted" if cause == "accepted" else "failed",
                        JUDGE_ONE,
                        JUDGE_ONE if attempt == 2 else None,
                        attempt,
                        json.dumps({"terminal_status": cause}),
                        f"arena/{ROUND_ID}/scores/{assignment}:{attempt}.json"
                        if cause == "accepted"
                        else None,
                        cause,
                        json.dumps({"fixture": cause}),
                        scope["cache_key"],
                        scope["scoring_input_hash"],
                        json.dumps(scope),
                        submission_id == min(sid for sid, _, _ in PARTICIPANTS),
                        sorted(miners.values()),
                    ),
                )
        cursor.execute("ALTER TABLE public.lab_arena_runs ENABLE TRIGGER lab_arena_runs_terminal")
        plan = {
            "schema_version": "leadpoet.lab_arena.scoring_plan.v1",
            "round_id": ROUND_ID,
            "stage": 1,
            "work_items": sorted(work_items, key=lambda item: str(item["scored_run_id"])),
            "zero_rows": [],
        }
        cursor.execute("ALTER TABLE public.lab_arena_rounds DISABLE TRIGGER lab_arena_rounds_write_once")
        cursor.execute("ALTER TABLE public.lab_arena_rounds DISABLE TRIGGER lab_arena_icp_set_date_write_once")
        cursor.execute(
            "UPDATE public.lab_arena_rounds SET status='cancelled',status_generation=5,"
            "stage_generation=4,evaluation_date='2026-09-14',icp_set_date='2026-09-13',"
            "cancel_reason='scoring_incomplete',stage1_scoring_plan_doc=%s::jsonb,"
            "configuration_doc=jsonb_set(configuration_doc,'{schedule}',%s::jsonb,false) "
            "WHERE round_id=%s",
            (
                json.dumps(plan),
                json.dumps(
                    {
                        "submission_open": "2026-09-13T00:00:00Z",
                        "submission_cutoff": "2026-09-14T00:00:00Z",
                        "stage_1_scoring_close": "2026-09-14T11:00:01Z",
                        "stage_2_start": "2026-09-14T11:00:02Z",
                    }
                ),
                ROUND_ID,
            ),
        )
        cursor.execute("ALTER TABLE public.lab_arena_rounds ENABLE TRIGGER lab_arena_icp_set_date_write_once")
        cursor.execute("ALTER TABLE public.lab_arena_rounds ENABLE TRIGGER lab_arena_rounds_write_once")
        target_miner = miners[TARGET_SUBMISSION]
        ledger_rows = (
            (348871, "reservation", {"request_hash": REQUEST_HASH}),
            (348872, "dispatch", {}),
            (
                348922,
                "uncertain",
                {"reason": "worker_reported", "call": {"reason": "transport_failure", "call_succeeded": False}},
            ),
        )
        for entry_id, kind, entry_doc in ledger_rows:
            cursor.execute(
                "INSERT INTO public.lab_arena_ledger (entry_id,entry_kind,miner_hotkey,"
                "round_id,submission_id,run_id,stage,call_identity,provider,operation_id,"
                "funding_source,amount_microusd,entry_doc,created_at) VALUES "
                "(%s,%s,%s,%s,%s,%s,1,%s,'deepline','scrapingdog.scrape','miner_key',"
                "%s,%s::jsonb,%s::timestamptz)",
                (
                    entry_id,
                    kind,
                    target_miner,
                    ROUND_ID,
                    TARGET_SUBMISSION,
                    TARGET_RUN,
                    TARGET_IDENTITY,
                    reservation_amount if entry_id == 348871 else 49_945_650,
                    json.dumps(entry_doc),
                    {
                        348871: "2026-09-14T00:43:55.468147Z",
                        348872: "2026-09-14T00:43:55.521619Z",
                        348922: "2026-09-14T00:44:20.528031Z",
                    }[entry_id],
                ),
            )
        cursor.execute(
            "SELECT pg_catalog.setval(pg_get_serial_sequence('public.lab_arena_ledger','entry_id'),"
            "(SELECT max(entry_id) FROM public.lab_arena_ledger),true)"
        )
    return connection, store, transport


def test_recovery_is_append_only_claimable_retryable_and_replay_safe(database):
    connection, store, transport = _prepare(database)
    try:
        with connection.cursor() as cursor:
            old_runs = _row_hash(cursor, "lab_arena_runs", "round_id=%s", (ROUND_ID,))
            old_ledger = _row_hash(cursor, "lab_arena_ledger", "round_id=%s", (ROUND_ID,))
            old_submissions = _row_hash(cursor, "lab_arena_submissions", "round_id=%s", (ROUND_ID,))
            old_credentials = _row_hash(
                cursor,
                "lab_arena_submission_credentials",
                "submission_id=ANY(%s)",
                (list(PARTICIPANT_IDS),),
            )
            old_unrelated = _row_hash(cursor, "lab_arena_rounds", "round_id=%s", (UNRELATED_ROUND_ID,))
            cursor.execute(MIGRATION.read_text(encoding="utf-8"))
            assert _row_hash(
                cursor,
                "lab_arena_runs",
                "round_id=%s AND assignment_id NOT LIKE '%%:recovery244'",
                (ROUND_ID,),
            ) == old_runs
            assert _row_hash(cursor, "lab_arena_submissions", "round_id=%s", (ROUND_ID,)) == old_submissions
            assert _row_hash(
                cursor,
                "lab_arena_submission_credentials",
                "submission_id=ANY(%s)",
                (list(PARTICIPANT_IDS),),
            ) == old_credentials
            assert _row_hash(cursor, "lab_arena_rounds", "round_id=%s", (UNRELATED_ROUND_ID,)) == old_unrelated
            cursor.execute(
                "SELECT count(*),count(DISTINCT assignment_id),"
                "bool_and(attempt=1 AND run_id=assignment_id||':1' AND status='pending' "
                "AND stage_generation=5) FROM public.lab_arena_runs "
                "WHERE round_id=%s AND assignment_id LIKE '%%:recovery244'",
                (ROUND_ID,),
            )
            assert cursor.fetchone() == (93, 93, True)
            cursor.execute(
                "SELECT count(*) FROM (SELECT judgment_cache_key FROM public.lab_arena_runs "
                "WHERE round_id=%s AND assignment_id LIKE '%%:recovery244' "
                "GROUP BY judgment_cache_key HAVING count(*) FILTER "
                "(WHERE judgment_group_leader)<>1) bad",
                (ROUND_ID,),
            )
            assert cursor.fetchone()[0] == 0
            cursor.execute(
                "SELECT amount_microusd,entry_doc,terminal_response FROM public.lab_arena_ledger "
                "WHERE call_identity=%s AND entry_kind='settlement'",
                (TARGET_IDENTITY,),
            )
            amount, entry_doc, terminal = cursor.fetchone()
            assert amount == 2_000
            assert entry_doc["accounting_amount_kind"] == "provider_ledger_time_correlated"
            assert entry_doc["provider_attribution_exact"] is False
            assert terminal["provider_cost"]["units"] == "0.02"
            assert _row_hash(
                cursor,
                "lab_arena_ledger",
                "round_id=%s AND entry_doc->>'arena_244_reconciliation' IS DISTINCT FROM 'true'",
                (ROUND_ID,),
            ) == old_ledger
            cursor.execute("SELECT status,status_generation,stage_generation,cancel_reason FROM public.lab_arena_rounds WHERE round_id=%s", (ROUND_ID,))
            assert cursor.fetchone() == ("stage1_scoring", 6, 5, None)

        first, token, _request_id, _request_hash = claim(store, ROUND_ID, JUDGE_ONE)
        assert first["status"] == "leased"
        failed = complete(store, first["run_id"], hash_lease_token(token), "judge_error")
        assert failed["status"] == "failed"
        assert failed["confirmation_attempt"] == 2
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT run_id,attempt,status,previous_runner_hotkey FROM public.lab_arena_runs "
                "WHERE assignment_id=%s ORDER BY attempt",
                (first["assignment_id"],),
            )
            attempts = cursor.fetchall()
            assert attempts[-1] == (first["assignment_id"] + ":2", 2, "pending", JUDGE_ONE)
            before_replay = _row_hash(cursor, "lab_arena_runs", "round_id=%s", (ROUND_ID,))
            before_ledger_replay = _row_hash(cursor, "lab_arena_ledger", "round_id=%s", (ROUND_ID,))
            cursor.execute(MIGRATION.read_text(encoding="utf-8"))
            assert _row_hash(cursor, "lab_arena_runs", "round_id=%s", (ROUND_ID,)) == before_replay
            assert _row_hash(cursor, "lab_arena_ledger", "round_id=%s", (ROUND_ID,)) == before_ledger_replay

            cursor.execute("ALTER TABLE public.lab_arena_runs DISABLE TRIGGER lab_arena_runs_terminal")
            cursor.execute(
                "UPDATE public.lab_arena_runs SET status='accepted',terminal_cause='accepted',"
                "output_ref='arena/recovered/'||run_id||'.json' WHERE round_id=%s "
                "AND assignment_id LIKE '%%:recovery244' AND assignment_id<>%s "
                "AND status='pending'",
                (ROUND_ID, first["assignment_id"]),
            )
            cursor.execute("ALTER TABLE public.lab_arena_runs ENABLE TRIGGER lab_arena_runs_terminal")
        second, second_token, _request_id, _request_hash = claim(
            store, ROUND_ID, JUDGE_TWO
        )
        assert second["status"] == "leased"
        assert second["run_id"] == first["assignment_id"] + ":2"
        exhausted = complete(
            store,
            second["run_id"],
            hash_lease_token(second_token),
            "judge_error",
        )
        assert exhausted["status"] == "failed"
        assert "confirmation_attempt" not in exhausted
        closed = store.close_scoring(ROUND_ID, 1)
        assert closed["status"] == "closed"
        assert closed["round_status"] == "stage1_judged"
        assert closed["incomplete_assignments"] == 1
    finally:
        connection.close()
        transport.close()


def test_recovery_rolls_back_on_exact_cost_binding_mismatch(database):
    connection, _store, transport = _prepare(database, reservation_amount=49_945_649)
    try:
        with connection.cursor() as cursor:
            before_runs = _row_hash(cursor, "lab_arena_runs", "round_id=%s", (ROUND_ID,))
            before_ledger = _row_hash(cursor, "lab_arena_ledger", "round_id=%s", (ROUND_ID,))
        with pytest.raises(connection.Error, match="cost reservation binding differs"):
            with connection.cursor() as cursor:
                cursor.execute(MIGRATION.read_text(encoding="utf-8"))
        with connection.cursor() as cursor:
            cursor.execute("ROLLBACK")
        with connection.cursor() as cursor:
            assert _row_hash(cursor, "lab_arena_runs", "round_id=%s", (ROUND_ID,)) == before_runs
            assert _row_hash(cursor, "lab_arena_ledger", "round_id=%s", (ROUND_ID,)) == before_ledger
    finally:
        connection.close()
        transport.close()
