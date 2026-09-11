"""Database-backed coverage for per-company accepted judgments."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import threading

import pytest

from lab_arena import company_judgments, contracts, scoring
from lab_arena.owner_admission import OwnerAdmission
from lab_arena.store import ArenaStore, PsycopgTransport, hash_lease_token
from tests.lab_arena.lab_arena_pg_harness import (
    POSTGREST_MIGRATIONS,
    database_with_lab_arena_migration,
)
from tests.lab_arena.test_lab_arena_migration_postgres import (
    _commit_plan,
    _execute_everything,
    claim,
    commit_round,
    encrypted_runtime_credentials,
    frozen_participants as historical_frozen_participants,
    hotkey,
    pass_code_review,
    round_config,
    source_submission_doc,
)


MIGRATION = "20260911200103_lab_arena_company_judgments.sql"


@pytest.fixture(scope="module")
def database():
    migrations = (*POSTGREST_MIGRATIONS, "216-lab-arena-validator-participation.sql", MIGRATION)
    yield from database_with_lab_arena_migration(migrations)


@pytest.fixture()
def store(database):
    psycopg2, dsn = database
    yield ArenaStore(PsycopgTransport(lambda: psycopg2.connect(**dsn)))


def _scope(round_id: str, marker: str, index: int) -> dict:
    input_hash = contracts.document_hash({"company": marker})
    scope = {
        "schema_version": company_judgments.CACHE_SCOPE_SCHEMA_VERSION,
        "integrity_policy": "arena_integrity_v1",
        "company_quality_policy": "company_quality_v1",
        "round_id": round_id,
        "network_name": "finney",
        "netuid": 71,
        "scorer_image_digest": "sha256:" + "a" * 64,
        "scorer_image_reference": (
            "registry.example/lab/scorer@sha256:" + "a" * 64
        ),
        "evaluation_date": "2026-09-02",
        "company_input_hash": input_hash,
    }
    cache_key = contracts.document_hash(scope)
    return {
        "schema_version": company_judgments.COMPANY_REF_SCHEMA_VERSION,
        "company_index": index,
        "cache_key": cache_key,
        "company_input_hash": input_hash,
        "scope_doc": {**scope, "cache_key": cache_key},
    }


def _open(
    store: ArenaStore,
    *,
    round_id: str,
    markers_by_submission: list[list[str]],
    contacts: bool = False,
) -> tuple[list[dict], list[dict]]:
    runner = hotkey("company-cache-runner-" + round_id[-2:])
    config = round_config(
        round_id, [runner], cost_per_company_microusd=1_000
    )
    config.update({
        "integrity_policy": "arena_integrity_v1",
        "company_quality_policy": "company_quality_v1",
        "scorer_policy": scoring.build_scorer_policy(
            scoring_adapter_version=(
                "qualification_contacts_v3"
                if contacts
                else "qualification_integrity_v2"
            ),
            company_quality=True,
        ),
    })
    if contacts:
        config["contact_policy"] = "contacts_v1"
    assert store.create_round(round_id, config)["status"] == "created"
    confirmation_hash = "sha256:" + "f" * 64
    assert store.prepare_confirmation_bank(
        round_id,
        "arena/%s/confirmation/%s.json"
        % (round_id, confirmation_hash.removeprefix("sha256:")),
        confirmation_hash,
    )["status"] == "ok"
    participants = _frozen_participants(
        store, round_id, len(markers_by_submission), prefix=round_id[-2:]
    )
    commit_round(store, round_id, participants)
    assert store.open_stage(
        round_id, 1, participants, list(contracts.stage_positions(1))
    )["status"] == "ok"
    executed = _execute_everything(store, round_id, runner)
    assert store.close_stage(round_id, 1)["status"] == "closed"
    _commit_plan(store, round_id, 1)
    items = []
    for participant, markers in zip(participants, markers_by_submission):
        scored_run_id = executed[(participant["submission_id"], 0)]
        items.append({
            "scored_run_id": scored_run_id,
            "submission_id": participant["submission_id"],
            "icp_position": 0,
            "output_ref": "arena/x/outputs/%s.json" % scored_run_id,
            "company_judgment_refs": [
                _scope(round_id, marker, index)
                for index, marker in enumerate(markers)
            ],
        })
    opened = store.open_scoring(
        round_id, 1, items, company_quality_cache=True
    )
    assert opened["status"] == "ok"
    return participants, items


def _frozen_participants(
    store: ArenaStore, round_id: str, count: int, *, prefix: str
) -> list[dict]:
    participants = []
    for index in range(count):
        miner = hotkey("%s-miner-%d" % (prefix, index))
        submission_id = "%s-sub-%d" % (prefix, index)
        registered = store.register_submission(
            round_id,
            submission_id,
            miner,
            source_submission_doc(round_id, submission_id),
            owner_admission=OwnerAdmission(
                coldkey=hotkey("%s-owner-%d" % (prefix, index)),
                block_number=123,
                block_hash="0x" + "%064x" % (index + 1),
            ),
        )
        assert registered["status"] == "registered"
        assert store.accept_submission_with_credentials(
            round_id,
            submission_id,
            miner,
            encrypted_runtime_credentials(submission_id),
        )["status"] == "ok"
        pass_code_review(store, submission_id, miner)
        assert store.update_submission(
            round_id, submission_id, "accepted", "frozen", {"is_king": False}
        )["status"] == "ok"
        participants.append({
            "submission_id": submission_id,
            "miner_hotkey": miner,
            "is_king": False,
        })
    return participants


def _raw(marker: str) -> dict:
    return {
        "final_score": 72.0,
        "failure_reason": "" if marker != "negative" else "company mismatch",
        "company_identity_key": "domain:%s.example|name:%s" % (marker, marker),
        "company_identity_alias_keys": [
            "domain:%s.example|name:%s" % (marker, marker)
        ],
        "intent_signals_detail": [],
        "verifier_gate_receipts": [{
            "gate": "company_fit",
            "decision": "match" if marker != "negative" else "mismatch",
            "dimension_evidence": {
                "identity": {
                    "web_identity_receipt": {
                        "decision": "match",
                        "evidence_source": "company_web_reverification",
                        "observed_name": marker,
                        "observed_domain": "%s.example" % marker,
                        "observed_linkedin_slug": marker,
                    }
                }
            },
        }],
    }


def _complete(
    store: ArenaStore,
    leased: dict,
    token: str,
    *,
    refs: list[dict],
    completion_marker: str = "f",
) -> dict:
    context = company_judgments.validate_lease_context(
        leased["company_judgment_cache"]
    )
    output = {
        "schema_version": scoring.SCORING_OUTPUT_SCHEMA_VERSION,
        "scored_run_id": leased["scored_run_id"],
        "breakdowns": [],
        "company_judgments": [],
    }
    output_hash = contracts.document_hash(output)
    by_index = {ref["company_index"]: ref for ref in refs}
    source_run = store.get_run(leased["run_id"])
    assert source_run is not None
    evidence_rows = []
    seen = set()
    for miss in context["misses"]:
        if miss["cache_key"] in seen:
            continue
        seen.add(miss["cache_key"])
        ref = by_index[miss["company_index"]]
        new = {
            **miss,
            "raw_judgment": _raw(
                str(ref["scope_doc"]["company_input_hash"])[-8:]
            ),
        }
        evidence = company_judgments.build_evidence_snapshot(
            new_judgment=new,
            company_ref=ref,
            source_score_run_id=leased["run_id"],
            source_scored_run_id=leased["scored_run_id"],
            source_output_ref="arena/x/scores/%s.json" % leased["run_id"],
            source_output_hash=output_hash,
            source_runner_hotkey=source_run["runner_hotkey"],
            source_claim_request_id=source_run["claim_request_id"],
            source_claim_request_hash=source_run["claim_request_hash"],
            source_lease_generation=leased["lease_generation"],
            source_completion_request_hash=(
                "sha256:" + completion_marker * 64
            ),
            runner_authority_exclusions=leased[
                "runner_authority_exclusions"
            ],
        )
        evidence_rows.append({
            "cache_key": evidence["cache_key"],
            "company_input_hash": evidence["company_input_hash"],
            "authority_slot": evidence["authority_slot"],
            "evidence_hash": contracts.document_hash(evidence),
            "evidence_doc": evidence,
        })
    return store.complete_attempt(
        run_id=leased["run_id"],
        lease_token_hash=hash_lease_token(token),
        result={"terminal_status": "accepted"},
        terminal_cause="accepted",
        output_ref="arena/x/scores/%s.json" % leased["run_id"],
        output_hash=output_hash,
        company_judgment_evidence=evidence_rows,
        completion_request_hash="sha256:" + completion_marker * 64,
    )


def test_partial_reuse_and_substantive_negative_are_shared(store):
    round_id = "arena-2026-09-11-c1"
    participants, items = _open(
        store,
        round_id=round_id,
        markers_by_submission=[
            ["same-a", "negative", "same-c"],
            ["same-a", "negative", "changed-c"],
        ],
    )
    runner = hotkey("c1-validator")
    first, token, _, _ = claim(store, round_id, runner, excluded=[runner])
    assert first["status"] == "leased"
    first_item = next(
        item for item in items
        if item["submission_id"] == first["submission_id"]
    )
    assert len(first["company_judgment_cache"]["misses"]) == 3
    assert _complete(store, first, token, refs=first_item["company_judgment_refs"])[
        "company_judgments_stored"
    ] == 3
    accepted_at = store.get_run(first["run_id"])["participation_accepted_at"]
    assert accepted_at is not None

    second, token2, _, _ = claim(store, round_id, runner, excluded=[runner])
    assert second["status"] == "leased"
    assert len(second["company_judgment_cache"]["hits"]) == 2
    assert len(second["company_judgment_cache"]["misses"]) == 1
    second_item = next(
        item for item in items
        if item["submission_id"] == second["submission_id"]
    )
    assert _complete(
        store, second, token2, refs=second_item["company_judgment_refs"],
        completion_marker="e",
    )["company_judgments_reused"] == 2
    assert store.get_run(first["run_id"])["participation_accepted_at"] == accepted_at


def test_simultaneous_overlapping_reservations_do_not_duplicate_or_block(
    store, database
):
    round_id = "arena-2026-09-11-c2"
    _participants, items = _open(
        store,
        round_id=round_id,
        markers_by_submission=[["overlap"], ["overlap"], ["independent"]],
    )
    psycopg2, dsn = database
    barrier = threading.Barrier(2)

    def simultaneous_claim(label: str) -> dict:
        runner = hotkey("c2-validator-" + label)
        transport = PsycopgTransport(lambda: psycopg2.connect(**dsn))
        local_store = ArenaStore(transport)
        try:
            barrier.wait(timeout=10)
            return claim(
                local_store, round_id, runner, excluded=[runner]
            )[0]
        finally:
            transport.close()

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(simultaneous_claim, ("a", "b")))
    leased = [result for result in results if result["status"] == "leased"]
    assert leased
    if len(leased) == 1:
        runner_c = hotkey("c2-validator-c")
        next_result, _token, _, _ = claim(
            store, round_id, runner_c, excluded=[runner_c]
        )
        assert next_result["status"] == "leased"
        leased.append(next_result)
    assert len(leased) == 2
    independent = next(
        item["submission_id"] for item in items
        if item["company_judgment_refs"][0]["cache_key"]
        != leased[0]["company_judgment_cache"]["misses"][0]["cache_key"]
    )
    assert independent in {result["submission_id"] for result in leased}
    leased_keys = [
        result["company_judgment_cache"]["misses"][0]["cache_key"]
        for result in leased
    ]
    assert len(set(leased_keys)) == 2


def test_incompatible_authority_gets_a_new_slot_and_cannot_overwrite(store):
    round_id = "arena-2026-09-11-c3"
    participants, items = _open(
        store, round_id=round_id, markers_by_submission=[["same"], ["same"]]
    )
    runner = hotkey("c3-validator")
    blocked_miner = participants[1]["miner_hotkey"]
    first, token, _, _ = claim(
        store, round_id, runner, excluded=[runner, blocked_miner]
    )
    first_item = next(
        item for item in items
        if item["submission_id"] == first["submission_id"]
    )
    assert _complete(store, first, token, refs=first_item["company_judgment_refs"])[
        "status"
    ] == "accepted"
    second_runner = hotkey("c3-validator-2")
    second, _token2, _, _ = claim(
        store, round_id, second_runner, excluded=[second_runner]
    )
    assert second["submission_id"] == participants[1]["submission_id"]
    assert second["company_judgment_cache"]["hits"] == []
    assert second["company_judgment_cache"]["misses"][0][
        "authority_slot"
    ] == 1

    row = store.get_company_judgments(
        first_item["company_judgment_refs"][0]["cache_key"]
    )[0]
    assert row["authority_slot"] == 0


def test_expired_leader_retries_same_refs_and_stale_completion_loses(
    store, database
):
    round_id = "arena-2026-09-11-c4"
    _participants, items = _open(
        store, round_id=round_id, markers_by_submission=[["expire"]]
    )
    runner = hotkey("c4-validator")
    leased, token, _, _ = claim(store, round_id, runner, excluded=[runner])
    original = store.get_run(leased["run_id"])
    psycopg2, dsn = database
    with psycopg2.connect(**dsn) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "UPDATE public.lab_arena_runs "
                "SET lease_expires_at = clock_timestamp() - interval '1 second' "
                "WHERE run_id = %s",
                (leased["run_id"],),
            )
    assert store.expire_leases(round_id)["retried"] == 1
    retry = next(
        row for row in store.list_runs(round_id, stage=1, kind="score")
        if row["attempt"] == 2
    )
    assert retry["company_judgment_refs"] == original["company_judgment_refs"]
    stale = _complete(
        store, leased, token, refs=items[0]["company_judgment_refs"]
    )
    assert stale["status"] == "failed" and stale["idempotent"] is True
    assert store.get_company_judgments(
        items[0]["company_judgment_refs"][0]["cache_key"]
    ) == []


def test_contact_quality_round_uses_the_same_company_cache_protocol(store):
    round_id = "arena-2026-09-11-c5"
    _open(
        store,
        round_id=round_id,
        markers_by_submission=[["contact"]],
        contacts=True,
    )
    runner = hotkey("c5-validator")
    leased, _token, _, _ = claim(store, round_id, runner, excluded=[runner])
    assert leased["status"] == "leased"
    assert len(leased["company_judgment_cache"]["misses"]) == 1


def test_historical_round_claim_has_no_company_cache_payload(store):
    round_id = "arena-2026-09-11-c6"
    runner = hotkey("c6-runner")
    assert store.create_round(round_id, round_config(round_id, [runner]))[
        "status"
    ] == "created"
    participants = historical_frozen_participants(
        store, round_id, 1, prefix="c6"
    )
    commit_round(store, round_id, participants)
    assert store.open_stage(
        round_id, 1, participants, list(contracts.stage_positions(1))
    )["status"] == "ok"
    executed = _execute_everything(store, round_id, runner)
    assert store.close_stage(round_id, 1)["status"] == "closed"
    _commit_plan(store, round_id, 1)
    scored_run_id = executed[(participants[0]["submission_id"], 0)]
    assert store.open_scoring(round_id, 1, [{
        "scored_run_id": scored_run_id,
        "submission_id": participants[0]["submission_id"],
        "icp_position": 0,
        "output_ref": "arena/x/outputs/%s.json" % scored_run_id,
    }])["status"] == "ok"
    leased, _token, _, _ = claim(
        store, round_id, hotkey("c6-score-validator")
    )
    assert leased["status"] == "leased"
    assert "company_judgment_cache" not in leased
