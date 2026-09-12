"""Adversarial persistence tests for daily champion credential funding."""

from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor

import pytest

from lab_arena import contracts
from lab_arena.store import ArenaStore, ArenaStoreError, PsycopgTransport, hash_lease_token
from tests.lab_arena.lab_arena_pg_harness import (
    POSTGREST_MIGRATIONS,
    database_with_lab_arena_migration,
)
from tests.lab_arena.participation_original_judgments_postgres_test import (
    _has_recent_participation,
)
from tests.lab_arena.test_lab_arena_migration_postgres import (
    claim,
    commit_round,
    frozen_participants,
    hotkey,
    round_config,
    sha,
    source_submission_doc,
)


MIGRATION = "227-lab-arena-champion-funding.sql"
CURRENT_MIGRATIONS = tuple(
    dict.fromkeys(POSTGREST_MIGRATIONS + (MIGRATION,))
)


@pytest.fixture()
def database():
    yield from database_with_lab_arena_migration(CURRENT_MIGRATIONS)


@pytest.fixture(scope="module")
def old_database():
    yield from database_with_lab_arena_migration(
        tuple(name for name in POSTGREST_MIGRATIONS if name != MIGRATION)
    )


def _store(database):
    psycopg2, dsn = database
    return ArenaStore(PsycopgTransport(lambda: psycopg2.connect(**dsn)))


def _seed_champion_run(database, label: str, *, execution_cap=5_000_000):
    psycopg2, dsn = database
    connect = lambda: psycopg2.connect(**dsn)
    store = ArenaStore(PsycopgTransport(connect))
    prior_id = "arena-2026-09-10-%s" % label
    current_id = "arena-2026-09-11-%s" % label
    prior_runner = hotkey(label + "-prior-runner")
    assert store.create_round(
        prior_id,
        round_config(prior_id, [prior_runner], rewards_enabled=True),
    )["status"] == "created"
    champion = frozen_participants(
        store, prior_id, 1, prefix=label + "-champion"
    )[0]
    commit_round(store, prior_id, [champion])
    with connect() as connection, connection.cursor() as cursor:
        # The fixture needs only a completed promotion owner. Publication
        # contract behavior has separate end-to-end tests.
        cursor.execute(
            "ALTER TABLE public.lab_arena_rounds DISABLE TRIGGER USER"
        )
        cursor.execute(
            "UPDATE public.lab_arena_rounds SET status='published', "
            "promotion_required=TRUE, published_at=clock_timestamp(), "
            "baseline_promoted_at=clock_timestamp(), publication_doc=%s::jsonb "
            "WHERE round_id=%s",
            (
                contracts.canonical_json(
                    {
                        "king_decision": {
                            "outcome": "crowned",
                            "winner_submission_id": champion["submission_id"],
                            "king_hotkey": champion["miner_hotkey"],
                        }
                    }
                ),
                prior_id,
            ),
        )
        cursor.execute(
            "ALTER TABLE public.lab_arena_rounds ENABLE TRIGGER USER"
        )

    runner = hotkey(label + "-runner")
    assert store.create_round(
        current_id,
        round_config(
            current_id,
            [runner],
            execution_cap_microusd=execution_cap,
            quotas={"openrouter": 30, "deepline": 30, "scrapingdog": 30},
            rewards_enabled=True,
        ),
    )["status"] == "created"
    with connect() as connection, connection.cursor() as cursor:
        cursor.execute(
            "SELECT submission.submission_id, submission.miner_hotkey "
            "FROM public.lab_arena_rounds AS prior "
            "JOIN public.lab_arena_rounds AS current ON current.round_id=%s "
            "JOIN public.lab_arena_submissions AS submission ON "
            "submission.submission_id=prior.publication_doc #>> "
            "'{king_decision,winner_submission_id}' "
            "AND submission.round_id=prior.round_id "
            "AND submission.miner_hotkey=prior.publication_doc #>> "
            "'{king_decision,king_hotkey}' "
            "AND submission.status='frozen' AND NOT submission.is_king "
            "WHERE prior.arena_network_name=current.arena_network_name "
            "AND prior.arena_netuid=current.arena_netuid "
            "AND prior.configuration_doc ->> 'mode'="
            "current.configuration_doc ->> 'mode' "
            "AND prior.status='published' "
            "AND prior.baseline_promoted_at IS NOT NULL "
            "AND prior.publication_doc #>> '{king_decision,outcome}'='crowned'",
            (current_id,),
        )
        assert cursor.fetchone() == (
            champion["submission_id"], champion["miner_hotkey"]
        )
    assert store.freeze_champion_funding(current_id)["status"] == "frozen"
    baseline_id = "baseline-" + current_id.removeprefix("arena-")
    baseline_hotkey = hotkey("baseline")
    assert store.register_submission(
        current_id,
        baseline_id,
        baseline_hotkey,
        source_submission_doc(current_id, baseline_id, is_king=True),
    )["status"] == "registered"
    assert store.update_submission(
        current_id, baseline_id, "uploading", "accepted"
    )["status"] == "ok"
    assert store.update_submission(
        current_id, baseline_id, "accepted", "frozen", {"is_king": True}
    )["status"] == "ok"
    participant = {
        "submission_id": baseline_id,
        "miner_hotkey": baseline_hotkey,
        "is_king": True,
    }
    commit_round(store, current_id, [participant])
    assert store.open_stage(
        current_id, 1, [participant], list(contracts.stage_positions(1))
    )["status"] == "ok"
    run, token, _, _ = claim(store, current_id, runner)
    assert run["status"] == "leased"
    stored_run = store.get_run(run["run_id"])
    assert stored_run["champion_funding_sources"] == {
        "openrouter": "miner_key",
        "deepline": "miner_key",
        "scrapingdog": "miner_key",
    }, stored_run
    return store, connect, current_id, champion, runner, run, token


def _account_call(
    store: ArenaStore,
    run,
    token: str,
    *,
    provider: str,
    action_sequence: int,
    provider_attempt: int,
    provider_status: int,
):
    operation = (
        "openrouter.chat" if provider == "openrouter" else "deepline.execute"
    )
    request_hash = sha("%s-%s" % (run["run_id"], action_sequence))
    base = contracts.provider_call_identity(
        attempt=run["attempt"],
        assignment_id=run["assignment_id"],
        icp_position=run["icp_position"],
        action_sequence=action_sequence,
        operation_id=operation,
        request_hash=request_hash,
    )
    identity = (
        base
        if provider_attempt == 1
        else contracts.document_hash(
            {"base_call_identity": base, "provider_attempt": provider_attempt}
        )
    )
    call_doc = {
        "request_hash": request_hash,
        "base_call_identity": base,
        "provider_attempt": provider_attempt,
        "action_sequence": action_sequence,
    }
    reserved = store.reserve_call(
        run_id=run["run_id"],
        lease_token_hash=hash_lease_token(token),
        call_identity=identity,
        operation_id=operation,
        provider=provider,
        funding_source="miner_key",
        amount_microusd=1 if provider == "openrouter" else 0,
        call_doc={
            **call_doc,
            **({"reserve_remaining_budget": True} if provider == "deepline" else {}),
        },
    )
    if reserved["status"] != "reserved":
        return reserved
    assert store.mark_dispatched(
        run_id=run["run_id"],
        lease_token_hash=hash_lease_token(token),
        call_identity=identity,
    )["status"] == "dispatched"
    evidence = {
        "error_class": "account_credential_failure",
        "provider_status": provider_status,
        "base_call_identity": base,
        "provider_attempt": provider_attempt,
        "action_sequence": action_sequence,
    }
    if provider == "deepline":
        assert store.mark_uncertain(
            run_id=run["run_id"],
            lease_token_hash=hash_lease_token(token),
            call_identity=identity,
            call_doc={
                "reason": "missing_provider_cost",
                "provider_status": provider_status,
                "account_failure_evidence": evidence,
            },
        )["status"] == "uncertain"
    else:
        body = base64.b64encode(
            b'{"error":{"code":"miner_credentials_unavailable"}}'
        ).decode()
        assert store.settle_call(
            run_id=run["run_id"],
            lease_token_hash=hash_lease_token(token),
            call_identity=identity,
            actual_microusd=0,
            terminal_response={
                "status": 402,
                "headers": {"content-type": "application/json"},
                "body_b64": body,
                "account_failure_evidence": evidence,
            },
        )["status"] == "settled"
    return {
        "status": "recorded",
        "base_call_identity": base,
        "call_identity": identity,
        "request_hash": request_hash,
        "evidence": evidence,
    }


def test_exact_owner_snapshot_has_no_secret_and_is_immutable(database):
    store, connect, round_id, champion, _runner, run, _token = (
        _seed_champion_run(database, "owner")
    )
    funding = store.provider_funding(run["run_id"], "openrouter")
    assert funding == {
        "status": "available",
        "funding_source": "miner_key",
        "champion_funding": True,
        "credential_submission_id": champion["submission_id"],
        "credential_miner_hotkey": champion["miner_hotkey"],
        "restart_required": False,
    }
    assert "ciphertext" not in contracts.canonical_json(funding)
    with connect() as connection, connection.cursor() as cursor:
        with pytest.raises(Exception, match="champion funding owner is immutable"):
            cursor.execute(
                "UPDATE public.lab_arena_rounds SET champion_hotkey=%s "
                "WHERE round_id=%s",
                (hotkey("forged-owner"), round_id),
            )


def test_four_settled_account_failures_latch_once_under_parallel_calls(database):
    store, _connect, round_id, champion, _runner, run, token = (
        _seed_champion_run(database, "parallel")
    )
    base = None
    for attempt in range(1, 5):
        recorded = _account_call(
            store,
            run,
            token,
            provider="openrouter",
            action_sequence=4,
            provider_attempt=attempt,
            provider_status=401,
        )
        assert recorded["status"] == "recorded"
        base = recorded["base_call_identity"]
    evidence = {
        "error_class": "account_credential_failure",
        "provider_status": 401,
        "base_call_identity": base,
        "provider_attempts": 4,
        "action_sequence": 4,
    }
    with pytest.raises(ArenaStoreError):
        store.mark_champion_provider_fallback(
            run["run_id"],
            hash_lease_token(token),
            "deepline",
            {**evidence, "provider_status": 503},
        )

    def mark():
        local = _store(database)
        try:
            return local.mark_champion_provider_fallback(
                run["run_id"], hash_lease_token(token), "openrouter", evidence
            )
        finally:
            local.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _index: mark(), range(2)))
    assert sorted(result["status"] for result in results) == ["existing", "marked"]
    funding = store.provider_funding(run["run_id"], "openrouter")
    assert funding["funding_source"] == "miner_key"
    assert funding["restart_required"] is True
    assert funding["credential_submission_id"] == champion["submission_id"]
    assert store.mark_champion_provider_fallback(
        run["run_id"],
        hash_lease_token(token),
        "deepline",
        evidence,
    )["status"] == "existing"
    round_row = store.get_round(round_id)
    assert round_row["champion_fallback_providers"] == ["openrouter"]


@pytest.mark.parametrize("account_status", [401, 429])
def test_uncertain_auth_liability_allows_host_fallback_without_erasing_cost(
    database, account_status
):
    store, _connect, round_id, _champion, runner, run, token = (
        _seed_champion_run(
            database,
            "unc%s" % account_status,
            execution_cap=5_000_000,
        )
    )
    first = _account_call(
        store,
        run,
        token,
        provider="deepline",
        action_sequence=2,
        provider_attempt=1,
        provider_status=account_status,
    )
    assert first["status"] == "recorded"
    replay = store.reserve_call(
        run_id=run["run_id"],
        lease_token_hash=hash_lease_token(token),
        call_identity=first["call_identity"],
        operation_id="deepline.execute",
        provider="deepline",
        funding_source="miner_key",
        amount_microusd=0,
        call_doc={
            "request_hash": first["request_hash"],
            "base_call_identity": first["base_call_identity"],
            "provider_attempt": 1,
            "action_sequence": 2,
            "reserve_remaining_budget": True,
        },
    )
    assert replay["status"] == "uncertain"
    assert replay["account_failure_evidence"] == first["evidence"]
    for attempt in range(2, 5):
        refused = _account_call(
            store,
            run,
            token,
            provider="deepline",
            action_sequence=2,
            provider_attempt=attempt,
            provider_status=account_status,
        )
        assert refused["status"] == "refused"
        assert refused["prior_miner_credential_refusal"] is True
    assert store.mark_champion_provider_fallback(
        run["run_id"],
        hash_lease_token(token),
        "deepline",
        {
            "error_class": "account_credential_failure",
            "provider_status": account_status,
            "base_call_identity": first["base_call_identity"],
            "provider_attempts": 4,
            "action_sequence": 2,
        },
    )["status"] == "marked"
    before_restart = len(store.list_ledger(run_id=run["run_id"]))
    refused_old_run = store.reserve_call(
        run_id=run["run_id"],
        lease_token_hash=hash_lease_token(token),
        call_identity=sha("old-run-must-not-charge"),
        operation_id="deepline.execute",
        provider="deepline",
        funding_source="miner_key",
        amount_microusd=1,
        call_doc={"request_hash": sha("old-run-request")},
    )
    assert refused_old_run["status"] == "champion_restart_required"
    assert len(store.list_ledger(run_id=run["run_id"])) == before_restart
    completed = store.complete_attempt(
        run_id=run["run_id"],
        lease_token_hash=hash_lease_token(token),
        result={"terminal_status": "accepted"},
        terminal_cause="accepted",
        output_ref="arena/forged-accepted-output.json",
    )
    assert completed["status"] == "failed"
    retry, retry_token, _, _ = claim(store, round_id, runner)
    assert retry["assignment_id"] == run["assignment_id"]
    assert store.provider_funding(retry["run_id"], "deepline")[
        "funding_source"
    ] == "host"
    host_identity = sha("host-fallback-call")
    admitted = store.reserve_call(
        run_id=retry["run_id"],
        lease_token_hash=hash_lease_token(retry_token),
        call_identity=host_identity,
        operation_id="deepline.execute",
        provider="deepline",
        funding_source="host",
        amount_microusd=0,
        call_doc={"reserve_remaining_budget": True},
    )
    assert admitted["status"] == "reserved"
    assert admitted["amount_microusd"] == 5_000_000
    assert store.mark_dispatched(
        run_id=retry["run_id"],
        lease_token_hash=hash_lease_token(retry_token),
        call_identity=host_identity,
    )["status"] == "dispatched"
    assert store.settle_call(
        run_id=retry["run_id"],
        lease_token_hash=hash_lease_token(retry_token),
        call_identity=host_identity,
        actual_microusd=5_000_000,
        terminal_response={
            "status": 200,
            "headers": {"content-type": "application/json"},
            "body_b64": base64.b64encode(b"{}").decode(),
        },
    )["status"] == "settled"
    settled_spend_is_gated = store.reserve_call(
        run_id=retry["run_id"],
        lease_token_hash=hash_lease_token(retry_token),
        call_identity=sha("host-settled-spend-gated"),
        operation_id="deepline.execute",
        provider="deepline",
        funding_source="host",
        amount_microusd=0,
        call_doc={"reserve_remaining_budget": True},
    )
    assert settled_spend_is_gated["status"] == "refused"
    assert settled_spend_is_gated["reason"] == "money_cap"
    old_heads = store.list_ledger(run_id=run["run_id"])
    uncertain = [row for row in old_heads if row["entry_kind"] == "uncertain"]
    assert len(uncertain) == 1
    assert uncertain[0]["amount_microusd"] == 5_000_000


def test_fallback_retry_participation_is_owned_and_immutable(database):
    store, _connect, round_id, _champion, failed_runner, run, token = (
        _seed_champion_run(database, "participation")
    )
    base_identity = None
    for provider_attempt in range(1, 5):
        recorded = _account_call(
            store,
            run,
            token,
            provider="openrouter",
            action_sequence=3,
            provider_attempt=provider_attempt,
            provider_status=401,
        )
        assert recorded["status"] == "recorded"
        base_identity = recorded["base_call_identity"]
    assert store.mark_champion_provider_fallback(
        run["run_id"],
        hash_lease_token(token),
        "openrouter",
        {
            "error_class": "account_credential_failure",
            "provider_status": 401,
            "base_call_identity": base_identity,
            "provider_attempts": 4,
            "action_sequence": 3,
        },
    )["status"] == "marked"

    forced_failure = store.complete_attempt(
        run_id=run["run_id"],
        lease_token_hash=hash_lease_token(token),
        result={"terminal_status": "accepted"},
        terminal_cause="accepted",
        output_ref="arena/forged-fallback-acceptance.json",
    )
    assert forced_failure["status"] == "failed"
    failed_row = store.get_run(run["run_id"])
    assert failed_row["status"] == "failed"
    assert failed_row["participation_accepted_at"] is None
    assert not _has_recent_participation(database, failed_runner)

    accepted_runner = hotkey("champion-participation-retry-runner")
    retry, retry_token, _, _ = claim(store, round_id, accepted_runner)
    assert retry["assignment_id"] == run["assignment_id"]
    assert retry["attempt"] == 2
    retry_row = store.get_run(retry["run_id"])
    assert retry_row["runner_hotkey"] == accepted_runner
    assert retry_row["participation_accepted_at"] is None
    assert store.provider_funding(retry["run_id"], "openrouter")[
        "funding_source"
    ] == "host"
    assert not _has_recent_participation(database, accepted_runner)

    generation_id = "gen-champion-participation"
    credential_fingerprint = "sha256:" + "f" * 64
    request_hash = sha("champion-participation-host-call")
    call_identity = contracts.provider_call_identity(
        attempt=retry["attempt"],
        assignment_id=retry["assignment_id"],
        icp_position=retry["icp_position"],
        action_sequence=4,
        operation_id="openrouter.chat",
        request_hash=request_hash,
    )
    retry_token_hash = hash_lease_token(retry_token)
    assert store.reserve_call(
        run_id=retry["run_id"],
        lease_token_hash=retry_token_hash,
        call_identity=call_identity,
        operation_id="openrouter.chat",
        provider="openrouter",
        funding_source="host",
        amount_microusd=500_000,
        call_doc={"model": "fixture", "request_hash": request_hash},
    )["status"] == "reserved"
    assert store.mark_dispatched(
        run_id=retry["run_id"],
        lease_token_hash=retry_token_hash,
        call_identity=call_identity,
    )["status"] == "dispatched"
    assert store.mark_uncertain(
        run_id=retry["run_id"],
        lease_token_hash=retry_token_hash,
        call_identity=call_identity,
        call_doc={
            "reason": "missing_provider_cost",
            "provider_status": 502,
            "openrouter_generation_id": generation_id,
            "credential_fingerprint": credential_fingerprint,
        },
    )["status"] == "uncertain"

    completion = {
        "run_id": retry["run_id"],
        "lease_token_hash": retry_token_hash,
        "result": {"terminal_status": "accepted"},
        "terminal_cause": "accepted",
        "output_ref": "arena/champion-participation-retry.json",
    }
    assert store.complete_attempt(**completion)["status"] == "accepted"
    accepted_row = store.get_run(retry["run_id"])
    accepted_at = accepted_row["participation_accepted_at"]
    assert accepted_at is not None
    assert accepted_row["runner_hotkey"] == accepted_runner
    assert _has_recent_participation(database, accepted_runner)
    assert not _has_recent_participation(database, failed_runner)

    replayed_completion = store.complete_attempt(**completion)
    assert replayed_completion["status"] == "accepted"
    assert replayed_completion["idempotent"] is True
    assert store.get_run(retry["run_id"]) == accepted_row

    candidates = store.list_openrouter_cost_reconciliations(
        round_id, run_id=retry["run_id"]
    )
    assert len(candidates) == 1
    reconciliation = {
        "round_id": round_id,
        "run_id": retry["run_id"],
        "call_identity": call_identity,
        "uncertain_entry_id": candidates[0]["uncertain_entry_id"],
        "generation_id": generation_id,
        "credential_fingerprint": credential_fingerprint,
        "actual_microusd": 123,
        "cost_units": "0.000123",
    }
    settled = store.reconcile_openrouter_cost(**reconciliation)
    assert settled["status"] == "settled"
    assert settled["idempotent"] is False
    assert store.get_run(retry["run_id"]) == accepted_row

    replayed_settlement = store.reconcile_openrouter_cost(**reconciliation)
    assert replayed_settlement["status"] == "settled"
    assert replayed_settlement["idempotent"] is True
    final_row = store.get_run(retry["run_id"])
    assert final_row == accepted_row
    assert final_row["participation_accepted_at"] == accepted_at
    settlements = [
        entry
        for entry in store.list_ledger(call_identity=call_identity)
        if entry["entry_kind"] == "settlement"
    ]
    assert len(settlements) == 1


def test_host_network_uncertainty_is_not_excluded_after_fallback(database):
    store, _connect, round_id, _champion, runner, run, token = (
        _seed_champion_run(database, "network", execution_cap=5_000_000)
    )
    first = _account_call(
        store,
        run,
        token,
        provider="deepline",
        action_sequence=2,
        provider_attempt=1,
        provider_status=401,
    )
    for attempt in range(2, 5):
        assert _account_call(
            store,
            run,
            token,
            provider="deepline",
            action_sequence=2,
            provider_attempt=attempt,
            provider_status=401,
        )["prior_miner_credential_refusal"] is True
    assert store.mark_champion_provider_fallback(
        run["run_id"],
        hash_lease_token(token),
        "deepline",
        {
            "error_class": "account_credential_failure",
            "provider_status": 401,
            "base_call_identity": first["base_call_identity"],
            "provider_attempts": 4,
            "action_sequence": 2,
        },
    )["status"] == "marked"
    assert store.complete_attempt(
        run_id=run["run_id"],
        lease_token_hash=hash_lease_token(token),
        result={"terminal_status": "provider_error"},
        terminal_cause="provider_error",
        output_ref="",
    )["status"] == "failed"
    retry, retry_token, _, _ = claim(store, round_id, runner)
    network_identity = sha("host-network-uncertain")
    assert store.reserve_call(
        run_id=retry["run_id"],
        lease_token_hash=hash_lease_token(retry_token),
        call_identity=network_identity,
        operation_id="deepline.execute",
        provider="deepline",
        funding_source="host",
        amount_microusd=0,
        call_doc={"reserve_remaining_budget": True},
    )["status"] == "reserved"
    assert store.mark_dispatched(
        run_id=retry["run_id"],
        lease_token_hash=hash_lease_token(retry_token),
        call_identity=network_identity,
    )["status"] == "dispatched"
    assert store.mark_uncertain(
        run_id=retry["run_id"],
        lease_token_hash=hash_lease_token(retry_token),
        call_identity=network_identity,
        call_doc={"reason": "network_timeout"},
    )["status"] == "uncertain"
    blocked = store.reserve_call(
        run_id=retry["run_id"],
        lease_token_hash=hash_lease_token(retry_token),
        call_identity=sha("host-network-liability-gated"),
        operation_id="deepline.execute",
        provider="deepline",
        funding_source="host",
        amount_microusd=0,
        call_doc={"reserve_remaining_budget": True},
    )
    assert blocked["status"] == "refused"
    assert blocked["reason"] == "provider_cost_uncertain"


def test_old_schema_has_no_silent_champion_funding_fallback(old_database):
    store = _store(old_database)
    try:
        with pytest.raises(ArenaStoreError):
            store.champion_funding_schema()
    finally:
        store.close()
