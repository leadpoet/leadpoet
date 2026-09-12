"""End-to-end proof for durable daily champion funding and reward recovery.

The fake provider boundary is free and deterministic. Everything behind it is
the production round service, runner socket/shim, broker, credential selector,
ledger, PostgreSQL functions, promoter, signed reward, and canonical weights.
"""

from __future__ import annotations

import json
import subprocess
import threading
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from lab_arena import broker as br, contracts, scoring, service as svc
from lab_arena import source_bundle, submission_runtime, weight_state
from lab_arena.code_review_runtime import SubmissionCodeReviewer
from lab_arena.promotion import GitPromoter
from leadpoet_canonical import arena_weights
from leadpoet_canonical.lab_arena_rewards import champion_values
from tests.lab_arena.lab_arena_pg_harness import (
    POSTGREST_MIGRATIONS,
    database_with_lab_arena_migration,
)
from tests.lab_arena.test_lab_arena_service_round import (
    CANARY_OPENROUTER_KEY,
    FakeCredentialManager,
    FakeProviderTransport,
    Harness,
    SCORER_IMAGE_DIGEST,
    SCORER_IMAGE_REFERENCE,
    _run_stage_one_to_scoring,
    keypair,
    price_table,
    promotion_repository,
    wallet_verify,
)


HOST_KEYS = {
    "deepline": "dl_host_" + "h" * 32,
    "openrouter": "sk-or-v1-" + "h" * 40,
    "scrapingdog": "doghost" + "h" * 32,
}


@pytest.fixture
def champion_database():
    yield from database_with_lab_arena_migration(
        POSTGREST_MIGRATIONS + ("227-lab-arena-champion-funding.sql",)
    )


@pytest.fixture
def champion_connect(champion_database):
    psycopg2, dsn = champion_database
    return lambda: psycopg2.connect(**dsn)


class IdentityCredentialManager(FakeCredentialManager):
    """Return a stable fake secret for each submitted credential identity."""

    def runtime_key(self, row, provider):
        assert row["provider"] == provider
        if provider == "openrouter":
            # The code-review fake recognizes this value too. The identity
            # distinction needed by this proof is on model execution.
            return CANARY_OPENROUTER_KEY
        return "%s-miner-%s" % (provider, row["submission_id"])


class ChampionTransport(FakeProviderTransport):
    """A deterministic Deepline boundary with controllable account failures."""

    def __init__(self):
        super().__init__()
        self.champion_key = ""
        self.failure_status = 0
        self.failure_icp = "financial services companies"
        self.failure_count = 0
        self.sent = []
        self.infrastructure_failure = ""

    def arm_account_failure(self, status: int) -> None:
        self.failure_status = int(status)
        self.failure_count = 0

    def arm_infrastructure_failure(self, failure: str) -> None:
        assert failure in ("status_500", "network")
        self.infrastructure_failure = failure

    def send(self, *, method, url, headers, body, timeout_seconds, max_response_bytes=None):
        authorization = str(headers.get("authorization") or headers.get("Authorization") or "")
        self.sent.append((method, url, authorization))
        if method == "GET" and url.startswith(br.DEEPLINE_BILLING_HISTORY_URL):
            with self._deepline_lock:
                entries = list(reversed(self._deepline_jobs[-50:]))
            return br.ProviderResponse(
                200,
                {"content-type": "application/json"},
                json.dumps({"recent": {"entries": entries, "has_more": False, "next_cursor": None}}).encode(),
            )
        request = json.loads(body.decode("utf-8"))
        if request.get("model"):
            return super().send(
                method=method,
                url=url,
                headers=headers,
                body=body,
                timeout_seconds=timeout_seconds,
                max_response_bytes=max_response_bytes,
            )
        if self.infrastructure_failure:
            failure = self.infrastructure_failure
            self.infrastructure_failure = ""
            if failure == "network":
                raise br.ProviderTransportError("synthetic network failure")
            return br.ProviderResponse(
                500,
                {"content-type": "application/json"},
                b'{"error":"synthetic provider failure"}',
            )
        query = json.dumps(request.get("payload") or {}, sort_keys=True)
        if (
            self.failure_status
            and authorization == "Bearer " + self.champion_key
            and self.failure_icp in query
            and self.failure_count < br.CHAMPION_CREDENTIAL_PROVIDER_ATTEMPTS
        ):
            self.failure_count += 1
            return br.ProviderResponse(
                self.failure_status,
                {"content-type": "application/json"},
                json.dumps({"error": "champion account unavailable"}).encode(),
            )
        assert method == "POST"
        operation = request["operation"]
        with self._deepline_lock:
            job_id = "champion-deepline-job-%d" % (len(self._deepline_jobs) + 1)
            self._deepline_jobs.append(
                {
                    "request_id": job_id,
                    "operation": operation,
                    "provider": "fake",
                    "credits": 0,
                    "charge_state": "posted",
                }
            )
        return br.ProviderResponse(
            200,
            {"content-type": "application/json"},
            json.dumps(
                {
                    "job_id": job_id,
                    "results": [{"url": "https://co1.example.com", "title": "Co"}],
                    "status": "completed",
                }
            ).encode(),
        )


class ChampionHarness(Harness):
    def __init__(self, connect, tmp_path: Path, *, challengers, runners):
        self.provider_transport = ChampionTransport()
        self.credential_manager = IdentityCredentialManager()
        super().__init__(connect, tmp_path, challengers=challengers, runners=runners)

    def objects_key(self) -> str:
        return "champion-funding"

    def build_service(self) -> svc.ArenaService:
        store = self.make_store()
        payer = submission_runtime.SubmissionProviderKeys(
            store=store,
            credentials=self.credential_manager,
            organizer_keys=HOST_KEYS,
        )

        def broker_factory(service, round_row):
            return br.Broker(
                store=store,
                key_for=lambda provider: HOST_KEYS[provider],
                credential_for=payer.credential_for,
                funding_source_for=payer.funding_source_for,
                provider_funding_source_for=payer.provider_funding_source_for,
                retry_miner_credential_for=payer.retry_miner_credential_for,
                mark_provider_fallback=payer.mark_provider_fallback,
                provider_restart_required_for=payer.provider_restart_required_for,
                price_table=price_table(),
                judge_models=tuple(scoring.DEFAULT_JUDGE_MODELS.values()),
                transport=self.provider_transport,
                clock=self.clock,
            )

        return svc.ArenaService(
            svc.ServiceConfig(
                mode="live",
                store=store,
                object_store=self.objects,
                signer=self.signer,
                chain=self.chain,
                verify_signature=wallet_verify,
                daily_icp_source=lambda **kwargs: {
                    "status": "ready",
                    "set_id": int(kwargs["set_id"]),
                    "icps": __import__(
                        "tests.lab_arena.icp_fixtures", fromlist=["daily_icps"]
                    ).daily_icps(),
                },
                banned_hotkeys_source=lambda: list(self.banned),
                broker_factory=broker_factory,
                defaults=svc.RoundDefaults(
                    runner_hotkeys=tuple(self.runner_keys),
                    baseline_hotkey=self.baseline_hotkey,
                    baseline_source_url=svc.DEFAULT_BASELINE_SOURCE_URL,
                    max_challengers=self.max_challengers,
                    daily_cutoff_hour_utc=self.daily_cutoff_hour_utc,
                    rewards_enabled=True,
                    scorer_image_digest=SCORER_IMAGE_DIGEST,
                    scorer_image_reference=SCORER_IMAGE_REFERENCE,
                ),
                clock=self.clock,
                baseline_source_fetcher=lambda _url, _limit: self.baseline_source,
                credential_manager=self.credential_manager,
                code_reviewer=SubmissionCodeReviewer(
                    store=store,
                    objects=self.objects,
                    credential_for=payer.code_review_key,
                    price_table=price_table(),
                    transport=self.provider_transport,
                ),
            )
        )


def _start_round(harness: ChampionHarness, *, day: int, epoch: int, challengers=(), pool_percent=5):
    harness.challengers = list(challengers)
    harness.chain.epoch = epoch
    harness.service._config.defaults = replace(
        harness.service._config.defaults,
        max_challengers=max(1, len(challengers)),
        pool_percent=pool_percent,
    )
    configuration = harness.service.create_round(
        harness.clock.now + timedelta(hours=12),
        round_id="arena-2026-11-%02d" % day,
    )
    harness.round_id = configuration["round_id"]
    for flavor in challengers:
        harness.submit(flavor, harness.round_id)
    harness.clock.advance_to(harness.schedule()["submission_cutoff"])
    assert harness.service.advance_round(harness.round_id)["status"] == "ok"
    for participant in harness.service.store.get_round(harness.round_id)["participants"]:
        harness.flavors.setdefault(participant["submission_id"], "PublicBaseline")
    return harness.service.store.get_round(harness.round_id)


def _publish_round(harness: ChampionHarness, participants: int) -> dict:
    _run_stage_one_to_scoring(harness, participants, runners=3)
    harness.advance_until("published", runners=3)
    return harness.service.store.get_round(harness.round_id)


def _promote_first_winner(harness: ChampionHarness, tmp_path: Path, row: dict):
    repository_root = tmp_path / "champion-promotion-repository"
    repository_root.mkdir()
    remote = promotion_repository(repository_root)
    harness.service._config.baseline_promoter_factory = lambda: GitPromoter(
        str(remote), tmp_path / "champion-promotion-objects"
    )

    def promoted_source(_url, _limit):
        return subprocess.run(
            ("git", "--git-dir", str(remote), "archive", "--format=tar.gz", "lab"),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout

    harness.service._config.baseline_source_fetcher = promoted_source
    harness.clock.now = datetime.fromisoformat(
        str(row["published_at"]).replace("Z", "+00:00")
    )
    assert harness.service.promote_pending_baselines() == {"status": "ok", "promoted": 1}
    return repository_root, remote, promoted_source


def _activate(harness: ChampionHarness, expected_factor: int) -> dict:
    result = harness.service.activate_reward(harness.round_id)
    assert result["status"] == "activated", result
    row = harness.service.store.get_round(harness.round_id)
    basis = row["reward_basis_doc"]
    assert basis["champion_reward_factor_ppm"] == expected_factor
    return basis


def _baseline_id(row: dict) -> str:
    return next(item["submission_id"] for item in row["participants"] if item["is_king"])


def _assert_complete_miner_cycle(harness: ChampionHarness, row: dict) -> None:
    baseline_id = _baseline_id(row)
    executions = harness.service.store.list_runs(
        row["round_id"], submission_id=baseline_id, kind="execute"
    )
    scores = harness.service.store.list_runs(
        row["round_id"], submission_id=baseline_id, kind="score"
    )
    assert len(executions) == contracts.BENCHMARK_ICP_COUNT
    assert len(scores) == contracts.BENCHMARK_ICP_COUNT
    assert all(run["status"] == "accepted" and run["per_icp_score"] is not None for run in executions)
    assert all(run["status"] == "accepted" for run in scores)
    assert all(
        run["champion_funding_sources"]
        == {provider: "miner_key" for provider in contracts.PROVIDERS}
        for run in executions
    )
    assert all(run["champion_funding_sources"] is None for run in scores)


@pytest.mark.parametrize("account_status", (401, 402), ids=("unauthorized", "credit"))
def test_champion_funding_survives_rounds_restart_and_source_edits(
    champion_connect, tmp_path, account_status
):
    harness = ChampionHarness(
        champion_connect,
        tmp_path,
        challengers=["Alpha", "Bravo", "Charlie"],
        runners=["alpha", "beta", "gamma"],
    )
    # Submission admission is checked by PostgreSQL's real clock, while all
    # later round transitions use the injected clock.
    harness.clock.now = datetime.now(timezone.utc)

    first = _start_round(
        harness,
        day=1 if account_status == 401 else 11,
        epoch=41000 if account_status == 401 else 42000,
        challengers=("Alpha", "Bravo", "Charlie"),
    )
    first = _publish_round(harness, len(first["participants"]))
    assert first["king_outcome"] == "crowned"
    winner_id = first["publication_doc"]["king_decision"]["winner_submission_id"]
    winner_hotkey = first["publication_doc"]["king_decision"]["king_hotkey"]
    repository_root, _remote, promoted_source = _promote_first_winner(
        harness, tmp_path, first
    )
    full_first_basis = _activate(harness, 1_000_000)
    harness.provider_transport.champion_key = "deepline-miner-" + winner_id

    full = _start_round(
        harness,
        day=2 if account_status == 401 else 12,
        epoch=harness.chain.epoch + 10,
    )
    assert full["champion_funding_frozen"] is True
    assert full["champion_submission_id"] == winner_id
    assert full["champion_hotkey"] == winner_hotkey
    full = _publish_round(harness, 1)
    full_basis = _activate(harness, 1_000_000)
    _assert_complete_miner_cycle(harness, full)
    assert full_basis["king_hotkey"] == full_first_basis["king_hotkey"]

    # The source repository can change after promotion. It does not alter the
    # immutable original submission and hotkey that fund later baselines.
    edited = repository_root / "owner-edit"
    subprocess.run(("git", "clone", "--branch", "lab", str(_remote), str(edited)), check=True, capture_output=True)
    subprocess.run(("git", "config", "user.name", "Owner"), cwd=edited, check=True)
    subprocess.run(("git", "config", "user.email", "owner@example.test"), cwd=edited, check=True)
    (edited / "old.txt").write_text("owner edited source", encoding="utf-8")
    subprocess.run(("git", "add", "old.txt"), cwd=edited, check=True, capture_output=True)
    subprocess.run(("git", "commit", "-m", "owner edit"), cwd=edited, check=True, capture_output=True)
    subprocess.run(("git", "push", "origin", "lab"), cwd=edited, check=True, capture_output=True)
    harness.baseline_source = promoted_source("", source_bundle.MAX_SOURCE_ARCHIVE_BYTES)

    penalized = _start_round(
        harness,
        day=3 if account_status == 401 else 13,
        epoch=harness.chain.epoch + 10,
        pool_percent=25,
    )
    assert penalized["champion_submission_id"] == winner_id
    assert penalized["champion_hotkey"] == winner_hotkey
    stored_benchmark_input = harness.objects.get(penalized["benchmark_ref"])
    harness.provider_transport.arm_account_failure(account_status)
    harness.clock.advance_to(harness.schedule()["stage_1_start"])
    assert harness.service.advance_round(harness.round_id)["assignments"] == contracts.STAGE_1_ICP_COUNT
    first_runner = harness.runner(0, parallel=1)
    scheduled_now = harness.clock.now
    harness.clock.now = datetime.now(timezone.utc)
    try:
        for _ in range(9):
            assert first_runner.run_once(max_claims=1) == 1
    finally:
        first_runner.close()
        harness.clock.now = scheduled_now
    before_restart = harness.service.store.list_runs(
        harness.round_id, submission_id=_baseline_id(penalized), kind="execute"
    )
    stable = {
        run["run_id"]: harness.objects.get(run["output_ref"])
        for run in before_restart
        if run["icp_position"] < 8 and run["status"] == "accepted"
    }
    assert len(stable) == 8
    failed = [run for run in before_restart if run["icp_position"] == 8]
    assert len(failed) == 2
    failed_attempt = min(failed, key=lambda run: run["attempt"])
    assert failed_attempt["status"] == "failed"
    assert failed_attempt["champion_restart_required"] is True
    # Deepline cannot report the exact cost of a refused request. The initial
    # refusal keeps the full reservation uncertain. The durable latch below
    # can only follow the bounded initial broker attempt plus three retries.
    assert harness.provider_transport.failure_count == 1
    connection = champion_connect()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """SELECT DISTINCT entry_doc ->> 'provider_attempt'
                   FROM public.lab_arena_ledger
                   WHERE run_id = %s AND entry_kind = 'reservation'
                   ORDER BY 1""",
                (failed_attempt["run_id"],),
            )
            provider_attempts = [row[0] for row in cursor.fetchall()]
    finally:
        connection.close()
    assert provider_attempts == ["1"]
    assert failed_attempt["result_doc"]["resource_summary"]["provider_call_count"] == 1
    assert harness.service.store.get_round(harness.round_id)["champion_fallback_providers"] == ["deepline"]

    # Recreate both the gateway service and runner before the fresh attempt.
    harness.service = harness.build_service()
    assert harness.objects.get(penalized["benchmark_ref"]) == stored_benchmark_input
    harness.run_stage_with_runners(1)
    after_restart = harness.service.store.list_runs(
        harness.round_id, submission_id=_baseline_id(penalized), kind="execute"
    )
    for run_id, output in stable.items():
        row = harness.service.store.get_run(run_id)
        assert row["status"] == "accepted"
        assert harness.objects.get(row["output_ref"]) == output
    for position in range(8):
        assert len([run for run in after_restart if run["icp_position"] == position]) == 1
    recovered_nine = [run for run in after_restart if run["icp_position"] == 8 and run["status"] == "accepted"]
    assert len(recovered_nine) == 1
    assert recovered_nine[0]["attempt"] == 2
    assert recovered_nine[0]["champion_funding_sources"] == {
        "deepline": "host",
        "openrouter": "miner_key",
        "scrapingdog": "miner_key",
    }
    assert harness.service.advance_round(harness.round_id)["status"] == "ok"
    assert harness.service.advance_round(harness.round_id)["assignments"] == contracts.STAGE_1_ICP_COUNT
    harness.run_stage_with_runners(1)
    harness.advance_until("published", runners=1)
    half_basis = _activate(harness, 500_000)
    assert champion_values(
        half_basis, half_basis["effective_reward_epoch"], [winner_hotkey]
    )["champion_share"] == 0.125

    repeated = _start_round(
        harness,
        day=4 if account_status == 401 else 14,
        epoch=harness.chain.epoch + 10,
        pool_percent=20,
    )
    harness.provider_transport.arm_account_failure(account_status)
    repeated = _publish_round(harness, len(repeated["participants"]))
    repeated_basis = _activate(harness, 500_000)
    assert champion_values(
        repeated_basis, repeated_basis["effective_reward_epoch"], [winner_hotkey]
    )["champion_share"] == 0.10

    restored = _start_round(
        harness,
        day=5 if account_status == 401 else 15,
        epoch=harness.chain.epoch + 10,
        pool_percent=20,
    )
    harness.provider_transport.failure_status = 0
    restored = _publish_round(harness, len(restored["participants"]))
    restored_basis = _activate(harness, 1_000_000)
    _assert_complete_miner_cycle(harness, restored)
    assert champion_values(
        restored_basis, restored_basis["effective_reward_epoch"], [winner_hotkey]
    )["champion_share"] == 0.20

    burn_hotkey = keypair("champion-funding-burn").ss58_address
    accepted = weight_state.build_accepted_weight_state(
        harness.signer,
        network="finney",
        genesis_hash="1" * 64,
        netuid=71,
        epoch=int(half_basis["effective_reward_epoch"]),
        valid_from_block=1,
        valid_until_block=360,
        reward_basis=half_basis,
        burn_hotkey=burn_hotkey,
        issued_at=half_basis["published_at"],
    )
    arena_weights.verify_accepted_weight_state_signature(
        accepted,
        public_key_der=harness.signer.public_key_der,
        expected_public_key_hash=harness.signer.public_key_hash,
    )
    vector = arena_weights.derive_arena_weights(
        accepted, [winner_hotkey, burn_hotkey]
    )
    assert vector["champion_share_ppb"] == 125_000_000
    assert vector["burned_residual_ppb"] == 875_000_000
    assert vector["sparse_uids"] == [0, 1]

    connection = champion_connect()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """SELECT funding_source,count(*),coalesce(sum(amount_microusd),0)
                   FROM public.lab_arena_ledger
                   WHERE round_id = %s AND provider = 'deepline'
                   GROUP BY funding_source ORDER BY funding_source""",
                (penalized["round_id"],),
            )
            ledger = cursor.fetchall()
    finally:
        connection.close()
    assert {row[0] for row in ledger} == {"host", "miner_key"}
    assert all(row[1] > 0 and row[2] >= 0 for row in ledger)

    # Infrastructure failures use the same real broker and ledger, but never
    # authorize an account fallback. Two validators hold separate real leases
    # so both cases can be proved without completing an extra daily cycle.
    infrastructure = _start_round(
        harness,
        day=6 if account_status == 401 else 16,
        epoch=harness.chain.epoch + 10,
    )
    assert infrastructure["champion_fallback_providers"] == []
    harness.provider_transport.failure_status = 0
    harness.clock.advance_to(harness.schedule()["stage_1_start"])
    assert harness.service.advance_round(harness.round_id)["assignments"] == contracts.STAGE_1_ICP_COUNT
    scheduled_now = harness.clock.now
    harness.clock.now = datetime.now(timezone.utc)
    try:
        for runner_index, failure in enumerate(("status_500", "network")):
            runner = harness.runner(runner_index, parallel=1)
            try:
                lease = runner.claim_one()
                assert lease["status"] == "leased"
                harness.provider_transport.arm_infrastructure_failure(failure)
                response = harness.service.handle_provider(
                    lease["run_id"],
                    lease["lease_token"],
                    {
                        "operation_id": "deepline.execute",
                        "parameters": {
                            "tool": "exa_search",
                            "payload": {"query": "synthetic infrastructure proof"},
                        },
                        "timeout_ms": 5000,
                        "action_sequence": 0,
                    },
                )
                assert response["call"]["funding_source"] == "miner_key"
                assert response["call"]["error_code"] == "provider_unavailable"
                assert response["call"]["provider_attempt"] == 1
                assert harness.service.store.get_round(harness.round_id)[
                    "champion_fallback_providers"
                ] == []
            finally:
                runner.close()
    finally:
        harness.clock.now = scheduled_now
