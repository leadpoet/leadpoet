"""Real-client proof that an unplanned validator can pick up Arena work.

The test keeps the chain, provider, and sandbox boundaries controlled.  It
does not replace round discovery, signed claim authorization, PostgreSQL lease
selection, source delivery, execution, or signed completion.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from lab_arena import contracts, runner as rn, runtime, scoring
from lab_arena.api import create_app
from tests.lab_arena.lab_arena_pg_harness import (
    CURRENT_SERVICE_MIGRATIONS,
    database_with_lab_arena_migration,
)
from tests.lab_arena.test_lab_arena_service_round import (
    Harness,
    _run_stage_one_to_scoring,
    _start_round,
    keypair,
)


@pytest.fixture()
def database():
    yield from database_with_lab_arena_migration(CURRENT_SERVICE_MIGRATIONS)


def _http_runner(
    harness: Harness,
    http: TestClient,
    tmp_path,
    *,
    key_label: str,
    round_id: str | None,
    parallelism: int,
) -> rn.Runner:
    key = keypair(key_label)
    api = rn.HttpArenaApiClient("http://localhost", client=http)
    cache_key = key.ss58_address[:12]
    image_cache = rn.ImageCache(
        tmp_path / ("pickup-images-" + cache_key),
        lambda _reference, _digest, target: (target / "rootfs").mkdir(),
    )
    source_cache = rn.SourceCache(
        tmp_path / ("pickup-sources-" + cache_key),
        api.source,
        dependency_installer=lambda _requirements, _target: None,
    )
    work_dir = tmp_path / ("pickup-work-" + cache_key)
    work_dir.mkdir()
    return rn.Runner(
        rn.RunnerConfig(
            round_id=round_id,
            identity=rn.RunnerIdentity(
                hotkey=key.ss58_address,
                sign=lambda message: key.sign(message.encode("utf-8")).hex(),
            ),
            api=api,
            sandbox_runtime=harness.sandbox,
            image_cache=image_cache,
            source_cache=source_cache,
            work_dir=work_dir,
            max_parallel_runs=parallelism,
            evaluation_date="2026-09-12",
            clock=harness.clock,
            completion_retry_seconds=(0.0, 0.0),
            claim_retry_seconds=(),
        )
    )


def test_unplanned_validator_discovers_and_completes_work_while_primary_is_full(
    database, tmp_path
):
    """Exercise the production runner/app/service/SQL pickup path end to end."""

    psycopg2, dsn = database
    connect = lambda: psycopg2.connect(**dsn)
    harness = Harness(
        connect,
        tmp_path,
        challengers=["PickupCandidate"],
        runners=["alpha"],
    )
    participant_count = _start_round(harness, day=13, epoch=50_000)
    working_round_id = harness.round_id
    harness.clock.advance_to(harness.schedule()["stage_1_start"])
    opened = harness.service.advance_round(working_round_id)
    assert opened["assignments"] == (
        participant_count * contracts.STAGE_1_ICP_COUNT
    )
    # PostgreSQL owns lease expiry.  Use wall-clock time while leases are live,
    # as the existing round harness does; the schedule has already been opened.
    harness.clock.now = datetime.now(timezone.utc)

    external_key = keypair("arena-unplanned-validator")
    harness.chain.runners.append(external_key.ss58_address)
    harness.chain.permits[external_key.ss58_address] = True
    harness.chain.stakes[external_key.ss58_address] = 100_000.0
    harness.chain.active[external_key.ss58_address] = False

    frozen_configuration = harness.service.store.get_round(working_round_id)[
        "configuration_doc"
    ]
    assert frozen_configuration["runner_hotkeys"] == [harness.runner_keys[0]]
    assert external_key.ss58_address not in frozen_configuration["runner_hotkeys"]
    snapshot = harness.chain.metagraph(finalized=True)
    external_uid = snapshot.hotkeys.index(external_key.ss58_address)
    assert snapshot.validator_permit[external_uid] is True
    assert snapshot.stake[external_uid] >= 75_000.0

    observed_sources: list[dict[str, str]] = []
    original_run_icp = harness.sandbox.run_icp

    def run_icp(spec, **kwargs):
        if spec.source_dir is not None:
            observed_sources.append(
                {
                    "submission_id": spec.source_dir.parent.name.removeprefix(
                        "submission-"
                    ),
                    "flavor": (spec.source_dir / "flavor.txt").read_text(
                        encoding="utf-8"
                    ),
                }
            )
        return original_run_icp(spec, **kwargs)

    harness.sandbox.run_icp = run_icp

    with TestClient(create_app(harness.service)) as http:
        primary = _http_runner(
            harness,
            http,
            tmp_path,
            key_label="svc-runner-alpha",
            round_id=working_round_id,
            parallelism=frozen_configuration["runner_slot_ceiling"],
        )
        try:
            primary_leases = [
                primary.claim_one()
                for _ in range(frozen_configuration["runner_slot_ceiling"])
            ]
            assert {lease["status"] for lease in primary_leases} == {"leased"}

            before = harness.service.store.list_runs(
                working_round_id, stage=1, kind="execute"
            )
            assert sum(
                run["status"] == "leased"
                and run["runner_hotkey"] == harness.runner_keys[0]
                for run in before
            ) == frozen_configuration["runner_slot_ceiling"]
            assert sum(run["status"] == "pending" for run in before) > 0
            assert not any(
                run["runner_hotkey"] == external_key.ss58_address
                for run in before
            )

            open_round_id = "arena-2026-10-14"
            open_configuration = harness.service.create_round(
                harness.clock.now + timedelta(hours=12),
                round_id=open_round_id,
            )
            assert open_configuration["round_id"] == open_round_id
            discovery = rn.HttpArenaApiClient(
                "http://localhost", client=http
            ).current()
            assert discovery["open_round"]["round_id"] == open_round_id
            assert discovery["round"]["round_id"] == open_round_id
            assert [
                row["round_id"] for row in discovery["running_rounds"]
            ] == [working_round_id]

            external = _http_runner(
                harness,
                http,
                tmp_path,
                key_label="arena-unplanned-validator",
                round_id=None,
                parallelism=1,
            )
            try:
                assert external.run_once(max_claims=1) == 1
                assert external.round_ids == [working_round_id]
                assert len(external.completed) == 1
                assert "error" not in external.completed[0], external.completed[0]
                assert external.completed[0]["result"]["status"] == "accepted"
            finally:
                external.close()
        finally:
            primary.close()

    completed_run = harness.service.store.get_run(external.completed[0]["run_id"])
    assert completed_run["status"] == "accepted"
    assert completed_run["kind"] == "execute"
    assert completed_run["runner_hotkey"] == external_key.ss58_address
    submission = harness.service.store.get_submission(
        completed_run["submission_id"]
    )
    assert completed_run["claim_response"]["source_ref"] == submission[
        "source_ref"
    ]
    assert observed_sources == [
        {
            "submission_id": completed_run["submission_id"],
            "flavor": harness.flavors[completed_run["submission_id"]],
        }
    ]
    output = json.loads(
        harness.objects.get(completed_run["output_ref"]).decode("utf-8")
    )
    assert output["companies"]
    assert output["companies"][0]["company_name"].startswith(
        observed_sources[0]["flavor"] + " Company "
    )

    after = harness.service.store.list_runs(
        working_round_id, stage=1, kind="execute"
    )
    assert sum(
        run["status"] == "leased"
        and run["runner_hotkey"] == harness.runner_keys[0]
        for run in after
    ) == frozen_configuration["runner_slot_ceiling"]
    assert sum(run["status"] == "pending" for run in after) > 0
    assert sum(
        run["status"] == "accepted"
        and run["runner_hotkey"] == external_key.ss58_address
        for run in after
    ) == 1


def test_unplanned_validator_discovers_and_completes_scoring_while_primary_is_full(
    database, tmp_path
):
    """Bind a real HTTP scoring lease to its accepted execution output."""

    psycopg2, dsn = database
    connect = lambda: psycopg2.connect(**dsn)
    harness = Harness(
        connect,
        tmp_path,
        challengers=["PickupScoreCandidate"],
        runners=["alpha"],
    )
    participant_count = _start_round(harness, day=15, epoch=50_001)
    working_round_id = harness.round_id
    _run_stage_one_to_scoring(harness, participant_count, runners=1)
    assert harness.status() == "stage1_scoring"
    assert all(
        run["status"] == "accepted" and run["output_ref"]
        for run in harness.service.store.list_runs(
            working_round_id, stage=1, kind="execute"
        )
    )
    harness.clock.now = datetime.now(timezone.utc)

    external_key = keypair("arena-unplanned-score-validator")
    harness.chain.runners.append(external_key.ss58_address)
    harness.chain.permits[external_key.ss58_address] = True
    harness.chain.stakes[external_key.ss58_address] = 100_000.0
    harness.chain.active[external_key.ss58_address] = False
    configuration = harness.service.store.get_round(working_round_id)[
        "configuration_doc"
    ]
    assert configuration["runner_hotkeys"] == [harness.runner_keys[0]]
    assert external_key.ss58_address not in configuration["runner_hotkeys"]
    snapshot = harness.chain.metagraph(finalized=True)
    external_uid = snapshot.hotkeys.index(external_key.ss58_address)
    assert snapshot.validator_permit[external_uid] is True
    assert snapshot.stake[external_uid] >= 75_000.0

    observed_scoring_inputs: list[dict] = []
    original_run_icp = harness.sandbox.run_icp

    def run_icp(spec, **kwargs):
        input_document = json.loads(
            (spec.input_dir / runtime.INPUT_FILE_NAME).read_text(
                encoding="utf-8"
            )
        )
        if input_document.get("schema_version") == scoring.SCORING_INPUT_SCHEMA_VERSION:
            observed_scoring_inputs.append(input_document)
        return original_run_icp(spec, **kwargs)

    harness.sandbox.run_icp = run_icp

    primary_capacity = 1
    with TestClient(create_app(harness.service)) as http:
        primary = _http_runner(
            harness,
            http,
            tmp_path,
            key_label="svc-runner-alpha",
            round_id=working_round_id,
            parallelism=primary_capacity,
        )
        try:
            primary_leases = [
                primary.claim_one() for _ in range(primary_capacity)
            ]
            assert {lease["status"] for lease in primary_leases} == {"leased"}
            assert {lease["kind"] for lease in primary_leases} == {"score"}

            before = harness.service.store.list_runs(
                working_round_id, stage=1, kind="score"
            )
            assert sum(
                run["status"] == "leased"
                and run["runner_hotkey"] == harness.runner_keys[0]
                for run in before
            ) == primary_capacity
            assert sum(run["status"] == "pending" for run in before) > 0
            assert not any(
                run["runner_hotkey"] == external_key.ss58_address
                for run in before
            )

            external = _http_runner(
                harness,
                http,
                tmp_path,
                key_label="arena-unplanned-score-validator",
                round_id=None,
                parallelism=1,
            )
            try:
                assert external.run_once(max_claims=1) == 1
                assert external.round_ids == [working_round_id]
                assert len(external.completed) == 1
                assert "error" not in external.completed[0], external.completed[0]
                assert external.completed[0]["result"]["status"] == "accepted"
            finally:
                external.close()
        finally:
            primary.close()

    completed_score = harness.service.store.get_run(
        external.completed[0]["run_id"]
    )
    assert completed_score["status"] == "accepted"
    assert completed_score["kind"] == "score"
    assert completed_score["runner_hotkey"] == external_key.ss58_address
    scored_execution = harness.service.store.get_run(
        completed_score["scored_run_id"]
    )
    assert scored_execution["status"] == "accepted"
    assert scored_execution["kind"] == "execute"
    source_output = json.loads(
        harness.objects.get(scored_execution["output_ref"]).decode("utf-8")
    )
    assert len(observed_scoring_inputs) == 1
    assert observed_scoring_inputs[0]["scored_run_id"] == scored_execution[
        "run_id"
    ]
    assert observed_scoring_inputs[0]["companies"] == source_output["companies"]
    assert completed_score["output_ref"]
    score_output = json.loads(
        harness.objects.get(completed_score["output_ref"]).decode("utf-8")
    )
    assert score_output["scored_run_id"] == scored_execution["run_id"]

    after = harness.service.store.list_runs(
        working_round_id, stage=1, kind="score"
    )
    assert sum(
        run["status"] == "leased"
        and run["runner_hotkey"] == harness.runner_keys[0]
        for run in after
    ) == primary_capacity
    assert sum(run["status"] == "pending" for run in after) > 0
    assert sum(
        run["status"] == "accepted"
        and run["runner_hotkey"] == external_key.ss58_address
        for run in after
    ) == 1
