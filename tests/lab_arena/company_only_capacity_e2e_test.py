"""A busy company-only day through signed intake, HTTP runners, and publication."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
import subprocess
import threading
from urllib.parse import urlsplit

import pytest
from fastapi.testclient import TestClient

from lab_arena import contracts, intent_details_policy, runtime, scoring, source_bundle
from lab_arena.api import create_app
from lab_arena.promotion import GitPromoter
from qualification.scoring.competition import apply_company_judgment_context
from tests.lab_arena import test_lab_arena_service_round as fixtures
from tests.lab_arena.company_quality_round_test import QualityHarness
from tests.lab_arena.icp_fixtures import daily_icps
from tests.lab_arena.lab_arena_pg_harness import (
    CURRENT_SERVICE_MIGRATIONS,
    database_with_lab_arena_migration,
)
from tests.lab_arena.parallel_twenty_runtime_e2e_test import (
    _http_runner,
    _postgres_lease_clock,
)
from tests.test_arena_company_quality_policy import _positive_breakdown


@pytest.fixture()
def database():
    yield from database_with_lab_arena_migration(CURRENT_SERVICE_MIGRATIONS)


def _submit_without_review(harness, flavor):
    """Issue both signed API writes; review the burst once after all uploads."""
    round_id = harness.round_id
    miner = fixtures.keypair("svc-miner-" + flavor)
    payload = fixtures.flavor_source_archive(flavor)
    facts = source_bundle.validate_source_archive(payload)
    presign = contracts.build_signed_request(
        scope=contracts.SCOPE_SUBMISSION_PRESIGN,
        round_id=round_id,
        hotkey=miner.ss58_address,
        body={
            "source_size_bytes": facts["source_size_bytes"],
            "consent": {"public_rerun": True},
        },
        timestamp=int(harness.clock().timestamp()),
        sign_message=lambda message: miner.sign(message.encode()).hex(),
    )
    target = harness.service.handle_submission_presign(presign)
    harness.flavors[target["submission_id"]] = flavor
    harness.objects.put(target["source_ref"], payload)
    finalize = contracts.build_signed_request(
        scope=contracts.SCOPE_SUBMISSION_FINALIZE,
        round_id=round_id,
        hotkey=miner.ss58_address,
        body={
            "submission_id": target["submission_id"],
            "source_ref": target["source_ref"],
            "source_size_bytes": facts["source_size_bytes"],
            "credentials": {
                "openrouter_api_key": fixtures.CANARY_OPENROUTER_KEY,
                "openrouter_management_key": fixtures.CANARY_OPENROUTER_MANAGEMENT_KEY,
                "deepline_api_key": fixtures.CANARY_DEEPLINE_KEY,
            },
        },
        timestamp=int(harness.clock().timestamp()),
        sign_message=lambda message: miner.sign(message.encode()).hex(),
    )
    assert harness.service.handle_submission_finalize(
        target["submission_id"], finalize
    )["status"] == "accepted"
    return target["submission_id"]


def _install_company_only_sandbox(harness):
    original = harness.sandbox.run_icp
    lock = threading.Lock()
    counts = {"execute": 0, "score": 0}

    def run_icp(spec, **kwargs):
        document = json.loads(
            (spec.input_dir / runtime.INPUT_FILE_NAME).read_text(encoding="utf-8")
        )
        if document.get("schema_version") == scoring.SCORING_INPUT_SCHEMA_VERSION:
            assert "contact_source_evidence" not in document
            assert all("contact" not in row for row in document["companies"])

            def judge(companies, _buyer, _reference):
                judgments = []
                for company in companies:
                    name = str(company["company_name"])
                    slug = name.casefold().replace(" ", "-")
                    value = 30.0 if name.startswith("PublicBaseline") else 60.0
                    raw = _positive_breakdown(
                        name, str(urlsplit(company["company_website"]).hostname), slug
                    )
                    raw.update(
                        final_score=value,
                        intent_signal_raw=value,
                        intent_signal_final=value,
                    )
                    raw["intent_signals_detail"][0].update(
                        raw=value, after_decay=value
                    )
                    raw["verifier_gate_receipts"].append(
                        {"gate": "intent_details", "decision": "match"}
                    )
                    judgments.append(raw)
                return apply_company_judgment_context(
                    companies, judgments, contacts_required=False
                )

            judge.company_quality = False
            judge.integrity_policy = True
            judge.contacts_required = False
            item = {"scored_run_id": document["scored_run_id"]}
            scores = scoring.score_work_item(
                item, icp=document["icp"], companies=document["companies"],
                scorer=judge,
            )
            output = scoring.build_scoring_output(document["scored_run_id"], scores)
            with lock:
                counts["score"] += 1
            return runtime.fake_result(
                exit_code=0, output_bytes=json.dumps(output).encode()
            )

        assert document["output_schema_version"] == (
            intent_details_policy.COMPANY_ONLY_OUTPUT_SCHEMA
        )
        assert "contact_policy" not in document["icp"]
        base = original(spec, **kwargs)
        rows = json.loads(base.output_bytes)["companies"]
        for row in rows:
            slug = row["company_name"].casefold().replace(" ", "-")
            row.update(
                company_linkedin=f"https://linkedin.com/company/{slug}",
                state="CA",
                intent_details=(
                    f"{row['company_name']} announced a product launch that "
                    "matches the requested signal."
                ),
                company_stage_evidence=[],
            )
            row.pop("fit_summary", None)
            row.pop("fit_evidence_urls", None)
            for signal in row["intent_signals"]:
                signal.pop("why_now", None)
                signal.pop("snippet", None)
        with lock:
            counts["execute"] += 1
        return runtime.fake_result(
            exit_code=0,
            output_bytes=json.dumps({
                "schema_version": intent_details_policy.COMPANY_ONLY_OUTPUT_SCHEMA,
                "companies": rows,
            }).encode(),
        )

    harness.sandbox.run_icp = run_icp
    return counts


def _drain_both(harness, runners):
    with _postgres_lease_clock(harness):
        with ThreadPoolExecutor(max_workers=2) as pool:
            def drain(runner):
                taken = 0
                while True:
                    count = runner.run_once(max_claims=1000)
                    taken += count
                    if not count:
                        return taken

            counts = list(pool.map(drain, runners))
    return counts


def test_64_miner_company_only_round_preserves_all_work_and_publishes(
    database, tmp_path
):
    psycopg2, dsn = database
    flavors = [f"Miner{index:02d}" for index in range(64)]
    harness = QualityHarness(
        lambda: psycopg2.connect(**dsn), tmp_path,
        challengers=flavors, runners=["alpha", "beta"],
    )
    bank = deepcopy(daily_icps()[:10])
    harness.service.config.defaults = replace(
        harness.service.config.defaults,
        benchmark_icp_count=10,
        max_challengers=contracts.DEFAULT_MAX_CHALLENGERS,
        # The synthetic 40-slot validators and stage windows can carry all
        # 64 challengers through full execution and scoring retries.
        runner_slot_ceiling=40,
        runner_capacity_slots={hotkey: 40 for hotkey in harness.runner_keys},
        stage_minutes={
            "benchmark": 30, "stage_1": 61, "stage_1_scoring": 16,
            "stage_2": 976, "final_scoring": 260,
        },
        promotion_margin=0.5,
        rewards_enabled=True,
        per_icp_cost_policy=True,
        contacts_from=None,
        company_quality_from=None,
        intent_details_from="2026-01-01T00:00:00Z",
        execution_sequence_from="2026-01-01T00:00:00Z",
        benchmark_disclosure_from="2026-01-01T00:00:00Z",
    )
    harness.service.config.daily_icp_source = lambda **kwargs: {
        "status": "ready", "set_id": int(kwargs["set_id"]), "icps": deepcopy(bank),
    }
    observed = _install_company_only_sandbox(harness)
    harness.chain.epoch = 64_000
    harness.clock.now = datetime.now(timezone.utc) + timedelta(seconds=1)
    harness.round_id = "arena-2098-10-06-capacity"
    configuration = harness.service.create_round(
        harness.clock.now + timedelta(minutes=30), round_id=harness.round_id
    )
    assert configuration["max_challengers"] == 64
    assert (configuration["stage_1_icp_count"], configuration["stage_2_icp_count"]) == (5, 5)
    assert configuration["execution_sequence_policy"] == (
        contracts.BASELINE_SCORED_FIRST_POLICY
    )
    assert "contact_policy" not in configuration
    with ThreadPoolExecutor(max_workers=8) as pool:
        submissions = list(pool.map(
            lambda flavor: _submit_without_review(harness, flavor), flavors
        ))
    assert len(set(submissions)) == len(flavors)
    assert harness.service.review_pending_submissions()["reviewed"] == 64
    harness.clock.advance_to(harness.schedule()["submission_cutoff"])
    assert harness.service.advance_round(harness.round_id)["status"] == "ok"
    committed = harness.service.store.get_round(harness.round_id)
    assert len(committed["participants"]) == 65
    baseline_id = next(
        row["submission_id"] for row in committed["participants"] if row["is_king"]
    )
    harness.flavors[baseline_id] = "PublicBaseline"
    harness.clock.advance_to(harness.schedule()["stage_1_start"])
    assert harness.service.advance_round(harness.round_id)["assignments"] == 10
    assert {row["submission_id"] for row in harness.service.store.list_runs(
        harness.round_id, kind="execute"
    )} == {baseline_id}

    with TestClient(create_app(harness.service)) as http:
        # Leave one real claimed lease unstarted, then expire it through the
        # database path. The replacement attempt must be the only accepted one.
        abandoned = _http_runner(
            harness, http, tmp_path / "validator-abandoned",
            key_label="svc-runner-alpha", local_capacity=10,
        )
        try:
            with _postgres_lease_clock(harness):
                abandoned_lease = abandoned.claim_one()
            assert abandoned_lease["status"] == "leased"
        finally:
            abandoned.close()
        with harness.connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                "UPDATE public.lab_arena_runs "
                "SET lease_expires_at=clock_timestamp()-interval '1 second' "
                "WHERE run_id=%s",
                (abandoned_lease["run_id"],),
            )
        expired = harness.service.store.expire_leases(harness.round_id)
        assert expired["expired"] == expired["retried"] == 1
        small = _http_runner(
            harness, http, tmp_path / "validator-small",
            key_label="svc-runner-alpha", local_capacity=40,
        )
        large = _http_runner(
            harness, http, tmp_path / "validator-large",
            key_label="svc-runner-beta", local_capacity=40,
        )
        try:
            for _ in range(30):
                status = harness.status()
                if status == "published":
                    break
                if status in ("stage1", "stage1_scoring", "stage2", "stage2_scoring"):
                    _drain_both(harness, (small, large))
                if status == "stage1_scored":
                    assert harness.service._baseline_fully_succeeded(
                        harness.service.store.get_round(harness.round_id)
                    )
                    assert harness.service.store.list_runs(
                        harness.round_id, stage=2, kind="execute"
                    ) == []
                    harness.clock.advance_to(harness.schedule()["stage_2_start"])
                result = harness.service.advance_round(harness.round_id)
                assert result.get("status") not in (
                    "cancelled", "terminal", "retry", "stale"
                ), (status, result)
            else:
                raise AssertionError("round did not publish")
        finally:
            small.close()
            large.close()

    published = harness.service.store.get_round(harness.round_id)
    assert published["status"] == "published"
    assert observed == {"execute": 650, "score": 650}
    for kind in ("execute", "score"):
        runs = harness.service.store.list_runs(harness.round_id, kind=kind)
        assert len(runs) == 650 + (kind == "execute")
        assert len({row["assignment_id"] for row in runs}) == 650
        assert sum(row["status"] == "accepted" for row in runs) == 650
        assert {row["icp_position"] for row in runs} == set(range(10))
        assert {row["runner_hotkey"] for row in runs} == set(harness.runner_keys)
        assert all(row["lease_token_hash"] for row in runs)
    attempts = [
        row for row in harness.service.store.list_runs(harness.round_id, kind="execute")
        if row["assignment_id"] == abandoned_lease["assignment_id"]
    ]
    assert sorted((row["attempt"], row["status"]) for row in attempts) == [
        (1, "failed"), (2, "accepted"),
    ]
    assert len(harness.service.store.list_runs(harness.round_id)) == 1301
    ranking = published["publication_doc"]["final_ranking"]
    assert len(ranking) == 65
    assert {entry["final_score"] for entry in ranking if not entry["is_baseline"]} == {60.0}
    assert next(entry["final_score"] for entry in ranking if entry["is_baseline"]) == 30.0
    assert all(entry["eligible"] for entry in ranking)
    for entry in ranking:
        cost = entry["cost_summary"]
        assert len(cost["per_icp"]) == 10
        assert cost["returned_company_count"] == 10
        assert cost["qualified_company_count"] == 50
        assert cost["competition_sourcing_microusd"] == cost["execution"][
            "successful_microusd"
        ]
    assert published["king_outcome"] == "crowned"
    winner = published["publication_doc"]["king_decision"]["winner_submission_id"]
    assert winner in submissions
    remote_root = tmp_path / "promotion"
    remote_root.mkdir()
    remote = fixtures.promotion_repository(remote_root)
    harness.service.config.baseline_promoter_factory = lambda: GitPromoter(
        str(remote), tmp_path / "promotion-objects"
    )
    assert harness.service.promote_pending_baselines() == {
        "status": "ok", "promoted": 1,
    }
    assert subprocess.check_output([
        "git", "--git-dir", str(remote), "show", "lab:flavor.txt",
    ]).decode() == harness.flavors[winner]
