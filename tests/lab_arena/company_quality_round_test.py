"""Frozen activation and complete saved/public company-quality rounds."""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from copy import deepcopy
import io
import json
import subprocess
import tarfile
from urllib.parse import urlsplit

import pytest

from lab_arena import (
    company_judgments,
    contracts,
    contact_policy,
    public_dashboard,
    quality_policy,
    runtime,
    scoring,
    verify,
)
from lab_arena.promotion import GitPromoter
from lab_arena.service import ArenaService, RoundDefaults, ServiceConfig, ServiceError
from qualification.scoring.competition import (
    COMPANY_VERIFICATION_EXHAUSTED_FAILURE_CLASS,
    apply_company_judgment_context,
)
from tests.lab_arena import test_lab_arena_service_round as fixtures
from tests.lab_arena.lab_arena_pg_harness import (
    CURRENT_SERVICE_MIGRATIONS,
    database_with_lab_arena_migration,
)
from tests.lab_arena.test_integrity_round import IntegrityHarness
from tests.test_arena_company_quality_policy import _positive_breakdown


def config(defaults, store=None):
    return ServiceConfig(mode="shadow", store=store or object(), object_store=object(), signer=None, chain=object(), verify_signature=lambda *_: True, daily_icp_source=lambda **_: {}, banned_hotkeys_source=lambda: (), broker_factory=lambda *_: None, defaults=defaults)


@pytest.mark.parametrize("integrity_from,quality_from", [(None, "2026-12-01T00:00:00Z"), ("2026-12-03T00:00:00Z", "2026-12-01T00:00:00Z"), ("2026-01-01T00:00:00Z", "2026-12-01T00:00:00")])
def test_quality_activation_requires_announced_integrity_and_timezone(integrity_from, quality_from):
    with pytest.raises(ServiceError, match="company_quality_activation_invalid"):
        config(RoundDefaults(integrity_from=integrity_from, company_quality_from=quality_from))


def test_quality_activation_is_frozen_before_intake():
    class Store:
        def create_round(self, round_id, configuration):
            return {"status": "created"}
    runner = fixtures.keypair("quality-runner").ss58_address
    defaults = RoundDefaults(runner_hotkeys=(runner,), baseline_hotkey=fixtures.keypair("quality-baseline").ss58_address, scorer_image_digest=fixtures.SCORER_IMAGE_DIGEST, scorer_image_reference=fixtures.SCORER_IMAGE_REFERENCE, integrity_from="2026-01-01T00:00:00Z", company_quality_from="2026-12-01T00:00:00Z")
    service = ArenaService(config(defaults, Store()))
    service.runner_settings = lambda: ([runner], [])
    service._require_integrity_schema = lambda: None
    service._require_company_quality_schema = lambda: None
    activation = datetime(2026, 12, 1, tzinfo=timezone.utc)
    old = service.create_round(activation + timedelta(days=1, seconds=-1), round_id="arena-2026-12-01-qualitybefore")
    new = service.create_round(activation + timedelta(days=1), round_id="arena-2026-12-02-qualityafter")
    assert "company_quality_policy" not in old
    assert new["company_quality_policy"] == quality_policy.POLICY
    assert new["scorer_policy"]["company_quality_policy"] == quality_policy.POLICY
    assert "company_quality_policy" not in old["scorer_policy"]


class QualityHarness(IntegrityHarness):
    def objects_key(self):
        return "company-quality-round-" + self.tmp.name

    def build_service(self):
        service = super().build_service()
        service.config.defaults = replace(service.config.defaults, company_quality_from="2026-01-01T00:00:00Z")
        return service

    def run_stage_with_runners(self, count=2):
        """Give each configured validator a claim opportunity per pass."""
        scheduled_now = self.clock.now
        self.clock.now = datetime.now(timezone.utc)
        runners = [self.runner(index) for index in range(count)]
        try:
            while any([runner.run_once() for runner in runners]):
                pass
        finally:
            for runner in runners:
                runner.close()
            self.clock.now = scheduled_now
        abandoned = [
            completed
            for runner in runners
            for completed in runner.completed
            if completed.get("error")
        ]
        assert not abandoned, (
            "runners abandoned work: %s; api errors: %s"
            % (abandoned[:3], fixtures.InProcessApi.errors[:3])
        )


@pytest.fixture()
def database():
    yield from database_with_lab_arena_migration(CURRENT_SERVICE_MIGRATIONS)


@pytest.mark.parametrize(
    "with_contacts", [False, True, "exhausted"]
)
def test_quality_round_publishes_coverage_winner_after_restart(
    database, tmp_path, with_contacts
):
    exhausted_company = "Padded Company 3" if with_contacts == "exhausted" else ""
    with_contacts = bool(with_contacts)
    psycopg2, dsn = database
    challengers = ["Broad", "Sparse", "Padded"]
    harness = QualityHarness(lambda: psycopg2.connect(**dsn), tmp_path, challengers=challengers, runners=["alpha", "beta"])
    if with_contacts:
        from tests.lab_arena.contact_round_test import _contact_icps
        source = harness.service.config.daily_icp_source
        harness.service.config.defaults = replace(harness.service.config.defaults, contacts_from="2026-01-01T00:00:00Z")
        harness.service.config.daily_icp_source = lambda **kwargs: {**source(**kwargs), "icps": _contact_icps(source(**kwargs)["icps"])}
    original = harness.sandbox.run_icp
    scoring_leases = []
    judged_companies = []
    exhausted_attempts = []

    def run_icp(spec, **kwargs):
        document = json.loads((spec.input_dir / runtime.INPUT_FILE_NAME).read_text())
        if document.get("schema_version") != scoring.SCORING_INPUT_SCHEMA_VERSION:
            assert document["icp"]["company_quality_policy"] == quality_policy.POLICY
            assert "company_linkedin" in document["icp"]["company_requirements"]
            base = original(spec, **kwargs)
            if not base.output_bytes:
                return base
            rows = json.loads(base.output_bytes)["companies"]
            flavor = rows[0]["company_name"].split(" Company ", 1)[0]
            if flavor == "Sparse":
                rows = rows[:2]
            elif flavor == "Padded":
                # Padded repeats the two exact Sparse winners, then repeats the
                # first winner and adds two rows with invalid intent evidence.
                # Reusable raw judgments therefore cross validator assignments.
                for index in (0, 1):
                    rows[index].update(
                        company_name=f"Sparse Company {index}",
                        company_website=f"https://sparse-{index}.example.com",
                        fit_evidence_urls=[f"https://sparse-{index}.example.com/about"],
                    )
                    rows[index]["intent_signals"][0].update(
                        url=f"https://news.example.com/Sparse/{index}",
                    )
                rows[2] = deepcopy(rows[0])
            for row in rows:
                slug = row["company_name"].lower().replace(" ", "-")
                row.update(company_linkedin=f"https://linkedin.com/company/{slug}", state="CA")
                if with_contacts:
                    from tests.lab_arena.contact_round_test import _claim
                    row["contact"] = _claim(row, valid_role=True)
            return runtime.fake_result(exit_code=0, output_bytes=json.dumps({"companies": rows}).encode())
        icp = document["icp"]
        lease = document["company_judgment_cache"]
        scoring_leases.append({
            "hits": [item["company_index"] for item in lease["hits"]],
            "misses": [item["company_index"] for item in lease["misses"]],
            "hit_keys": [item["cache_key"] for item in lease["hits"]],
            "miss_keys": [
                item["cache_key"]
                for item in lease["misses"]
                if document["companies"][item["company_index"]]["company_name"]
                == exhausted_company
            ],
        })
        def judge(companies, buyer, _reference):
            c = companies[0]
            name = c["company_name"]
            flavor = name.split(" Company ", 1)[0]
            score = 20 if flavor == "PublicBaseline" else 35
            if name in ("Sparse Company 0", "Sparse Company 1"):
                score = 100
            slug = name.lower().replace(" ", "-")
            raw = _positive_breakdown(name, urlsplit(c["company_website"]).hostname, slug)
            raw.update(final_score=score, intent_signal_raw=score, intent_signal_final=score)
            raw["intent_signals_detail"][0].update(raw=score, after_decay=score)
            if with_contacts:
                import asyncio
                from lab_arena.contact_evidence import source_key
                from tests.lab_arena.contact_round_test import _contact_result
                from qualification.scoring.competition import _merge_contact_breakdown
                evidence = document["contact_source_evidence"][source_key(c)]
                _merge_contact_breakdown(raw, asyncio.run(_contact_result(c, buyer, evidence)))
            if name in ("Padded Company 3", "Padded Company 4"):
                # These are valid output rows but have no verified primary
                # intent. High intrinsic score text cannot turn them into
                # qualified padding.
                raw["intent_signals_detail"] = []
                raw["failure_reason"] = "primary intent was not verified"
            if exhausted_company and name == exhausted_company:
                # This is the typed provider/source outcome produced by the
                # real company verifier. The scoring seam should retry this
                # one company three times, then cache a terminal zero.
                raw["verifier_gate_receipts"][0].update(
                    decision="unavailable",
                    reason="source_blocked",
                    failure_reason_code="source_blocked",
                )
                raw.update(
                    final_score=0.0,
                    intent_signal_raw=0.0,
                    intent_signal_final=0.0,
                    failure_reason="source_blocked",
                    intent_signals_detail=[],
                )
                exhausted_attempts.append((document["scored_run_id"], name))
            judged_companies.append(name)
            return apply_company_judgment_context(companies, [raw], contacts_required=with_contacts)
        judge.company_quality = judge.integrity_policy = True
        judge.contacts_required = with_contacts
        full, new = scoring.score_quality_work_item({"scored_run_id": document["scored_run_id"]}, icp=icp, companies=document["companies"], scorer=judge, cache_context=document["company_judgment_cache"])
        return runtime.fake_result(exit_code=0, output_bytes=json.dumps(scoring.build_scoring_output(document["scored_run_id"], full, company_judgments=new)).encode())

    harness.sandbox.run_icp = run_icp
    harness.clock.now = datetime.now(timezone.utc)
    round_id = "arena-2026-12-02-qualitycomplete"
    harness.service.create_round(harness.clock.now + timedelta(hours=12), round_id=round_id)
    harness.round_id = round_id
    broad = harness.submit("Broad", round_id)
    sparse = harness.submit("Sparse", round_id)
    padded = harness.submit("Padded", round_id)
    harness.clock.advance_to(harness.schedule()["submission_cutoff"])
    harness.advance_until("scored", runners=2, max_steps=120)

    persisted = harness.service.store.list_runs(round_id, kind="execute")
    main_positions = set(range(contracts.BENCHMARK_ICP_COUNT))
    broad_main = [run for run in persisted if run["submission_id"] == broad and run["icp_position"] in main_positions]
    sparse_main = [run for run in persisted if run["submission_id"] == sparse and run["icp_position"] in main_positions]
    padded_main = [run for run in persisted if run["submission_id"] == padded and run["icp_position"] in main_positions]
    assert len(broad_main) == len(sparse_main) == len(padded_main) == contracts.BENCHMARK_ICP_COUNT
    assert all(run["terminal_cause"] == "accepted" and float(run["per_icp_score"]) == 35 for run in broad_main)
    assert all(run["terminal_cause"] == "accepted" and float(run["per_icp_score"]) == 16 for run in sparse_main)
    assert all(run["terminal_cause"] == "accepted" and float(run["per_icp_score"]) == 16 for run in padded_main)
    assert all(
        [row["company_qualified"] for row in run["qualification_doc"]["companies"]]
        == [True, True]
        for run in sparse_main
    )
    if with_contacts:
        assert all(
            [row["contact_qualified"] for row in run["qualification_doc"]["companies"]]
            == [True, True]
            for run in sparse_main
        )
        assert all(
            [row["contact_qualified"] for row in run["qualification_doc"]["companies"]]
            == [True, True, False, False, False]
            for run in padded_main
        )
    else:
        assert all(
            all("contact_qualified" not in row for row in run["qualification_doc"]["companies"])
            for run in sparse_main + padded_main
        )
    assert all(
        [row["company_qualified"] for row in run["qualification_doc"]["companies"]]
        == [True, True, False, False, False]
        for run in padded_main
    )
    assert all(
        [row["duplicate_company"] for row in run["qualification_doc"]["companies"]]
        == [False, False, True, False, False]
        for run in padded_main
    )

    harness.service = harness.build_service()
    if exhausted_company:
        # Rebuild one accepted scorer lease after the service restart as an
        # all-hit lease. A scorer that is called here is a regression: the
        # exhausted company judgment must be reusable from PostgreSQL.
        replay_run = next(
            run
            for run in harness.service.store.list_runs(round_id, kind="score")
            if run["submission_id"] == padded
            and any(
                ref["company_index"] == 3
                for ref in (run.get("claim_response") or {})
                .get("company_judgment_cache", {})
                .get("misses", [])
            )
        )
        replay_lease = (replay_run["claim_response"] or {})["company_judgment_cache"]
        replay_refs = sorted(
            replay_lease["hits"] + replay_lease["misses"],
            key=lambda ref: ref["company_index"],
        )
        replay_hits = []
        for ref in replay_refs:
            evidence = harness.service.store.get_company_judgment(
                ref["cache_key"], ref["authority_slot"]
            )
            assert evidence is not None
            replay_hits.append({
                **ref,
                "evidence_hash": evidence["evidence_hash"],
                "evidence_doc": evidence["evidence_doc"],
            })
        replay_context = {
            "schema_version": company_judgments.LEASE_SCHEMA_VERSION,
            "hits": replay_hits,
            "misses": [],
        }
        companies = json.loads(
            harness.objects.get_bounded(
                next(run for run in padded_main if run["run_id"] == replay_run["scored_run_id"])["output_ref"],
                scoring.MAX_SCORING_OUTPUT_BYTES,
            ).decode("utf-8")
        )["companies"]

        def forbidden_scorer(*_args, **_kwargs):
            raise AssertionError("accepted company judgment cache miss after restart")

        forbidden_scorer.company_quality = True
        forbidden_scorer.integrity_policy = True
        forbidden_scorer.contacts_required = with_contacts
        full, new = scoring.score_quality_work_item(
            {"scored_run_id": replay_run["scored_run_id"]},
            icp=harness.service.evaluation_icps(round_id)[replay_run["icp_position"]],
            companies=companies,
            scorer=forbidden_scorer,
            cache_context=replay_context,
        )
        assert new == []
        assert full[3]["final_score"] == 0
        assert full[3]["company_qualified"] is False
        assert any(
            receipt.get("failure_class") == COMPANY_VERIFICATION_EXHAUSTED_FAILURE_CLASS
            for receipt in full[3]["verifier_gate_receipts"]
        )
    published = harness.advance_until("published", runners=2, max_steps=120)
    assert published["king_outcome"] == "crowned"
    ranking = {row["submission_id"]: row for row in published["publication_doc"]["final_ranking"]}
    assert ranking[broad]["main_score"] == ranking[broad]["final_score"] == 35
    assert ranking[sparse]["main_score"] == 16
    assert ranking[sparse]["final_score"] is None
    assert ranking[padded]["main_score"] == 16
    assert ranking[padded]["final_score"] is None
    assert published["publication_doc"]["final_ranking"][0]["submission_id"] == broad
    results = harness.service.public_results(round_id, broad)
    assert len(results["outputs"]) == contracts.BENCHMARK_ICP_COUNT
    assert all(len(output["companies"]) == 5 for output in results["outputs"].values())
    assert all(
        float(row["per_icp_score"]) == 35
        for rows in results["scores"].values() for row in rows
    )
    expected_schema = contact_policy.output_schema(published["configuration_doc"])
    assert all(output["schema_version"] == expected_schema for output in results["outputs"].values())
    if with_contacts:
        assert len(results["contact_verifications"]) == contracts.BENCHMARK_ICP_COUNT
        assert all(
            [row["contact_qualified"] for row in rows] == [True] * 5
            for rows in results["contact_verifications"].values()
        )
    assert public_dashboard._stage1_scores(harness.service, published)[broad] == 35

    score_runs = harness.service.store.list_runs(round_id, kind="score")
    assert len({run["runner_hotkey"] for run in score_runs if run["status"] == "accepted"}) == 2
    assert any(item["hits"] for item in scoring_leases)
    assert judged_companies.count("Sparse Company 0") < 2 * contracts.BENCHMARK_ICP_COUNT

    if exhausted_company:
        attempts_by_run = {}
        for scored_run_id, name in exhausted_attempts:
            if name == exhausted_company:
                attempts_by_run[scored_run_id] = attempts_by_run.get(scored_run_id, 0) + 1
        assert attempts_by_run and set(attempts_by_run.values()) == {scoring.MAX_JUDGE_RETRIES}
        exhausted_keys = {
            key for lease in scoring_leases for key in lease["miss_keys"]
        }
        cache_hit_keys = {
            key for lease in scoring_leases for key in lease["hit_keys"]
        }
        assert exhausted_keys and cache_hit_keys

        failed_rows = [
            run["qualification_doc"]["companies"][3]
            for run in padded_main
        ]
        assert failed_rows
        assert all(row["company_qualified"] is False for row in failed_rows)
        persisted_judgments = [
            harness.service.store.get_company_judgment(
                ref["cache_key"], ref["authority_slot"]
            )
            for ref in replay_hits
            if ref["company_index"] == 3
        ]
        assert any(persisted_judgments)
        assert all(
            judgment["evidence_doc"]["raw_judgment"]["final_score"] == 0
            and any(
                receipt["decision"] == "unavailable"
                and receipt["reason"] == "source_blocked"
                and receipt["failure_class"] == COMPANY_VERIFICATION_EXHAUSTED_FAILURE_CLASS
                for receipt in judgment["evidence_doc"]["raw_judgment"]["verifier_gate_receipts"]
                if receipt["gate"] == "company_fit"
            )
            for judgment in persisted_judgments
            if judgment is not None
        )
        round_config = harness.service.store.get_round(round_id)["configuration_doc"]
        for run in persisted:
            if run["submission_id"] != padded or run["icp_position"] >= contracts.BENCHMARK_ICP_COUNT:
                continue
            score = verify.per_icp_score(
                harness.service.evaluation_icps(round_id)[run["icp_position"]],
                run["qualification_doc"]["companies"],
                round_config["scorer_policy"],
            )
            assert score["fp_gate_count"] == score["fp_unverified_primary_count"] == 0
        fixtures.assert_canary_absent(harness, lambda: psycopg2.connect(**dsn))

    repository_root = tmp_path / "quality-promotion-repository"
    repository_root.mkdir()
    remote = fixtures.promotion_repository(repository_root)
    harness.service._config.baseline_promoter_factory = lambda: GitPromoter(
        str(remote), tmp_path / "quality-promotion-objects"
    )
    harness.clock.now = datetime.fromisoformat(
        str(published["published_at"]).replace("Z", "+00:00")
    )
    assert harness.service.promote_pending_baselines() == {"status": "ok", "promoted": 1}
    promoted = subprocess.run(
        ("git", "--git-dir", str(remote), "archive", "--format=tar.gz", "lab"),
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout
    with tarfile.open(fileobj=io.BytesIO(promoted), mode="r:gz") as archive:
        assert archive.extractfile("flavor.txt").read().decode("utf-8") == "Broad"


def test_historical_rounds_still_publish_after_quality_migration(database, tmp_path, monkeypatch):
    psycopg2, dsn = database
    monkeypatch.setattr(fixtures.Harness, "objects_key", lambda self: "quality-historical-" + self.tmp.name)
    fixtures.test_full_round_publishes_results_and_next_day_uses_the_public_baseline(
        lambda: psycopg2.connect(**dsn), tmp_path,
    )
