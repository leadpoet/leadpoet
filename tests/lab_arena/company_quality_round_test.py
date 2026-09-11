"""Frozen activation and complete saved/public company-quality rounds."""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
import random
from urllib.parse import urlsplit

import pytest

from lab_arena import contracts, contact_policy, public_dashboard, quality_policy, runtime, scoring, verify
from lab_arena.service import ArenaService, RoundDefaults, ServiceConfig, ServiceError
from qualification.scoring.competition import apply_company_judgment_context
from tests.lab_arena import test_lab_arena_service_round as fixtures
from tests.lab_arena.lab_arena_pg_harness import database_with_lab_arena_migration
from tests.lab_arena.test_integrity_round import IntegrityHarness, MIGRATIONS
from tests.test_arena_company_quality_policy import _positive_breakdown

QUALITY_MIGRATION = "20260911200103_lab_arena_company_judgments.sql"


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


@pytest.fixture()
def database():
    yield from database_with_lab_arena_migration(MIGRATIONS + ("215-lab-arena-contacts.sql", "216-lab-arena-validator-participation.sql", QUALITY_MIGRATION))


def test_database_coverage_score_matches_validator_exactly(database):
    psycopg2, dsn = database
    random_values = random.Random(190)
    cases = [[100, 100, 10, 0, 0], [40] * 5, [40] * 4 + [0], [100] * 10 + [0] * 10]
    cases += [[value] * 5 for value in (0, 100, 0.1234567890125, 0.1234567890135, 1e-28)]
    cases += [[random_values.random() * 100 for _ in range(n)] for n in (5, 10, 20) for _ in range(100)]
    with psycopg2.connect(**dsn) as connection, connection.cursor() as cursor:
        for values in cases:
            cursor.execute("SELECT public.lab_arena__company_quality_stage_score(%s::double precision[])", (values,))
            assert cursor.fetchone()[0] == verify.stage_score(values, len(values), company_quality=True), values
        for role in ("anon", "authenticated", "service_role", "lab_arena_service"):
            cursor.execute("SELECT has_function_privilege(%s, 'public.lab_arena__company_quality_stage_score(double precision[])', 'EXECUTE')", (role,))
            assert cursor.fetchone()[0] is False


@pytest.mark.parametrize("with_contacts", [False, True])
def test_quality_round_publishes_coverage_winner_after_restart(database, tmp_path, with_contacts):
    psycopg2, dsn = database
    harness = QualityHarness(lambda: psycopg2.connect(**dsn), tmp_path, challengers=["Broad", "Spiky"], runners=["alpha", "beta"])
    if with_contacts:
        from tests.lab_arena.contact_round_test import _contact_icps
        source = harness.service.config.daily_icp_source
        confirmation_source = harness.service.config.confirmation_icp_source
        harness.service.config.defaults = replace(harness.service.config.defaults, contacts_from="2026-01-01T00:00:00Z")
        harness.service.config.daily_icp_source = lambda **kwargs: {**source(**kwargs), "icps": _contact_icps(source(**kwargs)["icps"])}
        harness.service.config.confirmation_icp_source = lambda **kwargs: _contact_icps(confirmation_source(**kwargs))
    original = harness.sandbox.run_icp
    observed = []

    def run_icp(spec, **kwargs):
        document = json.loads((spec.input_dir / runtime.INPUT_FILE_NAME).read_text())
        if document.get("schema_version") != scoring.SCORING_INPUT_SCHEMA_VERSION:
            assert document["icp"]["company_quality_policy"] == quality_policy.POLICY
            assert "company_linkedin" in document["icp"]["company_requirements"]
            base = original(spec, **kwargs)
            if not base.output_bytes:
                return base
            rows = json.loads(base.output_bytes)["companies"]
            for row in rows:
                slug = row["company_name"].lower().replace(" ", "-")
                row.update(company_linkedin=f"https://linkedin.com/company/{slug}", state="CA")
                if with_contacts:
                    from tests.lab_arena.contact_round_test import _claim
                    row["contact"] = _claim(row, valid_role=True)
            return runtime.fake_result(exit_code=0, output_bytes=json.dumps({"companies": rows}).encode())
        icp = document["icp"]
        confirmation = str(icp["icp_id"]).startswith("confirmation_")
        position = int(str(icp["icp_id"]).rsplit("_", 1)[-1]) - 1
        def judge(companies, buyer, _reference):
            c = companies[0]
            name = c["company_name"]
            flavor = name.split(" Company ", 1)[0]
            score = 40 if flavor == "PublicBaseline" else (50 if confirmation else 60)
            if flavor == "Spiky":
                score = [100, 100, 10, 0, 0][position % 5] if confirmation else 80
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
            observed.append((flavor, confirmation, position))
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
    spiky = harness.submit("Spiky", round_id)
    harness.clock.advance_to(harness.schedule()["submission_cutoff"])
    harness.advance_until("scored", runners=2, max_steps=120)
    harness.service = harness.build_service()
    published = harness.advance_until("published", runners=2, max_steps=120)
    assert published["king_outcome"] == "crowned"
    ranking = {row["submission_id"]: row for row in published["publication_doc"]["final_ranking"]}
    assert ranking[broad]["final_score"] == 50
    assert ranking[spiky]["main_score"] == 80
    assert ranking[spiky]["final_score"] == verify.stage_score([100, 100, 10, 0, 0], 5, company_quality=True)
    assert published["publication_doc"]["final_ranking"][0]["submission_id"] == broad
    results = harness.service.public_results(round_id, broad)
    assert len(results["outputs"]) == contracts.MAX_EVALUATION_ICP_COUNT
    expected_schema = contact_policy.output_schema(published["configuration_doc"])
    assert all(output["schema_version"] == expected_schema for output in results["outputs"].values())
    if with_contacts:
        assert len(results["contact_verifications"]) == contracts.MAX_EVALUATION_ICP_COUNT
    assert public_dashboard._stage1_scores(harness.service, published)[broad] == 60
    assert observed


def test_historical_rounds_still_publish_after_quality_migration(database, tmp_path, monkeypatch):
    psycopg2, dsn = database
    monkeypatch.setattr(fixtures.Harness, "objects_key", lambda self: "quality-historical-" + self.tmp.name)
    fixtures.test_full_round_publishes_results_and_next_day_uses_the_public_baseline(
        lambda: psycopg2.connect(**dsn), tmp_path,
    )
