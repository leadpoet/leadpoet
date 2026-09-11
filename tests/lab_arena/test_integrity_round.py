"""Saved-result verification of integrity rounds on disposable PostgreSQL."""
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from lab_arena import contracts, integrity, service as svc, verify
from lab_arena.chain import MetagraphSnapshot
from qualification.scoring.arena_integrity import canonical_company_identity
from tests.lab_arena import test_lab_arena_service_round as fixtures
from tests.lab_arena.lab_arena_pg_harness import DEFAULT_MIGRATIONS, database_with_lab_arena_migration
from tests.lab_arena.test_integrity_policy import fresh_icps

MIGRATIONS = DEFAULT_MIGRATIONS + (
    "211-lab-arena-owner-admission.sql", "212-lab-arena-accepted-judgment-cache.sql", "213-lab-arena-score-integrity.sql",
)


@pytest.fixture()
def database():
    yield from database_with_lab_arena_migration(MIGRATIONS)


class IntegrityHarness(fixtures.Harness):
    def objects_key(self):
        return "integrity-round"

    def build_service(self):
        service = super().build_service()
        service._config.defaults = replace(service._config.defaults, integrity_from="2026-01-01T00:00:00Z")
        service._config.confirmation_icp_source = lambda **kwargs: fresh_icps()
        def metagraph(*, finalized=True):
            assert finalized
            hotkeys = tuple(key.ss58_address for key in fixtures.KEYS.values())
            return MetagraphSnapshot(netuid=71, block_number=123, block_hash="0x" + "a" * 64,
                hotkeys=hotkeys, coldkeys=hotkeys, validator_permit=tuple(key in self.runner_keys for key in hotkeys))
        self.chain.metagraph = metagraph
        return service

    def advance_until(self, target, *, runners=1, max_steps=80):
        for _ in range(max_steps):
            status = self.status()
            if status == target:
                return self.service.store.get_round(self.round_id)
            if status in ("stage1", "stage1_scoring", "stage2", "stage2_scoring", "stage3", "stage3_scoring"):
                self.run_stage_with_runners(runners)
            outcome = self.service.advance_round(self.round_id)
            assert outcome.get("status") not in ("cancelled", "terminal", "retry", "stale"), (status, outcome)
        raise AssertionError((target, self.status()))


@pytest.mark.parametrize("case,main_score,confirmation_score,expected", [
    ("regression", 80.0, 39.0, "no_king"),
    ("confirmed", 80.0, 60.0, "crowned"),
    ("belowmargin", 40.9, 60.0, "no_king"),
])
def test_confirmation_controls_winner_and_survives_service_restart(database, tmp_path, monkeypatch, case, main_score, confirmation_score, expected):
    psycopg2, dsn = database
    connect = lambda: psycopg2.connect(**dsn)
    harness = IntegrityHarness(connect, tmp_path, challengers=["Challenger"], runners=["alpha"])
    def judge(companies, icp, reference):
        assert not reference and "verified_example_company" not in icp
        indexes, _ = verify.bucket_skip(icp, companies)
        rows = []
        for index in indexes:
            baseline = companies[index]["company_name"].startswith("PublicBaseline")
            score = 40.0 if baseline else confirmation_score if icp["icp_id"].startswith("confirmation_") else main_score
            rows.append({"final_score": score, "company_index": index,
                "company_identity_key": canonical_company_identity(companies[index]).key,
                "company_qualified": True, "duplicate_company": False,
                "verifier_gate_receipts": [{"gate": "company_fit", "decision": "match"}],
                "intent_signals_detail": [], "failure_reason": ""})
        return rows
    monkeypatch.setattr(fixtures, "deterministic_scorer", judge)
    harness.chain.epoch = 25000 + int(confirmation_score)
    harness.clock.now = datetime.now(timezone.utc)
    round_id = "arena-2026-11-01-" + case
    config = harness.service.create_round(datetime.now(timezone.utc) + timedelta(hours=12), round_id=round_id)
    harness.round_id = round_id
    assert config["integrity_policy"] == integrity.POLICY
    challenger = harness.submit("Challenger", round_id)
    failed = harness.submit("Broken", round_id)
    harness.broken.add(failed)
    harness.clock.advance_to(harness.schedule()["submission_cutoff"])
    harness.advance_until("scored")
    before = harness.service.store.get_round(round_id)
    bank_hash = before["confirmation_bank_hash"]
    assert bank_hash and not harness.service.public_round(round_id).get("confirmation_submission_ids")
    assert "confirmation_bank" not in harness.service.public_benchmark(round_id)
    harness.service = harness.build_service()
    confirmation_required = main_score >= 41
    harness.advance_until("stage3" if confirmation_required else "confirmed")
    chosen = harness.service.store.get_round(round_id)
    assert (challenger in chosen["confirmation_cohort"]["submission_ids"]) is confirmation_required
    harness.service = harness.build_service()
    published = harness.advance_until("published")
    assert published["king_outcome"] == expected
    assert published["confirmation_bank_hash"] == bank_hash
    assert len(harness.service.store.list_runs(round_id, stage=3, kind="execute")) == (10 if confirmation_required else 0)
    ranking = {row["submission_id"]: row for row in published["publication_doc"]["final_ranking"]}
    assert ranking[challenger]["main_score"] == main_score
    assert ranking[challenger]["final_score"] == (confirmation_score if confirmation_required else main_score)
    assert ranking[challenger]["cost_summary"]["qualified_company_count"] == (125 if confirmation_required else 100)
    assert ranking[failed]["main_score"] is None
    assert ranking[failed]["eligible"] is False
    public = harness.service.public_results(round_id, challenger)
    assert len(public["scores"]["confirmation"]) == (5 if confirmation_required else 0)
    bank = harness.service.public_benchmark(round_id)["confirmation_bank"]
    assert contracts.document_hash(bank) == bank_hash
    assert "verified_example_company" not in str(bank)
    fixtures.assert_canary_absent(harness, connect)
    if expected == "crowned":
        # Publication must still permit the existing downstream baseline
        # promotion writes. Exercise them against an isolated local git remote.
        from lab_arena.promotion import GitPromoter
        repository_root = tmp_path / "promotion-repository"
        repository_root.mkdir()
        remote = fixtures.promotion_repository(repository_root)
        harness.service._config.baseline_promoter_factory = lambda: GitPromoter(str(remote), tmp_path / "promotion-objects")
        harness.clock.now = datetime.fromisoformat(str(published["published_at"]).replace("Z", "+00:00"))
        assert harness.service.promote_pending_baselines() == {"status": "ok", "promoted": 1}
        assert harness.service.store.get_round(round_id)["baseline_promoted_at"] is not None
