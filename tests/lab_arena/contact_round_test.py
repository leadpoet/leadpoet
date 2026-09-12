"""Saved and public end-to-end journey for an opt-in contact round."""

from __future__ import annotations

import asyncio
import json
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from urllib.parse import urlsplit

import pytest

from lab_arena import contracts, runtime, scoring, service as svc, verify
from lab_arena.promotion import GitPromoter
from lab_arena.contact_evidence import source_key
from qualification.scoring.arena_integrity import (
    canonical_company_identity,
    company_identity_alias_keys,
)
from qualification.scoring.contact_verification import verify_contact
from tests.lab_arena import test_lab_arena_service_round as fixtures
from tests.lab_arena.icp_fixtures import daily_icps
from tests.lab_arena.lab_arena_pg_harness import database_with_lab_arena_migration
from tests.lab_arena.test_integrity_policy import fresh_icps
from tests.lab_arena.test_integrity_round import IntegrityHarness, MIGRATIONS

MIGRATION = "215-lab-arena-contacts.sql"


@pytest.fixture()
def database():
    yield from database_with_lab_arena_migration(
        MIGRATIONS
        + (
            MIGRATION,
            "223-lab-arena-cancelled-call-late-settlement.sql",
            "225-lab-arena-openrouter-delayed-cost-reconciliation.sql",
        )
    )


def _contact_icps(rows: list[dict]) -> list[dict]:
    result = deepcopy(rows)
    for row in result:
        row.update(
            {
                "contact_policy": "contacts_v1",
                "target_roles": ["VP Engineering"],
                "target_seniority": "VP+",
                "contact_geography": {
                    "countries": ["US"],
                    "regions": [],
                    "cities": [],
                },
            }
        )
        row["prompt"] += " Find a VP Engineering contact at each company."
    return result


class ContactHarness(IntegrityHarness):
    def objects_key(self):
        return "contact-round"

    def build_service(self):
        service = super().build_service()
        service._config.defaults = replace(
            service._config.defaults,
            contacts_from="2026-01-01T00:00:00Z",
        )
        service._config.daily_icp_source = lambda **kwargs: {
            "status": "ready",
            "set_id": int(kwargs["set_id"]),
            "icps": _contact_icps(daily_icps()),
        }
        service._config.confirmation_icp_source = (
            lambda **_kwargs: _contact_icps(fresh_icps())
        )
        return service


def _claim(company: dict, *, valid_role: bool) -> dict:
    name = str(company["company_name"])
    slug = "-".join(name.casefold().split())
    domain = (urlsplit(company["company_website"]).hostname or "example.com")
    return {
        "full_name": "Ada Lovelace" if valid_role else "Grace Hopper",
        "role": "VP Engineering" if valid_role else "Engineering Intern",
        "linkedin_url": f"https://www.linkedin.com/in/{slug}/",
        "location": {"country": "US"},
        "email": f"{'ada' if valid_role else 'grace'}@{domain}",
        "email_source": {
            "provider": "harvestapi",
            "tool": "harvestapi_get_profile",
            "record_id": f"profile-{slug}",
        },
    }


def test_contact_generation_does_not_activate_an_existing_company_round(
    database, tmp_path
) -> None:
    psycopg2, dsn = database
    harness = fixtures.Harness(
        lambda: psycopg2.connect(**dsn), tmp_path,
        challengers=["CompanyOnly"], runners=["alpha"],
    )
    harness.service.config.daily_icp_source = lambda **kwargs: {
        "status": "ready", "set_id": int(kwargs["set_id"]),
        "icps": _contact_icps(daily_icps()),
    }
    original = harness.sandbox.run_icp
    observed = []

    def run_icp(spec, **kwargs):
        document = json.loads((spec.input_dir / runtime.INPUT_FILE_NAME).read_text())
        assert "contact_policy" not in document["icp"]
        observed.append(document["schema_version"])
        return original(spec, **kwargs)

    harness.sandbox.run_icp = run_icp
    harness.clock.now = datetime.now(timezone.utc)
    round_id = "arena-2026-12-01-companyonly"
    configuration = harness.service.create_round(
        harness.clock.now + timedelta(hours=12), round_id=round_id,
    )
    assert "contact_policy" not in configuration
    harness.round_id = round_id
    submission = harness.submit("CompanyOnly", round_id)
    harness.clock.advance_to(harness.schedule()["submission_cutoff"])
    harness.advance_until("published", runners=1)
    outputs = harness.service.public_results(round_id, submission)["outputs"]
    assert len(outputs) == contracts.BENCHMARK_ICP_COUNT
    assert observed
    assert all(
        output["schema_version"] == contracts.OUTPUT_DOCUMENT_SCHEMA_VERSION
        for output in outputs.values()
    )


async def _contact_result(company: dict, icp: dict, source: dict) -> dict:
    contact = company["contact"]
    first, last = contact["full_name"].split(" ", 1)
    domain = urlsplit(company["company_website"]).hostname

    async def execute(tool: str, _payload: dict):
        if tool == "harvestapi_get_profile":
            return {
                "status": "completed",
                "result": {
                    "elements": [
                        {
                            "id": contact["email_source"]["record_id"],
                            "linkedinUrl": contact["linkedin_url"],
                            "firstName": first,
                            "lastName": last,
                            "email": contact["email"],
                            "location": {"country": "United States"},
                            "currentPosition": {
                                "title": contact["role"],
                                "companyName": company["company_name"],
                                "companyDomain": domain,
                                "current": True,
                            },
                        }
                    ]
                },
            }
        if tool == "zerobounce_validate":
            return {"status": "catch_all"}
        raise AssertionError(tool)

    return await verify_contact(
        company,
        icp,
        source_evidence=source,
        execute=execute,
    )


def _install_contact_sandbox(harness: ContactHarness) -> None:
    original = harness.sandbox.run_icp

    def run_icp(spec, **kwargs):
        input_document = json.loads(
            (spec.input_dir / runtime.INPUT_FILE_NAME).read_text(encoding="utf-8")
        )
        if input_document.get("schema_version") != scoring.SCORING_INPUT_SCHEMA_VERSION:
            base = original(spec, **kwargs)
            if base.exit_code != 0 or not base.output_bytes:
                return base
            output = json.loads(base.output_bytes)
            companies = output["companies"][:2]
            for index, company in enumerate(companies):
                company["contact"] = _claim(company, valid_role=index == 0)
            document = {
                "schema_version": contracts.CONTACT_OUTPUT_DOCUMENT_SCHEMA_VERSION,
                "companies": companies,
            }
            return runtime.fake_result(
                exit_code=0,
                output_bytes=json.dumps(document).encode("utf-8"),
            )

        icp = input_document["icp"]
        evidence = input_document.get("contact_source_evidence") or {}
        companies = input_document["companies"]
        breakdowns = []
        for index, company in enumerate(companies):
            identity = canonical_company_identity(company)
            flavor = str(company["company_name"]).split(" Company ", 1)[0]
            base_score = (
                60.0
                if str(icp["icp_id"]).startswith("confirmation_")
                and flavor != "PublicBaseline"
                else 80.0 if flavor != "PublicBaseline" else 40.0
            )
            contact = asyncio.run(
                _contact_result(company, icp, evidence[source_key(company)])
            )
            qualified = contact["contact_qualified"] is True
            row = {
                "company_index": index,
                "company_identity_key": identity.key,
                "company_identity_alias_keys": list(
                    company_identity_alias_keys(identity)
                ),
                "company_qualified": qualified,
                "duplicate_company": False,
                "final_score": base_score if qualified else 0.0,
                "failure_reason": "" if qualified else "contact qualification failed",
                "intent_signals_detail": [],
                "verifier_gate_receipts": [
                    {"gate": "company_fit", "decision": "match"}
                ] + list(contact["verifier_gate_receipts"]),
                "contact_qualified": contact["contact_qualified"],
                "contact_identity_key": contact["contact_identity_key"],
                "email_status": contact["email_status"],
                "contact_verification": contact["contact_verification"],
            }
            breakdowns.append(row)
        output = scoring.build_scoring_output(
            input_document["scored_run_id"], breakdowns
        )
        return runtime.fake_result(
            exit_code=0,
            output_bytes=json.dumps(output).encode("utf-8"),
        )

    harness.sandbox.run_icp = run_icp


@pytest.mark.parametrize("delayed_disclosure", [False, True])
def test_contact_round_saves_scores_budget_counts_and_public_receipts(
    database, tmp_path, delayed_disclosure
) -> None:
    psycopg2, dsn = database
    connect = lambda: psycopg2.connect(**dsn)
    harness = ContactHarness(
        connect, tmp_path, challengers=["ContactWinner"], runners=["alpha"]
    )
    _install_contact_sandbox(harness)
    harness.chain.epoch = 29_000
    harness.clock.now = datetime.now(timezone.utc)
    harness.service.config.defaults = replace(
        harness.service.config.defaults,
        rewards_enabled=True,
        benchmark_disclosure_from=(
            "2026-01-01T00:00:00Z" if delayed_disclosure else None
        ),
    )
    round_id = (
        "arena-2026-12-01-contactsdelay"
        if delayed_disclosure else "arena-2026-12-01-contacts"
    )
    configuration = harness.service.create_round(
        harness.clock.now + timedelta(hours=12), round_id=round_id
    )
    harness.round_id = round_id
    assert configuration["contact_policy"] == "contacts_v1"
    assert configuration["scorer_policy"]["scoring_adapter_version"] == (
        "qualification_contacts_v3"
    )
    winner = harness.submit("ContactWinner", round_id)
    with pytest.raises(svc.ServiceError, match="results_not_public"):
        harness.service.public_results(round_id, winner)
    harness.clock.advance_to(harness.schedule()["submission_cutoff"])
    harness.advance_until("scored", runners=1, max_steps=100)
    harness.service = harness.build_service()
    published = harness.advance_until("published", runners=1, max_steps=100)

    assert published["king_outcome"] == "crowned"
    ranking = next(
        row for row in published["publication_doc"]["final_ranking"]
        if row["submission_id"] == winner
    )
    assert ranking["cost_summary"]["qualified_company_count"] == 25

    saved = [
        run
        for run in harness.service.store.list_runs(
            round_id, submission_id=winner, kind="execute"
        )
        if run.get("qualification_doc")
    ]
    assert len(saved) == contracts.MAX_EVALUATION_ICP_COUNT
    assert all(
        [row["contact_qualified"] for row in run["qualification_doc"]["companies"]]
        == [True, False]
        for run in saved
    )

    # Contact qualification must still permit the existing winner promotion
    # and reward path. Neither operation waits for public benchmark details.
    repository_root = tmp_path / "promotion-repository"
    repository_root.mkdir()
    remote = fixtures.promotion_repository(repository_root)
    harness.service.config.baseline_promoter_factory = lambda: GitPromoter(
        str(remote), tmp_path / "promotion-objects"
    )
    assert harness.service.promote_pending_baselines() == {
        "status": "ok", "promoted": 1,
    }
    assert harness.service.activate_reward(round_id)["status"] == "activated"
    promoted = harness.service.store.get_round(round_id)
    preserved = {
        key: promoted[key] for key in (
            "publication_doc", "promotion_doc", "baseline_promoted_at",
            "reward_basis_doc", "reward_activated_at", "effective_reward_epoch",
        )
    }
    assert harness.service.public_reward_basis(
        int(promoted["effective_reward_epoch"])
    ) == promoted["reward_basis_doc"]

    public = harness.service.public_results(round_id, winner)
    if delayed_disclosure:
        assert public["outputs"] == {}
        assert public["contact_verifications"] == {}
        assert public["run_results"] == []
        with pytest.raises(svc.ServiceError, match="benchmark_not_public"):
            harness.service.public_benchmark(round_id)
        harness.service = harness.build_service()
        reveal_at = datetime.fromisoformat(
            configuration["schedule"]["submission_cutoff"].replace("Z", "+00:00")
        ) + timedelta(hours=24)
        harness.clock.now = reveal_at - timedelta(microseconds=1)
        assert harness.service.public_results(round_id, winner) == public
        harness.clock.now = reveal_at
        public = harness.service.public_results(round_id, winner)
        assert len(harness.service.public_benchmark(round_id)["icps"]) == 20
    assert len(public["outputs"]) == contracts.MAX_EVALUATION_ICP_COUNT
    assert len(public["contact_verifications"]) == contracts.MAX_EVALUATION_ICP_COUNT
    first_output = next(iter(public["outputs"].values()))
    assert first_output["schema_version"] == contracts.CONTACT_OUTPUT_DOCUMENT_SCHEMA_VERSION
    assert first_output["companies"][0]["contact"]["email"].startswith("ada@")
    first_receipts = next(iter(public["contact_verifications"].values()))
    assert [row["contact_qualified"] for row in first_receipts] == [True, False]
    assert first_receipts[0]["email_status"] == "catch_all"
    assert first_receipts[0]["contact_verification"]["decision"] == "verified"
    assert first_receipts[1]["contact_verification"]["decision"] == "mismatch"
    verification = first_receipts[0]["contact_verification"]
    assert set(verification) <= {
        "decision",
        "reason",
        "verified_at",
        "subchecks",
        "evidence_hashes",
        "evidence_timestamps",
    }
    assert verification["evidence_hashes"]
    assert all(
        isinstance(value, str) and len(value) <= 100
        for value in verification["evidence_hashes"].values()
    )
    assert "elements" not in json.dumps(verification)
    assert "currentPosition" not in json.dumps(verification)
    after = harness.service.store.get_round(round_id)
    assert {key: after[key] for key in preserved} == preserved
    fixtures.assert_canary_absent(harness, connect)
