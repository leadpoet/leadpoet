"""Company-only V6 lifecycle proof on the disposable Arena database."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
from urllib.parse import urlsplit

import pytest

from lab_arena import (
    company_judgments,
    contracts,
    intent_details_policy,
    quality_policy,
    runtime,
    scoring,
)
from lab_arena.output import validate_output_document
from lab_arena.promotion import GitPromoter
from leadpoet_canonical import arena_weights
from qualification.scoring.competition import apply_company_judgment_context
from tests.lab_arena.icp_fixtures import daily_icps
from tests.lab_arena.lab_arena_pg_harness import (
    CURRENT_SERVICE_MIGRATIONS,
    database_with_lab_arena_migration,
)
from tests.lab_arena.company_quality_round_test import QualityHarness
from tests.lab_arena.parallel_twenty_icp_execution_postgres_test import (
    _verified_test_pool,
)
from tests.lab_arena import test_lab_arena_service_round as fixtures
from tests.test_arena_company_quality_policy import _positive_breakdown


@pytest.fixture()
def database():
    yield from database_with_lab_arena_migration(CURRENT_SERVICE_MIGRATIONS)


def _company(name: str = "Acme") -> dict:
    slug = name.casefold().replace(" ", "-")
    return {
        "company_name": name,
        "company_website": f"https://{slug}.example.com/",
        "company_linkedin": f"https://linkedin.com/company/{slug}",
        "industry": "Software",
        "employee_count": "51-200",
        "company_stage": "Series A",
        "country": "United States",
        "state": "CA",
        "intent_details": (
            f"{name} announced a product launch that matches the requested signal."
        ),
        "intent_signals": [
            {
                "matched_icp_signal": 0,
                "description": "Announced a product launch",
                "date": "2026-09-01",
                "url": f"https://{slug}.example.com/launch",
            }
        ],
        "company_stage_evidence": [],
        "required_attribute": None,
    }


def _company_only_policy() -> dict:
    return scoring.build_scorer_policy(
        scoring_adapter_version="qualification_integrity_v2",
        company_quality=True,
        intent_details=True,
    )


def test_v6_discards_legacy_contact_claims_before_cache_identity():
    expected_schema = intent_details_policy.COMPANY_ONLY_OUTPUT_SCHEMA
    normalized = []
    for contact in (
        None,
        "invalid legacy contact",
        {"email": "different@example.com", "untrusted": [1, 2, 3]},
    ):
        row = _company()
        if contact is not None:
            row["contact"] = contact
        document = validate_output_document(
            {"schema_version": expected_schema, "companies": [row]},
            expected_schema_version=expected_schema,
        )
        assert "contact" not in document["companies"][0]
        normalized.append(document["companies"])
    assert normalized[0] == normalized[1] == normalized[2]

    scopes = []
    for companies in normalized:
        scoring_input = scoring.build_scoring_input(
            scored_run_id="execute-1",
            icp={
                "icp_id": "public-icp-1",
                "industry": "Software",
                "employee_count": ["51-200"],
                "intent_signals": ["Product launch"],
                "max_companies": 5,
            },
            companies=companies,
            policy=_company_only_policy(),
            evaluation_date="2026-09-24",
        )
        assert "contact_source_evidence" not in scoring_input
        scopes.append(
            company_judgments.build_company_scopes(
                scoring_input=scoring_input,
                round_id="arena-2026-09-24-companyonly",
                network_name="finney",
                netuid=71,
                scorer_image_digest="sha256:" + "a" * 64,
                scorer_image_reference=(
                    "registry.example/scorer@sha256:" + "a" * 64
                ),
                integrity_policy="arena_integrity_v1",
                company_quality_policy=quality_policy.POLICY,
            )
        )
    assert [scope[0]["cache_key"] for scope in scopes] == [
        scopes[0][0]["cache_key"]
    ] * 3


@pytest.mark.parametrize(
    "normalized_scale", [False, True], ids=["historical-scale", "reachable-100"]
)
@pytest.mark.parametrize(
    "company_quality",
    [False, True],
    ids=["current-production-policy", "company-quality-policy"],
)
def test_company_only_ten_icp_baseline_and_miner_complete_full_lifecycle(
    database, tmp_path, company_quality, normalized_scale
):
    psycopg2, dsn = database
    connect = lambda: psycopg2.connect(**dsn)
    harness = QualityHarness(
        connect,
        tmp_path,
        challengers=["Miner"],
        runners=["alpha", "beta"],
    )
    frozen_bank = []
    for position, source in enumerate(daily_icps()[:10]):
        row = deepcopy(source)
        row.update(
            icp_id=f"icp_20260923_{position + 1:03d}",
            prompt=(
                f"Public fixture ICP {position + 1}."
                " Target contacts: VP Engineering in the United States."
            ),
            contact_policy="contacts_v1",
            target_roles=["VP Engineering"],
            target_seniority="VP",
            contact_geography={
                "countries": ["US"],
                "regions": [],
                "cities": [],
            },
        )
        frozen_bank.append(row)
    bank_snapshot = deepcopy(frozen_bank)
    harness.service.config.defaults = replace(
        harness.service.config.defaults,
        benchmark_icp_count=10,
        promotion_margin=0.5,
        runner_slot_ceiling=10,
        pool_percent=30,
        rewards_enabled=True,
        per_icp_cost_policy=True,
        normalize_intent_scale=normalized_scale,
        contacts_from=None,
        company_quality_from=(
            "2026-01-01T00:00:00Z" if company_quality else None
        ),
        intent_details_from="2026-01-01T00:00:00Z",
        execution_sequence_from="2026-01-01T00:00:00Z",
        benchmark_disclosure_from="2026-01-01T00:00:00Z",
    )
    harness.service.config.daily_icp_source = lambda **kwargs: {
        "status": "ready",
        "set_id": int(kwargs["set_id"]),
        "icps": deepcopy(frozen_bank),
    }
    original_runner = harness.runner

    def verified_runner(index, parallel=4):
        runner = original_runner(index, parallel)
        runner._config.proxy_worker_pool = _verified_test_pool(parallel)
        return runner

    harness.runner = verified_runner
    original_run_icp = harness.sandbox.run_icp
    observed_execution_inputs = []
    observed_scoring_inputs = []

    def run_icp(spec, **kwargs):
        document = json.loads(
            (spec.input_dir / runtime.INPUT_FILE_NAME).read_text(encoding="utf-8")
        )
        base = original_run_icp(spec, **kwargs)
        if document.get("schema_version") == scoring.SCORING_INPUT_SCHEMA_VERSION:
            observed_scoring_inputs.append(document)
            assert "contact_source_evidence" not in document
            assert all("contact" not in row for row in document["companies"])
            assert ("company_judgment_cache" in document) is company_quality

            def judge(companies, _buyer, _reference):
                judgments = []
                for company in companies:
                    name = str(company["company_name"])
                    slug = name.casefold().replace(" ", "-")
                    score = 30.0 if name.startswith("PublicBaseline") else 60.0
                    raw = _positive_breakdown(
                        name,
                        str(urlsplit(company["company_website"]).hostname),
                        slug,
                    )
                    raw.update(
                        final_score=score,
                        intent_signal_raw=score,
                        intent_signal_final=score,
                    )
                    raw["intent_signals_detail"][0].update(
                        raw=score, after_decay=score
                    )
                    raw["verifier_gate_receipts"].append(
                        {"gate": "intent_details", "decision": "match"}
                    )
                    judgments.append(raw)
                return apply_company_judgment_context(
                    companies, judgments, contacts_required=False
                )

            judge.company_quality = company_quality
            judge.integrity_policy = True
            judge.contacts_required = False
            item = {"scored_run_id": document["scored_run_id"]}
            if company_quality:
                full, new = scoring.score_quality_work_item(
                    item,
                    icp=document["icp"],
                    companies=document["companies"],
                    scorer=judge,
                    cache_context=document["company_judgment_cache"],
                )
            else:
                full = scoring.score_work_item(
                    item,
                    icp=document["icp"],
                    companies=document["companies"],
                    scorer=judge,
                )
                new = None
            output = scoring.build_scoring_output(
                document["scored_run_id"],
                full,
                **({"company_judgments": new} if new is not None else {}),
            )
            return runtime.fake_result(
                exit_code=0, output_bytes=json.dumps(output).encode()
            )

        observed_execution_inputs.append(document)
        assert document["output_schema_version"] == (
            intent_details_policy.COMPANY_ONLY_OUTPUT_SCHEMA
        )
        assert "contact_policy" not in document["icp"]
        assert "target_roles" not in document["icp"]
        assert "Target contacts:" not in document["icp"]["prompt"]
        rows = json.loads(base.output_bytes)["companies"]
        flavor = rows[0]["company_name"].split(" Company ", 1)[0]
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
                # V6 accepts old agents that still emit a contact, then removes
                # it before persistence and scoring.
                contact={"legacy": flavor, "invalid": True},
            )
            row.pop("fit_summary", None)
            row.pop("fit_evidence_urls", None)
            for signal in row["intent_signals"]:
                signal.pop("why_now", None)
                signal.pop("snippet", None)
        return runtime.fake_result(
            exit_code=0,
            output_bytes=json.dumps(
                {
                    "schema_version": (
                        intent_details_policy.COMPANY_ONLY_OUTPUT_SCHEMA
                    ),
                    "companies": rows,
                }
            ).encode(),
        )

    harness.sandbox.run_icp = run_icp
    harness.chain.epoch = 40_000
    harness.clock.now = datetime.now(timezone.utc)
    round_id = "arena-2026-09-24-companyonly"
    configuration = harness.service.create_round(
        harness.clock.now + timedelta(minutes=30), round_id=round_id
    )
    harness.round_id = round_id
    assert configuration["stage_1_icp_count"] == 5
    assert configuration["stage_2_icp_count"] == 5
    assert configuration["promotion_margin"] == 0.5
    assert (configuration["scorer_policy"]["env_bindings"].get(
        contracts.SCORE_NORMALIZATION_BINDING
    ) == contracts.AVAILABLE_INTENT_CAP_NORMALIZATION) is normalized_scale
    assert configuration["scorer_policy"]["env_bindings"].get(
        contracts.PROVIDER_OBSERVATION_HANDOFF_BINDING
    ) == contracts.AUTHENTICATED_PROVIDER_OBSERVATION_HANDOFF
    assert configuration["intent_details_policy"] == intent_details_policy.POLICY
    assert (configuration.get("company_quality_policy") == quality_policy.POLICY) is (
        company_quality
    )
    assert configuration["sourcing_cost_eligibility_policy"] == (
        contracts.PER_ICP_SUCCESSFUL_CALLS_COST_POLICY
    )
    assert configuration["reward_constants"]["pool_percent"] == 30
    assert "contact_policy" not in configuration
    assert harness.service.public_round(round_id)["output_schema_version"] == (
        intent_details_policy.COMPANY_ONLY_OUTPUT_SCHEMA
    )

    miner_submission = harness.submit("Miner", round_id)
    harness.clock.advance_to(harness.schedule()["submission_cutoff"])
    harness.advance_until("published", runners=2, max_steps=120)
    assert frozen_bank == bank_snapshot
    assert len(observed_execution_inputs) == 20
    assert len(observed_scoring_inputs) == 20

    published = harness.service.store.get_round(round_id)
    assert published["status"] == "published"
    assert published["king_outcome"] == "crowned"
    assert published["publication_doc"]["king_decision"][
        "winner_submission_id"
    ] == miner_submission
    participants = published["participants"]
    assert len(participants) == 2
    for participant in participants:
        submission_id = participant["submission_id"]
        execution_runs = harness.service.store.list_runs(
            round_id, submission_id=submission_id, kind="execute"
        )
        score_runs = harness.service.store.list_runs(
            round_id, submission_id=submission_id, kind="score"
        )
        assert len(execution_runs) == len(score_runs) == 10
        assert all(run["status"] == "accepted" for run in execution_runs + score_runs)
        for run in execution_runs:
            stored = json.loads(
                harness.objects.get_bounded(
                    run["output_ref"], 512 * 1024
                ).decode("utf-8")
            )
            assert stored["schema_version"] == (
                intent_details_policy.COMPANY_ONLY_OUTPUT_SCHEMA
            )
            assert all("contact" not in company for company in stored["companies"])
            assert all(
                "contact_qualified" not in company
                for company in (run.get("qualification_doc") or {}).get(
                    "companies", []
                )
            )
        public = harness.service.public_results(round_id, submission_id)
        assert public["benchmark_icp_count"] == 10
        assert len(public["outputs"]) == 10
        assert "contact_verifications" not in public
        assert all(
            "contact" not in company
            for output in public["outputs"].values()
            for company in output["companies"]
        )

    ranking = published["publication_doc"]["final_ranking"]
    assert len(ranking) == 2
    assert sorted(entry["final_score"] for entry in ranking) == (
        [50.0, 100.0] if normalized_scale else [30.0, 60.0]
    )
    assert all(entry["eligible"] for entry in ranking)
    # The fixture has five LinkedIn identities per ICP, but all five company
    # websites share one registrable example.com domain. The cost denominator
    # therefore counts ten returned domains and fifty qualified identities.
    assert [
        entry["cost_summary"]["returned_company_count"] for entry in ranking
    ] == [10, 10]
    assert [
        entry["cost_summary"]["qualified_company_count"] for entry in ranking
    ] == [50, 50]
    assert all(len(entry["cost_summary"]["per_icp"]) == 10 for entry in ranking)

    repository_root = tmp_path / "promotion-repository"
    repository_root.mkdir()
    remote = fixtures.promotion_repository(repository_root)
    harness.service._config.baseline_promoter_factory = lambda: GitPromoter(
        str(remote), tmp_path / "promotion-objects"
    )
    assert harness.service.promote_pending_baselines() == {
        "status": "ok",
        "promoted": 1,
    }
    assert harness.service.activate_reward(round_id)["status"] == "activated"
    rewarded = harness.service.store.get_round(round_id)
    basis = rewarded["reward_basis_doc"]
    assert basis["king_hotkey"] == rewarded["king_hotkey"]
    epoch = int(basis["effective_reward_epoch"])
    assert harness.service.public_reward_basis(epoch)["reward_basis_hash"] == (
        basis["reward_basis_hash"]
    )

    burn_hotkey = harness.runner_keys[0]
    harness.service._config.accepted_burn_hotkey = burn_hotkey
    harness.chain.accepted_weight_epoch_scope = lambda: {
        "epoch": epoch,
        "genesis_hash": "1" * 64,
        "valid_from_block": 1,
        "valid_until_block": 360,
    }
    state = harness.service.public_weight_state(epoch)["state"]
    vector = arena_weights.derive_arena_weights(
        state, [basis["king_hotkey"], burn_hotkey]
    )
    assert vector["champion_share_ppb"] > 0
    assert vector["burned_residual_ppb"] > 0
