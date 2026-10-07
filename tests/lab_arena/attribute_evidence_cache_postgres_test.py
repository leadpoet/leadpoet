"""Source hints require new judgments; stored cache plans survive an upgrade."""

import copy
from types import SimpleNamespace
import pytest
from lab_arena import contracts, judgment_cache, scoring
from lab_arena.service import ArenaService
from lab_arena.store import ArenaStore, PsycopgTransport, hash_lease_token
from tests.lab_arena import idle_worker_claims416_postgres_test as current
from tests.lab_arena.test_lab_arena_migration_postgres import (
    round_config,
    hotkey,
    commit_round,
    _execute_everything,
    _commit_plan,
    claim,
)
from tests.lab_arena.company_judgments_postgres_test import _frozen_participants
from qualification.scoring import competition, lead_scorer
from qualification.scoring.company_fit_decision import company_fit_match
from gateway.qualification.models import CompanyOutput

database = current.database
migrated = current.migrated
from tests.lab_arena.attribute_evidence_cache_test import COMPANY, ICP


@pytest.mark.parametrize("field", ["evidence_quote", "evidence_url", "upgrade"])
def test_service_preserves_old_plans_and_judges_changed_hints(
    database, migrated, monkeypatch, field
):
    psycopg, dsn = database
    transport = PsycopgTransport(lambda: psycopg.connect(**dsn))
    store = ArenaStore(transport)
    try:
        round_id = (
            "arena-2026-10-07-"
            + {"evidence_quote": "quote", "evidence_url": "url", "upgrade": "upgrade"}[
                field
            ]
        )
        runner = hotkey("hintcache-runner")
        cfg = round_config(round_id, [runner], cost_per_company_microusd=1_000_000)
        cfg.update(
            {
                "integrity_policy": "arena_integrity_v1",
                "intent_details_policy": "intent_details_v1",
                "scorer_policy": scoring.build_scorer_policy(
                    scoring_adapter_version="qualification_integrity_v2",
                    intent_details=True,
                ),
            }
        )
        assert store.create_round(round_id, cfg)["status"] == "created"
        parts = _frozen_participants(store, round_id, 3, prefix=field.replace("_", ""))
        commit_round(store, round_id, parts)
        assert (
            store.open_stage(round_id, 1, parts, list(contracts.stage_positions(1)))[
                "status"
            ]
            == "ok"
        )
        executed = _execute_everything(store, round_id, runner)
        assert store.close_stage(round_id, 1)["status"] == "closed"
        _commit_plan(store, round_id, 1)
        frozen = store.get_round(round_id)
        original = copy.deepcopy(COMPANY)
        changed = copy.deepcopy(original)
        changed["required_attribute"][
            field if field != "upgrade" else "evidence_quote"
        ] = (
            "https://www.ibm.com/different"
            if field == "evidence_url"
            else "A quote absent from the retained page."
        )
        icp = ICP
        policy = cfg["scorer_policy"]
        outputs = {}
        items = []
        for p in parts:
            for pos in contracts.stage_positions(1):
                run = store.get_run(executed[(p["submission_id"], pos)])
                companies = [changed if p is parts[2] else original] if pos == 0 else []
                outputs[run["output_ref"]] = {
                    "schema_version": "leadpoet.lab_arena.output.v6",
                    "companies": companies,
                }
                items.append(
                    {
                        "scored_run_id": run["run_id"],
                        "submission_id": p["submission_id"],
                        "icp_position": pos,
                        "output_ref": run["output_ref"],
                    }
                )
        service = object.__new__(ArenaService)
        service._store = store
        service._objects = SimpleNamespace(
            get_bounded=lambda ref, limit: contracts.canonical_json(
                outputs[ref]
            ).encode()
        )
        service._round = lambda rid: store.get_round(rid)
        service._load_scoring_plan = lambda r, stage: {"work_items": items}
        service._require_code_review = (
            lambda *a: None
        )  # real fixture reviews passed above; this skips service-object config dependencies only
        service.evaluation_icps = lambda rid: [icp for _ in range(20)]
        current_projection = competition.effective_competition_input

        def legacy_projection(*args, **kwargs):
            effective = current_projection(*args, **kwargs)
            for row in effective["companies"]:
                row.pop("required_attribute", None)
            return effective

        if field == "upgrade":
            monkeypatch.setattr(
                competition, "effective_competition_input", legacy_projection
            )
        assert service.open_scoring(round_id, 1)["status"] == "ok"
        monkeypatch.setattr(
            competition, "effective_competition_input", current_projection
        )
        pair = [
            r
            for r in store.list_runs(round_id, kind="score", stage=1)
            if r["icp_position"] == 0
        ]
        by_submission = {r["submission_id"]: r for r in pair}
        assert (
            len(pair) == 3
            and by_submission[parts[0]["submission_id"]]["judgment_cache_key"]
            == by_submission[parts[1]["submission_id"]]["judgment_cache_key"]
        )
        assert (
            by_submission[parts[0]["submission_id"]]["judgment_cache_key"]
            == by_submission[parts[2]["submission_id"]]["judgment_cache_key"]
        ) is (field == "upgrade")
        # The identical follower uses its source leader without a new review.
        leased, token, *_ = claim(store, round_id, runner, excluded=[runner])
        assert leased["kind"] == "score" and leased["icp_position"] == 0
        assert leased["submission_id"] == parts[0]["submission_id"]
        source = store.get_run(leased["run_id"])
        identity = competition.canonical_company_identity(
            competition._normalized_company(original, integrity_policy=True)
        )
        prior = {
            "company_index": 0,
            "company_identity_key": identity.key,
            "company_identity_alias_keys": list(
                competition.company_identity_alias_keys(identity)
            ),
            "company_qualified": False,
            "duplicate_company": False,
            "company_name": "IBM",
            "final_score": 0.0,
            "failure_reason": "Original cached reason from occurring source quote",
            "verifier_gate_receipts": [
                {
                    "gate": "local_fixture_evidence",
                    "source_url": original["required_attribute"]["evidence_url"],
                    "source_quote": original["required_attribute"]["evidence_quote"],
                }
            ],
        }
        output = scoring.build_scoring_output(leased["scored_run_id"], [prior])
        evidence = judgment_cache.build_evidence_snapshot(
            output=output,
            cache_scope=source["judgment_scope_doc"],
            source_score_run_id=leased["run_id"],
            source_scored_run_id=leased["scored_run_id"],
            source_output_ref="arena/local/score.json",
            source_runner_hotkey=runner,
            runner_authority_exclusions=leased["runner_authority_exclusions"],
        )
        assert (
            store.complete_attempt(
                run_id=leased["run_id"],
                lease_token_hash=hash_lease_token(token),
                result={"terminal_status": "accepted"},
                terminal_cause="accepted",
                output_ref=evidence["source_output_ref"],
                judgment_evidence=evidence,
                judgment_evidence_hash=contracts.document_hash(evidence),
            )["status"]
            == "accepted"
        )
        pair = [
            r
            for r in store.list_runs(round_id, kind="score", stage=1)
            if r["icp_position"] == 0
        ]
        follower = next(
            r for r in pair if r["submission_id"] == parts[1]["submission_id"]
        )
        assert follower["status"] == "accepted" and follower["runner_hotkey"] is None
        assert follower["judgment_cache_source_run_id"] == leased["run_id"]
        monkeypatch.setattr(
            lead_scorer,
            "score_company_competition_intent",
            lambda *a, **k: (_ for _ in ()).throw(
                AssertionError("Fresh review forbidden")
            ),
        )
        restored = service._verified_breakdowns(
            follower, icp=icp, companies=[original], policy=policy
        )
        assert restored == [prior]
        assert (
            restored[0]["verifier_gate_receipts"][0]["source_quote"]
            == original["required_attribute"]["evidence_quote"]
        )
        # Actual context consumer sees different evidence for the otherwise same input.
        fit = company_fit_match(
            details={
                "dimension_evidence": {
                    "identity": {
                        "decision": "match",
                        "web_identity_receipt": {
                            "decision": "match",
                            "observed_name": "IBM",
                            "observed_domain": "ibm.com",
                            "observed_linkedin_slug": "ibm",
                            "evidence_source": "company_web_reverification",
                        },
                    }
                }
            }
        )
        url = original["required_attribute"]["evidence_url"]
        cache = {
            url: {
                "final_url": url,
                "text": "IBM. " + original["required_attribute"]["evidence_quote"],
            }
        }

        def contexts(c):
            model = CompanyOutput(
                **competition._normalized_company(c, integrity_policy=True)
            )
            return lead_scorer._matched_company_source_contexts(
                fit, None, None, cache, paragraph=model.intent_details, company=model
            )

        assert contexts(original) != contexts(changed)
        changed_run = next(
            r for r in pair if r["submission_id"] == parts[2]["submission_id"]
        )
        if field == "upgrade":
            # An old stored key/lease remains authoritative after the new projection loads.
            assert changed_run["status"] == "accepted"
            assert service._verified_breakdowns(
                changed_run, icp=icp, companies=[changed], policy=policy
            ) == [prior]
            assert (
                store.get_run(leased["run_id"])["judgment_scope_doc"]
                == source["judgment_scope_doc"]
            )
        else:
            assert changed_run["status"] == "pending"
            fresh, fresh_token, *_ = claim(store, round_id, runner, excluded=[runner])
            assert fresh["run_id"] == changed_run["run_id"]
            assert changed_run["judgment_cache_key"] != source["judgment_cache_key"]
            fresh_row = copy.deepcopy(prior)
            fresh_row["failure_reason"] = "Fresh reviewed changed evidence"
            fresh_row["verifier_gate_receipts"][0]["source_url"] = changed[
                "required_attribute"
            ]["evidence_url"]
            fresh_row["verifier_gate_receipts"][0]["source_quote"] = changed[
                "required_attribute"
            ]["evidence_quote"]
            fresh_evidence = judgment_cache.build_evidence_snapshot(
                output=scoring.build_scoring_output(
                    fresh["scored_run_id"], [fresh_row]
                ),
                cache_scope=changed_run["judgment_scope_doc"],
                source_score_run_id=fresh["run_id"],
                source_scored_run_id=fresh["scored_run_id"],
                source_output_ref="arena/local/fresh.json",
                source_runner_hotkey=runner,
                runner_authority_exclusions=fresh["runner_authority_exclusions"],
            )
            assert (
                store.complete_attempt(
                    run_id=fresh["run_id"],
                    lease_token_hash=hash_lease_token(fresh_token),
                    result={"terminal_status": "accepted"},
                    terminal_cause="accepted",
                    output_ref=fresh_evidence["source_output_ref"],
                    judgment_evidence=fresh_evidence,
                    judgment_evidence_hash=contracts.document_hash(fresh_evidence),
                )["status"]
                == "accepted"
            )
            assert service._verified_breakdowns(
                store.get_run(fresh["run_id"]),
                icp=icp,
                companies=[changed],
                policy=policy,
            ) == [fresh_row]
            assert (
                store.get_judgment_cache(source["judgment_cache_key"])["evidence_doc"]
                == evidence
            )
    finally:
        transport.close()
