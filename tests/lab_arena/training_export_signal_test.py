"""Offline source attribution for frozen signal judgment reuse."""

from __future__ import annotations

import copy
import json

import pytest

from lab_arena import company_judgments, contracts, scoring, signal_judgments, verify
from lab_arena.training_export import ExportError, _signal_authority_evidence, assemble_example
from scripts.export_arena_training import export_run
from tests.lab_arena.training_export_test import (
    OUTPUT_REF, add_call, per_icp_cost_fixture, save_input,
)


def _signal_fixture():
    values = per_icp_cost_fixture()
    execute, judge = values["runs"]
    config = values["round_row"]["configuration_doc"]
    policy = scoring.build_scorer_policy(
        scoring_adapter_version="qualification_integrity_v2",
        company_judgment_cache=True,
    )
    policy = contracts.validate_scorer_policy({**policy, "signal_judgment_policy": signal_judgments.POLICY})
    config["scorer_policy"] = policy
    values["round_row"]["arena_netuid"] = 71
    icp = json.loads(values["objects"].values["benchmark"])["icps"][0]
    companies = json.loads(values["objects"].values[OUTPUT_REF])["companies"]
    scoring_input = scoring.build_scoring_input(
        scored_run_id=execute["run_id"], icp=icp, companies=companies,
        policy=policy, evaluation_date=values["round_row"]["evaluation_date"],
    )
    save_input(values["objects"], judge["run_id"], "score", scoring_input)
    identity = {"decision": "match", "observed_name": "Acme",
                "observed_domain": "acme.example"}
    scope = signal_judgments.build_scope(
        scoring_input=scoring_input, company_index=0, signal_index=0,
        verified_company_identity=identity, verified_funding_stage_context=[],
        round_id=execute["round_id"], network_name="finney", netuid=71,
        scorer_image_digest=config["scorer_image_digest"],
        scorer_image_reference=config["scorer_image_reference"],
    )
    url = "https://acme.example/launch"
    quote = "Acme announces a new product launch for its customers."
    judgment = {
        "raw": 54.0, "after_decay": 54.0, "decay": 1.0,
        "confidence": 90, "date_status": "in_window", "date_verdict": "in_window",
        "claim_support_verdict": "supported", "matched_icp_signal": 0,
        "evidence_type": scope["expected_evidence_type"], "evidence_urls": [url],
        "judge_verdict": {
            "decision": "verified", "pipeline_decision": "approve",
            "claim_support_verdict": "supported", "client_ready": True,
            "verification_trace": {
                "evidence_url": url, "evidence_urls": [url],
                "evidence_source": "company_website", "extracted_signal_date": None,
                "verified_source_context": [{"url": url, "text": quote}],
                "intent_verdict": {"signal_evaluations": [{
                    "signal_status": "supported", "same_entity_check": "pass",
                    "verification_mode": "source_grounded", "supporting_quotes": [quote],
                    "evidence_urls_used": [url],
                }]},
                "final_disposition": "approve",
            },
        },
    }
    signal_judgments.validate_judgment(judgment, scope)
    reference = signal_judgments.reference(scope, 0)
    breakdown = {
        "company_index": 0, "company_identity_key": "company-acme",
        "company_identity_alias_keys": ["company-acme"],
        "company_qualified": True, "duplicate_company": False,
        "final_score": 51.0,
        "intent_signals_detail": [{**judgment, "signal_judgment_reference": reference,
                                   "counted_in_aggregate": True}],
    }
    scored = verify.scored_row(execute["submission_id"], 0, execute["run_id"],
                               icp, companies, [breakdown], policy)
    execute["per_icp_score"] = scored["per_icp_score"]
    values["objects"].put(judge["output_ref"], contracts.canonical_json(
        scoring.build_scoring_output(execute["run_id"], [breakdown])).encode())
    source = copy.deepcopy(judge)
    source.update(run_id="judge:source", scored_run_id="execute:source",
                  runner_hotkey="validator-source")
    values["runs"].append(source)
    judgment_hash = contracts.document_hash(judgment)
    values["signal_judgment_uses"] = [{
        "run_id": judge["run_id"], "company_index": 0, "signal_index": 0,
        "cache_key": scope["cache_key"], "authority_slot": 0,
        "signal_input_hash": scope["signal_input_hash"],
        "source_score_run_id": source["run_id"], "judgment_hash": judgment_hash,
    }]
    values["signal_judgment_rows"] = [{
        "cache_key": scope["cache_key"], "authority_slot": 0,
        "signal_input_hash": scope["signal_input_hash"],
        "scope_doc": scope,
        "judgment_hash": judgment_hash, "judgment_doc": judgment,
        "source_score_run_id": source["run_id"],
        "source_scored_run_id": source["scored_run_id"],
        "source_runner_hotkey": source["runner_hotkey"],
        "source_claim_request_hash": source["claim_request_hash"],
        "source_lease_generation": source["lease_generation"],
        "runner_authority_exclusions": [source["runner_hotkey"]],
    }]
    add_call(values)
    values["signal_source_ledger_rows"] = [
        {**row, "run_id": source["run_id"], "call_identity": "source-" + row["call_identity"]}
        for row in values["ledger_rows"]
    ]
    values["ledger_rows"] = []
    execute["result_doc"]["resource_summary"]["provider_call_count"] = 0
    source["result_doc"]["resource_summary"]["provider_call_count"] = 1
    return values


def test_signal_hit_exports_original_judgment_and_source_provider_cost_separately():
    values = _signal_fixture()
    result = assemble_example(**values)
    evidence = result["judgment"]["signal_authority_evidence"]
    assert len(evidence) == 1
    assert evidence[0]["authority"]["judgment_doc"] == values["signal_judgment_rows"][0]["judgment_doc"]
    assert evidence[0]["authority"]["source_score_run_id"] == "judge:source"
    assert result["judgment"]["signal_source_provider_calls"][0]["run_id"] == "judge:source"
    assert result["judgment"]["signal_source_cost_attribution"]["confirmed_actual_microusd"] == 42
    assert result["costs"]["confirmed_actual_microusd"] == 0


def test_signal_export_rejects_changed_frozen_input_and_missing_source_authority():
    values = _signal_fixture()
    values["signal_judgment_rows"][0]["scope_doc"]["verified_company_identity"]["observed_name"] = "Other"
    invalid = assemble_example(**values)
    assert "judgment" not in invalid
    assert "signal_authority_unverified" in invalid["completeness"]["incomplete_reasons"]
    values = _signal_fixture()
    values["signal_judgment_rows"] = []
    invalid = assemble_example(**values)
    assert "signal_authority_unverified" in invalid["completeness"]["incomplete_reasons"]


def test_no_signal_policy_keeps_historical_export_shape():
    values = per_icp_cost_fixture()
    result = assemble_example(**values)
    assert "signal_authority_evidence" not in result["judgment"]
    assert "signal_source_provider_calls" not in result["judgment"]


def test_failed_or_inflight_source_signal_still_has_offline_authority():
    for status in ("failed", "leased"):
        values = _signal_fixture()
        values["runs"][-1]["status"] = status
        values["runs"][-1]["terminal_cause"] = "judge_error" if status == "failed" else None
        result = assemble_example(**values)
        assert result["judgment"]["signal_authority_evidence"][0]["authority"]["source_score_run_id"] == "judge:source"


def test_whole_company_cache_hit_resolves_original_signal_use_after_index_change():
    values = _signal_fixture()
    execute, judge, signal_source = values["runs"]
    from lab_arena.training_input import read_input
    scoring_input = read_input(values["objects"], judge["run_id"], "score",
                               judge["lease_generation"], judge["claim_request_hash"])["input"]
    ordered_input = {key: scoring_input[key] for key in (
        "schema_version", "scored_run_id", "icp", "companies",
        "scorer_policy", "evaluation_date")}
    company_ref = company_judgments.build_company_scopes(
        scoring_input=ordered_input, round_id=judge["round_id"],
        network_name="finney", netuid=71,
        scorer_image_digest=values["round_row"]["configuration_doc"]["scorer_image_digest"],
        scorer_image_reference=values["round_row"]["configuration_doc"]["scorer_image_reference"],
        integrity_policy="arena_integrity_v1", company_judgment_policy="company_judgment_v1",
    )[0]
    judge["company_judgment_refs"] = [company_ref]
    judge["claim_response"] = {"company_judgment_cache": {"hits": [{
        "company_index": 0, "cache_key": company_ref["cache_key"],
        "company_input_hash": company_ref["company_input_hash"],
        "authority_slot": 0,
    }]}}
    source_ref = {**company_ref, "company_index": 1}
    # Two submitted rows have the same company cache key. Only the second
    # source position actually ran this signal and has a persisted use.
    signal_source["company_judgment_refs"] = [company_ref, source_ref]
    company_row = {"cache_key": company_ref["cache_key"], "authority_slot": 0,
                   "source_score_run_id": signal_source["run_id"]}
    use = {**values["signal_judgment_uses"][0], "run_id": signal_source["run_id"],
           "company_index": 1}
    detail = {**values["signal_judgment_rows"][0]["judgment_doc"],
              "signal_judgment_reference": signal_judgments.reference(
                  values["signal_judgment_rows"][0]["scope_doc"], 0),
              "counted_in_aggregate": True}
    evidence = _signal_authority_evidence(
        judge=judge, breakdowns=[{"intent_signals_detail": [detail]}],
        scored_company_indexes=[0], scoring_input=scoring_input,
        uses=[use], authority_rows=values["signal_judgment_rows"],
        company_rows=[company_row],
        runs={row["run_id"]: row for row in values["runs"]},
        round_row=values["round_row"],
    )
    assert evidence[0]["use"]["company_index"] == 1
    assert evidence[0]["company_index"] == 0
    with pytest.raises(ExportError, match="signal_reference_without_use"):
        _signal_authority_evidence(
            judge=judge, breakdowns=[{"intent_signals_detail": [detail]}],
            scored_company_indexes=[0], scoring_input=scoring_input,
            uses=[use, {**use, "company_index": 0}],
            authority_rows=values["signal_judgment_rows"],
            company_rows=[company_row],
            runs={row["run_id"]: row for row in values["runs"]},
            round_row=values["round_row"],
        )


def test_export_entrypoint_fetches_original_signal_ledger_without_current_charge():
    values = _signal_fixture()

    class Store:
        def get_run(self, run_id):
            return next((row for row in values["runs"] if row["run_id"] == run_id), None)

        def get_round(self, round_id):
            return values["round_row"] if round_id == values["round_row"]["round_id"] else None

        def get_submission(self, submission_id):
            return values["submission_row"] if submission_id == values["submission_row"]["submission_id"] else None

        def list_runs(self, round_id):
            return values["runs"] if round_id == values["round_row"]["round_id"] else []

        def list_ledger(self, *, run_id, after_entry_id=None, limit=1000):
            rows = values["signal_source_ledger_rows"] if run_id == "judge:source" else []
            return [row for row in rows if after_entry_id is None or row["entry_id"] > after_entry_id]

        def get_judgment_cache(self, key):
            return None

        def get_signal_judgment_uses(self, run_id):
            return [row for row in values["signal_judgment_uses"] if row["run_id"] == run_id]

        def get_signal_judgment(self, key, slot):
            return next((row for row in values["signal_judgment_rows"]
                         if (row["cache_key"], row["authority_slot"]) == (key, slot)), None)

        def icp_cost_eligibility(self, **_identity):
            return {}

    result = export_run(Store(), values["objects"], values["run_id"])
    assert result["judgment"]["signal_source_provider_calls"][0]["run_id"] == "judge:source"
    assert result["judgment"]["signal_source_provider_call_refs"][0]["location"] == "signal_source_provider_calls"
    assert result["costs"]["confirmed_actual_microusd"] == 0


def test_original_signal_from_current_score_links_existing_charge_once():
    values = _signal_fixture()
    judge = values["runs"][1]
    row = values["signal_judgment_rows"][0]
    row.update(source_score_run_id=judge["run_id"],
               source_scored_run_id=judge["scored_run_id"],
               source_runner_hotkey=judge.get("runner_hotkey"))
    values["signal_judgment_uses"][0]["source_score_run_id"] = judge["run_id"]
    values["ledger_rows"] = [
        {**entry, "run_id": judge["run_id"],
         "call_identity": "source-" + entry["call_identity"]}
        for entry in values["signal_source_ledger_rows"]
    ]
    values["signal_source_ledger_rows"] = []
    judge["result_doc"]["resource_summary"]["provider_call_count"] = 1
    result = assemble_example(**values)
    assert result["costs"]["confirmed_actual_microusd"] == 42
    assert result["judgment"]["signal_source_provider_calls"] == []
    assert result["judgment"]["signal_source_provider_call_refs"][0]["location"] == "provider_calls"
    assert result["judgment"]["signal_source_cost_attribution"]["confirmed_actual_microusd"] == 42
