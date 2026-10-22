"""Offline source attribution for frozen signal judgment reuse."""

from __future__ import annotations

import copy
import json

import pytest

from lab_arena import company_judgments, contracts, scoring, signal_judgments, verify
from lab_arena.training_export import ExportError, _signal_authority_evidence, assemble_example
from scripts.export_arena_training import export_run
from tests.lab_arena.training_export_test import (
    OUTPUT_REF, add_call, per_icp_cost_fixture, per_icp_cost_reader, save_input,
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


def _company_cache_fixture(*, whole_company_hit=False, signal_policy=True):
    """Actual company authority validation, including a transitive signal hit."""
    from tests.test_arena_score_integrity import _positive_breakdown

    values = _signal_fixture()
    execute, judge, signal_source = values["runs"]
    config = values["round_row"]["configuration_doc"]
    config["company_judgment_policy"] = company_judgments.POLICY
    policy = config["scorer_policy"]
    if not signal_policy:
        policy = {key: value for key, value in policy.items() if key != "signal_judgment_policy"}
        config["scorer_policy"] = policy
        values["signal_judgment_uses"] = []
        values["signal_judgment_rows"] = []
        values["signal_source_ledger_rows"] = []
    icp = json.loads(values["objects"].values["benchmark"])["icps"][0]
    companies = json.loads(values["objects"].values[OUTPUT_REF])["companies"]
    document = scoring.build_scoring_input(
        scored_run_id=execute["run_id"], icp=icp, companies=companies,
        policy=policy, evaluation_date=values["round_row"]["evaluation_date"],
    )
    ref = company_judgments.build_company_scopes(
        scoring_input=document, round_id=judge["round_id"], network_name="finney", netuid=71,
        scorer_image_digest=config["scorer_image_digest"],
        scorer_image_reference=config["scorer_image_reference"],
        integrity_policy="arena_integrity_v1", company_judgment_policy=company_judgments.POLICY,
    )[0]
    raw = _positive_breakdown("Acme", "acme.example")
    raw["company_judgment_policy"] = company_judgments.POLICY
    if signal_policy:
        signal = values["signal_judgment_rows"][0]
        raw["intent_signals_detail"] = [{
            **signal["judgment_doc"],
            "signal_judgment_reference": signal_judgments.reference(signal["scope_doc"], 0),
            "counted_in_aggregate": True,
        }]
    breakdowns = scoring.project_company_judgments(companies, [raw], company_quality=False)
    from qualification.scoring.competition import raw_company_judgment
    raw = {**raw_company_judgment(breakdowns[0]),
           "company_judgment_policy": company_judgments.POLICY}
    source = judge
    source_index = 0
    if whole_company_hit:
        source = copy.deepcopy(judge)
        source.update(run_id="company:source", scored_run_id="execute:company-source",
                      output_ref="company-source-output")
        values["runs"].append(source)
        source_index = 1
        # The current row is first; its immutable source used the second slot.
        source["company_judgment_refs"] = [ref, {**ref, "company_index": source_index}]
    source.update(runner_hotkey="validator-company", claim_request_id="company-claim")
    source["claim_response"] = {"runner_authority_exclusions": [source["runner_hotkey"]]}
    judge["company_judgment_refs"] = [ref]
    item = {key: ref[key] for key in ("company_index", "cache_key", "company_input_hash")}
    item["authority_slot"] = 0
    new = {**item, "company_index": source_index, "raw_judgment": raw}
    source_output = scoring.build_scoring_output(source["scored_run_id"], breakdowns,
                                                company_judgments=[new])
    values["objects"].put(source["output_ref"], contracts.canonical_json(source_output).encode())
    exclusions = [source["runner_hotkey"]]
    if signal_policy:
        exclusions = [signal_source["runner_hotkey"], source["runner_hotkey"]]
        values["signal_judgment_uses"][0].update(run_id=source["run_id"], company_index=source_index)
    evidence = company_judgments.build_evidence_snapshot(
        new_judgment=new, company_ref={**ref, "company_index": source_index},
        source_score_run_id=source["run_id"], source_scored_run_id=source["scored_run_id"],
        source_output_ref=source["output_ref"], source_output_hash=contracts.document_hash(source_output),
        source_runner_hotkey=source["runner_hotkey"], source_claim_request_id=source["claim_request_id"],
        source_claim_request_hash=source["claim_request_hash"], source_lease_generation=source["lease_generation"],
        source_completion_request_hash="sha256:" + "e" * 64,
        runner_authority_exclusions=exclusions,
    )
    values["company_judgment_rows"] = [{
        "cache_key": ref["cache_key"], "authority_slot": 0,
        "company_input_hash": ref["company_input_hash"], "scope_doc": ref["scope_doc"],
        "evidence_doc": evidence, "evidence_hash": contracts.document_hash(evidence),
        **{key: evidence[key] for key in (
            "source_score_run_id", "source_scored_run_id", "source_runner_hotkey")},
    }]
    lease = {"schema_version": company_judgments.LEASE_SCHEMA_VERSION,
             "hits": [], "misses": [item]}
    if whole_company_hit:
        lease.update(hits=[{**item, "evidence_doc": evidence,
                            "evidence_hash": contracts.document_hash(evidence)}], misses=[])
        values["objects"].put(judge["output_ref"], contracts.canonical_json(
            scoring.build_scoring_output(execute["run_id"], breakdowns, company_judgments=[])).encode())
    judge.setdefault("claim_response", {})["company_judgment_cache"] = lease
    judge["result_doc"]["company_judgment_consumed_indexes"] = [0]
    document["company_judgment_cache"] = lease
    save_input(values["objects"], judge["run_id"], "score", document)
    scored = verify.scored_row(
        execute["submission_id"], 0, execute["run_id"], icp, companies, breakdowns, policy,
    )
    record = scoring.run_scores_for_store(
        {"rows": [scored], "integrity_policy": "arena_integrity_v1"}, [execute],
    )[0]
    execute.update(per_icp_score=record["per_icp_score"], qualification_doc=record["qualification_doc"])
    return values


class _CompanyExportStore:
    def __init__(self, values):
        self.values = values

    def get_run(self, run_id):
        return next((row for row in self.values["runs"] if row["run_id"] == run_id), None)

    def get_round(self, round_id):
        row = self.values["round_row"]
        return row if row["round_id"] == round_id else None

    def get_submission(self, submission_id):
        row = self.values["submission_row"]
        return row if row["submission_id"] == submission_id else None

    def list_runs(self, round_id):
        return [row for row in self.values["runs"] if row["round_id"] == round_id]

    def list_ledger(self, *, run_id, after_entry_id=None, limit=1000):
        return [row for row in self.values["signal_source_ledger_rows"]
                if row["run_id"] == run_id and (after_entry_id is None or row["entry_id"] > after_entry_id)][:limit]

    def get_judgment_cache(self, key):
        return None

    def get_company_judgment(self, key, slot):
        return next((row for row in self.values["company_judgment_rows"]
                     if (row["cache_key"], row["authority_slot"]) == (key, slot)), None)

    def get_signal_judgment_uses(self, run_id):
        return [row for row in self.values["signal_judgment_uses"] if row["run_id"] == run_id]

    def get_signal_judgment(self, key, slot):
        return next((row for row in self.values["signal_judgment_rows"]
                     if (row["cache_key"], row["authority_slot"]) == (key, slot)), None)

    def icp_cost_eligibility(self, **identity):
        return per_icp_cost_reader(spend=0, reason="eligible")(**identity)["eligibility"]


@pytest.mark.parametrize("whole_company_hit", [False, True])
def test_export_entrypoint_validates_company_and_transitive_signal_authority(whole_company_hit):
    values = _company_cache_fixture(whole_company_hit=whole_company_hit)
    result = export_run(_CompanyExportStore(values), values["objects"], values["run_id"])
    evidence = result["judgment"]["company_authority_evidence"][0]["evidence_doc"]
    assert evidence["runner_authority_exclusions"] == ["validator-source", "validator-company"]
    assert result["judgment"]["per_icp_score"] == values["runs"][0]["per_icp_score"]
    signal = result["judgment"]["signal_authority_evidence"][0]
    assert signal["company_index"] == 0
    assert signal["use"]["company_index"] == int(whole_company_hit)
    assert signal["authority"]["source_score_run_id"] == "judge:source"
    assert result["competition_cost_eligibility"]["status"] == "available"


@pytest.mark.parametrize("fault", [
    "missing_source", "wrong_round", "source_claim", "source_output",
    "missing_signal", "signal_exclusions", "source_exclusions",
    "use_conflict", "duplicate_use", "missing_use",
])
def test_export_entrypoint_keeps_company_signal_authority_fail_closed(fault):
    values = _company_cache_fixture(whole_company_hit=True)
    source = values["runs"][-1]
    if fault == "missing_source":
        values["runs"].remove(source)
    elif fault == "wrong_round":
        source["round_id"] = "other-round"
    elif fault == "source_claim":
        source["claim_request_hash"] = "sha256:" + "f" * 64
    elif fault == "source_output":
        values["objects"].put(source["output_ref"], b"{}")
    elif fault == "missing_signal":
        values["signal_judgment_rows"] = []
    elif fault == "signal_exclusions":
        values["signal_judgment_rows"][0]["runner_authority_exclusions"] = ["forged-validator"]
    elif fault == "source_exclusions":
        source["claim_response"]["runner_authority_exclusions"] = ["forged-validator"]
    elif fault == "use_conflict":
        values["signal_judgment_uses"][0]["source_score_run_id"] = "forged-source"
    elif fault == "duplicate_use":
        values["signal_judgment_uses"].append(copy.deepcopy(values["signal_judgment_uses"][0]))
    else:
        values["signal_judgment_uses"] = []
    result = export_run(_CompanyExportStore(values), values["objects"], values["run_id"])
    assert "judgment" not in result
    assert "judge_evidence_unverified" in result["completeness"]["incomplete_reasons"]


@pytest.mark.parametrize("whole_company_hit", [False, True])
def test_export_entrypoint_validates_historical_company_authority_without_signal_policy(whole_company_hit):
    values = _company_cache_fixture(whole_company_hit=whole_company_hit, signal_policy=False)
    result = export_run(_CompanyExportStore(values), values["objects"], values["run_id"])
    assert result["judgment"]["company_authority_evidence"]
    assert "signal_authority_evidence" not in result["judgment"]


def _add_frozen_catalog(values):
    from lab_arena.training_input import read_input
    from tests.lab_arena.test_deepline_catalog import frozen

    catalog = frozen()
    values["round_row"]["configuration_doc"]["deepline_catalog"] = catalog
    judge = values["runs"][1]
    document = read_input(values["objects"], judge["run_id"], "score",
                          judge["lease_generation"], judge["claim_request_hash"])["input"]
    document["deepline_catalog"] = copy.deepcopy(catalog)
    return judge, document


@pytest.mark.parametrize("whole_company_hit", [False, True])
def test_export_entrypoint_accepts_runner_injected_frozen_catalog(whole_company_hit):
    values = _company_cache_fixture(whole_company_hit=whole_company_hit)
    judge, document = _add_frozen_catalog(values)
    save_input(values["objects"], judge["run_id"], "score", document)
    result = export_run(_CompanyExportStore(values), values["objects"], values["run_id"])
    assert result["judgment"]["per_icp_score"] == values["runs"][0]["per_icp_score"]
    assert len(result["judgment"]["signal_authority_evidence"]) == 1
    assert result["competition_cost_eligibility"]["status"] == "available"


@pytest.mark.parametrize("fault", [
    "changed_catalog", "catalog_json_type", "missing_catalog", "unfrozen_catalog",
    "null_catalog", "unknown_field", "changed_effective_input",
])
def test_export_entrypoint_rejects_changed_catalog_or_unrecognized_score_input(fault):
    values = _company_cache_fixture(whole_company_hit=True)
    judge, document = _add_frozen_catalog(values)
    if fault == "changed_catalog":
        document["deepline_catalog"]["tools"][0]["description"] = "changed"
    elif fault == "catalog_json_type":
        # Python equality treats False and 0 as equal; exact JSON does not.
        document["deepline_catalog"]["allow_people"] = 0
    elif fault == "missing_catalog":
        del document["deepline_catalog"]
    elif fault == "unfrozen_catalog":
        del values["round_row"]["configuration_doc"]["deepline_catalog"]
    elif fault == "null_catalog":
        document["deepline_catalog"] = None
    elif fault == "unknown_field":
        document["unrecognized_context"] = {}
    else:
        document["companies"][0]["intent_signals"][0]["snippet"] = "changed effective evidence"
    save_input(values["objects"], judge["run_id"], "score", document)
    result = export_run(_CompanyExportStore(values), values["objects"], values["run_id"])
    assert "judgment" not in result
    assert "signal_authority_unverified" in result["completeness"]["incomplete_reasons"]


def _mixed_fit_fixture(*, signal_policy=True):
    """One positive company and one terminal fit failure with null details."""
    from qualification.scoring.competition import raw_company_judgment
    from tests.test_arena_score_integrity import _positive_breakdown

    values = _company_cache_fixture(signal_policy=signal_policy)
    execute, judge = values["runs"][:2]
    config = values["round_row"]["configuration_doc"]
    output = json.loads(values["objects"].values[OUTPUT_REF])
    failed_company = copy.deepcopy(output["companies"][0])
    failed_company.update(company_name="Other", company_website="https://other.example")
    output["companies"].append(failed_company)
    values["objects"].put(OUTPUT_REF, contracts.canonical_json(output).encode())
    icp = json.loads(values["objects"].values["benchmark"])["icps"][0]
    document = scoring.build_scoring_input(
        scored_run_id=execute["run_id"], icp=icp, companies=output["companies"],
        policy=config["scorer_policy"], evaluation_date=values["round_row"]["evaluation_date"],
    )
    refs = company_judgments.build_company_scopes(
        scoring_input=document, round_id=judge["round_id"], network_name="finney", netuid=71,
        scorer_image_digest=config["scorer_image_digest"],
        scorer_image_reference=config["scorer_image_reference"],
        integrity_policy="arena_integrity_v1", company_judgment_policy=company_judgments.POLICY,
    )
    raw = values["company_judgment_rows"][0]["evidence_doc"]["raw_judgment"]
    failed = _positive_breakdown("Other", "other.example")
    failed.update(final_score=0.0, intent_signal_raw=0.0, intent_signal_final=0.0,
                  intent_signals_detail=None, failure_reason="company_fit_mismatch",
                  company_judgment_policy=company_judgments.POLICY)
    failed["verifier_gate_receipts"][0]["decision"] = "mismatch"
    breakdowns = scoring.project_company_judgments(output["companies"], [raw, failed], company_quality=False)
    failed = {**raw_company_judgment(breakdowns[1]), "company_judgment_policy": company_judgments.POLICY}
    misses = [{**{key: ref[key] for key in ("company_index", "cache_key", "company_input_hash")},
               "authority_slot": 0} for ref in refs]
    new = [{**item, "raw_judgment": value} for item, value in zip(misses, [raw, failed])]
    score_output = scoring.build_scoring_output(execute["run_id"], breakdowns, company_judgments=new)
    values["objects"].put(judge["output_ref"], contracts.canonical_json(score_output).encode())
    first_evidence = values["company_judgment_rows"][0]["evidence_doc"]
    provenance = {key: first_evidence[key] for key in first_evidence if key.startswith("source_")}
    provenance["source_output_hash"] = contracts.document_hash(score_output)
    values["company_judgment_rows"] = []
    for ref, judgment in zip(refs, new):
        evidence = company_judgments.build_evidence_snapshot(
            new_judgment=judgment, company_ref=ref, **provenance,
            runner_authority_exclusions=first_evidence["runner_authority_exclusions"],
        )
        values["company_judgment_rows"].append({
            "cache_key": ref["cache_key"], "company_input_hash": ref["company_input_hash"],
            "authority_slot": 0, "scope_doc": ref["scope_doc"],
            "evidence_doc": evidence, "evidence_hash": contracts.document_hash(evidence),
            **{key: evidence[key] for key in (
                "source_score_run_id", "source_scored_run_id", "source_runner_hotkey")},
        })
    judge["company_judgment_refs"] = refs
    lease = {"schema_version": company_judgments.LEASE_SCHEMA_VERSION, "hits": [], "misses": misses}
    judge["claim_response"]["company_judgment_cache"] = lease
    judge["result_doc"]["company_judgment_consumed_indexes"] = [0, 1]
    document["company_judgment_cache"] = lease
    save_input(values["objects"], judge["run_id"], "score", document)
    scored = verify.scored_row(execute["submission_id"], 0, execute["run_id"], icp,
                               output["companies"], breakdowns, config["scorer_policy"])
    record = scoring.run_scores_for_store(
        {"rows": [scored], "integrity_policy": "arena_integrity_v1"}, [execute],
    )[0]
    execute.update(per_icp_score=record["per_icp_score"], qualification_doc=record["qualification_doc"])
    return values


def test_export_entrypoint_keeps_positive_and_null_fit_failure_signal_labels():
    values = _mixed_fit_fixture()
    judge, document = _add_frozen_catalog(values)
    save_input(values["objects"], judge["run_id"], "score", document)
    result = export_run(_CompanyExportStore(values), values["objects"], values["run_id"])
    assert len(result["judgment"]["signal_authority_evidence"]) == 1
    assert result["companies"][0]["judgment"]["company_qualified"] is True
    assert result["companies"][1]["judgment"]["company_qualified"] is False
    assert result["companies"][1]["judgment"]["intent_signals_detail"] is None
    assert result["competition_cost_eligibility"]["status"] == "available"


def test_historical_export_keeps_positive_and_null_fit_failure_without_signal_policy():
    values = _mixed_fit_fixture(signal_policy=False)
    result = export_run(_CompanyExportStore(values), values["objects"], values["run_id"])
    assert result["companies"][1]["judgment"]["intent_signals_detail"] is None
    assert "signal_authority_evidence" not in result["judgment"]


def test_export_entrypoint_keeps_unused_signal_use_after_terminal_fit_failure():
    values = _mixed_fit_fixture()
    values["signal_judgment_uses"].append({**values["signal_judgment_uses"][0], "company_index": 1})
    result = export_run(_CompanyExportStore(values), values["objects"], values["run_id"])
    assert len(result["companies"]) == 2
    assert len(result["judgment"]["signal_authority_evidence"]) == 1
    assert result["judgment"]["signal_judgment_uses"] == values["signal_judgment_uses"]
    assert result["companies"][1]["judgment"]["intent_signals_detail"] is None


def test_current_judge_use_survives_terminal_unavailable_without_signal_verdict():
    values = _mixed_fit_fixture()
    values["signal_judgment_uses"].append({**values["signal_judgment_uses"][0], "company_index": 1})
    judge = values["runs"][1]
    breakdowns = json.loads(values["objects"].values[judge["output_ref"]])["breakdowns"]
    breakdowns[1]["verifier_gate_receipts"][0]["decision"] = "unavailable"
    from lab_arena.training_input import read_input
    scoring_input = read_input(values["objects"], judge["run_id"], "score",
                               judge["lease_generation"], judge["claim_request_hash"])["input"]
    evidence = _signal_authority_evidence(
        judge=judge, breakdowns=breakdowns, scored_company_indexes=[0, 1],
        scoring_input=scoring_input, uses=values["signal_judgment_uses"],
        authority_rows=values["signal_judgment_rows"], company_rows=values["company_judgment_rows"],
        runs={row["run_id"]: row for row in values["runs"]}, round_row=values["round_row"],
    )
    assert len(evidence) == 1
    assert evidence[0]["company_index"] == 0


@pytest.mark.parametrize("decision", ["mismatch", "unavailable"])
def test_cached_source_use_survives_terminal_fit_failure(decision):
    values = _company_cache_fixture(whole_company_hit=True)
    judge = values["runs"][1]
    breakdowns = json.loads(values["objects"].values[judge["output_ref"]])["breakdowns"]
    breakdowns[0]["company_qualified"] = False
    breakdowns[0]["intent_signals_detail"] = None
    breakdowns[0]["verifier_gate_receipts"][0]["decision"] = decision
    from lab_arena.training_input import read_input
    scoring_input = read_input(values["objects"], judge["run_id"], "score",
                               judge["lease_generation"], judge["claim_request_hash"])["input"]
    evidence = _signal_authority_evidence(
        judge=judge, breakdowns=breakdowns, scored_company_indexes=[0],
        scoring_input=scoring_input, uses=values["signal_judgment_uses"],
        authority_rows=values["signal_judgment_rows"], company_rows=values["company_judgment_rows"],
        runs={row["run_id"]: row for row in values["runs"]}, round_row=values["round_row"],
    )
    assert evidence == []


@pytest.mark.parametrize("signal_policy", [False, True])
@pytest.mark.parametrize("details", [{}, "", 0, False])
def test_signal_authority_rejects_malformed_non_list_details(details, signal_policy):
    values = _mixed_fit_fixture(signal_policy=signal_policy)
    judge = values["runs"][1]
    score_output = json.loads(values["objects"].values[judge["output_ref"]])
    score_output["breakdowns"][1]["intent_signals_detail"] = details
    from lab_arena.training_input import read_input
    scoring_input = read_input(values["objects"], judge["run_id"], "score",
                               judge["lease_generation"], judge["claim_request_hash"])["input"]
    with pytest.raises(ExportError, match="signal_details_invalid"):
        _signal_authority_evidence(
            judge=judge, breakdowns=score_output["breakdowns"], scored_company_indexes=[0, 1],
            scoring_input=scoring_input, uses=values["signal_judgment_uses"],
            authority_rows=values["signal_judgment_rows"], company_rows=values["company_judgment_rows"],
            runs={row["run_id"]: row for row in values["runs"]}, round_row=values["round_row"],
        )


@pytest.mark.parametrize("details", [None, []])
def test_empty_company_hit_details_reject_inherited_signal_uses(details):
    values = _company_cache_fixture(whole_company_hit=True)
    judge = values["runs"][1]
    breakdowns = json.loads(values["objects"].values[judge["output_ref"]])["breakdowns"]
    breakdowns[0]["intent_signals_detail"] = details
    from lab_arena.training_input import read_input
    scoring_input = read_input(values["objects"], judge["run_id"], "score",
                               judge["lease_generation"], judge["claim_request_hash"])["input"]
    with pytest.raises(ExportError, match="signal_use_without_reference"):
        _signal_authority_evidence(
            judge=judge, breakdowns=breakdowns, scored_company_indexes=[0],
            scoring_input=scoring_input, uses=values["signal_judgment_uses"],
            authority_rows=values["signal_judgment_rows"], company_rows=values["company_judgment_rows"],
            runs={row["run_id"]: row for row in values["runs"]}, round_row=values["round_row"],
        )
