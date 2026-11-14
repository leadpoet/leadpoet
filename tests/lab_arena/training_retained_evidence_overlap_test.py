"""Readiness detects retained evidence reuse without changing dataset splits."""
import copy

import pytest

from lab_arena import training_dataset as dataset
from tests.lab_arena.training_dataset_test import claim, example, row, seal


QUOTE = {
    "content_kind": "verifier_source_quote", "dimension": "industry",
    "url": "https://source.example/manufacturers",
    "text": "These manufacturers sell industrial hardware, not the required software service.",
    "provenance": "company_fit_receipt",
}
FACT = {
    "content_kind": "structured_provider_fact", "dimension": "employee_size",
    "url": "https://www.linkedin.com/company/synthetic-manufacturer",
    "website": "https://synthetic-manufacturer.example",
    "employee_count": "51-200", "provider": "harvestapi_get_company",
    "source_field": "employeeCountRange", "provenance": "company_fit_receipt",
}


def records(first, second):
    """Already-derived records with independent company, ICP and episode keys."""
    result = []
    for index, evidence in enumerate((first, second)):
        result.append({
            "meta": {"round_id": "arena-2026-10-18", "submission_id": "source-%s" % index,
                     "stage": 1, "icp_position": 0},
            "group_keys": {"company_cluster": "company-%s" % index,
                           "icp_fingerprint": "icp-%s" % index,
                           "source_family": "family-%s" % index},
            "input": {"evaluation_date": "2026-10-18", "source_evidence": copy.deepcopy(evidence)},
        })
    return result


def checks(first, second):
    first, second = records(first, second)
    return dataset.evaluation_checks({"train": [first], "eval": [second], "analysis": []}, dataset.Rules())


def corpus():
    result = []
    for index in range(24):
        company = claim("Synthetic Manufacturer %s" % index,
                        "synthetic-%s.invalid" % index, "synthetic-%s" % index)
        company_row = row(0, company, "wrong_fit")
        company_row["judgment"]["intent_signals_detail"] = None
        receipt = company_row["judgment"]["verifier_gate_receipts"][0]
        for evidence in receipt["dimension_evidence"].values():
            evidence["web_evidence"] = {"url": QUOTE["url"], "quote": QUOTE["text"]}
        document = example("arena-2026-10-18:synthetic-%s:1:0:1" % index,
                           [company_row], submission_id="synthetic-%s" % index)
        document["icp"]["industry"] = "Synthetic required industry %s" % index
        result.append(seal(document))
    return result


def test_builder_reports_substantive_source_reuse_without_changing_records_or_assignment(monkeypatch):
    examples = corpus()
    original = copy.deepcopy(examples)
    splits, manifest = dataset.build(examples)
    with monkeypatch.context() as patched:
        patched.setattr(dataset, "evaluation_checks", lambda splits, rules: {})
        unchecked_splits, unchecked_manifest = dataset.build(examples)
    assert splits == unchecked_splits
    assert {k: v for k, v in manifest.items() if k != "evaluation_checks"} == {
        k: v for k, v in unchecked_manifest.items() if k != "evaluation_checks"}
    assert examples == original
    assert splits["train"] and splits["eval"]
    evaluation = manifest["evaluation_checks"]
    assert evaluation["train_eval_group_overlap_counts"]["company_cluster"] == 0
    assert evaluation["train_eval_group_overlap_counts"]["icp_fingerprint"] == 0
    assert evaluation["train_eval_group_overlap_counts"]["episode"] == 0
    assert evaluation["split_viable"] is False
    assert evaluation["train_eval_retained_evidence_overlap_counts"]["text_source"] == 1
    assert "retained_evidence_overlap" in evaluation["insufficient_split_reasons"]


@pytest.mark.parametrize("kind", ["verifier_source_quote", "retained_page_excerpt"])
def test_text_identity_ignores_label_and_incidental_metadata(kind):
    second = {**QUOTE, "content_kind": kind, "dimension": "geography",
              "matched_icp_signal": 0, "retrieved_at": "2026-10-19T12:00:00Z",
              "source_publication_date": "2026-10-18", "provenance": "other"}
    first_record, second_record = records([QUOTE], [second])
    first_record["label"] = {"class": "wrong_fit", "sub_reason": "industry"}
    second_record["label"] = {"class": "qualified", "sub_reason": None}
    evaluation = dataset.evaluation_checks(
        {"train": [first_record], "eval": [second_record], "analysis": []}, dataset.Rules())
    assert evaluation["split_viable"] is False
    assert evaluation["train_eval_retained_evidence_overlap_counts"] == {
        "text_source": 1, "provider_fact_source": 0}


def test_provider_fact_identity_requires_the_source_and_actual_value():
    evaluation = checks([FACT], [{**FACT, "dimension": "industry", "provenance": "other"}])
    assert evaluation["split_viable"] is False
    assert evaluation["train_eval_retained_evidence_overlap_counts"]["provider_fact_source"] == 1
    for change in ({"employee_count": "201-500"}, {"url": "https://www.linkedin.com/company/other"},
                   {"website": "https://other.example"}, {"provider": "other_provider"},
                   {"source_field": "other_field"}):
        assert checks([FACT], [{**FACT, **change}])["split_viable"] is True


@pytest.mark.parametrize("first,second", [
    ([QUOTE], [{**QUOTE, "text": "An independently retained different source passage."}]),
    ([QUOTE], [{**QUOTE, "url": "https://independent.example/manufacturers"}]),
    ([{"url": QUOTE["url"], "retrieved_at": "2026-10-18", "provenance": "shared"}],
     [{"url": QUOTE["url"], "retrieved_at": "2026-10-18", "provenance": "shared"}]),
    ([{**QUOTE, "text": "  "}], [{**QUOTE, "text": "  "}]),
    ([{**QUOTE, "text": ""}], [{**QUOTE, "text": ""}]),
    ([{"content_kind": "company_identity_observation", "observed_name": "Shared name"}],
     [{"content_kind": "company_identity_observation", "observed_name": "Shared name"}]),
    ([{**FACT, "employee_count": ""}], [{**FACT, "employee_count": ""}]),
])
def test_shared_trivial_metadata_or_independent_evidence_does_not_block(first, second):
    evaluation = checks(first, second)
    assert evaluation["split_viable"] is True
    assert not any(evaluation["train_eval_retained_evidence_overlap_counts"].values())


def test_analysis_evidence_is_not_a_train_eval_overlap():
    first, second = records([QUOTE], [{**QUOTE, "url": "https://independent.example/source"}])
    analysis = copy.deepcopy(first)
    evaluation = dataset.evaluation_checks(
        {"train": [first], "eval": [second], "analysis": [analysis]}, dataset.Rules())
    assert evaluation["split_viable"] is True


def test_temporal_split_still_reports_reused_evidence_and_keeps_dates():
    first, second = records([QUOTE], [QUOTE])
    first["input"]["evaluation_date"] = "2026-10-17"
    second["input"]["evaluation_date"] = "2026-10-19"
    evaluation = dataset.evaluation_checks(
        {"train": [first], "eval": [second], "analysis": []},
        dataset.Rules(eval_after="2026-10-18"))
    assert evaluation["date_ranges"]["train"]["max"] == "2026-10-17"
    assert evaluation["date_ranges"]["eval"]["min"] == "2026-10-19"
    assert evaluation["split_viable"] is False
    second["input"]["source_evidence"] = [{**QUOTE, "url": "https://independent.example/source"}]
    independent = dataset.evaluation_checks(
        {"train": [first], "eval": [second], "analysis": []}, dataset.Rules(eval_after="2026-10-18"))
    assert independent["split_viable"] is True
