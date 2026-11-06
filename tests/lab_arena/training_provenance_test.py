"""Training context retains recorded authority without changing observed targets."""

from copy import deepcopy
import json

from lab_arena import training_export, training_reader, training_steps
from tests.lab_arena.training_export_test import fixture as export_fixture
from tests.lab_arena.training_steps_test import fixture, records, seal
from tests.lab_arena.training_thinking_test import full_example, rows as thinking_rows
from tests.lab_arena.training_export_forward_only_test import preserved_fixture, PreservedStore


def annotated(example):
    example.update(evaluation_date="2026-10-20", round_dates={
        "created_at": "2026-10-19T12:00:00+00:00", "published_at": "2026-10-21T01:02:03Z"},
        source={"ref": "private/source.tar.gz", "sha256": "sha256:" + "a" * 64,
                "source_correction": {"request_id": "correction", "request_hash": "sha256:" + "b" * 64,
                                      "commit": "c" * 40}})
    example["companies"] = [
        {"company_index": 7, "label_status": "scored", "judgment": {"company_qualified": True}},
        {"company_index": 2, "label_status": "scored", "judgment": {
            "company_qualified": False, "verifier_gate_receipts": [{"gate": "company_fit", "decision": "mismatch"}]}},
        {"company_index": 4, "label_status": "scored", "judgment": {
            "company_qualified": False, "verifier_gate_receipts": [{"gate": "company_fit", "decision": "unavailable",
                "failure_class": "insufficient_fit_evidence"}]}},
        {"company_index": 5, "label_status": "scored", "judgment": {
            "company_qualified": False, "verifier_gate_receipts": [{"gate": "company_fit", "decision": "unavailable",
                "failure_class": "company_verification_exhausted"}]}},
        {"company_index": 9, "label_status": "unjudged", "judgment": None},
    ]
    example["judgment"].update(per_icp_score=0, company_judgment_refs=[{"company_index": 7, "cache_key": "cached"}],
        company_authority_evidence=[{"cache_key": "cached", "evidence_doc": {"source_score_run_id": "prior-score"}}])
    return seal(example)


def test_raw_export_keeps_only_recorded_round_dates_and_automated_authority():
    values = export_fixture()
    values["round_row"].update(created_at="2026-10-19T12:00:00+00:00", published_at=None)
    before = deepcopy(values["round_row"])
    example = training_export.assemble_example(**values)
    assert example["round_dates"] == {"created_at": before["created_at"], "published_at": None}
    assert example["label_authority"]["origin"] == "automated_native_scorer"
    assert example["label_authority"]["ground_truth"] == "not_established"
    assert values["round_row"] == before


def test_round_date_context_still_uses_the_existing_whole_export_privacy_pass():
    values = export_fixture()
    private_url = "https://user:private-date-value@example.test/page"
    values["round_row"]["created_at"] = private_url
    example = training_export.assemble_example(**values)
    assert private_url not in json.dumps(example)
    assert example["privacy"]["privacy_filtered"] is True


def test_steps_keep_position_bound_outcomes_dates_and_corrections_without_target_changes():
    original = fixture()
    baseline, _ = records(deepcopy(original))
    example = annotated(original)
    got, _ = records(seal(example))
    assert len(got) == 1
    row, prior = got[0], baseline[0]
    assert (row["input"], row["target"], row["candidate_id"]) == (prior["input"], prior["target"], prior["candidate_id"])
    context = row["source_refs"][0]["training_provenance"]
    assert context["dates"]["round_published_at"] == "2026-10-21T01:02:03Z"
    assert "source_publication_date" not in context["dates"]
    assert context["source"]["source_correction"] == example["source"]["source_correction"]
    assert context["label_authority"]["ground_truth"] == "not_established"
    outcomes = context["company_outcomes"]
    assert [(item["company_index"], item["class"]) for item in outcomes] == [
        (7, "qualified"), (2, "wrong_fit"), (4, "insufficient_evidence"), (5, "provider_error"), (9, "unjudged")]
    assert outcomes[0]["label_provenance"]["source_score_run_id"] == "prior-score"
    assert outcomes[1]["label_provenance"]["source_score_run_id"] == example["score_run_id"]
    assert row["quality"] == prior["quality"]


def test_missing_dates_and_zero_without_company_verdict_never_become_negative_gold():
    example = fixture()
    example["judgment"]["per_icp_score"] = 0
    got, _ = records(seal(example))
    context = got[0]["source_refs"][0]["training_provenance"]
    assert all(value is None for value in context["dates"].values())
    assert context["company_outcomes"] == []
    assert context["label_authority"]["external_adjudication"] == "not_established_by_export"


def test_full_thinking_keeps_the_same_provenance_without_changing_training_content():
    original = full_example()
    baseline, _ = thinking_rows(deepcopy(original))
    example = annotated(original)
    got, _ = thinking_rows(example)
    assert len(got) == 1
    for key in ("input", "target", "generated_thinking", "candidate_id"):
        assert got[0][key] == baseline[0][key]
    context = got[0]["source_refs"][0]["training_provenance"]
    assert context["dates"]["evaluation_date"] == example["evaluation_date"]
    assert context["company_outcomes"][2]["class"] == "insufficient_evidence"


def test_corrected_labels_retain_the_verified_original_scorer_policy(monkeypatch):
    values, authority = preserved_fixture(monkeypatch)
    example = training_reader.export_run(PreservedStore(values, authority), values["objects"], values["run_id"])
    assert example["completeness"]["complete"], example["completeness"]["incomplete_reasons"]
    context = training_steps._training_provenance(example)
    assert context["label_policy_status"] == "recorded"
    assert context["scorer_policy_sha256"] == training_export.contracts.document_hash(authority["scorer_policy"])
    assert context["scorer_image_digest"] == authority["scorer_image_digest"]
    assert context["scorer_image_digest"] != example["configuration"]["scorer_image_digest"]


def test_unverified_ancillary_policy_does_not_remove_a_proven_observed_step():
    example = fixture()
    original, _ = records(deepcopy(example))
    example["judgment"]["preserved_judgment_policy"] = {"unverified": True}
    got, _ = records(seal(example))
    assert len(got) == len(original) == 1
    assert got[0]["candidate_id"] == original[0]["candidate_id"]
    context = got[0]["source_refs"][0]["training_provenance"]
    assert context["label_policy_status"] == "unverified"
    assert context["scorer_policy_sha256"] is None
