"""Company-qualification records: labels, leakage, dedup, splits and manifest."""

from __future__ import annotations

import copy
import hashlib
import json
import stat

import pytest

from lab_arena import training_dataset as dataset
from scripts import prepare_arena_training_dataset as cli

ADAPTER = "qualification_integrity_v2"
ICP = {"icp_id": "icp_20261017_001", "prompt": "B2B software companies hiring sales leaders.",
       "industry": "Software", "employee_count": ["51-200"], "country": "United States",
       "intent_signals": ["Hiring a VP of Sales"], "verified_example_company": "generator-secret.com"}
VERIFIED = [{"matched_icp_signal": 0, "after_decay": 40.0, "judge_verdict": {"decision": "verified"},
             "date_verdict": "in_window", "evidence_urls": ["https://acme-robotics.com/careers"]}]


def claim(name="Acme Robotics", domain="acme-robotics.com", slug="acme-robotics"):
    return {"company_name": name, "company_website": "https://" + domain,
            "company_linkedin": "https://www.linkedin.com/company/" + slug,
            "industry": "Software", "employee_count": "51-200",
            "intent_signals": [{"matched_icp_signal": 0, "description": "Hiring a VP of Sales",
                                "date": "2026-10-10", "url": "https://" + domain + "/careers"}]}


def fit(decision="match", **extra):
    return {"gate": "company_fit", "decision": decision, **extra}


JUDGMENTS = {
    "qualified": {"company_qualified": True, "final_score": 80, "failure_reason": None,
                  "verifier_gate_receipts": [fit()], "intent_signals_detail": VERIFIED},
    "wrong_fit": {"company_qualified": False, "final_score": 0, "failure_reason": "Company fit failed: industry",
                  "verifier_gate_receipts": [fit("mismatch", company_fit_dimensions={
                      "industry": "mismatch", "geography": "match"}, required_attribute_decision="mismatch")],
                  "intent_signals_detail": None},
    "fit_unproven": {"company_qualified": False, "final_score": 0,
                     "failure_reason": "Company fit unavailable: unproven dimensions industry",
                     "verifier_gate_receipts": [fit("unavailable", failure_class="insufficient_fit_evidence")],
                     "intent_signals_detail": None},
    "intent_unverified": {"company_qualified": False, "final_score": 0,
                          "failure_reason": "Primary intent evidence unverified: no dated source",
                          "verifier_gate_receipts": [fit()],
                          "intent_signals_detail": [{"matched_icp_signal": 0, "after_decay": 0.0,
                                                     "judge_verdict": {"decision": "rejected_three_stage"}}]},
    "duplicate": {"company_qualified": False, "duplicate_company": True, "final_score": 0,
                  "failure_reason": "Duplicate company identity", "verifier_gate_receipts": []},
    "exhausted": {"company_qualified": False, "final_score": 0, "failure_reason": "Company verification exhausted",
                  "verifier_gate_receipts": [fit("unavailable", failure_class="company_verification_exhausted")]},
    "contract": {"company_qualified": False, "final_score": 0, "failure_reason": "company model contract incompatible",
                 "verifier_gate_receipts": [fit("unavailable", failure_class="model_contract_incompatible")]},
}


def row(index, company, kind=None, status="scored"):
    return {"company_index": index, "company": company, "label_status": status,
            "judgment": copy.deepcopy(JUDGMENTS[kind]) if kind else None}


def seal(document):
    """Hash an example the way the exporter does, after a test edits it."""
    document["example_hash"] = dataset._hash({key: value for key, value in document.items() if key != "example_hash"})
    return document


def example(run="arena-2026-10-18:sub-1:1:0:1", rows=(), *, round_id="arena-2026-10-18", **changes):
    document = {
        "schema_version": dataset.EXAMPLE_SCHEMA_VERSION,
        "example_id": "sha256:" + hashlib.sha256(run.encode()).hexdigest(),
        "run_id": run, "score_run_id": run + ":score:1", "round_id": round_id, "round_status": "published",
        "submission_id": "sub-1", "role": "miner", "stage": 1, "icp_position": 0,
        "evaluation_date": "2026-10-18",
        "configuration": {"scorer_policy": {"scoring_adapter_version": ADAPTER},
                          "scorer_image_digest": "sha256:" + "a" * 64, "integrity_policy": "arena_integrity_v1"},
        "icp": copy.deepcopy(ICP), "input_snapshots": {},
        "output": {"companies": [item["company"] for item in rows]}, "companies": list(rows),
        "judgment": {}, "source": {"normalized_sha256": "sha256:" + "b" * 64},
        "privacy": {"context_redactions": []},
        "completeness": {"complete": False, "incomplete_reasons": ["direct_web_content_not_captured"],
                         "dimensions": {"context": True, "input_output_label": True, "exact_inputs": True}},
    }
    document.update(changes)
    return seal(document)


@pytest.mark.parametrize("kind,label,sub_reason", [
    ("qualified", "qualified", None),
    ("wrong_fit", "wrong_fit", "industry,required_attribute"),
    ("fit_unproven", "insufficient_evidence", "fit_insufficient_fit_evidence"),
    ("intent_unverified", "insufficient_evidence", "primary_intent_unverified"),
    ("duplicate", "duplicate", None),
    ("exhausted", "provider_error", "company_verification_exhausted"),
    ("contract", "invalid_submission", "model_contract_incompatible"),
])
def test_labels_come_from_the_stored_verdict(kind, label, sub_reason):
    assert dataset.classify(row(0, claim(), kind)) == (label, sub_reason)


def test_non_verdict_rows_are_never_negatives():
    assert dataset.classify(row(3, claim(), status="outside_evaluated_slice")) == ("outside_scored_slice", None)
    assert dataset.classify(row(0, claim(), status="skipped")) == ("bucket_skipped", "submitted_employee_bucket")
    assert dataset.classify(row(0, claim())) == ("unjudged", "scored")


def test_input_is_what_the_agent_had_and_never_the_label():
    records = dataset.records_from_example(example(rows=[row(0, claim(), "qualified")]))
    record = records[0]
    assert set(record["input"]) == {"icp", "icp_source", "evaluation_date", "company"}
    assert record["input"]["icp_source"] == "agent_visible_projection"
    rendered = json.dumps(record["input"])
    for leaked in ("generator-secret", "company_qualified", "verifier_gate_receipts", "judge_verdict", "label_status"):
        assert leaked not in rendered
    assert record["label"]["class"] == "qualified" and record["label"]["primary_intent_verified"] is True
    snapshot = {"input": {"icp": {"industry": "Software", "output_schema_version": "v6"}}, "privacy": {"redacted": False}}
    with_snapshot = example(rows=[row(0, claim(), "qualified")])
    with_snapshot["input_snapshots"] = {with_snapshot["run_id"]: snapshot}
    exact = dataset.records_from_example(with_snapshot)[0]["input"]
    assert exact["icp"] == snapshot["input"]["icp"] and exact["icp_source"] == "input_snapshot"


def test_unjudged_execution_keeps_its_companies_for_analysis_only():
    document = example(rows=[row(0, claim(), "qualified")])
    document["companies"] = []
    splits, manifest = dataset.build([seal(document)])
    assert [record["label"]["class"] for record in splits["analysis"]] == ["unjudged"]
    assert splits["train"] == splits["eval"] == []


@pytest.mark.parametrize("changes,reason", [
    ({"schema_version": "other"}, "example_schema_unsupported"),
    ({"round_id": "arena-2026-10-18-probe"}, "not_a_published_production_round"),
    ({"round_status": "stage1_scoring"}, "not_a_published_production_round"),
    ({"configuration": {"scorer_policy": {"scoring_adapter_version": "qualification_style_v1"}}},
     "scorer_adapter_without_company_labels"),
    ({"completeness": {"dimensions": {"context": True, "input_output_label": False}, "incomplete_reasons": []}},
     "label_unverified"),
    ({"completeness": {"dimensions": {"context": True, "input_output_label": True},
                       "incomplete_reasons": ["scorer_image_identity_missing"]}}, "label_authority_unverified"),
    ({"completeness": {"dimensions": {"context": True, "input_output_label": True},
                       "incomplete_reasons": ["runtime_input_context_mismatch"]}}, "input_conflict"),
    ({"privacy": {"context_redactions": ["credential"]}}, "claim_privacy_filtered"),
])
def test_unusable_examples_are_excluded_and_counted(changes, reason):
    document = example(rows=[row(0, claim(), "qualified")], **changes)
    assert dataset.example_exclusion(document, dataset.Rules()) == reason
    splits, manifest = dataset.build([document])
    assert not any(splits.values())
    assert manifest["counts"]["examples_excluded"] == {reason: 1}
    cohort = dataset.Rules(scorer_image_digest="sha256:" + "c" * 64)
    assert dataset.example_exclusion(example(rows=[row(0, claim(), "qualified")]), cohort) == "outside_scorer_cohort"


def test_an_edited_example_no_longer_matches_its_hash():
    document = example(rows=[row(0, claim(), "wrong_fit")])
    document["companies"][0]["judgment"] = copy.deepcopy(JUDGMENTS["qualified"])
    assert dataset.example_exclusion(document, dataset.Rules()) == "example_hash_mismatch"
    splits, manifest = dataset.build([document])
    assert not any(splits.values())
    assert manifest["counts"]["examples_excluded"] == {"example_hash_mismatch": 1}
    assert dataset.example_exclusion(seal(document), dataset.Rules()) is None


def test_repeated_claims_count_once_and_conflicting_verdicts_are_left_out():
    same = [example("run-a", [row(0, claim(), "qualified")]), example("run-b", [row(0, claim(), "qualified")])]
    splits, manifest = dataset.build(same)
    assert sum(len(items) for items in splits.values()) == 1
    assert manifest["counts"]["duplicate_records_merged"] == 1
    conflict = [example("run-a", [row(0, claim(), "qualified")]), example("run-b", [row(0, claim(), "wrong_fit")])]
    splits, manifest = dataset.build(conflict)
    assert not any(splits.values()) and manifest["counts"]["conflicting_label_groups_excluded"] == 1
    # Falling outside the scored slice is not a verdict and does not conflict.
    mixed = [example("run-a", [row(0, claim(), "qualified")]),
             example("run-b", [row(0, claim(), status="outside_evaluated_slice")])]
    splits, _ = dataset.build(mixed)
    assert len(splits["train"] + splits["eval"]) == 1 and len(splits["analysis"]) == 1


def test_the_same_claim_on_another_date_is_a_separate_input():
    # Intent recency depends on the date, so a later verdict is not a conflict.
    dated = [example("run-a", [row(0, claim(), "qualified")]),
             example("run-b", [row(0, claim(), "intent_unverified")], round_id="arena-2026-10-19",
                     evaluation_date="2026-10-19")]
    splits, manifest = dataset.build(dated)
    labels = sorted(record["label"]["class"] for name in ("train", "eval") for record in splits[name])
    assert labels == ["insufficient_evidence", "qualified"]
    assert manifest["counts"]["conflicting_label_groups_excluded"] == 0
    assert manifest["counts"]["duplicate_records_merged"] == 0
    # Both dates describe one company, so they still share a split.
    assert len({record["group_keys"]["company_cluster"] for name in ("train", "eval") for record in splits[name]}) == 1


def test_splits_never_share_a_company_cluster():
    companies = [claim("Company %d" % index, "company%d.com" % index, "company-%d" % index) for index in range(40)]
    examples = [example("run-%d" % index, [row(0, company, "qualified" if index % 2 else "wrong_fit")])
                for index, company in enumerate(companies)]
    # The same company under another name and ICP shares its LinkedIn slug.
    alias = claim("Company Zero Inc", "zero-alias.com", "company-0")
    examples.append(example("run-alias", [row(0, alias, "wrong_fit")],
                            icp={**ICP, "industry": "Fintech"}))
    splits, manifest = dataset.build(examples, dataset.Rules(eval_percent=50))
    clusters = {name: {record["group_keys"]["company_cluster"] for record in splits[name]} for name in ("train", "eval")}
    assert splits["train"] and splits["eval"]
    assert not clusters["train"] & clusters["eval"]
    zero = [record for name in ("train", "eval") for record in splits[name]
            if record["input"]["company"]["company_linkedin"].endswith("/company-0")]
    assert len(zero) == 2 and len({record["group_keys"]["company_cluster"] for record in zero}) == 1


def test_manifest_is_deterministic_and_records_permission_status():
    examples = [example("run-%d" % index, [row(0, claim("C%d" % index, "c%d.com" % index, "c-%d" % index),
                                                "qualified")]) for index in range(6)]
    first = dataset.build(examples)
    second = dataset.build(list(reversed(examples)))
    assert first == second
    manifest = first[1]
    assert manifest["training_permission"]["status"] == "unconfirmed"
    assert manifest["rules"] == {"eval_percent": 20, "seed": "leadpoet-arena-dataset-v1",
                                 "production_rounds_only": True, "scorer_image_digest": None}
    other_seed = dataset.build(examples, dataset.Rules(seed="other"))[1]
    assert other_seed["dataset_version"] != manifest["dataset_version"]


def test_real_exporter_output_produces_checked_labels():
    from scripts.export_arena_training import export_run
    from tests.lab_arena.training_export_signal_test import _CompanyExportStore, _add_frozen_catalog, _mixed_fit_fixture
    from tests.lab_arena.training_export_test import save_input

    values = _mixed_fit_fixture()
    judge, document = _add_frozen_catalog(values)
    save_input(values["objects"], judge["run_id"], "score", document)
    exported = export_run(_CompanyExportStore(values), values["objects"], values["run_id"])
    records = dataset.records_from_example(exported)
    assert [record["label"]["class"] for record in records] == ["qualified", "wrong_fit"]
    assert records[0]["label"]["primary_intent_verified"] is True
    assert records[0]["input"]["icp_source"] == "input_snapshot"
    assert "verified_example_company" not in json.dumps(records[0]["input"])
    # The shared fixture's judge input differs from its frozen context. A judge
    # that saw a different context cannot supply a label; a consistent one can.
    assert "judge_input_context_mismatch" in exported["completeness"]["incomplete_reasons"]
    assert dataset.example_exclusion(exported, dataset.Rules()) == "input_conflict"
    consistent = copy.deepcopy(exported)
    consistent["completeness"]["incomplete_reasons"].remove("judge_input_context_mismatch")
    assert dataset.example_exclusion(consistent, dataset.Rules()) == "example_hash_mismatch"
    assert dataset.example_exclusion(seal(consistent), dataset.Rules()) is None


def test_cli_writes_a_new_private_dataset(tmp_path, capsys):
    source = tmp_path / "exports"
    source.mkdir()
    for index in range(4):
        document = example("run-%d" % index, [row(0, claim("C%d" % index, "c%d.com" % index, "c-%d" % index),
                                                   "qualified")])
        (source / ("example-%d.json" % index)).write_text(json.dumps(document))
    output = tmp_path / "dataset"
    assert cli.main(["--examples", str(source), "--output", str(output)]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["training_permission"] == "unconfirmed"
    assert sum(summary["records"].values()) == 4
    for name in ("train.jsonl", "eval.jsonl", "analysis.jsonl", "manifest.json"):
        assert stat.S_IMODE((output / name).stat().st_mode) == 0o600
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["dataset_version"] == summary["dataset_version"]
    # An existing dataset is never replaced or removed.
    assert cli.main(["--examples", str(source), "--output", str(output)]) == 1
    assert json.loads((output / "manifest.json").read_text()) == manifest


def test_cli_removes_its_partial_output_after_a_failed_write(tmp_path, monkeypatch, capsys):
    source = tmp_path / "exports"
    source.mkdir()
    (source / "example.json").write_text(json.dumps(example(rows=[row(0, claim(), "qualified")])))
    output = tmp_path / "dataset"
    write = cli._write_private

    def fail_on_manifest(path, text):
        if path.name == "manifest.json":
            raise OSError("disk full")
        write(path, text)

    monkeypatch.setattr(cli, "_write_private", fail_on_manifest)
    assert cli.main(["--examples", str(source), "--output", str(output)]) == 1
    assert capsys.readouterr().err.strip() == "Arena training dataset failed: OSError"
    assert not output.exists()
    monkeypatch.setattr(cli, "_write_private", write)
    assert cli.main(["--examples", str(source), "--output", str(output)]) == 0
