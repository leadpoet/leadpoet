"""Training-only review cannot change raw candidates or bypass proof gates."""

import copy
import json

import pytest

from lab_arena import contracts, training_curation, training_dataset, training_steps, training_thinking
from scripts.prepare_arena_agent_dataset import main as agent_main
from scripts.prepare_arena_full_thinking_dataset import main as thinking_main
from scripts.prepare_arena_training_dataset import main as qualification_main
from tests.lab_arena.training_dataset_test import claim, example, row
from tests.lab_arena.training_steps_test import fixture
from tests.lab_arena.training_thinking_test import full_example


def _decision(record, split, manifest, curation, status="accepted"):
    return {"record_id": record.get("record_id") or record.get("candidate_id"),
            "record_hash": contracts.document_hash(record), "record_file": split + ".jsonl",
            "source_dataset_version": manifest["dataset_version"],
            "curation_version": curation["curation_version"],
            **({"input_provenance": copy.deepcopy(record["meta"])} if "record_id" in record
               else {"source_refs": copy.deepcopy(record["source_refs"])}),
            "human_review_status": status, "reviewed_label": copy.deepcopy(record.get("label")) if status == "accepted" else None,
            "reviewer": "human-reviewer" if status != "unreviewed" else None,
            "review_reason": "Checked retained evidence and target" if status != "unreviewed" else None}


def _qual():
    source = example(rows=[row(0, claim(), "qualified")])
    projections = {}
    splits, manifest = training_dataset.build([source], curation_metadata=projections)
    curation = training_curation.flag_manifest(splits, manifest["dataset_version"], projections)
    split = "train" if splits["train"] else "eval"
    return source, splits, manifest, curation, split


def test_structural_verdict_cases_flag_required_primary_and_semantic_review():
    # Sanitized shapes reproduce uncertain primary with a valid bonus, an
    # overall disqualification, and a disputed geography negative.
    judgment = {"intent_signals_detail": [
        {"matched_icp_signal": 0, "after_decay": 42, "date_verdict": "uncertain",
         "evidence_urls": ["https://example.com/undated"], "judge_verdict": {
             "decision": "verified", "verification_trace": {"intent_verdict": {
                 "overall_verdict": "needs_review", "missing_or_risks": ["Date not established"],
                 "signal_evaluations": [{"signal_status": "supported",
                                         "risk_notes": ["Date not established"],
                                         "evidence_urls_used": ["https://example.com/undated"]}]}}}},
        {"matched_icp_signal": 1, "after_decay": 12, "date_verdict": "in_window",
         "evidence_urls": ["https://example.com/bonus"], "judge_verdict": {
             "decision": "verified", "verification_trace": {"intent_verdict": {
                 "overall_verdict": "qualified", "signal_evaluations": []}}}}]}
    projection = training_curation.project_judgment(judgment)
    qualified = {"schema_version": training_dataset.RECORD_SCHEMA_VERSION,
                 "label": {"class": "qualified"}}
    assert projection["signals"][0]["overall_risks"] == ["Date not established"]
    assert projection["signals"][0]["signal_evaluations"][0]["evidence_urls_used"] == [
        "https://example.com/undated"]
    assert projection["signals"][1]["date_verdict"] == "in_window"
    reasons = training_curation.qualification_flags(qualified, projection)
    assert "required_primary_without_verified_in_window_qualified_support" in reasons
    assert "primary_overall_needs_review" in reasons
    assert "primary_intent_risk_notes" in reasons
    judgment["intent_signals_detail"][0]["date_verdict"] = "in_window"
    judgment["intent_signals_detail"][0]["judge_verdict"]["verification_trace"]["intent_verdict"]["overall_verdict"] = "disqualified"
    assert "primary_overall_disqualified" in training_curation.qualification_flags(
        qualified, training_curation.project_judgment(judgment))
    assert "negative_fit_semantics_need_human_review" in training_curation.qualification_flags(
        {"schema_version": training_dataset.RECORD_SCHEMA_VERSION,
         "label": {"class": "wrong_fit", "sub_reason": "geography"}})
    assert training_curation.qualification_flags({"schema_version": training_steps.RECORD_SCHEMA_VERSION}) == [
        "observed_action_quality_unreviewed"]
    assert training_curation.qualification_flags({"schema_version": training_thinking.RECORD_SCHEMA_VERSION}) == [
        "generated_thinking_quality_unreviewed"]


def test_raw_identity_and_reviewed_selection_fail_closed():
    source, splits, manifest, curation, split = _qual()
    plain, plain_manifest = training_dataset.build([source])
    assert splits == plain and manifest == plain_manifest
    record = splits[split][0]
    decision = _decision(record, split, manifest, curation)
    reviewed, reviewed_manifest = training_curation.apply_review(splits, manifest, [decision], curation)
    assert reviewed[split][0]["label"] == record["label"]
    assert reviewed[split][0]["reviewed_label"] == record["label"]
    assert reviewed[split][0]["group_keys"] == record["group_keys"]
    assert reviewed_manifest["source_dataset_version"] == manifest["dataset_version"]
    assert reviewed_manifest["source_curation_version"] == curation["curation_version"]
    assert reviewed_manifest["training_permission"]["status"] == "unconfirmed"
    empty, _ = training_curation.apply_review(splits, manifest, [], curation)
    assert not empty["train"] and not empty["eval"] and len(empty["analysis"]) == 1
    changed_record = copy.deepcopy(splits)
    changed_record[split][0]["input"]["company"]["company_name"] = "Different Company"
    renewed_decision = _decision(changed_record[split][0], split, manifest, curation)
    with pytest.raises(ValueError, match="curation_split_mismatch"):
        training_curation.apply_review(changed_record, manifest, [renewed_decision], curation)
    with pytest.raises(ValueError, match="curation_split_mismatch"):
        training_curation.review_sample(changed_record, 1, manifest, curation)
    for change in (
        {"record_hash": "sha256:" + "0" * 64},
        {"source_dataset_version": "sha256:" + "0" * 64},
        {"source_dataset_version": False},
        {"curation_version": "sha256:" + "0" * 64},
        {"source_dataset_version": None},
        {"curation_version": False},
        {"record_file": "analysis.jsonl"},
        {"human_review_status": "approved"},
        {"reviewer": ""},
        {"reviewed_label": {"class": "wrong_fit"}},
        {"original_label": {"class": "wrong_fit"}},
        {"input_provenance": {"example_hash": "tampered"}},
    ):
        bad = dict(decision, **change)
        with pytest.raises(ValueError):
            training_curation.apply_review(splits, manifest, [bad], curation)
    with pytest.raises(ValueError):
        training_curation.apply_review(splits, manifest, [decision, decision], curation)
    for field in ("source_dataset_version", "curation_version"):
        missing = dict(decision)
        missing.pop(field)
        with pytest.raises(ValueError, match="stale_review_manifest"):
            training_curation.apply_review(splits, manifest, [missing], curation)
    for changed_manifest, changed_curation in (
        (dict(manifest, dataset_version="sha256:" + "1" * 64), curation),
        (manifest, dict(curation, curation_version="sha256:" + "1" * 64)),
        (manifest, dict(curation, flagged_records=[])),
        (manifest, dict(curation, training_permission={"status": "confirmed"})),
        (dict(manifest, rules={"seed": "changed"}), curation),
    ):
        with pytest.raises(ValueError):
            training_curation.apply_review(splits, changed_manifest, [decision], changed_curation)
    unknown = dict(manifest, schema_version="unknown.manifest.v1")
    unknown["dataset_version"] = contracts.document_hash({
        k: v for k, v in unknown.items() if k != "dataset_version"})
    unknown_curation = training_curation.flag_manifest(splits, unknown["dataset_version"])
    with pytest.raises(ValueError, match="curation_manifest_mismatch"):
        training_curation.review_sample(splits, 1, unknown, unknown_curation)
    with pytest.raises(ValueError, match="curation_manifest_mismatch"):
        training_curation.apply_review(splits, unknown, [], unknown_curation)
    fake_permission = {"status": "confirmed"}
    spoofed_source = dict(manifest, training_permission=fake_permission)
    spoofed_curation = dict(curation, training_permission=fake_permission)
    spoofed_curation["curation_version"] = contracts.document_hash({
        k: v for k, v in spoofed_curation.items() if k != "curation_version"})
    with pytest.raises(ValueError, match="curation_manifest_mismatch"):
        training_curation.apply_review(splits, spoofed_source, [], spoofed_curation)
    changed = copy.deepcopy(curation)
    changed["flagged_records"][0]["reasons"] = ["changed_policy_flag"]
    changed["curation_version"] = contracts.document_hash({k: v for k, v in changed.items()
                                                            if k != "curation_version"})
    with pytest.raises(ValueError, match="stale_review_manifest"):
        training_curation.apply_review(splits, manifest, [decision], changed)
    rejected = _decision(record, split, manifest, curation, status="rejected")
    rejected["curation_version"] = False
    with pytest.raises(ValueError, match="stale_review_manifest"):
        training_curation.apply_review(splits, manifest, [rejected], curation)


def test_review_cannot_promote_analysis_or_proof_failure():
    _, splits, manifest, curation, split = _qual()
    bad = copy.deepcopy(splits)
    bad[split][0]["selection_exclusion"] = "missing_company_fit_source_evidence"
    bad_manifest = copy.deepcopy(manifest)
    bad_manifest["record_hashes"] = {name: [contracts.document_hash(record) for record in rows]
                                     for name, rows in bad.items()}
    keys = ("schema_version", "record_schema_version", "rules", "sources", "record_hashes")
    bad_manifest["dataset_version"] = contracts.document_hash({key: bad_manifest[key] for key in keys})
    bad_curation = training_curation.flag_manifest(bad, bad_manifest["dataset_version"])
    decision = _decision(bad[split][0], split, bad_manifest, bad_curation)
    with pytest.raises(ValueError, match="ineligible_review_acceptance"):
        training_curation.apply_review(bad, bad_manifest, [decision], bad_curation)


@pytest.mark.parametrize("main,source", [
    (qualification_main, lambda: example(rows=[row(0, claim(), "qualified")])),
    (agent_main, fixture),
    (thinking_main, full_example),
])
def test_all_cli_review_modes_keep_raw_and_write_separate_reviewed(tmp_path, main, source):
    path = tmp_path / "source.json"
    path.write_text(json.dumps(source()))
    raw = tmp_path / "raw"
    assert main(["--examples", str(path), "--output", str(raw), "--review-sample-size", "10"]) == 0
    curation = json.loads((raw / "curation.json").read_text())
    assert curation["training_permission"]["status"] == "unconfirmed"
    review = [json.loads(line) for line in (raw / "review.jsonl").read_text().splitlines()]
    assert review
    assert review[0]["source_dataset_version"] == json.loads((raw / "manifest.json").read_text())["dataset_version"]
    assert review[0]["curation_version"] == curation["curation_version"]
    review[0].update(human_review_status="accepted", reviewer="human-reviewer",
                     review_reason="Checked retained evidence and target")
    if review[0].get("original_label") is not None:
        review[0]["reviewed_label"] = review[0]["original_label"]
    decisions = tmp_path / "decisions.jsonl"
    decisions.write_text(contracts.canonical_json(review[0]) + "\n")
    reviewed = tmp_path / "reviewed"
    assert main(["--examples", str(path), "--output", str(reviewed),
                 "--review-decisions", str(decisions)]) == 0
    manifest = json.loads((reviewed / "manifest.json").read_text())
    assert manifest["source_dataset_version"] == json.loads((raw / "manifest.json").read_text())["dataset_version"]
    assert manifest["source_curation_version"] == curation["curation_version"]
    assert sum(len((reviewed / (name + ".jsonl")).read_text().splitlines()) for name in ("train", "eval")) == 1
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    empty_output = tmp_path / "empty_reviewed"
    assert main(["--examples", str(path), "--output", str(empty_output),
                 "--review-decisions", str(empty)]) == 2
    assert not (empty_output / "train.jsonl").read_text()
    assert not (empty_output / "eval.jsonl").read_text()
