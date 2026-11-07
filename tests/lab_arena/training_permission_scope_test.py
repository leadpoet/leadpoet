"""Operator permission does not clear record rights or establish data quality."""

import copy
import json
from functools import partial

import pytest

from lab_arena import (
    contracts, internal_capture, training_curation, training_dataset,
    training_readiness, training_steps, training_thinking,
)
from scripts.export_arena_internal_training import main as internal_main
from scripts.prepare_arena_agent_dataset import main as agent_main
from scripts.prepare_arena_full_thinking_dataset import main as thinking_main
from scripts.prepare_arena_training_dataset import main as qualification_main
from tests.lab_arena.internal_capture_core_test import captured, request
from tests.lab_arena.judge_thinking_export_test import judge_example
from tests.lab_arena.training_curation_test import _decision
from tests.lab_arena.training_dataset_test import claim, example, row
from tests.lab_arena.training_steps_test import fixture
from tests.lab_arena.training_thinking_test import full_example


def qualification_example():
    return example(rows=[row(0, claim(), "qualified")])


def assert_operator_scope(permission):
    assert permission["status"] == "confirmed"
    assert permission["scope"] == "operator_authorization_only"
    assert permission["purpose"] == "leadpoet_foundational_sales_llm_cpt_and_sft"
    assert permission["confirmed_on"] == "2026-11-07"
    assert permission["record_rights"] == "not_cleared_by_operator_authorization"
    assert permission["restrictions"] == "provider_terms_source_rights_and_customer_privacy_permissions_still_apply"
    assert permission["redistribution"] == "not_authorized_by_this_record"


@pytest.mark.parametrize("builder,source", [
    (training_dataset.build, qualification_example),
    (training_steps.build, fixture),
    (training_thinking.build, full_example),
    (partial(training_thinking.build, role="judge"), judge_example),
])
def test_candidate_and_reviewed_manifests_keep_operator_scope(builder, source):
    splits, manifest = builder([source()])
    assert_operator_scope(manifest["training_permission"])
    split = "train" if splits["train"] else "eval"
    record = splits[split][0]
    curation = training_curation.flag_manifest(splits, manifest["dataset_version"])
    assert_operator_scope(curation["training_permission"])
    reviewed, reviewed_manifest = training_curation.apply_review(
        splits, manifest, [_decision(record, split, manifest, curation)], curation)
    assert_operator_scope(reviewed_manifest["training_permission"])
    assert reviewed[split][0]["input"] == record["input"]
    if "quality" in record:
        assert record["quality"]["status"] == "unreviewed"
        assert reviewed[split][0]["quality"] == record["quality"]
    else:
        assert reviewed[split][0]["label"] == record["label"]


def test_readiness_does_not_turn_operator_permission_into_training_approval():
    report = training_readiness.build([qualification_example()])
    assert report["qualification"]["candidate_records"] == 1
    assert_operator_scope(report["readiness"]["training_permission"])
    assert_operator_scope(report["agent_steps"]["manifest"]["training_permission"])
    assert_operator_scope(report["full_thinking"]["manifest"]["training_permission"])
    assert report["readiness"]["training_ready"] is False
    assert report["readiness"]["agent_action_quality"] == "unreviewed"
    assert report["readiness"]["human_label_review"] == "not_proved_by_storage"


def test_internal_export_retains_scope_without_waiving_privacy_or_quality(tmp_path):
    objects, call = captured(tmp_path / "objects")
    _, candidate = internal_capture.read_call(objects, call.prefix + "request.json")
    assert_operator_scope(candidate["training_permission"])
    assert candidate["quality"] == {"status": "unreviewed", "correct_action": "not_established"}
    output = tmp_path / "export"
    assert internal_main(["--object-root", str(tmp_path / "objects"), "--refs",
                          call.prefix + "request.json", "--output", str(output)]) == 0
    manifest = json.loads((output / "manifest.json").read_text())
    assert_operator_scope(manifest["training_permission"])
    assert manifest["quality"] == "unreviewed"
    exported = json.loads((output / "sft.jsonl").read_text())
    assert_operator_scope(exported["training_permission"])
    secret = "test-private-customer-value"
    payload = request()
    payload["messages"][0]["content"] += secret
    private_objects, private_call = captured(tmp_path / "private", payload=payload, secrets=(secret,))
    raw, excluded = internal_capture.read_call(private_objects, private_call.prefix + "request.json")
    assert excluded is None
    assert "privacy_nonexact" in raw["sft_exclusions"]
    assert secret not in json.dumps(raw)


@pytest.mark.parametrize("main,source,extra_args", [
    (qualification_main, qualification_example, []),
    (agent_main, fixture, []),
    (thinking_main, full_example, []),
    (thinking_main, judge_example, ["--role", "judge"]),
])
def test_cli_summary_preserves_scope_and_uncleared_record_rights(tmp_path, capsys, main, source, extra_args):
    path = tmp_path / "example.json"
    path.write_text(json.dumps(source()))
    assert main(["--examples", str(path), "--output", str(tmp_path / "dataset"), *extra_args]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["training_permission"] == "confirmed"
    assert summary["training_permission_scope"] == "operator_authorization_only"
    assert summary["training_record_rights"] == "not_cleared_by_operator_authorization"


@pytest.mark.parametrize("permission", [
    {"status": "unconfirmed"},
    {"scope": "all_record_rights_cleared"},
])
def test_curation_rejects_historical_or_forged_permission_even_when_rehashed(permission):
    splits, manifest = training_dataset.build([qualification_example()])
    curation = training_curation.flag_manifest(splits, manifest["dataset_version"])
    old_manifest, old_curation = copy.deepcopy(manifest), copy.deepcopy(curation)
    replacement = copy.deepcopy(manifest["training_permission"])
    replacement.update(permission)
    old_manifest["training_permission"] = replacement
    old_curation["training_permission"] = copy.deepcopy(replacement)
    old_curation["curation_version"] = contracts.document_hash({
        key: value for key, value in old_curation.items() if key != "curation_version"})
    with pytest.raises(ValueError, match="curation_manifest_mismatch"):
        training_curation.apply_review(splits, old_manifest, [], old_curation)
    with pytest.raises(ValueError, match="curation_manifest_mismatch"):
        training_curation.review_sample(splits, 1, old_manifest, old_curation)
