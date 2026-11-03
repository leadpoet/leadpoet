"""The November 3 exception preserves exact completed grading identities."""

import copy
import json

import pytest

from lab_arena import contracts
from lab_arena.service import ServiceError
from lab_arena.training_reader import export_run
from tests.lab_arena import training_export_signal_test as signal_tests


def preserved_fixture(monkeypatch, *, signal_policy=True, current=False):
    original = signal_tests.per_icp_cost_fixture
    old_round, round_id = "arena-2026-10-16", "arena-2026-11-03"
    output_ref = signal_tests.OUTPUT_REF.replace(old_round, round_id)
    original_policy_builder = signal_tests.scoring.build_scorer_policy

    def current_policy_builder(**kwargs):
        policy = original_policy_builder(**kwargs)
        policy["judge_models"]["role_batch_check"] = "google/gemini-replacement"
        policy["env_bindings"]["ARENA_ROLE_BATCH_MODEL"] = "google/gemini-replacement"
        return policy

    def november_base():
        values = original()
        for row in [values["round_row"], values["submission_row"], *values["runs"]]:
            row["round_id"] = round_id
            if "submission_id" in row:
                row["submission_id"] = row["submission_id"].replace("2026-10-16", "2026-11-03")
        objects = values["objects"]
        for old_ref in list(objects.values):
            if old_round in old_ref:
                objects.values[old_ref.replace("2026-10-16", "2026-11-03")] = objects.values.pop(old_ref)
        for row in [values["submission_row"], *values["round_row"]["participants"]]:
            row["source_ref"] = row["source_ref"].replace("2026-10-16", "2026-11-03")
            row["submission_id"] = row["submission_id"].replace("2026-10-16", "2026-11-03")
        values["runs"][0]["output_ref"] = output_ref
        if current:
            config = values["round_row"]["configuration_doc"]
            config["scorer_image_digest"] = "sha256:" + "d" * 64
            config["scorer_image_reference"] = "registry.example/scorer@" + config["scorer_image_digest"]
        benchmark = json.loads(objects.values["benchmark"])
        benchmark["round_id"] = round_id
        objects.put("benchmark", contracts.canonical_json(benchmark).encode())
        return values

    with monkeypatch.context() as patch:
        patch.setattr(signal_tests, "per_icp_cost_fixture", november_base)
        patch.setattr(signal_tests, "OUTPUT_REF", output_ref)
        if current:
            patch.setattr(signal_tests.scoring, "build_scorer_policy", current_policy_builder)
        values = signal_tests._company_cache_fixture(signal_policy=signal_policy)
    config = values["round_row"]["configuration_doc"]
    authority = {"operation": "nov03_forward_only_verifier_v1", "scoring_generation_floor": 0,
                 **{key: copy.deepcopy(config[key]) for key in
                    ("scorer_policy", "scorer_image_digest", "scorer_image_reference")}}
    config["scorer_policy"] = copy.deepcopy(config["scorer_policy"])
    config["scorer_policy"]["judge_models"]["role_batch_check"] = "google/gemini-replacement"
    config["scorer_policy"]["env_bindings"]["ARENA_ROLE_BATCH_MODEL"] = "google/gemini-replacement"
    config["scorer_image_digest"] = "sha256:" + "d" * 64
    config["scorer_image_reference"] = "registry.example/scorer@" + config["scorer_image_digest"]
    authority.update({"current_" + key: config[key] for key in
                      ("scorer_image_digest", "scorer_image_reference")})
    return values, authority


class PreservedStore(signal_tests._CompanyExportStore):
    def __init__(self, values, authority):
        super().__init__(values)
        self.authority = authority
        self.bound_rows = copy.deepcopy(values["runs"][:2])
        self.reads = []

    def preserved_judgment_policy(self, round_id, judge_run_id, source_run_id, refs):
        self.reads.append((round_id, judge_run_id, source_run_id, copy.deepcopy(refs)))
        source, judge = self.bound_rows
        if (self.get_run(source_run_id) != source or self.get_run(judge_run_id) != judge
                or refs != judge["company_judgment_refs"]):
            return None
        return copy.deepcopy(self.authority)

    def icp_cost_eligibility(self, **identity):
        return signal_tests.per_icp_cost_reader(
            spend=0, reason="eligible", round_id=self.values["round_row"]["round_id"],
            submission_id=self.values["submission_row"]["submission_id"])(**identity)["eligibility"]


@pytest.mark.parametrize("signal_policy", [False, True])
def test_reader_keeps_original_completed_policy_runtime_and_paid_evidence(monkeypatch, signal_policy):
    values, authority = preserved_fixture(monkeypatch, signal_policy=signal_policy)
    store = PreservedStore(values, authority)
    before = copy.deepcopy(values["runs"])
    result = export_run(store, values["objects"], values["run_id"])
    assert result["completeness"]["complete"], result["completeness"]
    assert result["configuration"] == values["round_row"]["configuration_doc"]
    assert result["judgment"]["runtime_identity"]["grading_image_digest"] == authority["scorer_image_digest"]
    assert result["judgment"]["preserved_judgment_policy"] == authority
    attempt = next(row for row in result["attempts"] if row["run_id"] == "judge:1")
    assert attempt["scoring_generation_floor"] == 0
    assert attempt["scoring_policy_hash"] == contracts.document_hash(authority["scorer_policy"])
    assert result["input_snapshots"]["judge:1"]["input"]["scorer_policy"] == authority["scorer_policy"]
    assert result["judgment"]["company_authority_evidence"] == values["company_judgment_rows"]
    assert values["runs"] == before
    assert len(store.reads) == 1
    if signal_policy:
        assert result["judgment"]["signal_source_cost_attribution"]["confirmed_actual_microusd"] == 42
        call = result["judgment"]["signal_source_provider_calls"][0]
        assert len(call["request"]["admitted_parameters"]["messages"][0]["content"]) == 12000
        assert call["response"]["choices"][0]["message"]["content"] == "result"


@pytest.mark.parametrize("mutation", ["unlisted", "judge_result", "source_score", "refs", "refs_current"])
def test_unlisted_or_changed_old_rows_cannot_get_complete_export(monkeypatch, mutation):
    values, authority = preserved_fixture(monkeypatch)
    store = PreservedStore(values, authority)
    if mutation == "unlisted":
        store.bound_rows[1]["run_id"] = "not-listed"
    elif mutation == "judge_result":
        values["runs"][1]["result_doc"]["unlisted_change"] = True
    elif mutation == "source_score":
        values["runs"][0]["per_icp_score"] += 1
    elif mutation == "refs":
        values["runs"][1]["company_judgment_refs"][0]["scope_doc"]["scorer_image_digest"] = "sha256:" + "e" * 64
    else:
        config = values["round_row"]["configuration_doc"]
        values["runs"][1]["company_judgment_refs"][0]["scope_doc"].update(
            {key: config[key] for key in ("scorer_image_digest", "scorer_image_reference")})
    result = export_run(store, values["objects"], values["run_id"])
    assert not result["completeness"]["complete"]
    assert "preserved_judgment_policy" not in result.get("judgment", {})
    judge_attempt = next(row for row in result["attempts"] if row["run_id"] == "judge:1")
    assert "runtime_identity" not in judge_attempt


def test_current_policy_judgment_does_not_request_preservation(monkeypatch):
    values, authority = preserved_fixture(monkeypatch, current=True)
    store = PreservedStore(values, authority)
    result = export_run(store, values["objects"], values["run_id"])
    assert result["completeness"]["complete"]
    assert store.reads == []
    assert "preserved_judgment_policy" not in result["judgment"]
    assert result["judgment"]["runtime_identity"]["grading_image_digest"] == values["round_row"]["configuration_doc"]["scorer_image_digest"]


def test_unrelated_round_does_not_request_preservation():
    values = signal_tests._company_cache_fixture()
    store = PreservedStore(values, {})
    result = export_run(store, values["objects"], values["run_id"])
    assert result["completeness"]["complete"]
    assert store.reads == []
    assert "preserved_judgment_policy" not in result["judgment"]


def test_wrong_current_binding_cannot_apply_original_policy(monkeypatch):
    values, authority = preserved_fixture(monkeypatch)
    authority["current_scorer_image_digest"] = "sha256:" + "e" * 64
    with pytest.raises(ServiceError, match="preserved_judgment_policy_invalid"):
        export_run(PreservedStore(values, authority), values["objects"], values["run_id"])
