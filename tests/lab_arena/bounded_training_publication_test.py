"""Publication consumes exact exports without retaining the complete bank."""
import copy
import gc
import weakref
from types import SimpleNamespace

import pytest
from botocore.exceptions import ReadTimeoutError

from lab_arena import contracts, service, training_coverage, training_eligibility, training_reader
from tests.lab_arena.training_eligibility_test import (
    _unsettled_selected_call, refresh_response_links, submission, ten_selected_examples,
)
from tests.lab_arena.training_steps_test import seal
from tests.lab_arena.training_call_semantics_test import _source_timeout_and_cost_example


def harness(examples, percent=80, missing_percent=None, objects=None):
    config = {"stage_1_icp_count": 5, "stage_2_icp_count": 5,
              "training_minimum_icp_coverage_percent": percent}
    if missing_percent is not None:
        config["training_maximum_missing_reasoning_percent"] = missing_percent
    row = {"round_id": examples["run-0"]["round_id"], "configuration_doc": config}
    obj = SimpleNamespace(
        _store=SimpleNamespace(get_submission=lambda _: submission()),
        _objects=objects or object(), _is_daily_baseline=lambda *_: False,
        _selected_accepted_execution_runs=lambda *_: {
            position: {"run_id": run_id} for position, run_id in enumerate(examples)},
    )
    return obj, row


def eager_reference(obj, row, export, *, publication=None):
    """Original eager handoff, with the same canonical aggregate and read view."""
    examples, unavailable_outputs, unavailable_artifacts, unavailable_exports = {}, {}, {}, {}
    try:
        for selected in obj._selected_accepted_execution_runs([], "sub").values():
            run_id = selected["run_id"]
            output_ref = "arena/%s/outputs/%s.json" % (row["round_id"], run_id)
            fallback = {}
            if publication is not None and output_ref in publication.outputs_by_ref:
                cached = publication.raw_outputs_by_ref.get(output_ref)
                if cached is not None:
                    fallback[output_ref] = cached
            view = training_reader.AvailabilityReadView(obj._objects, fallback)
            try:
                examples[run_id] = export(obj._store, view, run_id)
            except Exception as exc:
                examples[run_id] = None
                unavailable_exports[run_id] = type(exc).__name__[:64]
                continue
            transient = {kind: ref for kind, ref in view.artifact_refs.items()
                         if view.transient_failures.get(ref) == "transient"}
            if transient:
                unavailable_artifacts[run_id] = transient
            if view.unavailable in ({output_ref: "missing"}, {output_ref: "transient"}):
                unavailable_outputs[run_id] = output_ref
        source = training_eligibility.assess_source(submission(), enabled=True)
        capture = training_eligibility.assess_capture(
            submission(), examples, list(examples), enabled=True,
            minimum_coverage_percent=training_coverage.percent_from_configuration(row["configuration_doc"]),
            maximum_missing_reasoning_percent=training_coverage.missing_reasoning_percent_from_configuration(row["configuration_doc"]),
            unavailable_exports=unavailable_exports,
            unavailable_outputs=unavailable_outputs, unavailable_artifacts=unavailable_artifacts)
        return {"eligible": source["eligible"] and capture["eligible"],
                "reasons": sorted(set(source["reasons"] + capture["reasons"])),
                "source_review": source, "capture": capture}
    except Exception:
        return {"eligible": False, "reasons": ["training_evidence_check_failed"]}


def actual(obj, row, publication=None):
    return service.ArenaService._submission_training_eligibility(
        obj, row, "sub", [], publication_reads=publication)


class TrackedDict(dict):
    pass


class TrackedList(list):
    pass


def test_previous_export_and_call_graph_released_before_next_read(monkeypatch):
    templates = ten_selected_examples()
    obj, row = harness(templates)
    previous = []
    reads = []

    def export(_store, _view, run_id):
        gc.collect()
        assert not any(ref() is not None for ref in previous), "prior export graph retained"
        previous.clear()
        example = TrackedDict(copy.deepcopy(templates[run_id]))
        example["provider_calls"] = TrackedList(
            TrackedDict(call) for call in example["provider_calls"])
        previous.extend(weakref.ref(value) for value in
                        [example, example["provider_calls"], *example["provider_calls"]])
        reads.append(run_id)
        return example

    monkeypatch.setattr(training_reader, "export_run", export)
    result = actual(obj, row)
    assert result["eligible"] is True, result
    assert reads == list(templates)
    gc.collect()
    assert not any(ref() is not None for ref in previous)


@pytest.mark.parametrize("percent", [80, 100])
@pytest.mark.parametrize("case", ["complete", "two_cost_gaps", "three_cost_gaps", "identity", "hash"])
def test_full_result_matches_eager_aggregate(monkeypatch, percent, case):
    examples = ten_selected_examples()
    if case in {"two_cost_gaps", "three_cost_gaps"}:
        for index in range(8 if case == "two_cost_gaps" else 7, 10):
            _unsettled_selected_call(examples["run-%d" % index])
    elif case == "identity":
        examples["run-9"]["submission_id"] = "wrong-submission"
        seal(examples["run-9"])
    elif case == "hash":
        examples["run-9"]["example_hash"] = "sha256:" + "0" * 64
    obj, row = harness(examples, percent)
    reads = []

    def export(_store, _view, run_id):
        reads.append(run_id)
        return copy.deepcopy(examples[run_id])

    expected = eager_reference(obj, row, export)
    reads.clear()
    monkeypatch.setattr(training_reader, "export_run", export)
    assert actual(obj, row) == expected
    assert reads == list(examples)
    assert expected["eligible"] is (case in {"complete", "two_cost_gaps", "three_cost_gaps"} or percent == 80)


@pytest.mark.parametrize("percent", [80, 100])
@pytest.mark.parametrize("fault", ["missing", "transient"])
def test_publication_fallback_availability_is_visible_to_same_aggregate(monkeypatch, percent, fault):
    examples = ten_selected_examples()
    damaged = examples["run-9"]
    ref = "arena/%s/outputs/run-9.json" % damaged["round_id"]
    raw = contracts.canonical_json(damaged["output"]).encode()
    damaged["output_identity"] = {"ref": ref, "document_hash": contracts.document_hash(damaged["output"])}
    seal(damaged)

    class Objects:
        def get_bounded(self, *_):
            if fault == "missing":
                raise FileNotFoundError("synthetic missing output")
            raise ReadTimeoutError(endpoint_url="https://objects.example.test")

    obj, row = harness(examples, percent, objects=Objects())
    publication = SimpleNamespace(outputs_by_ref={ref: damaged["output"]}, raw_outputs_by_ref={ref: raw})

    def export(_store, view, run_id):
        if run_id == "run-9":
            assert view.get_bounded(ref, 100) == raw
        return copy.deepcopy(examples[run_id])

    expected = eager_reference(obj, row, export, publication=publication)
    monkeypatch.setattr(training_reader, "export_run", export)
    assert actual(obj, row, publication) == expected
    assert expected["eligible"] is (percent == 80)
    if percent == 80:
        assert expected["capture"]["counts"]["complete_runs"] == 9


@pytest.mark.parametrize("percent", [80, 100])
@pytest.mark.parametrize("fault", ["missing", "transient"])
def test_completed_export_artifact_markers_match_eager(monkeypatch, percent, fault):
    examples = ten_selected_examples()
    examples.pop("run-9")
    damaged = _source_timeout_and_cost_example()
    examples["run"] = damaged
    ref = "arena/%s/sources/sub.tar.gz" % damaged["round_id"]
    obj, row = harness(examples, percent)

    def export(_store, view, run_id):
        if run_id == "run":
            view.artifact_refs["source"] = ref
            if fault == "transient":
                view.transient_failures[ref] = "transient"
        return copy.deepcopy(examples[run_id])

    expected = eager_reference(obj, row, export)
    monkeypatch.setattr(training_reader, "export_run", export)
    assert actual(obj, row) == expected
    assert expected["eligible"] is (percent == 80)


@pytest.mark.parametrize("failure", [KeyError, ValueError, training_reader.ExportError, FileNotFoundError])
def test_whole_export_errors_exclude_only_the_affected_icp(monkeypatch, failure):
    examples = ten_selected_examples()
    obj, row = harness(examples)
    reads = []

    def export(_store, _view, run_id):
        reads.append(run_id)
        if run_id == "run-9":
            raise failure("synthetic exporter error")
        return copy.deepcopy(examples[run_id])

    expected = eager_reference(obj, row, export)
    reads.clear()
    monkeypatch.setattr(training_reader, "export_run", export)
    assert actual(obj, row) == expected
    assert expected["eligible"] is True
    assert expected["capture"]["counts"]["complete_runs"] == 9
    assert expected["capture"]["counts"]["rejected_runs"] == [{
        "run_id": "run-9", "reasons": ["selected_run_export_invalid"], "error_class": failure.__name__}]
    assert reads == list(examples)


@pytest.mark.parametrize("percent", [80, 100])
@pytest.mark.parametrize("missing,eligible", [(1, True), (2, False)])
def test_frozen_missing_reasoning_boundary_matches_eager(monkeypatch, percent, missing, eligible):
    examples = ten_selected_examples()
    for example in examples.values():
        example["configuration"]["training_maximum_missing_reasoning_percent"] = 1
        for index, tokens in enumerate((missing, 40, 60 - missing)):
            example["provider_calls"][index]["response"]["usage"]["completion_tokens_details"]["reasoning_tokens"] = tokens
            refresh_response_links(example, index)
    obj, row = harness(examples, percent, missing_percent=1)

    def export(_store, _view, run_id):
        return copy.deepcopy(examples[run_id])

    expected = eager_reference(obj, row, export)
    monkeypatch.setattr(training_reader, "export_run", export)
    assert actual(obj, row) == expected
    assert expected["eligible"] is eligible
    assert expected["capture"]["counts"]["missing_reasoning_tokens"] == missing * 10
