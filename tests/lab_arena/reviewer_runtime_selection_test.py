"""Runtime repair selects one image while retaining exact grading identities."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from lab_arena import contracts, reviewer_runtime, runner, scoring, trajectory
from lab_arena.service import ServiceError
from tests.lab_arena.test_scorer_image_access import _service


def bound_run(run, configuration):
    run.update(kind="score", lease_generation=1)
    digest = "sha256:" + "d" * 64
    run["reviewer_runtime_doc"] = {
        "schema_version": reviewer_runtime.SCHEMA_VERSION,
        "run_id": run["run_id"], "lease_generation": run["lease_generation"],
        "grading_image_digest": configuration["scorer_image_digest"],
        "grading_image_reference": configuration["scorer_image_reference"],
        "runtime_image_digest": digest,
        "runtime_image_reference": "registry.example/arena/runtime@" + digest,
    }
    return run


def test_image_access_uses_persisted_pin_and_keeps_frozen_round_unchanged():
    service, run, calls = _service()
    frozen = deepcopy(service._test_round["configuration_doc"])
    bound_run(run, frozen)
    image = service.handle_scorer_image_access("run-1", "a" * 64)
    assert calls == [(run["reviewer_runtime_doc"]["runtime_image_reference"],
                      run["reviewer_runtime_doc"]["runtime_image_digest"])]
    assert image["image_digest"] != frozen["scorer_image_digest"]
    assert service._test_round["configuration_doc"] == frozen


@pytest.mark.parametrize("field,value", [
    ("run_id", "other-run"), ("lease_generation", 2),
    ("grading_image_digest", "sha256:" + "e" * 64),
    ("runtime_image_reference", "registry.example/wrong@sha256:" + "e" * 64),
    ("schema_version", "wrong"),
])
def test_image_access_rejects_unbound_runtime(field, value):
    service, run, calls = _service()
    bound_run(run, service._test_round["configuration_doc"])
    run["reviewer_runtime_doc"][field] = value
    with pytest.raises(ServiceError, match="reviewer_runtime_invalid"):
        service.handle_scorer_image_access("run-1", "a" * 64)
    assert calls == []


def test_runner_audit_round_trips_and_bound_completion_requires_exact_audit():
    service, run, _ = _service()
    configuration = service._test_round["configuration_doc"]
    bound_run(run, configuration)
    lease = {**run, **reviewer_runtime.selected_image(run, configuration)}
    metadata = runner._runtime_source_metadata(lease)
    for field in reviewer_runtime.AUDIT_FIELDS:
        assert metadata[field] == run["reviewer_runtime_doc"][field]
    result = {
        "schema_version": contracts.RUN_RESULT_SCHEMA_VERSION,
        "resource_summary": {
            "wall_seconds": 1.0, "cpu_seconds": 1.0, "max_rss_bytes": 1,
            "stdout_bytes": 0, "stderr_bytes": 0, "provider_call_count": 0,
            **metadata,
        },
        "started_at": "2026-10-16T12:00:00Z", "finished_at": "2026-10-16T12:00:01Z",
        "terminal_status": "accepted",
    }
    assert contracts.validate_run_result(result) == result
    reviewer_runtime.validate_completion_audit(run, configuration, result)
    for field in reviewer_runtime.AUDIT_FIELDS:
        bad = deepcopy(result)
        bad["resource_summary"].pop(field)
        with pytest.raises(reviewer_runtime.ReviewerRuntimeError):
            reviewer_runtime.validate_completion_audit(run, configuration, bad)
        bad["resource_summary"][field] = "wrong"
        with pytest.raises(reviewer_runtime.ReviewerRuntimeError):
            reviewer_runtime.validate_completion_audit(run, configuration, bad)


def test_old_leases_and_execution_keep_original_image_without_new_audit():
    service, run, _ = _service()
    configuration = service._test_round["configuration_doc"]
    for kind in ("execute", "score"):
        run["kind"] = kind
        assert reviewer_runtime.selected_image(run, configuration) == {
            "image_digest": configuration["scorer_image_digest"],
            "image_reference": configuration["scorer_image_reference"],
        }
        assert reviewer_runtime.audit_metadata(run) == {}
        reviewer_runtime.validate_completion_audit(run, configuration, {})


def test_service_rejects_partial_composed_runtime_audit_before_output_acceptance():
    service, run, _ = _service()
    bound_run(run, service._test_round["configuration_doc"])
    result = {
        "schema_version": contracts.RUN_RESULT_SCHEMA_VERSION,
        "resource_summary": {"wall_seconds": 1.0, "cpu_seconds": 1.0,
                             "max_rss_bytes": 1, "stdout_bytes": 0,
                             "stderr_bytes": 0, "provider_call_count": 0,
                             "runtime_image_digest": "sha256:" + "d" * 64},
        "started_at": "2026-10-16T12:00:00Z", "finished_at": "2026-10-16T12:00:01Z",
        "terminal_status": "accepted",
    }
    service._request_round = lambda *_a, **_k: (
        {"hotkey": run["runner_hotkey"], "body": {"run_id": run["run_id"], "result": result}},
        service._test_round,
    )
    with pytest.raises(ServiceError, match="reviewer_runtime_audit_invalid"):
        service.handle_complete({})


def test_old_external_validator_can_complete_composed_image_without_new_audit_fields():
    service, run, _ = _service()
    configuration = service._test_round["configuration_doc"]
    configuration["scorer_policy"] = scoring.build_scorer_policy()
    bound_run(run, configuration)
    run.update(scored_run_id="execution-1", icp_position=0)
    original_get_run = service._store.get_run
    service._store.get_run = lambda rid: (
        {"output_ref": "execution.json"} if rid == "execution-1" else original_get_run(rid)
    )
    service._objects = SimpleNamespace(
        get_bounded=lambda *_: json.dumps({
            "schema_version": "leadpoet.lab_arena.output.v1", "companies": [],
        }).encode(), put=lambda *_: None,
    )
    service.evaluation_icps = lambda *_: [{"employee_count": ["51-200"]}]
    completed = []
    service._store.complete_attempt = lambda **kwargs: (
        completed.append(kwargs) or {"status": "accepted", "idempotent": True}
    )
    result = {
        "schema_version": contracts.RUN_RESULT_SCHEMA_VERSION,
        "resource_summary": {"wall_seconds": 1.0, "cpu_seconds": 1.0,
                             "max_rss_bytes": 1, "stdout_bytes": 0,
                             "stderr_bytes": 0, "provider_call_count": 0},
        "started_at": "2026-10-16T12:00:00Z", "finished_at": "2026-10-16T12:00:01Z",
        "terminal_status": "accepted",
    }
    service._request_round = lambda *_a, **_k: (
        {"hotkey": run["runner_hotkey"], "body": {
            "run_id": run["run_id"], "lease_token": "a" * 64, "result": result,
            "output": scoring.build_scoring_output("execution-1", []),
        }}, service._test_round,
    )
    assert service.handle_complete({}) == {"status": "accepted", "idempotent": True}
    assert completed[0]["result"] == result


def test_gateway_records_selected_image_for_old_validator_started_event():
    service, run, _ = _service()
    bound_run(run, service._test_round["configuration_doc"])
    recorded = []
    service._store.append_trajectory_events = lambda _run, _token, events: (
        recorded.extend(events) or {"status": "accepted", "accepted": len(events)}
    )
    event = trajectory.event("runtime.started", {"status": "starting"})
    service.handle_trajectory(run["run_id"], "a" * 64, {"events": [event]})
    assert recorded[0]["content"]["reviewer_runtime_selection"] == {
        key: run["reviewer_runtime_doc"][key] for key in reviewer_runtime.AUDIT_FIELDS
    }
