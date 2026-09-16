"""The harness input and output validator use the same frozen round schema."""
from __future__ import annotations

import json
import sys

import pytest

from lab_arena import agent_entrypoint, contact_policy, lab_arena_checkpoint, runner, runtime
from tests.lab_arena.test_lab_arena_runner import (
    BridgingRuntime, FakeApi, lease, make_config,
)


@pytest.mark.parametrize("policies,version", [
    ({}, 1),
    ({"contact_policy": "contacts_v1"}, 2),
    ({"company_quality_policy": "company_quality_v1"}, 3),
    ({"company_quality_policy": "company_quality_v1", "contact_policy": "contacts_v1"}, 4),
    ({"intent_details_policy": "intent_details_v1"}, 5),
    ({"intent_details_policy": "intent_details_v1", "contact_policy": "contacts_v1"}, 5),
])
def test_round_schema_reaches_actual_harness_and_matches_validation(
    tmp_path, monkeypatch, policies, version,
):
    run_lease = lease()
    run_lease.update(policies)
    run_lease["integrity_policy"] = "arena_integrity_v1"
    run_lease["icp"].update({
        "output_schema_version": "untrusted-stale-version",
        "generation_receipt": "PRIVATE_GENERATOR_RECEIPT",
    })
    expected = f"leadpoet.lab_arena.output.v{version}"
    observed = []

    def harness(icp):
        observed.append(icp)
        assert icp["output_schema_version"] == expected
        assert "generation_receipt" not in icp
        for key, value in policies.items():
            assert icp[key] == value
        return []

    monkeypatch.setattr(agent_entrypoint, "_load_run_icp", lambda _: harness)
    monkeypatch.setitem(sys.modules, "lab_arena_checkpoint", lab_arena_checkpoint)

    class HarnessRuntime(BridgingRuntime):
        def run_icp(self, spec, **_):
            document = json.loads((spec.input_dir / runtime.INPUT_FILE_NAME).read_text())
            assert document["output_schema_version"] == expected
            agent_entrypoint.run(
                source_dir=spec.source_dir,
                input_path=spec.input_dir / runtime.INPUT_FILE_NAME,
                output_path=spec.output_path,
            )
            return runtime.fake_result(output_bytes=runtime.read_output(spec))

    api = FakeApi([run_lease])
    (tmp_path / "work").mkdir()
    assert runner.Runner(make_config(tmp_path, api, HarnessRuntime())).run_once() == 1
    assert len(observed) == 1
    completion = api.completions[0]["body"]
    assert completion["result"]["terminal_status"] == "accepted"
    assert completion["output"] == {"schema_version": expected, "companies": []}
    assert expected == contact_policy.output_schema(run_lease)
