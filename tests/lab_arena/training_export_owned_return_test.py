"""Returning an owned export must not allocate another whole evidence tree."""

import copy
from types import SimpleNamespace

import pytest

from lab_arena import contracts, training_export, training_input
from tests.lab_arena.training_export_receipts_test import save, values_with_receipts
from tests.lab_arena.training_journal_candidates_test import journal_values


def _values(kind):
    if kind == "partial":
        values, _ = journal_values()
    else:
        values, payload = values_with_receipts()
        snapshot = save(values, payload)
        if kind == "unverified":
            values["runs"][0]["result_doc"]["resource_summary"]["worker_observations"] = {
                "status": "stored", "payload_sha256": "sha256:" + "0" * 64,
                "observation_count": snapshot["payload"]["observation_count"],
                "local_refusal_count": snapshot["payload"]["local_refusal_count"],
                "complete": snapshot["payload"]["complete"],
            }
    values["round_row"]["configuration_doc"]["caller_context"] = {"rows": ["original"]}
    values["input_snapshots"] = {
        row["run_id"]: training_input.read_input(
            values["objects"], row["run_id"], row["kind"],
            row["lease_generation"], row["claim_request_hash"])
        for row in values["runs"]
    }
    return values


def _containers(value, seen=None):
    seen = set() if seen is None else seen
    if not isinstance(value, (dict, list, tuple)) or id(value) in seen:
        return []
    seen.add(id(value))
    children = value.values() if isinstance(value, dict) else value
    result = [value] if isinstance(value, (dict, list)) else []
    for child in children:
        result.extend(_containers(child, seen))
    return result


def _mutate(containers):
    for item in containers:
        if isinstance(item, dict):
            item["ownership_probe"] = ["changed"]
        else:
            item.append({"ownership_probe": "changed"})


def test_export_finishes_when_a_second_whole_export_allocation_is_unavailable(monkeypatch):
    values = _values("complete")
    expected = training_export.assemble_example(**values)
    real_deepcopy = copy.deepcopy

    def bounded_copy(value, *args, **kwargs):
        # Reproduce the observed allocation failure at the sealed final export,
        # without depending on machine RAM, elapsed time or a large fixture.
        if isinstance(value, dict) and {"example_hash", "provider_calls",
                                       "worker_observation_artifacts"} <= value.keys():
            raise MemoryError("second whole export tree is unavailable")
        return real_deepcopy(value, *args, **kwargs)

    monkeypatch.setattr(training_export, "copy", SimpleNamespace(deepcopy=bounded_copy))
    actual = training_export.assemble_example(**values)
    assert actual == expected
    assert actual["completeness"]["complete"]
    assert actual["example_hash"] == contracts.document_hash(
        {key: value for key, value in actual.items() if key != "example_hash"})


@pytest.mark.parametrize("kind,field", [
    ("complete", "worker_observation_artifacts"),
    ("partial", "partial_worker_observation_artifacts"),
    ("unverified", "unverified_worker_observation_artifacts"),
])
def test_owned_export_isolated_from_all_caller_containers_in_both_directions(kind, field):
    values = _values(kind)
    exported = training_export.assemble_example(**values)
    artifact = exported[field]["execute:1"]
    assert artifact["payload_sha256"] == contracts.document_hash(artifact["payload"])
    assert artifact["lease_generation"] == 1
    assert artifact["claim_request_hash"] == values["runs"][0]["claim_request_hash"]
    assert exported["completeness"]["complete"] is (kind == "complete")
    if kind == "partial":
        assert artifact["complete"] is artifact["payload"]["complete"] is False
        assert artifact["segment_refs"] and artifact["segment_sha256"]
    if kind == "unverified":
        assert artifact["complete"] is False
        assert artifact["unverified_reasons"] == ["worker_observation_summary_invalid"]

    caller = [{key: value for key, value in values.items() if key != "objects"},
              values["objects"].values]
    callers, outputs = _containers(caller), _containers(exported)
    assert not ({id(value) for value in callers} & {id(value) for value in outputs})
    expected_export = copy.deepcopy(exported)
    _mutate(callers)
    assert exported == expected_export
    expected_callers = copy.deepcopy(caller)
    _mutate(outputs)
    assert caller == expected_callers


@pytest.mark.parametrize("base", [int, float])
def test_supplied_snapshot_numeric_subclass_remains_isolated(base):
    class Number(base):
        pass

    value = Number(7)
    value.metadata = {"rows": ["original"]}
    values = _values("complete")
    snapshot = values["input_snapshots"]["execute:1"]
    snapshot["input"]["caller_number"] = value
    snapshot["input_sha256"] = contracts.document_hash(snapshot["input"])
    exported = training_export.assemble_example(**values)
    retained = exported["input_snapshots"]["execute:1"]["input"]["caller_number"]
    assert type(retained) is Number and retained == value
    assert retained is not value
    value.metadata["rows"].append("caller")
    assert retained.metadata == {"rows": ["original"]}
    retained.metadata["rows"].append("export")
    assert value.metadata == {"rows": ["original", "caller"]}


@pytest.mark.parametrize("location", ["configuration", "input_snapshot"])
def test_custom_string_keys_keep_caller_attributes_isolated(location):
    class Key(str):
        pass

    key = Key("caller_key")
    key.metadata = {"rows": ["original"]}
    values = _values("complete")
    if location == "configuration":
        values["round_row"]["configuration_doc"][key] = "public"
    else:
        snapshot = values["input_snapshots"]["execute:1"]
        snapshot["input"][key] = "public"
        snapshot["input_sha256"] = contracts.document_hash(snapshot["input"])
    exported = training_export.assemble_example(**values)
    mapping = (exported["configuration"] if location == "configuration" else
               exported["input_snapshots"]["execute:1"]["input"])
    retained = next(item for item in mapping if item == key)
    assert type(retained) is Key and retained is not key
    key.metadata["rows"].append("caller")
    assert retained.metadata == {"rows": ["original"]}
    retained.metadata["rows"].append("export")
    assert key.metadata == {"rows": ["original", "caller"]}


@pytest.mark.parametrize("kind,field", [
    ("complete", "worker_observation_artifacts"),
    ("partial", "partial_worker_observation_artifacts"),
    ("unverified", "unverified_worker_observation_artifacts"),
])
def test_post_privacy_identity_metadata_detaches_custom_strings(kind, field):
    class Text(str):
        pass

    values = _values(kind)
    run = values["runs"][0]
    for name in ("run_id", "kind", "status", "claim_request_hash"):
        value = Text(run[name])
        value.metadata = {"rows": [name]}
        run[name] = value
    exported = training_export.assemble_example(**values)
    artifact_map = exported[field]
    artifact_key = next(key for key in artifact_map if key == run["run_id"])
    gap_map = exported["related_run_gap_attribution"]
    gap_key = next(key for key in gap_map if key == run["run_id"])
    retained = [(artifact_key, run["run_id"]), (gap_key, run["run_id"])]
    retained.extend((gap_map[gap_key][name], run[name]) for name in ("kind", "status"))
    if kind == "partial":
        retained.extend((artifact_map[artifact_key][name], run[name])
                        for name in ("run_id", "kind", "claim_request_hash"))
    for value, original in retained:
        assert type(value) is Text and value is not original
        assert value.metadata == original.metadata
    for original in (run[name] for name in ("run_id", "kind", "status", "claim_request_hash")):
        original.metadata["rows"].append("caller")
    for value, original in retained:
        assert "caller" not in value.metadata["rows"]
        value.metadata["rows"].append("export")
        assert "export" not in original.metadata["rows"]
