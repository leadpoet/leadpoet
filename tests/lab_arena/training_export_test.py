from __future__ import annotations

import base64
import copy
import json
import stat

import pytest

from lab_arena import contracts, scoring, training_capture, training_input, verify
from lab_arena.training_export import ExportError, assemble_example, without_hidden_reasoning
from lab_arena.output import validate_output_document
from scripts.export_arena_training import _ledger, write_private


class Objects:
    def __init__(self):
        self.values = {}

    def put(self, ref, raw):
        self.values[ref] = raw

    def get_bounded(self, ref, maximum):
        raw = self.values[ref]
        if len(raw) > maximum:
            raise ValueError("too large")
        return raw


def fixture():
    objects = Objects()
    icp = {"icp_id": "icp-0", "prompt": "Find software companies", "industry": "Software",
           "employee_count": ["51-200"], "intent_signals": ["Product launch"], "max_companies": 1}
    company = {"company_name": "Acme", "company_website": "https://acme.example",
               "industry": "Software", "employee_count": "51-200", "country": "United States",
               "fit_summary": "Software company", "fit_evidence_urls": ["https://acme.example/about"],
               "intent_signals": [{"matched_icp_signal": 0, "description": "Product launch",
                                   "url": "https://acme.example/launch", "date": None,
                                   "snippet": "A new product", "why_now": "Recent launch"}]}
    policy = scoring.build_scorer_policy()
    config = {"scorer_policy": policy, "scorer_image_digest": "sha256:" + "a" * 64,
              "scorer_image_reference": "registry/scorer@sha256:" + "a" * 64}
    round_row = {"round_id": "arena-2026-10-16", "configuration_doc": config,
                 "benchmark_ref": "benchmark", "evaluation_date": "2026-10-16",
                 "participants": [{"submission_id": "sub", "source_ref": "source"}]}
    submission = {"submission_id": "sub", "round_id": round_row["round_id"], "miner_hotkey": "miner",
                  "source_ref": "source", "source_archive_sha256": contracts.hash_bytes(b"source")}
    objects.put("source", b"source")
    objects.put("benchmark", contracts.canonical_json({"schema_version": "leadpoet.lab_arena.benchmark.v1",
                "round_id": round_row["round_id"], "icps": [icp] * contracts.benchmark_icp_count(config)}).encode())
    resource = {"validator_source_commit": "a" * 40, "validator_source_dirty": "clean", "provider_call_count": 0}
    execute = {"run_id": "execute:1", "round_id": round_row["round_id"], "submission_id": "sub",
               "miner_hotkey": "miner", "stage": 1, "icp_position": 0, "assignment_id": "execute",
               "attempt": 1, "kind": "execute", "status": "accepted", "terminal_cause": "accepted",
               "output_ref": "output", "result_doc": {"resource_summary": copy.deepcopy(resource)}}
    judge = dict(execute, run_id="judge:1", kind="score", assignment_id="judge", scored_run_id="execute:1",
                 output_ref="judge-output", result_doc={"resource_summary": copy.deepcopy(resource)})
    output = validate_output_document({"schema_version": contracts.OUTPUT_DOCUMENT_SCHEMA_VERSION, "companies": [company]}, require_intent_dates=False)
    company = output["companies"][0]
    breakdowns = [{"final_score": 60.0}]
    execute["per_icp_score"] = verify.scored_row("sub", 0, execute["run_id"], icp, [company], breakdowns, policy)["per_icp_score"]
    objects.put("output", contracts.canonical_json(output).encode())
    objects.put("judge-output", contracts.canonical_json(scoring.build_scoring_output(execute["run_id"], breakdowns)).encode())
    for row, document in ((execute, {"schema_version": "leadpoet.lab_arena.icp_input.v1", "icp": icp, "evaluation_date": "2026-10-16"}),
                          (judge, scoring.build_scoring_input(scored_run_id=execute["run_id"], icp=icp,
                                                             companies=[company], policy=policy, evaluation_date="2026-10-16"))):
        _, compressed = training_input.build_snapshot(row["run_id"], row["kind"], document)
        objects.put(training_input.object_ref(row["run_id"]), compressed)
    return dict(run_id=execute["run_id"], round_row=round_row, submission_row=submission,
                runs=[execute, judge], ledger_rows=[], objects=objects)


def test_complete_no_external_call_example_is_deterministic_and_has_labels():
    values = fixture()
    first = assemble_example(**values)
    assert first["completeness"]["complete"] is True
    assert first["companies"][0]["label_status"] == "scored"
    assert first["judgment"]["per_icp_score"] == 60
    assert assemble_example(**dict(values, runs=list(reversed(values["runs"])))) == first
    unhashed = dict(first)
    del unhashed["example_hash"]
    assert first["example_hash"] == contracts.document_hash(unhashed)


def test_missing_snapshot_preserves_verified_labels_but_cannot_claim_full_process():
    values = fixture()
    del values["objects"].values[training_input.object_ref("execute:1")]
    result = assemble_example(**values)
    assert result["completeness"]["dimensions"]["input_output_label"] is True
    assert result["completeness"]["dimensions"]["exact_inputs"] is False
    assert result["completeness"]["dimensions"]["research_process"] is False


def test_partial_judge_is_not_a_negative_or_complete_label():
    values = fixture()
    values["runs"][1].update(status="failed", terminal_cause="judge_timeout")
    result = assemble_example(**values)
    assert result["companies"] == []
    assert "final_judge_unavailable" in result["completeness"]["incomplete_reasons"]


def test_ambiguous_judge_and_cross_run_context_fail_closed():
    values = fixture()
    values["runs"].append(dict(values["runs"][1], run_id="judge-duplicate"))
    with pytest.raises(ExportError, match="ambiguous_judge"):
        assemble_example(**values)
    values = fixture()
    values["submission_row"]["miner_hotkey"] = "different"
    with pytest.raises(ExportError, match="context_membership"):
        assemble_example(**values)


def test_direct_web_and_count_gap_are_explicit():
    values = fixture()
    values["runs"][0]["result_doc"]["resource_summary"].update(web_egress={"connection_count": 1}, provider_call_count=1)
    result = assemble_example(**values)
    assert result["completeness"]["dimensions"]["input_output_label"] is True
    assert set(result["completeness"]["incomplete_reasons"]) == {"direct_web_content_not_captured", "provider_capture_count_unverified"}


def add_call(values, *, status="settlement", pointer=True):
    run = values["runs"][0]
    run["result_doc"]["resource_summary"]["provider_call_count"] = 1
    identity, hashed_parameters = "sha256:" + "1" * 64, {"model": "test"}
    request_hash = contracts.document_hash(hashed_parameters)
    captured = training_capture.RequestCapture(values["objects"])(
        {"hashed_parameters": hashed_parameters, "request_hash": request_hash, "operation_id": "openrouter.chat",
         "admitted_parameters": {"messages": [{"role": "user", "content": "x" * 12000}]},
         "effective_parameters": {"model": "test"}, "outbound": {"body": {"model": "test"}}})
    context = {key: run[key] for key in ("run_id", "round_id", "submission_id", "miner_hotkey")}
    reserve = dict(context, entry_id=1, entry_kind="reservation", call_identity=identity,
                   provider="openrouter", operation_id="openrouter.chat", amount_microusd=0,
                   entry_doc={"request_hash": request_hash, **({"request_capture": captured} if pointer else {})})
    body = {"choices": [{"message": {"content": "result", "reasoning_content": "PRIVATE THOUGHT"}}]}
    terminal = dict(reserve, entry_id=3, entry_kind=status, amount_microusd=42,
                    terminal_response={"body_b64": base64.b64encode(json.dumps(body).encode()).decode(), "status": 200})
    values["ledger_rows"] = [reserve, dict(reserve, entry_id=2, entry_kind="dispatch"), terminal]


def test_full_request_not_excerpt_and_hidden_response_reasoning_removed():
    values = fixture()
    add_call(values)
    result = assemble_example(**values)
    assert result["completeness"]["complete"] is True
    assert len(result["provider_calls"][0]["request"]["admitted_parameters"]["messages"][0]["content"]) == 12000
    assert "PRIVATE THOUGHT" not in json.dumps(result)
    assert "hidden_reasoning" in result["provider_calls"][0]["response_privacy_redactions"]
    assert result["privacy"]["hidden_reasoning_removed"] is True
    assert result["costs"]["confirmed_actual_microusd"] == 42


@pytest.mark.parametrize("change,reason", [("pointer", "full_request_unavailable"), ("uncertain", "provider_cost_unresolved"), ("response", "full_response_unavailable")])
def test_call_gaps_are_not_complete(change, reason):
    values = fixture()
    add_call(values, status="uncertain" if change == "uncertain" else "settlement", pointer=change != "pointer")
    if change == "response":
        values["ledger_rows"][-1]["terminal_response"] = {}
    result = assemble_example(**values)
    assert result["completeness"]["complete"] is False
    assert reason in result["completeness"]["incomplete_reasons"]


def test_typed_reasoning_and_nested_hidden_details_are_removed():
    result, removed = without_hidden_reasoning({"output": [{"type": "reasoning", "summary": ["private"]}],
                                              "reasoning_details": ["private"], "reasoning": {"effort": "high"}})
    assert removed and "private" not in json.dumps(result)
    assert result["reasoning"] == {"effort": "high"}


def test_private_artifact_does_not_overwrite(tmp_path):
    path = tmp_path / "example.json"
    write_private(path, {"safe": True})
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    with pytest.raises(FileExistsError):
        write_private(path, {"safe": False})
    assert json.loads(path.read_text()) == {"safe": True}


def test_ledger_reader_pages_and_refuses_nonadvancing_cursor():
    class Store:
        def list_ledger(self, **kwargs):
            return [{"entry_id": 1}] * 1000
    with pytest.raises(ValueError, match="pagination"):
        _ledger(Store(), "run")


def test_request_hash_binding_and_missing_object_fail_closed():
    values = fixture()
    add_call(values)
    values["ledger_rows"][0]["entry_doc"]["request_hash"] = "sha256:" + "f" * 64
    assert "full_request_unavailable" in assemble_example(**values)["completeness"]["incomplete_reasons"]
    values = fixture()
    add_call(values)
    pointer = values["ledger_rows"][0]["entry_doc"]["request_capture"]
    del values["objects"].values[pointer["ref"]]
    assert "full_request_unavailable" in assemble_example(**values)["completeness"]["incomplete_reasons"]


def test_request_privacy_redaction_is_explicit_nonexact_gap():
    values = fixture()
    add_call(values)
    entry = values["ledger_rows"][0]["entry_doc"]
    entry["request_capture"] = training_capture.RequestCapture(values["objects"])(
        {"operation_id": "openrouter.chat", "request_hash": entry["request_hash"],
         "hashed_parameters": {"api_key": "not-an-exportable-key"}})
    result = assemble_example(**values)
    assert "request_privacy_nonexact" in result["completeness"]["incomplete_reasons"]
    assert "not-an-exportable-key" not in json.dumps(result)


def test_exact_snapshot_hash_does_not_override_wrong_frozen_context():
    values = fixture()
    _, compressed = training_input.build_snapshot("execute:1", "execute",
                                                 {"icp": {"icp_id": "different"}, "evaluation_date": "2026-10-16"})
    values["objects"].put(training_input.object_ref("execute:1"), compressed)
    result = assemble_example(**values)
    assert result["completeness"]["dimensions"]["input_output_label"] is True
    assert "runtime_input_context_mismatch" in result["completeness"]["incomplete_reasons"]


def test_unresolved_money_is_not_reported_as_confirmed_actual_cost():
    values = fixture()
    add_call(values, status="uncertain")
    result = assemble_example(**values)
    assert result["provider_calls"][0]["actual_microusd"] is None
    assert result["costs"]["confirmed_actual_microusd"] == 0
    assert result["completeness"]["dimensions"]["costs"] is False


def test_missing_source_identity_does_not_fabricate_or_drop_valid_labels():
    values = fixture()
    values["submission_row"].pop("source_archive_sha256")
    result = assemble_example(**values)
    assert result["completeness"]["dimensions"]["source_identity"] is False
    assert result["completeness"]["dimensions"]["input_output_label"] is True


def test_foreign_ledger_context_is_not_attached_to_example():
    values = fixture()
    add_call(values)
    values["ledger_rows"][0]["submission_id"] = "foreign"
    with pytest.raises(ExportError, match="ledger_context_mismatch"):
        assemble_example(**values)


def test_duplicate_ledger_entries_are_not_counted_twice():
    values = fixture()
    add_call(values)
    values["ledger_rows"].append(copy.deepcopy(values["ledger_rows"][-1]))
    with pytest.raises(ExportError, match="duplicate_ledger_entry"):
        assemble_example(**values)


def test_filtered_input_is_not_exact_but_valid_labels_remain():
    values = fixture()
    _, compressed = training_input.build_snapshot("execute:1", "execute", {
        "icp": json.loads(values["objects"].values["benchmark"])["icps"][0],
        "evaluation_date": "2026-10-16", "authorization": "secret credential"})
    values["objects"].put(training_input.object_ref("execute:1"), compressed)
    result = assemble_example(**values)
    assert result["privacy"]["privacy_filtered"] is True
    assert result["privacy"]["exact_replay"] is False
    assert result["completeness"]["dimensions"]["input_output_label"] is True
    assert "runtime_input_privacy_nonexact" in result["completeness"]["incomplete_reasons"]


def test_read_only_cli_does_not_construct_a_service(monkeypatch, tmp_path, capsys):
    from lab_arena import service, store, wiring
    from scripts.export_arena_training import main

    values = fixture()
    class Store:
        closed = False
        def __init__(self, transport):
            pass
        def get_run(self, run_id):
            return next(row for row in values["runs"] if row["run_id"] == run_id)
        def get_round(self, round_id):
            return values["round_row"]
        def get_submission(self, submission_id):
            return values["submission_row"]
        def list_runs(self, round_id):
            return values["runs"]
        def list_ledger(self, **kwargs):
            return []
        def close(self):
            Store.closed = True

    monkeypatch.setattr(store, "ArenaStore", Store)
    monkeypatch.setattr(store, "PostgrestTransport", lambda *args, **kwargs: None)
    monkeypatch.setattr(service, "S3ObjectStore", lambda *args, **kwargs: values["objects"])
    monkeypatch.setattr(wiring, "build_service_from_environment", lambda *args: pytest.fail("service construction"))
    for name in ("LAB_ARENA_SUPABASE_URL", "LAB_ARENA_SERVICE_KEY", "LAB_ARENA_BUCKET"):
        monkeypatch.setenv(name, "test-only")
    path = tmp_path / "result.json"
    assert main(["--run-id", "execute:1", "--output", str(path)]) == 0
    assert Store.closed
    assert json.loads(path.read_text())["completeness"]["complete"]
    assert "test-only" not in capsys.readouterr().out
