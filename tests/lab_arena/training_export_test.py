from __future__ import annotations

import base64
import copy
import json
import io
import random
import stat
import tarfile

import pytest

from lab_arena import contracts, judgment_cache, reviewer_runtime, scoring, training_capture, training_input, verify
from lab_arena.training_export import ExportError, _competition_cost_snapshot, assemble_example, without_hidden_reasoning
from lab_arena.output import validate_output_document
from scripts.export_arena_training import _ledger, export_run, write_private

CLAIM_HASH = "sha256:" + "c" * 64
OUTPUT_REF = "arena/arena-2026-10-16/outputs/execute:1.json"


def save_input(objects, run_id, kind, document, *, generation=1, claim_hash=CLAIM_HASH):
    _, compressed = training_input.build_snapshot(run_id, kind, document,
                                                 lease_generation=generation, claim_request_hash=claim_hash)
    objects.put(training_input.object_ref(run_id, generation), compressed)


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
              "scorer_image_reference": "registry.example/scorer@sha256:" + "a" * 64}
    round_row = {"round_id": "arena-2026-10-16", "configuration_doc": config,
                 "status": "published", "rewards_enabled": True, "arena_network_name": "finney",
                 "benchmark_ref": "benchmark", "evaluation_date": "2026-10-16",
                 "participants": [{"submission_id": "sub"}]}
    archive = io.BytesIO()
    harness = b"def run_icp(icp):\n    return []\n"
    with tarfile.open(fileobj=archive, mode="w:gz") as source_tar:
        member = tarfile.TarInfo("harness.py")
        member.size = len(harness)
        source_tar.addfile(member, io.BytesIO(harness))
    source = archive.getvalue()
    source_ref = "arena/%s/sources/sub.tar.gz" % round_row["round_id"]
    round_row["participants"][0].update(source_ref=source_ref, source_size_bytes=len(source))
    submission = {"submission_id": "sub", "round_id": round_row["round_id"], "miner_hotkey": "miner",
                  "source_ref": source_ref, "source_size_bytes": len(source), "source_archive_sha256": contracts.hash_bytes(source)}
    objects.put(source_ref, source)
    objects.put("benchmark", contracts.canonical_json({"schema_version": "leadpoet.lab_arena.benchmark.v1",
                "round_id": round_row["round_id"], "icps": [icp] * contracts.benchmark_icp_count(config)}).encode())
    resource = {"validator_source_commit": "a" * 40, "validator_source_origin": "git_checkout", "validator_source_dirty": "clean", "provider_call_count": 0, "provider_local_refusal_count": 0, "web_egress": {"connection_count": 0}}
    execute = {"run_id": "execute:1", "round_id": round_row["round_id"], "submission_id": "sub",
               "miner_hotkey": "miner", "stage": 1, "icp_position": 0, "assignment_id": "execute",
               "attempt": 1, "kind": "execute", "status": "accepted", "terminal_cause": "accepted",
               "lease_generation": 1, "claim_request_hash": CLAIM_HASH,
               "output_ref": OUTPUT_REF, "result_doc": {"resource_summary": copy.deepcopy(resource)}}
    judge = dict(execute, run_id="judge:1", kind="score", assignment_id="judge", scored_run_id="execute:1",
                 output_ref="judge-output", result_doc={"resource_summary": copy.deepcopy(resource)})
    output = validate_output_document({"schema_version": contracts.OUTPUT_DOCUMENT_SCHEMA_VERSION, "companies": [company]}, require_intent_dates=False)
    company = output["companies"][0]
    breakdowns = [{"final_score": 60.0}]
    execute["per_icp_score"] = verify.scored_row("sub", 0, execute["run_id"], icp, [company], breakdowns, policy)["per_icp_score"]
    objects.put(OUTPUT_REF, contracts.canonical_json(output).encode())
    objects.put("judge-output", contracts.canonical_json(scoring.build_scoring_output(execute["run_id"], breakdowns)).encode())
    for row, document in ((execute, {"schema_version": "leadpoet.lab_arena.icp_input.v1", "icp": icp, "evaluation_date": "2026-10-16"}),
                          (judge, scoring.build_scoring_input(scored_run_id=execute["run_id"], icp=icp,
                                                             companies=[company], policy=policy, evaluation_date="2026-10-16"))):
        save_input(objects, row["run_id"], row["kind"], document)
    return dict(run_id=execute["run_id"], round_row=round_row, submission_row=submission,
                runs=[execute, judge], ledger_rows=[], objects=objects)


def test_complete_no_external_call_example_is_deterministic_and_has_labels():
    values = fixture()
    first = assemble_example(**values)
    assert first["completeness"]["complete"] is True
    assert first["companies"][0]["label_status"] == "scored"
    assert (first["round_status"], first["rewards_enabled"], first["arena_network_name"]) == ("published", True, "finney")
    assert first["judgment"]["per_icp_score"] == 60
    identity = first["judgment"]["runtime_identity"]
    assert identity["selection_basis"] == "frozen_round"
    assert identity["runtime_image_digest"] == identity["grading_image_digest"] == values["round_row"]["configuration_doc"]["scorer_image_digest"]
    assert assemble_example(**dict(values, runs=list(reversed(values["runs"])))) == first
    unhashed = dict(first)
    del unhashed["example_hash"]
    assert first["example_hash"] == contracts.document_hash(unhashed)


def runtime_binding(values, judge):
    config = values["round_row"]["configuration_doc"]
    judge["reviewer_runtime_doc"] = {
        "schema_version": reviewer_runtime.SCHEMA_VERSION,
        "run_id": judge["run_id"], "lease_generation": judge["lease_generation"],
        "grading_image_digest": config["scorer_image_digest"],
        "grading_image_reference": config["scorer_image_reference"],
        "runtime_image_digest": "sha256:" + "d" * 64,
        "runtime_image_reference": "registry.example/runtime@sha256:" + "d" * 64,
    }
    return judge["reviewer_runtime_doc"]


def test_repaired_reviewer_runtime_is_retained_separately_from_frozen_grading():
    values = fixture()
    binding = runtime_binding(values, values["runs"][1])
    result = assemble_example(**values)
    assert result["completeness"]["complete"]
    identity = result["judgment"]["runtime_identity"]
    assert identity == {"selection_basis": "persisted_lease_binding", **binding}
    assert identity["runtime_image_digest"] != identity["grading_image_digest"]
    assert next(row for row in result["attempts"] if row["kind"] == "score")["runtime_identity"] == identity


@pytest.mark.parametrize("field,value", [
    ("run_id", "wrong-run"), ("lease_generation", 2),
    ("grading_image_digest", "sha256:" + "e" * 64),
    ("runtime_image_reference", "registry.example/runtime@sha256:" + "e" * 64),
])
def test_invalid_reviewer_binding_prevents_complete_export(field, value):
    values = fixture()
    runtime_binding(values, values["runs"][1])[field] = value
    result = assemble_example(**values)
    assert result["completeness"]["dimensions"]["input_output_label"]
    assert not result["privacy"]["exact_replay"]
    assert result["completeness"]["reasons_by_dimension"]["source_identity"] == ["reviewer_runtime_identity_unverified"]
    assert "runtime_identity" not in result["judgment"]


def test_reviewer_completion_audit_must_match_persisted_runtime_binding():
    values = fixture()
    judge = values["runs"][1]
    binding = runtime_binding(values, judge)
    judge["result_doc"]["resource_summary"].update({key: binding[key] for key in reviewer_runtime.AUDIT_FIELDS})
    assert assemble_example(**values)["completeness"]["complete"]
    judge["result_doc"]["resource_summary"]["runtime_image_digest"] = "sha256:" + "e" * 64
    result = assemble_example(**values)
    assert "reviewer_runtime_identity_unverified" in result["completeness"]["incomplete_reasons"]


def test_prior_launched_judge_keeps_its_own_reviewer_runtime_identity():
    values = fixture()
    prior = copy.deepcopy(values["runs"][1])
    prior.update(run_id="prior-judge", status="failed", terminal_cause="judge_timeout", output_ref=None)
    prior["result_doc"]["started_at"] = "2026-10-16T00:00:00Z"
    values["runs"][1]["attempt"] = 2
    binding = runtime_binding(values, prior)
    values["runs"].append(prior)
    document = training_input.read_input(values["objects"], "judge:1", "score", 1, CLAIM_HASH)["input"]
    save_input(values["objects"], prior["run_id"], "score", document)
    result = assemble_example(**values)
    assert result["completeness"]["complete"]
    assert next(row for row in result["attempts"] if row["run_id"] == prior["run_id"])["runtime_identity"] == {"selection_basis": "persisted_lease_binding", **binding}
    binding["lease_generation"] = 2
    result = assemble_example(**values)
    assert result["completeness"]["dimensions"]["source_identity"]
    assert "reviewer_runtime_identity_unverified" in result["completeness"]["reasons_by_dimension"]["research_process"]


def test_missing_snapshot_preserves_verified_labels_but_cannot_claim_full_process():
    values = fixture()
    del values["objects"].values[training_input.object_ref("execute:1", 1)]
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
    assert set(result["completeness"]["incomplete_reasons"]) == {"direct_web_content_not_captured", "provider_capture_count_unverified", "worker_observations_unavailable"}


def add_call(values, *, status="settlement", pointer=True, provider="openrouter", operation="openrouter.chat"):
    run = values["runs"][0]
    run["result_doc"]["resource_summary"]["provider_call_count"] = 1
    identity, hashed_parameters = "sha256:" + "1" * 64, {"model": "test"}
    request_hash = contracts.document_hash(hashed_parameters)
    captured = training_capture.RequestCapture(values["objects"])(
        {"hashed_parameters": hashed_parameters, "request_hash": request_hash, "operation_id": operation,
         "admitted_parameters": {"messages": [{"role": "user", "content": "x" * 12000}]},
         "effective_parameters": {"model": "test"}, "outbound": {"body": {"model": "test"}}})
    context = {key: run[key] for key in ("run_id", "round_id", "submission_id", "miner_hotkey")}
    reserve = dict(context, entry_id=1, entry_kind="reservation", call_identity=identity,
                   provider=provider, operation_id=operation, amount_microusd=0,
                   entry_doc={"request_hash": request_hash, "action_sequence": 0, "base_call_identity": identity,
                              **({"request_capture": captured} if pointer else {})})
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


def test_empty_credential_fields_and_nul_do_not_create_false_export_gaps():
    values = fixture()
    add_call(values)
    body = {"content": "before\x00after", "api_key": None, "authorization": "",
            "credentials": {}, "private_key": []}
    values["ledger_rows"][-1]["terminal_response"]["body_b64"] = base64.b64encode(json.dumps(body).encode()).decode()
    result = assemble_example(**values)
    assert result["provider_calls"][0]["response"] == body
    assert result["provider_calls"][0]["response_privacy_redactions"] == []
    assert result["completeness"]["complete"] is True
    assert result["privacy"]["exact_replay"] is True


def test_nonempty_auth_url_still_prevents_exact_export():
    values = fixture()
    add_call(values)
    auth = "a" * 64
    body = {"url": "https://example.test/file?auth=" + auth}
    values["ledger_rows"][-1]["terminal_response"]["body_b64"] = base64.b64encode(json.dumps(body).encode()).decode()
    result = assemble_example(**values)
    assert "response_privacy_nonexact" in result["completeness"]["incomplete_reasons"]
    assert result["privacy"]["exact_replay"] is False
    assert auth not in json.dumps(result["provider_calls"][0]["response"])


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
    save_input(values["objects"], "execute:1", "execute",
               {"icp": {"icp_id": "different"}, "evaluation_date": "2026-10-16"})
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
    save_input(values["objects"], "execute:1", "execute", {
        "icp": json.loads(values["objects"].values["benchmark"])["icps"][0],
        "evaluation_date": "2026-10-16", "authorization": "secret credential"})
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


def baseline_fixture():
    values = fixture()
    submission = values["submission_row"]
    baseline_id = "baseline-2026-10-16"
    old_ref = submission["source_ref"]
    new_ref = "arena/arena-2026-10-16/sources/%s.tar.gz" % baseline_id
    values["objects"].values[new_ref] = values["objects"].values.pop(old_ref)
    submission.update(submission_id=baseline_id, source_ref=new_ref, is_king=True, status="frozen")
    submission.pop("source_archive_sha256")
    participant = values["round_row"]["participants"][0]
    participant.update(submission_id=baseline_id, source_ref=new_ref, is_king=True)
    values["round_row"]["configuration_doc"]["baseline_hotkey"] = "miner"
    for row in values["runs"]:
        row["submission_id"] = baseline_id
    return values


def test_baseline_without_recorded_digest_uses_valid_frozen_object_authority():
    result = assemble_example(**baseline_fixture())
    assert result["completeness"]["complete"]
    assert result["source"]["hash_origin"] == "computed_from_frozen_object"
    assert result["source"]["recorded_hash_present"] is False
    assert result["source"]["sha256"].startswith("sha256:")


@pytest.mark.parametrize("failure", ["size", "pin", "archive", "unfrozen", "identity"])
def test_baseline_fallback_does_not_accept_wrong_or_unfrozen_source(failure):
    values = baseline_fixture()
    submission = values["submission_row"]
    if failure == "size":
        values["round_row"]["participants"][0]["source_size_bytes"] += 1
    elif failure == "pin":
        values["round_row"]["participants"][0]["source_ref"] = "other"
    elif failure == "archive":
        values["objects"].values[submission["source_ref"]] = b"not an archive"
    elif failure == "unfrozen":
        submission["status"] = "accepted"
    else:
        values["round_row"]["configuration_doc"]["baseline_hotkey"] = "other"
    result = assemble_example(**values)
    assert "source_unverified" in result["completeness"]["incomplete_reasons"]
    assert "source" not in result


def test_runtime_metadata_is_sanitized_at_every_export_location():
    values = fixture()
    secret = "sk-or-v1-" + "a" * 40
    values["runs"][0]["result_doc"]["resource_summary"]["validator_source_commit"] = secret
    result = assemble_example(**values)
    assert secret not in json.dumps(result)
    assert "runtime_source_unverified" in result["completeness"]["incomplete_reasons"]
    assert result["privacy"]["privacy_filtered"]


@pytest.mark.parametrize("origin,dirty,valid", [("git_checkout", "unknown", False),
    ("git_checkout", "dirty", False), (None, "clean", False),
    ("release_marker", "unknown", True), ("verified_archive_environment", "unknown", True)])
def test_runtime_provenance_requires_verified_origin(origin, dirty, valid):
    values = fixture()
    values["runs"][0]["result_doc"]["resource_summary"].update(validator_source_origin=origin, validator_source_dirty=dirty)
    assert assemble_example(**values)["completeness"]["dimensions"]["source_identity"] is valid


def test_provider_retry_receipts_count_one_logical_call_but_keep_every_cost():
    values = fixture()
    add_call(values)
    first = values["ledger_rows"]
    retry_id = "sha256:" + "7" * 64
    retries = [dict(row, entry_id=row["entry_id"] + 3, call_identity=retry_id,
                    entry_doc=dict(row["entry_doc"], provider_attempt=2)) for row in first]
    first.extend(retries)
    result = assemble_example(**values)
    assert result["completeness"]["complete"]
    assert len(result["provider_calls"]) == 2
    assert result["costs"]["confirmed_actual_microusd"] == 84


def test_judge_snapshot_contains_full_output_before_company_goal_slicing():
    values = fixture()
    output = json.loads(values["objects"].values[OUTPUT_REF])
    output["companies"].append(copy.deepcopy(output["companies"][0]))
    values["objects"].put(OUTPUT_REF, contracts.canonical_json(output).encode())
    icp = json.loads(values["objects"].values["benchmark"])["icps"][0]
    policy = values["round_row"]["configuration_doc"]["scorer_policy"]
    save_input(values["objects"], "judge:1", "score", scoring.build_scoring_input(
        scored_run_id="execute:1", icp=icp, companies=output["companies"], policy=policy, evaluation_date="2026-10-16"))
    result = assemble_example(**values)
    assert result["completeness"]["complete"]
    assert [row["label_status"] for row in result["companies"]] == ["scored", "outside_evaluated_slice"]


def test_tool_html_response_is_complete_and_not_a_diagnostic_excerpt():
    values = fixture()
    add_call(values, provider="scrapingdog", operation="scrapingdog.scrape")
    text = "<html>" + "source content " * 1000 + "</html>"
    values["ledger_rows"][-1]["terminal_response"]["body_b64"] = base64.b64encode(text.encode()).decode()
    result = assemble_example(**values)
    assert result["completeness"]["complete"]
    assert result["provider_calls"][0]["response"]["text"] == text


def test_input_from_previous_lease_does_not_claim_exact_reconstruction():
    values = fixture()
    values["runs"][0]["lease_generation"] = 2
    result = assemble_example(**values)
    assert result["completeness"]["dimensions"]["input_output_label"]
    assert "runtime_input_snapshot_missing" in result["completeness"]["incomplete_reasons"]


def test_launched_prior_judge_input_is_required_for_complete_retry_process():
    values = fixture()
    prior = copy.deepcopy(values["runs"][1])
    prior.update(run_id="judge-prior:1", status="failed", terminal_cause="judge_timeout", output_ref=None)
    prior["result_doc"]["started_at"] = "2026-10-16T00:00:00Z"
    values["runs"][1]["attempt"] = 2
    values["runs"].append(prior)
    first = assemble_example(**values)
    assert first["completeness"]["dimensions"]["input_output_label"]
    assert first["completeness"]["dimensions"]["exact_inputs"]
    assert "prior_attempt_input_missing" in first["completeness"]["incomplete_reasons"]
    current_input = training_input.read_input(values["objects"], "judge:1", "score", 1, CLAIM_HASH)["input"]
    save_input(values["objects"], prior["run_id"], "score", current_input)
    second = assemble_example(**values)
    assert second["completeness"]["complete"]
    assert prior["run_id"] in second["input_snapshots"]
    assert {row["lease_generation"] for row in second["attempts"]} == {1}


def test_whole_example_sanitizer_keeps_request_reasoning_control():
    values = fixture()
    add_call(values)
    entry = values["ledger_rows"][0]["entry_doc"]
    parameters = {"model": "test", "reasoning": {"effort": "high"}}
    request_hash = contracts.document_hash(parameters)
    pointer = training_capture.RequestCapture(values["objects"])(
        {"operation_id": "openrouter.chat", "request_hash": request_hash,
         "hashed_parameters": parameters, "admitted_parameters": parameters,
         "effective_parameters": parameters, "outbound": {"body": parameters}})
    entry.update(request_hash=request_hash, request_capture=pointer)
    result = assemble_example(**values)
    assert result["provider_calls"][0]["request"]["effective_parameters"]["reasoning"] == {"effort": "high"}


def test_shuffled_ledger_preserves_action_order_lifecycle_and_reconciled_cost_provenance():
    values = fixture()
    add_call(values)
    base_rows = values["ledger_rows"]
    rows = []
    identities = ["sha256:" + digit * 64 for digit in ("f", "e", "0", "1")]
    for index, (identity, sequence, attempt, row_run) in enumerate(zip(identities, (0, 0, 2, 0), (1, 2, 1, 1),
                                                                    ("execute:1", "execute:1", "execute:1", "judge:1"))):
        offset = index * 4
        for original in base_rows:
            row = copy.deepcopy(original)
            row.update(entry_id=offset + original["entry_id"], run_id=row_run, call_identity=identity,
                       funding_source="host", created_at="2026-10-16T00:%02d:00Z" % (20 - offset - original["entry_id"]))
            row["entry_doc"].update(action_sequence=sequence, provider_attempt=attempt,
                                     base_call_identity=identities[0] if index == 1 else identity)
            if row["entry_kind"] == "settlement":
                row["terminal_response"].update(call_succeeded=True, headers={"content-type": "application/json"},
                    provider_cost={"basis": "openrouter_generation_cost", "units": "0.000042", "unit_name": "usd",
                                   "operation": "openrouter.chat", "request_id": "generation-%d" % index})
            rows.append(row)
        if index == 0:
            # Retain the successful response while later billing establishes
            # its exact charge. The equal payload must appear only once.
            settlement = copy.deepcopy(rows[-1])
            rows[-1].update(entry_kind="uncertain", amount_microusd=100)
            settlement.update(entry_id=4, created_at="2026-10-16T00:30:00Z")
            settlement["entry_doc"].update(openrouter_delayed_reconciliation=True,
                                            reconciled_uncertainty_entry_id=3, late_reconciliation=True)
            rows.append(settlement)
    values["runs"][0]["result_doc"]["resource_summary"]["provider_call_count"] = 2
    values["runs"][1]["result_doc"]["resource_summary"]["provider_call_count"] = 1
    random.Random(42).shuffle(rows)
    values["ledger_rows"] = rows
    result = assemble_example(**values)
    assert result["completeness"]["complete"]
    assert [call["call_identity"] for call in result["provider_calls"]] == identities
    first = result["provider_calls"][0]
    assert first["funding_source"] == "host"
    assert [event["entry_id"] for event in first["lifecycle"]] == [1, 2, 3, 4]
    assert [event["created_at"] for event in first["lifecycle"]] == [
        "2026-10-16T00:19:00Z", "2026-10-16T00:18:00Z", "2026-10-16T00:17:00Z", "2026-10-16T00:30:00Z"]
    assert first["lifecycle"][-1]["accounting"]["reconciled_uncertainty_entry_id"] == 3
    assert first["terminal_entry_id"] == 4
    assert first["provider_cost"]["request_id"] == "generation-0"
    assert first["response_provenance"]["delivery_to_worker"] == "not_proved"
    assert result["costs"]["confirmed_actual_microusd"] == 168
    assert all("response" not in receipt for receipt in first["response_receipts"])
    assert all(receipt["response_ref"] == {"field": "response", "entry_id": 4} for receipt in first["response_receipts"])
    values["ledger_rows"] = list(reversed(rows))
    assert assemble_example(**values) == result


def test_distinct_earlier_response_is_kept_without_claiming_worker_delivery():
    values = fixture()
    add_call(values)
    earlier = copy.deepcopy(values["ledger_rows"][-1])
    earlier.update(entry_kind="uncertain")
    values["ledger_rows"][-1].update(entry_id=4)
    earlier["terminal_response"]["body_b64"] = base64.b64encode(json.dumps({"result": "earlier distinct response"}).encode()).decode()
    values["ledger_rows"].append(earlier)
    result = assemble_example(**values)
    receipts = result["provider_calls"][0]["response_receipts"]
    assert receipts[0]["response"] == {"result": "earlier distinct response"}
    assert receipts[1]["response_ref"]["field"] == "response"
    assert receipts[0]["response_sha256"] != receipts[1]["response_sha256"]


def test_foreign_execution_output_pointer_cannot_supply_another_runs_bytes():
    values = fixture()
    foreign_ref = "arena/arena-2026-10-16/outputs/foreign-execution:1.json"
    values["objects"].put(foreign_ref, values["objects"].values[OUTPUT_REF])
    values["runs"][0]["output_ref"] = foreign_ref
    result = assemble_example(**values)
    assert "execution_output_reference_mismatch" in result["completeness"]["incomplete_reasons"]
    assert result["completeness"]["dimensions"]["input_output_label"] is False
    assert "output" not in result and result["companies"] == []


def test_optional_recorded_execution_hash_detects_changed_valid_json():
    values = fixture()
    original = json.loads(values["objects"].values[OUTPUT_REF])
    values["runs"][0]["output_hash"] = contracts.document_hash(original)
    original["companies"][0]["fit_summary"] = "Different submitted company evidence"
    values["objects"].put(OUTPUT_REF, contracts.canonical_json(original).encode())
    result = assemble_example(**values)
    assert "execution_output_hash_mismatch" in result["completeness"]["incomplete_reasons"]
    assert result["completeness"]["dimensions"]["input_output_label"] is False
    assert "output" not in result


def test_execution_hash_uses_canonical_parsed_document_not_transport_whitespace():
    values = fixture()
    document = json.loads(values["objects"].values[OUTPUT_REF])
    digest = contracts.document_hash(document)
    values["runs"][0]["output_hash"] = digest
    values["objects"].put(OUTPUT_REF, json.dumps(document, indent=4).encode())
    result = assemble_example(**values)
    assert result["completeness"]["complete"]
    assert result["output_identity"] == {"ref": OUTPUT_REF, "document_hash": digest, "recorded_hash_present": True}


@pytest.mark.parametrize("absent_hash", [None, ""])
def test_canonical_legacy_output_does_not_invent_a_recorded_hash(absent_hash):
    values = fixture()
    values["runs"][0]["output_hash"] = absent_hash
    result = assemble_example(**values)
    assert result["completeness"]["complete"]
    assert result["output_identity"]["recorded_hash_present"] is False


def test_round_filters_report_stored_state_without_inferring_reward_authority():
    values = fixture()
    published = assemble_example(**values)
    values["round_row"].update(status="cancelled", rewards_enabled=False, arena_network_name="finney-shadow")
    cancelled = assemble_example(**values)
    assert (cancelled["round_status"], cancelled["rewards_enabled"], cancelled["arena_network_name"]) == (
        "cancelled", False, "finney-shadow")
    assert cancelled["example_id"] == published["example_id"]
    assert cancelled["example_hash"] != published["example_hash"]


def per_icp_cost_fixture():
    """One qualified company earns 12.75 before the separate cost decision."""
    values = baseline_fixture()
    config = values["round_row"]["configuration_doc"]
    config.update(integrity_policy="arena_integrity_v1",
                  sourcing_cost_eligibility_policy=contracts.PER_ICP_SUCCESSFUL_CALLS_COST_POLICY,
                  execution_icp_cap_microusd=4_000_000,
                  cost_per_company_microusd=800_000)
    benchmark = json.loads(values["objects"].values["benchmark"])
    icp = dict(benchmark["icps"][0], max_companies=4)
    benchmark["icps"] = [icp] * len(benchmark["icps"])
    values["objects"].put("benchmark", contracts.canonical_json(benchmark).encode())
    execute, judge = values["runs"]
    output = json.loads(values["objects"].values[OUTPUT_REF])
    breakdown = {"company_index": 0, "company_identity_key": "company-acme",
                 "company_identity_alias_keys": ["company-acme"],
                 "company_qualified": True, "duplicate_company": False,
                 "final_score": 51.0}
    row = verify.scored_row(execute["submission_id"], 0, execute["run_id"], icp,
                            output["companies"], [breakdown], config["scorer_policy"])
    record = scoring.run_scores_for_store({"rows": [row], "integrity_policy": "arena_integrity_v1"},
                                          [execute])[0]
    execute.update(per_icp_score=record["per_icp_score"],
                   qualification_doc=record["qualification_doc"])
    values["objects"].put(judge["output_ref"], contracts.canonical_json(
        scoring.build_scoring_output(execute["run_id"], [breakdown])).encode())
    save_input(values["objects"], execute["run_id"], "execute",
               {"schema_version": "leadpoet.lab_arena.icp_input.v1", "icp": icp,
                "evaluation_date": "2026-10-16"})
    save_input(values["objects"], judge["run_id"], "score",
               scoring.build_scoring_input(scored_run_id=execute["run_id"], icp=icp,
                                           companies=output["companies"], policy=config["scorer_policy"],
                                           evaluation_date="2026-10-16"))
    return values


def per_icp_cost_reader(*, spend=1_145_340, reason="cost_per_company_exceeded",
                        secret=None, round_id="arena-2026-10-16",
                        submission_id="baseline-2026-10-16", icp_position=0):
    unresolved = 42 if reason == "provider_cost_uncertain" else 0
    unresolved_calls = 1 if reason in ("provider_cost_uncertain", "provider_calls_inflight") else 0
    def read(**identity):
        assert identity == {"round_id": round_id,
                            "submission_id": submission_id,
                            "icp_position": icp_position, "qualified_company_count": 1}
        return {"observed_at": "2026-10-18T04:20:00+00:00",
                "eligibility": {"icp_position": icp_position, "eligible": reason == "eligible",
                                "eligibility_reason": reason,
                                "competition_sourcing_microusd": spend,
                                "execution_icp_cap_microusd": 4_000_000,
                                "cost_per_company_cap_microusd": 800_000,
                                "eligibility_cap_microusd": 800_000,
                                "execution": {"successful_microusd": spend - unresolved,
                                              "success_unresolved_microusd": unresolved,
                                              "inflight_calls": int(reason == "provider_calls_inflight"),
                                              "success_unresolved_calls": unresolved_calls},
                                "private_marker": secret}}
    return read


def test_raw_quality_and_export_time_cost_ineligibility_are_separate():
    values = per_icp_cost_fixture()
    secret = "never-export-this-cost-authority-field"
    result = assemble_example(**values, cost_eligibility_reader=per_icp_cost_reader(secret=secret))
    assert result["judgment"]["per_icp_score"] == 12.75
    assert result["companies"][0]["judgment"]["company_qualified"] is True
    decision = result["competition_cost_eligibility"]
    assert decision == {"status": "available", "scope": "export_time_per_icp_sourcing",
                        "source_rpc": "lab_arena_icp_cost_eligibility",
                        "observed_at": "2026-10-18T04:20:00+00:00",
                        "policy": contracts.PER_ICP_SUCCESSFUL_CALLS_COST_POLICY,
                        "icp_position": 0, "qualified_company_count": 1,
                        "competition_sourcing_microusd": 1_145_340,
                        "execution_icp_cap_microusd": 4_000_000,
                        "cost_per_company_cap_microusd": 800_000,
                        "eligibility_cap_microusd": 800_000,
                        "eligible": False, "eligibility_reason": "cost_per_company_exceeded"}
    assert result["costs"]["confirmed_actual_microusd"] == 0
    assert secret not in json.dumps(result)


def test_oct18_baseline_icp7_observed_quality_and_cost_example():
    """The observed 12.75 quality label survives the separate 1,145,340 cost refusal."""
    values = per_icp_cost_fixture()
    run = values["runs"][0]
    run.update(run_id="arena-2026-10-18:baseline-2026-10-18:1:7:1",
               round_id="arena-2026-10-18", submission_id="baseline-2026-10-18",
               icp_position=7)
    icp = json.loads(values["objects"].values["benchmark"])["icps"][0]
    output = json.loads(values["objects"].values[OUTPUT_REF])
    breakdowns = json.loads(values["objects"].values["judge-output"])["breakdowns"]
    scored = verify.scored_row(run["submission_id"], 7, run["run_id"], icp,
                               output["companies"], breakdowns,
                               values["round_row"]["configuration_doc"]["scorer_policy"])
    run["qualification_doc"] = scoring.run_scores_for_store(
        {"rows": [scored], "integrity_policy": "arena_integrity_v1"}, [run]
    )[0]["qualification_doc"]
    decision = _competition_cost_snapshot(
        run=run, configuration=values["round_row"]["configuration_doc"],
        scored=scored, related=[run], reader=per_icp_cost_reader(
            round_id="arena-2026-10-18", submission_id="baseline-2026-10-18",
            icp_position=7))
    assert scored["per_icp_score"] == 12.75
    assert decision["eligible"] is False
    assert decision["competition_sourcing_microusd"] == 1_145_340
    assert decision["eligibility_reason"] == "cost_per_company_exceeded"


def test_eligible_snapshot_is_distinct_from_actual_paid_cost():
    result = assemble_example(**per_icp_cost_fixture(),
                              cost_eligibility_reader=per_icp_cost_reader(spend=500_000, reason="eligible"))
    assert result["competition_cost_eligibility"]["eligible"] is True
    assert result["competition_cost_eligibility"]["competition_sourcing_microusd"] == 500_000
    assert result["judgment"]["per_icp_score"] == 12.75


def test_export_run_wires_one_read_only_cost_rpc_with_verified_qualified_count():
    values = per_icp_cost_fixture()
    calls = []
    class Store:
        def get_run(self, run_id):
            assert run_id == values["run_id"]
            return values["runs"][0]
        def get_round(self, round_id):
            assert round_id == values["round_row"]["round_id"]
            return values["round_row"]
        def get_submission(self, submission_id):
            assert submission_id == values["submission_row"]["submission_id"]
            return values["submission_row"]
        def list_runs(self, round_id):
            assert round_id == values["round_row"]["round_id"]
            return values["runs"]
        def list_ledger(self, **identity):
            assert identity["run_id"] in {row["run_id"] for row in values["runs"]}
            return []
        def icp_cost_eligibility(self, **identity):
            calls.append(identity)
            return per_icp_cost_reader()(**identity)["eligibility"]
    result = export_run(Store(), values["objects"], values["run_id"])
    assert result["competition_cost_eligibility"]["status"] == "available"
    assert result["competition_cost_eligibility"]["source_rpc"] == "lab_arena_icp_cost_eligibility"
    assert calls == [{"round_id": "arena-2026-10-16",
                      "submission_id": "baseline-2026-10-16",
                      "icp_position": 0, "qualified_company_count": 1}]


@pytest.mark.parametrize("spend,reason", [
    (1_145_340, "eligible"), (500_000, "cost_per_company_exceeded")])
def test_contradictory_cost_authority_reply_is_unavailable(spend, reason):
    result = assemble_example(**per_icp_cost_fixture(),
                              cost_eligibility_reader=per_icp_cost_reader(spend=spend, reason=reason))
    decision = result["competition_cost_eligibility"]
    assert decision["status"] == "unavailable"
    assert "eligible" not in decision and "competition_sourcing_microusd" not in decision


def test_missing_or_unfinished_cost_authority_never_invents_an_eligible_zero():
    values = per_icp_cost_fixture()
    missing = assemble_example(**values)["competition_cost_eligibility"]
    assert missing["status"] == "unavailable" and missing["reason"] == "cost_authority_unavailable"
    assert "eligible" not in missing and "competition_sourcing_microusd" not in missing
    values["runs"][1]["status"] = "leased"
    unfinished = assemble_example(**values, cost_eligibility_reader=per_icp_cost_reader())
    assert unfinished["competition_cost_eligibility"]["status"] == "unavailable"
    assert unfinished["competition_cost_eligibility"]["reason"] == "quality_or_attempt_not_final"
    assert "judgment" not in unfinished


def test_persisted_qualification_mismatch_blocks_cost_snapshot_without_changing_raw_label():
    values = per_icp_cost_fixture()
    values["runs"][0]["qualification_doc"]["companies"][0]["company_qualified"] = False
    result = assemble_example(**values, cost_eligibility_reader=per_icp_cost_reader())
    assert result["judgment"]["per_icp_score"] == 12.75
    assert result["competition_cost_eligibility"] == {
        "status": "unavailable", "reason": "persisted_qualification_mismatch",
        "policy": contracts.PER_ICP_SUCCESSFUL_CALLS_COST_POLICY}


def test_verified_cached_judgment_controls_cost_snapshot_and_bad_cache_blocks_it():
    values = per_icp_cost_fixture()
    execute, judge = values["runs"]
    config = values["round_row"]["configuration_doc"]
    output = json.loads(values["objects"].values[OUTPUT_REF])
    icp = json.loads(values["objects"].values["benchmark"])["icps"][0]
    scoring_input = scoring.build_scoring_input(
        scored_run_id=execute["run_id"], icp=icp, companies=output["companies"],
        policy=config["scorer_policy"], evaluation_date="2026-10-16")
    scope = judgment_cache.build_cache_scope(
        scoring_input=scoring_input, round_id=execute["round_id"], network_name="finney",
        netuid=71, scorer_image_digest=config["scorer_image_digest"],
        scorer_image_reference=config["scorer_image_reference"],
        integrity_policy="arena_integrity_v1")
    source_score_run_id = "judge:source"
    source_scored_run_id = "execute:source"
    source_output = scoring.build_scoring_output(
        source_scored_run_id,
        json.loads(values["objects"].values[judge["output_ref"]])["breakdowns"])
    evidence = judgment_cache.build_evidence_snapshot(
        output=source_output, cache_scope=scope,
        source_score_run_id=source_score_run_id,
        source_scored_run_id=source_scored_run_id,
        source_output_ref="arena/arena-2026-10-16/scores/items/judge:source.json",
        source_runner_hotkey="validator-source",
        runner_authority_exclusions=["validator-source"])
    cache_row = {"cache_key": scope["cache_key"],
                 "scoring_input_hash": scope["scoring_input_hash"],
                 "evidence_hash": contracts.document_hash(evidence),
                 "evidence_doc": evidence,
                 "source_score_run_id": source_score_run_id}
    source = dict(judge, run_id=source_score_run_id,
                  scored_run_id=source_scored_run_id,
                  runner_hotkey="validator-source")
    values["runs"].append(source)
    values["cache_rows"] = [cache_row]
    judge.update(judgment_cache_key=scope["cache_key"],
                 judgment_cache_source_run_id=source_score_run_id)
    result = assemble_example(**values, cost_eligibility_reader=per_icp_cost_reader())
    assert result["judgment"]["judgment_cache_source_run_id"] == source_score_run_id
    assert result["competition_cost_eligibility"]["status"] == "available"
    cache_row["evidence_hash"] = "sha256:" + "0" * 64
    def forbidden_reader(**_identity):
        pytest.fail("invalid cached judgment must not call cost authority")
    invalid = assemble_example(**values, cost_eligibility_reader=forbidden_reader)
    assert "judgment" not in invalid
    assert invalid["competition_cost_eligibility"]["status"] == "unavailable"


@pytest.mark.parametrize("reason", ["provider_calls_inflight", "provider_cost_uncertain"])
def test_unsettled_cost_authority_is_explicitly_unavailable(reason):
    result = assemble_example(**per_icp_cost_fixture(),
                              cost_eligibility_reader=per_icp_cost_reader(reason=reason))
    decision = result["competition_cost_eligibility"]
    assert decision["status"] == "unavailable" and decision["reason"] == reason
    assert "eligible" not in decision and "competition_sourcing_microusd" not in decision


def test_failed_call_unknown_fee_does_not_hide_final_successful_sourcing_decision():
    values = per_icp_cost_fixture()
    add_call(values, status="uncertain")
    values["ledger_rows"][-1]["terminal_response"]["status"] = 502
    values["ledger_rows"][-1]["terminal_response"]["body_b64"] = base64.b64encode(
        b'{"error":"provider_unavailable"}').decode()
    values["ledger_rows"][-1]["entry_doc"].update(
        reason="worker_reported",
        call={"call_succeeded": False, "reason": "transport_failure", "provider_status": 502})
    result = assemble_example(**values, cost_eligibility_reader=per_icp_cost_reader(
        spend=500_000, reason="eligible"))
    assert result["provider_calls"][0]["accounting_state"] == "uncertain"
    assert result["provider_calls"][0]["actual_microusd"] is None
    assert result["completeness"]["dimensions"]["costs"] is False
    assert result["competition_cost_eligibility"]["status"] == "available"
    assert result["competition_cost_eligibility"]["eligible"] is True


def test_successful_call_unknown_fee_keeps_successful_sourcing_decision_unavailable():
    values = per_icp_cost_fixture()
    add_call(values, status="uncertain")
    values["ledger_rows"][-1]["entry_doc"].update(
        reason="worker_reported", call={"call_succeeded": True,
                                        "reason": "missing_provider_cost", "provider_status": 200})
    result = assemble_example(**values, cost_eligibility_reader=per_icp_cost_reader(
        reason="provider_cost_uncertain"))
    assert result["provider_calls"][0]["accounting_state"] == "uncertain"
    assert result["completeness"]["dimensions"]["costs"] is False
    decision = result["competition_cost_eligibility"]
    assert decision["status"] == "unavailable"
    assert decision["reason"] == "provider_cost_uncertain"
    assert "eligible" not in decision
