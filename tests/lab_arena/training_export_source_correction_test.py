"""A source revision is private receipt authority, not a path exemption."""

from copy import deepcopy
import io
import tarfile

import pytest

from lab_arena import contracts, submission_similarity
from lab_arena.training_export import assemble_example
from lab_arena.training_reader import export_run
from tests.lab_arena.training_export_test import baseline_fixture, fixture


COMMIT = "24d9201c3c5fdc86cb1676628774223205c87166"


def corrected():
    values = baseline_fixture()
    run = values["runs"][0]
    submission = values["submission_row"]
    participant = values["round_row"]["participants"][0]
    participant["miner_hotkey"] = run["miner_hotkey"]
    old_ref = submission["source_ref"]
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz", format=tarfile.PAX_FORMAT,
                      pax_headers={"comment": COMMIT}) as archive:
        body = b"def run_icp(icp):\n    return []\n"
        member = tarfile.TarInfo("harness.py")
        member.size = len(body)
        archive.addfile(member, io.BytesIO(body))
    payload = stream.getvalue()
    ref = old_ref.removesuffix(".tar.gz") + "-source-" + COMMIT + ".tar.gz"
    values["objects"].put(ref, payload)
    identity = submission_similarity.inspect_archive(payload)
    source = dict(source_ref=ref, source_size_bytes=len(payload),
                  source_archive_sha256=identity.archive_sha256,
                  source_normalized_sha256=identity.normalized_sha256)
    submission.update(source)
    participant.update(source_ref=ref, source_size_bytes=len(payload))
    config = values["round_row"]["configuration_doc"]
    config.update(baseline_source_url="https://github.com/leadpoet/leadpoet-sales-agent/archive/" + COMMIT + ".tar.gz",
                  reasoning_baseline_source_sha256=identity.archive_sha256)
    proof = dict(operation="baseline_source_supersession_v1",
                 request_id="source-correction-fixture", request_hash="sha256:" + "c" * 64,
                 round_id=run["round_id"], submission_id=run["submission_id"],
                 miner_hotkey=run["miner_hotkey"], **source,
                 baseline_source_url=config["baseline_source_url"],
                 reasoning_baseline_source_sha256=config["reasoning_baseline_source_sha256"],
                 run_ids=[run["run_id"]])
    values["scoring_policy_history"] = dict(round_id=run["round_id"], revisions=[],
                                            source_corrections=[proof])
    return values, proof, old_ref


def test_receipt_bound_source_exports_without_changing_saved_inputs_or_old_object():
    values, proof, old_ref = corrected()
    original = dict(values["objects"].values)
    example = assemble_example(**values)
    assert example["completeness"]["complete"]
    assert example["source"]["sha256"] == proof["source_archive_sha256"]
    assert example["source"]["source_correction"] == {
        "request_id": proof["request_id"], "request_hash": proof["request_hash"], "commit": COMMIT}
    assert example["judgment"]["per_icp_score"] == 60
    assert old_ref in original and values["objects"].values == original
    assert example["example_hash"] == contracts.document_hash(
        {key: value for key, value in example.items() if key != "example_hash"})


@pytest.mark.parametrize("fault", [
    "missing_history", "missing_proofs", "wrong_round", "duplicate_proof", "unnamed_run",
    "duplicate_run", "wrong_operation", "missing_request", "bad_receipt_hash",
    "wrong_submission", "wrong_miner", "wrong_ref", "wrong_size", "wrong_archive_hash",
    "wrong_normalized_hash", "wrong_url", "wrong_config_hash", "changed_bytes",
    "different_commit", "forged_normalized_hash", "participant", "unfrozen", "miner",
])
def test_noncanonical_source_rejects_missing_or_changed_authority(fault):
    values, proof, _ = corrected()
    history = values["scoring_policy_history"]
    if fault == "missing_history": values.pop("scoring_policy_history")
    elif fault == "missing_proofs": history.pop("source_corrections")
    elif fault == "wrong_round": history["round_id"] = "another-round"
    elif fault == "duplicate_proof": history["source_corrections"].append(deepcopy(proof))
    elif fault == "unnamed_run": proof["run_ids"] = ["another-execution"]
    elif fault == "duplicate_run": proof["run_ids"] *= 2
    elif fault == "wrong_operation": proof["operation"] = "unapproved"
    elif fault == "missing_request": proof.pop("request_id")
    elif fault == "bad_receipt_hash": proof["request_hash"] = "unbound"
    elif fault == "wrong_submission": proof["submission_id"] = "another-submission"
    elif fault == "wrong_miner": proof["miner_hotkey"] = "another-miner"
    elif fault == "wrong_ref": proof["source_ref"] += ".other"
    elif fault == "wrong_size": proof["source_size_bytes"] += 1
    elif fault == "wrong_archive_hash": proof["source_archive_sha256"] = "sha256:" + "d" * 64
    elif fault == "wrong_normalized_hash": proof["source_normalized_sha256"] = "sha256:" + "d" * 64
    elif fault == "wrong_url": proof["baseline_source_url"] += "?changed"
    elif fault == "wrong_config_hash": values["round_row"]["configuration_doc"]["reasoning_baseline_source_sha256"] = "sha256:" + "d" * 64
    elif fault == "changed_bytes": values["objects"].values[proof["source_ref"]] += b"tampered"
    elif fault == "different_commit":
        proof["baseline_source_url"] = proof["baseline_source_url"].replace(COMMIT, "d" * 40)
        values["round_row"]["configuration_doc"]["baseline_source_url"] = proof["baseline_source_url"]
    elif fault == "forged_normalized_hash":
        proof["source_normalized_sha256"] = "sha256:" + "d" * 64
        values["submission_row"]["source_normalized_sha256"] = proof["source_normalized_sha256"]
    elif fault == "participant": values["round_row"]["participants"][0]["miner_hotkey"] = "foreign"
    elif fault == "unfrozen": values["submission_row"]["status"] = "pending"
    elif fault == "miner": values["submission_row"]["is_king"] = False
    example = assemble_example(**values)
    assert "source" not in example
    assert "source_unverified" in example["completeness"]["incomplete_reasons"]


class Store:
    def __init__(self, values, history=None):
        self.values, self.history, self.history_reads = values, history, []

    def get_run(self, run_id):
        return next(row for row in self.values["runs"] if row["run_id"] == run_id)

    def get_round(self, round_id):
        assert round_id == self.values["round_row"]["round_id"]
        return self.values["round_row"]

    def get_submission(self, submission_id):
        assert submission_id == self.values["submission_row"]["submission_id"]
        return self.values["submission_row"]

    def list_runs(self, round_id):
        return self.values["runs"]

    def list_ledger(self, **kwargs):
        return []

    def scoring_policy_history(self, round_id):
        self.history_reads.append(round_id)
        return self.history


def test_native_reader_fetches_persisted_source_proof_at_floor_zero():
    values, _, _ = corrected()
    store = Store(values, values.pop("scoring_policy_history"))
    result = export_run(store, values["objects"], values["run_id"])
    assert result["completeness"]["complete"]
    assert store.history_reads == [values["round_row"]["round_id"]]
    store.history = {"round_id": values["round_row"]["round_id"], "revisions": []}
    assert "source_unverified" in export_run(
        store, values["objects"], values["run_id"])["completeness"]["incomplete_reasons"]


@pytest.mark.parametrize("make", [fixture, baseline_fixture])
def test_canonical_sources_preserve_exact_output_and_skip_extra_history_read(make):
    values = make()
    expected = assemble_example(**values)
    store = Store(values)
    actual = export_run(store, values["objects"], values["run_id"])
    assert actual == expected
    assert store.history_reads == []
