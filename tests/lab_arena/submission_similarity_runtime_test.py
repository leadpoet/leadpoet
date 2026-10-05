from __future__ import annotations

import gzip
import io
import json
import tarfile

import pytest

from lab_arena import broker, code_review, submission_similarity
from lab_arena.code_review_runtime import SubmissionCodeReviewer


def archive(source: str) -> bytes:
    output = io.BytesIO()
    with gzip.GzipFile(fileobj=output, mode="wb", mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w") as tar:
            for name, payload in (
                ("harness.py", b"from agent import run\n"),
                ("agent.py", source.encode()),
            ):
                member = tarfile.TarInfo(name)
                member.size = len(payload)
                tar.addfile(member, io.BytesIO(payload))
    return output.getvalue()


class Objects:
    def __init__(self, values):
        self.values = values

    def get_bounded(self, ref, _limit):
        return self.values[ref]


class Store:
    def __init__(self):
        self.begins = []
        self.finishes = []

    def begin_submission_review(self, *args):
        self.begins.append(args)
        return {"status": "claimed"}

    def finish_submission_review(self, *args):
        self.finishes.append(args)
        return {"status": args[3]}


class Transport:
    def __init__(self, finding=None):
        self.calls = []
        self.finding = finding

    def send(self, **kwargs):
        self.calls.append(kwargs)
        parameters = json.loads(kwargs["body"])
        submitted = json.loads(parameters["messages"][1]["content"])
        body = {
            "model": parameters["model"],
            "usage": {"cost": "0.001"},
            "choices": [{"finish_reason": "stop", "message": {"content": json.dumps({
                "verdict": "reject" if self.finding else "pass",
                "reviewed_files": [item["path"] for item in submitted["submission_files"]],
                "findings": [self.finding(submitted)] if self.finding else [],
                **({"reviewed_comparisons": [
                    item["comparison_id"] for item in submitted["similarity_contexts"]
                ]} if submitted["similarity_contexts"] else {}),
            })}}],
        }
        return broker.ProviderResponse(200, {"content-type": "application/json"}, json.dumps(body).encode())


def reviewer(candidate, references, *, transport=None, credential=None):
    values = {"candidate": candidate}
    entries = []
    for index, (payload, public) in enumerate(references):
        name = f"reference-{index}"
        values[name] = payload
        identity = submission_similarity.inspect_archive(payload)
        entries.append({"row": {
            "submission_id": name,
            "source_ref": name,
            "source_size_bytes": len(payload),
            "source_archive_sha256": identity.archive_sha256,
        }, "source_public": public})
    store = Store()
    transport = transport or Transport()
    credential_calls = []

    def get_credential(row):
        credential_calls.append(row["submission_id"])
        return credential or "miner-only-key"

    price_table = {
        "schema_version": broker.PRICE_TABLE_SCHEMA_VERSION,
        "fetched_at": "2026-10-05T00:00:00Z",
        "source": broker.OPENROUTER_MODELS_URL,
        "models": {code_review.DEFAULT_REVIEW_MODEL: {
            "prompt": "0.000002", "completion": "0.00001", "request": "0",
            "image": "0", "web_search": "0", "internal_reasoning": "0",
        }},
    }
    runtime = SubmissionCodeReviewer(
        store=store, objects=Objects(values), credential_for=get_credential,
        price_table=price_table, transport=transport,
        similarity_references_for=lambda _row: entries,
    )
    identity = submission_similarity.inspect_archive(candidate)
    row = {
        "submission_id": "candidate", "miner_hotkey": "miner",
        "source_ref": "candidate", "source_size_bytes": len(candidate),
        "source_archive_sha256": identity.archive_sha256,
    }
    return runtime, row, store, transport, credential_calls, entries


def test_exact_duplicate_claims_zero_cost_and_never_reads_miner_key():
    source = archive("def run(icp):\n    return icp.industry\n")
    runtime, row, store, transport, credentials, _ = reviewer(source, [(source, False)])
    assert runtime.review(row)["status"] == "rejected"
    assert store.begins[0][3] == 0
    assert store.finishes[0][5] == 0
    assert store.finishes[0][4]["categories"] == ["duplicate_submission"]
    assert transport.calls == []
    assert credentials == []


def test_ambiguous_private_reference_sends_only_candidate_and_one_miner_key():
    old = archive("def run(icp):\n    item = icp.industry\n    return item\n")
    new = archive("def run(icp):\n    target = icp.industry\n    return target\n")
    runtime, row, store, transport, credentials, _ = reviewer(new, [(old, False)])
    assert runtime.review(row)["status"] == "passed"
    assert len(transport.calls) == 1
    assert credentials == ["candidate"]
    assert transport.calls[0]["headers"]["Authorization"] == "Bearer miner-only-key"
    outbound = transport.calls[0]["body"].decode()
    assert "item = icp.industry" not in outbound
    assert "target = icp.industry" in outbound
    assert "reference-0" not in outbound
    assert "item = icp.industry" not in repr(store.finishes)


def test_duplicate_finding_needs_eligible_id_and_high_confidence():
    old = archive("def run(icp):\n    item = icp.industry\n    return item\n")
    new = archive("def run(icp):\n    target = icp.industry\n    return target\n")

    def finding(submitted):
        return {
            "category": "duplicate_submission", "file": "agent.py",
            "evidence": "return target", "explanation": "Only an identifier changed.",
            "comparison_id": submitted["similarity_contexts"][0]["comparison_id"],
            "confidence": 0.99,
        }

    runtime, row, store, transport, _, _ = reviewer(new, [(old, False)], transport=Transport(finding))
    assert runtime.review(row)["status"] == "rejected"
    assert store.finishes[0][4]["categories"] == ["duplicate_submission"]
    assert "return target" not in repr(store.finishes)

    def invalid(submitted):
        result = finding(submitted)
        result["confidence"] = 0.97
        return result

    runtime, row, store, _, _, _ = reviewer(new, [(old, False)], transport=Transport(invalid))
    assert runtime.review(row)["status"] == "error"
    assert store.finishes[0][4]["error_reason"] == "finding_classification"


def test_missing_reference_and_candidate_digest_mismatch_fail_before_paid_call():
    source = archive("def run(icp):\n    return icp.industry\n")
    runtime, row, store, transport, credentials, entries = reviewer(source, [(source, False)])
    entries[0]["row"]["source_ref"] = "missing"
    assert runtime.review(row)["status"] == "error"
    assert store.finishes[0][4]["retryable"] is True
    assert transport.calls == [] and credentials == []

    # Even a confirmed exact match cannot hide a later missing reference.
    runtime, row, store, transport, credentials, entries = reviewer(
        source, [(source, False), (source, False)]
    )
    entries[1]["row"]["source_ref"] = "missing"
    assert runtime.review(row)["status"] == "error"
    assert store.finishes[0][4]["retryable"] is True
    assert transport.calls == [] and credentials == []

    runtime, row, store, transport, credentials, _ = reviewer(source, [])
    row["source_archive_sha256"] = "0" * 64
    assert runtime.review(row)["status"] == "error"
    assert store.finishes[0][4] == {
        "error_code": "code_review_source_digest_mismatch",
        "retryable": False,
        "model": code_review.DEFAULT_REVIEW_MODEL,
        "file_count": 2,
        "source_bytes": len("from agent import run\n") + len("def run(icp):\n    return icp.industry\n"),
    }
    assert transport.calls == [] and credentials == []


def test_meaningful_behavior_change_has_no_duplicate_context():
    old = archive("def run(icp):\n    return icp.industry\n")
    new = archive("def run(icp):\n    return icp.company\n")
    runtime, row, store, transport, _, _ = reviewer(new, [(old, False)])
    assert runtime.review(row)["status"] == "passed"
    request = json.loads(transport.calls[0]["body"])
    submitted = json.loads(request["messages"][1]["content"])
    assert submitted["similarity_contexts"] == []
    assert store.finishes[0][4]["categories"] == []


def test_all_ambiguous_contexts_fit_one_request_and_only_public_source_is_sent():
    candidate = archive("def run(icp):\n    target = icp.industry\n    return target\n")
    references = [
        (archive(f"def run(icp):\n    item{index} = icp.industry\n    return item{index}\n"), index == 0)
        for index in range(4)
    ]
    runtime, row, store, transport, _, _ = reviewer(candidate, references)
    assert runtime.review(row)["status"] == "passed"
    assert len(transport.calls) == 1
    submitted = json.loads(json.loads(transport.calls[0]["body"])["messages"][1]["content"])
    contexts = submitted["similarity_contexts"]
    assert len(contexts) == 4
    assert "public_reference_files" in contexts[0]
    assert all("public_reference_files" not in context for context in contexts[1:])
    assert "item0 = icp.industry" in repr(submitted)
    assert "item1 = icp.industry" not in repr(submitted)
    assert "item0 = icp.industry" not in repr(store.finishes)


def test_public_bootstrap_payload_is_compared_without_object_reference():
    source = archive("def run(icp):\n    return icp.industry\n")
    runtime, row, store, transport, credentials, entries = reviewer(source, [])
    entries.append({
        "row": {"source_size_bytes": len(source)},
        "payload": source,
        "source_public": True,
    })
    assert runtime.review(row)["status"] == "rejected"
    assert store.begins[0][3] == 0
    assert transport.calls == [] and credentials == []


def test_duplicate_verdict_without_comparison_context_is_invalid():
    prepared = code_review.prepare_request(archive("def run(icp):\n    return icp.industry\n"))
    content = {
        "verdict": "reject", "reviewed_files": list(prepared.reviewed_files),
        "findings": [{
            "category": "duplicate_submission", "file": "agent.py",
            "evidence": "return icp.industry", "explanation": "Duplicate claim.",
            "comparison_id": "unknown", "confidence": 1.0,
        }],
    }
    response = {
        "model": prepared.parameters["model"],
        "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(content)}}],
    }
    with pytest.raises(code_review.CodeReviewError) as failure:
        code_review.parse_response(response, prepared)
    assert failure.value.response_reason == "finding_classification"


def test_review_cannot_silently_skip_similarity_comparisons():
    payload = archive("def run(icp):\n    target = icp.industry\n    return target\n")
    old = archive("def run(icp):\n    item = icp.industry\n    return item\n")
    context = submission_similarity.comparison_context(
        submission_similarity.inspect_archive(payload),
        submission_similarity.inspect_archive(old),
        reference_public=False, comparison_id="private-1",
    )
    prepared = code_review.prepare_request(payload, similarity_contexts=(context,))
    for coverage in (None, [], ["unknown"], ["private-1", "private-1"]):
        doc = {"verdict": "pass", "reviewed_files": list(prepared.reviewed_files), "findings": []}
        if coverage is not None:
            doc["reviewed_comparisons"] = coverage
        response = {"model": prepared.parameters["model"], "choices": [{
            "finish_reason": "stop", "message": {"content": json.dumps(doc)},
        }]}
        with pytest.raises(code_review.CodeReviewError):
            code_review.parse_response(response, prepared)
