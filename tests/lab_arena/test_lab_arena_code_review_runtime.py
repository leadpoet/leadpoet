from __future__ import annotations

import gzip
import io
import json
import tarfile

import pytest

from lab_arena import broker, code_review, contracts
from lab_arena.code_review_runtime import SubmissionCodeReviewer
from lab_arena.submission_runtime import SubmissionProviderKeys


MINER = "5" + "A" * 47
OTHER_MINER = "5" + "B" * 47
MINER_KEY = "sk-or-v1-miner-review-key"
ORGANIZER_KEY = "sk-or-v1-organizer-key"
SUBMISSION = {
    "submission_id": "miner-submission",
    "round_id": "arena-2026-09-10",
    "miner_hotkey": MINER,
    "status": "accepted",
    "source_ref": "private/source.tar.gz",
}


def _archive(members: dict[str, bytes]) -> bytes:
    raw = io.BytesIO()
    with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w") as archive:
            for name, content in members.items():
                info = tarfile.TarInfo(name)
                info.size = len(content)
                archive.addfile(info, io.BytesIO(content))
    return raw.getvalue()


def _source() -> dict[str, bytes]:
    return {
        "harness.py": b"from src.agent import run_icp\n",
        "src/agent.py": b"def run_icp(icp, tools): return tools.search(icp.industry)\n",
        "prompts/evidence.txt": b"Use only facts supported by the supplied pages.\n",
    }


def _price_table():
    return broker.validate_price_table({
        "schema_version": broker.PRICE_TABLE_SCHEMA_VERSION,
        "fetched_at": "2026-09-10T00:00:00Z",
        "source": broker.OPENROUTER_MODELS_URL,
        "models": {
            code_review.DEFAULT_REVIEW_MODEL: {
                "prompt": "0.000002",
                "completion": "0.00001",
                "request": "0",
                "image": "0",
                "web_search": "0",
                "internal_reasoning": "0",
            }
        },
    })


class Objects:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def get_bounded(self, ref, max_bytes):
        self.calls.append((ref, max_bytes))
        return self.payload


class ReviewStore:
    def __init__(self):
        self.begin_calls = []
        self.finish_calls = []
        self.terminal = None

    def begin_submission_review(self, *args):
        self.begin_calls.append(args)
        if self.terminal is not None:
            return {
                "status": "existing",
                "code_review_status": self.terminal,
                "attempt": 1,
            }
        return {
            "status": "claimed",
            "code_review_status": "reviewing",
            "attempt": 1,
            "reserved_microusd": args[3],
        }

    def finish_submission_review(self, *args):
        self.finish_calls.append(args)
        self.terminal = args[3]
        return {
            "status": args[3],
            "code_review_status": args[3],
            "actual_microusd": args[5],
        }


class ReviewTransport:
    def __init__(self, response_builder):
        self.response_builder = response_builder
        self.sent = []

    def send(self, **request):
        self.sent.append(request)
        parameters = json.loads(request["body"].decode("utf-8"))
        response = self.response_builder(parameters)
        return broker.ProviderResponse(
            response.get("status", 200),
            {"content-type": "application/json"},
            json.dumps(response["body"]).encode("utf-8"),
        )


def _review_body(parameters, *, verdict="pass", findings=None, cost="0.001234"):
    supplied = json.loads(parameters["messages"][1]["content"])
    paths = [item["path"] for item in supplied["submission_files"]]
    body = {
        "model": parameters["model"],
        "choices": [{
            "finish_reason": "stop",
            "message": {
                "role": "assistant",
                "content": json.dumps({
                    "verdict": verdict,
                    "summary": "All files reviewed.",
                    "reviewed_files": paths,
                    "findings": [] if findings is None else findings,
                }),
            },
        }],
    }
    if cost is not None:
        body["usage"] = {"cost": cost}
    return body


def _reviewer(payload, transport, *, store=None, credential_for=None):
    store = store or ReviewStore()
    objects = Objects(payload)
    credential_for = credential_for or (lambda row: MINER_KEY)
    reviewer = SubmissionCodeReviewer(
        store=store,
        objects=objects,
        credential_for=credential_for,
        price_table=_price_table(),
        transport=transport,
    )
    row = {**SUBMISSION, "source_size_bytes": len(payload)}
    return reviewer, row, store, objects


def test_runtime_sends_every_file_and_only_the_submitting_miners_key():
    members = _source()
    seen_rows = []
    transport = ReviewTransport(lambda parameters: {"body": _review_body(parameters)})
    reviewer, row, store, objects = _reviewer(
        _archive(members),
        transport,
        credential_for=lambda submitted: seen_rows.append(dict(submitted)) or MINER_KEY,
    )

    result = reviewer.review(row)

    assert result["status"] == "passed"
    assert seen_rows == [row]
    assert len(transport.sent) == 1
    sent = transport.sent[0]
    assert sent["headers"]["Authorization"] == "Bearer " + MINER_KEY
    assert ORGANIZER_KEY not in repr(sent)
    parameters = json.loads(sent["body"].decode("utf-8"))
    assert sent["body"] == contracts.canonical_json(parameters).encode("utf-8")
    supplied = json.loads(parameters["messages"][1]["content"])
    assert {
        item["path"]: item["content"].encode("utf-8")
        for item in supplied["submission_files"]
    } == members
    assert parameters["transforms"] == []
    assert parameters["plugins"] == [
        {"id": "context-compression", "enabled": False}
    ]
    assert store.begin_calls[0][3] > 0
    assert store.finish_calls[0][3] == "passed"
    assert store.finish_calls[0][5] == 1_234
    assert objects.calls == [(row["source_ref"], 10 * 1024 * 1024)]


@pytest.mark.parametrize(
    ("damage", "reason"),
    (
        ("malformed_json", "content_json"),
        ("truncated", "not_finished"),
        ("incomplete_coverage", "coverage"),
        ("missing_cost", None),
        ("wrong_model", "model_mismatch"),
    ),
)
def test_incomplete_or_unaccounted_reviews_never_pass(damage, reason):
    def response(parameters):
        body = _review_body(parameters, cost=None if damage == "missing_cost" else "0.0001")
        if damage == "malformed_json":
            body["choices"][0]["message"]["content"] = "not json"
        elif damage == "truncated":
            body["choices"][0]["finish_reason"] = "length"
        elif damage == "incomplete_coverage":
            document = json.loads(body["choices"][0]["message"]["content"])
            document["reviewed_files"].pop()
            body["choices"][0]["message"]["content"] = json.dumps(document)
        elif damage == "wrong_model":
            body["model"] = "anthropic/not-the-requested-model"
        return {"body": body}

    transport = ReviewTransport(response)
    reviewer, row, store, _objects = _reviewer(_archive(_source()), transport)
    result = reviewer.review(row)

    assert result["status"] == "error"
    assert store.finish_calls[0][3] == "error"
    assert store.finish_calls[0][4]["error_code"].startswith("code_review_") or (
        store.finish_calls[0][4]["error_code"] == "review_response_invalid"
    )
    if reason is None:
        assert "error_reason" not in store.finish_calls[0][4]
    else:
        assert store.finish_calls[0][4]["error_reason"] == reason
        assert set(store.finish_calls[0][4]) == {
            "error_code",
            "error_reason",
            "model",
            "file_count",
            "source_bytes",
        }


def test_preparation_failure_makes_no_api_or_credential_call():
    members = _source()
    members["assets/unreviewable.bin"] = b"valid utf8\x00binary"
    transport = ReviewTransport(lambda _parameters: pytest.fail("API must not be called"))
    credential_calls = []
    reviewer, row, store, _objects = _reviewer(
        _archive(members),
        transport,
        credential_for=lambda submitted: credential_calls.append(submitted) or MINER_KEY,
    )

    result = reviewer.review(row)

    assert result["status"] == "error"
    assert transport.sent == []
    assert credential_calls == []
    assert store.begin_calls[0][3] == 0
    assert store.finish_calls[0][4] == {
        "error_code": "review_source_not_text", "model": code_review.DEFAULT_REVIEW_MODEL,
        "file_count": 0, "source_bytes": 0, "retryable": False,
    }


def test_credential_failure_is_redacted_from_persisted_result():
    secret_detail = "decryption failed for " + MINER_KEY
    transport = ReviewTransport(lambda _parameters: pytest.fail("API must not be called"))

    def fail_with_secret(_submitted):
        raise ValueError(secret_detail)

    reviewer, row, store, _objects = _reviewer(
        _archive(_source()), transport, credential_for=fail_with_secret
    )
    result = reviewer.review(row)

    assert result["status"] == "error"
    assert MINER_KEY not in repr(store.finish_calls)
    assert secret_detail not in repr(store.finish_calls)
    assert store.finish_calls[0][4]["error_code"] == "code_review_provider_unavailable"
    assert "retryable" not in store.finish_calls[0][4]


@pytest.mark.parametrize(
    ("broker_code", "error_code", "retryable"),
    (
        ("broker_unavailable", "code_review_preparation_unavailable", True),
        ("miner_credentials_unavailable", "code_review_provider_authentication", False),
    ),
)
def test_typed_credential_failure_controls_retry_without_dispatch(
    broker_code, error_code, retryable
):
    transport = ReviewTransport(lambda _parameters: pytest.fail("API must not be called"))

    def fail(_submitted):
        raise broker.BrokerError(broker_code)

    reviewer, row, store, _objects = _reviewer(
        _archive(_source()), transport, credential_for=fail
    )
    assert reviewer.review(row)["status"] == "error"
    assert transport.sent == []
    assert store.finish_calls[0][4]["error_code"] == error_code
    assert store.finish_calls[0][4]["retryable"] is retryable
    assert store.finish_calls[0][5] == 0


@pytest.mark.parametrize(
    ("http_status", "error_code", "retryable"),
    (
        (401, "code_review_provider_authentication", False),
        (403, "code_review_provider_authentication", False),
        (402, "code_review_provider_credit", False),
        (400, "code_review_provider_request_rejected", False),
        (408, "code_review_provider_timeout", True),
        (429, "code_review_provider_rate_limited", True),
        (503, "code_review_provider_unavailable", True),
    ),
)
def test_http_failures_persist_only_bounded_classification(
    http_status, error_code, retryable
):
    provider_prose = "private provider diagnostic that must not persist"
    transport = ReviewTransport(lambda _parameters: {
        "status": http_status,
        "body": {"error": {"message": provider_prose}},
    })
    reviewer, row, store, _objects = _reviewer(_archive(_source()), transport)

    assert reviewer.review(row)["status"] == "error"
    document = store.finish_calls[0][4]
    assert document["error_code"] == error_code
    assert document["provider_http_status"] == http_status
    assert document["retryable"] is retryable
    assert provider_prose not in repr(store.finish_calls)
    assert store.finish_calls[0][5] is None


@pytest.mark.parametrize("http_status", [200, 403])
@pytest.mark.parametrize("error_type", ["content_policy_violation", "refusal"])
def test_review_content_refusal_is_not_a_credential_failure(error_type, http_status):
    private_text = "private flagged content"
    transport = ReviewTransport(lambda _parameters: {
        "status": http_status,
        "body": {"error": {
            "code": 403,
            "message": private_text,
            "metadata": {"error_type": error_type, "flagged_input": private_text},
        }},
    })
    reviewer, row, store, _objects = _reviewer(_archive(_source()), transport)

    assert reviewer.review(row)["status"] == "error"
    document = store.finish_calls[0][4]
    assert document["error_code"] == "code_review_provider_request_rejected"
    assert document["provider_http_status"] == 403
    assert document["retryable"] is False
    assert private_text not in repr(store.finish_calls)
    assert store.finish_calls[0][5] is None  # Preserve uncertain accounting.
    assert len(transport.sent) == 1


@pytest.mark.parametrize("error_status", [401, 402, 403, 429, 503])
def test_review_embedded_provider_error_uses_same_typed_diagnostic(error_status):
    from lab_arena import code_review_policy

    transport = ReviewTransport(lambda _parameters: {
        "status": 200,
        "body": {"error": {"code": error_status, "message": "private error"}},
    })
    reviewer, row, store, _objects = _reviewer(_archive(_source()), transport)

    assert reviewer.review(row)["status"] == "error"
    document = store.finish_calls[0][4]
    expected = code_review_policy.provider_http_diagnostic(error_status)
    assert {key: document[key] for key in expected} == expected
    assert "private error" not in repr(store.finish_calls)
    assert store.finish_calls[0][5] is None
    assert len(transport.sent) == 1


def test_non_json_provider_error_does_not_persist_body():
    provider_prose = b"upstream secret prose"

    class NonJsonTransport:
        def send(self, **_request):
            return broker.ProviderResponse(
                503, {"content-type": "text/plain"}, provider_prose
            )

    reviewer, row, store, _objects = _reviewer(
        _archive(_source()), NonJsonTransport()
    )
    assert reviewer.review(row)["status"] == "error"
    assert store.finish_calls[0][4]["error_code"] == "code_review_provider_unavailable"
    assert store.finish_calls[0][4]["retryable"] is True
    assert provider_prose.decode() not in repr(store.finish_calls)


def test_idempotent_claim_never_dispatches_or_finishes_again():
    class IdempotentStore(ReviewStore):
        def begin_submission_review(self, *args):
            self.begin_calls.append(args)
            return {
                "status": "claimed",
                "idempotent": True,
                "code_review_status": "reviewing",
                "attempt": 1,
            }

    transport = ReviewTransport(
        lambda _parameters: pytest.fail("idempotent claim must not dispatch")
    )
    store = IdempotentStore()
    reviewer, row, _store, _objects = _reviewer(
        _archive(_source()), transport, store=store
    )
    result = reviewer.review(row)
    assert result["idempotent"] is True
    assert transport.sent == []
    assert store.finish_calls == []


def test_terminal_review_is_not_dispatched_or_charged_twice():
    transport = ReviewTransport(lambda parameters: {"body": _review_body(parameters)})
    store = ReviewStore()
    reviewer, row, _store, _objects = _reviewer(
        _archive(_source()), transport, store=store
    )

    first = reviewer.review(row)
    second = reviewer.review(row)

    assert first["status"] == "passed"
    assert second == {
        "status": "existing",
        "code_review_status": "passed",
        "attempt": 1,
    }
    assert len(transport.sent) == 1
    assert len(store.finish_calls) == 1


class KeyStore:
    def __init__(self, row, *, round_row=None, encrypted=None):
        self.row = row
        self.round_row = round_row or {}
        self.encrypted = encrypted
        self.credential_calls = []

    def get_submission(self, submission_id):
        return self.row if self.row and self.row["submission_id"] == submission_id else None

    def get_round(self, round_id):
        return self.round_row

    def get_submission_credential(self, submission_id, miner_hotkey, provider):
        self.credential_calls.append((submission_id, miner_hotkey, provider))
        return self.encrypted


class KeyManager:
    def __init__(self, value=MINER_KEY):
        self.value = value
        self.calls = []

    def runtime_key(self, encrypted, provider):
        self.calls.append((encrypted, provider))
        return self.value


def test_code_review_key_resolves_only_the_submission_owners_openrouter_key():
    encrypted = {
        "submission_id": SUBMISSION["submission_id"],
        "miner_hotkey": MINER,
        "provider": "openrouter",
        "ciphertext_b64": "opaque",
    }
    store = KeyStore(dict(SUBMISSION), encrypted=encrypted)
    manager = KeyManager()
    keys = SubmissionProviderKeys(
        store=store,
        credentials=manager,
        organizer_keys={"openrouter": ORGANIZER_KEY},
    )

    assert keys.code_review_key(SUBMISSION) == MINER_KEY
    assert store.credential_calls == [
        (SUBMISSION["submission_id"], MINER, "openrouter")
    ]
    assert manager.calls == [(encrypted, "openrouter")]


@pytest.mark.parametrize("change", ("wrong_hotkey", "unaccepted", "baseline"))
def test_code_review_key_never_falls_back_to_organizer_credentials(change):
    row = dict(SUBMISSION)
    submitted = dict(SUBMISSION)
    round_row = {}
    if change == "wrong_hotkey":
        submitted["miner_hotkey"] = OTHER_MINER
    elif change == "unaccepted":
        row["status"] = "rejected"
    else:
        row.update({
            "submission_id": "baseline-2026-09-10",
            "is_king": True,
        })
        submitted = dict(row)
        round_row = {"configuration_doc": {"baseline_hotkey": MINER}}
    store = KeyStore(row, round_row=round_row)
    manager = KeyManager()
    keys = SubmissionProviderKeys(
        store=store,
        credentials=manager,
        organizer_keys={"openrouter": ORGANIZER_KEY},
    )

    with pytest.raises(broker.BrokerError, match="miner_credentials_unavailable"):
        keys.code_review_key(submitted)
    assert store.credential_calls == []
    assert manager.calls == []
