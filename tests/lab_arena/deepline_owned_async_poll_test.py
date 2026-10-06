"""Provider job status reads need a start receipt owned by the same run."""

import base64
import json

import pytest

from lab_arena import broker as br
from tests.lab_arena.test_lab_arena_broker import CONTEXT


FLOW = {"poll_actions": ["firecrawl_get_batch_scrape_status"],
        "job_id_paths": ["job_id", "id", "data.id"], "poll_input": "id"}
POLL = {"tool_id": "firecrawl_get_batch_scrape_status", "async_parent": "firecrawl_batch_scrape"}
JOB_ID = "external-batch-job"
CALL = "sha256:" + "a" * 64


def row(entry_id, *, kind="reservation", call_identity=CALL, call_doc=None, terminal=None, **patch):
    result = {"entry_id": entry_id, "run_id": CONTEXT.run_id, "provider": "deepline",
        "operation_id": "deepline.execute", "entry_kind": kind, "call_identity": call_identity,
        "entry_doc": {"call": call_doc or {}} if kind == "uncertain" else call_doc or {},
        "terminal_response": terminal}
    result.update(patch)
    return result


def owner(rows, monkeypatch):
    class Store:
        def __init__(self):
            self.reads = []
        def list_ledger(self, **arguments):
            self.reads.append(arguments)
            assert arguments["run_id"] == CONTEXT.run_id
            assert arguments["operation_id"] == "deepline.execute"
            assert arguments["provider"] == "deepline"
            return [r for r in rows if r["entry_id"] > arguments["after_entry_id"]][:arguments["limit"]]
    store = Store()
    broker = object.__new__(br.Broker)
    broker._store = store
    context = br.RunContext(**dict(CONTEXT.__dict__, deepline_catalog={"fixture": True}))
    monkeypatch.setattr(br.deepline_catalog, "tool_entry", lambda _catalog, _parent: {"async_flow": FLOW})
    return broker, context, store


def test_declared_paths_extract_provider_job_and_ignore_billing_envelope_id():
    document = {"job_id": "native-billing-id", "status": "completed",
        "result": {"data": {"id": JOB_ID}}}
    assert br._deepline_async_job_ids(document, FLOW) == (JOB_ID,)
    assert br._deepline_async_job_ids({"job_id": "native-billing-id", "status": "completed"}, FLOW) == ()
    assert br._deepline_async_job_ids({"toolResponse": {"rawV2": {"data": {"id": JOB_ID}}}}, FLOW) == (JOB_ID,)


@pytest.mark.parametrize("terminal,doc", [
    ({"deepline_async_job_ids": [JOB_ID]}, {}),
    (None, {"deepline_async_job_ids": [JOB_ID]}),
    ({"body_b64": base64.b64encode(json.dumps({"job_id": "native-billing-id",
        "result": {"data": {"id": JOB_ID}}}).encode()).decode()}, {}),
])
def test_settled_and_uncertain_start_receipts_authorize_same_run_poll(terminal, doc, monkeypatch):
    broker, context, store = owner([
        row(1, call_doc={"tool": POLL["async_parent"]}),
        row(2, kind="settlement" if terminal else "uncertain", call_doc=doc, terminal=terminal),
    ], monkeypatch)
    assert broker._owns_deepline_async_job(context, POLL, {"id": JOB_ID}) is True
    assert len(store.reads) == 1


@pytest.mark.parametrize("patch,payload", [
    ({}, {"id": "foreign-job"}), ({}, {"id": JOB_ID, "next": "https://example.com/page"}),
    ({"call_identity": "sha256:" + "b" * 64}, {"id": JOB_ID}),
    ({"run_id": "another-run"}, {"id": JOB_ID}),
    ({"provider": "other-provider"}, {"id": JOB_ID}),
    ({"operation_id": "other.operation"}, {"id": JOB_ID}),
])
def test_foreign_jobs_other_calls_and_paging_urls_are_not_owned(patch, payload, monkeypatch):
    broker, context, _store = owner([
        row(1, call_doc={"tool": POLL["async_parent"]}),
        row(2, kind="uncertain", call_doc={"deepline_async_job_ids": [JOB_ID]}, **patch),
    ], monkeypatch)
    assert broker._owns_deepline_async_job(context, POLL, payload) is False


def test_an_unowned_start_cannot_grant_job_access(monkeypatch):
    broker, context, _store = owner([
        row(1, call_doc={"tool": "unrelated_tool"}),
        row(2, kind="uncertain", call_doc={"deepline_async_job_ids": [JOB_ID]}),
    ], monkeypatch)
    assert broker._owns_deepline_async_job(context, POLL, {"id": JOB_ID}) is False


def test_owned_start_survives_ledger_pagination(monkeypatch):
    rows = [row(1, call_doc={"tool": POLL["async_parent"]})]
    rows += [row(i, kind="settlement", call_identity="other-call") for i in range(2, 1001)]
    rows += [row(1001, kind="uncertain", call_doc={"deepline_async_job_ids": [JOB_ID]})]
    broker, context, store = owner(rows, monkeypatch)
    assert broker._owns_deepline_async_job(context, POLL, {"id": JOB_ID}) is True
    assert [read["after_entry_id"] for read in store.reads] == [0, 1000]
