import json
import uuid
from zipfile import ZipFile

import httpx
import pytest

from lab_arena import collection_intake as intake
from lab_arena import collection_trajectories as traces
from lab_arena import store


def test_export_can_read_signed_grants_in_a_new_operator_process(monkeypatch):
    monkeypatch.delitem(
        store.FUNCTION_SIGNATURES, "collection_training_consents_v1", raising=False
    )
    batch = "dv-" + "a" * 64

    def consent(request):
        assert request.url.path.endswith("/rpc/collection_training_consents_v1")
        assert json.loads(request.content) == {"p_batch": batch}
        return httpx.Response(200, json=[])

    with httpx.Client(transport=httpx.MockTransport(consent)) as client:
        db = store.PostgrestTransport(
            "https://db.test.invalid",
            service_key="sb_secret_fixture",
            http_client=client,
        )
        assert (
            traces.verified_grants(
                db,
                batch,
                [],
                mode="live",
                scope=intake.TASK_SCOPE,
                digest=intake.PAID_WORKFLOW_TERMS_HASH,
            )
            == {}
        )


def test_first_paid_receipt_reads_only_its_own_job_from_the_real_archive_layout(
    tmp_path,
):
    from scripts.collection_batch import paid_capture_counts

    batch = "dv-" + "b" * 64
    first, other = str(uuid.uuid4()), str(uuid.uuid4())
    path = tmp_path / "training.zip"
    with ZipFile(path, "w") as z:
        for rid, n in [(other, 2), (first, 3)]:
            row = {
                "run_id": rid,
                "batch_id": batch,
                "job_status": "accepted",
                "verified_task_pass": True,
                "verification_method": "verified_p7_v1",
                "length": {"reasoning_bytes": n, "tool_result_bytes": 123},
            }
            z.writestr(f"trajectories/{rid}.jsonl", json.dumps(row) + "\n")
    assert paid_capture_counts(path, first, batch)["reasoning_bytes"] == 3
    with pytest.raises((KeyError, ValueError)):
        paid_capture_counts(path, str(uuid.uuid4()), batch)
    with pytest.raises(ValueError, match="does not prove"):
        paid_capture_counts(path, first, "wrong-offer")
