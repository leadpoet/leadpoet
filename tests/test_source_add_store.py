import pytest

from gateway.research_lab import store


@pytest.mark.asyncio
async def test_persist_source_add_submission_writes_precheck_only_on_latest_stage(monkeypatch):
    rows = []

    async def fake_insert_row(table, row):
        rows.append((table, dict(row)))
        return dict(row)

    monkeypatch.setattr(store, "insert_row", fake_insert_row)

    await store.persist_source_add_submission(
        {
            "submission_id": "source_add_submission:abc",
            "adapter_id": "adapter:abc",
            "miner_hotkey": "hotkey-a",
            "stage": "provenance_precheck_passed",
            "stage_history": ["submitted", "manifest_validated", "provenance_precheck_passed"],
            "measured_trial_yield": -1.0,
            "precheck_status": "provenance_precheck_passed",
            "precheck_doc": {"reasons": ["provenance_reference_backed"]},
        }
    )

    assert [table for table, _row in rows] == ["research_lab_source_add_submissions"] * 3
    assert rows[0][1]["precheck_status"] == ""
    assert rows[0][1]["precheck_doc"] == {}
    assert rows[1][1]["submission_doc"] == {}
    assert rows[2][1]["precheck_status"] == "provenance_precheck_passed"
    assert rows[2][1]["precheck_doc"] == {"reasons": ["provenance_reference_backed"]}
    assert rows[2][1]["submission_doc"]["stage"] == "provenance_precheck_passed"
