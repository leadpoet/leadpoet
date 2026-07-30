from __future__ import annotations

from tests.restart_rehearsal.gateway_boundary_service import _matches_filter


def test_json_filters_match_postgrest_text_and_json_semantics() -> None:
    row = {
        "allocation_doc": {
            "historical_compute_fallback": True,
            "historical_compute_fallback_source_epoch": None,
            "reimbursement_allocations": [],
        }
    }

    assert _matches_filter(
        row,
        "allocation_doc->>historical_compute_fallback",
        "eq.true",
    )
    assert _matches_filter(
        row,
        "allocation_doc->>historical_compute_fallback_source_epoch",
        "is.null",
    )
    assert not _matches_filter(
        row,
        "allocation_doc->reimbursement_allocations",
        "not.eq.[]",
    )


def test_json_array_filter_selects_nonempty_compute_allocations() -> None:
    row = {
        "allocation_doc": {
            "reimbursement_allocations": [{"uid": 2}],
        }
    }

    assert _matches_filter(
        row,
        "allocation_doc->reimbursement_allocations",
        "not.eq.[]",
    )
