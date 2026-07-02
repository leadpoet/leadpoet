#!/usr/bin/env python3
"""Verify OpenRouter reasoning metadata projects into trajectory trace pointers."""

from __future__ import annotations

import asyncio
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gateway.research_lab.trajectory_projector import (  # noqa: E402
    EXECUTION_TRACES_TABLE,
    LOOP_EVENTS_TABLE,
    _collect_engine_trace_calls,
    _merge_trace_calls,
    _upsert_execution_trace_row,
    summarize_reasoning_capture_coverage,
)


class _FakeStore:
    def __init__(self, rows: dict[str, list[dict[str, Any]]] | None = None):
        self.rows = rows or {}
        self.updated: list[tuple[str, dict[str, Any], tuple[Any, ...]]] = []
        self.inserted: list[tuple[str, dict[str, Any]]] = []

    async def select_all(self, table: str, **_kwargs: Any) -> list[dict[str, Any]]:
        return [dict(row) for row in self.rows.get(table, [])]

    async def select_one(self, table: str, *, filters: tuple[Any, ...], **_kwargs: Any) -> dict[str, Any] | None:
        field, value = filters[0]
        for row in self.rows.get(table, []):
            if row.get(field) == value:
                return dict(row)
        return None

    async def insert_row(self, table: str, row: dict[str, Any]) -> dict[str, Any]:
        self.rows.setdefault(table, []).append(dict(row))
        self.inserted.append((table, dict(row)))
        return dict(row)

    async def update_row(self, table: str, values: dict[str, Any], *, filters: tuple[Any, ...]) -> dict[str, Any]:
        field, value = filters[0]
        for index, row in enumerate(self.rows.get(table, [])):
            if row.get(field) == value:
                updated = {**row, **values}
                self.rows[table][index] = updated
                self.updated.append((table, dict(values), filters))
                return dict(updated)
        raise RuntimeError(f"{table}: update returned no rows")


def main() -> int:
    errors: list[str] = []
    loop_events = [
        {
            "event_id": "event-1",
            "run_id": "11111111-1111-4111-8111-111111111111",
            "seq": 1,
            "event_type": "code_edit_validation_failed",
            "node_id": "node-1",
            "provider_usage": [
                {
                    "provider": "openrouter",
                    "model": "test/reasoning-model",
                    "response_id": "resp-1",
                    "call_stage": "code_edit_draft",
                    "loop_iteration": 2,
                    "raw_trace_ref": {
                        "s3_ref": "s3://bucket/traces/resp-1.json.enc",
                        "sha256": "sha256:raw-1",
                    },
                    "reasoning_capture": {
                        "requested": True,
                        "returned": True,
                        "fields_present": ["reasoning", "reasoning_details"],
                        "reasoning_hashes": ["sha256:reason-1", "sha256:details-1"],
                        "reasoning_token_count": 42,
                        "storage_policy": "redacted_bounded_openrouter_message_reasoning",
                    },
                    "reasoning_logs": {
                        "reasoning_hashes": ["sha256:reason-1", "sha256:details-1"],
                        "fields_present": ["reasoning", "reasoning_details"],
                    },
                }
            ],
            "event_doc": {"iteration": 2},
        },
        {
            "event_id": "event-2",
            "run_id": "11111111-1111-4111-8111-111111111111",
            "seq": 2,
            "event_type": "loop_direction_planned",
            "provider_usage": [
                {
                    "provider": "openrouter",
                    "model": "test/metadata-only-model",
                    "response_id": "resp-2",
                    "reasoning_logs": {
                        "reasoning_hash": "sha256:reason-2",
                        "fields_present": ["reasoning"],
                    },
                }
            ],
            "event_doc": {},
        },
    ]
    calls = _collect_engine_trace_calls(loop_events)
    if len(calls) != 2:
        errors.append(f"expected two projected reasoning calls, got {len(calls)}")
    raw_call = next((call for call in calls if call.get("s3_ref")), None)
    metadata_call = next((call for call in calls if call.get("storage_state") == "metadata_only"), None)
    if not raw_call or raw_call.get("reasoning_capture", {}).get("reasoning_token_count") != 42:
        errors.append("raw trace call did not preserve reasoning_capture metadata")
    if not metadata_call or metadata_call.get("call_kind") != "engine_reasoning_metadata":
        errors.append("metadata-only reasoning call was not preserved")
    if metadata_call and metadata_call.get("reasoning_capture", {}).get("reasoning_hashes") != ["sha256:reason-2"]:
        errors.append("metadata-only reasoning hashes were not projected")

    changed, merged = _merge_trace_calls(
        [{"s3_ref": "s3://bucket/traces/resp-1.json.enc", "sha256": "sha256:raw-1"}],
        calls,
    )
    if not changed or len(merged) != 2:
        errors.append("trace call merge did not append only missing calls")
    changed_again, merged_again = _merge_trace_calls(merged, calls)
    if changed_again or len(merged_again) != 2:
        errors.append("trace call merge did not dedupe repeated calls")

    async def _verify_store_merge() -> None:
        store = _FakeStore(
            {
                EXECUTION_TRACES_TABLE: [
                    {
                        "run_id": "trace-row-1",
                        "calls": [
                            {
                                "s3_ref": "s3://bucket/traces/resp-1.json.enc",
                                "sha256": "sha256:raw-1",
                            }
                        ],
                    }
                ]
            }
        )
        status = await _upsert_execution_trace_row(
            store,
            {
                "run_id": "trace-row-1",
                "calls": calls,
            },
        )
        if status != "updated":
            errors.append(f"expected existing trace row to update, got {status}")
        if not store.updated:
            errors.append("existing trace row was not updated during merge")

    async def _verify_coverage() -> None:
        store = _FakeStore(
            {
                LOOP_EVENTS_TABLE: loop_events,
                EXECUTION_TRACES_TABLE: [
                    {
                        "run_id": "trace-row-1",
                        "calls": calls,
                    }
                ],
            }
        )
        coverage = await summarize_reasoning_capture_coverage(store=store)
        if coverage.get("openrouter_calls") != 2:
            errors.append("coverage did not count OpenRouter calls")
        if coverage.get("reasoning_returned") != 2:
            errors.append("coverage did not count returned reasoning")
        if coverage.get("raw_trace_projected") != 1:
            errors.append("coverage did not match raw trace refs into execution_traces")
        if coverage.get("metadata_only_reasoning") != 1:
            errors.append("coverage did not count metadata-only reasoning")

    asyncio.run(_verify_store_merge())
    asyncio.run(_verify_coverage())

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print("research lab reasoning trajectory capture verification passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
