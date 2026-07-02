"""S3 pointer reconciliation + expanded capture coverage (P6/P7).

trajectoryimprovements.md:

* P6 — raw/scorer/in-container trace recorders return OPTIMISTIC pointers
  while uploads run asynchronously; a failed upload leaves a dangling (never
  wrong) reference. The reconciler HEADs every recent pointer and classifies
  it ``verified`` / ``missing`` / ``hash_mismatch`` / ``unchecked`` so the
  coverage report can distinguish "pointer present" from "object verified".

* P7 — the legacy ``--reasoning-coverage`` command covers only the engine
  channel. ``summarize_capture_coverage`` adds scorer traces, in-container
  traces (incl. dropped/truncated), build diagnostics, judge verdicts, and
  negative-example rows, per channel — converging on the target metric:
  % of production LLM interactions ending as verified, projected corpus
  records.

Both entry points are operator commands (see the projector CLI) and are
idempotent / read-only apart from log output.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from typing import Any, Iterable, Mapping, Sequence

from gateway.research_lab.trajectory_projector import (
    EVALUATION_EVENTS_TABLE,
    EVIDENCE_BUNDLES_TABLE,
    EXECUTION_TRACES_TABLE,
    LOOP_EVENTS_TABLE,
    SCORE_BUNDLES_TABLE,
    GatewayProjectorStore,
    _bundle_per_icp_rows,
    _doc,
    _iter_provider_usage_items,
)

logger = logging.getLogger(__name__)

_POINTER_KEY_PAIRS = (
    ("raw_trace_ref", None),  # nested {s3_ref, sha256}
    ("scorer_trace_ref", "scorer_trace_sha256"),
    ("incontainer_trace_ref", "incontainer_trace_sha256"),
    ("diagnostic_artifact_uri", "diagnostic_artifact_hash"),
    ("s3_ref", "sha256"),
)


def _walk_pointers(value: Any, source: str, out: dict[tuple[str, str], dict[str, Any]]) -> None:
    if isinstance(value, Mapping):
        for ref_key, sha_key in _POINTER_KEY_PAIRS:
            ref_value = value.get(ref_key)
            if ref_key == "raw_trace_ref" and isinstance(ref_value, Mapping):
                s3_ref = str(ref_value.get("s3_ref") or "")
                sha256 = str(ref_value.get("sha256") or "")
            elif isinstance(ref_value, str) and ref_value.startswith("s3://"):
                s3_ref = ref_value
                sha256 = str(value.get(sha_key) or "") if sha_key else ""
            else:
                continue
            if s3_ref:
                out.setdefault(
                    (s3_ref, sha256),
                    {"s3_ref": s3_ref, "sha256": sha256, "source": source},
                )
        for item in value.values():
            _walk_pointers(item, source, out)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _walk_pointers(item, source, out)


async def collect_trace_pointers(
    *,
    store: Any | None = None,
    max_rows: int = 2000,
) -> list[dict[str, Any]]:
    """Gather every S3 trace pointer reachable from the recent corpus rows."""
    store = store or GatewayProjectorStore()
    out: dict[tuple[str, str], dict[str, Any]] = {}
    sources = (
        (LOOP_EVENTS_TABLE, "loop_event", "run_id,provider_usage,event_doc,created_at"),
        (EVALUATION_EVENTS_TABLE, "evaluation_event", "run_id,event_doc,created_at"),
        (SCORE_BUNDLES_TABLE, "score_bundle", "run_id,score_bundle_doc,created_at"),
        (EXECUTION_TRACES_TABLE, "execution_trace", "run_id,calls,judge_verdicts,created_at"),
        (EVIDENCE_BUNDLES_TABLE, "evidence_bundle", "bundle_id,snapshots,bundle_doc,created_at"),
    )
    for table, source, columns in sources:
        try:
            rows = await store.select_all(
                table,
                columns=columns,
                filters=(),
                order_by=(("created_at", True),),
                max_rows=max(1, int(max_rows)),
            )
        except Exception as exc:  # noqa: BLE001 - report-only tool
            logger.warning("trace_reconciler_source_unavailable table=%s error=%s", table, str(exc)[:160])
            continue
        for row in rows:
            _walk_pointers(row, source, out)
    return list(out.values())


def _head_object(s3_client: Any, s3_ref: str) -> str:
    bucket, _sep, key = s3_ref[5:].partition("/")
    if not bucket or not key:
        return "invalid_ref"
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return "verified"
    except Exception as exc:  # noqa: BLE001 - classified below
        code = str(getattr(getattr(exc, "response", None), "get", lambda *_: "")("Error") or "")
        text = f"{type(exc).__name__}: {exc}"
        if "404" in text or "Not Found" in text or "NoSuchKey" in text or "404" in code:
            return "missing"
        return "unchecked"


def _hash_object(s3_client: Any, s3_ref: str) -> str:
    bucket, _sep, key = s3_ref[5:].partition("/")
    body = s3_client.get_object(Bucket=bucket, Key=key)["Body"].read()
    return "sha256:" + hashlib.sha256(body).hexdigest()


async def reconcile_trace_pointers(
    *,
    store: Any | None = None,
    s3_client: Any | None = None,
    verify_hash: bool = False,
    max_rows: int = 2000,
) -> dict[str, Any]:
    """P6: verify pointer→object integrity for recent corpus pointers.

    Idempotent and safe to run repeatedly. Returns counts plus the first few
    missing/mismatched refs for operator output; alerts are the caller's job.
    """
    pointers = await collect_trace_pointers(store=store, max_rows=max_rows)
    if s3_client is None:
        try:
            import boto3  # type: ignore

            s3_client = boto3.client("s3")
        except Exception as exc:  # noqa: BLE001
            return {
                "schema_version": "1.0",
                "status": "s3_unavailable",
                "error": str(exc)[:200],
                "pointer_count": len(pointers),
            }

    counts = {"verified": 0, "missing": 0, "hash_mismatch": 0, "unchecked": 0, "invalid_ref": 0}
    by_source: dict[str, dict[str, int]] = {}
    problems: list[dict[str, Any]] = []

    def _check(pointer: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
        status = _head_object(s3_client, pointer["s3_ref"])
        if status == "verified" and verify_hash and pointer.get("sha256"):
            try:
                actual = _hash_object(s3_client, pointer["s3_ref"])
                if actual != pointer["sha256"]:
                    status = "hash_mismatch"
            except Exception:  # noqa: BLE001
                status = "unchecked"
        return dict(pointer), status

    for pointer in pointers:
        checked, status = await asyncio.to_thread(_check, pointer)
        counts[status] = counts.get(status, 0) + 1
        source_counts = by_source.setdefault(str(checked.get("source")), {})
        source_counts[status] = source_counts.get(status, 0) + 1
        if status in {"missing", "hash_mismatch"} and len(problems) < 50:
            problems.append({**checked, "status": status})
    if counts["missing"] or counts["hash_mismatch"]:
        logger.error(
            "research_lab_trace_pointer_reconcile_problems missing=%s hash_mismatch=%s "
            "(dangling pointers mean the optimistic upload failed — the corpus raw "
            "layer is incomplete for these records)",
            counts["missing"],
            counts["hash_mismatch"],
        )
    return {
        "schema_version": "1.0",
        "status": "completed",
        "pointer_count": len(pointers),
        "verify_hash": bool(verify_hash),
        "counts": counts,
        "by_source": by_source,
        "problems": problems,
    }


SHADOW_WINDOWS_TABLE = "research_lab_shadow_monitor_windows"
PROMOTION_EVENTS_TABLE = "research_lab_candidate_promotion_events"


async def erase_evidence_bundle_content(
    bundle_id: str,
    *,
    deletion_request_ref: str,
    store: Any | None = None,
    s3_client: Any | None = None,
    update_row: Any | None = None,
    dry_run: bool = True,
) -> dict[str, Any]:
    """P20 / v5 §8.5 deletion-with-hash-retention for one evidence bundle.

    Deletes the S3 objects the bundle's snapshots point at, KEEPS the row with
    its ``bundle_hash`` and anchor ref, and flips ``verification_state`` to
    ``content_deleted`` with the deletion request recorded. L0 verification of
    the bundle then reports ``hash_attested``
    (research_lab.retention_policy.verification_state_for_bundle). Run the P6
    reconciler afterwards so pointer state and object state never diverge
    silently.
    """
    if not str(deletion_request_ref or "").strip():
        raise ValueError("deletion_request_ref is required for content erasure")
    store = store or GatewayProjectorStore()
    bundle = await store.select_one(
        EVIDENCE_BUNDLES_TABLE, filters=(("bundle_id", str(bundle_id)),)
    )
    if not bundle:
        return {"status": "bundle_not_found", "bundle_id": str(bundle_id)}
    pointers: dict[tuple[str, str], dict[str, Any]] = {}
    _walk_pointers(bundle.get("snapshots"), "evidence_bundle", pointers)
    _walk_pointers(bundle.get("bundle_doc"), "evidence_bundle", pointers)
    refs = sorted({pointer["s3_ref"] for pointer in pointers.values()})
    deleted: list[str] = []
    failed: list[str] = []
    if not dry_run:
        if s3_client is None:
            import boto3  # type: ignore

            s3_client = boto3.client("s3")
        for s3_ref in refs:
            bucket, _sep, key = s3_ref[5:].partition("/")
            try:
                await asyncio.to_thread(s3_client.delete_object, Bucket=bucket, Key=key)
                deleted.append(s3_ref)
            except Exception as exc:  # noqa: BLE001 - report, don't abort
                logger.warning(
                    "research_lab_evidence_erasure_delete_failed ref=%s error=%s",
                    s3_ref,
                    str(exc)[:160],
                )
                failed.append(s3_ref)
        if update_row is None:
            from gateway.research_lab.store import update_row as _store_update_row

            update_row = _store_update_row
        await update_row(
            EVIDENCE_BUNDLES_TABLE,
            {
                "verification_state": "content_deleted",
                "deletion_request_ref": str(deletion_request_ref),
            },
            filters=(("bundle_id", str(bundle_id)),),
        )
    return {
        "status": "dry_run" if dry_run else "content_deleted",
        "bundle_id": str(bundle_id),
        "bundle_hash_retained": str(bundle.get("bundle_hash") or ""),
        "deletion_request_ref": str(deletion_request_ref),
        "objects_targeted": refs,
        "objects_deleted": deleted,
        "objects_failed": failed,
    }


def _shadow_window_id(active_version_id: str) -> str:
    import uuid as _uuid

    return str(
        _uuid.uuid5(_uuid.NAMESPACE_URL, f"research-lab-shadow-window:{active_version_id}")
    )


async def import_shadow_windows(
    *,
    store: Any | None = None,
    get_json: Any | None = None,
    max_events: int = 200,
) -> dict[str, Any]:
    """P18: persist shadow-monitor windows as DB rows keyed to the promotion
    they adjudicate (requires scripts/64).

    The shadow process is read-only by design — this gateway-side importer
    reads its S3 report/state docs and upserts one row per candidate-derived
    merge; a merge whose window docs cannot be located gets an explicit
    ``not_monitored`` row, so "this merge was never monitored" is recordable
    instead of being an absence.
    """
    from gateway.research_lab.shadow_monitor import (
        ShadowMonitorSettings,
        window_report_uri,
        window_state_uri,
        shadow_window_prefix,
    )

    store = store or GatewayProjectorStore()
    settings = ShadowMonitorSettings.from_env()

    if get_json is None:
        def _default_get_json(uri: str) -> Mapping[str, Any] | None:
            try:
                import json as _json

                import boto3  # type: ignore

                bucket, _sep, key = uri[5:].partition("/")
                body = boto3.client("s3").get_object(Bucket=bucket, Key=key)["Body"].read()
                return _json.loads(body.decode("utf-8"))
            except Exception:  # noqa: BLE001 - absence is a classification
                return None

        get_json = _default_get_json

    events = await store.select_many(
        PROMOTION_EVENTS_TABLE,
        columns="promotion_event_id,candidate_id,private_model_version_id,event_type,event_doc,created_at",
        filters=(("event_type", "active_version_created"),),
        order_by=(("created_at", True),),
        limit=max(1, int(max_events)),
    )
    imported = 0
    not_monitored = 0
    skipped = 0
    for event in events:
        version_id = str(event.get("private_model_version_id") or "")
        if not version_id:
            continue
        row: dict[str, Any] = {
            "window_id": _shadow_window_id(version_id),
            "active_version_id": version_id,
            "promotion_event_ref": (
                f"promotion_event:{event.get('promotion_event_id')}"
                if event.get("promotion_event_id")
                else None
            ),
            "window_status": "not_monitored",
            "comparable_day_count": 0,
            "alert_count": 0,
            "window_report_uri": None,
            "window_doc": {"reason": "window_docs_not_found"},
        }
        try:
            prefix = shadow_window_prefix(
                active_version_id=version_id,
                live_manifest_uri=str(
                    (_doc(event) or {}).get("new_model_manifest_uri") or ""
                ),
                settings=settings,
            )
        except Exception:  # noqa: BLE001 - unlocatable window → not_monitored
            prefix = ""
        if prefix:
            report = get_json(window_report_uri(prefix))
            state = None if report else get_json(window_state_uri(prefix))
            doc = report or state
            if isinstance(doc, Mapping):
                alerts = doc.get("alerts") if isinstance(doc.get("alerts"), list) else []
                row.update(
                    {
                        "window_status": (
                            "alerted"
                            if alerts
                            else ("completed" if report else "open")
                        ),
                        "comparable_day_count": int(
                            doc.get("comparable_day_count")
                            or len(doc.get("days") or [])
                            or 0
                        ),
                        "cumulative_shadow_live_diff_points": doc.get(
                            "cumulative_shadow_live_diff_points"
                        ),
                        "mean_shadow_live_diff_points": doc.get(
                            "mean_shadow_live_diff_points"
                        ),
                        "alert_count": len(alerts),
                        "window_report_uri": window_report_uri(prefix),
                        "window_doc": dict(doc),
                    }
                )
        try:
            await store.insert_row(SHADOW_WINDOWS_TABLE, row)
            imported += 1
            if row["window_status"] == "not_monitored":
                not_monitored += 1
        except Exception as exc:  # noqa: BLE001 - duplicates are idempotence
            if "duplicate" in str(exc).lower() or "unique" in str(exc).lower():
                skipped += 1
            else:
                logger.warning(
                    "research_lab_shadow_window_import_failed version=%s error=%s "
                    "(is scripts/64 applied?)",
                    version_id,
                    str(exc)[:160],
                )
    return {
        "schema_version": "1.0",
        "merges_seen": len(events),
        "imported": imported,
        "not_monitored": not_monitored,
        "skipped_existing": skipped,
    }


async def summarize_capture_coverage(
    *,
    store: Any | None = None,
    max_rows: int = 5000,
) -> dict[str, Any]:
    """P7: capture → persistence → projection coverage across all channels."""
    store = store or GatewayProjectorStore()

    loop_events = await store.select_all(
        LOOP_EVENTS_TABLE,
        columns="run_id,event_id,seq,event_type,node_id,provider_usage,event_doc,created_at",
        filters=(),
        order_by=(("created_at", True),),
        max_rows=max(1, int(max_rows)),
    )
    evaluation_events = await store.select_all(
        EVALUATION_EVENTS_TABLE,
        columns="run_id,candidate_id,event_type,event_doc,created_at",
        filters=(),
        order_by=(("created_at", True),),
        max_rows=max(1, int(max_rows)),
    )
    score_bundles = await store.select_all(
        SCORE_BUNDLES_TABLE,
        columns="run_id,score_bundle_id,score_bundle_doc,created_at",
        filters=(),
        order_by=(("created_at", True),),
        max_rows=max(1, int(max_rows)),
    )
    execution_traces = await store.select_all(
        EXECUTION_TRACES_TABLE,
        columns="run_id,status,calls,judge_verdicts,trace_doc,created_at",
        filters=(),
        order_by=(("created_at", True),),
        max_rows=max(1, int(max_rows)),
    )

    # -- captured side ------------------------------------------------------
    engine_raw_captured = 0
    for row in loop_events:
        for item, _model in _iter_provider_usage_items([row.get("provider_usage"), _doc(row)]):
            pointer = item.get("raw_trace_ref")
            if isinstance(pointer, Mapping) and (pointer.get("s3_ref") or pointer.get("sha256")):
                engine_raw_captured += 1

    diagnostics_written: set[str] = set()
    for row in loop_events:
        refs: dict[tuple[str, str], dict[str, Any]] = {}
        _walk_pointers(_doc(row), "loop_event", refs)
        for (s3_ref, _sha) in refs:
            if "/diagnostics/" in s3_ref:
                diagnostics_written.add(s3_ref)

    scorer_captured: set[str] = set()
    for row in evaluation_events:
        raw_refs = _doc(row).get("scorer_trace_refs")
        if isinstance(raw_refs, Mapping):
            for pointer in raw_refs.values():
                if isinstance(pointer, Mapping) and pointer.get("s3_ref"):
                    scorer_captured.add(str(pointer["s3_ref"]))

    incontainer_captured = 0
    incontainer_dropped = 0
    incontainer_truncated = 0
    for row in score_bundles:
        for icp_row in _bundle_per_icp_rows(row):
            ref = str(icp_row.get("incontainer_trace_ref") or "")
            if ref:
                incontainer_captured += 1
            if icp_row.get("incontainer_trace_dropped"):
                incontainer_dropped += 1
            if icp_row.get("incontainer_trace_truncated_count"):
                incontainer_truncated += 1
            scorer_ref = str(icp_row.get("scorer_trace_ref") or "")
            if scorer_ref:
                scorer_captured.add(scorer_ref)

    # -- projected side -----------------------------------------------------
    projected = {
        "engine_raw_calls": 0,
        "incontainer_calls": 0,
        "build_diagnostic_calls": 0,
        "judge_verdicts": 0,
        "crash_rows": 0,
        "dropped_incontainer_calls": 0,
    }
    projected_scorer_refs: set[str] = set()
    projected_diag_refs: set[str] = set()
    for trace in execution_traces:
        if str(trace.get("status")) == "crash":
            projected["crash_rows"] += 1
        calls = trace.get("calls") if isinstance(trace.get("calls"), list) else []
        for call in calls:
            if not isinstance(call, Mapping):
                continue
            kind = str(call.get("call_kind") or "")
            if kind in {"engine_raw_trace", "engine_reasoning_metadata"}:
                projected["engine_raw_calls"] += 1
            elif kind == "incontainer_trace":
                projected["incontainer_calls"] += 1
                if call.get("dropped"):
                    projected["dropped_incontainer_calls"] += 1
            elif kind == "build_diagnostic_artifact":
                projected["build_diagnostic_calls"] += 1
                if call.get("s3_ref"):
                    projected_diag_refs.add(str(call["s3_ref"]))
        verdicts = trace.get("judge_verdicts") if isinstance(trace.get("judge_verdicts"), list) else []
        for verdict in verdicts:
            if isinstance(verdict, Mapping) and verdict.get("verdict_kind") == "scorer_judgment_trace":
                projected["judge_verdicts"] += 1
                if verdict.get("s3_ref"):
                    projected_scorer_refs.add(str(verdict["s3_ref"]))

    return {
        "schema_version": "1.0",
        "source": "research_lab_capture_coverage",
        "rows_scanned": {
            "loop_events": len(loop_events),
            "evaluation_events": len(evaluation_events),
            "score_bundles": len(score_bundles),
            "execution_traces": len(execution_traces),
        },
        "captured": {
            "engine_raw_traces": engine_raw_captured,
            "scorer_traces": len(scorer_captured),
            "incontainer_traces": incontainer_captured,
            "incontainer_dropped": incontainer_dropped,
            "incontainer_truncated": incontainer_truncated,
            "build_diagnostics": len(diagnostics_written),
        },
        "projected": projected,
        "projection_gaps": {
            "scorer_traces_unprojected": len(scorer_captured - projected_scorer_refs),
            "build_diagnostics_unprojected": len(diagnostics_written - projected_diag_refs),
        },
        # P6 caveat, by design: "captured/projected" means pointer present —
        # run reconcile_trace_pointers for object-verified rates.
        "verification_note": "pointer_presence_only; run --reconcile-pointers for object verification",
    }
