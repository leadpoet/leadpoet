#!/usr/bin/env python3
"""Replay historical candidate images and report dev/live score calibration.

This command is deliberately read-only. It selects immutable candidate image
manifests and realized calibration rows from Supabase, evaluates the images
against one verified development snapshot with Docker networking disabled, and
writes only a local JSON report (or stdout). It never calls a provider and
never inserts, updates, or deletes database rows.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _finite(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _latest_unique_calibrations(
    rows: Sequence[Mapping[str, Any]], *, limit: int
) -> list[dict[str, Any]]:
    """Return newest finite realized row per candidate without double counting."""
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in sorted(
        rows,
        key=lambda item: str(item.get("created_at") or ""),
        reverse=True,
    ):
        candidate_id = str(row.get("candidate_id") or "").strip()
        if not candidate_id or candidate_id in seen:
            continue
        if _finite(row.get("realized_mean_delta")) is None:
            continue
        seen.add(candidate_id)
        selected.append(dict(row))
        if len(selected) >= limit:
            break
    return selected


async def _load_inputs(limit: int) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    from gateway.research_lab import store

    scan_limit = max(1000, limit * 20)
    calibration_rows = await store.select_all(
        "research_lab_score_calibration",
        columns=(
            "candidate_id,run_id,node_id,lane,score_bundle_id,"
            "realized_mean_delta,realized_delta_lcb,created_at"
        ),
        filters=(),
        order_by=(("created_at", True),),
        batch_size=500,
        max_rows=scan_limit,
        allow_partial=True,
    )
    selected = _latest_unique_calibrations(calibration_rows, limit=limit)
    candidate_ids = [row["candidate_id"] for row in selected]
    candidates: dict[str, dict[str, Any]] = {}
    for offset in range(0, len(candidate_ids), 100):
        batch_ids = candidate_ids[offset : offset + 100]
        rows = await store.select_all(
            "research_lab_candidate_artifacts",
            columns=(
                "candidate_id,candidate_kind,candidate_model_manifest_doc,"
                "candidate_model_manifest_hash,created_at"
            ),
            filters=(("candidate_id", "in", batch_ids),),
            order_by=(("created_at", True),),
            batch_size=100,
            max_rows=max(100, len(batch_ids) + 1),
            allow_partial=True,
        )
        for row in rows:
            candidate_id = str(row.get("candidate_id") or "")
            if candidate_id:
                candidates[candidate_id] = dict(row)
    return selected, candidates


def _candidate_for_replay(
    calibration: Mapping[str, Any], candidate_row: Mapping[str, Any]
) -> Any:
    from research_lab.eval.artifacts import (
        PrivateModelArtifactManifest,
        validate_private_model_artifact_manifest,
    )

    if str(candidate_row.get("candidate_kind") or "") != "image_build":
        raise ValueError("candidate is not an immutable image_build artifact")
    manifest_doc = candidate_row.get("candidate_model_manifest_doc")
    if not isinstance(manifest_doc, Mapping):
        raise ValueError("candidate is missing candidate_model_manifest_doc")
    artifact = PrivateModelArtifactManifest.from_mapping(manifest_doc)
    errors = validate_private_model_artifact_manifest(artifact)
    if errors:
        raise ValueError("invalid candidate manifest: " + ",".join(errors))
    return SimpleNamespace(
        node_id=str(calibration.get("node_id") or calibration.get("candidate_id") or ""),
        iteration=0,
        draft=SimpleNamespace(lane=str(calibration.get("lane") or "")),
        build=SimpleNamespace(candidate_model_manifest=artifact),
    )


async def _evaluate_one(
    *,
    evaluator: Any,
    calibration: Mapping[str, Any],
    candidate_row: Mapping[str, Any],
) -> dict[str, Any]:
    candidate_id = str(calibration.get("candidate_id") or "")
    try:
        candidate = _candidate_for_replay(calibration, candidate_row)
        result = dict(await evaluator(candidate))
    except Exception as exc:  # noqa: BLE001 - report each unusable historical row
        return {
            "candidate_id": candidate_id,
            "eligible": False,
            "reason": f"{type(exc).__name__}:{str(exc)[:300]}",
        }
    replay_score = _finite(result.get("aggregate_dev_score"))
    eligible = bool(result.get("eligible")) and replay_score is not None
    return {
        "candidate_id": candidate_id,
        "run_id": str(calibration.get("run_id") or ""),
        "node_id": str(calibration.get("node_id") or ""),
        "lane": str(calibration.get("lane") or ""),
        "score_bundle_id": str(calibration.get("score_bundle_id") or ""),
        "image_digest": str(candidate.build.candidate_model_manifest.image_digest),
        "manifest_hash": str(candidate.build.candidate_model_manifest.manifest_hash),
        "eligible": eligible,
        "reason": str(result.get("eligibility_reason") or ""),
        "replayed_dev_score": replay_score,
        "realized_mean_delta": _finite(calibration.get("realized_mean_delta")),
        "realized_delta_lcb": _finite(calibration.get("realized_delta_lcb")),
        "snapshot_manifest_hash": str(result.get("snapshot_manifest_hash") or ""),
        "dev_set_hash": str(result.get("dev_set_hash") or ""),
        "score_commitment": str(result.get("score_commitment") or ""),
        "execution_coverage": _finite(result.get("execution_coverage")),
        "snapshot_miss_count": int(result.get("snapshot_miss_count") or 0),
        "failure_count": int(result.get("failure_count") or 0),
        "zero_output_count": int(result.get("zero_output_count") or 0),
    }


async def build_report(
    *, snapshot_uri: str, limit: int, concurrency: int = 1
) -> dict[str, Any]:
    from gateway.research_lab.dev_eval_runner import (
        DockerReplayDevEvaluator,
        snapshot_readiness,
    )
    from gateway.research_lab.calibration_metrics import (
        MIN_HISTORICAL_PAIR_COUNT,
        MIN_SPEARMAN_RHO,
        spearman_correlation,
        top_quartile_lift,
    )

    readiness = await asyncio.to_thread(snapshot_readiness, snapshot_uri)
    if not readiness.get("ready"):
        raise RuntimeError(
            "historical replay requires a verified snapshot: "
            + str(readiness.get("reason") or "not_ready")
        )
    calibrations, candidates = await _load_inputs(limit)
    evaluator = DockerReplayDevEvaluator(snapshot_uri=snapshot_uri)
    semaphore = asyncio.Semaphore(max(1, min(4, int(concurrency))))

    async def _bounded(row: Mapping[str, Any]) -> dict[str, Any]:
        candidate_id = str(row.get("candidate_id") or "")
        candidate_row = candidates.get(candidate_id)
        if candidate_row is None:
            return {
                "candidate_id": candidate_id,
                "eligible": False,
                "reason": "candidate_artifact_not_found",
            }
        async with semaphore:
            return await _evaluate_one(
                evaluator=evaluator,
                calibration=row,
                candidate_row=candidate_row,
            )

    results = await asyncio.gather(*(_bounded(row) for row in calibrations))
    pairs = [
        (float(row["replayed_dev_score"]), float(row["realized_mean_delta"]))
        for row in results
        if row.get("eligible")
        and _finite(row.get("replayed_dev_score")) is not None
        and _finite(row.get("realized_mean_delta")) is not None
    ]
    rho = spearman_correlation(pairs)
    lift = top_quartile_lift(pairs)
    snapshot_hashes = sorted(
        {str(row.get("snapshot_manifest_hash") or "") for row in results if row.get("eligible")}
        - {""}
    )
    set_hashes = sorted(
        {str(row.get("dev_set_hash") or "") for row in results if row.get("eligible")}
        - {""}
    )
    return {
        "schema_version": "research_lab.historical_dev_calibration_report.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "read_only": True,
        "provider_network_access": "disabled_by_docker_network_none",
        "snapshot": readiness,
        "requested_candidate_limit": limit,
        "selected_calibration_count": len(calibrations),
        "candidate_artifact_count": len(candidates),
        "eligible_pair_count": len(pairs),
        "ineligible_count": len(results) - len(pairs),
        "snapshot_manifest_hashes": snapshot_hashes,
        "dev_set_hashes": set_hashes,
        "spearman_rho": rho,
        "top_quartile_realized_lift": lift,
        "calibration_gate": {
            "minimum_pair_count": MIN_HISTORICAL_PAIR_COUNT,
            "minimum_spearman_rho": MIN_SPEARMAN_RHO,
            "requires_positive_top_quartile_lift": True,
            "passed": bool(
                len(pairs) >= MIN_HISTORICAL_PAIR_COUNT
                and rho is not None
                and rho >= MIN_SPEARMAN_RHO
                and lift is not None
                and lift > 0.0
                and len(snapshot_hashes) == 1
                and len(set_hashes) == 1
            ),
        },
        "results": results,
    }


def _write_report(path: Path, report: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshot-uri",
        default=os.getenv("RESEARCH_LAB_DEV_SNAPSHOT_URI", ""),
        help="Verified local or S3 snapshot-set URI.",
    )
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    snapshot_uri = str(args.snapshot_uri or "").strip()
    if not snapshot_uri:
        parser.error("--snapshot-uri or RESEARCH_LAB_DEV_SNAPSHOT_URI is required")
    if args.limit < 1:
        parser.error("--limit must be positive")
    report = asyncio.run(
        build_report(
            snapshot_uri=snapshot_uri,
            limit=min(1000, args.limit),
            concurrency=args.concurrency,
        )
    )
    if args.output:
        _write_report(args.output, report)
        print(str(args.output.expanduser().resolve()))
    else:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["eligible_pair_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
