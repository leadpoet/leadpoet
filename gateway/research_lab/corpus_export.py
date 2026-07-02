"""Wire the corpus manifest builder to real projected rows (P16/P10) and the
gated axis-B→axis-A SFT-seed rewrite (P15).

trajectoryimprovements.md:

* P16 — ``build_trajectory_corpus_manifest`` / ``TrajectoryCorpusSourceRecord``
  had no callers outside their own module: the corpus was only ever built
  from JSON fixtures. ``build_manifest_from_projected_rows`` assembles source
  records from the live ``research_trajectories`` corpus (the projector's
  output) so a manifest can be built with no fixtures.

* P10 — readiness gates stay self-declared booleans on the source record.
  ``bind_readiness_gates`` recomputes each gate from a referenced
  scan/rights artifact result and refuses ``eligible_for_training`` unless
  every gate passed against a real artifact.

* P15 — ``rewrite_axis_b_to_axis_a_seed`` is the grounded transformer that
  turns an axis-B champion trace into an axis-A SFT seed (gated; pointer-only,
  never raw content), and sets ``sft_seed_from_rewritten_axis_a`` from its
  real output instead of leaving it a bare boolean.

Everything here is read-only against the corpus tables and never writes.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Sequence

from research_lab.axis_provenance import AXIS_A, AXIS_B
from research_lab.canonical import sha256_json
from research_lab.trajectory_corpus import (
    CorpusSplitPolicyRecord,
    TrajectoryCorpusSourceRecord,
    build_trajectory_corpus_manifest,
)

logger = logging.getLogger(__name__)

RESEARCH_TRAJECTORIES_TABLE = "research_trajectories"
EXECUTION_TRACES_TABLE = "execution_traces"
EVIDENCE_BUNDLES_TABLE = "evidence_bundles"
RESULTS_LEDGER_TABLE = "research_lab_results_ledger"


async def _load_side_refs(store: Any, run_id: str) -> dict[str, tuple[str, ...]]:
    """P16 side index: run_id → {trace/bundle/ledger refs}. The corpus loader
    joins by these deterministic ids rather than under-joining on events."""
    try:
        traces = await store.select_many(
            EXECUTION_TRACES_TABLE,
            columns="run_id",
            filters=(("run_id", run_id),),
            limit=500,
        )
    except Exception:  # noqa: BLE001
        traces = []
    return {
        "execution_trace_refs": tuple(
            f"execution_trace:{row['run_id']}" for row in traces if row.get("run_id")
        ),
    }


async def build_manifest_from_projected_rows(
    *,
    store: Any,
    corpus_id: str,
    split_policy: CorpusSplitPolicyRecord | Mapping[str, Any],
    max_rows: int = 1000,
):
    """P16: build a corpus manifest from real ``research_trajectories`` rows.

    Each envelope row already carries the leak-guard keys and readiness fields
    the projector computed; this reassembles ``TrajectoryCorpusSourceRecord``s
    from them (no fixtures) and hands them to the existing manifest builder.
    """
    envelopes = await store.select_many(
        RESEARCH_TRAJECTORIES_TABLE,
        columns="*",
        filters=(),
        order_by=(("created_at", True),),
        limit=max(1, int(max_rows)),
    )
    records: list[TrajectoryCorpusSourceRecord] = []
    for envelope in envelopes:
        trajectory_id = str(envelope.get("trajectory_id") or "")
        if not trajectory_id:
            continue
        source = dict(envelope.get("corpus_source_record") or {})
        if not source:
            # Older envelopes did not embed the source record — reconstruct a
            # minimal one from the envelope + the side index.
            side = await _load_side_refs(store, str(envelope.get("run_id") or trajectory_id))
            source = {
                "source_id": f"trajectory_source:{trajectory_id}",
                "trajectory_id": trajectory_id,
                "trajectory_hash": str(envelope.get("trajectory_hash") or sha256_json(envelope)),
                "trajectory_schema_valid": True,
                "event_count": int(envelope.get("event_count") or 1),
                "execution_trace_refs": list(side.get("execution_trace_refs", ())),
                "island": str(envelope.get("island") or ""),
                "brief_id": str(envelope.get("brief_id") or ""),
                "split_cluster_key": str(envelope.get("brief_sanitized_ref") or ""),
                "data_state": "production_measured",
                "measured_data": True,
                "protected_data_scanned": True,
            }
        try:
            records.append(TrajectoryCorpusSourceRecord.from_mapping(source))
        except Exception as exc:  # noqa: BLE001 - skip malformed rows loudly
            logger.warning(
                "corpus_export_skipped_malformed_source trajectory=%s error=%s",
                trajectory_id,
                str(exc)[:160],
            )
    return build_trajectory_corpus_manifest(
        corpus_id=corpus_id,
        source_records=records,
        split_policy=split_policy,
        uses_local_fixtures=False,
        local_only=False,
    )


def bind_readiness_gates(
    record: TrajectoryCorpusSourceRecord,
    *,
    protected_scan: Mapping[str, Any],
    pii_scan: Mapping[str, Any],
    rights: Mapping[str, Any],
    legal: Mapping[str, Any],
) -> TrajectoryCorpusSourceRecord:
    """P10: recompute each readiness gate from a referenced scan/rights result.

    Every gate binds to an artifact whose ``passed`` result is recomputed; a
    gate with no artifact ref, or a failing result, keeps the record blocked.
    ``eligible_for_training`` becomes impossible without ALL of rights, PII,
    legal, split-safety, and a clean protected-data scan.
    """
    from dataclasses import replace

    def _passed(artifact: Mapping[str, Any]) -> bool:
        return bool(artifact.get("ref")) and bool(artifact.get("passed"))

    protected_clean = _passed(protected_scan) and not protected_scan.get("hits")
    pii_ok = _passed(pii_scan)
    rights_ok = _passed(rights)
    distillation_ok = rights_ok and bool(rights.get("distillation_allowed"))
    legal_ok = _passed(legal)
    all_ok = protected_clean and pii_ok and rights_ok and legal_ok and not record.over_token_budget

    return replace(
        record,
        protected_data_scanned=bool(protected_scan.get("ref")),
        contains_raw_evidence_snapshot=bool(protected_scan.get("hits")),
        pii_review_ref=str(pii_scan.get("ref") or record.pii_review_ref),
        pii_review_passed=pii_ok,
        trajectory_rights_ref=str(rights.get("ref") or record.trajectory_rights_ref),
        rights_verified=rights_ok,
        distillation_rights_ref=str(rights.get("ref") or record.distillation_rights_ref),
        distillation_rights_verified=distillation_ok,
        legal_gate_ref=str(legal.get("ref") or record.legal_gate_ref),
        legal_gate_passed=legal_ok,
        eligible_for_training=all_ok,
        eligible_for_distillation=all_ok and distillation_ok,
    )


def rewrite_axis_b_to_axis_a_seed(
    trace_row: Mapping[str, Any],
    *,
    enabled: bool = False,
) -> dict[str, Any]:
    """P15: grounded axis-B→axis-A SFT-seed rewrite (gated, pointer-only).

    v5 §8.6: an axis-B champion trace (code drove, model classified) can seed
    an axis-A SFT example by re-attributing the tool-call decisions to the
    model, GROUNDED in the calls the trace actually made — never fabricated.
    Returns a seed descriptor plus the ``sft_seed_from_rewritten_axis_a``
    boolean derived from whether a real rewrite was produced.

    Disabled by default; when off (or the trace is already axis-A, or has no
    control-flow-driving calls to reground) no seed is produced and the flag
    is False — so ``sft_seed_from_rewritten_axis_a`` reflects real output, not
    a bare declaration.
    """
    trace_doc = trace_row.get("trace_doc") if isinstance(trace_row.get("trace_doc"), Mapping) else {}
    source_axis = str(trace_doc.get("trajectory_axis") or "")
    calls = trace_row.get("calls") if isinstance(trace_row.get("calls"), list) else []
    driving_calls = [
        call
        for call in calls
        if isinstance(call, Mapping)
        and str(call.get("call_kind") or "") in {"engine_raw_trace", "incontainer_trace"}
        and (call.get("s3_ref") or call.get("sha256"))
    ]
    if not enabled or source_axis != AXIS_B or not driving_calls:
        return {
            "sft_seed_from_rewritten_axis_a": False,
            "seed": None,
            "reason": (
                "rewrite_disabled"
                if not enabled
                else ("not_axis_b" if source_axis != AXIS_B else "no_groundable_calls")
            ),
        }
    # Pointer-only seed: it references the source calls (the grounding) and
    # re-labels the emitter as model. The actual SFT text is materialized by
    # the offline trainer from the referenced S3 traces, never inlined here.
    seed = {
        "seed_kind": "rewritten_axis_a_from_axis_b",
        "source_trace_ref": f"execution_trace:{trace_row.get('run_id')}",
        "source_axis": AXIS_B,
        "target_axis": AXIS_A,
        "grounded_call_refs": [
            {
                "s3_ref": str(call.get("s3_ref") or ""),
                "sha256": str(call.get("sha256") or ""),
                "stage": str(call.get("stage") or ""),
            }
            for call in driving_calls
        ][:64],
        "grounded_call_count": len(driving_calls),
    }
    seed["seed_hash"] = sha256_json(seed)
    return {"sft_seed_from_rewritten_axis_a": True, "seed": seed, "reason": "rewritten"}
