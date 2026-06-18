#!/usr/bin/env python3
"""Verify Research Lab P0.4 notary/evidence/trace helpers."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_lab import (
    LocalSnapshotStore,
    NotarySigner,
    build_evidence_bundle,
    build_execution_trace,
    capture_snapshot,
    evidence_refs_from_bundle,
    make_trace_call,
    run_l0_replay,
    validate_schema_record,
)


def main() -> int:
    tmp_dir = Path(tempfile.mkdtemp(prefix="leadpoet_p04_", dir="/private/tmp"))
    try:
        store = LocalSnapshotStore(tmp_dir / "snapshot_store")
        signer = NotarySigner("phase-0-local-test-key", key_id="p0.4-test")
        content = (
            "Acme announced Series B funding to expand its compliance automation "
            "platform on June 1, 2026."
        )
        snapshot = capture_snapshot(
            url="https://news.example.com/acme-series-b-2026-06-01",
            content=content,
            store=store,
            signer=signer,
            fetch_ts="2026-06-16T00:01:05Z",
            metadata={"status_code": 200, "provider": "local-fixture"},
        )
        _assert(snapshot.signature.startswith("local_hmac:"), "local signatures use dev prefix")
        _assert(not snapshot.signature.startswith("fabric_sig:"), "local signatures avoid fabric namespace")

        bundle = build_evidence_bundle(
            run_id="33333333-3333-4333-8333-333333333333",
            artifact_hash="sha256:artifact-example",
            created_at="2026-06-16T00:01:10Z",
            snapshots=[snapshot],
            merkle_anchor_ref="arweave:pending-batch-2026-06-16T00",
        )
        _assert(not validate_schema_record("evidence_bundle.schema.json", bundle), "bundle validates")
        _assert("content" not in bundle["snapshots"][0], "bundle excludes raw content")
        _assert(store.read_content(snapshot.snapshot_ref) == content, "snapshot store preserves raw content")
        raw_snapshot = {**snapshot.to_schema_snapshot(), "content": content}
        _assert_raises(
            lambda: build_evidence_bundle(
                artifact_hash="sha256:artifact-example",
                snapshots=[raw_snapshot],
                validate=False,
            ),
            "evidence no-raw guard runs when schema validation is disabled",
        )

        call = make_trace_call(
            seq=1,
            ts="2026-06-16T00:01:00Z",
            provider="exa",
            model="exa-search",
            purpose="search",
            component="qualification.discovery",
            request={"query": "Acme Series B compliance automation"},
            response={"url": snapshot.url},
            cost_usd=0.02,
        )
        trace = build_execution_trace(
            run_id="33333333-3333-4333-8333-333333333333",
            artifact_hash="sha256:artifact-example",
            role="baseline_arm",
            rung="L1",
            status="completed",
            icp_set_hash="sha256:icp-set-example",
            eval_version={
                "verifier_hash": "sha256:verifier-example",
                "judge_version_hash": "sha256:judge-version-example",
            },
            calls=[call],
            evidence_refs=evidence_refs_from_bundle(bundle, signal_indices={snapshot.snapshot_ref: 0}),
            outputs_payload={"companies": ["Acme"]},
            score_bundle_payload={"score": 90.0},
            attestation_ref="sha256:attestation-example",
        )
        _assert(not validate_schema_record("execution_trace.schema.json", trace), "trace validates")
        _assert("request" not in trace["calls"][0], "trace excludes raw request")
        _assert("response" not in trace["calls"][0], "trace excludes raw response")
        _assert(trace["cost_ledger"]["total_usd"] == 0.02, "cost ledger totals calls")
        raw_call = {
            **call.to_schema_call(),
            "request": {"query": "raw should not be stored"},
        }
        _assert_raises(
            lambda: build_execution_trace(
                artifact_hash="sha256:artifact-example",
                role="baseline_arm",
                rung="L1",
                status="completed",
                icp_set_hash="sha256:icp-set-example",
                eval_version={
                    "verifier_hash": "sha256:verifier-example",
                    "judge_version_hash": "sha256:judge-version-example",
                },
                calls=[raw_call],
                evidence_refs=evidence_refs_from_bundle(bundle),
                validate=False,
            ),
            "trace no-raw guard runs when schema validation is disabled",
        )

        signal = {
            "url": snapshot.url,
            "source": "news",
            "company_website": "https://acme.ai",
            "description": "Acme announced Series B funding to expand its compliance automation platform.",
            "snippet": content,
            "date": "2026-06-01",
            "matched_icp_signal": "recently raised funding for compliance automation",
        }
        replay = run_l0_replay(
            evidence_bundle=bundle,
            execution_trace=trace,
            signals_by_index={0: signal},
            snapshot_store=store,
        )
        _assert(len(replay) == 1, "one replay result")
        _assert(replay[0].passed, "L0 replay passes against stored snapshot")
        snapshot_path = store.path_for_ref(snapshot.snapshot_ref)
        with snapshot_path.open("r", encoding="utf-8") as f:
            stored_payload = json.load(f)
        stored_payload["content"] = stored_payload["content"] + " tampered"
        with snapshot_path.open("w", encoding="utf-8") as f:
            json.dump(stored_payload, f, sort_keys=True, indent=2)
        _assert_raises(
            lambda: run_l0_replay(
                evidence_bundle=bundle,
                execution_trace=trace,
                signals_by_index={0: signal},
                snapshot_store=store,
            ),
            "L0 replay rejects tampered snapshot content",
        )
        store.write(snapshot, content=content, metadata={"status_code": 200, "provider": "local-fixture"})

        deleted = store.delete_content(
            snapshot.snapshot_ref,
            deletion_reason_ref="erasure-request:test",
            deleted_at="2026-06-17T00:00:00Z",
        )
        deleted_bundle = build_evidence_bundle(
            run_id="33333333-3333-4333-8333-333333333333",
            artifact_hash="sha256:artifact-example",
            created_at="2026-06-17T00:01:10Z",
            snapshots=[deleted],
            verification_state="content_deleted",
            deletion_request_ref="erasure-request:test",
        )
        _assert(
            deleted_bundle["snapshots"][0]["l0_verification_state"] == "content_deleted",
            "deletion-with-hash-retention state is represented",
        )

        print("Research Lab notary/evidence/trace helpers verified.")
        return 0
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _assert(condition: bool, label: str) -> None:
    if not condition:
        raise AssertionError(label)


def _assert_raises(fn, label: str) -> None:
    try:
        fn()
    except ValueError:
        return
    raise AssertionError(label)


if __name__ == "__main__":
    raise SystemExit(main())
