#!/usr/bin/env python3
"""Legacy manual SOURCE_ADD bundle trial.

Operator driver for sourceexperiments.md §5.3 step 4 (the lab-authored test
adapter) and for early manual funnel runs before a queue consumer exists:

    manifest validation → static scan → LLM review → sandboxed docker trial
    → acceptance evaluation → (with --persist) submission rows + catalog
    entry.

This script is intentionally disconnected from production SOURCE_ADD. It uses
the legacy Docker runner, accepts executable adapter bundles, and must never be
used for automatic miner submissions, catalog writes, or rewards.

Local mode needs no database:

    python3 scripts/run_source_add_trial.py \
        --manifest fixtures/adapter_manifest.json \
        --bundle ./adapter_bundle \
        --icp-refs icp:trial:1,icp:trial:2 \
        --llm-review-verdict pass \
        --human-gate-passed

Credential options (credential_ref_only manifests): --credential-env VAR to
read a raw key from the environment for the trial proxy, or --kms-decrypt to
decrypt the submission's stored envelope. Evidence classification: pass
--evidence-category-map path.json ({evidence_ref: category}) to plug the real
verification stack's output; --classify-as-declared treats all evidence as
the declared category (lab smoke tests only).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a legacy manual SOURCE_ADD trial funnel")
    parser.add_argument(
        "--allow-unsafe-legacy-manual-trial",
        action="store_true",
        help="acknowledge that this disconnected runner uses legacy unsafe networking",
    )
    parser.add_argument("--manifest", required=True, help="adapter manifest JSON path")
    parser.add_argument("--bundle", required=True, help="adapter bundle directory (must contain adapter.py)")
    parser.add_argument("--miner-hotkey", default="lab-operator")
    parser.add_argument("--icp-refs", required=True, help="comma-separated lab-scheduled trial ICP refs")
    parser.add_argument("--llm-review-verdict", choices=["pass", "fail"], required=True,
                        help="operator-attested LLM review verdict (automated review lands with the queue consumer)")
    parser.add_argument("--llm-review-reason", default="operator reviewed")
    parser.add_argument("--human-gate-passed", action="store_true")
    parser.add_argument("--credential-env", default="", help="env var holding the raw provider key for the trial")
    parser.add_argument("--kms-decrypt", action="store_true", help="decrypt the stored credential envelope via KMS")
    parser.add_argument("--evidence-category-map", default="", help="JSON file: {evidence_ref: category}")
    parser.add_argument("--classify-as-declared", action="store_true",
                        help="treat ALL evidence as the declared category (lab smoke tests only)")
    parser.add_argument("--auth-kind", default="header", choices=["header", "query", "bearer", "none"])
    parser.add_argument("--auth-name", default="x-api-key")
    parser.add_argument("--sandbox-image", default=os.getenv("RESEARCH_LAB_SOURCE_ADD_SANDBOX_IMAGE") or "python:3.11-slim")
    parser.add_argument("--timeout-seconds", type=int, default=int(os.getenv("RESEARCH_LAB_SOURCE_ADD_TRIAL_TIMEOUT_SECONDS") or 300))
    parser.add_argument("--acceptance-floor", type=float,
                        default=float(os.getenv("RESEARCH_LAB_SOURCE_ADD_ACCEPTANCE_FLOOR_YIELD") or 0.10))
    parser.add_argument("--registry-provider-id", default="", help="evidence-proxy registry id once the source is provisioned")
    parser.add_argument("--start-epoch", type=int, default=0, help="deprecated; Leg 1 is issued by the measured functional workflow")
    parser.add_argument("--persist", action="store_true", help="write submission/catalog rows to Supabase")
    args = parser.parse_args()

    if not args.allow_unsafe_legacy_manual_trial or os.getenv(
        "RESEARCH_LAB_ALLOW_UNSAFE_LEGACY_SOURCE_ADD_TRIAL"
    ) != "I_UNDERSTAND_THIS_IS_DISCONNECTED":
        print(
            "legacy SOURCE_ADD trial is disabled; use the measured V2 functional probe workflow",
            file=sys.stderr,
        )
        return 2
    if args.persist:
        print(
            "legacy SOURCE_ADD trial persistence is permanently disabled",
            file=sys.stderr,
        )
        return 2

    from gateway.research_lab.key_vault import decrypt_source_add_credential, encrypt_source_add_credential
    from gateway.research_lab.source_add_trial_runner import (
        build_source_add_sandbox_runner,
        build_trial_registry_entry,
    )
    from research_lab.source_add_execution import (
        apply_trial_result,
        evaluate_source_add_acceptance,
        intake_source_add_submission,
        run_llm_review_stage,
        run_sandboxed_trial,
        run_static_scan_stage,
    )
    manifest_doc = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    bundle_dir = Path(args.bundle)
    trial_icp_refs = tuple(ref.strip() for ref in args.icp_refs.split(",") if ref.strip())
    if not trial_icp_refs:
        print("no trial ICP refs supplied", file=sys.stderr)
        return 2

    raw_credential = os.getenv(args.credential_env, "") if args.credential_env else ""
    kms_key_id = (
        os.getenv("RESEARCH_LAB_SOURCE_ADD_CREDENTIAL_KMS_KEY_ID")
        or os.getenv("RESEARCH_LAB_OPENROUTER_KEY_KMS_KEY_ID")
        or ""
    )

    def _kms_encrypt(raw: str, hotkey: str, adapter_ref: str) -> dict[str, str]:
        return encrypt_source_add_credential(
            raw_credential=raw, kms_key_id=kms_key_id, miner_hotkey=hotkey, adapter_ref=adapter_ref
        )

    # 1. Intake (manifest validation + caps; catalog dedupe left to the API path).
    record, errors = intake_source_add_submission(
        manifest_doc,
        miner_hotkey=args.miner_hotkey,
        raw_credential=raw_credential if kms_key_id else "",
        submitted_at=_now_iso(),
        kms_encrypt=_kms_encrypt if kms_key_id else None,
    )
    if errors or record is None:
        print(json.dumps({"stage": "intake", "rejected": errors}, indent=2))
        return 1

    # 2. Static scan over the bundle source files.
    bundle_files = {
        str(path.relative_to(bundle_dir)): path.read_text(encoding="utf-8", errors="replace")
        for path in bundle_dir.rglob("*.py")
    }
    record = run_static_scan_stage(record, bundle_files)
    if record.stage == "rejected":
        print(json.dumps({"stage": "static_scan", "rejected": list(record.rejection_reasons)}, indent=2))
        return 1

    # 3. LLM review (operator-attested verdict for now).
    record = run_llm_review_stage(
        record,
        llm_reviewer=lambda _r: {"verdict": args.llm_review_verdict, "reasons": [args.llm_review_reason]},
    )
    if record.stage == "rejected":
        print(json.dumps({"stage": "llm_review", "rejected": list(record.rejection_reasons)}, indent=2))
        return 1

    # 4. Sandboxed metered docker trial.
    trial_credential = raw_credential
    if args.kms_decrypt and record.credential_envelope.get("ciphertext_b64"):
        trial_credential = decrypt_source_add_credential(
            ciphertext_b64=record.credential_envelope["ciphertext_b64"],
            miner_hotkey=record.miner_hotkey,
            adapter_ref=f"source_add:{record.adapter_id}",
        )
    category_map: dict[str, str] = {}
    if args.evidence_category_map:
        category_map = json.loads(Path(args.evidence_category_map).read_text(encoding="utf-8"))

    def _classifier(evidence_ref: str) -> str:
        if args.classify_as_declared:
            return record.manifest.source_kind
        return str(category_map.get(evidence_ref, ""))

    with tempfile.TemporaryDirectory(prefix="source-add-trial-") as work:
        runner, shutdown = build_source_add_sandbox_runner(
            record=record,
            bundle_dir=bundle_dir,
            work_dir=Path(work),
            registry_entry=build_trial_registry_entry(record, auth_kind=args.auth_kind, auth_name=args.auth_name),
            miner_credential=trial_credential,
            sandbox_image=args.sandbox_image,
            timeout_seconds=args.timeout_seconds,
        )
        try:
            trial = run_sandboxed_trial(
                record,
                trial_icp_refs=trial_icp_refs,
                sandbox_runner=runner,
                evidence_classifier=_classifier,
            )
        finally:
            shutdown()
    record = apply_trial_result(record, trial)
    print(json.dumps({"stage": "trial", **trial.to_dict()}, indent=2))
    if record.stage == "rejected":
        print(json.dumps({"stage": "trial", "rejected": list(record.rejection_reasons)}, indent=2))
        return 1

    # 5. Acceptance.
    record, catalog_entry = evaluate_source_add_acceptance(
        record,
        human_gate_passed=bool(args.human_gate_passed),
        acceptance_floor_yield=args.acceptance_floor,
        accepted_at=_now_iso(),
        registry_provider_id=args.registry_provider_id,
    )
    outcome = {
        "stage": record.stage,
        "measured_trial_yield": record.measured_trial_yield,
        "rejection_reasons": list(record.rejection_reasons),
        "catalog_entry": catalog_entry.to_dict() if catalog_entry else None,
    }

    if args.persist:
        from gateway.research_lab.store import insert_row, persist_source_add_submission

        async def _persist() -> None:
            await persist_source_add_submission(record.to_dict())
            if catalog_entry is not None:
                await insert_row(
                    "research_lab_source_catalog",
                    {**catalog_entry.to_dict(), "catalog_doc": {"submission_id": record.submission_id}},
                )
        asyncio.run(_persist())
        outcome["persisted"] = True

    print(json.dumps(outcome, indent=2))
    return 0 if record.stage == "accepted" else 1


if __name__ == "__main__":
    raise SystemExit(main())
