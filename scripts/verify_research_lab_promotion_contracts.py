#!/usr/bin/env python3
"""Verify Research Lab promotion and public benchmark contracts locally."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sys
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gateway.research_lab.public_benchmarks import build_public_benchmark_report, sanitize_benchmark_item_summary
from gateway.research_lab.config import ResearchLabGatewayConfig
from gateway.research_lab import promotion as promotion_module
from gateway.research_lab.promotion import ResearchLabPromotionController, _daily_counts_from_score_bundle
from leadpoet_verifier.economics import build_champion_reward_obligation
from research_lab.canonical import sha256_json
from research_lab.eval import PrivateModelArtifactManifest


def main() -> int:
    errors: list[str] = []
    report = _public_report()
    encoded = json.dumps(report, sort_keys=True).lower()
    for forbidden in (
        "sk-or-",
        "https://",
        "http://",
        "image_digest",
        "candidate_patch_manifest",
        "private_model_manifest_doc",
        ".dkr.ecr",
        "judge_prompt",
    ):
        if forbidden in encoded:
            errors.append(f"public benchmark report leaked forbidden marker: {forbidden}")
    if "intent_signals" not in encoded:
        errors.append("public benchmark report must expose exact intent_signals for public ICPs")
    if report["schema_version"] != "1.2":
        errors.append(f"public benchmark report schema was not 1.2: {report['schema_version']}")
    if report["item_count"] != 6:
        errors.append(f"public benchmark report expected 6 total ICPs, got {report['item_count']}")
    if report["public_icp_count"] != 3:
        errors.append(f"public benchmark report expected 3 public ICPs, got {report['public_icp_count']}")
    if report["private_holdout_icp_count"] != 3:
        errors.append(
            f"public benchmark report expected 3 private holdout ICPs, got {report['private_holdout_icp_count']}"
        )
    split = report["visibility_split"]
    if split["public_strength_counts"] != {"strong": 1, "weak": 2}:
        errors.append(f"public split did not expose 2 weak / 1 strong ICPs: {split['public_strength_counts']}")
    if split["private_strength_counts"] != {"strong": 2, "weak": 1}:
        errors.append(f"private split did not reserve 1 weak / 2 strong ICPs: {split['private_strength_counts']}")
    if report["zero_lead_icp_count"] != 1:
        errors.append("zero-lead ICP count did not match expected value")
    if report["low_intent_fit_icp_count"] != 1:
        errors.append("public low-intent ICP count did not match expected value")
    if report["low_icp_fit_count"] != 0:
        errors.append("ICP mismatch count must not be derived from avg_icp_fit")
    if report["failure_category_counts"].get("hallucinated_or_generic_intent") != 1:
        errors.append("hallucinated/generic intent failure was not counted")
    issue_counts = report.get("model_issue_counts", {})
    if issue_counts.get("zero_company_results") != 1:
        errors.append("model_issue_counts did not include the public zero-company ICP")
    if issue_counts.get("low_intent_fit") != 1:
        errors.append("model_issue_counts did not include the public low-intent ICP")
    if issue_counts.get("hallucinated_or_generic_intent") != 1:
        errors.append("model_issue_counts did not include the public hallucinated/generic ICP")
    issue_icps = report.get("model_issue_public_icps", {})
    for issue_key in ("zero_company_results", "low_intent_fit", "hallucinated_or_generic_intent"):
        rows = issue_icps.get(issue_key) if isinstance(issue_icps, dict) else None
        if not isinstance(rows, list) or not rows:
            errors.append(f"model issue {issue_key} did not map to public ICP rows")

    daily_counts = _daily_counts_from_score_bundle(_score_bundle())
    if daily_counts != {str(day): 6 for day in range(100, 110)}:
        errors.append(f"daily ICP counts did not parse from score bundle refs: {daily_counts}")

    obligation = build_champion_reward_obligation(
        {
            "uid": 9,
            "miner_hotkey": "5Fminer111111111111111111111111111111111111",
            "island": "generalist",
            "candidate_id": "candidate:" + "a" * 64,
            "score_bundle_id": "score_bundle:" + "b" * 64,
            "run_id": "11111111-1111-4111-8111-111111111111",
            "evaluation_epoch": 1000,
            "start_epoch": 1001,
            "improvement_points": 1.25,
            "threshold_points": 1.0,
            "daily_icp_counts": {str(day): 6 for day in range(100, 110)},
        },
        {
            "champion_threshold_points": 1.0,
            "champion_min_alpha_percent": 4.0,
            "champion_extra_alpha_percent_per_point": 0.2,
            "champion_max_alpha_percent": 10.0,
            "champion_eval_days": 10,
            "champion_icps_per_day": 6,
            "reward_epochs": 20,
        },
    )
    if obligation["status"] != "active":
        errors.append(f"champion obligation was not active: {obligation}")
    if float(obligation["threshold_points"]) != 1.0:
        errors.append("champion obligation did not use the 1-point threshold")
    if not str(obligation.get("champion_reward_id") or "").startswith("champion_reward:sha256:"):
        errors.append("champion reward id must keep champion_reward:sha256 prefix")
    anchored_hash = str(obligation.get("anchored_hash") or "")
    if not anchored_hash.startswith("sha256:") or anchored_hash.startswith("champion_reward:"):
        errors.append("champion reward anchored_hash must satisfy DB sha256 check constraint")

    try:
        asyncio.run(_test_stale_scored_candidate_requires_rebase())
    except Exception as exc:
        errors.append(f"stale scored candidate promotion contract failed: {exc}")
    try:
        asyncio.run(_test_private_source_reconciler_skips_recovered_stale_parent())
    except Exception as exc:
        errors.append(f"private-source recovered stale-parent idempotency failed: {exc}")
    try:
        asyncio.run(_test_disabled_auto_promotion_writes_terminal_decision())
    except Exception as exc:
        errors.append(f"disabled auto-promotion decision contract failed: {exc}")
    try:
        asyncio.run(_test_daily_baseline_delta_overrides_legacy_mean_delta())
    except Exception as exc:
        errors.append(f"daily-baseline promotion gate contract failed: {exc}")
    migration_56 = (ROOT / "scripts" / "56-research-lab-promotion-decision-events.sql").read_text(encoding="utf-8")
    if "stale_parent_needs_rescore" not in migration_56:
        errors.append("script 56 must preserve existing stale_parent_needs_rescore promotion_status rows")

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print(
        "Research Lab promotion contracts verified: sanitized public report, daily ICP count parsing, "
        "and 1-point champion obligation threshold."
    )
    return 0


def _public_report() -> dict[str, object]:
    benchmark_items = []
    summaries = []
    # The v3 hash-rotation public split is deterministic and score-independent:
    # for this window hash it puts icp_b/icp_c/icp_f in the public split. The
    # issue-bearing rows must therefore BE those rows — icp_b carries the
    # zero-company + hallucinated-intent failure, icp_c the low-intent row
    # (score 22 -> intent_signal 11 < 15), and the private rows stay healthy
    # (a weak, d/e strong) to keep the expected strength distribution.
    rows = [
        ("icp_a", "Healthcare", "Revenue Cycle", "hiring revenue operations leaders", 35.0, 2),
        ("icp_b", "Manufacturing", "Industrial Automation", "expansion into new facilities", 0.0, 0),
        ("icp_c", "Financial Services", "Risk", "new compliance audit program", 22.0, 3),
        ("icp_d", "Software", "Developer Tools", "migration to cloud data warehouse", 72.0, 4),
        ("icp_e", "Logistics", "Cold Chain", "opening new fulfillment centers", 83.0, 4),
        ("icp_f", "Cybersecurity", "IAM", "security platform implementation", 91.0, 5),
    ]
    for rank, (icp_id, industry, sub_industry, signal, score, company_count) in enumerate(rows, start=1):
        item = {
            "icp_ref": f"qualification_private_icp_sets:100:{icp_id}",
            "icp_hash": "sha256:" + f"{rank:064x}"[-64:],
            "set_id": 100,
            "day_index": 1,
            "day_rank": rank,
            "intent_signal_signature": signal,
            "icp": {
                "icp_id": icp_id,
                "prompt": f"Find companies for {industry}; see https://example.com/private",
                "industry": industry,
                "sub_industry": sub_industry,
                "country": "United States",
                "employee_count": "51-200",
                "product_service": f"{industry} platform",
                "intent_signals": [signal],
            },
        }
        benchmark_items.append(item)
        summaries.append(
            sanitize_benchmark_item_summary(
                item=item,
                score=score,
                company_count=company_count,
                score_breakdowns=[
                    {
                        "final_score": score,
                        "icp_fit": min(score / 2, 50.0),
                        "intent_signal_final": min(score / 2, 50.0),
                        "failure_reason": "Intent fabrication detected (generic claim)" if score == 0.0 else None,
                    }
                ],
            )
        )
    return build_public_benchmark_report(
        benchmark_date="2026-06-23",
        rolling_window_hash="sha256:" + "3" * 64,
        aggregate_score=36.0,
        per_icp_summaries=summaries,
        benchmark_items=benchmark_items,
        public_icps_per_day=3,
        public_weak_per_day=2,
    )


def _score_bundle() -> dict[str, object]:
    per_icp_results = []
    for set_id in range(100, 110):
        for idx in range(6):
            per_icp_results.append(
                {
                    "icp_ref": f"qualification_private_icp_sets:{set_id}:icp_{idx}",
                    "icp_hash": "sha256:" + f"{set_id:064x}"[-64:],
                    "base_company_scores": [50],
                    "candidate_company_scores": [55],
                }
            )
    return {"aggregates": {"per_icp_results": per_icp_results}}


def _manifest(name: str) -> PrivateModelArtifactManifest:
    payload = {
        "model_artifact_hash": sha256_json({"model": name}),
        "git_commit_sha": "abcdef1234567890",
        "image_digest": f"493765492819.dkr.ecr.us-east-1.amazonaws.com/research-lab/{name}@sha256:"
        + ("a" if name == "parent" else "b") * 64,
        "config_hash": sha256_json({"config": name}),
        "component_registry_version": "components:v1",
        "scoring_adapter_version": "adapter:v1",
        "manifest_uri": f"s3://leadpoet-private-model-artifacts-493765492819/research-lab/test/{name}.json",
        "signature_ref": f"kms://test/{name}",
        "build_id": f"build-{name}",
    }
    payload["manifest_hash"] = sha256_json(payload)
    return PrivateModelArtifactManifest.from_mapping(payload)


async def _test_stale_scored_candidate_requires_rebase() -> None:
    old_parent = _manifest("parent")
    active_parent = _manifest("active-parent")
    candidate = {
        "candidate_id": "candidate:" + "1" * 64,
        "candidate_kind": "image_build",
        "parent_artifact_hash": old_parent.model_artifact_hash,
        "candidate_model_manifest_doc": _manifest("candidate").to_dict(),
        "candidate_source_diff_hash": sha256_json({"diff": "miner"}),
        "miner_hotkey": "5EFakeMinerHotkey111111111111111111111111111",
        "run_id": "11111111-1111-4111-8111-111111111111",
        "ticket_id": "22222222-2222-4222-8222-222222222222",
        "island": "generalist",
    }
    score_bundle_row = {"score_bundle_id": "score_bundle:" + "2" * 64}
    score_bundle = {
        "parent_artifact_hash": old_parent.model_artifact_hash,
        "icp_set_hash": "sha256:" + "3" * 64,
        "aggregates": {
            "mean_delta": 2.0,
            "delta_lcb": 1.2,
        },
    }
    events: list[dict[str, object]] = []

    async def fake_load_active_private_model(_config, *, register_bootstrap=False):
        return SimpleNamespace(artifact=active_parent, version_row=None)

    async def fake_create_candidate_promotion_event(**kwargs):
        events.append(kwargs)
        return kwargs

    original_load_active = promotion_module.load_active_private_model
    original_promotion_event = promotion_module.create_candidate_promotion_event
    try:
        promotion_module.load_active_private_model = fake_load_active_private_model
        promotion_module.create_candidate_promotion_event = fake_create_candidate_promotion_event
        result = await ResearchLabPromotionController(
            ResearchLabGatewayConfig(auto_promotion_enabled=True),
            worker_ref="test-worker",
        ).process_scored_candidate(
            candidate=candidate,
            score_bundle_row=score_bundle_row,
            score_bundle=score_bundle,
        )
    finally:
        promotion_module.load_active_private_model = original_load_active
        promotion_module.create_candidate_promotion_event = original_promotion_event

    if result.get("status") != "stale_parent_needs_rescore":
        raise AssertionError(f"expected stale_parent_needs_rescore, got {result}")
    stale_events = [event for event in events if event.get("event_type") == "stale_parent_detected"]
    if not stale_events:
        raise AssertionError("stale parent event was not written")
    if stale_events[0].get("promotion_status") != "rebase_required":
        raise AssertionError(f"stale parent event used invalid status: {stale_events[0]}")
    if any(event.get("event_type") == "active_version_created" for event in events):
        raise AssertionError("stale scored candidate must not create an active version directly")


async def _test_private_source_reconciler_skips_recovered_stale_parent() -> None:
    candidate_id = "candidate:" + "9" * 64
    writes: list[dict[str, object]] = []

    async def fake_select_many(table, **_kwargs):
        if table != "research_loop_run_queue_events":
            return []
        filters = dict(_kwargs.get("filters") or ())
        if str(filters.get("ticket_id") or "") == "66666666-6666-4666-8666-666666666666":
            return []
        return [
            {
                "run_id": "33333333-3333-4333-8333-333333333333",
                "reason": "regenerate_after_rebase_unavailable",
                "event_doc": {"regenerated_from_candidate_id": candidate_id},
                "created_at": "2026-07-06T05:31:54+00:00",
            }
        ]

    async def fake_create_candidate_evaluation_event(**kwargs):
        writes.append({"kind": "candidate_evaluation", **kwargs})
        return kwargs

    async def fake_create_scoring_dispatch_event(**kwargs):
        writes.append({"kind": "scoring_dispatch", **kwargs})
        return kwargs

    original_select_many = promotion_module.select_many
    original_candidate_event = promotion_module.create_candidate_evaluation_event
    original_dispatch_event = promotion_module.create_scoring_dispatch_event
    try:
        promotion_module.select_many = fake_select_many
        promotion_module.create_candidate_evaluation_event = fake_create_candidate_evaluation_event
        promotion_module.create_scoring_dispatch_event = fake_create_scoring_dispatch_event
        for reason in ("stale_parent_rebase_unavailable", "stale_parent_needs_rescore"):
            result = await promotion_module._mark_private_source_push_stale_parent_for_rebase(
                candidate={
                    "candidate_id": candidate_id,
                    "run_id": "11111111-1111-4111-8111-111111111111",
                    "ticket_id": "22222222-2222-4222-8222-222222222222",
                    "current_candidate_status": "rejected",
                    "current_reason": reason,
                    "parent_artifact_hash": "sha256:" + "a" * 64,
                    "candidate_artifact_hash": "sha256:" + "b" * 64,
                },
                score_bundle_id="score_bundle:" + "c" * 64,
                worker_ref="test-worker",
                failed_promotion_event_id="44444444-4444-4444-8444-444444444444",
                latest_promotion_event_id="55555555-5555-4555-8555-555555555555",
            )
            if result.get("status") != "stale_parent_rebase_already_recovered":
                raise AssertionError(f"expected already recovered for {reason}, got {result}")
            if result.get("stale_parent_recovery_event_status") != "already_recovered":
                raise AssertionError(f"expected already_recovered marker for {reason}, got {result}")
            if result.get("stale_parent_rebase_eligible"):
                raise AssertionError(f"recovered stale-parent candidate must not be rebase eligible: {result}")
        unrecovered = await promotion_module._mark_private_source_push_stale_parent_for_rebase(
            candidate={
                "candidate_id": "candidate:" + "8" * 64,
                "run_id": "77777777-7777-4777-8777-777777777777",
                "ticket_id": "66666666-6666-4666-8666-666666666666",
                "current_candidate_status": "rejected",
                "current_reason": "stale_parent_needs_rescore",
                "parent_artifact_hash": "sha256:" + "d" * 64,
                "candidate_artifact_hash": "sha256:" + "e" * 64,
            },
            score_bundle_id="score_bundle:" + "f" * 64,
            worker_ref="test-worker",
            failed_promotion_event_id="88888888-8888-4888-8888-888888888888",
            latest_promotion_event_id="99999999-9999-4999-8999-999999999999",
        )
        if unrecovered.get("status") != "stale_parent_needs_rescore":
            raise AssertionError(f"unrecovered stale-parent marker should remain actionable: {unrecovered}")
        if not unrecovered.get("stale_parent_rebase_eligible"):
            raise AssertionError(f"unrecovered stale-parent marker should queue rebase: {unrecovered}")
    finally:
        promotion_module.select_many = original_select_many
        promotion_module.create_candidate_evaluation_event = original_candidate_event
        promotion_module.create_scoring_dispatch_event = original_dispatch_event

    if writes:
        raise AssertionError(f"recovered stale-parent reconciler wrote new events: {writes}")


async def _test_disabled_auto_promotion_writes_terminal_decision() -> None:
    parent = _manifest("parent")
    candidate = {
        "candidate_id": "candidate:" + "4" * 64,
        "candidate_kind": "image_build",
        "parent_artifact_hash": parent.model_artifact_hash,
        "miner_hotkey": "5EFakeMinerHotkey111111111111111111111111111",
        "run_id": "11111111-1111-4111-8111-111111111111",
        "ticket_id": "22222222-2222-4222-8222-222222222222",
        "island": "generalist",
    }
    score_bundle_row = {"score_bundle_id": "score_bundle:" + "5" * 64}
    score_bundle = {
        "parent_artifact_hash": parent.model_artifact_hash,
        "icp_set_hash": "sha256:" + "6" * 64,
        "aggregates": {
            "mean_delta": 3.0,
            "delta_lcb": 2.5,
        },
    }
    events: list[dict[str, object]] = []

    async def fake_create_candidate_promotion_event(**kwargs):
        events.append(kwargs)
        return kwargs

    original_promotion_event = promotion_module.create_candidate_promotion_event
    try:
        promotion_module.create_candidate_promotion_event = fake_create_candidate_promotion_event
        result = await ResearchLabPromotionController(
            ResearchLabGatewayConfig(auto_promotion_enabled=False),
            worker_ref="test-worker",
        ).process_scored_candidate(
            candidate=candidate,
            score_bundle_row=score_bundle_row,
            score_bundle=score_bundle,
        )
    finally:
        promotion_module.create_candidate_promotion_event = original_promotion_event

    if result.get("status") != "disabled":
        raise AssertionError(f"expected disabled, got {result}")
    if [event.get("event_type") for event in events] != ["promotion_checked", "promotion_disabled"]:
        raise AssertionError(f"disabled promotion did not write checked+terminal events: {events}")
    if events[1].get("promotion_status") != "disabled":
        raise AssertionError(f"disabled promotion event used invalid status: {events[1]}")


async def _test_daily_baseline_delta_overrides_legacy_mean_delta() -> None:
    parent = _manifest("parent")
    candidate = {
        "candidate_id": "candidate:" + "7" * 64,
        "candidate_kind": "image_build",
        "parent_artifact_hash": parent.model_artifact_hash,
        "candidate_model_manifest_doc": _manifest("candidate").to_dict(),
        "miner_hotkey": "5EFakeMinerHotkey111111111111111111111111111",
        "run_id": "11111111-1111-4111-8111-111111111111",
        "ticket_id": "22222222-2222-4222-8222-222222222222",
        "island": "generalist",
    }
    score_bundle_row = {"score_bundle_id": "score_bundle:" + "8" * 64}
    score_bundle = {
        "parent_artifact_hash": parent.model_artifact_hash,
        "icp_set_hash": "sha256:" + "9" * 64,
        "candidate_artifact_hash": _manifest("candidate").model_artifact_hash,
        "aggregates": {
            "mean_delta": 2.0,
            "delta_lcb": 1.5,
        },
        "private_holdout_gate": {
            "decision": "private_holdout_approved",
            "private_holdout_evaluated": True,
            "baseline_aggregate_score": 16.46,
            "candidate_total_score": 9.70,
            "candidate_delta_vs_daily_baseline": -6.76,
            "reference_evaluation_mode": "stored_daily_baseline",
        },
    }
    events: list[dict[str, object]] = []

    async def fake_load_active_private_model(_config, *, register_bootstrap=False):
        return SimpleNamespace(artifact=parent, version_row=None)

    async def fake_create_candidate_promotion_event(**kwargs):
        events.append(kwargs)
        return kwargs

    original_load_active = promotion_module.load_active_private_model
    original_promotion_event = promotion_module.create_candidate_promotion_event
    try:
        promotion_module.load_active_private_model = fake_load_active_private_model
        promotion_module.create_candidate_promotion_event = fake_create_candidate_promotion_event
        result = await ResearchLabPromotionController(
            ResearchLabGatewayConfig(auto_promotion_enabled=True),
            worker_ref="test-worker",
        ).process_scored_candidate(
            candidate=candidate,
            score_bundle_row=score_bundle_row,
            score_bundle=score_bundle,
        )
    finally:
        promotion_module.load_active_private_model = original_load_active
        promotion_module.create_candidate_promotion_event = original_promotion_event

    if result.get("status") != "rejected_below_threshold":
        raise AssertionError(f"expected daily-baseline rejection, got {result}")
    terminal = events[-1]
    if terminal.get("event_type") != "below_threshold":
        raise AssertionError(f"expected below_threshold event, got {terminal}")
    if terminal.get("improvement_points") != -6.76:
        raise AssertionError(f"promotion used legacy mean_delta instead of daily baseline delta: {terminal}")
    metric = ((terminal.get("event_doc") or {}).get("promotion_metric") or {})
    if metric.get("improvement_basis") != "stored_daily_baseline_total_delta":
        raise AssertionError(f"promotion metric did not record daily-baseline basis: {metric}")


if __name__ == "__main__":
    raise SystemExit(main())
