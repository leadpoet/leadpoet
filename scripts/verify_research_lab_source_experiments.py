#!/usr/bin/env python3
"""Verify the source-experiments build (sourceexperiments.md W1–W6).

Offline deploy check: exercises every workstream's contracts end-to-end with
fixtures (no network, docker, or database) and confirms the SQL bundle carries
the expected objects. Run before/after deploy alongside the pytest suite:

    python3 scripts/verify_research_lab_source_experiments.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _assert(condition: bool, label: str) -> None:
    if not condition:
        raise AssertionError(label)


def verify_w1_ranged_reads(tmp: Path) -> str:
    from gateway.research_lab.code_build import (
        ParentImageSourceContext,
        resolve_source_inspection_requests,
    )
    from research_lab.code_editing import CodeEditSourceInspectionRequest

    target = tmp / "sourcing_model" / "big.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("".join(f"line {i}\n" for i in range(1, 901)), encoding="utf-8")
    context = ParentImageSourceContext(
        source_root=tmp,
        source_mode="parent_image_extract",
        parent_image_digest_hash="sha256:x",
        source_tree_hash="sha256:y",
        top_level_paths=("sourcing_model/",),
        editable_files=("sourcing_model/big.py",),
        file_previews=(),
    )
    kwargs = dict(
        already_read_paths=(),
        max_files=12,
        max_file_bytes=24_000,
        max_total_bytes=120_000,
        max_search_matches=30,
    )
    ranged = resolve_source_inspection_requests(
        context,
        [CodeEditSourceInspectionRequest(operation="read_file", path="sourcing_model/big.py", start_line=400, max_lines=50)],
        source_access_v2=True,
        **kwargs,
    )
    (result,) = ranged.model_context["results"]
    _assert(result["start_line"] == 400 and result["end_line"] == 449, "W1 ranged read honors start_line/max_lines")
    _assert(result["total_line_count"] == 900, "W1 returns full-file total_line_count")
    legacy = resolve_source_inspection_requests(
        context,
        [CodeEditSourceInspectionRequest(operation="read_file", path="sourcing_model/big.py", start_line=400, max_lines=50)],
        **kwargs,
    )
    (legacy_result,) = legacy.model_context["results"]
    _assert("start_line" not in legacy_result, "W1 v2-off keeps legacy read shape")
    return "W1 ranged reads + legacy parity"


def verify_w2_digest() -> str:
    from gateway.research_lab.provider_outcome_digest import (
        build_provider_outcome_digest,
        parse_provider_error_marker_line,
    )

    parsed = parse_provider_error_marker_line(
        "research_lab_private_runtime_provider_error HTTPError: x; status=429; url=https://api.exa.ai/search"
    )
    _assert(parsed == {"provider": "exa", "error_class": "HTTPError", "http_status": 429}, "W2 marker parse")
    digest = build_provider_outcome_digest(
        usage_ledger_rows=[{"provider_id": "exa", "endpoint_class": "/search", "status": 429, "evidence": "recorded"}],
        candidate_snapshot_miss_counts={"node-1": 3},
    )
    _assert(digest["providers"]["exa"]["status_histogram"]["429"] == 1, "W2 aggregates ledger rows")
    _assert("snapshot_miss_note" in digest, "W2 snapshot-miss interpretation note")
    return "W2 provider-outcome digest"


def verify_w3_proxy() -> str:
    from gateway.research_lab.provider_evidence_proxy import (
        provider_registry_hash,
        resolve_provider_credential,
        seed_provider_registry,
        validate_provider_registry_entries,
    )

    entries = seed_provider_registry()
    _assert(validate_provider_registry_entries(entries) == [], "W3 seed registry valid")
    _assert(provider_registry_hash(entries).startswith("sha256:"), "W3 registry hash")
    import os

    saved = {name: os.environ.pop(name, None) for name in (
        "RESEARCH_LAB_SCRAPINGDOG_API_KEY", "SCRAPINGDOG_API_KEY", "QUALIFICATION_SCRAPINGDOG_API_KEY",
    )}
    try:
        os.environ["QUALIFICATION_SCRAPINGDOG_API_KEY"] = "fulfillment"
        sd = next(entry for entry in entries if entry.id == "sd")
        value, _ = resolve_provider_credential(sd, key_split=True)
        _assert(value == "", "W3 key split removes QUALIFICATION_* fallback")
        value, source = resolve_provider_credential(sd, key_split=False)
        _assert(source == "QUALIFICATION_SCRAPINGDOG_API_KEY", "W3 legacy fallback preserved when split off")
    finally:
        os.environ.pop("QUALIFICATION_SCRAPINGDOG_API_KEY", None)
        for name, value in saved.items():
            if value is not None:
                os.environ[name] = value
    return "W3 registry + key split"


def verify_w4_probes() -> str:
    from gateway.research_lab.provider_probe import (
        ProbeBudgetState,
        hash_private_window_terms,
        probe_query_guard,
        resolve_provider_probe,
    )
    from research_lab.code_editing import CodeEditSourceInspectionRequest
    from research_lab.probe_catalog import default_probe_catalog, validate_probe_catalog

    _assert(validate_probe_catalog(default_probe_catalog()) == [], "W4 default catalog valid")
    hashes = hash_private_window_terms(["Acme Robotics"])
    _assert(
        probe_query_guard({"query": "intent for acme robotics"}, private_window_term_hashes=hashes)
        == "private_window_term",
        "W4 private-window guard",
    )
    budget = ProbeBudgetState(max_probes=0)
    resolution = resolve_provider_probe(
        CodeEditSourceInspectionRequest(
            operation="probe_provider", endpoint="exa.search", params={"query": "b2b data"}
        ),
        catalog=default_probe_catalog(),
        proxy_url="http://127.0.0.1:1",
        budget=budget,
        live_enabled=False,
    )
    _assert(resolution.outcome == "budget_exhausted", "W4 budget cap short-circuits")
    return "W4 probe catalog + guard + budget"


def verify_w5_funnel() -> str:
    from research_lab.source_add_execution import (
        SourceAddFunnelStage,
        apply_trial_result,
        evaluate_source_add_acceptance,
        intake_source_add_submission,
        run_llm_review_stage,
        run_sandboxed_trial,
        run_static_scan_stage,
    )

    manifest = {
        "adapter_id": "adapter:verify-1",
        "miner_ref": "miner:verify",
        "source_name": "Verify Source",
        "source_kind": "news",
        "declared_base_domains": ["verify.example"],
        "output_schema_ref": "schema:source-add-output:v1",
        "allowed_output_fields": ["evidence_refs", "snapshot_refs", "content_hashes", "normalized_text_hashes"],
        "submitted_artifact_ref": "artifact:verify",
        "code_bundle_hash": "sha256:" + "a" * 64,
        "sandbox_policy_ref": "policy:sandbox-v1",
        "max_trial_cost_cents": 100,
        "max_request_cost_cents": 5,
        "max_latency_ms": 30_000,
        "fixture_refs": ["fixture:verify"],
    }
    record, errors = intake_source_add_submission(manifest, miner_hotkey="hk-verify")
    _assert(not errors and record is not None, "W5 intake admits a valid submission")
    record = run_static_scan_stage(record, {"adapter.py": "import json"})
    record = run_llm_review_stage(record, llm_reviewer=lambda _r: {"verdict": "pass"})

    def _runner(rec, icp_ref):
        return {
            "output": {
                "output_ref": f"output:{icp_ref}",
                "adapter_id": rec.adapter_id,
                "icp_ref": icp_ref,
                "evidence_refs": [f"evidence:{icp_ref}:0"],
                "snapshot_refs": [f"snapshot:{icp_ref}"],
                "content_hashes": ["sha256:" + "b" * 64],
                "normalized_text_hashes": ["sha256:" + "c" * 64],
            },
            "cost_cents": 1,
        }

    trial = run_sandboxed_trial(
        record,
        trial_icp_refs=("icp:1", "icp:2"),
        sandbox_runner=_runner,
        evidence_classifier=lambda _ref: "news",
    )
    record = apply_trial_result(record, trial)
    accepted, entry = evaluate_source_add_acceptance(
        record, human_gate_passed=True, accepted_at="2026-07-06T00:00:00Z"
    )
    _assert(accepted.stage == SourceAddFunnelStage.ACCEPTED.value and entry is not None, "W5 acceptance")
    # Category mismatch scores ~0.
    mismatch = run_sandboxed_trial(
        record,
        trial_icp_refs=("icp:1",),
        sandbox_runner=_runner,
        evidence_classifier=lambda _ref: "firmographic",
    )
    _assert(mismatch.failure_reason == "category_mismatch", "W5 category-scoped yield")
    return "W5 funnel end-to-end"


def verify_w6_rewards() -> str:
    from research_lab.source_add_rewards import (
        create_leg1_reward,
        create_leg2_reward,
        evaluate_leg2_trigger,
        validate_source_add_reward_record,
    )

    leg1 = create_leg1_reward(adapter_id="adapter:verify-1", miner_ref="hk-verify", start_epoch=10)
    _assert(leg1.alpha_percent == 1.0 and validate_source_add_reward_record(leg1) == [], "W6 leg-1")
    armed, blockers, evidence = evaluate_leg2_trigger(
        adapter_id="adapter:verify-1",
        catalog_registry_ids=("verifysource",),
        merged=True,
        merged_diff_routed_registry_ids=("verifysource",),
        merge_cleared_score_bar=True,
        shadow_monitor_live=True,
        shadow_window_days_elapsed=7.1,
        shadow_window_survived=True,
        ablation_adapter_on_score=6.0,
        ablation_adapter_off_score=5.3,
        now="2026-08-01T00:00:00Z",
        accepted_at="2026-07-06T00:00:00Z",
    )
    _assert(armed, f"W6 leg-2 trigger arms: {blockers}")
    leg2 = create_leg2_reward(
        adapter_id="adapter:verify-1",
        adapter_owner_miner_ref="hk-verify",
        start_epoch=200,
        trigger_evidence=evidence,
    )
    _assert(leg2.alpha_percent == 5.0 and leg2.reward_kind == "source_implementation", "W6 leg-2")
    row = leg2.champion_reward_row()
    _assert(row["reward_kind"] == "source_implementation" and row["desired_alpha_percent"] == 5.0, "W6 rails row")
    return "W6 two-leg rewards"


def verify_sql_bundle() -> str:
    sql = (REPO_ROOT / "scripts" / "72-research-lab-source-experiments.sql").read_text(encoding="utf-8")
    for expected in (
        "research_lab_provider_registry",
        "research_lab_provider_usage_ledger",
        "research_lab_source_add_submissions",
        "research_lab_source_catalog",
        "research_lab_source_add_reward_obligations",
        "research_lab_source_add_reward_current",
        "'probe_requested'",
        "'probe_resolved'",
        "'probe_blocked'",
    ):
        _assert(expected in sql, f"SQL bundle carries {expected}")
    return "SQL bundle 72"


def main() -> int:
    passed = []
    with tempfile.TemporaryDirectory(prefix="source-experiments-verify-") as tmp:
        passed.append(verify_w1_ranged_reads(Path(tmp)))
    passed.append(verify_w2_digest())
    passed.append(verify_w3_proxy())
    passed.append(verify_w4_probes())
    passed.append(verify_w5_funnel())
    passed.append(verify_w6_rewards())
    passed.append(verify_sql_bundle())
    print(json.dumps({"verified": passed}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
