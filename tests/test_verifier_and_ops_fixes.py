"""Tests for the N2 verifier version gate, the unbiased public split (bug #13
residual), and the operator health script hardening.

Covers:
- paired-mode bundles keep the existing (byte-identical) advisory verdict;
- daily-baseline bundles WITH stored per-ICP baseline scores get a recomputed
  candidate-minus-baseline advisory gate (hand-computed expectations);
- daily-baseline bundles WITHOUT per-ICP baseline scores are explicitly
  not_applicable and never eligible (candidate-vs-zero must not read eligible);
- provider_excluded_icp_ids are skipped on both sides and count as unsuccessful;
- the public benchmark split is deterministic per day, rotates across days, and
  is no longer biased to the baseline's weakest ICPs (env-gated with legacy
  fallback);
- the health script isolates failing checks and queries the arweave anchor view
  with the real column names.
"""

from __future__ import annotations

import importlib.util
import math
import os
import sys
from datetime import datetime, timedelta, timezone

import pytest

from leadpoet_verifier.research_evaluation import (
    IMPROVEMENT_GATE_NOT_APPLICABLE,
    REFERENCE_EVALUATION_MODE_DAILY_BASELINE,
    REFERENCE_EVALUATION_MODE_PAIRED,
    SCORE_BUNDLE_SCHEMA_VERSION,
    SCORE_BUNDLE_TYPE,
    SUPERSEDED_METRIC_BLOCKER,
    build_research_evaluation_score_bundle,
    compute_evaluation_aggregates,
    detect_reference_evaluation_mode,
    evaluate_daily_baseline_improvement_gate,
    evaluate_improvement_gate,
    score_bundle_hash,
    score_bundle_to_weight_input,
    verify_research_evaluation_score_bundle,
)
from gateway.research_lab.public_benchmarks import (
    PUBLIC_SPLIT_UNBIASED_ENV,
    SPLIT_POLICY_LEGACY,
    SPLIT_POLICY_UNBIASED,
    build_benchmark_visibility_split,
)


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HEALTH_SCRIPT_PATH = os.path.join(REPO_ROOT, "scripts", "check_research_lab_operator_health.py")


def _sha(label: str) -> str:
    return "sha256:" + (label * 64)[:64]


def _bundle_kwargs(per_icp_results, policy):
    return dict(
        run_id="run-1",
        ticket_id="ticket-1",
        miner_hotkey="hotkey-1",
        island="core",
        evaluation_epoch=7,
        parent_artifact_hash=_sha("a"),
        candidate_artifact_hash=_sha("b"),
        private_model_manifest_hash=_sha("c"),
        candidate_patch_hash=_sha("d"),
        icp_set_hash=_sha("e"),
        scoring_version="scoring-v1",
        evaluator_version="evaluator-v1",
        per_icp_results=per_icp_results,
        evidence_bundle_refs=("evidence:1",),
        execution_trace_ref="trace:1",
        cost_ledger_ref="ledger:1",
        benchmark_split_ref="split:1",
        policy=policy,
        signature_ref="sig:1",
    )


def _paired_rows():
    return [
        {"icp_ref": "icp:a", "icp_hash": _sha("1"), "base_company_scores": [80, 60], "candidate_company_scores": [90, 70]},
        {"icp_ref": "icp:b", "icp_hash": _sha("2"), "base_company_scores": [40], "candidate_company_scores": [70, 20]},
        {"icp_ref": "icp:c", "icp_hash": _sha("3"), "base_company_scores": [50, 0], "candidate_company_scores": [65, 5]},
    ]


def _daily_rows():
    return [
        {"icp_ref": "icp:a", "icp_hash": _sha("1"), "base_company_scores": [], "candidate_company_scores": [50, 50, 50, 50, 50]},
        {"icp_ref": "icp:b", "icp_hash": _sha("2"), "base_company_scores": [], "candidate_company_scores": [30, 30, 30, 30, 30]},
        {"icp_ref": "icp:c", "icp_hash": _sha("3"), "base_company_scores": [], "candidate_company_scores": [20, 20, 20, 20, 20]},
    ]


def _legacy_daily_bundle(per_icp_results, policy):
    """Replicate the pre-fix builder: naive candidate-vs-zero improvement gate.

    This is the exact shape of post-52a9edce bundles created before the version
    gate landed: empty base scores, no reference_evaluation_mode annotation, and
    an always-eligible recorded gate.
    """
    aggregates = compute_evaluation_aggregates(per_icp_results)
    aggregates = {**aggregates, "total_cost_usd": round(float(policy.get("observed_cost_usd", 0.0)), 6)}
    gate = evaluate_improvement_gate(aggregates, policy)
    bundle = {
        "schema_version": SCORE_BUNDLE_SCHEMA_VERSION,
        "bundle_type": SCORE_BUNDLE_TYPE,
        "run_id": "run-1",
        "ticket_id": "ticket-1",
        "miner_hotkey": "hotkey-1",
        "island": "core",
        "evaluation_epoch": 7,
        "parent_artifact_hash": _sha("a"),
        "candidate_artifact_hash": _sha("b"),
        "private_model_manifest_hash": _sha("c"),
        "candidate_patch_hash": _sha("d"),
        "icp_set_hash": _sha("e"),
        "scoring_version": "scoring-v1",
        "evaluator_version": "evaluator-v1",
        "benchmark_split_ref": "split:1",
        "evidence_bundle_refs": ["evidence:1"],
        "execution_trace_ref": "trace:1",
        "cost_ledger_ref": "ledger:1",
        "aggregates": aggregates,
        "improvement_gate": gate,
        "score_bundle_hash": "",
        "anchored_hash": "",
        "signature_ref": "sig:1",
        "reward_path": {
            "eligible_for_probation": gate["eligible_for_probation"],
            "eligible_for_crown": False,
            "eligible_for_improvement_grant": False,
            "reason": "probation_gate_only; crown/grant require later gated workflow",
        },
    }
    bundle_hash = score_bundle_hash(bundle)
    return {**bundle, "score_bundle_hash": bundle_hash, "anchored_hash": bundle_hash}


# ---------------------------------------------------------------------------
# 1. Paired-mode bundles: unchanged verdict.
# ---------------------------------------------------------------------------


class TestPairedModeUnchanged:
    POLICY = {
        "min_delta": 2.0,
        "min_successful_icps": 3,
        "max_hard_failures": 0,
        "min_candidate_score": 15.0,
        "observed_cost_usd": 1.25,
    }

    def test_paired_bundle_detected_and_verdict_unchanged(self):
        bundle = build_research_evaluation_score_bundle(**_bundle_kwargs(_paired_rows(), self.POLICY))
        assert detect_reference_evaluation_mode(bundle) == REFERENCE_EVALUATION_MODE_PAIRED
        # Recorded gate is exactly the legacy paired computation.
        expected_gate = evaluate_improvement_gate(bundle["aggregates"], self.POLICY)
        assert bundle["improvement_gate"] == expected_gate
        assert bundle["improvement_gate"]["decision"] == "eligible_for_probation"
        # Hand-computed paired deltas: (32-28) + (18-8) + (14-10) over 3 = 6.0.
        assert bundle["aggregates"]["mean_delta"] == pytest.approx(6.0)
        # No mode annotations leak into paired aggregates (hash compatibility).
        assert "reference_evaluation_mode" not in bundle["aggregates"]

        verification = verify_research_evaluation_score_bundle(bundle, policy=self.POLICY)
        assert verification["passed"], verification["errors"]
        assert verification["reference_evaluation_mode"] == REFERENCE_EVALUATION_MODE_PAIRED
        assert verification["advisory_improvement_gate"] is None
        assert verification["eligible_for_probation"] is True

        weight_input = score_bundle_to_weight_input(bundle)
        assert weight_input["mean_delta"] == pytest.approx(6.0)
        assert weight_input["eligible_for_probation"] is True
        assert "recorded_mean_delta" not in weight_input

    def test_paired_not_eligible_stays_not_eligible(self):
        policy = {**self.POLICY, "min_delta": 100.0}
        bundle = build_research_evaluation_score_bundle(**_bundle_kwargs(_paired_rows(), policy))
        assert bundle["improvement_gate"]["decision"] == "not_eligible"
        verification = verify_research_evaluation_score_bundle(bundle, policy=policy)
        assert verification["passed"], verification["errors"]
        assert verification["eligible_for_probation"] is False


# ---------------------------------------------------------------------------
# 2. Daily-baseline bundles WITH stored per-ICP baseline scores.
# ---------------------------------------------------------------------------


class TestDailyBaselineWithPerIcpScores:
    BASELINES = {"icp:a": 40.0, "icp:b": 28.0, "icp:c": 25.0}
    POLICY = {
        "min_delta": 1.0,
        "min_successful_icps": 2,
        "baseline_per_icp_scores": BASELINES,
    }

    # Hand-computed: candidate per-ICP scores are 50, 30, 20 (capped sum / 5);
    # deltas vs stored baseline are +10, +2, -5 -> mean 7/3.
    EXPECTED_MEAN = 7.0 / 3.0
    EXPECTED_SD = math.sqrt(((10 - 7 / 3) ** 2 + (2 - 7 / 3) ** 2 + (-5 - 7 / 3) ** 2) / 2)
    EXPECTED_LCB = EXPECTED_MEAN - 1.96 * (EXPECTED_SD / math.sqrt(3))

    def test_recomputed_deltas_and_eligibility(self):
        bundle = build_research_evaluation_score_bundle(**_bundle_kwargs(_daily_rows(), self.POLICY))
        assert detect_reference_evaluation_mode(bundle) == REFERENCE_EVALUATION_MODE_DAILY_BASELINE
        assert bundle["aggregates"]["reference_evaluation_mode"] == REFERENCE_EVALUATION_MODE_DAILY_BASELINE
        assert bundle["aggregates"]["baseline_per_icp_scores"] == {
            "icp:a": 40.0,
            "icp:b": 28.0,
            "icp:c": 25.0,
        }

        gate = bundle["improvement_gate"]
        assert gate["advisory_basis"] == "recomputed_candidate_vs_stored_daily_baseline_per_icp"
        assert gate["mean_delta"] == pytest.approx(self.EXPECTED_MEAN, abs=1e-6)
        assert gate["delta_lcb"] == pytest.approx(self.EXPECTED_LCB, abs=1e-5)
        assert gate["baseline_score"] == pytest.approx((40.0 + 28.0 + 25.0) / 3.0, abs=1e-6)
        assert gate["candidate_score"] == pytest.approx(100.0 / 3.0, abs=1e-6)
        assert gate["eligible_for_probation"] is True  # 2.33 >= min_delta 1.0

        verification = verify_research_evaluation_score_bundle(bundle, policy=self.POLICY)
        assert verification["passed"], verification["errors"]
        advisory = verification["advisory_improvement_gate"]
        assert advisory is not None
        assert advisory["mean_delta"] == pytest.approx(self.EXPECTED_MEAN, abs=1e-6)
        assert verification["eligible_for_probation"] is True

        weight_input = score_bundle_to_weight_input(bundle)
        assert weight_input["mean_delta"] == pytest.approx(self.EXPECTED_MEAN, abs=1e-6)
        assert weight_input["delta_lcb"] == pytest.approx(self.EXPECTED_LCB, abs=1e-5)
        assert weight_input["delta_metrics_basis"] == "recomputed_vs_stored_daily_baseline"
        # Candidate-vs-zero values stay visible only under recorded_* keys.
        assert weight_input["recorded_mean_delta"] == pytest.approx(100.0 / 3.0, abs=1e-5)

    def test_below_threshold_recomputed_delta_not_eligible(self):
        policy = {**self.POLICY, "min_delta": 5.0}
        bundle = build_research_evaluation_score_bundle(**_bundle_kwargs(_daily_rows(), policy))
        assert bundle["improvement_gate"]["eligible_for_probation"] is False
        assert "delta_below_minimum" in bundle["improvement_gate"]["blockers"]
        verification = verify_research_evaluation_score_bundle(bundle, policy=policy)
        assert verification["passed"], verification["errors"]
        assert verification["eligible_for_probation"] is False


# ---------------------------------------------------------------------------
# 3. Daily-baseline bundles WITHOUT per-ICP baseline scores: never eligible.
# ---------------------------------------------------------------------------


class TestDailyBaselineWithoutPerIcpScores:
    POLICY = {"min_delta": 1.0, "min_successful_icps": 2}

    def test_new_builder_records_not_applicable(self):
        bundle = build_research_evaluation_score_bundle(**_bundle_kwargs(_daily_rows(), self.POLICY))
        gate = bundle["improvement_gate"]
        assert gate["decision"] == IMPROVEMENT_GATE_NOT_APPLICABLE
        assert gate["eligible_for_probation"] is False
        assert SUPERSEDED_METRIC_BLOCKER in gate["blockers"]
        assert gate["reason"]
        assert bundle["reward_path"]["eligible_for_probation"] is False
        # Recorded aggregates are candidate-vs-zero (~33.3) and would trivially
        # clear min_delta -- the whole point is that they must not.
        assert bundle["aggregates"]["mean_delta"] == pytest.approx(100.0 / 3.0, abs=1e-5)

        verification = verify_research_evaluation_score_bundle(bundle, policy=self.POLICY)
        assert verification["passed"], verification["errors"]
        assert verification["eligible_for_probation"] is False
        assert verification["eligible_for_probation_reason"]
        assert verification["advisory_improvement_gate"]["decision"] == IMPROVEMENT_GATE_NOT_APPLICABLE

    def test_legacy_always_eligible_daily_bundle_reads_not_eligible(self):
        """Regression for N2: historical daily bundles recorded always-eligible."""
        bundle = _legacy_daily_bundle(_daily_rows(), {**self.POLICY, "observed_cost_usd": 0.0})
        # The legacy recorded gate really is the inflated candidate-vs-zero one.
        assert bundle["improvement_gate"]["eligible_for_probation"] is True
        assert detect_reference_evaluation_mode(bundle) == REFERENCE_EVALUATION_MODE_DAILY_BASELINE

        verification = verify_research_evaluation_score_bundle(bundle, policy=self.POLICY)
        # Integrity holds (never rewrite historical bundles)...
        assert verification["passed"], verification["errors"]
        # ...but the candidate-vs-zero delta must never read as eligible.
        assert verification["eligible_for_probation"] is False
        assert verification["advisory_improvement_gate"]["decision"] == IMPROVEMENT_GATE_NOT_APPLICABLE

        weight_input = score_bundle_to_weight_input(bundle)
        assert weight_input["eligible_for_probation"] is False
        assert weight_input["mean_delta"] == 0.0
        assert weight_input["delta_lcb"] == 0.0
        assert weight_input["delta_metrics_basis"] == "superseded_metric_not_applicable"
        assert weight_input["recorded_mean_delta"] == pytest.approx(100.0 / 3.0, abs=1e-5)

    def test_declared_mode_wins_over_telltale(self):
        rows = _paired_rows()  # non-empty base scores
        aggregates = compute_evaluation_aggregates(rows)
        bundle = {"aggregates": {**aggregates, "reference_evaluation_mode": "stored_daily_baseline"}}
        assert detect_reference_evaluation_mode(bundle) == REFERENCE_EVALUATION_MODE_DAILY_BASELINE
        bundle_paired = {"aggregates": aggregates, "reference_evaluation_mode": "paired_base"}
        assert detect_reference_evaluation_mode(bundle_paired) == REFERENCE_EVALUATION_MODE_PAIRED


# ---------------------------------------------------------------------------
# 4. provider_excluded_icp_ids handling.
# ---------------------------------------------------------------------------


class TestProviderExcludedIcps:
    def test_excluded_icps_skipped_on_both_sides(self):
        policy = {
            "min_delta": 1.0,
            "min_successful_icps": 2,
            # No baseline for icp:c on purpose: exclusion must remove it from
            # both sides before coverage is checked.
            "baseline_per_icp_scores": {"icp:a": 40.0, "icp:b": 28.0},
            "provider_excluded_icp_ids": ["icp:c"],
        }
        bundle = build_research_evaluation_score_bundle(**_bundle_kwargs(_daily_rows(), policy))
        gate = bundle["improvement_gate"]
        assert gate["advisory_basis"] == "recomputed_candidate_vs_stored_daily_baseline_per_icp"
        # deltas: +10 (a) and +2 (b); icp:c skipped entirely -> mean 6.0
        assert gate["mean_delta"] == pytest.approx(6.0)
        assert gate["compared_icp_count"] == 2
        assert gate["provider_excluded_icp_count"] == 1
        assert gate["provider_excluded_icp_ids"] == ["icp:c"]
        assert gate["eligible_for_probation"] is True
        assert bundle["aggregates"]["provider_excluded_icp_ids"] == ["icp:c"]

        verification = verify_research_evaluation_score_bundle(bundle, policy=policy)
        assert verification["passed"], verification["errors"]
        assert verification["advisory_improvement_gate"]["mean_delta"] == pytest.approx(6.0)

    def test_excluded_icps_count_as_unsuccessful(self):
        policy = {
            "min_delta": 1.0,
            "min_successful_icps": 3,
            "baseline_per_icp_scores": {"icp:a": 40.0, "icp:b": 28.0, "icp:c": 25.0},
            "provider_excluded_icp_ids": ["icp:c"],
        }
        bundle = build_research_evaluation_score_bundle(**_bundle_kwargs(_daily_rows(), policy))
        gate = bundle["improvement_gate"]
        assert gate["successful_icp_count"] == 2
        assert gate["eligible_for_probation"] is False
        assert "insufficient_successful_icps" in gate["blockers"]

    def test_absence_is_tolerated(self):
        aggregates = compute_evaluation_aggregates(_daily_rows())
        gate = evaluate_daily_baseline_improvement_gate(
            aggregates,
            {"min_delta": 1.0, "min_successful_icps": 2},
            baseline_per_icp_scores={"icp:a": 40.0, "icp:b": 28.0, "icp:c": 25.0},
        )
        assert gate["provider_excluded_icp_count"] == 0
        assert gate["compared_icp_count"] == 3

    def test_evaluator_contract_shape_gate_and_top_level(self):
        """The evaluator writes provider_excluded_icp_ids into the bundle's
        private_holdout_gate and at top level (never in aggregates); per-ICP
        baseline scores may arrive in the gate as visibility_split-style rows.
        The verifier must consume that exact shape."""
        legacy = _legacy_daily_bundle(_daily_rows(), {"min_delta": 1.0, "min_successful_icps": 2, "observed_cost_usd": 0.0})
        bundle = {
            **{key: value for key, value in legacy.items() if key not in {"score_bundle_hash", "anchored_hash"}},
            "private_holdout_gate": {
                "gate_type": "public_score_before_private_holdout",
                "decision": "private_holdout_approved",
                "reference_evaluation_mode": "stored_daily_baseline",
                "provider_excluded_icp_ids": ["icp:c"],
                "baseline_per_icp_scores": [
                    {"icp_ref": "icp:a", "score": 40.0},
                    {"icp_ref": "icp:b", "score": 28.0},
                ],
            },
            "provider_excluded_icp_ids": ["icp:c"],
            "score_bundle_hash": "",
            "anchored_hash": "",
        }
        bundle_hash = score_bundle_hash(bundle)
        bundle = {**bundle, "score_bundle_hash": bundle_hash, "anchored_hash": bundle_hash}

        verification = verify_research_evaluation_score_bundle(
            bundle, policy={"min_delta": 1.0, "min_successful_icps": 2}
        )
        assert verification["passed"], verification["errors"]
        assert verification["reference_evaluation_mode"] == REFERENCE_EVALUATION_MODE_DAILY_BASELINE
        advisory = verification["advisory_improvement_gate"]
        # icp:c excluded on both sides; deltas +10, +2 -> mean 6.0, eligible.
        assert advisory["mean_delta"] == pytest.approx(6.0)
        assert advisory["provider_excluded_icp_ids"] == ["icp:c"]
        assert verification["eligible_for_probation"] is True

    def test_missing_baseline_coverage_fails_closed(self):
        aggregates = compute_evaluation_aggregates(_daily_rows())
        gate = evaluate_daily_baseline_improvement_gate(
            aggregates,
            {"min_delta": 1.0},
            baseline_per_icp_scores={"icp:a": 40.0},  # icp:b / icp:c missing
        )
        assert gate["decision"] == IMPROVEMENT_GATE_NOT_APPLICABLE
        assert gate["eligible_for_probation"] is False
        assert "baseline_per_icp_coverage_incomplete" in gate["blockers"]
        assert "icp:b" in gate["reason"]


# ---------------------------------------------------------------------------
# 5. Public benchmark split: deterministic, per-day rotating, unbiased.
# ---------------------------------------------------------------------------


def _split_fixture(count: int = 10):
    benchmark_items = []
    per_icp_summaries = []
    for index in range(count):
        ref = f"icp:1:{index}"
        icp_hash = _sha(str(index % 10))
        benchmark_items.append(
            {
                "icp_ref": ref,
                "icp_hash": icp_hash,
                "set_id": 1,
                "day_index": 0,
                "day_rank": index + 1,
                "icp": {"industry": f"industry-{index}"},
            }
        )
        per_icp_summaries.append(
            {
                "icp_ref": ref,
                "icp_hash": icp_hash,
                "score": float((index + 1) * 10),
                "company_count": 5,
            }
        )
    return benchmark_items, per_icp_summaries


def _public_refs(split):
    return {str(item["icp_ref"]) for item in split["items"] if item["visibility"] == "public"}


WINDOW_HASHES = [_sha(ch) for ch in "123456"]


class TestPublicSplitUnbiased:
    def _split(self, window_hash):
        items, summaries = _split_fixture()
        return build_benchmark_visibility_split(
            rolling_window_hash=window_hash,
            benchmark_items=items,
            per_icp_summaries=summaries,
        )

    def test_same_day_same_window_is_deterministic(self, monkeypatch):
        monkeypatch.delenv(PUBLIC_SPLIT_UNBIASED_ENV, raising=False)
        first = self._split(WINDOW_HASHES[0])
        second = self._split(WINDOW_HASHES[0])
        assert _public_refs(first) == _public_refs(second)
        assert first["split_policy"] == SPLIT_POLICY_UNBIASED

    def test_different_day_window_rotates_subset(self, monkeypatch):
        monkeypatch.delenv(PUBLIC_SPLIT_UNBIASED_ENV, raising=False)
        subsets = {frozenset(_public_refs(self._split(window))) for window in WINDOW_HASHES}
        assert len(subsets) > 1, "public subset must rotate across daily windows"

    def test_not_biased_to_weakest_icps(self, monkeypatch):
        monkeypatch.delenv(PUBLIC_SPLIT_UNBIASED_ENV, raising=False)
        # Legacy behavior always exposes the two absolute weakest ICPs plus the
        # absolute strongest one. Across several daily windows the unbiased
        # selection must diverge from those extremes.
        legacy_extremes = frozenset({"icp:1:0", "icp:1:1", "icp:1:9"})
        subsets = [frozenset(_public_refs(self._split(window))) for window in WINDOW_HASHES]
        assert any(subset != legacy_extremes for subset in subsets)

    def test_pool_composition_still_honored(self, monkeypatch):
        monkeypatch.delenv(PUBLIC_SPLIT_UNBIASED_ENV, raising=False)
        split = self._split(WINDOW_HASHES[0])
        assert split["public_count"] == 3
        assert split["public_strength_counts"] == {"strong": 1, "weak": 2}
        weak_refs = {f"icp:1:{i}" for i in range(5)}  # bottom half by score
        public_weak = {
            str(item["icp_ref"])
            for item in split["items"]
            if item["visibility"] == "public" and item["strength_label"] == "weak"
        }
        assert public_weak <= weak_refs and len(public_weak) == 2

    def test_legacy_fallback_flag(self, monkeypatch):
        monkeypatch.setenv(PUBLIC_SPLIT_UNBIASED_ENV, "false")
        split = self._split(WINDOW_HASHES[0])
        assert split["split_policy"] == SPLIT_POLICY_LEGACY
        assert _public_refs(split) == {"icp:1:0", "icp:1:1", "icp:1:9"}

    def test_flag_default_is_unbiased(self, monkeypatch):
        monkeypatch.delenv(PUBLIC_SPLIT_UNBIASED_ENV, raising=False)
        split = self._split(WINDOW_HASHES[0])
        assert split["split_policy"] == SPLIT_POLICY_UNBIASED


# ---------------------------------------------------------------------------
# 6. Health script: check isolation, fixed arweave query, new checks.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def health():
    spec = importlib.util.spec_from_file_location("check_research_lab_operator_health", HEALTH_SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestHealthScript:
    def test_failing_check_is_isolated_and_reported(self, health):
        def ok_check():
            return ["ok_line_one"]

        def boom_check():
            raise ValueError("simulated schema drift")

        def second_ok_check():
            return ["ok_line_two"]

        printed: list[str] = []
        code = health.run_health_checks(
            checks=[("ok", ok_check), ("boom", boom_check), ("ok2", second_ok_check)],
            printer=printed.append,
        )
        assert code == 1
        assert "ok_line_one" in printed
        assert "ok_line_two" in printed, "checks after a failure must still run"
        failed_lines = [line for line in printed if line.startswith("CHECK-FAILED boom")]
        assert failed_lines and "simulated schema drift" in failed_lines[0]

    def test_all_checks_passing_returns_zero(self, health):
        printed: list[str] = []
        code = health.run_health_checks(checks=[("ok", lambda: ["fine"])], printer=printed.append)
        assert code == 0
        assert printed == ["fine"]

    def test_arweave_query_uses_current_view_columns(self, health):
        captured: dict = {}

        def fake_fetch(table, params):
            captured["table"] = table
            captured["params"] = params
            return [
                {
                    "epoch": 12,
                    "audit_kind": "active",
                    "current_anchor_status": "checkpointed",
                    "current_arweave_tx_id": "tx-abc123",
                    "current_status_at": "2026-07-01T00:00:00Z",
                    "payload_hash": _sha("f"),
                }
            ]

        lines = health.check_arweave_anchors(fetch=fake_fetch)
        assert captured["table"] == "research_lab_arweave_epoch_audit_anchor_current"
        select = captured["params"]["select"]
        assert "current_arweave_tx_id" in select
        # The old bare column caused the HTTP 400 (schema drift): the _current
        # view only exposes the events-side alias.
        assert ",arweave_tx_id" not in f",{select}".replace("current_arweave_tx_id", "")
        assert any("tx=tx-abc123" in line for line in lines)
        assert any("checkpointed" in line for line in lines)

    def test_main_without_env_exits_2(self, health, monkeypatch, capsys):
        monkeypatch.delenv("SUPABASE_URL", raising=False)
        monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)
        assert health.main() == 2
        assert "missing required env vars" in capsys.readouterr().out

    def test_stale_scoring_cards(self, health):
        cards = [
            {"card_id": "public_loop_card:1", "current_event_type": "scoring", "current_run_id": "run-active"},
            {"card_id": "public_loop_card:2", "current_event_type": "scoring", "current_run_id": "run-dead"},
            {"card_id": "public_loop_card:3", "current_event_type": "scored", "current_run_id": "run-dead"},
            {"card_id": "public_loop_card:4", "current_event_type": "scoring", "current_run_id": ""},
        ]
        candidates = [
            {"candidate_id": "c1", "run_id": "run-active", "current_candidate_status": "evaluating"},
            {"candidate_id": "c2", "run_id": "run-dead", "current_candidate_status": "rejected"},
        ]
        stale = health._stale_scoring_cards(cards, candidates)
        assert {card["card_id"] for card in stale} == {"public_loop_card:2", "public_loop_card:4"}

    def test_queue_and_loop_health_flags_stale_active_rows(self, health):
        stale_at = (datetime.now(timezone.utc) - timedelta(minutes=20)).isoformat()

        def fake_fetch(table, params):
            if table == "research_loop_run_queue_current":
                return [
                    {
                        "run_id": "run-stale",
                        "ticket_id": "ticket-1",
                        "current_queue_status": "started",
                        "worker_ref": "worker-9",
                        "current_status_at": stale_at,
                    }
                ]
            if table == "research_lab_auto_research_loop_current":
                return [
                    {
                        "run_id": "run-stale",
                        "current_loop_status": "running",
                        "current_event_type": "source_inspection_requested",
                        "current_worker_ref": "worker-9",
                        "current_status_at": stale_at,
                    }
                ]
            return []

        lines = health.check_queue_and_loops(fetch=fake_fetch)
        assert "alert_stale_started_queue_count=1" in lines
        assert "alert_stale_running_loop_count=1" in lines
        assert any("stale_started_queue run=run-stale" in line for line in lines)
        assert any("stale_running_loop run=run-stale" in line for line in lines)

    def test_candidate_health_flags_stale_active_scoring(self, health):
        stale_at = (datetime.now(timezone.utc) - timedelta(minutes=20)).isoformat()

        def fake_fetch(table, params):
            if table == "research_lab_candidate_evaluation_current":
                return [
                    {
                        "candidate_id": "candidate:stale",
                        "run_id": "run-stale",
                        "current_candidate_status": "evaluating",
                        "current_status_at": stale_at,
                        "candidate_patch_manifest": {},
                    }
                ]
            if table == "research_lab_candidate_evaluation_events":
                return []
            return []

        lines = health.check_candidates(fetch=fake_fetch)
        assert "active_scoring_count=1" in lines
        assert "alert_stale_active_scoring_count=1" in lines
        assert any("stale_active_scoring candidate=candidate:stale" in line for line in lines)

    def test_baseline_day_jumps_and_duplicates(self, health):
        rows = [
            {"benchmark_bundle_id": "b1", "benchmark_date": "2026-06-28", "aggregate_score": 27.04, "created_at": "2026-06-28T01:00:00Z", "current_benchmark_status": "completed"},
            {"benchmark_bundle_id": "b2", "benchmark_date": "2026-06-29", "aggregate_score": 19.51, "created_at": "2026-06-29T01:00:00Z", "current_benchmark_status": "completed"},
            {"benchmark_bundle_id": "b3", "benchmark_date": "2026-06-29", "aggregate_score": 16.48, "created_at": "2026-06-29T02:00:00Z", "current_benchmark_status": "completed"},
            {"benchmark_bundle_id": "b4", "benchmark_date": "2026-06-30", "aggregate_score": 17.08, "created_at": "2026-06-30T01:00:00Z", "current_benchmark_status": "completed"},
            {"benchmark_bundle_id": "b5", "benchmark_date": "2026-07-01", "aggregate_score": 16.46, "created_at": "2026-07-01T01:00:00Z", "current_benchmark_status": "failed"},
        ]
        score_by_day = health._latest_completed_score_by_day(rows)
        # The newest completed bundle wins the day; failed bundles are ignored.
        assert score_by_day == {"2026-06-28": 27.04, "2026-06-29": 16.48, "2026-06-30": 17.08}
        jumps = health._baseline_day_jumps(score_by_day, 3.0)
        assert jumps == [("2026-06-28", "2026-06-29", pytest.approx(-10.56))]
        assert health._duplicate_day_counts(rows) == {"2026-06-29": 2}

    def test_baseline_icp_summary_alerts(self, health):
        summaries = [
            {"icp_ref": "icp:1", "score": 0.0, "company_count": 0, "diagnostics": {"failure_categories": ["provider_http_4xx"]}},
            {"icp_ref": "icp:2", "score": 12.0, "company_count": 5, "diagnostics": {"failure_categories": []}},
            {"icp_ref": "icp:3", "score": 0.0, "company_count": 0, "diagnostics": {"failure_categories": []}},
            {"icp_ref": "icp:4", "score": 0.0, "company_count": 0, "diagnostics": {"failure_categories": ["runtime_provider_error"]}},
        ]
        assert health._unresolved_provider_error_icps(summaries) == ["icp:1", "icp:4"]
        assert health._zero_company_no_error_icps(summaries) == ["icp:3"]

    def test_promotions_check_flags_reward_pending_and_missing_activation(self, health):
        def fake_fetch(table, params):
            if table == "research_lab_candidate_evaluation_current":
                return [
                    {"candidate_id": "cand-1", "run_id": "run-1", "current_candidate_status": "scored"},
                ]
            if table == "research_lab_candidate_promotion_events":
                return [
                    {"candidate_id": "cand-1", "event_type": "promotion_passed", "promotion_status": "passed", "created_at": "2026-07-01T00:00:00Z"},
                    {"candidate_id": "cand-2", "event_type": "champion_reward_pending_uid", "promotion_status": "reward_pending_uid", "created_at": "2026-07-01T00:00:00Z"},
                ]
            raise AssertionError(f"unexpected table {table}")

        lines = health.check_promotions(fetch=fake_fetch)
        assert any(line == "alert_reward_pending_uid_events=1" for line in lines)
        assert any(line == "alert_promotion_passed_without_active_version_created=1" for line in lines)
        assert any("promotion_passed_no_activation candidate=cand-1" in line for line in lines)
