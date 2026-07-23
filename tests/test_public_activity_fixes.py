"""Tests for the Research Lab public-activity fixes (audit bugs 7, 33, 34).

Covers:
- latest CHECK-constraint allowlist: every label the projection can emit is
  allowlisted (labels extracted from the module AST, allowlists parsed from the
  migration SQL), and the new allowlist is a strict superset of scripts/61.
- derive_public_loop_outcome precedence fixes (bug 33a-e).
- safe_project_public_loop_activity bounded retry + escalated warning (bug 34b).
- reproject_stale_public_cards sweep: stale selection, batch cap, log-only mode,
  env gate (bug 34c).
- recovery operators fire best-effort public projections (bug 34a).
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

import gateway.research_lab.public_activity as public_activity
import gateway.research_lab.recovery as recovery
from gateway.research_lab.public_activity import (
    derive_public_loop_outcome,
    public_loop_outcome_closes_ticket,
    public_loop_ticket_id_matches_lookup,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
MIGRATION_69 = REPO_ROOT / "scripts" / "69-research-lab-candidate-generation-diagnostics.sql"
MIGRATION_85 = REPO_ROOT / "scripts" / "85-research-lab-unpaid-ticket-expiry.sql"
MIGRATION_61 = REPO_ROOT / "scripts" / "61-research-lab-public-status-check-allowlist.sql"
PUBLIC_ACTIVITY_SOURCE = REPO_ROOT / "gateway" / "research_lab" / "public_activity.py"

T1 = "2026-07-01T00:00:00+00:00"
T2 = "2026-07-01T01:00:00+00:00"
T3 = "2026-07-01T02:00:00+00:00"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _sql_allowlist(sql_text: str, column: str) -> set[str]:
    match = re.search(rf"{column}\s+IN\s*\(([^)]*)\)", sql_text)
    assert match, f"no {column} IN (...) allowlist found"
    return set(re.findall(r"'([^']+)'", match.group(1)))


def _all_sql_allowlisted_values(sql_text: str, column: str) -> set[str]:
    matches = re.findall(rf"{column}\s+IN\s*\(([^)]*)\)", sql_text)
    assert matches, f"no {column} IN (...) allowlist found"
    return {
        value
        for match in matches
        for value in re.findall(r"'([^']+)'", match)
    }


def _labels_writable_by_projection() -> tuple[set[str], set[str], set[str]]:
    """Extract every (event_type, outcome_label, outcome_band) literal triple the
    projection can emit, from PublicLoopOutcome(...) and _result(...) calls."""
    tree = ast.parse(PUBLIC_ACTIVITY_SOURCE.read_text())
    event_types: set[str] = set()
    outcome_labels: set[str] = set()
    outcome_bands: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func_name = node.func.id if isinstance(node.func, ast.Name) else None
        if func_name not in {"PublicLoopOutcome", "_result"}:
            continue
        values = [
            arg.value
            for arg in node.args[:3]
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
        ]
        if len(values) == 3:
            event_types.add(values[0])
            outcome_labels.add(values[1])
            outcome_bands.add(values[2])
    assert event_types, "no PublicLoopOutcome/_result literals found in public_activity.py"
    return event_types, outcome_labels, outcome_bands


def _ticket(status: str = "running") -> dict:
    return {"current_ticket_status": status, "created_at": T1}


def _queue(status: str, reason: str = "", at: str = T2) -> dict:
    return {
        "run_id": "run-1",
        "current_queue_status": status,
        "current_reason": reason,
        "current_status_at": at,
    }


def _candidate(status: str, reason: str = "", at: str = T2, artifact_hash: str = "") -> dict:
    return {
        "candidate_id": f"cand-{status}-{reason or 'none'}",
        "run_id": "run-1",
        "receipt_id": "receipt-1",
        "current_candidate_status": status,
        "current_reason": reason,
        "current_status_at": at,
        "candidate_artifact_hash": artifact_hash,
        "redacted_public_summary": "improved intent scoring",
    }


def _bundle(delta: float, artifact_hash: str = "hash-1", at: str = T2) -> dict:
    return {
        "score_bundle_id": "bundle-1",
        "candidate_artifact_hash": artifact_hash,
        "current_status_at": at,
        "score_bundle_doc": {
            "private_holdout_gate": {
                "decision": "private_holdout_approved",
                "private_holdout_evaluated": True,
                "baseline_aggregate_score": 20.0,
                "candidate_total_score": 20.0 + delta,
                "candidate_delta_vs_daily_baseline": delta,
            }
        },
    }


def _promotion(event_type: str, promotion_status: str, at: str = T2, event_doc: dict | None = None) -> dict:
    return {
        "promotion_event_id": f"promo-{event_type}-{at}",
        "candidate_id": "cand-scored-none",
        "event_type": event_type,
        "promotion_status": promotion_status,
        "created_at": at,
        "event_doc": event_doc or {},
    }


def _derive(
    *,
    ticket: dict | None = None,
    queue_rows: list | None = None,
    receipt_rows: list | None = None,
    candidate_rows: list | None = None,
    score_bundle_rows: list | None = None,
    promotion_event_rows: list | None = None,
    auto_loop_event_rows: list | None = None,
):
    return derive_public_loop_outcome(
        ticket=ticket or _ticket(),
        queue_rows=queue_rows or [],
        receipt_rows=receipt_rows or [],
        candidate_rows=candidate_rows or [],
        score_bundle_rows=score_bundle_rows or [],
        promotion_event_rows=promotion_event_rows or [],
        auto_loop_event_rows=auto_loop_event_rows or [],
        improvement_threshold_points=1.0,
    )


def _sweep_config() -> SimpleNamespace:
    return SimpleNamespace(
        api_enabled=True,
        reports_enabled=True,
        public_activity_enabled=True,
        improvement_threshold_points=1.0,
    )


def _opened_ticket_inputs() -> dict:
    """Projection inputs deriving to awaiting_payment (with canonical fields)."""
    return {
        "ticket": _ticket("opened"),
        "queue_rows": [],
        "receipt_rows": [],
        "candidate_rows": [],
        "score_bundle_rows": [],
        "promotion_event_rows": [],
        "auto_loop_event_rows": [],
    }


def _card(ticket_id: str, stored_label: str = "", event_doc: dict | None = None) -> dict:
    return {
        "card_id": f"public_loop_card:{ticket_id}",
        "ticket_id": ticket_id,
        "current_outcome_label": stored_label,
        "current_event_doc": event_doc or {},
        "current_last_activity_at": T2,
        "created_at": T1,
    }


# --------------------------------------------------------------------------- #
# Bug 7 — latest allowlist covers every writable label
# --------------------------------------------------------------------------- #
class TestMigration69Allowlist:
    def test_every_writable_label_is_allowlisted(self):
        sql = MIGRATION_85.read_text()
        allowed_event_types = _all_sql_allowlisted_values(sql, "event_type")
        allowed_outcome_labels = _all_sql_allowlisted_values(sql, "outcome_label")
        allowed_outcome_bands = _all_sql_allowlisted_values(sql, "outcome_band")
        event_types, outcome_labels, outcome_bands = _labels_writable_by_projection()
        assert event_types <= allowed_event_types, (
            f"writable event_type not allowlisted: {sorted(event_types - allowed_event_types)}"
        )
        assert outcome_labels <= allowed_outcome_labels, (
            f"writable outcome_label not allowlisted: {sorted(outcome_labels - allowed_outcome_labels)}"
        )
        assert outcome_bands <= allowed_outcome_bands, (
            f"writable outcome_band not allowlisted: {sorted(outcome_bands - allowed_outcome_bands)}"
        )

    def test_awaiting_payment_allowlisted(self):
        sql = MIGRATION_69.read_text()
        assert "awaiting_payment" in _sql_allowlist(sql, "event_type")
        assert "awaiting_payment" in _sql_allowlist(sql, "outcome_label")

    def test_no_buildable_candidate_allowlisted(self):
        sql = MIGRATION_69.read_text()
        assert "no_buildable_candidate" in _sql_allowlist(sql, "event_type")
        assert "no_buildable_candidate" in _sql_allowlist(sql, "outcome_label")

    def test_strict_superset_of_migration_61(self):
        sql_69 = MIGRATION_69.read_text()
        sql_61 = MIGRATION_61.read_text()
        for column in ("event_type", "outcome_label", "outcome_band"):
            old = _sql_allowlist(sql_61, column)
            new = _sql_allowlist(sql_69, column)
            assert old <= new, f"{column}: migration 69 dropped labels {sorted(old - new)}"
        # Strictly larger overall (no_buildable_candidate added), so old writers keep working
        # while the new label becomes legal.
        assert _sql_allowlist(sql_61, "event_type") < _sql_allowlist(sql_69, "event_type")


# --------------------------------------------------------------------------- #
# Bug 33 — projection precedence
# --------------------------------------------------------------------------- #
class TestDerivationPrecedence:
    def test_scored_beats_failed_sibling(self):
        outcome = _derive(
            queue_rows=[_queue("completed")],
            candidate_rows=[
                _candidate("scored", at=T1, artifact_hash="hash-1"),
                _candidate("failed", reason="candidate_scoring_runtime_error", at=T2),
            ],
            score_bundle_rows=[_bundle(2.0)],
        )
        assert outcome.outcome_label == "scored_promising"
        assert outcome.outcome_band == "passed_threshold"

    def test_all_failed_still_fails_without_scored_history(self):
        outcome = _derive(
            queue_rows=[_queue("completed")],
            candidate_rows=[_candidate("failed", reason="candidate_scoring_runtime_error")],
        )
        assert outcome.outcome_label == "failed"

    @pytest.mark.parametrize(
        "promotion_row",
        [
            # New-style status written by the scoring worker after the rename.
            _promotion("promotion_failed", "post_score_side_effect_failed", at=T3),
            # Legacy shape: bare failed status with the side-effect reason marker.
            _promotion(
                "promotion_failed",
                "failed",
                at=T3,
                event_doc={"reason": "post_score_side_effect_failed"},
            ),
            # Legacy shape: scored-status-preserved marker only.
            _promotion(
                "promotion_failed",
                "failed",
                at=T3,
                event_doc={"candidate_status_preserved": "scored"},
            ),
        ],
    )
    def test_post_score_side_effect_failure_is_ignored(self, promotion_row):
        outcome = _derive(
            queue_rows=[_queue("completed")],
            candidate_rows=[_candidate("scored", at=T1, artifact_hash="hash-1")],
            score_bundle_rows=[_bundle(2.0)],
            promotion_event_rows=[promotion_row],
        )
        assert outcome.outcome_label == "scored_promising"

    def test_bare_promotion_failure_does_not_mask_scored_candidate(self):
        outcome = _derive(
            queue_rows=[_queue("completed")],
            candidate_rows=[_candidate("scored", at=T1, artifact_hash="hash-1")],
            score_bundle_rows=[_bundle(0.5)],
            promotion_event_rows=[
                _promotion("promotion_failed", "failed", at=T3, event_doc={"reason": "merge_crashed"})
            ],
        )
        assert outcome.outcome_label == "scored_promising"
        assert outcome.outcome_band == "small_gain"

    def test_bare_promotion_failure_without_scored_candidate_still_fails(self):
        outcome = _derive(
            queue_rows=[_queue("completed")],
            candidate_rows=[_candidate("rejected", reason="stale_parent_rebase_failed")],
            promotion_event_rows=[
                _promotion("promotion_failed", "failed", at=T3, event_doc={"reason": "merge_crashed"})
            ],
        )
        assert outcome.outcome_label == "failed"

    def test_champion_reward_pending_uid_does_not_regress_merged_champion(self):
        outcome = _derive(
            queue_rows=[_queue("completed")],
            candidate_rows=[_candidate("scored", at=T1, artifact_hash="hash-1")],
            score_bundle_rows=[_bundle(3.0)],
            promotion_event_rows=[
                _promotion("active_version_created", "merged", at=T2),
                _promotion("champion_reward_pending_uid", "reward_pending_uid", at=T3),
            ],
        )
        assert outcome.outcome_label == "promoted"
        assert outcome.outcome_band == "promoted"

    def test_reward_pending_alone_reads_as_promoted(self):
        outcome = _derive(
            queue_rows=[_queue("completed")],
            candidate_rows=[_candidate("scored", at=T1, artifact_hash="hash-1")],
            score_bundle_rows=[_bundle(3.0)],
            promotion_event_rows=[
                _promotion("champion_reward_pending_uid", "reward_pending_uid", at=T3),
            ],
        )
        assert outcome.outcome_label == "promoted"

    def test_all_rejected_loop_is_terminal_failed_regardless_of_reason(self):
        # Reasons outside the old 3-reason allowlist must still be terminal.
        outcome = _derive(
            queue_rows=[_queue("completed")],
            candidate_rows=[
                _candidate("rejected", reason="novelty_duplicate", at=T1),
                _candidate("rejected", reason="judge_rejected", at=T2),
            ],
        )
        assert outcome.outcome_label == "failed"
        assert outcome.outcome_band == "failed"

    def test_requeued_run_does_not_show_complete(self):
        outcome = _derive(
            queue_rows=[_queue("queued", reason="operator_resume_from_checkpoint", at=T3)],
            candidate_rows=[_candidate("rejected", reason="novelty_duplicate", at=T1)],
        )
        assert outcome.outcome_label == "queued"

    def test_restarted_run_shows_running_not_complete(self):
        outcome = _derive(
            queue_rows=[_queue("started", at=T3)],
            candidate_rows=[_candidate("rejected", reason="novelty_duplicate", at=T1)],
        )
        assert outcome.outcome_label == "running"

    def test_requeued_run_with_failed_candidates_shows_queued(self):
        outcome = _derive(
            queue_rows=[_queue("queued", reason="operator_resume_from_checkpoint", at=T3)],
            candidate_rows=[_candidate("failed", reason="candidate_scoring_runtime_error", at=T1)],
        )
        assert outcome.outcome_label == "queued"

    def test_needs_rescore_does_not_mask_active_derived_candidate(self):
        outcome = _derive(
            queue_rows=[_queue("completed")],
            candidate_rows=[
                _candidate("rejected", reason="stale_parent_needs_rescore", at=T1),
                _candidate("evaluating", at=T2),
            ],
        )
        assert outcome.outcome_label == "scoring"

    def test_needs_rescore_does_not_mask_scored_derived_candidate(self):
        outcome = _derive(
            queue_rows=[_queue("completed")],
            candidate_rows=[
                _candidate("rejected", reason="stale_parent_needs_rescore", at=T1),
                _candidate("scored", at=T2, artifact_hash="hash-1"),
            ],
            score_bundle_rows=[_bundle(2.0)],
        )
        assert outcome.outcome_label == "scored_promising"

    def test_needs_rescore_still_surfaces_when_nothing_else_progresses(self):
        outcome = _derive(
            queue_rows=[_queue("completed")],
            candidate_rows=[_candidate("rejected", reason="stale_parent_needs_rescore")],
        )
        assert outcome.outcome_label == "needs_rescore"

    def test_awaiting_payment_emits_canonical_fields(self):
        outcome = _derive(ticket=_ticket("opened"))
        assert outcome.event_type == "awaiting_payment"
        assert outcome.outcome_label == "awaiting_payment"
        assert outcome.event_doc["public_status"] == "awaiting_payment"
        assert outcome.event_doc["payment_state"] == "no_payment"

    def test_completed_no_candidate_preserved(self):
        outcome = _derive(queue_rows=[_queue("completed")])
        assert outcome.outcome_label == "completed_no_candidate"

    def test_failed_zero_candidate_with_loop_evidence_is_no_buildable_candidate(self):
        outcome = _derive(
            queue_rows=[_queue("failed", "no_valid_image_build_finalists")],
            auto_loop_event_rows=[
                {
                    "event_type": "no_viable_patch",
                    "seq": 1,
                    "event_doc": {"reason": "planner returned no_new_safe_path"},
                }
            ],
        )
        assert outcome.event_type == "no_buildable_candidate"
        assert outcome.outcome_label == "no_buildable_candidate"
        assert outcome.outcome_band == "failed"
        assert outcome.event_doc["public_status_label"] == "No buildable candidate"
        assert outcome.event_doc["candidate_generation_failure"]["primary_reason"] == "no_viable_patch"

    def test_active_candidate_still_shows_scoring(self):
        outcome = _derive(
            queue_rows=[_queue("completed")],
            candidate_rows=[_candidate("evaluating")],
        )
        assert outcome.outcome_label == "scoring"


class TestPublicLoopLookupAndTicketCap:
    def test_short_ticket_prefix_matches_full_ticket_id(self):
        assert public_loop_ticket_id_matches_lookup(
            "49a0d110-1234-4567-89ab-123456789abc",
            "49a0d110",
        )

    def test_invalid_short_ticket_lookup_does_not_match(self):
        assert not public_loop_ticket_id_matches_lookup(
            "49a0d110-1234-4567-89ab-123456789abc",
            "not-a-uuid-prefix",
        )

    @pytest.mark.parametrize(
        "row",
        [
            {"current_outcome_label": "completed_no_candidate", "current_outcome_band": "pending"},
            {"current_outcome_label": "failed", "current_outcome_band": "failed"},
            {"current_outcome_label": "no_buildable_candidate", "current_outcome_band": "failed"},
            {"current_outcome_label": "scored_no_gain", "current_outcome_band": "no_gain"},
            {"current_outcome_label": "scored_promising", "current_outcome_band": "small_gain"},
            {"current_outcome_label": "scored_promising", "current_outcome_band": "passed_threshold"},
            {"current_outcome_label": "promotion_passed", "current_outcome_band": "passed_threshold"},
            {"current_outcome_label": "promoted", "current_outcome_band": "promoted"},
        ],
    )
    def test_terminal_public_outcomes_close_ticket_cap(self, row):
        assert public_loop_outcome_closes_ticket(row)

    @pytest.mark.parametrize(
        "row",
        [
            {"current_outcome_label": "awaiting_payment", "current_outcome_band": "pending"},
            {"current_outcome_label": "queued", "current_outcome_band": "pending"},
            {"current_outcome_label": "running", "current_outcome_band": "pending"},
            {"current_outcome_label": "scoring", "current_outcome_band": "pending"},
            {"current_outcome_label": "waiting_for_baseline", "current_outcome_band": "pending"},
            {"current_outcome_label": "candidate_generation_complete", "current_outcome_band": "pending"},
            {"current_outcome_label": "blocked_for_credit", "current_outcome_band": "blocked"},
            {"current_outcome_label": "needs_rescore", "current_outcome_band": "blocked"},
        ],
    )
    def test_live_or_blocked_public_outcomes_do_not_close_ticket_cap(self, row):
        assert not public_loop_outcome_closes_ticket(row)


# --------------------------------------------------------------------------- #
# Bug 34b — bounded projection retry
# --------------------------------------------------------------------------- #
class TestProjectionRetry:
    @pytest.fixture(autouse=True)
    def _no_backoff(self, monkeypatch):
        monkeypatch.setattr(public_activity, "PROJECTION_RETRY_BACKOFF_SECONDS", 0)

    async def test_retry_recovers_from_transient_failure(self, monkeypatch):
        calls: list[str] = []

        async def fake_project(ticket_id, *, source_ref, reason, config):
            calls.append(ticket_id)
            if len(calls) == 1:
                raise RuntimeError("transient postgrest blip")
            return {"card": {}, "event": {}}

        monkeypatch.setattr(public_activity, "project_public_loop_activity", fake_project)
        result = await public_activity.safe_project_public_loop_activity(
            "ticket-1", source_ref="src", reason="test", config=_sweep_config()
        )
        assert result is not None
        assert len(calls) == 2

    async def test_second_failure_escalates_and_returns_none(self, monkeypatch, caplog):
        calls: list[str] = []

        async def fake_project(ticket_id, *, source_ref, reason, config):
            calls.append(ticket_id)
            raise RuntimeError("persistent failure")

        monkeypatch.setattr(public_activity, "project_public_loop_activity", fake_project)
        with caplog.at_level("ERROR", logger="gateway.research_lab.public_activity"):
            result = await public_activity.safe_project_public_loop_activity(
                "ticket-1", source_ref="src", reason="test", config=_sweep_config()
            )
        assert result is None
        assert len(calls) == 2
        assert any(
            "research_lab_public_activity_projection_failed_after_retry" in record.message
            for record in caplog.records
        )


# --------------------------------------------------------------------------- #
# Bug 34c — reprojection sweep
# --------------------------------------------------------------------------- #
class TestReprojectionSweep:
    @pytest.fixture(autouse=True)
    def _sweep_env(self, monkeypatch):
        monkeypatch.delenv(public_activity.PUBLIC_REPROJECTION_SWEEP_ENABLED_ENV, raising=False)
        monkeypatch.delenv(public_activity.PUBLIC_REPROJECTION_LOG_ONLY_ENV, raising=False)

    def _patch_sweep(self, monkeypatch, *, cards, inputs_by_ticket):
        async def fake_select_many(table, **kwargs):
            assert table == "research_lab_public_loop_card_current"
            return cards

        async def fake_fetch_inputs(ticket_id):
            value = inputs_by_ticket.get(ticket_id)
            if isinstance(value, Exception):
                raise value
            return value

        projected: list[str] = []

        async def fake_project(ticket_id, *, source_ref, reason, config):
            projected.append(ticket_id)
            return {"card": {}, "event": {}}

        monkeypatch.setattr(public_activity, "select_many", fake_select_many)
        monkeypatch.setattr(public_activity, "_fetch_projection_inputs", fake_fetch_inputs)
        monkeypatch.setattr(public_activity, "project_public_loop_activity", fake_project)
        return projected

    async def test_selects_stale_cards_and_reprojects(self, monkeypatch):
        fresh_doc = {"public_status": "awaiting_payment", "payment_state": "no_payment"}
        cards = [
            _card("t-stale", stored_label=""),  # no event ever landed (bug 7)
            _card("t-fresh", stored_label="awaiting_payment", event_doc=fresh_doc),
        ]
        projected = self._patch_sweep(
            monkeypatch,
            cards=cards,
            inputs_by_ticket={
                "t-stale": _opened_ticket_inputs(),
                "t-fresh": _opened_ticket_inputs(),
            },
        )
        result = await public_activity.reproject_stale_public_cards(config=_sweep_config())
        assert result["enabled"] is True
        assert result["cards_checked"] == 2
        assert result["stale_found"] == 1
        assert result["reprojected"] == 1
        assert projected == ["t-stale"]

    async def test_canonical_fields_missing_marks_card_stale(self, monkeypatch):
        # Stored label matches the derived one, but the canonical lifecycle fields
        # never landed (the historical CHECK-constraint failure).
        cards = [_card("t-1", stored_label="awaiting_payment", event_doc={})]
        projected = self._patch_sweep(
            monkeypatch, cards=cards, inputs_by_ticket={"t-1": _opened_ticket_inputs()}
        )
        result = await public_activity.reproject_stale_public_cards(config=_sweep_config())
        assert result["stale_found"] == 1
        assert result["stale"][0]["canonical_fields_missing"] is True
        assert projected == ["t-1"]

    async def test_honors_batch_cap(self, monkeypatch):
        cards = [_card(f"t-{i}", stored_label="") for i in range(3)]
        projected = self._patch_sweep(
            monkeypatch,
            cards=cards,
            inputs_by_ticket={f"t-{i}": _opened_ticket_inputs() for i in range(3)},
        )
        result = await public_activity.reproject_stale_public_cards(
            config=_sweep_config(), batch_size=2
        )
        assert result["stale_found"] == 3
        assert result["reprojected"] == 2
        assert result["deferred_to_next_sweep"] == 1
        assert len(projected) == 2

    async def test_log_only_mode_reports_without_writing(self, monkeypatch):
        monkeypatch.setenv(public_activity.PUBLIC_REPROJECTION_LOG_ONLY_ENV, "true")
        cards = [_card("t-1", stored_label="")]
        projected = self._patch_sweep(
            monkeypatch, cards=cards, inputs_by_ticket={"t-1": _opened_ticket_inputs()}
        )
        result = await public_activity.reproject_stale_public_cards(config=_sweep_config())
        assert result["log_only"] is True
        assert result["stale_found"] == 1
        assert result["reprojected"] == 0
        assert projected == []

    async def test_disabled_by_env_flag(self, monkeypatch):
        monkeypatch.setenv(public_activity.PUBLIC_REPROJECTION_SWEEP_ENABLED_ENV, "false")

        async def exploding_select_many(table, **kwargs):
            raise AssertionError("sweep must not query when disabled")

        monkeypatch.setattr(public_activity, "select_many", exploding_select_many)
        result = await public_activity.reproject_stale_public_cards(config=_sweep_config())
        assert result["enabled"] is False
        assert result["reprojected"] == 0

    async def test_per_card_failures_are_contained(self, monkeypatch):
        cards = [_card("t-broken", stored_label=""), _card("t-ok", stored_label="")]
        projected = self._patch_sweep(
            monkeypatch,
            cards=cards,
            inputs_by_ticket={
                "t-broken": RuntimeError("row fetch exploded"),
                "t-ok": _opened_ticket_inputs(),
            },
        )
        result = await public_activity.reproject_stale_public_cards(config=_sweep_config())
        assert len(result["failed"]) == 1
        assert result["failed"][0]["ticket_id"] == "t-broken"
        assert projected == ["t-ok"]


# --------------------------------------------------------------------------- #
# Bug 34a — recovery paths project the public card
# --------------------------------------------------------------------------- #
class _ProjectionRecorder:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def __call__(self, ticket_id, *, source_ref, reason, config=None, force=False):
        self.calls.append({"ticket_id": ticket_id, "source_ref": source_ref, "reason": reason})
        return {"card": {}, "event": {}}


class TestRecoveryProjections:
    async def test_resume_failed_runs_projects_after_requeue(self, monkeypatch):
        recorder = _ProjectionRecorder()
        queue_row = {
            "run_id": "run-1",
            "ticket_id": "ticket-1",
            "current_queue_status": "failed",
            "queue_priority": 0,
            "current_event_hash": "sha256:" + "0" * 64,
            "current_status_at": T1,
        }

        async def fake_select_one(table, **kwargs):
            assert table == "research_loop_run_queue_current"
            return queue_row

        async def fake_select_many(table, **kwargs):
            return []  # _latest_queue_event_error

        async def fake_checkpoint(run_id):
            return {"checkpoint_hash": "sha256:" + "1" * 64}

        async def fake_create_queue_event(**kwargs):
            return {"event_id": "event-1", "seq": 1, "anchored_hash": "sha256:" + "2" * 64}

        monkeypatch.setattr(recovery, "select_one", fake_select_one)
        monkeypatch.setattr(recovery, "select_many", fake_select_many)
        monkeypatch.setattr(recovery, "latest_auto_research_checkpoint", fake_checkpoint)
        monkeypatch.setattr(recovery, "create_queue_event", fake_create_queue_event)
        monkeypatch.setattr(recovery, "safe_project_public_loop_activity", recorder)

        result = await recovery.resume_failed_runs_from_checkpoint(
            run_ids=["run-1"], dry_run=False
        )
        assert result["resumed"] == 1
        assert [call["ticket_id"] for call in recorder.calls] == ["ticket-1"]
        assert recorder.calls[0]["reason"] == "resume_failed_runs_from_checkpoint"

    async def test_requeue_baseline_not_ready_projects_after_requeue(self, monkeypatch):
        recorder = _ProjectionRecorder()
        candidate_row = {
            "candidate_id": "cand-1",
            "run_id": "run-1",
            "ticket_id": "ticket-1",
            "current_candidate_status": "queued",
            "current_reason": "baseline_not_ready",
            "private_model_manifest_doc": {"schema_version": "1.0"},
        }

        class FakeWorker:
            def __init__(self, config, worker_ref=None):
                pass

            async def _daily_candidate_scoring_window_and_gate(self, *, artifact):
                return SimpleNamespace(window_hash="sha256:" + "3" * 64), {}

        async def fake_select_one(table, **kwargs):
            assert table == "research_lab_candidate_evaluation_current"
            return candidate_row

        async def fake_create_candidate_event(**kwargs):
            return {"event_id": "event-1", "seq": 1}

        monkeypatch.setattr(recovery, "select_one", fake_select_one)
        monkeypatch.setattr(recovery, "ResearchLabGatewayScoringWorker", FakeWorker)
        monkeypatch.setattr(
            recovery,
            "PrivateModelArtifactManifest",
            SimpleNamespace(from_mapping=lambda doc: "artifact"),
        )
        monkeypatch.setattr(recovery, "create_candidate_evaluation_event", fake_create_candidate_event)
        monkeypatch.setattr(recovery, "safe_project_public_loop_activity", recorder)

        result = await recovery.requeue_baseline_not_ready_candidates(
            candidate_ids=["cand-1"], dry_run=False
        )
        assert result["requeued"] == 1
        assert [call["ticket_id"] for call in recorder.calls] == ["ticket-1"]

    async def test_recover_rebase_failed_projects_after_terminal_mark(self, monkeypatch):
        recorder = _ProjectionRecorder()
        candidate_row = {
            "candidate_id": "cand-1",
            "run_id": "run-1",
            "ticket_id": "ticket-1",
            "current_candidate_status": "rejected",
            "current_reason": "stale_parent_rebase_failed",
            "candidate_kind": "code_edit",
        }

        async def fake_select_one(table, **kwargs):
            assert table == "research_lab_candidate_evaluation_current"
            return candidate_row

        async def fake_select_many(table, **kwargs):
            return []  # _existing_regeneration_run

        async def fake_create_candidate_event(**kwargs):
            return {"event_id": "event-1", "seq": 1}

        monkeypatch.setattr(recovery, "select_one", fake_select_one)
        monkeypatch.setattr(recovery, "select_many", fake_select_many)
        monkeypatch.setattr(recovery, "create_candidate_evaluation_event", fake_create_candidate_event)
        monkeypatch.setattr(recovery, "safe_project_public_loop_activity", recorder)

        result = await recovery.recover_rebase_failed_candidates(
            candidate_ids=["cand-1"], dry_run=False, regenerate=False
        )
        assert result["recovered"] == 1
        assert [call["ticket_id"] for call in recorder.calls] == ["ticket-1"]

    async def test_projection_failure_never_fails_recovery(self, monkeypatch):
        async def exploding_projection(*args, **kwargs):
            raise RuntimeError("projection exploded")

        monkeypatch.setattr(recovery, "safe_project_public_loop_activity", exploding_projection)
        # Direct wrapper check: the guard must swallow anything the projector raises.
        await recovery._project_after_recovery(
            "ticket-1",
            source_ref="src",
            reason="test",
            config=_sweep_config(),
        )
