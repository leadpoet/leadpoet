"""Authenticated source reconstruction for measured Research Lab rewards."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Callable, Dict, Mapping

from gateway.tee.coordinator_chain_source_v2 import CoordinatorChainSourceV2
from gateway.tee.execution_job_manager_v2 import ExecutionContextV2
from gateway.tee.source_add_runtime_v2 import (
    build_source_add_runtime_catalog_v2,
)
from gateway.tee.supabase_source_v2 import SupabaseSourceReaderV2
from leadpoet_canonical.attested_v2 import sha256_json
from leadpoet_canonical.chain_source_v2 import CHAIN_FINALIZATION_EPOCH_BLOCKS


class CoordinatorRewardSourceV2Error(RuntimeError):
    """A reward proposal differs from authenticated database or chain state."""


class CoordinatorRewardSourceV2:
    """Replace host-selected SOURCE_ADD inputs with measured source values."""

    def __init__(
        self,
        *,
        reader: SupabaseSourceReaderV2,
        chain_source: CoordinatorChainSourceV2,
        config_supplier: Callable[[], Any],
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self._reader = reader
        self._chain_source = chain_source
        self._config_supplier = config_supplier
        self._clock = clock

    def resolve(
        self,
        *,
        payload: Mapping[str, Any],
        context: ExecutionContextV2,
    ) -> Dict[str, Any]:
        if not isinstance(payload, Mapping) or set(payload) != {
            "decision_kind",
            "decision_payload",
        }:
            raise CoordinatorRewardSourceV2Error(
                "reward authority payload fields are invalid"
            )
        kind = str(payload.get("decision_kind") or "")
        if kind == "reimbursement":
            return self._resolve_reimbursement(payload=payload, context=context)
        if kind not in {"source_add_leg1", "source_add_leg2"}:
            raise CoordinatorRewardSourceV2Error(
                "reward source kind is unsupported"
            )
        proposed = payload.get("decision_payload")
        if not isinstance(proposed, Mapping):
            raise CoordinatorRewardSourceV2Error("reward decision input is invalid")
        decision = dict(proposed)
        adapter_id = str(decision.get("adapter_id") or "")
        if not adapter_id:
            raise CoordinatorRewardSourceV2Error("SOURCE_ADD adapter is missing")

        config = self._config_supplier()
        chain_state = self._chain_source.read_finalized_metagraph(
            netuid=int(getattr(config, "netuid", 71) or 71),
            context=context,
        )
        chain_epoch = (
            int(chain_state["header"]["block"])
            // CHAIN_FINALIZATION_EPOCH_BLOCKS
        )
        if chain_epoch != int(context.epoch_id):
            raise CoordinatorRewardSourceV2Error(
                "reward execution epoch differs from finalized chain state"
            )
        expected_start_epoch = chain_epoch + 1
        if int(decision.get("start_epoch") or -1) != expected_start_epoch:
            raise CoordinatorRewardSourceV2Error(
                "SOURCE_ADD start epoch differs from finalized chain state"
            )

        expected_alpha = float(
            getattr(
                config,
                "source_add_leg1_alpha_percent"
                if kind == "source_add_leg1"
                else "source_add_leg2_alpha_percent",
                1.0 if kind == "source_add_leg1" else 5.0,
            )
            or (1.0 if kind == "source_add_leg1" else 5.0)
        )
        expected_epochs = int(getattr(config, "lab_reward_epochs", 20) or 20)
        if (
            float(decision.get("alpha_percent") or 0.0) != expected_alpha
            or int(decision.get("reward_epochs") or 0) != expected_epochs
        ):
            raise CoordinatorRewardSourceV2Error(
                "SOURCE_ADD reward policy differs from measured configuration"
            )

        decision["existing_rewards"] = self._read(
            "source_add_rewards_by_adapter",
            {"adapter_id": adapter_id},
            context,
        )
        if kind == "source_add_leg1":
            provenance = decision.get("provenance_result")
            if (
                not isinstance(provenance, Mapping)
                or provenance.get("precheck_status")
                != "provenance_precheck_passed"
            ):
                raise CoordinatorRewardSourceV2Error(
                    "SOURCE_ADD Leg 1 provenance is invalid"
                )
            submission_id = str(provenance.get("submission_id") or "")
            submission_rows = self._read(
                "source_add_submission_by_id",
                {"submission_id": submission_id},
                context,
            )
            if len(submission_rows) != 1:
                raise CoordinatorRewardSourceV2Error(
                    "SOURCE_ADD submission owner is missing or ambiguous"
                )
            submission = submission_rows[0]
            if (
                str(submission.get("adapter_id") or "") != adapter_id
                or str(submission.get("miner_hotkey") or "")
                != str(decision.get("miner_ref") or "")
                or str(submission.get("precheck_status") or "")
                != "provenance_precheck_passed"
            ):
                raise CoordinatorRewardSourceV2Error(
                    "SOURCE_ADD Leg 1 owner or status differs from measured submission"
                )
            now = self._clock()
            if now.tzinfo is None:
                now = now.replace(tzinfo=timezone.utc)
            day_start = now.astimezone(timezone.utc).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            rows = self._read(
                "source_add_leg1_events_since",
                {"day_start": day_start.isoformat()},
                context,
            )
            daily_cap = max(
                1,
                int(getattr(config, "source_add_leg1_max_per_utc_day", 10) or 10),
            )
            if len(rows) >= daily_cap:
                raise CoordinatorRewardSourceV2Error(
                    "SOURCE_ADD Leg 1 daily cap is reached"
                )
        else:
            judge_result = decision.get("judge_result")
            verdict = (
                judge_result.get("verdict")
                if isinstance(judge_result, Mapping)
                else None
            )
            if (
                not isinstance(verdict, Mapping)
                or verdict.get("verdict") != "helped"
                or verdict.get("source_used") is not True
            ):
                raise CoordinatorRewardSourceV2Error(
                    "SOURCE_ADD Leg 2 signed judge did not approve the reward"
                )
            rows = self._read(
                "source_add_provisioning_by_adapter",
                {"adapter_id": adapter_id},
                context,
            )
            if len(rows) != 1:
                raise CoordinatorRewardSourceV2Error(
                    "SOURCE_ADD provisioned owner is missing or ambiguous"
                )
            owner = str(rows[0].get("miner_hotkey") or "")
            if not owner or owner != str(decision.get("miner_ref") or ""):
                raise CoordinatorRewardSourceV2Error(
                    "SOURCE_ADD reward owner differs from measured provisioning"
                )
            matched_adapter = str(verdict.get("adapter_id") or "") == adapter_id
            matched_registry = str(verdict.get("registry_provider_id") or "") == str(
                rows[0].get("registry_provider_id") or ""
            )
            if not matched_adapter and not matched_registry:
                raise CoordinatorRewardSourceV2Error(
                    "SOURCE_ADD Leg 2 judge differs from measured provisioning"
                )
            trigger = decision.get("trigger_evidence")
            if not isinstance(trigger, Mapping):
                raise CoordinatorRewardSourceV2Error(
                    "SOURCE_ADD Leg 2 trigger evidence is invalid"
                )
            expected_trigger = {
                "llm_judge_passed": True,
                "llm_verdict": "helped",
                "llm_confidence": float(verdict.get("confidence") or 0.0),
                "source_used": True,
                "adapter_id": str(verdict.get("adapter_id") or ""),
                "registry_provider_id": str(
                    verdict.get("registry_provider_id") or ""
                ),
                "evidence_summary": str(verdict.get("evidence_summary") or "")[:1000],
                "reason_codes": [
                    str(item) for item in (verdict.get("reason_codes") or ())
                ][:20],
                "judge_model": str(verdict.get("model_id") or ""),
                "judge_doc_hash": str(verdict.get("judge_doc_hash") or ""),
                "provider_usage": dict(verdict.get("provider_usage") or {}),
            }
            for field, expected in expected_trigger.items():
                if trigger.get(field) != expected:
                    raise CoordinatorRewardSourceV2Error(
                        "SOURCE_ADD Leg 2 trigger differs from signed judge"
                    )
        return {
            "decision_kind": kind,
            "decision_payload": decision,
        }

    def _resolve_reimbursement(
        self,
        *,
        payload: Mapping[str, Any],
        context: ExecutionContextV2,
    ) -> Dict[str, Any]:
        proposed = payload.get("decision_payload")
        if not isinstance(proposed, Mapping) or set(proposed) != {
            "source_request",
            "autoresearch_result",
        }:
            raise CoordinatorRewardSourceV2Error(
                "reimbursement source request fields are invalid"
            )
        source_request = proposed.get("source_request")
        autoresearch_result = proposed.get("autoresearch_result")
        if not isinstance(source_request, Mapping) or set(source_request) != {
            "run_id",
            "ticket_id",
            "receipt_id",
        }:
            raise CoordinatorRewardSourceV2Error(
                "reimbursement source references are invalid"
            )
        if not isinstance(autoresearch_result, Mapping):
            raise CoordinatorRewardSourceV2Error(
                "reimbursement autoresearch result is missing"
            )
        run_id = str(source_request["run_id"])
        ticket_id = str(source_request["ticket_id"])
        receipt_id = str(source_request["receipt_id"])
        ticket = self._one(
            "reimbursement_ticket_by_id", {"ticket_id": ticket_id}, context
        )
        receipt = self._one(
            "reimbursement_receipt_by_id", {"receipt_id": receipt_id}, context
        )
        if (
            str(receipt.get("run_id") or "") != run_id
            or str(receipt.get("ticket_id") or "") != ticket_id
            or str(ticket.get("ticket_id") or "") != ticket_id
        ):
            raise CoordinatorRewardSourceV2Error(
                "reimbursement ticket, receipt, and run differ"
            )

        payment = None
        payment_id = str(receipt.get("loop_start_payment_id") or "")
        if payment_id:
            payment = self._one(
                "reimbursement_payment_by_id",
                {"payment_id": payment_id},
                context,
            )
            if (
                str(payment.get("ticket_id") or "") != ticket_id
                or str(payment.get("payment_status") or "") != "verified"
            ):
                raise CoordinatorRewardSourceV2Error(
                    "reimbursement payment is not the verified ticket payment"
                )
        queue_events = self._read(
            "reimbursement_queue_events_by_run", {"run_id": run_id}, context
        )
        queue_doc = {}
        for row in queue_events:
            if isinstance(row.get("event_doc"), Mapping):
                queue_doc = dict(row["event_doc"])
                break
        ticket_doc = (
            dict(ticket["ticket_doc"])
            if isinstance(ticket.get("ticket_doc"), Mapping)
            else {}
        )
        payment_doc = (
            dict(payment["verification_doc"])
            if isinstance(payment, Mapping)
            and isinstance(payment.get("verification_doc"), Mapping)
            else {}
        )
        config = self._config_supplier()
        requested_budget = (
            queue_doc.get("requested_compute_budget_usd")
            or payment_doc.get("compute_budget_usd")
            or payment_doc.get("requested_compute_budget_usd")
            or ticket_doc.get("requested_compute_budget_usd")
            or config.default_compute_budget_usd
        )
        funded_budget = float(config.clamp_compute_budget_usd(requested_budget))

        issued_at = self._parent_issued_at(context)
        run_day = issued_at.date().isoformat()
        island = str(ticket.get("island") or config.reimbursement_default_island)
        snapshot = self._participation_snapshot(
            island=island,
            lookback_end=issued_at,
            context=context,
        )
        cap_usage = self._cap_usage(
            run_day=run_day,
            miner_hotkey=str(ticket.get("miner_hotkey") or ""),
            island=island,
            context=context,
        )
        try:
            actual_microusd = max(
                0, int(autoresearch_result["actual_openrouter_cost_microusd"])
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise CoordinatorRewardSourceV2Error(
                "autoresearch cost commitment is invalid"
            ) from exc
        policy = dict(
            config.reimbursement_policy_doc(
                enabled=config.reimbursements_enabled
                or config.shadow_reimbursements_enabled
            )
        )
        chain_state = self._chain_source.read_finalized_metagraph(
            netuid=int(getattr(config, "netuid", 71) or 71),
            context=context,
        )
        chain_epoch = (
            int(chain_state["header"]["block"])
            // CHAIN_FINALIZATION_EPOCH_BLOCKS
        )
        if chain_epoch != int(context.epoch_id):
            raise CoordinatorRewardSourceV2Error(
                "reimbursement epoch differs from finalized chain state"
            )
        start_epoch = chain_epoch + 1
        run_cost = {
            "run_id": run_id,
            "miner_hotkey": str(ticket.get("miner_hotkey") or ""),
            "island": island,
            "run_day": run_day,
            "funded_compute_budget_usd": funded_budget,
            "actual_openrouter_cost_usd": round(actual_microusd / 1_000_000, 6),
            "loop_start_tao_fee_usd": float(config.loop_start_fee_usd),
            "paid_research_loop": True,
            "valid_receipt": str(receipt.get("current_receipt_status") or "")
            in {"queued", "completed", "failed"},
            "verified_loop_start_payment": payment is not None,
            "preserved_loop_start_credit": bool(
                str(receipt.get("loop_start_credit_id") or "")
            ),
            "miner_openrouter_key_present": bool(
                str(ticket.get("miner_openrouter_key_ref") or "").strip()
            ),
            "trusted_cost_ledger": True,
            "passed_abuse_checks": True,
            "refunded": False,
            "voided": False,
            "duplicate": False,
            "novelty_rejected": False,
            "self_cancelled_before_minimum_work": False,
            "banned_hotkey": False,
        }
        source_state = {
            "schema_version": "leadpoet.reimbursement_source_state.v2",
            "run_cost": run_cost,
            "participation_snapshot": snapshot,
            "policy": policy,
            "cap_usage": cap_usage,
            "start_epoch": start_epoch,
            "source_refs": {
                "run_id": run_id,
                "ticket_id": ticket_id,
                "receipt_id": receipt_id,
            },
        }
        return {
            "decision_kind": "reimbursement",
            "decision_payload": {
                "run_cost": run_cost,
                "participation_snapshot": snapshot,
                "policy": policy,
                "cap_usage": cap_usage,
                "start_epoch": start_epoch,
                "autoresearch_result": dict(autoresearch_result),
                "source_state": source_state,
            },
        }

    def _participation_snapshot(
        self,
        *,
        island: str,
        lookback_end: datetime,
        context: ExecutionContextV2,
    ) -> Dict[str, Any]:
        lookback_start = lookback_end - timedelta(days=7)
        tickets = self._read(
            "reimbursement_participation_tickets", {"island": island}, context
        )
        tickets = [
            row
            for row in tickets
            if self._row_dt(row.get("created_at") or row.get("current_status_at"))
            >= lookback_start
        ]
        ticket_ids = sorted(
            {str(row.get("ticket_id") or "") for row in tickets if row.get("ticket_id")}
        )
        queue_rows = []
        for ticket_id in ticket_ids:
            queue_rows.extend(
                self._read(
                    "reimbursement_queue_by_ticket",
                    {"ticket_id": ticket_id},
                    context,
                )
            )
        paid = [
            row
            for row in queue_rows
            if str(row.get("ticket_id") or "") in set(ticket_ids)
            and str(row.get("current_queue_status") or "")
            in {"queued", "started", "paused", "completed"}
            and self._row_dt(row.get("current_status_at")) >= lookback_start
        ]
        hotkeys = {
            str(row.get("miner_hotkey") or "")
            for row in tickets
            if row.get("miner_hotkey")
        }
        briefs = {
            str(row.get("brief_sanitized_ref") or "")
            for row in tickets
            if row.get("brief_sanitized_ref")
        }
        return {
            "snapshot_id": "participation:%s:%s"
            % (island, lookback_end.date().isoformat()),
            "island": island,
            "lookback_start": lookback_start.isoformat(),
            "lookback_end": lookback_end.isoformat(),
            "distinct_funded_hotkeys": len(hotkeys),
            "paid_loop_count": len(paid),
            "unique_brief_count": len(briefs),
        }

    def _cap_usage(
        self,
        *,
        run_day: str,
        miner_hotkey: str,
        island: str,
        context: ExecutionContextV2,
    ) -> Dict[str, float]:
        rows = self._read(
            "reimbursement_cap_awards_by_day", {"run_day": run_day}, context
        )
        eligible = [
            row
            for row in rows
            if str(row.get("run_day") or "") == run_day
            and str(
                row.get("current_award_status") or row.get("award_status") or ""
            )
            == "awarded"
        ]
        return {
            "hotkey_day_awarded_usd": self._sum_award_usd(
                row
                for row in eligible
                if str(row.get("miner_hotkey") or "") == miner_hotkey
            ),
            "island_day_awarded_usd": self._sum_award_usd(
                row
                for row in eligible
                if str(row.get("island") or "") == island
            ),
            "global_awarded_usd": self._sum_award_usd(eligible),
        }

    @staticmethod
    def _parent_issued_at(context: ExecutionContextV2) -> datetime:
        roots = []
        for graph in context.external_receipt_graphs:
            root_hash = str(graph.get("root_receipt_hash") or "")
            for receipt in graph.get("receipts") or ():
                if (
                    isinstance(receipt, Mapping)
                    and receipt.get("receipt_hash") == root_hash
                    and receipt.get("purpose") == "research_lab.candidate_decision.v2"
                ):
                    roots.append(receipt)
        if len(roots) != 1:
            raise CoordinatorRewardSourceV2Error(
                "reimbursement autoresearch parent is missing or ambiguous"
            )
        try:
            return datetime.strptime(
                str(roots[0]["issued_at"]), "%Y-%m-%dT%H:%M:%SZ"
            ).replace(tzinfo=timezone.utc)
        except (KeyError, ValueError) as exc:
            raise CoordinatorRewardSourceV2Error(
                "reimbursement parent timestamp is invalid"
            ) from exc

    @staticmethod
    def _row_dt(value: Any) -> datetime:
        if not value:
            return datetime.fromtimestamp(0, timezone.utc)
        text = str(value)
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return datetime.fromtimestamp(0, timezone.utc)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    @staticmethod
    def _sum_award_usd(rows: Any) -> float:
        total = Decimal("0")
        for row in rows:
            try:
                total += Decimal(
                    str(row.get("target_reimbursement_microusd", 0))
                ) / Decimal("1000000")
            except Exception:
                continue
        return float(
            total.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)
        )

    def _one(
        self,
        policy_id: str,
        parameters: Mapping[str, Any],
        context: ExecutionContextV2,
    ) -> Dict[str, Any]:
        rows = self._read(policy_id, parameters, context)
        if len(rows) != 1:
            raise CoordinatorRewardSourceV2Error(
                "%s source row is missing or ambiguous" % policy_id
            )
        return rows[0]

    def catalog_snapshot(
        self,
        *,
        payload: Mapping[str, Any],
        context: ExecutionContextV2,
    ) -> Dict[str, Any]:
        if not isinstance(payload, Mapping) or set(payload) != {"limit"}:
            raise CoordinatorRewardSourceV2Error(
                "SOURCE_ADD catalog snapshot payload is invalid"
            )
        if int(payload.get("limit") or 0) != 200:
            raise CoordinatorRewardSourceV2Error(
                "SOURCE_ADD catalog snapshot limit is invalid"
            )
        rows = self._read("source_add_provisioning_eligible", {}, context)
        private_registry_rows = self._read(
            "provider_registry_recent",
            {},
            context,
        )
        runtime_catalog = build_source_add_runtime_catalog_v2(rows)
        result = {
            "schema_version": "leadpoet.source_add_catalog_snapshot.v2",
            "provisioned_sources": rows,
            "provisioned_sources_hash": sha256_json(rows),
            "private_registry_rows": private_registry_rows,
            "private_registry_rows_hash": sha256_json(private_registry_rows),
            "runtime_catalog": runtime_catalog,
            "runtime_catalog_hash": str(runtime_catalog["catalog_hash"]),
        }
        return result

    def _read(
        self,
        policy_id: str,
        parameters: Mapping[str, Any],
        context: ExecutionContextV2,
    ) -> list[Dict[str, Any]]:
        return self._reader.read(
            policy_id=policy_id,
            parameters=parameters,
            job_id=context.job_id,
            purpose=context.purpose,
            record_transport=context.record_transport,
            record_artifact=context.record_artifact,
        )
