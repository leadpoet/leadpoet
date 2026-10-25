"""Read one persisted Arena execution for training evidence without writes."""

from __future__ import annotations

from datetime import datetime, timezone
import time

from lab_arena.training_export import ExportError, _select_judge, assemble_example


class AvailabilityReadView:
    """Bounded read view for training evidence; never changes stored objects."""

    def __init__(self, objects, verified_fallback=None):
        self.objects = objects
        self.verified_fallback = verified_fallback or {}
        self.unavailable = {}
        self.transient_failures: dict[str, str] = {}
        self.artifact_refs: dict[str, str] = {}

    def _fallback(self, ref, limit, reason):
        cached = self.verified_fallback.get(ref)
        if cached is None:
            return None
        if len(cached) > limit:
            return None
        self.unavailable[ref] = reason
        return cached

    @staticmethod
    def _retryable(error):
        from lab_arena.training_capture import retryable_storage_error
        return retryable_storage_error(error)

    def get_bounded(self, ref, limit):
        self.transient_failures.pop(ref, None)
        for attempt in range(2):
            try:
                optional = getattr(self.objects, "get_optional_bounded", None)
                value = optional(ref, limit) if callable(optional) else self.objects.get_bounded(ref, limit)
                if value is None:
                    cached = self._fallback(ref, limit, "missing")
                    if cached is not None:
                        return cached
                    raise FileNotFoundError("training evidence object is absent")
                self.unavailable.pop(ref, None)
                return value
            except FileNotFoundError:
                cached = self._fallback(ref, limit, "missing")
                if cached is not None:
                    return cached
                raise
            except Exception as exc:
                if self._retryable(exc):
                    if attempt == 0:
                        time.sleep(0.05)
                        continue
                    self.transient_failures[ref] = "transient"
                    cached = self._fallback(ref, limit, "transient")
                    if cached is not None:
                        return cached
                raise
        raise AssertionError("unreachable")

    def get_optional_bounded(self, ref, limit):
        try:
            return self.get_bounded(ref, limit)
        except FileNotFoundError:
            return None


def _ledger(store, run_id):
    rows, cursor = [], None
    while True:
        page = store.list_ledger(run_id=run_id, after_entry_id=cursor, limit=1000)
        rows.extend(page)
        if len(page) < 1000:
            return rows
        next_cursor = int(page[-1]["entry_id"])
        if cursor is not None and next_cursor <= cursor:
            raise ValueError("ledger_pagination_failed")
        cursor = next_cursor


def export_run(store, objects, run_id):
    """Read only the requested round and its referenced accepted authorities."""
    execution = store.get_run(run_id)
    if execution is None or execution.get("kind") != "execute":
        raise ExportError("execution_run_missing")
    round_row = store.get_round(execution["round_id"])
    submission = store.get_submission(execution["submission_id"])
    if not round_row or not submission:
        raise ExportError("context_missing")
    generation_floor = round_row.get("scoring_generation_floor", 0)
    policy_history = store.scoring_policy_history(execution["round_id"]) if generation_floor else None
    runs = store.list_runs(execution["round_id"])
    related = [row for row in runs if row.get("assignment_id") == execution["assignment_id"]
               or row.get("kind") == "score" and row.get("scored_run_id") == run_id]
    ledger = [entry for row in related for entry in _ledger(store, row["run_id"])]
    caches, company_rows = {}, {}
    for row in related:
        if row.get("kind") != "score":
            continue
        if row.get("judgment_cache_key"):
            key = row["judgment_cache_key"]
            cached = store.get_judgment_cache(key)
            if cached is not None:
                caches[key] = cached
        lease = (row.get("claim_response") or {}).get("company_judgment_cache") or {}
        for item in list(lease.get("hits") or []) + list(lease.get("misses") or []):
            identity = (item["cache_key"], item["authority_slot"])
            cached = store.get_company_judgment(*identity)
            if cached is not None:
                company_rows[identity] = cached
    judge = _select_judge(execution, runs, scoring_generation_floor=generation_floor)
    if isinstance(objects, AvailabilityReadView):
        from lab_arena.training_input import object_ref

        def input_ref(row):
            try:
                return object_ref(row["run_id"], row.get("lease_generation"))
            except ValueError:
                # Assembly still validates invalid saved input identities.
                return None

        refs = {"source": submission.get("source_ref"),
                "benchmark": round_row.get("benchmark_ref"),
                "input": input_ref(execution)}
        if judge is not None:
            refs.update(judge_output=judge.get("output_ref"), judge_input=input_ref(judge))
        objects.artifact_refs = {kind: ref for kind, ref in refs.items()
                                 if isinstance(ref, str) and ref}
    signal_uses, signal_rows, signal_source_ledger = [], {}, []
    if ((round_row.get("configuration_doc") or {}).get("scorer_policy") or {}).get("signal_judgment_policy") == "signal_judgment_v1":
        if judge is not None:
            origin_ids = {row["source_score_run_id"] for row in company_rows.values()}
            for score_id in sorted({judge["run_id"]} | origin_ids):
                signal_uses.extend(store.get_signal_judgment_uses(score_id))
            for use in signal_uses:
                identity = (use["cache_key"], use["authority_slot"])
                if identity not in signal_rows:
                    signal_rows[identity] = store.get_signal_judgment(*identity)
            source_ids = {row["source_score_run_id"] for row in signal_rows.values() if row is not None}
            related_ids = {row["run_id"] for row in related}
            signal_source_ledger = [entry for source_id in sorted(source_ids - related_ids)
                                    for entry in _ledger(store, source_id)]

    def cost_eligibility_reader(**identity):
        result = store.icp_cost_eligibility(**identity)
        return {"observed_at": datetime.now(timezone.utc).isoformat(),
                "eligibility": result}

    return assemble_example(run_id=run_id, round_row=round_row, submission_row=submission,
                            runs=runs, ledger_rows=ledger, objects=objects,
                            cache_rows=list(caches.values()), company_judgment_rows=list(company_rows.values()),
                            signal_judgment_uses=signal_uses,
                            signal_judgment_rows=[row for row in signal_rows.values() if row is not None],
                            signal_source_ledger_rows=signal_source_ledger,
                            scoring_policy_history=policy_history,
                            cost_eligibility_reader=cost_eligibility_reader)


__all__ = ["export_run"]
