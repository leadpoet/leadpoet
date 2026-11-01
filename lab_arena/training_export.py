"""Read-only, fail-closed assembly of one Arena training example.

Labeled results and a complete research process are separate claims. Diagnostic
trajectory excerpts cannot establish the latter. No provider or chain call is
made by this module, and no service lifecycle is started.
"""

from __future__ import annotations

import base64
import copy
import json
import re
from datetime import datetime
from decimal import Decimal
from typing import Any, Callable, Mapping, Sequence

from lab_arena import company_judgments, contracts, contact_policy, integrity, reviewer_runtime, scoring, signal_judgments, source_bundle, verify
from lab_arena.output import MAX_OUTPUT_BYTES, validate_output_document

SCHEMA_VERSION = "leadpoet.lab_arena.training_example.v1"
MAX_RECEIPT_BYTES = 16 * 1024 * 1024
# The final privacy pass bounds each joined call and input snapshot separately,
# plus one shared envelope budget. Per-request/response limits remain unchanged.
MAX_EXAMPLE_NODES = 2_000_000
_RETAINED_ERROR_STATUSES = {
    "openrouter": (400, 402, 403, 404, 422, 502),
    "deepline": (400, 402, 403, 404, 422),
}


class ExportError(ValueError):
    """The supplied identity or immutable join is ambiguous or invalid."""


def _json_object(objects: Any, ref: str, limit: int) -> dict:
    raw = objects.get_bounded(ref, limit)
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ExportError("object_not_document")
    return value


def without_hidden_reasoning(value: Any) -> tuple[Any, bool]:
    """Copy a JSON value, omitting opaque reasoning while keeping readable text."""
    from lab_arena.training_capture import (approved_history_reasoning_content, empty_value,
                                            opaque_reasoning_field, readable_reasoning_item, reasoning_kind)
    removed = False

    def walk(item, path=()):
        nonlocal removed
        if isinstance(item, dict):
            result = {}
            kind = reasoning_kind(item)
            for key, child in item.items():
                normalized = str(key).lower().replace("-", "_")
                safe_config = normalized == "reasoning" and isinstance(child, dict) and set(child) <= {"effort", "summary", "context", "max_tokens", "exclude", "enabled"}
                if (normalized == "reasoning_content" and isinstance(child, str)
                        and approved_history_reasoning_content(value, path, item)):
                    result[key] = walk(child, (*path, normalized))
                elif opaque_reasoning_field(normalized, child, kind, safe_config, path, value):
                    # Native readable history can include these empty optional
                    # fields. Removing them changes the retained request hash,
                    # although there are no private bytes to exclude.
                    if (item.get("type") == "reasoning" and kind == "readable"
                            and (key == "summary" and isinstance(child, list) and not child
                                 or key == "encrypted_content" and child is None)):
                        result[key] = walk(child, (*path, normalized))
                    else:
                        removed = removed or not empty_value(child)
                else:
                    result[key] = walk(child, (*path, normalized))
            return result
        if isinstance(item, list):
            kept = []
            for index, child in enumerate(item):
                opaque = (not readable_reasoning_item(child) if path and path[-1] == "reasoning_details"
                          else isinstance(child, dict) and reasoning_kind(child) == "opaque")
                if opaque:
                    removed = removed or not empty_value(child)
                else:
                    kept.append(walk(child, (*path, str(index))))
            return kept
        return item

    return walk(value), removed


class _Rows:
    """Minimal read-only store for the existing accepted-evidence validators."""

    def __init__(self, runs, submission, caches, companies, *, round_row,
                 signal_uses, signal_rows):
        self.runs = runs
        self.submission = submission
        self.round_row = round_row
        self.caches = {row["cache_key"]: row for row in caches}
        self.companies = {(row["cache_key"], row["authority_slot"]): row for row in companies}
        self.signal_uses = signal_uses
        self.signals = {(row["cache_key"], row["authority_slot"]): row for row in signal_rows}
        if len(self.caches) != len(caches) or len(self.companies) != len(companies):
            raise ExportError("ambiguous_cache_rows")
        if len(self.signals) != len(signal_rows):
            raise ExportError("signal_authority_ambiguous")

    def get_round(self, round_id):
        return self.round_row if self.round_row.get("round_id") == round_id else None

    def get_run(self, run_id):
        return self.runs.get(run_id)

    def get_submission(self, submission_id):
        return self.submission if self.submission.get("submission_id") == submission_id else None

    def get_judgment_cache(self, key):
        return self.caches.get(key)

    def get_company_judgment(self, key, slot):
        return self.companies.get((key, slot))

    def signal_authority_exclusions(self, run_id):
        """Replay the immutable union in signal_authority_exclusions_v1."""
        source = self.get_run(run_id)
        if source is None or self.get_round(source.get("round_id")) is None:
            raise ExportError("signal_source_missing")
        exclusions = (source.get("claim_response") or {}).get("runner_authority_exclusions")
        policy = (self.round_row.get("configuration_doc") or {}).get("scorer_policy") or {}
        if policy.get("signal_judgment_policy") != signal_judgments.POLICY:
            return exclusions
        groups = [exclusions]
        positions = set()
        for use in self.signal_uses:
            if use.get("run_id") != run_id:
                continue
            position = (use.get("company_index"), use.get("signal_index"))
            if position in positions:
                raise ExportError("signal_use_invalid")
            positions.add(position)
            authority = self.signals.get((use.get("cache_key"), use.get("authority_slot")))
            if authority is None:
                raise ExportError("signal_authority_missing")
            if any(use.get(key) != authority.get(key) for key in (
                    "source_score_run_id", "signal_input_hash", "judgment_hash")):
                raise ExportError("signal_source_mismatch")
            groups.append(authority.get("runner_authority_exclusions"))
        if any(not isinstance(group, list) or any(not isinstance(key, str) for key in group)
               for group in groups):
            raise ExportError("signal_authority_exclusions_invalid")
        members = {key for group in groups for key in group}
        # SQL orders by the database collation. Use the recorded ordering only
        # after proving exact membership from the independent immutable rows.
        for row in self.companies.values():
            evidence = row.get("evidence_doc") or {}
            recorded = evidence.get("runner_authority_exclusions")
            if (evidence.get("source_score_run_id") == run_id
                    and isinstance(recorded, list)
                    and all(isinstance(key, str) for key in recorded)
                    and len(recorded) == len(members) and set(recorded) == members):
                return list(recorded)
        raise ExportError("signal_authority_exclusions_mismatch")


def _evidence_reader(store, objects):
    # These methods only read the injected store/object interfaces. Deliberately
    # do not construct ArenaService, which also owns production lifecycle writes.
    from lab_arena.service import ArenaService

    class EvidenceReader:
        _verified_breakdowns = ArenaService._verified_breakdowns
        _company_quality_breakdowns = ArenaService._company_quality_breakdowns
        _validated_company_judgment_lease = ArenaService._validated_company_judgment_lease
        _company_judgment_evidence = ArenaService._company_judgment_evidence
        _round_from_context = ArenaService._round_from_context

        def _round(self, round_id):
            # Offline assembly has one exact frozen round, with no service
            # ownership, lifecycle, or current-policy lookup.
            row = self._store.get_round(round_id)
            if row is None:
                raise ExportError("source_round_missing")
            return row

    class BoundedObjects:
        def get(self, ref):
            return objects.get_bounded(ref, scoring.MAX_SCORING_OUTPUT_BYTES)

        def get_bounded(self, ref, limit):
            return objects.get_bounded(ref, limit)

    reader = EvidenceReader()
    reader._store, reader._objects = store, BoundedObjects()
    return reader


def _signal_details(breakdown):
    # Terminal company-fit failures have no signal judgment and encode null.
    details = breakdown.get("intent_signals_detail")
    if details is None:
        return []
    if not isinstance(details, list):
        raise ExportError("signal_details_invalid")
    return details


def _terminal_company_fit_nonmatch(breakdown):
    """A final fit failure has no signal verdict to reference."""
    if (breakdown.get("intent_signals_detail") is not None
            or breakdown.get("company_qualified") is not False):
        return False
    fit = [receipt for receipt in breakdown.get("verifier_gate_receipts") or []
           if isinstance(receipt, Mapping) and receipt.get("gate") == "company_fit"]
    return len(fit) == 1 and fit[0].get("decision") in {"mismatch", "unavailable"}


def _signal_authority_evidence(*, judge, breakdowns, scored_company_indexes,
                               scoring_input, uses, authority_rows, company_rows, runs,
                               round_row):
    """Resolve every emitted signal reference against its original frozen row."""
    if (scoring_input.get("scorer_policy") or {}).get("signal_judgment_policy") != signal_judgments.POLICY:
        if uses or any(
            "signal_judgment_reference" in detail
            for breakdown in breakdowns
            for detail in _signal_details(breakdown)
        ):
            raise ExportError("signal_policy_not_frozen")
        return None
    if len(scored_company_indexes) != len(breakdowns):
        raise ExportError("signal_company_index_mismatch")
    # Persisted input snapshots are canonical JSON and therefore sorted by
    # key. The gateway's effective-company projector requires construction
    # order, so restore only its exact frozen fields and drop runtime context.
    fields = ("schema_version", "scored_run_id", "icp", "companies",
              "scorer_policy", "evaluation_date")
    if scoring_input.get("scorer_policy", {}).get("scoring_adapter_version") == "qualification_contacts_v3":
        fields += ("contact_source_evidence",)
    config = round_row["configuration_doc"]
    runtime_fields = {"company_judgment_cache", "prior_terminal_breakdowns"}
    # The runner adds this frozen discovery catalog to both execution and
    # score inputs. It is not an effective input to the gateway's signal judge.
    if config.get("deepline_catalog") is not None:
        if ("deepline_catalog" not in scoring_input
                or contracts.canonical_json(scoring_input["deepline_catalog"])
                != contracts.canonical_json(config["deepline_catalog"])):
            raise ExportError("signal_frozen_catalog_mismatch")
        runtime_fields.add("deepline_catalog")
    if set(scoring_input) - runtime_fields != set(fields):
        raise ExportError("signal_scoring_input_changed")
    frozen_input = {field: scoring_input[field] for field in fields}
    by_use = {}
    for use in uses:
        position = (use.get("run_id"), use.get("company_index"), use.get("signal_index"))
        if (position in by_use or any(type(value) is not int or value < 0 for value in position[1:])):
            raise ExportError("signal_use_invalid")
        by_use[position] = use
    by_authority = {}
    for row in authority_rows:
        identity = (row.get("cache_key"), row.get("authority_slot"))
        if identity in by_authority:
            raise ExportError("signal_authority_ambiguous")
        by_authority[identity] = row
    evidence = []
    seen = set()
    unused_terminal_uses = set()
    company_by_identity = {(row["cache_key"], row["authority_slot"]): row for row in company_rows}
    lease = (judge.get("claim_response") or {}).get("company_judgment_cache") or {}
    lease_by_index = {row["company_index"]: row for row in
                      list(lease.get("hits") or []) + list(lease.get("misses") or [])}
    def source_positions(company_index, signal_index, reference):
        current_refs = judge.get("company_judgment_refs") or []
        if company_index >= len(current_refs):
            return []
        current_ref = company_judgments.validate_company_ref(current_refs[company_index])
        item = lease_by_index.get(company_index)
        if (item is None or item.get("cache_key") != current_ref["cache_key"]
                or item.get("company_input_hash") != current_ref["company_input_hash"]):
            return []
        company_row = company_by_identity.get((item["cache_key"], item["authority_slot"]))
        if company_row is None:
            return []
        source = runs.get(company_row.get("source_score_run_id"))
        if source is None:
            return []
        positions = []
        for ref in source.get("company_judgment_refs") or []:
            if (ref.get("cache_key") != current_ref["cache_key"]
                    or ref.get("company_input_hash") != current_ref["company_input_hash"]):
                continue
            source_ref = company_judgments.validate_company_ref(ref)
            position = (source["run_id"], source_ref["company_index"], signal_index)
            use = by_use.get(position)
            if use is None:
                continue
            if reference is not None and (
                use.get("cache_key") != reference.get("cache_key")
                or use.get("authority_slot") != reference.get("authority_slot")
                or use.get("signal_input_hash") != reference.get("signal_input_hash")
            ):
                continue
            positions.append(position)
        return positions
    for company_index, breakdown in zip(scored_company_indexes, breakdowns):
        details = _signal_details(breakdown)
        unmatched = {position for position in by_use
                     if position[0] == judge["run_id"] and position[1] == company_index}
        for signal_index in {position[2] for position in by_use}:
            unmatched.update(source_positions(company_index, signal_index, None))
        if not details and unmatched:
            if not _terminal_company_fit_nonmatch(breakdown):
                raise ExportError("signal_use_without_reference")
            unused_terminal_uses.update(unmatched)
        for signal_index, detail in enumerate(details):
            if not isinstance(detail, Mapping):
                raise ExportError("signal_detail_invalid")
            reference = detail.get("signal_judgment_reference")
            if reference is not None and not isinstance(reference, Mapping):
                raise ExportError("signal_reference_invalid")
            position = (judge["run_id"], company_index, signal_index)
            use = by_use.get(position)
            if use is None:
                positions = source_positions(company_index, signal_index, reference)
                position = positions[0] if len(positions) == 1 else None
                use = by_use.get(position)
            if reference is None:
                if use is not None:
                    raise ExportError("signal_use_without_reference")
                continue
            if use is None or position in seen:
                raise ExportError("signal_reference_without_use")
            seen.add(position)
            identity = (use.get("cache_key"), use.get("authority_slot"))
            row = by_authority.get(identity)
            if row is None or not isinstance(row.get("scope_doc"), Mapping):
                raise ExportError("signal_authority_missing")
            scope_doc = row["scope_doc"]
            scope = signal_judgments.build_scope(
                scoring_input=frozen_input, company_index=company_index,
                signal_index=signal_index,
                verified_company_identity=scope_doc.get("verified_company_identity"),
                verified_funding_stage_context=scope_doc.get("verified_funding_stage_context"),
                verified_source_pages=scope_doc.get("verified_source_pages"),
                round_id=judge["round_id"],
                network_name=round_row["arena_network_name"],
                netuid=int(round_row.get("arena_netuid") or 0),
                scorer_image_digest=config["scorer_image_digest"],
                scorer_image_reference=config["scorer_image_reference"],
            )
            if (scope_doc != scope
                    or signal_judgments.validate_reference(reference, scope) != reference
                    or identity != (reference["cache_key"], reference["authority_slot"])
                    or use.get("signal_input_hash") != scope["signal_input_hash"]
                    or row.get("signal_input_hash") != scope["signal_input_hash"]):
                raise ExportError("signal_scope_mismatch")
            raw = copy.deepcopy(dict(detail))
            raw.pop("signal_judgment_reference", None)
            raw.pop("counted_in_aggregate", None)
            judgment = signal_judgments.validate_judgment(row.get("judgment_doc"), scope)
            source_id = row.get("source_score_run_id")
            source = runs.get(source_id)
            if (raw != judgment
                    or row.get("judgment_hash") != contracts.document_hash(judgment)
                    or use.get("judgment_hash") != row["judgment_hash"]
                    or use.get("source_score_run_id") != source_id
                    or source is None or source.get("kind") != "score"
                    or source.get("status") not in {"accepted", "failed", "leased"}
                    or source.get("round_id") != judge["round_id"]
                    or source.get("scored_run_id") != row.get("source_scored_run_id")
                    or source.get("runner_hotkey") != row.get("source_runner_hotkey")
                    or source.get("claim_request_hash") != row.get("source_claim_request_hash")
                    or source.get("lease_generation") != row.get("source_lease_generation")):
                raise ExportError("signal_source_mismatch")
            evidence.append({
                "company_index": company_index, "signal_index": signal_index,
                "reference": copy.deepcopy(reference),
                "use": copy.deepcopy(use), "authority": copy.deepcopy(row),
            })
    if not {key for key in by_use if key[0] == judge["run_id"]}.issubset(seen | unused_terminal_uses):
        raise ExportError("signal_use_without_reference")
    return evidence


def _scoring_policy_versions(round_row, history):
    """Bind retained attempts to the owner-recorded model revision history."""
    floor = round_row.get("scoring_generation_floor", 0)
    if type(floor) is not int or floor < 0:
        raise ExportError("scoring_generation_floor_invalid")
    policy = (round_row.get("configuration_doc") or {}).get("scorer_policy")
    if floor == 0:
        return {0: policy}
    if (not isinstance(history, Mapping) or history.get("round_id") != round_row.get("round_id")
            or not isinstance(history.get("revisions"), list) or not history["revisions"]):
        raise ExportError("scoring_policy_history_missing")
    versions, previous = {}, 0
    for revision in history["revisions"]:
        if not isinstance(revision, Mapping):
            raise ExportError("scoring_policy_history_invalid")
        generation = revision.get("generation_floor")
        old, new = revision.get("old_scorer_policy"), revision.get("new_scorer_policy")
        if (type(generation) is not int or not previous < generation <= floor
                or type(revision.get("previous_generation_floor")) is not int
                or revision.get("previous_generation_floor") != previous
                or not isinstance(old, Mapping) or not isinstance(new, Mapping)
                or old == new or previous in versions and versions[previous] != old):
            raise ExportError("scoring_policy_history_invalid")
        versions[previous], versions[generation] = old, new
        previous = generation
    if previous != floor or versions[floor] != policy:
        raise ExportError("scoring_policy_history_invalid")
    return versions


def _scoring_revision(run):
    revision = run.get("scoring_generation_floor", 0)
    if type(revision) is not int or revision < 0:
        raise ExportError("scoring_generation_floor_invalid")
    return revision


def _select_judge(run, runs, *, scoring_generation_floor=0):
    if type(scoring_generation_floor) is not int or scoring_generation_floor < 0:
        raise ExportError("scoring_generation_floor_invalid")
    candidates = [row for row in runs if row.get("kind") == "score"
                  and row.get("scored_run_id") == run["run_id"]]
    for row in candidates:
        if any(row.get(key) != run.get(key) for key in
               ("round_id", "submission_id", "miner_hotkey", "stage", "icp_position")):
            raise ExportError("judge_membership_mismatch")
    candidates = [row for row in candidates
                  if _scoring_revision(row) == scoring_generation_floor]
    if not candidates:
        return None
    rank = lambda row: (row.get("status") == "accepted", int(row.get("stage_generation") or 0), int(row.get("attempt") or 0))
    best = max(map(rank, candidates))
    selected = [row for row in candidates if rank(row) == best]
    if len(selected) != 1:
        raise ExportError("ambiguous_judge")
    return selected[0]


def _runtime(row):
    result = row.get("result_doc") or {}
    return {key: copy.deepcopy(result[key]) for key in
            ("started_at", "finished_at", "terminal_status", "resource_summary", "failure_diagnostic")
            if key in result}


def _reviewer_runtime_identity(row, configuration):
    """Retain lease selection separately from frozen grading, not attestation."""
    binding = reviewer_runtime.lease_binding(row, configuration)
    reviewer_runtime.validate_completion_audit(
        row, configuration, row.get("result_doc") or {},
    )
    if binding is not None:
        return {"selection_basis": "persisted_lease_binding", **binding}
    # Historical score leases predate runtime repairs and used the frozen image.
    return {
        "selection_basis": "frozen_round",
        "grading_image_digest": configuration["scorer_image_digest"],
        "grading_image_reference": configuration["scorer_image_reference"],
        "runtime_image_digest": configuration["scorer_image_digest"],
        "runtime_image_reference": configuration["scorer_image_reference"],
    }


def _verified_runtime_source(resource):
    commit = resource.get("validator_source_commit")
    if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        return False
    origin, dirty = resource.get("validator_source_origin"), resource.get("validator_source_dirty")
    return origin == "git_checkout" and dirty == "clean" or origin in {"release_marker", "verified_archive_environment"} and dirty == "unknown"


def _source_evidence(objects, run, round_row, submission, participant):
    source_ref = participant["source_ref"]
    expected_ref = "arena/%s/sources/%s.tar.gz" % (run["round_id"], run["submission_id"])
    if source_ref != submission.get("source_ref") or source_ref != expected_ref:
        raise ExportError("source_ref_mismatch")
    source = objects.get_bounded(source_ref, source_bundle.MAX_SOURCE_ARCHIVE_BYTES)
    facts = source_bundle.validate_source_archive(source)
    if (type(participant.get("source_size_bytes")) is not int
            or type(submission.get("source_size_bytes")) is not int
            or participant["source_size_bytes"] != facts["source_size_bytes"]
            or submission["source_size_bytes"] != facts["source_size_bytes"]):
        raise ExportError("source_size_mismatch")
    digest = contracts.hash_bytes(source)
    recorded = submission.get("source_archive_sha256")
    if recorded:
        if recorded != digest:
            raise ExportError("source_hash_mismatch")
        origin = "recorded_archive_hash"
    else:
        # Daily baselines predate recorded archive digests. The source endpoint
        # and runner both use the frozen source object and exact frozen size.
        # Derive a new hash only after validating that same immutable boundary.
        expected_baseline = "baseline-" + run["round_id"].removeprefix("arena-")
        if (submission.get("is_king") is not True or participant.get("is_king") is not True
                or submission.get("status") != "frozen"
                or run["submission_id"] != expected_baseline
                or (round_row.get("configuration_doc") or {}).get("baseline_hotkey") != run.get("miner_hotkey")):
            raise ExportError("source_recorded_hash_missing")
        origin = "computed_from_frozen_object"
    return {"ref": source_ref, "sha256": digest, "size_bytes": len(source),
            "hash_origin": origin, "recorded_hash_present": bool(recorded),
            "normalized_sha256": submission.get("source_normalized_sha256"),
            "archive_commit_comment": source_bundle.source_archive_commit(source) or None}


def _response_value(terminal, provider):
    raw = base64.b64decode(terminal["body_b64"], validate=True)
    if len(raw) > MAX_RECEIPT_BYTES:
        raise ExportError("receipt_too_large")
    try:
        response = json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        if provider == "openrouter":
            raise ExportError("opaque_model_response")
        response = {"text": raw.decode("utf-8")}
    from lab_arena.training_capture import sanitize_training_value
    response, redactions = sanitize_training_value(response)
    response, removed = without_hidden_reasoning(response)
    return response, sorted(set(redactions) | ({"hidden_reasoning"} if removed else set()))


def _call_order(call, run_order):
    def numeric(value):
        return value if type(value) is int and value >= 0 else 2 ** 63
    return (run_order.get(call["run_id"], 2 ** 63),
            numeric(call.get("action_sequence")), numeric(call.get("provider_attempt", 1)),
            numeric(call.get("reservation_entry_id", call.get("refusal_entry_id"))), call["call_identity"])


def response_entry_id(call):
    """Select retained delivery evidence without changing the accounting head."""
    terminal = call.get("terminal_entry_id")
    if "response_entry_id" not in call:
        return terminal
    selected = call["response_entry_id"]
    provider = call.get("provider")
    receipts = [row for row in call.get("response_receipts", []) if row.get("entry_id") == selected]
    if (provider not in _RETAINED_ERROR_STATUSES or type(selected) is not int
        or type(terminal) is not int or not 0 < selected <= terminal or len(receipts) != 1):
        raise ExportError("retained_provider_error_identity_invalid")
    receipt = receipts[0]
    if (receipt.get("terminal_source") != f"entry_doc.call.{provider}_terminal_response"
        or receipt.get("entry_kind") != "uncertain"
        or (receipt.get("metadata") or {}).get("call_succeeded") is not False
        or (receipt.get("metadata") or {}).get("status") != call.get("response_status")
        or call.get("response_status") not in _RETAINED_ERROR_STATUSES[provider]
        or receipt.get("response_sha256") != call.get("response_sha256")
        or receipt.get("response_ref") != {"field": "response", "entry_id": selected}):
        raise ExportError("retained_provider_error_identity_invalid")
    if selected != terminal:
        heads = [row for row in call.get("lifecycle", []) if row.get("entry_id") == terminal]
        if (len(heads) != 1 or heads[0].get("entry_kind") != "settlement"
            or (heads[0].get("accounting") or {}).get(f"{provider}_delayed_reconciliation") is not True
            or any(f"{name}_delayed_reconciliation" in (heads[0].get("accounting") or {})
                   for name in _RETAINED_ERROR_STATUSES if name != provider)
            or (heads[0].get("accounting") or {}).get("reconciled_uncertainty_entry_id") != selected):
            raise ExportError("retained_provider_error_identity_invalid")
    return selected


def _calls(rows, run_ids, objects, run_order):
    from lab_arena.broker import BrokerError, _decode_terminal, broker_observations

    def observations(document):
        try:
            return broker_observations(document)
        except BrokerError as exc:
            raise ExportError("broker_observation_invalid") from exc

    grouped, calls, reasons = {}, [], set()
    if len({row.get("entry_id") for row in rows}) != len(rows):
        raise ExportError("duplicate_ledger_entry")
    for row in rows:
        if row.get("run_id") not in run_ids:
            raise ExportError("ledger_run_mismatch")
        identity = row.get("call_identity")
        if not isinstance(identity, str) or not identity:
            reasons.add("ledger_call_identity_missing")
            continue
        # call_identity has a global unique reservation index in Arena SQL.
        # Multiple reservations or cross-run reuse fail closed below.
        grouped.setdefault(identity, []).append(row)
    for identity, entries in sorted(grouped.items()):
        reserves = [row for row in entries if row.get("entry_kind") == "reservation"]
        dispatches = [row for row in entries if row.get("entry_kind") == "dispatch"]
        terminals = [row for row in entries if row.get("entry_kind") in
                     {"settlement", "uncertain", "recovery", "refusal"}]
        response_terminals = terminals
        retained = []
        provider = reserves[0].get("provider") if reserves else None
        if provider in _RETAINED_ERROR_STATUSES:
            terminal_field = f"{provider}_terminal_response"
            for row in terminals:
                saved = (row.get("entry_doc") or {}).get("call") or {}
                if row["entry_kind"] != "uncertain" or terminal_field not in saved:
                    continue
                terminal = saved[terminal_field]
                try:
                    status, _, body = _decode_terminal(terminal)
                    if (saved.get("reason") != "missing_provider_cost"
                        or saved.get("call_succeeded") is not False
                        or terminal.get("call_succeeded") is not False
                        or status not in _RETAINED_ERROR_STATUSES[provider]
                        or provider == "openrouter" and saved.get("openrouter_generation_present") is not True):
                        raise BrokerError("broker_unavailable")
                    if provider == "deepline" and status == 402:
                        from lab_arena.operations import GENERIC_UNAVAILABLE_BODY
                        if (reserves[0].get("funding_source") != "host"
                            or saved.get("deepline_host_payment_refusal") is not True
                            or saved.get("provider_status") != 402
                            or body != GENERIC_UNAVAILABLE_BODY):
                            raise BrokerError("broker_unavailable")
                except BrokerError as exc:
                    raise ExportError("retained_provider_error_invalid") from exc
                retained.append(dict(row, terminal_response=terminal,
                    retained_terminal_source=f"entry_doc.call.{terminal_field}"))
            if retained:
                if len(retained) != 1:
                    raise ExportError("retained_provider_error_ambiguous")
                saved = retained[0]
                for row in terminals:
                    accounting = row.get("entry_doc") or {}
                    markers = {f"{name}_delayed_reconciliation" for name in _RETAINED_ERROR_STATUSES
                               if f"{name}_delayed_reconciliation" in accounting}
                    if markers and (
                        markers != {f"{provider}_delayed_reconciliation"}
                        or accounting.get(f"{provider}_delayed_reconciliation") is not True
                        or accounting.get("reconciled_uncertainty_entry_id") != saved["entry_id"]):
                        raise ExportError("retained_provider_error_identity_invalid")
                # A delayed bill's generic 502 is an accounting placeholder,
                # not another response delivered to the model. Its immutable
                # cost and lifecycle remain authoritative below.
                response_terminals = [saved] + [row for row in terminals
                    if row["entry_id"] != saved["entry_id"]
                    and not ((row.get("entry_doc") or {}).get(f"{provider}_delayed_reconciliation") is True
                        and (row.get("entry_doc") or {}).get("reconciled_uncertainty_entry_id") == saved["entry_id"])]
        standalone_refusal = (not reserves and not dispatches and len(entries) == 1
                              and entries[0].get("entry_kind") == "refusal"
                              and entries[0].get("amount_microusd") == 0
                              and (entries[0].get("entry_doc") or {}).get("reason")
                                  in {"per_icp_quota", "stage_quota", "money_cap"}
                              and isinstance((entries[0].get("entry_doc") or {}).get("call"), Mapping))
        if not standalone_refusal and (len(reserves) != 1 or len(dispatches) > 1):
            reasons.add("ambiguous_call_lifecycle")
            continue
        reserve = entries[0] if standalone_refusal else reserves[0]
        if any(any(row.get(key) != reserve.get(key) for key in
                   ("run_id", "round_id", "submission_id", "provider", "operation_id")) for row in entries):
            raise ExportError("ledger_membership_mismatch")
        entry = ((reserve.get("entry_doc") or {}).get("call") if standalone_refusal
                 else reserve.get("entry_doc")) or {}
        call = {"call_identity": identity, "run_id": reserve["run_id"],
                "provider": reserve.get("provider"), "operation_id": reserve.get("operation_id"),
                "funding_source": reserve.get("funding_source"),
                **({"refusal_entry_id": reserve["entry_id"], "refusal_receipt": copy.deepcopy(reserve)}
                   if standalone_refusal else {"reservation_entry_id": reserve["entry_id"]}),
                **{key: entry[key] for key in ("base_call_identity", "provider_attempt", "action_sequence", "request_hash") if key in entry}}
        if isinstance(entry.get("reasoning_capture"), Mapping):
            call["reasoning_capture"] = copy.deepcopy(entry["reasoning_capture"])
        call_reasons = set()
        upstream_receipts = []
        for row in sorted(entries, key=lambda row: int(row["entry_id"])):
            saved_call = (row.get("entry_doc") or {}).get("call") or {}
            terminal = row.get("terminal_response") or {}
            captured = (saved_call.get("upstream_response_receipt")
                        if isinstance(saved_call, Mapping) else None)
            captured = (captured or terminal.get("upstream_response_receipt")
                        or (row.get("entry_doc") or {}).get("upstream_response_receipt"))
            if not captured:
                continue
            if isinstance(captured, Mapping) and "broker_observations" in captured:
                observations({"upstream_response_receipt": captured})
                if set(captured) == {"broker_observations"}:
                    continue  # Timing/route evidence does not claim an upstream body exists.
            if isinstance(captured, Mapping) and captured.get("status") == "unavailable":
                if captured.get("reason") in {"response_capture_failed", "response_size_limit"}:
                    call["upstream_response_capture"] = {"status": "unavailable", "reason": captured["reason"]}
                if captured.get("reason") == "response_size_limit":
                    call_reasons.add("upstream_response_size_limit")
                call_reasons.add("upstream_response_capture_unavailable")
                continue
            try:
                if (not isinstance(captured, Mapping)
                        or captured.get("capture_kind") != "upstream_response"
                        or captured.get("call_identity") != identity
                        or captured.get("operation_id") != reserve.get("operation_id")
                        or captured.get("request_hash") != entry.get("request_hash")
                        or type(captured.get("status")) is not int
                        or not 100 <= captured["status"] <= 599
                        or type(captured.get("wire_body_bytes")) is not int
                        or captured["wire_body_bytes"] < 0
                        or not isinstance(captured.get("wire_body_sha256"), str)
                        or not contracts.SHA256_RE.fullmatch(captured["wire_body_sha256"])):
                    raise ExportError("upstream_response_identity_invalid")
                raw = base64.b64decode(captured["body_b64"], validate=True)
                if (len(raw) != captured.get("body_bytes")
                        or contracts.hash_bytes(raw) != captured.get("body_sha256")):
                    raise ExportError("upstream_response_hash_invalid")
                privacy = captured.get("privacy")
                if (not isinstance(privacy, Mapping)
                        or type(privacy.get("redacted")) is not bool
                        or not isinstance(privacy.get("redactions"), list)
                        or any(reason not in {"credential", "hidden_reasoning"}
                               for reason in privacy["redactions"])
                        or privacy["redacted"] != bool(privacy["redactions"])):
                    raise ExportError("upstream_response_privacy_invalid")
                if not privacy["redacted"] and (len(raw) != captured["wire_body_bytes"]
                        or contracts.hash_bytes(raw) != captured["wire_body_sha256"]):
                    raise ExportError("upstream_response_hash_invalid")
                # Failed model replies may be malformed JSON. Retain their
                # privacy-filtered text without treating it as a model answer.
                response, removed = _response_value(captured, "upstream")
                redactions = sorted(set(privacy["redactions"]) | set(removed))
                upstream_receipts.append({
                    "entry_id": row["entry_id"], "status": captured["status"],
                    "response": response, "privacy_redactions": redactions,
                    "body_sha256": captured["body_sha256"], "body_bytes": len(raw),
                    "wire_body_sha256": captured["wire_body_sha256"],
                    "wire_body_bytes": captured["wire_body_bytes"],
                    "provenance": "private_upstream_response_receipt",
                    "delivery_to_worker": "not_proved",
                    "response_sha256": contracts.document_hash(response),
                })
                if redactions:
                    call_reasons.add("upstream_response_privacy_nonexact")
            except Exception:
                call_reasons.add("upstream_response_capture_unavailable")
        if upstream_receipts:
            # This is provider evidence. It must never replace the bridge
            # response or establish what the agent received or paid.
            call["upstream_response_receipts"] = upstream_receipts
        accounting_fields = {"reserved_microusd", "released_microusd", "variance_microusd", "late_reconciliation",
                             "openrouter_delayed_reconciliation", "deepline_delayed_reconciliation",
                             "reconciled_uncertainty_entry_id", "reconciled_uncertainty_reason",
                             "openrouter_generation_id", "deepline_request_id", "deepline_operation",
                             "request_hash", "base_call_identity", "action_sequence", "provider_attempt", "provider_cost",
                             "transport_elapsed_ms", "timing_scope"}
        call["lifecycle"] = []
        for row in sorted(entries, key=lambda row: int(row["entry_id"])):
            metadata = {key: copy.deepcopy(row[key]) for key in
                        ("entry_id", "entry_kind", "created_at", "funding_source", "amount_microusd") if key in row}
            metadata["accounting"] = {key: copy.deepcopy(value) for key, value in (row.get("entry_doc") or {}).items()
                                      if key in accounting_fields}
            if (row.get("entry_kind") == "recovery"
                    and (row.get("entry_doc") or {}).get("reason") == "attempt_completed_before_dispatch"):
                metadata["accounting"]["reason"] = "attempt_completed_before_dispatch"
            metadata["accounting"].update({key: copy.deepcopy(value) for key, value in
                observations(row.get("entry_doc") or {}).items() if key in accounting_fields})
            recorded_call = (row.get("entry_doc") or {}).get("call")
            if isinstance(recorded_call, Mapping):
                call_accounting = {key: copy.deepcopy(value) for key, value in recorded_call.items() if key in accounting_fields}
                call_accounting.update({key: copy.deepcopy(value) for key, value in
                    observations(recorded_call).items() if key in accounting_fields})
                if call_accounting:
                    metadata["accounting"]["call"] = call_accounting
            call["lifecycle"].append(metadata)
        call["response_receipts"] = []
        if standalone_refusal:
            terminals = []  # Admission denial has no provider terminal or generation.
            response_terminals = []
        latest_id = max((int(row["entry_id"]) for row in response_terminals), default=None)
        latest_terminal = next((row.get("terminal_response") or {} for row in response_terminals if row["entry_id"] == latest_id), {})
        latest_response, latest_redactions, latest_digest = None, [], None
        try:
            latest_response, latest_redactions = _response_value(latest_terminal, reserve.get("provider"))
            latest_digest = contracts.document_hash(latest_response)
        except Exception:
            pass  # Missing receipt is recorded below; no failed proof is hidden.
        saved_bodies = {}
        for row in sorted(response_terminals, key=lambda row: int(row["entry_id"])):
            terminal = row.get("terminal_response") or {}
            receipt = {"entry_id": row["entry_id"], "entry_kind": row["entry_kind"],
                       "created_at": row.get("created_at"),
                       "metadata": {key: copy.deepcopy(value) for key, value in terminal.items()
                                    if key not in {"body_b64", "upstream_response_receipt"}}}
            if row.get("retained_terminal_source"):
                receipt["terminal_source"] = row["retained_terminal_source"]
            receipt["metadata"].update(copy.deepcopy(observations(terminal)))
            try:
                # The latest receipt references the response already validated
                # above. Keep wire proof and every other receipt's checks.
                reuse_latest = terminal is latest_terminal and latest_digest is not None
                if reuse_latest:
                    response, redactions = latest_response, latest_redactions
                else:
                    response, redactions = _response_value(terminal, reserve.get("provider"))
                wire_body = base64.b64decode(terminal["body_b64"], validate=True)
                receipt["wire_body_sha256"] = contracts.hash_bytes(wire_body)
                receipt["wire_body_bytes"] = len(wire_body)
                digest = latest_digest if reuse_latest else contracts.document_hash(response)
                receipt["response_sha256"], receipt["privacy_redactions"] = digest, redactions
                if digest == latest_digest:
                    receipt["response_ref"] = {"field": "response", "entry_id": latest_id}
                elif digest in saved_bodies:
                    receipt["response_ref"] = {"entry_id": saved_bodies[digest]}
                else:
                    receipt["response"] = response
                    saved_bodies[digest] = row["entry_id"]
                if "credential" in redactions:
                    call_reasons.add("response_privacy_nonexact")
            except Exception:
                receipt["response_unavailable"] = True
            call["response_receipts"].append(receipt)
        call["response_provenance"] = {"source": "stored_provider_ledger_receipt", "delivery_to_worker": "not_proved"}
        pointer = entry.get("request_capture") or {}
        try:
            from lab_arena.training_capture import read_request
            if pointer.get("status") not in {"stored", "referenced"}:
                raise ExportError("request_not_stored")
            call["request"] = read_request(objects, pointer)
            call["request_capture"] = pointer
            request = call["request"]
            if request.get("request_hash") != entry.get("request_hash") or request.get("operation_id") != reserve.get("operation_id"):
                raise ExportError("request_identity_mismatch")
            if pointer.get("redacted") is True:
                call_reasons.add("request_privacy_nonexact")
            elif not isinstance(request.get("hashed_parameters"), dict) or contracts.document_hash(request["hashed_parameters"]) != entry.get("request_hash"):
                raise ExportError("request_parameters_hash_mismatch")
        except Exception:
            call.pop("request", None)
            call_reasons.add("full_request_unavailable")
        if standalone_refusal:
            call["accounting_state"] = "refusal"
            call["actual_microusd"] = 0
            call["response_provenance"]["source"] = "stored_admission_refusal"
        elif not terminals:
            call_reasons.add("provider_call_inflight")
        else:
            # A later recovery can settle a previous uncertain entry. The
            # append-only ledger's highest entry_id is the current authority.
            latest = max(terminals, key=lambda row: int(row["entry_id"]))
            if len({row["entry_id"] for row in terminals}) != len(terminals):
                raise ExportError("duplicate_ledger_entry")
            call["accounting_state"] = latest["entry_kind"]
            call["terminal_entry_id"] = latest["entry_id"]
            if type(latest.get("amount_microusd")) is not int or latest["amount_microusd"] < 0:
                raise ExportError("call_cost_invalid")
            call["actual_microusd"] = latest["amount_microusd"] if latest["entry_kind"] != "uncertain" else None
            if latest["entry_kind"] == "uncertain":
                call_reasons.add("provider_cost_unresolved")
            terminal = latest.get("terminal_response") or {}
            if "provider_cost" in terminal:
                call["provider_cost"] = copy.deepcopy(terminal["provider_cost"])
            if latest_digest is not None:
                call["response"] = latest_response
                call["response_sha256"] = latest_digest
                call["response_hidden_reasoning_removed"] = "hidden_reasoning" in latest_redactions
                call["response_privacy_redactions"] = latest_redactions
                call["response_status"] = latest_terminal.get("status")
                if retained and latest_id == retained[0]["entry_id"]:
                    call["response_entry_id"] = latest_id
            else:
                call_reasons.add("full_response_unavailable")
            for key in ("judgment_cache_source_call_identity", "judgment_cache_source_run_id"):
                if key in terminal:
                    call[key] = terminal[key]
        call["evidence_gaps"] = sorted(call_reasons)
        reasons.update(call_reasons)
        calls.append(call)
    return sorted(calls, key=lambda call: _call_order(call, run_order)), reasons


def _reported_phase_coverage(payload):
    """Check the declared baseline phase contract, not arbitrary local memory."""
    states = [row["data"] for row in payload.get("agent_observations", []) if row["kind"] == "state_snapshot"]
    # The runner adds its recorder status after the agent counts its own states.
    if sum(row["name"] == "automatic_capture_summary" for row in states) > 1:
        return False
    states = [row for row in states if row["name"] != "automatic_capture_summary"]
    summaries = [row["value"] for row in states if row["name"] == "training_capture_summary"]
    required = {"after_discovery", "after_verification", "after_admission", "final"}
    if len(summaries) != 1 or any(sum(row["name"] == name for row in states) != 1 for name in required):
        return False
    summary = summaries[0]
    return (isinstance(summary, dict) and summary.get("contract_version") == 1
            and summary.get("scope") == "agent_direct_egress_and_worker_socket"
            and summary.get("sdk_available") is True
            and summary.get("state_recorded") == len(states) - 1
            and all(summary.get(key, 0) == 0 for key in (
                "state_record_errors", "state_local_errors", "state_overflow_rows", "state_overflow_manifests")))


def _not_dispatched_host_observations(run, paid, snapshot, configuration, ledger_rows):
    """Prove counted local broker errors without creating provider ledger calls."""
    from lab_arena import training_coverage, training_observations
    if ledger_rows is None or not training_coverage.uses_host_result_evidence(configuration or {}):
        return {}
    payload = snapshot.get("payload") or {}
    candidates = [row for row in payload.get("observations") or []
                  if (row.get("outcome") or {}).get("broker_disposition")]
    if not candidates:
        return {}
    try:
        if (snapshot.get("schema_version") != training_observations.SCHEMA_VERSION
                or snapshot.get("run_id") != run["run_id"] or snapshot.get("kind") != run["kind"]
                or snapshot.get("payload_sha256") != contracts.document_hash(payload)):
            return {}
        found = {}
        observations = payload["observations"]
        # Both callers validate the full hash-bound host payload. The runner
        # stamps callbacks with its latest started action, which can advance
        # while an earlier call is in flight. Only host rows in this snapshot
        # can corroborate that frontier; callbacks cannot corroborate themselves.
        host_sequences = {item["action_sequence"]
                          for rows in (observations, payload["response_receipts"]) for item in rows
                          if type(item.get("action_sequence")) is int}
        for row in candidates:
            identity, sequence = row.get("call_identity"), row.get("action_sequence")
            disposition = row["outcome"]["broker_disposition"]
            if (training_observations.clean_observation(row) != row
                    or row["kind"] != "nonstandard_response"
                    or row["contributes_to_provider_call_count"] is not True
                    or not isinstance(identity, str) or contracts.SHA256_RE.fullmatch(identity) is None
                    or type(sequence) is not int or sequence < 0
                    or disposition != {"outcome": "not_dispatched", "reason": disposition.get("reason")}
                    or disposition.get("reason") not in training_observations.BROKER_NOT_DISPATCHED_REASONS
                    or not training_observations._expected_not_dispatched_response(row["outcome"].get("response"))
                    or sum(item.get("call_identity") == identity for item in observations) != 1
                    or sum(item.get("action_sequence") == sequence for item in observations) != 1
                    or any(call.get("call_identity") == identity or call.get("base_call_identity") == identity
                           or call.get("action_sequence") == sequence for call in paid)):
                continue
            # Supplied raw rows remain authority even if no paid call could be
            # assembled from them. A broker tag cannot erase accounting.
            collision = False
            for entry in ledger_rows:
                doc = entry.get("entry_doc") or {}
                call = doc.get("call") or {}
                if (entry.get("call_identity") == identity
                        or doc.get("base_call_identity") == identity
                        or call.get("base_call_identity") == identity
                        or entry.get("run_id") == run["run_id"] and (
                            doc.get("action_sequence") == sequence or call.get("action_sequence") == sequence)):
                    collision = True
                    break
            if collision:
                continue
            writes = [item for item in payload["response_receipts"]
                      if item.get("call_identity") == identity or item.get("ordinal") == row["ordinal"]
                      or item.get("action_sequence") == sequence]
            if len(writes) != 1:
                continue
            write = writes[0]
            response = row["outcome"]["response"]
            raw = base64.b64decode(response["body_b64"], validate=True)
            if (training_observations.clean_response_receipt(write) != write
                    or any(write[key] != row[key] for key in ("ordinal", "action_sequence", "operation_id", "call_identity"))
                    or write["protocol"] != row["outcome"]["protocol"]
                    or write["write_state"] != "socket_written" or write["status"] != response["status"]
                    or write["body_bytes"] != len(raw) or write["body_sha256"] != contracts.hash_bytes(raw)):
                continue
            reads = [item for item in payload["agent_observations"] if item.get("kind") == "response_received"
                     and (item.get("data") or {}).get("call_identity") == identity]
            # Host-result rounds do not require cooperative miner callbacks.
            # Every callback that is present must still agree with the write.
            if any(training_observations.clean_agent_observation(read) != read
                   or read["privacy"] != {"redacted": False, "redactions": []}
                   or type(read["after_action_sequence"]) is not int
                   or read["after_action_sequence"] < sequence
                   or read["after_action_sequence"] not in host_sequences
                   or any(read["data"].get(key) != write[key]
                          for key in ("call_identity", "status", "body_sha256", "body_bytes"))
                   for read in reads):
                continue
            found[identity] = row
        return found
    except (KeyError, IndexError, TypeError, ValueError, AttributeError):
        return {}


def _link_worker_receipts(run, paid, snapshot, *, require_completed_phases=True, require_client_ack=True, configuration=None, ledger_rows=None, update_provenance=True):
    """Join host writes and cooperative read acknowledgments without guessing.

    A host write is not consumption. An SDK acknowledgment is an agent report,
    bound to the exact response bytes and lease; it does not prove use of them.
    Web text and state stay once in the private artifact, referenced by ID.
    """
    payload = snapshot["payload"]
    errors, linked = set(), {}
    not_dispatched = _not_dispatched_host_observations(run, paid, snapshot, configuration, ledger_rows)
    by_identity = {call["call_identity"]: call if update_provenance else {
        **call, "response_provenance": copy.deepcopy(call.get("response_provenance") or {})}
        for call in paid}
    error_observations = {row["ordinal"]: row for row in payload.get("observations", [])
                          if row.get("call_identity") is not None}
    for receipt in payload.get("response_receipts", []):
        call = by_identity.get(receipt["call_identity"])
        if call is None and receipt["call_identity"] in not_dispatched:
            linked[receipt["call_identity"]] = receipt
            continue
        if (call is None or call.get("action_sequence") != receipt["action_sequence"]
                or call.get("operation_id") != receipt["operation_id"]):
            errors.add("worker_response_receipt_link_invalid")
            continue
        matches = [item for item in call["response_receipts"]
                   if item.get("wire_body_sha256") == receipt["body_sha256"]
                   and item.get("wire_body_bytes") == receipt["body_bytes"]
                   and item.get("metadata", {}).get("status") == receipt["status"]]
        observed = error_observations.get(receipt["ordinal"])
        worker_error_body = False
        if observed is not None:
            response = observed["outcome"].get("response") or {}
            worker_error_body = (observed["call_identity"] == receipt["call_identity"]
                                 and observed["action_sequence"] == receipt["action_sequence"]
                                 and observed["operation_id"] == receipt["operation_id"]
                                 and response.get("status") == receipt["status"]
                                 and isinstance(response.get("body_b64"), str)
                                 and (observed["privacy"]["redacted"]
                                      or len(base64.b64decode(response["body_b64"], validate=True)) == receipt["body_bytes"])
                                 and (observed["privacy"]["redacted"]
                                      or contracts.hash_bytes(base64.b64decode(response["body_b64"], validate=True))
                                      == receipt["body_sha256"]))
        identity = receipt["call_identity"]
        if (not matches and not worker_error_body) or identity in linked:
            errors.add("worker_response_receipt_link_invalid")
            continue
        if worker_error_body and not matches and any(
                item.get("wire_body_sha256") for item in call["response_receipts"]):
            errors.add("worker_provider_response_mismatch")
        linked[identity] = receipt
        call["response_provenance"].update({
            "delivery_to_worker": receipt["write_state"],
            "worker_receipt_ref": {"run_id": run["run_id"], "ordinal": receipt["ordinal"],
                                   "payload_sha256": snapshot["payload_sha256"]},
            "agent_receipt": "not_reported",
            "worker_received_receipt_entry_ids": sorted(item["entry_id"] for item in matches),
            "worker_body_source": "ledger_terminal" if matches else "worker_error_observation",
            "worker_body_exact_in_artifact": bool(matches or worker_error_body and not observed["privacy"]["redacted"]),
        })
        if worker_error_body:
            call["response_provenance"]["worker_error_observation_ref"] = {
                "run_id": run["run_id"], "ordinal": observed["ordinal"],
                "payload_sha256": snapshot["payload_sha256"]}
    acknowledged = set()
    for row in payload.get("agent_observations", []):
        if row["privacy"]["redacted"]:
            errors.add("worker_observation_privacy_nonexact")
        if row["kind"] == "web_fetch_binary":
            errors.add("binary_privacy_review_required")
        for reference in row["depends_on"]:
            if reference.startswith("sha256:"):
                dependency = by_identity.get(reference) or not_dispatched.get(reference)
                if (dependency is None or type(dependency.get("action_sequence")) is not int
                        or type(row["after_action_sequence"]) is not int
                        or dependency["action_sequence"] > row["after_action_sequence"]):
                    errors.add("agent_dependency_link_invalid")
        if row["kind"] != "response_received":
            continue
        data = row["data"]
        receipt = linked.get(data["call_identity"])
        if (receipt is None or receipt["write_state"] != "socket_written"
                or any(receipt[key] != data[key] for key in ("status", "body_sha256", "body_bytes"))):
            errors.add("agent_response_receipt_link_invalid")
            continue
        if data["call_identity"] in not_dispatched:
            acknowledged.add(data["call_identity"])
            continue
        call = by_identity[data["call_identity"]]
        ref = {"run_id": run["run_id"], "id": row["id"], "payload_sha256": snapshot["payload_sha256"]}
        call["response_provenance"].setdefault("agent_receipt_refs", []).append(ref)
        call["response_provenance"]["agent_receipt"] = "reported_exact_body_received"
        latest_acknowledged = response_entry_id(call) in call["response_provenance"]["worker_received_receipt_entry_ids"]
        call["response_provenance"]["selected_response_acknowledged"] = latest_acknowledged
        if not latest_acknowledged and call["response_provenance"].get("worker_body_source") == "worker_error_observation":
            call["response_provenance"]["worker_error_body_acknowledged"] = True
        elif not latest_acknowledged:
            errors.add("agent_selected_response_unacknowledged")
        acknowledged.add(data["call_identity"])
    # The caller sets this flag only for the selected accepted execution.
    # Related judge attempts may stop before delivery; every present link
    # still passes the unconditional identity and byte checks above.
    if not require_client_ack and require_completed_phases:
        from lab_arena import training_thinking
        proof = {"run_id": run["run_id"], "configuration": configuration or {},
                 "provider_calls": paid, "worker_observation_artifacts": {run["run_id"]: snapshot}}
        if any(call["call_identity"] not in linked
               and not training_thinking.verified_recovered_provider_retry(proof, call)
               and not training_thinking.verified_undispatched_transport(
                   proof, call, require_observation_ref=False) for call in paid):
            errors.add("worker_response_receipt_link_invalid")
    if run["kind"] == "execute":
        expected = {call["call_identity"] for call in paid
                    if call.get("accounting_state") == "refusal" or
                    type(call.get("response_status")) is int and
                    (200 <= call["response_status"] < 300 or call["call_identity"] in linked)}
        if require_client_ack and expected - acknowledged:
            errors.add("agent_response_receipts_unverified")
        if require_client_ack and paid and require_completed_phases and not _reported_phase_coverage(payload):
            errors.add("agent_task_state_unverified")
    return errors


def _worker_evidence(related, calls, objects, *, selected_run_id, configuration=None, ledger_rows=None):
    from lab_arena import training_observations
    from lab_arena import training_coverage
    require_client_ack = not training_coverage.uses_host_result_evidence(configuration or {})
    artifacts, partial_artifacts, unverified_artifacts = {}, {}, {}
    errors, counts = set(), {}
    per_run_errors, verified_counts = {}, {}
    selected = next((row for row in related if row["run_id"] == selected_run_id
                     and row["kind"] == "execute" and row.get("status") == "accepted"), {})

    def record_error(code, run_id):
        errors.add(code)
        per_run_errors.setdefault(run_id, set()).add(code)
    for run in related:
        run_id = run["run_id"]
        summary = (run.get("result_doc") or {}).get("resource_summary") or {}
        capture = summary.get("worker_observations")
        paid = [call for call in calls if call["run_id"] == run_id]
        # A failed predecessor can stop before admission/final states. Its
        # receipts and observations still pass through every check below.
        require_completed_phases = (run_id == selected_run_id and run["kind"] == "execute"
                                    and run.get("status") == "accepted")
        sequences = {call["action_sequence"] for call in paid if type(call.get("action_sequence")) is int}
        counts[run_id] = set(sequences)
        local_refusals = summary.get("provider_local_refusal_count")
        if type(local_refusals) is not int or local_refusals < 0:
            record_error("worker_local_refusals_unverified", run_id)
        needed = (type(summary.get("provider_call_count")) is not int or summary.get("provider_call_count") != len(sequences)
                  or type(local_refusals) is int and local_refusals > 0
                  or any("full_response_unavailable" in call["evidence_gaps"] for call in paid))
        try:
            snapshot = training_observations.read_snapshot(objects, run_id, run["kind"], run.get("lease_generation"), run.get("claim_request_hash"))
        except Exception:
            try:
                if not callable(getattr(objects, "get_optional_bounded", None)):
                    raise NotImplementedError("legacy object reader has no journal")
                segments, partial, journal_error = training_observations.read_journal_prefix(
                    objects, run_id, run["kind"], run.get("lease_generation"),
                    run.get("claim_request_hash"))
                if partial is not None:
                    partial_artifacts[run_id] = {
                        "schema_version": training_observations.JOURNAL_SCHEMA_VERSION,
                        "run_id": run_id, "kind": run["kind"],
                        "lease_generation": run["lease_generation"],
                        "claim_request_hash": run["claim_request_hash"],
                        "complete": False, "payload": partial,
                        "payload_sha256": contracts.document_hash(partial),
                        "segment_refs": [training_observations.segment_ref(
                            run_id, run["lease_generation"], run["claim_request_hash"],
                            segment["number"]) for segment in segments],
                        "segment_sha256": [segment["segment_sha256"] for segment in segments],
                    }
                    record_error("worker_observations_incomplete", run_id)
                    if not require_client_ack:
                        # The reader verified every retained segment and its
                        # lease-bound chain. Link only its persisted writes;
                        # this prefix cannot establish the absent final tail.
                        prefix = partial_artifacts[run_id]
                        for error in _link_worker_receipts(
                                run, paid, prefix, require_completed_phases=False,
                                require_client_ack=False, configuration=configuration,
                                ledger_rows=ledger_rows):
                            record_error(error, run_id)
                        for call in paid:
                            ref = (call.get("response_provenance") or {}).get("worker_receipt_ref")
                            if ref is not None:
                                ref["artifact_field"] = "partial_worker_observation_artifacts"
                if journal_error:
                    record_error("worker_observation_journal_unavailable", run_id)
            except NotImplementedError:
                pass
            except Exception:
                record_error("worker_observation_journal_unavailable", run_id)
            if needed or capture is not None or (paid and not require_client_ack):
                record_error("worker_observations_unavailable", run_id)
            if run["kind"] == "execute" and paid:
                if require_client_ack:
                    record_error("agent_response_receipts_unverified", run_id)
                if require_client_ack and require_completed_phases:
                    record_error("agent_task_state_unverified", run_id)
            continue
        payload = snapshot["payload"]
        # Native host-result proof comes from this exact immutable, validated
        # snapshot. An unavailable upload acknowledgement is not a conflicting
        # claim about its bytes. Require every supplied claim to agree, and
        # keep the legacy acknowledgement policy unchanged.
        unconfirmed_native_capture = (
            not require_client_ack and isinstance(capture, dict)
            and set(capture) <= {"status", "payload_sha256", "observation_count", "local_refusal_count", "complete"}
            and capture.get("status") == "unavailable"
            and ("payload_sha256" not in capture or capture["payload_sha256"] == snapshot["payload_sha256"])
            and all(type(capture.get(key)) is int and capture[key] == payload[key]
                    for key in ("observation_count", "local_refusal_count"))
            and type(capture.get("complete")) is bool and capture["complete"] == payload["complete"])
        if capture is not None and (not isinstance(capture, dict)
                or capture.get("status") != "stored" or capture.get("payload_sha256") != snapshot["payload_sha256"]
                or capture.get("observation_count") != payload["observation_count"]
                or capture.get("local_refusal_count") != payload["local_refusal_count"]
                or capture.get("complete") != payload["complete"]) and not unconfirmed_native_capture:
            # The object passed identity, hash, schema and privacy validation.
            # Retain it for inspection, but a failed or conflicting completion
            # summary cannot authorize receipt links or verified counters.
            # Journal prefixes have separate eligibility semantics, so keep
            # this unconfirmed snapshot outside their partial collection.
            # Outer completeness describes that unverified audit status; the
            # original payload and its claimed completeness remain unchanged.
            unverified_artifacts[run_id] = {
                **snapshot, "complete": False,
                "unverified_reasons": ["worker_observation_summary_invalid"],
            }
            record_error("worker_observation_summary_invalid", run_id)
            continue
        if not payload["complete"]:
            record_error("worker_observations_incomplete", run_id)
        if payload["local_refusal_count"] != local_refusals:
            record_error("worker_local_refusals_unverified", run_id)
        artifacts[run_id] = snapshot
        for error in _link_worker_receipts(
                run, paid, snapshot, require_completed_phases=require_completed_phases,
                require_client_ack=require_client_ack, configuration=configuration, ledger_rows=ledger_rows):
            record_error(error, run_id)
        not_dispatched = _not_dispatched_host_observations(run, paid, snapshot, configuration, ledger_rows)
        for observation in payload["observations"]:
            if observation["privacy"]["redacted"]:
                record_error("worker_observation_privacy_nonexact", run_id)
            sequence, identity = observation["action_sequence"], observation["call_identity"]
            matches = [call for call in paid if call.get("action_sequence") == sequence]
            if identity is not None:
                sequence_matches = matches
                matches = [call for call in matches if call["call_identity"] == identity]
                if not matches and (sequence_matches or any(call["call_identity"] == identity for call in paid)):
                    record_error("worker_observation_link_invalid", run_id)
                    continue
            if matches:
                if len(matches) != 1 or matches[0]["operation_id"] != observation["operation_id"] or not observation["contributes_to_provider_call_count"]:
                    record_error("worker_observation_link_invalid", run_id)
                    continue
                call = matches[0]
                if isinstance(call.get("request"), dict) and call["request"].get("admitted_parameters") != observation["request"]["admitted_parameters"]:
                    record_error("worker_observation_request_mismatch", run_id)
                    continue
                call["worker_observation_ref"] = {"run_id": run_id, "ordinal": observation["ordinal"], "payload_sha256": snapshot["payload_sha256"]}
                # Only a returned bridge response can fill the missing-response
                # gap. An exception-only observation does not prove a body.
                # Neither outcome becomes an upstream receipt or known cost.
                if isinstance(observation["outcome"].get("response"), dict):
                    call["evidence_gaps"] = [gap for gap in call["evidence_gaps"] if gap != "full_response_unavailable"]
            elif observation["contributes_to_provider_call_count"]:
                if type(sequence) is not int:
                    record_error("worker_observation_link_invalid", run_id)
                    continue
                counts[run_id].add(sequence)
                if observation["kind"] != "local_catalog" and identity not in not_dispatched:
                    record_error("worker_call_accounting_unverified", run_id)
        # A lease can expire after the immutable capture upload but before the
        # completion summary. Only an earlier failed execution in a validated
        # host-result round may use exact complete capture for absent counters.
        # Present counters remain independent checks, including invalid values
        # and contradictions.
        run_errors = per_run_errors.get(run_id, set())
        missing_local = "provider_local_refusal_count" not in summary
        expected_calls = summary.get("provider_call_count", len(counts[run_id]))
        if (not require_client_ack and run_id != selected_run_id
                and run["kind"] == "execute" and run.get("status") == "failed"
                and type(run.get("attempt")) is int and type(selected.get("attempt")) is int
                and 0 < run["attempt"] < selected["attempt"]
                and payload["complete"] and "response_receipts" in payload
                and not run_errors - ({"worker_local_refusals_unverified"} if missing_local else set())
                and type(expected_calls) is int and expected_calls == len(counts[run_id])
                and ledger_rows is not None
                and {row["entry_id"] for row in ledger_rows if row.get("run_id") == run_id}
                    == {entry["entry_id"] for call in paid for entry in call["lifecycle"]}
                and all(type(call.get("action_sequence")) is int and call["action_sequence"] >= 0
                        and type(call.get("actual_microusd")) is int and call["actual_microusd"] >= 0
                        and not call["evidence_gaps"]
                        and not call.get("response_hidden_reasoning_removed")
                        and not call.get("response_privacy_redactions")
                        and call["response_provenance"].get("worker_receipt_ref")
                        and call["response_provenance"].get("delivery_to_worker") == "socket_written"
                        for call in paid)):
            verified_counts[run_id] = len(counts[run_id])
            if missing_local:
                run_errors.discard("worker_local_refusals_unverified")
    errors = set().union(*per_run_errors.values()) if per_run_errors else set()
    return artifacts, partial_artifacts, errors, counts, per_run_errors, verified_counts, unverified_artifacts


def _competition_cost_snapshot(*, run, configuration, scored, related,
                               reader: Callable[..., Mapping[str, Any]] | None) -> dict:
    """Keep the export-time cost decision separate from the quality label.

    The existing SQL authority calculates successful-sourcing cost. Its result
    is a snapshot, not a persisted publication decision or actual paid cost.
    """

    policy = configuration.get("sourcing_cost_eligibility_policy")
    coverage_percent = configuration.get("training_minimum_icp_coverage_percent")
    confirmed_full_thinking_cost = (
        configuration.get("reasoning_capture_policy") == "full_generated_thinking_v1"
        and type(coverage_percent) is int and 1 <= coverage_percent <= 100
    )
    unavailable = lambda reason: {"status": "unavailable", "reason": reason,
                                  "policy": policy if isinstance(policy, str) else None}
    if policy not in contracts.PER_ICP_SUCCESSFUL_CALLS_COST_POLICIES:
        return unavailable("per_icp_policy_not_configured")
    if (configuration.get("integrity_policy") != "arena_integrity_v1"
            or scored is None or run.get("status") != "accepted"
            or any(row.get("status") not in ("accepted", "failed") for row in related)):
        return unavailable("quality_or_attempt_not_final")
    try:
        # This is the exact projection used when stage scores are persisted.
        record = scoring.run_scores_for_store(
            {"rows": [scored], "integrity_policy": "arena_integrity_v1"}, [run]
        )[0]
        qualification = record["qualification_doc"]
        if qualification != run.get("qualification_doc"):
            return unavailable("persisted_qualification_mismatch")
        qualified = {
            row["company_identity_key"] for row in qualification["companies"]
            if row["company_qualified"] is True and row["duplicate_company"] is False
        }
        if any(type(row.get("company_qualified")) is not bool
                or type(row.get("duplicate_company")) is not bool
                or not isinstance(row.get("company_identity_key"), str)
                or not row["company_identity_key"]
                for row in qualification["companies"]):
            return unavailable("persisted_qualification_invalid")
    except (KeyError, IndexError, TypeError, ValueError, scoring.ScoringError):
        return unavailable("persisted_qualification_unavailable")
    if reader is None:
        return unavailable("cost_authority_unavailable")
    try:
        observed = reader(
            round_id=run["round_id"], submission_id=run["submission_id"],
            icp_position=int(run["icp_position"]),
            qualified_company_count=len(qualified),
        )
        timestamp = observed["observed_at"]
        moment = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        if moment.tzinfo is None:
            raise ValueError("cost snapshot has no timezone")
        snapshot = observed["eligibility"]
        if not isinstance(snapshot, Mapping):
            raise ValueError("cost snapshot is not a mapping")
        execution = snapshot["execution"]
        if not isinstance(execution, Mapping):
            raise ValueError("execution cost state is not a mapping")
        successful = execution["successful_microusd"]
        split = ("competition_successful_microusd",
                 "excluded_infrastructure_successful_microusd")
        if any(key in execution for key in split):
            # Migration 478 preserves raw spend and excludes proved failed
            # infrastructure attempts only from competition spend. Both parts
            # must account for the unchanged raw successful total.
            if (any(type(execution.get(key)) is not int or execution[key] < 0 for key in split)
                    or successful != sum(execution[key] for key in split)):
                raise ValueError("execution cost split is invalid")
            successful = execution["competition_successful_microusd"]
        confirmed = execution.get("settled_microusd")
        if confirmed_full_thinking_cost:
            if type(confirmed) is not int or confirmed < 0:
                raise ValueError("confirmed cost is invalid")
            settled_split = ("competition_settled_microusd",
                             "excluded_infrastructure_settled_microusd")
            if any(key in execution for key in settled_split):
                if (any(type(execution.get(key)) is not int or execution[key] < 0
                        for key in settled_split)
                        or confirmed != sum(execution[key] for key in settled_split)):
                    raise ValueError("confirmed competition cost split is invalid")
                confirmed = execution["competition_settled_microusd"]
        numbers = (
            "competition_sourcing_microusd", "execution_icp_cap_microusd",
            "cost_per_company_cap_microusd", "eligibility_cap_microusd",
        )
        flat_policy = policy == contracts.FLAT_PER_ICP_SUCCESSFUL_CALLS_COST_POLICY
        expected_cap = (configuration.get("execution_icp_cap_microusd") if flat_policy
                        else configuration.get("cost_per_company_microusd", 0) * len(qualified))
        if (type(snapshot.get("icp_position")) is not int
                or snapshot["icp_position"] != run["icp_position"]
                or any(type(snapshot.get(key)) is not int or snapshot[key] < 0
                       for key in numbers)
                or any(type(execution.get(key)) is not int or execution[key] < 0
                       for key in ("successful_microusd", "success_unresolved_microusd",
                                   "inflight_calls", "success_unresolved_calls"))
                or snapshot["competition_sourcing_microusd"] != (
                    confirmed if confirmed_full_thinking_cost else
                    successful + execution["success_unresolved_microusd"])
                or snapshot["execution_icp_cap_microusd"] != configuration.get("execution_icp_cap_microusd")
                or snapshot["cost_per_company_cap_microusd"] != configuration.get("cost_per_company_microusd")
                or snapshot["eligibility_cap_microusd"] != (
                    expected_cap)):
            raise ValueError("cost snapshot does not match frozen policy")
        reason = snapshot.get("eligibility_reason")
        if reason in ("provider_calls_inflight", "provider_cost_uncertain"):
            return {**unavailable(reason), "observed_at": timestamp,
                    "source_rpc": "lab_arena_icp_cost_eligibility"}
        exceeded_reason = "execution_cap_exceeded" if flat_policy else "cost_per_company_exceeded"
        if (reason not in ("eligible", exceeded_reason)
                or type(snapshot.get("eligible")) is not bool
                or snapshot["eligible"] != (reason == "eligible")
                or snapshot["eligible"] != (
                    snapshot["competition_sourcing_microusd"]
                    <= snapshot["eligibility_cap_microusd"])
                or execution["inflight_calls"]
                or (execution["success_unresolved_calls"] and not confirmed_full_thinking_cost)):
            raise ValueError("cost snapshot decision is not final")
        return {
            "status": "available", "scope": "export_time_per_icp_sourcing",
            "source_rpc": "lab_arena_icp_cost_eligibility",
            "observed_at": timestamp, "policy": policy,
            "icp_position": run["icp_position"],
            "qualified_company_count": len(qualified),
            "competition_sourcing_microusd": snapshot["competition_sourcing_microusd"],
            "execution_icp_cap_microusd": snapshot["execution_icp_cap_microusd"],
            "cost_per_company_cap_microusd": snapshot["cost_per_company_cap_microusd"],
            "eligibility_cap_microusd": snapshot["eligibility_cap_microusd"],
            "eligible": snapshot["eligible"], "eligibility_reason": reason,
        }
    except Exception:
        # Never put an RPC exception, URL, or service credential in an export.
        return unavailable("cost_authority_unavailable")


def assemble_example(*, run_id: str, round_row: Mapping[str, Any],
                     submission_row: Mapping[str, Any], runs: Sequence[Mapping[str, Any]],
                     ledger_rows: Sequence[Mapping[str, Any]], objects: Any,
                     cache_rows: Sequence[Mapping[str, Any]] = (),
                     company_judgment_rows: Sequence[Mapping[str, Any]] = (),
                     signal_judgment_uses: Sequence[Mapping[str, Any]] = (),
                     signal_judgment_rows: Sequence[Mapping[str, Any]] = (),
                     signal_source_ledger_rows: Sequence[Mapping[str, Any]] = (),
                     input_snapshots: Mapping[str, Mapping[str, Any]] | None = None,
                     scoring_policy_history: Mapping[str, Any] | None = None,
                     cost_eligibility_reader: Callable[..., Mapping[str, Any]] | None = None) -> dict:
    """Assemble one exact execution. Missing evidence produces explicit gaps.

    Invalid identity/ambiguous joins raise ExportError. ``input_snapshots`` must
    contain verified persisted payloads, never a newly reconstructed input.
    All other inputs are plain immutable rows or a read-only object interface.
    """
    by_id = {row["run_id"]: row for row in runs}
    if len(by_id) != len(runs) or run_id not in by_id:
        raise ExportError("ambiguous_or_missing_run")
    run = by_id[run_id]
    if run.get("kind") != "execute":
        raise ExportError("execution_run_required")
    if (run.get("round_id") != round_row.get("round_id")
            or run.get("submission_id") != submission_row.get("submission_id")
            or run.get("miner_hotkey") != submission_row.get("miner_hotkey")
            or submission_row.get("round_id") != round_row.get("round_id")):
        raise ExportError("context_membership_mismatch")
    config = round_row.get("configuration_doc") or {}
    policy_versions = _scoring_policy_versions(round_row, scoring_policy_history)
    generation_floor = round_row.get("scoring_generation_floor", 0)
    reasons = {name: set() for name in ("context", "source_identity", "input_output_label", "exact_inputs", "costs", "research_process")}
    example = {"schema_version": SCHEMA_VERSION,
               "example_id": contracts.document_hash({"schema_version": SCHEMA_VERSION, "run_id": run_id}),
               **{key: run[key] for key in ("run_id", "round_id", "submission_id", "stage", "icp_position", "assignment_id", "attempt")},
               "round_status": round_row.get("status"), "rewards_enabled": round_row.get("rewards_enabled"),
               "arena_network_name": round_row.get("arena_network_name"),
               "configuration": copy.deepcopy(config), "evaluation_date": round_row.get("evaluation_date"),
               "role": "baseline" if submission_row.get("is_king") is True else "miner",
               "runtime": _runtime(run), "companies": []}
    from lab_arena import baseline_reasoning
    if (baseline_reasoning.FIELD in config
            and submission_row.get("is_king") is True):
        example[baseline_reasoning.EXPORT_FIELD] = baseline_reasoning.for_run(round_row, run)
    if generation_floor:
        example["scoring_generation_floor"] = generation_floor
        example["scoring_policy_history"] = copy.deepcopy(scoring_policy_history)
    icp = output = None
    try:
        benchmark = _json_object(objects, round_row["benchmark_ref"], 4 * 1024 * 1024)
        if (benchmark.get("schema_version") != "leadpoet.lab_arena.benchmark.v1"
                or benchmark.get("round_id") != run["round_id"]
                or len(benchmark["icps"]) != contracts.benchmark_icp_count(config)):
            raise ExportError("benchmark_invalid")
        icp = benchmark["icps"][int(run["icp_position"])]
        example["icp"] = icp
    except Exception:
        reasons["context"].add("frozen_icp_unavailable")
    participants = [row for row in round_row.get("participants") or [] if row.get("submission_id") == run["submission_id"]]
    if len(participants) != 1:
        raise ExportError("ambiguous_frozen_participant")
    participant = participants[0]
    try:
        example["source"] = _source_evidence(objects, run, round_row, submission_row, participant)
    except Exception:
        reasons["source_identity"].add("source_unverified")
    resource = (run.get("result_doc") or {}).get("resource_summary") or {}
    if not _verified_runtime_source(resource):
        reasons["source_identity"].add("runtime_source_unverified")
    if not config.get("scorer_image_digest") or not config.get("scorer_image_reference"):
        reasons["source_identity"].add("scorer_image_identity_missing")
    try:
        expected_output_ref = "arena/%s/outputs/%s.json" % (run["round_id"], run_id)
        if run.get("output_ref") != expected_output_ref:
            raise ExportError("execution_output_reference_mismatch")
        stored_output = _json_object(objects, expected_output_ref, MAX_OUTPUT_BYTES)
        # Production replay authority hashes the parsed stored document, not
        # literal JSON whitespace or a newly normalized validation projection.
        output_hash = contracts.document_hash(stored_output)
        recorded_hash = run.get("output_hash")
        if recorded_hash not in (None, "") and (
                not isinstance(recorded_hash, str) or contracts.SHA256_RE.fullmatch(recorded_hash) is None
                or recorded_hash != output_hash):
            raise ExportError("execution_output_hash_mismatch")
        output = validate_output_document(stored_output,
                                          expected_schema_version=contact_policy.output_schema(config),
                                          require_intent_dates=False)
        if run.get("status") != "accepted" or run.get("terminal_cause") != "accepted":
            raise ExportError("execution_not_accepted")
        example["output"] = output
        example["output_identity"] = {
            "ref": expected_output_ref, "document_hash": output_hash,
            "normalized_document_hash": contracts.document_hash(output),
            "recorded_hash_present": recorded_hash not in (None, ""),
        }
    except ExportError as exc:
        output = None
        reasons["input_output_label"].add(str(exc))
    except Exception:
        output = None
        reasons["input_output_label"].add("accepted_output_unavailable")
    scored = None
    judge = _select_judge(run, runs, scoring_generation_floor=generation_floor)
    if judge is None or judge.get("status") != "accepted" or judge.get("terminal_cause") != "accepted":
        reasons["input_output_label"].add("final_judge_unavailable")
    elif output is not None and icp is not None:
        try:
            policy = config["scorer_policy"]
            sliced = verify.slice_first_n(output["companies"], verify.icp_company_goal(icp))
            reader = _evidence_reader(_Rows(
                by_id, submission_row, cache_rows, company_judgment_rows,
                round_row=round_row, signal_uses=signal_judgment_uses,
                signal_rows=signal_judgment_rows,
            ), objects)
            breakdowns = reader._verified_breakdowns(judge, icp=icp, companies=sliced, policy=policy)
            result = verify.scored_row(run["submission_id"], run["icp_position"], run_id, icp, output["companies"], breakdowns, policy)
            if run.get("per_icp_score") is None or abs(Decimal(str(result["per_icp_score"])) - Decimal(str(run["per_icp_score"]))) > Decimal("0.000001"):
                raise ExportError("score_mismatch")
            example["score_run_id"] = judge["run_id"]
            example["judgment"] = {"breakdowns": breakdowns, "per_icp_score": result["per_icp_score"],
                                   "runtime": _runtime(judge),
                                   "judgment_cache_source_run_id": judge.get("judgment_cache_source_run_id"),
                                   "company_judgment_refs": judge.get("company_judgment_refs")}
            if judge.get("company_judgment_refs") is not None:
                needed = {(row["cache_key"], row["authority_slot"]) for row in
                          ((judge.get("claim_response") or {}).get("company_judgment_cache") or {}).get("hits", [])}
                needed.update((row["cache_key"], row["authority_slot"]) for row in
                              ((judge.get("claim_response") or {}).get("company_judgment_cache") or {}).get("misses", []))
                example["judgment"]["company_authority_evidence"] = [copy.deepcopy(row) for row in company_judgment_rows
                                                                       if (row["cache_key"], row["authority_slot"]) in needed]
            if judge.get("judgment_cache_key"):
                example["judgment"]["cache_authority_evidence"] = [copy.deepcopy(row) for row in cache_rows
                                                                   if row["cache_key"] == judge["judgment_cache_key"]]
            indexed = dict(zip(result["scored_company_indexes"], breakdowns))
            for index, company in enumerate(output["companies"]):
                state = "scored" if index in indexed else "skipped" if index in result["skipped_company_indexes"] else "outside_evaluated_slice"
                example["companies"].append({"company_index": index, "company": company, "label_status": state,
                                              "judgment": indexed.get(index)})
            scored = result
        except Exception:
            reasons["input_output_label"].add("judge_evidence_unverified")
    related = [row for row in runs if row.get("assignment_id") == run["assignment_id"]
               or row.get("kind") == "score" and row.get("scored_run_id") == run_id]
    # This is a deterministic grouping, not a claim that parallel runs or calls
    # occurred serially. Preserve original ledger timestamps in each lifecycle.
    related.sort(key=lambda row: (row.get("kind") != "execute", int(row.get("stage_generation") or 0),
                                  int(row.get("attempt") or 0), row["run_id"]))
    run_ids = {row["run_id"] for row in related}
    example["attempts"] = [{"run_id": row["run_id"], "kind": row["kind"], "attempt": row.get("attempt"),
                             "lease_generation": row.get("lease_generation"), "status": row.get("status"),
                             "terminal_cause": row.get("terminal_cause"), "runtime": _runtime(row)}
                            for row in related]
    for row in ledger_rows:
        if row.get("run_id") in run_ids and any(row.get(key) != run.get(key) for key in ("round_id", "submission_id", "miner_hotkey")):
            raise ExportError("ledger_context_mismatch")
    calls, gaps = _calls([row for row in ledger_rows if row.get("run_id") in run_ids], run_ids, objects,
                         {row["run_id"]: index for index, row in enumerate(related)})
    (observations, partial_observations, observation_gaps, observed_actions,
     worker_gaps_by_run, verified_worker_counts, unverified_observations) = _worker_evidence(
        related, calls, objects, selected_run_id=run_id, configuration=config, ledger_rows=ledger_rows)
    # Recompute payload gaps after exact worker-side fallback joins. Structural
    # ledger failures and every accounting gap retain their original authority.
    gaps -= {"full_request_unavailable", "full_response_unavailable", "request_privacy_nonexact", "response_privacy_nonexact", "provider_call_inflight", "provider_cost_unresolved"}
    gaps.update(gap for call in calls for gap in call["evidence_gaps"])
    related_gaps = {row["run_id"]: {"research_process": set(), "costs": set()}
                    for row in related}
    for source_id, source_gaps in worker_gaps_by_run.items():
        related_gaps[source_id]["research_process"].update(source_gaps)
        related_gaps[source_id]["costs"].update(source_gaps & {"worker_call_accounting_unverified"})
    for call in calls:
        source_gaps = set(call["evidence_gaps"])
        related_gaps[call["run_id"]]["research_process"].update(source_gaps)
        related_gaps[call["run_id"]]["costs"].update(source_gaps & {
            "provider_call_inflight", "provider_cost_unresolved", "ambiguous_call_lifecycle",
            "ledger_call_identity_missing"})
    example["provider_calls"] = calls
    reasons["research_process"].update(gaps | observation_gaps)
    reasons["costs"].update(observation_gaps & {"worker_call_accounting_unverified"})
    reasons["costs"].update(gaps & {"provider_call_inflight", "provider_cost_unresolved", "ambiguous_call_lifecycle", "ledger_call_identity_missing"})
    example["costs"] = {"confirmed_actual_microusd": sum(call.get("actual_microusd") or 0 for call in calls),
                         "competition_cost": "not_recomputed", "all_related_attempts_included": True}
    example["competition_cost_eligibility"] = _competition_cost_snapshot(
        run=run, configuration=config, scored=scored, related=related,
        reader=cost_eligibility_reader,
    )
    if (config.get("sourcing_cost_eligibility_policy")
            in contracts.PER_ICP_SUCCESSFUL_CALLS_COST_POLICIES
            and example["competition_cost_eligibility"]["status"] != "available"):
        reasons["costs"].add("competition_cost_eligibility_unavailable")
    snapshots = dict(input_snapshots or {})
    selected_ids = {run_id} | ({judge["run_id"]} if judge else set())
    launched_ids = set()
    for row in related:
        result = row.get("result_doc") or {}
        resource = result.get("resource_summary") or {}
        if (result.get("started_at") or row.get("output_ref") or resource.get("provider_call_count", 0)
                or resource.get("wall_seconds", 0)):
            launched_ids.add(row["run_id"])
    snapshot_ids = selected_ids | launched_ids
    attempt_by_id = {row["run_id"]: row for row in example["attempts"]}
    for selected_id in sorted(snapshot_ids):
        attempt_policy = config.get("scorer_policy")
        if by_id[selected_id]["kind"] == "score":
            generation = int(by_id[selected_id].get("stage_generation") or 0)
            if generation < 0:
                raise ExportError("scoring_generation_invalid")
            revision = _scoring_revision(by_id[selected_id])
            if revision not in policy_versions:
                raise ExportError("scoring_policy_history_invalid")
            attempt_policy = policy_versions[revision]
            if generation_floor:
                attempt_by_id[selected_id].update(stage_generation=generation,
                    scoring_generation_floor=revision,
                    scoring_policy_hash=contracts.document_hash(attempt_policy))
            try:
                identity = _reviewer_runtime_identity(by_id[selected_id], config)
                attempt_by_id[selected_id]["runtime_identity"] = identity
                if selected_id == example.get("score_run_id"):
                    example["judgment"]["runtime_identity"] = copy.deepcopy(identity)
            except (reviewer_runtime.ReviewerRuntimeError, KeyError, TypeError, ValueError):
                dimension = "source_identity" if selected_id in selected_ids else "research_process"
                reasons[dimension].add("reviewer_runtime_identity_unverified")
        selected_resource = (by_id[selected_id].get("result_doc") or {}).get("resource_summary") or {}
        if selected_id in selected_ids and not _verified_runtime_source(selected_resource):
            reasons["source_identity"].add("runtime_source_unverified")
        elif selected_id not in selected_ids and not _verified_runtime_source(selected_resource):
            reasons["research_process"].add("prior_attempt_source_unverified")
        if selected_id not in snapshots:
            try:
                from lab_arena.training_input import read_input
                selected = by_id[selected_id]
                snapshots[selected_id] = read_input(objects, selected_id, selected["kind"],
                                                     selected.get("lease_generation"), selected.get("claim_request_hash"))
            except Exception:
                if selected_id in selected_ids:
                    reasons["exact_inputs"].add("runtime_input_snapshot_missing")
                else:
                    reasons["research_process"].add("prior_attempt_input_missing")
        if selected_id in snapshots:
            snapshot = snapshots[selected_id]
            document = snapshot.get("input")
            privacy = snapshot.get("privacy")
            if (not isinstance(privacy, dict) or type(privacy.get("redacted")) is not bool
                    or not isinstance(privacy.get("redactions"), list)
                    or privacy["redacted"] != bool(privacy["redactions"])):
                reasons["exact_inputs"].add("runtime_input_snapshot_invalid")
            if (snapshot.get("privacy") or {}).get("redacted") is True:
                reasons["exact_inputs"].add("runtime_input_privacy_nonexact")
            if (snapshot.get("schema_version") != "leadpoet.lab_arena.training_input.v1"
                    or snapshot.get("run_id") != selected_id or snapshot.get("kind") != by_id[selected_id]["kind"]
                    or snapshot.get("lease_generation") != by_id[selected_id].get("lease_generation")
                    or snapshot.get("claim_request_hash") != by_id[selected_id].get("claim_request_hash")
                    or not isinstance(document, dict) or snapshot.get("input_sha256") != contracts.document_hash(document)):
                reasons["exact_inputs"].add("runtime_input_snapshot_invalid")
                del snapshots[selected_id]
            elif by_id[selected_id]["kind"] == "score" and (
                    document.get("scored_run_id") != run_id or output is None
                    or document.get("companies") != output["companies"]
                    or document.get("scorer_policy") != attempt_policy
                    or document.get("evaluation_date") != round_row.get("evaluation_date")):
                reasons["exact_inputs"].add("judge_input_context_mismatch")
            if isinstance(document, dict) and icp is not None:
                actual_icp = dict(document.get("icp") or {})
                expected_icp = integrity.agent_visible_icp(icp, contacts_required=contact_policy.enabled(config)) if integrity.enabled(config) else dict(icp)
                if not contact_policy.enabled(config):
                    expected_icp.pop("contact_policy", None)
                for key in ("output_schema_version", "company_quality_policy", "company_requirements", "intent_details_policy", "_arena_provider_observations"):
                    actual_icp.pop(key, None)
                if actual_icp != expected_icp or document.get("evaluation_date") != round_row.get("evaluation_date"):
                    reasons["exact_inputs"].add("runtime_input_context_mismatch")
    signal_policy = (config.get("scorer_policy") or {}).get("signal_judgment_policy") == signal_judgments.POLICY
    unmarked_signal_evidence = bool(signal_judgment_uses or signal_judgment_rows or signal_source_ledger_rows) or any(
        "signal_judgment_reference" in detail
        for breakdown in (example.get("judgment") or {}).get("breakdowns", [])
        for detail in _signal_details(breakdown)
        if isinstance(detail, Mapping)
    )
    if signal_policy or unmarked_signal_evidence:
        try:
            if judge is None or "judgment" not in example or judge["run_id"] not in snapshots:
                raise ExportError("signal_judge_evidence_missing")
            evidence = _signal_authority_evidence(
                judge=judge, breakdowns=example["judgment"]["breakdowns"],
                scored_company_indexes=scored["scored_company_indexes"],
                scoring_input=snapshots[judge["run_id"]]["input"],
                uses=signal_judgment_uses, authority_rows=signal_judgment_rows,
                company_rows=company_judgment_rows, runs=by_id,
                round_row=round_row,
            )
            example["judgment"]["signal_authority_evidence"] = evidence
            # Keep persisted actions as process evidence even when the final
            # company-fit failure has no signal verdict to validate against.
            example["judgment"]["signal_judgment_uses"] = copy.deepcopy(signal_judgment_uses)
            source_ids = {item["authority"]["source_score_run_id"] for item in evidence}
            external_source_ids = source_ids - run_ids
            selected_source_ledger = [row for row in signal_source_ledger_rows
                                      if row.get("run_id") in external_source_ids]
            source_calls, source_gaps = _calls(
                selected_source_ledger, external_source_ids, objects,
                {source_id: index for index, source_id in enumerate(sorted(external_source_ids))},
            )
            for source_id in external_source_ids:
                source = by_id[source_id]
                if any(row.get(key) != source.get(key) for row in selected_source_ledger
                       if row.get("run_id") == source_id
                       for key in ("round_id", "submission_id", "miner_hotkey")):
                    raise ExportError("signal_source_ledger_context_mismatch")
                expected_count = ((source.get("result_doc") or {}).get("resource_summary") or {}).get("provider_call_count")
                logical_calls = {call.get("action_sequence") for call in source_calls
                                 if call["run_id"] == source_id and type(call.get("action_sequence")) is int}
                if type(expected_count) is not int or expected_count != len(logical_calls):
                    source_gaps.add("signal_source_provider_capture_unverified")
            example["judgment"]["signal_source_provider_calls"] = source_calls
            source_call_refs = [
                {"source_score_run_id": call["run_id"],
                 "call_identity": call["call_identity"],
                 "location": "provider_calls" if call in calls else "signal_source_provider_calls"}
                for call in calls + source_calls if call["run_id"] in source_ids
            ]
            example["judgment"]["signal_source_provider_call_refs"] = source_call_refs
            example["judgment"]["signal_source_cost_attribution"] = {
                "scope": "original_source_score_runs",
                "source_score_run_ids": sorted(source_ids),
                "confirmed_actual_microusd": sum(
                    call.get("actual_microusd") or 0
                    for call in calls + source_calls if call["run_id"] in source_ids
                ),
                "included_in_execution_cost_run_ids": sorted(source_ids & run_ids),
                "external_source_run_ids": sorted(external_source_ids),
            }
            reasons["research_process"].update(source_gaps)
        except (ExportError, signal_judgments.SignalJudgmentError, KeyError, IndexError, TypeError, ValueError):
            reasons["input_output_label"].add("signal_authority_unverified")
            example.pop("judgment", None)
            example["companies"] = []
            scored = None
    example["input_snapshots"] = {key: snapshots[key] for key in sorted(snapshot_ids) if key in snapshots}
    if not selected_ids.issubset(snapshots):
        reasons["exact_inputs"].add("runtime_input_snapshot_missing")
    for row in related:
        summary = (row.get("result_doc") or {}).get("resource_summary") or {}
        web = summary.get("web_egress")
        if "web_egress" not in summary and row.get("kind") == "score" and _verified_runtime_source(summary):
            # The trusted scorer starts with no web bridge and network=none.
            pass
        elif not isinstance(web, dict) or type(web.get("connection_count")) is not int or web["connection_count"] < 0:
            reasons["research_process"].add("direct_web_usage_unverified")
            related_gaps[row["run_id"]]["research_process"].add("direct_web_usage_unverified")
        elif web["connection_count"]:
            reported_pages = (observations.get(row["run_id"], {}).get("payload", {}).get("agent_observations", []))
            web_gap = "direct_web_coverage_unverified" if any(
                item["kind"] in {"web_fetch", "web_fetch_binary"}
                for item in reported_pages) else "direct_web_content_not_captured"
            reasons["research_process"].add(web_gap)
            related_gaps[row["run_id"]]["research_process"].add(web_gap)
        logical_calls = set()
        for call in calls:
            if call["run_id"] != row["run_id"]:
                continue
            sequence = call.get("action_sequence")
            base = call.get("base_call_identity")
            if type(sequence) is int and sequence >= 0:
                logical_calls.add(("action", sequence))
            elif isinstance(base, str) and base:
                logical_calls.add(("base", base))
            else:
                reasons["research_process"].add("logical_call_identity_unverified")
                related_gaps[row["run_id"]]["research_process"].add("logical_call_identity_unverified")
                logical_calls.add(("unproved", call["call_identity"]))
        logical_calls.update(("action", sequence) for sequence in observed_actions.get(row["run_id"], set()))
        captured = len(logical_calls)
        expected_count = summary.get("provider_call_count", verified_worker_counts.get(row["run_id"]))
        if type(expected_count) is not int or expected_count != captured:
            reasons["research_process"].add("provider_capture_count_unverified")
            reasons["costs"].add("provider_capture_count_unverified")
            related_gaps[row["run_id"]]["research_process"].add("provider_capture_count_unverified")
            related_gaps[row["run_id"]]["costs"].add("provider_capture_count_unverified")
    reasons["research_process"].update(reasons["exact_inputs"])
    reasons["research_process"].update(reasons["input_output_label"])
    if reasons["context"]:
        reasons["input_output_label"].add("frozen_context_unverified")
    # Diagnostic logs and model-reported decisions are deliberately excluded.
    example["excluded"] = ["diagnostic_trajectory_excerpts", "model_reported_decision_excerpts", "hidden_reasoning"]
    from lab_arena.training_capture import sanitize_training_value
    example, redactions = sanitize_training_value(
        example, max_nodes=MAX_EXAMPLE_NODES, aggregate_parts=True)
    privacy_redactions = set(redactions)
    clean, removed = without_hidden_reasoning(example)
    clean["worker_observation_artifacts"] = observations
    if partial_observations:
        clean["partial_worker_observation_artifacts"] = partial_observations
    if unverified_observations:
        clean["unverified_worker_observation_artifacts"] = unverified_observations
    worker_artifacts = (list(observations.values()) + list(partial_observations.values())
                        + list(unverified_observations.values()))
    observation_redactions = {reason for artifact in worker_artifacts
                             for item in artifact["payload"]["observations"] + artifact["payload"].get("agent_observations", [])
                             for reason in item["privacy"]["redactions"]}
    hidden_removed = removed or "hidden_reasoning" in observation_redactions or "hidden_reasoning" in privacy_redactions or any(
        call.get("response_hidden_reasoning_removed") or "hidden_reasoning" in call.get("response_privacy_redactions", [])
        or any("hidden_reasoning" in receipt.get("privacy_redactions", []) for receipt in call.get("response_receipts", []))
        or any("hidden_reasoning" in receipt.get("privacy_redactions", []) for receipt in call.get("upstream_response_receipts", [])) for call in calls)
    clean["privacy"] = {"hidden_reasoning_removed": hidden_removed, "hidden_reasoning_excluded": True,
                        "context_redactions": sorted(privacy_redactions)}
    privacy_filtered = bool(privacy_redactions | observation_redactions) or hidden_removed or any(
        (snapshot.get("privacy") or {}).get("redacted") for snapshot in snapshots.values()) or any(
        (call.get("request_capture") or {}).get("redacted") or call.get("response_privacy_redactions")
        or any(receipt.get("privacy_redactions") for receipt in call.get("upstream_response_receipts", []))
        or any(receipt.get("privacy_redactions") for receipt in call.get("response_receipts", [])) for call in calls)
    if "credential" in privacy_redactions:
        reasons["research_process"].add("context_privacy_nonexact")
    clean["privacy"]["privacy_filtered"] = privacy_filtered
    clean["privacy"]["exact_replay"] = not privacy_filtered and not any(reasons.values())
    clean["completeness"] = {"complete": not any(reasons.values()),
                             "dimensions": {key: not value for key, value in reasons.items()},
                             "incomplete_reasons": sorted({reason for values in reasons.values() for reason in values}),
                             "reasons_by_dimension": {key: sorted(value) for key, value in reasons.items()}}
    clean["related_run_gap_attribution"] = {
        row["run_id"]: {"kind": row["kind"], "status": row["status"],
                        "research_process_gaps": sorted(related_gaps[row["run_id"]]["research_process"]),
                        "cost_gaps": sorted(related_gaps[row["run_id"]]["costs"])}
        for row in related}
    clean["example_hash"] = contracts.document_hash(clean)
    return json.loads(contracts.canonical_json(clean))
