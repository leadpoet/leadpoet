"""Verify ordinary synthetic research decisions and derive genuine pre-error checkpoints."""

from __future__ import annotations

import json
import re

from lab_arena.collection_judge import digest
from lab_arena.collection_journal import WorkflowJournal


def _memo(raw):
    if not isinstance(raw, str):
        return {}

    def unique(pairs):
        result = dict(pairs)
        if len(pairs) != len(result):
            raise ValueError("ambiguous JSON object")
        return result

    try:
        value = json.loads(raw, object_pairs_hook=unique)
    except ValueError:
        return {}
    return value if isinstance(value, dict) else {}


def _findings(memo):
    items = memo.get("findings")
    if (
        not isinstance(items, list)
        or not items
        or not all(
            isinstance(f, dict)
            and set(f) == {"field", "value", "citations"}
            and isinstance(f["field"], str)
            for f in items
        )
    ):
        return {}
    result = {f["field"]: f for f in items}
    return result if len(result) == len(items) else {}


def _check_memo(memo, reference, state, visible):
    gold, finals = _findings(reference), _findings(memo)
    if not gold:
        raise ValueError("missing pinned sales research reference findings")
    grounded = {}
    facts = {}
    citations = {}
    for name, needed in gold.items():
        field = finals.get(name, {})
        value = field.get("value")
        facts[name] = (
            name in finals
            and type(value) is type(needed["value"])
            and value == needed["value"]
        )
        expected = {c["source_id"]: c for c in needed["citations"]}
        src = field.get("citations")
        valid = isinstance(src, list) and len(src) == len(expected)
        cited = []
        for c in src if valid else ():
            if (
                not isinstance(c, dict)
                or set(c) != {"source_id", "published_on", "quote"}
                or not isinstance(c["source_id"], str)
            ):
                valid = False
                break
            key = c["source_id"]
            doc = state["documents"].get(key)
            quote = c["quote"]
            if (
                key not in expected
                or key not in visible
                or not doc
                or doc["entity"] != state["entity"]
                or doc["field"] != name
                or c["published_on"] != doc["published_on"]
                or c["published_on"] > state["as_of"]
                or not isinstance(quote, str)
                or not quote
                or quote not in doc["body"]
                or doc["fact"] not in quote
            ):
                valid = False
                break
            cited.append(key)
        citations[name] = cited
        grounded[name] = bool(valid and set(cited) == set(expected))
    valid = (
        set(memo) == {"account_id", "as_of", "decision", "findings"}
        and memo["account_id"] == state["entity"]
        and memo["as_of"] == state["as_of"]
        and memo["decision"] == reference["decision"]
        and finals.keys() == gold.keys()
        and all(facts.values())
        and all(grounded.values())
    )
    return {
        "passed": bool(valid),
        "claim_checks": facts,
        "citation_checks": grounded,
        "citation_ids": citations,
    }


def _history(objects, job, manifest, row, packet, *, classify_grounding=False):
    from lab_arena.collection_workflow import read_artifact

    if packet.get("projection_error") and packet.get("full_evidence"):
        packet = read_artifact(
            objects,
            job["batch_id"],
            job["job_id"],
            "full-judge-evidence",
            packet["full_evidence"],
            128 * 1024 * 1024,
        )
    reference = _memo(
        json.loads(packet["reference_answer"])["files"]["qualification.json"]
    )
    state = row["evidence"]["initial_state"]
    journal = WorkflowJournal(objects, job["batch_id"], job["job_id"])
    events = (json.loads(data) for data, _ in journal.read(manifest["events"]))
    before = None
    action = None
    seen = set()
    discovered = set()
    first = None
    first_mistake = None
    persistent_errors = set()
    repair = None
    queries = []
    opened = []
    last_check = _check_memo({}, reference, state, set())
    memo_seen_at = set()
    inherited_prefix = False
    for e in events:
        if e["kind"] == "start":
            tools = e["tools"]
            # A continuation inherits the exact prior transcript. Restore which articles it had already seen.
            inherited = {}
            inherited_writer = set()
            for msg in e["messages"]:
                if msg["role"] == "assistant":
                    for c in msg.get("tool_calls", []):
                        function = c["function"]
                        inherited[c["id"]] = function["name"]
                        if (
                            function["name"] == "write_file"
                            and _memo(function["arguments"]).get("path")
                            == "qualification.json"
                        ):
                            inherited_writer = set(seen)
                if msg["role"] == "tool":
                    d = _memo(msg["content"])
                    kind = inherited.get(msg["tool_call_id"])
                    if kind == "archive_search":
                        discovered.update(v["source_id"] for v in d.get("results", []))
                    if kind == "archive_open":
                        key = d.get("source_id")
                        if (
                            key in discovered
                            and key in state["documents"]
                            and d.get("body") == state["documents"][key]["body"]
                        ):
                            seen.add(key)
            inherited_prefix = bool(row.get("native_correction") or row.get("correction"))
            continue
        if e["kind"] in ("before", "finish"):
            if inherited_prefix and before is None:
                memo_seen_at = inherited_writer
                last_check = _check_memo(
                    _memo(e["files"].get("qualification.json")),
                    reference,
                    state,
                    memo_seen_at,
                )
                inherited_prefix = False
            if before is not None and action is not None:
                turn = before["turn"]
                raw = e["files"].get("qualification.json")
                wrote = any(
                    call["function"]["name"] == "write_file"
                    and _memo(call["function"]["arguments"]).get("path")
                    == "qualification.json"
                    for call in action.get("tool_calls", [])
                )
                if wrote:
                    memo_seen_at = seen_before_turn
                    memo = _memo(raw)
                    last_check = _check_memo(memo, reference, state, memo_seen_at)
                    if classify_grounding:
                        # Track errors in submitted fields of the first wrong memo.
                        # Missing/malformed memos and subsequently repaired fields are not examples.
                        errors = {
                            (field, kind)
                            for field in _findings(memo)
                            for kind in ("claim", "citation")
                            if memo.get("account_id") == state["entity"]
                            and memo.get("as_of") == state["as_of"]
                            and last_check[kind + "_checks"].get(field) is False
                        }
                        if first is None:
                            persistent_errors = errors
                        else:
                            persistent_errors.intersection_update(errors)
                    if not last_check["passed"] and first is None:
                        first = (before, action)
                        first_mistake = "unsupported_account_memo"
                    if last_check["passed"] and first is not None and repair is None:
                        repair = turn
                if (
                    e["kind"] == "finish"
                    and row["evidence"]["finish"] == "answered"
                    and (
                        not last_check["passed"]
                        or e.get("answer", "").strip() != reference["decision"]
                    )
                    and first is None
                ):
                    first = (before, action)
                    first_mistake = "incomplete_or_incorrect_final_decision"
            before = e if e["kind"] == "before" else None
            seen_before_turn = set(seen)
            action = None
        elif e["kind"] == "reply":
            action = e["message"]
        elif e["kind"] == "tool":
            if e["name"] == "archive_search":
                d = _memo(e["output"])
                found = [record["source_id"] for record in d.get("results", [])]
                discovered.update(found)
                call = next(
                    (
                        c
                        for c in (action or {}).get("tool_calls", [])
                        if c["id"] == e["call_id"]
                    ),
                    None,
                )
                if not call:
                    raise ValueError("search has no matching original action")
                queries.append(
                    {
                        "turn": before["turn"],
                        "query": _memo(call["function"]["arguments"]).get("query"),
                        "returned": found,
                    }
                )
            if e["name"] == "archive_open":
                d = _memo(e["output"])
                key = d.get("source_id")
                if key is not None:
                    doc = state["documents"].get(key)
                    if (
                        key not in discovered
                        or not doc
                        or doc["body"] != d.get("body")
                        or any(
                            doc[k] != d.get(k)
                            for k in ("entity", "published_on", "source_type", "issuer")
                        )
                    ):
                        raise ValueError(
                            "opened sales evidence differs from the original source"
                        )
                    seen.add(key)
                    opened.append(
                        {
                            "turn": before["turn"],
                            "source_id": key,
                            "body_sha256": digest(doc["body"]),
                        }
                    )
    empty = {
        "passed": False,
        "claim_checks": last_check["claim_checks"],
        "citation_checks": last_check["citation_checks"],
        "citation_ids": last_check["citation_ids"],
    }
    # Only new funded continuations opt into reusing verified, previously opened evidence.
    # Historical manifests must replay to their original native decisions.
    inherited_research = (
        row.get("correction") and manifest.get("uses_checkpoint_evidence") is True
    )
    if not queries and not (row.get("native_correction") or inherited_research):
        last_check = empty
    details = {
        "verified_decision": bool(
            last_check["passed"]
            and row["evidence"]["grade"]["score"] == 1
            and not row["evidence"]["grade"].get("hard_fail")
            and row["evidence"]["out"]["answer"].strip() == reference["decision"]
        ),
        "field_evidence": last_check,
        "searches": queries,
        "opened_sources": opened,
        "first_failed_decision": None,
        "observed_repair": None,
    }
    pre = None
    if first is not None:
        entry, rejected = first
        prefix = entry["prefix_messages"]
        cp = {
            "messages": row["messages"][:prefix],
            "files": entry["files"],
            "state": entry["state"],
            "log": entry["log"],
            "tools": tools,
        }
        if (
            len(row["messages"]) <= prefix
            or row["messages"][prefix]["role"] != "assistant"
        ):
            raise ValueError("failed action has no prior transcript boundary")
        # The journal may retain a null provider content which the model-visible transcript represents as empty.
        rejected = row["messages"][prefix]
        anchor = {
            "turn": entry["turn"],
            "event": entry["index"],
            "checkpoint_sha256": digest(cp),
            "rejected_sha256": digest(rejected),
            "reason": first_mistake,
        }
        details["first_failed_decision"] = anchor
        if repair is not None:
            details["observed_repair"] = {
                "rejected_turn": entry["turn"],
                "repair_turn": repair,
                "kind": "observed_revision",
            }
        pre = {"checkpoint": cp, "rejected": rejected, **anchor}
        if (
            persistent_errors
            and first_mistake == "unsupported_account_memo"
            and repair is None
            and not details["verified_decision"]
        ):
            pre["unrepaired_grounding"] = {
                name: [
                    k for k in ("claim", "citation") if (name, k) in persistent_errors
                ]
                for name in sorted({name for name, _ in persistent_errors})
            }
    return details, pre


def sales_correction_source(objects, job):
    """Reread an immutable, originally owner-funded model mistake. Never turn an example into a failure."""
    from lab_arena.collection_worker import native_payment_evidence, verification_matches
    from lab_arena.collection_workflow import SALES_EVIDENCE_RULE, read_artifact

    if (
        not isinstance(job, dict)
        or job.get("mode") != "owner"
        or not re.fullmatch("owner-p8-[a-f0-9]{64}", job["batch_id"])
    ):
        raise ValueError("expected a separately funded original owner case")
    b, r = job["batch_id"], job["job_id"]
    m = read_artifact(
        objects, b, r, "manifest", {"sha256": job["artifact_sha256"]}, 1024 * 1024
    )
    if (
        m.get("funding_mode") != "owner"
        or m.get("payment_rule") != SALES_EVIDENCE_RULE
        or m.get("correction_source")
    ):
        raise ValueError("expected an original owner sales assignment")
    checked = native_payment_evidence(objects, job, m)
    recorded = read_artifact(objects, b, r, "native-verification", m["verification"], 4 * 1024 * 1024)
    if not verification_matches(recorded, checked):
        raise ValueError("original owner verification differs")
    if not checked["first_failed_decision"]:
        raise ValueError("no observed failed decision to correct")
    row = read_artifact(objects, b, r, "generator", m["generator"], 128 * 1024 * 1024)
    packet = read_artifact(objects, b, r, "judge-input", m["packet"], 16 * 1024 * 1024)
    _, source = _history(objects, job, m, row, packet)
    if not source or not checked["calls"] or checked["cost_microusd"] <= 0:
        raise ValueError("no verified real model error")
    return {
        **source,
        "source_final_passed": checked["verified_decision"],
        "scenario_id": job["scenario_id"],
        "student_model": job["model"],
        "source_job_id": r,
        "source_batch_id": b,
        "source_sha256": job["artifact_sha256"],
    }


def verify_account_memo(objects, job, manifest, row, packet):
    detail, _ = _history(objects, job, manifest, row, packet)
    if manifest.get("correction_source"):
        source = sales_correction_source(objects, manifest["correction_source"])
        meta = row.get("native_correction") or {}
        cut = len(source["checkpoint"]["messages"])
        if (
            job.get("mode") != "owner"
            or source["student_model"] == job["model"]
            or meta.get("prefix_messages") != cut
            or meta.get("checkpoint_sha256") != source["checkpoint_sha256"]
            or row["messages"][:cut] != source["checkpoint"]["messages"]
            or meta.get("source_sha256") != source["source_sha256"]
        ):
            raise ValueError("correction does not share the genuine source checkpoint")
        journal = WorkflowJournal(objects, job["batch_id"], job["job_id"])
        it = journal.read(manifest["events"])
        start = json.loads(next(it)[0])
        before = json.loads(next(it)[0])
        cp = source["checkpoint"]
        if (
            start["tools"] != cp["tools"]
            or before["kind"] != "before"
            or any(before[k] != cp[k] for k in ("files", "state", "log"))
        ):
            raise ValueError("correction started from a different world")

        def decision(m):
            calls = [
                (c["function"]["name"], _memo(c["function"]["arguments"]))
                for c in m.get("tool_calls", [])
            ]
            return m.get("content") or "", calls

        detail["original_final_passed"] = source["source_final_passed"]
        detail["changed_first_action"] = len(row["messages"]) > cut and decision(
            row["messages"][cut]
        ) != decision(source["rejected"])
    return detail
