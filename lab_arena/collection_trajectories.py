"""Release model-visible workflow evidence. Keep private worlds and reference answers out."""

from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
import json
import math
import re
from uuid import UUID

from lab_arena import collection_judge as judge, contracts
from lab_arena.collection_checkpoint import require_private_evidence

SCHEMA = "collection.trajectories.v1"
LABEL = re.compile(r"[A-Za-z0-9_ .:-]{1,128}")


def verified_grants(database, batch, jobs, *, mode, scope, digest, allow_pending=False, subset=False):
    """Use the original signed registration of each actual job, including retries."""
    from lab_arena import collection_intake as intake
    from lab_arena.chain import verify_hotkey_signature

    from lab_arena.collection_pool import POOL_SCOPE, POOL_TERMS_HASH, POOL_TERMS_BY_RATE, job_reward_rate
    pool = mode == "live" and scope == POOL_SCOPE and re.fullmatch(r"pool-[0-9a-f]{64}", batch)
    if scope == POOL_SCOPE and not pool:
        raise ValueError("standing release requires a live pool experiment")
    permitted = {POOL_TERMS_HASH} if pool else {intake.SYNTHETIC_TRAINING_TERMS_HASH}
    if mode == "live" and scope == intake.TASK_SCOPE and intake.PAID_OFFER_ID.fullmatch(batch):
        permitted = {intake.PAID_WORKFLOW_TERMS_HASH}
    if scope == intake.CORRECTION_SCOPE:
        permitted.add(intake.LEGACY_CORRECTION_TERMS_HASHES[mode])
    if digest not in permitted:
        raise ValueError("this enrollment did not grant training release rights")
    intake.register_signatures()
    rows = database.rpc("collection_training_consents_v1", {"p_batch": batch})
    if not isinstance(rows, list):
        raise ValueError("invalid job consent readback")
    if subset:
        wanted = {job["job_id"] for job in jobs}
        rows = [row for row in rows if row["job_id"] in wanted]
    by_id = {row["job_id"]: row for row in rows}
    if (
        len(by_id) != len(rows)
        or len(by_id) != len(jobs)
        or set(by_id) != {j["job_id"] for j in jobs}
    ):
        raise ValueError("missing or duplicate job funding consent")
    from lab_arena.collection_intake import SCOPE, scope_id

    for j in jobs:
        auth = by_id[j["job_id"]]
        envelope = auth["consent"]
        # This re-verifies a previously admitted signature. Freshness was checked
        # on enrollment; the original signature must stay usable years later.
        signed = contracts.validate_signed_request(
            envelope,
            expected_scope=SCOPE["register"],
            expected_round_id=scope_id(mode),
            now=envelope["timestamp"],
            verify_signature=verify_hotkey_signature,
        )
        body = signed["body"]
        terms_hash = body.get("terms_sha256")
        if pool:
            cap = intake.standing_limit(body)
            # A pre-existing grant can receive a prospective increase. A job
            # must never be assigned below the miner's signed standing terms.
            rate = job_reward_rate(j)
            if terms_hash != POOL_TERMS_HASH and terms_hash != POOL_TERMS_BY_RATE[rate][1]:
                raise ValueError("job rate is below its signed standing terms")
            valid = (type(auth.get("spend_limit_microusd")) is int and cap == auth["spend_limit_microusd"]
                     and type(auth.get("priority")) is int and body["priority"] == auth["priority"])
        else:
            valid = (intake.signed_daily_slots(body) == auth.get("max_slots_per_utc_day", 1)
                     and type(body.get("daily_cap_microusd")) is int and body["daily_cap_microusd"] > 0)
        if not valid:
            raise ValueError("invalid original enrollment body")
        if (
            auth["mode"] != mode
            or auth["miner_hotkey"] != j["miner_hotkey"]
            or signed["hotkey"] != j["miner_hotkey"]
            or str(UUID(hex=signed["request_id"])) != auth["registration_id"]
            or signed["body"].get("task_scope") != scope
            or (not pool and terms_hash != digest)
            or (digest == intake.PAID_WORKFLOW_TERMS_HASH and signed["body"].get("offer_id") != batch)
        ):
            raise ValueError("signed release consent differs from funded job")
        for k in ("unresolved", "generator_calls", "miner_cost_microusd"):
            if type(auth[k]) is not int or auth[k] < 0 or (k in j and j[k] != auth[k]):
                raise ValueError("funding evidence differs from original job")
        if auth["unresolved"] and not allow_pending:
            raise ValueError("unconfirmed miner charges")
    return by_id


def _native(grade):
    """Only release scoring fields, never graders' reference files or private notes."""
    if not isinstance(grade, dict):
        return None
    score = grade.get("score")
    if score is not None and (
        type(score) not in (int, float) or not math.isfinite(score)
    ):
        raise ValueError("invalid native score")
    fail = grade.get("hard_fail")
    if fail and not (isinstance(fail, str) and LABEL.fullmatch(fail)):
        fail = "native_hard_failure"
    credits = {
        k: v
        for k, v in (grade.get("credits") or {}).items()
        if isinstance(k, str)
        and LABEL.fullmatch(k)
        and type(v) in (int, float)
        and math.isfinite(v)
    }
    return {"score": score, "hard_fail": fail or None, "credits": credits}


def _steps(evidence, messages):
    steps = evidence.get("step_checks")
    if steps is None:  # Old checkpointed runs already recorded both boundaries.
        states = [
            e for e in evidence.get("events", []) if e["kind"] in ("before", "finish")
        ]
        steps = [
            {"turn": a["turn"], "before": a["checks"], "after": b["checks"]}
            for a, b in zip(states, states[1:])
            if a["kind"] == "before"
        ]
    indexes = [i for i, m in enumerate(messages) if m["role"] == "assistant"]
    by_turn = {s["turn"]: s for s in steps}
    if len(by_turn) != len(steps) or any(
        type(k) is not int or k < 0 or k >= len(indexes) for k in by_turn
    ):
        raise ValueError("step check does not identify an assistant decision")
    return [
        {
            "turn": i,
            "message_index": at,
            "before": _native(by_turn.get(i, {}).get("before")),
            "after": _native(by_turn.get(i, {}).get("after")),
        }
        for i, at in enumerate(indexes)
    ]


def experiment_calls(calls, events):
    """Recover omitted usage details from the journal entry for the same provider generation."""
    replies = {e["receipt"]: e for e in events if e.get("kind") == "reply" and e.get("receipt")}
    result = []
    for call in calls:
        usage = dict(call.get("usage") or {})
        original = replies.get(call.get("provider_id"))
        if original:
            if original["model"] != call["model"]:
                raise ValueError("original usage model mismatch")
            for key, value in (original.get("usage") or {}).items():
                if key not in ("prompt_tokens", "completion_tokens", "total_tokens", "completion_tokens_details"):
                    continue
                if key == "completion_tokens_details":
                    # The experiment meter deliberately retains only numeric reasoning tokens.
                    detail = value.get("reasoning_tokens") if isinstance(value, dict) else None
                    if type(detail) is not int or detail < 0:
                        continue
                    recorded = (usage.get(key) or {}).get("reasoning_tokens")
                    if recorded is not None and recorded != detail:
                        raise ValueError("original generation usage mismatch")
                    usage[key] = {"reasoning_tokens": detail}
                else:
                    if key in usage and usage[key] != value:
                        raise ValueError("original generation usage mismatch")
                    usage[key] = value
        result.append({**call, "usage": usage})
    return result


def _total_tokens(usage):
    total = usage.get("total_tokens")
    if type(total) is int and total >= 0:
        return total
    prompt, completion = usage.get("prompt_tokens"), usage.get("completion_tokens")
    if all(type(v) is int and v >= 0 for v in (prompt, completion)):
        return prompt + completion
    return None


def _tokens(calls, successful_calls):
    result = {}
    expected = max(len(calls), successful_calls)
    for key in (
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "reasoning_tokens",
    ):
        values = []
        for call in calls:
            usage = call.get("usage") or {}
            value = (
                (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
                if key == "reasoning_tokens"
                else _total_tokens(usage) if key == "total_tokens" else usage.get(key)
            )
            if type(value) is int and value >= 0:
                values.append(value)
        unknown = expected - len(values)
        result[key] = sum(values) if expected and not unknown else None
        result["known_" + key] = sum(values)
        result["unknown_" + key + "_calls"] = unknown
    return result


def trajectory_row(
    *,
    job,
    generator,
    packet,
    calls,
    settings_sha256,
    harness,
    builder,
    revision,
    route,
    split,
    terms_sha256,
    assessment=None,
    experiment=None,
    source=None,
    partial=False,
    hold=None,
):
    """Project a verified artifact or privacy-cleared journal. Missing labels stay null."""
    rid = job["job_id"]
    if generator is not None and (
        generator.get("run") != rid or generator.get("teacher") != job["model"]
    ):
        raise ValueError("generator identity differs from funded job")
    if generator is not None and (
        generator.get("task", job["scenario_id"]) != job["scenario_id"]
        or not isinstance(generator.get("messages"), list)
        or not isinstance(generator.get("tools"), list)
    ):
        raise ValueError("malformed or wrongly assigned generator transcript")
    if packet is not None and (
        packet["checks"].get("privacy") is not True
        or packet["checks"].get("training_split") is not True
    ):
        raise ValueError("unverified privacy or training split")
    if packet is not None and (
        packet.get("run_id", rid) != rid
        or packet.get("scenario_id", job["scenario_id"]) != job["scenario_id"]
    ):
        raise ValueError("grading packet does not belong to the same run")
    row = generator or {}
    evidence = row.get("evidence") or {}
    messages = row.get("messages", [])
    tools = row.get("tools", [])
    for message in messages:
        if message.get("role") == "assistant":
            require_private_evidence({"choices": [{"message": message}]})
        else:
            require_private_evidence(message)
    checks = dict(packet["checks"]) if packet else {}
    if any(type(v) is not bool for v in checks.values()):
        raise ValueError("invalid task checks")
    generation = evidence.get("generation") or {}
    finish = generation.get("finish_reason") or evidence.get("finish")
    final = bool(
        messages
        and messages[-1].get("role") == "assistant"
        and not messages[-1].get("tool_calls")
        and (messages[-1].get("content") or "").strip()
    )
    if generation.get("visible_training_safe") is False:
        raise ValueError("private captured content is outside the training partition")
    answered = checks.get("answered")
    final = (
        final
        and not partial
        and (
            generation.get("final_turn_present") is True
            if "final_turn_present" in generation
            else answered is True or finish == "answered"
        )
    )
    timed_out = answered is False or finish in (
        "max_steps",
        "max_tokens",
        "length",
        "unanswered",
        "token_cap",
        "run_token_cap",
        "output_limit",
        "interrupted",
    )
    native = _native(evidence.get("grade"))
    steps = _steps(evidence, messages)
    failure = next(
        (
            {
                "turn": s["turn"],
                "check": s["after"]["hard_fail"],
                "kind": "native_hard_failure",
            }
            for s in steps
            if s["after"] and s["after"]["hard_fail"]
        ),
        None,
    )
    if failure is None and final and answered is True:
        failed = next(
            (
                k
                for k in ("no_hard_failure", "code_grade", "shape")
                if checks.get(k) is False
            ),
            None,
        )
        if failed:
            # The terminal check proves this outcome, not which earlier action caused it.
            failure = {"turn": None, "check": failed, "kind": "terminal_check"}
    verdict = judge.claim_verdict(packet) if packet else None
    status = assessment.get("status") if assessment else None
    if not status and packet and packet.get("projection_error"):
        status = "cannot_judge"
    passed = None
    if not partial and answered is True and final:
        if (
            any(
                checks.get(k) is False
                for k in ("code_grade", "shape", "no_hard_failure")
            )
            or verdict == "failed"
        ):
            passed = False
        elif all(
            checks.get(k) is True for k in ("code_grade", "shape", "no_hard_failure")
        ) and verdict in ("passed", "not_applicable"):
            passed = True
    group = {
        "case": job["scenario_id"],
        "harness": harness,
        "builder": builder,
        "split": split,
        "checkpoint": (
            source.get("checkpoint_sha256")
            if source and experiment and experiment.get("role") == "continuation"
            else None
        ),
    }
    if experiment and job["batch_id"].startswith("pool-"):
        # Contrast repeats of the same frozen arm. Never confound its outcome with a
        # different model, checkpoint, reasoning setting or budget.
        group.update(settings=settings_sha256, arm=experiment["arm"], model=revision, route=route)
    final_context = _total_tokens(calls[-1].get("usage") or {}) if (
        final and calls and len(calls) == job["generator_calls"]
    ) else None
    tokens = _tokens(calls, job["generator_calls"])
    ins = {
        "schema": "collection.trajectory.v1",
        "run_id": rid,
        "batch_id": job["batch_id"],
        "scenario_id": job["scenario_id"],
        "slot": job["slot"],
        "attempt": job.get("attempt", 1),
        "job_status": job["status"],
        "model": job["model"],
        "model_revision": revision,
        "provider_route": route,
        "harness_commit": harness,
        "builder_commit": builder,
        "settings_sha256": settings_sha256,
        "manifest_sha256": job.get("artifact_sha256"),
        "terms_sha256": terms_sha256,
        "split": split,
        "repeat_group_id": "sha256:" + judge.digest(group),
        "finish_reason": finish,
        "final_turn_present": final,
        "unanswered": timed_out,
        "partial": partial,
        "verified_task_pass": passed,
        "checks": checks,
        "claim_verdict": verdict,
        "native_grade": native,
        "judgment_status": status,
        "first_failed_check": failure,
        "step_checks": steps,
        "projection_error": packet.get("projection_error") if packet else None,
        "judge_segment_count": (
            (packet.get("judge_segments") or {}).get("parts") if packet else None
        ),
        "provider_finish_reasons": generation.get(
            "response_finish_reasons", [c.get("finish_reason") for c in calls]
        ),
        "output_token_upper_bound": generation.get("output_token_upper_bound"),
        "opaque_sidecars_withheld": generation.get("opaque_sidecars_withheld"),
        "miner_cost_microusd": job["miner_cost_microusd"],
        "generator_calls": job["generator_calls"],
        "tokens": tokens,
        "length": {
            # Final context includes the final prompt (prior messages/tools) and completion.
            # Summing every request would count the same conversation prefixes repeatedly.
            "tokens": final_context,
            "generated_tokens": tokens["completion_tokens"],
            "token_basis": "provider_final_prompt_and_completion",
            "message_count": len(messages),
            "assistant_turns": len(steps),
            "message_bytes": len(judge._json(messages).encode()),
            "reasoning_bytes": sum(
                len((m.get("reasoning") or m.get("reasoning_content") or "").encode())
                for m in messages
                if m["role"] == "assistant"
            ),
            "tool_result_bytes": sum(
                len(m["content"].encode())
                for m in messages
                if m["role"] == "tool" and isinstance(m.get("content"), str)
            ),
        },
        "messages": messages,
        "tools": tools,
        "label_turns": row.get("label_turns", []),
        "no_loss_turns": row.get("no_loss_turns", []),
    }
    if hold:
        ins["capture_hold"] = hold
    if experiment:
        ins["experiment"] = experiment
    if source:
        ins["student_source_sha256"] = source["source_sha256"]
    if assessment:
        ins["judge_labels"] = [
            {
                "judge": vote["judge"],
                "judge_version": vote["judge_version"],
                "cannot_judge": vote["cannot_judge"],
                "scores": vote.get("scores"),
                "first_bad_step": vote.get("first_bad_step"),
                "failure_tags": vote.get("failure_tags", []),
                "hard_steps": vote.get("hard_steps", []),
            }
            for vote in assessment.get("records", [])
        ]
    if passed is False and failure is None and verdict == "failed":
        ins["first_failed_check"] = {
            "turn": None,
            "check": "claim_verdict",
            "kind": "terminal_check",
        }
    if assessment:
        first_bad = [
            x
            for x in assessment.get("records", [])
            if not x.get("cannot_judge") and type(x.get("first_bad_step")) is int
        ]
        ins["judge_first_bad_steps"] = [
            {"judge": x["judge"], "turn": x["first_bad_step"] - 1} for x in first_bad
        ]
    require_private_evidence({k: v for k, v in ins.items() if k != "messages"})
    return ins


def apply_native_labels(row, checked, reward_microusd=None):
    """Take task and payment outcomes from the same replay of the original funded work."""
    complete = row["final_turn_present"] and checked["complete_verified"]
    row["verification_method"] = checked["payment_rule"]
    row["verified_complete"] = bool(complete)
    row["verified_task_pass"] = checked["task_pass"] if complete else None
    row["payment_eligible"] = checked["eligible"]
    if reward_microusd is not None:
        row["earned_target_microusd"] = reward_microusd if checked["eligible"] else 0
    if row["verified_task_pass"] is False and row.get("first_failed_check") is None:
        row["first_failed_check"] = {"turn": None, "check": "native_task", "kind": "terminal_check"}
    if "verified_decision" in checked:
        row["sales_evidence"] = {k: checked[k] for k in (
            "verified_decision", "field_evidence", "searches", "opened_sources",
            "first_failed_decision", "observed_repair",
        )}


class TrajectoryArchive:
    """Write a deterministic release digest covering full runs and both sides of each preference."""

    def __init__(self, archive):
        self.archive = archive
        self.digest = sha256()
        self.stats = Counter()
        self.groups = defaultdict(lambda: {True: [], False: []})

    def emit(self, path, value):
        data = (judge._json(value) + "\n").encode()
        self.archive.writestr(path, data)
        h = "sha256:" + sha256(data).hexdigest()
        self.digest.update((path + "\0" + h + "\n").encode())
        return {"path": path, "sha256": h}

    def write(self, row, pair=None):
        run = row["run_id"]
        if not isinstance(run, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", run):
            raise ValueError("invalid run path")
        ref = self.emit("trajectories/" + run + ".jsonl", row)
        self.stats["trajectory_rows"] += 1
        self.stats["partial_runs"] += row["partial"]
        self.stats["unanswered_runs"] += row["unanswered"]
        self.stats["verified_passes"] += row["verified_task_pass"] is True
        self.stats["verified_failures"] += row["verified_task_pass"] is False
        self.stats["unverified_outcomes"] += row["verified_task_pass"] is None
        self.stats["oversized_projections"] += bool(row.get("projection_error"))
        self.stats["segmented_runs"] += (row.get("judge_segment_count") or 0) > 1
        self.stats["final_turns_present"] += row["final_turn_present"]
        self.stats["cannot_judge_runs"] += row["judgment_status"] == "cannot_judge"
        self.stats["capture_holds"] += bool(row.get("capture_hold"))
        self.stats["model_transcripts"] += bool(row["messages"])
        if (
            row["verified_task_pass"] is not None
            and row["final_turn_present"] and not row["partial"] and not row.get("capture_hold")
            and (row["judgment_status"] != "cannot_judge" or row.get("verified_complete") is True)
        ):
            self.groups[row["repeat_group_id"]][row["verified_task_pass"]].append(
                dict(ref, run_id=run)
            )
        if pair:
            self.emit("pairs/" + run + ".jsonl", pair)
            self.stats["correction_pairs"] += 1

    def finish(self):
        preferences = []
        for group, values in sorted(self.groups.items()):
            good, bad = values[True], values[False]
            if good and bad:
                # Pair linearly; either arm may have more repeats than the other.
                for i in range(max(len(good), len(bad))):
                    preferences.append(
                        {
                            "repeat_group_id": group,
                            "chosen": good[i % len(good)],
                            "rejected": bad[i % len(bad)],
                        }
                    )
        if preferences:
            path = "preferences.jsonl"
            data = b"".join((judge._json(x) + "\n").encode() for x in preferences)
            self.archive.writestr(path, data)
            self.digest.update(
                (path + "\0sha256:" + sha256(data).hexdigest() + "\n").encode()
            )
        return {
            **self.stats,
            "preference_pairs": len(preferences),
            "dataset_sha256": "sha256:" + self.digest.hexdigest(),
        }


def partial_generator(objects, job, expected_events=None):
    """Recover only checked model-visible events. Private world and gold data stay in storage."""
    from lab_arena.collection_journal import WorkflowJournal

    journal = WorkflowJournal(objects, job["batch_id"], job["job_id"])
    messages, tools, calls, states = [], [], [], []
    safe = True
    saw_start = False
    finish = None
    for data, _ in journal.read(expected_events):
        event = json.loads(data)
        kind = event["kind"]
        if not saw_start and kind != "start":
            raise ValueError("partial workflow journal lacks its origin")
        if (
            kind in ("start", "reply", "tool")
            and event.get("visible_training_safe") is not True
        ):
            safe = False
        if kind == "start":
            if saw_start or event["model"] != job["model"]:
                raise ValueError("partial journal model or start differs from job")
            saw_start = True
            messages = list(event["messages"])
            tools = event["tools"]
        elif kind == "reply":
            if (
                event.get("model") != job["model"]
                or event["message"].get("role") != "assistant"
            ):
                raise ValueError("partial reply does not bind to assigned generator")
            m = event["message"]
            messages.append(
                {
                    k: v
                    for k, v in m.items()
                    if k
                    in (
                        "role",
                        "content",
                        "tool_calls",
                        "reasoning",
                        "reasoning_content",
                        "reasoning_details",
                    )
                }
            )
            calls.append(
                {k: event.get(k) for k in ("turn", "usage", "finish_reason", "model")}
            )
        elif kind == "tool":
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": event["call_id"],
                    "content": event["output"][:20000],
                }
            )
        elif kind == "interrupted":
            finish = "interrupted"
        elif kind in ("before", "finish"):
            if "code_checks" in event:
                states.append(
                    {
                        "kind": kind,
                        "turn": event.get("turn"),
                        "checks": event["code_checks"],
                    }
                )
            if kind == "finish":
                finish = event.get("finish")
    if not saw_start or not safe:
        return (
            None,
            calls,
            "missing_journal" if not saw_start else "no_visible_privacy_attestation",
        )
    return (
        {
            "run": job["job_id"],
            "teacher": job["model"],
            "messages": messages,
            "tools": tools,
            "label_turns": [
                i for i, m in enumerate(messages) if m.get("role") == "assistant"
            ],
            "no_loss_turns": [],
            "evidence": {"finish": finish, "events": states},
        },
        calls,
        None,
    )
