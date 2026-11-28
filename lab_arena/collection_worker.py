"""Run a locked generator and its private judges; publish immutable evidence before completing the lease."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import re

from lab_arena import collection_judge as judge
from lab_arena.collection_intake import MODELS
from lab_arena.collection_pool import POOL_PAYMENT_RULE, job_reward_rate, pool_target_microusd
from lab_arena.collection_sales_evidence import sales_correction_source
from lab_arena.collection_workflow import (
    COMPARISON_CAPTURE,
    TRAIN_ONLY_PARTITION_SHA256,
    SALES_EVIDENCE_PARTITION_SHA256,
    SALES_EVIDENCE_RULE,
    comparison_revisions,
    read_artifact,
    play_workflow,
    store_artifact,
)


def process_job(
    *,
    job,
    assignment,
    policy,
    prompt,
    intake,
    broker,
    objects,
    sources,
    epoch,
    research=None,
    control_packet=None,
    claim_pack=None,
    persist=store_artifact,
    prepare_launch=False,
    split=None,
    claim_protocol="collection.fresh-claims.v1",
    capture_version=None,
    payment_rule=None,
    correction_source_job=None,
):
    """The journal stores judge bytes separately. Only scores and tags reach the public statement."""
    from lab_arena.collection_journal import WorkflowJournal

    run_id, batch = job["job_id"], job["batch_id"]
    capture = (
        {"capture_version": capture_version} if capture_version is not None else {}
    )
    agent, builder = comparison_revisions(
        {**capture, "scenario_ids": [job["scenario_id"]]}
    )
    owner = job.get("mode") == "owner"
    if owner and (
        not policy.pilot
        or intake is not None
        or prepare_launch
        or payment_rule != SALES_EVIDENCE_RULE
        or not re.fullmatch(r"owner-p8-[0-9a-f]{64}", batch)
        or any(
            job.get(k)
            for k in (
                "miner_hotkey",
                "registration_id",
                "openrouter_key_id",
                "deepline_key_id",
                "price_microusd",
                "lease_token",
            )
        )
    ):
        raise ValueError("owner reference cannot access miner enrollment or settlement")
    if correction_source_job is not None and not owner:
        raise ValueError("a native counterfactual run requires separate owner funding")
    journal = None
    if job.get("deepline_key_id") is None and job["scenario_id"].startswith(
        "research-"
    ):
        raise ValueError("live research is outside synthetic funding scope")
    if (
        job["harness_commit"] != agent
        or job["model"] not in MODELS
        or (not owner and job["mode"] != ("pilot" if policy.pilot else "live"))
        or batch != policy.batch_id
        or assignment["run_id"] != run_id
    ):
        raise ValueError("frozen collection assignment mismatch")
    if re.fullmatch(r"dv-[a-f0-9]{64}", batch) and payment_rule is None:
        raise ValueError("an approved funded offer requires its native payment rule")
    if payment_rule is not None:
        if (
            payment_rule != task_payment_rule(job)
            or (policy.pilot and not owner)
            or capture_version != COMPARISON_CAPTURE
            or (not owner and not re.fullmatch(r"dv-[a-f0-9]{64}", batch))
            or control_packet is not None
            or claim_pack is not None
            or split is None
        ):
            raise ValueError("invalid deterministic payment assignment")
    else:
        judge.validate_prompt(policy, prompt)
    control_ref = None
    row_ref = None
    if control_packet is not None:
        packet = control_packet["packet"]
        if (
            judge.digest(control_packet["case"]) != control_packet["case_sha256"]
            or packet["run_id"] != run_id
            or packet["scenario_id"] != job["scenario_id"]
        ):
            raise ValueError("invalid blind control assignment")
        assignment = dict(assignment, overlap=True)
        control_ref = persist(
            objects, batch, run_id, "control-case", control_packet["case"]
        )
    else:
        if job["scenario_id"].startswith("research-"):
            if research is None:
                raise ValueError("missing trusted research runner")
            row, packet = research(job, broker)
        else:
            if capture and (
                getattr(sources[0], "source_revision", None),
                getattr(sources[1], "source_revision", None),
            ) != (agent, builder):
                raise ValueError("comparison worker loaded the wrong pinned sources")
            journal = WorkflowJournal(objects, batch, run_id)
            if split is not None:
                journal.save("split", split)
            source = (
                sales_correction_source(objects, correction_source_job)
                if correction_source_job is not None
                else None
            )
            try:
                row, packet = play_workflow(
                    *sources,
                    job,
                    broker,
                    record=journal.append,
                    save=journal.save,
                    capture_version=capture_version,
                    **({"native_continuation": source} if source else {}),
                )
            except Exception as exc:
                journal.finish("interrupted")
                journal.save(
                    "interruption",
                    {
                        "job_id": run_id,
                        "cause": type(exc).__name__,
                        "events": journal.hashes,
                    },
                )
                raise
            journal.finish("completed")
        if (
            row["run"] != run_id
            or row["teacher"] != job["model"]
            or packet["scenario_id"] != job["scenario_id"]
            or packet["run_id"] != run_id
        ):
            raise ValueError("collection artifact identity mismatch")
        if any(
            x is not None and (x["teacher"] != job["model"] or x["run"] != run_id)
            for x in row.get("research_rows", [])
        ):
            raise ValueError("incorrect research segment identity")
        row_ref = persist(objects, batch, run_id, "generator", row)
    if split is not None:
        if control_ref and packet.get("split") != split:
            raise ValueError("control source does not belong to the approved split")
        packet = {**packet, "split": split}
    if not control_ref and claim_pack is not None:
        from lab_arena.collection_claims import apply_pack

        packet = apply_pack(packet, batch_id=batch, pack=claim_pack)
    if (
        not payment_rule
        and not control_ref
        and not job["scenario_id"].startswith("research-")
    ):
        from lab_arena.collection_claims import read_fresh

        packet = read_fresh(
            packet,
            policy=policy,
            generator=job["model"],
            broker=broker,
            record_vote=lambda vote: persist(
                objects, batch, run_id, "claim-vote", vote
            ),
            protocol=claim_protocol,
        )
    packet_ref = persist(objects, batch, run_id, "judge-input", packet)
    manifest = {
        "job_id": run_id,
        "batch_id": batch,
        "model": job["model"],
        "scenario_id": job["scenario_id"],
        "harness_commit": agent,
        "builder_commit": builder,
        **capture,
        **({"events": journal.hashes} if journal is not None else {}),
        "judge_policy": asdict(policy),
        "generator": row_ref,
        "packet": packet_ref,
    }
    if owner:
        manifest["funding_mode"] = "owner"
    if correction_source_job:
        manifest["correction_source"] = {
            k: correction_source_job[k]
            for k in (
                "job_id",
                "batch_id",
                "scenario_id",
                "model",
                "harness_commit",
                "mode",
                "artifact_sha256",
            )
        }
    if payment_rule:
        evidence = native_payment_evidence(objects, job, manifest)
        manifest["payment_rule"] = payment_rule
        manifest["verification"] = persist(
            objects, batch, run_id, "native-verification", evidence
        )
        receipt = persist(objects, batch, run_id, "manifest", manifest)
        if owner:
            return manifest
        accepted = evidence["eligible"]
        quality = (
            {"training_status": "needs_review", "payment_rule": payment_rule}
            if accepted
            else {"training_status": "code_rejected"}
        )
        intake.complete(
            run_id,
            job["lease_token"],
            "accepted" if accepted else "rejected",
            quality,
            receipt["sha256"],
            epoch(),
            **({"prepare_launch": True} if accepted and prepare_launch else {}),
        )
        return manifest

    assessment = judge.judge_run(
        policy,
        assignment,
        packet,
        broker.judge,
        prompt=prompt,
        record_segment=(
            (
                lambda role, entry: journal.save(
                    f"judge-part-{role}-{entry['part']:04d}", entry
                )
            )
            if journal is not None
            else None
        ),
    )
    judge_ref = persist(objects, batch, run_id, "judge-output", assessment)
    manifest["assessment"] = judge_ref
    from lab_arena.collection_ranking import rank_pack_if_ready

    if control_ref:
        manifest["control"] = control_ref
    else:
        ranking = rank_pack_if_ready(
            job, packet, assessment, policy, prompt, broker, objects, broker.database
        )
        if ranking is not None:
            manifest["ranking"] = ranking
    receipt = persist(objects, batch, run_id, "manifest", manifest)
    public = judge.public_result(assessment)["judges"]
    summary = public[0] if public else {}
    summary = dict(summary, training_status=assessment["status"])
    checks = packet["checks"]
    complete_execution = all(
        checks.get(k) is True
        for k in ("answered", "shape", "privacy", "training_split")
    )
    status = "accepted" if complete_execution and not control_ref else "rejected"
    if status == "accepted" and prepare_launch:
        intake.complete(
            run_id,
            job["lease_token"],
            status,
            summary,
            receipt["sha256"],
            epoch(),
            prepare_launch=True,
        )
    else:
        intake.complete(
            run_id, job["lease_token"], status, summary, receipt["sha256"], epoch()
        )
    return manifest


P7_PAYMENT_RULE = "verified_p7_v1"


def task_payment_rule(job):
    if re.fullmatch(r"pool-[0-9a-f]{64}", str(job.get("batch_id"))) and re.fullmatch(
        r"(P7|P8|P9|P10)-s[1-9][0-9]{0,7}", job["scenario_id"]
    ):
        return POOL_PAYMENT_RULE
    match = re.fullmatch(r"(P7|P8)-s[1-9]\d{0,7}", job["scenario_id"])
    if not match or job.get("experiment"):
        return None
    return P7_PAYMENT_RULE if match[1] == "P7" else SALES_EVIDENCE_RULE


def _paid_reply(objects, job, index, event, message):
    """Bind the archived assistant turn to the original, broker-settled provider generation."""
    from lab_arena.collection_generation import visible_reply

    batch, run = job["batch_id"], job["job_id"]
    request = f"generate-{index}"
    kind = f"provider-{index:04d}"
    pointer = objects.get_bounded(f"collection/{batch}/{run}/saved/{kind}.json", 8192)
    sent = read_artifact(
        objects, batch, run, kind, json.loads(pointer), 128 * 1024 * 1024
    )
    if (
        sent["request_id"] != request
        or sent["run_id"] != run
        or not sent.get("response_ref")
    ):
        raise ValueError("missing original generator response")
    wire = read_artifact(
        objects, batch, run, "wire-" + request, sent["response_ref"], 128 * 1024 * 1024
    )
    if (
        wire["http_status"] != 200
        or wire["run_id"] != run
        or wire["request_id"] != request
        or type(wire["miner_cost_microusd"]) is not int
        or wire["miner_cost_microusd"] < 0
    ):
        raise ValueError("unsettled or unrelated generator receipt")
    if wire.get("opaque_fields_withheld") is True:
        reply = wire["permitted_response"]
    else:
        body = wire["body_utf8"].encode()
        if hashlib.sha256(body).hexdigest() != wire["response_sha256"]:
            raise ValueError("original generator response hash mismatch")
        reply = json.loads(body)
    (choice,) = reply["choices"]
    visible = visible_reply(choice["message"])
    if not visible.get("reasoning") and not visible.get("reasoning_content"):
        exposed = "\n".join(
            part["text"]
            for part in choice["message"].get("reasoning_details") or []
            if isinstance(part, dict)
            and part.get("type") == "reasoning.text"
            and isinstance(part.get("text"), str)
            and part["text"].strip()
        )
        if exposed:
            visible["reasoning"] = exposed
    generation = wire.get("provider_generation_id")
    if (
        not isinstance(generation, str)
        or not generation
        or reply.get("id") != generation
        or event.get("receipt") != generation
        or reply.get("model") != job["model"]
        or event["model"] != job["model"]
        or event["finish_reason"] != choice["finish_reason"]
        or (batch.startswith("pool-") and event.get("usage") != reply.get("usage"))
        or visible != message
    ):
        raise ValueError("original generator receipt differs from the trajectory")
    return generation, wire["miner_cost_microusd"], json.loads(pointer)["sha256"]


def native_payment_evidence(objects, job, manifest, *, receipts_out=None):
    """Bind the terminal trajectory and outcome to the assigned task, funding route and original receipts."""
    from lab_arena.collection_journal import WorkflowJournal

    run, batch, model = job["job_id"], job["batch_id"], job["model"]
    rule = task_payment_rule(job)
    family = job["scenario_id"].split("-s")[0]
    from lab_arena.collection_workflow import NATIVE_PARTITIONS

    capture = {
        "capture_version": COMPARISON_CAPTURE,
        "scenario_ids": [job["scenario_id"]],
    }
    agent, builder = comparison_revisions(capture)
    if (
        not rule
        or manifest.get("control")
        or manifest["job_id"] != run
        or manifest["batch_id"] != batch
        or manifest["model"] != model
        or manifest["scenario_id"] != job["scenario_id"]
        or manifest["capture_version"] != COMPARISON_CAPTURE
        or manifest["builder_commit"] != builder
        or manifest["harness_commit"] != agent
        or job["harness_commit"] != agent
    ):
        raise ValueError("deterministic work differs from its pinned assignment")
    row = read_artifact(
        objects, batch, run, "generator", manifest["generator"], 128 * 1024 * 1024
    )
    packet = read_artifact(
        objects, batch, run, "judge-input", manifest["packet"], 16 * 1024 * 1024
    )
    split = packet.get("split")
    if (
        row["run"] != run
        or packet["run_id"] != run
        or row["teacher"] != model
        or row["source"] != "locked_workflow"
        or row["task"] != job["scenario_id"]
        or packet["scenario_id"] != job["scenario_id"]
        or not isinstance(split, dict)
        or split
        != {
            **row["split"],
            "manifest_sha256": NATIVE_PARTITIONS[family],
        }
        or row["split"]["template_id"] != job["scenario_id"].split("-s")[0]
    ):
        raise ValueError("generator, scenario or training partition mismatch")
    checkpoint = None
    if rule == POOL_PAYMENT_RULE and row.get("correction"):
        student = read_artifact(
            objects, batch, run, "student-source", manifest["student_source"], 128 << 20
        )
        link = row["correction"]
        checkpoint = student["checkpoint"]
        if (
            row["experiment"]["role"] != "continuation"
            or link["student_model"] == model
            or link["student_model"] != student["student_model"]
            or link["rejected"] != student["rejected"]
            or link["source_sha256"] != student["source_sha256"]
            or link["checkpoint_sha256"] != judge.digest(checkpoint)
        ):
            raise ValueError("continuation does not match its archived source")
    WorkflowJournal(objects, batch, run).comparison_index(
        complete=True,
        model=model,
        manifest_events=manifest["events"],
        generator=row,
        verify_terminal=rule == POOL_PAYMENT_RULE,
        checkpoint=checkpoint,
    )
    ev = row["evidence"]
    replies = [e for e in ev["events"] if e["kind"] == "reply"]
    if bool(manifest.get("correction_source")) != ("native_correction" in row):
        raise ValueError(
            "correction prefix must be tied to its original observed source"
        )
    prefix = (row.get("native_correction") or row.get("correction") or {}).get(
        "prefix_messages", 0
    )
    messages = [m for m in row["messages"][prefix:] if m["role"] == "assistant"]
    if (
        len(replies) != len(messages)
        or len(replies) != ev["generation"]["generated_calls"]
    ):
        raise ValueError("incomplete generator receipt set")
    receipts = [
        _paid_reply(objects, job, i, event, message)
        for i, (event, message) in enumerate(zip(replies, messages), 1)
    ]
    if len({rid for rid, _, _ in receipts}) != len(receipts):
        raise ValueError("duplicate generation receipt")
    total = sum(cost for _, cost, _ in receipts)
    native = ev["grade"]
    checks = packet["checks"]
    eligible = all(
        checks.get(k) is True
        for k in (
            "answered",
            "shape",
            "code_grade",
            "no_hard_failure",
            "privacy",
            "training_split",
        )
    )
    if packet.get("full_evidence") and packet.get("projection_error"):
        # Segmentation limits only gate subjective review. Recover the full reference from the original upload.
        packet = read_artifact(
            objects,
            batch,
            run,
            "full-judge-evidence",
            packet["full_evidence"],
            128 * 1024 * 1024,
        )
    expected = json.loads(packet["reference_answer"])["answer"]
    additional = {}
    if family == "P8":
        from lab_arena.collection_sales_evidence import verify_account_memo

        additional = verify_account_memo(objects, job, manifest, row, packet)
        task_pass = additional["verified_decision"]
    elif family == "P7":
        posting = (ev["out"]["state"] or {}).get("posting") or {}
        task_pass = native["credits"] == {
            "sealed": 1,
            "charter": 1,
            "receipt": 1,
        } and expected == posting.get("receipt")
    else:
        # These desks grade the committed world and cited event IDs. A model may
        # summarize its decision in its own words after committing the plan.
        task_pass = native["score"] == 1 and not native.get("hard_fail")

    final_answer = ev["out"].get("answer")
    complete_verified = bool(
        total > 0
        and ev["finish"] == "answered"
        and messages
        and isinstance(final_answer, str)
        and final_answer.strip()
        and not messages[-1].get("tool_calls")
        and final_answer.strip() == (messages[-1].get("content") or "").strip()
        and all(
            checks.get(k) is True
            for k in ("answered", "shape", "privacy", "training_split")
        )
    )
    eligible = bool(
        eligible
        and total > 0
        and ev["finish"] == "answered"
        and not native.get("hard_fail")
        and native["score"] == 1
        and isinstance(expected, str)
        and expected
        and task_pass
        and (family in {"P9", "P10"} or expected == ev["out"]["answer"].strip())
        and messages
        and (family in {"P9", "P10"} or expected == messages[-1]["content"].strip())
        and any(
            (m.get("reasoning") or m.get("reasoning_content") or "").strip()
            for m in messages
        )
        and any(m["role"] == "tool" for m in row["messages"])
    )
    if manifest.get("correction_source"):
        additional["verified_counterfactual"] = (
            eligible and additional["changed_first_action"]
        )
    if receipts_out is not None:
        receipts_out.extend(receipts)
    # For standing work the economic outcome uses only independently verifiable
    # completion. Preserve exposed responses even when a model takes no tools,
    # omits reasoning on a turn, or makes an unsafe synthetic task decision.
    if rule == POOL_PAYMENT_RULE:
        task_pass = bool(
            complete_verified
            and task_pass
            and checks["code_grade"]
            and checks["no_hard_failure"]
            and (family in {"P9", "P10"} or expected == final_answer.strip())
        )
        rate = job_reward_rate(job)
        if rate != 12500:
            # Preserve the original proof shape for replay of frozen 1.25x jobs.
            additional.update(
                reward_rate_bps=rate,
                earned_target_microusd=pool_target_microusd(job, total) if complete_verified else 0,
            )
    else:
        task_pass = eligible
    return {
        **additional,
        "payment_rule": rule,
        "complete_verified": complete_verified,
        "task_pass": task_pass,
        "eligible": (
            complete_verified
            if rule == POOL_PAYMENT_RULE
            else eligible and not manifest.get("correction_source")
        ),
        "calls": len(receipts),
        "cost_microusd": total,
        "provider_calls_sha256": judge.digest([sha for _, _, sha in receipts]),
    }


def verification_matches(recorded, evidence):
    if recorded == evidence:
        return True
    # Original fixed-offer manifests predate the separate completion and grade labels.
    return evidence["payment_rule"] != POOL_PAYMENT_RULE and recorded == {
        key: value
        for key, value in evidence.items()
        if key not in ("complete_verified", "task_pass")
    }


def registered_native_result(pending, objects, policy, approved):
    if (
        policy.pilot
        or pending["batch_id"] != policy.batch_id
        or not re.fullmatch(r"dv-[a-f0-9]{64}", policy.batch_id)
        or not task_payment_rule(pending)
        or approved.get("payment_rule") != task_payment_rule(pending)
        or approved.get("capture_version") != COMPARISON_CAPTURE
    ):
        raise ValueError("incorrect deterministic payment policy")
    run, batch = pending["job_id"], pending["batch_id"]
    manifest = read_artifact(
        objects,
        batch,
        run,
        "manifest",
        {"sha256": pending["artifact_sha256"]},
        1024 * 1024,
    )
    if (
        manifest.get("payment_rule") != approved["payment_rule"]
        or manifest["judge_policy"] != asdict(policy)
        or any(manifest[k] != approved[k] for k in ("builder_commit", "harness_commit"))
    ):
        raise ValueError("incorrect deterministic payment manifest")
    evidence = native_payment_evidence(objects, pending, manifest)
    recorded = read_artifact(
        objects,
        batch,
        run,
        "native-verification",
        manifest["verification"],
        1024 * 1024,
    )
    if not verification_matches(recorded, evidence):
        raise ValueError("stored verification differs from the funded work")
    return manifest, evidence


def verify_paid_job(pending, objects, policy, approved):
    _, evidence = registered_native_result(pending, objects, policy, approved)
    if not evidence["eligible"]:
        raise ValueError("first paid run did not pass the native task")
    return evidence
