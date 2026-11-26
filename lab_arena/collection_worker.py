"""Run a locked generator and its private judges; publish immutable evidence before completing the lease."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import re

from lab_arena import collection_judge as judge
from lab_arena.collection_intake import MODELS
from lab_arena.collection_workflow import (
    COMPARISON_CAPTURE,
    TRAIN_ONLY_PARTITION_SHA256,
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
):
    """The journal stores judge bytes separately. Only scores and tags reach the public statement."""
    from lab_arena.collection_journal import WorkflowJournal

    run_id, batch = job["job_id"], job["batch_id"]
    capture = (
        {"capture_version": capture_version} if capture_version is not None else {}
    )
    agent, builder = comparison_revisions(capture)
    journal = None
    if job.get("deepline_key_id") is None and job["scenario_id"].startswith(
        "research-"
    ):
        raise ValueError("live research is outside synthetic funding scope")
    if (
        job["harness_commit"] != agent
        or job["model"] not in MODELS
        or job["mode"] != ("pilot" if policy.pilot else "live")
        or batch != policy.batch_id
        or assignment["run_id"] != run_id
    ):
        raise ValueError("frozen collection assignment mismatch")
    if payment_rule is not None:
        if (
            payment_rule != P7_PAYMENT_RULE
            or policy.pilot
            or capture_version != COMPARISON_CAPTURE
            or not re.fullmatch(r"dv-[a-f0-9]{64}", batch)
            or not _p7(job)
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
            try:
                row, packet = play_workflow(
                    *sources,
                    job,
                    broker,
                    record=journal.append,
                    save=journal.save,
                    capture_version=capture_version,
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
    if payment_rule:
        evidence = native_payment_evidence(objects, job, manifest)
        manifest["payment_rule"] = payment_rule
        manifest["verification"] = persist(
            objects, batch, run_id, "native-verification", evidence
        )
        receipt = persist(objects, batch, run_id, "manifest", manifest)
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


def _p7(job):
    return re.fullmatch(r"P7-s[1-9]\d{0,7}", job["scenario_id"]) and not job.get(
        "experiment"
    )


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
        or visible != message
    ):
        raise ValueError("original generator receipt differs from the trajectory")
    return generation, wire["miner_cost_microusd"], json.loads(pointer)["sha256"]


def native_payment_evidence(objects, job, manifest):
    """Verify a locked ordinary P7 trajectory using the pinned producer's sealed ledger answer."""
    from lab_arena.collection_journal import WorkflowJournal

    run, batch, model = job["job_id"], job["batch_id"], job["model"]
    capture = {"capture_version": COMPARISON_CAPTURE}
    agent, builder = comparison_revisions(capture)
    if (
        not _p7(job)
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
        or split != {**row["split"], "manifest_sha256": TRAIN_ONLY_PARTITION_SHA256}
        or row["split"]["template_id"] != "P7"
    ):
        raise ValueError("generator, scenario or training partition mismatch")
    WorkflowJournal(objects, batch, run).comparison_index(
        complete=True, model=model, manifest_events=manifest["events"], generator=row
    )
    ev = row["evidence"]
    replies = [e for e in ev["events"] if e["kind"] == "reply"]
    messages = [m for m in row["messages"] if m["role"] == "assistant"]
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
    posting = (ev["out"]["state"] or {}).get("posting") or {}
    expected = json.loads(packet["reference_answer"])["answer"]
    eligible = bool(
        eligible
        and total > 0
        and ev["finish"] == "answered"
        and not native.get("hard_fail")
        and native["score"] == 1
        and native["credits"] == {"sealed": 1, "charter": 1, "receipt": 1}
        and isinstance(expected, str)
        and expected
        and expected == posting.get("receipt")
        and expected == ev["out"]["answer"].strip()
        and expected == messages[-1]["content"].strip()
        and any(
            (m.get("reasoning") or m.get("reasoning_content") or "").strip()
            for m in messages
        )
        and any(m["role"] == "tool" for m in row["messages"])
    )
    return {
        "payment_rule": P7_PAYMENT_RULE,
        "eligible": eligible,
        "calls": len(receipts),
        "cost_microusd": total,
        "provider_calls_sha256": judge.digest([sha for _, _, sha in receipts]),
    }


def registered_native_result(pending, objects, policy, approved):
    if (
        policy.pilot
        or pending["batch_id"] != policy.batch_id
        or not re.fullmatch(r"dv-[a-f0-9]{64}", policy.batch_id)
        or approved.get("payment_rule") != P7_PAYMENT_RULE
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
        manifest.get("payment_rule") != P7_PAYMENT_RULE
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
    if recorded != evidence:
        raise ValueError("stored verification differs from the funded work")
    return manifest, evidence


def verify_paid_job(pending, objects, policy, approved):
    _, evidence = registered_native_result(pending, objects, policy, approved)
    if not evidence["eligible"]:
        raise ValueError("first paid run did not pass the native task")
    return evidence
