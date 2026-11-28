"""Run one bounded cohort on the trusted collection host. All network calls are metered."""

from __future__ import annotations

import argparse
from lab_arena.collection_controls import ControlPack, PACK_SCHEMA
import json
import os
import re
import sys
import time
from tempfile import NamedTemporaryFile
from functools import partial
from contextlib import closing, contextmanager, redirect_stdout, ExitStack
from pathlib import Path
from uuid import UUID, uuid5

from lab_arena import chain, collection_judge as judge
from lab_arena.broker import HttpxProviderTransport
from lab_arena.collection_prices import claim_at_current_price, endpoint_prices, prices_for_assignment
from lab_arena.collection_broker import (
    CollectionBroker,
    REFERENCE_MODEL,
    REFERENCE_ROUTE,
    batch_versions,
)
from lab_arena.collection_intake import (
    GENERATOR_ROUTES, LEGACY_COMPARISON_TERMS_HASH, MODELS, SYNTHETIC_TRAINING_TERMS_HASH,
    PAID_TASK_FAMILIES, PAID_WORKFLOW_TERMS_HASH, VERIFIED_PRICE_SCHEMA, VERIFIED_TASK_PRICE_SCHEMA, CollectionIntake, live_finney_arena,
)
from lab_arena.collection_worker import P7_PAYMENT_RULE, process_job, verify_paid_job
from lab_arena.collection_workflow import (
    CHAMPION_REV,
    POET_REV,
    comparison_revisions,
    build_scenario,
    workflow_sources,
)
from lab_arena.credentials import CredentialManager
from lab_arena.owner_admission import resolve_finalized_owner
from lab_arena.service import S3ObjectStore
from lab_arena.store import PostgrestTransport


def paid_family(rule):
    if rule not in PAID_TASK_FAMILIES:
        raise ValueError("unknown verified workflow payment rule")
    return PAID_TASK_FAMILIES[rule]


def load_batch(raw, prompt):
    from lab_arena.collection_claims import protocol_of
    protocol_of(raw)
    agent, builder = comparison_revisions(raw)
    if set(raw) - {
        "research_index_sha256",
        "split_manifest",
        "panel_qualification_sha256",
        "claim_protocol",
        "capture_version",
        "thresholds",
        "terms_sha256",
        "reference_successor",
        "route_witness_sha256",
        "pricing_pilot",
        "control_artifacts",
        "claim_pack_sha256",
        "payment_rule",
        "fixed_quotes_microusd",
    } != {
        "mode",
        "seed",
        "routes",
        "scenario_ids",
        "budgets_microusd",
        "control_slots",
        "drift_slots",
        "name",
    }:
        raise ValueError("invalid batch manifest")
    if raw["mode"] not in {"pilot", "live"} or not re.fullmatch(
        r"[a-zA-Z0-9_-]{1,40}", raw["name"]
    ):
        raise ValueError("invalid collection batch")
    scenarios = raw["scenario_ids"]
    paid = "payment_rule" in raw
    terms = PAID_WORKFLOW_TERMS_HASH if paid else SYNTHETIC_TRAINING_TERMS_HASH
    if "terms_sha256" in raw and raw["terms_sha256"] != terms:
        raise ValueError("unrecognized synthetic collection training terms")
    if "terms_sha256" in raw and any(isinstance(s, str) and s.startswith("research-") for s in scenarios):
        raise ValueError("synthetic funding terms cannot authorize research")
    if (
        type(scenarios) is not list
        or not 1 <= len(scenarios) <= 4096
        or any(
            not isinstance(s, str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,100}", s)
            for s in scenarios
        )
        or len(scenarios) != len(set(scenarios))
    ):
        raise ValueError("scenario pack must contain unique identifiers")
    if "capture_version" in raw and any(s.startswith("research-") for s in scenarios):
        raise ValueError("checkpoint comparison requires synthetic workflows")
    if any(s.startswith("research-") for s in scenarios) and not re.fullmatch(
        "[0-9a-f]{64}", raw.get("research_index_sha256", "")
    ):
        raise ValueError("research pack needs a frozen case index")
    models = set(raw["budgets_microusd"])
    if paid:
        from lab_arena.collection_generation import CAPTURE_V2
        family = paid_family(raw["payment_rule"])
        task = "P7" if family == "P" else family
        quotes = raw.get("fixed_quotes_microusd")
        if (
            raw["mode"] != "live"
            or not models or not models <= set(MODELS)
            or not isinstance(quotes, dict) or set(quotes) != models
            or any(type(v) is not int or not 0 < v <= 10**10 for v in quotes.values())
            or raw.get("terms_sha256") != terms or raw.get("capture_version") != CAPTURE_V2
            or not raw.get("split_manifest")
            or not all(re.fullmatch(task + r"-s[1-9][0-9]{0,7}", sid) for sid in scenarios)
            or raw["control_slots"] or raw["drift_slots"] or raw.get("control_artifacts")
            or any(k in raw for k in ("panel_qualification_sha256", "route_witness_sha256",
                                      "reference_successor", "pricing_pilot", "claim_pack_sha256",
                                      "research_index_sha256", "thresholds"))
        ):
            raise ValueError("invalid deterministic workflow payment offer")
    elif "fixed_quotes_microusd" in raw or models != set(MODELS):
        raise ValueError("every model needs a bounded positive budget")
    if any(
        type(n) is not int or not 0 < n <= 10**10
        for n in raw["budgets_microusd"].values()
    ):
        raise ValueError("every model needs a bounded positive budget")
    revisions = batch_versions(raw["routes"])
    qualification = set(judge.QUALIFICATION_JUDGES) <= revisions.keys()
    certificate = raw.get("panel_qualification_sha256")
    if qualification and raw["mode"] == "live" and not paid:
        if not re.fullmatch(r"[0-9a-f]{64}", str(certificate)) or "split_manifest" not in raw:
            raise ValueError("live open judges require independent qualification and a frozen split")
    elif certificate is not None:
        raise ValueError("unexpected qualification approval")

    if not all(
        re.fullmatch(r"[a-z0-9][a-z0-9._/-]{0,127}", v) for v in raw["routes"].values()
    ):
        raise ValueError("every generator and judge needs one pinned provider route")
    successor = raw.get("reference_successor")
    full_witness = raw.get("route_witness_sha256")
    if "pricing_pilot" in raw and (not full_witness or raw["mode"] != "live"):
        raise ValueError("pricing pilot requires a witnessed paid batch")
    if "route_witness_sha256" in raw:
        from lab_arena.collection_route_witness import WITNESS
        from lab_arena.collection_generation import CAPTURE_V2

        if (
            successor
            or full_witness != WITNESS
            or (raw["mode"] == "live" and (
                not re.fullmatch(r"rq-[0-9a-f]{64}", str(raw.get("pricing_pilot")))
                or not certificate or raw["control_slots"] or raw["drift_slots"]
                or raw.get("control_artifacts")
            ))
            or not qualification
            or raw.get("capture_version") != CAPTURE_V2
            or raw.get("terms_sha256") != SYNTHETIC_TRAINING_TERMS_HASH
            or not raw.get("split_manifest")
            or not all(s.startswith("P7-s") for s in scenarios)
        ):
            raise ValueError("invalid full witnessed reference cohort")

    if "reference_successor" in raw:
        from lab_arena.collection_generation import CAPTURE_V2

        if (
            not isinstance(successor, dict)
            or set(successor) != {"parent", "next_slot", "owner_witness_sha256"}
            or not isinstance(successor["parent"], str)
            or not re.fullmatch(r"[A-Za-z0-9_-]+-[a-f0-9]{24}", successor["parent"])
            or type(successor["next_slot"]) is not int
            or not 0 < successor["next_slot"] < len(MODELS) * len(scenarios)
            or MODELS[successor["next_slot"] % len(MODELS)] != REFERENCE_MODEL
            or not isinstance(successor["owner_witness_sha256"], str)
            or not re.fullmatch(
                r"sha256:[0-9a-f]{64}", successor["owner_witness_sha256"]
            )
            or raw["mode"] != "pilot"
            or raw.get("capture_version") != CAPTURE_V2
            or raw.get("terms_sha256") != SYNTHETIC_TRAINING_TERMS_HASH
            or not qualification
            or not raw.get("split_manifest")
            or raw["control_slots"]
            or raw["drift_slots"]
            or raw.get("control_artifacts")
        ):
            raise ValueError("invalid witnessed reference successor")
        parent_raw = {k: v for k, v in raw.items() if k != "reference_successor"}
        parent_raw["routes"] = dict(raw["routes"])
        parent_raw["routes"][REFERENCE_MODEL] = GENERATOR_ROUTES[REFERENCE_MODEL].tag
        if load_batch(parent_raw, prompt).batch_id != successor["parent"]:
            raise ValueError("successor changes earlier frozen batch settings")
    for model, route in GENERATOR_ROUTES.items():
        expected = (
            REFERENCE_ROUTE
            if (successor or full_witness) and model == REFERENCE_MODEL
            else route.tag
        )
        if paid and model in models and model == REFERENCE_MODEL and raw["routes"][model] == REFERENCE_ROUTE:
            continue
        if raw["routes"][model] != expected:
            raise ValueError("unverified generator route")
    bundle_hash = judge.digest(
        {
            "manifest": raw,
            "prompt": judge.digest(prompt),
            "champion": agent,
            "builder": builder,
            "model_revisions": revisions,
        }
    )
    thresholds = raw.get("thresholds", {})
    if not isinstance(thresholds, dict) or set(thresholds) - {
        "accept_sum",
        "salvage_min_steps",
        "agreement_percent",
    }:
        raise ValueError("unknown batch threshold")
    versions = {}
    if qualification:
        versions["fable_version"], versions["astra_version"] = judge.QUALIFICATION_JUDGES
    if paid:
        batch_id = "dv-" + bundle_hash
    elif successor:
        batch_id = "rs-" + bundle_hash
    elif full_witness:
        batch_id = ("lq-" if raw["mode"] == "live" else "rq-") + bundle_hash
    else:
        batch_id = raw["name"] + "-" + bundle_hash[:24]
    policy = judge.BatchPolicy(
        batch_id,
        raw["seed"],
        judge.digest(prompt),
        pilot=raw["mode"] == "pilot",
        **versions,
        **thresholds,
    )
    for slot in raw["drift_slots"]:
        if not isinstance(slot, str) or not re.fullmatch(r"0|[1-9][0-9]{0,4}", slot) or int(slot) >= len(MODELS) * len(scenarios):
            raise ValueError("drift slot outside batch")
        generator = MODELS[int(slot) % len(MODELS)]
        if len(judge.independent_roles(policy, generator)) != len(policy.roles):
            raise ValueError("drift slot requires both independent judges")
    judge.validate_prompt(policy, prompt)
    return policy


def assignment(job):
    return {
        "run_id": job["job_id"],
        "generator": job["model"],
        "harness": job["harness_commit"],
        "miner": job["miner_hotkey"],
        "slot": job["slot"],
        "cohort": job["slot"],
    }


def ready_launch_evidence(pending, objects, policy, approved):
    from lab_arena.collection_judge_review import registered_bundle

    agent, builder = comparison_revisions(approved)
    if (
        pending["batch_id"] != policy.batch_id
        or policy.pilot
        or pending["harness_commit"] != agent
        or approved["harness_commit"] != agent
        or approved["builder_commit"] != builder
        or approved["judge_prompt_sha256"] != policy.prompt_sha256
    ):
        raise ValueError("staged launch differs from calibrated live batch")
    job = pending["job_id"]
    evidence = registered_bundle(
        policy,
        [
            {
                "run_id": job,
                "generator": pending["model"],
                "harness": pending["harness_commit"],
                "miner": pending["miner_hotkey"],
            }
        ],
        jobs=[pending],
        objects=objects,
        planted={},
        drift={},
        expected_builder=builder,
        capture_version=approved.get("capture_version"),
    )
    checks = evidence["runs"][0]["packet"]["checks"]
    if not all(
        checks.get(k) is True
        for k in ("answered", "shape", "privacy", "training_split")
    ):
        raise ValueError("first live collection execution is incomplete")
    return evidence["runs"][0]


def complete_launch_verdict(assessment):
    return (
        bool(assessment["records"])
        and not assessment.get("verifier_unavailable")
        and all(not v["cannot_judge"] for v in assessment["records"])
    )


def finish_launch(database, pending, objects, policy, approved):
    from scripts.collection_batch import cutover_anchor

    if approved.get("schema") == VERIFIED_PRICE_SCHEMA:
        verify_paid_job(pending, objects, policy, approved)
    else:
        evidence = ready_launch_evidence(pending, objects, policy, approved)
        if not complete_launch_verdict(evidence["assessment"]):
            raise ValueError("first live collection job awaits owner verification")
    anchor = cutover_anchor(database, launch_job=pending)
    result = database.rpc(
        "collection_launch_finish_v1", {"p_job": pending["job_id"], "p_anchor": anchor}
    )
    if result.get("status") != "accepted":
        raise ValueError("collection launch acceptance unavailable")


class IncompleteOwnerReview(ValueError):
    """The bounded owner attempt returned no complete verified verdict."""


def rejudge_ready(
    database, objects, transport, raw, policy, prompt, judge_key, job_id, attempt_id,
    *, interrupted_request=None,
):
    """Retry only an unavailable owner verdict against the exact stored execution."""
    from lab_arena.collection_launch_review import LaunchReviewBroker, finish_review
    from lab_arena.collection_workflow import store_artifact
    from scripts.collection_batch import read_artifact

    if policy.pilot:
        from lab_arena import collection_judge_review as review
        from lab_arena.collection_quality import verify_split

        rows = database.rpc("collection_batch_jobs_v1", {"p_batch": policy.batch_id})
        pending = next((r for r in rows if r["job_id"] == job_id), None)
        if (
            not pending
            or pending["mode"] != "pilot"
            or pending["status"] != "accepted"
            or str(pending["slot"]) in raw.get("control_artifacts", {})
            or not raw.get("capture_version")
            or not raw.get("split_manifest")
        ):
            raise ValueError("expected one complete funded ordinary pilot job")
        _, builder = comparison_revisions(raw)
        bundle = review.registered_bundle(
            policy,
            [assignment(pending)],
            jobs=[pending],
            objects=objects,
            expected_builder=builder,
            capture_version=raw["capture_version"],
            planted={},
            drift={},
        )
        bundle["split_manifest_sha256"] = judge.digest(raw["split_manifest"])
        if raw.get("claim_protocol"):
            bundle["claim_protocol"] = raw["claim_protocol"]
        review._read_bundle(bundle)
        (evidence,) = bundle["runs"]
        packet = evidence["packet"]
        verify_split(packet["split"], raw["split_manifest"])
        if not all(packet["checks"].values()) or judge.claim_verdict(packet) not in (
            "passed",
            "not_applicable",
        ):
            if interrupted_request is not None:
                return False
            raise ValueError("pilot lacks complete native, privacy or factual checks")
        if pending.get("pilot_review_sha256"):
            return  # Already verified the only registered recovery above.
    else:
        schedule = database.rpc("collection_reward_schedule_v1", {})
        pending = schedule.get("pending") or {}
        if (
            not schedule.get("staged")
            or pending.get("status") != "ready"
            or pending.get("job_id") != job_id
        ):
            raise ValueError("requested staged job is not ready for owner adjudication")
        approved = schedule["pilot_evidence"]
        verify_live_panel(raw, policy, approved)
        evidence = ready_launch_evidence(pending, objects, policy, approved)
    previous = evidence["assessment"]
    if interrupted_request is not None:
        missing = previous.get("verifier_unavailable") or {}
        if (missing.get("reason") != "http_error" or missing.get("request_id") != interrupted_request
                or any(v["cannot_judge"] for v in previous["records"])):
            return False
    if not previous.get("verifier_unavailable") or any(
        v["cannot_judge"] for v in previous["records"]
    ):
        raise ValueError(
            "only an unavailable verifier on the original execution may be retried"
        )
    roles = {policy.version(r): r for r in policy.roles}
    missing_model = previous["verifier_unavailable"]["model"]
    primary = (
        previous["records"][0]["judge"] if previous["records"] else roles[missing_model]
    )
    prices = endpoint_prices(
        raw["routes"], models=[missing_model] if previous["records"] else roles
    )
    channel = LaunchReviewBroker(
        database,
        transport,
        pending,
        prices,
        routes=raw["routes"],
        judge_key=judge_key,
        attempt_id=attempt_id,
    )
    channel.panel_qualification = raw.get("panel_qualification_sha256")
    replacement = judge.judge_run(
        policy,
        {"run_id": job_id, "primary": primary, "overlap": previous["overlap"], "generator":pending["model"]},
        evidence["packet"],
        channel.judge,
        prompt=prompt,
        previous=previous,
    )
    if not complete_launch_verdict(replacement):
        raise IncompleteOwnerReview(
            "owner adjudication did not produce a complete verified record"
        )
    batch = policy.batch_id
    original = read_artifact(
        objects,
        batch,
        job_id,
        "manifest",
        {"sha256": evidence["manifest_sha256"]},
        1024 * 1024,
    )
    revised = dict(
        original,
        assessment=store_artifact(objects, batch, job_id, "judge-output", replacement),
        judge_recovery={
            "attempt_id": attempt_id,
            "prior_manifest_sha256": evidence["manifest_sha256"],
        },
    )
    link = store_artifact(objects, batch, job_id, "manifest", revised)
    quality = dict(
        judge.public_result(replacement)["judges"][0],
        training_status=replacement["status"],
    )
    finish_review(
        database,
        pending,
        attempt_id,
        link["sha256"],
        quality,
        channel.completed_requests,
    )
    if policy.pilot:
        return True
    current = database.rpc("collection_reward_schedule_v1", {}).get("pending") or {}
    if (
        current.get("job_id") != job_id
        or current.get("artifact_sha256") != link["sha256"]
    ):
        raise ValueError("revised launch evidence is not current")
    finish_launch(database, current, objects, policy, approved)


def recover_idle_judge(database, objects, transport, raw, policy, prompt, judge_key):
    """Use at most one owner retry on an idle pilot tick. Existing reservations fence restarts."""
    candidates = database.rpc("collection_interrupted_pilot_judges_v1", {"p_batch": policy.batch_id})
    if not candidates:
        return
    # Rotate so one unready packet or an exhausted budget cannot hold up other jobs.
    candidate = candidates[int(time.time() // 300) % len(candidates)]
    job, request = candidate["job"], candidate["request"]
    attempt = str(uuid5(UUID(job), "owner-pilot-outage:" + request))
    try:
        recovered = rejudge_ready(
            database, objects, transport, raw, policy, prompt, judge_key,
            job, attempt, interrupted_request=request,
        )
    except IncompleteOwnerReview:
        status = "pilot_judge_held"
    else:
        if not recovered:
            return
        status = "pilot_judge_recovered"
    print(json.dumps({"batch": policy.batch_id, "job": job, "status": status}), flush=True)


class ComparisonSplitHold(ValueError):
    """The pinned comparison pack cannot use a miner's funding."""


def pack_split(raw, builders, scenario_id):
    from lab_arena.collection_quality import admit_built_split

    if not raw.get("split_manifest"):
        raise ComparisonSplitHold("missing_split_manifest")
    try:
        task = build_scenario(builders, scenario_id)
        return admit_built_split(builders, task, None, raw["split_manifest"])
    except (ValueError, builders.guard.Refused) as exc:
        raise ComparisonSplitHold("scenario_not_in_training_partition") from exc


def report_split_hold(policy, exc):
    print(
        json.dumps(
            {"batch": policy.batch_id, "status": "comparison_held", "reason": str(exc)}
        ),
        flush=True,
    )
    return 0


def verify_live_panel(raw, policy, approved):
    schemas = (VERIFIED_PRICE_SCHEMA, VERIFIED_TASK_PRICE_SCHEMA)
    if "payment_rule" in raw or approved.get("schema") in schemas:
        family = paid_family(raw.get("payment_rule"))
        if (
            approved.get("schema") not in schemas
            or (approved["schema"] == VERIFIED_PRICE_SCHEMA and family != "P")
            or approved.get("payment_rule") != raw["payment_rule"]
            or approved.get("live_batch_id") != policy.batch_id
            or approved.get("terms_sha256") != raw["terms_sha256"]
            or approved.get("routes") != {m: raw["routes"][m] for m in raw["budgets_microusd"]}
            or approved.get("prices") != {family: raw["fixed_quotes_microusd"]}
            or approved.get("budgets") != {family: raw["budgets_microusd"]}
            or approved.get("capture_version") != raw["capture_version"]
            or approved.get("split_manifest_sha256") != judge.digest(raw["split_manifest"])
            or approved.get("judge_prompt_sha256") != policy.prompt_sha256
            or comparison_revisions(raw) != (approved.get("harness_commit"), approved.get("builder_commit"))
        ):
            raise ValueError("paid workflow differs from the fixed offer")
        return
    certificate = raw.get("panel_qualification_sha256")
    witness = raw.get("route_witness_sha256")
    priced_id = approved.get("batch_id", "")
    priced_witness = bool(re.fullmatch(r"rq-[0-9a-f]{64}", str(priced_id)))
    if (bool(witness) != priced_witness or witness != approved.get("route_witness_sha256")
            or (witness and raw.get("pricing_pilot") != priced_id)):
        raise ValueError("generator route differs from the qualified pricing pilot")
    if (
        comparison_revisions(raw) != comparison_revisions(approved)
        or certificate != approved.get("panel_qualification_sha256")
        or raw.get("claim_protocol", "collection.fresh-claims.v1")
        != approved.get("claim_protocol", "collection.fresh-claims.v1")
        or policy.qualification != bool(certificate)
        or (
            approved.get("split_manifest_sha256")
            and (
                "split_manifest" not in raw
                or judge.digest(raw["split_manifest"])
                != approved["split_manifest_sha256"]
            )
        )
    ):
        raise ValueError("live panel or split differs from approved pilot")
    return certificate


def run_cohort(
    *,
    raw,
    policy,
    prompt,
    intake,
    database,
    objects,
    transport,
    pricing,
    epoch,
    sources,
    owner_lookup,
    judge_key,
    max_jobs,
    research=None,
    controls=None,
    claim_pack=None,
    recover_idle_pilot=False,
    before_claim=None,
):
    if raw.get("route_witness_sha256"):
        from lab_arena.collection_route_witness import verified_witness

        verified_witness(raw, objects)
    agent, builder = comparison_revisions(raw)
    if raw.get("capture_version") and (
        getattr(sources[0], "source_revision", None),
        getattr(sources[1], "source_revision", None),
    ) != (agent, builder):
        raise ValueError("comparison worker loaded the wrong pinned sources")
    staged = False
    ceiling = dict(raw["budgets_microusd"])
    fees = dict.fromkeys(MODELS, 0)
    if not judge_key and not raw.get("payment_rule"):
        raise ValueError("private verifier credential unavailable")
    if not policy.pilot:
        schedule = (
            database.rpc("collection_offer_schedule_v1", {"p_offer": policy.batch_id})
            if raw.get("payment_rule")
            else database.rpc("collection_reward_schedule_v1", {})
        )
        if not schedule or epoch() < schedule["first_epoch"] - 1:
            raise ValueError("payout cutover not ready")
        verify_live_panel(raw, policy, schedule["pilot_evidence"])
        staged = schedule.get("staged", False)
        if staged and schedule.get("pending"):
            pending = schedule["pending"]
            if pending["status"] == "ready":
                finish_launch(database, pending, objects, policy, schedule["pilot_evidence"])
            return 0
        approved = schedule["pilot_evidence"]
        if raw.get("payment_rule"):
            family = paid_family(raw["payment_rule"])
        else:
            (family,) = {("research" if sid.startswith("research-") else sid[0]) for sid in raw["scenario_ids"]}
        if (
            approved["harness_commit"] != agent
            or approved["builder_commit"] != builder
            or approved["judge_prompt_sha256"] != policy.prompt_sha256
        ):
            raise ValueError("live arm differs from the measured pilot")
        fees = approved["prices"][family]
        if approved["budgets"][family] != ceiling:
            raise ValueError("live cap differs from pilot")
    splits = {}
    for sid in raw["scenario_ids"]:
        if sid.startswith("research-"):
            if research is None:
                raise ValueError("research runner and label evidence required")
        else:
            try:
                splits[sid] = pack_split(raw, sources[1], sid)
            except ComparisonSplitHold as exc:
                return report_split_hold(policy, exc)
    completed = 0
    for _ in range(max_jobs):
        claim = partial(
            intake.claim, policy.batch_id, raw["scenario_ids"], agent, fees, ceiling,
            terms_hash=raw.get("terms_sha256", LEGACY_COMPARISON_TERMS_HASH),
        )
        job, job_prices = claim_at_current_price(claim, pricing, before_claim=before_claim)
        if not job:
            if recover_idle_pilot and policy.pilot and raw.get("capture_version") and raw.get("split_manifest"):
                recover_idle_judge(database, objects, transport, raw, policy, prompt, judge_key)
            break
        try:
            if owner_lookup(job["miner_hotkey"]).coldkey != job["owner_coldkey"]:
                raise ValueError("collection funding owner changed")
            model_broker = CollectionBroker(
                intake,
                database,
                transport,
                job,
                job_prices,
                routes=raw["routes"],
                judge_key=judge_key,
            )
            model_broker.panel_qualification = raw.get("panel_qualification_sha256")
            model_broker.pricing_pilot = raw.get("pricing_pilot")
            a = assignment(job)
            manifest = process_job(
                job=job,
                assignment=a if raw.get("payment_rule") else judge.assign_judges(policy, [a])[a["run_id"]],
                policy=policy,
                prompt=prompt,
                intake=intake,
                broker=model_broker,
                objects=objects,
                sources=sources,
                epoch=epoch,
                research=research,
                control_packet=controls.bind(job) if controls else None,
                claim_pack=claim_pack,
                prepare_launch=staged,
                **({"payment_rule": raw["payment_rule"]} if "payment_rule" in raw else {}),
                capture_version=raw.get("capture_version"),
                split=splits.get(job["scenario_id"]),
                claim_protocol=raw.get("claim_protocol", "collection.fresh-claims.v1"),
            )
        except Exception as exc:
            # Never print raw model output, arguments, credentials or provider messages.
            # If a completion response was lost, the terminal row is immutable.
            try:
                intake.complete(
                    job["job_id"], job["lease_token"], "failed", {}, None, epoch()
                )
            except Exception as recovery:
                raise RuntimeError(
                    "collection completion requires readback"
                ) from recovery
            raise RuntimeError("collection job failed: " + type(exc).__name__) from None
        print(
            json.dumps(
                {
                    "job": manifest["job_id"],
                    "batch": policy.batch_id,
                    "status": "completed",
                }
            ),
            flush=True,
        )
        completed += 1
        if staged:
            pending = database.rpc("collection_reward_schedule_v1", {}).get("pending")
            if pending and pending["status"] == "ready":
                finish_launch(database, pending, objects, policy, approved)
            break
    return completed


def evidence_stores():
    return (
        PostgrestTransport(
            os.environ["LAB_ARENA_SUPABASE_URL"],
            service_key=os.environ["LAB_ARENA_SERVICE_KEY"],
        ),
        S3ObjectStore(
            os.environ["LAB_ARENA_BUCKET"],
            prefix=os.getenv("LAB_ARENA_OBJECT_PREFIX", ""),
        ),
    )


def stage_failed_pool_continuation(spec, job, *, database, objects, directory, inputs, prompt):
    """At most one outstanding continuation per ordinary program and one per scenario."""
    from lab_arena.collection_experiment import load_experiment, pool_program_id
    from lab_arena.collection_pool_source import funded_source
    from scripts.collection_review import read_file
    from scripts.prepare_collection_experiment import (
        lock_pool_directory,
        matches_continuation,
        store_private_json,
    )

    corrector = spec.raw["correction_model"]
    scenario = job["scenario_id"]
    seed = scenario.split("-s")[1]
    if job["batch_id"] != spec.batch_id or job["model"] == corrector:
        raise ValueError("invalid student or independent correction model")
    program = pool_program_id(spec.raw)
    # Serialize competing trusted workers' decisions on this existing directory.
    # Source bytes and manifests are hash-addressed; no new queue or mutable job state.
    with lock_pool_directory(directory):
        latest = None
        for path in directory.glob("pool-*.json"):
            raw = json.loads(read_file(path))
            if raw.get("auto_from_program") != program:
                # A manually staged correction can have a different pack ID and
                # source projection but still resume this same original paid case.
                if matches_continuation(
                    raw, inputs, scenario, spec.batch_id, {corrector}
                ):
                    return None
                continue
            case = raw["cases"][0]["scenario_id"]
            if case == scenario:
                return None
            other = int(case.split("-s")[1])
            if latest is None or other > latest[0]:
                latest = other,path.stem
        if latest:
            if latest[0] >= int(seed):
                return None
            previous = database.rpc("collection_batch_jobs_v1",{"p_batch":latest[1]})
            if not any(j["status"] in {"accepted","rejected","failed"} for j in previous):
                return None
        source = funded_source(
            database,
            objects,
            spec.batch_id,
            job["job_id"],
            prompt,
            classify_grounding=True,
        )
        if not (
            source["rejected_fatal"] or source["failure"].get("unrepaired_grounding")
        ):
            return None
        source_hash = "sha256:" + judge.digest(source)
        expert = next(a for a in spec.raw["arms"] if a["model"]==corrector)
        raw = {k:v for k,v in spec.raw.items() if k not in ("successor_seeds","correction_model","series_arm_order")}
        raw.update(id="continue-"+program[:24],auto_from_program=program,max_slots=1,
                   cases=[{"scenario_id":scenario,"source_sha256":source_hash}],
                   arms=[dict(expert,id="correction",role="continuation",repeats=1)],
                   routes={corrector:spec.routes[corrector]},budget_microusd=expert["budget_microusd"],
                   question="Can the expert recover from the observed failing action at the same state?")
        continuation=load_experiment(raw,prompt)
        store_private_json(inputs/(source_hash[7:]+".json"),source)
        store_private_json(directory/(continuation.batch_id+".json"),raw)
        print(json.dumps({"batch":continuation.batch_id,"source_job":job["job_id"],"status":"correction_staged"}),flush=True)
        return continuation.batch_id


def _freeze_pool_successor(parent, directory, prompt, leader):
    from lab_arena.collection_experiment import load_experiment, pool_program_id
    from scripts.collection_review import read_file
    from scripts.prepare_collection_experiment import (
        lock_pool_directory,
        store_private_json,
    )

    family, last = parent.raw["cases"][-1]["scenario_id"].split("-s")
    seed = int(last) + 1
    size = len(parent.raw["cases"])
    cases = [{"scenario_id": f"{family}-s{i}"} for i in range(seed, seed + size)]
    raw = dict(parent.raw, cases=cases)
    group = pool_program_id(parent.raw)
    if leader and leader[0] == raw["id"]:
        model = leader[1]
        arms = raw["arms"]
        if model not in {arm["model"] for arm in arms}:
            raise ValueError("successor leader must already be in the frozen program")
        if arms[0]["model"] != model:
            raw.setdefault("series_arm_order", [arm["id"] for arm in arms])
            raw["arms"] = sorted(arms, key=lambda arm: arm["model"] != model)
    successor = load_experiment(raw, prompt)
    with lock_pool_directory(directory):
        # Two operators can configure different leaders at the same pack boundary.
        # Reuse the already-frozen roster before publishing a second batch.
        for path in directory.glob("pool-*.json"):
            if path.stem == parent.batch_id:
                continue
            try:
                candidate = json.loads(read_file(path))
            except (OSError, ValueError):
                continue  # Reported by the poller's normal manifest validation.
            if (
                not isinstance(candidate, dict)
                or not candidate.get("successor_seeds")
                or candidate.get("cases") != cases
                or candidate.get("id") != raw["id"]
            ):
                continue
            existing = load_experiment(candidate, prompt)
            if existing.batch_id != path.stem:
                raise ValueError("successor filename differs from its frozen identity")
            if pool_program_id(candidate) == group:
                return existing
        store_private_json(directory / (successor.batch_id + ".json"), raw)
    return successor


def poll_pool_experiments(prompt, directory, database, run, *, successor_leader=None, lane=None, fundable=None):
    """Try a disjoint healthy program after a pre-lease refusal, never after an ambiguous claim."""
    from lab_arena.collection_experiment import load_experiment, pool_program_id
    from scripts.collection_review import read_file
    from lab_arena.collection_prices import CatalogError
    from lab_arena.collection_governor import HeldRoute

    def unavailable(name, exc):
        reason = exc.code if isinstance(exc, CatalogError) else type(exc).__name__
        print(json.dumps({"batch": name, "status": "pool_unavailable", "reason": reason}),
              file=sys.stderr, flush=True)

    active = {}
    failures = False
    for path in sorted(directory.glob("pool-*.json")):
        try:
            raw = json.loads(read_file(path))
            spec = load_experiment(raw, prompt)
            if spec.batch_id != path.stem:
                raise ValueError("pool manifest differs from its frozen identity")
        except (OSError, ValueError, TypeError, KeyError) as exc:
            unavailable(path.stem, exc)
            failures = True
            continue
        # Old series members remain immutable for exports. Only the most recent
        # published seed range can receive additional assignments.
        group = (raw.get("auto_from_program") or "") + "-correction" if raw.get("auto_from_program") else (
            pool_program_id(raw) if raw.get("successor_seeds") else spec.batch_id
        )
        seed = int(raw["cases"][0]["scenario_id"].split("-s")[1])
        if group in active and seed == active[group][0]:
            # Hold a conflicting range without denying unrelated funded work.
            active[group] = (seed, None)
            unavailable(path.stem, ValueError("conflicting frozen packs"))
            failures = True
            continue
        if group not in active or seed > active[group][0]:
            active[group] = seed, spec
    if not active:
        raise ValueError("active standing pool requires a valid operator experiment")
    cursor = directory / (
        ".last-pool-program" if not lane else f".last-pool-program-{lane}"
    )
    groups = sorted(k for k, (_, spec) in active.items() if spec is not None)
    if lane and groups:
        offset = lane % len(groups)
        groups = groups[offset:] + groups[:offset]
    if cursor.exists():
        previous = read_file(cursor, maximum=128).decode()
        if previous in groups:
            index = groups.index(previous) + 1
            groups = groups[index:] + groups[:index]
    for group in groups:
        seed, spec = active[group]
        attempted = False

        def on_claim():
            nonlocal attempted
            attempted = True
            with NamedTemporaryFile(dir=directory, delete=False) as out:
                out.write(group.encode())
                out.flush()
                os.fsync(out.fileno())
            os.replace(out.name, cursor)

        def dispatch(member):
            held = set()
            while True:
                try:
                    return run(member.raw, before_claim=on_claim)
                except HeldRoute as exc:
                    if lane is None or attempted or exc.model in held:
                        raise
                    unavailable(member.batch_id, exc.cause)
                    held.add(exc.model)
                    if len(held) >= len({arm["model"] for arm in member.raw["arms"]}):
                        return 0

        try:
            if (
                fundable is not None
                and min(a["budget_microusd"] for a in spec.raw["arms"]) > fundable
            ):
                continue
            state = (
                "open"
                if lane is None
                else database.rpc(
                    "collection_pool_pack_state_v1",
                    {"p_batch": spec.batch_id, "p_slots": len(spec.slots)},
                )
            )
            if state == "running":
                continue
            if state == "open":
                if dispatch(spec):
                    return 1
                if attempted and lane is None:
                    return 0
                if lane is not None:
                    # Other concurrent lanes may just have acquired the preview.
                    # A qualified empty response confirms we hold no new lease.
                    continue
            elif state != "complete":
                raise ValueError("unknown governed pack status")
            if not spec.raw.get("successor_seeds"):
                continue
            if lane is None:
                jobs = database.rpc(
                    "collection_batch_jobs_v1", {"p_batch": spec.batch_id}
                )
                finished = {
                    j["slot"]
                    for j in jobs
                    if j["status"] in {"accepted", "rejected", "failed"}
                }
                if finished != set(range(len(spec.slots))):
                    continue
            successor = _freeze_pool_successor(
                spec, directory, prompt, successor_leader
            )
            if dispatch(successor):
                return 1
            if attempted and lane is None:
                return 0
        except Exception as exc:
            # The same chain is used for every program. Trying unrelated packs
            # cannot restore finality or ownership evidence during an RPC outage.
            if attempted or (lane is not None and isinstance(exc, chain.ArenaChainError)):
                raise
            unavailable(spec.batch_id, exc)
            failures = True
    if failures:
        raise ValueError("one or more standing programs could not be polled")
    return 0


def poll_paid_offers(baseline, prompt, directory, run):
    """Use one existing worker slot across the immutable manifests miners have signed."""
    from scripts.collection_review import read_file

    base_id = load_batch(baseline, prompt).batch_id
    if baseline.get("mode") != "live" or "payment_rule" not in baseline:
        raise ValueError("paid offer polling requires a verified live batch")
    paths = {p.stem: p for p in directory.glob("dv-*.json")}
    paths.setdefault(base_id, None)
    order = sorted(paths)
    cursor = directory / ".last-claimed-offer"
    if cursor.exists():
        previous = read_file(cursor, maximum=128).decode()
        if previous in paths:
            cut = order.index(previous) + 1
            order = order[cut:] + order[:cut]

    failures = []
    for offer_id in order:
        attempted = False

        def before_claim():
            nonlocal attempted
            # A lost claim response cannot authorize a second dispatch in this poll.
            attempted = True
            with NamedTemporaryFile(dir=directory, delete=False) as f:
                f.write(offer_id.encode())
                f.flush()
                os.fsync(f.fileno())
            os.replace(f.name, cursor)

        try:
            manifest = baseline if paths[offer_id] is None else json.loads(read_file(paths[offer_id]))
            if (
                manifest.get("mode") != "live" or "payment_rule" not in manifest
                or load_batch(manifest, prompt).batch_id != offer_id
            ):
                raise ValueError("paid manifest differs from its immutable offer ID")
            if run(manifest, before_claim=before_claim):
                return 1
            if attempted:
                break
        except Exception as exc:
            if attempted:
                raise
            failures.append(type(exc).__name__)
            print(json.dumps({"offer": offer_id, "status": "offer_unavailable", "reason": type(exc).__name__}),
                  file=sys.stderr, flush=True)
    if failures:
        raise ValueError("one or more paid offers could not be polled")
    return 0


def _worker_timer_active():
    import subprocess

    try:
        return (
            subprocess.run(
                [
                    "systemctl",
                    "is-active",
                    "--quiet",
                    "leadpoet-collection-pilot.timer",
                ],
                timeout=3,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            ).returncode
            == 0
        )
    except (OSError, subprocess.TimeoutExpired):
        return False


def poll_funded_burst(
    limit,
    poll,
    *,
    productive,
    monotonic=time.monotonic,
    timer_active=_worker_timer_active,
):
    """Reselect funding for each new job. Let an authorized run finish before draining."""
    deadline = monotonic() + 30 * 60 if productive else None
    completed = 0
    while completed < limit:
        if productive and completed:
            if monotonic() >= deadline or (completed % 8 == 0 and not timer_active()):
                break
        if not poll():
            break
        completed += 1
    return completed


class LaneDraining(Exception):
    pass


def governed_tick(args, lane, deadline):
    """One authenticated experiment in this process; no shared harness module globals."""
    from lab_arena.collection_intake import register_signatures

    register_signatures()
    db, _ = evidence_stores()
    settled = []
    prompt = json.loads(args.prompt.read_bytes())

    tick_epoch = None

    def epoch_for_tick(client, cutover, database, lock):
        nonlocal tick_epoch
        if tick_epoch is None:
            from lab_arena.collection_governor import prepare_epoch
            tick_epoch = prepare_epoch(client, cutover, database, lock)
        return tick_epoch

    def run(raw, *, before_claim):
        if time.monotonic() >= deadline or not _worker_timer_active():
            raise LaneDraining()

        def mark_claim():
            if time.monotonic() >= deadline or not _worker_timer_active():
                raise LaneDraining()
            before_claim()

        return _run_options(
            args,
            argparse.ArgumentParser(),
            raw,
            before_claim=mark_claim,
            on_pool_settled=settled.append, epoch_for_tick=epoch_for_tick,
        )

    try:
        funds = db.rpc("collection_pool_fundable_v1", {})
        if type(funds) is not int or funds < 0:
            raise ValueError("invalid signed pool capacity")
        if not funds:
            return 0
        return poll_pool_experiments(
            prompt,
            args.offer_directory,
            db,
            run,
            lane=lane,
            fundable=funds,
            successor_leader=args.successor_first_model,
        )
    except LaneDraining:
        return 0
    except RuntimeError:
        # Only a finished failure acknowledged by its own immutable lease can
        # advance the lane. Neither an ambiguous claim nor uncertain completion
        # can trigger an immediate repeat of that operation.
        if settled == ["failed"]:
            return 1
        raise
    finally:
        db.close()


def _init_collection_process():
    # The SDK registers a regular atexit hook. A multiprocessing child closes its
    # log pipe earlier, so drain that child's listener before queue finalizers.
    import atexit
    from multiprocessing.util import Finalize

    def stop_sdk_listener():
        logs = sys.modules.get("bittensor.utils.btlogging")
        listener = getattr(getattr(logs, "logging", None), "_listener", None)
        if listener is not None and getattr(listener, "_thread", None) is not None:
            atexit.unregister(listener.stop)
            listener.stop()
    from lab_arena.collection_chain_access import close_lane_chain
    Finalize(None, close_lane_chain, exitpriority=25)
    Finalize(None, stop_sdk_listener, exitpriority=20)


def _fundable_program(directory, database, funds):
    """Scheduling hint only. The atomic DB preview and lease still authorize every charged call."""
    from scripts.collection_review import read_file
    if funds == 0:
        return False
    configured = False
    for path in sorted(directory.glob("pool-*.json")):
        try:
            raw = json.loads(read_file(path))
            arms = raw["arms"]
            caps = [a["budget_microusd"] for a in arms]
            slots = len(raw["cases"]) * sum(a["repeats"] for a in arms)
            if not caps or not all(type(c) is int and c > 0 for c in caps) or not 1 <= slots <= 4096:
                continue
        except (ValueError, KeyError, OSError, TypeError):
            continue
        configured = True
        if min(caps) > funds:
            continue
        state = database.rpc("collection_pool_pack_state_v1", {"p_batch": path.stem, "p_slots": slots})
        if state not in ("complete", "open", "running"):
            raise ValueError("invalid funded pack state")
        if state == "open" or (state == "complete" and raw.get("successor_seeds")):
            return True
    if not configured:
        raise ValueError("no standing program is configured")
    return False


def run_governed_lanes(
    args,
    lanes,
    *,
    executor=None,
    tick=governed_tick,
    clock=time.monotonic,
    timer_active=_worker_timer_active,
):
    from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
    from multiprocessing import get_context

    if type(lanes) is not int or lanes not in (4, 16, 40, 66):
        raise ValueError("unapproved worker tier")
    limit = args.max_jobs
    args = argparse.Namespace(**{**vars(args), "max_jobs": 1})
    deadline = clock() + 1800
    counts = [0] * lanes
    stopped = set()
    if not timer_active():
        return 0
    if executor is None:
        executor = ProcessPoolExecutor(
            max_workers=lanes, mp_context=get_context("spawn"), initializer=_init_collection_process
        )
    with executor as workers:
        running = {
            workers.submit(tick, args, lane, deadline): lane for lane in range(lanes)
        }
        while running:
            done, _ = wait(running, return_when=FIRST_COMPLETED)
            progress = False
            for future in done:
                lane = running.pop(future)
                try:
                    count = future.result()
                    if type(count) is not int or count not in (0, 1):
                        raise ValueError("invalid governed worker result")
                except Exception as exc:
                    print(
                        json.dumps(
                            {
                                "lane": lane,
                                "status": "lane_stopped",
                                "reason": type(exc).__name__,
                            }
                        ),
                        file=sys.stderr,
                        flush=True,
                    )
                    stopped.add(lane)
                    continue
                counts[lane] += count
                progress |= bool(count)
            if progress and clock() < deadline and timer_active():
                busy = set(running.values())
                for lane in range(lanes):
                    if (
                        lane not in busy
                        and lane not in stopped
                        and counts[lane] < limit
                    ):
                        running[workers.submit(tick, args, lane, deadline)] = lane
    if stopped:
        raise RuntimeError(
            "governed worker cycle had an unconfirmed operation; inspect its journal"
        )
    return sum(counts)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--environment-file", type=Path, help="Protected Arena gateway environment"
    )
    p.add_argument("--batch", type=Path, required=True)
    p.add_argument("--offer-directory", type=Path,
                   help="Private directory of subsequent paid manifests named by their signed offer ID")
    p.add_argument("--prompt", type=Path, required=True)
    p.add_argument("--champion-repo", type=Path)
    p.add_argument("--poet-repo", type=Path)
    p.add_argument("--max-jobs", type=int, default=1)
    p.add_argument("--governed-workers", action="store_true",
                   help="Use the approved database worker tier, epoch budget and route backoff on paid work")
    p.add_argument("--productive-burst", action="store_true",
                   help="Paid service only: admit up to 64 jobs or 30 minutes of new work, and drain on timer stop")
    p.add_argument("--successor-first-model", nargs=2, metavar=("PROGRAM", "MODEL"),
                   help="Lead only future renewable packs of this program with an existing assigned model")
    p.add_argument("--research-case-index", type=Path)
    p.add_argument(
        "--research-stdio",
        action="store_true",
        help="relay a single research assignment to the isolated validator worker",
    )
    p.add_argument("--claim-pack", type=Path)
    p.add_argument("--control-pack", type=Path)
    p.add_argument("--preflight", action="store_true")
    p.add_argument(
        "--rejudge-ready",
        type=UUID,
        nargs=2,
        metavar=("JOB_UUID", "ATTEMPT_UUID"),
        help="Retry unavailable owner judgments for one recorded first live job",
    )
    p.add_argument("--pilot-only", action="store_true", help="Accept only unpaid synthetic qualification jobs")
    p.add_argument("--experiment-inputs", type=Path, help="Protected hash-addressed student sources and split policy")
    p.add_argument("--next-correction", type=Path,
                   help="Optional staged correction pilot to try before the qualification pack")
    p.add_argument(
        "--next-reference",
        type=Path,
        help="Optional private binding from a settled pilot successor to one frozen complete reference",
    )
    args = p.parse_args(argv)
    if args.successor_first_model and (
        not args.offer_directory
        or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", args.successor_first_model[0])
        or args.successor_first_model[1] not in GENERATOR_ROUTES
    ):
        p.error(
            "successor-first-model requires offer-directory, an experiment id and an approved model"
        )
    if args.environment_file:
        from scripts.run_lab_arena_service import load_scoped_environment

        load_scoped_environment(args.environment_file)
    if args.governed_workers and (not args.offer_directory or not args.productive_burst):
        p.error("governed-workers requires the existing paid service burst")
    if args.productive_burst and not args.offer_directory:
        p.error("productive-burst requires offer-directory")
    job_limit = 64 if args.productive_burst else (8 if args.offer_directory else 6)
    if not 1 <= args.max_jobs <= job_limit:
        p.error("invalid job limit for this worker")
    baseline = json.loads(args.batch.read_text())
    if args.offer_directory:
        if (args.preflight or args.pilot_only or args.next_reference or args.next_correction
                or args.research_stdio or args.rejudge_ready or not args.offer_directory.is_dir()):
            p.error("offer-directory requires the bounded paid comparison worker")
        prompt = json.loads(args.prompt.read_text())
        database, _ = evidence_stores()
        limit = args.max_jobs
        args.max_jobs = 1  # Re-select the program and live funds after every actual job.
        if database.rpc("collection_pool_active_v1", {}) is True:
            args.experiment_inputs = (
                args.experiment_inputs
                or args.offer_directory.parent / "experiment-inputs"
            )
            if args.governed_workers:
                args.max_jobs = limit
                # No provider or chain traffic when every authorized limit is empty.
                funds = database.rpc("collection_worker_funding_v1", {})
                if type(funds) is not int or funds<0:
                    raise ValueError("invalid signed pool capacity")
                from lab_arena.collection_chain_access import chain_cooling
                if not chain_cooling(args.offer_directory) and _fundable_program(args.offer_directory, database, funds):
                    run_governed_lanes(args,database.rpc("collection_worker_lanes_v1",{}))
                return 0
            def scan():
                settled = []
                handled = poll_pool_experiments(
                    prompt, args.offer_directory, database,
                    partial(_run_options, args, p, on_pool_settled=settled.append),
                    successor_leader=args.successor_first_model,
                )
                # A clipped, unpayable output is retained. Resume at the ordinary timer
                # rather than immediately funding more potentially unusable work.
                return handled and (not args.productive_burst or settled == ["accepted"])

            poll_funded_burst(limit, scan, productive=args.productive_burst)
        else:
            poll_paid_offers(
                baseline, prompt, args.offer_directory, partial(_run_options, args, p)
            )
        return 0
    if args.next_reference and (
        args.max_jobs != 1
        or not args.pilot_only
        or args.preflight
        or args.rejudge_ready
        or args.research_stdio
        or baseline.get("mode") != "pilot"
        or args.next_reference == args.batch
    ):
        p.error("reference progression requires the bounded one-job pilot worker")
    if args.next_correction:
        if (args.max_jobs != 1 or args.preflight or args.rejudge_ready
                or args.research_stdio or not args.experiment_inputs or args.next_correction == args.batch):
            p.error("correction-first polling requires one job and distinct private batch files")
        from lab_arena import collection_intake as funding
        from lab_arena.collection_experiment import SCHEMA
        from scripts.collection_review import read_file

        if args.next_correction.exists():
            base_mode = baseline.get("mode", "pilot")
            if base_mode == "live" and args.pilot_only:
                p.error("pilot-only polling cannot run live work")
            try:
                funding.correction_terms_hash(base_mode)
            except funding.IntakeError as exc:
                if exc.code != "collection_correction_terms_unavailable":
                    raise
            else:
                admission_attempted = False

                def before_claim():
                    nonlocal admission_attempted
                    admission_attempted = True

                try:
                    candidate = json.loads(read_file(args.next_correction))
                    if candidate.get("schema") != SCHEMA or candidate.get("mode") != base_mode:
                        raise ValueError("staged correction differs from the approved consent")
                    funding.require_correction_grant(base_mode, candidate.get("terms_sha256"))
                    if _run_options(args, p, candidate, before_claim=before_claim):
                        return 0
                except Exception as exc:
                    # A faulty optional pack must not starve the qualification queue.
                    # Once a claim is sent, its outcome may be ambiguous on disconnect.
                    if admission_attempted:
                        raise
                    print(json.dumps({"status": "correction_held", "reason": type(exc).__name__}),
                          file=sys.stderr, flush=True)
    if args.next_reference and args.next_reference.exists():
        from scripts.collection_reference_followup import reference_for_poll
        from lab_arena.store import ArenaStoreError
        from scripts.collection_review import read_file

        try:
            database, objects = evidence_stores()
            baseline = reference_for_poll(
                baseline,
                json.loads(read_file(args.prompt)),
                args.next_reference,
                database,
                objects,
            )
            from scripts.collection_qualification_followup import qualification_for_poll

            if baseline.get("route_witness_sha256"):
                baseline, controls = qualification_for_poll(
                    baseline,
                    json.loads(read_file(args.prompt)),
                    args.next_reference.parent,
                    database,
                    objects,
                )
                if controls is not None:
                    args.control_pack = controls
        except (ValueError, OSError, KeyError, TypeError, ArenaStoreError) as exc:
            print(
                json.dumps(
                    {"status": "reference_followup_held", "reason": type(exc).__name__}
                ),
                file=sys.stderr,
                flush=True,
            )
    _run_options(args, p, baseline)
    return 0


def _run_options(args, p, raw, *, before_claim=None, on_pool_settled=None, epoch_for_tick=None):
    """Return how many new jobs were executed, for bounded pilot polling."""
    prompt = json.loads(args.prompt.read_text())
    from lab_arena import collection_experiment as experiments
    experiment = None
    if raw.get("schema") == experiments.SCHEMA:
        if args.research_stdio or args.rejudge_ready or args.claim_pack or args.control_pack or args.research_case_index:
            p.error("correction experiments require synthetic inputs only")
        experiment = experiments.load_experiment(raw, prompt)
        policy = experiment.policy
        scenarios = [c["scenario_id"] for c in raw["cases"]]
    else:
        policy = load_batch(raw, prompt)
        scenarios = raw["scenario_ids"]
    judge_key = os.getenv("LEADPOET_COLLECTION_JUDGE_KEY") or os.getenv(
        "LAB_ARENA_OPENROUTER_API_KEY"
    )
    if args.rejudge_ready:
        if (
            (args.pilot_only and not policy.pilot)
            or args.preflight
            or args.research_stdio
            or os.getenv("LAB_ARENA_COLLECTION_MODE")
            != ("pilot" if policy.pilot else "live")
            or not live_finney_arena()
            or not judge_key
        ):
            p.error(
                "owner adjudication requires the original Finney/71 mode, batch and owner credential"
            )
        database, objects = evidence_stores()
        job_id, attempt_id = map(str, args.rejudge_ready)
        with closing(HttpxProviderTransport()) as transport:
            rejudge_ready(
                database,
                objects,
                transport,
                raw,
                policy,
                prompt,
                judge_key,
                job_id,
                attempt_id,
            )
        print(json.dumps({"job": job_id, "status": "verified"}))
        return 0
    standing = bool(experiment and experiment.batch_id.startswith("pool-"))
    if not args.champion_repo or not args.poet_repo:
        p.error("running generators requires --champion-repo and --poet-repo")
    if args.pilot_only and (
        raw["mode"] != "pilot" or not policy.pilot or not policy.qualification
        or any(s.startswith("research-") for s in scenarios)
    ):
        p.error("the pilot worker requires a synthetic qualification batch")
    controls = None if experiment else ControlPack(
        raw,
        (
            json.loads(args.control_pack.read_text())
            if args.control_pack
            else {"schema": PACK_SCHEMA, "cases": {}}
        ),
    )
    claim_pack = None
    if args.claim_pack or "claim_pack_sha256" in raw:
        from lab_arena.collection_claims import verify_pack
        from scripts.collection_review import read_file

        if not args.claim_pack:
            p.error("frozen claim report pack is required")
        claim_pack = verify_pack(
            json.loads(read_file(args.claim_pack)),
            Path(os.environ["LAB_ARENA_COLLECTION_CLAIM_PUBLIC_KEY_FILE"]).read_bytes(),
            raw["claim_pack_sha256"],
        )
    research_index = None
    if any(s.startswith("research-") for s in scenarios):
        if not args.research_case_index:
            p.error("supply the committed research case index")
        research_index = json.loads(args.research_case_index.read_text())
        if judge.digest(research_index) != raw["research_index_sha256"] or not {
            s for s in scenarios if s.startswith("research-")
        } <= set(research_index):
            raise ValueError("research case index differs from frozen pack")
        if not args.preflight and (not args.research_stdio or args.max_jobs != 1):
            raise ValueError(
                "research requires one authenticated validator relay per invocation"
            )
    agent, builder = (
        experiments.source_revisions(raw) if experiment else comparison_revisions(raw)
    )
    source_args = (
        {"corrections": True, "builder_revision": builder}
        if experiment or "capture_version" in raw
        else {}
    )
    with workflow_sources(args.champion_repo, args.poet_repo, **source_args) as sources:
        experiment_inputs = experiments.read_inputs(experiment, args.experiment_inputs, sources, prompt=prompt) if experiment else None
        try:
            for sid in ([] if experiment else scenarios):
                if not sid.startswith("research-"):
                    pack_split(raw, sources[1], sid)
        except ComparisonSplitHold as exc:
            if args.preflight:
                raise
            return report_split_hold(policy, exc)
        if args.preflight:
            print(
                json.dumps(
                    {
                        "batch": policy.batch_id,
                        "models": (
                            len({a["model"] for a in raw["arms"]})
                            if experiment
                            else len(raw["budgets_microusd"])
                        ),
                        "scenarios": len(scenarios),
                        "champion": agent,
                        "builder": builder,
                    }
                )
            )
            return 0
        mode = os.environ.get("LAB_ARENA_COLLECTION_MODE", "off")
        if (
            mode != raw["mode"]
            or not live_finney_arena()
        ):
            raise ValueError("collection environment/scope mismatch")
        db, objects = evidence_stores()
        cm = CredentialManager(kms_key_id=os.environ["LAB_ARENA_CREDENTIAL_KMS_KEY_ID"])
        config = chain.ArenaChainConfig(
            os.environ["LAB_ARENA_CHAIN_ENDPOINT"], 71, "finney", 30
        )
        governed = standing and getattr(args, "governed_workers", False)
        with ExitStack() as stack:
            if governed:
                from lab_arena.collection_chain_access import lane_chain
                stack.callback(db.close)
                client = stack.enter_context(lane_chain(config, args.offer_directory))
            else:
                client = stack.enter_context(closing(chain.ArenaChain(config, chain.connect_substrate(config))))
            cutover = chain.load_arena_cutover()
            research = None
            protocol_out = sys.stdout.buffer
            if research_index is not None:
                from lab_arena.collection_research import (
                    FrozenResearchCases,
                    RemoteResearch,
                )
                from lab_arena.store import ArenaStore

                cases = FrozenResearchCases(ArenaStore(db), objects, research_index)

                @contextmanager
                def open_channel(job):
                    yield sys.stdin.buffer, protocol_out

                from lab_arena.collection_research_capture import ResearchGrant
                research = RemoteResearch(open_channel, cases.load, cases.grade, grant=ResearchGrant.deployed(), objects=objects)
            if not governed:
                chain.current_settlement_epoch(client, cutover)
            transport = stack.enter_context(closing(HttpxProviderTransport()))
            if args.research_stdio:
                stack.enter_context(redirect_stdout(sys.stderr))
            if experiment:
                if not standing and not judge_key:
                    raise ValueError("private claim-reader credential unavailable")
                intake = CollectionIntake(db, cm, mode)
                price = (lambda model: endpoint_prices(experiment.routes,models=[model])) if standing else prices_for_assignment(experiment.routes,policy)
                if standing and getattr(args, "governed_workers", False):
                    from lab_arena.collection_governor import GovernedIntake, prepare_epoch
                    from scripts.prepare_collection_experiment import lock_pool_directory
                    epoch_id = (epoch_for_tick or prepare_epoch)(client,cutover,db,partial(lock_pool_directory,args.offer_directory))
                    intake = GovernedIntake(db,cm,epoch_id,experiment.routes,lambda: chain.current_settlement_epoch(client,cutover))
                    price = intake.prices
                completed = experiments.run_experiment(
                    spec=experiment, sources=sources, inputs=experiment_inputs,
                    intake=intake, database=db, objects=objects, transport=transport, pricing=price,
                    owner_lookup=lambda hk: resolve_finalized_owner(client, hk), judge_key=judge_key,
                    max_jobs=args.max_jobs, epoch=lambda: chain.current_settlement_epoch(client, cutover),
                    before_claim=before_claim, prompt=prompt, on_pool_settled=on_pool_settled,
                    on_pool_failure=(partial(stage_failed_pool_continuation, database=db, objects=objects,
                                             directory=args.offer_directory, inputs=args.experiment_inputs, prompt=prompt)
                                     if standing and args.offer_directory else None),
                )
            else:
                completed = run_cohort(
                    raw=raw,
                    policy=policy,
                    prompt=prompt,
                    intake=CollectionIntake(db, cm, mode),
                    database=db,
                    objects=objects,
                    transport=transport,
                    pricing=(
                        (lambda model: endpoint_prices(raw["routes"], models=[model]))
                        if "payment_rule" in raw else prices_for_assignment(raw["routes"], policy)
                    ),
                    epoch=lambda: chain.current_settlement_epoch(client, cutover),
                    sources=sources,
                    owner_lookup=lambda hk: resolve_finalized_owner(client, hk),
                    judge_key=judge_key,
                    max_jobs=args.max_jobs,
                    research=research,
                    controls=controls,
                    claim_pack=claim_pack,
                    recover_idle_pilot=args.pilot_only and args.max_jobs == 1,
                    before_claim=before_claim,
                )
            if completed == 0 and not args.research_stdio:
                print(json.dumps({"batch": policy.batch_id, "status": "idle"}), flush=True)
            if args.research_stdio and completed == 0:
                from lab_arena.collection_research import send

                send(protocol_out, {"type": "idle"})
    return completed


if __name__ == "__main__":
    raise SystemExit(main())
