"""Run one bounded cohort on the trusted collection host. All network calls are metered."""

from __future__ import annotations

import argparse
from lab_arena.collection_controls import ControlPack, PACK_SCHEMA
import json
import os
import re
import sys
from contextlib import closing, contextmanager, redirect_stdout, ExitStack
from pathlib import Path
from uuid import UUID

from lab_arena import chain, collection_judge as judge
from lab_arena.broker import HttpxProviderTransport
from lab_arena.collection_prices import endpoint_prices
from lab_arena.collection_broker import (
    CollectionBroker,
    REFERENCE_MODEL,
    REFERENCE_ROUTE,
    batch_versions,
)
from lab_arena.collection_intake import (
    GENERATOR_ROUTES, LEGACY_COMPARISON_TERMS_HASH, MODELS, SYNTHETIC_TRAINING_TERMS_HASH,
    CollectionIntake, live_finney_arena,
)
from lab_arena.collection_worker import process_job
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
        "control_artifacts",
        "claim_pack_sha256",
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
    if "terms_sha256" in raw and raw["terms_sha256"] != SYNTHETIC_TRAINING_TERMS_HASH:
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
    if set(raw["budgets_microusd"]) != set(MODELS) or any(
        type(n) is not int or not 0 < n <= 10**10
        for n in raw["budgets_microusd"].values()
    ):
        raise ValueError("every model needs a bounded positive budget")
    revisions = batch_versions(raw["routes"])
    qualification = set(judge.QUALIFICATION_JUDGES) <= revisions.keys()
    certificate = raw.get("panel_qualification_sha256")
    if qualification and raw["mode"] == "live":
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
    if "route_witness_sha256" in raw:
        from lab_arena.collection_route_witness import WITNESS
        from lab_arena.collection_generation import CAPTURE_V2

        if (
            successor
            or full_witness != WITNESS
            or raw["mode"] != "pilot"
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
    policy = judge.BatchPolicy(
        (
            "rs-" + bundle_hash
            if successor
            else (
                "rq-" + bundle_hash
                if full_witness
                else raw["name"] + "-" + bundle_hash[:24]
            )
        ),
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

    evidence = ready_launch_evidence(pending, objects, policy, approved)
    if not complete_launch_verdict(evidence["assessment"]):
        raise ValueError("first live collection job awaits owner verification")
    anchor = cutover_anchor(database, launch_job=pending)
    result = database.rpc(
        "collection_launch_finish_v1", {"p_job": pending["job_id"], "p_anchor": anchor}
    )
    if result.get("status") != "accepted":
        raise ValueError("collection launch acceptance unavailable")


def rejudge_ready(
    database, objects, transport, raw, policy, prompt, judge_key, job_id, attempt_id
):
    """Retry only an unavailable owner verdict against the exact stored live execution."""
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
        raise ValueError(
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
        return
    current = database.rpc("collection_reward_schedule_v1", {}).get("pending") or {}
    if (
        current.get("job_id") != job_id
        or current.get("artifact_sha256") != link["sha256"]
    ):
        raise ValueError("revised launch evidence is not current")
    finish_launch(database, current, objects, policy, approved)


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
    certificate = raw.get("panel_qualification_sha256")
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
    if not judge_key:
        raise ValueError("private verifier credential unavailable")
    if not policy.pilot:
        schedule = database.rpc("collection_reward_schedule_v1", {})
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
        (family,) = {
            ("research" if sid.startswith("research-") else sid[0])
            for sid in raw["scenario_ids"]
        }
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
        job = intake.claim(
            policy.batch_id, raw["scenario_ids"], agent, fees, ceiling,
            terms_hash=raw.get("terms_sha256", LEGACY_COMPARISON_TERMS_HASH),
        )
        if not job:
            break
        try:
            if owner_lookup(job["miner_hotkey"]).coldkey != job["owner_coldkey"]:
                raise ValueError("collection funding owner changed")
            model_broker = CollectionBroker(
                intake,
                database,
                transport,
                job,
                pricing,
                routes=raw["routes"],
                judge_key=judge_key,
            )
            model_broker.panel_qualification = raw.get("panel_qualification_sha256")
            a = assignment(job)
            manifest = process_job(
                job=job,
                assignment=judge.assign_judges(policy, [a])[a["run_id"]],
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


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--environment-file", type=Path, help="Protected Arena gateway environment"
    )
    p.add_argument("--batch", type=Path, required=True)
    p.add_argument("--prompt", type=Path, required=True)
    p.add_argument("--champion-repo", type=Path)
    p.add_argument("--poet-repo", type=Path)
    p.add_argument("--max-jobs", type=int, default=1)
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
    if args.environment_file:
        from scripts.run_lab_arena_service import load_scoped_environment

        load_scoped_environment(args.environment_file)
    if not 1 <= args.max_jobs <= 6:
        p.error("select 1 to 6 jobs per invocation")
    baseline = json.loads(args.batch.read_text())
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


def _run_options(args, p, raw, *, before_claim=None):
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
        experiment_inputs = experiments.read_inputs(experiment, args.experiment_inputs, sources) if experiment else None
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
                            else len(MODELS)
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
        client = chain.ArenaChain(config, chain.connect_substrate(config))
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
        with ExitStack() as stack:
            stack.callback(client.close)
            # Confirm the finalized genesis and subnet before even an unpaid lease.
            chain.current_settlement_epoch(client, cutover)
            transport = stack.enter_context(closing(HttpxProviderTransport()))
            if args.research_stdio:
                stack.enter_context(redirect_stdout(sys.stderr))
            if experiment:
                if not judge_key:
                    raise ValueError("private claim-reader credential unavailable")
                active_models = {arm["model"] for arm in raw["arms"]} | set(judge.QUALIFICATION_JUDGES)
                completed = experiments.run_experiment(
                    spec=experiment, sources=sources, inputs=experiment_inputs,
                    intake=CollectionIntake(db, cm, mode), database=db, objects=objects,
                    transport=transport, pricing=endpoint_prices(experiment.routes, models=sorted(active_models)),
                    owner_lookup=lambda hk: resolve_finalized_owner(client, hk), judge_key=judge_key,
                    max_jobs=args.max_jobs, epoch=lambda: chain.current_settlement_epoch(client, cutover),
                    before_claim=before_claim, prompt=prompt,
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
                    pricing=endpoint_prices(raw["routes"]),
                    epoch=lambda: chain.current_settlement_epoch(client, cutover),
                    sources=sources,
                    owner_lookup=lambda hk: resolve_finalized_owner(client, hk),
                    judge_key=judge_key,
                    max_jobs=args.max_jobs,
                    research=research,
                    controls=controls,
                    claim_pack=claim_pack,
                )
            if completed == 0 and not args.research_stdio:
                print(json.dumps({"batch": policy.batch_id, "status": "idle"}), flush=True)
            if args.research_stdio and completed == 0:
                from lab_arena.collection_research import send

                send(protocol_out, {"type": "idle"})
    return completed


if __name__ == "__main__":
    raise SystemExit(main())
