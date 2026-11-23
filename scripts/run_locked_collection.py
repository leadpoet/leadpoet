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
from lab_arena.collection_broker import CollectionBroker, batch_versions
from lab_arena.collection_intake import GENERATOR_ROUTES, MODELS, CollectionIntake, live_finney_arena
from lab_arena.collection_worker import process_job
from lab_arena.collection_workflow import (
    CHAMPION_REV,
    POET_REV,
    build_scenario,
    workflow_sources,
)
from lab_arena.credentials import CredentialManager
from lab_arena.owner_admission import resolve_finalized_owner
from lab_arena.service import S3ObjectStore
from lab_arena.store import PostgrestTransport


def load_batch(raw, prompt):
    if set(raw) - {
        "research_index_sha256",
        "thresholds",
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
    if qualification and raw["mode"] != "pilot":
        raise ValueError("candidate judges require a qualification pilot")
    if not all(
        re.fullmatch(r"[a-z0-9][a-z0-9._/-]{0,127}", v) for v in raw["routes"].values()
    ):
        raise ValueError("every generator and judge needs one pinned provider route")
    if any(
        raw["routes"][model] != route.tag for model, route in GENERATOR_ROUTES.items()
    ):
        raise ValueError("unverified generator route")
    bundle_hash = judge.digest(
        {
            "manifest": raw,
            "prompt": judge.digest(prompt),
            "champion": CHAMPION_REV,
            "builder": POET_REV,
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
        raw["name"] + "-" + bundle_hash[:24],
        raw["seed"],
        judge.digest(prompt),
        pilot=raw["mode"] == "pilot",
        **versions,
        **thresholds,
    )
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

    if (
        pending["batch_id"] != policy.batch_id
        or policy.pilot
        or pending["harness_commit"] != CHAMPION_REV
        or approved["harness_commit"] != CHAMPION_REV
        or approved["builder_commit"] != POET_REV
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
        expected_builder=POET_REV,
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

    schedule = database.rpc("collection_reward_schedule_v1", {})
    pending = schedule.get("pending") or {}
    if (
        not schedule.get("staged")
        or pending.get("status") != "ready"
        or pending.get("job_id") != job_id
    ):
        raise ValueError("requested staged job is not ready for owner adjudication")
    approved = schedule["pilot_evidence"]
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
    replacement = judge.judge_run(
        policy,
        {"run_id": job_id, "primary": primary, "overlap": previous["overlap"]},
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
    current = database.rpc("collection_reward_schedule_v1", {}).get("pending") or {}
    if (
        current.get("job_id") != job_id
        or current.get("artifact_sha256") != link["sha256"]
    ):
        raise ValueError("revised launch evidence is not current")
    finish_launch(database, current, objects, policy, approved)


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
    staged = False
    ceiling = dict(raw["budgets_microusd"])
    fees = dict.fromkeys(MODELS, 0)
    if not judge_key:
        raise ValueError("private verifier credential unavailable")
    if not policy.pilot:
        schedule = database.rpc("collection_reward_schedule_v1", {})
        if not schedule or epoch() < schedule["first_epoch"] - 1:
            raise ValueError("payout cutover not ready")
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
            approved["harness_commit"] != CHAMPION_REV
            or approved["builder_commit"] != POET_REV
            or approved["judge_prompt_sha256"] != policy.prompt_sha256
        ):
            raise ValueError("live arm differs from the measured pilot")
        fees = approved["prices"][family]
        if approved["budgets"][family] != ceiling:
            raise ValueError("live cap differs from pilot")
    for sid in raw["scenario_ids"]:
        if sid.startswith("research-"):
            if research is None:
                raise ValueError("research runner and label evidence required")
        else:
            build_scenario(sources[1], sid)
    completed = 0
    for _ in range(max_jobs):
        job = intake.claim(
            policy.batch_id, raw["scenario_ids"], CHAMPION_REV, fees, ceiling
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
    args = p.parse_args(argv)
    if args.environment_file:
        from scripts.run_lab_arena_service import load_scoped_environment

        load_scoped_environment(args.environment_file)
    if not 1 <= args.max_jobs <= 6:
        p.error("select 1 to 6 jobs per invocation")
    raw = json.loads(args.batch.read_text())
    prompt = json.loads(args.prompt.read_text())
    policy = load_batch(raw, prompt)
    judge_key = os.getenv("LEADPOET_COLLECTION_JUDGE_KEY") or os.getenv(
        "LAB_ARENA_OPENROUTER_API_KEY"
    )
    if args.rejudge_ready:
        if (
            args.pilot_only
            or args.preflight
            or args.research_stdio
            or policy.pilot
            or os.getenv("LAB_ARENA_COLLECTION_MODE") != "live"
            or not live_finney_arena()
            or not judge_key
        ):
            p.error(
                "owner adjudication requires the original live Finney/71 batch and owner credential"
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
        not policy.pilot or not policy.qualification
        or any(s.startswith("research-") for s in raw["scenario_ids"])
    ):
        p.error("the pilot worker requires a synthetic qualification batch")
    controls = ControlPack(
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
    if any(s.startswith("research-") for s in raw["scenario_ids"]):
        if not args.research_case_index:
            p.error("supply the committed research case index")
        research_index = json.loads(args.research_case_index.read_text())
        if judge.digest(research_index) != raw["research_index_sha256"] or not {
            s for s in raw["scenario_ids"] if s.startswith("research-")
        } <= set(research_index):
            raise ValueError("research case index differs from frozen pack")
        if not args.preflight and (not args.research_stdio or args.max_jobs != 1):
            raise ValueError(
                "research requires one authenticated validator relay per invocation"
            )
    with workflow_sources(args.champion_repo, args.poet_repo) as sources:
        for sid in raw["scenario_ids"]:
            if not sid.startswith("research-"):
                build_scenario(sources[1], sid)
        if args.preflight:
            print(
                json.dumps(
                    {
                        "batch": policy.batch_id,
                        "models": len(MODELS),
                        "scenarios": len(raw["scenario_ids"]),
                        "champion": CHAMPION_REV,
                        "builder": POET_REV,
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

            research = RemoteResearch(open_channel, cases.load, cases.grade)
        with ExitStack() as stack:
            stack.callback(client.close)
            # Confirm the finalized genesis and subnet before even an unpaid lease.
            chain.current_settlement_epoch(client, cutover)
            transport = stack.enter_context(closing(HttpxProviderTransport()))
            if args.research_stdio:
                stack.enter_context(redirect_stdout(sys.stderr))
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
