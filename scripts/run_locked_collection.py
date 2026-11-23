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

from lab_arena import chain, collection_judge as judge
from lab_arena.broker import HttpxProviderTransport
from lab_arena.collection_prices import endpoint_prices
from lab_arena.collection_broker import CollectionBroker, VERSIONS
from lab_arena.collection_intake import MODELS, CollectionIntake
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
    judges = {"anthropic/claude-fable-5.2", "openai/gpt-6.2-astra"}
    if set(raw["routes"]) != set(MODELS) | judges or not all(
        re.fullmatch(r"[a-z0-9][a-z0-9._/-]{0,127}", v) for v in raw["routes"].values()
    ):
        raise ValueError("every generator and judge needs one pinned provider route")
    bundle_hash = judge.digest(
        {
            "manifest": raw,
            "prompt": judge.digest(prompt),
            "champion": CHAMPION_REV,
            "builder": POET_REV,
            "model_revisions": VERSIONS,
        }
    )
    thresholds = raw.get("thresholds", {})
    if not isinstance(thresholds, dict) or set(thresholds) - {
        "accept_sum",
        "salvage_min_steps",
        "agreement_percent",
    }:
        raise ValueError("unknown batch threshold")
    policy = judge.BatchPolicy(
        raw["name"] + "-" + bundle_hash[:24],
        raw["seed"],
        judge.digest(prompt),
        pilot=raw["mode"] == "pilot",
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
    ceiling = dict(raw["budgets_microusd"])
    fees = dict.fromkeys(MODELS, 0)
    if not judge_key:
        raise ValueError("private verifier credential unavailable")
    if not policy.pilot:
        schedule = database.rpc("collection_reward_schedule_v1", {})
        if not schedule or epoch() < schedule["first_epoch"] - 1:
            raise ValueError("payout cutover not ready")
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
    return completed


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--environment-file", type=Path, help="Protected Arena gateway environment"
    )
    p.add_argument("--batch", type=Path, required=True)
    p.add_argument("--prompt", type=Path, required=True)
    p.add_argument("--champion-repo", type=Path, required=True)
    p.add_argument("--poet-repo", type=Path, required=True)
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
    args = p.parse_args(argv)
    if args.environment_file:
        from scripts.run_lab_arena_service import load_scoped_environment

        load_scoped_environment(args.environment_file)
    if not 1 <= args.max_jobs <= 6:
        p.error("select 1 to 6 jobs per invocation")
    raw = json.loads(args.batch.read_text())
    prompt = json.loads(args.prompt.read_text())
    policy = load_batch(raw, prompt)
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
            or os.environ.get("LAB_ARENA_NETWORK") != "finney"
            or os.environ.get("LAB_ARENA_NETUID") != "71"
        ):
            raise ValueError("collection environment/scope mismatch")
        db = PostgrestTransport(
            os.environ["LAB_ARENA_SUPABASE_URL"],
            service_key=os.environ["LAB_ARENA_SERVICE_KEY"],
        )
        cm = CredentialManager(kms_key_id=os.environ["LAB_ARENA_CREDENTIAL_KMS_KEY_ID"])
        config = chain.ArenaChainConfig(
            os.environ["LAB_ARENA_CHAIN_ENDPOINT"], 71, "finney", 30
        )
        client = chain.ArenaChain(config, chain.connect_substrate(config))
        cutover = chain.load_arena_cutover()
        objects = S3ObjectStore(
            os.environ["LAB_ARENA_BUCKET"],
            prefix=os.getenv("LAB_ARENA_OBJECT_PREFIX", ""),
        )
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
                judge_key=os.getenv("LEADPOET_COLLECTION_JUDGE_KEY")
                or os.environ.get("LAB_ARENA_OPENROUTER_API_KEY"),
                max_jobs=args.max_jobs,
                research=research,
                controls=controls,
                claim_pack=claim_pack,
            )
            if args.research_stdio and completed == 0:
                from lab_arena.collection_research import send

                send(protocol_out, {"type": "idle"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
