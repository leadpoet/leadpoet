"""Frozen experimental arms and repeat slots in the existing collection queue."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import partial
import json
import re
import base64
import os
from pathlib import Path
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from cryptography.hazmat.primitives.serialization import load_pem_public_key

from lab_arena import collection_judge as judge
from lab_arena.collection_claims import CLAIM_RUBRIC
from lab_arena.collection_intake import (
    GENERATOR_ROUTES,
    SYNTHETIC_SCENARIO,
    MAX_MICROUSD,
)
from lab_arena.collection_pool import POOL_PAYMENT_RULE, POOL_TERMS_HASH
from lab_arena.collection_workflow import store_artifact
from lab_arena.collection_journal import WorkflowJournal
from lab_arena.collection_generation import (
    CAPTURE_V2,
    MAX_RUN_TOKENS,
    OutputLimitReached,
    output_charge,
)

SCHEMA = "collection.experiment.v1"
SHA = re.compile(r"sha256:[0-9a-f]{64}")
NAME = re.compile(r"[a-zA-Z0-9_-]{1,64}")
DEFAULT_MAX_SLOTS = 12
MAX_EXPERIMENT_SLOTS = 4096


def _hash(value):
    return "sha256:" + judge.digest(value)


def _integer(value, minimum, maximum):
    return type(value) is int and minimum <= value <= maximum


@dataclass(frozen=True)
class Experiment:
    raw: dict
    settings_hash: str
    batch_id: str
    routes: dict
    slots: tuple
    policy: judge.BatchPolicy

    @property
    def pilot(self):
        return self.raw["mode"] == "pilot"


def source_revisions(raw=None):
    from lab_arena.collection_workflow import experiment_revisions

    return experiment_revisions(c["scenario_id"] for c in (raw or {}).get("cases", []))


def load_experiment(raw, prompt):
    from lab_arena.collection_claims import protocol_of

    protocol_of(raw)
    pool = raw.get("payment_rule") == POOL_PAYMENT_RULE
    if "payment_rule" in raw and not pool:
        raise ValueError("unknown experiment payment rule")

    if (
        set(raw)
        - (
            {
                "payment_rule",
                "routes",
                "successor_seeds",
                "correction_model",
                "auto_from_program",
            }
            if pool
            else set()
        )
        - {
            "claim_protocol",
            "min_pricing_runs",
            "pricing_pilot",
            "max_slots",
            "provenance_sha256",
            "capture_version",
        }
        != {
            "schema",
            "id",
            "mode",
            "question",
            "pass_mark",
            "budget_microusd",
            "arms",
            "cases",
            "split_sha256",
            "terms_sha256",
        }
        or ("capture_version" in raw and raw["capture_version"] != CAPTURE_V2)
        or raw["schema"] != SCHEMA
        or raw["mode"] not in {"pilot", "live"}
    ):
        raise ValueError("invalid correction pilot manifest")
    if (
        not NAME.fullmatch(raw["id"])
        or not all(
            isinstance(raw[k], str) and 1 <= len(raw[k].strip()) <= 1024
            for k in ("question", "pass_mark")
        )
        or not all(SHA.fullmatch(raw[k]) for k in ("split_sha256", "terms_sha256"))
    ):
        raise ValueError("missing experiment question, pass mark, split or consent")
    if pool and (
        raw["mode"] != "live"
        or raw["terms_sha256"] != POOL_TERMS_HASH
        or raw.get("capture_version") != CAPTURE_V2
        or "pricing_pilot" in raw
        or "min_pricing_runs" in raw
    ):
        raise ValueError(
            "standing experiments require their own live consent and capture"
        )
    cases, arms = raw["cases"], raw["arms"]
    slot_limit = raw.get("max_slots", DEFAULT_MAX_SLOTS)
    if not _integer(slot_limit, 1, MAX_EXPERIMENT_SLOTS) or not all(
        isinstance(items, list) and 1 <= len(items) <= slot_limit
        for items in (cases, arms)
    ):
        raise ValueError("invalid experiment size")
    for case in cases:
        if (
            not isinstance(case, dict)
            or set(case) not in ({"scenario_id"}, {"scenario_id", "source_sha256"})
            or not isinstance(case["scenario_id"], str)
            or not (
                re.fullmatch(r"(?:P7|P8|P9|P10)-s[1-9][0-9]{0,7}", case["scenario_id"])
                if pool
                else SYNTHETIC_SCENARIO.fullmatch(case["scenario_id"])
            )
            or (
                "source_sha256" in case
                and (
                    not isinstance(case["source_sha256"], str)
                    or not SHA.fullmatch(case["source_sha256"])
                )
            )
        ):
            raise ValueError("invalid experiment source")
    source_ids = [case["source_sha256"] for case in cases if "source_sha256" in case]
    native_ids = [case["scenario_id"] for case in cases if "source_sha256" not in case]
    if len(set(source_ids)) != len(source_ids) or len(set(native_ids)) != len(
        native_ids
    ):
        raise ValueError("duplicate experiment case; use the arm repeat count")
    if (not pool and bool(source_ids) != ("provenance_sha256" in raw)) or (
        "provenance_sha256" in raw
        and (not source_ids or not SHA.fullmatch(str(raw["provenance_sha256"])))
    ):
        raise ValueError("student sources require their original provenance")
    caps = {}
    arm_names = set()
    for arm in arms:
        if set(arm) - {"max_run_tokens"} != {
            "id",
            "role",
            "model",
            "repeats",
            "max_steps",
            "max_tokens",
            "reasoning_effort",
            "capture_reasoning",
            "budget_microusd",
        }:
            raise ValueError("invalid experiment arm fields")
        model = arm["model"]
        if (
            not NAME.fullmatch(arm["id"])
            or arm["id"] in arm_names
            or arm["role"] not in {"expert", "continuation"}
            or model not in GENERATOR_ROUTES
            or not _integer(arm["repeats"], 1, slot_limit)
            or not _integer(arm["max_steps"], 1, 90)
            or not _integer(arm["max_tokens"], 1, 8192)
            or not _integer(
                arm.get("max_run_tokens", arm["max_tokens"]), 1, MAX_RUN_TOKENS
            )
            or not _integer(arm["budget_microusd"], 1, MAX_MICROUSD)
            or caps.setdefault(model, arm["budget_microusd"]) != arm["budget_microusd"]
            or arm["reasoning_effort"] not in {"low", "medium", "high"}
            or type(arm["capture_reasoning"]) is not bool
            or (pool and (not arm["capture_reasoning"] or "max_run_tokens" not in arm))
        ):
            raise ValueError("invalid or unapproved experiment arm")
        arm_names.add(arm["id"])
    if native_ids and any(arm["role"] == "continuation" for arm in arms):
        raise ValueError(
            "a continuation requires a verified student failure for every case"
        )
    if len(cases) * sum(arm["repeats"] for arm in arms) > slot_limit:
        raise ValueError("experiment exceeds its configured slot limit")
    # Present the same task to all arms before advancing to the next case.
    slots = tuple(
        (case, arm, repeat)
        for case in cases
        for repeat in range(1, max(a["repeats"] for a in arms) + 1)
        for arm in arms
        if repeat <= arm["repeats"]
    )
    liability = sum(a["budget_microusd"] for _, a, _ in slots)
    if (
        not _integer(raw["budget_microusd"], 1, MAX_MICROUSD)
        or liability > raw["budget_microusd"]
    ):
        raise ValueError("experiment exceeds canary slots or budget")
    if "min_pricing_runs" in raw and (
        raw["mode"] != "pilot" or not _integer(raw["min_pricing_runs"], 1, len(slots))
    ):
        raise ValueError("pricing requires a predeclared achievable sample size")
    if (raw["mode"] == "live" and not pool) or "min_pricing_runs" in raw:
        if (
            len(arms) != 1
            or arms[0]["role"] != "continuation"
            or len({c["scenario_id"].split("-s")[0] for c in cases}) != 1
        ):
            raise ValueError(
                "paid calibration requires one comparable correction class"
            )
    if raw["mode"] == "live" and not pool:
        if not isinstance(raw.get("pricing_pilot"), str) or not re.fullmatch(
            r"xp-[0-9a-f]{64}", raw["pricing_pilot"]
        ):
            raise ValueError("live corrections require an approved pricing pilot")
    elif "pricing_pilot" in raw:
        raise ValueError("a pilot cannot spend an approved live tariff")
    routes = {m: r.tag for m, r in GENERATOR_ROUTES.items()}
    routes.update({m: r.route for m, r in judge.QUALIFICATION_JUDGES.items()})
    if pool:
        from lab_arena.collection_broker import REFERENCE_MODEL, REFERENCE_ROUTE
        from lab_arena.collection_quality import require_release_partition

        require_release_partition(raw)
        selected = raw.get("routes")
        if (
            not isinstance(selected, dict)
            or set(selected) != set(caps)
            or not all(
                route == GENERATOR_ROUTES[m].tag
                or (m, route) == (REFERENCE_MODEL, REFERENCE_ROUTE)
                for m, route in selected.items()
            )
        ):
            raise ValueError(
                "standing experiment routes differ from approved pinned endpoints"
            )
        routes.update(selected)
        if "correction_model" in raw:
            if (
                source_ids
                or len(selected) < 2
                or raw["correction_model"] not in selected
                or any(a["role"] != "expert" for a in arms)
            ):
                raise ValueError(
                    "automatic corrections require a distinct approved fresh expert"
                )
        if "auto_from_program" in raw:
            if (
                not isinstance(raw["auto_from_program"], str)
                or not re.fullmatch(r"[0-9a-f]{64}", raw["auto_from_program"])
                or len(source_ids) != 1
                or len(slots) != 1
                or arms[0]["role"] != "continuation"
                or raw.get("successor_seeds")
                or "correction_model" in raw
            ):
                raise ValueError("invalid automatic continuation")

        if "successor_seeds" in raw:
            # A rolling full-task pack advances exactly one disjoint seed range. Private
            # student checkpoint experiments can never silently turn into new cases.
            seeds = [int(s.split("-s")[1]) for s in native_ids]
            if (
                raw["successor_seeds"] is not True
                or source_ids
                or not seeds
                or seeds != list(range(seeds[0], seeds[0] + len(seeds)))
                or seeds[0] < 300
            ):
                raise ValueError("invalid renewable training seed range")
    agent_revision, builder_revision = source_revisions(raw)
    identity = judge.digest(
        {
            "manifest": raw,
            "agent": agent_revision,
            "builder": builder_revision,
            "models": {m: list(GENERATOR_ROUTES[m]) for m in caps},
            "readers": {m: list(r) for m, r in judge.QUALIFICATION_JUDGES.items()},
            "claim_reader_protocol": CLAIM_RUBRIC,
            "prompt": judge.digest(prompt),
        }
    )
    batch = ("pool-" if pool else "xp-") + identity
    policy = judge.BatchPolicy(
        batch,
        identity,
        judge.digest(prompt),
        pilot=raw["mode"] == "pilot",
        **dict(zip(("fable_version", "astra_version"), judge.QUALIFICATION_JUDGES)),
    )
    return Experiment(raw, "sha256:" + identity, batch, routes, slots, policy)


def verify_provenance(spec, receipt, sources, *, objects=None):
    """Authenticate the Poet producer independently of the funding and release operators."""
    from scripts.collection_review import read_file
    from cryptography.exceptions import InvalidSignature

    from lab_arena.collection_gateway_student import (
        RECEIPT,
        object_store,
        verify_gateway_source,
    )

    if receipt.get("schema") == RECEIPT:
        return verify_gateway_source(
            spec, receipt, sources, objects if objects is not None else object_store()
        )

    path = os.environ.get("LAB_ARENA_COLLECTION_TRAINER_PUBLIC_KEY_FILE")
    if not path:
        raise ValueError("trusted Poet producer key is required")
    key = load_pem_public_key(read_file(Path(path), maximum=8192, private=False))
    body = {k: v for k, v in receipt.items() if k != "signature"}
    if not isinstance(key, Ed25519PublicKey) or set(body) != {
        "schema",
        "producer",
        "split_sha256",
        "sources",
    }:
        raise ValueError("invalid Poet source attestation")
    if (
        body["schema"] != "collection.poet-student-sources.v1"
        or not NAME.fullmatch(body["producer"])
        or body["split_sha256"] != spec.raw["split_sha256"]
        or not isinstance(body["sources"], dict)
        or set(body["sources"])
        != {c["source_sha256"] for c in spec.raw["cases"] if "source_sha256" in c}
        or _hash(receipt) != spec.raw["provenance_sha256"]
    ):
        raise ValueError("Poet source attestation differs from the experiment")
    try:
        key.verify(
            base64.b64decode(receipt["signature"], validate=True),
            judge._json(body).encode(),
        )
    except (InvalidSignature, KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid Poet source signature") from exc
    for digest, source in sources.items():
        info = body["sources"][digest]
        if (
            set(info)
            != {"role", "model", "model_revision", "source_run_id", "rights", "split"}
            or info["role"] != "untuned_student"
            or info["model"] != source["student_model"]
            or not NAME.fullmatch(info["source_run_id"])
            or not isinstance(info["model_revision"], str)
            or not re.fullmatch(
                r"(?:[a-f0-9]{40}|sha256:[a-f0-9]{64})", info["model_revision"]
            )
            or info["rights"] != "owned_synthetic_training_and_review"
            or info["split"] != "training"
            or _hash(source) != digest
        ):
            raise ValueError(
                "student run lacks authenticated origin or training rights"
            )
    return body


def read_inputs(spec, directory, sources, *, objects=None, database=None, prompt=None):
    """Verify every source and split before leasing or charging any key."""
    from scripts.collection_review import read_file
    from lab_arena.collection_workflow import (
        validate_student_source,
        correction_scenario,
        build_scenario,
    )
    from lab_arena.collection_quality import admit_built_split

    def read(digest):
        if directory is None or not SHA.fullmatch(digest):
            raise ValueError("private hash-addressed experiment inputs required")
        value = json.loads(read_file(directory / (digest[7:] + ".json")))
        if _hash(value) != digest:
            raise ValueError("private experiment input digest changed")
        return value

    if (sources[0].source_revision, sources[1].source_revision) != source_revisions(
        spec.raw
    ):
        raise ValueError("experiment loaded the wrong pinned sources")
    split = read(spec.raw["split_sha256"])
    resolved = {}
    for case in spec.raw["cases"]:
        if "source_sha256" not in case:
            sid = case["scenario_id"]
            task = build_scenario(sources[1], sid)
            resolved[sid] = (None, admit_built_split(sources[1], task, None, split))
            continue
        source = read(case["source_sha256"])
        from lab_arena.collection_pool_source import (
            SCHEMA as POOL_STUDENT,
            verify_funded_source,
        )

        if source.get("schema") == POOL_STUDENT:
            if (
                not spec.batch_id.startswith("pool-")
                or prompt is None
                or "provenance_sha256" in spec.raw
            ):
                raise ValueError(
                    "a funded source requires its original pool scope and rubric"
                )
            if database is None or objects is None:
                from scripts.run_locked_collection import evidence_stores

                database, objects = evidence_stores()
            verify_funded_source(database, objects, source, prompt)
            if any(a["role"] != "continuation" for a in spec.raw["arms"]):
                raise ValueError("use a fresh native case for a fresh expert arm")
        else:
            validate_student_source(*sources, source)
            if "provenance_sha256" not in spec.raw:
                raise ValueError(
                    "external student source requires its original provenance"
                )
        if source["scenario_id"] != case["scenario_id"]:
            raise ValueError("student checkpoint origin differs from assignment")
        if any(
            a["model"] == source["student_model"] and a["role"] == "continuation"
            for a in spec.raw["arms"]
        ):
            raise ValueError("a correction needs a distinct approved expert")
        native = source.get("schema") == POOL_STUDENT
        task = (
            build_scenario(sources[1], source["scenario_id"])
            if native
            else correction_scenario(sources[1], source)
        )
        # These identities are re-derived from the pinned builder, never miner labels.
        stamp = admit_built_split(
            sources[1],
            task,
            (
                None
                if native
                else json.loads(source["original"]["trace.json"]).get("shape")
            ),
            split,
        )
        resolved[case["source_sha256"]] = (source, stamp)
    provenance = None
    if "provenance_sha256" in spec.raw:
        provenance = read(spec.raw["provenance_sha256"])
        verify_provenance(
            spec,
            provenance,
            {k: v[0] for k, v in resolved.items() if v[0] is not None},
            objects=objects,
        )
    return resolved, split, provenance


class ExperimentMeter:
    """Enforce the frozen joint output-token cap across all generator calls."""

    def __init__(self, broker, arm):
        self.broker = broker
        self.arm = arm
        self.remaining = arm.get("max_run_tokens", arm["max_tokens"])
        self.calls = []

    def model_call(self, request_id, body):
        if self.remaining < 1 or len(self.calls) >= self.arm["max_steps"]:
            raise OutputLimitReached("experiment generation cap reached")
        limit = min(self.remaining, self.arm["max_tokens"], body["max_tokens"])
        params = dict(
            body,
            max_tokens=limit,
            reasoning={
                "effort": self.arm["reasoning_effort"],
                "exclude": not self.arm["capture_reasoning"],
            },
        )
        result = self.broker.model_call(request_id, params)
        usage = result.get("usage") if isinstance(result.get("usage"), dict) else {}
        counter = output_charge(usage, limit)
        # Unknown output is privately labelled and charged its maximum only for admission.
        reported = usage.get("completion_tokens")
        if type(reported) is int and reported > limit:
            self.remaining = 0
        else:
            self.remaining -= limit if counter is None else counter
        # Only publish the numeric token details used by the recorder. Do not copy
        # arbitrary provider metadata into the training-cost manifest.
        captured_usage = {
            k: usage[k]
            for k in ("prompt_tokens", "completion_tokens", "total_tokens", "cost")
            if k in usage
        }
        for name, fields in (
            ("completion_tokens_details", ("reasoning_tokens",)),
            ("prompt_tokens_details", ("cached_tokens", "cache_write_tokens")),
        ):
            detail = usage.get(name)
            if isinstance(detail, dict):
                captured_usage[name] = {
                    k: detail[k]
                    for k in fields
                    if type(detail.get(k)) is int and detail[k] >= 0
                }
        self.calls.append(
            {
                "request_id": request_id,
                "provider_id": result.get("id"),
                "model": result["model"],
                "revision": self.broker.versions[result["model"]],
                "provider": result.get("provider"),
                "output_allowance": limit,
                "usage": captured_usage,
            }
        )
        return result


def finish_pool_job(intake, database, job, status, quality, artifact, epoch):
    # A lost HTTP response or epoch-boundary race never warrants a second model
    # call. Check for the immutable original completion before retrying settlement.
    args = (job["job_id"], job["lease_token"], status, quality, artifact)
    try:
        return intake.complete(*args, epoch())
    except Exception:
        rows = database.rpc("collection_batch_jobs_v1", {"p_batch": job["batch_id"]})
        prior = next((j for j in rows if j["job_id"] == job["job_id"]), {})
        if prior.get("status") == status and prior.get("artifact_sha256") == artifact:
            return prior
        if prior.get("status") != "running":
            raise
        return intake.complete(*args, epoch())


def run_experiment(
    *,
    spec,
    sources,
    inputs,
    intake,
    database,
    objects,
    transport,
    pricing,
    owner_lookup,
    judge_key,
    max_jobs,
    epoch,
    before_claim=None,
    prompt=None,
    on_pool_failure=None,
):
    from lab_arena.collection_prices import claim_at_current_price
    from lab_arena.collection_broker import CollectionBroker
    from lab_arena.collection_workflow import (
        play_workflow,
    )
    from lab_arena.collection_claims import read_fresh

    agent, builder = source_revisions(spec.raw)
    stored, split, provenance = inputs
    live = spec.raw["mode"] == "live"
    pool = spec.raw.get("payment_rule") == POOL_PAYMENT_RULE
    tariff = None
    if live and not pool:
        from lab_arena.collection_correction_tariff import load_approved

        if prompt is None:
            raise ValueError("paid correction needs its frozen rubric")
        tariff = load_approved(spec, inputs, prompt, database, objects, epoch)
    fixed_quote = tariff["measurement"]["quote_microusd"] if tariff else 0
    count = 0
    for _ in range(max_jobs):
        job, job_prices = claim_at_current_price(
            (
                partial(intake.claim_pool, spec, agent)
                if pool
                else partial(
                    intake.claim_experiment, spec, agent, fixed_quote=fixed_quote
                )
            ),
            pricing,
            before_claim=before_claim,
        )
        if not job:
            break
        run_id = job["job_id"]
        case, arm, repeat = spec.slots[job["slot"]]
        batch = spec.batch_id
        meta = {
            "id": spec.raw["id"],
            "arm": arm["id"],
            "settings_sha256": spec.settings_hash,
            "repeat": repeat,
            "role": arm["role"],
            "capture_reasoning": arm["capture_reasoning"],
            "reasoning_effort": arm["reasoning_effort"],
        }
        meter = None
        generator_finished = False
        journal = WorkflowJournal(objects, batch, run_id)
        archived = {}
        vote_refs = []

        def save(kind, doc):
            ref = store_artifact(objects, batch, run_id, kind, doc)
            # A process loss before finalization still leaves verifiable object pointers.
            if kind != "interruption":
                objects.put(
                    f"collection/{batch}/{run_id}/saved/{kind}.json",
                    judge._json(ref).encode(),
                )
                archived[kind] = ref
            return ref

        def record_vote(vote):
            ref = save("claim-vote-" + vote["request_id"], vote)
            vote_refs.append(ref)

        try:
            if (
                job["model"] != arm["model"]
                or job["scenario_id"] != case["scenario_id"]
                or job["mode"] != spec.raw["mode"]
                or job["price_microusd"] != fixed_quote
                or job["harness_commit"] != agent
                or job["batch_id"] != batch
                or owner_lookup(job["miner_hotkey"]).coldkey != job["owner_coldkey"]
            ):
                raise ValueError("invalid experiment assignment or funding owner")
            source, stamp = stored[case.get("source_sha256", case["scenario_id"])]
            source_refs = {}
            if source is not None:
                source_refs["student_source"] = save("student-source", source)
                if provenance is not None:
                    source_refs["producer_attestation"] = save(
                        "producer-attestation", provenance
                    )
            settings = save("settings", spec.raw)
            split_ref = save("split", split)
            save("split-stamp", stamp)
            adapter = CollectionBroker(
                intake,
                database,
                transport,
                job,
                job_prices,
                routes=spec.routes,
                judge_key=judge_key,
            )
            if tariff:
                adapter.panel_qualification = tariff["panel_sha256"]
            meter = ExperimentMeter(adapter, arm)
            role_option = (
                "continuation" if arm["role"] == "continuation" else "student_source"
            )
            row, packet = play_workflow(
                *sources,
                dict(job, experiment=meta),
                meter,
                max_steps=arm["max_steps"],
                max_tokens=arm["max_tokens"],
                record=journal.append,
                save=save,
                capture_version=spec.raw.get("capture_version"),
                **({role_option: source} if source is not None else {}),
            )
            if (
                row["run"] != run_id
                or row["teacher"] != job["model"]
                or packet["run_id"] != run_id
                or packet["scenario_id"] != job["scenario_id"]
            ):
                raise ValueError("experiment output identity mismatch")
            generator_finished = True
            row_ref = archived.get("generator") or save("generator", row)
            calls_ref = save("generation-receipts", meter.calls)
            save("pre-claim-input", packet)
            if not pool and all(packet["checks"].values()):
                packet = read_fresh(
                    packet,
                    policy=spec.policy,
                    generator=job["model"],
                    broker=adapter,
                    record_vote=record_vote,
                    protocol=spec.raw.get(
                        "claim_protocol", "collection.fresh-claims.v1"
                    ),
                )
            if pool:
                packet["split"] = stamp
            packet_ref = save("judge-input", packet)
            claim = judge.claim_verdict(packet)
            code_pass = all(packet["checks"].values())
            manifest = {
                "schema": SCHEMA,
                "batch_id": batch,
                "job_id": run_id,
                "model": job["model"],
                "scenario_id": job["scenario_id"],
                "harness_commit": agent,
                "builder_commit": builder,
                "experiment": meta,
                "generator": row_ref,
                "packet": packet_ref,
                **source_refs,
                "split": split_ref,
                "split_stamp": stamp,
                "settings": settings,
                "calls": calls_ref,
                "claim_verdict": claim,
                "events": journal.hashes,
                "claim_votes": vote_refs,
            }
            if tariff:
                manifest["tariff_sha256"] = _hash(tariff)
                manifest["fixed_quote_microusd"] = fixed_quote
            if pool:
                from lab_arena.collection_worker import native_payment_evidence

                manifest.update(
                    packet=packet_ref,
                    payment_rule=POOL_PAYMENT_RULE,
                    capture_version=CAPTURE_V2,
                    judge_policy=asdict(spec.policy),
                )
                if arm["role"] == "continuation" and job["scenario_id"].startswith("P8-"):
                    manifest["uses_checkpoint_evidence"] = True
                journal.finish("completed")
                evidence = native_payment_evidence(objects, job, manifest)
                usage = adapter.usage()
                if (
                    usage["settled_microusd"] != evidence["cost_microusd"]
                    or usage["inflight_calls"]
                    or usage["reserved_or_uncertain_microusd"]
                    or usage["providers"]
                    != {"openrouter": {"used": evidence["calls"], "inflight": 0}}
                ):
                    raise ValueError(
                        "original transcript differs from miner-funded receipts"
                    )
                manifest["verification"] = save("native-verification", evidence)
                status = "accepted" if evidence["complete_verified"] else "rejected"
                label = "accepted" if evidence["task_pass"] else "code_rejected"
                quality = (
                    {"training_status": label, "payment_rule": POOL_PAYMENT_RULE}
                    if status == "accepted"
                    else {"training_status": label}
                )
            else:
                complete = all(
                    packet["checks"].get(k) is True
                    for k in ("answered", "shape", "privacy", "training_split")
                )
                status = "accepted" if complete else "rejected"
                label = (
                    "code_rejected"
                    if not code_pass
                    else (
                        "accepted"
                        if claim in {"passed", "not_applicable"}
                        else "needs_review"
                    )
                )
                quality = {"training_status": label}
            manifest_ref = save("manifest", manifest)
            if pool:
                finish_pool_job(
                    intake,
                    database,
                    job,
                    status,
                    quality,
                    manifest_ref["sha256"],
                    epoch,
                )
            else:
                intake.complete(
                    run_id,
                    job["lease_token"],
                    status,
                    quality,
                    manifest_ref["sha256"],
                    epoch(),
                )
        except Exception as exc:
            # Keep the original recorder journal and charge receipts even when the
            # worker cannot construct a complete row. Never echo a provider exception.
            interruption = save(
                "interruption",
                {
                    "job_id": run_id,
                    "experiment": meta,
                    "archived": dict(archived),
                    "cause": type(exc).__name__,
                    "generator_finished": generator_finished,
                    "calls": meter.calls if meter else [],
                    "events": journal.hashes,
                },
            )
            objects.put(
                f"collection/{batch}/{run_id}/interruption.json",
                judge._json(interruption).encode(),
            )
            intake.complete(
                run_id,
                job["lease_token"],
                "failed",
                {},
                interruption["sha256"] if generator_finished else None,
                epoch(),
            )
            raise RuntimeError(
                "experiment interrupted: " + type(exc).__name__
            ) from None
        if (
            pool
            and on_pool_failure
            and spec.raw.get("correction_model")
            and job["model"] != spec.raw["correction_model"]
            and evidence["complete_verified"]
            and not evidence["task_pass"]
        ):
            # Settlement is already durable. Follow-up preparation has no provider side effect.
            on_pool_failure(spec, job)
        count += 1
        print(json.dumps({"job": run_id, "batch": batch, "status": status}), flush=True)
    return count


def pool_verified_result(objects, job, spec):
    """Independently rebuild a standing job's native outcome from its frozen experiment and original wires."""
    from lab_arena.collection_workflow import read_artifact
    from lab_arena.collection_worker import native_payment_evidence

    if (
        spec.raw.get("payment_rule") != POOL_PAYMENT_RULE
        or spec.pilot
        or job["mode"] != "live"
        or job["batch_id"] != spec.batch_id
        or not 0 <= job["slot"] < len(spec.slots)
    ):
        raise ValueError("incorrect standing experiment identity")
    case, arm, repeat = spec.slots[job["slot"]]
    batch, run = spec.batch_id, job["job_id"]
    manifest = read_artifact(
        objects, batch, run, "manifest", {"sha256": job["artifact_sha256"]}, 1 << 20
    )
    if (
        manifest["schema"] != SCHEMA
        or manifest["payment_rule"] != POOL_PAYMENT_RULE
        or manifest["settings"]["sha256"] != _hash(spec.raw)
        or manifest["experiment"]["settings_sha256"] != spec.settings_hash
        or manifest["experiment"]["arm"] != arm["id"]
        or manifest["experiment"]["repeat"] != repeat
        or manifest["experiment"]["role"] != arm["role"]
        or job["model"] != arm["model"]
        or job["scenario_id"] != case["scenario_id"]
        or manifest["judge_policy"] != asdict(spec.policy)
        or case.get("source_sha256")
        != (manifest.get("student_source") or {}).get("sha256")
    ):
        raise ValueError("standing assignment differs from its original manifest")
    proof = native_payment_evidence(objects, job, manifest)
    recorded = read_artifact(
        objects, batch, run, "native-verification", manifest["verification"], 4 << 20
    )
    if proof != recorded or (
        job["status"] == "accepted"
        and (
            not proof["complete_verified"]
            or job["price_microusd"] != (proof["cost_microusd"] * 5 + 3) // 4
        )
    ):
        raise ValueError(
            "stored native result or earned reward differs from original compute"
        )
    return manifest, proof
