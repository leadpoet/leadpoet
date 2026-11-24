"""A bounded, unpaid experiment in the existing collection queue."""

from __future__ import annotations

from dataclasses import dataclass
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
MAX_SLOTS = 12


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

    if (
        set(raw)
        - {"claim_protocol", "min_pricing_runs", "pricing_pilot", "capture_version"}
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
            "provenance_sha256",
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
        or not all(
            SHA.fullmatch(raw[k])
            for k in ("split_sha256", "terms_sha256", "provenance_sha256")
        )
    ):
        raise ValueError("missing experiment question, pass mark, split or consent")
    cases, arms = raw["cases"], raw["arms"]
    if not all(isinstance(v, list) and 1 <= len(v) <= MAX_SLOTS for v in (cases, arms)):
        raise ValueError("invalid experiment size")
    if any(
        set(c) != {"scenario_id", "source_sha256"}
        or not SYNTHETIC_SCENARIO.fullmatch(c["scenario_id"])
        or not SHA.fullmatch(c["source_sha256"])
        for c in cases
    ):
        raise ValueError("invalid experiment source")
    if len({c["source_sha256"] for c in cases}) != len(cases):
        raise ValueError("duplicate student source")
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
            or not _integer(arm["repeats"], 1, MAX_SLOTS)
            or not _integer(arm["max_steps"], 1, 90)
            or not _integer(arm["max_tokens"], 1, 8192)
            or not _integer(
                arm.get("max_run_tokens", arm["max_tokens"]), 1, MAX_RUN_TOKENS
            )
            or not _integer(arm["budget_microusd"], 1, MAX_MICROUSD)
            or caps.setdefault(model, arm["budget_microusd"]) != arm["budget_microusd"]
            or arm["reasoning_effort"] not in {"low", "medium", "high"}
            or type(arm["capture_reasoning"]) is not bool
        ):
            raise ValueError("invalid or unapproved experiment arm")
        arm_names.add(arm["id"])
    # Compare arms on the same fixed student case before moving to a new case.
    slots = tuple(
        (case, arm, repeat)
        for case in cases
        for repeat in range(1, max(a["repeats"] for a in arms) + 1)
        for arm in arms
        if repeat <= arm["repeats"]
    )
    liability = sum(a["budget_microusd"] for _, a, _ in slots)
    if (
        len(slots) > MAX_SLOTS
        or not _integer(raw["budget_microusd"], 1, MAX_MICROUSD)
        or liability > raw["budget_microusd"]
    ):
        raise ValueError("experiment exceeds canary slots or budget")
    if "min_pricing_runs" in raw and (
        raw["mode"] != "pilot" or not _integer(raw["min_pricing_runs"], 1, len(slots))
    ):
        raise ValueError("pricing requires a predeclared achievable sample size")
    if raw["mode"] == "live" or "min_pricing_runs" in raw:
        if (
            len(arms) != 1
            or arms[0]["role"] != "continuation"
            or len({c["scenario_id"].split("-s")[0] for c in cases}) != 1
        ):
            raise ValueError(
                "paid calibration requires one comparable correction class"
            )
    if raw["mode"] == "live":
        if not isinstance(raw.get("pricing_pilot"), str) or not re.fullmatch(
            r"xp-[0-9a-f]{64}", raw["pricing_pilot"]
        ):
            raise ValueError("live corrections require an approved pricing pilot")
    elif "pricing_pilot" in raw:
        raise ValueError("a pilot cannot spend an approved live tariff")
    routes = {m: r.tag for m, r in GENERATOR_ROUTES.items()}
    routes.update({m: r.route for m, r in judge.QUALIFICATION_JUDGES.items()})
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
    batch = "xp-" + identity
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

    from lab_arena.collection_gateway_student import RECEIPT, object_store, verify_gateway_source
    if receipt.get("schema") == RECEIPT:
        return verify_gateway_source(spec, receipt, sources, objects if objects is not None else object_store())

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
        or set(body["sources"]) != {c["source_sha256"] for c in spec.raw["cases"]}
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


def read_inputs(spec, directory, sources, *, objects=None):
    """Verify every source and split before leasing or charging any key."""
    from scripts.collection_review import read_file
    from lab_arena.collection_workflow import (
        validate_student_source,
        correction_scenario,
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
        source = read(case["source_sha256"])
        validate_student_source(*sources, source)
        if source["scenario_id"] != case["scenario_id"]:
            raise ValueError("student checkpoint origin differs from assignment")
        if any(
            a["model"] == source["student_model"] and a["role"] == "continuation"
            for a in spec.raw["arms"]
        ):
            raise ValueError("a correction needs a distinct approved expert")
        task = correction_scenario(sources[1], source)
        # These identities are re-derived from the pinned builder, never miner labels.
        stamp = admit_built_split(
            sources[1],
            task,
            json.loads(source["original"]["trace.json"]).get("shape"),
            split,
        )
        resolved[case["source_sha256"]] = (source, stamp)
    provenance = read(spec.raw["provenance_sha256"])
    verify_provenance(spec, provenance, {k: v[0] for k, v in resolved.items()}, objects=objects)
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
        self.calls.append(
            {
                "request_id": request_id,
                "provider_id": result.get("id"),
                "model": result["model"],
                "revision": self.broker.versions[result["model"]],
                "provider": result.get("provider"),
                "output_allowance": limit,
                "usage": {
                    k: usage[k]
                    for k in (
                        "prompt_tokens",
                        "completion_tokens",
                        "total_tokens",
                        "cost",
                    )
                    if k in usage
                },
            }
        )
        return result


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
):
    from lab_arena.collection_broker import CollectionBroker
    from lab_arena.collection_workflow import (
        play_workflow,
    )
    from lab_arena.collection_claims import read_fresh

    agent, builder = source_revisions(spec.raw)
    stored, split, provenance = inputs
    live = spec.raw["mode"] == "live"
    tariff = None
    if live:
        from lab_arena.collection_correction_tariff import load_approved

        if prompt is None:
            raise ValueError("paid correction needs its frozen rubric")
        tariff = load_approved(spec, inputs, prompt, database, objects, epoch)
    fixed_quote = tariff["measurement"]["quote_microusd"] if live else 0
    count = 0
    for _ in range(max_jobs):
        if before_claim is not None:
            before_claim()
        job = intake.claim_experiment(spec, agent, fixed_quote=fixed_quote)
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
            source, stamp = stored[case["source_sha256"]]
            source_ref = save("student-source", source)
            settings = save("settings", spec.raw)
            split_ref = save("split", split)
            save("split-stamp", stamp)
            provenance_ref = save("producer-attestation", provenance)
            adapter = CollectionBroker(
                intake,
                database,
                transport,
                job,
                pricing,
                routes=spec.routes,
                judge_key=judge_key,
            )
            if live:
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
            if all(packet["checks"].values()):
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
                "student_source": source_ref,
                "split": split_ref,
                "split_stamp": stamp,
                "producer_attestation": provenance_ref,
                "settings": settings,
                "calls": calls_ref,
                "claim_verdict": claim,
                "events": journal.hashes,
                "claim_votes": vote_refs,
            }
            if live:
                manifest["tariff_sha256"] = _hash(tariff)
                manifest["fixed_quote_microusd"] = fixed_quote
            manifest_ref = save("manifest", manifest)
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
            intake.complete(
                run_id,
                job["lease_token"],
                status,
                {"training_status": label},
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
        count += 1
        print(json.dumps({"job": run_id, "batch": batch, "status": status}), flush=True)
    return count
