"""Versioned, train-only sandbox programs. Preserved historical manifests have no program block."""

from __future__ import annotations

import base64
from functools import lru_cache
import json
import os
from pathlib import Path
import re

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from cryptography.hazmat.primitives.serialization import load_pem_public_key

from lab_arena import collection_judge as judge
from lab_arena.collection_intake import GENERATOR_ROUTES

SCHEMA = "collection.sales-program.v3"
REGISTRY_SCHEMA = "collection.sales-sandbox.v3"
ROUTE_POLICY = "openrouter.approved_revisions.distillable_zdr_no_fallback.v1"
STUDENT = "qwen/qwen3.5-4b"
CLOSE = "qwen/qwen3.8-27b"
FAR = "deepseek/deepseek-v4-pro-0813"
TEACHERS = (
    "moonshotai/kimi-k3.1-pro",
    FAR,
    "deepseek/deepseek-v4.1-flash-1017",
    "mistralai/mistral-large-4-0",
    CLOSE,
    "nvidia/nemotron-3.5-super-vl",
)
VARIANTS = {1: "base", 2: "brief", 3: "review"}
BAND = 10_000_000
SOURCES_PER_ORDINAL = {"funded_failure": 3, "node_student": 4}
SHA = re.compile(r"sha256:[a-f0-9]{64}")
FAMILY = re.compile(r"P[1-9][0-9]{0,3}")
SCENARIO = re.compile(r"(P[1-9][0-9]{0,3})-s([1-9][0-9]{0,7})")

# None of the six approved teacher endpoints serves these exact student weights.
# S9 becomes available only after a real route/rights review pins the exact
# student revision in GENERATOR_ROUTES and this reviewed identifier matches.
STUDENT_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"


def hosted_student_route():
    route = GENERATOR_ROUTES.get(STUDENT)
    if route is None or route.revision != STUDENT_REVISION:
        raise ValueError(
            "no approved hosted route for the exact untuned Qwen3.5-4B student"
        )
    return route


def _hash(value):
    return "sha256:" + judge.digest(value)


def _signed_registry_record(record, public_key):
    body = {k: v for k, v in record.items() if k != "signature"}
    if not isinstance(public_key, Ed25519PublicKey):
        raise ValueError("invalid registry publisher key")
    try:
        public_key.verify(
            base64.b64decode(record["signature"], validate=True),
            judge._json(body).encode(),
        )
    except (InvalidSignature, KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid registry publisher signature") from exc
    return body


def validate_catalog_entry(entry):
    required = {
        "schema",
        "family",
        "builder",
        "partition",
        "grader",
        "kind",
        "patterns",
        "aliases",
        "allowed_variants",
        "route_policy",
        "evidence",
    }
    if (
        not isinstance(entry, dict)
        or set(entry) != required
        or entry["schema"] != REGISTRY_SCHEMA
        or not FAMILY.fullmatch(str(entry["family"]))
        or entry["kind"] not in {"train", "evaluation"}
        or not re.fullmatch(r"[0-9a-f]{40}", str(entry["builder"]))
        or not SHA.fullmatch(str(entry["partition"]))
        or entry["route_policy"] != ROUTE_POLICY
        or entry["grader"] not in {"posting", "qualification", "outreach", "native"}
        or type(entry["patterns"]) is not int
        or not 24 <= entry["patterns"] <= 10000
        or not isinstance(entry["allowed_variants"], list)
        or len(set(entry["allowed_variants"])) != len(entry["allowed_variants"])
        or not set(entry["allowed_variants"])
        <= (set(VARIANTS.values()) if entry["kind"] == "train" else {"heldout"})
        or (entry["kind"] == "train" and "base" not in entry["allowed_variants"])
        or not isinstance(entry["aliases"], dict)
        or set(entry["aliases"]) != set(entry["allowed_variants"])
    ):
        raise ValueError("invalid sandbox catalog record")
    for aliases in entry["aliases"].values():
        if (
            not isinstance(aliases, dict)
            or len(set(aliases.values())) != len(aliases)
            or any(
                not re.fullmatch(r"[a-z][a-z0-9_]{0,80}", str(s))
                for pair in aliases.items()
                for s in pair
            )
        ):
            raise ValueError("invalid producer tool alias table")
    return entry


def verify_admission(record, key):
    """A signed prospective sandbox admission, with independently collected gate evidence."""
    entry = validate_catalog_entry(_signed_registry_record(record, key))
    gates = entry["evidence"]
    if entry["kind"] == "evaluation":
        if (
            not isinstance(gates, dict)
            or gates.get("partition", {}).get("split_sha256") != entry["partition"]
        ):
            raise ValueError("held-out sandbox lacks its signed frozen partition")
        return entry
    required = {
        "independent_solver",
        "teacher",
        "lazy_policies",
        "patterns",
        "student",
        "partition",
    }
    if (
        not isinstance(gates, dict)
        or set(gates) != required
        or any(
            not isinstance(v, dict) or not SHA.fullmatch(str(v.get("artifact_sha256")))
            for v in gates.values()
        )
    ):
        raise ValueError("sandbox lacks the six reviewed gate artifacts")
    solver, teacher, student = (
        gates[k] for k in ("independent_solver", "teacher", "student")
    )

    def exact_count(x):
        return type(x) is int and x >= 0

    if (
        not all(
            exact_count(x)
            for x in (
                solver.get("tested"),
                solver.get("matched"),
                teacher.get("tested"),
                teacher.get("passed"),
                teacher.get("misses_reviewed"),
                student.get("tested"),
                student.get("passed"),
            )
        )
        or solver["tested"] < 200
        or solver["matched"] != solver["tested"]
        or teacher["tested"] <= 0
        or not 0.90 <= teacher["passed"] / teacher["tested"] <= 1
        or teacher["misses_reviewed"] != teacher["tested"] - teacher["passed"]
        or student["tested"] <= 0
        or not 0.15 <= student["passed"] / student["tested"] <= 0.75
        or str(student.get("model")).lower() != STUDENT
        or gates["lazy_policies"].get("failures")
        != ["suppress_all", "release_all", "copy_first", "cite_nothing"]
        or gates["patterns"].get("distinct") != entry["patterns"]
        or gates["partition"].get("split_sha256") != entry["partition"]
    ):
        raise ValueError("sandbox admission gates did not pass")
    return entry


@lru_cache(maxsize=1)
def _builtins():
    path = Path(__file__).with_name("sandbox_programs_v3.json")
    return {
        entry["family"]: validate_catalog_entry(entry)
        for entry in json.loads(path.read_bytes())
    }


def sandbox(family):
    if not isinstance(family, str) or not FAMILY.fullmatch(family):
        raise ValueError("unknown sandbox family")
    known = _builtins().get(family)
    if known:
        return known
    directory, key = os.getenv("LAB_ARENA_SANDBOX_REGISTRY"), os.getenv(
        "LAB_ARENA_COLLECTION_TRAINER_PUBLIC_KEY_FILE"
    )
    if not directory or not key:
        raise ValueError("unknown sandbox family")
    from scripts.collection_review import read_file

    path = Path(directory) / (family + ".json")
    if not path.is_file():
        raise ValueError("unknown sandbox family")
    result = verify_admission(
        json.loads(read_file(path, maximum=1 << 20)),
        load_pem_public_key(read_file(Path(key), maximum=8192, private=False)),
    )
    if result["family"] != family:
        raise ValueError("registry identity changed")
    return result


def identity(scenario):
    match = SCENARIO.fullmatch(str(scenario))
    if not match:
        raise ValueError("invalid registered scenario")
    family, seed = match[1], int(match[2])
    band, ordinal = divmod(seed, BAND)
    if band not in VARIANTS:
        raise ValueError("scenario is outside the registered training variants")
    return family, VARIANTS[band], ordinal


def program(collection, family, variant, *, source_kind=None, selection=None):
    entry = sandbox(family)
    if entry["kind"] != "train" or variant not in entry["allowed_variants"]:
        raise ValueError("evaluation sandbox or variant cannot run in the funding pool")
    doc = {
        "schema": SCHEMA,
        "collection": collection,
        "sandbox": family,
        "variant": variant,
        "use": "pass_rate_only" if collection == "C5" else "training",
        "registry_sha256": _hash(entry),
    }
    if source_kind is not None:
        doc["source_kind"] = source_kind
    if selection is not None:
        doc["selection_sha256"] = selection
    return doc


def validate_program(raw):
    doc = raw["program"]
    if not isinstance(doc, dict):
        raise ValueError("invalid registered program")
    collection, family, variant = (
        doc.get(k) for k in ("collection", "sandbox", "variant")
    )
    source_kind = doc.get("source_kind")
    selection = doc.get("selection_sha256")
    if (selection is not None) != (collection == "C5") or (
        selection and not SHA.fullmatch(str(selection))
    ):
        raise ValueError("effort comparisons require the frozen pass-rate selection")
    if collection not in {"C1", "C2", "C3", "C4", "C5"} or doc != program(
        collection, family, variant, source_kind=source_kind, selection=selection
    ):
        raise ValueError("registered program or provenance changed")
    if any(
        k in raw
        for k in (
            "successor_seeds",
            "correction_model",
            "auto_from_program",
            "series_arm_order",
        )
    ):
        raise ValueError("a registered research program has finite assigned work")
    if raw["split_sha256"] != sandbox(family)["partition"]:
        raise ValueError("program split differs from its registry")
    cases, arms = raw["cases"], raw["arms"]
    models = [arm["model"] for arm in arms]
    if len(models) != len(set(models)) or any(
        identity(c["scenario_id"])[:2] != (family, variant) for c in cases
    ):
        raise ValueError("registered case or model roster differs")
    if collection == "C4":
        if len({c.get("decision_sha256") for c in cases}) != len(cases):
            raise ValueError("duplicate original correction decision")
        if (
            source_kind not in {"node_student", "funded_failure"}
            or len(models) != 2
            or not set(models) <= set(TEACHERS)
            or any(
                not SHA.fullmatch(str(c.get("decision_sha256")))
                or "source_sha256" not in c
                for c in cases
            )
            or ("provenance_sha256" in raw) != (source_kind == "node_student")
        ):
            raise ValueError(
                "corrections require an authenticated source and two distinct teachers"
            )
    elif (
        source_kind is not None
        or "provenance_sha256" in raw
        or any("source_sha256" in c for c in cases)
    ):
        raise ValueError("fresh program cannot include correction evidence")
    if collection == "C3":
        hosted_student_route()
        if models != [STUDENT]:
            raise ValueError("a hosted student assignment must use the untuned student")
    if collection in {"C1", "C5"} and (
        set(models) != set(TEACHERS) or variant != "base"
    ):
        raise ValueError("the pass-rate map requires all six base teacher arms")
    if collection == "C2" and (
        variant not in {"brief", "review"} or set(models) != {CLOSE, FAR}
    ):
        raise ValueError(
            "the variant comparison requires the paired close and far teachers"
        )
    if "followups" in raw and (collection != "C1" or raw["followups"] != ["C4", "C5"]):
        raise ValueError("only baseline programs may request observed follow-ups")
    role = {"C3": "student", "C4": "continuation"}.get(collection, "expert")
    repeat = {"C3": 8, "C4": 2}.get(collection, 4)
    effort = "high" if collection == "C5" else "low"
    if any(
        a["role"] != role or a["repeats"] != repeat or a["reasoning_effort"] != effort
        for a in arms
    ):
        raise ValueError(
            "registered arm effort, role or repeats differ from the research design"
        )


def require_registered(database, doc):
    record = database.rpc(
        "collection_registered_sandbox_v1", {"p_family": doc["sandbox"]}
    )
    if record != {"kind": "train", "registry_sha256": doc["registry_sha256"]}:
        raise ValueError("database and pinned sandbox registry do not match")


def run_metadata(raw, scenario):
    if "program" not in raw:
        return {}
    family, _, ordinal = identity(scenario)
    return {
        "program": raw["program"],
        "ordinal": ordinal,
        "case_pattern": ordinal % sandbox(family)["patterns"],
    }


def canonical_tool(doc, name):
    if not doc:
        return name
    aliases = sandbox(doc["sandbox"])["aliases"][doc["variant"]]
    return next(
        (canonical for canonical, actual in aliases.items() if actual == name), name
    )


def registered_builder(revision):
    if any(
        entry["kind"] == "train" and entry["builder"] == revision
        for entry in _builtins().values()
    ):
        return True
    directory = os.getenv("LAB_ARENA_SANDBOX_REGISTRY")
    return bool(
        directory
        and any(
            sandbox(p.stem)["kind"] == "train"
            and sandbox(p.stem)["builder"] == revision
            for p in Path(directory).glob("P*.json")
            if FAMILY.fullmatch(p.stem)
        )
    )


def producer_partition(builders, family, seed):
    return (
        builders.training_partition(family, seed=seed)
        if seed >= BAND
        else builders.training_partition(family)
    )


def verify_selection(proof, raw):
    """Rebind a high-effort purchase to complete low-effort results from the private watcher."""
    from zipfile import ZipFile
    from hashlib import sha256
    from scripts.collection_review import read_file

    (sid,) = (c["scenario_id"] for c in raw["cases"])
    if (
        not isinstance(proof, dict)
        or set(proof) != {"schema", "snapshot", "scenario_id", "observations"}
        or proof["schema"] != "collection.pass-rate-selection.v1"
        or proof["scenario_id"] != sid
        or not re.fullmatch(r"[0-9a-f]{64}", str(proof["snapshot"]))
    ):
        raise ValueError("invalid pass-rate selection")
    observations = proof["observations"]
    if (
        not isinstance(observations, list)
        or len(observations) != 4 * len(TEACHERS)
        or len({r["run_id"] for r in observations}) != len(observations)
        or {(r["model"], r["repeat"]) for r in observations}
        != {(m, i) for m in TEACHERS for i in range(1, 5)}
        or any(type(r["pass"]) is not bool for r in observations)
        or not 0.25 <= sum(r["pass"] for r in observations) / len(observations) <= 0.75
    ):
        raise ValueError(
            "high-effort selection requires four verified low-effort runs for every teacher"
        )
    root = Path(
        os.environ.get(
            "LAB_ARENA_COLLECTION_EXPORT_DIRECTORY",
            Path.home() / ".local/share/leadpoet/collection/watch/exports",
        )
    )
    data = read_file(
        root / "snapshots" / (proof["snapshot"] + ".json"), maximum=128 << 20
    )
    if sha256(data).hexdigest() != proof["snapshot"]:
        raise ValueError("calibration index changed")
    index = json.loads(data)["offers"]
    batches = {r["batch"] for r in observations}
    if len(batches) != 1:
        raise ValueError("case was mixed across multiple baseline programs")
    batch = batches.pop()
    g = index[batch][sid]
    filename = g["dataset_sha256"].removeprefix("sha256:") + ".zip"
    if g["file"] != filename or not re.fullmatch(r"[0-9a-f]{64}\.zip", filename):
        raise ValueError("calibration archive differs")
    path = root / batch / filename
    if sha256(read_file(path, maximum=256 << 20)).hexdigest() != g["archive_sha256"]:
        raise ValueError("calibration archive changed")
    with ZipFile(path) as archive:
        for obs in observations:
            rid = obs["run_id"]
            if rid not in g["released"]:
                raise ValueError("calibration row was not released")
            blob = archive.read("trajectories/" + rid + ".jsonl")
            r = json.loads(blob)
            if (
                "sha256:" + sha256(blob).hexdigest() != obs["row_sha256"]
                or r["model"] != obs["model"]
                or r["run_id"] != rid
                or r["scenario_id"] != sid
                or r["experiment"]["repeat"] != obs["repeat"]
                or r["experiment"]["program"]
                != program("C1", raw["program"]["sandbox"], "base")
                or not r["verified_complete"]
                or r["verified_task_pass"] is not obs["pass"]
            ):
                raise ValueError(
                    "calibration row differs from original verified outcome"
                )


def source_case(source):
    rejected = source["rejected"]
    tools = []
    for call in rejected.get("tool_calls") or []:
        function = call["function"]
        arguments = function.get("arguments")
        try:
            value = json.loads(arguments)
        except (ValueError, TypeError):
            value = arguments
        tools.append({"name": function["name"], "arguments": value})
    action = {"content": rejected.get("content") or "", "tools": tools}
    return {
        "scenario_id": source["scenario_id"],
        "source_sha256": _hash(source),
        "decision_sha256": _hash(
            [source["student_model"], source["checkpoint_sha256"], action]
        ),
    }
