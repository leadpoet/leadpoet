"""Verify the operator's immutable, owner-funded P7 receipt before using its exact route."""

from hashlib import sha256
from lab_arena.collection_broker import REFERENCE_MODEL, REFERENCE_ROUTE
from lab_arena.collection_intake import GENERATOR_ROUTES
from lab_arena.collection_workflow import comparison_revisions

# Only this named source has passed the complete private route check. Registering a
# different source is a reviewed policy change, as is changing any generator pin.
OWNER_BATCH = "owner-alternative-p7-1125"
OWNER_RUN = "owner-deepseek-coreweave-4096-1125"
WITNESS = "sha256:9771941a1713b1f8e3a14af8409f4490fcc5e5e8a555c4343196ed788cce9465"


def verified_witness(raw, objects):
    from scripts.collection_batch import read_artifact

    successor = raw.get("reference_successor")
    digest = (
        successor["owner_witness_sha256"]
        if successor
        else raw.get("route_witness_sha256")
    )
    if digest != WITNESS or raw["routes"][REFERENCE_MODEL] != REFERENCE_ROUTE:
        raise ValueError("unknown generator route witness")

    def read(kind, ref):
        return read_artifact(
            objects, OWNER_BATCH, OWNER_RUN, kind, ref, 128 * 1024 * 1024
        )

    doc = read("route-witness", {"sha256": digest})
    revision = GENERATOR_ROUTES[REFERENCE_MODEL].revision
    if (
        doc["schema"] != "collection.owner-route-witness.v1"
        or doc["owner_only"] is not True
        or (
            doc["run_id"],
            doc["model"],
            doc["revision"],
            doc["provider"],
            doc["route"],
            doc["quantization"],
        )
        != (OWNER_RUN, REFERENCE_MODEL, revision, "CoreWeave", REFERENCE_ROUTE, "fp8")
        or doc["native_checks_passed"] is not True
        or doc["finish"] != "answered"
        or doc["strict_terms"]
        != {
            "author_distillable": True,
            "fallback": False,
            "request_fee": 0,
            "safety_retention": False,
            "zdr": True,
        }
    ):
        raise ValueError("witness is not the approved private route observation")
    gen = read("generator", doc["objects"]["generator"])
    out = read("workflow-output", doc["objects"]["workflow-output"])
    if (
        out["run_id"] != OWNER_RUN
        or out["scenario_id"] != doc["scope"]
        or out["finish"] != "answered"
        or gen["run"] != OWNER_RUN
        or gen["teacher"] != REFERENCE_MODEL
        or gen["task"] != doc["scope"]
        or (out["harness_commit"], out["builder_commit"]) != comparison_revisions(raw)
        or gen["evidence"]["grade"]["score"] != 1
        or gen["evidence"]["grade"].get("hard_fail")
        or not (0 < doc["usage_microusd"] <= doc["cap_microusd"])
    ):
        raise ValueError("witnessed run does not match frozen sources or measurements")
    total = 0
    observed = set()
    for n, receipt in enumerate(doc["receipts"], 1):
        kind = f"provider-{n:04d}"
        original = read(kind, doc["objects"][kind])
        reply = original["response"]
        if (
            original["run_id"] != OWNER_RUN
            or original["request_id"] != receipt["request"]
            or original["request_id"] != f"generate-{n}"
            or receipt["status"] != 200
            or receipt["model"] != REFERENCE_MODEL
            or receipt["source_revision"] != revision
            or receipt["route"] != REFERENCE_ROUTE
            or receipt["provider"] != doc["provider"]
            or reply["model"] != REFERENCE_MODEL
            or reply["provider"] != doc["provider"]
            or sha256(reply["id"].encode()).hexdigest() != receipt["gen_sha256"]
            or receipt["gen_sha256"] in observed
        ):
            raise ValueError("provider response differs from witnessed generation")
        observed.add(receipt["gen_sha256"])
        total += receipt["known_microusd"]
    if total != doc["usage_microusd"] or not any(
        c["returned_reasoning"] for c in doc["receipts"]
    ):
        raise ValueError("owner witness lacks measured generation or exposed reasoning")
    return {
        "owner_witness_sha256": digest,
        "owner_run_id": OWNER_RUN,
        "model": REFERENCE_MODEL,
        "model_revision": revision,
        "provider_route": REFERENCE_ROUTE,
        "precision": "fp8",
    }


def verify_matching_panel(raw, policy, qualification, prompt, database, *, successor):
    """The independent human-qualified cohort must have measured this exact route and cost class."""
    from scripts.run_locked_collection import load_batch
    from lab_arena.collection_broker import batch_versions
    from lab_arena.collection_intake import MODELS, TASK_SCOPE
    from lab_arena.collection_trajectories import verified_grants

    panel = load_batch(qualification, prompt)
    if (
        not panel.pilot
        or not panel.qualification
        or not panel.batch_id.startswith("rq-")
        or qualification.get("reference_successor")
        or qualification.get("route_witness_sha256") != WITNESS
        or (not policy.pilot and raw.get("pricing_pilot") != panel.batch_id)
        or (successor and panel.batch_id == policy.batch_id)
        or comparison_revisions(qualification) != comparison_revisions(raw)
        or batch_versions(qualification["routes"]) != batch_versions(raw["routes"])
        or any(
            qualification.get(k) != raw.get(k)
            for k in (
                "mode",
                "routes",
                "budgets_microusd",
                "split_manifest",
                "terms_sha256",
                "capture_version",
                "claim_protocol",
                "thresholds",
            )
        )
        or panel.prompt_sha256 != policy.prompt_sha256
    ):
        raise ValueError(
            "no complete independently qualified cohort matches the witnessed route"
        )
    # The signed panel verifies all ordinary cases, controls and blind marks. Join
    # actual settled charges separately; matching caps alone cannot establish cost coverage.
    rows = database.rpc("collection_pilot_costs_v1", {"p_batch": panel.batch_id})
    verified_grants(
        database,
        panel.batch_id,
        rows,
        mode="pilot",
        scope=TASK_SCOPE,
        digest=qualification["terms_sha256"],
    )
    expected = len(qualification["scenario_ids"]) * len(MODELS)
    if (
        not isinstance(rows, list)
        or {r["slot"] for r in rows} != set(range(expected))
        or any(
            r["unresolved"]
            or r["batch_id"] != panel.batch_id
            or r["model"] != MODELS[r["slot"] % len(MODELS)]
            or r["scenario_id"]
            != qualification["scenario_ids"][r["slot"] // len(MODELS)]
            for r in rows
        )
    ):
        raise ValueError("qualified cohort lacks settled complete route measurements")
    excluded = set(qualification["control_slots"]) | set(qualification["drift_slots"])
    measured = {
        r["model"]
        for r in rows
        if str(r["slot"]) not in excluded
        and r["status"] in ("accepted", "rejected")
        and r["artifact_sha256"]
        and r["generator_calls"] > 0
        and r["miner_cost_microusd"] > 0
    }
    if measured != set(MODELS):
        raise ValueError("missing paid generator evidence for the qualifying route")
