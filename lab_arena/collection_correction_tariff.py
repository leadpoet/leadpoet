"""Verify an immutable fixed quote for one bounded post-cutover correction batch."""

from __future__ import annotations

import base64
import os
from pathlib import Path
import tempfile

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from cryptography.hazmat.primitives.serialization import load_pem_public_key

from lab_arena import collection_judge as judge, contracts
from lab_arena.collection_claims import (
    CLAIM_RUBRIC,
    KEYED_RUBRIC,
    KEYED_V2,
    protocol_of,
)
from lab_arena.collection_experiment import (
    load_experiment,
    read_inputs,
    source_revisions,
)
from lab_arena.collection_experiment_export import write_dataset
from lab_arena.collection_intake import (
    GENERATOR_ROUTES,
    LIVE_CORRECTION_TERMS,
    require_correction_grant,
)
from lab_arena.collection_workflow import store_artifact

SCHEMA = "collection.compensated-correction.v1"


def class_definition(spec, inputs):
    if len(spec.raw["arms"]) != 1 or spec.raw["arms"][0]["role"] != "continuation":
        raise ValueError("a correction tariff covers one continuation arm")
    sources, _, provenance = inputs
    identities = {
        (
            s["student_model"],
            provenance["sources"][source_hash]["model_revision"],
            split["template_id"],
            split["rule_pool"],
            split["tool_variant"],
            split["manifest_sha256"],
        )
        for source_hash, (s, split) in sources.items()
    }
    if len(identities) != 1:
        raise ValueError("tariff requires a single source model, split and task class")
    student, revision, template, pool, variant, partition = identities.pop()
    arm = spec.raw["arms"][0]
    return {
        "student_model": student,
        "student_revision": revision,
        "template": template,
        "rule_pool": pool,
        "tools": variant,
        "split": partition,
        "harness": source_revisions(spec.raw),
        "prompt": spec.policy.prompt_sha256,
        "reader_protocol": protocol_of(spec.raw),
        "reader_rubric": judge.digest(
            KEYED_RUBRIC if protocol_of(spec.raw) == KEYED_V2 else CLAIM_RUBRIC
        ),
        "generator_revision": GENERATOR_ROUTES[arm["model"]].revision,
        "route": spec.routes[arm["model"]],
        "arm": {k: v for k, v in arm.items() if k not in ("id", "repeats")},
    }


def _measurement_matches(measured, release, minimum):
    observations = release.get("price_observations", [])
    if len(observations) != 1 or release.get("rows", 0) < 1:
        raise ValueError("no released correction evidence for this class")
    o = observations[0]
    if (
        o["role"] != "continuation"
        or o["model"] != measured["model"]
        or o["task_class"] != measured["template"]
        or o["measured_runs"] != measured["completed"]
        or o["miner_cost_microusd"] != measured["cost_microusd"]
        or not 1 <= minimum <= measured["completed"]
        or not 1 <= measured["quote_microusd"] <= 10**10
        or o["proposed_fixed_microusd"] != measured["quote_microusd"]
        or measured["quote_microusd"]
        != (5 * measured["cost_microusd"] + 4 * measured["completed"] - 1)
        // (4 * measured["completed"])
    ):
        raise ValueError("pilot observations differ from confirmed original charges")


def _active_panel(database, panel_hash):
    cutover = database.rpc("collection_reward_schedule_v1", {}) or {}
    if (
        cutover.get("staged")
        or type(cutover.get("first_epoch")) is not int
        or cutover.get("pilot_evidence", {}).get("panel_qualification_sha256")
        != panel_hash
    ):
        raise ValueError("ordinary collection cutover with this paid panel is required")
    return cutover["first_epoch"]


def prepare(live, pilot, directory, sources, database, objects, qualification):
    """Called before signing. All evidence and class identity are re-derived."""
    if (
        live.raw["mode"] != "live"
        or pilot.raw["mode"] != "pilot"
        or live.raw["pricing_pilot"] != pilot.batch_id
        or not pilot.raw.get("min_pricing_runs")
        or live.raw["terms_sha256"]
        != contracts.hash_bytes(LIVE_CORRECTION_TERMS.encode())
    ):
        raise ValueError(
            "correction tariff requires predeclared calibration and approved live consent"
        )
    cert = qualification["approval"]
    if cert.get("allow_live") is not True:
        raise ValueError("release approval does not authorize paid collection")
    panel_hash = judge.digest(cert)
    _active_panel(database, panel_hash)
    basis = class_definition(pilot, read_inputs(pilot, directory, sources))
    if basis != class_definition(live, read_inputs(live, directory, sources)):
        raise ValueError("live correction class differs from the measured class")
    # Reconstruct the real registered jobs, signed blind review and source permissions.
    with tempfile.TemporaryDirectory() as tmp:
        release = write_dataset(
            pilot,
            database,
            objects,
            Path(tmp) / "release.zip",
            qualification=qualification,
        )
    measured = database.rpc(
        "collection_correction_measurement_v1", {"p_batch": pilot.batch_id}
    )
    minimum = pilot.raw["min_pricing_runs"]
    _measurement_matches(measured, release, minimum)
    evidence = {
        "pilot_manifest": pilot.raw,
        "panel_approval": cert,
        "release_sha256": "sha256:" + judge.digest(release),
    }
    ref = store_artifact(objects, pilot.batch_id, "tariff", "evidence", evidence)
    return {
        "schema": SCHEMA,
        "live_batch": live.batch_id,
        "pilot_batch": pilot.batch_id,
        "minimum_runs": minimum,
        "class_sha256": judge.digest(basis),
        "cases": [c["scenario_id"] for c, _, _ in live.slots],
        "panel_sha256": panel_hash,
        "terms_sha256": live.raw["terms_sha256"],
        "measurement": measured,
        "evidence_sha256": ref["sha256"],
    }


def verify_pricing_signature(approval):
    from scripts.collection_review import read_file

    path = os.environ.get("LAB_ARENA_COLLECTION_TARIFF_PUBLIC_KEY_FILE")
    if not path:
        raise ValueError("missing pinned correction pricing authority")
    key = load_pem_public_key(read_file(Path(path), maximum=8192, private=False))
    if not isinstance(key, Ed25519PublicKey):
        raise ValueError("invalid pricing authority")
    body = {k: v for k, v in approval.items() if k != "signature"}
    key.verify(
        base64.b64decode(approval["signature"], validate=True),
        judge._json(body).encode(),
    )
    return body


def load_approved(live, inputs, prompt, database, objects, epoch):
    """Fail closed before asking for a miner credential or reserving work."""
    from scripts.collection_batch import read_artifact

    approval = database.rpc(
        "collection_correction_tariff_v1", {"p_batch": live.batch_id}
    )
    if not approval:
        raise ValueError("no authorized fixed correction quote")
    body = verify_pricing_signature(approval)
    first = _active_panel(database, body["panel_sha256"])
    if epoch() < first:
        raise ValueError("ordinary cutover epoch has not started")
    pilot = live.raw["pricing_pilot"]
    require_correction_grant("live", body["terms_sha256"])
    if (
        body["schema"] != SCHEMA
        or body["live_batch"] != live.batch_id
        or body["pilot_batch"] != pilot
        or body["terms_sha256"] != live.raw["terms_sha256"]
        or body["class_sha256"] != judge.digest(class_definition(live, inputs))
        or body["cases"] != [c["scenario_id"] for c, _, _ in live.slots]
    ):
        raise ValueError("tariff differs from this live batch")
    evidence = read_artifact(
        objects,
        pilot,
        "tariff",
        "evidence",
        {"sha256": body["evidence_sha256"]},
        2 * 1024 * 1024,
    )
    calibration = load_experiment(evidence["pilot_manifest"], prompt)

    # A price never crosses inference settings or held-out partitions, even when
    # an otherwise valid certificate exists for the same template and model.
    def inference_arm(exp):
        return {
            k: v for k, v in exp.raw["arms"][0].items() if k not in {"id", "repeats"}
        }

    if (
        protocol_of(calibration.raw) != protocol_of(live.raw)
        or protocol_of(evidence["panel_approval"]) != protocol_of(live.raw)
        or inference_arm(calibration) != inference_arm(live)
        or calibration.raw["split_sha256"] != live.raw["split_sha256"]
    ):
        raise ValueError("paid settings differ from measured correction class")
    cert = evidence["panel_approval"]
    if (
        calibration.batch_id != pilot
        or calibration.raw["min_pricing_runs"] != body["minimum_runs"]
        or cert.get("allow_live") is not True
        or judge.digest(cert) != body["panel_sha256"]
    ):
        raise ValueError("tariff qualification or pricing commitment changed")
    release = read_artifact(
        objects,
        pilot,
        "export",
        "release",
        {"sha256": evidence["release_sha256"]},
        2 * 1024 * 1024,
    )
    if (
        release["batch_id"] != pilot
        or release["settings_sha256"] != calibration.settings_hash
        or release["qualification_sha256"]
        != "sha256:" + judge.digest({k: v for k, v in cert.items() if k != "signature"})
    ):
        raise ValueError("tariff differs from the qualified dataset")
    measured = database.rpc("collection_correction_measurement_v1", {"p_batch": pilot})
    if measured != body["measurement"]:
        raise ValueError("approved correction receipts changed")
    _measurement_matches(measured, release, body["minimum_runs"])
    return body
