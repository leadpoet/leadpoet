"""Publish private, immutable comparison packs from the existing assigned-job ledger."""

from __future__ import annotations

from collections import Counter, defaultdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from zipfile import ZipFile

from lab_arena import collection_intake as intake, collection_judge as judge
from lab_arena.collection_pool import POOL_PAYMENT_RULE
from lab_arena.collection_experiment import load_experiment
from lab_arena.collection_experiment_export import write_dataset
from lab_arena.collection_workflow import store_artifact
from scripts.collection_batch import write_exports
from scripts.collection_review import read_file
from scripts.run_locked_collection import load_batch, verify_live_panel


TERMINAL = {"accepted", "failed", "rejected"}


class ExportWindowExpired(TimeoutError):
    """The watcher's allotted export time has elapsed."""


class OfferRows:
    """Read one consistent job/grant roster across the pack's validation and export."""

    def __init__(self, db, batch, experiment=None):
        self.batch = batch
        self.db = db
        self.experiment = experiment
        self.approval = None if experiment else db.rpc("collection_offer_schedule_v1", {"p_offer": batch})
        self.rows = db.rpc("collection_batch_jobs_v1", {"p_batch": batch})
        self.funding = db.rpc("collection_training_consents_v1", {"p_batch": batch})

    def rpc(self, name, args):
        if args == {"p_offer": self.batch} and name == "collection_offer_schedule_v1":
            return self.approval
        if args == {"p_batch": self.batch}:
            if name == "collection_batch_jobs_v1":
                return self.rows
            if name == "collection_training_consents_v1":
                return self.funding
        return self.db.rpc(name, args)


def _reason(exc):
    if isinstance(exc, FileNotFoundError) or getattr(exc, "response", {}).get("Error", {}).get(
        "Code"
    ) in {
        "NoSuchKey",
        "NotFound",
        "404",
    }:
        return "missing_artifact"
    if isinstance(exc, (ValueError, KeyError, TypeError)):
        return "unverified_evidence"
    raise exc


def _pack(raw, prompt, source, objects, directory, scenario, members, previous):
    """A missing or unsafe attempt must not hide its independently verified siblings."""
    ready = {rid for rid, row in members.items() if row["state"] in {"settled", "partial_billing_pending"}}
    held = {}
    result = {"members": members, "held": held, "export_version": 2}
    if not ready:
        return result
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=directory) as td:
        file = Path(td) / "pack.zip"

        def build(ids, output):
            if source.experiment:
                return write_dataset(source.experiment, source, objects, output, full_trajectories=True,
                                     only_jobs=ids, publish=False, prompt=prompt)
            return write_exports(raw, prompt, source, objects, output, only_jobs=ids, publish=False)

        try:
            meta = build(ready, file)
        except Exception as exc:
            reason = _reason(exc)
            if len(ready) == 1:
                held[next(iter(ready))] = reason
                return result
            for index, rid in enumerate(sorted(ready)):
                try:
                    build({rid}, Path(td) / f"check-{index}.zip")
                except Exception as single:
                    held[rid] = _reason(single)
            ready -= held.keys()
            if not ready:
                return result
            meta = build(ready, file)

        target = directory / (meta["dataset_sha256"].removeprefix("sha256:") + ".zip")
        try:
            os.link(file, target)
        except FileExistsError:
            # A repeated build can have different ZIP timestamps. Compare every member.
            with ZipFile(file) as candidate, ZipFile(target) as saved:
                if candidate.namelist() != saved.namelist() or any(
                    candidate.read(name) != saved.read(name) for name in candidate.namelist()
                ):
                    raise ValueError("previous private archive differs from verified rows")
        with ZipFile(target) as z:
            if z.testzip() or json.loads(z.read("metadata.json")) != meta:
                raise ValueError("invalid stored private trajectory archive")
            for rid in sorted(ready):
                row = json.loads(z.read(f"trajectories/{rid}.jsonl"))
                if row["run_id"] != rid or row["scenario_id"] != scenario:
                    raise ValueError("private archive contains a different assigned run")
                if row.get("capture_hold"):
                    if row["messages"]:
                        raise ValueError("held transcript in private archive")
                    held[rid] = row["capture_hold"]
        with target.open("rb") as stream:
            archive_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        if previous.get("dataset_sha256") != meta["dataset_sha256"]:
            if source.experiment:
                # All chunks from this experiment share its frozen, private partition.
                # Re-read the immutable input stored with a released job rather than copying held-out IDs into the ZIP.
                from lab_arena.collection_workflow import read_artifact
                split = read_artifact(objects, source.batch, min(ready - held.keys() or ready), "split",
                                      {"sha256": raw["split_sha256"]}, 128 << 20)
            else:
                split = raw["split_manifest"]
            store_artifact(objects, source.batch, "export", "split", split)
            store_artifact(objects, source.batch, "export", "release", meta)
        result.update(
            file=target.name,
            archive_sha256=archive_hash,
            bytes=target.stat().st_size,
            released=sorted(ready - held.keys()),
            dataset_sha256=meta["dataset_sha256"],
            trajectory_rows=meta["rows"],
            preference_pairs=meta["preference_pairs"],
        )
    return result


def _save(directory, data):
    encoded = judge._json(data).encode()
    digest = hashlib.sha256(encoded).hexdigest()
    snapshots = directory / "snapshots"
    snapshots.mkdir(mode=0o700, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=directory) as tmp:
        tmp.write(encoded)
        tmp.flush()
        os.fsync(tmp.fileno())
        target = snapshots / (digest + ".json")
        try:
            os.link(tmp.name, target)
        except FileExistsError:
            if read_file(target, maximum=64 << 20) != encoded:
                raise ValueError("private snapshot manifest hash changed")
    # The current pointer is published only after all referenced files and the manifest exist.
    with tempfile.NamedTemporaryFile(dir=directory, mode="w", delete=False) as pointer:
        json.dump({"snapshot": digest}, pointer)
        pointer.flush()
        os.fsync(pointer.fileno())
    os.replace(pointer.name, directory / "current.json")
    return digest


def _partial_pointer(objects, key):
    try:
        pointer = objects.get_optional_bounded(key, 8192)
    except ValueError:
        # The exporter will isolate an oversized or malformed artifact to its row.
        return "invalid"
    return hashlib.sha256(pointer).hexdigest() if pointer is not None else None


def observe(private_root, directory, db, objects, previous=None, *, max_groups=3, deadline=None):
    """One bounded watcher tick. Backfills and holds use the same public batch exporter."""
    intake.register_signatures()
    directory.mkdir(mode=0o700, exist_ok=True, parents=True)
    with (directory / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        current = directory / "current.json"
        old_digest = None
        old = {}
        if current.exists():
            old_digest = json.loads(read_file(current))["snapshot"]
            if not intake.SHA.fullmatch("sha256:" + old_digest):
                raise ValueError("invalid private snapshot identifier")
            contents = read_file(directory / "snapshots" / (old_digest + ".json"), maximum=64 << 20)
            if hashlib.sha256(contents).hexdigest() != old_digest:
                raise ValueError("private export index changed")
            old = json.loads(contents)["offers"]
        prompt = json.loads(read_file(private_root / "prompt.json"))
        paid = private_root / "paid"
        legacy = paid / "batch.json"
        manifests = ([legacy] if legacy.is_file() else []) + sorted(
            path for path in (paid / "offers").glob("*.json") if path.name.startswith(("dv-", "pool-"))
        )
        inputs = {}
        work = []
        offers = {}
        for path in manifests:
            raw = json.loads(read_file(path))
            experiment = load_experiment(raw, prompt) if raw.get("payment_rule") == POOL_PAYMENT_RULE else None
            batch = experiment.batch_id if experiment else load_batch(raw, prompt).batch_id
            if path != legacy and path.stem != batch:
                raise ValueError("paid manifest name differs from its contents")
            if batch in inputs:
                continue
            data = OfferRows(db, batch, experiment)
            if not experiment and not data.approval:
                if data.rows or batch in old:
                    raise ValueError("missing historical export approval")
                continue  # A measured offer may be installed before staging.
            if not experiment:
                verify_live_panel(raw, load_batch(raw, prompt), data.approval["pilot_evidence"])
            inputs[batch] = (raw, data)
            grants = {r["job_id"]: r for r in data.funding}
            if len(grants) != len(data.funding):
                raise ValueError("duplicate original funding grant")
            by_case = defaultdict(dict)
            for j in data.rows:
                rid = j["job_id"]
                auth = grants.get(rid)
                if not auth or rid in by_case[j["scenario_id"]]:
                    raise ValueError("job funding changed while preparing export")
                state = (
                    j["status"]
                    if j["status"] not in TERMINAL
                    else ("billing_pending" if auth["unresolved"] else "settled")
                )
                if experiment and state == "billing_pending" and j["status"] == "failed":
                    state = "partial_billing_pending"
                member = {
                    "state": state,
                    "job_status": j["status"],
                    "evidence_sha256": judge.digest([j, auth]),
                }
                if experiment and j["status"] == "failed":
                    prefix = f"collection/{batch}/{rid}/"
                    member["interruption"] = _partial_pointer(objects, prefix + "interruption.json")
                    member["quarantine"] = _partial_pointer(objects, prefix + "saved/quarantine.json")
                by_case[j["scenario_id"]][rid] = member
            groups = offers[batch] = {}
            prior = old.get(batch, {})
            if not set(prior) <= set(by_case):
                raise ValueError("previously registered work disappeared")
            for case, members in sorted(by_case.items()):
                prior_group = prior.get(case, {})
                same_members = members == prior_group.get("members")
                same = same_members and prior_group.get("export_version") == 2
                exists = (
                    not prior_group.get("file") or (directory / batch / prior_group["file"]).is_file()
                )
                if same and exists:
                    groups[case] = prior_group
                else:
                    # A change to a queued attempt cannot invalidate its published siblings.
                    # Check all prior released or held members, including billing evidence.
                    before = prior_group.get("members", {})
                    published = set(prior_group.get("released", ())) | set(prior_group.get("held", {}))
                    # An older index without the failed row's pointer fingerprints
                    # cannot prove it is still safe. Rebuild before linking that ZIP.
                    kept = exists and all(i in before and members.get(i) == before[i] for i in published)
                    groups[case] = {**(prior_group if kept else {}), "members": members, "queued": True}
                # A verified interruption seals its journal. Recovery and quarantine
                # pointer changes still queue this case just like a billing change.
                unsealed = any(
                    m["job_status"] == "failed" and not m.get("interruption") for m in members.values()
                )
                retry_hold = any(
                    reason != "quarantine" or not members[rid].get("quarantine")
                    for rid, reason in prior_group.get("held", {}).items()
                )
                if groups[case].get("queued"):
                    # A verified archive is already available during a format upgrade.
                    # New funded work comes first; the regular retry pass backfills old packs.
                    upgrade_only = same_members and exists and prior_group.get("file") and not same
                    work.append((1 if upgrade_only else 0, batch, case))
                elif retry_hold or unsealed:
                    work.append((1, batch, case))
        if not set(old) <= set(offers):
            raise ValueError("a historical paid export manifest is missing")

        cursor = (previous or {}).get("last_group", "")
        pass_number = (previous or {}).get("pass", 0) + 1
        # Retry missing uploads even while new scenarios keep arriving.
        if pass_number % 4 == 0:
            work = [(1 - priority, batch, case) for priority, batch, case in work]
        work.sort(key=lambda item: (item[0], "/".join(item[1:]) <= cursor, item[1:]))
        last_pack_seconds = None
        for _, batch, case in work[:max_groups]:
            started = time.monotonic()
            # Leave room to write the atomic index before the watch deadline.
            if deadline is not None and last_pack_seconds is not None and deadline - started <= last_pack_seconds + 1:
                break
            raw, data = inputs[batch]
            group = offers[batch][case]
            try:
                packed = _pack(
                    raw, prompt, data, objects, directory / batch, case,
                    group["members"], old.get(batch, {}).get(case, {}),
                )
            except ExportWindowExpired:
                if last_pack_seconds is None:
                    raise
                # Publish earlier verified groups. The interrupted group retains
                # its previous archive or queued state and is retried next time.
                break
            offers[batch][case] = packed
            last_pack_seconds = time.monotonic() - started
            cursor = batch + "/" + case

        manifest = {"schema": "collection.private-export-index.v1", "offers": offers}
        digest = _save(directory, manifest) if offers != old or old_digest is None else old_digest
        counts = Counter()
        for groups in offers.values():
            for group in groups.values():
                counts["registered"] += len(group["members"])
                counts["released"] += len(group.get("released", ()))
                counts["held"] += len(group.get("held", {}))
                counts["pending"] += (
                    len(group["members"]) - len(group.get("released", ())) - len(group.get("held", {}))
                )
        return {"snapshot": digest, "last_group": cursor, "pass": pass_number, **counts}
