"""Prepare the bounded release authority needed by an exact V2 restart.

The lifetime release catalog is not restart authority.  These entrypoints
derive the small set of releases that can still authorize live receipt or
publication state, fetch those exact immutable channel objects, and fail
closed if the selection moves before it can be installed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

from gateway.tee.active_release_requirements_v2 import (
    MAX_ACTIVE_RELEASE_COMMITS,
    MAX_ACTIVE_RELEASE_ROOTS,
    build_active_release_requirements_v2,
    validate_active_release_requirements_v2,
)
from gateway.tee.release_channel_v2 import (
    DEFAULT_BUCKET,
    DEFAULT_PREFIX,
    fetch_historical_release_lineage_v2,
    fetch_release_lineage_v2,
    git_ancestor_commits_v2,
)
from gateway.tee.release_lineage_v2 import (
    build_compact_release_lineage_boot_verifier_v2,
    build_historical_compact_release_lineage_boot_verifier_v2,
    validate_compact_release_lineage_v2,
    validate_historical_compact_release_lineage_v2,
    validate_prior_compact_release_lineage_v2,
)
from gateway.tee.release_manifest_v2 import (
    validate_historical_release_manifest,
    validate_prior_release_manifest,
)
from leadpoet_canonical.attested_v2 import (
    canonical_json,
    sha256_json,
    validate_boot_identity,
    validate_receipt_graph,
)
from leadpoet_canonical.hotkey_authority_v2 import (
    MAX_WEIGHT_TRANSPORT_LOGICAL_BYTES,
    validate_chain_signing_profile,
)


RESULT_SCHEMA_VERSION = "leadpoet.prepare_active_release_lineage.v2"
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_INVOCATION_RE = re.compile(r"^[a-z0-9][a-z0-9_.:-]{0,127}$")
_MAX_SIDECAR_JSON_INPUT_BYTES = 4 * 1024 * 1024
_MAX_PUBLICATION_JOURNAL_BYTES = MAX_WEIGHT_TRANSPORT_LOGICAL_BYTES
_FALLBACK_CONTEXTS = frozenset({"standalone", "cutover", "full-parity"})
_SOURCE_ADD_ACTIVE_INTENT_STATUSES = ("leased", "queued", "retry_wait")
_SOURCE_ADD_ACTIVE_REWARD_STATUSES = ("active", "partially_paid", "queued")
_SOURCE_ADD_QUERY_CHUNK = 50
_SOURCE_ADD_INTENT_RE = re.compile(r"^source_add_reward_intent:[0-9a-f]{16}$")
_SOURCE_ADD_SUBMISSION_RE = re.compile(r"^source_add_submission:[0-9a-f]{16}$")
_SOURCE_ADD_REWARD_RE = re.compile(r"^source_add_reward:[0-9a-f]{16}$")


class PrepareActiveReleaseLineageV2Error(RuntimeError):
    """The active release authority could not be frozen exactly."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PrepareActiveReleaseLineageV2Error(message)


def _commit(value: Any, label: str) -> str:
    _require(
        isinstance(value, str) and _COMMIT_RE.fullmatch(value) is not None,
        "%s is invalid" % label,
    )
    return value


def _lineage_id(value: Any) -> str:
    _require(
        isinstance(value, str) and _HASH_RE.fullmatch(value) is not None,
        "ancestry lineage id is invalid",
    )
    return value


def _invocation_id(value: Any) -> str:
    _require(
        isinstance(value, str) and _INVOCATION_RE.fullmatch(value) is not None,
        "restart invocation id is invalid",
    )
    return value


def _load_json(
    path: Path,
    label: str,
    *,
    max_bytes: int = _MAX_SIDECAR_JSON_INPUT_BYTES,
) -> Dict[str, Any]:
    _require(
        isinstance(max_bytes, int)
        and not isinstance(max_bytes, bool)
        and max_bytes > 0,
        "%s byte bound is invalid" % label,
    )
    nofollow = getattr(os, "O_NOFOLLOW", None)
    _require(nofollow is not None, "no-follow JSON reads are unavailable")
    descriptor = -1
    try:
        descriptor = os.open(
            str(Path(path)),
            os.O_RDONLY | os.O_CLOEXEC | nofollow,
        )
        metadata = os.fstat(descriptor)
        _require(
            stat.S_ISREG(metadata.st_mode)
            and 0 < metadata.st_size <= max_bytes,
            "%s is not a bounded regular file" % label,
        )
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            raw = handle.read(max_bytes + 1)
        _require(
            0 < len(raw) <= max_bytes,
            "%s is not a bounded regular file" % label,
        )
        value = json.loads(raw)
    except PrepareActiveReleaseLineageV2Error:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PrepareActiveReleaseLineageV2Error(
            "%s is unavailable or invalid" % label
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    _require(isinstance(value, Mapping), "%s must be an object" % label)
    return dict(value)


def _load_optional_journal(path: Path) -> Optional[Dict[str, Any]]:
    candidate = Path(path)
    try:
        candidate.lstat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise PrepareActiveReleaseLineageV2Error(
            "validator publication journal is unavailable"
        ) from exc
    return _load_json(
        candidate,
        "validator publication journal",
        max_bytes=_MAX_PUBLICATION_JOURNAL_BYTES,
    )


def _load_validator_authority_context(
    *,
    hotkey_config_path: Path,
    chain_signing_profile_path: Path,
) -> Dict[str, Any]:
    from validator_tee.enclave.hotkey_authority_v2 import (
        validate_hotkey_authority_configuration,
    )

    try:
        configuration = validate_hotkey_authority_configuration(
            _load_json(
                Path(hotkey_config_path),
                "validator hotkey configuration",
            )
        )
        profile = validate_chain_signing_profile(
            _load_json(
                Path(chain_signing_profile_path),
                "validator chain signing profile",
            )
        )
    except Exception as exc:
        raise PrepareActiveReleaseLineageV2Error(
            "validator authority configuration is unavailable or invalid"
        ) from exc
    hotkey = str(configuration.get("validator_hotkey") or "")
    _require(bool(hotkey), "validator hotkey configuration is invalid")
    _require(
        configuration["chain_signing_profile_hash"] == sha256_json(profile),
        "validator chain signing profile differs from hotkey configuration",
    )
    return {
        "validator_hotkey": hotkey,
        "chain_signing_profile": profile,
    }


def _stage_atomic_json(path: Path, value: Mapping[str, Any]) -> Path:
    destination = Path(path)
    payload = (canonical_json(dict(value)) + "\n").encode("ascii")
    _require(
        len(payload) <= _MAX_SIDECAR_JSON_INPUT_BYTES,
        "active release output exceeds sidecar byte bound",
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".%s." % destination.name,
        dir=str(destination.parent),
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        return temporary
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_json_documents(
    documents: Sequence[tuple[Path, Mapping[str, Any]]],
) -> None:
    destinations = [Path(path) for path, _value in documents]
    _require(
        len(destinations) == len(set(destinations)),
        "active release output paths are duplicated",
    )
    staged: list[tuple[Path, Path]] = []
    try:
        for destination, value in documents:
            staged.append(
                (Path(destination), _stage_atomic_json(Path(destination), value))
            )
        for destination, temporary in staged:
            os.replace(temporary, destination)
        for directory in sorted({path.parent for path in destinations}):
            descriptor = os.open(str(directory), os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    finally:
        for _destination, temporary in staged:
            temporary.unlink(missing_ok=True)


def _normalize_journal_requirements(value: Any) -> Dict[str, Any]:
    _require(
        isinstance(value, Mapping)
        and set(value) == {"journal_hash", "required_commits"},
        "publication journal release requirements are invalid",
    )
    journal_hash = value.get("journal_hash")
    _require(
        journal_hash is None
        or (isinstance(journal_hash, str) and _HASH_RE.fullmatch(journal_hash)),
        "publication journal hash is invalid",
    )
    commits = value.get("required_commits")
    _require(
        isinstance(commits, list),
        "publication journal required commits are invalid",
    )
    normalized = [
        _commit(item, "publication journal required commit") for item in commits
    ]
    _require(
        normalized == sorted(set(normalized)),
        "publication journal required commits are not canonical",
    )
    _require(
        len(normalized) <= MAX_ACTIVE_RELEASE_COMMITS,
        "publication journal required commits exceed bound",
    )
    if journal_hash is None:
        _require(
            not normalized,
            "missing publication journal cannot select releases",
        )
    return {"journal_hash": journal_hash, "required_commits": normalized}


def _journal_requirements(
    journal: Optional[Mapping[str, Any]],
    *,
    expected_lineage_id: str,
    expected_validator_hotkey: str,
    chain_signing_profile: Mapping[str, Any],
    boot_verifier: Optional[Callable[[Mapping[str, Any]], Any]] = None,
) -> Dict[str, Any]:
    from validator_tee.host.publication_journal_v2 import (
        publication_journal_release_requirements_v2,
    )

    return _normalize_journal_requirements(
        publication_journal_release_requirements_v2(
            journal,
            expected_lineage_id=expected_lineage_id,
            expected_validator_hotkey=expected_validator_hotkey,
            boot_verifier=boot_verifier,
            chain_profile=chain_signing_profile,
        )
    )


def _fetch_exact_release_lineage_v2(
    *,
    candidate_commit_sha: str,
    authority_commit_sha: str,
    required_commits: Sequence[str],
    repository: Path,
    bucket: str,
    prefix: str,
    historical_topology_hash: str | None = None,
    s3_client: Any = None,
) -> Dict[str, Any]:
    candidate = _commit(candidate_commit_sha, "candidate commit")
    authority = _commit(authority_commit_sha, "release authority commit")
    required = list(required_commits)
    _require(
        required
        and required == sorted(set(required))
        and all(
            isinstance(item, str) and _COMMIT_RE.fullmatch(item) for item in required
        )
        and len(required) <= MAX_ACTIVE_RELEASE_COMMITS,
        "required release set is invalid",
    )
    allowed = git_ancestor_commits_v2(
        repository=Path(repository),
        current_commit=authority,
    )
    _require(
        set(required).issubset(set(allowed)),
        "required release is outside release authority Git ancestry",
    )
    if historical_topology_hash is None:
        lineage = fetch_release_lineage_v2(
            bucket=str(bucket),
            current_commit=candidate,
            prefix=str(prefix),
            s3_client=s3_client,
            allowed_commits=allowed,
            required_commits=required,
        )
        normalized = validate_compact_release_lineage_v2(
            lineage,
            expected_current_commit=candidate,
        )
    else:
        lineage = fetch_historical_release_lineage_v2(
            bucket=str(bucket),
            current_commit=candidate,
            expected_topology_hash=historical_topology_hash,
            prefix=str(prefix),
            s3_client=s3_client,
            allowed_commits=allowed,
            required_commits=required,
        )
        normalized = validate_historical_compact_release_lineage_v2(
            lineage,
            expected_topology_hash=historical_topology_hash,
            expected_current_commit=candidate,
        )
    _require(
        sorted(normalized["releases"]) == required,
        "compact release lineage differs from required set",
    )
    return normalized


def _structural_boot_verifier(identity: Mapping[str, Any]) -> Dict[str, Any]:
    try:
        validate_boot_identity(identity)
    except Exception as exc:
        raise PrepareActiveReleaseLineageV2Error(
            "active ancestry boot identity is invalid"
        ) from exc
    return dict(identity)


def _compact_boot_verifier(
    lineage: Mapping[str, Any],
    *,
    historical_topology_hash: str | None = None,
) -> Callable[[Mapping[str, Any]], Mapping[str, Any]]:
    if historical_topology_hash is not None:
        return build_historical_compact_release_lineage_boot_verifier_v2(
            lineage,
            expected_topology_hash=historical_topology_hash,
        )
    return build_compact_release_lineage_boot_verifier_v2(lineage)


def _selected_compact_boot_verifier(
    lineage: Mapping[str, Any],
    *,
    historical_topology_hash: str | None,
) -> Callable[[Mapping[str, Any]], Mapping[str, Any]]:
    if historical_topology_hash is None:
        return _compact_boot_verifier(lineage)
    return _compact_boot_verifier(
        lineage,
        historical_topology_hash=historical_topology_hash,
    )


def _validate_selected_gateway_release(
    value: Mapping[str, Any],
    *,
    historical_topology_hash: str | None,
) -> Dict[str, Any]:
    if historical_topology_hash is not None:
        return validate_historical_release_manifest(
            value,
            expected_topology_hash=historical_topology_hash,
        )
    return validate_prior_release_manifest(value)


def _validate_selected_lineage(
    value: Mapping[str, Any],
    *,
    historical_topology_hash: str | None,
    expected_current_commit: str,
    expected_current_gateway_release_hash: str | None = None,
    allow_historical_current: bool = False,
) -> Dict[str, Any]:
    if historical_topology_hash is not None:
        return validate_historical_compact_release_lineage_v2(
            value,
            expected_topology_hash=historical_topology_hash,
            expected_current_commit=expected_current_commit,
            expected_current_gateway_release_hash=(
                expected_current_gateway_release_hash
            ),
        )
    validator = (
        validate_prior_compact_release_lineage_v2
        if allow_historical_current
        else validate_compact_release_lineage_v2
    )
    return validator(
        value,
        expected_current_commit=expected_current_commit,
        expected_current_gateway_release_hash=(
            expected_current_gateway_release_hash
        ),
    )


async def _load_active_source_add_graphs_v2(
    *,
    current_epoch: int,
    select_rows: Any = None,
    load_business_graphs: Any = None,
) -> list[dict[str, Any]]:
    """Load SOURCE_ADD ancestry that can authorize work after this restart.

    Pending Leg 1 work consumes its provenance receipt. Every nonterminal
    reward consumes its decision receipt in an allocation. Select the full
    payment life because a reward can finalize after the current epoch's
    allocation frontier was already created.
    """

    _require(
        isinstance(current_epoch, int)
        and not isinstance(current_epoch, bool)
        and current_epoch >= 0,
        "SOURCE_ADD active release epoch is invalid",
    )
    if select_rows is None:
        from gateway.research_lab.store import select_all

        select_rows = select_all
    if load_business_graphs is None:
        from gateway.research_lab.attested_v2_store import (
            load_business_artifact_graphs_v2,
        )

        load_business_graphs = load_business_artifact_graphs_v2

    pending_intents, active_rewards = await asyncio.gather(
        select_rows(
            "research_lab_source_add_reward_intents",
            columns=(
                "intent_id,submission_id,adapter_id,miner_hotkey,leg,"
                "intent_status,approval_kind,provenance_receipt_hash,"
                "provenance_artifact_hash"
            ),
            filters=(
                ("leg", 1),
                ("approval_kind", "provenance_precheck_passed"),
                (
                    "intent_status",
                    "in",
                    list(_SOURCE_ADD_ACTIVE_INTENT_STATUSES),
                ),
            ),
            order_by=(("intent_id", False),),
            max_rows=MAX_ACTIVE_RELEASE_ROOTS + 1,
            allow_partial=False,
        ),
        select_rows(
            "research_lab_source_add_reward_current",
            columns=(
                "reward_ref,adapter_id,miner_hotkey,leg,reward_kind,"
                "alpha_percent,reward_epochs,start_epoch,trigger_evidence_doc,"
                "public_label,current_reward_status"
            ),
            filters=(
                (
                    "current_reward_status",
                    "in",
                    list(_SOURCE_ADD_ACTIVE_REWARD_STATUSES),
                ),
            ),
            order_by=(("reward_ref", False),),
            max_rows=MAX_ACTIVE_RELEASE_ROOTS + 1,
            allow_partial=False,
        ),
    )
    _require(
        isinstance(pending_intents, list) and isinstance(active_rewards, list),
        "SOURCE_ADD active release rows are invalid",
    )
    _require(
        len(pending_intents) + len(active_rewards) <= MAX_ACTIVE_RELEASE_ROOTS,
        "SOURCE_ADD active release root count exceeds bound",
    )

    pending_by_submission: dict[str, dict[str, Any]] = {}
    intent_ids: set[str] = set()
    pending_adapters: set[str] = set()
    for raw_intent in pending_intents:
        _require(
            isinstance(raw_intent, Mapping),
            "SOURCE_ADD pending reward intent is invalid",
        )
        intent = dict(raw_intent)
        intent_id = str(intent.get("intent_id") or "")
        submission_id = str(intent.get("submission_id") or "")
        adapter_id = str(intent.get("adapter_id") or "")
        miner_hotkey = str(intent.get("miner_hotkey") or "")
        receipt_hash = str(intent.get("provenance_receipt_hash") or "")
        artifact_hash = str(intent.get("provenance_artifact_hash") or "")
        _require(
            _SOURCE_ADD_INTENT_RE.fullmatch(intent_id) is not None
            and _SOURCE_ADD_SUBMISSION_RE.fullmatch(submission_id) is not None
            and bool(adapter_id)
            and bool(miner_hotkey)
            and int(intent.get("leg") or 0) == 1
            and intent.get("approval_kind") == "provenance_precheck_passed"
            and intent.get("intent_status") in _SOURCE_ADD_ACTIVE_INTENT_STATUSES
            and _HASH_RE.fullmatch(receipt_hash) is not None
            and _HASH_RE.fullmatch(artifact_hash) is not None,
            "SOURCE_ADD pending reward intent differs from provenance authority",
        )
        _require(
            intent_id not in intent_ids
            and submission_id not in pending_by_submission
            and adapter_id not in pending_adapters,
            "SOURCE_ADD pending reward intent is duplicated",
        )
        intent_ids.add(intent_id)
        pending_adapters.add(adapter_id)
        pending_by_submission[submission_id] = intent

    authority_by_submission: dict[str, dict[str, Any]] = {}
    pending_submission_ids = sorted(pending_by_submission)
    for offset in range(0, len(pending_submission_ids), _SOURCE_ADD_QUERY_CHUNK):
        chunk = pending_submission_ids[offset : offset + _SOURCE_ADD_QUERY_CHUNK]
        rows = await select_rows(
            "research_lab_source_add_provenance_leg1_authority_v1",
            columns=(
                "submission_id,adapter_id,miner_hotkey,precheck_status,"
                "provenance_receipt_hash,provenance_artifact_hash"
            ),
            filters=(("submission_id", "in", chunk),),
            order_by=(("submission_id", False),),
            max_rows=len(chunk) + 1,
            allow_partial=False,
        )
        _require(
            isinstance(rows, list),
            "SOURCE_ADD provenance authority rows are invalid",
        )
        for raw_authority in rows:
            _require(
                isinstance(raw_authority, Mapping),
                "SOURCE_ADD provenance authority row is invalid",
            )
            authority = dict(raw_authority)
            submission_id = str(authority.get("submission_id") or "")
            _require(
                submission_id in chunk and submission_id not in authority_by_submission,
                "SOURCE_ADD provenance authority is missing or ambiguous",
            )
            authority_by_submission[submission_id] = authority
    _require(
        set(authority_by_submission) == set(pending_by_submission),
        "SOURCE_ADD provenance authority is missing or ambiguous",
    )

    artifact_requests: list[tuple[str, str, str]] = []
    expected_roots: dict[tuple[str, str, str], tuple[str, str]] = {}
    for submission_id, intent in pending_by_submission.items():
        authority = authority_by_submission[submission_id]
        receipt_hash = str(intent["provenance_receipt_hash"])
        artifact_hash = str(intent["provenance_artifact_hash"])
        _require(
            authority.get("submission_id") == submission_id
            and authority.get("adapter_id") == intent.get("adapter_id")
            and authority.get("miner_hotkey") == intent.get("miner_hotkey")
            and authority.get("precheck_status") == "provenance_precheck_passed"
            and authority.get("provenance_receipt_hash") == receipt_hash
            and authority.get("provenance_artifact_hash") == artifact_hash,
            "SOURCE_ADD pending reward intent differs from provenance authority",
        )
        artifact = ("source_add_provenance", submission_id, artifact_hash)
        artifact_requests.append(artifact)
        expected_roots[artifact] = (
            receipt_hash,
            "research_lab.source_add_provenance.v2",
        )

    from gateway.tee.reward_executor_v2 import source_add_reward_row_projection_v2

    active_reward_ids: set[str] = set()
    active_adapter_legs: set[tuple[str, int]] = set()
    for raw_reward in active_rewards:
        _require(
            isinstance(raw_reward, Mapping),
            "SOURCE_ADD active reward row is invalid",
        )
        reward = dict(raw_reward)
        reward_ref = str(reward.get("reward_ref") or "")
        adapter_id = str(reward.get("adapter_id") or "")
        miner_hotkey = str(reward.get("miner_hotkey") or "")
        try:
            leg = int(reward.get("leg") or 0)
            start_epoch = int(reward.get("start_epoch") or -1)
            projection = source_add_reward_row_projection_v2(
                "source_add_leg%d" % leg,
                {**reward, "initial_reward_status": "active"},
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise PrepareActiveReleaseLineageV2Error(
                "SOURCE_ADD active reward row is invalid"
            ) from exc
        expected_kind = {
            1: "source_acceptance",
            2: "source_implementation",
        }.get(leg)
        _require(
            _SOURCE_ADD_REWARD_RE.fullmatch(reward_ref) is not None
            and bool(adapter_id)
            and bool(miner_hotkey)
            and expected_kind is not None
            and reward.get("reward_kind") == expected_kind
            and reward.get("current_reward_status")
            in _SOURCE_ADD_ACTIVE_REWARD_STATUSES
            and start_epoch >= 0,
            "SOURCE_ADD active reward row is invalid",
        )
        _require(
            reward_ref not in active_reward_ids
            and (adapter_id, leg) not in active_adapter_legs,
            "SOURCE_ADD active reward row is duplicated",
        )
        active_reward_ids.add(reward_ref)
        active_adapter_legs.add((adapter_id, leg))
        artifact_hash = sha256_json(projection)
        artifact = ("source_add_reward_decision", reward_ref, artifact_hash)
        artifact_requests.append(artifact)
        expected_roots[artifact] = ("", "research_lab.reward_decision.v2")

    if not artifact_requests:
        return []
    try:
        loaded = await load_business_graphs(tuple(artifact_requests))
    except Exception as exc:
        raise PrepareActiveReleaseLineageV2Error(
            "SOURCE_ADD active release graph lookup failed"
        ) from exc
    _require(
        isinstance(loaded, Mapping) and set(loaded) == set(artifact_requests),
        "SOURCE_ADD active release graphs are incomplete",
    )

    selected: dict[str, dict[str, Any]] = {}
    for artifact in sorted(artifact_requests):
        raw_graph = loaded[artifact]
        _require(
            isinstance(raw_graph, Mapping),
            "SOURCE_ADD active release graph is invalid",
        )
        graph = dict(raw_graph)
        root_hash = str(graph.get("root_receipt_hash") or "")
        expected_root, expected_purpose = expected_roots[artifact]
        try:
            validate_receipt_graph(graph, required_purposes={expected_purpose})
        except Exception as exc:
            raise PrepareActiveReleaseLineageV2Error(
                "SOURCE_ADD active release graph is invalid"
            ) from exc
        roots = [
            receipt
            for receipt in graph.get("receipts") or ()
            if isinstance(receipt, Mapping) and receipt.get("receipt_hash") == root_hash
        ]
        root = roots[0] if len(roots) == 1 else None
        _require(
            _HASH_RE.fullmatch(root_hash) is not None
            and isinstance(root, Mapping)
            and root.get("role") == "gateway_coordinator"
            and root.get("purpose") == expected_purpose
            and root.get("status") == "succeeded"
            and root.get("output_root") == artifact[2]
            and (not expected_root or root_hash == expected_root)
            and (
                expected_purpose != "research_lab.source_add_provenance.v2"
                or root.get("parent_receipt_hashes") == []
            ),
            "SOURCE_ADD active release graph differs from durable authority",
        )
        previous = selected.get(root_hash)
        _require(
            previous is None or canonical_json(previous) == canonical_json(graph),
            "SOURCE_ADD active release graph conflicts for one immutable root",
        )
        selected[root_hash] = graph
    return [selected[root] for root in sorted(selected)]


def prepare_validator_initial_active_lineage_v2(
    *,
    candidate_commit_sha: str,
    authority_commit_sha: str,
    restart_invocation_id: str,
    running_validator_commit_sha: str,
    expected_validator_hotkey: str,
    chain_signing_profile: Mapping[str, Any],
    journal_loader: Callable[[], Optional[Mapping[str, Any]]],
    repository: Path,
    expected_lineage_id: str,
    bucket: str = DEFAULT_BUCKET,
    prefix: str = DEFAULT_PREFIX,
    historical_topology_hash: str | None = None,
    s3_client: Any = None,
) -> Dict[str, Any]:
    """Freeze validator recovery authority while the old validator runs."""

    candidate = _commit(candidate_commit_sha, "candidate commit")
    authority = _commit(authority_commit_sha, "release authority commit")
    invocation_id = _invocation_id(restart_invocation_id)
    running = _commit(running_validator_commit_sha, "running validator commit")
    lineage_id = _lineage_id(expected_lineage_id)
    validator_hotkey = str(expected_validator_hotkey or "")
    profile = validate_chain_signing_profile(chain_signing_profile)
    _require(
        1 <= len(validator_hotkey) <= 128
        and not any(character.isspace() for character in validator_hotkey),
        "expected validator hotkey is invalid",
    )
    _require(callable(journal_loader), "publication journal loader is unavailable")

    first = _journal_requirements(
        journal_loader(),
        expected_lineage_id=lineage_id,
        expected_validator_hotkey=validator_hotkey,
        chain_signing_profile=profile,
    )
    transitions = sorted({running, *first["required_commits"]})
    provisional = build_active_release_requirements_v2(
        candidate_commit_sha=candidate,
        authority_commit_sha=authority,
        restart_invocation_id=invocation_id,
        transition_commit_shas=transitions,
        active_graphs={},
        expected_lineage_id=lineage_id,
        boot_verifier=_structural_boot_verifier,
    )
    lineage = _fetch_exact_release_lineage_v2(
        candidate_commit_sha=candidate,
        authority_commit_sha=authority,
        required_commits=provisional["required_commits"],
        repository=Path(repository),
        bucket=bucket,
        prefix=prefix,
        historical_topology_hash=historical_topology_hash,
        s3_client=s3_client,
    )
    verifier = _selected_compact_boot_verifier(
        lineage,
        historical_topology_hash=historical_topology_hash,
    )
    second = _journal_requirements(
        journal_loader(),
        expected_lineage_id=lineage_id,
        expected_validator_hotkey=validator_hotkey,
        chain_signing_profile=profile,
        boot_verifier=verifier,
    )
    _require(
        second == first,
        "publication journal changed during release selection",
    )
    requirements = build_active_release_requirements_v2(
        candidate_commit_sha=candidate,
        authority_commit_sha=authority,
        restart_invocation_id=invocation_id,
        transition_commit_shas=sorted({running, *second["required_commits"]}),
        active_graphs={},
        expected_lineage_id=lineage_id,
        boot_verifier=verifier,
    )
    _require(
        requirements == provisional,
        "validator active release selection changed during verification",
    )
    return {
        "requirements": requirements,
        "lineage": lineage,
        "journal_hash": second["journal_hash"],
    }


async def prepare_gateway_final_active_lineage_v2(
    *,
    candidate_commit_sha: str,
    authority_commit_sha: str,
    restart_invocation_id: str,
    running_gateway_release_manifest: Mapping[str, Any],
    validator_requirements: Optional[Mapping[str, Any]] = None,
    fallback_lineage: Optional[Mapping[str, Any]] = None,
    fallback_context: Optional[str] = None,
    epoch_id: int,
    netuid: int,
    policy: Mapping[str, Any],
    repository: Path,
    expected_lineage_id: str,
    bucket: str = DEFAULT_BUCKET,
    prefix: str = DEFAULT_PREFIX,
    historical_topology_hash: str | None = None,
    s3_client: Any = None,
    load_allocation_graphs: Any = None,
    load_sourcing_graphs: Any = None,
    load_source_add_graphs: Any = None,
) -> Dict[str, Any]:
    """Select active gateway roots twice and build one exact paired lineage."""

    from gateway.tee.bootstrap_active_ancestry_checkpoints_v2 import (
        _load_frontier_bounded_allocation_graphs,
        _select_active_graphs,
    )

    candidate = _commit(candidate_commit_sha, "candidate commit")
    authority = _commit(authority_commit_sha, "release authority commit")
    invocation_id = _invocation_id(restart_invocation_id)
    lineage_id = _lineage_id(expected_lineage_id)
    _require(
        isinstance(epoch_id, int) and not isinstance(epoch_id, bool) and epoch_id >= 0,
        "active release epoch is invalid",
    )
    _require(
        isinstance(netuid, int) and not isinstance(netuid, bool) and netuid > 0,
        "active release netuid is invalid",
    )
    _require(isinstance(policy, Mapping), "active release policy is invalid")
    release = _validate_selected_gateway_release(
        running_gateway_release_manifest,
        historical_topology_hash=historical_topology_hash,
    )
    running_commit = _commit(release.get("commit_sha"), "running gateway commit")
    _require(
        (validator_requirements is None) != (fallback_lineage is None),
        "exactly one validator requirements or fallback lineage is required",
    )
    if validator_requirements is not None:
        _require(
            fallback_context is None,
            "paired validator requirements cannot use fallback context",
        )
        validator = validate_active_release_requirements_v2(validator_requirements)
        _require(
            validator["candidate_commit_sha"] == candidate,
            "validator requirements target another candidate",
        )
        _require(
            validator["authority_commit_sha"] == authority,
            "validator requirements target another release authority",
        )
        _require(
            validator["ancestry_lineage_id"] == lineage_id,
            "validator requirements target another ancestry lineage",
        )
        _require(
            validator["restart_invocation_id"] == invocation_id,
            "validator requirements target another restart invocation",
        )
        validator_commits = validator["required_commits"]
    else:
        _require(
            fallback_context in _FALLBACK_CONTEXTS,
            "installed lineage fallback requires an explicit safe context",
        )
        fallback = _validate_selected_lineage(
            fallback_lineage,
            historical_topology_hash=historical_topology_hash,
            expected_current_commit=running_commit,
            expected_current_gateway_release_hash=release["release_hash"],
            allow_historical_current=True,
        )
        validator_commits = sorted(fallback["releases"])

    if load_allocation_graphs is None:
        load_allocation_graphs = _load_frontier_bounded_allocation_graphs
    if load_sourcing_graphs is None:
        from gateway.research_lab.attested_v2_store import (
            load_sourcing_epoch_graphs_v2,
        )

        load_sourcing_graphs = load_sourcing_epoch_graphs_v2
    if load_source_add_graphs is None:
        load_source_add_graphs = _load_active_source_add_graphs_v2
    transitions = sorted({running_commit, *validator_commits})
    first_graphs = await _select_active_graphs(
        epoch_id=epoch_id,
        netuid=netuid,
        policy=dict(policy),
        load_allocation_graphs=load_allocation_graphs,
        load_sourcing_graphs=load_sourcing_graphs,
        load_source_add_graphs=load_source_add_graphs,
    )
    first = build_active_release_requirements_v2(
        candidate_commit_sha=candidate,
        authority_commit_sha=authority,
        restart_invocation_id=invocation_id,
        transition_commit_shas=transitions,
        active_graphs=first_graphs,
        expected_lineage_id=lineage_id,
        boot_verifier=_structural_boot_verifier,
    )
    lineage = _fetch_exact_release_lineage_v2(
        candidate_commit_sha=candidate,
        authority_commit_sha=authority,
        required_commits=first["required_commits"],
        repository=Path(repository),
        bucket=bucket,
        prefix=prefix,
        historical_topology_hash=historical_topology_hash,
        s3_client=s3_client,
    )
    verifier = _selected_compact_boot_verifier(
        lineage,
        historical_topology_hash=historical_topology_hash,
    )
    second_graphs = await _select_active_graphs(
        epoch_id=epoch_id,
        netuid=netuid,
        policy=dict(policy),
        load_allocation_graphs=load_allocation_graphs,
        load_sourcing_graphs=load_sourcing_graphs,
        load_source_add_graphs=load_source_add_graphs,
    )
    _require(
        canonical_json(first_graphs) == canonical_json(second_graphs),
        "active receipt graph selection changed during release verification",
    )
    second = build_active_release_requirements_v2(
        candidate_commit_sha=candidate,
        authority_commit_sha=authority,
        restart_invocation_id=invocation_id,
        transition_commit_shas=transitions,
        active_graphs=second_graphs,
        expected_lineage_id=lineage_id,
        boot_verifier=verifier,
    )
    _require(
        second == first,
        "active release requirement set changed during verification",
    )
    return {"requirements": second, "lineage": lineage}


def prepare_validator_final_active_lineage_v2(
    *,
    candidate_commit_sha: str,
    authority_commit_sha: str,
    restart_invocation_id: str,
    initial_requirements: Mapping[str, Any],
    final_requirements: Mapping[str, Any],
    handed_lineage: Mapping[str, Any],
    journal_loader: Callable[[], Optional[Mapping[str, Any]]],
    expected_validator_hotkey: str,
    chain_signing_profile: Mapping[str, Any],
    repository: Path,
    expected_lineage_id: str,
    bucket: str = DEFAULT_BUCKET,
    prefix: str = DEFAULT_PREFIX,
    historical_topology_hash: str | None = None,
    s3_client: Any = None,
) -> Dict[str, Any]:
    """Independently rebuild and install-check the gateway's final lineage."""

    candidate = _commit(candidate_commit_sha, "candidate commit")
    authority = _commit(authority_commit_sha, "release authority commit")
    invocation_id = _invocation_id(restart_invocation_id)
    lineage_id = _lineage_id(expected_lineage_id)
    validator_hotkey = str(expected_validator_hotkey or "")
    profile = validate_chain_signing_profile(chain_signing_profile)
    _require(
        1 <= len(validator_hotkey) <= 128
        and not any(character.isspace() for character in validator_hotkey),
        "expected validator hotkey is invalid",
    )
    initial = validate_active_release_requirements_v2(initial_requirements)
    final = validate_active_release_requirements_v2(final_requirements)
    for label, requirements in (("initial", initial), ("final", final)):
        _require(
            requirements["candidate_commit_sha"] == candidate,
            "%s requirements target another candidate" % label,
        )
        _require(
            requirements["authority_commit_sha"] == authority,
            "%s requirements target another release authority" % label,
        )
        _require(
            requirements["ancestry_lineage_id"] == lineage_id,
            "%s requirements target another ancestry lineage" % label,
        )
        _require(
            requirements["restart_invocation_id"] == invocation_id,
            "%s requirements target another restart invocation" % label,
        )
    _require(
        set(initial["required_commits"]).issubset(set(final["transition_commit_shas"])),
        "final requirements omit validator transition authority",
    )

    handed = _validate_selected_lineage(
        handed_lineage,
        historical_topology_hash=historical_topology_hash,
        expected_current_commit=candidate,
    )
    _require(
        sorted(handed["releases"]) == final["required_commits"],
        "handed release lineage differs from final requirements",
    )
    independent = _fetch_exact_release_lineage_v2(
        candidate_commit_sha=candidate,
        authority_commit_sha=authority,
        required_commits=final["required_commits"],
        repository=Path(repository),
        bucket=bucket,
        prefix=prefix,
        historical_topology_hash=historical_topology_hash,
        s3_client=s3_client,
    )
    _require(
        canonical_json(handed) == canonical_json(independent),
        "handed release lineage differs from independent readback",
    )
    verifier = _selected_compact_boot_verifier(
        independent,
        historical_topology_hash=historical_topology_hash,
    )
    first_journal = _journal_requirements(
        journal_loader(),
        expected_lineage_id=lineage_id,
        expected_validator_hotkey=validator_hotkey,
        chain_signing_profile=profile,
        boot_verifier=verifier,
    )
    _require(
        set(first_journal["required_commits"]).issubset(set(final["required_commits"])),
        "current publication journal requires an uncovered release",
    )
    second_journal = _journal_requirements(
        journal_loader(),
        expected_lineage_id=lineage_id,
        expected_validator_hotkey=validator_hotkey,
        chain_signing_profile=profile,
        boot_verifier=verifier,
    )
    _require(
        second_journal == first_journal,
        "publication journal changed before install",
    )
    _require(
        set(second_journal["required_commits"]).issubset(
            set(final["required_commits"])
        ),
        "current publication journal requires an uncovered release",
    )
    return {
        "requirements": final,
        "lineage": independent,
        "journal_hash": second_journal["journal_hash"],
    }


def _gateway_final_authority_commit(
    requested_authority_commit: str,
    validator_requirements: Optional[Mapping[str, Any]],
) -> str:
    """Select the paired validator's already-validated release authority."""

    requested = _commit(requested_authority_commit, "release authority commit")
    if validator_requirements is None:
        return requested
    requirements = validate_active_release_requirements_v2(
        validator_requirements
    )
    # The installed N-1 gateway controller removes restart-only authority
    # variables at the miner-maintenance bootstrap boundary. Its shell fallback
    # can therefore observe a newer origin/main. The invocation-bound validator
    # sidecar is the paired authority; the full gateway-final verifier below
    # still binds its candidate, invocation, lineage, and selected roots.
    return str(requirements["authority_commit_sha"])


def _result(*, mode: str, prepared: Mapping[str, Any]) -> Dict[str, Any]:
    requirements = prepared["requirements"]
    lineage = prepared["lineage"]
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "mode": mode,
        "status": "complete",
        "candidate_commit_sha": requirements["candidate_commit_sha"],
        "authority_commit_sha": requirements["authority_commit_sha"],
        "restart_invocation_id": requirements["restart_invocation_id"],
        "selection_hash": requirements["selection_hash"],
        "required_release_count": len(requirements["required_commits"]),
        "lineage_hash": lineage["lineage_hash"],
        "journal_hash": prepared.get("journal_hash"),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("validator-initial", "gateway-final", "validator-final"),
        required=True,
    )
    parser.add_argument("--candidate-commit", required=True)
    parser.add_argument("--authority-commit", required=True)
    parser.add_argument("--restart-invocation-id", required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--lineage-id", required=True)
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--historical-topology-hash")
    parser.add_argument("--running-validator-commit")
    parser.add_argument("--running-gateway-manifest", type=Path)
    gateway_authority = parser.add_mutually_exclusive_group()
    gateway_authority.add_argument("--validator-requirements", type=Path)
    gateway_authority.add_argument("--fallback-lineage", type=Path)
    parser.add_argument(
        "--fallback-context",
        choices=tuple(sorted(_FALLBACK_CONTEXTS)),
    )
    parser.add_argument("--epoch", type=int)
    parser.add_argument("--netuid", type=int)
    parser.add_argument("--initial-requirements", type=Path)
    parser.add_argument("--final-requirements-input", type=Path)
    parser.add_argument("--lineage-input", type=Path)
    parser.add_argument("--journal", type=Path)
    parser.add_argument("--validator-hotkey-config", type=Path)
    parser.add_argument("--chain-signing-profile", type=Path)
    parser.add_argument("--requirements-output", type=Path)
    parser.add_argument("--lineage-output", type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = list(os.sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] in {
        "validator-initial",
        "gateway-final",
        "validator-final",
    }:
        arguments = ["--phase", arguments[0], *arguments[1:]]
    args = _parser().parse_args(arguments)
    if args.phase in {"validator-initial", "validator-final"}:
        _require(
            args.validator_hotkey_config is not None
            and args.chain_signing_profile is not None,
            "%s requires validator authority configuration" % args.phase,
        )
        validator_context = _load_validator_authority_context(
            hotkey_config_path=args.validator_hotkey_config,
            chain_signing_profile_path=args.chain_signing_profile,
        )
        expected_validator_hotkey = validator_context["validator_hotkey"]
        chain_signing_profile = validator_context["chain_signing_profile"]
    else:
        _require(
            args.validator_hotkey_config is None,
            "gateway-final cannot receive validator hotkey configuration",
        )
        _require(
            args.chain_signing_profile is None,
            "gateway-final cannot receive validator chain signing profile",
        )
        expected_validator_hotkey = None
        chain_signing_profile = None
    if args.phase == "validator-initial":
        _require(
            args.running_validator_commit is not None
            and args.journal is not None
            and args.requirements_output is not None,
            "validator-initial arguments are incomplete",
        )
        prepared = prepare_validator_initial_active_lineage_v2(
            candidate_commit_sha=args.candidate_commit,
            authority_commit_sha=args.authority_commit,
            restart_invocation_id=args.restart_invocation_id,
            running_validator_commit_sha=args.running_validator_commit,
            expected_validator_hotkey=str(expected_validator_hotkey),
            chain_signing_profile=dict(chain_signing_profile or {}),
            journal_loader=lambda: _load_optional_journal(args.journal),
            repository=args.repository,
            expected_lineage_id=args.lineage_id,
            bucket=args.bucket,
            prefix=args.prefix,
            historical_topology_hash=args.historical_topology_hash,
        )
        _atomic_json_documents(((args.requirements_output, prepared["requirements"]),))
    elif args.phase == "gateway-final":
        _require(
            args.running_gateway_manifest is not None
            and (args.validator_requirements is None) != (args.fallback_lineage is None)
            and (
                (args.fallback_lineage is not None)
                == (args.fallback_context is not None)
            )
            and args.epoch is not None
            and args.netuid is not None
            and args.requirements_output is not None
            and args.lineage_output is not None,
            "gateway-final arguments are incomplete",
        )
        from gateway.research_lab.config import ResearchLabGatewayConfig

        policy = ResearchLabGatewayConfig.from_env().reimbursement_policy_doc(
            enabled=True
        )
        validator_requirements = (
            _load_json(
                args.validator_requirements,
                "validator active release requirements",
            )
            if args.validator_requirements is not None
            else None
        )
        authority_commit = _gateway_final_authority_commit(
            args.authority_commit,
            validator_requirements,
        )
        prepared = asyncio.run(
            prepare_gateway_final_active_lineage_v2(
                candidate_commit_sha=args.candidate_commit,
                authority_commit_sha=authority_commit,
                restart_invocation_id=args.restart_invocation_id,
                running_gateway_release_manifest=_load_json(
                    args.running_gateway_manifest,
                    "running gateway release manifest",
                ),
                validator_requirements=validator_requirements,
                fallback_lineage=(
                    _load_json(
                        args.fallback_lineage,
                        "installed fallback compact release lineage",
                    )
                    if args.fallback_lineage is not None
                    else None
                ),
                fallback_context=args.fallback_context,
                epoch_id=args.epoch,
                netuid=args.netuid,
                policy=policy,
                repository=args.repository,
                expected_lineage_id=args.lineage_id,
                bucket=args.bucket,
                prefix=args.prefix,
                historical_topology_hash=args.historical_topology_hash,
            )
        )
        _atomic_json_documents(
            (
                (args.requirements_output, prepared["requirements"]),
                (args.lineage_output, prepared["lineage"]),
            )
        )
    else:
        _require(
            args.initial_requirements is not None
            and args.final_requirements_input is not None
            and args.lineage_input is not None
            and args.journal is not None
            and args.requirements_output is not None
            and args.lineage_output is not None,
            "validator-final arguments are incomplete",
        )
        prepared = prepare_validator_final_active_lineage_v2(
            candidate_commit_sha=args.candidate_commit,
            authority_commit_sha=args.authority_commit,
            restart_invocation_id=args.restart_invocation_id,
            initial_requirements=_load_json(
                args.initial_requirements,
                "initial validator active release requirements",
            ),
            final_requirements=_load_json(
                args.final_requirements_input,
                "final active release requirements",
            ),
            handed_lineage=_load_json(
                args.lineage_input,
                "handed compact release lineage",
            ),
            journal_loader=lambda: _load_optional_journal(args.journal),
            expected_validator_hotkey=str(expected_validator_hotkey),
            chain_signing_profile=dict(chain_signing_profile or {}),
            repository=args.repository,
            expected_lineage_id=args.lineage_id,
            bucket=args.bucket,
            prefix=args.prefix,
            historical_topology_hash=args.historical_topology_hash,
        )
        _atomic_json_documents(
            (
                (args.requirements_output, prepared["requirements"]),
                (args.lineage_output, prepared["lineage"]),
            )
        )
    print(json.dumps(_result(mode=args.phase, prepared=prepared), sort_keys=True))
    return 0


def cli(argv: Optional[Sequence[str]] = None) -> int:
    try:
        return main(argv)
    except Exception as exc:
        print("active release lineage preparation failed: %s" % exc, file=os.sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(cli())


__all__ = [
    "PrepareActiveReleaseLineageV2Error",
    "prepare_gateway_final_active_lineage_v2",
    "prepare_validator_final_active_lineage_v2",
    "prepare_validator_initial_active_lineage_v2",
]
