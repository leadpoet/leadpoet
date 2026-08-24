"""Derive the exact release set required by active bounded V2 ancestry.

This module deliberately performs no I/O.  Callers must supply the exact
active root-to-graph mapping and a boot verifier that binds every disclosed
identity to an approved release and Nitro attestation.
"""

from __future__ import annotations

from copy import deepcopy
import re
from typing import Any, Callable, Dict, Mapping, Sequence

from leadpoet_canonical.ancestry_checkpoint_v2 import (
    validate_compact_ancestry_proof_v2,
)
from leadpoet_canonical.attested_v2 import (
    CHECKPOINTED_RECEIPT_GRAPH_SCHEMA_VERSIONS,
    RECEIPT_GRAPH_SCHEMA_VERSION,
    ROLE_PURPOSES,
    canonical_json,
    sha256_json,
    validate_boot_identity,
    validate_receipt_graph,
)
from leadpoet_canonical.hotkey_authority_v2 import (
    MAX_WEIGHT_TRANSPORT_LOGICAL_BYTES,
)


ACTIVE_RELEASE_REQUIREMENTS_SCHEMA_VERSION = "leadpoet.active_release_requirements.v2"
ACTIVE_RELEASE_ROOT_SET_SCHEMA_VERSION = "leadpoet.active_release_root_set.v2"
ACTIVE_RELEASE_COMMIT_SET_SCHEMA_VERSION = "leadpoet.active_release_commit_set.v2"
MAX_ACTIVE_RELEASE_COMMITS = 512
MAX_ACTIVE_RELEASE_ROOTS = 10_000
MAX_ACTIVE_RELEASE_GRAPH_BYTES = MAX_WEIGHT_TRANSPORT_LOGICAL_BYTES

_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_INVOCATION_RE = re.compile(r"^[a-z0-9][a-z0-9_.:-]{0,127}$")
_REQUIREMENTS_FIELDS = frozenset(
    {
        "schema_version",
        "candidate_commit_sha",
        "authority_commit_sha",
        "restart_invocation_id",
        "transition_commit_shas",
        "ancestry_lineage_id",
        "commits_by_root",
        "root_set_hash",
        "required_commits",
        "required_set_hash",
        "selection_hash",
    }
)


class ActiveReleaseRequirementsV2Error(RuntimeError):
    """Active ancestry cannot be reduced to one exact approved release set."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ActiveReleaseRequirementsV2Error(message)


def _commit(value: Any, label: str) -> str:
    _require(
        isinstance(value, str) and _COMMIT_RE.fullmatch(value) is not None,
        "%s is invalid" % label,
    )
    return value


def _hash(value: Any, label: str) -> str:
    _require(
        isinstance(value, str) and _HASH_RE.fullmatch(value) is not None,
        "%s is invalid" % label,
    )
    return value


def _invocation(value: Any) -> str:
    _require(
        isinstance(value, str) and _INVOCATION_RE.fullmatch(value) is not None,
        "restart invocation id is invalid",
    )
    return value


def _commit_list(value: Any, label: str) -> list[str]:
    _require(isinstance(value, list), "%s must be an array" % label)
    commits = [_commit(item, "%s commit" % label) for item in value]
    _require(
        commits == sorted(set(commits)),
        "%s is not sorted and unique" % label,
    )
    return commits


def _root_set_document(roots: Sequence[str]) -> Dict[str, Any]:
    return {
        "schema_version": ACTIVE_RELEASE_ROOT_SET_SCHEMA_VERSION,
        "root_receipt_hashes": list(roots),
    }


def _required_set_document(commits: Sequence[str]) -> Dict[str, Any]:
    return {
        "schema_version": ACTIVE_RELEASE_COMMIT_SET_SCHEMA_VERSION,
        "commit_shas": list(commits),
    }


def validate_active_release_requirements_v2(
    value: Mapping[str, Any],
) -> Dict[str, Any]:
    """Validate one canonical, hash-bound active release selection sidecar."""

    _require(
        isinstance(value, Mapping) and set(value) == _REQUIREMENTS_FIELDS,
        "active release requirements fields are invalid",
    )
    _require(
        value.get("schema_version") == ACTIVE_RELEASE_REQUIREMENTS_SCHEMA_VERSION,
        "active release requirements schema is invalid",
    )
    candidate = _commit(value.get("candidate_commit_sha"), "candidate commit")
    authority = _commit(value.get("authority_commit_sha"), "authority commit")
    invocation_id = _invocation(value.get("restart_invocation_id"))
    transitions = _commit_list(
        value.get("transition_commit_shas"), "transition commits"
    )
    lineage_id = _hash(value.get("ancestry_lineage_id"), "ancestry lineage id")

    raw_by_root = value.get("commits_by_root")
    _require(
        isinstance(raw_by_root, Mapping),
        "active release root commit mapping is invalid",
    )
    _require(
        len(raw_by_root) <= MAX_ACTIVE_RELEASE_ROOTS,
        "active release root count exceeds bound",
    )
    commits_by_root: Dict[str, list[str]] = {}
    for raw_root, raw_commits in raw_by_root.items():
        root = _hash(raw_root, "active receipt root")
        _require(
            raw_root == root,
            "active receipt root is not canonical",
        )
        commits_by_root[root] = _commit_list(raw_commits, "active receipt root commits")
        _require(
            bool(commits_by_root[root]),
            "active receipt root commits are empty",
        )
    roots = sorted(commits_by_root)
    expected_root_set_hash = sha256_json(_root_set_document(roots))
    _require(
        value.get("root_set_hash") == expected_root_set_hash,
        "active release root set hash differs",
    )

    required = _commit_list(value.get("required_commits"), "required commits")
    expected_required = sorted(
        {
            candidate,
            *transitions,
            *(
                commit
                for root_commits in commits_by_root.values()
                for commit in root_commits
            ),
        }
    )
    _require(
        required == expected_required,
        "active release required commits differ from selected ancestry",
    )
    _require(
        len(required) <= MAX_ACTIVE_RELEASE_COMMITS,
        "active release required commits exceed bound",
    )
    expected_required_set_hash = sha256_json(_required_set_document(required))
    _require(
        value.get("required_set_hash") == expected_required_set_hash,
        "active release required set hash differs",
    )

    body = {
        "schema_version": ACTIVE_RELEASE_REQUIREMENTS_SCHEMA_VERSION,
        "candidate_commit_sha": candidate,
        "authority_commit_sha": authority,
        "restart_invocation_id": invocation_id,
        "transition_commit_shas": transitions,
        "ancestry_lineage_id": lineage_id,
        "commits_by_root": {root: commits_by_root[root] for root in roots},
        "root_set_hash": expected_root_set_hash,
        "required_commits": required,
        "required_set_hash": expected_required_set_hash,
    }
    _require(
        value.get("selection_hash") == sha256_json(body),
        "active release selection hash differs",
    )
    return {**body, "selection_hash": str(value["selection_hash"])}


def build_active_release_requirements_v2(
    *,
    candidate_commit_sha: str,
    authority_commit_sha: str,
    restart_invocation_id: str,
    transition_commit_shas: Sequence[str],
    active_graphs: Mapping[str, Mapping[str, Any]],
    expected_lineage_id: str,
    boot_verifier: Callable[[Mapping[str, Any]], Any],
) -> Dict[str, Any]:
    """Validate active bounded graphs and select every disclosed release.

    Parent authority descriptors are intentionally not traversed.  Their
    hashes authenticate omitted history, but only full boot identities
    disclosed by the active graph or its certificate issuer select releases.
    """

    candidate = _commit(candidate_commit_sha, "candidate commit")
    authority = _commit(authority_commit_sha, "authority commit")
    invocation_id = _invocation(restart_invocation_id)
    _require(
        isinstance(transition_commit_shas, Sequence)
        and not isinstance(transition_commit_shas, (str, bytes)),
        "transition commits must be a sequence",
    )
    transitions = sorted(
        {_commit(item, "transition commit") for item in transition_commit_shas}
    )
    lineage_id = _hash(expected_lineage_id, "expected ancestry lineage id")
    _require(callable(boot_verifier), "boot verifier is unavailable")
    _require(
        isinstance(active_graphs, Mapping),
        "active receipt graph mapping is invalid",
    )
    _require(
        len(active_graphs) <= MAX_ACTIVE_RELEASE_ROOTS,
        "active release root count exceeds bound",
    )
    verified_identities: Dict[str, Dict[str, Any]] = {}

    def checked_boot_verifier(identity: Mapping[str, Any]) -> Mapping[str, Any]:
        try:
            validate_boot_identity(identity)
            identity_hash = str(identity["boot_identity_hash"])
            previous = verified_identities.get(identity_hash)
            if previous is not None:
                _require(
                    previous == dict(identity),
                    "active ancestry boot identity conflicts",
                )
                return dict(previous)
            verified = boot_verifier(identity)
            _require(
                isinstance(verified, Mapping),
                "active ancestry boot verifier returned no evidence",
            )
            stored = dict(identity)
            verified_identities[identity_hash] = stored
            return dict(stored)
        except ActiveReleaseRequirementsV2Error:
            raise
        except Exception as exc:
            raise ActiveReleaseRequirementsV2Error(
                "active ancestry boot identity is invalid or unapproved"
            ) from exc

    graph_mapping = dict(active_graphs)
    raw_roots = list(graph_mapping)
    for raw_root in raw_roots:
        _hash(raw_root, "active receipt root")

    commits_by_root: Dict[str, list[str]] = {}
    active_graph_bytes = 0
    for raw_root in sorted(raw_roots):
        root = _hash(raw_root, "active receipt root")
        _require(raw_root == root, "active receipt root is not canonical")
        raw_graph = graph_mapping[raw_root]
        _require(
            isinstance(raw_graph, Mapping),
            "active receipt graph is invalid",
        )
        try:
            graph_bytes = len(canonical_json(raw_graph).encode("utf-8"))
            graph = deepcopy(raw_graph)
        except Exception as exc:
            raise ActiveReleaseRequirementsV2Error(
                "active receipt graph is not canonical JSON"
            ) from exc
        active_graph_bytes += graph_bytes
        _require(
            active_graph_bytes <= MAX_ACTIVE_RELEASE_GRAPH_BYTES,
            "active receipt graph bytes exceed bound",
        )
        graph_schema = graph.get("schema_version")
        _require(
            graph_schema
            in {
                RECEIPT_GRAPH_SCHEMA_VERSION,
                *CHECKPOINTED_RECEIPT_GRAPH_SCHEMA_VERSIONS,
            },
            "active receipt graph schema is invalid",
        )
        _require(
            graph.get("root_receipt_hash") == root,
            "active receipt graph root differs from mapping key",
        )
        if graph_schema in CHECKPOINTED_RECEIPT_GRAPH_SCHEMA_VERSIONS:
            _require(
                graph.get("ancestry_lineage_id") == lineage_id,
                "active receipt graph lineage differs",
            )
        try:
            validate_receipt_graph(
                graph,
                boot_attestation_verifier=checked_boot_verifier,
                require_boot_attestation_verification=True,
            )
            proof = None
            if graph_schema in CHECKPOINTED_RECEIPT_GRAPH_SCHEMA_VERSIONS:
                proof = validate_compact_ancestry_proof_v2(
                    graph.get("ancestry_proof"),
                    expected_lineage_id=lineage_id,
                    boot_attestation_verifier=checked_boot_verifier,
                    allowed_issuer_roles=ROLE_PURPOSES,
                    required_receipt_hashes=(root,),
                )
        except Exception as exc:
            raise ActiveReleaseRequirementsV2Error(
                "active receipt graph or ancestry proof is invalid"
            ) from exc

        identities = list(graph.get("boot_identities") or ())
        if proof is not None:
            identities.extend(proof.get("disclosed_boot_identities") or ())
            certificate = proof.get("certificate")
            issuer = (
                certificate.get("issuer_boot_identity")
                if isinstance(certificate, Mapping)
                else None
            )
            _require(
                isinstance(issuer, Mapping),
                "active ancestry proof certificate issuer is missing",
            )
            identities.append(issuer)
        root_commits = set()
        for identity in identities:
            _require(
                isinstance(identity, Mapping),
                "active ancestry boot identity is invalid",
            )
            checked_boot_verifier(identity)
            root_commits.add(
                _commit(
                    identity.get("commit_sha"),
                    "active ancestry boot commit",
                )
            )
        commits_by_root[root] = sorted(root_commits)

    roots = sorted(commits_by_root)
    required = sorted(
        {
            candidate,
            *transitions,
            *(
                commit
                for root_commits in commits_by_root.values()
                for commit in root_commits
            ),
        }
    )
    _require(
        len(required) <= MAX_ACTIVE_RELEASE_COMMITS,
        "active release required commits exceed bound",
    )
    body = {
        "schema_version": ACTIVE_RELEASE_REQUIREMENTS_SCHEMA_VERSION,
        "candidate_commit_sha": candidate,
        "authority_commit_sha": authority,
        "restart_invocation_id": invocation_id,
        "transition_commit_shas": transitions,
        "ancestry_lineage_id": lineage_id,
        "commits_by_root": {root: commits_by_root[root] for root in roots},
        "root_set_hash": sha256_json(_root_set_document(roots)),
        "required_commits": required,
        "required_set_hash": sha256_json(_required_set_document(required)),
    }
    return validate_active_release_requirements_v2(
        {**body, "selection_hash": sha256_json(body)}
    )


__all__ = [
    "ACTIVE_RELEASE_REQUIREMENTS_SCHEMA_VERSION",
    "ActiveReleaseRequirementsV2Error",
    "MAX_ACTIVE_RELEASE_GRAPH_BYTES",
    "MAX_ACTIVE_RELEASE_COMMITS",
    "MAX_ACTIVE_RELEASE_ROOTS",
    "build_active_release_requirements_v2",
    "validate_active_release_requirements_v2",
]
