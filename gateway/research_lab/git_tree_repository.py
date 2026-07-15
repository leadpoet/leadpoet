"""Private SHA-256 Git repository used by one autoresearch tree."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any, Iterator, Mapping, Sequence

from gateway.research_lab.git_tree_models import TreeChildSlot
from leadpoet_canonical.attested_v2 import sha256_bytes, sha256_json
from research_lab.code_editing import CodeEditDraft
from research_lab.eval.private_runtime import compute_private_source_tree_hash


GIT_TREE_METADATA_SCHEMA_VERSION = "research_lab.git_tree_repository.v1"
GIT_TREE_COMMIT_SCHEMA_VERSION = "research_lab.git_tree_commit.v1"
GIT_TREE_CONTROL_SCHEMA_VERSION = "research_lab.git_tree_control.v2"
GIT_TREE_RECOVERY_SCHEMA_VERSION = "research_lab.git_tree_recovery.v1"

_GIT_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_TREE_ID_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_NODE_ID_RE = re.compile(r"^tree-node:[0-9a-f]{64}$")


class GitTreeRepositoryError(RuntimeError):
    """A Git tree operation failed or produced an unverifiable state."""


@dataclass(frozen=True)
class GitTreeCommit:
    tree_id: str
    node_id: str
    parent_node_id: str
    root_branch_id: str
    depth: int
    slot_index: int
    git_commit: str
    parent_git_commit: str
    source_tree_hash: str
    draft_patch_hash: str
    incremental_patch: str
    incremental_patch_hash: str
    cumulative_patch: str
    cumulative_patch_hash: str
    changed_files: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": GIT_TREE_COMMIT_SCHEMA_VERSION,
            "tree_id": self.tree_id,
            "node_id": self.node_id,
            "parent_node_id": self.parent_node_id,
            "root_branch_id": self.root_branch_id,
            "depth": self.depth,
            "slot_index": self.slot_index,
            "git_commit": self.git_commit,
            "parent_git_commit": self.parent_git_commit,
            "source_tree_hash": self.source_tree_hash,
            "draft_patch_hash": self.draft_patch_hash,
            "incremental_patch_hash": self.incremental_patch_hash,
            "cumulative_patch_hash": self.cumulative_patch_hash,
            "changed_files": list(self.changed_files),
        }


def _run_git(
    args: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str] | None = None,
    timeout_seconds: int = 120,
    check: bool = True,
) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        env={**os.environ, **dict(env or {})},
        text=True,
        capture_output=True,
        timeout=timeout_seconds,
        check=False,
    )
    if check and completed.returncode != 0:
        raise GitTreeRepositoryError(
            "git %s failed (%s): %s"
            % (
                " ".join(args[:4]),
                completed.returncode,
                str(completed.stderr or completed.stdout or "")[-1200:],
            )
        )
    return str(completed.stdout or "")


def _copy_source_tree(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(
        source,
        destination,
        symlinks=False,
        ignore=shutil.ignore_patterns(".git", ".research_lab", "__pycache__", "*.pyc"),
    )


def _safe_ref(node_id: str) -> str:
    if not _NODE_ID_RE.fullmatch(str(node_id or "")):
        raise GitTreeRepositoryError("tree node id is invalid")
    return "refs/heads/nodes/" + node_id.split(":", 1)[1]


def operation_settlement_commitment(
    *, tree_id: str, operations: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Normalize and hash terminal operation rows from either persistence layer."""

    if not _TREE_ID_RE.fullmatch(str(tree_id or "")):
        raise GitTreeRepositoryError("tree id is invalid")
    projections: list[dict[str, Any]] = []
    for raw_operation in operations:
        payload = dict(raw_operation)
        operation_id = str(
            payload.get("operation_id")
            or payload.get("logical_operation_id")
            or ""
        )
        status = str(
            payload.get("status") or payload.get("operation_status") or ""
        )
        if status == "reserved":
            raise GitTreeRepositoryError(
                "tree checkpoint cannot commit a reserved operation"
            )
        if (
            not _TREE_ID_RE.fullmatch(operation_id)
            or str(payload.get("tree_id") or "") != tree_id
            or status not in {"succeeded", "failed", "indeterminate"}
        ):
            raise GitTreeRepositoryError(
                "tree operation terminal commitment is invalid"
            )
        projections.append(
            {
                "operation_id": operation_id,
                "node_id": str(payload.get("node_id") or ""),
                "operation_kind": str(payload.get("operation_kind") or ""),
                "status": status,
                "request_hash": str(payload.get("request_hash") or ""),
                "result_hash": str(payload.get("result_hash") or ""),
                "settled_cost_microusd": max(
                    0, int(payload.get("settled_cost_microusd") or 0)
                ),
                "provider_call_count": max(
                    0, int(payload.get("provider_call_count") or 0)
                ),
            }
        )
    projections.sort(key=lambda item: item["operation_id"])
    if len({item["operation_id"] for item in projections}) != len(projections):
        raise GitTreeRepositoryError(
            "tree operation settlement contains duplicate identities"
        )
    document = {
        "schema_version": "research_lab.git_tree_operation_settlements.v1",
        "tree_id": tree_id,
        "operations": projections,
    }
    return {
        "operation_count": len(projections),
        "settled_cost_microusd": sum(
            item["settled_cost_microusd"] for item in projections
        ),
        "provider_call_count": sum(
            item["provider_call_count"] for item in projections
        ),
        "operation_settlement_hash": sha256_json(document),
    }


class GitTreeRepository:
    """Crash-safe direct-parent Git lineage for one immutable tree root."""

    def __init__(self, *, workspace: Path, tree_id: str) -> None:
        if not _TREE_ID_RE.fullmatch(str(tree_id or "")):
            raise GitTreeRepositoryError("tree id is invalid")
        self.workspace = Path(workspace).resolve()
        self.tree_id = str(tree_id)
        self.repo_dir = self.workspace / "repository"
        self.worktree_dir = self.workspace / "worktrees"
        self.metadata_path = self.workspace / "TREE.json"
        self.control_path = self.workspace / "CONTROL.json"
        self.lock_path = self.workspace / ".tree.lock"

    @contextmanager
    def _locked(self) -> Iterator[None]:
        self.workspace.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+b") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def initialize(
        self,
        *,
        source_root: Path,
        root_artifact_hash: str,
        policy_hash: str,
        run_id: str = "",
        root_manifest_hash: str = "",
        root_image_digest: str = "",
        evaluator_commitment_hash: str = "",
        tree_doc: Mapping[str, Any] | None = None,
    ) -> str:
        source_root = Path(source_root).resolve()
        if not source_root.is_dir():
            raise GitTreeRepositoryError("tree root source directory is missing")
        with self._locked():
            if self.metadata_path.exists():
                if not self.repo_dir.is_dir() or not self.control_path.is_file():
                    raise GitTreeRepositoryError(
                        "existing tree workspace is incomplete"
                    )
                metadata = self._load_metadata()
                control = self._load_control()
                expected = {
                    "tree_id": self.tree_id,
                    "root_artifact_hash": str(root_artifact_hash),
                    "policy_hash": str(policy_hash),
                }
                optional_expected = {
                    "run_id": str(run_id),
                    "root_manifest_hash": str(root_manifest_hash),
                    "root_image_digest": str(root_image_digest),
                    "evaluator_commitment_hash": str(evaluator_commitment_hash),
                    "tree_doc_hash": sha256_json(dict(tree_doc or {})),
                }
                expected.update(optional_expected)
                for key, value in expected.items():
                    if metadata.get(key) != value:
                        raise GitTreeRepositoryError(
                            f"existing tree {key} differs from requested root"
                        )
                root_commit = str(metadata.get("root_git_commit") or "")
                self._verify_repository_state(metadata=metadata, control=control)
                return root_commit

            if self.repo_dir.exists():
                raise GitTreeRepositoryError(
                    "tree repository exists without authoritative metadata"
                )
            self.workspace.mkdir(parents=True, exist_ok=True)
            _copy_source_tree(source_root, self.repo_dir)
            _run_git(["init", "--quiet", "--object-format=sha256"], cwd=self.repo_dir)
            object_format = _run_git(
                ["rev-parse", "--show-object-format"], cwd=self.repo_dir
            ).strip()
            if object_format != "sha256":
                raise GitTreeRepositoryError("Git tree repository is not SHA-256")
            _run_git(["config", "user.name", "Leadpoet Research Lab"], cwd=self.repo_dir)
            _run_git(
                ["config", "user.email", "research-lab@leadpoet.local"],
                cwd=self.repo_dir,
            )
            _run_git(["add", "-A"], cwd=self.repo_dir)
            _run_git(
                ["commit", "--quiet", "-m", "Research Lab immutable tree root"],
                cwd=self.repo_dir,
                env=self._commit_env(0),
            )
            root_commit = _run_git(["rev-parse", "HEAD"], cwd=self.repo_dir).strip()
            self._verify_commit(root_commit)
            source_tree_hash = compute_private_source_tree_hash(self.repo_dir)
            metadata_body = {
                "schema_version": GIT_TREE_METADATA_SCHEMA_VERSION,
                "tree_id": self.tree_id,
                "root_artifact_hash": str(root_artifact_hash),
                "policy_hash": str(policy_hash),
                "root_git_commit": root_commit,
                "root_source_tree_hash": source_tree_hash,
                "object_format": object_format,
                "run_id": str(run_id),
                "root_manifest_hash": str(root_manifest_hash),
                "root_image_digest": str(root_image_digest),
                "evaluator_commitment_hash": str(evaluator_commitment_hash),
                "tree_doc_hash": sha256_json(dict(tree_doc or {})),
            }
            metadata = {
                **metadata_body,
                "metadata_hash": sha256_json(metadata_body),
            }
            self._atomic_json_write(self.metadata_path, metadata)
            self._write_control(
                {
                    "schema_version": GIT_TREE_CONTROL_SCHEMA_VERSION,
                    "tree_id": self.tree_id,
                    "records": {},
                }
            )
            self.worktree_dir.mkdir(parents=True, exist_ok=True)
            return root_commit

    def state_status(self) -> str:
        """Return ``missing`` or ``complete``; reject a partial/tampered cache."""

        present = {
            "repository": self.repo_dir.exists(),
            "metadata": self.metadata_path.exists(),
            "control": self.control_path.exists(),
        }
        if not any(present.values()):
            return "missing"
        if not all(present.values()):
            raise GitTreeRepositoryError(
                "tree workspace is partial: "
                + ", ".join(
                    name for name, exists in present.items() if not exists
                )
            )
        with self._locked():
            metadata = self._load_metadata()
            control = self._load_control()
            self._verify_repository_state(metadata=metadata, control=control)
        return "complete"

    def export_recovery_state(
        self,
        *,
        checkpoint_hash: str,
        bundle_uri: str,
        bundle_hash: str,
        bundle_size_bytes: int,
    ) -> dict[str, Any]:
        """Export private control state bound to one verified Git bundle."""

        if not re.fullmatch(r"sha256:[0-9a-f]{64}", str(checkpoint_hash or "")):
            raise GitTreeRepositoryError("tree recovery checkpoint hash is invalid")
        if not str(bundle_uri or "").startswith("s3://"):
            raise GitTreeRepositoryError("tree recovery bundle URI is invalid")
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", str(bundle_hash or "")):
            raise GitTreeRepositoryError("tree recovery bundle hash is invalid")
        if int(bundle_size_bytes) <= 0:
            raise GitTreeRepositoryError("tree recovery bundle is empty")
        with self._locked():
            metadata = self._load_metadata()
            control = self._load_control()
            checkpoint_key = f"checkpoint:{checkpoint_hash}"
            if checkpoint_key not in dict(control.get("records") or {}):
                raise GitTreeRepositoryError(
                    "tree recovery checkpoint is not committed locally"
                )
            return {
                "schema_version": GIT_TREE_RECOVERY_SCHEMA_VERSION,
                "tree_id": self.tree_id,
                "checkpoint_hash": str(checkpoint_hash),
                "git_bundle": {
                    "uri": str(bundle_uri),
                    "content_hash": str(bundle_hash),
                    "size_bytes": int(bundle_size_bytes),
                },
                "tree_metadata": metadata,
                "control_state": control,
            }

    def restore_recovery_state(
        self,
        *,
        recovery_state: Mapping[str, Any],
        bundle_path: Path,
    ) -> str:
        """Atomically restore a missing workspace from committed private state."""

        document = dict(recovery_state)
        required = {
            "schema_version",
            "tree_id",
            "checkpoint_hash",
            "git_bundle",
            "tree_metadata",
            "control_state",
        }
        if set(document) != required:
            raise GitTreeRepositoryError("tree recovery state fields are invalid")
        if (
            document.get("schema_version") != GIT_TREE_RECOVERY_SCHEMA_VERSION
            or document.get("tree_id") != self.tree_id
        ):
            raise GitTreeRepositoryError("tree recovery state identity differs")
        checkpoint_hash = str(document.get("checkpoint_hash") or "")
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", checkpoint_hash):
            raise GitTreeRepositoryError("tree recovery checkpoint hash is invalid")
        bundle = document.get("git_bundle")
        if not isinstance(bundle, Mapping) or set(bundle) != {
            "uri",
            "content_hash",
            "size_bytes",
        }:
            raise GitTreeRepositoryError("tree recovery bundle fields are invalid")
        bundle_path = Path(bundle_path).resolve()
        if not bundle_path.is_file():
            raise GitTreeRepositoryError("tree recovery bundle is missing")
        bundle_bytes = bundle_path.read_bytes()
        if (
            sha256_bytes(bundle_bytes) != bundle.get("content_hash")
            or len(bundle_bytes) != int(bundle.get("size_bytes") or 0)
        ):
            raise GitTreeRepositoryError("tree recovery bundle commitment differs")
        metadata = self._validate_metadata_document(document.get("tree_metadata"))
        control = self._validate_control_document(document.get("control_state"))
        if f"checkpoint:{checkpoint_hash}" not in control["records"]:
            raise GitTreeRepositoryError(
                "tree recovery control state lacks its checkpoint"
            )

        with self._locked():
            present = {
                "repository": self.repo_dir.exists(),
                "metadata": self.metadata_path.exists(),
                "control": self.control_path.exists(),
            }
            if any(present.values()):
                if not all(present.values()):
                    raise GitTreeRepositoryError(
                        "tree recovery refuses partial existing local state"
                    )
                local_metadata = self._load_metadata()
                local_control = self._load_control()
                if local_metadata != metadata or local_control != control:
                    raise GitTreeRepositoryError(
                        "tree recovery existing local state differs"
                    )
                self._verify_repository_state(
                    metadata=local_metadata,
                    control=local_control,
                )
                return str(local_metadata.get("root_git_commit") or "")
            self.workspace.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(
                prefix=self.workspace.name + ".restore-",
                dir=str(self.workspace.parent),
            ) as temporary:
                staged = Path(temporary)
                staged_repo = staged / "repository"
                _run_git(
                    ["clone", "--quiet", "--no-checkout", str(bundle_path), str(staged_repo)],
                    cwd=staged,
                    timeout_seconds=300,
                )
                object_format = _run_git(
                    ["rev-parse", "--show-object-format"], cwd=staged_repo
                ).strip()
                if object_format != "sha256":
                    raise GitTreeRepositoryError(
                        "tree recovery bundle is not a SHA-256 repository"
                    )
                heads = _run_git(
                    ["bundle", "list-heads", str(bundle_path)], cwd=staged_repo
                ).splitlines()
                for line in heads:
                    parts = line.strip().split()
                    if len(parts) != 2 or not parts[1].startswith("refs/heads/"):
                        continue
                    self._verify_commit(parts[0])
                    _run_git(
                        ["update-ref", parts[1], parts[0]], cwd=staged_repo
                    )
                root_commit = str(metadata.get("root_git_commit") or "")
                self._verify_commit(root_commit)
                _run_git(["reset", "--hard", root_commit], cwd=staged_repo)
                _run_git(
                    ["config", "user.name", "Leadpoet Research Lab"],
                    cwd=staged_repo,
                )
                _run_git(
                    ["config", "user.email", "research-lab@leadpoet.local"],
                    cwd=staged_repo,
                )
                staged_metadata = staged / "TREE.json"
                staged_control = staged / "CONTROL.json"
                self._atomic_json_write(staged_metadata, metadata)
                self._atomic_json_write(staged_control, control)
                os.replace(staged_repo, self.repo_dir)
                os.replace(staged_metadata, self.metadata_path)
                os.replace(staged_control, self.control_path)
            self.worktree_dir.mkdir(parents=True, exist_ok=True)
            restored_metadata = self._load_metadata()
            restored_control = self._load_control()
            restored_root = str(restored_metadata.get("root_git_commit") or "")
            self._verify_repository_state(
                metadata=restored_metadata,
                control=restored_control,
            )
            return restored_root

    def plan_slot(
        self,
        *,
        slot: TreeChildSlot,
        request_hash: str,
        operation_id: str,
        node_doc: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        payload = {
            "tree_id": self.tree_id,
            "node_id": slot.node_id,
            "operation_kind": "generation",
            "slot": slot.to_dict(),
            "request_hash": str(request_hash),
            "operation_id": str(operation_id),
            "node_doc": dict(node_doc),
            "status": "reserved",
        }
        return self._commit_control_record(
            record_kind="operation",
            record_id=str(operation_id),
            payload=payload,
        )

    def inspect_operation(self, *, operation_id: str) -> Mapping[str, Any]:
        """Return one committed local operation state without changing it."""

        with self._locked():
            control = self._load_control()
            record = dict(control.get("records") or {}).get(
                f"operation:{operation_id}"
            )
            if record is None:
                return {"exists": False, "operation_id": str(operation_id)}
            if not isinstance(record, Mapping) or not isinstance(
                record.get("payload"), Mapping
            ):
                raise GitTreeRepositoryError("tree operation record is invalid")
            payload = dict(record["payload"])
            if str(payload.get("operation_id") or "") != str(operation_id):
                raise GitTreeRepositoryError("tree operation identity differs")
            return {
                "exists": True,
                "operation_id": str(operation_id),
                "operation": payload,
            }

    def operation_settlement_commitment(self) -> Mapping[str, Any]:
        """Commit every terminal logical operation without exposing payloads."""

        with self._locked():
            control = self._load_control()
            operations: list[dict[str, Any]] = []
            for key, raw_record in sorted(
                dict(control.get("records") or {}).items()
            ):
                if not key.startswith("operation:"):
                    continue
                if not isinstance(raw_record, Mapping) or not isinstance(
                    raw_record.get("payload"), Mapping
                ):
                    raise GitTreeRepositoryError(
                        "tree operation record is invalid"
                    )
                operations.append(dict(raw_record["payload"]))
            return operation_settlement_commitment(
                tree_id=self.tree_id,
                operations=operations,
            )

    def reconcile_operations(
        self, operations: Sequence[Mapping[str, Any]]
    ) -> int:
        """Rehydrate or advance local operation state from authoritative DB rows."""

        with self._locked():
            control = self._load_control()
            records = dict(control.get("records") or {})
            created = 0
            for raw in operations:
                row = dict(raw)
                operation_id = str(row.get("logical_operation_id") or "")
                operation_kind = str(row.get("operation_kind") or "")
                status = str(row.get("operation_status") or "")
                node_id = str(row.get("node_id") or "")
                request_hash = str(row.get("request_hash") or "")
                result_hash = str(row.get("result_hash") or "")
                settlement_doc = dict(row.get("settlement_doc") or {})
                if (
                    not _TREE_ID_RE.fullmatch(operation_id)
                    or row.get("tree_id") != self.tree_id
                    or operation_kind
                    not in {
                        "generation",
                        "build",
                        "provider",
                        "evaluation",
                        "artifact",
                        "checkpoint",
                    }
                    or status
                    not in {"reserved", "succeeded", "failed", "indeterminate"}
                    or not _TREE_ID_RE.fullmatch(request_hash)
                    or (node_id and not _NODE_ID_RE.fullmatch(node_id))
                    or (result_hash and not _TREE_ID_RE.fullmatch(result_hash))
                ):
                    raise GitTreeRepositoryError(
                        "persisted tree operation is invalid"
                    )
                payload = {
                    "tree_id": self.tree_id,
                    "node_id": node_id,
                    "operation_kind": operation_kind,
                    "request_hash": request_hash,
                    "operation_id": operation_id,
                    "status": status,
                }
                if status == "reserved":
                    payload["reservation_doc"] = settlement_doc
                else:
                    payload.update(
                        {
                            "result_hash": result_hash,
                            "settled_cost_microusd": max(
                                0, int(row.get("settled_cost_microusd") or 0)
                            ),
                            "provider_call_count": max(
                                0, int(row.get("provider_call_count") or 0)
                            ),
                            "settlement_doc": settlement_doc,
                        }
                    )
                key = f"operation:{operation_id}"
                existing = records.get(key)
                if existing is not None:
                    if not isinstance(existing, Mapping) or not isinstance(
                        existing.get("payload"), Mapping
                    ):
                        raise GitTreeRepositoryError(
                            "local tree operation record is invalid"
                        )
                    existing_payload = dict(existing["payload"])
                    identity_fields = (
                        "tree_id",
                        "node_id",
                        "operation_kind",
                        "request_hash",
                        "operation_id",
                    )
                    if any(
                        str(existing_payload.get(field) or "")
                        != str(payload.get(field) or "")
                        for field in identity_fields
                    ):
                        raise GitTreeRepositoryError(
                            "local tree operation differs from persistence"
                        )
                    if self._operation_comparison(existing_payload) == (
                        self._operation_comparison(payload)
                    ):
                        continue
                    if (
                        existing_payload.get("status") == "reserved"
                        and status in {"succeeded", "failed", "indeterminate"}
                    ):
                        advanced_payload = {**existing_payload, **payload}
                        records[key] = {
                            "payload": advanced_payload,
                            "record_hash": sha256_json(advanced_payload),
                        }
                        created += 1
                        continue
                    raise GitTreeRepositoryError(
                        "local tree operation differs from persistence"
                    )
                records[key] = {
                    "payload": payload,
                    "record_hash": sha256_json(payload),
                }
                created += 1
            if created:
                control["records"] = records
                self._write_control(control)
            return created

    def settle_operation(
        self,
        *,
        operation_id: str,
        operation_status: str,
        request_hash: str,
        result_hash: str,
        settled_cost_microusd: int,
        provider_call_count: int,
        settlement_doc: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        status = str(operation_status or "")
        if status not in {"succeeded", "failed", "indeterminate"}:
            raise GitTreeRepositoryError("tree operation terminal status is invalid")
        with self._locked():
            control = self._load_control()
            records = dict(control.get("records") or {})
            key = f"operation:{operation_id}"
            prior = records.get(key)
            if not isinstance(prior, Mapping):
                raise GitTreeRepositoryError("tree operation was not reserved")
            prior_payload = dict(prior.get("payload") or {})
            payload = {
                **prior_payload,
                "operation_id": str(operation_id),
                "request_hash": str(request_hash),
                "status": status,
                "result_hash": str(result_hash),
                "settled_cost_microusd": max(0, int(settled_cost_microusd)),
                "provider_call_count": max(0, int(provider_call_count)),
                "settlement_doc": dict(settlement_doc),
            }
            if prior_payload.get("status") == status:
                if (
                    str(prior_payload.get("operation_id") or "")
                    == str(operation_id)
                    and self._operation_comparison(prior_payload)
                    == self._operation_comparison(payload)
                ):
                    return {"created": False, "record": dict(prior_payload)}
                raise GitTreeRepositoryError(
                    "tree operation terminal evidence differs"
                )
            if (
                prior_payload.get("operation_id") != operation_id
                or prior_payload.get("request_hash") != request_hash
                or prior_payload.get("status") != "reserved"
            ):
                raise GitTreeRepositoryError("tree operation state differs")
            records[key] = {"payload": payload, "record_hash": sha256_json(payload)}
            control["records"] = records
            self._write_control(control)
            return {"created": True, "record": payload}

    def reserve_operation(
        self,
        *,
        operation_id: str,
        operation_kind: str,
        request_hash: str,
        node_id: str = "",
        reservation_doc: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        kind = str(operation_kind or "").strip().lower()
        if kind not in {"build", "provider", "evaluation", "artifact", "checkpoint"}:
            raise GitTreeRepositoryError("tree operation reservation kind is invalid")
        payload = {
            "tree_id": self.tree_id,
            "node_id": str(node_id or ""),
            "operation_kind": kind,
            "request_hash": str(request_hash),
            "operation_id": str(operation_id),
            "reservation_doc": dict(reservation_doc or {}),
            "status": "reserved",
        }
        return self._commit_control_record(
            record_kind="operation",
            record_id=str(operation_id),
            payload=payload,
        )

    def record_node(self, *, node_doc: Mapping[str, Any]) -> Mapping[str, Any]:
        node_id = str(node_doc.get("node_id") or "")
        if not _NODE_ID_RE.fullmatch(node_id):
            raise GitTreeRepositoryError("tree node projection identity is invalid")
        node_hash = sha256_json(dict(node_doc))
        return self._commit_control_record(
            record_kind="node",
            record_id=f"{node_id}:{node_hash}",
            payload=dict(node_doc),
        )

    def commit_checkpoint(
        self, *, checkpoint_hash: str, checkpoint_doc: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        return self._commit_control_record(
            record_kind="checkpoint",
            record_id=str(checkpoint_hash),
            payload=dict(checkpoint_doc),
        )

    def select_final(
        self, *, selection_hash: str, selection_doc: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        with self._locked():
            control = self._load_control()
            records = dict(control.get("records") or {})
            if any(key.startswith("tree_terminal:") for key in records):
                raise GitTreeRepositoryError(
                    "failed tree cannot select a finalist"
                )
            prior = [
                value
                for key, value in records.items()
                if key.startswith("final_selection:")
            ]
            if prior:
                prior_payload = dict(prior[0].get("payload") or {})
                if sha256_json(prior_payload) != str(selection_hash):
                    raise GitTreeRepositoryError("tree finalist changed")
                return {"created": False, "record": prior_payload}
            payload = dict(selection_doc)
            if sha256_json(payload) != str(selection_hash):
                raise GitTreeRepositoryError("tree selection hash differs")
            records[f"final_selection:{selection_hash}"] = {
                "payload": payload,
                "record_hash": str(selection_hash),
            }
            control["records"] = records
            self._write_control(control)
            return {"created": True, "record": payload}

    def fail_tree(
        self, *, failure_hash: str, failure_doc: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        """Commit one immutable no-finalist terminal record."""

        payload = dict(failure_doc)
        if sha256_json(payload) != str(failure_hash):
            raise GitTreeRepositoryError("tree failure hash differs")
        with self._locked():
            control = self._load_control()
            records = dict(control.get("records") or {})
            prior = [
                value
                for key, value in records.items()
                if key.startswith("tree_terminal:")
            ]
            if prior:
                prior_payload = dict(prior[0].get("payload") or {})
                if sha256_json(prior_payload) != str(failure_hash):
                    raise GitTreeRepositoryError("tree terminal state changed")
                return {"created": False, "record": prior_payload}
            if any(key.startswith("final_selection:") for key in records):
                raise GitTreeRepositoryError(
                    "selected tree cannot become a no-finalist failure"
                )
            records[f"tree_terminal:{failure_hash}"] = {
                "payload": payload,
                "record_hash": str(failure_hash),
            }
            control["records"] = records
            self._write_control(control)
            return {"created": True, "record": payload}

    def commit_child(
        self,
        *,
        slot: TreeChildSlot,
        draft: CodeEditDraft,
        expected_parent_source_tree_hash: str,
    ) -> GitTreeCommit:
        if slot.tree_id != self.tree_id:
            raise GitTreeRepositoryError("child slot belongs to another tree")
        with self._locked():
            metadata = self._load_metadata()
            root_commit = str(metadata["root_git_commit"])
            parent_commit = (
                root_commit
                if slot.parent_node_id == "root"
                else self.resolve_node_commit(slot.parent_node_id)
            )
            node_ref = _safe_ref(slot.node_id)
            existing = _run_git(
                ["show-ref", "--hash", node_ref],
                cwd=self.repo_dir,
                check=False,
            ).strip()
            if existing:
                return self._reconcile_existing_child(
                    slot=slot,
                    draft=draft,
                    root_commit=root_commit,
                    parent_commit=parent_commit,
                    existing_commit=existing,
                    expected_parent_source_tree_hash=expected_parent_source_tree_hash,
                )

            self.worktree_dir.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(
                prefix=slot.node_id.split(":", 1)[1][:12] + "-",
                dir=self.worktree_dir,
            ) as tmp:
                checkout = Path(tmp) / "source"
                _run_git(
                    ["worktree", "add", "--quiet", "--detach", str(checkout), parent_commit],
                    cwd=self.repo_dir,
                )
                try:
                    parent_source_tree_hash = compute_private_source_tree_hash(checkout)
                    if parent_source_tree_hash != expected_parent_source_tree_hash:
                        raise GitTreeRepositoryError(
                            "tree parent source commitment differs before patch"
                        )
                    diff_path = Path(tmp) / "incremental.diff"
                    diff_path.write_text(draft.unified_diff, encoding="utf-8")
                    _run_git(["apply", "--check", "--recount", str(diff_path)], cwd=checkout)
                    _run_git(["apply", "--recount", str(diff_path)], cwd=checkout)
                    if not _run_git(
                        ["status", "--porcelain=v1"], cwd=checkout
                    ).strip():
                        raise GitTreeRepositoryError("tree child patch produced no changes")
                    _run_git(["add", "-A"], cwd=checkout)
                    _run_git(
                        [
                            "commit",
                            "--quiet",
                            "-m",
                            f"Research Lab tree node {slot.node_id}",
                        ],
                        cwd=checkout,
                        env=self._commit_env(slot.depth * 256 + slot.slot_index + 1),
                    )
                    commit = _run_git(["rev-parse", "HEAD"], cwd=checkout).strip()
                    self._verify_commit(commit)
                    parent_line = _run_git(
                        ["rev-list", "--parents", "-n", "1", commit], cwd=checkout
                    ).strip().split()
                    if len(parent_line) != 2 or parent_line[1] != parent_commit:
                        raise GitTreeRepositoryError(
                            "tree child commit is not directly parented"
                        )
                    changed_files = tuple(
                        sorted(
                            line.strip()
                            for line in _run_git(
                                ["diff", "--name-only", parent_commit, commit],
                                cwd=checkout,
                            ).splitlines()
                            if line.strip()
                        )
                    )
                    if not changed_files:
                        raise GitTreeRepositoryError(
                            "tree child commit produced no changed files"
                        )
                    _run_git(["update-ref", node_ref, commit], cwd=self.repo_dir)
                    source_tree_hash = compute_private_source_tree_hash(checkout)
                    incremental_patch = _run_git(
                        ["diff", "--binary", "--full-index", parent_commit, commit],
                        cwd=checkout,
                    )
                    cumulative_patch = _run_git(
                        ["diff", "--binary", "--full-index", root_commit, commit],
                        cwd=checkout,
                    )
                finally:
                    _run_git(
                        ["worktree", "remove", "--force", str(checkout)],
                        cwd=self.repo_dir,
                        check=False,
                    )
                    _run_git(["worktree", "prune"], cwd=self.repo_dir, check=False)

            if not incremental_patch.startswith("diff --git "):
                raise GitTreeRepositoryError("incremental Git patch is invalid")
            if not cumulative_patch.startswith("diff --git "):
                raise GitTreeRepositoryError("cumulative Git patch is invalid")
            normalized_incremental = incremental_patch.rstrip() + "\n"
            normalized_cumulative = cumulative_patch.rstrip() + "\n"
            return GitTreeCommit(
                tree_id=self.tree_id,
                node_id=slot.node_id,
                parent_node_id=slot.parent_node_id,
                root_branch_id=slot.root_branch_id,
                depth=slot.depth,
                slot_index=slot.slot_index,
                git_commit=commit,
                parent_git_commit=parent_commit,
                source_tree_hash=source_tree_hash,
                draft_patch_hash=sha256_json(
                    {"unified_diff": draft.unified_diff}
                ),
                incremental_patch=normalized_incremental,
                incremental_patch_hash=sha256_json(
                    {"unified_diff": normalized_incremental}
                ),
                cumulative_patch=normalized_cumulative,
                cumulative_patch_hash=sha256_json(
                    {"unified_diff": normalized_cumulative}
                ),
                changed_files=changed_files,
            )

    def _reconcile_existing_child(
        self,
        *,
        slot: TreeChildSlot,
        draft: CodeEditDraft,
        root_commit: str,
        parent_commit: str,
        existing_commit: str,
        expected_parent_source_tree_hash: str,
    ) -> GitTreeCommit:
        """Recover the commit-before-checkpoint crash window idempotently."""

        self._verify_commit(existing_commit)
        parents = _run_git(
            ["rev-list", "--parents", "-n", "1", existing_commit],
            cwd=self.repo_dir,
        ).strip().split()
        if len(parents) != 2 or parents[1] != parent_commit:
            raise GitTreeRepositoryError(
                "existing tree child has a different direct parent"
            )
        self.worktree_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="reconcile-" + slot.node_id.split(":", 1)[1][:12] + "-",
            dir=self.worktree_dir,
        ) as tmp:
            parent_checkout = Path(tmp) / "parent"
            actual_checkout = Path(tmp) / "actual"
            _run_git(
                ["worktree", "add", "--quiet", "--detach", str(parent_checkout), parent_commit],
                cwd=self.repo_dir,
            )
            _run_git(
                [
                    "worktree",
                    "add",
                    "--quiet",
                    "--detach",
                    str(actual_checkout),
                    existing_commit,
                ],
                cwd=self.repo_dir,
            )
            try:
                parent_source_tree_hash = compute_private_source_tree_hash(
                    parent_checkout
                )
                if parent_source_tree_hash != expected_parent_source_tree_hash:
                    raise GitTreeRepositoryError(
                        "existing tree parent source commitment differs"
                    )
                diff_path = Path(tmp) / "incremental.diff"
                diff_path.write_text(draft.unified_diff, encoding="utf-8")
                _run_git(
                    ["apply", "--check", "--recount", str(diff_path)],
                    cwd=parent_checkout,
                )
                _run_git(
                    ["apply", "--recount", str(diff_path)], cwd=parent_checkout
                )
                expected_source_tree_hash = compute_private_source_tree_hash(
                    parent_checkout
                )
                actual_source_tree_hash = compute_private_source_tree_hash(
                    actual_checkout
                )
                if expected_source_tree_hash != actual_source_tree_hash:
                    raise GitTreeRepositoryError(
                        "existing tree child source differs from planned draft"
                    )
            finally:
                _run_git(
                    ["worktree", "remove", "--force", str(parent_checkout)],
                    cwd=self.repo_dir,
                    check=False,
                )
                _run_git(
                    ["worktree", "remove", "--force", str(actual_checkout)],
                    cwd=self.repo_dir,
                    check=False,
                )
                _run_git(["worktree", "prune"], cwd=self.repo_dir, check=False)

        incremental_patch = _run_git(
            ["diff", "--binary", "--full-index", parent_commit, existing_commit],
            cwd=self.repo_dir,
        ).rstrip() + "\n"
        cumulative_patch = _run_git(
            ["diff", "--binary", "--full-index", root_commit, existing_commit],
            cwd=self.repo_dir,
        ).rstrip() + "\n"
        changed_files = tuple(
            sorted(
                line.strip()
                for line in _run_git(
                    ["diff", "--name-only", parent_commit, existing_commit],
                    cwd=self.repo_dir,
                ).splitlines()
                if line.strip()
            )
        )
        return GitTreeCommit(
            tree_id=self.tree_id,
            node_id=slot.node_id,
            parent_node_id=slot.parent_node_id,
            root_branch_id=slot.root_branch_id,
            depth=slot.depth,
            slot_index=slot.slot_index,
            git_commit=existing_commit,
            parent_git_commit=parent_commit,
            source_tree_hash=actual_source_tree_hash,
            draft_patch_hash=sha256_json({"unified_diff": draft.unified_diff}),
            incremental_patch=incremental_patch,
            incremental_patch_hash=sha256_json(
                {"unified_diff": incremental_patch}
            ),
            cumulative_patch=cumulative_patch,
            cumulative_patch_hash=sha256_json(
                {"unified_diff": cumulative_patch}
            ),
            changed_files=changed_files,
        )

    def resolve_node_commit(self, node_id: str) -> str:
        commit = _run_git(
            ["show-ref", "--hash", _safe_ref(node_id)], cwd=self.repo_dir
        ).strip()
        self._verify_commit(commit)
        return commit

    def materialize_node(self, *, node_id: str, destination: Path) -> str:
        with self._locked():
            commit = self.resolve_node_commit(node_id)
            destination = Path(destination).resolve()
            if destination.exists():
                shutil.rmtree(destination)
            destination.parent.mkdir(parents=True, exist_ok=True)
            _run_git(
                ["worktree", "add", "--quiet", "--detach", str(destination), commit],
                cwd=self.repo_dir,
            )
            return compute_private_source_tree_hash(destination)

    def release_materialized_node(self, destination: Path) -> None:
        with self._locked():
            _run_git(
                ["worktree", "remove", "--force", str(Path(destination).resolve())],
                cwd=self.repo_dir,
                check=False,
            )
            _run_git(["worktree", "prune"], cwd=self.repo_dir, check=False)

    def create_bundle(self, destination: Path) -> dict[str, Any]:
        with self._locked():
            destination = Path(destination).resolve()
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(destination.name + ".tmp")
            if temporary.exists():
                temporary.unlink()
            _run_git(
                ["bundle", "create", str(temporary), "--all"], cwd=self.repo_dir
            )
            _run_git(
                ["bundle", "verify", str(temporary)], cwd=self.repo_dir
            )
            bundle_hash = sha256_bytes(temporary.read_bytes())
            os.replace(temporary, destination)
            return {
                "tree_id": self.tree_id,
                "bundle_path": str(destination),
                "bundle_hash": bundle_hash,
                "bundle_size_bytes": destination.stat().st_size,
            }

    def verify_node(self, *, commit: GitTreeCommit) -> None:
        with self._locked():
            actual = self.resolve_node_commit(commit.node_id)
            if actual != commit.git_commit:
                raise GitTreeRepositoryError("tree node ref differs from commitment")
            expected_parent = (
                str(self._load_metadata()["root_git_commit"])
                if commit.parent_node_id == "root"
                else self.resolve_node_commit(commit.parent_node_id)
            )
            parents = _run_git(
                ["rev-list", "--parents", "-n", "1", actual], cwd=self.repo_dir
            ).strip().split()
            if len(parents) != 2 or parents[1] != expected_parent:
                raise GitTreeRepositoryError("tree direct-parent verification failed")

    def verify_node_identity(
        self,
        *,
        node_id: str,
        git_commit: str,
        parent_node_id: str,
    ) -> None:
        """Verify a checkpoint without requiring private patch bytes in memory."""

        with self._locked():
            actual = self.resolve_node_commit(node_id)
            if actual != str(git_commit or ""):
                raise GitTreeRepositoryError(
                    "tree checkpoint node ref differs from Git"
                )
            expected_parent = (
                str(self._load_metadata()["root_git_commit"])
                if parent_node_id == "root"
                else self.resolve_node_commit(parent_node_id)
            )
            parents = _run_git(
                ["rev-list", "--parents", "-n", "1", actual], cwd=self.repo_dir
            ).strip().split()
            if len(parents) != 2 or parents[1] != expected_parent:
                raise GitTreeRepositoryError(
                    "tree checkpoint direct-parent verification failed"
                )

    def _load_metadata(self) -> dict[str, Any]:
        if not self.metadata_path.is_file():
            raise GitTreeRepositoryError("tree metadata is missing")
        value = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        return self._validate_metadata_document(value)

    def _validate_metadata_document(self, value: Any) -> dict[str, Any]:
        if not isinstance(value, Mapping):
            raise GitTreeRepositoryError("tree metadata is invalid")
        value = dict(value)
        metadata_hash = str(value.get("metadata_hash") or "")
        body = {key: item for key, item in value.items() if key != "metadata_hash"}
        if (
            value.get("schema_version") != GIT_TREE_METADATA_SCHEMA_VERSION
            or value.get("tree_id") != self.tree_id
            or sha256_json(body) != metadata_hash
        ):
            raise GitTreeRepositoryError("tree metadata commitment differs")
        return value

    def _load_control(self) -> dict[str, Any]:
        if not self.control_path.is_file():
            raise GitTreeRepositoryError("tree control state is missing")
        value = json.loads(self.control_path.read_text(encoding="utf-8"))
        return self._validate_control_document(value)

    def _validate_control_document(self, value: Any) -> dict[str, Any]:
        if (
            not isinstance(value, Mapping)
            or value.get("schema_version") != GIT_TREE_CONTROL_SCHEMA_VERSION
            or value.get("tree_id") != self.tree_id
            or not isinstance(value.get("records"), dict)
        ):
            raise GitTreeRepositoryError("tree control state is invalid")
        value = dict(value)
        control_hash = str(value.get("control_hash") or "")
        body = {key: item for key, item in value.items() if key != "control_hash"}
        if sha256_json(body) != control_hash:
            raise GitTreeRepositoryError("tree control commitment differs")
        for record in value["records"].values():
            if (
                not isinstance(record, Mapping)
                or not isinstance(record.get("payload"), Mapping)
                or record.get("record_hash")
                != sha256_json(dict(record["payload"]))
            ):
                raise GitTreeRepositoryError("tree control record is invalid")
        return value

    def _verify_repository_state(
        self,
        *,
        metadata: Mapping[str, Any],
        control: Mapping[str, Any],
    ) -> None:
        """Verify all committed Git state before reuse after a restart."""

        object_format = _run_git(
            ["rev-parse", "--show-object-format"], cwd=self.repo_dir
        ).strip()
        if object_format != "sha256":
            raise GitTreeRepositoryError(
                "restored Git tree repository is not SHA-256"
            )
        root_commit = str(metadata.get("root_git_commit") or "")
        self._verify_commit(root_commit)
        _run_git(["cat-file", "-e", root_commit + "^{commit}"], cwd=self.repo_dir)
        if _run_git(["rev-parse", "HEAD"], cwd=self.repo_dir).strip() != root_commit:
            raise GitTreeRepositoryError("tree repository HEAD differs from its root")
        if _run_git(["status", "--porcelain"], cwd=self.repo_dir).strip():
            raise GitTreeRepositoryError("tree repository root checkout is modified")
        if (
            compute_private_source_tree_hash(self.repo_dir)
            != metadata.get("root_source_tree_hash")
        ):
            raise GitTreeRepositoryError("tree repository root source differs")
        _run_git(["fsck", "--strict", "--no-dangling"], cwd=self.repo_dir)

        records = control.get("records")
        if not isinstance(records, Mapping):
            raise GitTreeRepositoryError("tree control records are invalid")
        for key, record in records.items():
            if not str(key).startswith("node:") or not isinstance(record, Mapping):
                continue
            payload = record.get("payload")
            if not isinstance(payload, Mapping):
                raise GitTreeRepositoryError("tree node control record is invalid")
            node_id = str(payload.get("node_id") or "")
            git_commit = str(payload.get("git_commit") or "")
            if not git_commit:
                continue
            if not _NODE_ID_RE.fullmatch(node_id):
                raise GitTreeRepositoryError("tree node control identity is invalid")
            self._verify_commit(git_commit)
            actual_commit = _run_git(
                ["show-ref", "--hash", _safe_ref(node_id)], cwd=self.repo_dir
            ).strip()
            if actual_commit != git_commit:
                raise GitTreeRepositoryError(
                    "tree node control commitment differs from Git"
                )

    def _commit_control_record(
        self,
        *,
        record_kind: str,
        record_id: str,
        payload: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        with self._locked():
            control = self._load_control()
            records = dict(control.get("records") or {})
            key = f"{record_kind}:{record_id}"
            normalized = dict(payload)
            record_hash = sha256_json(normalized)
            existing = records.get(key)
            if existing is not None:
                if (
                    not isinstance(existing, Mapping)
                    or existing.get("record_hash") != record_hash
                ):
                    raise GitTreeRepositoryError(
                        f"tree {record_kind} commitment changed"
                    )
                return {"created": False, "record": normalized}
            records[key] = {"payload": normalized, "record_hash": record_hash}
            control["records"] = records
            self._write_control(control)
            return {"created": True, "record": normalized}

    def _write_control(self, value: Mapping[str, Any]) -> None:
        body = {
            key: item
            for key, item in dict(value).items()
            if key != "control_hash"
        }
        if (
            body.get("schema_version") != GIT_TREE_CONTROL_SCHEMA_VERSION
            or body.get("tree_id") != self.tree_id
            or not isinstance(body.get("records"), dict)
        ):
            raise GitTreeRepositoryError("tree control write is invalid")
        self._atomic_json_write(
            self.control_path,
            {**body, "control_hash": sha256_json(body)},
        )

    @staticmethod
    def _operation_comparison(value: Mapping[str, Any]) -> dict[str, Any]:
        status = str(value.get("status") or "")
        return {
            "tree_id": str(value.get("tree_id") or ""),
            "node_id": str(value.get("node_id") or ""),
            "operation_kind": str(value.get("operation_kind") or ""),
            "request_hash": str(value.get("request_hash") or ""),
            "status": status,
            "result_hash": str(value.get("result_hash") or ""),
            "settled_cost_microusd": int(
                value.get("settled_cost_microusd") or 0
            ),
            "provider_call_count": int(value.get("provider_call_count") or 0),
            "settlement_doc": dict(
                value.get("settlement_doc")
                or (value.get("reservation_doc") if status == "reserved" else {})
                or {}
            ),
        }

    @staticmethod
    def _verify_commit(commit: str) -> None:
        if not _GIT_SHA256_RE.fullmatch(str(commit or "")):
            raise GitTreeRepositoryError("Git commit is not a SHA-256 object")

    @staticmethod
    def _commit_env(offset_seconds: int) -> dict[str, str]:
        timestamp = 946684800 + max(0, int(offset_seconds))
        value = f"{timestamp} +0000"
        return {
            "GIT_AUTHOR_DATE": value,
            "GIT_COMMITTER_DATE": value,
        }

    @staticmethod
    def _atomic_json_write(path: Path, value: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(path.parent),
            prefix=path.name + ".",
            suffix=".tmp",
            delete=False,
        ) as handle:
            json.dump(dict(value), handle, sort_keys=True, separators=(",", ":"))
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.replace(temporary, path)
