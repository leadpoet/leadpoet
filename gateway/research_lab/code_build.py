"""Gateway-side builder for code-edit Research Lab candidate images."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

from gateway.research_lab.config import ResearchLabGatewayConfig
from gateway.research_lab.source_symbol_index import (
    build_source_symbol_index,
    compact_file_inventory,
)
from gateway.research_lab.store import canonical_hash
from research_lab.canonical import sha256_json
from research_lab.code_editing import (
    CodeEditDraft,
    CodeEditSourceInspectionRequest,
    code_edit_candidate_manifest,
    extract_unified_diff_paths,
    validate_code_edit_draft,
)
from research_lab.eval import (
    PrivateModelArtifactManifest,
    compute_private_source_tree_hash,
    validate_private_model_artifact_manifest,
)


logger = logging.getLogger(__name__)


def _env_flag(name: str, default: bool) -> bool:
    """Local env-flag read; gateway config.py is intentionally not touched here."""

    raw = str(os.getenv(name, "") or "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


class CodeEditBuildError(RuntimeError):
    """Raised when a code-edit candidate cannot be built safely."""

    default_failure_stage = "candidate_build_failed"

    def __init__(
        self,
        message: str,
        *,
        failure_stage: str | None = None,
        stderr: str = "",
        stdout: str = "",
        exit_code: int | None = None,
    ):
        super().__init__(message)
        self.failure_stage = failure_stage or self.default_failure_stage
        self.stderr = stderr
        self.stdout = stdout
        self.exit_code = exit_code


class CodeEditPatchApplyError(CodeEditBuildError):
    default_failure_stage = "candidate_patch_apply_failed"


class CodeEditEmptyOrNoopPatchError(CodeEditPatchApplyError):
    default_failure_stage = "candidate_patch_empty_or_noop"


class CodeEditPrivateTestError(CodeEditBuildError):
    default_failure_stage = "candidate_patch_test_failed"


class CodeEditImageBuildError(CodeEditBuildError):
    default_failure_stage = "candidate_image_build_failed"


class CodeEditArtifactMissingError(CodeEditImageBuildError):
    default_failure_stage = "candidate_artifact_missing"


class CodeEditInfraFailureError(CodeEditImageBuildError):
    """Registry/auth/network failure — not the candidate's fault (bug #30).

    Subclasses ``CodeEditImageBuildError`` so every existing handler keeps
    catching it; ``failure_stage`` marks the outcome as infra-failed so
    higher-level callers can requeue instead of charging the candidate.
    """

    default_failure_stage = "candidate_build_infra_failed"
    retryable = True


@dataclass(frozen=True)
class CodeEditBuildResult:
    candidate_model_manifest: PrivateModelArtifactManifest
    code_edit_manifest: dict[str, Any]
    source_diff_hash: str
    build_doc: dict[str, Any]


@dataclass(frozen=True)
class SourceInspectionBatch:
    model_context: dict[str, Any]
    event_doc: dict[str, Any]
    read_paths: tuple[str, ...]
    bytes_returned: int
    # Per-path covered line ranges (source-access v2). Full-file coverage is
    # recorded with the FULL_FILE_RANGE_END sentinel so unranged re-reads skip.
    read_ranges: dict[str, tuple[tuple[int, int], ...]] = field(default_factory=dict)


@dataclass(frozen=True)
class ParentImageSourceContext:
    """Sanitized source inventory extracted from the parent runtime image."""

    source_root: Path
    source_mode: str
    parent_image_digest_hash: str
    source_tree_hash: str
    top_level_paths: tuple[str, ...]
    editable_files: tuple[str, ...]
    file_previews: tuple[dict[str, Any], ...]
    planner_source_index: dict[str, Any] = field(default_factory=dict)

    def prompt_context(self) -> dict[str, Any]:
        return {
            "source_mode": self.source_mode,
            "parent_image_digest_hash": self.parent_image_digest_hash,
            "source_tree_hash": self.source_tree_hash,
            "extracted_top_level_paths": list(self.top_level_paths),
            "editable_file_count": len(self.editable_files),
            "editable_files": list(self.editable_files),
            "file_previews_available": len(self.file_previews),
            "rules": [
                "Only edit files listed in editable_files.",
                "Every edited file must first be read through source inspection.",
                "Do not invent paths or create new files.",
            ],
        }

    def inspection_index(self) -> dict[str, Any]:
        files = compact_file_inventory(self.planner_source_index)
        if not files:
            files = _source_file_inventory(self.source_root, self.editable_files)
        return {
            "source_mode": self.source_mode,
            "parent_image_digest_hash": self.parent_image_digest_hash,
            "source_tree_hash": self.source_tree_hash,
            "extracted_top_level_paths": list(self.top_level_paths),
            "editable_file_count": len(self.editable_files),
            "editable_files": list(self.editable_files),
            "file_inventory": files,
            "rules": [
                "Search and read only files in editable_files.",
                "Do not request dependency, env, credential, Docker, CI, or generated files.",
                "A final patch may edit only files returned by read_file in this iteration.",
            ],
        }

    def planner_index(self) -> dict[str, Any]:
        """The one-time AST projection for the loop-direction planner."""

        if self.planner_source_index:
            return json.loads(json.dumps(self.planner_source_index, default=str))
        return {
            **self.inspection_index(),
            "symbol_index_status": "unavailable",
        }


class CodeEditCandidateBuilder:
    """Build a candidate private model image from a validated unified diff."""

    def __init__(self, config: ResearchLabGatewayConfig):
        self.config = config

    def enabled(self) -> bool:
        return bool(
            self.config.private_test_cmd
            and self.config.private_build_cmd
            and self.config.private_artifact_manifest_output
        )

    def prepare_parent_source_context(
        self,
        *,
        parent_artifact: PrivateModelArtifactManifest,
        workspace_dir: Path,
    ) -> ParentImageSourceContext:
        """Extract the parent image /app once before the first model draft call."""

        image_digest = str(parent_artifact.image_digest or "").strip()
        if not image_digest:
            raise CodeEditBuildError("parent artifact image_digest is required for code-edit source context")
        source_root = workspace_dir / "parent_image_app"
        source_root.mkdir(parents=True, exist_ok=True)
        source_tree_hash, top_level_paths = _extract_parent_image_source(
            image_digest=image_digest,
            source_dir=source_root,
            timeout_seconds=self.config.code_edit_build_timeout_seconds,
        )
        editable_files = _editable_runtime_files(
            source_root,
            allowed_prefixes=self.config.code_edit_allowed_path_prefixes(),
            allowed_exact_paths=self.config.code_edit_allowed_exact_paths(),
            allowed_suffixes=self.config.code_edit_allowed_suffixes(),
        )
        if not editable_files:
            raise CodeEditBuildError("parent image /app has no editable runtime files in the code-edit allowlist")
        parent_image_digest_hash = canonical_hash({"image_digest": image_digest})
        planner_source_index: dict[str, Any] = {}
        if bool(getattr(self.config, "planner_symbol_index_enabled", True)):
            try:
                planner_source_index = build_source_symbol_index(
                    source_root=source_root,
                    editable_files=editable_files,
                    source_tree_hash=source_tree_hash,
                    parent_image_digest_hash=parent_image_digest_hash,
                )
            except Exception as exc:
                logger.warning(
                    "research_lab_source_symbol_index_build_failed source_tree_hash=%s error=%s",
                    source_tree_hash,
                    type(exc).__name__,
                )
        return ParentImageSourceContext(
            source_root=source_root,
            source_mode="parent_image_extract",
            parent_image_digest_hash=parent_image_digest_hash,
            source_tree_hash=source_tree_hash,
            top_level_paths=tuple(top_level_paths),
            editable_files=tuple(editable_files),
            file_previews=tuple(_source_file_previews(source_root, editable_files)),
            planner_source_index=planner_source_index,
        )

    def validate_draft_against_source_context(
        self,
        draft: CodeEditDraft,
        source_context: ParentImageSourceContext,
        *,
        read_paths: Sequence[str] | None = None,
        require_read: bool = False,
    ) -> list[str]:
        allowed = set(source_context.editable_files)
        read = set(read_paths or ())
        paths = set(draft.target_files) | extract_unified_diff_paths(draft.unified_diff)
        errors: list[str] = []
        for path in sorted(paths):
            if path not in allowed:
                errors.append(f"code_edit_path_not_in_extracted_source:{path}")
            if require_read and path not in read:
                errors.append(f"code_edit_unread_source_file:{path}")
        return errors

    def check_patch_applies(
        self,
        *,
        draft: CodeEditDraft,
        parent_artifact: PrivateModelArtifactManifest,
        source_context: ParentImageSourceContext | None = None,
    ) -> None:
        """Validate that a draft patch applies before starting the full build."""

        try:
            validate_code_edit_draft(
                draft,
                allowed_prefixes=self.config.code_edit_allowed_path_prefixes(),
                allowed_exact_paths=self.config.code_edit_allowed_exact_paths(),
                allowed_suffixes=self.config.code_edit_allowed_suffixes(),
            )
        except ValueError as exc:
            raise CodeEditPatchApplyError(str(exc)) from exc
        with tempfile.TemporaryDirectory(prefix="research-lab-code-edit-check-") as tmp:
            tmp_dir = Path(tmp)
            repo_dir = tmp_dir / "repo"
            _prepare_parent_image_workspace(
                image_digest=parent_artifact.image_digest,
                repo_dir=repo_dir,
                timeout_seconds=self.config.code_edit_build_timeout_seconds,
                source_context=source_context,
            )
            if source_context is not None:
                context_errors = self.validate_draft_against_source_context(draft, source_context)
                if context_errors:
                    raise CodeEditPatchApplyError("; ".join(context_errors))
            diff_path = tmp_dir / "candidate.diff"
            diff_path.write_text(draft.unified_diff, encoding="utf-8")
            try:
                _run_git_apply(
                    diff_path,
                    cwd=repo_dir,
                    timeout_seconds=120,
                    check=True,
                )
            except CodeEditBuildError as exc:
                raise CodeEditPatchApplyError(
                    str(exc),
                    stderr=exc.stderr,
                    stdout=exc.stdout,
                    exit_code=exc.exit_code,
                ) from exc

    def build(
        self,
        *,
        draft: CodeEditDraft,
        parent_artifact: PrivateModelArtifactManifest,
        run_id: str,
        candidate_index: int,
        source_context: ParentImageSourceContext | None = None,
    ) -> CodeEditBuildResult:
        if not self.enabled():
            missing = [
                name
                for name, value in {
                    "RESEARCH_LAB_PRIVATE_TEST_CMD": self.config.private_test_cmd,
                    "RESEARCH_LAB_PRIVATE_BUILD_CMD": self.config.private_build_cmd,
                    "RESEARCH_LAB_PRIVATE_ARTIFACT_MANIFEST_OUTPUT": self.config.private_artifact_manifest_output,
                }.items()
                if not str(value or "").strip()
            ]
            raise CodeEditBuildError("code-edit image builder is missing config: " + ", ".join(missing))
        validate_code_edit_draft(
            draft,
            allowed_prefixes=self.config.code_edit_allowed_path_prefixes(),
            allowed_exact_paths=self.config.code_edit_allowed_exact_paths(),
            allowed_suffixes=self.config.code_edit_allowed_suffixes(),
        )
        source_diff_hash = sha256_json({"unified_diff": draft.unified_diff})
        parent_image_digest_hash = canonical_hash({"image_digest": parent_artifact.image_digest})
        repo_ref_hash = canonical_hash(
            {
                "source_mode": "parent_image_extract",
                "parent_manifest_hash": parent_artifact.manifest_hash,
                "parent_image_digest_hash": parent_image_digest_hash,
            }
        )
        with tempfile.TemporaryDirectory(prefix="research-lab-code-edit-") as tmp:
            tmp_dir = Path(tmp)
            repo_dir = tmp_dir / "repo"
            extracted_source_tree_hash_before_patch, extracted_top_level_paths = _prepare_parent_image_workspace(
                image_digest=parent_artifact.image_digest,
                repo_dir=repo_dir,
                timeout_seconds=self.config.code_edit_build_timeout_seconds,
                source_context=source_context,
            )
            if source_context is not None:
                context_errors = self.validate_draft_against_source_context(draft, source_context)
                if context_errors:
                    raise CodeEditPatchApplyError("; ".join(context_errors))
            diff_path = tmp_dir / "candidate.diff"
            draft_path = tmp_dir / "code_edit_draft.json"
            parent_manifest_path = tmp_dir / "parent_manifest.json"
            diff_path.write_text(draft.unified_diff, encoding="utf-8")
            draft_path.write_text(json.dumps(draft.to_dict(), sort_keys=True), encoding="utf-8")
            parent_manifest_path.write_text(json.dumps(parent_artifact.to_dict(), sort_keys=True), encoding="utf-8")
            try:
                _run_git_apply(
                    diff_path,
                    cwd=repo_dir,
                    timeout_seconds=120,
                    check=True,
                )
                _run_git_apply(
                    diff_path,
                    cwd=repo_dir,
                    timeout_seconds=120,
                    check=False,
                )
            except CodeEditBuildError as exc:
                raise CodeEditPatchApplyError(
                    str(exc),
                    stderr=exc.stderr,
                    stdout=exc.stdout,
                    exit_code=exc.exit_code,
                ) from exc
            changed_files = _changed_files(repo_dir)
            if not changed_files:
                raise CodeEditEmptyOrNoopPatchError("code edit produced no repository changes")
            try:
                _py_compile_changed_files(repo_dir, changed_files)
                _run_shell(
                    self.config.private_test_cmd,
                    cwd=repo_dir,
                    env=self._build_env(
                        draft_path=draft_path,
                        parent_manifest_path=parent_manifest_path,
                        diff_path=diff_path,
                        run_id=run_id,
                        candidate_index=candidate_index,
                        include_aws=False,
                    ),
                    timeout_seconds=self.config.code_edit_build_timeout_seconds,
                )
            except CodeEditBuildError as exc:
                raise CodeEditPrivateTestError(
                    str(exc),
                    stderr=exc.stderr,
                    stdout=exc.stdout,
                    exit_code=exc.exit_code,
                ) from exc
            _run(["git", "config", "user.name", "Leadpoet Research Lab"], cwd=repo_dir, timeout_seconds=60)
            _run(["git", "config", "user.email", "research-lab@leadpoet.local"], cwd=repo_dir, timeout_seconds=60)
            _run(["git", "add", "-A"], cwd=repo_dir, timeout_seconds=120)
            _run(
                [
                    "git",
                    "commit",
                    "-m",
                    f"Research Lab candidate {str(run_id).split('-')[0]}:{candidate_index}",
                ],
                cwd=repo_dir,
                timeout_seconds=120,
            )
            git_commit_sha = _run(["git", "rev-parse", "HEAD"], cwd=repo_dir, timeout_seconds=60).strip()
            recorded_commit_sha, recorded_commit_sha_source = _resolve_recorded_commit_sha(
                workspace_sha=git_commit_sha,
                parent_artifact=parent_artifact,
            )
            commit_excluded_paths = _git_ignored_paths(repo_dir)
            if commit_excluded_paths:
                # Bug #29(b): these files exist in the built image (docker COPY)
                # but git ignore rules keep them out of any pushed commit, so a
                # promotion push would silently diverge from the scored image.
                logger.warning(
                    "research-lab code-edit build: %d extracted path(s) excluded from git commits "
                    "by ignore rules; a promotion push will not contain them: %s",
                    len(commit_excluded_paths),
                    ", ".join(commit_excluded_paths[:20]),
                )
            # §5.3: make sure the parent image is cached locally BEFORE the build
            # command starts, so a cold registry pull never eats the build budget
            # (matters especially on the source_context path, which touches no
            # docker image until this point).
            _prepull_parent_image_for_build(parent_artifact.image_digest)
            _run_private_build_cmd_with_infra_retry(
                cmd=self.config.private_build_cmd,
                cwd=repo_dir,
                env={
                    **self._build_env(
                        draft_path=draft_path,
                        parent_manifest_path=parent_manifest_path,
                        diff_path=diff_path,
                        run_id=run_id,
                        candidate_index=candidate_index,
                        include_aws=True,
                    ),
                    "RESEARCH_LAB_PRIVATE_COMMIT_SHA": recorded_commit_sha,
                    "RESEARCH_LAB_PRIVATE_ARTIFACT_MANIFEST_OUTPUT": self.config.private_artifact_manifest_output,
                },
                timeout_seconds=self.config.code_edit_build_timeout_seconds,
            )
            manifest_path = Path(self.config.private_artifact_manifest_output)
            if not manifest_path.is_absolute():
                manifest_path = repo_dir / manifest_path
            if not manifest_path.exists():
                raise CodeEditArtifactMissingError("private build did not produce artifact manifest output")
            candidate_manifest = PrivateModelArtifactManifest.from_mapping(
                json.loads(manifest_path.read_text(encoding="utf-8"))
            )
            errors = validate_private_model_artifact_manifest(candidate_manifest)
            if errors:
                raise CodeEditImageBuildError("candidate artifact manifest failed validation: " + "; ".join(errors))
            if candidate_manifest.model_artifact_hash == parent_artifact.model_artifact_hash:
                raise CodeEditImageBuildError("candidate artifact hash must differ from parent artifact hash")
            build_doc = {
                "schema_version": "1.1",
                "candidate_kind": "image_build",
                "source_mode": "parent_image_extract",
                "repo_ref_hash": repo_ref_hash,
                "parent_artifact_hash": parent_artifact.model_artifact_hash,
                "parent_manifest_hash": parent_artifact.manifest_hash,
                "parent_git_commit_sha": parent_artifact.git_commit_sha,
                "parent_image_digest_hash": parent_image_digest_hash,
                "extracted_source_tree_hash_before_patch": extracted_source_tree_hash_before_patch,
                "generated_build_scaffold": True,
                "extracted_top_level_paths": extracted_top_level_paths,
                "candidate_model_artifact_hash": candidate_manifest.model_artifact_hash,
                "candidate_model_manifest_hash": candidate_manifest.manifest_hash,
                "candidate_git_commit_sha": candidate_manifest.git_commit_sha,
                "build_workspace_git_commit_sha": git_commit_sha,
                "recorded_git_commit_sha": recorded_commit_sha,
                "recorded_git_commit_sha_source": recorded_commit_sha_source,
                "commit_excluded_paths": commit_excluded_paths[:200],
                "commit_excluded_path_count": len(commit_excluded_paths),
                "source_diff_hash": source_diff_hash,
                "changed_files": changed_files,
                "test_command_hash": canonical_hash({"cmd": self.config.private_test_cmd}),
                "build_command_hash": canonical_hash({"cmd": self.config.private_build_cmd}),
                "build_validation": "passed",
            }
            build_doc.update(
                _write_private_code_edit_diff_artifact(
                    parent_artifact=parent_artifact,
                    run_id=run_id,
                    candidate_index=candidate_index,
                    draft=draft,
                    source_diff_hash=source_diff_hash,
                )
            )
            build_doc_hash = sha256_json(build_doc)
            code_manifest = code_edit_candidate_manifest(
                draft=draft,
                parent_artifact_hash=parent_artifact.model_artifact_hash,
                candidate_artifact_hash=candidate_manifest.model_artifact_hash,
                candidate_model_manifest_hash=candidate_manifest.manifest_hash,
                source_diff_hash=source_diff_hash,
                build_doc_hash=build_doc_hash,
            )
            return CodeEditBuildResult(
                candidate_model_manifest=candidate_manifest,
                code_edit_manifest=code_manifest,
                source_diff_hash=source_diff_hash,
                build_doc={**build_doc, "build_doc_hash": build_doc_hash},
            )

    def _build_env(
        self,
        *,
        draft_path: Path,
        parent_manifest_path: Path,
        diff_path: Path,
        run_id: str,
        candidate_index: int,
        include_aws: bool,
    ) -> dict[str, str]:
        env: dict[str, str] = {}
        for key, value in os.environ.items():
            lowered = key.lower()
            if any(marker in lowered for marker in _PROVIDER_OR_SECRET_ENV_MARKERS):
                continue
            env[key] = value
        if include_aws:
            for key in _AWS_BUILD_ENV_NAMES:
                value = os.environ.get(key)
                if value:
                    env[key] = value
            env.setdefault("DOCKER_BUILDKIT", os.getenv("RESEARCH_LAB_DOCKER_BUILDKIT", "0"))
            env.setdefault("BUILDKIT_PROGRESS", os.getenv("RESEARCH_LAB_BUILDKIT_PROGRESS", "plain"))
        env.update(
            {
                "RESEARCH_LAB_CODE_EDIT_DRAFT_PATH": str(draft_path),
                "RESEARCH_LAB_ACTIVE_MANIFEST_PATH": str(parent_manifest_path),
                "RESEARCH_LAB_CODE_EDIT_DIFF_PATH": str(diff_path),
                "RESEARCH_LAB_RUN_ID": str(run_id),
                "RESEARCH_LAB_CANDIDATE_INDEX": str(candidate_index),
                "RESEARCH_LAB_CANDIDATE_KIND": "image_build",
            }
        )
        return env


_PROVIDER_OR_SECRET_ENV_MARKERS = (
    "openrouter",
    "scrapingdog",
    "exa",
    "supabase",
    "proxy",
    "secret",
    "token",
    "password",
    "private_key",
    "webshare",
)

_AWS_BUILD_ENV_NAMES = (
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_SECURITY_TOKEN",
    "AWS_REGION",
    "AWS_DEFAULT_REGION",
)

# Conservative stderr/stdout markers for registry/auth/network failures that a
# candidate's diff cannot cause (bug #30). Kept narrow on purpose: anything
# unrecognized stays charged to the candidate as today.
_INFRA_FAILURE_MARKERS = (
    "no basic auth credentials",
    "authorization token has expired",
    "unable to locate credentials",
    "is not authorized to perform",
    "not authorized",
    "accessdenied",
    "access denied",
    "denied: user",
    "ecr:initiatelayerupload",
    "ecr:uploadlayerpart",
    "ecr:completelayerupload",
    "ecr:putimage",
    "ecr:batchchecklayeravailability",
    "ecr:describeimages",
    "ecr:batchgetimage",
    "ecr:getdownloadurlforlayer",
    "expiredtoken",
    "expired token",
    "toomanyrequests",
    "too many requests",
    "tls handshake timeout",
    "connection refused",
    "connection reset by peer",
    "i/o timeout",
    "temporary failure in name resolution",
    "no such host",
    "dial tcp",
    "request canceled while waiting for connection",
    "received unexpected http status: 5",
    "502 bad gateway",
    "503 service unavailable",
    "504 gateway",
    "unexpected eof",
    "blob upload unknown",
    "cannot connect to the docker daemon",
    "error during connect",
)


def _is_infra_failure_text(*texts: str) -> bool:
    blob = " ".join(str(text or "") for text in texts).lower()
    return any(marker in blob for marker in _INFRA_FAILURE_MARKERS)


def _infra_retry_enabled() -> bool:
    return _env_flag("RESEARCH_LAB_BUILD_INFRA_RETRY_ENABLED", True)


def _infra_retry_backoff_seconds() -> float:
    raw = str(os.getenv("RESEARCH_LAB_BUILD_INFRA_RETRY_BACKOFF_SECONDS", "") or "").strip()
    try:
        return max(0.0, float(raw)) if raw else 5.0
    except ValueError:
        return 5.0


def _parent_image_prepull_enabled() -> bool:
    return _env_flag("RESEARCH_LAB_PARENT_IMAGE_PREPULL", True)


def _parent_image_pull_timeout_seconds() -> int:
    raw = str(os.getenv("RESEARCH_LAB_PARENT_IMAGE_PULL_TIMEOUT_SECONDS", "") or "").strip()
    try:
        return max(1, int(raw)) if raw else 900
    except ValueError:
        return 900


_REQUIRED_PARENT_APP_DIRS = (
    "gateway",
    "qualification",
    "sourcing_model",
    "validator_models",
)
_REQUIRED_PARENT_APP_FILES = (
    "research_lab_adapter.py",
    "requirements.txt",
)


def _prepare_parent_image_workspace(
    *,
    image_digest: str,
    repo_dir: Path,
    timeout_seconds: int,
    source_context: ParentImageSourceContext | None = None,
) -> tuple[str, list[str]]:
    if source_context is not None:
        if not source_context.source_root.is_dir():
            raise CodeEditBuildError("prepared parent image source context is missing its source root")
        if repo_dir.exists():
            shutil.rmtree(repo_dir)
        shutil.copytree(source_context.source_root, repo_dir)
        _validate_parent_app_runtime(repo_dir)
        extracted_top_level_paths = list(source_context.top_level_paths)
        extracted_source_tree_hash_before_patch = source_context.source_tree_hash
    else:
        extracted_source_tree_hash_before_patch, extracted_top_level_paths = _extract_parent_image_source(
            image_digest=image_digest,
            source_dir=repo_dir,
            timeout_seconds=timeout_seconds,
        )
    _write_research_lab_build_scaffold(repo_dir, base_image_ref=image_digest)
    _initialize_temporary_git_repo(repo_dir)
    return extracted_source_tree_hash_before_patch, extracted_top_level_paths


def _extract_parent_image_source(
    *,
    image_digest: str,
    source_dir: Path,
    timeout_seconds: int,
) -> tuple[str, list[str]]:
    image_ref = str(image_digest or "").strip()
    if not image_ref:
        raise CodeEditBuildError("parent artifact image_digest is required for code-edit candidate builds")
    if source_dir.exists() and any(source_dir.iterdir()):
        shutil.rmtree(source_dir)
    source_dir.mkdir(parents=True, exist_ok=True)
    _ensure_parent_image_available(image_ref, timeout_seconds=timeout_seconds)
    _extract_parent_image_app(image_ref, repo_dir=source_dir, timeout_seconds=timeout_seconds)
    _validate_parent_app_runtime(source_dir)
    return compute_private_source_tree_hash(source_dir), _top_level_paths(source_dir)


def _ensure_parent_image_available(image_ref: str, *, timeout_seconds: int) -> None:
    try:
        _run(["docker", "image", "inspect", image_ref], cwd=Path.cwd(), timeout_seconds=120)
        return
    except CodeEditBuildError:
        pass
    pull_cmd = ["docker", "pull", "--platform", _docker_platform(), image_ref]
    try:
        _run(pull_cmd, cwd=Path.cwd(), timeout_seconds=timeout_seconds)
        return
    except CodeEditBuildError as exc:
        if not _infra_retry_enabled():
            raise
        # A pull failure (auth, network, registry, timeout during pull) cannot
        # be caused by the candidate's diff — retry once, then surface it as
        # an infra failure instead of charging the candidate (bug #30).
        logger.warning(
            "research-lab code-edit build: parent image pull failed, retrying once after backoff: %s",
            str(exc)[:300],
        )
        time.sleep(_infra_retry_backoff_seconds())
        try:
            _run(pull_cmd, cwd=Path.cwd(), timeout_seconds=timeout_seconds)
        except CodeEditBuildError as retry_exc:
            raise CodeEditInfraFailureError(
                f"parent image pull failed after infra retry: {retry_exc}",
                stderr=retry_exc.stderr,
                stdout=retry_exc.stdout,
                exit_code=retry_exc.exit_code,
            ) from retry_exc


def _parent_image_cached_locally(image_ref: str) -> bool:
    try:
        _run(["docker", "image", "inspect", image_ref], cwd=Path.cwd(), timeout_seconds=120)
        return True
    except CodeEditBuildError:
        return False


def _prepull_parent_image_for_build(image_ref: str) -> None:
    """Pull the parent image as its own pre-step, outside the build-cmd budget (§5.3).

    The private build script's ``docker build`` re-pulls the parent image when it
    is not cached locally, and that cold pull used to burn a large slice of the
    single ``code_edit_build_timeout_seconds`` budget the whole build command
    shares — cold workers unfairly killed their first candidates. Pre-pulling
    here with a dedicated generous timeout
    (``RESEARCH_LAB_PARENT_IMAGE_PULL_TIMEOUT_SECONDS``, default 900) means the
    script's own pull hits the local cache and the build budget pays for the
    build only. A pull failure is a registry/auth/network problem the
    candidate's diff cannot cause, so it always classifies as infra (bug #30):
    retried once with the existing backoff, then surfaced as
    ``CodeEditInfraFailureError`` so higher levels requeue instead of charging
    the candidate. Disable via ``RESEARCH_LAB_PARENT_IMAGE_PREPULL=false`` to
    restore the pull-inside-the-build-budget behavior.
    """

    if not _parent_image_prepull_enabled():
        return
    image = str(image_ref or "").strip()
    if not image:
        return
    if _parent_image_cached_locally(image):
        return
    pull_cmd = ["docker", "pull", "--platform", _docker_platform(), image]
    timeout_seconds = _parent_image_pull_timeout_seconds()
    try:
        _run(pull_cmd, cwd=Path.cwd(), timeout_seconds=timeout_seconds)
        return
    except CodeEditBuildError as exc:
        if not _infra_retry_enabled():
            raise CodeEditInfraFailureError(
                f"parent image pre-pull failed: {exc}",
                stderr=exc.stderr,
                stdout=exc.stdout,
                exit_code=exc.exit_code,
            ) from exc
        logger.warning(
            "research-lab code-edit build: parent image pre-pull failed, retrying once after backoff: %s",
            str(exc)[:300],
        )
        time.sleep(_infra_retry_backoff_seconds())
        try:
            _run(pull_cmd, cwd=Path.cwd(), timeout_seconds=timeout_seconds)
        except CodeEditBuildError as retry_exc:
            raise CodeEditInfraFailureError(
                f"parent image pre-pull failed after infra retry: {retry_exc}",
                stderr=retry_exc.stderr,
                stdout=retry_exc.stdout,
                exit_code=retry_exc.exit_code,
            ) from retry_exc


def _extract_parent_image_app(image_ref: str, *, repo_dir: Path, timeout_seconds: int) -> None:
    container_id = ""
    try:
        container_id = _run(
            ["docker", "create", "--platform", _docker_platform(), image_ref, "sh", "-c", "true"],
            cwd=repo_dir.parent,
            timeout_seconds=120,
        ).strip()
        if not container_id:
            raise CodeEditBuildError("docker create did not return a container id for parent image extraction")
        _run(
            ["docker", "cp", f"{container_id}:/app/.", str(repo_dir)],
            cwd=repo_dir.parent,
            timeout_seconds=timeout_seconds,
        )
    finally:
        if container_id:
            try:
                _run(["docker", "rm", "-f", container_id], cwd=repo_dir.parent, timeout_seconds=120)
            except CodeEditBuildError:
                pass


def _validate_parent_app_runtime(repo_dir: Path) -> None:
    missing: list[str] = []
    for rel in _REQUIRED_PARENT_APP_DIRS:
        if not (repo_dir / rel).is_dir():
            missing.append(rel + "/")
    for rel in _REQUIRED_PARENT_APP_FILES:
        if not (repo_dir / rel).is_file():
            missing.append(rel)
    if missing:
        raise CodeEditBuildError("parent image /app missing required runtime paths: " + ", ".join(sorted(missing)))


def _top_level_paths(repo_dir: Path) -> list[str]:
    return sorted(
        path.name + ("/" if path.is_dir() else "")
        for path in repo_dir.iterdir()
        if path.name not in {".git", ".research_lab"}
    )


_SOURCE_CONTEXT_MAX_FILES = 300
_SOURCE_CONTEXT_MAX_PREVIEW_FILES = 12
_SOURCE_CONTEXT_MAX_PREVIEW_CHARS = 12000
_SOURCE_SEARCH_MAX_SNIPPET_CHARS = 320
_SOURCE_SECRET_LINE_MARKERS = (
    "sk-or-",
    "sb_secret",
    "service_role_key",
    "aws_secret_access_key",
    "aws_access_key_id",
    "password=",
    "password:",
    "private_key",
    "authorization:",
    "bearer ",
    "api_key=",
    "api-key",
    "webshare",
)
_SOURCE_DISALLOWED_PATH_PATTERNS = (
    r"(^|/)Dockerfile(\.[^/]*)?$",
    r"(^|/)docker-compose[^/]*\.ya?ml$",
    r"(^|/)\.github/",
    r"(^|/)\.git/",
    r"(^|/)\.env",
    r"(^|/)requirements[^/]*\.txt$",
    r"(^|/)pyproject\.toml$",
    r"(^|/)poetry\.lock$",
    r"(^|/)uv\.lock$",
    r"(^|/)Pipfile(\.lock)?$",
    r"(^|/)package(-lock)?\.json$",
    r"(^|/)\.research_lab/",
)


def _editable_runtime_files(
    source_dir: Path,
    *,
    allowed_prefixes: Sequence[str],
    allowed_exact_paths: Sequence[str],
    allowed_suffixes: Sequence[str],
) -> list[str]:
    allowed_exact = set(allowed_exact_paths)
    allowed = []
    for path in source_dir.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(source_dir).as_posix()
        if rel.startswith(".git/") or rel.startswith(".research_lab/"):
            continue
        if _source_path_disallowed(rel):
            continue
        if rel in allowed_exact or any(rel.startswith(prefix) for prefix in allowed_prefixes):
            if rel.endswith(tuple(allowed_suffixes)):
                allowed.append(rel)
    return sorted(allowed)[:_SOURCE_CONTEXT_MAX_FILES]


# Sentinel end-line marking full-file coverage: an unranged repeat read of a
# fully-read file is contained in (1, FULL_FILE_RANGE_END) and skips.
FULL_FILE_RANGE_END = 10**9


def _normalize_covered_ranges(
    already_read_paths: Sequence[str],
    already_read_ranges: Mapping[str, Sequence[Sequence[int]]] | None,
) -> dict[str, list[tuple[int, int]]]:
    covered: dict[str, list[tuple[int, int]]] = {}
    for path, ranges in (already_read_ranges or {}).items():
        normalized: list[tuple[int, int]] = []
        for item in ranges:
            try:
                start, end = int(item[0]), int(item[1])
            except (TypeError, ValueError, IndexError):
                continue
            if start >= 1 and end >= start:
                normalized.append((start, end))
        if normalized:
            covered[str(path)] = normalized
    for path in already_read_paths:
        # Legacy path-only bookkeeping: treat as full-file coverage.
        covered.setdefault(str(path), [(1, FULL_FILE_RANGE_END)])
    return covered


def _requested_range(request: CodeEditSourceInspectionRequest) -> tuple[int, int]:
    start = int(request.start_line) if request.start_line > 0 else 1
    if request.max_lines > 0:
        return start, start + int(request.max_lines) - 1
    return start, FULL_FILE_RANGE_END


def _range_contained(requested: tuple[int, int], covered: Sequence[tuple[int, int]]) -> bool:
    return any(start <= requested[0] and requested[1] <= end for start, end in covered)


def resolve_source_inspection_requests(
    source_context: ParentImageSourceContext,
    requests: Sequence[CodeEditSourceInspectionRequest],
    *,
    already_read_paths: Sequence[str],
    max_files: int,
    max_file_bytes: int,
    max_total_bytes: int,
    max_search_matches: int,
    source_access_v2: bool = False,
    already_read_ranges: Mapping[str, Sequence[Sequence[int]]] | None = None,
    max_ranges_per_path: int = 3,
) -> SourceInspectionBatch:
    allowed = set(source_context.editable_files)
    read_paths = set(already_read_paths)
    covered_ranges = _normalize_covered_ranges(already_read_paths, already_read_ranges)
    remaining_bytes = max(0, int(max_total_bytes))
    max_files = max(1, int(max_files))
    max_file_bytes = max(1, int(max_file_bytes))
    max_search_matches = max(1, int(max_search_matches))
    max_ranges_per_path = max(1, int(max_ranges_per_path))
    results: list[dict[str, Any]] = []
    event_results: list[dict[str, Any]] = []
    bytes_returned = 0

    for request in requests:
        operation = request.operation
        if operation == "finish":
            result = {
                "operation": "finish",
                "rationale_hash": sha256_json({"rationale": request.rationale}) if request.rationale else "",
            }
            results.append({key: value for key, value in result.items() if value})
            event_results.append({key: value for key, value in result.items() if value})
            continue
        if operation == "search":
            search_results = _search_source_files(
                source_context.source_root,
                source_context.editable_files,
                query=request.query,
                max_matches=max_search_matches,
            )
            model_result = {
                "operation": "search",
                "query_hash": sha256_json({"query": request.query}),
                "matches": search_results,
                "match_count": len(search_results),
                "truncated": len(search_results) >= max_search_matches,
            }
            results.append(model_result)
            event_results.append(
                {
                    "operation": "search",
                    "query_hash": model_result["query_hash"],
                    "match_count": len(search_results),
                    "result_hash": sha256_json(model_result),
                }
            )
            continue
        if operation != "read_file":
            raise CodeEditBuildError(f"unsupported source-inspection operation:{operation}")
        rel = request.path
        if rel not in allowed:
            raise CodeEditBuildError(f"source_inspection_path_not_editable:{rel}")
        if len(read_paths) >= max_files and rel not in read_paths:
            event_results.append(
                {
                    "operation": "read_file",
                    "path": rel,
                    "skipped": "max_files_reached",
                }
            )
            continue
        if not source_access_v2 and rel in read_paths:
            event_results.append(
                {
                    "operation": "read_file",
                    "path": rel,
                    "skipped": "already_read",
                }
            )
            continue
        if source_access_v2 and rel in covered_ranges:
            requested_range = _requested_range(request)
            if _range_contained(requested_range, covered_ranges[rel]):
                event_results.append(
                    {
                        "operation": "read_file",
                        "path": rel,
                        "start_line": requested_range[0],
                        "skipped": "range_already_read",
                    }
                )
                continue
            if len(covered_ranges[rel]) >= max_ranges_per_path:
                event_results.append(
                    {
                        "operation": "read_file",
                        "path": rel,
                        "start_line": requested_range[0],
                        "skipped": "max_ranges_reached",
                    }
                )
                continue
        if remaining_bytes <= 0:
            event_results.append(
                {
                    "operation": "read_file",
                    "path": rel,
                    "skipped": "max_total_bytes_reached",
                }
            )
            continue
        model_result = _read_source_file_for_model(
            source_context.source_root,
            rel,
            max_bytes=min(max_file_bytes, remaining_bytes),
            start_line=request.start_line if source_access_v2 else 0,
            max_lines=request.max_lines if source_access_v2 else 0,
            line_based=source_access_v2,
        )
        returned = int(model_result.get("bytes_returned") or 0)
        read_paths.add(rel)
        if source_access_v2:
            start = int(model_result.get("start_line") or 1)
            end = int(model_result.get("end_line") or 0)
            if not model_result.get("truncated") and start == 1:
                # Full-file read: cover all ranges so unranged repeats skip.
                covered_ranges[rel] = [(1, FULL_FILE_RANGE_END)]
            elif end >= start:
                covered_ranges.setdefault(rel, []).append((start, end))
            else:
                # Empty range (start past EOF) still counts as an attempt.
                covered_ranges.setdefault(rel, []).append((start, start))
        remaining_bytes = max(0, remaining_bytes - returned)
        bytes_returned += returned
        results.append(model_result)
        read_event: dict[str, Any] = {
            "operation": "read_file",
            "path": rel,
            "bytes_returned": returned,
            "truncated": bool(model_result.get("truncated")),
            "line_count": model_result.get("line_count"),
            "result_hash": sha256_json(model_result),
        }
        if source_access_v2:
            read_event["start_line"] = model_result.get("start_line")
            read_event["end_line"] = model_result.get("end_line")
            read_event["total_line_count"] = model_result.get("total_line_count")
            read_event["range_truncated"] = bool(model_result.get("range_truncated"))
        event_results.append(read_event)

    read_ranges = {path: tuple(ranges) for path, ranges in sorted(covered_ranges.items()) if path in read_paths}
    model_context = {
        "schema_version": "1.0",
        "source_tree_hash": source_context.source_tree_hash,
        "read_files": sorted(read_paths),
        "results": results,
        "bytes_returned": bytes_returned,
    }
    if source_access_v2:
        model_context["read_ranges"] = {
            path: [list(item) for item in ranges] for path, ranges in read_ranges.items()
        }
    event_doc = {
        "source_tree_hash": source_context.source_tree_hash,
        "read_files": sorted(read_paths),
        "read_file_count": len(read_paths),
        "result_count": len(results),
        "bytes_returned": bytes_returned,
        "results": event_results,
        "result_hash": sha256_json(model_context),
    }
    return SourceInspectionBatch(
        model_context=model_context,
        event_doc=event_doc,
        read_paths=tuple(sorted(read_paths)),
        bytes_returned=bytes_returned,
        read_ranges=read_ranges if source_access_v2 else {},
    )


def _search_source_files(
    source_root: Path,
    editable_files: Sequence[str],
    *,
    query: str,
    max_matches: int,
) -> list[dict[str, Any]]:
    needle = str(query or "").strip().lower()
    if not needle:
        return []
    terms = [term for term in re.split(r"\W+", needle) if len(term) >= 3]
    matches: list[dict[str, Any]] = []
    for rel in editable_files:
        path = source_root / rel
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for line_number, line in enumerate(lines, start=1):
            lowered = line.lower()
            if needle not in lowered and not (terms and all(term in lowered for term in terms[:4])):
                continue
            snippet = _redact_source_excerpt(line.strip())[:_SOURCE_SEARCH_MAX_SNIPPET_CHARS]
            matches.append({"path": rel, "line": line_number, "snippet": snippet})
            if len(matches) >= max_matches:
                return matches
    return matches


def _read_source_file_for_model(
    source_root: Path,
    rel: str,
    *,
    max_bytes: int,
    start_line: int = 0,
    max_lines: int = 0,
    line_based: bool = False,
) -> dict[str, Any]:
    path = source_root / rel
    try:
        raw_bytes = path.read_bytes()
    except OSError as exc:
        raise CodeEditBuildError(f"source_inspection_read_failed:{rel}") from exc
    if not line_based:
        # Legacy behavior (source-access v2 off): whole-file-from-top, byte-clipped.
        clipped = raw_bytes[: max(1, int(max_bytes))]
        content = clipped.decode("utf-8", errors="replace")
        redacted = _redact_source_excerpt(content)
        return {
            "operation": "read_file",
            "path": rel,
            "size_bytes": len(raw_bytes),
            "bytes_returned": len(clipped),
            "truncated": len(raw_bytes) > len(clipped),
            "line_count": content.count("\n") + (1 if content else 0),
            "content": redacted,
            "content_hash": sha256_json({"path": rel, "content": redacted}),
        }
    # Source-access v2: slice by lines after decode so drafts get exact hunk
    # context (no mid-line or mid-multibyte-char cuts), then apply the byte cap
    # to the slice at line granularity.
    text = raw_bytes.decode("utf-8", errors="replace")
    lines = text.splitlines(keepends=True)
    total_line_count = len(lines)
    start = max(1, int(start_line) if start_line > 0 else 1)
    requested = lines[start - 1 : start - 1 + int(max_lines)] if max_lines > 0 else lines[start - 1 :]
    budget = max(1, int(max_bytes))
    kept: list[str] = []
    used = 0
    for line in requested:
        encoded_len = len(line.encode("utf-8"))
        if kept and used + encoded_len > budget:
            break
        kept.append(line)
        used += encoded_len
        if used >= budget:
            break
    content = "".join(kept)
    end_line = start - 1 + len(kept)
    range_truncated = len(kept) < len(requested)
    redacted = _redact_source_excerpt(content)
    return {
        "operation": "read_file",
        "path": rel,
        "size_bytes": len(raw_bytes),
        "bytes_returned": used,
        "start_line": start,
        "end_line": end_line,
        "line_count": len(kept),
        "total_line_count": total_line_count,
        "truncated": end_line < total_line_count,
        "range_truncated": range_truncated,
        "content": redacted,
        "content_hash": sha256_json({"path": rel, "content": redacted}),
    }


def _source_file_inventory(source_dir: Path, editable_files: Sequence[str]) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    for rel in editable_files:
        path = source_dir / rel
        try:
            stat = path.stat()
            raw = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        files.append(
            {
                "path": rel,
                "size_bytes": int(stat.st_size),
                "line_count": raw.count("\n") + (1 if raw else 0),
            }
        )
    return files


def _source_file_previews(source_dir: Path, editable_files: Sequence[str]) -> list[dict[str, Any]]:
    previews: list[dict[str, Any]] = []
    for rel in sorted(editable_files, key=_preview_priority)[:_SOURCE_CONTEXT_MAX_PREVIEW_FILES]:
        path = source_dir / rel
        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        excerpt = raw[:_SOURCE_CONTEXT_MAX_PREVIEW_CHARS]
        previews.append(
            {
                "path": rel,
                "line_count": raw.count("\n") + (1 if raw else 0),
                "size_bytes": path.stat().st_size,
                "excerpt_truncated": len(raw) > len(excerpt),
                "content_excerpt": _redact_source_excerpt(excerpt),
            }
        )
    return previews


def _preview_priority(path: str) -> tuple[int, str]:
    if path == "research_lab_adapter.py":
        return (0, path)
    if path.startswith("sourcing_model/"):
        return (1, path)
    if path.startswith("gateway/research_lab/"):
        return (2, path)
    if path.startswith("qualification/scoring/"):
        return (3, path)
    if path.startswith("validator_models/"):
        return (4, path)
    return (5, path)


# Patterns matching secret VALUES (literal keys, tokens, connection strings).
# Bug #19: the old behavior replaced every line that merely *mentioned* a
# marker keyword, corrupting the source shown to the model → hunks built
# against text that does not match the real file → guaranteed-futile repairs.
# Value-level masking keeps line structure byte-stable: masked characters are
# replaced 1:1 with '*' so line counts and column offsets survive.
_SOURCE_SECRET_VALUE_PATTERNS = (
    re.compile(r"sk-or-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"sb_secret_[A-Za-z0-9_\-]{8,}"),
    re.compile(r"\beyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{4,}\.[A-Za-z0-9_\-]{4,}\b"),  # JWTs
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"(?i)\b[a-z][a-z0-9+.\-]{1,20}://[^\s'\"@/]{1,64}:[^\s'\"@]{4,}@[^\s'\"]+"),  # URL creds
    re.compile(r"(?i)(\bbearer\s+)([A-Za-z0-9_\-.=]{16,})"),
    re.compile(
        r"(?i)(\b(?:api[_-]?key|apikey|api[_-]?secret|secret[_-]?key|access[_-]?key|auth[_-]?token"
        r"|token|secret|passwd|password|pwd|service_role_key|aws_secret_access_key|private_key)\b"
        r"\s*[:=]\s*[\"']?)([A-Za-z0-9_\-]{16,})"
    ),
)


def _mask_secret_chars(value: str) -> str:
    return re.sub(r"[^\n]", "*", value)


def _redact_secret_values(text: str) -> str:
    """Mask secret literals in place, preserving line count and offsets.

    Residual risk (documented): a hard-coded secret in a format not covered by
    ``_SOURCE_SECRET_VALUE_PATTERNS`` is now shown to the model where the old
    line-blanking would have hidden it. Source here is extracted from a built
    image whose env/credential files are excluded from editable inventory, so
    literal secrets in scanned files should already be rare; the pattern list
    covers the provider/registry credentials this runtime actually uses.
    """

    def _sub(match: re.Match[str]) -> str:
        groups = match.groups()
        if not groups:
            return _mask_secret_chars(match.group(0))
        prefix = groups[0] or ""
        value = groups[1] if len(groups) > 1 else ""
        if value and not re.search(r"\d", value) and "secret" not in value.lower():
            # Identifier-like (no digits) — likely code, not a literal secret.
            return match.group(0)
        return prefix + _mask_secret_chars(match.group(0)[len(prefix):])

    for pattern in _SOURCE_SECRET_VALUE_PATTERNS:
        text = pattern.sub(_sub, text)
    return text


def _redact_source_excerpt(text: str) -> str:
    if _env_flag("RESEARCH_LAB_REDACT_VALUES_ONLY", True):
        return _redact_secret_values(text)
    # Legacy behavior (flag off): blank whole keyword-mentioning lines.
    redacted_lines = []
    for line in text.splitlines():
        lowered = line.lower()
        if any(marker in lowered for marker in _SOURCE_SECRET_LINE_MARKERS):
            redacted_lines.append("[redacted secret-like source line]")
        else:
            redacted_lines.append(line)
    return "\n".join(redacted_lines)


def _source_path_disallowed(rel: str) -> bool:
    return any(re.search(pattern, rel) for pattern in _SOURCE_DISALLOWED_PATH_PATTERNS)


def _write_research_lab_build_scaffold(repo_dir: Path, *, base_image_ref: str) -> None:
    base_image = _safe_docker_base_image_ref(base_image_ref)
    (repo_dir / "Dockerfile.research-lab").write_text(
        f"""FROM {base_image}
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
WORKDIR /app
COPY . /app
RUN python - <<'PY'
import research_lab_adapter
import sourcing_model
metadata = research_lab_adapter.adapter_metadata()
assert metadata.get("adapter_version")
assert sourcing_model is not None
PY
CMD ["python", "-c", "import json, research_lab_adapter; print(json.dumps(research_lab_adapter.adapter_metadata(), sort_keys=True))"]
""",
        encoding="utf-8",
    )
    (repo_dir / ".dockerignore").write_text(
        """.git
.research_lab
__pycache__/
*.pyc
*.pyo
.env
.env.*
*.pem
*.key
.pytest_cache/
.mypy_cache/
candidate.diff
code_edit_draft.json
parent_manifest.json
""",
        encoding="utf-8",
    )


def _safe_docker_base_image_ref(value: str) -> str:
    image_ref = str(value or "").strip()
    if not image_ref or any(char.isspace() for char in image_ref):
        raise CodeEditBuildError("invalid parent image digest for generated candidate Dockerfile")
    if not image_ref.startswith(("public.ecr.aws/", "localhost/", "127.0.0.1/")) and ".dkr.ecr." not in image_ref:
        raise CodeEditBuildError("parent image digest must be an ECR image reference for code-edit candidate builds")
    return image_ref


def _initialize_temporary_git_repo(repo_dir: Path) -> None:
    _run(["git", "init"], cwd=repo_dir, timeout_seconds=120)
    _run(["git", "config", "user.name", "Leadpoet Research Lab"], cwd=repo_dir, timeout_seconds=60)
    _run(["git", "config", "user.email", "research-lab@leadpoet.local"], cwd=repo_dir, timeout_seconds=60)
    _run(["git", "add", "-A"], cwd=repo_dir, timeout_seconds=120)
    _run(["git", "commit", "-m", "Research Lab parent image source"], cwd=repo_dir, timeout_seconds=120)


def _docker_platform() -> str:
    return os.getenv("RESEARCH_LAB_PRIVATE_MODEL_DOCKER_PLATFORM", "linux/amd64").strip() or "linux/amd64"


def _changed_files(repo_dir: Path) -> list[str]:
    output = _run(["git", "diff", "--name-only"], cwd=repo_dir, timeout_seconds=60)
    return sorted(line.strip() for line in output.splitlines() if line.strip())


def _git_ignored_paths(repo_dir: Path) -> list[str]:
    """Extracted files that git ignore rules keep out of commits (bug #29b).

    Never fails the build: this is diagnostic-only so a push can be audited
    against the built image.
    """

    try:
        output = _run(
            ["git", "ls-files", "--others", "--ignored", "--exclude-standard"],
            cwd=repo_dir,
            timeout_seconds=60,
        )
    except CodeEditBuildError:
        return []
    return sorted(line.strip() for line in output.splitlines() if line.strip())


def _looks_like_git_sha(value: str) -> bool:
    return bool(re.fullmatch(r"[0-9a-f]{7,40}", str(value or "").strip().lower()))


def _resolve_recorded_commit_sha(
    *,
    workspace_sha: str,
    parent_artifact: PrivateModelArtifactManifest,
) -> tuple[str, str]:
    """Pick the commit sha recorded into the candidate artifact manifest.

    Bug #29(a): the sha from the throwaway ``git init`` workspace never exists
    in the real private source repo, so auto-commit promotion wedges on the
    head check after the first push. Prefer a real source-repo sha when one is
    available: an explicit env input first, then the parent manifest's sha.
    Controlled by RESEARCH_LAB_BUILD_RECORD_REAL_HEAD_SHA (default true);
    disabling restores the previous throwaway-workspace behavior.
    """

    if not _env_flag("RESEARCH_LAB_BUILD_RECORD_REAL_HEAD_SHA", True):
        return workspace_sha, "build_workspace"
    env_sha = str(os.getenv("RESEARCH_LAB_PRIVATE_SOURCE_HEAD_SHA", "") or "").strip()
    if _looks_like_git_sha(env_sha):
        return env_sha.lower(), "env"
    parent_sha = str(parent_artifact.git_commit_sha or "").strip()
    if _looks_like_git_sha(parent_sha):
        return parent_sha.lower(), "parent_manifest"
    return workspace_sha, "build_workspace"


def _run_private_build_cmd_with_infra_retry(
    *,
    cmd: str,
    cwd: Path,
    env: Mapping[str, str],
    timeout_seconds: int,
) -> str:
    """Run the private build/push command, retrying once on infra failures.

    Genuine build/test failures stay charged to the candidate as
    ``CodeEditImageBuildError``. Registry/auth/network failures (which the
    candidate's diff cannot cause) are retried once with a short backoff and,
    if persistent, surfaced as ``CodeEditInfraFailureError`` so higher levels
    can requeue instead of rejecting the candidate (bug #30). Timeouts of the
    build command itself remain charged to the candidate.
    """

    try:
        return _run_shell(cmd, cwd=cwd, env=env, timeout_seconds=timeout_seconds)
    except CodeEditBuildError as exc:
        infra = _infra_retry_enabled() and _is_infra_failure_text(str(exc), exc.stderr, exc.stdout)
        if not infra:
            raise CodeEditImageBuildError(
                str(exc),
                stderr=exc.stderr,
                stdout=exc.stdout,
                exit_code=exc.exit_code,
            ) from exc
        logger.warning(
            "research-lab code-edit build: private build command hit an infra-classified failure, "
            "retrying once after backoff: %s",
            str(exc)[:300],
        )
        time.sleep(_infra_retry_backoff_seconds())
        try:
            return _run_shell(cmd, cwd=cwd, env=env, timeout_seconds=timeout_seconds)
        except CodeEditBuildError as retry_exc:
            if _is_infra_failure_text(str(retry_exc), retry_exc.stderr, retry_exc.stdout):
                raise CodeEditInfraFailureError(
                    f"private build command failed on infrastructure after one retry: {retry_exc}",
                    stderr=retry_exc.stderr,
                    stdout=retry_exc.stdout,
                    exit_code=retry_exc.exit_code,
                ) from retry_exc
            raise CodeEditImageBuildError(
                str(retry_exc),
                stderr=retry_exc.stderr,
                stdout=retry_exc.stdout,
                exit_code=retry_exc.exit_code,
            ) from retry_exc


def _py_compile_changed_files(repo_dir: Path, changed_files: list[str]) -> None:
    py_files = [str(repo_dir / path) for path in changed_files if path.endswith(".py")]
    if not py_files:
        return
    _run([sys.executable, "-m", "py_compile", *py_files], cwd=repo_dir, timeout_seconds=240)


def _write_private_code_edit_diff_artifact(
    *,
    parent_artifact: PrivateModelArtifactManifest,
    run_id: str,
    candidate_index: int,
    draft: CodeEditDraft,
    source_diff_hash: str,
) -> dict[str, Any]:
    """Persist the successful raw diff privately for stale-parent rebase."""

    payload = {
        "schema_version": "1.0",
        "artifact_type": "research_lab_code_edit_source_diff",
        "run_id": str(run_id),
        "candidate_index": int(candidate_index),
        "parent_artifact_hash": parent_artifact.model_artifact_hash,
        "parent_manifest_hash": parent_artifact.manifest_hash,
        "source_diff_hash": source_diff_hash,
        "target_files": list(draft.target_files),
        "unified_diff": draft.unified_diff,
        "draft_hash": sha256_json(draft.to_dict()),
    }
    artifact_hash = sha256_json(payload)
    manifest_uri = str(parent_artifact.manifest_uri or "")
    if not manifest_uri.startswith("s3://"):
        try:
            local_dir = Path(tempfile.gettempdir()) / "research-lab-source-diffs"
            local_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
            local_path = local_dir / (
                f"{sha256_json({'run_id': str(run_id), 'candidate_index': int(candidate_index), 'hash': source_diff_hash})}.json"
            )
            local_path.write_text(
                json.dumps({**payload, "artifact_hash": artifact_hash}, sort_keys=True),
                encoding="utf-8",
            )
            return {
                "source_diff_artifact_uri": local_path.resolve().as_uri(),
                "source_diff_artifact_hash": artifact_hash,
                "source_diff_artifact_storage": "local_file",
            }
        except Exception as exc:
            return {
                "source_diff_artifact_hash": artifact_hash,
                "source_diff_artifact_error": str(exc)[:300],
            }
    try:
        bucket, key = _parse_s3_uri(manifest_uri)
    except ValueError as exc:
        return {"source_diff_artifact_error": str(exc)[:200]}
    base_prefix = key.rsplit("/", 1)[0] if "/" in key else "research-lab/sourcing-model"
    object_key = f"{base_prefix}/candidates/{run_id}/{int(candidate_index)}/source_diff.json"
    try:
        import boto3  # type: ignore

        boto3.client("s3").put_object(
            Bucket=bucket,
            Key=object_key,
            Body=json.dumps({**payload, "artifact_hash": artifact_hash}, sort_keys=True).encode("utf-8"),
            ContentType="application/json",
        )
    except Exception as exc:
        return {
            "source_diff_artifact_hash": artifact_hash,
            "source_diff_artifact_error": str(exc)[:300],
        }
    return {
        "source_diff_artifact_uri": f"s3://{bucket}/{object_key}",
        "source_diff_artifact_hash": artifact_hash,
    }


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    raw = str(uri or "")
    if not raw.startswith("s3://"):
        raise ValueError("expected s3:// URI")
    without_scheme = raw[5:]
    bucket, sep, key = without_scheme.partition("/")
    if not bucket or not sep or not key:
        raise ValueError("invalid s3 URI")
    return bucket, key


def _run(cmd: list[str], *, cwd: Path, timeout_seconds: int) -> str:
    try:
        completed = subprocess.run(
            cmd,
            cwd=str(cwd),
            check=True,
            text=True,
            capture_output=True,
            timeout=max(1, int(timeout_seconds)),
        )
        return completed.stdout.strip()
    except subprocess.TimeoutExpired as exc:
        raise CodeEditBuildError(f"command timed out: {_safe_cmd(cmd)}") from exc
    except subprocess.CalledProcessError as exc:
        raise CodeEditBuildError(
            f"command failed exit={exc.returncode}: {_safe_cmd(cmd)} stderr={_safe_text(exc.stderr)}",
            stderr=_safe_text(exc.stderr, limit=12000),
            stdout=_safe_text(exc.stdout, limit=12000),
            exit_code=int(exc.returncode),
        ) from exc


def _run_git_apply(
    diff_path: Path,
    *,
    cwd: Path,
    timeout_seconds: int,
    check: bool,
) -> str:
    args = ["git", "apply", "--recount"]
    if check:
        args.append("--check")
    args.append(str(diff_path))
    try:
        return _run(args, cwd=cwd, timeout_seconds=timeout_seconds)
    except CodeEditBuildError as strict_error:
        unified_diff = diff_path.read_text(encoding="utf-8")
        if not _can_retry_git_apply_without_edge_context(unified_diff):
            raise

        fallback_args = ["git", "apply", "--recount", "--unidiff-zero"]
        if check:
            fallback_args.append("--check")
        fallback_args.append(str(diff_path))
        try:
            result = _run(fallback_args, cwd=cwd, timeout_seconds=timeout_seconds)
        except CodeEditBuildError:
            raise strict_error
        logger.info(
            "research_lab_patch_apply_zero_edge_context_fallback check=%s diff_hash=%s",
            int(check),
            sha256_json({"unified_diff": unified_diff}),
        )
        return result


def _can_retry_git_apply_without_edge_context(unified_diff: str) -> bool:
    """Allow Git's zero-context mode only for exact replacement hunks.

    Git rejects model-generated hunks that omit unchanged context at either
    edge, even when every removed byte matches the parent source. Requiring a
    removal in each edge-context-free hunk keeps the fallback anchored to
    existing source and excludes line-number-only insertion patches. Hunks
    with normal context at both edges remain safely anchored as usual.
    """

    hunk_prefixes: list[str] = []
    saw_hunk = False
    needs_fallback = False

    def finish_hunk() -> bool:
        nonlocal needs_fallback
        if not hunk_prefixes:
            return False
        missing_edge_context = hunk_prefixes[0] != " " or hunk_prefixes[-1] != " "
        if missing_edge_context and "-" not in hunk_prefixes:
            return False
        if missing_edge_context:
            needs_fallback = True
        return True

    for line in str(unified_diff or "").rstrip("\r\n").splitlines():
        if line.startswith("@@ "):
            if hunk_prefixes and not finish_hunk():
                return False
            hunk_prefixes = []
            saw_hunk = True
            continue
        if not saw_hunk:
            continue
        if line.startswith("diff --git "):
            if not finish_hunk():
                return False
            hunk_prefixes = []
            saw_hunk = False
            continue
        if line.startswith("\\ No newline at end of file"):
            continue
        if not line or line[0] not in {" ", "+", "-"}:
            return False
        hunk_prefixes.append(line[0])

    if not saw_hunk or not finish_hunk():
        return False
    return needs_fallback


def _run_shell(cmd: str, *, cwd: Path, env: Mapping[str, str], timeout_seconds: int) -> str:
    try:
        completed = subprocess.run(
            cmd,
            cwd=str(cwd),
            env=dict(env),
            shell=True,
            check=True,
            text=True,
            capture_output=True,
            timeout=max(1, int(timeout_seconds)),
        )
        return completed.stdout.strip()
    except subprocess.TimeoutExpired as exc:
        raise CodeEditBuildError("configured private build/test command timed out") from exc
    except subprocess.CalledProcessError as exc:
        stderr_summary = _safe_text(exc.stderr, tail=True)
        raise CodeEditBuildError(
            f"configured private build/test command failed exit={exc.returncode} stderr_tail={stderr_summary}",
            stderr=_safe_text(exc.stderr, limit=12000),
            stdout=_safe_text(exc.stdout, limit=12000),
            exit_code=int(exc.returncode),
        ) from exc


def _safe_cmd(cmd: list[str]) -> str:
    redacted: list[str] = []
    for item in cmd:
        text = str(item)
        if "://" in text or "@" in text:
            redacted.append("[redacted-ref]")
        else:
            redacted.append(text)
    return " ".join(redacted)


def _safe_text(value: str | None, *, limit: int = 500, tail: bool = False) -> str:
    text = str(value or "")
    for marker in ("sk-or-", "service_role", "openrouter_api_key", "raw_secret"):
        text = text.replace(marker, "[redacted]")
    compacted = " ".join(text.split())
    limit = max(1, int(limit))
    if len(compacted) <= limit:
        return compacted
    if tail:
        return compacted[-limit:]
    return compacted[:limit]
