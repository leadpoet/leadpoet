"""Gateway-side builder for code-edit Research Lab candidate images."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any, Mapping, Sequence

from gateway.research_lab.config import ResearchLabGatewayConfig
from gateway.research_lab.store import canonical_hash
from research_lab.canonical import sha256_json
from research_lab.code_editing import (
    CodeEditDraft,
    code_edit_candidate_manifest,
    extract_unified_diff_paths,
    validate_code_edit_draft,
)
from research_lab.eval import (
    PrivateModelArtifactManifest,
    compute_private_source_tree_hash,
    validate_private_model_artifact_manifest,
)


class CodeEditBuildError(RuntimeError):
    """Raised when a code-edit candidate cannot be built safely."""


@dataclass(frozen=True)
class CodeEditBuildResult:
    candidate_model_manifest: PrivateModelArtifactManifest
    code_edit_manifest: dict[str, Any]
    source_diff_hash: str
    build_doc: dict[str, Any]


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

    def prompt_context(self) -> dict[str, Any]:
        return {
            "source_mode": self.source_mode,
            "parent_image_digest_hash": self.parent_image_digest_hash,
            "source_tree_hash": self.source_tree_hash,
            "extracted_top_level_paths": list(self.top_level_paths),
            "editable_file_count": len(self.editable_files),
            "editable_files": list(self.editable_files),
            "file_previews": [dict(item) for item in self.file_previews],
            "rules": [
                "Only edit files listed in editable_files.",
                "Prefer files in file_previews because exact source excerpts are included.",
                "Do not invent paths or create new files.",
            ],
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
        return ParentImageSourceContext(
            source_root=source_root,
            source_mode="parent_image_extract",
            parent_image_digest_hash=canonical_hash({"image_digest": image_digest}),
            source_tree_hash=source_tree_hash,
            top_level_paths=tuple(top_level_paths),
            editable_files=tuple(editable_files),
            file_previews=tuple(_source_file_previews(source_root, editable_files)),
        )

    def validate_draft_against_source_context(
        self,
        draft: CodeEditDraft,
        source_context: ParentImageSourceContext,
    ) -> list[str]:
        allowed = set(source_context.editable_files)
        paths = set(draft.target_files) | extract_unified_diff_paths(draft.unified_diff)
        errors: list[str] = []
        for path in sorted(paths):
            if path not in allowed:
                errors.append(f"code_edit_path_not_in_extracted_source:{path}")
        return errors

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
                    raise CodeEditBuildError("; ".join(context_errors))
            diff_path = tmp_dir / "candidate.diff"
            draft_path = tmp_dir / "code_edit_draft.json"
            parent_manifest_path = tmp_dir / "parent_manifest.json"
            diff_path.write_text(draft.unified_diff, encoding="utf-8")
            draft_path.write_text(json.dumps(draft.to_dict(), sort_keys=True), encoding="utf-8")
            parent_manifest_path.write_text(json.dumps(parent_artifact.to_dict(), sort_keys=True), encoding="utf-8")
            _run(["git", "apply", "--recount", "--check", str(diff_path)], cwd=repo_dir, timeout_seconds=120)
            _run(["git", "apply", "--recount", str(diff_path)], cwd=repo_dir, timeout_seconds=120)
            changed_files = _changed_files(repo_dir)
            if not changed_files:
                raise CodeEditBuildError("code edit produced no repository changes")
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
            _run_shell(
                self.config.private_build_cmd,
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
                    "RESEARCH_LAB_PRIVATE_COMMIT_SHA": git_commit_sha,
                    "RESEARCH_LAB_PRIVATE_ARTIFACT_MANIFEST_OUTPUT": self.config.private_artifact_manifest_output,
                },
                timeout_seconds=self.config.code_edit_build_timeout_seconds,
            )
            manifest_path = Path(self.config.private_artifact_manifest_output)
            if not manifest_path.is_absolute():
                manifest_path = repo_dir / manifest_path
            if not manifest_path.exists():
                raise CodeEditBuildError("private build did not produce artifact manifest output")
            candidate_manifest = PrivateModelArtifactManifest.from_mapping(
                json.loads(manifest_path.read_text(encoding="utf-8"))
            )
            errors = validate_private_model_artifact_manifest(candidate_manifest)
            if errors:
                raise CodeEditBuildError("candidate artifact manifest failed validation: " + "; ".join(errors))
            if candidate_manifest.model_artifact_hash == parent_artifact.model_artifact_hash:
                raise CodeEditBuildError("candidate artifact hash must differ from parent artifact hash")
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
                "source_diff_hash": source_diff_hash,
                "changed_files": changed_files,
                "test_command_hash": canonical_hash({"cmd": self.config.private_test_cmd}),
                "build_command_hash": canonical_hash({"cmd": self.config.private_build_cmd}),
                "build_validation": "passed",
            }
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
    _write_research_lab_build_scaffold(repo_dir)
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
    _run(
        ["docker", "pull", "--platform", _docker_platform(), image_ref],
        cwd=Path.cwd(),
        timeout_seconds=timeout_seconds,
    )


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
_SOURCE_CONTEXT_MAX_PREVIEW_CHARS = 3500


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
        if rel in allowed_exact or any(rel.startswith(prefix) for prefix in allowed_prefixes):
            if rel.endswith(tuple(allowed_suffixes)):
                allowed.append(rel)
    return sorted(allowed)[:_SOURCE_CONTEXT_MAX_FILES]


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


def _redact_source_excerpt(text: str) -> str:
    redacted_lines = []
    for line in text.splitlines():
        lowered = line.lower()
        if any(marker in lowered for marker in ("sk-or-", "sb_secret", "aws_secret_access_key", "password=")):
            redacted_lines.append("[redacted secret-like source line]")
        else:
            redacted_lines.append(line)
    return "\n".join(redacted_lines)


def _write_research_lab_build_scaffold(repo_dir: Path) -> None:
    (repo_dir / "Dockerfile.research-lab").write_text(
        """FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
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


def _py_compile_changed_files(repo_dir: Path, changed_files: list[str]) -> None:
    py_files = [str(repo_dir / path) for path in changed_files if path.endswith(".py")]
    if not py_files:
        return
    _run(["python3", "-m", "py_compile", *py_files], cwd=repo_dir, timeout_seconds=240)


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
            f"command failed exit={exc.returncode}: {_safe_cmd(cmd)} stderr={_safe_text(exc.stderr)}"
        ) from exc


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
        raise CodeEditBuildError(
            f"configured private build/test command failed exit={exc.returncode} stderr={_safe_text(exc.stderr)}"
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


def _safe_text(value: str | None) -> str:
    text = str(value or "")
    for marker in ("sk-or-", "service_role", "openrouter_api_key", "raw_secret"):
        text = text.replace(marker, "[redacted]")
    return " ".join(text.split())[:500]
