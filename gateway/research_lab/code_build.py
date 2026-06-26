"""Gateway-side builder for code-edit Research Lab candidate images."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Mapping

from gateway.research_lab.config import ResearchLabGatewayConfig
from gateway.research_lab.store import canonical_hash
from research_lab.canonical import sha256_json
from research_lab.code_editing import (
    CodeEditDraft,
    code_edit_candidate_manifest,
    validate_code_edit_draft,
)
from research_lab.eval import PrivateModelArtifactManifest, validate_private_model_artifact_manifest


class CodeEditBuildError(RuntimeError):
    """Raised when a code-edit candidate cannot be built safely."""


@dataclass(frozen=True)
class CodeEditBuildResult:
    candidate_model_manifest: PrivateModelArtifactManifest
    code_edit_manifest: dict[str, Any]
    source_diff_hash: str
    build_doc: dict[str, Any]


class CodeEditCandidateBuilder:
    """Build a candidate private model image from a validated unified diff."""

    def __init__(self, config: ResearchLabGatewayConfig):
        self.config = config

    def enabled(self) -> bool:
        return bool(
            self.config.private_repo_url
            and self.config.private_test_cmd
            and self.config.private_build_cmd
            and self.config.private_artifact_manifest_output
        )

    def build(
        self,
        *,
        draft: CodeEditDraft,
        parent_artifact: PrivateModelArtifactManifest,
        run_id: str,
        candidate_index: int,
    ) -> CodeEditBuildResult:
        if not self.enabled():
            missing = [
                name
                for name, value in {
                    "RESEARCH_LAB_PRIVATE_REPO_URL": self.config.private_repo_url,
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
        repo_ref_hash = canonical_hash({"private_repo_url": self.config.private_repo_url})
        with tempfile.TemporaryDirectory(prefix="research-lab-code-edit-") as tmp:
            tmp_dir = Path(tmp)
            repo_dir = tmp_dir / "repo"
            _run(
                [
                    "git",
                    "clone",
                    "--branch",
                    self.config.private_repo_branch or "main",
                    "--single-branch",
                    self.config.private_repo_url,
                    str(repo_dir),
                ],
                cwd=tmp_dir,
                timeout_seconds=self.config.code_edit_build_timeout_seconds,
            )
            _checkout_parent_commit(repo_dir, parent_artifact.git_commit_sha)
            diff_path = tmp_dir / "candidate.diff"
            draft_path = tmp_dir / "code_edit_draft.json"
            parent_manifest_path = tmp_dir / "parent_manifest.json"
            diff_path.write_text(draft.unified_diff, encoding="utf-8")
            draft_path.write_text(json.dumps(draft.to_dict(), sort_keys=True), encoding="utf-8")
            parent_manifest_path.write_text(json.dumps(parent_artifact.to_dict(), sort_keys=True), encoding="utf-8")
            _run(["git", "apply", "--check", str(diff_path)], cwd=repo_dir, timeout_seconds=120)
            _run(["git", "apply", str(diff_path)], cwd=repo_dir, timeout_seconds=120)
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
                    ),
                    "RESEARCH_LAB_PRIVATE_COMMIT_SHA": git_commit_sha,
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
                "schema_version": "1.0",
                "candidate_kind": "image_build",
                "repo_ref_hash": repo_ref_hash,
                "parent_artifact_hash": parent_artifact.model_artifact_hash,
                "parent_manifest_hash": parent_artifact.manifest_hash,
                "parent_git_commit_sha": parent_artifact.git_commit_sha,
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
    ) -> dict[str, str]:
        env: dict[str, str] = {}
        for key, value in os.environ.items():
            lowered = key.lower()
            if any(marker in lowered for marker in _PROVIDER_OR_SECRET_ENV_MARKERS):
                continue
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


def _checkout_parent_commit(repo_dir: Path, commit_sha: str) -> None:
    commit = str(commit_sha or "").strip()
    if len(commit) >= 7:
        _run(["git", "checkout", "--detach", commit], cwd=repo_dir, timeout_seconds=120)


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
