"""Fail-closed publication of an Arena winner to Git ``main`` and ``lab``."""

from __future__ import annotations

import datetime as dt
import io
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import tarfile
from typing import Any, Mapping

from lab_arena import source_bundle


class PromotionError(RuntimeError):
    """A promotion could not be prepared or safely published."""


_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")
_OBJECT_ID = re.compile(r"[0-9a-f]{40}")
_SAFE_GIT_ENVIRONMENT = {
    "HOME", "PATH", "TMPDIR", "TMP", "TEMP", "LANG", "LC_ALL", "SSH_AUTH_SOCK",
    "GIT_CONFIG_GLOBAL", "GIT_CONFIG_NOSYSTEM", "GIT_SSH", "GIT_SSH_COMMAND",
    "GIT_ASKPASS", "SSH_ASKPASS",
}


class GitPromoter:
    """Build one ordinary commit and atomically advance two remote branches."""

    def __init__(
        self,
        repo_url: str,
        work_dir: str | Path,
        git_environment: Mapping[str, str] | None = None,
    ) -> None:
        if not isinstance(repo_url, str) or not repo_url:
            raise PromotionError("promotion_repository_invalid")
        self.repo_url = repo_url
        self.work_dir = Path(work_dir)
        inherited = {
            key: value for key, value in os.environ.items()
            if key in _SAFE_GIT_ENVIRONMENT or not key.startswith("GIT_")
        }
        if git_environment is not None:
            for key, value in git_environment.items():
                if not isinstance(key, str) or not isinstance(value, str) or "\x00" in value:
                    raise PromotionError("promotion_git_environment_invalid")
                inherited[key] = value
        inherited["GIT_TERMINAL_PROMPT"] = "0"
        try:
            config_count = int(inherited.get("GIT_CONFIG_COUNT", "0"))
        except ValueError as exc:
            raise PromotionError("promotion_git_environment_invalid") from exc
        if not 0 <= config_count <= 100:
            raise PromotionError("promotion_git_environment_invalid")
        inherited.update({
            "GIT_CONFIG_COUNT": str(config_count + 2),
            f"GIT_CONFIG_KEY_{config_count}": "core.hooksPath",
            f"GIT_CONFIG_VALUE_{config_count}": os.devnull,
            f"GIT_CONFIG_KEY_{config_count + 1}": "advice.detachedHead",
            f"GIT_CONFIG_VALUE_{config_count + 1}": "false",
        })
        self._environment = inherited

    def prepare(
        self,
        source_bytes: bytes,
        *,
        round_id: str,
        submission_id: str,
        timestamp: str | dt.datetime,
    ) -> dict[str, str]:
        round_id = self._identifier(round_id)
        submission_id = self._identifier(submission_id)
        canonical_time = self._timestamp(timestamp)
        self._initialize()
        heads = self._remote_heads()
        self._fetch_heads(heads)
        commit = self._build_commit(
            source_bytes,
            main_before=heads["main"],
            lab_before=heads["lab"],
            round_id=round_id,
            submission_id=submission_id,
            timestamp=canonical_time,
        )
        return {
            "commit": commit,
            "main_before": heads["main"],
            "lab_before": heads["lab"],
            "timestamp": canonical_time,
        }

    def publish(
        self,
        source_bytes: bytes,
        *,
        plan: Mapping[str, str],
        round_id: str,
        submission_id: str,
    ) -> str:
        round_id = self._identifier(round_id)
        submission_id = self._identifier(submission_id)
        expected = self._plan(plan)
        self._initialize()
        heads = self._remote_heads()
        already_published = (
            heads["main"] == expected["commit"]
            and heads["lab"] == expected["commit"]
        )
        if not already_published and (
            heads["main"] != expected["main_before"]
            or heads["lab"] != expected["lab_before"]
        ):
            raise PromotionError("promotion_remote_changed")
        self._fetch_heads(heads)
        actual = self._build_commit(
            source_bytes,
            main_before=expected["main_before"],
            lab_before=expected["lab_before"],
            round_id=round_id,
            submission_id=submission_id,
            timestamp=expected["timestamp"],
        )
        if actual != expected["commit"]:
            raise PromotionError("promotion_plan_mismatch")
        if already_published:
            return actual
        try:
            self._git(
                "push", "--porcelain", "--atomic", self.repo_url,
                f"{actual}:refs/heads/main", f"{actual}:refs/heads/lab",
                timeout=60,
            )
        except PromotionError:
            after = self._remote_heads()
            if after["main"] == actual and after["lab"] == actual:
                return actual
            raise
        after = self._remote_heads()
        if after["main"] != actual or after["lab"] != actual:
            raise PromotionError("promotion_publish_unconfirmed")
        return actual

    def _initialize(self) -> None:
        self.work_dir.mkdir(parents=True, exist_ok=True)
        if not (self.work_dir / "HEAD").exists():
            if any(self.work_dir.iterdir()):
                raise PromotionError("promotion_work_dir_invalid")
            self._git("init", "--bare", ".")
        if not (self.work_dir / "objects").is_dir():
            raise PromotionError("promotion_work_dir_invalid")

    def _remote_heads(self) -> dict[str, str]:
        output = self._git(
            "ls-remote", "--heads", self.repo_url,
            "refs/heads/main", "refs/heads/lab", timeout=30,
        )
        heads: dict[str, str] = {}
        for line in output.decode("ascii", "strict").splitlines():
            fields = line.split("\t")
            if len(fields) == 2 and fields[1] in ("refs/heads/main", "refs/heads/lab"):
                heads[fields[1].rsplit("/", 1)[1]] = fields[0]
        if set(heads) != {"main", "lab"} or any(not _OBJECT_ID.fullmatch(v) for v in heads.values()):
            raise PromotionError("promotion_remote_heads_invalid")
        return heads

    def _fetch_heads(self, heads: Mapping[str, str]) -> None:
        self._git(
            "fetch", "--quiet", "--no-tags", "--atomic", self.repo_url,
            "+refs/heads/main:refs/remotes/promotion/main",
            "+refs/heads/lab:refs/remotes/promotion/lab",
            timeout=60,
        )
        for branch in ("main", "lab"):
            fetched = self._git("rev-parse", f"refs/remotes/promotion/{branch}").decode().strip()
            if fetched != heads[branch]:
                raise PromotionError("promotion_remote_changed")

    def _build_commit(
        self,
        source_bytes: bytes,
        *,
        main_before: str,
        lab_before: str,
        round_id: str,
        submission_id: str,
        timestamp: str,
    ) -> str:
        files = self._archive_files(source_bytes)
        tree = self._write_tree(files)
        arguments = ["commit-tree", tree, "-p", main_before]
        if lab_before != main_before:
            arguments.extend(("-p", lab_before))
        message = f"Promote Arena winner {submission_id} from round {round_id}\n"
        environment = {
            "GIT_AUTHOR_NAME": "Leadpoet Arena",
            "GIT_AUTHOR_EMAIL": "arena@leadpoet.com",
            "GIT_AUTHOR_DATE": timestamp,
            "GIT_COMMITTER_NAME": "Leadpoet Arena",
            "GIT_COMMITTER_EMAIL": "arena@leadpoet.com",
            "GIT_COMMITTER_DATE": timestamp,
        }
        commit = self._git(*arguments, input_bytes=message.encode(), environment=environment).decode().strip()
        if not _OBJECT_ID.fullmatch(commit):
            raise PromotionError("promotion_commit_invalid")
        return commit

    def _archive_files(self, source_bytes: bytes) -> dict[tuple[str, ...], tuple[str, bytes]]:
        try:
            facts = source_bundle.validate_source_archive(source_bytes)
        except source_bundle.SourceBundleError as exc:
            raise PromotionError("promotion_archive_invalid") from exc
        root = str(facts["source_root"])
        prefix = root + "/" if root else ""
        files: dict[tuple[str, ...], tuple[str, bytes]] = {}
        try:
            with tarfile.open(fileobj=io.BytesIO(bytes(source_bytes)), mode="r|gz") as archive:
                for member in archive:
                    if not member.isfile():
                        continue
                    name = member.name[len(prefix):] if prefix else member.name
                    path = PurePosixPath(name)
                    lower = tuple(part.lower() for part in path.parts)
                    if (
                        ".git" in lower
                        or ".gitattributes" in lower
                        or lower[:2] == (".github", "workflows")
                    ):
                        raise PromotionError("promotion_path_forbidden")
                    source = archive.extractfile(member)
                    if source is None:
                        raise PromotionError("promotion_archive_invalid")
                    data = source.read(int(member.size) + 1)
                    if len(data) != int(member.size):
                        raise PromotionError("promotion_archive_invalid")
                    mode = "100755" if member.mode & 0o111 else "100644"
                    files[path.parts] = (mode, data)
        except PromotionError:
            raise
        except (OSError, EOFError, tarfile.TarError) as exc:
            raise PromotionError("promotion_archive_invalid") from exc
        return files

    def _write_tree(self, files: Mapping[tuple[str, ...], tuple[str, bytes]]) -> str:
        nodes: dict[str, Any] = {}
        for parts, value in files.items():
            node = nodes
            for part in parts[:-1]:
                node = node.setdefault(part, {})
            node[parts[-1]] = value

        def write(node: Mapping[str, Any]) -> str:
            records = bytearray()
            for name in sorted(node, key=lambda item: item.encode("utf-8")):
                value = node[name]
                if isinstance(value, dict):
                    mode, kind, object_id = "040000", "tree", write(value)
                else:
                    mode, data = value
                    kind = "blob"
                    object_id = self._git("hash-object", "-w", "--stdin", input_bytes=data).decode().strip()
                records.extend(f"{mode} {kind} {object_id}\t{name}".encode("utf-8") + b"\0")
            return self._git("mktree", "-z", input_bytes=bytes(records)).decode().strip()

        return write(nodes)

    def _git(
        self,
        *arguments: str,
        input_bytes: bytes | None = None,
        environment: Mapping[str, str] | None = None,
        timeout: int = 30,
    ) -> bytes:
        env = dict(self._environment)
        if environment:
            env.update(environment)
        try:
            completed = subprocess.run(
                ("git", *arguments), cwd=self.work_dir, env=env, input=input_bytes,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise PromotionError("promotion_git_failed") from exc
        if completed.returncode:
            raise PromotionError("promotion_git_failed")
        return completed.stdout

    @staticmethod
    def _identifier(value: str) -> str:
        if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
            raise PromotionError("promotion_identifier_invalid")
        return value

    @staticmethod
    def _timestamp(value: str | dt.datetime) -> str:
        try:
            parsed = value if isinstance(value, dt.datetime) else dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
        except (AttributeError, ValueError) as exc:
            raise PromotionError("promotion_timestamp_invalid") from exc
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise PromotionError("promotion_timestamp_invalid")
        return parsed.astimezone(dt.timezone.utc).replace(microsecond=0).isoformat()

    @classmethod
    def _plan(cls, plan: Mapping[str, str]) -> dict[str, str]:
        if not isinstance(plan, Mapping) or set(plan) != {"commit", "main_before", "lab_before", "timestamp"}:
            raise PromotionError("promotion_plan_invalid")
        result = dict(plan)
        if any(not isinstance(value, str) for value in result.values()):
            raise PromotionError("promotion_plan_invalid")
        if any(not _OBJECT_ID.fullmatch(result[key]) for key in ("commit", "main_before", "lab_before")):
            raise PromotionError("promotion_plan_invalid")
        if cls._timestamp(result["timestamp"]) != result["timestamp"]:
            raise PromotionError("promotion_plan_invalid")
        return result
