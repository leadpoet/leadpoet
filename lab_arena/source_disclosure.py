"""Time-limited source privacy and a bounded, read-only code preview."""

from __future__ import annotations

import io
import tarfile
from datetime import datetime, timedelta, timezone
from pathlib import PurePosixPath
from typing import Any, Mapping
from urllib.parse import quote

from lab_arena import source_bundle

DISCLOSURE_DELAY = timedelta(hours=24)
MAX_PREVIEW_BYTES = 2 * 1024 * 1024
MAX_PREVIEW_FILE_BYTES = 256 * 1024
MAX_PREVIEW_FILES = 100
TEXT_EXTENSIONS = frozenset({
    ".py", ".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx", ".json",
    ".toml", ".yaml", ".yml", ".md", ".txt", ".sh", ".ini", ".cfg",
})


class SourceDisclosureError(ValueError):
    def __init__(self, code: str, status: int = 403) -> None:
        self.code = code
        self.status = status
        super().__init__(code)


def _timestamp(value: Any) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            return None
        return parsed.astimezone(timezone.utc)
    except (ValueError, TypeError):
        return None


def disclosure_status(submission: Mapping[str, Any], now: datetime) -> dict:
    """Do not use the upload reservation time as the submission time.

    Old frozen rows have no acceptance timestamp. Their freeze time is a
    conservative upper bound; old accepted rows retain their last update time.
    Missing or malformed timestamps keep the source private.
    """
    submitted = _timestamp(submission.get("accepted_at"))
    if submitted is None:
        if submission.get("status") == "frozen":
            submitted = _timestamp(submission.get("frozen_at"))
        elif submission.get("status") == "accepted":
            submitted = _timestamp(submission.get("updated_at"))
    available_at = submitted + DISCLOSURE_DELAY if submitted is not None else None
    allowed = (
        submission.get("status") in ("accepted", "frozen")
        and (submission.get("consent") or {}).get("public_rerun") is True
        and bool(submission.get("source_ref"))
        and available_at is not None
        and now.tzinfo is not None
        and now.astimezone(timezone.utc) >= available_at
    )
    submission_id = str(submission.get("submission_id") or "")
    return {
        "available": bool(allowed),
        "available_at": available_at.isoformat().replace("+00:00", "Z") if available_at else None,
        "url": "/arena/v1/submissions/%s/code" % quote(submission_id, safe="") if allowed else None,
    }


def public_source_code(objects: Any, submission: Mapping[str, Any], now: datetime) -> dict:
    """Read validated source as inert text. Never extract, import, or run it."""
    if not disclosure_status(submission, now)["available"]:
        raise SourceDisclosureError("source_not_public")
    payload = objects.get_bounded(
        str(submission["source_ref"]), source_bundle.MAX_SOURCE_ARCHIVE_BYTES
    )
    if len(payload) != int(submission.get("source_size_bytes") or 0):
        raise SourceDisclosureError("source_unavailable", 503)
    try:
        facts = source_bundle.validate_source_archive(payload)
        prefix = str(facts["source_root"])
        prefix = prefix + "/" if prefix else ""
        files = []
        total = 0
        truncated = False
        with tarfile.open(fileobj=io.BytesIO(payload), mode="r|gz") as archive:
            for member in archive:
                if not member.isfile():
                    continue
                path = member.name.removeprefix(prefix)
                name = PurePosixPath(path)
                # Environment templates and arbitrary binary/data files are
                # not code. Keep the preview narrow even after release.
                if name.name.startswith(".env") or (
                    name.suffix.lower() not in TEXT_EXTENSIONS
                    and name.name not in ("Dockerfile", "Makefile", "LICENSE")
                ):
                    truncated = True
                    continue
                if (member.size > MAX_PREVIEW_FILE_BYTES
                    or total + member.size > MAX_PREVIEW_BYTES
                    or len(files) >= MAX_PREVIEW_FILES):
                    truncated = True
                    continue
                handle = archive.extractfile(member)
                if handle is None:
                    raise SourceDisclosureError("source_unavailable", 503)
                content = handle.read(MAX_PREVIEW_FILE_BYTES + 1)
                if len(content) != member.size:
                    raise SourceDisclosureError("source_unavailable", 503)
                try:
                    text = content.decode("utf-8")
                except UnicodeDecodeError:
                    truncated = True
                    continue
                if "\x00" in text:
                    truncated = True
                    continue
                files.append({"path": path, "content": text})
                total += len(content)
    except (source_bundle.SourceBundleError, tarfile.TarError, EOFError, OSError) as exc:
        raise SourceDisclosureError("source_unavailable", 503) from exc
    files.sort(key=lambda file: (file["path"] != "harness.py", file["path"]))
    return {"submission_id": submission["submission_id"], "files": files, "truncated": truncated}
