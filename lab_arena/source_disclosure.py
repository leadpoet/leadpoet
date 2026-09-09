"""Publish evaluated submissions as bounded, read-only source previews."""

from __future__ import annotations

import io
import tarfile
from datetime import datetime, timezone
from pathlib import PurePosixPath
from typing import Any, Mapping
from urllib.parse import quote

from lab_arena import icp_disclosure, source_bundle

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


def disclosure_status(
    submission: Mapping[str, Any], now: datetime, *,
    round_row: Mapping[str, Any] | None = None,
) -> dict:
    """Only a completed, published evaluation releases its frozen source.

    A late Day 0 submission can be released on Day 1 without waiting another
    24 hours. Upload age alone never releases an unscored submission.
    """
    row = round_row or {}
    publication = row.get("publication_doc") or {}
    submission_id = str(submission.get("submission_id") or "")
    participant_ids = {
        str(item.get("submission_id") or "")
        for item in publication.get("participants") or []
        if isinstance(item, Mapping)
    }
    available_at = _timestamp(row.get("published_at")) if row.get("status") == "published" else None
    metadata = icp_disclosure.disclosure_metadata(row)
    public_at = _timestamp(metadata.get("public_at")) if metadata else None
    # Historical rounds evaluated on their bank's creation day. Their source
    # becomes eligible on the new next-day boundary, not permanently private.
    if row.get("icp_set_date") is None and available_at is not None and public_at is not None:
        available_at = max(available_at, public_at)
    allowed = (
        submission.get("status") == "frozen"
        and submission.get("round_id") == row.get("round_id")
        and submission_id in participant_ids
        and (submission.get("consent") or {}).get("public_rerun") is True
        and bool(submission.get("source_ref"))
        and available_at is not None
        and public_at is not None
        and available_at >= public_at
        and now.tzinfo is not None
        and now.astimezone(timezone.utc) >= available_at
    )
    return {
        "available": bool(allowed),
        "available_at": available_at.isoformat().replace("+00:00", "Z") if available_at else None,
        "url": "/arena/v1/submissions/%s/code" % quote(submission_id, safe="") if allowed else None,
    }


def public_source_code(
    objects: Any, submission: Mapping[str, Any], now: datetime, *,
    round_row: Mapping[str, Any] | None = None,
) -> dict:
    """Read validated source as inert text. Never extract, import, or run it."""
    if not disclosure_status(submission, now, round_row=round_row)["available"]:
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
