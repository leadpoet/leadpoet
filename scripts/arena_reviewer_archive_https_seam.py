"""Upgrade only a proved Wayback snapshot locator's transport in frozen code."""
from __future__ import annotations

import ast
import hashlib

from scripts.arena_reviewer_investigation_finality_seam import _named_bindings

CORRECTION_ID = "wayback_snapshot_https_transport_v1"
ORIGINAL_SHA256 = "eaf0f25773688dcc49679a68a60f2f7b295a5308932e63d989f9103b6a11e4f7"
REPAIRED_SHA256 = "2b6101f02ef5d9badc97b7f52379e83feb4310ed60faa526ea823bba506e4f23"
ANCHOR = '            r = await cli.get(snap["url"])\n'
REPLACEMENT = '''            snapshot_url = str(snap["url"])
            snapshot_parts = urlsplit(snapshot_url)
            timestamp = snap.get("timestamp")
            if (
                snapshot_parts.scheme == "http"
                and snapshot_parts.netloc == "web.archive.org"
                and "#" not in snapshot_url
                and snap.get("available") is True
                and snap.get("status") == "200"
                and isinstance(timestamp, str)
                and re.fullmatch(r"[0-9]{14}", timestamp)
                and re.match(r"/web/" + timestamp + r"(?:[a-z_]+)?/", snapshot_parts.path)
                and (archived_original := _wayback_original_url(snapshot_url))
                and _normalize_url(archived_original) == _normalize_url(url)
            ):
                # Wayback publishes HTTP locators for its own HTTPS service.
                # Preserve the exact source and timestamp; only its transport
                # scheme changes. Other locators retain existing URL guards.
                from ipaddress import ip_address
                from qualification.competition_models import public_http_url

                public_http_url(archived_original)
                try:
                    ip_address(urlsplit(archived_original).hostname or "")
                except ValueError:
                    snapshot_url = "https" + snapshot_url[4:]
            r = await cli.get(snapshot_url)
'''


class ArchiveHttpsSeamError(ValueError):
    """The complete frozen archive function is not a reviewed version."""


def convert_archive_https_seam(source: bytes) -> bytes:
    try:
        text = source.decode("utf-8")
        module = ast.parse(text)
    except (UnicodeDecodeError, SyntaxError) as exc:
        raise ArchiveHttpsSeamError("archive source must be UTF-8 Python") from exc
    if "\r" in text:
        raise ArchiveHttpsSeamError("archive source must use LF newlines")
    nodes = [node for node in module.body if isinstance(node, ast.AsyncFunctionDef)
             and node.name == "_try_wayback"]
    if len(nodes) != 1 or _named_bindings(module, "_try_wayback") != nodes:
        raise ArchiveHttpsSeamError("archive function binding is missing or ambiguous")
    node = nodes[0]
    if node.decorator_list:
        raise ArchiveHttpsSeamError("archive function must not be decorated")
    lines = text.splitlines(keepends=True)
    body = "".join(lines[node.lineno - 1:node.end_lineno])
    digest = hashlib.sha256(body.encode()).hexdigest()
    if digest == REPAIRED_SHA256:
        return source
    if digest != ORIGINAL_SHA256 or body.count(ANCHOR) != 1:
        raise ArchiveHttpsSeamError("unrecognized archive function source")
    repaired = body.replace(ANCHOR, REPLACEMENT, 1)
    if hashlib.sha256(repaired.encode()).hexdigest() != REPAIRED_SHA256:
        raise ArchiveHttpsSeamError("archive transport repair differs from reviewed source")
    lines[node.lineno - 1:node.end_lineno] = [repaired]
    result = "".join(lines).encode()
    convert_archive_https_seam(result)
    return result
