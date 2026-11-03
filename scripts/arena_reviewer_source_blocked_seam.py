"""Normalize one existing exact-profile diagnostic in frozen reviewer source."""

from scripts.arena_reviewer_source_access_seam import (
    SourceAccessSeamError, _convert, convert_source_access_seam,
)


CORRECTION_ID = "typed_linkedin_source_blocked_alias_v1"


class SourceBlockedSeamError(ValueError):
    """Frozen source does not have the reviewed diagnostic boundary."""


OLD = '''    elif raw_error == "http_410":
        error_class = "source_gone"
    elif raw_error.startswith("http_"):
'''
NEW = '''    elif raw_error == "http_410":
        error_class = "source_gone"
    elif raw_error == "source_blocked":
        # The exact-profile reader already distinguishes source-local blocks
        # from outer transport, malformed replies, and unbound crawl failures.
        # Reuse the canonical class recognized by both finality consumers.
        error_class = "source_access_denied"
    elif raw_error.startswith("http_"):
'''
DELTAS = (("_fetch_outcome", OLD, NEW),)


def convert_source_blocked_seam(source: bytes) -> bytes:
    try:
        # Reuse the existing reviewed canonical classification. An older base
        # can select --source-access-denied first; this option never adds it
        # implicitly or changes its API/authentication/error distinctions.
        if convert_source_access_seam(source) != source:
            raise SourceBlockedSeamError("source-blocked alias requires source-access correction")
        return _convert(source, DELTAS)
    except SourceAccessSeamError as exc:
        raise SourceBlockedSeamError("unrecognized source-blocked alias correction shape") from exc
