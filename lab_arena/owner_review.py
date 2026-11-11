"""Bounded owner-only excerpts from a source-validated admission review."""

from collections.abc import Mapping

from lab_arena import code_review, contracts, training_capture

MAX_FINDINGS = 8
MAX_EVIDENCE_BYTES = 1000
MAX_FILE_BYTES = 512


def _bounded(text, maximum):
    return text.encode("utf-8")[:maximum].decode("utf-8", errors="ignore")


def from_validated_findings(findings, *, secrets=()):
    """Call only after parse_response has proved each excerpt in own source.

    Redact before slicing so a secret cannot become an unrecognized prefix.
    Never retain model explanations or another submission's comparison data.
    """
    result = []
    truncated = len(findings) > MAX_FINDINGS
    redacted = False
    for finding in findings[:MAX_FINDINGS]:
        value = {key: finding[key] for key in ("category", "file", "evidence")}
        cleaned, reasons = training_capture.sanitize_training_value(value, secrets)
        redacted = redacted or bool(reasons)
        # PostgreSQL JSONB cannot represent NUL. Omit this optional excerpt;
        # never let a source character prevent admission result settlement.
        if "\x00" in cleaned["file"] or "\x00" in cleaned["evidence"]:
            truncated = True
            continue
        # A path cannot be truncated without referring to a different file.
        if len(cleaned["file"].encode("utf-8")) > MAX_FILE_BYTES:
            truncated = True
            continue
        evidence = _bounded(cleaned["evidence"], MAX_EVIDENCE_BYTES)
        truncated = truncated or evidence != cleaned["evidence"]
        result.append({**cleaned, "evidence": evidence})
    document = {"findings": result, "truncated": truncated, "redacted": redacted}
    # JSON escapes can exceed the excerpt's UTF-8 size (for example controls).
    # Keep the existing SQL review document limit, even for adversarial source.
    while len(contracts.canonical_json(document).encode("utf-8")) > 16000:
        result.pop()
        document["truncated"] = True
    return document


def private_projection(row):
    """Fail closed for absent, historical, malformed, or nonfinal findings."""
    unavailable = {"available": False, "findings": [], "truncated": False, "redacted": False}
    if row.get("code_review_status") != "rejected":
        return unavailable
    document = row.get("code_review_doc")
    value = document.get("owner_findings") if isinstance(document, Mapping) else None
    categories = document.get("categories") if isinstance(document, Mapping) else None
    if (not isinstance(categories, list) or not all(
            isinstance(category, str) and category in code_review.REVIEW_CATEGORIES
            for category in categories)):
        return unavailable
    if (not isinstance(value, Mapping) or set(value) != {"findings", "truncated", "redacted"}
            or type(value["truncated"]) is not bool or type(value["redacted"]) is not bool
            or not isinstance(value["findings"], list) or len(value["findings"]) > MAX_FINDINGS):
        return unavailable
    for finding in value["findings"]:
        if (not isinstance(finding, Mapping) or set(finding) != {"category", "file", "evidence"}
                or not isinstance(finding["category"], str)
                or finding["category"] not in code_review.REVIEW_CATEGORIES
                or finding["category"] not in categories
                or not isinstance(finding["file"], str) or not isinstance(finding["evidence"], str)):
            return unavailable
        try:
            if not (1 <= len(finding["file"].encode("utf-8")) <= MAX_FILE_BYTES
                    and 1 <= len(finding["evidence"].encode("utf-8")) <= MAX_EVIDENCE_BYTES
                    and "\x00" not in finding["file"] and "\x00" not in finding["evidence"]):
                return unavailable
        except UnicodeError:
            return unavailable
    try:
        if len(contracts.canonical_json(value).encode("utf-8")) > 20000:
            return unavailable
        cleaned, reasons = training_capture.sanitize_training_value(dict(value))
    except (ValueError, TypeError, UnicodeError):
        return unavailable
    # Sanitization can expand a short credential into a longer marker.
    cleaned["findings"] = [
        {**finding, "evidence": _bounded(finding["evidence"], MAX_EVIDENCE_BYTES)}
        for finding in cleaned["findings"]
    ]
    cleaned["redacted"] = cleaned["redacted"] or bool(reasons)
    return {"available": True, **cleaned}
