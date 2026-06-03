"""Company-name normalization for cross-cycle / cross-chain dedup."""

import re

_COMPANY_SUFFIX_RE = re.compile(
    r"\b(inc\.?|llc|ltd\.?|corp\.?|corporation|co\.?|company|"
    r"plc|gmbh|ag|sa|sas|srl|bv|nv|pty|pvt)\b",
    flags=re.IGNORECASE,
)
_TRAILING_PUNCT_RE = re.compile(r"[,.\s]+$")
_WHITESPACE_RUN_RE = re.compile(r"\s+")


def normalize_company(name: str) -> str:
    if not name:
        return ""
    s = name.lower().strip()
    s = _COMPANY_SUFFIX_RE.sub("", s)
    s = _TRAILING_PUNCT_RE.sub("", s)
    s = _WHITESPACE_RUN_RE.sub(" ", s).strip()
    return s
