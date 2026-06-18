"""Open static gaming pattern checks seeded from model_evaluation_rules.md."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import List, Tuple


@dataclass(frozen=True)
class StaticCheckResult:
    passed: bool
    confidence: int
    red_flags: Tuple[str, ...]


PAYLOAD_INJECTION_PATTERNS = [
    r"tmp\s*=\s*decode\s*\([^)]+\)",
    r"[\"']description[\"']\s*:\s*[a-z_]+\s*\+",
    r"[\"']snippet[\"']\s*:\s*[a-z_]+\s*\+",
    r"decode\s*\([^)]+\)[^}]*[\"']description[\"']\s*:",
]

HIDDEN_CRYPTO_PATTERNS = [
    r"decode\s*\(\s*[a-z_]+\s*,\s*[\"'][^\"']+[\"']\s*\)",
    r"def\s+derive_key\s*\([^)]*secret",
    r"def\s+keystream\s*\(",
    r"bytes\s*\(\s*a\s*\^\s*b\s+for\s+a\s*,\s*b\s+in\s+zip",
]

DATA_FABRICATION_PATTERNS = [
    r"timedelta\s*\(\s*days\s*=\s*random\.(randint|uniform|choice)",
    r"random\.(randint|uniform)\s*\(\s*\d+\s*,\s*\d+\s*\)[\s\S]{0,300}date\.today\s*\(\s*\)\s*-\s*timedelta",
    r"date\.today\s*\(\s*\)\s*-\s*timedelta\s*\(\s*days\s*=\s*\d{1,3}\s*\)",
    r"[\"']date[\"']\s*:[^\n]*\bor\b[^\n]*date\.today\s*\(\s*\)\s*-\s*timedelta",
]

GENERIC_INTENT_FALLBACK_PATTERNS = [
    r"f[\"'][^\"']*\{[^}]*company[^}]*\}[^\"']*is\s+actively\s+operating",
    r"f[\"'][^\"']*visible\s+market\s+activity",
    r"[\"']description[\"']\s*:\s*f[\"'][^\"']*\{[^}]*(company|business)[^}]*\}[^\"']*(?:operating|active|activity|expanding)",
    r"[\"']snippet[\"']\s*:\s*f[\"'][^\"']*(?:company\s+updates|market\s+activity|business\s+operations)",
]

SOURCE_TYPE_INFLATION_PATTERNS = [
    r"\(\s*f?[\"'][^\"']*(?:/careers|/jobs)[^\"']*[\"']\s*,\s*[\"']job_board[\"']\s*\)",
    r"(?:/careers|/jobs)[^}]*[\"']job_board[\"']",
]

ICP_ECHO_PATTERNS = [
    r"[\"']industry[\"']\s*:\s*(?:_norm\s*\()?ctx\.get\s*\(\s*[\"']industry[\"']\s*\)",
    r"[\"']sub_industry[\"']\s*:\s*(?:_norm\s*\()?ctx\.get\s*\(\s*[\"']sub_industry[\"']\s*\)",
    r"_norm\s*\(\s*ctx\.get\s*\([^)]+\)\s*\)\s*or\s*_norm\s*\(\s*lead\.get",
]

EVIDENCE_MANIPULATION_PATTERNS = [
    r"(?:[\"']no\s+specific[\"'].*[\"']no\s+(?:relevant|evidence|indication)[\"'])",
    r"re\.sub\s*\([^)]*(?:no\s+specific|no\s+evidence|no\s+indication|does\s+not\s+indicate|lacks\s+specific)",
    r"(?:_emit_block|_bare_neg|_no_ev|neg_phrases|block_phrases)\s*=\s*[\(\[]",
]

HARDCODED_INTENT_DEFAULTS = [
    r"""or\s+['"](?:hiring|funding|expansion)[\s,]+(?:hiring|funding|expansion)""",
    r"""['"].*hiring.*funding.*expansion.*['"]""",
    r"""\bor\s+['"]hiring\s+funding""",
]


def run_static_gaming_checks(code_content: str) -> StaticCheckResult:
    red_flags: List[str] = []
    confidence = 0

    injection_matches = _count_matches(PAYLOAD_INJECTION_PATTERNS, code_content)
    if injection_matches >= 2:
        red_flags.append("payload injection mechanism detected")
        confidence = max(confidence, 90)

    crypto_matches = _count_matches(HIDDEN_CRYPTO_PATTERNS, code_content, dotall=True)
    if crypto_matches >= 2:
        red_flags.append("hidden crypto/obfuscation detected")
        confidence = max(confidence, 90)
    elif crypto_matches == 1:
        red_flags.append("suspicious crypto pattern")
        confidence = max(confidence, 50)

    fabrication_matches = _count_matches(DATA_FABRICATION_PATTERNS, code_content, dotall=True)
    if fabrication_matches >= 2:
        red_flags.append("date/data fabrication mechanism detected")
        confidence = max(confidence, 88)
    elif fabrication_matches == 1:
        red_flags.append("potential date/data fabrication")
        confidence = max(confidence, 70)

    generic_matches = _count_matches(GENERIC_INTENT_FALLBACK_PATTERNS, code_content, dotall=True)
    if generic_matches >= 3:
        red_flags.append("generic intent fallback templates detected")
        confidence = max(confidence, 75)
    elif generic_matches >= 1:
        red_flags.append("potential generic intent fallback")
        confidence = max(confidence, 35)

    source_inflation_matches = _count_matches(SOURCE_TYPE_INFLATION_PATTERNS, code_content, dotall=True)
    if source_inflation_matches:
        red_flags.append("source type inflation for /careers or /jobs page")
        confidence = max(confidence, 65)

    icp_echo_matches = _count_matches(ICP_ECHO_PATTERNS, code_content, dotall=True)
    if icp_echo_matches:
        red_flags.append("ICP echo-back detected")
        confidence = max(confidence, 85)

    evidence_manip_matches = _count_matches(EVIDENCE_MANIPULATION_PATTERNS, code_content, dotall=True)
    if evidence_manip_matches >= 2:
        red_flags.append("negative evidence manipulation detected")
        confidence = max(confidence, 90)
    elif evidence_manip_matches == 1:
        red_flags.append("possible negative evidence manipulation")
        confidence = max(confidence, 65)

    intent_default_matches = _count_matches(HARDCODED_INTENT_DEFAULTS, code_content, dotall=True)
    if intent_default_matches >= 2:
        red_flags.append("hardcoded intent keyword defaults detected")
        confidence = max(confidence, 88)
    elif intent_default_matches == 1:
        red_flags.append("possible hardcoded intent keyword default")
        confidence = max(confidence, 60)

    return StaticCheckResult(
        passed=confidence < 85,
        confidence=confidence,
        red_flags=tuple(red_flags),
    )


def _count_matches(patterns: List[str], content: str, *, dotall: bool = False) -> int:
    flags = re.IGNORECASE | (re.DOTALL if dotall else 0)
    return sum(1 for pattern in patterns if re.search(pattern, content or "", flags))
