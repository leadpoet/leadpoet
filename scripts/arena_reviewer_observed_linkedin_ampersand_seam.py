"""Compose only the reviewed observed ampersand locator support.

All replacements are exact AST-owned functions/imports plus one exact helper.
Unknown or mixed versions fail closed. No current grading module is imported.
"""
from __future__ import annotations

import ast

from scripts.arena_reviewer_investigation_finality_seam import _named_bindings

CORRECTION_ID = "observed_linkedin_ampersand_source_locator_v1"
FIT = "qualification/scoring/company_fit_decision.py"
HOME = "qualification/scoring/company_verification.py"
SIZE = "qualification/scoring/linkedin_company_size.py"
HELPER = "observed_linkedin_ampersand_slug"


class ObservedLinkedinAmpersandSeamError(ValueError):
    """Frozen source differs from the exact reviewed locator correction."""


OLD_FUNCTION_0 = r'''def _linkedin_slug(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    if "://" not in raw:
        raw = f"https://{raw}"
    try:
        parsed = urlsplit(raw)
    except ValueError:
        return ""
    host = (parsed.hostname or "").casefold().removeprefix("www.")
    parts = [part for part in parsed.path.split("/") if part]
    if (
        not (host == "linkedin.com" or host.endswith(".linkedin.com"))
        or len(parts) < 2
        or parts[0].casefold() != "company"
    ):
        return ""
    slug = parts[1].casefold()
    return slug if _LINKEDIN_SLUG_RE.fullmatch(slug) else ""
'''

NEW_FUNCTION_0 = r'''def _linkedin_slug(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    if "://" not in raw:
        raw = f"https://{raw}"
    try:
        parsed = urlsplit(raw)
    except ValueError:
        return ""
    host = (parsed.hostname or "").casefold().removeprefix("www.")
    parts = [part for part in parsed.path.split("/") if part]
    if (
        not (host == "linkedin.com" or host.endswith(".linkedin.com"))
        or len(parts) < 2
        or parts[0].casefold() != "company"
    ):
        return ""
    slug = parts[1].casefold()
    ampersand_slug = observed_linkedin_ampersand_slug(slug)
    if ampersand_slug:
        return ampersand_slug
    return slug if _LINKEDIN_SLUG_RE.fullmatch(slug) else ""
'''

OLD_FUNCTION_1 = r'''def _canonical_linkedin_company_url(value: str) -> str:
    """Return a direct, canonical LinkedIn company URL or an empty string."""

    raw = str(value or "").strip()
    if raw.startswith("//"):
        raw = f"https:{raw}"
    try:
        parsed = urlsplit(raw)
    except (TypeError, ValueError):
        return ""
    host = str(parsed.hostname or "").casefold().removeprefix("www.")
    parts = [part for part in parsed.path.split("/") if part]
    if (
        parsed.scheme.casefold() not in {"http", "https"}
        or parsed.username is not None
        or parsed.password is not None
        or not (host == "linkedin.com" or host.endswith(".linkedin.com"))
        or len(parts) < 2
        or parts[0].casefold() != "company"
    ):
        return ""
    slug = parts[1].casefold()
    if not re.fullmatch(r"[a-z0-9][a-z0-9._%+-]{0,99}", slug):
        return ""
    return f"https://www.linkedin.com/company/{slug}"
'''

NEW_FUNCTION_1 = r'''def _canonical_linkedin_company_url(value: str) -> str:
    """Return a direct, canonical LinkedIn company URL or an empty string."""

    raw = str(value or "").strip()
    if raw.startswith("//"):
        raw = f"https:{raw}"
    try:
        parsed = urlsplit(raw)
    except (TypeError, ValueError):
        return ""
    host = str(parsed.hostname or "").casefold().removeprefix("www.")
    parts = [part for part in parsed.path.split("/") if part]
    if (
        parsed.scheme.casefold() not in {"http", "https"}
        or parsed.username is not None
        or parsed.password is not None
        or not (host == "linkedin.com" or host.endswith(".linkedin.com"))
        or len(parts) < 2
        or parts[0].casefold() != "company"
    ):
        return ""
    slug = parts[1].casefold()
    ampersand_slug = observed_linkedin_ampersand_slug(slug)
    if ampersand_slug:
        return f"https://www.linkedin.com/company/{ampersand_slug}"
    if not re.fullmatch(r"[a-z0-9][a-z0-9._%+-]{0,99}", slug):
        return ""
    return f"https://www.linkedin.com/company/{slug}"
'''

OLD_FUNCTION_2 = r'''def linkedin_company_page_slug(value: Any) -> str:
    """Return one strict LinkedIn company slug from an HTTP(S) profile URL."""

    try:
        slug = candidate_linkedin_prompt_slug(
            value,
            "employee_size_evidence_url",
        )
        parsed = urlsplit(str(value))
    except (TypeError, ValueError):
        return ""
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) != 2 or parts[0].casefold() != "company":
        return ""
    return slug
'''

NEW_FUNCTION_2 = r'''def linkedin_company_page_slug(value: Any) -> str:
    """Return one strict LinkedIn company slug from an HTTP(S) profile URL."""

    try:
        slug = candidate_linkedin_prompt_slug(
            value,
            "employee_size_evidence_url",
        )
        parsed = urlsplit(str(value))
    except (TypeError, ValueError):
        # An independently observed company locator may contain an ampersand.
        # Keep the miner prompt grammar unchanged; validate the full URL before
        # admitting this one separately bounded, single-decoded component.
        try:
            canonical_candidate_prompt_url(value, "employee_size_evidence_url")
            parsed = urlsplit(value)
            host = str(parsed.hostname or "").casefold()
            parts = [part for part in parsed.path.split("/") if part]
            if (
                not (host == "linkedin.com" or host.endswith(".linkedin.com"))
                or len(parts) != 2 or parts[0].casefold() != "company"
            ):
                return ""
            slug = observed_linkedin_ampersand_slug(parts[1])
            if not slug:
                return ""
        except (TypeError, ValueError):
            return ""
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) != 2 or parts[0].casefold() != "company":
        return ""
    return slug
'''

NEW_HELPER = r'''def observed_linkedin_ampersand_slug(value: str) -> str:
    """Canonicalize one observed path component, without miner admission."""

    slug = unquote(value).casefold()
    if (
        "&" not in slug
        or re.fullmatch(r"[a-z0-9][a-z0-9_.&-]{0,99}", slug, re.ASCII) is None
    ):
        return ""
    return quote(slug, safe="._-")
'''

OLD_IMPORT_0 = r'''from urllib.parse import urlsplit
'''

NEW_IMPORT_0 = r'''from urllib.parse import quote, unquote, urlsplit
'''

OLD_IMPORT_1 = r'''from qualification.scoring.company_fit_decision import (
    CompanyFitDecisionResult,
    company_fit_match,
    company_fit_mismatch,
    company_fit_unavailable,
    evaluate_company_identity,
    _company_name,
)
'''

NEW_IMPORT_1 = r'''from qualification.scoring.company_fit_decision import (
    CompanyFitDecisionResult,
    company_fit_match,
    company_fit_mismatch,
    company_fit_unavailable,
    evaluate_company_identity,
    observed_linkedin_ampersand_slug,
    _company_name,
)
'''

OLD_IMPORT_2 = r'''from gateway.qualification.models import candidate_linkedin_prompt_slug
'''

NEW_IMPORT_2 = r'''from gateway.qualification.models import (
    candidate_linkedin_prompt_slug, canonical_candidate_prompt_url,
)
'''

OLD_IMPORT_3 = r'''from qualification.scoring.company_fit_decision import (
    _company_name, verified_homepage_brand_name,
)
'''

NEW_IMPORT_3 = r'''from qualification.scoring.company_fit_decision import (
    _company_name, observed_linkedin_ampersand_slug, verified_homepage_brand_name,
)
'''


FUNCTIONS = {
    FIT: ("_linkedin_slug", OLD_FUNCTION_0, NEW_FUNCTION_0),
    HOME: ("_canonical_linkedin_company_url", OLD_FUNCTION_1, NEW_FUNCTION_1),
    SIZE: ("linkedin_company_page_slug", OLD_FUNCTION_2, NEW_FUNCTION_2),
}
IMPORTS = {
    FIT: (("urllib.parse", OLD_IMPORT_0, NEW_IMPORT_0),),
    HOME: (("qualification.scoring.company_fit_decision", OLD_IMPORT_1, NEW_IMPORT_1),),
    SIZE: (("gateway.qualification.models", OLD_IMPORT_2, NEW_IMPORT_2),
           ("qualification.scoring.company_fit_decision", OLD_IMPORT_3, NEW_IMPORT_3)),
}
INTRODUCED_IMPORTS = {FIT: ("quote", "unquote"), HOME: (HELPER,),
                      SIZE: (HELPER, "canonical_candidate_prompt_url")}


def _source_node(lines: list[str], node: ast.AST) -> str:
    return "".join(lines[node.lineno - 1:node.end_lineno])


def _convert_file(path: str, source: bytes) -> tuple[bytes, set[str]]:
    try:
        text = source.decode("utf-8")
        module = ast.parse(text)
    except (UnicodeDecodeError, SyntaxError) as exc:
        raise ObservedLinkedinAmpersandSeamError("locator source must be UTF-8 Python") from exc
    if "\r" in text:
        raise ObservedLinkedinAmpersandSeamError("locator source must use LF newlines")
    lines = text.splitlines(keepends=True)
    name, old, new = FUNCTIONS[path]
    nodes = [node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == name]
    if (len(nodes) != 1 or _named_bindings(module, name) != nodes
            or nodes[0].decorator_list):
        raise ObservedLinkedinAmpersandSeamError("locator function binding is missing or ambiguous")
    function = nodes[0]
    states, replacements = set(), []
    helper_missing = False
    if path == FIT:
        helpers = [node for node in module.body if isinstance(node, ast.FunctionDef)
                   and node.name == HELPER]
        if _named_bindings(module, HELPER) != helpers or len(helpers) > 1:
            raise ObservedLinkedinAmpersandSeamError("observed helper binding is ambiguous")
        helper_missing = not helpers
        states.add("original" if helper_missing else "corrected")
        if helpers and (
            helpers[0].decorator_list or _source_node(lines, helpers[0]) != NEW_HELPER
            or "".join(lines[helpers[0].end_lineno:function.lineno - 1]) != "\n\n"
        ):
            raise ObservedLinkedinAmpersandSeamError("unrecognized observed helper or insertion point")
    for node, original, corrected in [(function, old, new)]:
        block = _source_node(lines, node)
        if block == original:
            states.add("original")
            replacement = (NEW_HELPER + "\n\n" if helper_missing else "") + corrected
            replacements.append((node.lineno - 1, node.end_lineno, replacement))
        elif block == corrected:
            states.add("corrected")
        else:
            raise ObservedLinkedinAmpersandSeamError("unrecognized locator function source")
    for module_name, original, corrected in IMPORTS[path]:
        imports = [node for node in module.body if isinstance(node, ast.ImportFrom)
                   and node.module == module_name]
        if len(imports) != 1:
            raise ObservedLinkedinAmpersandSeamError("locator import is missing or ambiguous")
        node = imports[0]
        block = _source_node(lines, node)
        if block == original:
            states.add("original")
            replacements.append((node.lineno - 1, node.end_lineno, corrected))
        elif block == corrected:
            states.add("corrected")
        else:
            raise ObservedLinkedinAmpersandSeamError("unrecognized locator import")
    for imported_name in INTRODUCED_IMPORTS[path]:
        expected = [alias for node in module.body if isinstance(node, ast.ImportFrom)
                    for alias in node.names if alias.name == imported_name and alias.asname is None]
        if _named_bindings(module, imported_name) != expected or len(expected) > 1:
            raise ObservedLinkedinAmpersandSeamError("locator dependency binding is ambiguous")
    if len(states) != 1:
        raise ObservedLinkedinAmpersandSeamError("partially corrected locator file")
    if states == {"corrected"}:
        return source, states
    for start, end, replacement in sorted(replacements, reverse=True):
        lines[start:end] = [replacement]
    result = "".join(lines).encode("utf-8")
    compile(result, "frozen_observed_linkedin_ampersand.py", "exec")
    return result, states


def convert_observed_linkedin_ampersand_files(files: dict[str, bytes]) -> dict[str, bytes]:
    if set(files) != set(FUNCTIONS):
        raise ObservedLinkedinAmpersandSeamError("exact locator source files are required")
    converted, states = {}, set()
    for path, source in files.items():
        corrected, state = _convert_file(path, source)
        converted[path] = corrected
        states.update(state)
    if len(states) != 1:
        raise ObservedLinkedinAmpersandSeamError("mixed original and corrected locator files")
    if states == {"original"}:
        if convert_observed_linkedin_ampersand_files(converted) != converted:
            raise ObservedLinkedinAmpersandSeamError("locator correction is not idempotent")
    return converted

