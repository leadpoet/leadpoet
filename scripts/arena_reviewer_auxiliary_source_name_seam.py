"""Align the frozen optional auxiliary producer with its existing consumer."""
from __future__ import annotations

import ast
import hashlib

from scripts.arena_reviewer_investigation_finality_seam import _named_bindings

CORRECTION_ID = "first_party_auxiliary_exact_name_binding_v1"
ORIGINAL_SHA256 = "8b3f6c65a0a7392dcf1125eb52fd894479dfa3b09bdb452b78af8afa3a6c086e"
REPAIRED_SHA256 = "b89d7776c6cab928cdf9450b8a7a04def66c74523623a7639811b0b0498cf37c"
FUNCTION = "_matched_first_party_source_context"
ANCHOR = '    identity = verified_identity_receipt([company_fit.receipt("company_fit")])\n'
GUARD = '''    # The paragraph reviewer requires this exact name binding for auxiliaries.
    if (
        not isinstance(identity, Mapping)
        or _compact_company_name(company.company_name)
        != _compact_company_name(identity.get("observed_name"))
    ):
        return None
'''


class AuxiliarySourceNameSeamError(ValueError):
    """The frozen auxiliary producer is not an exact reviewed version."""


def convert_auxiliary_source_name_seam(source: bytes) -> bytes:
    try:
        text = source.decode("utf-8")
        module = ast.parse(text)
    except (UnicodeDecodeError, SyntaxError) as exc:
        raise AuxiliarySourceNameSeamError("auxiliary source must be UTF-8 Python") from exc
    if "\r" in text:
        raise AuxiliarySourceNameSeamError("auxiliary source must use LF newlines")
    nodes = [node for node in module.body if isinstance(node, ast.FunctionDef)
             and node.name == FUNCTION]
    if len(nodes) != 1 or _named_bindings(module, FUNCTION) != nodes:
        raise AuxiliarySourceNameSeamError("auxiliary function binding is missing or ambiguous")
    node = nodes[0]
    if node.decorator_list:
        raise AuxiliarySourceNameSeamError("auxiliary function must not be decorated")
    lines = text.splitlines(keepends=True)
    body = "".join(lines[node.lineno - 1:node.end_lineno])
    digest = hashlib.sha256(body.encode()).hexdigest()
    if digest == REPAIRED_SHA256:
        return source
    if digest != ORIGINAL_SHA256 or body.count(ANCHOR) != 1:
        raise AuxiliarySourceNameSeamError("unrecognized auxiliary function source")
    repaired = body.replace(ANCHOR, ANCHOR + GUARD, 1)
    if hashlib.sha256(repaired.encode()).hexdigest() != REPAIRED_SHA256:
        raise AuxiliarySourceNameSeamError("auxiliary repair differs from reviewed source")
    lines[node.lineno - 1:node.end_lineno] = [repaired]
    result = "".join(lines).encode()
    convert_auxiliary_source_name_seam(result)
    return result
