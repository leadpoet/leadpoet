"""Apply only the approved exact-source Exa access-denial correction.

Frozen source stays authoritative. No current grading module is copied.
Unknown, ambiguous and partially corrected source shapes fail closed.
"""

from __future__ import annotations

import ast


CORRECTION_ID = "exact_exa_source_access_denied_v1"


class SourceAccessSeamError(ValueError):
    """The frozen source does not have the reviewed correction shape."""


_EXA_OLD = '''            return {"ok": False, "error": "invalid_contents_status"}
        if any(
'''
_EXA_NEW = '''            return {"ok": False, "error": "invalid_contents_status"}
        # Exa authenticated the API request; this exact target denied access.
        # Do not confuse a crawl status with API auth, credits or an outage.
        if len(statuses) == 1 and body.get("results") == []:
            source = statuses[0]
            error = source.get("error")
            if (
                source.get("id") == canonical_url
                and source.get("status") == "error"
                and isinstance(error, Mapping)
                and type(error.get("httpStatusCode")) is int
                and error["httpStatusCode"] in {401, 403}
                and error.get("tag") == "CRAWL_HTTP_%d" % error["httpStatusCode"]
            ):
                return {"ok": False, "error": "source_access_denied"}
        if any(
'''
_OUTCOME_OLD = '''        "provider_request_refused", "provider_diagnostic_body",
    }:
'''
_OUTCOME_NEW = '''        "provider_request_refused", "provider_diagnostic_body",
        "source_access_denied",
    }:
'''
_INVESTIGATION_OLD = '''                                    "source_gone",
                                }
'''
_INVESTIGATION_NEW = '''                                    "source_gone", "source_access_denied",
                                }
'''
_TIMEOUT_OLD = '''                if primary_timeout is not None and not result.get("ok"):
                    raise primary_timeout
'''
_TIMEOUT_NEW = '''                if (
                    primary_timeout is not None and not result.get("ok")
                    and result.get("error") != "source_access_denied"
                ):
                    raise primary_timeout
'''
_FINALITY_OLD = '''                "provider_request_refused", "source_not_found", "source_gone",
            }
'''
_FINALITY_NEW = '''                "provider_request_refused", "source_not_found", "source_gone",
                "source_access_denied",
            }
'''
INVESTIGATOR_DELTAS = (
    ("_fetch_exa_exact_page", _EXA_OLD, _EXA_NEW),
    ("_fetch_outcome", _OUTCOME_OLD, _OUTCOME_NEW),
    ("investigate_company_evidence", _INVESTIGATION_OLD, _INVESTIGATION_NEW),
    ("investigate_company_evidence", _TIMEOUT_OLD, _TIMEOUT_NEW),
)
FINALITY_DELTAS = (("_has_explicitly_unproven_fit_dimensions", _FINALITY_OLD, _FINALITY_NEW),)


def _convert(source: bytes, deltas: tuple) -> bytes:
    try:
        text = source.decode("utf-8")
        module = ast.parse(text)
    except (UnicodeDecodeError, SyntaxError) as exc:
        raise SourceAccessSeamError("grading source must be valid UTF-8 Python") from exc
    if "\r" in text:
        raise SourceAccessSeamError("grading source must use LF newlines")
    lines = text.splitlines(keepends=True)
    replacements = {}
    states = set()
    for name, old, new in deltas:
        nodes = [node for node in module.body
                 if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                 and node.name == name]
        if len(nodes) != 1:
            raise SourceAccessSeamError("correction function is missing or ambiguous")
        node = nodes[0]
        original = "".join(lines[node.lineno - 1:node.end_lineno])
        counts = original.count(old), original.count(new)
        if counts == (1, 0):
            states.add("original")
            key = (node.lineno - 1, node.end_lineno)
            replacements[key] = replacements.get(key, original).replace(old, new, 1)
        elif counts == (0, 1):
            states.add("corrected")
        else:
            raise SourceAccessSeamError("unrecognized or partial correction block")
    if len(states) != 1:
        raise SourceAccessSeamError("partially corrected grading source")
    for start, stop in sorted(replacements, reverse=True):
        lines[start:stop] = [replacements[(start, stop)]]
    result = "".join(lines)
    compile(result, "frozen_source_access.py", "exec")
    return result.encode("utf-8")


def convert_source_access_seam(source: bytes) -> bytes:
    return _convert(source, INVESTIGATOR_DELTAS)


def convert_source_access_finality_seam(source: bytes) -> bytes:
    return _convert(source, FINALITY_DELTAS)
