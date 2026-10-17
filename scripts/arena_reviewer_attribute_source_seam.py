"""Include a failed attribute source in one already planned bounded review.

Only target selection and its semantic-target mapping change. Evidence checks,
qualification, finality, research limits and all other frozen bytes stay intact.
"""

from __future__ import annotations

import ast


CORRECTION_ID = "failed_attribute_source_shared_review_v1"


class AttributeSourceSeamError(ValueError):
    """The frozen source does not contain the reviewed selection shape."""


_SELECTOR_OLD = '''            targets.append("geography")
    return tuple(targets)
'''
_SELECTOR_NEW = '''            targets.append("geography")
    grounding = details.get("required_attribute_grounding")
    if (
        targets
        and details.get("required_attribute_decision") == COMPANY_FIT_UNAVAILABLE
        and isinstance(grounding, Mapping)
        and grounding.get("status") == "source_unavailable"
        and grounding.get(VERIFIER_FAILURE_DETAIL_KEY)
        in {SOURCE_BLOCKED_FAILURE_REASON, PROVIDER_ERROR_FAILURE_REASON}
    ):
        # Share the already planned review and its unchanged research limits.
        # A failed source is still a failure until independent review resolves it.
        targets.append("required_attribute")
    return tuple(targets)
'''
_MAPPING_OLD = '''            "industry": "industry",
            "geography": "geography",
        }[target]
        for target in investigation_targets
'''
_MAPPING_NEW = '''            "industry": "industry",
            "geography": "geography",
            "required_attribute": "required_attribute",
        }[target]
        for target in investigation_targets
'''
DELTAS = (
    ("_targeted_company_investigation_dimensions", _SELECTOR_OLD, _SELECTOR_NEW),
    ("_positive_semantic_review_needed", _MAPPING_OLD, _MAPPING_NEW),
)


def convert_attribute_source_seam(source: bytes) -> bytes:
    try:
        text = source.decode("utf-8")
        module = ast.parse(text)
    except (UnicodeDecodeError, SyntaxError) as exc:
        raise AttributeSourceSeamError("grading source must be valid UTF-8 Python") from exc
    if "\r" in text:
        raise AttributeSourceSeamError("grading source must use LF newlines")
    lines = text.splitlines(keepends=True)
    replacements = {}
    states = set()
    for name, old, new in DELTAS:
        nodes = [node for node in module.body
                 if isinstance(node, ast.FunctionDef) and node.name == name]
        if len(nodes) != 1:
            raise AttributeSourceSeamError("correction function is missing or ambiguous")
        node = nodes[0]
        original = "".join(lines[node.lineno - 1:node.end_lineno])
        counts = original.count(old.rstrip("\n")), original.count(new.rstrip("\n"))
        if counts == (1, 0):
            states.add("original")
            replacements[(node.lineno - 1, node.end_lineno)] = original.replace(
                old.rstrip("\n"), new.rstrip("\n"), 1,
            )
        elif counts == (0, 1):
            states.add("corrected")
        else:
            raise AttributeSourceSeamError("unrecognized or partial correction block")
    if len(states) != 1:
        raise AttributeSourceSeamError("partially corrected grading source")
    for start, stop in sorted(replacements, reverse=True):
        lines[start:stop] = [replacements[(start, stop)]]
    result = "".join(lines)
    compile(result, "frozen_attribute_source.py", "exec")
    return result.encode("utf-8")
