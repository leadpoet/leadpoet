"""Apply only the reviewed completed-empty-domain negative fix to frozen79cd.

Unknown or mixed source generations fail closed. No current grader is read.
"""
from __future__ import annotations

import ast
import hashlib

CORRECTION_ID = "completed_negative_empty_observed_domain_v1"
FINALITY_FILE = "qualification/scoring/lead_scorer.py"
PREDICATE = "_has_explicitly_unproven_fit_dimensions"
ORIGINAL_SHA256 = '125c0303aa5e100df746499bc1ca76dc8e381b2bda8f3a1ad5132d688df15d58'
REPAIRED_SHA256 = '105f2108de3ff36f966764dcd8cbe5940c1d0fde0440dab4669906ad9b80daa2'
_ORIGINAL_GUARD = '''                    and _is_bound_homepage_name_conflict(identity_receipt)
'''
_CORRECTED_GUARD = '''                    and (
                        _is_bound_homepage_name_conflict(identity_receipt)
                        or (
                            # An explicit absent website is not identity proof.
                            # A completed targeted negative can exhaust it;
                            # missing, malformed or failed observations cannot.
                            isinstance(identity_receipt, Mapping)
                            and identity_receipt.get("decision") == COMPANY_FIT_UNAVAILABLE
                            and identity_receipt.get("reason_code") == "identity_not_proven"
                            and identity_receipt.get("evidence_source") == "company_web_reverification"
                            and verdict.get("observed_company_website") == ""
                            and identity_receipt.get("observed_domain") == ""
                            and all(
                                isinstance(identity_receipt.get(field), str)
                                and bool(identity_receipt[field].strip())
                                for field in (
                                    "submitted_name", "submitted_domain", "observed_name",
                                )
                            )
                            and all(
                                isinstance(identity_receipt.get(field), str)
                                and (
                                    identity_receipt[field] == ""
                                    or bool(identity_receipt[field].strip())
                                )
                                for field in (
                                    "submitted_linkedin_slug", "observed_linkedin_slug",
                                )
                            )
                        )
                    )
'''


class EmptyDomainFinalitySeamError(ValueError):
    """The source is not the exact reviewed frozen generation."""


def convert_empty_domain_finality_seam(source: bytes) -> bytes:
    digest = hashlib.sha256(source).hexdigest()
    if digest == REPAIRED_SHA256:
        return source
    if digest != ORIGINAL_SHA256:
        raise EmptyDomainFinalitySeamError("unrecognized or partial frozen empty-domain finality source")
    text = source.decode("utf-8")
    module = ast.parse(text)
    predicates = [node for node in module.body
                  if isinstance(node, ast.FunctionDef) and node.name == PREDICATE]
    if len(predicates) != 1:
        raise EmptyDomainFinalitySeamError("finality predicate is missing or ambiguous")
    predicate = ast.get_source_segment(text, predicates[0])
    if predicate is None or predicate.count(_ORIGINAL_GUARD) != 1:
        raise EmptyDomainFinalitySeamError("frozen finality guard differs")
    corrected = predicate.replace(_ORIGINAL_GUARD, _CORRECTED_GUARD, 1)
    result = text.replace(predicate, corrected, 1).encode("utf-8")
    compile(result, "frozen_empty_domain_finality.py", "exec")
    if hashlib.sha256(result).hexdigest() != REPAIRED_SHA256:
        raise EmptyDomainFinalitySeamError("empty-domain finality repair differs from reviewed source")
    return result
