"""Apply only the reviewed identity-repair ordering to exact frozen b699.

No current grader is read. Unknown and mixed generations fail closed.
"""
from __future__ import annotations

import ast
import hashlib

CORRECTION_ID = "missing_identity_repair_before_investigation_v1"
FINALITY_FILE = "qualification/scoring/lead_scorer.py"
FROZEN_IMAGE_DIGEST = "sha256:b69931916a8ddfae0836876f20c162799c0b26b1e3b160389885f9d5e656361b"
ORIGINAL_SHA256 = "559e5283eb9fe971bee6dc10abd106d08d02e39b2a9a2ee6bdb8c4eefe9b9af9"
REPAIRED_SHA256 = "27e7a0d915c0e65642968ba2e4ee719d486042820ebe1e6ece7e1cafc03b161b"
FUNCTION = "_llm_reverify_company"
_ANCHOR = '''    claims: Mapping[str, Any] = {}
    repair_missing_identity = False
'''
_INSERTION = '''    identity_fields = ("observed_company_name", "observed_company_website")
    if company.company_linkedin:
        identity_fields += ("observed_company_linkedin",)
    if (
        investigation_targets
        and "rebrand" not in investigation_targets
        and "identity" in _incomplete_company_reverify_dimensions(
            result, icp_attribute=icp_attribute, icp_stage=icp_stage,
        )
        and any(
            not isinstance(verdict.get(field), str) or not verdict[field].strip()
            for field in identity_fields
        )
    ):
        # Repair the missing identity before spending the single bounded
        # investigation. Its full response can leave a different fact gap
        # (for example geography), which must join that same investigation.
        # No completed finding exists yet; never copy proof from this
        # incomplete identity or add a second investigation budget.
        investigation_targets = ()
'''


class IdentityRepairOrderSeamError(ValueError):
    """The source is not the exact reviewed frozen generation."""


def convert_identity_repair_order_seam(source: bytes) -> bytes:
    digest = hashlib.sha256(source).hexdigest()
    if digest == REPAIRED_SHA256:
        return source
    if digest != ORIGINAL_SHA256:
        raise IdentityRepairOrderSeamError("unrecognized or partial frozen identity-repair-order source")
    text = source.decode("utf-8")
    functions = [node for node in ast.parse(text).body
                 if isinstance(node, ast.AsyncFunctionDef) and node.name == FUNCTION]
    if len(functions) != 1:
        raise IdentityRepairOrderSeamError("identity verifier is missing or ambiguous")
    function = ast.get_source_segment(text, functions[0])
    if function is None or function.count(_ANCHOR) != 1:
        raise IdentityRepairOrderSeamError("frozen identity repair anchor differs")
    corrected = function.replace(_ANCHOR, _INSERTION + _ANCHOR, 1)
    result = text.replace(function, corrected, 1).encode("utf-8")
    compile(result, "frozen_identity_repair_order.py", "exec")
    if hashlib.sha256(result).hexdigest() != REPAIRED_SHA256:
        raise IdentityRepairOrderSeamError("identity repair ordering differs from reviewed source")
    return result
