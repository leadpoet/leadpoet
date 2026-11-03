"""Apply only the reviewed homepage LinkedIn conflict repair to frozen source.

The exact two blocks are owned by distinct, unambiguous AST functions.
No current grading module or dependencies are imported into the image.
"""
from __future__ import annotations

import ast

from scripts.arena_reviewer_investigation_finality_seam import _named_bindings

CORRECTION_ID = "verified_homepage_linkedin_conflict_schema_repair_v1"
REASON = "web_linkedin_conflicts_with_verified_homepage"


class HomepageLinkedinConflictSeamError(ValueError):
    """Frozen source does not have the exact reviewed identity blocks."""


OLD_DISPATCH = r'''    if (
        receipt.get("decision") == COMPANY_FIT_MISMATCH
        and receipt.get("reason_code") == "identity_mismatch"
        and verified_anchor_receipt.get("decision") == COMPANY_FIT_MATCH
        and receipt.get("submitted_name") == receipt.get("observed_name")
        and receipt.get("submitted_domain")
        == verified_homepage_identity.get("registrable_dns_domain")
        and receipt.get("submitted_linkedin_slug")
        == verified_homepage_identity.get("linkedin_company_slug")
        and receipt.get("observed_domain")
        == verified_homepage_identity.get("registrable_dns_domain")
        and receipt.get("observed_linkedin_slug")
        and receipt.get("observed_linkedin_slug")
        != verified_homepage_identity.get("linkedin_company_slug")
    ):
        # A model-owned alternate LinkedIn slug is not stronger than the exact
        # identity triplet independently fetched from the first-party
        # homepage. It is also not enough to prove that the slugs are aliases.
        # Keep the model observation in the receipt, classify the conflict as
        # incomplete, and let the existing bounded repair request re-check it.
        receipt.update(
            decision=COMPANY_FIT_UNAVAILABLE,
            reason_code="web_linkedin_conflicts_with_verified_homepage",
        )
        return receipt
'''

NEW_DISPATCH = r'''    if (
        verified_anchor_receipt.get("decision") == COMPANY_FIT_MATCH
        and receipt.get("submitted_domain")
        == verified_homepage_identity.get("registrable_dns_domain")
        and receipt.get("submitted_linkedin_slug")
        in {"", verified_homepage_identity.get("linkedin_company_slug")}
        and receipt.get("observed_domain")
        == verified_homepage_identity.get("registrable_dns_domain")
        and receipt.get("observed_linkedin_slug")
        and receipt.get("observed_linkedin_slug")
        != verified_homepage_identity.get("linkedin_company_slug")
        and (
            receipt.get("decision") == COMPANY_FIT_MISMATCH
            and receipt.get("reason_code") == "identity_mismatch"
            or (
                receipt.get("decision") == COMPANY_FIT_UNAVAILABLE
                and not str(company.company_linkedin or "").strip()
                and receipt.get("reason_code")
                in {"identity_not_proven", "identity_name_alias_unresolved"}
                # Preserve unresolved routes unless the existing homepage
                # fallback would produce this exact terminal mismatch.
                and (fallback_identity := evaluate_company_identity(
                    submitted_name=company.company_name,
                    submitted_website=company.company_website,
                    submitted_linkedin=(
                        "https://www.linkedin.com/company/"
                        + verified_homepage_identity["linkedin_company_slug"]
                    ),
                    observed_name=observed_values["name"],
                    observed_website=observed_values["website"],
                    observed_linkedin=observed_values["linkedin"],
                    evidence_source="company_web_reverification",
                    company_quality=company_quality,
                )).get("decision") == COMPANY_FIT_MISMATCH
                and fallback_identity.get("reason_code") == "identity_mismatch"
            )
        )
    ):
        # A model-owned alternate LinkedIn slug is not stronger than the exact
        # identity triplet independently fetched from the first-party
        # homepage, even when the web model also changes the company name.
        # Neither a changed name nor an omitted submitted LinkedIn URL proves
        # the slugs are aliases. They still require independent identity repair.
        # Keep the model observation in the receipt, classify the conflict as
        # incomplete, and let the existing bounded repair request re-check it.
        receipt.update(
            decision=COMPANY_FIT_UNAVAILABLE,
            reason_code="web_linkedin_conflicts_with_verified_homepage",
        )
        return receipt
'''

OLD_PROMPT = r'''        identity_conflict_repair = (
            "\nIDENTITY CONFLICT REPAIR: the prior observed_company_linkedin "
            "conflicted with the server-verified first-party homepage anchor "
            "while the normalized company name and domain matched. Perform a "
            "fresh lookup. Return the exact LinkedIn company URL only when a "
            "public source proves it; otherwise return an empty string. Do not "
            "assume that two different slugs are aliases and do not repeat an "
            "ungrounded alternate slug."
        )
'''

NEW_PROMPT = r'''        identity_conflict_repair = (
            "\nIDENTITY CONFLICT REPAIR: the prior observed_company_linkedin "
            "conflicted with the server-verified first-party homepage anchor "
            "while the company domain matched. The observed company name may "
            "also differ. Perform a fresh lookup for the anchored entity and "
            "independently verify its name, domain, and LinkedIn URL. Return "
            "the exact LinkedIn company URL only when a "
            "public source proves it; otherwise return an empty string. Do not "
            "assume that two different slugs are aliases and do not repeat an "
            "ungrounded alternate slug."
        )
'''


def _function(module: ast.Module, name: str, kind: type) -> ast.AST:
    nodes = [node for node in module.body if isinstance(node, kind) and node.name == name]
    if len(nodes) != 1 or _named_bindings(module, name) != nodes:
        raise HomepageLinkedinConflictSeamError("identity function binding is missing or ambiguous")
    if nodes[0].decorator_list:
        raise HomepageLinkedinConflictSeamError("identity function must not be decorated")
    return nodes[0]


def convert_homepage_linkedin_conflict_seam(source: bytes) -> bytes:
    try:
        text = source.decode("utf-8")
        module = ast.parse(text)
    except (UnicodeDecodeError, SyntaxError) as exc:
        raise HomepageLinkedinConflictSeamError("identity source must be UTF-8 Python") from exc
    if "\r" in text:
        raise HomepageLinkedinConflictSeamError("identity source must use LF newlines")
    web = _function(module, "_web_identity_receipt", ast.FunctionDef)
    verifier = _function(module, "_llm_reverify_company", ast.AsyncFunctionDef)
    dispatches = [node for node in web.body if isinstance(node, ast.If)
                  and any(isinstance(child, ast.Constant) and child.value == REASON
                          for child in ast.walk(node))]
    expected_test = ast.parse("identity_conflict_reason == " + repr(REASON), mode="eval").body
    prompts = [node for node in ast.walk(verifier) if isinstance(node, ast.If)
               and ast.dump(node.test) == ast.dump(expected_test)]
    if len(dispatches) != 1 or len(prompts) != 1 or len(prompts[0].body) != 1:
        raise HomepageLinkedinConflictSeamError("identity correction blocks are missing or ambiguous")
    prompt = prompts[0].body[0]
    if (not isinstance(prompt, ast.Assign) or len(prompt.targets) != 1
            or not isinstance(prompt.targets[0], ast.Name)
            or prompt.targets[0].id != "identity_conflict_repair"):
        raise HomepageLinkedinConflictSeamError("identity repair prompt binding changed")
    lines = text.splitlines(keepends=True)
    replacements, states = [], set()
    for node, old, new in (
        (dispatches[0], OLD_DISPATCH, NEW_DISPATCH),
        (prompt, OLD_PROMPT, NEW_PROMPT),
    ):
        block = "".join(lines[node.lineno - 1:node.end_lineno])
        if block == old:
            states.add("original")
            replacements.append((node.lineno - 1, node.end_lineno, new))
        elif block == new:
            states.add("corrected")
        else:
            raise HomepageLinkedinConflictSeamError("unrecognized identity correction block")
    if len(states) != 1:
        raise HomepageLinkedinConflictSeamError("partially corrected identity source")
    if states == {"corrected"}:
        return source
    for start, end, replacement in sorted(replacements, reverse=True):
        lines[start:end] = [replacement]
    result = "".join(lines).encode("utf-8")
    compile(result, "frozen_homepage_linkedin_conflict.py", "exec")
    if convert_homepage_linkedin_conflict_seam(result) != result:
        raise HomepageLinkedinConflictSeamError("identity repair is not idempotent")
    return result

