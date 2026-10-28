"""Compose only the reviewed homepage brand proof and matching host contract.

The host must deploy the matching company_quality_receipt_matches_claim
validator before accepting this runtime's verified_homepage_brand_alias receipt.
The proof marker stays in server anchors and receipts; the two model prompt
copies omit only that marker. A fresh native full-scorer check is required.
This option does not deploy host source or import current grading modules.
"""
from __future__ import annotations

import ast
import hashlib

from scripts.arena_reviewer_investigation_finality_seam import _named_bindings

CORRECTION_ID = "verified_homepage_brand_legal_identity_v1"
SCORER = "qualification/scoring/lead_scorer.py"
FIT = "qualification/scoring/company_fit_decision.py"
INVESTIGATOR = "qualification/scoring/company_evidence_investigator.py"
HOST_VALIDATOR_REQUIREMENT = {
    "module": "qualification.scoring.company_fit_decision",
    "function": "company_quality_receipt_matches_claim",
    "function_source_sha256": "fada24a1e2a9f22e0be36d7099f7d4ace2daa99793239498ef00e0a5a76e9424",
    "receipt_reason_code": "verified_homepage_brand_alias",
    "proof_marker": "verified_homepage_brand_alias",
}


class HomepageBrandAliasSeamError(ValueError):
    """Source differs from the reviewed brand identity correction."""


OLD_PROJECTOR = r'''            if brand:
                anchor["verified_brand_name"] = brand
'''

NEW_PROJECTOR = r'''            if brand:
                anchor["verified_brand_name"] = brand
                # The new brand acceptance path must see the original proof,
                # before legacy alias truncation/filtering can hide a defect.
                if (
                    0 < len(raw_aliases) <= 3
                    and all(
                        isinstance(alias, str) and alias == alias.strip()
                        and bool(alias) and len(alias) <= 200
                        and _company_name(alias) == anchor["normalized_name"]
                        for alias in raw_aliases
                    )
                    and receipt.get("submitted_name")
                    == receipt.get("observed_name") == anchor["normalized_name"]
                    and receipt.get("submitted_domain")
                    == receipt.get("observed_domain") == anchor["registrable_dns_domain"]
                    and receipt.get("submitted_linkedin_slug")
                    == receipt.get("observed_linkedin_slug") == anchor["linkedin_company_slug"]
                ):
                    anchor["verified_homepage_brand_alias"] = True
'''

OLD_WEB = r'''    if (
        receipt.get("decision") != COMPANY_FIT_MATCH
        and verified_brand_receipt.get("decision") == COMPANY_FIT_MATCH
        and receipt.get("evidence_source") == "company_web_reverification"
        and receipt_submitted_domain
        == receipt_observed_domain
        == verified_alias_domain
        and receipt_submitted_slug
        == receipt_observed_slug
        == verified_alias_slug
        and verified_brand_legal_name_initialism(
            verified_alias_name,
            observed_values["name"],
            verified_alias_anchor.get("verified_legal_name_aliases"),
        )
    ):
        receipt.update(
            decision=COMPANY_FIT_MATCH,
            reason_code="verified_legal_name_initialism",
        )
        return receipt
'''

NEW_WEB = r'''    if (
        receipt.get("decision") != COMPANY_FIT_MATCH
        and verified_brand_receipt.get("decision") == COMPANY_FIT_MATCH
        and receipt.get("evidence_source") == "company_web_reverification"
        and receipt_submitted_domain
        == receipt_observed_domain
        == verified_alias_domain
        and receipt_submitted_slug
        == receipt_observed_slug
        == verified_alias_slug
        and verified_brand_legal_name_initialism(
            verified_alias_name,
            observed_values["name"],
            verified_alias_anchor.get("verified_legal_name_aliases"),
        )
    ):
        receipt.update(
            decision=COMPANY_FIT_MATCH,
            reason_code="verified_legal_name_initialism",
        )
        return receipt
    homepage_brand = verified_homepage_brand_name(verified_alias_anchor)
    homepage_legal_aliases = verified_alias_anchor.get("verified_legal_name_aliases")
    if (
        receipt.get("decision") != COMPANY_FIT_MATCH
        and verified_brand_receipt.get("decision") == COMPANY_FIT_MATCH
        and verified_alias_anchor.get("verified_homepage_brand_alias") is True
        and receipt.get("submitted_name") == verified_alias_name
        and receipt_submitted_domain == receipt_observed_domain == verified_alias_domain
        and receipt_submitted_slug == receipt_observed_slug == verified_alias_slug
        and homepage_brand
        and receipt.get("observed_name") == _company_name(homepage_brand)
        and isinstance(homepage_legal_aliases, list)
        and 0 < len(homepage_legal_aliases) <= 3
        and all(
            isinstance(alias, str) and alias.strip() and len(alias.strip()) <= 200
            and _company_name(alias) == verified_alias_name
            for alias in homepage_legal_aliases
        )
    ):
        receipt.update(
            decision=COMPANY_FIT_MATCH,
            reason_code="verified_homepage_brand_alias",
            verified_brand_name=homepage_brand,
            verified_legal_name_aliases=list(homepage_legal_aliases),
            verified_homepage_brand_alias=True,
        )
        return receipt
'''

OLD_HOST = r'''    raw_aliases = receipt.get("verified_legal_name_aliases")
    if not isinstance(raw_aliases, list) or len(raw_aliases) > 3:
        return False
    return any(
        isinstance(alias, str)
        and bool(alias.strip())
        and len(alias.strip()) <= 200
        and _company_name(alias) == observed_name
        for alias in raw_aliases
    )
'''

NEW_HOST = r'''    raw_aliases = receipt.get("verified_legal_name_aliases")
    if not isinstance(raw_aliases, list) or len(raw_aliases) > 3:
        return False
    if receipt.get("reason_code") == "verified_homepage_brand_alias":
        brand = verified_homepage_brand_name({
            "normalized_name": claimed_name,
            "verified_brand_name": receipt.get("verified_brand_name"),
            "verified_legal_name_aliases": raw_aliases,
        })
        return bool(
            receipt.get("evidence_source") == "company_web_reverification"
            and receipt.get("verified_homepage_brand_alias") is True
            and brand and _company_name(brand) == observed_name
            and raw_aliases
            and all(
                isinstance(alias, str) and alias.strip() and len(alias.strip()) <= 200
                and _company_name(alias) == claimed_name
                for alias in raw_aliases
            )
        )
    return any(
        isinstance(alias, str)
        and bool(alias.strip())
        and len(alias.strip()) <= 200
        and _company_name(alias) == observed_name
        for alias in raw_aliases
    )
'''

BRAND_HELPER = r'''def verified_homepage_brand_name(identity: Mapping[str, Any]) -> str:
    """Read the host-projected brand bound to this exact homepage legal name.

    Callers supply an independently verified homepage anchor, never model JSON.
    A brand cannot extend an unrelated legal identity merely by being present.
    """

    brand = identity.get("verified_brand_name")
    aliases = identity.get("verified_legal_name_aliases")
    name = _company_name(identity.get("normalized_name"))
    if (
        not isinstance(brand, str)
        or not 2 < len(brand.strip()) <= 200
        or not _company_name(brand)
        or not name
        or not isinstance(aliases, list)
        or not aliases
    ):
        return ""
    if name == _company_name(brand) or any(
        isinstance(alias, str) and len(alias.strip()) <= 200
        and _company_name(alias) == name
        for alias in aliases[:3]
    ):
        return brand.strip()
    return ""
'''


OLD_SCORER_PROMPT = '''            + json.dumps(
                verified_identity,
                sort_keys=True,
                separators=(",", ":"),
            )
'''
NEW_SCORER_PROMPT = '''            + json.dumps(
                {
                    key: value for key, value in verified_identity.items()
                    if key != "verified_homepage_brand_alias"
                },
                sort_keys=True,
                separators=(",", ":"),
            )
'''
OLD_INVESTIGATOR_PROMPT = '''        "verified_homepage_identity": dict(verified_homepage_identity or {}),
'''
NEW_INVESTIGATOR_PROMPT = '''        "verified_homepage_identity": {
            key: value for key, value in (verified_homepage_identity or {}).items()
            if key != "verified_homepage_brand_alias"
        },
'''

# Exact prompt owners after existing overlays on the 8d, 8761 and 8bd bases.
# Reverse only the owned copy filter before checking these original hashes.
PROMPT_FUNCTION_SHA256 = {
    "_llm_reverify_company": frozenset({
        "254737f6424356190f62f3496acce81379cbfb1d08609ce572b4dadc95943683",
        "8b6d0f580854704111f1720aa4789b5a6d4f492ad25cfa752b6a7771cd1cfb39",
    }),
    "investigate_company_evidence": frozenset({
        "d3b3848dc7751c6019b41b9a310aec64e2dc45343d5b96ccae0ab7e69146c01f",
        "34a42f7cf46b403d9125fbc403250a284923cac49b7a1c699e92e31fde5ce1a0",
        "5695fb8c525810d8e8f7e4eddb989c2a90ba6b1a5529c7625335e5cfb15a661f",
    }),
}


ORIGINAL_FUNCTION_SHA256 = {'_verified_homepage_identity_anchor': 'c244aefac41b72109b1b848df8fe02dfde6877d3abf002dba5c0c6de6eddbd08', '_web_identity_receipt': 'e257bd2a364b20039915d3ae26af11f4f38f0130450036183723afedb5cc6916', 'company_quality_receipt_matches_claim': '13fd832087df8fb3d701ff34254655c12ff6df22c9aee3d8974f4656d4c06044'}


SIGNATURES = {
    '_verified_homepage_identity_anchor': 'def _verified_homepage_identity_anchor(\n    identity: Optional[CompanyFitDecisionResult],\n) -> dict[str, Any]:\n    pass\n',
    '_web_identity_receipt': 'def _web_identity_receipt(\n    company: CompanyOutput,\n    verdict: Mapping[str, Any],\n    *,\n    verified_homepage_identity: Optional[Mapping[str, Any]] = None,\n    verified_homepage_transport_domain: str = "",\n    verified_homepage_navigation_locators: Sequence[Mapping[str, Any]] = (),\n    verified_rebrand_identity: Optional[Mapping[str, Any]] = None,\n    verified_rebrand_redirect: Optional[Mapping[str, str]] = None,\n    verified_structured_identity: Optional[Mapping[str, Any]] = None,\n    company_quality: bool = False,\n) -> dict[str, Any]:\n    pass\n',
    'company_quality_receipt_matches_claim': 'def company_quality_receipt_matches_claim(\n    receipt: Any,\n    company: Mapping[str, Any],\n) -> bool:\n    pass\n',
}


def _function(module: ast.Module, name: str, kind: type = ast.FunctionDef) -> ast.AST:
    nodes = [node for node in module.body
             if isinstance(node, kind) and node.name == name]
    if (len(nodes) != 1 or _named_bindings(module, name) != nodes
            or nodes[0].decorator_list):
        raise HomepageBrandAliasSeamError("brand function binding is missing or ambiguous")
    if name in SIGNATURES:
        signature = ast.parse(SIGNATURES[name]).body[0]
        if (ast.dump(nodes[0].args) != ast.dump(signature.args)
                or ast.dump(nodes[0].returns or ast.Constant(value=None))
                != ast.dump(signature.returns or ast.Constant(value=None))):
            raise HomepageBrandAliasSeamError("brand function signature changed")
    return nodes[0]


def _reason_if(function: ast.FunctionDef, reason: str) -> ast.If:
    nodes = [node for node in function.body if isinstance(node, ast.If)
             and any(isinstance(child, ast.Constant) and child.value == reason
                     for child in ast.walk(node))]
    if len(nodes) != 1:
        raise HomepageBrandAliasSeamError("brand correction anchor is missing or ambiguous")
    return nodes[0]


def _prompt_span(path: str, module: ast.Module) -> tuple:
    if path == SCORER:
        owner = _function(module, "_llm_reverify_company", ast.AsyncFunctionDef)
        guards = [node for node in ast.walk(owner) if isinstance(node, ast.If)
                  and isinstance(node.test, ast.Name) and node.test.id == "verified_identity"]
        if (len(guards) != 1 or guards[0].orelse or len(guards[0].body) != 1
                or not isinstance(guards[0].body[0], ast.Assign)
                or len(guards[0].body[0].targets) != 1
                or not isinstance(guards[0].body[0].targets[0], ast.Name)
                or guards[0].body[0].targets[0].id != "verified_identity_context"):
            raise HomepageBrandAliasSeamError("homepage prompt guard is missing or ambiguous")
        sites = [node for node in ast.walk(guards[0].body[0]) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
                 and node.func.value.id == "json" and node.func.attr == "dumps"]
        old, new = OLD_SCORER_PROMPT, NEW_SCORER_PROMPT
    else:
        owner = _function(module, "investigate_company_evidence", ast.AsyncFunctionDef)
        documents = [node for node in owner.body if isinstance(node, ast.Assign)
                     and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                     and node.targets[0].id == "input_document" and isinstance(node.value, ast.Dict)]
        if len(documents) != 1:
            raise HomepageBrandAliasSeamError("investigator prompt document is missing or ambiguous")
        sites = [value for key, value in zip(documents[0].value.keys, documents[0].value.values)
                 if isinstance(key, ast.Constant) and key.value == "verified_homepage_identity"]
        old, new = OLD_INVESTIGATOR_PROMPT, NEW_INVESTIGATOR_PROMPT
    if len(sites) != 1:
        raise HomepageBrandAliasSeamError("homepage prompt copy is missing or ambiguous")
    return owner, sites[0].lineno - 1, sites[0].end_lineno, old, new, ()


def _owned_spans(path: str, module: ast.Module) -> list[tuple]:
    if path == FIT:
        function = _function(module, "company_quality_receipt_matches_claim")
        assignments = [node for node in function.body if isinstance(node, ast.Assign)
                       and any(isinstance(target, ast.Name) and target.id == "raw_aliases"
                               for target in node.targets)]
        if len(assignments) != 1:
            raise HomepageBrandAliasSeamError("host alias projection is missing or ambiguous")
        return [(function, assignments[0].lineno - 1, function.end_lineno,
                 OLD_HOST, NEW_HOST, ("brand",))]
    if path == INVESTIGATOR:
        return [_prompt_span(path, module)]
    projector = _function(module, "_verified_homepage_identity_anchor")
    anchors = [node for node in ast.walk(projector) if isinstance(node, ast.If)
               and isinstance(node.test, ast.Name) and node.test.id == "brand"]
    if len(anchors) != 1:
        raise HomepageBrandAliasSeamError("raw homepage brand projection is missing or ambiguous")
    web = _function(module, "_web_identity_receipt")
    first = _reason_if(web, "verified_legal_name_initialism")
    following = _reason_if(web, "verified_formal_legal_name_alias")
    if first.lineno >= following.lineno:
        raise HomepageBrandAliasSeamError("brand insertion anchors are reversed")
    return [
        (projector, anchors[0].lineno - 1, anchors[0].end_lineno,
         OLD_PROJECTOR, NEW_PROJECTOR, ()),
        (web, first.lineno - 1, following.lineno - 1,
         OLD_WEB, NEW_WEB, ("homepage_brand", "homepage_legal_aliases")),
        _prompt_span(path, module),
    ]


def _convert_file(path: str, source: bytes) -> tuple[bytes, set[str]]:
    try:
        text = source.decode("utf-8")
        module = ast.parse(text)
    except (UnicodeDecodeError, SyntaxError) as exc:
        raise HomepageBrandAliasSeamError("brand source must be UTF-8 Python") from exc
    if "\r" in text:
        raise HomepageBrandAliasSeamError("brand source must use LF newlines")
    lines = text.splitlines(keepends=True)
    if path == FIT:
        helper = _function(module, "verified_homepage_brand_name")
        if "".join(lines[helper.lineno - 1:helper.end_lineno]) != BRAND_HELPER:
            raise HomepageBrandAliasSeamError("unrecognized verified brand dependency")
    elif path == SCORER:
        imports = [alias for node in module.body if isinstance(node, ast.ImportFrom)
                   and node.module == "qualification.scoring.company_fit_decision"
                   for alias in node.names
                   if alias.name == "verified_homepage_brand_name" and alias.asname is None]
        if len(imports) != 1 or _named_bindings(module, "verified_homepage_brand_name") != imports:
            raise HomepageBrandAliasSeamError("verified brand dependency binding is ambiguous")
    states, replacements = set(), []
    for owner, start, end, old, new, introduced in _owned_spans(path, module):
        block = "".join(lines[start:end])
        if block == old:
            state = "original"
            replacements.append((start, end, new))
        elif block == new:
            state = "corrected"
        else:
            raise HomepageBrandAliasSeamError("unrecognized brand correction block")
        original_function = "".join(lines[owner.lineno - 1:owner.end_lineno])
        if state == "corrected":
            original_function = original_function.replace(new, old, 1)
        if owner.name == "_web_identity_receipt":
            from scripts.arena_reviewer_homepage_linkedin_conflict_seam import OLD_DISPATCH, NEW_DISPATCH
            original_function = original_function.replace(NEW_DISPATCH, OLD_DISPATCH, 1)
        allowed = PROMPT_FUNCTION_SHA256.get(owner.name, (ORIGINAL_FUNCTION_SHA256.get(owner.name),))
        if hashlib.sha256(original_function.encode()).hexdigest() not in allowed:
            raise HomepageBrandAliasSeamError("unrecognized bytes in owned brand function")
        states.add(state)
        for name in introduced:
            uses = [node for node in ast.walk(owner)
                    if isinstance(node, ast.Name) and node.id == name]
            if (state == "original" and uses) or any(
                not start < node.lineno <= end for node in uses
            ) or any(
                isinstance(node, (ast.Global, ast.Nonlocal)) and name in node.names
                for node in ast.walk(owner)
            ):
                raise HomepageBrandAliasSeamError("brand local binding is ambiguous")
    if len(states) != 1:
        raise HomepageBrandAliasSeamError("partially corrected brand file")
    if states == {"corrected"}:
        return source, states
    for start, end, replacement in sorted(replacements, reverse=True):
        lines[start:end] = [replacement]
    result = "".join(lines).encode("utf-8")
    compile(result, "frozen_homepage_brand_alias.py", "exec")
    return result, states


def convert_homepage_brand_alias_files(files: dict[str, bytes]) -> dict[str, bytes]:
    if set(files) != {SCORER, FIT, INVESTIGATOR}:
        raise HomepageBrandAliasSeamError("exact scorer, investigator and host-validator sources are required")
    converted, states = {}, set()
    for path, source in files.items():
        converted[path], state = _convert_file(path, source)
        states.update(state)
    if len(states) != 1:
        raise HomepageBrandAliasSeamError("mixed original and corrected brand files")
    if states == {"original"} and convert_homepage_brand_alias_files(converted) != converted:
        raise HomepageBrandAliasSeamError("brand correction is not idempotent")
    return converted

