"""Carry only the reviewed structured numeric-homepage alias proof into quote binding.

The complete original or repaired identity function must match exactly. All
other frozen bytes remain unchanged; no current grader implementation is copied.
"""
from __future__ import annotations

import ast
import hashlib

from scripts.arena_reviewer_investigation_finality_seam import _named_bindings

CORRECTION_ID = "structured_homepage_numeric_alias_v1"
ORIGINAL_SHA256 = "bf386d626833f58d6ad298204e85408a8f748dd5a241d630baf7efc56b6d2147"
REPAIRED_SHA256 = "59b4c46c2cb74487c04b7d3cc4b16c086ad4689280a94b144961eecdd489c997"
ANCHOR = '    observed_slug = str(web_identity.get("observed_linkedin_slug") or "")\n'
INSERTION = '''    if (
        homepage_slug.isdigit()
        and observed_slug
        and not observed_slug.isdigit()
        and submitted_slug in {"", observed_slug}
        and set(evidence) == {
            "name", "provider", "source_field", "url", "website",
            "company_id", "requested_url",
        }
        and all(isinstance(value, str) for value in evidence.values())
        and evidence.get("provider") == STRUCTURED_PROFILE_PROVIDER
        and evidence.get("source_field") == STRUCTURED_PROFILE_IDENTITY_SOURCE_FIELD
        and evidence.get("company_id") == homepage_slug
        and _strict_linkedin_company_profile_url(evidence.get("requested_url"))
        and linkedin_company_page_slug(evidence.get("requested_url")) == homepage_slug
        and _strict_linkedin_company_profile_url(evidence.get("url"))
        and linkedin_company_page_slug(evidence.get("url")) == observed_slug
        and web_identity.get("decision") == COMPANY_FIT_MATCH
        and web_identity.get("evidence_source") == "company_web_reverification"
        and web_identity.get("submitted_linkedin_slug") == submitted_slug
        and transport_domain
        and homepage_identity.get("registrable_dns_domain") == transport_domain
        and _canonical_company_domain(evidence.get("website")) == transport_domain
        and web_identity.get("submitted_domain") == transport_domain
        and web_identity.get("observed_domain") == transport_domain
        and _company_name(evidence.get("name"))
        and _company_name(evidence.get("name"))
        == _company_name(homepage_identity.get("normalized_name"))
        == _company_name(web_identity.get("submitted_name"))
        == _company_name(web_identity.get("observed_name"))
    ):
        # The existing structured parser can prove the homepage's numeric ID
        # even when the submitted LinkedIn field is optional or already vanity.
        # Carry that exact proof into quote binding without changing the locator.
        return {
            **homepage_identity,
            "linkedin_company_slug": observed_slug,
        }, str(submitted_linkedin or "")
'''


class StructuredHomepageAliasSeamError(ValueError):
    """The frozen identity function is not an exact reviewed version."""


def convert_structured_homepage_alias_seam(source: bytes) -> bytes:
    try:
        text = source.decode("utf-8")
        module = ast.parse(text)
    except (UnicodeDecodeError, SyntaxError) as exc:
        raise StructuredHomepageAliasSeamError("grading source must be valid UTF-8 Python") from exc
    if "\r" in text:
        raise StructuredHomepageAliasSeamError("grading source must use LF newlines")
    nodes = [node for node in module.body if isinstance(node, ast.FunctionDef)
             and node.name == "_investigator_identity_context"]
    if len(nodes) != 1 or _named_bindings(module, "_investigator_identity_context") != nodes:
        raise StructuredHomepageAliasSeamError("identity context binding is missing or ambiguous")
    node = nodes[0]
    lines = text.splitlines(keepends=True)
    body = "".join(lines[node.lineno - 1:node.end_lineno])
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    if digest == REPAIRED_SHA256:
        return source
    if digest != ORIGINAL_SHA256 or body.count(ANCHOR) != 1:
        raise StructuredHomepageAliasSeamError("unrecognized identity context source")
    repaired = body.replace(ANCHOR, ANCHOR + INSERTION, 1)
    if hashlib.sha256(repaired.encode("utf-8")).hexdigest() != REPAIRED_SHA256:
        raise StructuredHomepageAliasSeamError("identity context repair differs from reviewed source")
    lines[node.lineno - 1:node.end_lineno] = [repaired]
    result = "".join(lines).encode("utf-8")
    convert_structured_homepage_alias_seam(result)
    return result


# This second delta stays in the existing identity seam and the same overlay.
SELF_NAME_CORRECTION_ID = "numeric_profile_leading_legal_self_name_v1"
SELF_NAME_CHANGES = {
    'qualification/scoring/linkedin_company_size.py': (
        'project_structured_linkedin_company_identity',
        '3da1227084b6b3db8e992a8fe793780c1a7a25a3a375bae6fe0bc9ce97726416',
        '527bd7c3018d9a849fa958f956505cc9b4e797e1dea0861668a633d580021d10',
        (
            (
                '    if (\n        not requested_slug.isdigit()\n',
                r"""    identity_name = name
    identity_source_field = STRUCTURED_PROFILE_IDENTITY_SOURCE_FIELD
    name_matches = _company_name(name) == _company_name(expected_company_name)
    if not name_matches:
        # One numeric main profile can publish a short display name and open
        # its own description with its full legal self-name. This is not a
        # search for customer/partner names elsewhere in provider prose.
        description = element.get("description")
        self_name = re.match(
            r"([A-Za-z0-9][A-Za-z0-9 .,'&()\-]{0,199}?)[ \t]+is[ \t]+([^\r\n]{1,300})",
            description,
        ) if isinstance(description, str) else None
        universal_name = element.get("universalName")
        if (
            website_matches
            and company_id.isascii()
            and self_name
            and name.isascii()
            and self_name[1].casefold().startswith(name.casefold() + " ")
            and _company_name(self_name[1]) == _company_name(expected_company_name)
            and isinstance(universal_name, str)
            and universal_name.isascii()
            and universal_name == returned_slug
            and re.match(r"(?:an?|the)[ \t]+[A-Za-z0-9]", self_name[2], re.I)
            and not re.search(
                r"\b(?:customers?|partners?|clients?|competitors?|subsidiar(?:y|ies)|division|"
                r"not|never|allegedly|quoted|mentioned)\b",
                self_name[2][:120], re.I,
            )
        ):
            identity_name = self_name[1]
            identity_source_field = "description.leading_legal_name"
            name_matches = True
    if (
        not requested_slug.isdigit()
""",
            ),
            (
                '        or _company_name(name) != _company_name(expected_company_name)\n',
                r"""        or not name_matches
""",
            ),
            (
                '    evidence = {\n        "name": name,\n        "provider": STRUCTURED_PROFILE_PROVIDER,\n        "source_field": STRUCTURED_PROFILE_IDENTITY_SOURCE_FIELD,\n',
                r"""    evidence = {
        "name": identity_name,
        "provider": STRUCTURED_PROFILE_PROVIDER,
        "source_field": identity_source_field,
""",
            ),
        ),
    ),
    'qualification/scoring/lead_scorer.py': (
        '_investigator_identity_context',
        '59b4c46c2cb74487c04b7d3cc4b16c086ad4689280a94b144961eecdd489c997',
        '24acb16cab038b79ea723466ead07cfe2c9c62518b51a9df1e6e2c3ad70d6dca',
        (
            (
                '        and evidence.get("source_field") == STRUCTURED_PROFILE_IDENTITY_SOURCE_FIELD\n        and evidence.get("company_id") == homepage_slug\n',
                r"""        and evidence.get("source_field") in {
            STRUCTURED_PROFILE_IDENTITY_SOURCE_FIELD,
            "description.leading_legal_name",
        }
        and evidence.get("company_id") == homepage_slug
""",
            ),
        ),
    ),
}


def convert_numeric_profile_self_name_files(files: dict[str, bytes]) -> dict[str, bytes]:
    """Change only two exact reviewed functions after the existing alias repair."""
    if set(files) != set(SELF_NAME_CHANGES):
        raise StructuredHomepageAliasSeamError("numeric self-name source set is incomplete")
    result = {}
    for path, (name, original_hash, repaired_hash, changes) in SELF_NAME_CHANGES.items():
        source = files[path]
        try:
            text = source.decode("utf-8")
            module = ast.parse(text)
        except (UnicodeDecodeError, SyntaxError) as exc:
            raise StructuredHomepageAliasSeamError("numeric self-name source must be valid Python") from exc
        if "\r" in text:
            raise StructuredHomepageAliasSeamError("numeric self-name source must use LF newlines")
        nodes = [node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == name]
        if len(nodes) != 1 or _named_bindings(module, name) != nodes:
            raise StructuredHomepageAliasSeamError("numeric self-name function binding is ambiguous")
        node = nodes[0]
        lines = text.splitlines(keepends=True)
        body = "".join(lines[node.lineno - 1:node.end_lineno])
        digest = hashlib.sha256(body.encode()).hexdigest()
        if digest == repaired_hash:
            result[path] = source
            continue
        if digest != original_hash:
            raise StructuredHomepageAliasSeamError("unrecognized numeric self-name function source")
        for before, after in changes:
            if body.count(before) != 1:
                raise StructuredHomepageAliasSeamError("numeric self-name repair anchor is ambiguous")
            body = body.replace(before, after, 1)
        if hashlib.sha256(body.encode()).hexdigest() != repaired_hash:
            raise StructuredHomepageAliasSeamError("numeric self-name repair differs from reviewed source")
        lines[node.lineno - 1:node.end_lineno] = [body]
        result[path] = "".join(lines).encode()
    return result
