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
