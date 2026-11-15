"""Preserve reviewed identity evidence through the exact frozen50de runtime.

Only the committed c3a1c17c6 -> c7d56e3ca identity handoff delta is applied.
Complete pair hashes reject source drift, partial repairs and other generations.
The existing four-option runtime composition must run before this correction.
"""
from __future__ import annotations

import hashlib
from typing import Mapping

CORRECTION_ID = "verified_identity_handoff_v1"
REVIEWED_BASE_COMMIT = "c3a1c17c6171138447832debce1ded9dd6b15590"
REVIEWED_SOURCE_COMMIT = "c7d56e3caed9a9c58f9ebda71caf8aac421fa198"
ORIGINAL_SHA256 = {
    'qualification/scoring/lead_scorer.py': '7994f7574b5e955d56c86d8a3a73a41f30d4dda302e7e0fcfa4a0e197f3d7bcd',
    'qualification/scoring/company_evidence_investigator.py': 'ca66a71b55b43f02d083d6cd24a3c6f0ebad1b06d8d57f16896717037a7f80d6',
}
REPAIRED_SHA256 = {
    'qualification/scoring/lead_scorer.py': 'c5a0495b38cb45038ee2afb3bd001e1779d13e383a6f2f9a77948d64bf2c874a',
    'qualification/scoring/company_evidence_investigator.py': 'ec74bdfb2cc4b657afc301cb183c799985915b25a8151ea056786c933c6113c5',
}
FILES = tuple(ORIGINAL_SHA256)

# Exact reviewed hunks with unique unchanged context. No repository source reads.
DELTAS = {
    'qualification/scoring/lead_scorer.py': (
        (
            (
                b'    verify_company_exists,\n'
                b')\n'
                b'from qualification.scoring.company_fit_decision import (\n'
                b'    verified_homepage_brand_name,\n'
                b'    observed_linkedin_ampersand_slug,\n'
                b'    COMPANY_FIT_MATCH,\n'
            ),
            (
                b'    verify_company_exists,\n'
                b')\n'
                b'from qualification.scoring.company_fit_decision import (\n'
                b'    company_quality_receipt_matches_claim,\n'
                b'    verified_homepage_brand_name,\n'
                b'    observed_linkedin_ampersand_slug,\n'
                b'    COMPANY_FIT_MATCH,\n'
            ),
        ),
        (
            (
                b'    verified_domain = str(\n'
                b'        recovery_identity.get("registrable_dns_domain") or ""\n'
                b'    ).strip()\n'
                b'    return _complete_verified_first_party_identity(\n'
                b'        evidence_url,\n'
                b'        {verified_domain} if verified_domain else set(),\n'
                b'        {\n'
                b'            "submitted_name": receipt.get("submitted_name") or company.company_name,\n'
                b'            "submitted_domain": receipt.get("submitted_domain"),\n'
                b'            "submitted_linkedin_slug": submitted_slug,\n'
                b'            "observed_name": receipt.get("observed_name"),\n'
                b'            "observed_domain": receipt.get("observed_domain"),\n'
                b'            "observed_linkedin_slug": receipt.get("observed_linkedin_slug"),\n'
                b'            "verified_name": recovery_identity.get("normalized_name"),\n'
            ),
            (
                b'    verified_domain = str(\n'
                b'        recovery_identity.get("registrable_dns_domain") or ""\n'
                b'    ).strip()\n'
                b'    legal_name = receipt.get("verified_parenthetical_legal_name")\n'
                b'    if legal_name is not None and not company_quality_receipt_matches_claim(\n'
                b'        receipt, {\n'
                b'            "company_name": company.company_name,\n'
                b'            "company_website": company.company_website,\n'
                b'            "company_linkedin": company.company_linkedin,\n'
                b'        },\n'
                b'    ):\n'
                b'        return False\n'
                b'    return _complete_verified_first_party_identity(\n'
                b'        evidence_url,\n'
                b'        {verified_domain} if verified_domain else set(),\n'
                b'        {\n'
                b'            "submitted_name": company.company_name if legal_name else (\n'
                b'                receipt.get("submitted_name") or company.company_name\n'
                b'            ),\n'
                b'            "submitted_domain": receipt.get("submitted_domain"),\n'
                b'            "submitted_linkedin_slug": submitted_slug,\n'
                b'            "observed_name": legal_name or receipt.get("observed_name"),\n'
                b'            "observed_domain": receipt.get("observed_domain"),\n'
                b'            "observed_linkedin_slug": receipt.get("observed_linkedin_slug"),\n'
                b'            "verified_name": recovery_identity.get("normalized_name"),\n'
            ),
        ),
        (
            (
                b'    web_identity: Mapping[str, Any],\n'
                b'    transport_domain: str,\n'
                b'    homepage_navigation_locators: Sequence[Mapping[str, Any]],\n'
                b') -> Mapping[str, str]:\n'
                b'    """Allow a lookup, never a match, from a fetched root\'s company-page link."""\n'
                b'\n'
                b'    submitted = str(web_identity.get("submitted_domain") or "")\n'
                b'    observed = str(web_identity.get("observed_domain") or "")\n'
            ),
            (
                b'    web_identity: Mapping[str, Any],\n'
                b'    transport_domain: str,\n'
                b'    homepage_navigation_locators: Sequence[Mapping[str, Any]],\n'
                b'    *,\n'
                b'    server_verified_homepage_receipt: bool = False,\n'
                b') -> Mapping[str, str]:\n'
                b'    """Locate a profile from a fetched root\'s company link or child redirect."""\n'
                b'\n'
                b'    submitted = str(web_identity.get("submitted_domain") or "")\n'
                b'    observed = str(web_identity.get("observed_domain") or "")\n'
            ),
        ),
        (
            (
                b'    slug = str(web_identity.get("observed_linkedin_slug") or "")\n'
                b'    submitted_slug = str(web_identity.get("submitted_linkedin_slug") or "")\n'
                b'    if (\n'
                b'        web_identity.get("evidence_source") != "company_web_reverification"\n'
                b'        or web_identity.get("decision")\n'
                b'        not in {COMPANY_FIT_MISMATCH, COMPANY_FIT_UNAVAILABLE}\n'
                b'        or web_identity.get("reason_code")\n'
                b'        not in {"identity_mismatch", "rebrand_continuity_unproven"}\n'
                b'        or not name or name != web_identity.get("submitted_name")\n'
                b'        or not slug or (submitted_slug and submitted_slug != slug)\n'
                b'        or not transport_domain or submitted != transport_domain\n'
                b'    ):\n'
                b'        return {}\n'
            ),
            (
                b'    slug = str(web_identity.get("observed_linkedin_slug") or "")\n'
                b'    submitted_slug = str(web_identity.get("submitted_linkedin_slug") or "")\n'
                b'    if (\n'
                b'        web_identity.get("evidence_source") != (\n'
                b'            "company_homepage" if server_verified_homepage_receipt\n'
                b'            else "company_web_reverification"\n'
                b'        )\n'
                b'        or web_identity.get("decision")\n'
                b'        not in ({COMPANY_FIT_UNAVAILABLE} if server_verified_homepage_receipt\n'
                b'                else {COMPANY_FIT_MISMATCH, COMPANY_FIT_UNAVAILABLE})\n'
                b'        or web_identity.get("reason_code")\n'
                b'        not in ({"identity_not_proven"} if server_verified_homepage_receipt\n'
                b'                else {"identity_mismatch", "rebrand_continuity_unproven"})\n'
                b'        or not name or name != web_identity.get("submitted_name")\n'
                b'        or not slug or (submitted_slug and submitted_slug != slug)\n'
                b'        or (server_verified_homepage_receipt and not submitted_slug)\n'
                b'        or not transport_domain or submitted != transport_domain\n'
                b'    ):\n'
                b'        return {}\n'
            ),
        ),
        (
            (
                b'        "registrable_dns_domain": transport_domain,\n'
                b'        "linkedin_company_slug": slug,\n'
                b'    }\n'
                b'    for locator in _validated_homepage_navigation_locators(\n'
                b'        homepage_navigation_locators,\n'
                b'        # This enables bounded locator validation, not a qualification verdict.\n'
            ),
            (
                b'        "registrable_dns_domain": transport_domain,\n'
                b'        "linkedin_company_slug": slug,\n'
                b'    }\n'
                b'    if server_verified_homepage_receipt:\n'
                b'        # The homepage verifier observed this exact name/profile at the child\n'
                b'        # reached from the submitted root. This permits only the existing\n'
                b'        # profile lookup; its returned raw website must independently bind the\n'
                b'        # root before any identity or employee-size evidence is admitted.\n'
                b'        return anchor\n'
                b'    for locator in _validated_homepage_navigation_locators(\n'
                b'        homepage_navigation_locators,\n'
                b'        # This enables bounded locator validation, not a qualification verdict.\n'
            ),
        ),
        (
            (
                b'        )\n'
                b'        if missing_link_lookup:\n'
                b'            return missing_link_lookup\n'
                b'        child_lookup = _official_child_structured_profile_lookup(\n'
                b'            web_identity, transport_domain, homepage_navigation_locators,\n'
                b'        )\n'
                b'        if child_lookup:\n'
                b'            return child_lookup\n'
                b'    observed_slug = str(web_identity.get("observed_linkedin_slug") or "").strip()\n'
                b'    submitted_slug = str(web_identity.get("submitted_linkedin_slug") or "").strip()\n'
                b'    expected_source = (\n'
            ),
            (
                b'        )\n'
                b'        if missing_link_lookup:\n'
                b'            return missing_link_lookup\n'
                b'    child_lookup = _official_child_structured_profile_lookup(\n'
                b'        web_identity, transport_domain, homepage_navigation_locators,\n'
                b'        server_verified_homepage_receipt=server_verified_homepage_receipt,\n'
                b'    )\n'
                b'    if child_lookup:\n'
                b'        return child_lookup\n'
                b'    observed_slug = str(web_identity.get("observed_linkedin_slug") or "").strip()\n'
                b'    submitted_slug = str(web_identity.get("submitted_linkedin_slug") or "").strip()\n'
                b'    expected_source = (\n'
            ),
        ),
        (
            (
                b'                         raw_observed_linkedin_slug=web_identity.get("observed_linkedin_slug", ""))\n'
                b'            return bound\n'
                b'    official_child = bool(\n'
                b'        not server_verified_homepage_receipt\n'
                b'        and _official_child_structured_profile_lookup(\n'
                b'            web_identity, transport_domain, homepage_navigation_locators,\n'
                b'        )\n'
                b'    )\n'
                b'    expected_source = (\n'
            ),
            (
                b'                         raw_observed_linkedin_slug=web_identity.get("observed_linkedin_slug", ""))\n'
                b'            return bound\n'
                b'    official_child = bool(\n'
                b'        _official_child_structured_profile_lookup(\n'
                b'            web_identity, transport_domain, homepage_navigation_locators,\n'
                b'            server_verified_homepage_receipt=server_verified_homepage_receipt,\n'
                b'        )\n'
                b'    )\n'
                b'    expected_source = (\n'
            ),
        ),
        (
            (
                b'        )\n'
                b'    )\n'
                b'    investigation_observed_linkedin = verdict.get("observed_company_linkedin")\n'
                b'    if investigation_web_identity.get("reason_code") == "structured_exact_profile_identity_verified":\n'
                b'        # Carry the fetched profile into research without rewriting the raw\n'
                b'        # model observation retained with the identity receipt.\n'
            ),
            (
                b'        )\n'
                b'    )\n'
                b'    investigation_observed_linkedin = verdict.get("observed_company_linkedin")\n'
                b'    investigation_legal_name = investigation_web_identity.get("verified_parenthetical_legal_name")\n'
                b'    if not (\n'
                b'        investigation_legal_name == verdict.get("observed_company_name")\n'
                b'        and company_quality_receipt_matches_claim(investigation_web_identity, {\n'
                b'            "company_name": company.company_name,\n'
                b'            "company_website": company.company_website,\n'
                b'            "company_linkedin": company.company_linkedin,\n'
                b'        })\n'
                b'    ):\n'
                b'        investigation_legal_name = None\n'
                b'    if investigation_web_identity.get("reason_code") == "structured_exact_profile_identity_verified":\n'
                b'        # Carry the fetched profile into research without rewriting the raw\n'
                b'        # model observation retained with the identity receipt.\n'
            ),
        ),
        (
            (
                b'                **{\n'
                b'                    key: investigation_web_identity.get(key)\n'
                b'                    for key in (\n'
                b'                        "submitted_name", "submitted_domain", "submitted_linkedin_slug",\n'
                b'                        "observed_name", "observed_domain", "observed_linkedin_slug",\n'
                b'                    )\n'
                b'                },\n'
                b'                "verified_name": investigation_verified_identity.get("normalized_name"),\n'
                b'                **{\n'
                b'                    key: investigation_verified_identity.get(key)\n'
            ),
            (
                b'                **{\n'
                b'                    key: investigation_web_identity.get(key)\n'
                b'                    for key in (\n'
                b'                        "submitted_domain", "submitted_linkedin_slug",\n'
                b'                        "observed_domain", "observed_linkedin_slug",\n'
                b'                    )\n'
                b'                },\n'
                b'                # The accepted receipt canonicalizes names. Preserve the raw\n'
                b'                # legal-name relation needed by the existing attribution gate.\n'
                b'                "submitted_name": company.company_name if investigation_legal_name else (\n'
                b'                    investigation_web_identity.get("submitted_name")\n'
                b'                ),\n'
                b'                "observed_name": investigation_legal_name or investigation_web_identity.get("observed_name"),\n'
                b'                "verified_name": investigation_verified_identity.get("normalized_name"),\n'
                b'                **{\n'
                b'                    key: investigation_verified_identity.get(key)\n'
            ),
        ),
        (
            (
                b'                collect_employee_size=False,\n'
                b'                collect_identity=True,\n'
                b'                preserve_identity_website=bool(\n'
                b'                    not server_verified_homepage_receipt\n'
                b'                    and (\n'
                b'                        _missing_link_structured_profile_lookup(\n'
                b'                            lookup_receipt, verified_transport_domain,\n'
                b'                        )\n'
                b'                        or _official_child_structured_profile_lookup(\n'
                b'                            lookup_receipt, verified_transport_domain,\n'
                b'                            verified_homepage_navigation_locators,\n'
                b'                        )\n'
                b'                    )\n'
                b'                ),\n'
                b'            )\n'
            ),
            (
                b'                collect_employee_size=False,\n'
                b'                collect_identity=True,\n'
                b'                preserve_identity_website=bool(\n'
                b'                    (\n'
                b'                        not server_verified_homepage_receipt\n'
                b'                        and\n'
                b'                        _missing_link_structured_profile_lookup(\n'
                b'                            lookup_receipt, verified_transport_domain,\n'
                b'                        )\n'
                b'                    )\n'
                b'                    or _official_child_structured_profile_lookup(\n'
                b'                        lookup_receipt, verified_transport_domain,\n'
                b'                        verified_homepage_navigation_locators,\n'
                b'                        server_verified_homepage_receipt=server_verified_homepage_receipt,\n'
                b'                    )\n'
                b'                ),\n'
                b'            )\n'
            ),
        ),
    ),
    'qualification/scoring/company_evidence_investigator.py': (
        (
            (
                b'\n'
                b'\n'
                b'def _reversed_parenthetical_identity_names(identity: Mapping[str, Any]) -> bool:\n'
                b'    """Keep an exact verified legal/initialism pair when its label is reversed."""\n'
                b'    submitted = str(identity.get("submitted_name") or "").strip()\n'
                b'    verified = str(identity.get("verified_name") or "").strip()\n'
                b'    observed = str(identity.get("observed_name") or "").strip()\n'
                b'    forward = _INITIALISM_LEGAL_NAME_RE.fullmatch(submitted)\n'
                b'    reverse = _PARENTHETICAL_INITIALISM_RE.fullmatch(observed)\n'
                b'    if (not forward or not reverse\n'
                b'            or _normalized_span(submitted) != _normalized_span(verified)\n'
                b'            or forward["initialism"] != reverse["initialism"]):\n'
                b'        return False\n'
                b"    # Reuse the identity gate's exact legal-name and real-initials parser.\n"
                b'    # The caller still requires the complete homepage/domain/profile binding.\n'
                b'    return _verified_parenthetical_name_alignment(\n'
                b'        submitted, reverse["legal_name"].strip(),\n'
                b'        observed_linkedin_slug=str(identity.get("observed_linkedin_slug") or "").strip().casefold(),\n'
                b'        submitted_linkedin_slug=str(identity.get("submitted_linkedin_slug") or "").strip().casefold(),\n'
                b'        evidence_source="company_homepage",\n'
            ),
            (
                b'\n'
                b'\n'
                b'def _reversed_parenthetical_identity_names(identity: Mapping[str, Any]) -> bool:\n'
                b'    """Retain an exact legal name, with or without its verified initialism."""\n'
                b'    submitted = str(identity.get("submitted_name") or "").strip()\n'
                b'    verified = str(identity.get("verified_name") or "").strip()\n'
                b'    observed = str(identity.get("observed_name") or "").strip()\n'
                b'    forward = _INITIALISM_LEGAL_NAME_RE.fullmatch(submitted)\n'
                b'    reverse = _PARENTHETICAL_INITIALISM_RE.fullmatch(observed)\n'
                b'    if (not forward\n'
                b'            or _normalized_span(submitted) != _normalized_span(verified)\n'
                b'            or (reverse and forward["initialism"] != reverse["initialism"])):\n'
                b'        return False\n'
                b"    # Reuse the identity gate's exact legal-name and real-initials parser.\n"
                b'    # The caller still requires the complete homepage/domain/profile binding.\n'
                b'    return _verified_parenthetical_name_alignment(\n'
                b'        submitted, reverse["legal_name"].strip() if reverse else observed,\n'
                b'        observed_linkedin_slug=str(identity.get("observed_linkedin_slug") or "").strip().casefold(),\n'
                b'        submitted_linkedin_slug=str(identity.get("submitted_linkedin_slug") or "").strip().casefold(),\n'
                b'        evidence_source="company_homepage",\n'
            ),
        ),
    ),
}


class IdentityHandoffSeamError(ValueError):
    """The supplied files are not an exact reviewed runtime pair."""


def convert_identity_handoff_files(sources: Mapping[str, bytes]) -> dict[str, bytes]:
    if set(sources) != set(FILES):
        raise IdentityHandoffSeamError("identity handoff requires the exact two-file pair")
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in sources.items()}
    if hashes == REPAIRED_SHA256:
        return dict(sources)
    if hashes != ORIGINAL_SHA256:
        raise IdentityHandoffSeamError("identity handoff source pair checksum mismatch")
    corrected = {}
    for name, source in sources.items():
        for before, after in DELTAS[name]:
            if source.count(before) != 1:
                raise IdentityHandoffSeamError("identity handoff anchor is not unique: " + name)
            source = source.replace(before, after, 1)
        compile(source, name, "exec")
        if hashlib.sha256(source).hexdigest() != REPAIRED_SHA256[name]:
            raise IdentityHandoffSeamError("identity handoff repaired checksum mismatch: " + name)
        corrected[name] = source
    return corrected
