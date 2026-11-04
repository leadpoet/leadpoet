"""Compose five exact frozen files with verified display-brand attribution."""
from __future__ import annotations

import hashlib

from scripts import arena_reviewer_homepage_profile_alias_seam as homepage

CORRECTION_ID = "verified_display_brand_quote_attribution_v1"
HOST_VALIDATOR_REQUIREMENT = homepage.HOST_VALIDATOR_REQUIREMENT
INVESTIGATOR_FILE = "qualification/scoring/company_evidence_investigator.py"
FILES = (*homepage.FILES, INVESTIGATOR_FILE)
ORIGINAL_SHA256 = {**homepage.ORIGINAL_SHA256, INVESTIGATOR_FILE: '1bc1c5244c2f0633cd4623b65657a0b3d802ce7827ecb7e8b302a52dd981357c'}
REPAIRED_SHA256 = {**homepage.REPAIRED_SHA256, INVESTIGATOR_FILE: '9137a7a39a49f24407fe9b894245a855218858c5a919c7759e08c9d79ce92518'}

# Apply only the reviewed 84463c66a investigator delta to the exact frozen
# f69461adb source. No current grader or runtime source is read.
REPLACEMENTS = (
    (110382, 110382, b'def _quote_names_verified_display_brand(\n    quote: str, url: str, final_url: str, domains: set[str], identity: Mapping[str, Any],\n) -> bool:\n    """Recognize a named brand in a verified \'Brand (part of Parent)\' label.\n\n    This is citation attribution, not a new identity or ownership verdict.\n    Keep the full identity checks for unnamed quotes and all other targets.\n    """\n    name = str(identity.get("verified_name") or "").strip()\n    match = re.fullmatch(r"([^()]+?)\\s+\\(part of ([^()]+)\\)", name, re.I)\n    if not match or _normalized_span(name) != _normalized_span(identity.get("submitted_name")):\n        return False\n    brand = match[1].replace("\xc2\xae", "").replace("\xe2\x84\xa2", "").strip()\n    brand_key = re.sub(r"[^a-z0-9]+", "", _normalized_span(brand))\n    slug = str(identity.get("verified_linkedin_slug") or "").strip().casefold()\n    if (\n        len(brand_key) < 3\n        or not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", slug)\n        or brand_key != slug.replace("-", "")\n        or identity.get("submitted_linkedin_slug") != slug\n        or identity.get("observed_linkedin_slug") not in (None, "", slug)\n    ):\n        return False\n    domain = identity.get("verified_domain")\n    if not domain or any(identity.get(key) != domain for key in ("submitted_domain", "observed_domain")):\n        return False\n    if not all(\n        _registrable_domain(value) == domain\n        and _independently_bound_first_party_url(value, domains, identity)\n        for value in (url, final_url)\n    ):\n        return False\n    # Require the leading named actor/product. Mentioning the parent or a\n    # related company\'s activity does not establish the brand\'s activity.\n    actor = re.match(\n        rf"^(?:through\\s+)?{re.escape(_normalized_span(brand))}"\n        r"(?=\\s|[,;:]|[\'\xe2\x80\x99]s\\s)(?P<rest>.*)",\n        _normalized_span(quote),\n    )\n    return bool(actor and not re.search(\n        r"\\b(?:parents?|owners?|subsidiar(?:y|ies)|sisters?|affiliates?|partners?|customers?)\\b",\n        actor["rest"],\n    ))\n\n\n'),
    (183757, 183838, b'    named_brand = _quote_names_verified_display_brand(\n        quote, url, str((fetched_final_urls or {}).get(url) or ""),\n        first_party_domains, identity_anchor,\n    )\n    if not (same_company_source or _quote_names_company(quote, identity_names)\n            or named_brand):\n'),
    (195738, 195738, b'                and not (\n                    target in {"industry", "required_attribute"}\n                    and status == "VERIFIED"\n                    and (target == "required_attribute" or finding["activity_role"] == "supplier_operator")\n                    and _quote_names_verified_display_brand(\n                        finding["evidence_quote"], evidence_url,\n                        str((fetched_final_urls or {}).get(evidence_url) or ""),\n                        first_party_domains, identity_anchor or {},\n                    )\n                )\n'),
)


class VerifiedDisplayBrandSeamError(ValueError):
    """The frozen five-file input is not one reviewed generation."""


def convert_verified_display_brand_files(sources: dict[str, bytes]) -> dict[str, bytes]:
    if set(sources) != set(FILES):
        raise VerifiedDisplayBrandSeamError("verified display brand requires exactly five frozen modules")
    hashes = {name: hashlib.sha256(source).hexdigest() for name, source in sources.items()}
    if hashes == REPAIRED_SHA256:
        return dict(sources)
    if hashes != ORIGINAL_SHA256:
        raise VerifiedDisplayBrandSeamError("unrecognized or partial frozen verified display brand source")
    repaired = homepage.convert_homepage_profile_alias_files(
        {name: sources[name] for name in homepage.FILES},
    )
    investigator = sources[INVESTIGATOR_FILE]
    for start, end, replacement in reversed(REPLACEMENTS):
        investigator = investigator[:start] + replacement + investigator[end:]
    repaired[INVESTIGATOR_FILE] = investigator
    if {name: hashlib.sha256(source).hexdigest() for name, source in repaired.items()} != REPAIRED_SHA256:
        raise VerifiedDisplayBrandSeamError("verified display brand repair differs from reviewed source")
    return repaired
