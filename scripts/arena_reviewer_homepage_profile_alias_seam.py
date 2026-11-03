"""Compose only exact old-homepage canonical profile and native denial repairs."""
from __future__ import annotations

import hashlib

from scripts import arena_reviewer_intent_target_denial_seam as intent

CORRECTION_ID = "exact_old_homepage_linkedin_canonical_alias_v1"
HOST_VALIDATOR_REQUIREMENT = "matching_gateway_and_executing_validators_target_denial_v1"
ALIAS_FILES = ("qualification/scoring/lead_scorer.py", "qualification/scoring/linkedin_company_size.py")
FILES = (*ALIAS_FILES, *intent.FILES)
ALIAS_ORIGINAL_SHA256 = {'qualification/scoring/lead_scorer.py': '203272ceb209a4b44f6ef91643cd819776486ea4a9cb894921488a7918f90c15', 'qualification/scoring/linkedin_company_size.py': '9c74e7da3d036ea550b2f4b84d6d4615b13d490d71cf200ab3c1cea1ffabc725'}
ALIAS_REPAIRED_SHA256 = {'qualification/scoring/lead_scorer.py': 'f2e31061630a73ff0c5238c1a952c94f0f50ff1da3b0d3e0892098e9696f1726', 'qualification/scoring/linkedin_company_size.py': '1362440d281b961f820b765fc3fbda4439709bf6d9500c82e09cd2d491612f67'}
ORIGINAL_SHA256 = {**ALIAS_ORIGINAL_SHA256, **intent.ORIGINAL_SHA256}
REPAIRED_SHA256 = {**ALIAS_REPAIRED_SHA256, **intent.REPAIRED_SHA256}

# Byte ranges index only the complete checksum-bound original modules.
# No current grader file is read; every other byte is retained verbatim.
REPLACEMENTS = {
    'qualification/scoring/lead_scorer.py': (
        (191536, 191658, b'        (\n            (slug.isascii() and slug.isdigit()\n             and observed.get("reason_code") == "identity_linkedin_alias_unresolved")\n            or (slug.isascii() and slug and not slug.isdigit()\n                and observed.get("reason_code") in {"identity_not_proven", "identity_linkedin_alias_unresolved"})\n        )\n'),
        (192059, 192103, b'        and (\n            (slug.isdigit() and proof.get("company_id") == slug)\n            or (not slug.isdigit()\n                and re.fullmatch(r"[1-9][0-9]*", proof.get("company_id", "")))\n        )\n'),
        (193005, 193231, b'        # The exact old-locator main profile proves its canonical web identity.\n        # A short homepage brand name cannot turn that independently bound\n        # alias into an unresolved conflict.\n'),
        (193845, 193924, b'    """Locate exact numeric or old-vanity proof; the lookup grants no match."""\n'),
        (194221, 194221, b'    numeric_locator = (\n        slug.isascii() and slug.isdigit()\n        and observed.get("reason_code") == "identity_linkedin_alias_unresolved"\n    )\n    vanity_locator = (\n        slug.isascii() and slug and not slug.isdigit()\n        and _strict_linkedin_company_profile_url(f"https://www.linkedin.com/company/{slug}")\n        and observed.get("reason_code") in {"identity_not_proven", "identity_linkedin_alias_unresolved"}\n    )\n'),
        (194230, 194358, b'        not (numeric_locator or vanity_locator)\n'),
    ),
    'qualification/scoring/linkedin_company_size.py': (
        (10729, 10802, b'def _structured_company_elements(value: Any, *, strict_main: bool = False) -> list[Mapping[str, Any]]:\n'),
        (10912, 10912, b'        if strict_main and _structured_provider_reported_error(current):\n            return []\n        if strict_main:\n            status = current.get("status")\n            if ("status" in current and not (\n                (type(status) is int and status == 200)\n                or status in ("completed", "success")\n            )) or ("job_id" in current and "status" not in current):\n                return []\n            wrapper_count = sum(isinstance(current.get(key), Mapping) for key in (\n                "toolResponse", "rawV2", "raw", "result", "data", "output",\n            ))\n            if (wrapper_count > 1\n                or (wrapper_count and ("element" in current or "elements" in current))\n                or ("element" in current and "elements" in current)):\n                return []\n'),
        (11323, 11323, b'            if strict_main and (len(elements) != 1 or not isinstance(elements[0], Mapping)):\n                return []\n'),
        (11404, 11404, b'        if strict_main and ("element" in current or "elements" in current):\n            return []\n'),
        (16348, 16427, b'    """Project one exact main profile, including a bound canonical alias."""\n'),
        (18391, 18391, b'    if not requested_slug.isdigit():\n        # An exact old vanity lookup can return the main profile\'s canonical\n        # URL. Require its full independent identity; a same-domain name or\n        # description alone cannot bind two different company-page slugs.\n        if (\n            not requested_slug.isascii()\n            or not observed_slug or observed_slug.isdigit()\n            or returned_profile != observed_profile\n            or element.get("universalName") != observed_slug\n            or not re.fullmatch(r"[1-9][0-9]*", company_id)\n            or _company_name(name) != _company_name(expected_company_name)\n            or not _company_name(expected_company_name)\n            or not website_matches\n            or _structured_company_elements(payload, strict_main=True) != [element]\n        ):\n            return None\n        return {\n            "name": name,\n            "provider": STRUCTURED_PROFILE_PROVIDER,\n            "source_field": STRUCTURED_PROFILE_IDENTITY_SOURCE_FIELD,\n            "url": returned_profile,\n            "website": f"https://{domain}/",\n            "company_id": company_id,\n            "requested_url": profile_url,\n        }\n'),
    ),
}


class HomepageProfileAliasSeamError(ValueError):
    """The frozen four-file composition is not a reviewed generation."""


def convert_homepage_profile_alias_files(sources: dict[str, bytes]) -> dict[str, bytes]:
    if set(sources) != set(FILES):
        raise HomepageProfileAliasSeamError("homepage profile alias requires exactly four frozen modules")
    aliases = {name: sources[name] for name in ALIAS_FILES}
    hashes = {name: hashlib.sha256(source).hexdigest() for name, source in aliases.items()}
    target_sources = {name: sources[name] for name in intent.FILES}
    target_hashes = {name: hashlib.sha256(source).hexdigest() for name, source in target_sources.items()}
    if hashes == ALIAS_REPAIRED_SHA256:
        if target_hashes != intent.REPAIRED_SHA256:
            raise HomepageProfileAliasSeamError("corrected alias source lacks the complete target-denial repair")
    elif hashes == ALIAS_ORIGINAL_SHA256:
        for name, replacements in REPLACEMENTS.items():
            for start, end, replacement in reversed(replacements):
                aliases[name] = aliases[name][:start] + replacement + aliases[name][end:]
    else:
        raise HomepageProfileAliasSeamError("unrecognized or partial frozen homepage profile alias source")
    repaired = {**aliases, **intent.convert_intent_target_denial_files(target_sources)}
    if {name: hashlib.sha256(source).hexdigest() for name, source in repaired.items()} != REPAIRED_SHA256:
        raise HomepageProfileAliasSeamError("homepage profile alias repair differs from reviewed source")
    return repaired
