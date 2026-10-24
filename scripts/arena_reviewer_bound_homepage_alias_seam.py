"""Apply only the reviewed exact first-party legal/brand identity deltas.

Known original and repaired module hashes are pinned for the frozen and default
source revisions. No whole current grader is copied over frozen grading code.
The image composer still verifies all overlay bytes and unchanged base layers.
"""
from __future__ import annotations

import hashlib

CORRECTION_ID = "bound_homepage_legal_brand_identity_v1"

class BoundHomepageAliasSeamError(ValueError):
    """The source bundle is not one reviewed complete correction state."""

ORIGINAL_TO_REPAIRED_SHA256 = {'qualification/scoring/company_verification.py': {'904cdfefdf180cb1d1345272e8c69aaffdcbf268d29a1bf35716309bd45f91fc': '6445f934b2c61338e3e934be0e78912698dd896c83fa7a59abf8d2450a9a56ad'},
 'qualification/scoring/company_fit_decision.py': {'ad7ce1546f79eccbd10319433404e1b1b53f831351613c4a2d05bdd4c74a0d0a': '1617ce93cd826ff07a70599132e61933861453fc3a26ddc0da4b9cda94ba3ecf'},
 'qualification/scoring/lead_scorer.py': {'7693b9eeb3661cd24f188b03502636771e7ccb40c2bfd5cb0488f42c31422f76': '4b9cf3c9effc93ceeeec72868a0bfe96f497a36c38364d4ee50d284fbafe1b87',
                                          '9bc43046a37ce8512044dd1822626ce3d5cb9f7ee878b9599cd816e4aab74e27': 'ffbf6c7a1f6053300c98e904a713cf7704628b30909832b2f0fdd6d8ccf94b58'},
 'qualification/scoring/company_evidence_investigator.py': {'d77c85831409c971bd1b5fb4419e0651e148f6208d26757e7be78cd7e91dd8dc': '6a48762b6ef873ae659fe95c711921d592957fc6b19da7b56427ee90f113b546',
                                                            '3c55748e82e65a55382aace470721ec39fd11f5085fe51cee017045977978411': '27f1d4558da4cdec377d1e53b2c27bca9de5118bbb845e78f86e2e8bfc0c2614'},
 'qualification/scoring/linkedin_company_size.py': {'b40c00254d9cf00c5868d379666383e2f7c1e7ec3720b4c55e28f71a5a484790': '9247339684c7349701cb38082620d1b628d38d436cace6f0194ebe79d03a9d63'}}

ORIGINAL_BUNDLES = ((('qualification/scoring/company_evidence_investigator.py',
   'd77c85831409c971bd1b5fb4419e0651e148f6208d26757e7be78cd7e91dd8dc'),
  ('qualification/scoring/company_fit_decision.py',
   'ad7ce1546f79eccbd10319433404e1b1b53f831351613c4a2d05bdd4c74a0d0a'),
  ('qualification/scoring/company_verification.py',
   '904cdfefdf180cb1d1345272e8c69aaffdcbf268d29a1bf35716309bd45f91fc'),
  ('qualification/scoring/lead_scorer.py',
   '7693b9eeb3661cd24f188b03502636771e7ccb40c2bfd5cb0488f42c31422f76'),
  ('qualification/scoring/linkedin_company_size.py',
   'b40c00254d9cf00c5868d379666383e2f7c1e7ec3720b4c55e28f71a5a484790')),
 (('qualification/scoring/company_evidence_investigator.py',
   '3c55748e82e65a55382aace470721ec39fd11f5085fe51cee017045977978411'),
  ('qualification/scoring/company_fit_decision.py',
   'ad7ce1546f79eccbd10319433404e1b1b53f831351613c4a2d05bdd4c74a0d0a'),
  ('qualification/scoring/company_verification.py',
   '904cdfefdf180cb1d1345272e8c69aaffdcbf268d29a1bf35716309bd45f91fc'),
  ('qualification/scoring/lead_scorer.py',
   '9bc43046a37ce8512044dd1822626ce3d5cb9f7ee878b9599cd816e4aab74e27'),
  ('qualification/scoring/linkedin_company_size.py',
   'b40c00254d9cf00c5868d379666383e2f7c1e7ec3720b4c55e28f71a5a484790')))

REPAIRED_BUNDLES = ((('qualification/scoring/company_evidence_investigator.py',
   '6a48762b6ef873ae659fe95c711921d592957fc6b19da7b56427ee90f113b546'),
  ('qualification/scoring/company_fit_decision.py',
   '1617ce93cd826ff07a70599132e61933861453fc3a26ddc0da4b9cda94ba3ecf'),
  ('qualification/scoring/company_verification.py',
   '6445f934b2c61338e3e934be0e78912698dd896c83fa7a59abf8d2450a9a56ad'),
  ('qualification/scoring/lead_scorer.py',
   '4b9cf3c9effc93ceeeec72868a0bfe96f497a36c38364d4ee50d284fbafe1b87'),
  ('qualification/scoring/linkedin_company_size.py',
   '9247339684c7349701cb38082620d1b628d38d436cace6f0194ebe79d03a9d63')),
 (('qualification/scoring/company_evidence_investigator.py',
   '27f1d4558da4cdec377d1e53b2c27bca9de5118bbb845e78f86e2e8bfc0c2614'),
  ('qualification/scoring/company_fit_decision.py',
   '1617ce93cd826ff07a70599132e61933861453fc3a26ddc0da4b9cda94ba3ecf'),
  ('qualification/scoring/company_verification.py',
   '6445f934b2c61338e3e934be0e78912698dd896c83fa7a59abf8d2450a9a56ad'),
  ('qualification/scoring/lead_scorer.py',
   'ffbf6c7a1f6053300c98e904a713cf7704628b30909832b2f0fdd6d8ccf94b58'),
  ('qualification/scoring/linkedin_company_size.py',
   '9247339684c7349701cb38082620d1b628d38d436cace6f0194ebe79d03a9d63')))

DELTAS = {
    'qualification/scoring/company_verification.py': [
        ('''    return f"https://www.linkedin.com/company/{slug}"


def _organization_identity_records(value) -> list[dict[str, object]]:
    """Extract bounded root Organization identity records from JSON-LD."""

''', '''    return f"https://www.linkedin.com/company/{slug}"


class _OrganizationIdentityOverflow(ValueError):
    """A JSON-LD document exceeded the bounded identity inspection."""


def _organization_identity_records(value) -> list[dict[str, object]]:
    """Extract bounded root Organization identity records from JSON-LD."""

'''),
        ('''        return []

    roots = document if isinstance(document, list) else [document]
    candidates: list[object] = []
    for root in roots[:20]:
        if not isinstance(root, dict):
            continue
        candidates.append(root)
        graph = root.get("@graph")
        if isinstance(graph, list):
            candidates.extend(graph[:20])

    records: list[dict[str, object]] = []
    for node in candidates:
''', '''        return []

    roots = document if isinstance(document, list) else [document]
    if len(roots) > 20:
        raise _OrganizationIdentityOverflow
    candidates: list[object] = []
    for root in roots:
        if not isinstance(root, dict):
            continue
        candidates.append(root)
        graph = root.get("@graph")
        if isinstance(graph, list):
            if len(graph) > 20:
                raise _OrganizationIdentityOverflow
            candidates.extend(graph)

    records: list[dict[str, object]] = []
    for node in candidates:
'''),
        ('''                "linkedin_urls": linkedins,
            }
        )
        if len(records) == 10:
            break
    return records


''', '''                "linkedin_urls": linkedins,
            }
        )
        if len(records) > 10:
            raise _OrganizationIdentityOverflow
    return records


'''),
        ('''        self.linkedin_urls: list[str] = []
        self.source_hrefs: list[str] = []
        self.organization_records: list[dict[str, object]] = []
        self._json_ld_parts: list[str] | None = None
        self._nonvisible_depth = 0

''', '''        self.linkedin_urls: list[str] = []
        self.source_hrefs: list[str] = []
        self.organization_records: list[dict[str, object]] = []
        self.organization_identity_overflow = False
        self._json_ld_parts: list[str] | None = None
        self._nonvisible_depth = 0

'''),
        ('''            self._in_title = False
        elif tag_name == "script" and self._json_ld_parts is not None:
            json_ld = "".join(self._json_ld_parts)
            records = _organization_identity_records(json_ld)
            self.organization_records.extend(records)
            self.linkedin_urls.extend(_organization_same_as_urls(json_ld))
            self._json_ld_parts = None
''', '''            self._in_title = False
        elif tag_name == "script" and self._json_ld_parts is not None:
            json_ld = "".join(self._json_ld_parts)
            try:
                records = _organization_identity_records(json_ld)
            except _OrganizationIdentityOverflow:
                self.organization_identity_overflow = True
                records = []
            self.organization_records.extend(records)
            self.linkedin_urls.extend(_organization_same_as_urls(json_ld))
            self._json_ld_parts = None
'''),
        ('''    return locators


def _homepage_company_names(
    page_text: str,
    *,
    observed_domain: str = "",
    observed_linkedin: str = "",
) -> list[str]:
    """Return names actually observed in title or first-party metadata."""

''', '''    return locators


def _homepage_organization_url(
    raw_url: object, observed_domain: str, homepage_url: str = "",
) -> str:
    """Resolve relative metadata only against the fetched homepage origin."""

    if not isinstance(raw_url, str) or not 0 < len(raw_url) <= 2048:
        return ""
    try:
        parts = urlsplit(raw_url)
        if not parts.scheme and not parts.netloc:
            base = public_http_url(homepage_url)
            base_parts = urlsplit(base)
            if base_parts.scheme != "https":
                return ""
            organization_url = public_http_url(urljoin(base, raw_url))
            resolved = urlsplit(organization_url)
            if (resolved.scheme, resolved.netloc) != (
                base_parts.scheme, base_parts.netloc
            ):
                return ""
        else:
            organization_url = public_http_url(raw_url)
        if _registrable_domain(organization_url) == observed_domain:
            return organization_url
    except (NormalizationError, TypeError, ValueError):
        pass
    return ""


def _bound_homepage_organization_records(
    parser: _HomepageIdentityParser,
    observed_domain: str,
    observed_linkedin: str,
    homepage_url: str = "",
) -> list[dict[str, object]]:
    """Bind one unambiguous legal/brand pair to the homepage and LinkedIn."""

    if (
        not observed_domain or not observed_linkedin
        or parser.organization_identity_overflow
        or len(parser.organization_records) > 10
    ):
        return []
    records: list[dict[str, object]] = []
    brands, legal_names = set(), set()
    for record in parser.organization_records:
        name = record.get("name")
        if (
            not isinstance(name, str)
            or not 2 < len(name.strip()) <= _MAX_ORGANIZATION_NAME_LENGTH
            or not _company_name(name)
            or record.get("linkedin_urls") != [observed_linkedin]
        ):
            continue
        organization_url = _homepage_organization_url(
            record.get("url"), observed_domain, homepage_url,
        )
        if not organization_url:
            continue
        brands.add(_company_name(name))
        legal = record.get("legal_name")
        if (
            isinstance(legal, str)
            and 2 < len(legal.strip()) <= _MAX_ORGANIZATION_NAME_LENGTH
            and _company_name(legal)
        ):
            legal_names.add(_company_name(legal))
            records.append({**record, "url": organization_url})
    return records if len(brands) == len(legal_names) == 1 else []


def _homepage_company_names(
    page_text: str,
    *,
    observed_domain: str = "",
    observed_linkedin: str = "",
    homepage_url: str = "",
) -> list[str]:
    """Return names actually observed in title or first-party metadata."""

'''),
        ('''    except Exception:
        return []
    candidates = [*parser.metadata_names, *parser.copyright_legal_names]
    # A tagline-only title can omit the operating brand even though the root
    # Organization record explicitly binds it to this domain and LinkedIn.
    # Do not promote legal names, nested organizations, or conflicting records.
    organization_names: dict[str, str] = {}
    if observed_domain and observed_linkedin:
        for record in parser.organization_records:
            name = record.get("name")
            organization_url = record.get("url")
            if (
                not isinstance(name, str)
                or not 2 < len(name.strip()) <= _MAX_ORGANIZATION_NAME_LENGTH
                or not isinstance(organization_url, str)
                or not 0 < len(organization_url) <= 2048
                or record.get("linkedin_urls") != [observed_linkedin]
            ):
                continue
            try:
                organization_domain = _registrable_domain(
                    public_http_url(organization_url)
                )
            except (NormalizationError, TypeError, ValueError):
                continue
            if organization_domain == observed_domain and _company_name(name):
                organization_names.setdefault(_company_name(name), name.strip())
        if len(organization_names) == 1:
            candidates = [*organization_names.values(), *candidates]
    if parser.title:
        candidates.append(parser.title)
        candidates.extend(
''', '''    except Exception:
        return []
    candidates = [*parser.metadata_names, *parser.copyright_legal_names]
    # Preserve brand discovery while requiring a unique explicit legal relation.
    organization_names: dict[str, str] = {}
    if observed_domain and observed_linkedin:
        for record in parser.organization_records:
            name = record.get("name")
            if (
                isinstance(name, str)
                and 2 < len(name.strip()) <= _MAX_ORGANIZATION_NAME_LENGTH
                and _company_name(name)
                and record.get("linkedin_urls") == [observed_linkedin]
                and _homepage_organization_url(
                    record.get("url"), observed_domain, homepage_url,
                )
            ):
                organization_names.setdefault(_company_name(name), name.strip())
    records = _bound_homepage_organization_records(
        parser, observed_domain, observed_linkedin, homepage_url,
    )
    candidates = [
        *(organization_names.values() if len(organization_names) == 1 else ()),
        *(str(record["legal_name"]).strip() for record in records), *candidates,
    ]
    if parser.title:
        candidates.append(parser.title)
        candidates.extend(
'''),
        ('''

def _bound_homepage_organization_identity(
    page_text: str, observed_domain: str, observed_linkedins: list[str]
) -> dict[str, str]:
    """Keep one first-party Organization identity for later conflict checks."""

''', '''

def _bound_homepage_organization_identity(
    page_text: str, observed_domain: str, observed_linkedins: list[str],
    homepage_url: str = "",
) -> dict[str, str]:
    """Keep one first-party Organization identity for later conflict checks."""

'''),
        ('''        return {}
    bound_names = set()
    for record in parser.organization_records:
        name, url = record.get("name"), record.get("url")
        if not isinstance(name, str) or not isinstance(url, str):
            continue
        try:
            if _registrable_domain(public_http_url(url)) != observed_domain:
                continue
        except (NormalizationError, TypeError, ValueError):
            continue
        linkedins = record.get("linkedin_urls")
        if linkedins and linkedins != observed_linkedins:
''', '''        return {}
    bound_names = set()
    for record in parser.organization_records:
        name = record.get("name")
        if not isinstance(name, str) or not _homepage_organization_url(
            record.get("url"), observed_domain, homepage_url,
        ):
            continue
        linkedins = record.get("linkedin_urls")
        if linkedins and linkedins != observed_linkedins:
'''),
        ('''    observed_name: str,
    observed_linkedin: str,
    company_quality: bool = False,
) -> list[str]:
    """Return legal names explicitly bound to the verified homepage entity."""

''', '''    observed_name: str,
    observed_linkedin: str,
    company_quality: bool = False,
    homepage_url: str = "",
) -> list[str]:
    """Return legal names explicitly bound to the verified homepage entity."""

'''),
        ('''    except Exception:
        return []
    aliases: list[str] = []
    for record in parser.organization_records[:10]:
        brand = record.get("name")
        legal_name = record.get("legal_name")
        organization_url = record.get("url")
        linkedins = record.get("linkedin_urls")
        if not all(
            isinstance(value, str) and value.strip() and len(value.strip()) <= limit
            for value, limit in (
                (brand, _MAX_ORGANIZATION_NAME_LENGTH),
                (legal_name, _MAX_ORGANIZATION_NAME_LENGTH),
                (organization_url, 2048),
            )
        ) or not isinstance(linkedins, list):
            continue
        try:
            organization_domain = _registrable_domain(organization_url)
        except Exception:
            continue
        if organization_domain != observed_domain or observed_linkedin not in linkedins:
            continue
        brand_receipt = evaluate_company_identity(
            submitted_name=observed_name,
            submitted_website=f"https://{observed_domain}",
            submitted_linkedin=observed_linkedin,
            observed_name=brand,
            observed_website=organization_url,
            observed_linkedin=observed_linkedin,
            evidence_source="company_homepage",
            company_quality=company_quality,
        )
        if brand_receipt["decision"] != "match":
            continue
        alias = legal_name.strip()
        if alias not in aliases:
            aliases.append(alias)
        if len(aliases) == _MAX_ORGANIZATION_LEGAL_NAME_ALIASES:
            break
    return aliases
''', '''    except Exception:
        return []
    aliases: list[str] = []
    for record in _bound_homepage_organization_records(
        parser, observed_domain, observed_linkedin, homepage_url,
    ):
        brand, legal_name = record.get("name"), record.get("legal_name")
        if (
            not isinstance(legal_name, str)
            or not 2 < len(legal_name.strip()) <= _MAX_ORGANIZATION_NAME_LENGTH
            or _company_name(observed_name) not in {
                _company_name(brand), _company_name(legal_name),
            }
        ):
            continue
        if legal_name.strip() not in aliases:
            aliases.append(legal_name.strip())
        if len(aliases) == _MAX_ORGANIZATION_LEGAL_NAME_ALIASES:
            break
    return aliases
'''),
        ('''    observed_names = _homepage_company_names(
        text,
        observed_domain=observed_domain,
        observed_linkedin=(
            observed_linkedins[0] if len(observed_linkedins) == 1 else ""
        ),
''', '''    observed_names = _homepage_company_names(
        text,
        observed_domain=observed_domain,
        homepage_url=observed_url,
        observed_linkedin=(
            observed_linkedins[0] if len(observed_linkedins) == 1 else ""
        ),
'''),
        ('''            verified_homepage_transport_domain=domain,
        )
    bound_organization = _bound_homepage_organization_identity(
        text, observed_domain, observed_linkedins
    )
    identity_receipts = [
        evaluate_company_identity(
''', '''            verified_homepage_transport_domain=domain,
        )
    bound_organization = _bound_homepage_organization_identity(
        text, observed_domain, observed_linkedins, observed_url
    )
    identity_receipts = [
        evaluate_company_identity(
'''),
        ('''            observed_name=matched["observed_name"],
            observed_linkedin=matched_linkedin,
            company_quality=company_quality,
        )
        if legal_name_aliases:
            matched_receipt["verified_legal_name_aliases"] = legal_name_aliases
        if homepage_navigation_locator_sink is not None:
            homepage_navigation_locator_sink.extend(
                _homepage_navigation_locators(
''', '''            observed_name=matched["observed_name"],
            observed_linkedin=matched_linkedin,
            company_quality=company_quality,
            homepage_url=observed_url,
        )
        if legal_name_aliases:
            matched_receipt["verified_legal_name_aliases"] = legal_name_aliases
            parser = _HomepageIdentityParser()
            parser.feed(text[:_MAX_BYTES])
            parser.close()
            bound_records = _bound_homepage_organization_records(
                parser, matched["observed_domain"], matched_linkedin, observed_url,
            )
            if bound_records:
                matched_receipt["verified_brand_name"] = str(
                    bound_records[0]["name"]
                ).strip()
        if homepage_navigation_locator_sink is not None:
            homepage_navigation_locator_sink.extend(
                _homepage_navigation_locators(
'''),
    ],
    'qualification/scoring/company_fit_decision.py': [
        ('''        else:
            break
    return "".join(words)


def _verified_parenthetical_name_alignment(
''', '''        else:
            break
    return "".join(words)


def verified_homepage_brand_name(identity: Mapping[str, Any]) -> str:
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


def _verified_parenthetical_name_alignment(
'''),
    ],
    'qualification/scoring/lead_scorer.py': [
        ('''    verify_company_exists,
)
from qualification.scoring.company_fit_decision import (
    COMPANY_FIT_MATCH,
    COMPANY_FIT_MISMATCH,
    COMPANY_FIT_UNAVAILABLE,
''', '''    verify_company_exists,
)
from qualification.scoring.company_fit_decision import (
    verified_homepage_brand_name,
    COMPANY_FIT_MATCH,
    COMPANY_FIT_MISMATCH,
    COMPANY_FIT_UNAVAILABLE,
'''),
        ('''        )
        if aliases:
            anchor["verified_legal_name_aliases"] = aliases
    return anchor


''', '''        )
        if aliases:
            anchor["verified_legal_name_aliases"] = aliases
            brand = verified_homepage_brand_name({
                **anchor, "verified_brand_name": receipt.get("verified_brand_name"),
            })
            if brand:
                anchor["verified_brand_name"] = brand
    return anchor


'''),
        ('''    return bool(
        final_profile and linkedin_company_page_slug(final_profile) == slug
        and quote and _quote_occurs(quote, pages.get(url, ""))
        and _quote_names_company(quote, {_compact_company_name(identity.get("normalized_name"))})
        and _quote_supports_headcount(quote, finding["observed_value"])
    )

''', '''    return bool(
        final_profile and linkedin_company_page_slug(final_profile) == slug
        and quote and _quote_occurs(quote, pages.get(url, ""))
        and _quote_names_company(quote, {
            _compact_company_name(identity.get("normalized_name")),
            _compact_company_name(verified_homepage_brand_name(identity)),
        } - {""})
        and _quote_supports_headcount(quote, finding["observed_value"])
    )

'''),
        ('''        and _dimension_web_evidence(projected, "employee_size") == {"url": url, "quote": quote}
        and quote and _quote_occurs(quote, pages.get(url, ""))
        and (
            _quote_names_company(quote, {_compact_company_name(identity.get("normalized_name"))})
            or _profile_body_matches_company_identity(pages.get(url, ""), identity)
        )
        and _quote_supports_headcount(quote, finding["observed_value"])
''', '''        and _dimension_web_evidence(projected, "employee_size") == {"url": url, "quote": quote}
        and quote and _quote_occurs(quote, pages.get(url, ""))
        and (
            _quote_names_company(quote, {
                _compact_company_name(identity.get("normalized_name")),
                _compact_company_name(verified_homepage_brand_name(identity)),
            } - {""})
            or _profile_body_matches_company_identity(pages.get(url, ""), identity)
        )
        and _quote_supports_headcount(quote, finding["observed_value"])
'''),
    ],
    'qualification/scoring/company_evidence_investigator.py': [
        ('''from lab_arena.operations import OPENROUTER_MAX_CONTENT_CHARS
from leadpoet_verifier.identity.normalization import NormalizationError, normalize_host
from qualification.competition_models import public_http_url
from qualification.scoring.company_fit_decision import _company_name
from qualification.scoring.checked_sources import checked_source_provenance
from qualification.scoring.country_data import US_STATES
from qualification.scoring.company_verification import (
''', '''from lab_arena.operations import OPENROUTER_MAX_CONTENT_CHARS
from leadpoet_verifier.identity.normalization import NormalizationError, normalize_host
from qualification.competition_models import public_http_url
from qualification.scoring.company_fit_decision import (
    _company_name, verified_homepage_brand_name,
)
from qualification.scoring.checked_sources import checked_source_provenance
from qualification.scoring.country_data import US_STATES
from qualification.scoring.company_verification import (
'''),
        ('''                    (verified_homepage_identity or {}).get("normalized_name")
                ),
            ),
        )
        if normalized
    }
''', '''                    (verified_homepage_identity or {}).get("normalized_name")
                ),
            ),
            re.sub(
                r"[^a-z0-9]+", "",
                _normalized_span(
                    verified_homepage_brand_name(verified_homepage_identity or {})
                ),
            ),
        )
        if normalized
    }
'''),
    ],
    'qualification/scoring/linkedin_company_size.py': [
        ('''from qualification.employee_buckets import LINKEDIN_EMPLOYEE_BUCKETS
from qualification.company_quality import canonical_us_state
from qualification.scoring.pre_checks import _resolve_country
from qualification.scoring.company_fit_decision import _company_name


logger = logging.getLogger(__name__)
''', '''from qualification.employee_buckets import LINKEDIN_EMPLOYEE_BUCKETS
from qualification.company_quality import canonical_us_state
from qualification.scoring.pre_checks import _resolve_country
from qualification.scoring.company_fit_decision import (
    _company_name, verified_homepage_brand_name,
)


logger = logging.getLogger(__name__)
'''),
        ('''    """Bind the profile's own website or title, never a name mentioned in prose."""

    domain = _canonical_company_domain(identity.get("registrable_dns_domain"))
    names = {_company_name(identity.get("normalized_name"))}
    aliases = identity.get("verified_legal_name_aliases")
    if isinstance(aliases, list):
        names.update(_company_name(value) for value in aliases[:3])
''', '''    """Bind the profile's own website or title, never a name mentioned in prose."""

    domain = _canonical_company_domain(identity.get("registrable_dns_domain"))
    names = {
        _company_name(identity.get("normalized_name")),
        _company_name(verified_homepage_brand_name(identity)),
    }
    aliases = identity.get("verified_legal_name_aliases")
    if isinstance(aliases, list):
        names.update(_company_name(value) for value in aliases[:3])
'''),
    ],
}

def convert_bound_homepage_legal_alias_files(sources: dict[str, bytes]) -> dict[str, bytes]:
    if set(sources) != set(DELTAS):
        raise BoundHomepageAliasSeamError("bound identity sources are incomplete")
    bundle = tuple(sorted((name, hashlib.sha256(source).hexdigest())
                          for name, source in sources.items()))
    if bundle not in ORIGINAL_BUNDLES and bundle not in REPAIRED_BUNDLES:
        raise BoundHomepageAliasSeamError(
            "unrecognized or partially corrected bound identity source bundle"
        )
    states = set()
    converted = {}
    for name, source in sources.items():
        digest = hashlib.sha256(source).hexdigest()
        known = ORIGINAL_TO_REPAIRED_SHA256[name]
        if digest in known.values():
            states.add("repaired")
            converted[name] = source
            continue
        if digest not in known:
            raise BoundHomepageAliasSeamError("unrecognized bound identity source: " + name)
        states.add("original")
        text = source.decode("utf-8")
        for before, after in DELTAS[name]:
            if text.count(before) != 1:
                raise BoundHomepageAliasSeamError("ambiguous bound identity correction: " + name)
            text = text.replace(before, after, 1)
        result = text.encode("utf-8")
        if hashlib.sha256(result).hexdigest() != known[digest]:
            raise BoundHomepageAliasSeamError("bound identity repair differs from reviewed source")
        compile(result, name, "exec")
        converted[name] = result
    if len(states) != 1:
        raise BoundHomepageAliasSeamError("partially corrected bound identity sources")
    return converted
