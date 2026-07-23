from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from html.parser import HTMLParser
from urllib.parse import urljoin

from .models import WebsiteObservation
from .network import FetchedPage
from .normalization import normalize_host, normalize_linkedin_company_url, normalize_url


_AGGREGATOR_DOMAINS = frozenset({
    "about.me", "beacons.ai", "bio.link", "crunchbase.com", "facebook.com",
    "instagram.com", "linktr.ee", "linkedin.com", "medium.com", "x.com",
})
_PARKED_MARKERS = (
    "domain is for sale", "buy this domain", "parked free", "sedo domain parking",
    "this domain may be for sale", "make an offer on this domain",
)


class _MetadataParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.canonical_urls: list[str] = []
        self.linkedin_urls: list[str] = []
        self.outbound_urls: list[str] = []
        self.names: list[str] = []
        self._json_ld = False
        self._json_ld_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = {key.lower(): value or "" for key, value in attrs}
        if tag.lower() == "link" and "canonical" in values.get("rel", "").lower().split():
            self.canonical_urls.append(values.get("href", ""))
        if tag.lower() == "meta":
            key = (values.get("property") or values.get("name") or "").lower()
            if key == "og:site_name" and values.get("content"):
                self.names.append(values["content"][:300])
        if tag.lower() == "a" and values.get("href"):
            self.outbound_urls.append(values["href"])
        if (
            tag.lower() == "script"
            and values.get("type", "").lower() == "application/ld+json"
        ):
            self._json_ld = True
            self._json_ld_parts = []

    def handle_data(self, data: str) -> None:
        if self._json_ld and sum(map(len, self._json_ld_parts)) < 131072:
            self._json_ld_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() != "script" or not self._json_ld:
            return
        self._json_ld = False
        try:
            payload = json.loads("".join(self._json_ld_parts))
        except (json.JSONDecodeError, UnicodeError):
            return
        self._visit_json_ld(payload)

    def _visit_json_ld(self, value: object) -> None:
        if isinstance(value, list):
            for item in value[:32]:
                self._visit_json_ld(item)
            return
        if not isinstance(value, dict):
            return
        graph = value.get("@graph")
        if graph is not None:
            self._visit_json_ld(graph)
        raw_type = value.get("@type")
        types = {str(item).lower() for item in raw_type} if isinstance(raw_type, list) else {str(raw_type).lower()}
        if types & {"organization", "corporation", "localbusiness"}:
            for key in ("legalName", "name", "alternateName"):
                name = value.get(key)
                if isinstance(name, str) and name.strip():
                    self.names.append(name.strip()[:300])
            same_as = value.get("sameAs")
            links = same_as if isinstance(same_as, list) else [same_as]
            self.outbound_urls.extend(link for link in links if isinstance(link, str))


def observation_from_page(page: FetchedPage, *, evidence_ref: str) -> WebsiteObservation:
    parser = _MetadataParser()
    text = page.body.decode("utf-8", errors="replace")
    parser.feed(text)
    canonical: list[str] = []
    for raw in parser.canonical_urls[:4]:
        try:
            canonical.append(normalize_url(urljoin(page.final_url, raw)).url)
        except ValueError:
            continue
    canonical = sorted(set(canonical))
    linkedin_urls: set[str] = set()
    outbound_hosts: set[str] = set()
    for raw in parser.outbound_urls[:256]:
        absolute = urljoin(page.final_url, raw)
        try:
            normalized = normalize_url(absolute)
            outbound_hosts.add(normalized.ascii_host)
        except ValueError:
            continue
        try:
            linkedin_urls.add(normalize_linkedin_company_url(absolute).canonical_url)
        except ValueError:
            pass
    final = normalize_url(page.final_url)
    lowered = " ".join(text.casefold().split())[:262144]
    registrable = normalize_host(final.ascii_host).registrable_domain
    return WebsiteObservation(
        requested_url=page.requested_url,
        final_url=page.final_url,
        status=page.status,
        fetched_at=datetime.fromtimestamp(page.fetched_at_epoch_ms / 1000, UTC),
        content_sha256=hashlib.sha256(page.body).hexdigest(),
        names=sorted(set(parser.names))[:24],
        linkedin_company_urls=sorted(linkedin_urls)[:16],
        outbound_hosts=sorted(outbound_hosts)[:64],
        canonical_url=canonical[0] if len(canonical) == 1 else None,
        redirects=page.redirects,
        parked=any(marker in lowered for marker in _PARKED_MARKERS),
        aggregator=registrable in _AGGREGATOR_DOMAINS,
        shared_infrastructure=final.domain.is_private_suffix,
        contradictory_names=[],
        source_evidence_ref=evidence_ref,
    )
