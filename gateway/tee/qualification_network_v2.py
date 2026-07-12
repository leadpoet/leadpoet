"""Authenticated DNS and registration-data adapters for qualification V2.

The protected qualification functions still call ``dns.resolver.resolve`` and
``whois.whois``.  A Nitro runner has no direct network, and port-43 WHOIS is
plaintext, so this measured adapter replaces only those I/O seams with DoH and
RDAP over the coordinator's authenticated provider broker.  Returned objects
retain the attributes consumed by the unchanged qualification code.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import re
import threading
from typing import Any, Mapping, Optional, Sequence
import urllib.error
import urllib.parse
import urllib.request


DOH_ENDPOINT = "https://cloudflare-dns.com/dns-query"
RDAP_ENDPOINT = "https://rdap.org/domain/"
MAX_DNS_RESPONSE_BYTES = 1024 * 1024
MAX_RDAP_RESPONSE_BYTES = 8 * 1024 * 1024
MAX_RDAP_REDIRECTS = 5
_DOMAIN_RE = re.compile(r"^[A-Za-z0-9_*-](?:[A-Za-z0-9_.*-]{0,251}[A-Za-z0-9_*.-])?$")


class QualificationNetworkV2Error(RuntimeError):
    """An authenticated qualification network response is malformed."""


@dataclass(frozen=True)
class _RdapWhoisRecord:
    creation_date: Optional[datetime]
    updated_date: Optional[datetime]
    registrar: Optional[str]
    name_servers: Optional[list]


def _domain(value: Any, *, allow_service_label: bool = True) -> str:
    raw = str(value or "").strip().rstrip(".")
    if not raw or not _DOMAIN_RE.fullmatch(raw):
        raise QualificationNetworkV2Error("DNS name is invalid")
    try:
        labels = raw.split(".")
        normalized = ".".join(
            label if (allow_service_label and label.startswith("_"))
            else label.encode("idna").decode("ascii")
            for label in labels
        ).lower()
    except (UnicodeError, ValueError) as exc:
        raise QualificationNetworkV2Error("DNS name is invalid") from exc
    if len(normalized) > 253 or any(not label for label in normalized.split(".")):
        raise QualificationNetworkV2Error("DNS name is invalid")
    return normalized


def _read_response(response: Any, limit: int) -> bytes:
    body = response.read(limit + 1)
    if len(body) > limit:
        raise QualificationNetworkV2Error("authenticated response exceeds limit")
    return body


def resolve_doh(name: Any, rdtype: Any, *args: Any, **kwargs: Any):
    """Return dnspython rdata objects from an authenticated DNS JSON answer."""

    import dns.rdata
    import dns.rdataclass
    import dns.rdatatype
    import dns.resolver

    if args:
        raise QualificationNetworkV2Error("positional resolver options are unsupported")
    unsupported = set(kwargs) - {
        "rdclass",
        "tcp",
        "source",
        "raise_on_no_answer",
        "source_port",
        "lifetime",
        "search",
    }
    if unsupported:
        raise QualificationNetworkV2Error(
            "unsupported resolver options: %s" % ",".join(sorted(unsupported))
        )
    query_name = _domain(name)
    try:
        query_type = dns.rdatatype.to_text(dns.rdatatype.from_text(str(rdtype)))
    except Exception as exc:
        raise QualificationNetworkV2Error("DNS record type is invalid") from exc
    if query_type not in {"A", "MX", "TXT"}:
        raise QualificationNetworkV2Error("DNS record type is not measured")
    query = urllib.parse.urlencode({"name": query_name, "type": query_type})
    request = urllib.request.Request(
        DOH_ENDPOINT + "?" + query,
        headers={"Accept": "application/dns-json"},
        method="GET",
    )
    timeout = kwargs.get("lifetime")
    try:
        with urllib.request.urlopen(request, timeout=timeout or 30) as response:
            payload = json.loads(_read_response(response, MAX_DNS_RESPONSE_BYTES))
    except urllib.error.URLError as exc:
        if "timeout" in str(exc).lower():
            raise dns.resolver.Timeout(str(exc)) from exc
        raise dns.resolver.NoNameservers(str(exc)) from exc
    except (json.JSONDecodeError, UnicodeDecodeError, TypeError) as exc:
        raise dns.resolver.NoNameservers("authenticated DNS response is malformed") from exc
    if not isinstance(payload, Mapping):
        raise dns.resolver.NoNameservers("authenticated DNS response is malformed")
    status = payload.get("Status")
    if status == 3:
        raise dns.resolver.NXDOMAIN(qnames=[query_name])
    if status != 0:
        raise dns.resolver.NoNameservers(
            "authenticated DNS response status is %s" % status
        )
    answers = payload.get("Answer") or []
    if not isinstance(answers, list):
        raise dns.resolver.NoNameservers("authenticated DNS answer is malformed")
    expected_type = dns.rdatatype.from_text(query_type)
    records = []
    for answer in answers:
        if not isinstance(answer, Mapping) or answer.get("type") != expected_type:
            continue
        data = answer.get("data")
        if not isinstance(data, str) or not data:
            raise dns.resolver.NoNameservers("authenticated DNS rdata is malformed")
        try:
            records.append(
                dns.rdata.from_text(
                    dns.rdataclass.IN,
                    expected_type,
                    data,
                )
            )
        except Exception as exc:
            raise dns.resolver.NoNameservers(
                "authenticated DNS rdata is malformed"
            ) from exc
    if not records:
        if kwargs.get("raise_on_no_answer", True):
            raise dns.resolver.NoAnswer()
        return []
    return records


def _rdap_datetime(value: Any) -> Optional[datetime]:
    if not isinstance(value, str) or not value.strip():
        return None
    normalized = value.strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def _entity_name(entity: Mapping[str, Any]) -> Optional[str]:
    vcard = entity.get("vcardArray")
    if not isinstance(vcard, list) or len(vcard) != 2 or not isinstance(vcard[1], list):
        return None
    for field in vcard[1]:
        if (
            isinstance(field, list)
            and len(field) >= 4
            and str(field[0]).lower() in {"fn", "org"}
            and isinstance(field[3], str)
            and field[3].strip()
        ):
            return field[3].strip()
    return None


def whois_via_rdap(domain_name: Any) -> _RdapWhoisRecord:
    """Return the legacy WHOIS attributes from authenticated RDAP JSON."""

    domain = _domain(domain_name, allow_service_label=False)
    if "." not in domain or "_" in domain or "*" in domain:
        raise QualificationNetworkV2Error("RDAP domain is invalid")
    url = RDAP_ENDPOINT + urllib.parse.quote(domain, safe=".-")
    for redirect_count in range(MAX_RDAP_REDIRECTS + 1):
        request = urllib.request.Request(
            url,
            headers={"Accept": "application/rdap+json, application/json"},
            method="GET",
        )
        try:
            response = urllib.request.urlopen(request, timeout=30)
        except urllib.error.HTTPError as exc:
            raise QualificationNetworkV2Error(
                "authenticated RDAP response status is %s" % exc.code
            ) from exc
        with response:
            status = int(getattr(response, "status", response.getcode()))
            if status in {301, 302, 303, 307, 308}:
                location = response.headers.get("location")
                if not location or redirect_count >= MAX_RDAP_REDIRECTS:
                    raise QualificationNetworkV2Error("RDAP redirect is invalid")
                next_url = urllib.parse.urljoin(url, location)
                if urllib.parse.urlsplit(next_url).scheme != "https":
                    raise QualificationNetworkV2Error(
                        "RDAP redirect attempted plaintext transport"
                    )
                url = next_url
                continue
            if status != 200:
                raise QualificationNetworkV2Error(
                    "authenticated RDAP response status is %s" % status
                )
            try:
                payload = json.loads(_read_response(response, MAX_RDAP_RESPONSE_BYTES))
            except (json.JSONDecodeError, UnicodeDecodeError, TypeError) as exc:
                raise QualificationNetworkV2Error(
                    "authenticated RDAP response is malformed"
                ) from exc
        if not isinstance(payload, Mapping):
            raise QualificationNetworkV2Error("authenticated RDAP response is malformed")
        events = payload.get("events") or []
        creation_date = None
        updated_date = None
        for event in events if isinstance(events, list) else ():
            if not isinstance(event, Mapping):
                continue
            action = str(event.get("eventAction") or "").lower()
            event_date = _rdap_datetime(event.get("eventDate"))
            if action in {"registration", "registered"} and creation_date is None:
                creation_date = event_date
            elif action in {"last changed", "last update", "changed"} and updated_date is None:
                updated_date = event_date
        registrar = None
        entities = payload.get("entities") or []
        for entity in entities if isinstance(entities, list) else ():
            if not isinstance(entity, Mapping):
                continue
            roles = {str(role).lower() for role in (entity.get("roles") or [])}
            if "registrar" in roles:
                registrar = _entity_name(entity)
                if registrar:
                    break
        nameservers = []
        for row in payload.get("nameservers") or []:
            if isinstance(row, Mapping) and isinstance(row.get("ldhName"), str):
                nameservers.append(row["ldhName"])
        return _RdapWhoisRecord(
            creation_date=creation_date,
            updated_date=updated_date,
            registrar=registrar,
            name_servers=nameservers or None,
        )
    raise QualificationNetworkV2Error("RDAP redirect loop did not terminate")


class SecureQualificationNetworkV2:
    """Install and restore the two raw-network seams used by qualification."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._restore = []
        self._installed = False

    def install(self) -> None:
        with self._lock:
            if self._installed:
                return
            import dns.resolver
            import whois

            original_resolve = dns.resolver.resolve
            original_whois = whois.whois
            dns.resolver.resolve = resolve_doh
            whois.whois = whois_via_rdap
            self._restore.extend(
                (
                    lambda: setattr(dns.resolver, "resolve", original_resolve),
                    lambda: setattr(whois, "whois", original_whois),
                )
            )
            self._installed = True

    def restore(self) -> None:
        with self._lock:
            while self._restore:
                self._restore.pop()()
            self._installed = False
