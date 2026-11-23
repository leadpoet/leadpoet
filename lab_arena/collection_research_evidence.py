"""Build judge evidence from passive, gateway-observed fetches of the current run."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from hashlib import sha256
import re
from urllib.parse import urlsplit

from lab_arena import contracts, deepline_catalog
from leadpoet_verifier.identity.normalization import (
    NormalizationError,
    normalize_linkedin_company_url,
    normalize_url,
)
from qualification.scoring.verification_helpers import (
    visible_html_links,
    visible_html_text,
    visible_html_body_empty,
)


@dataclass(frozen=True)
class Page:
    url: str
    text: str
    links: tuple
    request_id: str
    sha256: str


def _url(value):
    if (
        not isinstance(value, str)
        or not value.startswith("https://")
        or "?" in value
        or "#" in value
    ):
        raise ValueError("noncanonical evidence URL")
    deepline_catalog._public_url(value)
    return normalize_url(value).url


def _text(value):
    return " ".join(value.split()) if isinstance(value, str) else ""


def verified_pages(trace):
    """Only a metered, completed, noninteractive fetch bound to its requested URL.

    Source bodies from generator invocations, search results, user-controlled
    extraction prompts, redirects, and incomplete/error envelopes never qualify.
    """
    pages = {}
    conflicts = set()
    for seq, call in trace.items():
        if call.get("op") != "deepline.execute":
            continue
        req = call["request"]
        resp = call["response"]
        if (
            not isinstance(resp, dict)
            or resp.get("status") != "completed"
            or req.get("tool") != "firecrawl_scrape"
            or req.get("response_contract") is not None
            or resp.get("error")
            or (
                isinstance(resp.get("result"), dict)
                and (
                    resp["result"].get("error")
                    or resp["result"].get("success") is False
                )
            )
        ):
            continue
        args = req.get("payload")
        if not isinstance(args, dict) or set(args) - {
            "url",
            "formats",
            "onlyMainContent",
        }:
            continue
        if not isinstance(args.get("formats"), list) or set(args["formats"]) - {
            "markdown",
            "rawHtml",
        }:
            continue
        data = (
            (resp.get("result") or {}).get("data")
            if isinstance(resp.get("result"), dict)
            else None
        )
        if not isinstance(data, dict):
            continue
        meta = data.get("metadata") or {}
        if (
            not isinstance(meta, dict)
            or type(meta.get("statusCode")) is not int
            or not 200 <= meta["statusCode"] < 300
        ):
            continue
        url = args.get("url")
        if meta.get("sourceURL") != url or meta.get("url") != url:
            continue
        try:
            key = _url(url)
        except (
            TypeError,
            ValueError,
            NormalizationError,
            deepline_catalog.CatalogError,
        ):
            continue
        if "rawHtml" in args["formats"] and isinstance(data.get("rawHtml"), str):
            body = data["rawHtml"]
            if not (1 <= len(body) <= 1_048_576) or visible_html_body_empty(body):
                continue
            text = _text(visible_html_text(body))
            links = visible_html_links(body)
        elif "markdown" in args["formats"] and isinstance(data.get("markdown"), str):
            body = data["markdown"]
            if not 1 <= len(body) <= 1_048_576:
                continue
            # Firecrawl's passive Markdown representation only. Exclude code
            # examples/HTML comments before extracting visible text or links.
            md = re.sub(
                r"<!--.*?-->|(?ms:^\s*```[^\n]*\n.*?^\s*```[^\n]*$)",
                "",
                body,
                flags=re.S,
            )
            if "<" in md or ">" in md:
                continue
            matches = list(
                re.finditer(r"(?<!!)\[([^\[\]\n]*)\]\((https://[^\s()]+)\)", md)
            )
            links = tuple(x.group(2) for x in matches)
            text = _text(
                re.sub(r"(?<!!)\[([^\[\]\n]*)\]\((https://[^\s()]+)\)", r"\1", md)
            )
        else:
            continue
        if not text:
            continue
        pg = Page(
            key,
            text,
            links,
            f"generate-research-{seq}",
            sha256(body.encode()).hexdigest(),
        )
        if key in pages and pages[key].sha256 != pg.sha256:
            conflicts.add(key)
        pages[key] = pg
    for key in conflicts:
        pages.pop(key, None)
    return pages


def _excerpt(page, claim, *, minimum=1):
    term = _text(claim)
    if len(term) < minimum or len(term) > 2000:
        raise ValueError("invalid source claim")
    hit = re.search(r"(?<![\w-])" + re.escape(term) + r"(?![\w-])", page.text, re.I)
    if not hit:
        raise ValueError("missing source claim")
    return {
        "url": page.url,
        "request_id": page.request_id,
        "body_sha256": page.sha256,
        "excerpt": page.text[
            max(0, hit.start() - 100) : min(len(page.text), hit.end() + 100)
        ],
    }


def _first_excerpt(pages, claim):
    for p in pages:
        try:
            return _excerpt(p, claim)
        except ValueError:
            pass
    raise ValueError("missing source passage")


def _page(pages, url):
    found = pages.get(_url(url))
    if not found:
        raise ValueError("source not fetched")
    return found


def _site(value):
    p = normalize_url(_url(value))
    # A free hosting registrable domain cannot establish official source ownership.
    if p.domain.is_private_suffix:
        raise ValueError("shared publisher")
    return p.domain.registrable_domain


def _linked(page, url, *, linkedin=False):
    for link in page.links:
        try:
            if linkedin:
                if normalize_linkedin_company_url(
                    link
                ) == normalize_linkedin_company_url(url):
                    return True
            elif _url(link) == _url(url):
                return True
        except (ValueError, NormalizationError, deepline_catalog.CatalogError):
            continue
    return False


def evidence_for(company, case, pages):
    """Return numbered native-source excerpts, or None if verification needs more evidence.

    This proves attributable source material for the two independent judges to
    interpret. It does not assert an Arena competition verdict for a new company.
    """
    try:
        name = company["company_name"]
        site = company["company_website"]
        li = company["company_linkedin"]
        if (
            not isinstance(name, str)
            or len(name.strip()) < 2
            or urlsplit(site).path not in ("", "/")
        ):
            return None
        normalize_linkedin_company_url(li)
        homepage = _page(pages, site)
        profile = _page(pages, li)
        site_domain = _site(site)
        if not _linked(homepage, li, linkedin=True) or not _linked(profile, site):
            return None
        proof = [_excerpt(homepage, name), _excerpt(profile, name)]
        for field in ("industry", "employee_count", "country"):
            proof.append(_excerpt(profile, company[field]))
        if company.get("country", "").casefold() in {
            "us",
            "usa",
            "united states",
            "united states of america",
        } or company.get("state"):
            proof.append(_excerpt(profile, company["state"]))
        if company.get("company_stage"):
            proof.append(_first_excerpt((profile, homepage), company["company_stage"]))
        icp = case["icp"]
        if company["company_name"].casefold() in {
            str(x).casefold() for x in icp.get("excluded_companies", [])
        }:
            return None
        if any(
            _text(x).casefold() in {name.casefold(), site_domain}
            for x in icp.get("excluded_companies", [])
            if isinstance(x, str)
        ):
            return None
        signals = company.get("intent_signals") or []
        seen = set()
        signal_pages = []
        for s in signals:
            i = s["matched_icp_signal"]
            if (
                type(i) is not int
                or not 0 <= i < len(icp["intent_signals"])
                or i in seen
            ):
                return None
            seen.add(i)
            pg = _page(pages, s["url"])
            if _site(pg.url) != site_domain:
                return None
            signal_pages.append(pg)
            d = date.fromisoformat(s["date"])
            days = (date.fromisoformat(case["evaluation_date"]) - d).days
            max_age = icp.get("intent_max_age_days")
            if (
                type(max_age) is not int
                or not 1 <= max_age <= 3650
                or not 0 <= days <= max_age
            ):
                return None
            dd = d.strftime("%B %d, %Y").replace(" 0", " ")
            desc = _excerpt(pg, s["description"], minimum=12)
            # An unrelated date or company elsewhere on the page cannot support
            # the claim. Both must occur in the quoted event context.
            if _text(name).casefold() not in desc["excerpt"].casefold():
                return None
            if not any(x in desc["excerpt"] for x in (d.isoformat(), dd)):
                return None
            proof.append(desc)
        if 0 not in seen:
            return None
        details = company["intent_details"]
        for sentence in re.split(r"(?<=[.!?])\s+", details.strip()):
            if len(sentence) < 8:
                return None
            proof.append(_first_excerpt(signal_pages, sentence))
        required = icp.get("required_attribute")
        if isinstance(required, dict):
            required = required.get("text") or required.get("description")
        attr = company.get("required_attribute")
        if required and (
            not attr or _text(attr["text"]).casefold() != _text(required).casefold()
        ):
            return None
        if attr is not None:
            pg = _page(pages, attr["evidence_url"])
            if _site(pg.url) != site_domain or attr["passed"] is not True:
                return None
            proof.append(_excerpt(pg, attr["evidence_quote"], minimum=12))
        for s in company.get("company_stage_evidence", []):
            pg = _page(pages, s["url"])
            if _site(pg.url) != site_domain:
                return None
            proof.append(_excerpt(pg, s["quote"], minimum=12))
        return {
            "company": name,
            "origin": "same_run_verified_provider_sources",
            "company_sha256": contracts.document_hash(company),
            "source_evidence": proof,
        }
    except (
        KeyError,
        IndexError,
        AttributeError,
        TypeError,
        ValueError,
        NormalizationError,
        deepline_catalog.CatalogError,
    ):
        return None


def apply_fresh_evidence(native, case, companies, trace):
    """Mix original native and new grounded examples. Every company needs evidence."""
    import json

    pages = verified_pages(trace)
    bindings = []
    if not pages:
        return native
    for c in companies:
        known = case.get("labels", {}).get(contracts.document_hash(c))
        if known is not None:
            if known["class"] != "qualified":
                return native
            bindings.append(dict(known, origin="historical_native_judgment"))
            continue
        fresh = evidence_for(c, case, pages)
        if fresh is None:
            return native
        bindings.append(fresh)
    facts = json.dumps(
        bindings, separators=(",", ":"), sort_keys=True, ensure_ascii=False
    )
    if len(facts) > 65536:
        return native
    references = [
        {"company": v["company"], "class": v["class"]}
        for v in case.get("labels", {}).values()
    ]
    reference = json.dumps(
        sorted(references, key=lambda x: (str(x["company"]), x["class"]))[:20],
        ensure_ascii=False,
    )
    # Preserve distinct-company and shape checks. Replacing the native-only
    # gate requires affirmative source bindings for every uncovered company.
    checks = {
        k: v
        for k, v in native["checks"].items()
        if k not in {"native_coverage", "native_qualified"}
    }
    return {
        **native,
        "facts": facts,
        "reference_answer": reference,
        "grading_notes": "Frozen native references are nonexhaustive examples. Assess new companies independently against the original ICP and the cited same-run provider excerpts. Source coverage is not a native qualification verdict.",
        "checks": {**checks, "source_coverage": True},
    }
