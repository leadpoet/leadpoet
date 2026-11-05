"""Private, prospective reuse of native fit receipts after fresh source checks.

The host owns family identities and durable authority. This module never gives
an unvalidated candidate authority: it re-fetches the original bounded sources
before claiming its exact evidence variant. Missing capture falls back to native
fit verification. Source records contain hashes, not provider credentials.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from functools import wraps
import inspect
import json
import re
from typing import Any, Mapping

SCHEMA = "leadpoet.lab_arena.company_fit_evidence.v2"
MANIFEST_SCHEMA = "leadpoet.lab_arena.company_fit_sources.v2"
MAX_SOURCES = 40
MAX_SNAPSHOT_BYTES = 1_000_000
_exchange = ContextVar("arena_fit_evidence_exchange", default=None)
_capture = ContextVar("arena_fit_evidence_capture", default=None)


def _copy(value):
    from lab_arena import contracts
    return json.loads(contracts.canonical_json(value))


def _hash(value):
    from lab_arena import contracts
    return contracts.document_hash(value)


class FitEvidenceReuseError(ValueError):
    pass


@contextmanager
def install_fit_evidence_reuse(callback):
    token = _exchange.set(callback)
    try:
        yield
    finally:
        _exchange.reset(token)


def current_exchange():
    return _exchange.get()


class FitEvidenceCapture:
    def __init__(self):
        self.records = {}
        self.responses = {}
        self.complete = True

    def add(self, kind, request, response, *, ok=True):
        try:
            request, response = _copy(request), _copy(response)
            identity = _hash({"kind": kind, "request": request})
            record = {"kind": kind, "request": request, "response_hash": _hash(response)}
            previous = self.records.get(identity)
            if previous is not None and previous != record:
                self.complete = False
            self.records[identity] = record
            self.responses[identity] = response
            self.complete = self.complete and ok and len(self.records) <= MAX_SOURCES
        except (ValueError, TypeError):
            self.complete = False

    def manifest(self):
        return {"schema_version": MANIFEST_SCHEMA,
                "sources": [self.records[key] for key in sorted(self.records)]}


@contextmanager
def capture_fit_evidence():
    captured = FitEvidenceCapture()
    token = _capture.set(captured)
    try:
        yield captured
    finally:
        _capture.reset(token)


def record_source(kind, request, response, *, ok=True):
    captured = _capture.get()
    if captured is not None:
        captured.add(kind, request, response, ok=ok)


def unretained_source():
    """Keep native verification when evidence cannot be re-fetched exactly."""
    captured = _capture.get()
    if captured is not None:
        captured.complete = False


def observed_source(kind, request_fields, sink_fields=()):
    """Observe an existing retrieval without changing its return or exception."""
    def decorate(function):
        signature = inspect.signature(function)
        @wraps(function)
        async def wrapped(*args, **kwargs):
            if _capture.get() is None:
                return await function(*args, **kwargs)
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            request = {name: bound.arguments[name] for name in request_fields}
            if sink_fields:
                request["capture_sinks"] = [name for name in sink_fields if bound.arguments.get(name) is not None]
            try:
                result = await function(*args, **kwargs)
            except BaseException:
                _capture.get().complete = False
                raise
            response = {"result": result,
                        "contexts": {name: bound.arguments.get(name) or {} for name in sink_fields if bound.arguments.get(name) is not None}}
            ok = result is not None and (not isinstance(result, Mapping) or result.get("ok") is not False)
            record_source(kind, request, response, ok=ok)
            return result
        return wrapped
    return decorate


def canonical_fit_company(company):
    """Remove intent claim wording from prospective fit research, retain locators.

    Dates, stage packets and attribute quotes remain unchanged. Empty V5 details
    retain the V5/non-legacy distinction while avoiding paragraph-based selection
    of ancillary pages. The later paragraph and intent judges use the original.
    """
    result = company.model_copy(deep=True)
    if result.intent_details is not None:
        result.intent_details = ""
    for signal in result.intent_signals:
        signal.description = ""
    return result


def _receipt_fit(receipt):
    from qualification.scoring.company_fit_decision import CompanyFitDecisionResult
    return CompanyFitDecisionResult(receipt["decision"], receipt.get("reason"), details={
        key: deepcopy(value) for key, value in receipt.items()
        if key not in {"gate", "contract_id", "contract_version", "decision", "reason"}
    })


def _text_pages(captured):
    pages = {}
    def visit(value):
        if isinstance(value, Mapping):
            text = value.get("text")
            url = value.get("url") or value.get("final_url")
            if isinstance(text, str) and isinstance(url, str):
                pages[url] = text
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value: visit(item)
    for identity, response in captured.responses.items():
        record = captured.records[identity]
        if record["kind"] == "homepage":
            # The native investigator sees this exact plain-text projection.
            # Keep raw HTML in the response hash so markup/body changes still
            # invalidate reuse; quote checks must use the admitted text surface.
            from qualification.scoring.company_evidence_investigator import _plain_text
            text = _plain_text(response.get("text", ""))
            for url in (record["request"].get("url"), response.get("final_url")):
                if isinstance(url, str) and text:
                    pages[url] = text
            continue
        visit(response)
        result = response.get("result") if isinstance(response, Mapping) else None
        if isinstance(result, Mapping) and isinstance(result.get("text"), str):
            url = record["request"].get("url")
            if url: pages[url] = result["text"]
    return pages


def _grounded_finding(receipt, dimension, web, pages):
    """Find an existing completed review that produced this exact final fact."""
    from qualification.scoring.company_evidence_investigator import _quote_occurs
    from qualification.scoring import lead_scorer
    target = {"employee_size": "headcount"}.get(dimension, dimension)
    observed = receipt["dimension_evidence"].get("web_observations") or {}
    item = receipt["dimension_evidence"][dimension]
    decision = item.get("observed_decision", item.get("decision"))
    status = {"match": "VERIFIED", "mismatch": "CONTRADICTED"}.get(decision)
    for review in receipt.get("supporting_receipts") or []:
        if (not isinstance(review, Mapping)
            or review.get("gate") != "company_evidence_investigation"
            or review.get("completed_submitted_findings") is not True
            or review.get("failure_reason") or target not in (review.get("targets") or [])):
            continue
        finding = (review.get("claims") or {}).get(target)
        if (not isinstance(finding, Mapping) or finding.get("target") != target
            or finding.get("status") != status
            or finding.get("evidence_url") != web.get("url")
            or finding.get("evidence_quote") != web.get("quote")
            or (finding.get("supporting_evidence") or []) != (web.get("supporting_evidence") or [])):
            continue
        if dimension == "industry":
            industry = finding.get("_industry_review")
            if (not isinstance(industry, Mapping)
                or industry.get("schema") != "industry_evidence_review_v1"
                or industry.get("decision") != decision or not industry.get("model")
                or re.fullmatch(r"(?:sha256:)?[0-9a-f]{64}", str(industry.get("packet_sha256") or "")) is None
                or finding.get("activity_role") != observed.get("industry_activity_role")
                or finding.get("observed_industry") != observed.get("observed_industry")
                or (finding.get("observed_subindustry") or "") != (observed.get("observed_subindustry") or "")):
                continue
            sources = industry.get("evidence")
            if not isinstance(sources, list) or not sources or any(
                not isinstance(source, Mapping) or not isinstance(source.get("quote"), str)
                or not _quote_occurs(source["quote"], pages.get(source.get("url"), ""))
                for source in sources
            ):
                continue
        elif dimension == "stage":
            if (not observed.get("observed_company_stage")
                or lead_scorer._normalize_company_stage(finding.get("observed_value"))
                != lead_scorer._normalize_company_stage(observed["observed_company_stage"])):
                continue
        elif dimension == "employee_size":
            value = finding.get("observed_value")
            if (lead_scorer._normalize_linkedin_employee_bucket(value) or value) != observed.get("observed_employee_count"):
                continue
        elif dimension == "geography":
            if any(str(finding.get(field) or "").strip() != str(observed.get(native) or "").strip()
                   for field, native in (("observed_country", "observed_hq_country"),
                                         ("observed_state", "observed_hq_state"))):
                continue
        return True
    return False


def _captured_profile_fact(receipt, dimension, web, captured):
    """Use the existing native profile projections, never a model lookalike."""
    from qualification.scoring import company_evidence_investigator as cei
    observed = receipt["dimension_evidence"].get("web_observations") or {}
    identity = receipt["dimension_evidence"]["identity"].get("identity") or {}
    anchor = {"verified_name": identity.get("observed_name"),
              "verified_domain": identity.get("observed_domain"),
              "verified_linkedin_slug": identity.get("observed_linkedin_slug")}
    finding = {"status": "VERIFIED", "evidence_url": web.get("url"),
               "evidence_quote": web.get("quote"),
               "observed_country": observed.get("observed_hq_country"),
               "observed_state": observed.get("observed_hq_state")}
    for key, response in captured.responses.items():
        kind = captured.records[key]["kind"]
        if kind not in {"linkedin_contents", "linkedin_structured"}:
            continue
        result = response.get("result")
        contexts = response.get("contexts") or {}
        if dimension == "employee_size" and isinstance(result, Mapping):
            if kind == "linkedin_structured" and result == web:
                return True  # Exact native typed size projection and source request.
            if (kind == "linkedin_contents" and result.get("employee_count")
                and result.get("employee_count") == observed.get("observed_employee_count")
                and result.get("url") == web.get("url") and result.get("quote") == web.get("quote")):
                return True
        if dimension == "geography":
            source = contexts.get("company_headquarters_evidence")
            if isinstance(source, Mapping) and cei._bound_structured_headquarters_finding(finding, source, anchor):
                return True
            source = contexts.get("source_text_sink")
            if (isinstance(source, Mapping) and isinstance(source.get("text"), str)
                and cei._bound_linkedin_headquarters_finding(finding, source["text"], source.get("url"), anchor)):
                return True
    return False


def _source_complete(receipt, captured):
    """Require each final fact to have native proof independent of discovery.

    Sonar's initial lookup may select sources. Quote occurrence alone does not
    make its observations independently reviewed: retain only existing grounded
    findings or exact native profile projections, with unchanged criteria.
    """
    from qualification.scoring.company_fit_decision import COMPANY_FIT_REQUIRED_DIMENSIONS
    if not captured.complete:
        return False
    dimensions = receipt.get("company_fit_dimensions")
    evidence = receipt.get("dimension_evidence")
    if not isinstance(dimensions, Mapping) or not isinstance(evidence, Mapping):
        return False
    required = set(COMPANY_FIT_REQUIRED_DIMENSIONS)
    if receipt.get("company_fit_stage_required"):
        required.add("stage")
    decision = receipt.get("decision")
    if decision == "match":
        if any(dimensions.get(name) != "match" for name in required) or receipt.get("required_attribute_decision") != "match":
            return False
    elif decision == "mismatch":
        if not any(dimensions.get(name) == "mismatch" for name in required) and receipt.get("required_attribute_decision") != "mismatch":
            return False
    else:
        return False
    if receipt.get("failure_class") or receipt.get("failure_reason_code"):
        return False
    pages = _text_pages(captured)
    from qualification.scoring.company_evidence_investigator import _quote_occurs
    active = required | ({"required_attribute"} if "required_attribute" in evidence else set())
    decisive = []
    for name in active:
        item = evidence.get(name)
        if not isinstance(item, Mapping):
            return False
        if item.get("decision") == "unavailable":
            # A proved mismatch is terminal even when other dimensions have no
            # proof. Their unavailable state is retained verbatim, never changed.
            if decision == "mismatch": continue
            return False
        if item.get("submitted_decision") == "mismatch":
            decisive.append(name)
            continue
        if name == "identity":
            if not any(record["kind"] == "homepage" for record in captured.records.values()):
                return False
            homepage = item.get("identity")
            observed = item.get("web_identity_receipt", homepage)
            if (not isinstance(homepage, Mapping) or homepage.get("decision") != item.get("decision")
                or not isinstance(observed, Mapping)
                or any(homepage.get(field) != observed.get(field) for field in
                       ("observed_name", "observed_domain", "observed_linkedin_slug"))):
                # Structured aliases and unresolved homepage/web conflicts need
                # a broader identity manifest. Keep their native path for now.
                return False
            decisive.append(name)
            continue
        web = item.get("web_evidence")
        if (isinstance(web, Mapping) and name in {"employee_size", "geography"}
            and _captured_profile_fact(receipt, name, web, captured)):
            decisive.append(name)
            continue
        if isinstance(web, Mapping) and isinstance(web.get("url"), str) and isinstance(web.get("quote"), str):
            if not _quote_occurs(web["quote"], pages.get(web["url"], "")):
                return False
            support = web.get("supporting_evidence", [])
            if not isinstance(support, list) or any(
                not isinstance(source, Mapping)
                or not isinstance(source.get("url"), str)
                or not isinstance(source.get("quote"), str)
                or not _quote_occurs(source["quote"], pages.get(source["url"], ""))
                for source in support
            ):
                return False
            if not _grounded_finding(receipt, name, web, pages):
                return False
            decisive.append(name)
            continue
        # Neither a model quote nor a URL proves independent adjudication.
        return False
    return bool(decisive)


def _contexts_complete(contexts, captured):
    """Every restored source body must belong to a freshly captured page."""
    from qualification.scoring.company_evidence_investigator import _plain_text, _quote_occurs
    pages = []
    def collect(value, requested=None):
        if isinstance(value, Mapping):
            url = value.get("url") or requested
            final = value.get("final_url") or url
            body = value.get("text")
            if isinstance(url, str) and isinstance(final, str) and isinstance(body, str):
                pages.append((url, final, body))
                pages.append((url, final, _plain_text(body)))
            for item in value.values():
                collect(item, requested)
        elif isinstance(value, list):
            for item in value:
                collect(item, requested)
    for identity, response in captured.responses.items():
        request = captured.records[identity]["request"]
        collect(response, request.get("url") or request.get("profile_url"))
    def valid(value, requested=None):
        if not value:
            return True
        if not isinstance(value, Mapping):
            return False
        if "text" in value:
            url = value.get("url") or requested
            final = value.get("final_url") or url
            text = value["text"]
            return isinstance(text, str) and bool(text) and any(
                url == source and final == target and (text == body or _quote_occurs(text, body))
                for source, target, body in pages)
        return all(valid(item, key) for key, item in value.items())
    return all(valid(context) for context in contexts)


def build_fit_snapshot(fit, linkedin, matched, required, captured):
    receipt = fit.receipt("company_fit")
    if not _source_complete(receipt, captured) or not _contexts_complete((linkedin, matched, required), captured):
        return None
    manifest = captured.manifest()
    snapshot = {"schema_version": SCHEMA, "company_fit_receipt": receipt,
                "linkedin_profile_source": linkedin, "matched_company_sources": matched,
                "required_attribute_sources": required, "evidence_manifest": manifest,
                "source_snapshot_hash": _hash(manifest)}
    try:
        snapshot = validate_fit_snapshot(snapshot)
        # The wire request repeats the manifest outside the judgment. Reserve
        # room for the index, reference and protocol fields added by the host.
        from lab_arena import contracts, signal_cache_control
        if len(contracts.canonical_json({"judgment": snapshot, "evidence_input": manifest}).encode()) > signal_cache_control.MAX_BYTES - 4096:
            return None
        return snapshot
    except FitEvidenceReuseError:
        return None


def validate_fit_snapshot(snapshot):
    try:
        copied = _copy(snapshot)
        if set(copied) != {"schema_version", "company_fit_receipt", "linkedin_profile_source",
                           "matched_company_sources", "required_attribute_sources", "evidence_manifest", "source_snapshot_hash"}:
            raise ValueError("fields")
        receipt, manifest = copied["company_fit_receipt"], copied["evidence_manifest"]
        if (copied["schema_version"] != SCHEMA or receipt.get("gate") != "company_fit"
            or receipt.get("contract_id") != "company-fit-decision:v1"
            or receipt.get("contract_version") != "company-fit-decision:v1"
            or receipt.get("decision") not in {"match", "mismatch"}
            or receipt.get("company_fit_decision") != receipt["decision"]
            or receipt.get("failure_class") or receipt.get("failure_reason_code")
            or set(manifest) != {"schema_version", "sources"}
            or manifest["schema_version"] != MANIFEST_SCHEMA
            or not isinstance(manifest["sources"], list) or len(manifest["sources"]) > MAX_SOURCES
            or copied["source_snapshot_hash"] != _hash(manifest)
            or len(json.dumps(copied, ensure_ascii=False).encode()) > MAX_SNAPSHOT_BYTES):
            raise ValueError("envelope")
        from qualification.scoring.company_fit_decision import (
            COMPANY_FIT_DIMENSIONS, aggregate_company_fit_decisions, reconcile_company_fit_decisions,
        )
        dimensions = receipt.get("company_fit_dimensions")
        if (not isinstance(dimensions, Mapping) or set(dimensions) != set(COMPANY_FIT_DIMENSIONS)
            or type(receipt.get("company_fit_stage_required")) is not bool
            or any(value not in {"match", "mismatch", "unavailable"} for value in dimensions.values())
            or receipt.get("required_attribute_decision") not in {"match", "mismatch", "unavailable"}
            or reconcile_company_fit_decisions([
                aggregate_company_fit_decisions(dimensions, stage_required=receipt["company_fit_stage_required"]),
                receipt["required_attribute_decision"],
            ]) != receipt["decision"]):
            raise ValueError("native decision")
        evidence = receipt.get("dimension_evidence")
        if (not isinstance(evidence, Mapping) or any(
            not isinstance(evidence.get(name), Mapping)
            or evidence[name].get("decision") != dimensions[name]
            for name in COMPANY_FIT_DIMENSIONS
        )):
            raise ValueError("native evidence")
        request_fields = {
            "homepage": {"url"},
            "investigator_page": {"url", "stealth_mode", "archive_targets", "archive_company_names"},
            "investigator_exa": {"url"},
            "linkedin_contents": {"profile_url", "expected_company_identity", "capture_sinks"},
            "linkedin_structured": {"requested_domain", "profile_url", "company_identity_observed_profile_url",
                                    "expected_company_name", "preserve_identity_website", "capture_sinks"},
        }
        from qualification.competition_models import public_http_url
        seen = set()
        for entry in manifest["sources"]:
            if (set(entry) != {"kind", "request", "response_hash"}
                or entry["kind"] not in {"homepage", "investigator_page", "investigator_exa", "linkedin_contents", "linkedin_structured"}
                or not isinstance(entry["request"], Mapping)
                or not isinstance(entry["response_hash"], str) or re.fullmatch(r"sha256:[0-9a-f]{64}", entry["response_hash"]) is None):
                raise ValueError("source")
            if set(entry["request"]) != request_fields[entry["kind"]]:
                raise ValueError("source request")
            source_url = entry["request"].get("url") or entry["request"].get("profile_url")
            if not isinstance(source_url, str) or not public_http_url(source_url).startswith("https://"):
                raise ValueError("source URL")
            identity = _hash({"kind": entry["kind"], "request": entry["request"]})
            if identity in seen: raise ValueError("duplicate source")
            seen.add(identity)
            if "capture_sinks" in entry["request"] and (
                not isinstance(entry["request"]["capture_sinks"], list)
                or any(name not in {"source_text_sink", "public_company_evidence", "company_identity_evidence",
                                    "company_description_evidence", "company_headquarters_evidence"}
                       for name in entry["request"]["capture_sinks"])):
                raise ValueError("source context")
        if not manifest["sources"] and not (receipt["decision"] == "mismatch" and any(
            isinstance(item, Mapping) and item.get("submitted_decision") == "mismatch"
            for item in (receipt.get("dimension_evidence") or {}).values()
        )):
            raise ValueError("source coverage")
        for field in ("linkedin_profile_source", "matched_company_sources", "required_attribute_sources"):
            if not isinstance(copied[field], Mapping): raise ValueError("context")
        return copied
    except (TypeError, ValueError, KeyError) as exc:
        raise FitEvidenceReuseError("fit evidence snapshot is invalid") from exc


async def _read_source(entry):
    import aiohttp
    kind, request = entry["kind"], dict(entry["request"])
    sinks = request.pop("capture_sinks", [])
    if kind == "homepage":
        from qualification.scoring import company_verification as cv
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=cv._HTTP_TIMEOUT_SECS, connect=min(3.0, cv._HTTP_TIMEOUT_SECS)), headers=cv._HEADERS) as session:
            status, final, text = await cv._fetch_bounded_html(session, request["url"])
        record_source(kind, request, {"status": status, "final_url": final, "text": text}, ok=status < 400)
    elif kind in {"investigator_page", "investigator_exa"}:
        from qualification.scoring import company_evidence_investigator as cei
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=35)) as session:
            await (cei._fetch_page(session, **request) if kind == "investigator_page" else cei._fetch_exa_exact_page(session, **request))
    elif kind == "linkedin_contents":
        from qualification.scoring.linkedin_company_size import fetch_current_linkedin_company_size
        await fetch_current_linkedin_company_size(**request, diagnostic={}, **{name: {} for name in sinks})
    elif kind == "linkedin_structured":
        from qualification.scoring.linkedin_company_size import fetch_structured_linkedin_company_size
        await fetch_structured_linkedin_company_size(**request, diagnostic={}, **{name: {} for name in sinks})
    else:
        raise FitEvidenceReuseError("unknown evidence source")


async def revalidate_fit_snapshot(snapshot, *, captured_out=None):
    snapshot = validate_fit_snapshot(snapshot)
    try:
        with capture_fit_evidence() as fresh:
            for entry in snapshot["evidence_manifest"]["sources"]:
                await _read_source(entry)
        equal = (fresh.complete and fresh.manifest() == snapshot["evidence_manifest"]
                 and _source_complete(snapshot["company_fit_receipt"], fresh)
                 and _contexts_complete((snapshot["linkedin_profile_source"], snapshot["matched_company_sources"],
                                         snapshot["required_attribute_sources"]), fresh))
        if equal and captured_out is not None:
            captured_out.append(fresh)
        return equal
    except Exception:
        # A failed fresh fetch proves no equality and cannot admit a candidate.
        return False


def _restore_paragraph_source_context(company, fit, captured, matched):
    # Ranking ancillary retained pages is paragraph-specific. Run the original
    # source selector against freshly observed bodies after fit receipt reuse.
    from qualification.scoring import company_evidence_investigator as cei, lead_scorer
    pages = {}
    for identity, response in captured.responses.items():
        entry = captured.records[identity]
        if entry["kind"] == "homepage":
            result = {"final_url": response["final_url"], "text": cei._plain_text(response["text"])}
            url = entry["request"]["url"]
        else:
            result = response.get("result")
            url = entry["request"].get("url")
        if (isinstance(result, Mapping) and isinstance(result.get("text"), str)
            and result["text"] and isinstance(result.get("final_url"), str) and url):
            page = {"final_url": result["final_url"], "text": result["text"]}
            if entry["kind"] == "homepage":
                pages.setdefault(url, page)
            else:
                pages[url] = page
    if pages:
        lead_scorer._retain_matched_investigator_source_contexts(
            matched, fit, {cei.PRIVATE_FETCHED_PAGES_KEY: pages}, company=company,
        )


async def fit_with_evidence_reuse(native, *, company, linkedin, matched, required):
    """Return native fit or an exactly revalidated original native receipt."""
    exchange = current_exchange()
    if exchange is None:
        return await native(canonical_fit_company(company))
    response = await exchange({"operation": "candidates"})
    if not isinstance(response, Mapping) or not isinstance(response.get("candidates"), list) or len(response["candidates"]) > 8:
        raise FitEvidenceReuseError("fit evidence candidates are invalid")
    for candidate in response["candidates"]:
        snapshot = validate_fit_snapshot(candidate.get("judgment"))
        fresh_captures = []
        if not await revalidate_fit_snapshot(snapshot, captured_out=fresh_captures):
            continue
        claim = {"operation": "claim", "candidate_reference": candidate.get("reference"),
                 "source_snapshot_hash": snapshot["source_snapshot_hash"], "source_manifest": snapshot["evidence_manifest"]}
        result = await exchange(claim)
        if not isinstance(result, Mapping) or result.get("status") != "hit":
            raise FitEvidenceReuseError("revalidated fit authority is unavailable")
        stored = validate_fit_snapshot(result.get("judgment"))
        if stored != snapshot:
            raise FitEvidenceReuseError("fit evidence authority changed")
        linkedin.update(deepcopy(snapshot["linkedin_profile_source"]))
        matched.update(deepcopy(snapshot["matched_company_sources"]))
        required.update(deepcopy(snapshot["required_attribute_sources"]))
        fit = _receipt_fit(snapshot["company_fit_receipt"])
        _restore_paragraph_source_context(company, fit, fresh_captures[0], matched)
        return fit
    with capture_fit_evidence() as captured:
        fit = await native(canonical_fit_company(company))
    _restore_paragraph_source_context(company, fit, captured, matched)
    snapshot = build_fit_snapshot(fit, linkedin, matched, required, captured)
    if snapshot is None:
        return fit
    claim = {"operation": "claim", "source_snapshot_hash": snapshot["source_snapshot_hash"],
             "source_manifest": snapshot["evidence_manifest"]}
    result = await exchange(claim)
    if not isinstance(result, Mapping) or result.get("status") not in {"hit", "miss"}:
        raise FitEvidenceReuseError("fit evidence variant claim is invalid")
    if result["status"] == "hit":
        # An equal-source concurrent writer owns the first completed verdict.
        stored = validate_fit_snapshot(result.get("judgment"))
        if stored["evidence_manifest"] != snapshot["evidence_manifest"]:
            raise FitEvidenceReuseError("fit evidence variant differs")
        linkedin.clear(); linkedin.update(stored["linkedin_profile_source"])
        matched.clear(); matched.update(stored["matched_company_sources"])
        required.clear(); required.update(stored["required_attribute_sources"])
        fit = _receipt_fit(stored["company_fit_receipt"])
        _restore_paragraph_source_context(company, fit, captured, matched)
        return fit
    completed = await exchange({**claim, "operation": "complete", "reference": result.get("reference"), "judgment": snapshot})
    if not isinstance(completed, Mapping) or completed.get("status") not in {"stored", "existing"}:
        raise FitEvidenceReuseError("fit evidence completion is invalid")
    return fit
