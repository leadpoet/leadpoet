"""Miner-funded scoring routes for legacy judge provider calls.

The pinned judge still issues a small set of ScrapingDog-shaped requests.
Only a miner-funded scoring run translates those requests to the miner's
Deepline workspace.  Host-funded scoring and miner model execution do not use
this module's routes.

The broker keeps the requested operation identity in the ledger.  This module
only supplies the effective Deepline request and converts the bounded raw
Deepline envelope back to the response shape the unchanged judge expects.
"""

from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any, Mapping, Optional
from urllib.parse import urlsplit

from lab_arena import deepline_catalog, operations


COMPATIBILITY_VERSION = "miner-score-deepline-compat:v1"
EFFECTIVE_OPERATION_ID = "deepline.execute"
SHADOW_REPLAY_ROUTE_PREFIX = (
    "https://arena.invalid/saved-output-replay/miner-score-route/"
)

FIRECRAWL_TOOL = "firecrawl_scrape"
GENERIC_HTTP_TOOL = "generic_http_request"
SOURCE_HTML_ADAPTERS = frozenset({"firecrawl_raw_html", "generic_http_raw_html", "harvest_linkedin_post_html"})
HARVEST_JOB_TOOL = "harvestapi_get_job"
HARVEST_POST_TOOL = "harvestapi_get_post"
TWITTER_POST_TOOL = "twitterapi_tweets_by_ids"

_ROUND_ID_RE = re.compile(
    r"^arena-(\d{4}-\d{2}-\d{2})(?:-[a-z0-9]{1,16})?$"
)
_LINKEDIN_ACTIVITY_URL = (
    "https://www.linkedin.com/feed/update/urn:li:activity:{activity_id}/"
)
_ASHBY_API_PATH_RE = re.compile(
    r"^/posting-api/job-board/[A-Za-z0-9_-]{1,100}/?$"
)
_GREENHOUSE_API_PATH_RE = re.compile(
    r"^/v1/boards/[A-Za-z0-9_-]{1,100}/jobs/[0-9]{5,20}/?$"
)
_GREENHOUSE_BOARD_API_PATH_RE = re.compile(
    r"^/v1/boards/[A-Za-z0-9_-]{1,100}/jobs/?$"
)
_WORKDAY_API_PATH_RE = re.compile(
    r"^/wday/cxs/[A-Za-z0-9_-]{1,100}/[A-Za-z0-9_-]{1,100}/job/"
    r"[A-Za-z0-9_./-]{3,500}$"
)
_LEVER_POSTING_ID_PATTERN = (
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)
_LEVER_POSTING_ID_RE = re.compile(r"^" + _LEVER_POSTING_ID_PATTERN + r"$")
_LEVER_API_PATH_RE = re.compile(
    r"^/v0/postings/(?P<tenant>[A-Za-z0-9](?:[A-Za-z0-9_-]{0,98}[A-Za-z0-9])?)"
    r"(?:/(?P<posting>" + _LEVER_POSTING_ID_PATTERN + r"))?/?$"
)


class CompatibilityResponseError(ValueError):
    """The provider response cannot satisfy the legacy judge contract."""

    trusted_response_url: Optional[str] = None


@dataclass(frozen=True)
class MinerScoreRoute:
    requested_operation_id: str
    effective_operation_id: str
    effective_parameters: Mapping[str, Any]
    adapter: str
    requested_parameters: Mapping[str, Any]
    evaluation_date: Optional[date] = None

    def summary(self) -> dict[str, str]:
        return {
            "compatibility_version": COMPATIBILITY_VERSION,
            "effective_operation_id": self.effective_operation_id,
            "adapter": self.adapter,
        }


def _evaluation_date(round_id: str) -> Optional[date]:
    match = _ROUND_ID_RE.fullmatch(str(round_id or ""))
    if match is None:
        return None
    try:
        return date.fromisoformat(match.group(1))
    except ValueError:
        return None


def _ats_api_kind(value: Any) -> str:
    """Classify only exact public JSON transports used by scoring."""

    raw_url = str(value or "")
    try:
        parsed = urlsplit(raw_url)
        port = parsed.port
    except ValueError:
        return ""
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
        or (port is not None and port != 443)
    ):
        return ""
    host = parsed.hostname.casefold()
    if (
        host == "api.ashbyhq.com"
        and not parsed.query
        and _ASHBY_API_PATH_RE.fullmatch(parsed.path)
    ):
        return "ashby"
    if (
        host == "boards-api.greenhouse.io"
        and parsed.query == "content=true"
        and _GREENHOUSE_API_PATH_RE.fullmatch(parsed.path)
    ):
        return "greenhouse"
    if (
        host == "boards-api.greenhouse.io"
        and not parsed.query
        and "?" not in raw_url
        and _GREENHOUSE_BOARD_API_PATH_RE.fullmatch(parsed.path)
    ):
        return "greenhouse_board"
    if (
        host.endswith(".myworkdayjobs.com")
        and not parsed.query
        and _WORKDAY_API_PATH_RE.fullmatch(parsed.path)
    ):
        return "workday"
    if (
        host == "api.lever.co"
        and not parsed.query
        and _LEVER_API_PATH_RE.fullmatch(parsed.path)
    ):
        return "lever"
    return ""


def _linkedin_post_activity_id(value: Any) -> str:
    """Recognize only public, unambiguous LinkedIn post/activity URLs."""
    if not isinstance(value, str) or len(value) > 2000:
        return ""
    try:
        parsed = urlsplit(value)
        if (parsed.scheme != "https" or parsed.netloc not in {"www.linkedin.com", "linkedin.com"}
                or parsed.query or parsed.fragment):
            return ""
    except ValueError:
        return ""
    match = re.fullmatch(r"/feed/update/urn:li:activity:([0-9]{19})/?", parsed.path)
    if match:
        return match.group(1)
    match = re.fullmatch(r"/posts/[A-Za-z0-9_-]+[-_]activity-([0-9]{19})(?:-[A-Za-z0-9]+)?/?", parsed.path)
    if match and len(re.findall(r"activity-[0-9]+", parsed.path)) == 1:
        return match.group(1)
    return ""


def route_for(
    *,
    kind: str,
    funding_source: str,
    round_id: str,
    operation_id: str,
    parameters: Mapping[str, Any],
    timeout_ms: Optional[int] = None,
    allow_host_shadow_score: bool = False,
    source_content_fallback: bool = False,
    source_content_response_url: Optional[str] = None,
    frozen_catalog: Optional[Mapping[str, Any]] = None,
) -> Optional[MinerScoreRoute]:
    """Return one exact Deepline route for an eligible scoring request."""

    if kind != "score" or not (
        funding_source == "miner_key"
        or (funding_source == "host" and allow_host_shadow_score is True)
    ):
        return None
    # The upstream must finish before the broker stops reading its response.
    # Otherwise a normal provider timeout loses the billing receipt as well.
    upstream_timeout_ms = (
        60_000 if timeout_ms is None
        else min(60_000, max(1, timeout_ms - min(5_000, timeout_ms // 2)))
    )
    requested = dict(parameters)
    if operation_id == "scrapingdog.scrape":
        activity_id = _linkedin_post_activity_id(requested.get("url"))
        if activity_id and (
            frozen_catalog is None
            or HARVEST_POST_TOOL in deepline_catalog.allowed_tool_ids(frozen_catalog)
        ):
            return MinerScoreRoute(
                requested_operation_id=operation_id,
                effective_operation_id=EFFECTIVE_OPERATION_ID,
                effective_parameters={"tool": HARVEST_POST_TOOL, "payload": {
                    "url": _LINKEDIN_ACTIVITY_URL.format(activity_id=activity_id),
                }},
                adapter="harvest_linkedin_post_html",
                requested_parameters=requested,
            )
        ats_kind = _ats_api_kind(requested["url"])
        if ats_kind:
            return MinerScoreRoute(
                requested_operation_id=operation_id,
                effective_operation_id=EFFECTIVE_OPERATION_ID,
                effective_parameters={
                    "tool": GENERIC_HTTP_TOOL,
                    "payload": {
                        "url": requested["url"],
                        "method": "GET",
                        "follow_redirects": False,
                        "timeout_ms": upstream_timeout_ms,
                    },
                },
                adapter="generic_ats_json:" + ats_kind,
                requested_parameters=requested,
            )
        if source_content_fallback is True:
            # Only the broker may select this transport after retaining a
            # settled Firecrawl source-content failure. Keep the original
            # source and rendering intent; only its explicit validated final
            # URL may select a different transport destination.
            if not _safe_public_source_https_url(requested["url"]):
                raise operations.OperationRequestError("invalid_url", "url")
            response_url = (
                requested["url"]
                if source_content_response_url is None
                else _safe_public_source_https_url(source_content_response_url)
            )
            if not response_url:
                raise operations.OperationRequestError("invalid_url", "response_url")
            return MinerScoreRoute(
                requested_operation_id=operation_id,
                effective_operation_id=EFFECTIVE_OPERATION_ID,
                effective_parameters={
                    "tool": GENERIC_HTTP_TOOL,
                    "payload": {
                        "url": response_url,
                        "method": "GET",
                        "follow_redirects": True,
                        "timeout_ms": upstream_timeout_ms,
                    },
                },
                adapter="generic_http_raw_html",
                requested_parameters=requested,
            )
        payload = {
            "url": requested["url"],
            "formats": ["rawHtml"],
            "onlyMainContent": False,
            "maxAge": 0,
            "timeout": upstream_timeout_ms,
            "storeInCache": False,
        }
        return MinerScoreRoute(
            requested_operation_id=operation_id,
            effective_operation_id=EFFECTIVE_OPERATION_ID,
            effective_parameters={"tool": FIRECRAWL_TOOL, "payload": payload},
            adapter="firecrawl_raw_html",
            requested_parameters=requested,
        )
    if operation_id == "scrapingdog.x_post":
        tweet_id = str(requested["tweetId"])
        return MinerScoreRoute(
            requested_operation_id=operation_id,
            effective_operation_id=EFFECTIVE_OPERATION_ID,
            effective_parameters={
                "tool": TWITTER_POST_TOOL,
                "payload": {"tweet_ids": tweet_id},
            },
            adapter="twitter_x_post",
            requested_parameters=requested,
        )
    if operation_id == "scrapingdog.linkedinjobs":
        evaluation_date = _evaluation_date(round_id)
        if evaluation_date is None:
            return None
        return MinerScoreRoute(
            requested_operation_id=operation_id,
            effective_operation_id=EFFECTIVE_OPERATION_ID,
            effective_parameters={
                "tool": HARVEST_JOB_TOOL,
                "payload": {"jobId": requested["job_id"]},
            },
            adapter="harvest_linkedin_job",
            requested_parameters=requested,
            evaluation_date=evaluation_date,
        )
    if operation_id == "scrapingdog.profile_post":
        activity_id = str(requested["id"])
        return MinerScoreRoute(
            requested_operation_id=operation_id,
            effective_operation_id=EFFECTIVE_OPERATION_ID,
            effective_parameters={
                "tool": HARVEST_POST_TOOL,
                "payload": {
                    "url": _LINKEDIN_ACTIVITY_URL.format(activity_id=activity_id)
                },
            },
            adapter="harvest_linkedin_post",
            requested_parameters=requested,
        )
    return None


def _envelope_data(body: bytes) -> Any:
    try:
        document = json.loads(bytes(body).decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise CompatibilityResponseError("invalid_deepline_envelope") from exc
    if not isinstance(document, Mapping) or document.get("status") != "completed":
        raise CompatibilityResponseError("incomplete_deepline_envelope")
    result = document.get("result")
    data = result.get("data") if isinstance(result, Mapping) else None
    if not isinstance(data, (Mapping, list)):
        raise CompatibilityResponseError("missing_deepline_result")
    return data


def _target_status(data: Mapping[str, Any], *, default: int = 200) -> int:
    metadata = data.get("metadata")
    candidates = (
        metadata.get("statusCode") if isinstance(metadata, Mapping) else None,
        data.get("status"),
    )
    for candidate in candidates:
        if isinstance(candidate, int) and not isinstance(candidate, bool):
            if 100 <= candidate <= 599:
                return candidate
    return default


def _safe_https_url(value: Any) -> str:
    try:
        return operations.validate_https_url(
            value,
            max_length=2000,
            field="response_url",
        )
    except operations.OperationError:
        return ""


def _safe_public_source_https_url(value: Any) -> str:
    safe_url = _safe_https_url(value)
    if not safe_url:
        return ""
    try:
        # Reuse the frozen catalog's lexical public-target policy. This makes
        # no DNS requests and does not change Firecrawl's existing contract.
        deepline_catalog._public_url(safe_url)
    except deepline_catalog.CatalogError:
        return ""
    return safe_url


def _json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            dict(value) if isinstance(value, Mapping) else value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CompatibilityResponseError("invalid_compatibility_response") from exc


def _parse_datetime(value: Any) -> Optional[datetime]:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _firecrawl_response(
    route: MinerScoreRoute, data: Mapping[str, Any]
) -> tuple[int, dict[str, str], bytes, str]:
    raw_html = data.get("rawHtml")
    metadata = data.get("metadata")
    if not isinstance(raw_html, str) or not isinstance(metadata, Mapping):
        raise CompatibilityResponseError("missing_firecrawl_raw_html")
    requested_url = str(route.requested_parameters.get("url") or "")
    if str(metadata.get("sourceURL") or "") != requested_url:
        raise CompatibilityResponseError("firecrawl_source_url_mismatch")
    final_url_value = metadata.get("url")
    if isinstance(final_url_value, str) and "#" in final_url_value:
        # A fragment does not identify a different fetched resource. Preserve
        # the original metadata's length/character checks before removing it.
        if (
            len(final_url_value) <= 2000
            and "\\" not in final_url_value
            and all(0x21 <= ord(char) < 0x7F for char in final_url_value)
        ):
            final_url_value = final_url_value.split("#", 1)[0]
    final_url = _safe_https_url(final_url_value)
    if not final_url:
        raise CompatibilityResponseError("invalid_firecrawl_final_url")
    status = metadata.get("statusCode")
    if isinstance(status, bool) or not isinstance(status, int) or not 100 <= status <= 599:
        raise CompatibilityResponseError("missing_firecrawl_target_status")
    if 200 <= status < 300 and (
        not raw_html.strip()
        or re.sub(r"\s+", " ", raw_html).strip().casefold()
        in ("provider account capacity details redacted.", "[redacted]")
    ):
        # A completed provider job can still return a diagnostic or placeholder
        # instead of the requested page. Never turn that text into evidence.
        if not _safe_public_source_https_url(requested_url):
            raise CompatibilityResponseError("invalid_firecrawl_source_url")
        public_final_url = _safe_public_source_https_url(final_url)
        if not public_final_url:
            raise CompatibilityResponseError("invalid_firecrawl_final_url")
        failure = CompatibilityResponseError("firecrawl_source_content_unavailable")
        failure.trusted_response_url = public_final_url
        raise failure
    return (
        status,
        {"content-type": "text/html; charset=utf-8"},
        raw_html.encode("utf-8"),
        final_url,
    )


def _generic_http_raw_html_response(
    route: MinerScoreRoute, body: bytes,
) -> tuple[int, dict[str, str], bytes, str]:
    """Require the exact public GET receipt before exposing its source HTML."""

    try:
        document = json.loads(bytes(body).decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise CompatibilityResponseError("invalid_generic_html_envelope") from exc
    tool_response = (
        document.get("toolResponse") if isinstance(document, Mapping) else None
    )
    data = tool_response.get("rawV2") if isinstance(tool_response, Mapping) else None
    if (
        not isinstance(document, Mapping)
        or document.get("status") != "completed"
        or document.get("error") is not None
        or document.get("tool_error") is not None
        or not isinstance(tool_response, Mapping)
        or tool_response.get("view") != "rawV2"
        or not isinstance(data, Mapping)
        or data.get("provider") != "generic_http"
        or data.get("operation") != GENERIC_HTTP_TOOL
        or data.get("method") != "GET"
    ):
        raise CompatibilityResponseError("invalid_generic_html_envelope")
    effective_payload = route.effective_parameters.get("payload")
    effective_url = (
        effective_payload.get("url")
        if isinstance(effective_payload, Mapping) else None
    )
    if (
        not _safe_public_source_https_url(route.requested_parameters.get("url"))
        or not _safe_public_source_https_url(effective_url)
        or data.get("requested_url") != effective_url
        or not _safe_public_source_https_url(data.get("requested_url"))
    ):
        raise CompatibilityResponseError("generic_html_source_url_mismatch")
    final_url = _safe_public_source_https_url(data.get("final_url"))
    if not final_url:
        raise CompatibilityResponseError("invalid_generic_html_final_url")
    status = data.get("status_code")
    if (
        type(status) is not int or not 100 <= status <= 599
        or type(data.get("ok")) is not bool
        or data["ok"] != (200 <= status < 300)
    ):
        raise CompatibilityResponseError("invalid_generic_html_target_status")
    headers = data.get("headers")
    if not isinstance(headers, Mapping) or any(
        not isinstance(name, str) or not isinstance(value, str)
        for name, value in headers.items()
    ):
        raise CompatibilityResponseError("invalid_generic_html_headers")
    if not 200 <= status < 300:
        # Only proved source absence is terminal. Every other failed target
        # fetch must retain the caller's existing bounded Exa recovery path.
        # No error-body content becomes evidence.
        outward_status = status if status in {404, 410} else 502
        return outward_status, {"content-type": "text/html; charset=utf-8"}, b"", final_url
    content_types = [
        value for name, value in headers.items() if name.lower() == "content-type"
    ]
    if len(content_types) != 1:
        raise CompatibilityResponseError("invalid_generic_html_content_type")
    content_type = content_types[0].split(";", 1)[0].strip().casefold()
    if content_type not in {"text/html", "application/xhtml+xml"}:
        raise CompatibilityResponseError("invalid_generic_html_content_type")
    html = data.get("data")
    if not isinstance(html, str):
        raise CompatibilityResponseError("missing_generic_html_body")
    if (
        not html.strip()
        or re.sub(r"\s+", " ", html).strip().casefold()
        in ("provider account capacity details redacted.", "[redacted]")
    ):
        raise CompatibilityResponseError("generic_html_source_content_unavailable")
    return (
        status, {"content-type": content_type + "; charset=utf-8"},
        html.encode("utf-8"), final_url,
    )


def _harvest_data(data: Mapping[str, Any]) -> tuple[int, Mapping[str, Any]]:
    status = _target_status(data)
    element = data.get("element")
    if not isinstance(element, Mapping):
        if status < 400:
            raise CompatibilityResponseError("missing_harvest_element")
        return status, {}
    return status, element


def _harvest_job_response(
    route: MinerScoreRoute, data: Mapping[str, Any]
) -> tuple[int, dict[str, str], bytes]:
    status, element = _harvest_data(data)
    if status >= 400:
        return status, {"content-type": "application/json"}, b"{}"
    requested_id = str(route.requested_parameters.get("job_id") or "")
    if str(element.get("id") or "") != requested_id:
        raise CompatibilityResponseError("harvest_job_id_mismatch")
    title = element.get("title")
    if not isinstance(title, str) or not title.strip():
        raise CompatibilityResponseError("missing_harvest_job_title")
    location = element.get("location")
    location_text = (
        str(location.get("linkedinText") or "")
        if isinstance(location, Mapping)
        else ""
    )
    description = element.get("descriptionText") or element.get("descriptionHtml")
    if not isinstance(description, str):
        description = ""

    job_state = str(element.get("jobState") or "").strip()
    state_parts = [job_state]
    closed_at = _parse_datetime(element.get("closedAt"))
    if (
        element.get("jobApplicationLimitReached") is True
        or job_state.casefold() in {"closed", "suspended"}
        or closed_at is not None
    ):
        state_parts.append("applications closed")
    expires_at = _parse_datetime(element.get("expireAt"))
    evaluation_date = route.evaluation_date
    if expires_at is not None and evaluation_date is not None:
        if expires_at.date() < evaluation_date:
            state_parts.append("expired")

    posted = _parse_datetime(element.get("postedDate"))
    posted_relative = ""
    if posted is not None and evaluation_date is not None:
        age_days = (evaluation_date - posted.date()).days
        if age_days >= 0:
            posted_relative = f"{age_days} days ago"

    company = element.get("company")
    industries = element.get("industries")
    industry_names = [
        str(item.get("name") or "")
        for item in industries
        if isinstance(item, Mapping) and item.get("name")
    ] if isinstance(industries, list) else []
    response = {
        "job_position": title,
        "company_name": (
            str(company.get("name") or "")
            if isinstance(company, Mapping)
            else str(element.get("companyName") or "")
        ),
        "job_location": location_text,
        "Employment_type": str(element.get("employmentType") or ""),
        "Seniority_level": str(element.get("experienceLevel") or ""),
        "Industries": ", ".join(industry_names),
        "number_of_applicants": str(element.get("applicants") or ""),
        "jobs_status": "; ".join(part for part in state_parts if part),
        "job_posting_time": posted_relative,
        "job_description": description,
    }
    return status, {"content-type": "application/json"}, _json_bytes(response)


def _harvest_post_response(
    route: MinerScoreRoute, data: Mapping[str, Any],
) -> tuple[int, dict[str, str], bytes]:
    status, element = _harvest_data(data)
    if status >= 400:
        return status, {"content-type": "application/json"}, b"{}"
    requested_id = str(route.requested_parameters.get("id") or "")
    if str(element.get("id") or "") != requested_id:
        raise CompatibilityResponseError("harvest_post_id_mismatch")
    content = element.get("content")
    if not isinstance(content, str) or not content.strip():
        raise CompatibilityResponseError("missing_harvest_post_content")
    author = element.get("author")
    posted_at = element.get("postedAt")
    response = {
        "post_results": {
            "post_text": content,
            "author": {
                "name": (
                    str(author.get("name") or "")
                    if isinstance(author, Mapping)
                    else ""
                )
            },
            "activity_date": (
                str(posted_at.get("date") or "")
                if isinstance(posted_at, Mapping)
                else ""
            ),
        }
    }
    return status, {"content-type": "application/json"}, _json_bytes(response)


def _harvest_post_html_response(
    route: MinerScoreRoute, data: Mapping[str, Any],
) -> tuple[int, dict[str, str], bytes, str]:
    status, element = _harvest_data(data)
    if status != 200:
        if status >= 400:
            return status, {"content-type": "text/html; charset=utf-8"}, b"", ""
        raise CompatibilityResponseError("invalid_harvest_post_status")
    source_url = route.requested_parameters.get("url")
    activity_id = _linkedin_post_activity_id(source_url)
    # Share the established native post ID/nonempty-content validation without
    # passing its legacy JSON response to a page consumer.
    post_route = MinerScoreRoute(
        requested_operation_id=route.requested_operation_id,
        effective_operation_id=route.effective_operation_id,
        effective_parameters=route.effective_parameters,
        adapter="harvest_linkedin_post",
        requested_parameters={"id": activity_id},
    )
    _harvest_post_response(post_route, data)
    for field in ("url", "linkedinUrl", "postUrl"):
        if field in element and _linkedin_post_activity_id(element[field]) != activity_id:
            raise CompatibilityResponseError("harvest_post_url_mismatch")
    content = element["content"]
    if (not activity_id or len(content.encode("utf-8")) > 131072
            or any(ord(c) < 32 and c not in "\t\r\n" for c in content)):
        raise CompatibilityResponseError("invalid_harvest_post_content")
    # Only verbatim post text is projected. No author/date/HTML facts are added.
    body = ("<!doctype html><html><body><pre>" + html.escape(content)
            + "</pre></body></html>").encode("utf-8")
    return 200, {"content-type": "text/html; charset=utf-8"}, body, source_url


def _twitter_post_response(
    route: MinerScoreRoute, data: Mapping[str, Any]
) -> tuple[int, dict[str, str], bytes]:
    status = _target_status(data)
    if status >= 400:
        return status, {"content-type": "application/json"}, b"{}"
    tweets = data.get("tweets")
    requested_id = str(route.requested_parameters.get("tweetId") or "")
    match = next(
        (
            item
            for item in tweets
            if isinstance(item, Mapping) and str(item.get("id") or "") == requested_id
        ),
        None,
    ) if isinstance(tweets, list) else None
    if not isinstance(match, Mapping):
        raise CompatibilityResponseError("twitter_post_id_mismatch")
    text = match.get("text")
    if not isinstance(text, str) or not text.strip():
        raise CompatibilityResponseError("missing_twitter_post_text")
    author = match.get("author")
    response = {
        "full_tweet": text,
        "created_at": str(match.get("createdAt") or ""),
        "user": {
            "name": str(author.get("name") or "") if isinstance(author, Mapping) else "",
            "screen_name": (
                str(author.get("userName") or "")
                if isinstance(author, Mapping)
                else ""
            ),
        },
    }
    return status, {"content-type": "application/json"}, _json_bytes(response)


def _lever_posting_valid(
    value: Any, *, tenant: str, posting: Optional[str]
) -> bool:
    if not isinstance(value, Mapping):
        return False
    posting_id = value.get("id")
    title = value.get("text")
    hosted_url = value.get("hostedUrl")
    if (
        not isinstance(posting_id, str)
        or _LEVER_POSTING_ID_RE.fullmatch(posting_id) is None
        or posting is not None and posting_id.casefold() != posting.casefold()
        or not isinstance(title, str)
        or not title.strip()
        or not isinstance(hosted_url, str)
    ):
        return False
    expected_hosted_url = "https://jobs.lever.co/%s/%s" % (tenant, posting_id)
    return hosted_url == expected_hosted_url


def _generic_ats_response(
    route: MinerScoreRoute, data: Any
) -> tuple[int, dict[str, str], bytes]:
    reported_status = data.get("status") if isinstance(data, Mapping) else None
    if (
        isinstance(reported_status, int)
        and not isinstance(reported_status, bool)
        and 400 <= reported_status <= 599
    ):
        return reported_status, {"content-type": "application/json"}, _json_bytes(data)
    ats_kind = route.adapter.partition(":")[2]
    lever_match = None
    if ats_kind == "lever":
        try:
            parsed = urlsplit(str(route.requested_parameters.get("url") or ""))
            lever_match = _LEVER_API_PATH_RE.fullmatch(parsed.path)
        except ValueError:
            lever_match = None
        if lever_match is None:
            raise CompatibilityResponseError("invalid_generic_ats_response")
        tenant = lever_match.group("tenant")
        posting = lever_match.group("posting")
        valid_lever = (
            _lever_posting_valid(data, tenant=tenant, posting=posting)
            if posting is not None
            else isinstance(data, list)
            and all(
                _lever_posting_valid(item, tenant=tenant, posting=None)
                for item in data
            )
        )
        if not valid_lever:
            raise CompatibilityResponseError("invalid_generic_ats_response")
        return 200, {"content-type": "application/json"}, _json_bytes(data)
    if not isinstance(data, Mapping):
        raise CompatibilityResponseError("invalid_generic_ats_response")
    valid = (
        ats_kind == "ashby"
        and isinstance(data.get("jobs"), list)
        or ats_kind == "greenhouse"
        and all(key in data for key in ("id", "absolute_url", "title", "company_name", "content"))
        or ats_kind == "greenhouse_board"
        and isinstance(data.get("jobs"), list)
        or ats_kind == "workday"
        and isinstance(data.get("jobPostingInfo"), Mapping)
    )
    if not valid:
        raise CompatibilityResponseError("invalid_generic_ats_response")
    return 200, {"content-type": "application/json"}, _json_bytes(data)


def adapt_response_with_trusted_url(
    route: MinerScoreRoute,
    *,
    status: int,
    headers: Mapping[str, Any],
    body: bytes,
) -> tuple[int, dict[str, str], bytes, str]:
    """Convert a Deepline reply and retain only its validated page URL."""

    del headers
    if status < 200 or status >= 300:
        raise CompatibilityResponseError("deepline_http_error")
    if route.adapter == "generic_http_raw_html":
        return _generic_http_raw_html_response(route, body)
    data = _envelope_data(body)
    if route.adapter.startswith("generic_ats_json:"):
        response_status, response_headers, response_body = _generic_ats_response(
            route, data
        )
        return response_status, response_headers, response_body, ""
    if not isinstance(data, Mapping):
        raise CompatibilityResponseError("missing_deepline_result")
    if route.adapter == "firecrawl_raw_html":
        return _firecrawl_response(route, data)
    if route.adapter == "harvest_linkedin_job":
        response_status, response_headers, response_body = _harvest_job_response(
            route, data
        )
        return response_status, response_headers, response_body, ""
    if route.adapter == "harvest_linkedin_post_html":
        return _harvest_post_html_response(route, data)
    if route.adapter == "harvest_linkedin_post":
        response_status, response_headers, response_body = _harvest_post_response(
            route, data
        )
        return response_status, response_headers, response_body, ""
    if route.adapter == "twitter_x_post":
        response_status, response_headers, response_body = _twitter_post_response(
            route, data
        )
        return response_status, response_headers, response_body, ""
    raise CompatibilityResponseError("unknown_compatibility_adapter")


def adapt_response(
    route: MinerScoreRoute,
    *,
    status: int,
    headers: Mapping[str, Any],
    body: bytes,
) -> tuple[int, dict[str, str], bytes]:
    """Convert one Deepline reply to the requested legacy response shape."""

    response_status, response_headers, response_body, _response_url = (
        adapt_response_with_trusted_url(
            route,
            status=status,
            headers=headers,
            body=body,
        )
    )
    return response_status, response_headers, response_body


__all__ = [
    "COMPATIBILITY_VERSION",
    "CompatibilityResponseError",
    "EFFECTIVE_OPERATION_ID",
    "FIRECRAWL_TOOL",
    "GENERIC_HTTP_TOOL",
    "SOURCE_HTML_ADAPTERS",
    "HARVEST_JOB_TOOL",
    "HARVEST_POST_TOOL",
    "TWITTER_POST_TOOL",
    "MinerScoreRoute",
    "adapt_response",
    "adapt_response_with_trusted_url",
    "route_for",
]
