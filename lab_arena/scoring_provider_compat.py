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

import json
import re
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any, Mapping, Optional
from urllib.parse import urlsplit


COMPATIBILITY_VERSION = "miner-score-deepline-compat:v1"
EFFECTIVE_OPERATION_ID = "deepline.execute"

FIRECRAWL_TOOL = "firecrawl_scrape"
GENERIC_HTTP_TOOL = "generic_http_request"
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
_WORKDAY_API_PATH_RE = re.compile(
    r"^/wday/cxs/[A-Za-z0-9_-]{1,100}/[A-Za-z0-9_-]{1,100}/job/"
    r"[A-Za-z0-9_./-]{3,500}$"
)


class CompatibilityResponseError(ValueError):
    """The provider response cannot satisfy the legacy judge contract."""


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
    """Classify only the three exact public JSON transports used by scoring."""

    try:
        parsed = urlsplit(str(value or ""))
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
        host.endswith(".myworkdayjobs.com")
        and not parsed.query
        and _WORKDAY_API_PATH_RE.fullmatch(parsed.path)
    ):
        return "workday"
    return ""


def route_for(
    *,
    kind: str,
    funding_source: str,
    round_id: str,
    operation_id: str,
    parameters: Mapping[str, Any],
) -> Optional[MinerScoreRoute]:
    """Return one exact Deepline route for an eligible scoring request."""

    if kind != "score" or funding_source != "miner_key":
        return None
    requested = dict(parameters)
    if operation_id == "scrapingdog.scrape":
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
                        "timeout_ms": 60_000,
                    },
                },
                adapter="generic_ats_json:" + ats_kind,
                requested_parameters=requested,
            )
        payload = {
            "url": requested["url"],
            "formats": ["rawHtml"],
            "onlyMainContent": False,
            "maxAge": 0,
            "timeout": 60_000,
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


def _envelope_data(body: bytes) -> Mapping[str, Any]:
    try:
        document = json.loads(bytes(body).decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise CompatibilityResponseError("invalid_deepline_envelope") from exc
    if not isinstance(document, Mapping) or document.get("status") != "completed":
        raise CompatibilityResponseError("incomplete_deepline_envelope")
    result = document.get("result")
    data = result.get("data") if isinstance(result, Mapping) else None
    if not isinstance(data, Mapping):
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
    text = str(value or "")
    try:
        parts = urlsplit(text)
        port = parts.port
    except ValueError:
        return ""
    if (
        parts.scheme != "https"
        or not parts.hostname
        or parts.username is not None
        or parts.password is not None
        or parts.fragment
        or (port is not None and port != 443)
    ):
        return ""
    return text


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            dict(value),
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
) -> tuple[int, dict[str, str], bytes]:
    raw_html = data.get("rawHtml")
    metadata = data.get("metadata")
    if not isinstance(raw_html, str) or not isinstance(metadata, Mapping):
        raise CompatibilityResponseError("missing_firecrawl_raw_html")
    requested_url = str(route.requested_parameters.get("url") or "")
    if str(metadata.get("sourceURL") or "") != requested_url:
        raise CompatibilityResponseError("firecrawl_source_url_mismatch")
    final_url = _safe_https_url(metadata.get("url"))
    if not final_url:
        raise CompatibilityResponseError("invalid_firecrawl_final_url")
    status = metadata.get("statusCode")
    if isinstance(status, bool) or not isinstance(status, int) or not 100 <= status <= 599:
        raise CompatibilityResponseError("missing_firecrawl_target_status")
    return status, {"content-type": "text/html; charset=utf-8"}, raw_html.encode(
        "utf-8"
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


def _generic_ats_response(
    route: MinerScoreRoute, data: Mapping[str, Any]
) -> tuple[int, dict[str, str], bytes]:
    reported_status = data.get("status")
    if (
        isinstance(reported_status, int)
        and not isinstance(reported_status, bool)
        and 400 <= reported_status <= 599
    ):
        return reported_status, {"content-type": "application/json"}, _json_bytes(data)
    ats_kind = route.adapter.partition(":")[2]
    valid = (
        ats_kind == "ashby"
        and isinstance(data.get("jobs"), list)
        or ats_kind == "greenhouse"
        and all(key in data for key in ("id", "absolute_url", "title", "company_name", "content"))
        or ats_kind == "workday"
        and isinstance(data.get("jobPostingInfo"), Mapping)
    )
    if not valid:
        raise CompatibilityResponseError("invalid_generic_ats_response")
    return 200, {"content-type": "application/json"}, _json_bytes(data)


def adapt_response(
    route: MinerScoreRoute,
    *,
    status: int,
    headers: Mapping[str, Any],
    body: bytes,
) -> tuple[int, dict[str, str], bytes]:
    """Convert one Deepline reply to the requested legacy response shape."""

    del headers
    if status < 200 or status >= 300:
        raise CompatibilityResponseError("deepline_http_error")
    data = _envelope_data(body)
    if route.adapter == "firecrawl_raw_html":
        return _firecrawl_response(route, data)
    if route.adapter == "harvest_linkedin_job":
        return _harvest_job_response(route, data)
    if route.adapter == "harvest_linkedin_post":
        return _harvest_post_response(route, data)
    if route.adapter == "twitter_x_post":
        return _twitter_post_response(route, data)
    if route.adapter.startswith("generic_ats_json:"):
        return _generic_ats_response(route, data)
    raise CompatibilityResponseError("unknown_compatibility_adapter")


__all__ = [
    "COMPATIBILITY_VERSION",
    "CompatibilityResponseError",
    "EFFECTIVE_OPERATION_ID",
    "FIRECRAWL_TOOL",
    "GENERIC_HTTP_TOOL",
    "HARVEST_JOB_TOOL",
    "HARVEST_POST_TOOL",
    "TWITTER_POST_TOOL",
    "MinerScoreRoute",
    "adapt_response",
    "route_for",
]
