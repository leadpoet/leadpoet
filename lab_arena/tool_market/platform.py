"""Translate the site's single-operation Deepline payloads into metered Gamma calls."""

from typing import Any, Mapping

from .catalog import CatalogError

OPERATIONS = {
    "zerobounce_validate": ("zerobounce", "validate"),
    "harvestapi_get_profile": ("harvestapi", "linkedin_profile"),
    "harvestapi_search_leads": ("harvestapi", "search_leads"),
    "harvestapi_get_company": ("harvestapi", "linkedin_company"),
    "harvestapi_search_jobs": ("harvestapi", "search_jobs"),
    "harvestapi_get_job": ("harvestapi", "get_job"),
    "harvestapi_get_post_reactions": ("harvestapi", "post_reactions"),
    "harvestapi_get_post_comments": ("harvestapi", "post_comments"),
    "harvestapi_get_profile_posts": ("harvestapi", "profile_posts"),
    "harvestapi_search_posts": ("harvestapi", "search_posts"),
    "findymail_find_from_name": ("findymail", "find_from_name"),
    "findymail_find_from_business_profile": ("findymail", "find_from_linkedin"),
    "hunter_email_finder": ("hunter", "email_finder"),
    "prospeo_enrich_person": ("prospeo", "enrich_person"),
}
SHORT_PROFILE = "harvestapi_get_profile.main"
EMAIL_PROFILE = "harvestapi_get_profile.findEmail"


def _flag(params: dict, name: str) -> bool:
    """Pop an optional boolean flag, sent as a JSON boolean or its string."""

    value = params.pop(name, False)
    if value is not True and value is not False and value not in ("true", "false"):  # not 1 or 0
        raise CatalogError("param_invalid")
    return value is True or value == "true"


def platform_request(alias: str, payload: Mapping[str, Any]) -> tuple:
    """An alias in OPERATIONS and its payload as a catalog call, and the price key it is capped by."""

    tool, name = OPERATIONS[alias]
    params = dict(payload)
    price_key = alias
    if alias == "harvestapi_get_profile":
        if _flag(params, "main"):
            params["main"], price_key = "true", SHORT_PROFILE
        if _flag(params, "findEmail"):  # its price covers the profile, short or not
            name, price_key = "linkedin_profile_email", EMAIL_PROFILE
    elif alias == "harvestapi_get_company" and "universalName" in params:
        # The vendor's documented query accepts a URL, company ID or universal name.
        if "url" in params:
            raise CatalogError("param_invalid")
        params["url"] = params.pop("universalName")
    elif alias == "prospeo_enrich_person" and "enrich_mobile" in params:
        if params.pop("enrich_mobile") is not False:
            raise CatalogError("param_invalid")
    return tool, name, params, price_key
