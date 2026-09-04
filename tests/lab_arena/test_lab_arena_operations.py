"""Closed operation table tests (labarena.md sections 7.4 and 18.4)."""

from __future__ import annotations

import json
from urllib.parse import urlencode

import pytest

from lab_arena import contracts
from lab_arena import operations as ops

D_EXECUTE = "https://code.deepline.com/api/v2/integrations/exa_search/execute"

VALID = {
    "deepline.execute": {
        "tool": "exa_search",
        "payload": {"query": "fintech startups in berlin", "type": "fast", "numResults": 5, "includeDomains": ["Example.com"]},
    },
    "scrapingdog.scrape": {"url": "https://example.com/jobs?page=2", "dynamic": True, "wait": 5000, "premium": True, "stealth_mode": True},
    "scrapingdog.x_post": {"tweetId": "1234567890"},
    "scrapingdog.x_profile": {"profileId": "acme"},
    "scrapingdog.profile": {"type": "company", "id": "acme-inc"},
    "scrapingdog.profile_post": {"id": "7123456789"},
    "scrapingdog.linkedinjobs": {"job_id": "4012345678"},
    "scrapingdog.jobs": {"job_id": "abc123"},
    "scrapingdog.indeed": {"url": "https://www.indeed.com/viewjob?jk=1"},
    "scrapingdog.instagram_profile": {"username": "acme"},
    "scrapingdog.tiktok_profile": {"username": "acme"},
    "scrapingdog.youtube_video": {"v": "dQw4w9WgXcQ"},
    "scrapingdog.youtube_transcripts": {"v": "dQw4w9WgXcQ"},
    "scrapingdog.youtube_channel": {"channel_id": "UC123"},
    "scrapingdog.youtube_search": {"search_query": "acme funding"},
    "exa.search": {"query": "acme series b", "type": "auto", "numResults": 5, "contents": {"highlights": {"query": "series b", "maxCharacters": 400}}},
    "exa.contents": {"ids": ["https://example.com/news"], "text": {"maxCharacters": 12000}, "maxAgeHours": 0},
    "scrapingdog.google": {"query": "acme corp", "country": "gb"},
    "scrapingdog.google_news": {"query": "acme launch", "country": "gb"},
    "scrapingdog.google_jobs": {"query": "acme cloud jobs", "country": "gb"},
    "openrouter.chat": {
        "model": "openai/gpt-4o-mini",
        "messages": [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}],
        "temperature": 0.2,
        "max_tokens": 256,
        "top_p": 0.9,
        "stop": ["\n"],
        "seed": 7,
        "response_format": {"type": "text"},
    },
}

HOSTILE_VALUES = (
    "https://evil.example/x",
    "http://code.deepline.com/api/v2/integrations/exa_search/execute",
    "https://code.deepline.com@evil.example/",
    "https://user:pw@example.com/",
    "https://127.0.0.1/",
    "https://[::1]/",
    "https://2130706433/",
    "https://example.com:8443/",
    "https://example.com/#frag",
    "evil.example",
    "../../etc/passwd",
    "\x00",
    "a" * 100_000,
    10 ** 9,
    -1,
    1.5,
    True,
    None,
    [],
    ["https://evil.example"],
    {},
    {"host": "evil.example"},
    {"url": "https://evil.example"},
)

CREDENTIAL_FIELD_NAMES = (
    "host",
    "url",
    "headers",
    "Authorization",
    "apikey",
    "api_key",
    "x-api-key",
    "cookie",
    "redirect",
    "follow_redirects",
    "allow_redirects",
)


def request_for(operation_id: str, parameters=None):
    operation = ops.OPERATIONS[operation_id]
    parameters = VALID[operation_id] if parameters is None else parameters
    path = operation.path
    for name in operation.path_fields:
        path = path.replace("{%s}" % name, str(parameters[name]))
    url = "https://%s%s" % (operation.host, path)
    if operation.method == "POST":
        body = {name: value for name, value in parameters.items() if name not in operation.path_fields}
        return operation.method, url, json.dumps(body).encode("utf-8"), {"Content-Type": "application/json"}
    query = urlencode({name: ("true" if value is True else "false" if value is False else value) for name, value in parameters.items()})
    return operation.method, url + "?" + query, b"", {}


def reject(operation_id: str, parameters, code=None):
    with pytest.raises(ops.OperationRequestError) as excinfo:
        ops.validate_operation_request(operation_id, parameters)
    assert excinfo.value.code in ops.ERROR_CODES
    if code is not None:
        assert excinfo.value.code == code
    return excinfo.value


# ---------------------------------------------------------------------------
# Table integrity
# ---------------------------------------------------------------------------


def test_table_is_closed_and_every_operation_uses_host_credentials():
    judge_scrapingdog = {"scrapingdog.x_post", "scrapingdog.x_profile", "scrapingdog.profile", "scrapingdog.profile_post", "scrapingdog.linkedinjobs", "scrapingdog.jobs", "scrapingdog.indeed", "scrapingdog.instagram_profile", "scrapingdog.tiktok_profile", "scrapingdog.youtube_video", "scrapingdog.youtube_transcripts", "scrapingdog.youtube_channel", "scrapingdog.youtube_search"}
    assert set(ops.OPERATIONS) == {
        "deepline.execute",
        "exa.search",
        "exa.contents",
        "scrapingdog.scrape",
        "scrapingdog.google",
        "scrapingdog.google_news",
        "scrapingdog.google_jobs",
        "openrouter.chat",
    } | judge_scrapingdog
    assert ops.PROVIDERS == contracts.PROVIDERS == ("scrapingdog", "deepline", "openrouter")
    assert ops.FUNDING_SOURCES == ("host",)
    for operation_id, operation in ops.OPERATIONS.items():
        assert operation.operation_id == operation_id
        assert operation.provider in ops.PROVIDERS
        assert operation.host in ("code.deepline.com", "api.exa.ai", "api.scrapingdog.com", "openrouter.ai")
        if operation.host == "api.exa.ai":
            assert operation.provider == "deepline" and operation.outbound_host == "code.deepline.com" and operation.deepline_tool in ops.DEEPLINE_TOOLS
        assert operation.method in ops.METHODS
        assert operation.funding_source == "host"
        # A price-determining parameter is never model-controllable.
        assert not set(operation.fixed_params) & set(operation.request_fields)
        if operation.provider == "openrouter":
            assert operation.cost_rule["kind"] == "openrouter_price_table"
            assert operation.cost_rule["max_output_tokens"] == ops.OPENROUTER_MAX_OUTPUT_TOKENS
        else:
            assert operation.cost_rule == {"kind": "call_quota"}
    deepline = ops.OPERATIONS["deepline.execute"]
    assert deepline.path == "/api/v2/integrations/{tool}/execute" and deepline.path_fields == ("tool",)
    assert deepline.request_fields["tool"].choices == ops.DEEPLINE_TOOLS and deepline.request_fields["payload"].fields is None
    assert deepline.credential.location == "header" and deepline.credential.scheme == "Bearer"
    assert ops.OPERATIONS["scrapingdog.scrape"].fixed_params == {}
    assert ops.OPERATIONS["scrapingdog.google"].fixed_params == {"results": 10}
    assert ops.OPERATIONS["scrapingdog.google_news"].fixed_params == {"results": 10}
    assert ops.OPERATIONS["scrapingdog.google_jobs"].fixed_params == {}
    chat = ops.OPERATIONS["openrouter.chat"].fixed_params
    assert chat["stream"] is False
    assert chat["provider"] == {"data_collection": "deny", "allow_fallbacks": False, "zdr": True}
    assert not hasattr(ops, "PROVIDER_PRICE_LIST") and not hasattr(ops, "fixed_cost_microusd")


def test_deepline_path_field_is_closed_and_rendered_into_the_outbound_path():
    for tool in ops.DEEPLINE_TOOLS:
        outbound = ops.build_outbound_request("deepline.execute", {"tool": tool, "payload": {"query": "x"}})
        assert outbound.url == "https://code.deepline.com/api/v2/integrations/%s/execute" % tool
        assert json.loads(outbound.body) == {"provider": ops.DEEPLINE_TOOL_PROVIDERS[tool], "operation": tool, "payload": {"query": "x"}}  # the path field names the tool, never a body key
        assert dict(outbound.headers) == {"x-deepline-execute-response-intent": "raw"}
    reject("deepline.execute", {"tool": "exa_search/../admin", "payload": {}}, "invalid_field")
    reject("deepline.execute", {"tool": "unknown_tool", "payload": {}}, "invalid_field")
    reject("deepline.execute", {"payload": {"query": "x"}}, "missing_field")
    reject("deepline.execute", {"tool": "exa_search"}, "missing_field")
    reject("deepline.execute", {"tool": "exa_search", "payload": []}, "invalid_field")
    reject("deepline.execute", {"tool": "exa_search", "payload": {"query": "x", "api_key": "k"}}, "forbidden_field")
    reject("deepline.execute", {"tool": "exa_search", "payload": {"contents": {"x-api-key": "k"}}}, "forbidden_field")
    assert ops.validate_operation_request("deepline.execute", {"tool": "exa_contents", "payload": {"urls": ["https://example.com/"], "text": True}})["payload"]["urls"] == ["https://example.com/"]
    reject("deepline.execute", {"tool": "exa_search", "payload": {"n": float("inf")}}, "invalid_request")  # structural check first
    with pytest.raises(ops.OperationRequestError) as excinfo:
        ops.match_request("POST", D_EXECUTE, b'{"tool": "exa_contents", "payload": {}}', {})
    assert excinfo.value.code == "invalid_body"  # the body may not restate the path field


def test_operations_are_immutable():
    operation = ops.OPERATIONS["scrapingdog.google"]
    with pytest.raises(TypeError):
        operation.fixed_params["results"] = 100  # type: ignore[index]
    with pytest.raises(TypeError):
        ops.OPERATIONS["evil"] = operation  # type: ignore[index]
    with pytest.raises(Exception):
        operation.host = "evil.example"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Valid requests and normalization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("operation_id", sorted(ops.OPERATIONS))
def test_valid_request_matches_exactly_one_operation(operation_id):
    matched, parameters = ops.match_request(*request_for(operation_id))
    assert matched == operation_id
    assert set(parameters) <= set(ops.OPERATIONS[operation_id].request_fields)
    for name in ops.OPERATIONS[operation_id].defaults:
        assert name in parameters


def test_normalization_fills_defaults_and_passes_tool_payloads_through():
    parameters = ops.validate_operation_request("deepline.execute", {"tool": "exa_search", "payload": {"query": "x", "includeDomains": ["Example.COM"]}})
    assert parameters == {"tool": "exa_search", "payload": {"query": "x", "includeDomains": ["Example.COM"]}}
    chat = ops.validate_operation_request("openrouter.chat", {"model": "openai/gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]})
    assert chat["max_tokens"] == ops.OPENROUTER_MAX_OUTPUT_TOKENS
    assert ops.validate_operation_request("scrapingdog.google", {"query": "acme"})["country"] == "us"
    assert ops.validate_operation_request("scrapingdog.google_news", {"query": "acme"})["country"] == "us"
    assert ops.validate_operation_request("scrapingdog.google_jobs", {"query": "acme"})["country"] == "us"
    assert ops.validate_operation_request("scrapingdog.scrape", {"url": "https://example.com/"})["dynamic"] is False


def test_public_pydantic_baseline_provider_routes_are_allowed() -> None:
    for tool in (
        "hunter_discover",
        "free_simple_company_search",
        "predictleads_company_job_openings",
        "predictleads_company_financing_events",
        "predictleads_company_news_events",
    ):
        operation, parameters = ops.match_request(
            "POST",
            "https://code.deepline.com/api/v2/integrations/%s/execute" % tool,
            b'{"payload":{"query":"x"}}',
            {"content-type": "application/json"},
        )
        assert operation == "deepline.execute"
        assert parameters["tool"] == tool
    for endpoint, operation_id in (
        ("google", "scrapingdog.google"),
        ("google_news", "scrapingdog.google_news"),
        ("google_jobs", "scrapingdog.google_jobs"),
    ):
        operation, parameters = ops.match_request(
            "GET",
            "https://api.scrapingdog.com/%s?query=acme" % endpoint,
            b"",
            {},
        )
        assert operation == operation_id
        assert parameters["query"] == "acme"


def test_null_optional_fields_are_dropped_and_null_required_fields_are_missing():
    assert ops.validate_operation_request("scrapingdog.google", {"query": "x", "country": None}) == {"query": "x", "country": "us"}
    reject("scrapingdog.google", {"query": None}, "missing_field")


# ---------------------------------------------------------------------------
# Exact-operation rejections: path, method, query, header, credentials
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "method,url",
    [
        ("POST", "https://code.deepline.com/api/v2/integrations/exa_search/execute/"),
        ("POST", "https://code.deepline.com/api/v2/integrations/exa_search/execute/extra"),
        ("POST", "https://code.deepline.com/api/v2/integrations/exa_search/Execute"),
        ("POST", "https://code.deepline.com/api/v1/integrations/exa_search/execute"),
        ("POST", "https://code.deepline.com/api/v2/integrations/execute"),
        ("POST", "https://code.deepline.com/api/v2/integrations//execute"),
        ("POST", "https://code.deepline.com/api/v2/integrations/exa%20search/execute"),
        ("GET", "https://code.deepline.com/api/v2/integrations/exa_search/execute"),
        ("GET", "https://code.deepline.com/api/v2/integrations/exa_search/get"),
        ("PUT", "https://code.deepline.com/api/v2/integrations/exa_search/execute"),
        ("POST", "http://code.deepline.com/api/v2/integrations/exa_search/execute"),
        ("POST", "https://code.deepline.com:8443/api/v2/integrations/exa_search/execute"),
        ("POST", "https://code.deepline.com:80/api/v2/integrations/exa_search/execute"),
        ("POST", "https://user@code.deepline.com/api/v2/integrations/exa_search/execute"),
        ("POST", "https://code.deepline.com@evil.example/api/v2/integrations/exa_search/execute"),
        ("POST", "https://code.deepline.com/api/v2/integrations/exa_search/execute#fragment"),
        ("POST", "https://evil.example/api/v2/integrations/exa_search/execute"),
        ("POST", "https://code.deepline.com.evil.example/api/v2/integrations/exa_search/execute"),
        ("POST", "https://sub.code.deepline.com/api/v2/integrations/exa_search/execute"),
        ("POST", "https://code.deepline.com./api/v2/integrations/exa_search/execute"),
        ("POST", "https://127.0.0.1/api/v2/integrations/exa_search/execute"),
        ("POST", "https://[::1]/api/v2/integrations/exa_search/execute"),
        ("POST", "https://2130706433/api/v2/integrations/exa_search/execute"),
        ("POST", "https://code.deepline.com/api/v2/integrations/exa_search/execute\n"),
        ("GET", "https://api.scrapingdog.com/scrape/"),
        ("POST", "https://api.scrapingdog.com/scrape"),
        ("POST", "https://openrouter.ai/api/v1/completions"),
        ("POST", "https://openrouter.ai/api/v1/chat/completions/"),
        ("GET", "https://openrouter.ai/api/v1/models"),
    ],
)
def test_unknown_path_method_host_port_userinfo_and_fragment_match_nothing(method, url):
    with pytest.raises(ops.OperationRequestError) as excinfo:
        ops.match_request(method, url, b'{"payload": {"query": "x"}}', {})
    assert excinfo.value.code == "no_matching_operation"


def test_explicit_port_443_is_the_same_operation():
    assert ops.match_request("POST", "https://code.deepline.com:443/api/v2/integrations/exa_search/execute", b'{"payload": {"query": "x"}}', {})[0] == "deepline.execute"


def test_query_string_on_post_operation_is_rejected():
    with pytest.raises(ops.OperationRequestError) as excinfo:
        ops.match_request("POST", D_EXECUTE + "?numResults=100", b'{"payload": {"query": "x"}}', {})
    assert excinfo.value.code == "invalid_query"


@pytest.mark.parametrize(
    "query,code",
    [
        ("url=https%3A%2F%2Fexample.com%2F&api_key=abc", "forbidden_field"),
        ("url=https%3A%2F%2Fexample.com%2F&render=true", "unknown_field"),
        ("url=https%3A%2F%2Fexample.com%2F&url=https%3A%2F%2Fevil.example%2F", "invalid_query"),
        ("url", "invalid_query"),
        ("url=https%3A%2F%2Fexample.com%2F&dynamic=maybe", "invalid_field"),
        ("dynamic=true", "missing_field"),
    ],
)
def test_get_operation_query_rules(query, code):
    with pytest.raises(ops.OperationRequestError) as excinfo:
        ops.match_request("GET", "https://api.scrapingdog.com/scrape?" + query, b"", {})
    assert excinfo.value.code == code


def test_get_operation_rejects_a_body_and_coerces_booleans():
    with pytest.raises(ops.OperationRequestError) as excinfo:
        ops.match_request("GET", "https://api.scrapingdog.com/scrape?url=https%3A%2F%2Fexample.com%2F", b"{}", {})
    assert excinfo.value.code == "invalid_body"
    for raw, expected in (("true", True), ("1", True), ("false", False), ("0", False)):
        _, parameters = ops.match_request("GET", "https://api.scrapingdog.com/scrape?url=https%3A%2F%2Fexample.com%2F&dynamic=" + raw, b"", {})
        assert parameters["dynamic"] is expected


@pytest.mark.parametrize(
    "header,code",
    [
        ("Authorization", "forbidden_header"),
        ("authorization", "forbidden_header"),
        ("X-API-Key", "forbidden_header"),
        ("x-api-key", "forbidden_header"),
        ("Cookie", "forbidden_header"),
        ("Proxy-Authorization", "forbidden_header"),
        ("X-Auth-Token", "forbidden_header"),
        ("Origin", "unknown_header"),
        ("X-Request-Id", "unknown_header"),
        ("X-Forwarded-For", "unknown_header"),
        ("X-Stainless-Lang", "unknown_header"),
    ],
)
def test_caller_credentials_and_unknown_headers_are_refused(header, code):
    with pytest.raises(ops.OperationRequestError) as excinfo:
        ops.match_request("POST", D_EXECUTE, b'{"payload": {"query": "x"}}', {header: "value"})
    assert excinfo.value.code == code
    with pytest.raises(ops.OperationRequestError):
        ops.match_request("GET", "https://api.scrapingdog.com/google?query=x", b"", {header: "value"})


def test_benign_client_headers_are_accepted_and_never_forwarded():
    headers = {
        "Host": "code.deepline.com",
        "User-Agent": "python-requests/2.32",
        "Accept": "*/*",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
        "Content-Length": "14",
        "Content-Type": "application/json; charset=utf-8",
    }
    operation_id, parameters = ops.match_request("POST", D_EXECUTE, b'{"payload": {"query": "x"}}', headers)
    assert operation_id == "deepline.execute"
    outbound = ops.build_outbound_request(operation_id, parameters)
    assert "User-Agent" not in json.dumps(outbound.body.decode("utf-8"))
    assert outbound.content_type == "application/json"


@pytest.mark.parametrize("body", [b"", b"[]", b'"x"', b"not json", b"\xff\xfe", b"{\"query\": \"x\"} trailing"])
def test_post_body_must_be_a_json_object(body):
    with pytest.raises(ops.OperationRequestError) as excinfo:
        ops.match_request("POST", D_EXECUTE, body, {})
    assert excinfo.value.code in ("invalid_body", "missing_field")


def test_oversized_raw_body_and_nesting_bombs_are_rejected_before_validation():
    padded = json.dumps({"payload": {"query": "x", "pad": "y" * 70_000}}).encode("utf-8")
    with pytest.raises(ops.OperationRequestError) as excinfo:
        ops.match_request("POST", D_EXECUTE, padded, {})
    assert excinfo.value.code == "request_too_large"
    bomb: dict = {"query": "x"}
    cursor = bomb
    for _ in range(20):
        cursor["n"] = {}
        cursor = cursor["n"]
    reject("deepline.execute", {"tool": "exa_search", "payload": bomb}, "invalid_request")
    reject("deepline.execute", {"tool": "exa_search", "payload": {"query": "x\x00y"}}, "invalid_request")
    reject("deepline.execute", "not a mapping", "invalid_request")


# ---------------------------------------------------------------------------
# Field rules
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("operation_id", sorted(ops.OPERATIONS))
@pytest.mark.parametrize("name", CREDENTIAL_FIELD_NAMES)
def test_forbidden_fields_are_refused_on_every_operation(operation_id, name):
    if name in ops.OPERATIONS[operation_id].request_fields:
        pytest.skip("declared field")
    parameters = dict(VALID[operation_id])
    parameters[name] = "https://evil.example/"
    reject(operation_id, parameters, "forbidden_field")


@pytest.mark.parametrize("operation_id", sorted(ops.OPERATIONS))
def test_unknown_fields_are_refused_on_every_operation(operation_id):
    parameters = dict(VALID[operation_id])
    parameters["not_a_declared_field"] = 100
    reject(operation_id, parameters, "unknown_field")


@pytest.mark.parametrize(
    "url",
    [
        "http://example.com/",
        "https://example.com:8443/",
        "https://example.com:80/",
        "https://user@example.com/",
        "https://user:pw@example.com/",
        "https://example.com/#frag",
        "https://example.com/path#",
        "https://127.0.0.1/",
        "https://10.0.0.1/admin",
        "https://[::1]/",
        "https://[fe80::1]/",
        "https://2130706433/",
        "https://0x7f000001/",
        "https://localhost/",
        "https://intranet/",
        "https://example.com\\evil/",
        "https://exam ple.com/",
        "https://example.com/\n",
        "https://exämple.com/",
        "https://example.com:notaport/",
        "ftp://example.com/",
        "//example.com/",
        "",
        "https://",
        "https:///path",
    ],
)
def test_https_only_url_rules(url):
    # Strings shorter than the field minimum fail the length rule first; the
    # code is still generic and the request is still refused.
    expected = ("invalid_url", "invalid_field") if len(url) < 8 else ("invalid_url",)
    assert reject("scrapingdog.scrape", {"url": url}).code in expected
    assert reject("scrapingdog.scrape", {"url": url}).code in expected


@pytest.mark.parametrize(
    "url",
    ["https://example.com/", "https://example.com:443/path", "https://sub.example.co.uk/jobs?page=2&sort=asc", "https://example.com"],
)
def test_accepted_target_urls_are_returned_unchanged(url):
    assert ops.validate_operation_request("scrapingdog.scrape", {"url": url})["url"] == url


def test_scrapingdog_scrape_bounds_url():
    reject("scrapingdog.scrape", {"url": ""}, "invalid_field")
    reject("scrapingdog.scrape", {"url": ["https://example.com/"]}, "invalid_field")
    reject("scrapingdog.scrape", {"url": 5}, "invalid_field")
    reject("scrapingdog.scrape", {}, "missing_field")


def test_deepline_execute_field_rules():
    reject("deepline.execute", {"tool": "", "payload": {}}, "invalid_field")
    reject("deepline.execute", {"tool": "exa_search" * 20, "payload": {}}, "invalid_field")
    reject("deepline.execute", {"tool": 5, "payload": {}}, "invalid_field")
    reject("deepline.execute", {"tool": "exa_search", "payload": "text"}, "invalid_field")
    reject("deepline.execute", {"tool": "exa_search", "payload": {"query": "x", "extra": {"nested": {"api_key": "k"}}}}, "forbidden_field")
    reject("deepline.execute", {"tool": "exa_search", "payload": {"query": "x"}, "url": "https://evil.example/"}, "forbidden_field")
    # Deepline owns tool schemas: payload fields pass through untouched, including ones the old Exa table capped.
    passed = ops.validate_operation_request("deepline.execute", {"tool": "exa_people_search", "payload": {"query": "x", "numResults": 100, "includeDomains": ["Example.com"] * 21}})
    assert passed["payload"]["numResults"] == 100 and len(passed["payload"]["includeDomains"]) == 21


def test_scrapingdog_google_field_rules():
    reject("scrapingdog.google", {"query": "x" * 501}, "invalid_field")
    reject("scrapingdog.google", {"query": "x", "country": "zz"}, "invalid_field")
    reject("scrapingdog.google", {"query": "x", "results": 100}, "unknown_field")


@pytest.mark.parametrize(
    "extra,code",
    [
        ({"functions": []}, "forbidden_field"),
        ({"plugins": [{"id": "web"}]}, "forbidden_field"),
        ({"stream": True}, "forbidden_field"),
        ({"stream": False}, "forbidden_field"),
        ({"transforms": ["middle-out"]}, "forbidden_field"),
        ({"route": "fallback"}, "forbidden_field"),
        ({"models": ["openai/gpt-4o"]}, "forbidden_field"),
        ({"web_search_options": {}}, "forbidden_field"),
        ({"modalities": ["image"]}, "forbidden_field"),
        ({"response_format": {"type": "json"}}, "invalid_field"),
        ({"response_format": {"type": "json_schema", "schema": {}}}, "unknown_field"),
        ({"provider": {"allow_fallbacks": True}}, "invalid_request"),
        ({"provider": {"data_collection": "deny", "order": ["openai"]}}, "invalid_request"),
        ({"provider": {}}, "invalid_request"),
        ({"provider": "deny"}, "invalid_request"),
        ({"max_tokens": 0}, "invalid_field"),
        ({"max_tokens": 4097}, "invalid_field"),
        ({"max_tokens": 2.0}, "invalid_field"),
        ({"temperature": 2.1}, "invalid_field"),
        ({"temperature": -0.1}, "invalid_field"),
        ({"top_p": 1.5}, "invalid_field"),
        ({"stop": ["a", "b", "c", "d", "e"]}, "invalid_field"),
        ({"seed": "7"}, "invalid_field"),
        ({"model": "gpt-4o-mini"}, "invalid_field"),
        ({"model": "openai/gpt 4o"}, "invalid_field"),
        ({"messages": []}, "invalid_field"),
        ({"messages": [{"role": "user", "content": [{"type": "text", "text": "x"}]}]}, "invalid_field"),
        ({"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://x"}}]}]}, "invalid_field"),
        ({"tools": [{"type": "function", "function": "bad"}]}, "invalid_field"),
        ({"messages": [{"role": "assistant", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "search"}}]}]}, "missing_field"),
        ({"messages": [{"role": "user", "content": "x" * 32_001}]}, "invalid_field"),
        ({"messages": [{"role": "user", "content": "x"}] * 65}, "invalid_field"),
    ],
)
def test_openrouter_chat_rejections(extra, code):
    parameters = {"model": "openai/gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]}
    parameters.update(extra)
    reject("openrouter.chat", parameters, code)


def test_openrouter_chat_accepts_the_judge_request_shape():
    """The judge's privacy request is a subset of the pinned policy and is dropped; JSON reply formats pass."""

    base = {"model": "openai/gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}], "temperature": 0, "max_tokens": 400}
    superseded = ops.validate_operation_request("openrouter.chat", dict(base, provider={"data_collection": "deny", "zdr": True}))
    assert "provider" not in superseded and superseded["max_tokens"] == 400
    assert ops.validate_operation_request("openrouter.chat", dict(base, response_format={"type": "json_object"}))["response_format"] == {"type": "json_object"}
    schema = {"type": "json_schema", "json_schema": {"name": "verify", "strict": True, "schema": {"type": "object", "properties": {"verified": {"type": "boolean"}}, "required": ["verified"]}}}
    assert ops.validate_operation_request("openrouter.chat", dict(base, response_format=schema))["response_format"] == schema
    outbound = ops.build_outbound_request("openrouter.chat", ops.validate_operation_request("openrouter.chat", dict(base, provider={"zdr": True})))
    assert json.loads(outbound.body)["provider"] == {"data_collection": "deny", "allow_fallbacks": False, "zdr": True}
    ops.check_request_headers({"Content-Type": "application/json", "HTTP-Referer": "https://leadpoet.ai", "X-Title": "Leadpoet Qualification"})


def test_openrouter_chat_accepts_common_agent_tool_and_reasoning_fields():
    request = {
        "model": "openai/gpt-4o-mini",
        "messages": [
            {"role": "developer", "content": "Use the search tool."},
            {"role": "assistant", "tool_calls": [{"id": "call-1", "type": "function", "function": {"name": "search", "arguments": '{"query":"acme"}'}}]},
            {"role": "tool", "tool_call_id": "call-1", "content": "result"},
        ],
        "tools": [{"type": "function", "function": {"name": "search", "description": "Search", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}}}}],
        "tool_choice": "auto",
        "parallel_tool_calls": False,
        "reasoning": {"effort": "medium"},
        "reasoning_effort": "medium",
    }
    normalized = ops.validate_operation_request("openrouter.chat", request)
    assert normalized["messages"][1]["tool_calls"][0]["function"]["name"] == "search"
    assert normalized["tools"][0]["function"]["name"] == "search"
    assert normalized["tool_choice"] == "auto" and normalized["reasoning_effort"] == "medium"


def test_openrouter_chat_total_size_is_bounded():
    big = {"model": "openai/gpt-4o-mini", "messages": [{"role": "user", "content": "x" * 32_000}] * 40}
    reject("openrouter.chat", big, "request_too_large")


# ---------------------------------------------------------------------------
# No request field can alter host, path, method, headers, or credentials
# ---------------------------------------------------------------------------


def _mutations(operation: ops.Operation):
    for name in operation.request_fields:
        for value in HOSTILE_VALUES:
            yield name, value
    for name in sorted(ops.FORBIDDEN_FIELD_NAMES | {"numResults", "premium", "results", "contents", "text"}):
        yield name, "https://evil.example/"
        yield name, {"host": "evil.example"}


@pytest.mark.parametrize("operation_id", sorted(ops.OPERATIONS))
def test_outbound_target_is_constant_under_every_field_mutation(operation_id):
    operation = ops.OPERATIONS[operation_id]
    baseline = ops.outbound_target(operation_id, VALID[operation_id])
    rendered_path = operation.outbound_path or operation.path
    for field_name in operation.path_fields:
        rendered_path = rendered_path.replace("{%s}" % field_name, VALID[operation_id][field_name])
    expected_host = operation.outbound_host or operation.host
    assert baseline == ops.OutboundTarget(operation.method, "https", expected_host, 443, rendered_path)
    exercised = 0
    for name, value in _mutations(operation):
        parameters = dict(VALID[operation_id])
        parameters[name] = value
        try:
            target = ops.outbound_target(operation_id, parameters)
        except ops.OperationRequestError as exc:
            assert exc.code in ops.ERROR_CODES
            continue
        exercised += 1
        assert target == baseline
        outbound = ops.build_outbound_request(operation_id, parameters)
        assert outbound.url.startswith("https://%s%s" % (expected_host, rendered_path))
        assert outbound.target == baseline
        if operation.method == "POST":
            document = json.loads(outbound.body.decode("utf-8"))
            if operation.body_wrapper == "deepline_execute":
                assert set(document) == {"provider", "operation", "payload"} and document["operation"] == parameters["tool"]
            elif operation.body_wrapper == "deepline_exa_compat":
                assert set(document) == {"provider", "operation", "payload"} and document["operation"] == operation.deepline_tool
            else:
                assert set(document) <= (set(operation.request_fields) - set(operation.path_fields)) | set(operation.fixed_params)
            for fixed_name, fixed_value in operation.fixed_params.items():
                assert document[fixed_name] == fixed_value
        else:
            assert set(outbound.query) <= set(operation.request_fields) | set(operation.fixed_params)
            assert outbound.body == b""
    assert exercised > 0


@pytest.mark.parametrize("operation_id", sorted(ops.OPERATIONS))
def test_outbound_request_carries_no_credential_and_fixed_params_win(operation_id):
    operation = ops.OPERATIONS[operation_id]
    outbound = ops.build_outbound_request(operation_id, VALID[operation_id])
    serialized = outbound.url + outbound.body.decode("utf-8")
    for marker in ("api_key=", "Authorization", "Bearer", "x-api-key"):
        assert marker not in serialized
    assert outbound.credential == operation.credential
    assert operation.credential.location in ("header", "query")
    if operation.method == "GET":
        assert outbound.query == {**{k: ("true" if v is True else "false" if v is False else str(v)) for k, v in ops.validate_operation_request(operation_id, VALID[operation_id]).items()}, **{k: ("true" if v is True else "false" if v is False else str(v)) for k, v in operation.fixed_params.items()}}
        assert "api_key" not in outbound.query


def test_scrapingdog_scrape_query_is_url_encoded_and_credential_free():
    outbound = ops.build_outbound_request("scrapingdog.scrape", {"url": "https://example.com/a?b=1&c=2"})
    assert outbound.url == "https://api.scrapingdog.com/scrape?dynamic=false&url=https%3A%2F%2Fexample.com%2Fa%3Fb%3D1%26c%3D2"


# ---------------------------------------------------------------------------
# Response sanitizer
# ---------------------------------------------------------------------------


def test_sanitizer_strips_every_provider_header_and_recomputes_length():
    status, headers, body = ops.sanitize_response(
        "deepline.execute",
        200,
        {"Set-Cookie": "session=abc", "X-RateLimit-Remaining": "1", "Server": "nginx", "Content-Type": "application/json; charset=utf-8", "Content-Length": "999", "Via": "proxy"},
        b'{"results": []}',
    )
    assert status == 200
    assert headers == {"content-type": "application/json", "content-length": "15"}
    assert body == b'{"results": []}'


@pytest.mark.parametrize("status", sorted(ops.HOST_ACCOUNT_STATUSES | {429, 500, 503, 599}))
@pytest.mark.parametrize("operation_id", sorted(ops.OPERATIONS))
def test_provider_infrastructure_statuses_become_one_generic_response(operation_id, status):
    result = ops.sanitize_response(operation_id, status, {"WWW-Authenticate": "Bearer"}, b'{"error": "invalid api key sk-live-123"}')
    assert result == (502, {"content-type": "application/json", "content-length": str(len(ops.GENERIC_UNAVAILABLE_BODY))}, ops.GENERIC_UNAVAILABLE_BODY)
    assert b"sk-live" not in result[2]


def test_host_payment_required_is_an_infrastructure_response():
    for operation_id in ("deepline.execute", "scrapingdog.google", "openrouter.chat"):
        status, _, body = ops.sanitize_response(operation_id, 402, {}, b'{"error": {"code": 402}}')
        assert status == 502 and body == ops.GENERIC_UNAVAILABLE_BODY


def test_openrouter_body_passes_through_unchanged_when_valid_json():
    payload = b'{"id": "gen-1", "choices": [{"message": {"content": "hi"}}], "usage": {"prompt_tokens": 3, "completion_tokens": 1}}'
    status, headers, body = ops.sanitize_response("openrouter.chat", 200, {"x-openrouter-trace": "t"}, payload)
    assert (status, body) == (200, payload)
    assert set(headers) == {"content-type", "content-length"}


def test_json_sanitizer_fails_closed_on_invalid_or_oversized_bodies():
    with pytest.raises(ops.OperationResponseError) as excinfo:
        ops.sanitize_response("deepline.execute", 200, {}, b"<html>oops</html>")
    assert excinfo.value.code == "invalid_response"
    with pytest.raises(ops.OperationResponseError) as excinfo:
        ops.sanitize_response("deepline.execute", 200, {}, b'"just a string"')
    assert excinfo.value.code == "invalid_response"
    too_big = b'{"pad": "' + b"x" * ops.OPERATIONS["deepline.execute"].max_response_bytes + b'"}'
    with pytest.raises(ops.OperationResponseError) as excinfo:
        ops.sanitize_response("deepline.execute", 200, {}, too_big)
    assert excinfo.value.code == "response_too_large"
    for bad_status in (99, 600, "200", True, None):
        with pytest.raises(ops.OperationResponseError):
            ops.sanitize_response("deepline.execute", bad_status, {}, b"{}")
    with pytest.raises(ops.OperationResponseError):
        ops.sanitize_response("deepline.execute", 200, {}, "not bytes")


def test_text_sanitizer_truncates_on_a_utf8_boundary_and_bounds_content_type():
    limit = ops.OPERATIONS["scrapingdog.scrape"].max_response_bytes
    body = ("é" * (limit // 2 + 10)).encode("utf-8")
    status, headers, out = ops.sanitize_response("scrapingdog.scrape", 200, {"content-type": "text/html; charset=utf-8"}, body)
    assert status == 200
    assert len(out) <= limit
    out.decode("utf-8")
    assert headers == {"content-type": "text/html", "content-length": str(len(out))}
    _, headers, _ = ops.sanitize_response("scrapingdog.scrape", 200, {"content-type": "weird stuff\r\nset-cookie: x"}, b"ok")
    assert headers["content-type"] == "text/plain"


def test_error_codes_are_closed_and_messages_carry_no_values():
    exc = reject("deepline.execute", {"tool": "exa_search", "payload": {"query": "x"}, "url": "https://secret-host.example/token=abc"}, "forbidden_field")
    assert "secret-host" not in str(exc)
    assert str(exc).startswith("forbidden_field")
    with pytest.raises(ValueError):
        ops.OperationRequestError("provider said: sk-live-123")
    for value in ops.ERROR_CODES:
        assert value == value.lower() and " " not in value
