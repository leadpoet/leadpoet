"""Closed operation table tests (labarena.md sections 7.4 and 18.4)."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlencode

import pytest

from lab_arena import contracts
from lab_arena import operations as ops

REPO_ROOT = Path(__file__).resolve().parents[2]

VALID = {
    "exa.search": {
        "query": "fintech startups in berlin",
        "type": "auto",
        "category": "company",
        "includeDomains": ["Example.com"],
        "excludeDomains": ["bad.example"],
        "startPublishedDate": "2024-01-01",
        "endPublishedDate": "2025-01-01",
    },
    "exa.contents": {"urls": ["https://example.com/a", "https://example.org/b?x=1"]},
    "scrapingdog.scrape": {"url": "https://example.com/jobs?page=2", "dynamic": True},
    "scrapingdog.google": {"query": "acme corp", "country": "gb"},
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
    "http://api.exa.ai/search",
    "https://api.exa.ai@evil.example/",
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
    url = "https://%s%s" % (operation.host, operation.path)
    if operation.method == "POST":
        return operation.method, url, json.dumps(parameters).encode("utf-8"), {"Content-Type": "application/json"}
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
# Table integrity, price list, hashes
# ---------------------------------------------------------------------------


def test_table_is_closed_and_every_cost_is_fixed_by_parameters():
    assert set(ops.OPERATIONS) == {"exa.search", "exa.contents", "scrapingdog.scrape", "scrapingdog.google", "openrouter.chat"}
    for operation_id, operation in ops.OPERATIONS.items():
        assert operation.operation_id == operation_id
        assert operation.provider in ops.PROVIDERS
        assert operation.host in ("api.exa.ai", "api.scrapingdog.com", "openrouter.ai")
        assert operation.method in ops.METHODS
        assert operation.funding_source == ("openrouter" if operation.provider == "openrouter" else "tao")
        # A price-determining parameter is never model-controllable.
        assert not set(operation.fixed_params) & set(operation.request_fields)
        if operation.funding_source == "tao":
            assert operation.cost_rule["kind"] == "fixed_microusd"
            assert operation.cost_rule["microusd"] == ops.PROVIDER_PRICE_LIST[operation_id]
            assert ops.fixed_cost_microusd(operation_id) == ops.PROVIDER_PRICE_LIST[operation_id]
        else:
            assert operation.cost_rule["kind"] == "openrouter_price_table"
            assert operation.cost_rule["max_output_tokens"] == ops.OPENROUTER_MAX_OUTPUT_TOKENS
            assert ops.fixed_cost_microusd(operation_id) is None
    assert ops.OPERATIONS["exa.search"].fixed_params["numResults"] == 10
    assert ops.OPERATIONS["exa.search"].fixed_params["contents"] == {"text": {"maxCharacters": 2000}}
    assert ops.OPERATIONS["exa.contents"].fixed_params == {"text": {"maxCharacters": 4000}}
    assert ops.OPERATIONS["scrapingdog.scrape"].fixed_params == {"premium": False}
    assert ops.OPERATIONS["scrapingdog.google"].fixed_params == {"results": 10}
    chat = ops.OPERATIONS["openrouter.chat"].fixed_params
    assert chat["stream"] is False
    assert chat["provider"] == {"data_collection": "deny", "allow_fallbacks": False}
    assert set(ops.PROVIDER_PRICE_LIST) == {k for k, v in ops.OPERATIONS.items() if v.funding_source == "tao"}
    assert ops.PROVIDER_PRICE_LIST["exa.search"] == 5_000
    assert ops.PROVIDER_PRICE_LIST["exa.contents"] == 5_000
    assert ops.PROVIDER_PRICE_LIST["scrapingdog.scrape"] == 2_000
    assert ops.PROVIDER_PRICE_LIST["scrapingdog.google"] == 5_000


def test_operations_are_immutable():
    operation = ops.OPERATIONS["exa.search"]
    with pytest.raises(TypeError):
        operation.fixed_params["numResults"] = 100  # type: ignore[index]
    with pytest.raises(TypeError):
        ops.OPERATIONS["evil"] = operation  # type: ignore[index]
    with pytest.raises(Exception):
        operation.host = "evil.example"  # type: ignore[misc]


def test_table_and_price_hashes_are_deterministic_and_stable_across_interpreters():
    assert ops.OPERATION_TABLE_HASH == contracts.document_hash(ops.operation_table_document())
    assert ops.PROVIDER_PRICE_LIST_HASH == contracts.document_hash(ops.price_list_document())
    assert contracts.SHA256_RE.match(ops.OPERATION_TABLE_HASH)
    script = "from lab_arena import operations as o; print(o.OPERATION_TABLE_HASH, o.PROVIDER_PRICE_LIST_HASH)"
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    assert completed.stdout.split() == [ops.OPERATION_TABLE_HASH, ops.PROVIDER_PRICE_LIST_HASH]


def test_table_hash_changes_when_a_fixed_param_or_price_changes():
    document = ops.operation_table_document()
    assert contracts.document_hash(copy.deepcopy(document)) == ops.OPERATION_TABLE_HASH
    changed = copy.deepcopy(document)
    search = next(item for item in changed["operations"] if item["operation_id"] == "exa.search")
    search["fixed_params"]["numResults"] = 11
    assert contracts.document_hash(changed) != ops.OPERATION_TABLE_HASH
    changed = copy.deepcopy(document)
    chat = next(item for item in changed["operations"] if item["operation_id"] == "openrouter.chat")
    chat["fixed_params"]["provider"]["allow_fallbacks"] = True
    assert contracts.document_hash(changed) != ops.OPERATION_TABLE_HASH
    prices = ops.price_list_document()
    prices["prices"]["exa.search"] += 1
    assert contracts.document_hash(prices) != ops.PROVIDER_PRICE_LIST_HASH
    # The table document carries every constraint, so a schema change is visible.
    changed = copy.deepcopy(document)
    search = next(item for item in changed["operations"] if item["operation_id"] == "exa.search")
    search["request_fields"]["query"]["max_length"] = 5000
    assert contracts.document_hash(changed) != ops.OPERATION_TABLE_HASH


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


def test_normalization_lowercases_domains_and_fills_defaults():
    parameters = ops.validate_operation_request("exa.search", {"query": "x", "includeDomains": ["Example.COM"]})
    assert parameters == {"query": "x", "includeDomains": ["example.com"]}
    chat = ops.validate_operation_request("openrouter.chat", {"model": "openai/gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]})
    assert chat["max_tokens"] == ops.OPENROUTER_MAX_OUTPUT_TOKENS
    assert ops.validate_operation_request("scrapingdog.google", {"query": "acme"})["country"] == "us"
    assert ops.validate_operation_request("scrapingdog.scrape", {"url": "https://example.com/"})["dynamic"] is False


def test_null_optional_fields_are_dropped_and_null_required_fields_are_missing():
    assert ops.validate_operation_request("exa.search", {"query": "x", "category": None}) == {"query": "x"}
    reject("exa.search", {"query": None}, "missing_field")


# ---------------------------------------------------------------------------
# Exact-operation rejections: path, method, query, header, credentials
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "method,url",
    [
        ("POST", "https://api.exa.ai/search/"),
        ("POST", "https://api.exa.ai/search/extra"),
        ("POST", "https://api.exa.ai/Search"),
        ("POST", "https://api.exa.ai/v1/search"),
        ("GET", "https://api.exa.ai/search"),
        ("PUT", "https://api.exa.ai/search"),
        ("POST", "http://api.exa.ai/search"),
        ("POST", "https://api.exa.ai:8443/search"),
        ("POST", "https://api.exa.ai:80/search"),
        ("POST", "https://user@api.exa.ai/search"),
        ("POST", "https://api.exa.ai@evil.example/search"),
        ("POST", "https://api.exa.ai/search#fragment"),
        ("POST", "https://evil.example/search"),
        ("POST", "https://api.exa.ai.evil.example/search"),
        ("POST", "https://sub.api.exa.ai/search"),
        ("POST", "https://api.exa.ai./search"),
        ("POST", "https://127.0.0.1/search"),
        ("POST", "https://[::1]/search"),
        ("POST", "https://2130706433/search"),
        ("POST", "https://api.exa.ai/search\n"),
        ("GET", "https://api.scrapingdog.com/scrape/"),
        ("POST", "https://api.scrapingdog.com/scrape"),
        ("POST", "https://openrouter.ai/api/v1/completions"),
        ("POST", "https://openrouter.ai/api/v1/chat/completions/"),
        ("GET", "https://openrouter.ai/api/v1/models"),
    ],
)
def test_unknown_path_method_host_port_userinfo_and_fragment_match_nothing(method, url):
    with pytest.raises(ops.OperationRequestError) as excinfo:
        ops.match_request(method, url, b'{"query": "x"}', {})
    assert excinfo.value.code == "no_matching_operation"


def test_explicit_port_443_is_the_same_operation():
    assert ops.match_request("POST", "https://api.exa.ai:443/search", b'{"query": "x"}', {})[0] == "exa.search"


def test_query_string_on_post_operation_is_rejected():
    with pytest.raises(ops.OperationRequestError) as excinfo:
        ops.match_request("POST", "https://api.exa.ai/search?numResults=100", b'{"query": "x"}', {})
    assert excinfo.value.code == "invalid_query"


@pytest.mark.parametrize(
    "query,code",
    [
        ("url=https%3A%2F%2Fexample.com%2F&api_key=abc", "forbidden_field"),
        ("url=https%3A%2F%2Fexample.com%2F&premium=true", "unknown_field"),
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
        ("X-Title", "unknown_header"),
        ("HTTP-Referer", "unknown_header"),
        ("X-Forwarded-For", "unknown_header"),
        ("X-Stainless-Lang", "unknown_header"),
    ],
)
def test_caller_credentials_and_unknown_headers_are_refused(header, code):
    with pytest.raises(ops.OperationRequestError) as excinfo:
        ops.match_request("POST", "https://api.exa.ai/search", b'{"query": "x"}', {header: "value"})
    assert excinfo.value.code == code
    with pytest.raises(ops.OperationRequestError):
        ops.match_request("GET", "https://api.scrapingdog.com/google?query=x", b"", {header: "value"})


def test_benign_client_headers_are_accepted_and_never_forwarded():
    headers = {
        "Host": "api.exa.ai",
        "User-Agent": "python-requests/2.32",
        "Accept": "*/*",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
        "Content-Length": "14",
        "Content-Type": "application/json; charset=utf-8",
    }
    operation_id, parameters = ops.match_request("POST", "https://api.exa.ai/search", b'{"query": "x"}', headers)
    assert operation_id == "exa.search"
    outbound = ops.build_outbound_request(operation_id, parameters)
    assert "User-Agent" not in json.dumps(outbound.body.decode("utf-8"))
    assert outbound.content_type == "application/json"


@pytest.mark.parametrize("body", [b"", b"[]", b'"x"', b"not json", b"\xff\xfe", b"{\"query\": \"x\"} trailing"])
def test_post_body_must_be_a_json_object(body):
    with pytest.raises(ops.OperationRequestError) as excinfo:
        ops.match_request("POST", "https://api.exa.ai/search", body, {})
    assert excinfo.value.code in ("invalid_body", "missing_field")


def test_oversized_raw_body_and_nesting_bombs_are_rejected_before_validation():
    padded = json.dumps({"query": "x", "pad": "y" * 20_000}).encode("utf-8")
    with pytest.raises(ops.OperationRequestError) as excinfo:
        ops.match_request("POST", "https://api.exa.ai/search", padded, {})
    assert excinfo.value.code == "request_too_large"
    bomb: dict = {"query": "x"}
    cursor = bomb
    for _ in range(20):
        cursor["n"] = {}
        cursor = cursor["n"]
    reject("exa.search", bomb, "invalid_request")
    reject("exa.search", {"query": "x\x00y"}, "invalid_request")
    reject("exa.search", "not a mapping", "invalid_request")


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
    parameters["numResults"] = 100
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
    assert reject("exa.contents", {"urls": [url]}).code in expected


@pytest.mark.parametrize(
    "url",
    ["https://example.com/", "https://example.com:443/path", "https://sub.example.co.uk/jobs?page=2&sort=asc", "https://example.com"],
)
def test_accepted_target_urls_are_returned_unchanged(url):
    assert ops.validate_operation_request("scrapingdog.scrape", {"url": url})["url"] == url


def test_exa_contents_bounds_urls():
    reject("exa.contents", {"urls": []}, "invalid_field")
    reject("exa.contents", {"urls": ["https://example.com/"] * 6}, "invalid_field")
    reject("exa.contents", {"urls": "https://example.com/"}, "invalid_field")
    reject("exa.contents", {}, "missing_field")


def test_exa_search_field_rules():
    reject("exa.search", {"query": ""}, "invalid_field")
    reject("exa.search", {"query": "x" * 1001}, "invalid_field")
    reject("exa.search", {"query": "x", "type": "magic"}, "invalid_field")
    reject("exa.search", {"query": "x", "category": "everything"}, "invalid_field")
    reject("exa.search", {"query": "x", "includeDomains": ["example.com"] * 21}, "invalid_field")
    reject("exa.search", {"query": "x", "includeDomains": ["https://example.com"]}, "invalid_field")
    reject("exa.search", {"query": "x", "includeDomains": ["10.0.0.1"]}, "invalid_field")
    reject("exa.search", {"query": "x", "startPublishedDate": "2024-13-01"}, "invalid_field")
    reject("exa.search", {"query": "x", "startPublishedDate": "2024-01-01T00:00:00Z"}, "invalid_field")
    reject("exa.search", {"query": 5}, "invalid_field")


def test_scrapingdog_google_field_rules():
    reject("scrapingdog.google", {"query": "x" * 501}, "invalid_field")
    reject("scrapingdog.google", {"query": "x", "country": "zz"}, "invalid_field")
    reject("scrapingdog.google", {"query": "x", "results": 100}, "unknown_field")


@pytest.mark.parametrize(
    "extra,code",
    [
        ({"tools": []}, "forbidden_field"),
        ({"tool_choice": "auto"}, "forbidden_field"),
        ({"functions": []}, "forbidden_field"),
        ({"plugins": [{"id": "web"}]}, "forbidden_field"),
        ({"provider": {"allow_fallbacks": True}}, "forbidden_field"),
        ({"stream": True}, "forbidden_field"),
        ({"stream": False}, "forbidden_field"),
        ({"reasoning": {"effort": "high"}}, "forbidden_field"),
        ({"transforms": ["middle-out"]}, "forbidden_field"),
        ({"route": "fallback"}, "forbidden_field"),
        ({"models": ["openai/gpt-4o"]}, "forbidden_field"),
        ({"web_search_options": {}}, "forbidden_field"),
        ({"modalities": ["image"]}, "forbidden_field"),
        ({"response_format": {"type": "json_object"}}, "invalid_field"),
        ({"response_format": {"type": "json_schema", "json_schema": {}}}, "unknown_field"),
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
        ({"messages": [{"role": "tool", "content": "x"}]}, "invalid_field"),
        ({"messages": [{"role": "user", "content": [{"type": "text", "text": "x"}]}]}, "invalid_field"),
        ({"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://x"}}]}]}, "invalid_field"),
        ({"messages": [{"role": "user", "content": "x", "tool_calls": []}]}, "unknown_field"),
        ({"messages": [{"role": "user", "content": "x", "name": "n"}]}, "unknown_field"),
        ({"messages": [{"role": "user"}]}, "missing_field"),
        ({"messages": [{"role": "user", "content": "x" * 32_001}]}, "invalid_field"),
        ({"messages": [{"role": "user", "content": "x"}] * 65}, "invalid_field"),
    ],
)
def test_openrouter_chat_rejections(extra, code):
    parameters = {"model": "openai/gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]}
    parameters.update(extra)
    reject("openrouter.chat", parameters, code)


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
    assert baseline == ops.OutboundTarget(operation.method, "https", operation.host, 443, operation.path)
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
        assert outbound.url.startswith("https://%s%s" % (operation.host, operation.path))
        assert outbound.target == baseline
        if operation.method == "POST":
            document = json.loads(outbound.body.decode("utf-8"))
            assert set(document) <= set(operation.request_fields) | set(operation.fixed_params)
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
    assert outbound.url == "https://api.scrapingdog.com/scrape?dynamic=false&premium=false&url=https%3A%2F%2Fexample.com%2Fa%3Fb%3D1%26c%3D2"


# ---------------------------------------------------------------------------
# Response sanitizer
# ---------------------------------------------------------------------------


def test_sanitizer_strips_every_provider_header_and_recomputes_length():
    status, headers, body = ops.sanitize_response(
        "exa.search",
        200,
        {"Set-Cookie": "session=abc", "X-RateLimit-Remaining": "1", "Server": "nginx", "Content-Type": "application/json; charset=utf-8", "Content-Length": "999", "Via": "proxy"},
        b'{"results": []}',
    )
    assert status == 200
    assert headers == {"content-type": "application/json", "content-length": "15"}
    assert body == b'{"results": []}'


@pytest.mark.parametrize("status", sorted(ops.CREDENTIAL_STATUSES))
@pytest.mark.parametrize("operation_id", sorted(ops.OPERATIONS))
def test_credential_statuses_become_one_generic_response(operation_id, status):
    result = ops.sanitize_response(operation_id, status, {"WWW-Authenticate": "Bearer"}, b'{"error": "invalid api key sk-live-123"}')
    assert result == (502, {"content-type": "application/json", "content-length": str(len(ops.GENERIC_UNAVAILABLE_BODY))}, ops.GENERIC_UNAVAILABLE_BODY)
    assert b"sk-live" not in result[2]


def test_payment_required_is_generic_for_arena_credentials_but_visible_for_miner_keys():
    assert ops.sanitize_response("exa.search", 402, {}, b'{"error": "account out of credit"}')[0] == 502
    status, _, body = ops.sanitize_response("openrouter.chat", 402, {}, b'{"error": {"code": 402}}')
    assert status == 402 and body == b'{"error": {"code": 402}}'


def test_openrouter_body_passes_through_unchanged_when_valid_json():
    payload = b'{"id": "gen-1", "choices": [{"message": {"content": "hi"}}], "usage": {"prompt_tokens": 3, "completion_tokens": 1}}'
    status, headers, body = ops.sanitize_response("openrouter.chat", 200, {"x-openrouter-trace": "t"}, payload)
    assert (status, body) == (200, payload)
    assert set(headers) == {"content-type", "content-length"}


def test_json_sanitizer_fails_closed_on_invalid_or_oversized_bodies():
    with pytest.raises(ops.OperationResponseError) as excinfo:
        ops.sanitize_response("exa.search", 200, {}, b"<html>oops</html>")
    assert excinfo.value.code == "invalid_response"
    with pytest.raises(ops.OperationResponseError) as excinfo:
        ops.sanitize_response("exa.search", 200, {}, b'"just a string"')
    assert excinfo.value.code == "invalid_response"
    too_big = b'{"pad": "' + b"x" * ops.OPERATIONS["exa.search"].max_response_bytes + b'"}'
    with pytest.raises(ops.OperationResponseError) as excinfo:
        ops.sanitize_response("exa.search", 200, {}, too_big)
    assert excinfo.value.code == "response_too_large"
    for bad_status in (99, 600, "200", True, None):
        with pytest.raises(ops.OperationResponseError):
            ops.sanitize_response("exa.search", bad_status, {}, b"{}")
    with pytest.raises(ops.OperationResponseError):
        ops.sanitize_response("exa.search", 200, {}, "not bytes")


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
    exc = reject("exa.search", {"query": "x", "url": "https://secret-host.example/token=abc"}, "forbidden_field")
    assert "secret-host" not in str(exc)
    assert str(exc).startswith("forbidden_field")
    with pytest.raises(ValueError):
        ops.OperationRequestError("provider said: sk-live-123")
    for value in ops.ERROR_CODES:
        assert value == value.lower() and " " not in value
