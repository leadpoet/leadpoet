"""The production model catalog must stay inside the organizer account scope."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError, URLError

import pytest

from lab_arena import broker
from lab_arena.contracts import ArenaContractError


ALLOWED = "openai/gpt-4o-mini"
PUBLIC_ONLY = "example/public-only"


def _document(*rows):
    return {"data": [{"id": model, "pricing": {"prompt": prompt, "completion": completion}}
                     for model, prompt, completion in rows]}


class _Body:
    def __init__(self, document):
        self.data = json.dumps(document).encode("utf-8")

    def read(self, limit):
        return self.data[:limit]

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


class _Opener:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def __call__(self, request, timeout):
        self.calls.append((request.full_url, request.headers.get("Authorization")))
        response = self.responses[request.full_url]
        if isinstance(response, Exception):
            raise response
        return _Body(response)


def test_account_ids_filter_public_catalog_without_changing_shared_prices():
    opener = _Opener({
        broker.OPENROUTER_USER_MODELS_URL: _document((ALLOWED, "0.0000009", "0.000002")),
        broker.OPENROUTER_MODELS_URL: _document(
            (ALLOWED, "0.00000015", "0.0000006"),
            (PUBLIC_ONLY, "0.0000001", "0.0000002"),
        ),
    })

    table = broker.fetch_openrouter_price_table(urlopen=opener, api_key="test-key")

    assert list(table["models"]) == [ALLOWED]
    assert table["models"][ALLOWED]["prompt"] == "0.00000015"
    assert table["models"][ALLOWED]["completion"] == "0.0000006"
    assert table["source"] == broker.OPENROUTER_MODELS_URL
    assert opener.calls == [
        (broker.OPENROUTER_USER_MODELS_URL, "Bearer test-key"),
        (broker.OPENROUTER_MODELS_URL, None),
    ]


@pytest.mark.parametrize("account_reply", [
    HTTPError("https://openrouter.ai/api/v1/models/user", 401, "denied", {}, None),
    HTTPError("https://openrouter.ai/api/v1/models/user", 503, "unavailable", {}, None),
    URLError("unreachable"),
    TimeoutError("timeout"),
    {"data": []},
    {"data": [{"id": ALLOWED}, {}]},
    {"data": [{"id": ALLOWED}, {"id": 7}]},
    {"data": [{"id": ALLOWED}, {"id": "  "}]},
    {"wrong": "shape"},
])
def test_account_refusal_fails_closed_without_public_fallback(account_reply):
    opener = _Opener({
        broker.OPENROUTER_USER_MODELS_URL: account_reply,
        broker.OPENROUTER_MODELS_URL: _document((ALLOWED, "0.00000015", "0.0000006")),
    })

    with pytest.raises(ArenaContractError):
        broker.fetch_openrouter_price_table(urlopen=opener, api_key="test-key")
    assert opener.calls == [(broker.OPENROUTER_USER_MODELS_URL, "Bearer test-key")]


def test_public_refusal_keeps_existing_fail_closed_startup_contract():
    opener = _Opener({
        broker.OPENROUTER_USER_MODELS_URL: _document((ALLOWED, "0.0000009", "0.000002")),
        broker.OPENROUTER_MODELS_URL: URLError("unreachable"),
    })

    with pytest.raises(ArenaContractError):
        broker.fetch_openrouter_price_table(urlopen=opener, api_key="test-key")
    assert [url for url, _auth in opener.calls] == [
        broker.OPENROUTER_USER_MODELS_URL, broker.OPENROUTER_MODELS_URL,
    ]


def test_explicit_model_list_cannot_restore_public_only_model():
    opener = _Opener({
        broker.OPENROUTER_USER_MODELS_URL: _document((ALLOWED, "0.0000009", "0.000002")),
        broker.OPENROUTER_MODELS_URL: _document(
            (ALLOWED, "0.00000015", "0.0000006"),
            (PUBLIC_ONLY, "0.0000001", "0.0000002"),
        ),
    })

    with pytest.raises(ArenaContractError):
        broker.fetch_openrouter_price_table(
            [ALLOWED, PUBLIC_ONLY], urlopen=opener, api_key="test-key"
        )


def test_account_ids_do_not_depend_on_account_pricing_or_supported_syntax():
    opener = _Opener({
        broker.OPENROUTER_USER_MODELS_URL: {"data": [
            {"id": ALLOWED}, {"id": ALLOWED}, {"id": "example/unsupported@model"},
        ]},
        broker.OPENROUTER_MODELS_URL: _document(
            (ALLOWED, "0.00000015", "0.0000006"),
            (PUBLIC_ONLY, "0.0000001", "0.0000002"),
        ),
    })

    table = broker.fetch_openrouter_price_table(urlopen=opener, api_key="test-key")

    assert list(table["models"]) == [ALLOWED]
    assert table["models"][ALLOWED]["prompt"] == "0.00000015"


def test_no_key_keeps_explicit_public_catalog_helper_behavior():
    opener = _Opener({
        broker.OPENROUTER_MODELS_URL: _document((ALLOWED, "0.00000015", "0.0000006")),
    })

    table = broker.fetch_openrouter_price_table(urlopen=opener)

    assert list(table["models"]) == [ALLOWED]
    assert opener.calls == [(broker.OPENROUTER_MODELS_URL, None)]


def test_empty_supplied_account_key_fails_before_any_catalog_request():
    opener = _Opener({})

    with pytest.raises(ArenaContractError):
        broker.fetch_openrouter_price_table(urlopen=opener, api_key="")
    assert opener.calls == []


def test_authenticated_catalog_redirect_does_not_forward_key(monkeypatch):
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            seen.append(self.path)
            if self.path in ("/catalog", "/public"):
                self.send_response(302)
                self.send_header("Location", "/redirect-target")
                self.end_headers()
            else:
                body = json.dumps(_document((ALLOWED, "0.00000015", "0.0000006"))).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        monkeypatch.setattr(broker, "OPENROUTER_USER_MODELS_URL", base + "/catalog")
        monkeypatch.setattr(broker, "OPENROUTER_MODELS_URL", base + "/public")
        with pytest.raises(ArenaContractError):
            broker.fetch_openrouter_price_table(api_key="test-key", timeout_seconds=2)
        assert seen == ["/catalog"]
        public_table = broker.fetch_openrouter_price_table(timeout_seconds=2)
        assert list(public_table["models"]) == [ALLOWED]
        assert seen == ["/catalog", "/public", "/redirect-target"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
