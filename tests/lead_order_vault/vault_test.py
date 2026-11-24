"""The key vault opens a miner's key only for its own provider and host, and signs what it did."""

from __future__ import annotations

import base64
import hashlib
import json

import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, utils

from lead_order_vault import chain, cli
from lead_order_vault import envelope as env
from lead_order_vault.vault import (
    ENCRYPTION_DOMAIN, RECEIPTS_DOMAIN, UpstreamError, UpstreamResponse, Vault, VaultError, canonical,
)

HOTKEY = "5" + "F" * 47
OTHER_HOTKEY = "5" + "G" * 47
TOKEN = "t" * 40
OPENROUTER_KEY = "sk-or-v1-" + "a" * 64
MANAGEMENT_KEY = "sk-or-v1-" + "b" * 64
DEEPLINE_KEY = "dl_" + "c" * 40


def _key(seed: int) -> ec.EllipticCurvePrivateKey:
    return ec.derive_private_key(seed, ec.SECP256K1())


def _compressed(key) -> bytes:
    return key.public_key().public_bytes(serialization.Encoding.X962, serialization.PublicFormat.CompressedPoint)


ENCRYPTION = _key(0x1111)
RECEIPTS = _key(0x2222)
VAULT_KEY = _compressed(ENCRYPTION).hex()


class FakeProvider:
    def __init__(self, responses=None):
        self.requests = []
        self.responses = responses or {}

    def __call__(self, method, host, target, headers, body, timeout, max_bytes):
        self.requests.append({"method": method, "host": host, "target": target, "headers": dict(headers), "body": body})
        result = self.responses.get(target, UpstreamResponse(200, [("Content-Type", "application/json")], b'{"ok":true}'))
        if isinstance(result, Exception):
            raise result
        return result


def _vault(provider=None) -> Vault:
    return Vault(encryption_key=ENCRYPTION, receipts_key=RECEIPTS, identity={"app_id": "ab" * 20},
                 caller_token=TOKEN, upstream=provider or FakeProvider(), clock=lambda: 1_800_000_000)


def _seal(secret, provider="openrouter", purpose="runtime", hotkey=HOTKEY, vault_key=VAULT_KEY):
    return env.seal(secret, env.Header(provider, purpose, hotkey, vault_key))


def _call(url="https://openrouter.ai/api/v1/chat/completions", sealed=None, **extra):
    request = {"method": "POST", "url": url,
               "headers": {"authorization": "Bearer " + (sealed or _seal(OPENROUTER_KEY)), "content-type": "application/json"},
               "body_b64": base64.b64encode(b'{"model":"x"}').decode(), "timeout_seconds": 120}
    request.update(extra)
    return request


def _check_receipt(receipt):
    public = ec.EllipticCurvePublicKey.from_encoded_point(ec.SECP256K1(), bytes.fromhex(receipt["public_key"]))
    public.verify(bytes.fromhex(receipt["signature"]), canonical(receipt["document"]), ec.ECDSA(hashes.SHA256()))
    assert receipt["public_key"] == _compressed(RECEIPTS).hex()
    return receipt["document"]


# -- envelope -------------------------------------------------------------------


def test_envelope_round_trip_and_header_binding():
    sealed = _seal(OPENROUTER_KEY)
    assert OPENROUTER_KEY not in sealed
    secret, header = env.open_envelope(sealed, ENCRYPTION)
    assert secret == OPENROUTER_KEY and header == env.Header("openrouter", "runtime", HOTKEY, VAULT_KEY)


@pytest.mark.parametrize("field,value", [("provider", "deepline"), ("miner_hotkey", OTHER_HOTKEY), ("purpose", "management")])
def test_editing_the_public_header_breaks_the_envelope(field, value):
    _, document = env.read_header(_seal(OPENROUTER_KEY))
    document[field] = value
    raw = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    forged = env.PREFIX + base64.urlsafe_b64encode(raw).decode().rstrip("=")
    with pytest.raises(env.EnvelopeError, match="envelope_tampered"):
        env.open_envelope(forged, ENCRYPTION)


def test_envelope_for_another_vault_is_refused():
    with pytest.raises(env.EnvelopeError, match="sealed_to_another_vault"):
        env.open_envelope(_seal(OPENROUTER_KEY, vault_key=_compressed(_key(0x3333)).hex()), ENCRYPTION)


def test_deepline_cannot_have_a_management_envelope():
    with pytest.raises(env.EnvelopeError, match="purpose_invalid"):
        _seal(DEEPLINE_KEY, provider="deepline", purpose="management")


# -- chain ------------------------------------------------------------------------


def test_chain_matches_the_dstack_reference_vector():
    # From dstack's key-provenance spec: app root key 0x1a2b..., domain "storage-encryption".
    root = _key(0x1A2B3C4D5E6F7A8B9C0D1E2F3A4B5C6D7E8F9A0B1C2D3E4F5A6B7C8D9E0F1A2B)
    link0 = bytes.fromhex(
        "5b6193729ce7976ec67863f21692d4b98c69832698aae8e001a7d33a6f818b6e"
        "46ca950725b6e90e8ca9bcf394abd03ce264bf9b7eec1e91693247f9dd53c26901")
    public_key = bytes.fromhex("03d962450a41748021c8b02787ac36ce642ff0ae25f4c55019eb527e1112cfd764")
    assert chain.keccak256(b"").hex() == "c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470"
    assert chain.app_root_key(algorithm="secp256k1", domain="storage-encryption",
                              public_key=public_key, link0=link0) == _compressed(root)


def _sign(key, digest: bytes) -> bytes:
    """A 65-byte r||s||v signature like dstack's, with v found by recovery."""

    r, s = utils.decode_dss_signature(key.sign(digest, ec.ECDSA(utils.Prehashed(hashes.SHA256()))))
    n = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
    s = min(s, n - s)
    for v in range(4):
        candidate = r.to_bytes(32, "big") + s.to_bytes(32, "big") + bytes([v])
        try:
            if chain.recover(digest, candidate) == _compressed(key):
                return candidate
        except chain.ChainError:
            continue
    raise AssertionError("no recovery id")


KMS_ROOT = _key(0x4444)
APP_ROOT = _key(0x5555)
APP_ID = bytes.fromhex("ab" * 20)


def _issued(key, domain, app_root=APP_ROOT, kms_root=KMS_ROOT, app_id=APP_ID):
    public_key = _compressed(key)
    link0 = _sign(app_root, chain.keccak256(chain.claim_message("secp256k1", domain, public_key)))
    link1 = _sign(kms_root, chain.keccak256(chain.issued_message(app_id, _compressed(app_root))))
    return {"domain": domain, "algorithm": "secp256k1", "public_key": public_key.hex(),
            "signature_chain": [link0.hex(), link1.hex()]}


DEPLOYMENT = {"url": "https://vault.example", "app_id": APP_ID.hex(), "kms_root_public_key": _compressed(KMS_ROOT).hex(),
              "kms_contract": "0x" + "11" * 20, "chain": "base"}


def _identity(**changes):
    identity = {"app_id": APP_ID.hex(), "encryption": _issued(ENCRYPTION, ENCRYPTION_DOMAIN),
                "receipts": _issued(RECEIPTS, RECEIPTS_DOMAIN)}
    identity.update(changes)
    return identity


def test_cli_accepts_keys_the_kms_issued_to_our_app():
    assert cli.verified_identity(DEPLOYMENT, _identity())["encryption"]["public_key"] == VAULT_KEY


@pytest.mark.parametrize("changes,message", [
    ({"encryption": None}, "not the expected kind"),
    ({"app_id": "cd" * 20}, "different app id"),
])
def test_cli_refuses_a_wrong_identity(changes, message):
    identity = _identity(**changes)
    with pytest.raises(cli.CliError, match=message):
        cli.verified_identity(DEPLOYMENT, identity)


@pytest.mark.parametrize("issued", [
    lambda: _issued(ENCRYPTION, ENCRYPTION_DOMAIN, kms_root=_key(0x6666)),  # someone else's KMS
    lambda: _issued(ENCRYPTION, ENCRYPTION_DOMAIN, app_id=bytes.fromhex("cd" * 20)),  # another app
    lambda: _issued(ENCRYPTION, "leadpoet/other"),  # another key domain, relabelled
    lambda: dict(_issued(ENCRYPTION, ENCRYPTION_DOMAIN), public_key=_compressed(_key(0x7777)).hex()),  # swapped key
])
def test_cli_refuses_a_key_the_kms_did_not_issue_to_our_app(issued):
    key = dict(issued(), domain=ENCRYPTION_DOMAIN)
    with pytest.raises(cli.CliError, match="does not chain"):
        cli.verified_identity(DEPLOYMENT, _identity(encryption=key))


def test_cli_refuses_until_deployed(tmp_path):
    path = tmp_path / "deployment.json"
    path.write_text(json.dumps(dict(DEPLOYMENT, url=None)))
    with pytest.raises(cli.CliError, match="not deployed"):
        cli.load_deployment(path)
    assert json.loads(cli.DEPLOYMENT_FILE.read_text())["url"] is None


def test_cli_seals_both_openrouter_keys_to_the_verified_vault():
    secrets = iter([OPENROUTER_KEY, MANAGEMENT_KEY])
    sealed = cli.seal_key(provider="openrouter", hotkey=HOTKEY, identity=_identity(), read_secret=lambda _: next(secrets))
    assert env.open_envelope(sealed["envelope"], ENCRYPTION)[0] == OPENROUTER_KEY
    assert env.open_envelope(sealed["management_envelope"], ENCRYPTION)[1].purpose == "management"
    assert sealed["key_fingerprint"] == env.fingerprint("openrouter", OPENROUTER_KEY)


# -- call -------------------------------------------------------------------------


def test_call_swaps_the_envelope_for_the_key_and_signs_a_receipt():
    provider = FakeProvider()
    vault = _vault(provider)
    result = vault.call(_call())
    sent = provider.requests[0]
    assert sent["host"] == "openrouter.ai" and sent["target"] == "/api/v1/chat/completions"
    assert sent["headers"]["authorization"] == "Bearer " + OPENROUTER_KEY
    assert result["status"] == 200 and base64.b64decode(result["body_b64"]) == b'{"ok":true}'
    receipt = _check_receipt(result["receipt"])
    assert receipt["key_fingerprint"] == env.fingerprint("openrouter", OPENROUTER_KEY)
    assert receipt["miner_hotkey"] == HOTKEY and "sequence" not in receipt and receipt["key_sequence"] == 1
    assert receipt["request_sha256"] == hashlib.sha256(b'{"model":"x"}').hexdigest()
    assert receipt["response_sha256"] == hashlib.sha256(b'{"ok":true}').hexdigest()
    assert OPENROUTER_KEY not in json.dumps(result)


@pytest.mark.parametrize("url,code", [
    ("https://evil.example/api/v1/chat/completions", "host_not_allowed"),
    ("http://openrouter.ai/api/v1/chat/completions", "host_not_allowed"),
    ("https://openrouter.ai:8443/api/v1/chat/completions", "host_not_allowed"),
    ("https://user@openrouter.ai/api/v1/chat/completions", "host_not_allowed"),
    ("https://openrouter.ai.evil.example/api/v1/chat", "host_not_allowed"),
    ("https://openrouter.ai/api/v1/keys", "path_not_allowed"),
    ("https://openrouter.ai/api/v1/Keys/abc", "path_not_allowed"),
    ("https://openrouter.ai/api/v1/credits/coinbase", "path_not_allowed"),
    ("https://openrouter.ai/api/v1/%6beys", "path_not_allowed"),
    ("https://openrouter.ai/api/v1/x/../keys", "path_not_allowed"),
    ("https://openrouter.ai/api/v1/chat/completions?x=1", "path_not_allowed"),
    ("https://openrouter.ai/api/v1/responses", "path_not_allowed"),
])
def test_call_refuses_other_hosts_and_management_paths(url, code):
    provider = FakeProvider()
    with pytest.raises(VaultError, match=code):
        _vault(provider).call(_call(url=url))
    assert provider.requests == []


def test_a_key_works_only_on_its_own_provider_host():
    provider = FakeProvider()
    with pytest.raises(VaultError, match="provider_mismatch"):
        _vault(provider).call(_call(url="https://code.deepline.com/api/v2/integrations/harvestapi_get_profile/execute"))
    assert provider.requests == []


def test_the_management_key_never_makes_a_call():
    provider = FakeProvider()
    result = _vault(provider).call(_call(sealed=_seal(MANAGEMENT_KEY, purpose="management")))
    assert result["status"] == 401 and result["receipt"] is None and provider.requests == []


def test_an_unusable_envelope_answers_401_without_sending():
    provider = FakeProvider()
    other_vault = _compressed(_key(0x3333)).hex()
    result = _vault(provider).call(_call(sealed=_seal(OPENROUTER_KEY, vault_key=other_vault)))
    assert result["status"] == 401 and provider.requests == []


def test_a_plain_key_is_refused():
    request = _call()
    request["headers"]["authorization"] = "Bearer " + OPENROUTER_KEY
    with pytest.raises(VaultError, match="credential_missing"):
        _vault().call(request)


def test_caller_headers_cannot_smuggle_a_host_or_second_credential():
    provider = FakeProvider()
    request = _call()
    request["headers"].update({"Host": "evil.example", "Cookie": "a=b", "Accept-Encoding": "gzip"})
    _vault(provider).call(request)
    sent = provider.requests[0]["headers"]
    assert "host" not in sent and "cookie" not in sent and "accept-encoding" not in sent
    request["headers"]["Authorization"] = "Bearer x"
    with pytest.raises(VaultError, match="headers_invalid"):
        _vault(provider).call(request)


def test_a_response_that_echoes_the_key_is_withheld():
    echo = UpstreamResponse(200, [], b'{"your_key":"' + OPENROUTER_KEY.encode() + b'"}')
    result = _vault(FakeProvider({"/api/v1/chat/completions": echo})).call(_call())
    assert result["status"] == 502 and OPENROUTER_KEY not in base64.b64decode(result["body_b64"]).decode()
    assert _check_receipt(result["receipt"])["outcome"] == "key_echo_withheld"


def test_a_transport_error_still_gets_a_receipt():
    failing = FakeProvider({"/api/v1/chat/completions": UpstreamError("upstream_unavailable", 200)})
    result = _vault(failing).call(_call())
    assert result["transport_error"] == "upstream_unavailable" and result["observed_status"] == 200
    assert _check_receipt(result["receipt"])["outcome"] == "transport_error"


def test_bad_caller_token_is_refused():
    vault = _vault()
    for token in (None, "", "t" * 39, "u" * 40):
        with pytest.raises(VaultError, match="caller_unauthorized"):
            vault.authorize(token)
    vault.authorize(TOKEN)


# -- check ------------------------------------------------------------------------


def _openrouter_provider(current=None, managed=None, status=200):
    runtime_hash = hashlib.sha256(OPENROUTER_KEY.encode()).hexdigest()
    current = current if current is not None else {"data": {"disabled": False, "is_management_key": False, "limit_remaining": 5}}
    managed = managed if managed is not None else {"data": {"hash": runtime_hash, "disabled": False}}
    return FakeProvider({
        "/api/v1/key": UpstreamResponse(status, [], json.dumps(current).encode()),
        "/api/v1/keys/" + runtime_hash: UpstreamResponse(200, [], json.dumps(managed).encode()),
    })


def _check_request(**changes):
    request = {"provider": "openrouter", "envelope": _seal(OPENROUTER_KEY),
               "management_envelope": _seal(MANAGEMENT_KEY, purpose="management")}
    request.update(changes)
    return request


def test_check_validates_openrouter_ownership_with_the_management_key():
    provider = _openrouter_provider()
    result = _vault(provider).check(_check_request())
    assert result == {"status": "valid", "provider": "openrouter", "miner_hotkey": HOTKEY,
                      "key_fingerprint": env.fingerprint("openrouter", OPENROUTER_KEY)}
    assert [r["headers"]["authorization"] for r in provider.requests] == ["Bearer " + OPENROUTER_KEY, "Bearer " + MANAGEMENT_KEY]


@pytest.mark.parametrize("provider,reason", [
    (lambda: _openrouter_provider(status=401), "openrouter_api_key_invalid"),
    (lambda: _openrouter_provider(current={"data": {"is_management_key": True}}), "openrouter_api_key_invalid"),
    (lambda: _openrouter_provider(current={"data": {"is_management_key": False, "limit_remaining": 0}}), "openrouter_api_key_no_credit"),
    (lambda: _openrouter_provider(managed={"data": {"hash": "0" * 64}}), "openrouter_management_key_invalid"),
])
def test_check_reports_invalid_keys(provider, reason):
    result = _vault(provider()).check(_check_request())
    assert result["status"] == "invalid" and result["reason"] == reason


def test_check_refuses_envelopes_from_two_miners():
    with pytest.raises(VaultError, match="envelope_mismatch"):
        _vault(_openrouter_provider()).check(
            _check_request(management_envelope=_seal(MANAGEMENT_KEY, purpose="management", hotkey=OTHER_HOTKEY)))


def test_check_deepline_reads_the_balance():
    provider = FakeProvider()
    result = _vault(provider).check({"provider": "deepline", "envelope": _seal(DEEPLINE_KEY, provider="deepline")})
    assert result["status"] == "valid" and provider.requests[0]["target"] == "/api/v2/billing/balance"


def test_identity_reveals_no_key():
    vault = _vault()
    vault.call(_call())
    text = json.dumps(vault.identity())
    assert OPENROUTER_KEY not in text and vault.identity()["boot_id"] == vault.boot_id


def test_http_server_routes_authenticates_and_never_logs(capsys):
    import threading
    import urllib.error
    import urllib.request
    from http.server import ThreadingHTTPServer

    from lead_order_vault.vault import CALLER_HEADER, make_handler

    provider = FakeProvider()
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(_vault(provider)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = "http://127.0.0.1:%d" % server.server_address[1]

    def post(path, document, token=TOKEN):
        request = urllib.request.Request(base + path, data=json.dumps(document).encode(), method="POST",
                                         headers={CALLER_HEADER: token} if token else {})
        try:
            with urllib.request.urlopen(request) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read())

    try:
        with urllib.request.urlopen(base + "/v1/identity") as response:
            assert json.loads(response.read())["app_id"] == "ab" * 20
        assert post("/v1/call", _call(), token=None) == (401, {"error": "caller_unauthorized"})
        assert post("/v1/call", _call(url="https://evil.example/x")) == (400, {"error": "host_not_allowed"})
        status, result = post("/v1/call", _call())
        assert status == 200 and result["status"] == 200 and len(provider.requests) == 1
        assert post("/v1/nope", {})[0] == 404
    finally:
        server.shutdown()
        server.server_close()
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""


@pytest.mark.parametrize("method,url", [
    ("POST", "https://code.deepline.com/api/v2/integrations/Bad-Tool/execute"),
    ("POST", "https://code.deepline.com/api/v2/billing/top-up"),
    ("GET", "https://code.deepline.com/api/v2/billing/balance"),
    ("POST", "https://code.deepline.com/api/v2/plays/run"),
    ("POST", "https://code.deepline.com/api/v2/integrations/harvestapi_get_profile/execute/x"),
    ("GET", "https://code.deepline.com/api/v2/integrations/harvestapi_get_profile/execute"),
])
def test_deepline_keys_reach_only_tool_execute(method, url):
    provider = FakeProvider()
    request = _call(url=url, sealed=_seal(DEEPLINE_KEY, provider="deepline"), method=method)
    if method == "GET":
        request.pop("body_b64")
    with pytest.raises(VaultError, match="path_not_allowed"):
        _vault(provider).call(request)
    assert provider.requests == []


def test_generation_read_back_is_allowed_with_only_an_id():
    provider = FakeProvider()
    request = _call(url="https://openrouter.ai/api/v1/generation?id=gen-abc_123", method="GET")
    request.pop("body_b64")
    receipt = _check_receipt(_vault(provider).call(request)["receipt"])
    assert provider.requests[0]["target"] == "/api/v1/generation?id=gen-abc_123"
    assert receipt["target_sha256"] == hashlib.sha256(b"/api/v1/generation?id=gen-abc_123").hexdigest()
    for query in ("id=a&id=b", "id=a/b", ""):
        request["url"] = "https://openrouter.ai/api/v1/generation" + ("?" + query if query else "")
        with pytest.raises(VaultError, match="path_not_allowed"):
            _vault(provider).call(request)


@pytest.mark.parametrize("headers", [{"bad name": "x"}, {"x:y": "z"}, {"x-a": "caf\u00e9"}])
def test_bad_headers_are_refused_before_the_call_is_counted(headers):
    vault = _vault()
    request = _call()
    request["headers"].update(headers)
    with pytest.raises(VaultError, match="headers_invalid"):
        vault.call(request)
    assert _check_receipt(vault.call(_call())["receipt"])["key_sequence"] == 1


def test_an_envelope_has_one_spelling():
    sealed = _seal(OPENROUTER_KEY)
    for variant in (sealed + "=", sealed[:10] + "!" + sealed[10:], sealed + "+"):
        with pytest.raises(env.EnvelopeError):
            env.open_envelope(variant, ENCRYPTION)


def test_any_deepline_tool_works_without_a_vault_change():
    provider = FakeProvider()
    sealed = _seal(DEEPLINE_KEY, provider="deepline")
    receipts = [_check_receipt(_vault(provider).call(_call(url="https://code.deepline.com/api/v2/integrations/%s/execute" % tool,
                                                             sealed=sealed))["receipt"])
                for tool in ("harvestapi_get_profile", "a_tool_added_next_year")]
    assert [r["path"] for r in receipts] == ["/api/v2/integrations/harvestapi_get_profile/execute",
                                             "/api/v2/integrations/a_tool_added_next_year/execute"]


def test_every_request_the_broker_sends_fits_the_vault():
    from lab_arena.lead_orders import broker
    from lead_order_vault.vault import _route_allowed

    assert _route_allowed("openrouter", "POST", "/" + broker.OPENROUTER_CHAT_PATH, "")
    assert _route_allowed("openrouter", "POST", "/" + broker.OPENROUTER_DECISIONS_PATH, "")
    assert _route_allowed("openrouter", "GET", "/api/v1/generation", "id=gen-1")
    from urllib.parse import quote
    assert _route_allowed("openrouter", "GET", "/api/v1/generation", "id=" + quote("gen-1.a:b", safe=""))
    for tool in broker.DEEPLINE_OPERATIONS:
        assert _route_allowed("deepline", "POST", "/api/v2/integrations/%s/execute" % tool, "")


# -- the gateway's client, against the real HTTP server ---------------------------


@pytest.fixture
def served_vault():
    """The real vault server, ending TLS itself, with a KMS-issued identity."""

    import threading

    from lead_order_vault.vault import BoundedServer, make_handler, tls_server_context

    provider = FakeProvider({"/api/v2/integrations/boom/execute": UpstreamError("reset", observed_status=None)})
    context, spki = tls_server_context()
    vault = Vault(encryption_key=ENCRYPTION, receipts_key=RECEIPTS, identity=_identity(), caller_token=TOKEN,
                  upstream=provider, tls_spki_sha256=spki)
    server = BoundedServer(("127.0.0.1", 0), make_handler(vault))
    server.ssl_context = context
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield dict(DEPLOYMENT, url="https://127.0.0.1:%d" % server.server_address[1]), provider
    finally:
        server.shutdown()
        server.server_close()


def _send(transport, url="https://openrouter.ai/api/v1/chat/completions", sealed=None):
    return transport.send(method="POST", url=url, body=b'{"model":"x"}', timeout_seconds=30,
                          headers={"authorization": "Bearer " + (sealed or _seal(OPENROUTER_KEY)),
                                   "content-type": "application/json"})


def test_vault_transport_returns_the_answer_and_its_receipt(served_vault):
    from lab_arena.lead_orders.vault_client import VaultClient, VaultResponse, VaultTransport

    url, provider = served_vault
    response = _send(VaultTransport(VaultClient(url, TOKEN)))
    assert isinstance(response, VaultResponse) and response.status == 200 and response.body == b'{"ok":true}'
    assert _check_receipt(response.vault_receipt)["key_sequence"] == 1
    assert provider.requests[0]["headers"]["authorization"] == "Bearer " + OPENROUTER_KEY


def test_vault_transport_says_whether_the_provider_was_reached(served_vault):
    from lab_arena.lead_orders.vault_client import VaultClient, VaultTransport, VaultTransportError

    url, provider = served_vault
    with pytest.raises(VaultTransportError) as refused:
        _send(VaultTransport(VaultClient(url, TOKEN)), url="https://openrouter.ai/api/v1/keys")
    assert refused.value.forwarded is False and refused.value.vault_receipt is None and not provider.requests
    with pytest.raises(VaultTransportError) as unauthorized:
        _send(VaultTransport(VaultClient(url, "x" * 40)))
    assert unauthorized.value.forwarded is False
    with pytest.raises(VaultTransportError) as lost:
        _send(VaultTransport(VaultClient(url, TOKEN)), url="https://code.deepline.com/api/v2/integrations/boom/execute",
              sealed=_seal(DEEPLINE_KEY, "deepline"))
    assert lost.value.forwarded is True and _check_receipt(lost.value.vault_receipt)["outcome"] == "transport_error"
    with pytest.raises(VaultTransportError) as down:
        _send(VaultTransport(VaultClient(dict(url, url="https://127.0.0.1:9"), TOKEN)))
    assert down.value.forwarded is False


def test_vault_client_checks_a_sealed_key(served_vault):
    from lab_arena.lead_orders.vault_client import VaultClient

    url, _ = served_vault
    checked = VaultClient(url, TOKEN).check(provider="deepline", envelope=_seal(DEEPLINE_KEY, "deepline"),
                                            management_envelope=None)
    assert checked["status"] == "valid" and checked["key_fingerprint"] == env.fingerprint("deepline", DEEPLINE_KEY)


def test_vault_client_refuses_plain_http():
    from lab_arena.lead_orders.vault_client import VaultClient

    for url in ("http://vault.example", "http://127.0.0.1:8080"):
        with pytest.raises(ValueError):
            VaultClient(dict(DEPLOYMENT, url=url), TOKEN)


def test_vault_client_sends_nothing_through_a_gateway_that_ends_tls():
    from lab_arena.lead_orders.vault_client import VaultClient, VaultTransport, VaultTransportError

    vault, _ = _tls_vault()
    gateway, _, deployment = _tls_server(vault.identity)
    try:
        with pytest.raises(VaultTransportError) as refused:
            _send(VaultTransport(VaultClient(deployment, TOKEN)))
        assert refused.value.forwarded is False and str(refused.value) == "vault_unverified"
    finally:
        gateway.shutdown()
        gateway.server_close()


def test_re_verifying_leaves_requests_in_flight_alone(served_vault, monkeypatch):
    from lab_arena.lead_orders import vault_client
    from lab_arena.lead_orders.vault_client import VaultClient

    deployment, _ = served_vault
    verified = []
    real_connect = vault_client.vault_cli.connect
    monkeypatch.setattr(vault_client.vault_cli, "connect", lambda d: verified.append(1) or real_connect(d))
    client = VaultClient(deployment, TOKEN)
    first = client._pinned()
    # Two threads saw the same connect failure: one re-verifies, the other reuses its result.
    second = client._pinned(failed=first)
    assert client._pinned(failed=first) is second and len(verified) == 2
    # The replaced client stays open for requests still running on it.
    assert not first.is_closed and first.get(deployment["url"] + "/healthz").json() == {"ok": True}


def test_vault_client_follows_a_restarted_vault_to_its_new_tls_key():
    import threading

    from lab_arena.lead_orders.vault_client import VaultClient, VaultTransport
    from lead_order_vault.vault import BoundedServer, make_handler

    def serve(port=0):
        vault, (context, _) = _tls_vault()

        # One request per connection: a stopped test server's threads would
        # otherwise keep serving a kept-alive connection, which a real
        # restart (a new process) cannot.
        class Handler(make_handler(vault)):
            protocol_version = "HTTP/1.0"

        server = BoundedServer(("127.0.0.1", port), Handler)
        server.ssl_context = context
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return server, vault

    first, _ = serve()
    port = first.server_address[1]
    client = VaultClient(dict(DEPLOYMENT, url="https://127.0.0.1:%d" % port), TOKEN)
    transport = VaultTransport(client)
    try:
        assert _check_receipt(_send(transport).vault_receipt)["key_sequence"] == 1
    finally:
        first.shutdown()
        first.server_close()
    # Same address, new boot: a new TLS key and a new boot id.
    second, restarted = serve(port)
    try:
        assert _check_receipt(_send(transport).vault_receipt)["boot_id"] == restarted.boot_id
    finally:
        second.shutdown()
        second.server_close()


# -- the miner's receipts check ---------------------------------------------------


def _receipts_page(count=3):
    vault = _vault()
    rows = [{"sequence": n + 1, "call_id": "c%d" % n, "kind": "call", "receipt": vault.call(_call())["receipt"]}
            for n in range(count)]
    keys = [{"key_id": "k1", "provider": "openrouter", "key_fingerprint": env.fingerprint("openrouter", OPENROUTER_KEY),
             "sealed_to": VAULT_KEY}]
    return {"receipts": rows, "keys": keys}


def test_receipts_check_passes_a_complete_history():
    report = cli.audit_receipts(_receipts_page(), _identity())
    assert report["ok"] and report["sequences"][0]["highest"] == 3 and report["sequences"][0]["missing"] == []


def test_receipts_check_finds_a_hidden_call_a_forgery_and_a_stale_seal():
    page = _receipts_page(4)
    del page["receipts"][1]  # a call Leadpoet did not report
    forged = json.loads(json.dumps(page["receipts"][0]))
    forged["receipt"]["document"]["key_sequence"] = 9
    page["receipts"].append(forged)
    page["keys"].append({"key_id": "old", "provider": "deepline", "key_fingerprint": "sha256:" + "0" * 64,
                         "sealed_to": None})
    report = cli.audit_receipts(page, _identity())
    assert not report["ok"] and report["sequences"][0]["missing"] == [2]
    assert report["bad_signatures"] == [1] and report["keys_to_seal_again"] == ["old"]


def test_receipts_check_trusts_only_the_verified_receipts_key():
    page = _receipts_page(1)
    impostor = _key(0x9999)
    document = page["receipts"][0]["receipt"]["document"]
    page["receipts"][0]["receipt"] = {"document": document, "public_key": _compressed(impostor).hex(),
                                      "signature": impostor.sign(canonical(document), ec.ECDSA(hashes.SHA256())).hex()}
    assert cli.audit_receipts(page, _identity())["bad_signatures"] == [1]


def test_a_timeout_after_headers_keeps_the_generation_id_for_the_cost_read_back():
    from lab_arena.lead_orders.vault_client import VaultTransport, VaultTransportError

    seen = [("X-Generation-Id", "gen-abc.1"), ("Set-Cookie", "drop")]
    vault = _vault(FakeProvider({"/api/v1/chat/completions": UpstreamError("reset", 200, seen)}))
    result = vault.call(_call())
    assert result["observed_headers"] == [["x-generation-id", "gen-abc.1"]]

    class Direct:
        def call(self, document, timeout):
            return vault.call(document)

    with pytest.raises(VaultTransportError) as lost:
        _send(VaultTransport(Direct()))
    assert lost.value.forwarded is True and lost.value.openrouter_generation_id == "gen-abc.1"


def test_pre_failure_headers_that_echo_the_key_are_dropped():
    vault = _vault(FakeProvider({"/api/v1/chat/completions": UpstreamError("reset", 200, [("x-echo", OPENROUTER_KEY)])}))
    assert vault.call(_call())["observed_headers"] == []


def test_only_the_vaults_own_refusals_prove_nothing_was_sent():
    import httpx

    from lab_arena.lead_orders.vault_client import VaultClient, VaultTransport, VaultTransportError

    def answer(status, document):
        handler = lambda request: httpx.Response(status, json=document)
        return VaultTransport(VaultClient(DEPLOYMENT, TOKEN,
                                          client=httpx.Client(transport=httpx.MockTransport(handler))))

    cases = [(400, {"error": "path_not_allowed"}, False), (408, {"error": "timeout"}, None),
             (504, {"message": "gateway timeout"}, None), (500, {"error": "vault_error"}, None)]
    for status, document, forwarded in cases:
        with pytest.raises(VaultTransportError) as raised:
            _send(answer(status, document))
        assert raised.value.forwarded is forwarded, (status, document)


def test_a_withheld_answer_keeps_the_generation_id():
    echo = UpstreamResponse(200, [("X-Generation-Id", "gen-9"), ("X-Other", "x")], OPENROUTER_KEY.encode())
    result = _vault(FakeProvider({"/api/v1/chat/completions": echo})).call(_call())
    assert result["status"] == 502 and ["x-generation-id", "gen-9"] in result["headers"]
    assert OPENROUTER_KEY not in json.dumps(result)


# -- the vault ends TLS itself ------------------------------------------------------


def _tls_server(identity_source, tls=None):
    """A TLS server like the vault's. ``identity_source`` answers /v1/identity."""

    import threading
    from http.server import BaseHTTPRequestHandler

    from lead_order_vault.vault import BoundedServer, tls_server_context

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            data = json.dumps(identity_source()).encode()
            self.send_response(200)
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = BoundedServer(("127.0.0.1", 0), Handler)
    server.ssl_context, spki = tls or tls_server_context()
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, spki, dict(DEPLOYMENT, url="https://127.0.0.1:%d" % server.server_address[1])


def _tls_vault():
    from lead_order_vault.vault import tls_server_context

    context, spki = tls_server_context()
    vault = Vault(encryption_key=ENCRYPTION, receipts_key=RECEIPTS, identity=_identity(), caller_token=TOKEN,
                  upstream=FakeProvider(), tls_spki_sha256=spki)
    return vault, (context, spki)


def test_the_cli_trusts_only_a_tls_session_the_vault_ends_itself():
    import urllib.error
    import urllib.request

    vault, tls = _tls_vault()
    server, spki, deployment = _tls_server(vault.identity, tls)
    try:
        identity, certificate = cli.connect(deployment)
        assert identity["tls"]["document"]["spki_sha256"] == spki
        context = cli.pinned_context(certificate)
        with urllib.request.urlopen(deployment["url"] + "/v1/identity", context=context) as response:
            assert json.loads(response.read())["boot_id"] == vault.boot_id
        # After a restart the vault has a new TLS key: the old pin fails closed.
        restarted, _, again = _tls_server(vault.identity)
        try:
            with pytest.raises(urllib.error.URLError, match="CERTIFICATE_VERIFY_FAILED"):
                urllib.request.urlopen(again["url"] + "/v1/identity", context=context)
        finally:
            restarted.shutdown()
            restarted.server_close()
    finally:
        server.shutdown()
        server.server_close()


def test_a_gateway_that_ends_tls_itself_is_refused():
    vault, _ = _tls_vault()
    # The gateway relays the vault's real, KMS-signed identity over its own TLS key.
    gateway, _, deployment = _tls_server(vault.identity)
    try:
        with pytest.raises(cli.CliError, match="does not end in the vault"):
            cli.connect(deployment)
    finally:
        gateway.shutdown()
        gateway.server_close()


def test_a_gateway_cannot_rebind_the_tls_key_without_the_receipts_key():
    from lead_order_vault.vault import tls_server_context

    vault, _ = _tls_vault()
    context, spki = tls_server_context()

    def forged():
        identity = json.loads(json.dumps(vault.identity()))
        identity["tls"]["document"]["spki_sha256"] = spki
        return identity

    gateway, _, deployment = _tls_server(forged, (context, spki))
    try:
        with pytest.raises(cli.CliError, match="not signed by its receipts key"):
            cli.connect(deployment)
    finally:
        gateway.shutdown()
        gateway.server_close()
