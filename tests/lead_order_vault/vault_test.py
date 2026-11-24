"""The key vault opens a miner's key only for its own provider and host, and signs what it did."""

from __future__ import annotations

import base64
import contextlib
import hashlib
import http.client
import json
import socket
import ssl
import threading
import time
import urllib.error
import urllib.request

import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, utils

from lead_order_vault import chain, cli, trust
from lead_order_vault import envelope as env
from lead_order_vault.vault import (
    CALLER_HEADER, ENCRYPTION_DOMAIN, IDENTITY_SCHEMA, RECEIPTS_DOMAIN, BoundedServer, UpstreamError, UpstreamResponse, Vault,
    VaultError, canonical, make_handler, tls_server_context,
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
                 caller_token=TOKEN, tls_spki_sha256="0" * 64, upstream=provider or FakeProvider(),
                 clock=lambda: 1_800_000_000)


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
    identity = {"schema": IDENTITY_SCHEMA, "app_id": APP_ID.hex(), "encryption": _issued(ENCRYPTION, ENCRYPTION_DOMAIN),
                "receipts": _issued(RECEIPTS, RECEIPTS_DOMAIN)}
    identity.update(changes)
    return identity


def test_trust_accepts_keys_the_kms_issued_to_our_app():
    assert trust.verified_identity(DEPLOYMENT, _identity())["encryption"]["public_key"] == VAULT_KEY


@pytest.mark.parametrize("changes,message", [
    ({"encryption": None}, "not the expected kind"),
    ({"app_id": "cd" * 20}, "different app id"),
    ({"schema": "something.else"}, "not an identity document"),
])
def test_trust_refuses_a_wrong_identity(changes, message):
    identity = _identity(**changes)
    with pytest.raises(trust.TrustError, match=message):
        trust.verified_identity(DEPLOYMENT, identity)


@pytest.mark.parametrize("issued", [
    lambda: _issued(ENCRYPTION, ENCRYPTION_DOMAIN, kms_root=_key(0x6666)),  # someone else's KMS
    lambda: _issued(ENCRYPTION, ENCRYPTION_DOMAIN, app_id=bytes.fromhex("cd" * 20)),  # another app
    lambda: _issued(ENCRYPTION, "leadpoet/other"),  # another key domain, relabelled
    lambda: dict(_issued(ENCRYPTION, ENCRYPTION_DOMAIN), public_key=_compressed(_key(0x7777)).hex()),  # swapped key
])
def test_trust_refuses_a_key_the_kms_did_not_issue_to_our_app(issued):
    key = dict(issued(), domain=ENCRYPTION_DOMAIN)
    with pytest.raises(trust.TrustError, match="does not chain"):
        trust.verified_identity(DEPLOYMENT, _identity(encryption=key))


def test_trust_refuses_until_deployed(tmp_path):
    path = tmp_path / "deployment.json"
    path.write_text(json.dumps(dict(DEPLOYMENT, url=None)))
    with pytest.raises(trust.TrustError, match="not deployed"):
        trust.load_deployment(path)


COMPOSE_HASH = "cc" * 32


def _base(monkeypatch, *, root=None, registered=True, approved=True, chain_id=8453, epoch=None):
    """A fake Base RPC that answers what the KMS and app contracts would."""

    root = root if root is not None else bytes.fromhex(DEPLOYMENT["kms_root_public_key"])
    word = lambda flag: "0x" + (int(flag)).to_bytes(32, "big").hex()
    padded = root + b"\0" * (-len(root) % 32)
    kms_info = ((128).to_bytes(32, "big") + b"\0" * 96 + len(root).to_bytes(32, "big") + padded).hex()
    selector = lambda signature: "0x" + chain.keccak256(signature.encode())[:4].hex()
    answers = {(DEPLOYMENT["kms_contract"], selector("kmsInfo()")): "0x" + kms_info,
               (DEPLOYMENT["kms_contract"], selector("registeredApps(address)") + "00" * 12 + APP_ID.hex()): word(registered),
               ("0x" + APP_ID.hex(), selector("allowedComposeHashes(bytes32)") + COMPOSE_HASH): word(approved),
               ("0x" + APP_ID.hex(), selector("owner()")): "0x" + "00" * 12 + "99" * 20}
    if epoch is not None:  # a KMS from before app epochs has no currentAppEpoch
        answers[(DEPLOYMENT["kms_contract"], selector("currentAppEpoch(address)") + "00" * 12 + APP_ID.hex())] = \
            "0x" + epoch.to_bytes(32, "big").hex()

    def rpc(url, method, params):
        if method == "eth_chainId":
            return hex(chain_id)
        answer = answers.get((params[0]["to"], params[0]["data"]))
        if answer is None:
            raise chain.ChainError("rpc_error")  # what _rpc raises when a call reverts
        return answer

    monkeypatch.setattr(chain, "_rpc", rpc)


@pytest.mark.parametrize("epoch", [None, 0])
def test_trust_reads_base_itself(monkeypatch, epoch):
    _base(monkeypatch, epoch=epoch)
    assert trust.on_chain(DEPLOYMENT, _identity(compose_hash=COMPOSE_HASH), "rpc") == "0x" + "99" * 20


@pytest.mark.parametrize("changes,reason", [
    ({"root": _compressed(_key(0x6666))}, "kms_root_mismatch"),  # deployment.json names another KMS root
    ({"registered": False}, "app_not_registered"),
    ({"approved": False}, "compose_hash_not_approved"),  # a build the owner never approved on Base
    ({"chain_id": 1}, "wrong_chain"),
    ({"epoch": 1}, "app_epoch_unsupported"),  # approvals no longer live in the epoch-0 list
])
def test_trust_refuses_what_base_does_not_confirm(monkeypatch, changes, reason):
    _base(monkeypatch, **changes)
    with pytest.raises(trust.TrustError, match=reason):
        trust.on_chain(DEPLOYMENT, _identity(compose_hash=COMPOSE_HASH), "rpc")


def test_cli_verify_reports_what_base_confirmed(monkeypatch, capsys):
    _base(monkeypatch)
    identity = _identity(compose_hash=COMPOSE_HASH, tls={"document": {"spki_sha256": "ab" * 32}})
    monkeypatch.setattr(trust, "load_deployment", lambda: DEPLOYMENT)
    monkeypatch.setattr(trust, "connect", lambda deployment: (identity, None))
    assert cli.main(["verify"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["kms_contract"] == DEPLOYMENT["kms_contract"] and report["app_owner"] == "0x" + "99" * 20
    _base(monkeypatch, approved=False)
    assert cli.main(["verify"]) == 1


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


DEEPLINE_TOOL = "https://code.deepline.com/api/v2/integrations/harvestapi_get_profile/execute"


@pytest.mark.parametrize("method,url,provider,code", [
    ("POST", "https://evil.example/api/v1/chat/completions", "openrouter", "host_not_allowed"),
    ("POST", "http://openrouter.ai/api/v1/chat/completions", "openrouter", "host_not_allowed"),
    ("POST", "https://openrouter.ai:8443/api/v1/chat/completions", "openrouter", "host_not_allowed"),
    ("POST", "https://user@openrouter.ai/api/v1/chat/completions", "openrouter", "host_not_allowed"),
    ("POST", "https://openrouter.ai.evil.example/api/v1/chat", "openrouter", "host_not_allowed"),
    ("POST", "https://openrouter.ai/api/v1/keys", "openrouter", "path_not_allowed"),
    ("POST", "https://openrouter.ai/api/v1/Keys/abc", "openrouter", "path_not_allowed"),
    ("POST", "https://openrouter.ai/api/v1/credits/coinbase", "openrouter", "path_not_allowed"),
    ("POST", "https://openrouter.ai/api/v1/%6beys", "openrouter", "path_not_allowed"),
    ("POST", "https://openrouter.ai/api/v1/x/../keys", "openrouter", "path_not_allowed"),
    ("POST", "https://openrouter.ai/api/v1/chat/completions?x=1", "openrouter", "path_not_allowed"),
    ("POST", "https://openrouter.ai/api/v1/responses", "openrouter", "path_not_allowed"),
    ("POST", DEEPLINE_TOOL, "openrouter", "provider_mismatch"),  # a key works only on its own host
    ("POST", "https://code.deepline.com/api/v2/integrations/Bad-Tool/execute", "deepline", "path_not_allowed"),
    ("POST", "https://code.deepline.com/api/v2/billing/top-up", "deepline", "path_not_allowed"),
    ("GET", "https://code.deepline.com/api/v2/billing/balance", "deepline", "path_not_allowed"),
    ("POST", "https://code.deepline.com/api/v2/plays/run", "deepline", "path_not_allowed"),
    ("POST", DEEPLINE_TOOL + "/x", "deepline", "path_not_allowed"),
    ("GET", DEEPLINE_TOOL, "deepline", "path_not_allowed"),
])
def test_a_key_reaches_only_its_own_allowed_routes(method, url, provider, code):
    upstream = FakeProvider()
    key = OPENROUTER_KEY if provider == "openrouter" else DEEPLINE_KEY
    request = _call(url=url, sealed=_seal(key, provider=provider), method=method)
    if method == "GET":
        request.pop("body_b64")
    with pytest.raises(VaultError, match=code):
        _vault(upstream).call(request)
    assert upstream.requests == []


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


@pytest.mark.parametrize("headers,body", [
    ([], b'{"your_key":"' + OPENROUTER_KEY.encode() + b'"}'),
    ([], b'{"k":"' + "".join("\\u%04x" % ord(c) for c in OPENROUTER_KEY).encode() + b'"}'),
    ([], b"key=" + "".join("%%%02X" % ord(c) for c in OPENROUTER_KEY).encode()),
    ([("X-Echo", "%73" + OPENROUTER_KEY[1:])], b"{}"),
    # A streamed answer: one JSON document per data: line.
    ([], b'data: {"k":"' + "".join("\\u%04x" % ord(c) for c in OPENROUTER_KEY).encode() + b'"}\n\ndata: [DONE]\n'),
])
def test_a_response_that_echoes_the_key_is_withheld(headers, body):
    echo = UpstreamResponse(200, headers, body)
    result = _vault(FakeProvider({"/api/v1/chat/completions": echo})).call(_call())
    assert result["status"] == 502 and OPENROUTER_KEY not in json.dumps(result)
    assert _check_receipt(result["receipt"])["outcome"] == "key_echo_withheld"


@pytest.mark.parametrize("failure", [UpstreamError("upstream_unavailable", 200), AttributeError("surprise")])
def test_every_failed_call_still_gets_a_receipt(failure):
    result = _vault(FakeProvider({"/api/v1/chat/completions": failure})).call(_call())
    assert result["transport_error"] == "upstream_unavailable"
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


# -- the vault's HTTPS server ---------------------------------------------------------

# The tests reach the server without pinning; the trust tests pin.
UNVERIFIED = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
UNVERIFIED.check_hostname, UNVERIFIED.verify_mode = False, ssl.CERT_NONE


def _tls_vault(provider=None):
    context, spki = tls_server_context()
    vault = Vault(encryption_key=ENCRYPTION, receipts_key=RECEIPTS, identity=_identity(), caller_token=TOKEN,
                  tls_spki_sha256=spki, upstream=provider or FakeProvider())
    return vault, (context, spki)


@contextlib.contextmanager
def _serving(handler, tls, server_class=BoundedServer, port=0):
    """``handler`` behind the vault's own HTTPS server on a loopback port."""

    server = server_class(("127.0.0.1", port), handler, tls[0])
    thread = threading.Thread(target=server.serve_forever, args=(0.05,))
    thread.start()
    try:
        yield server, "https://127.0.0.1:%d" % server.server_address[1]
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def _connect(server):
    return http.client.HTTPSConnection("127.0.0.1", server.server_address[1], timeout=5, context=UNVERIFIED)


def test_http_server_routes_authenticates_and_never_logs(capsys):
    provider = FakeProvider()
    vault, tls = _tls_vault(provider)

    def post(base, path, document, token=TOKEN):
        request = urllib.request.Request(base + path, data=json.dumps(document).encode(), method="POST",
                                         headers={CALLER_HEADER: token} if token else {})
        try:
            with urllib.request.urlopen(request, context=UNVERIFIED) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read())

    with _serving(make_handler(vault), tls) as (_, base):
        with urllib.request.urlopen(base + "/v1/identity", context=UNVERIFIED) as response:
            assert json.loads(response.read())["boot_id"] == vault.boot_id
        assert post(base, "/v1/call", _call(), token=None) == (401, {"error": "caller_unauthorized"})
        assert post(base, "/v1/call", _call(url="https://evil.example/x")) == (400, {"error": "host_not_allowed"})
        status, result = post(base, "/v1/call", _call())
        assert status == 200 and result["status"] == 200 and len(provider.requests) == 1
        assert post(base, "/v1/nope", {})[0] == 404
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""

def test_a_request_head_is_capped_before_it_is_buffered():
    vault, tls = _tls_vault()
    with _serving(make_handler(vault), tls) as (server, _):
        connection = _connect(server)
        connection.putrequest("GET", "/healthz")
        for index in range(4):
            connection.putheader("x-pad-%d" % index, "a" * 10000)
        connection.endheaders()
        assert connection.getresponse().status == 431
        connection.close()
        connection = _connect(server)
        connection.request("GET", "/healthz")
        assert connection.getresponse().status == 200
        connection.close()


def test_a_client_that_resets_mid_request_leaves_no_trace(capsys):
    import struct

    vault, tls = _tls_vault()
    with _serving(make_handler(vault), tls) as (server, _):
        client = UNVERIFIED.wrap_socket(socket.create_connection(server.server_address))
        client.sendall(b"POST /v1/call HTTP/1.1\r\ncontent-length: 1000\r\n" + CALLER_HEADER.encode()
                       + b": " + TOKEN.encode() + b"\r\n\r\n{")
        time.sleep(0.1)
        # Close with a reset, not a clean shutdown.
        client.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
        client.close()
        time.sleep(0.2)
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""


def test_stopping_the_server_lets_a_call_in_flight_finish():
    started, release = threading.Event(), threading.Event()

    class HeldProvider(FakeProvider):
        def __call__(self, *args):
            started.set()
            release.wait(5)
            return super().__call__(*args)

    vault, tls = _tls_vault(HeldProvider())
    server = BoundedServer(("127.0.0.1", 0), make_handler(vault), tls[0])
    serving = threading.Thread(target=server.serve_forever)
    serving.start()
    answers = []

    def call():
        connection = _connect(server)
        connection.request("POST", "/v1/call", body=json.dumps(_call()).encode(), headers={CALLER_HEADER: TOKEN})
        answers.append(json.loads(connection.getresponse().read()))

    caller = threading.Thread(target=call)
    caller.start()
    assert started.wait(5)
    server.shutdown()
    serving.join()
    closing = threading.Thread(target=server.server_close)
    closing.start()
    closing.join(0.3)
    assert closing.is_alive()  # waiting for the call in flight
    release.set()
    closing.join(5)
    caller.join(5)
    assert answers[0]["receipt"]["document"]["outcome"] == "response"


def test_every_answer_closes_its_connection_and_slow_clients_are_cut(monkeypatch):
    from lead_order_vault import vault as vault_module

    monkeypatch.setattr(vault_module, "REQUEST_READ_SECONDS", 0.5)

    class OneSlot(BoundedServer):
        MAX_CONNECTIONS = 1

    vault, tls = _tls_vault()
    with _serving(make_handler(vault), tls, OneSlot) as (server, _):
        # A refusal reads the body first and still closes the connection.
        connection = _connect(server)
        connection.request("POST", "/v1/call", body=json.dumps(_call()).encode())
        refused = connection.getresponse()
        refused.read()
        assert refused.status == 401 and refused.will_close
        connection.close()
        # A client that never finishes the TLS handshake is cut too.
        silent = socket.create_connection(server.server_address)
        silent.settimeout(3)
        started = time.monotonic()
        try:
            assert silent.recv(1) == b""
        except ConnectionResetError:
            pass
        assert time.monotonic() - started < 2
        silent.close()
        # A client sending one byte at a time holds the only slot only until
        # the read deadline, not for as long as it keeps trickling.
        slow = UNVERIFIED.wrap_socket(socket.create_connection(server.server_address))
        slow.settimeout(0.2)
        started = time.monotonic()
        try:
            while time.monotonic() - started < 3:
                slow.sendall(b"G")
                try:
                    if slow.recv(1) == b"":
                        break
                except socket.timeout:
                    pass
        except OSError:
            pass
        assert time.monotonic() - started < 2
        slow.close()
        time.sleep(0.1)
        connection = _connect(server)
        connection.request("GET", "/healthz")
        answered = connection.getresponse()
        assert answered.status == 200 and answered.will_close
        connection.close()

def test_the_real_upstream_bounds_total_time_and_size():
    from lead_order_vault.vault import https_upstream

    server_context, _ = tls_server_context()
    listener = socket.create_server(("127.0.0.1", 0))
    port = listener.getsockname()[1]

    def serve(answer):
        raw, _ = listener.accept()
        with server_context.wrap_socket(raw, server_side=True) as connection:
            connection.recv(65536)
            answer(connection)

    def trickle(connection):
        connection.sendall(b"HTTP/1.1 200 OK\r\ncontent-length: 100\r\n\r\n")
        for _ in range(30):
            time.sleep(0.1)
            try:
                connection.sendall(b"x")
            except OSError:
                return

    def oversize(connection):
        connection.sendall(b"HTTP/1.1 200 OK\r\ncontent-length: 2048\r\n\r\n" + b"x" * 2048)

    def whole(connection):
        connection.sendall(b"HTTP/1.1 200 OK\r\ncontent-length: 2\r\n\r\nok")

    def closing(connection):
        # http.client drops the connection's socket when the answer says close.
        connection.sendall(b"HTTP/1.1 429 Too Many Requests\r\nconnection: close\r\ncontent-length: 2\r\n\r\nno")

    client_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    client_context.check_hostname, client_context.verify_mode = False, ssl.CERT_NONE
    try:
        for answer, error in ((trickle, "upstream_timeout"), (oversize, "response_too_large"),
                              (whole, b"ok"), (closing, b"no")):
            thread = threading.Thread(target=serve, args=(answer,), daemon=True)
            thread.start()
            started = time.monotonic()
            call = lambda: https_upstream("GET", "127.0.0.1", "/", {}, b"", 1.0, 1024, port=port,
                                          context=client_context)
            if isinstance(error, str):
                with pytest.raises(UpstreamError, match=error) as raised:
                    call()
                assert raised.value.observed_status == 200
            else:
                assert call().body == error
            assert time.monotonic() - started < 2
            thread.join(5)
    finally:
        listener.close()


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

    provider = FakeProvider({"/api/v2/integrations/boom/execute": UpstreamError("reset", observed_status=None)})
    vault, tls = _tls_vault(provider)
    with _serving(make_handler(vault), tls) as (_, url):
        yield dict(DEPLOYMENT, url=url), provider


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

    deployment, provider = served_vault
    with pytest.raises(VaultTransportError) as refused:
        _send(VaultTransport(VaultClient(deployment, TOKEN)), url="https://openrouter.ai/api/v1/keys")
    assert refused.value.forwarded is False and refused.value.vault_receipt is None and not provider.requests
    with pytest.raises(VaultTransportError) as unauthorized:
        _send(VaultTransport(VaultClient(deployment, "x" * 40)))
    assert unauthorized.value.forwarded is False
    with pytest.raises(VaultTransportError) as lost:
        _send(VaultTransport(VaultClient(deployment, TOKEN)), url="https://code.deepline.com/api/v2/integrations/boom/execute",
              sealed=_seal(DEEPLINE_KEY, "deepline"))
    assert lost.value.forwarded is True and _check_receipt(lost.value.vault_receipt)["outcome"] == "transport_error"
    with pytest.raises(VaultTransportError) as down:
        _send(VaultTransport(VaultClient(dict(deployment, url="https://127.0.0.1:9"), TOKEN)))
    assert down.value.forwarded is False


def test_vault_client_checks_a_sealed_key(served_vault):
    from lab_arena.lead_orders.vault_client import VaultClient

    url, _ = served_vault
    checked = VaultClient(url, TOKEN).check(provider="deepline", envelope=_seal(DEEPLINE_KEY, "deepline"),
                                            management_envelope=None)
    assert checked["status"] == "valid" and checked["key_fingerprint"] == env.fingerprint("deepline", DEEPLINE_KEY)


def test_re_verifying_leaves_requests_in_flight_alone(served_vault, monkeypatch):
    from lab_arena.lead_orders import vault_client
    from lab_arena.lead_orders.vault_client import VaultClient

    deployment, _ = served_vault
    verified = []
    real_connect = vault_client.trust.connect
    monkeypatch.setattr(vault_client.trust, "connect", lambda d: verified.append(1) or real_connect(d))
    client = VaultClient(deployment, TOKEN)
    first = client._pinned()
    # Two threads saw the same connect failure: one re-verifies, the other reuses its result.
    second = client._pinned(failed=first)
    assert client._pinned(failed=first) is second and len(verified) == 2
    # The replaced client stays open for requests still running on it.
    assert not first.is_closed and first.get(deployment["url"] + "/healthz").json() == {"ok": True}


def test_vault_client_follows_a_restarted_vault_to_its_new_tls_key():
    from lab_arena.lead_orders.vault_client import VaultClient, VaultTransport

    first, tls = _tls_vault()
    with _serving(make_handler(first), tls) as (server, url):
        port = server.server_address[1]
        transport = VaultTransport(VaultClient(dict(DEPLOYMENT, url=url), TOKEN))
        assert _check_receipt(_send(transport).vault_receipt)["key_sequence"] == 1
    # Same address, new boot: a new TLS key and a new boot id.
    restarted, tls = _tls_vault()
    with _serving(make_handler(restarted), tls, port=port):
        assert _check_receipt(_send(transport).vault_receipt)["boot_id"] == restarted.boot_id


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


def test_a_vault_fault_leaves_it_unknown_whether_the_call_was_sent():
    import httpx

    from lab_arena.lead_orders.vault_client import VaultClient, VaultUnavailable

    for status, document in ((500, {"error": "vault_error"}), (503, {"error": "check_unavailable"}), (200, [])):
        with pytest.raises(VaultUnavailable) as raised:
            VaultClient._document(httpx.Response(status, json=document))
        assert raised.value.forwarded is None, status


def test_a_withheld_answer_keeps_the_generation_id():
    echo = UpstreamResponse(200, [("X-Generation-Id", "gen-9"), ("X-Other", "x")], OPENROUTER_KEY.encode())
    result = _vault(FakeProvider({"/api/v1/chat/completions": echo})).call(_call())
    assert result["status"] == 502 and ["x-generation-id", "gen-9"] in result["headers"]
    assert OPENROUTER_KEY not in json.dumps(result)


# -- the vault ends TLS itself ------------------------------------------------------


@contextlib.contextmanager
def _tls_server(identity_source, tls=None):
    """A server like the vault's whose /v1/identity is ``identity_source()``: (deployment, spki)."""

    from http.server import BaseHTTPRequestHandler

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            self.server.request_read()
            data = json.dumps(identity_source()).encode()
            self.send_response(200)
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    tls = tls or tls_server_context()
    with _serving(Handler, tls) as (_, url):
        yield dict(DEPLOYMENT, url=url), tls[1]

def test_the_cli_trusts_only_a_tls_session_the_vault_ends_itself():
    vault, tls = _tls_vault()
    with _tls_server(vault.identity, tls) as (deployment, spki):
        identity, certificate = trust.connect(deployment)
        assert identity["tls"]["document"]["spki_sha256"] == spki
        context = trust.tls_context(certificate)
        with urllib.request.urlopen(deployment["url"] + "/v1/identity", context=context) as response:
            assert json.loads(response.read())["boot_id"] == vault.boot_id
        # After a restart the vault has a new TLS key: the old pin fails closed.
        with _tls_server(vault.identity) as (again, _):
            with pytest.raises(urllib.error.URLError, match="CERTIFICATE_VERIFY_FAILED"):
                urllib.request.urlopen(again["url"] + "/v1/identity", context=context)


def test_a_gateway_that_ends_tls_itself_is_refused():
    from lab_arena.lead_orders.vault_client import VaultClient, VaultTransport, VaultTransportError

    vault, _ = _tls_vault()
    # The gateway relays the vault's real, KMS-signed identity over its own TLS key.
    with _tls_server(vault.identity) as (deployment, _):
        with pytest.raises(trust.TrustError, match="does not end in the vault"):
            trust.connect(deployment)
        # Leadpoet's broker sends nothing through it either.
        with pytest.raises(VaultTransportError) as refused:
            _send(VaultTransport(VaultClient(deployment, TOKEN)))
        assert refused.value.forwarded is False and str(refused.value) == "vault_unverified"


def test_a_gateway_cannot_rebind_the_tls_key_without_the_receipts_key():
    vault, _ = _tls_vault()
    context, spki = tls_server_context()

    def forged():
        identity = json.loads(json.dumps(vault.identity()))
        identity["tls"]["document"]["spki_sha256"] = spki
        return identity

    with _tls_server(forged, (context, spki)) as (deployment, _):
        with pytest.raises(trust.TrustError, match="not signed by the vault.s receipts key"):
            trust.connect(deployment)
