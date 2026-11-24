"""The gateway's side of the lead-order key vault (``lead_order_vault``).

Miner keys never reach the gateway. It stores each miner's sealed envelope
and uses it as the request's credential: ``VaultTransport`` sends the request
to the vault, which swaps the envelope for the key, forwards it to the
provider, and returns the answer with a signed receipt. Leadpoet's own keys
keep the direct transport.

The client talks only to a TLS session the vault itself ends: it pins the
certificate that the vault's KMS-verified identity signed (``trust.connect``),
so Phala's gateway cannot read the caller token or the envelopes.
"""

from __future__ import annotations

import base64
import threading
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional
from urllib.parse import urlsplit

import httpx

from lab_arena import broker as arena_broker
from lab_arena.broker import ProviderResponse, ProviderTransportError
from lead_order_vault import trust
from lead_order_vault.vault import CALLER_HEADER, CHECK_TIMEOUT_SECONDS as _VAULT_PROBE_SECONDS

# Time for the vault itself on top of the provider's own timeout.
VAULT_OVERHEAD_SECONDS = 15.0
# A check probes the runtime key and then the management key.
CHECK_REQUEST_SECONDS = 2 * _VAULT_PROBE_SECONDS + VAULT_OVERHEAD_SECONDS
CONNECT_SECONDS = 10.0
# A failed connection, or none free in the pool, sent nothing.
_NOT_SENT = (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout)


class VaultUnavailable(RuntimeError):
    """The vault could not be reached or refused the request; carries a safe code."""

    def __init__(self, code: str, *, forwarded: Optional[bool], status: Optional[int] = None) -> None:
        super().__init__(code)
        self.code = code
        # The vault's HTTP status; None when it could not be reached.
        self.status = status
        # False: the vault did not send anything to the provider. None: unknown.
        self.forwarded = forwarded

    @property
    def retryable(self) -> bool:
        # 401 is our caller token, not the miner's key.
        return self.status is None or self.status >= 500 or self.status == 401


@dataclass(frozen=True)
class VaultResponse(ProviderResponse):
    """A provider answer that came through the vault, with the vault's receipt."""

    vault_receipt: Optional[Mapping[str, Any]] = None


class VaultTransportError(ProviderTransportError):
    """The provider call failed. ``forwarded`` False means nothing was sent."""

    def __init__(self, message: str, *, forwarded: Optional[bool], observed_status: Optional[int] = None,
                 vault_receipt: Optional[Mapping[str, Any]] = None, openrouter_generation_id: Optional[str] = None,
                 deepline_job_id: Optional[str] = None) -> None:
        super().__init__(message, observed_status=observed_status, openrouter_generation_id=openrouter_generation_id,
                         deepline_job_id=deepline_job_id)
        self.forwarded = forwarded
        self.vault_receipt = vault_receipt


class VaultClient:
    """``deployment`` comes from ``trust.load_deployment``: url, app id, KMS root."""

    def __init__(self, deployment: Mapping[str, Any], caller_token: str) -> None:
        self._url = str(deployment["url"]).rstrip("/")
        self._deployment = deployment
        self._token = caller_token
        self._client: Optional[httpx.Client] = None
        self._lock = threading.Lock()

    def _pinned(self, failed: Optional[httpx.Client] = None) -> httpx.Client:
        """The pinned client; with ``failed``, a freshly verified one unless another thread already replaced it.

        Verification runs outside the lock. A replaced client is left open so
        requests already in flight on it finish.
        """

        with self._lock:
            if self._client is not None and (self._client is not failed):
                return self._client
        try:
            _, certificate = trust.connect(self._deployment)
        except (trust.TrustError, OSError, ValueError, KeyError, TypeError):
            raise VaultUnavailable("vault_unverified", forwarded=False) from None
        fresh = httpx.Client(verify=trust.tls_context(certificate), follow_redirects=False, trust_env=False)
        with self._lock:
            if self._client is None or self._client is failed:
                self._client = fresh
                return fresh
            current = self._client
        fresh.close()
        return current

    def _post(self, path: str, document: Mapping[str, Any], timeout: float) -> Dict[str, Any]:
        client = self._pinned()
        try:
            return self._document(self._send(client, path, document, timeout))
        except _NOT_SENT:
            pass
        # Verify the vault again, which picks up its new TLS key after a
        # restart, and retry once.
        try:
            return self._document(self._send(self._pinned(failed=client), path, document, timeout))
        except _NOT_SENT:
            raise VaultUnavailable("vault_unreachable", forwarded=False) from None

    def _send(self, client: httpx.Client, path: str, document: Mapping[str, Any], timeout: float) -> httpx.Response:
        try:
            # A short connect limit, so an outage fails fast and is retried.
            return client.post(self._url + path, json=document, headers={CALLER_HEADER: self._token},
                               timeout=httpx.Timeout(timeout, connect=CONNECT_SECONDS))
        except _NOT_SENT:
            raise
        except httpx.HTTPError:
            raise VaultUnavailable("vault_unreachable", forwarded=None) from None

    @staticmethod
    def _document(response: httpx.Response) -> Dict[str, Any]:
        try:
            document = response.json()
        except ValueError:
            document = None
        if response.status_code != 200 or not isinstance(document, dict):
            code = document.get("error") if isinstance(document, dict) else None
            status = response.status_code
            # Over the pinned session only the vault answers, and it refuses
            # with a 4xx only before it contacts the provider.
            raise VaultUnavailable("vault_" + str(code or status), status=status,
                                   forwarded=False if 400 <= status < 500 else None)
        return document

    def check(self, *, provider: str, envelope: str, management_envelope: Optional[str]) -> Dict[str, Any]:
        document: Dict[str, Any] = {"provider": provider, "envelope": envelope}
        if management_envelope is not None:
            document["management_envelope"] = management_envelope
        return self._post("/v1/check", document, CHECK_REQUEST_SECONDS)

    def call(self, document: Mapping[str, Any], timeout: float) -> Dict[str, Any]:
        return self._post("/v1/call", document, timeout + VAULT_OVERHEAD_SECONDS)


def _headers(pairs: Any) -> Dict[str, str]:
    if not isinstance(pairs, list):
        return {}
    return {str(pair[0]).lower(): str(pair[1]) for pair in pairs if isinstance(pair, list) and len(pair) == 2}


class VaultTransport:
    """``ProviderTransport`` for miner keys: the credential is the sealed envelope."""

    def __init__(self, vault: VaultClient) -> None:
        self._vault = vault

    def send(self, *, method: str, url: str, headers: Mapping[str, str], body: bytes,
             timeout_seconds: float, max_response_bytes: Optional[int] = None) -> ProviderResponse:
        document: Dict[str, Any] = {
            "method": method, "url": url, "headers": dict(headers), "timeout_seconds": float(timeout_seconds),
            "body_b64": base64.b64encode(body).decode("ascii") if body else None,
        }
        if max_response_bytes is not None:
            document["max_response_bytes"] = int(max_response_bytes)
        try:
            result = self._vault.call(document, float(timeout_seconds))
        except VaultUnavailable as exc:
            raise VaultTransportError(exc.code, forwarded=exc.forwarded) from None
        receipt = result.get("receipt")
        if "transport_error" in result:
            observed = result.get("observed_status")
            # Request ids from headers that arrived before the failure, as the
            # direct transport reports them, so the cost can be read back.
            seen = _headers(result.get("observed_headers"))
            deepline = urlsplit(url).hostname == "code.deepline.com"
            raise VaultTransportError(
                "provider_unreachable", forwarded=True, vault_receipt=receipt,
                observed_status=observed if isinstance(observed, int) else None,
                openrouter_generation_id=None if deepline else arena_broker._openrouter_generation_identity(None, seen)[1],
                deepline_job_id=arena_broker._deepline_response_header_request_id(seen) if deepline else None)
        try:
            status = int(result["status"])
            response_headers = _headers(result.get("headers"))
            content = base64.b64decode(result.get("body_b64") or "", validate=True)
        except (KeyError, TypeError, ValueError):
            raise VaultTransportError("vault_answer_invalid", forwarded=None, vault_receipt=receipt) from None
        return VaultResponse(status=status, headers=response_headers, body=content, vault_receipt=receipt)


__all__ = ["VaultClient", "VaultResponse", "VaultTransport", "VaultTransportError", "VaultUnavailable"]
