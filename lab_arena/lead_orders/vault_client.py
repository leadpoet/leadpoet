"""The gateway's side of the lead-order key vault (``lead_order_vault``).

Miner keys never reach the gateway. It stores each miner's sealed envelope
and uses it as the request's credential: ``VaultTransport`` sends the request
to the vault, which swaps the envelope for the key, forwards it to the
provider, and returns the answer with a signed receipt. Leadpoet's own keys
keep the direct transport.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional
from urllib.parse import urlsplit

import httpx

from lab_arena import broker as arena_broker
from lab_arena.broker import ProviderResponse, ProviderTransportError

CALLER_HEADER = "x-lead-order-vault-token"
# Time for the vault itself on top of the provider's own timeout.
VAULT_OVERHEAD_SECONDS = 15.0
CHECK_TIMEOUT_SECONDS = 45.0
# The vault's own refusals, all raised before it contacts the provider. Any
# other error (a proxy's 408, a 5xx) may come after it already forwarded.
_REFUSED_BEFORE_SENDING = frozenset({
    "request_invalid", "method_not_allowed", "url_invalid", "host_not_allowed", "path_not_allowed",
    "headers_invalid", "body_invalid", "timeout_invalid", "max_response_bytes_invalid", "credential_missing",
    "provider_mismatch", "purpose_mismatch", "caller_unauthorized", "request_too_large", "not_found",
})


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
        return self.status is None or self.status >= 500


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
    def __init__(self, url: str, caller_token: str, *, client: Optional[httpx.Client] = None) -> None:
        if not url.startswith("https://") and not url.startswith("http://127.0.0.1"):
            raise ValueError("vault url must be https")
        self._url = url.rstrip("/")
        self._token = caller_token
        self._client = client or httpx.Client(follow_redirects=False, trust_env=False)

    def _post(self, path: str, document: Mapping[str, Any], timeout: float) -> Dict[str, Any]:
        try:
            response = self._client.post(self._url + path, json=document, timeout=timeout,
                                         headers={CALLER_HEADER: self._token})
        except (httpx.ConnectError, httpx.ConnectTimeout):
            raise VaultUnavailable("vault_unreachable", forwarded=False) from None
        except httpx.HTTPError:
            raise VaultUnavailable("vault_unreachable", forwarded=None) from None
        return self._document(response)

    @staticmethod
    def _document(response: httpx.Response) -> Dict[str, Any]:
        try:
            document = response.json()
        except ValueError:
            document = None
        if response.status_code != 200 or not isinstance(document, dict):
            code = document.get("error") if isinstance(document, dict) else None
            status = response.status_code
            refused = isinstance(code, str) and code in _REFUSED_BEFORE_SENDING
            raise VaultUnavailable("vault_" + str(code or status), status=status,
                                   forwarded=False if refused else None)
        return document

    def identity(self) -> Dict[str, Any]:
        try:
            response = self._client.get(self._url + "/v1/identity", timeout=10.0)
        except httpx.HTTPError:
            raise VaultUnavailable("vault_unreachable", forwarded=False) from None
        return self._document(response)

    def check(self, *, provider: str, envelope: str, management_envelope: Optional[str]) -> Dict[str, Any]:
        document: Dict[str, Any] = {"provider": provider, "envelope": envelope}
        if management_envelope is not None:
            document["management_envelope"] = management_envelope
        return self._post("/v1/check", document, CHECK_TIMEOUT_SECONDS)

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
