"""The gateway's side of the lead-order key vault (``lead_order_vault``).

Miner keys never reach the gateway. It stores each miner's sealed envelope
and uses it as the request's credential: ``VaultTransport`` sends the request
to the vault, which swaps the envelope for the key, forwards it to the
provider, and returns the answer with a signed receipt. Leadpoet's own keys
keep the direct transport.

The client talks only to a TLS session the vault itself ends: it pins the
certificate that the vault's KMS-verified identity signed (``cli.connect``),
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
from lead_order_vault import cli as vault_cli

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
    """``deployment`` is ``lead_order_vault/deployment.json``: url, app id, KMS root.

    ``client`` replaces the pinned client, for tests only.
    """

    def __init__(self, deployment: Mapping[str, Any], caller_token: str, *,
                 client: Optional[httpx.Client] = None) -> None:
        self._url = str(deployment.get("url") or "").rstrip("/")
        if not self._url.startswith("https://"):
            raise ValueError("vault url must be https")
        self._deployment = deployment
        self._token = caller_token
        self._client = client
        self._fixed = client is not None
        self._lock = threading.Lock()

    def _pinned(self, failed: Optional[httpx.Client] = None) -> httpx.Client:
        """The pinned client; with ``failed``, a freshly verified one unless another thread already replaced it.

        Verification runs outside the lock. A replaced client is left open so
        requests already in flight on it finish.
        """

        with self._lock:
            if self._client is not None and (self._client is not failed or self._fixed):
                return self._client
        try:
            _, certificate = vault_cli.connect(self._deployment)
        except (vault_cli.CliError, OSError, ValueError, KeyError, TypeError):
            raise VaultUnavailable("vault_unverified", forwarded=False) from None
        fresh = httpx.Client(verify=vault_cli.pinned_context(certificate), follow_redirects=False, trust_env=False)
        with self._lock:
            if self._client is None or self._client is failed:
                self._client = fresh
                return fresh
            current = self._client
        fresh.close()
        return current

    def _post(self, path: str, document: Mapping[str, Any], timeout: float) -> Dict[str, Any]:
        # A failed connect sent nothing. It is retried once after verifying the
        # vault again, which picks up the new TLS key after a restart.
        client = self._pinned()
        for retry in (False, True):
            try:
                response = client.post(self._url + path, json=document, timeout=timeout,
                                       headers={CALLER_HEADER: self._token})
            except (httpx.ConnectError, httpx.ConnectTimeout):
                if not retry:
                    client = self._pinned(failed=client)
                    continue
                raise VaultUnavailable("vault_unreachable", forwarded=False) from None
            except httpx.HTTPError:
                raise VaultUnavailable("vault_unreachable", forwarded=None) from None
            return self._document(response)
        raise AssertionError("unreachable")

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
