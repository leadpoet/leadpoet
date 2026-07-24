from __future__ import annotations

import asyncio
import ipaddress
import socket
import ssl
import time
import zlib
from dataclasses import dataclass
from typing import Awaitable, Callable
from urllib.parse import urljoin

import aiohttp
from aiohttp.abc import AbstractResolver

from .models import RedirectHop
from .normalization import NormalizationError, normalize_url


MAX_REDIRECTS = 5
MAX_COMPRESSED_BYTES = 256 * 1024
MAX_BODY_BYTES = 1024 * 1024
ALLOWED_PORTS = frozenset({80, 443})


class IdentityFetchError(RuntimeError):
    def __init__(self, code: str, *, transient: bool = False) -> None:
        super().__init__(code)
        self.code = code
        self.transient = transient


@dataclass(frozen=True)
class FetchedPage:
    requested_url: str
    final_url: str
    status: int
    headers: dict[str, str]
    body: bytes
    redirects: list[RedirectHop]
    fetched_at_epoch_ms: int


ResolveHost = Callable[[str, int], Awaitable[list[str]]]


def is_public_address(raw: str) -> bool:
    try:
        address = ipaddress.ip_address(raw)
    except ValueError:
        return False
    return bool(
        address.is_global
        and not address.is_private
        and not address.is_loopback
        and not address.is_link_local
        and not address.is_multicast
        and not address.is_reserved
        and not address.is_unspecified
    )


async def resolve_public_addresses(host: str, port: int) -> list[str]:
    try:
        rows = await asyncio.get_running_loop().getaddrinfo(
            host, port, type=socket.SOCK_STREAM, proto=socket.IPPROTO_TCP
        )
    except OSError as exc:
        raise IdentityFetchError("dns_unavailable", transient=True) from exc
    addresses = sorted({row[4][0] for row in rows})
    if not addresses:
        raise IdentityFetchError("dns_no_addresses", transient=True)
    # Reject mixed public/private answers instead of selecting the convenient one.
    if any(not is_public_address(address) for address in addresses):
        raise IdentityFetchError("dns_non_public_address")
    return addresses


class _PinnedResolver(AbstractResolver):
    def __init__(self, expected_host: str, addresses: list[str]) -> None:
        self.expected_host = expected_host
        self.addresses = tuple(addresses)

    async def resolve(
        self, host: str, port: int = 0, family: int = socket.AF_UNSPEC
    ) -> list[dict[str, object]]:
        if host != self.expected_host:
            raise OSError("resolver host changed after validation")
        result = []
        for address in self.addresses:
            version = ipaddress.ip_address(address).version
            address_family = socket.AF_INET6 if version == 6 else socket.AF_INET
            if family not in {socket.AF_UNSPEC, address_family}:
                continue
            result.append({
                "hostname": host,
                "host": address,
                "port": port,
                "family": address_family,
                "proto": socket.IPPROTO_TCP,
                "flags": socket.AI_NUMERICHOST,
            })
        return result

    async def close(self) -> None:
        return None


def _decode_bounded(body: bytes, encoding: str) -> bytes:
    def inflate(wbits: int) -> bytes:
        try:
            inflater = zlib.decompressobj(wbits)
            decoded_value = inflater.decompress(body, MAX_BODY_BYTES + 1)
            if len(decoded_value) > MAX_BODY_BYTES:
                raise IdentityFetchError("response_body_too_large")
            decoded_value += inflater.flush(MAX_BODY_BYTES + 1 - len(decoded_value))
        except zlib.error as exc:
            raise IdentityFetchError("invalid_compressed_response") from exc
        if not inflater.eof or inflater.unused_data:
            raise IdentityFetchError("invalid_compressed_response")
        return decoded_value

    if encoding in {"", "identity"}:
        decoded = body
    elif encoding == "gzip":
        decoded = inflate(16 + zlib.MAX_WBITS)
    elif encoding == "deflate":
        try:
            decoded = inflate(zlib.MAX_WBITS)
        except IdentityFetchError as exc:
            if exc.code != "invalid_compressed_response":
                raise
            # RFC-era servers use both zlib-wrapped and raw DEFLATE streams.
            decoded = inflate(-zlib.MAX_WBITS)
    else:
        raise IdentityFetchError("unsupported_content_encoding")
    if len(decoded) > MAX_BODY_BYTES:
        raise IdentityFetchError("response_body_too_large")
    return decoded


async def _request_one_hop(
    url: str,
    addresses: list[str],
    *,
    timeout_seconds: float,
) -> tuple[int, dict[str, str], bytes]:
    parsed = normalize_url(url)
    connector = aiohttp.TCPConnector(
        resolver=_PinnedResolver(parsed.ascii_host, addresses),
        use_dns_cache=False,
        ssl=ssl.create_default_context(),
        limit=1,
        force_close=True,
    )
    timeout = aiohttp.ClientTimeout(total=timeout_seconds, connect=min(5, timeout_seconds))
    try:
        async with aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            auto_decompress=False,
            headers={
                "Accept": "text/html,application/xhtml+xml",
                "Accept-Encoding": "gzip, deflate",
                "User-Agent": "LeadpoetIdentityResolver/1.0",
            },
            cookie_jar=aiohttp.DummyCookieJar(),
        ) as session:
            async with session.get(
                parsed.url,
                allow_redirects=False,
                max_line_size=8190,
                max_field_size=8190,
            ) as response:
                chunks: list[bytes] = []
                size = 0
                async for chunk in response.content.iter_chunked(16384):
                    size += len(chunk)
                    if size > MAX_COMPRESSED_BYTES:
                        raise IdentityFetchError("compressed_response_too_large")
                    chunks.append(chunk)
                headers = {key.lower(): value for key, value in response.headers.items()}
                body = _decode_bounded(b"".join(chunks), headers.get("content-encoding", "").lower())
                return response.status, headers, body
    except IdentityFetchError:
        raise
    except (asyncio.TimeoutError, aiohttp.ClientError, OSError) as exc:
        raise IdentityFetchError("http_transport_unavailable", transient=True) from exc


async def fetch_page(
    raw_url: str,
    *,
    resolve_host: ResolveHost = resolve_public_addresses,
    total_timeout_seconds: float = 15,
) -> FetchedPage:
    try:
        requested = normalize_url(raw_url)
    except NormalizationError as exc:
        raise IdentityFetchError("unsafe_or_invalid_url") from exc
    current = requested
    visited: set[str] = set()
    redirects: list[RedirectHop] = []
    deadline = time.monotonic() + total_timeout_seconds

    for hop_index in range(MAX_REDIRECTS + 1):
        if current.url in visited:
            raise IdentityFetchError("redirect_loop")
        visited.add(current.url)
        port = current.port or (443 if current.scheme == "https" else 80)
        if port not in ALLOWED_PORTS:
            raise IdentityFetchError("non_standard_port_forbidden")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise IdentityFetchError("fetch_deadline_exceeded", transient=True)
        # DNS is deliberately repeated immediately before each new connection.
        addresses = await resolve_host(current.ascii_host, port)
        if any(not is_public_address(address) for address in addresses):
            raise IdentityFetchError("dns_non_public_address")
        started = time.monotonic()
        status, headers, body = await _request_one_hop(
            current.url, addresses, timeout_seconds=remaining
        )
        if status not in {301, 302, 303, 307, 308}:
            content_type = headers.get("content-type", "").split(";", 1)[0].strip().lower()
            if body and content_type not in {"text/html", "application/xhtml+xml", ""}:
                raise IdentityFetchError("non_html_identity_response")
            return FetchedPage(
                requested_url=requested.url,
                final_url=current.url,
                status=status,
                headers=headers,
                body=body,
                redirects=redirects,
                fetched_at_epoch_ms=int(time.time() * 1000),
            )
        if hop_index >= MAX_REDIRECTS:
            raise IdentityFetchError("redirect_hop_limit")
        location = headers.get("location", "")
        if not location or len(location) > 2048:
            raise IdentityFetchError("invalid_redirect_location")
        try:
            target = normalize_url(urljoin(current.url, location))
        except NormalizationError as exc:
            raise IdentityFetchError("unsafe_redirect_target") from exc
        if current.scheme == "https" and target.scheme == "http":
            raise IdentityFetchError("https_downgrade_redirect")
        redirects.append(RedirectHop(
            source_url=current.url,
            target_url=target.url,
            status=status,
            elapsed_ms=round((time.monotonic() - started) * 1000),
        ))
        current = target

    raise IdentityFetchError("redirect_hop_limit")
