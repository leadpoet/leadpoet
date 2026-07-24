from __future__ import annotations

import asyncio
import gzip
import unittest
from unittest.mock import AsyncMock, patch

from leadpoet_verifier.identity.network import (
    IdentityFetchError,
    _decode_bounded,
    fetch_page,
    is_public_address,
)
from leadpoet_verifier.identity.normalization import (
    NormalizationError,
    exact_or_legal_name_match,
    is_label_subdomain,
    normalize_host,
    normalize_linkedin_company_url,
    normalize_name,
    normalize_url,
)


class CompanyIdentityNormalizationTests(unittest.TestCase):
    def test_domain_vectors_include_icann_private_and_idn_rules(self) -> None:
        vectors = {
            "Example.COM.": ("example.com", "com", "example.com", False),
            "jobs.example.co.uk": ("jobs.example.co.uk", "co.uk", "example.co.uk", False),
            "example.com.au": ("example.com.au", "com.au", "example.com.au", False),
            "tenant.blogspot.com": (
                "tenant.blogspot.com", "blogspot.com", "tenant.blogspot.com", True,
            ),
            "www.bücher.de": ("www.xn--bcher-kva.de", "de", "xn--bcher-kva.de", False),
        }
        for raw, expected in vectors.items():
            with self.subTest(raw=raw):
                actual = normalize_host(raw)
                self.assertEqual(
                    (actual.ascii_host, actual.public_suffix, actual.registrable_domain, actual.is_private_suffix),
                    expected,
                )

    def test_public_suffix_ip_single_label_and_malformed_hosts_fail_closed(self) -> None:
        for raw in (
            "com", "co.uk", "blogspot.com", "localhost", "127.0.0.1", "[::1]",
            "bad_domain.com", "evil.com\\@example.com", "", "exa mple.com",
        ):
            with self.subTest(raw=raw), self.assertRaises(NormalizationError):
                normalize_host(raw)

    def test_url_normalization_is_idempotent_and_drops_only_fragment(self) -> None:
        once = normalize_url(" HTTPS://BÜCHER.de:443/a%20b?q=1#frag ", allow_bare_domain=True)
        twice = normalize_url(once.url)
        self.assertEqual(once, twice)
        self.assertEqual(once.url, "https://xn--bcher-kva.de/a%20b?q=1")
        self.assertEqual(once.origin, "https://xn--bcher-kva.de")

    def test_url_rejects_parser_smuggling_and_unsafe_schemes(self) -> None:
        for raw in (
            "https://user@example.com", "https://user:pass@example.com",
            "file:///etc/passwd", "javascript:alert(1)", "https://example.com/%ZZ",
            "https://example.com/\r\nHost:evil.test", "https://example.com\\@evil.test",
            "//example.com/path", "https://example.com:99999/",
        ):
            with self.subTest(raw=raw), self.assertRaises(NormalizationError):
                normalize_url(raw)

    def test_bare_domains_are_accepted_only_when_explicitly_allowed(self) -> None:
        with self.assertRaises(NormalizationError):
            normalize_url("example.com")
        self.assertEqual(
            normalize_url("example.com", allow_bare_domain=True).url,
            "https://example.com/",
        )

    def test_linkedin_company_routes_are_exact(self) -> None:
        value = normalize_linkedin_company_url("https://uk.linkedin.com/company/Acme-42/?trk=x#top")
        self.assertEqual(value.canonical_url, "https://linkedin.com/company/acme-42")
        self.assertIsNone(value.company_id)
        numeric = normalize_linkedin_company_url("https://www.linkedin.com/company/12345")
        self.assertEqual(numeric.company_id, "12345")
        for raw in (
            "https://linkedin.com/in/acme", "https://linkedin.com/school/acme",
            "https://linkedin.com/company/acme/jobs", "https://evil-linkedin.com/company/acme",
            "https://linkedin.com/company/acme_showcase",
        ):
            with self.subTest(raw=raw), self.assertRaises(NormalizationError):
                normalize_linkedin_company_url(raw)

    def test_label_boundaries_prevent_suffix_confusion(self) -> None:
        self.assertTrue(is_label_subdomain("careers.example.com", "example.com"))
        self.assertFalse(is_label_subdomain("evil-example.com", "example.com"))
        self.assertFalse(is_label_subdomain("example.com.evil.test", "example.com"))
        self.assertFalse(is_label_subdomain("example.com", "example.com"))

    def test_name_comparison_keeps_digits_and_uses_legal_suffix_view_only(self) -> None:
        self.assertTrue(exact_or_legal_name_match("Acme, Inc.", "ACME"))
        self.assertTrue(exact_or_legal_name_match("Marks & Spencer Ltd", "Marks and Spencer"))
        self.assertFalse(exact_or_legal_name_match("Studio 42", "Studio 24"))
        self.assertFalse(exact_or_legal_name_match("AC", "Acme Corporation"))
        self.assertEqual(normalize_name("ACME GmbH"), "acme gmbh")

    def test_public_address_filter_blocks_all_special_ranges(self) -> None:
        for address in (
            "127.0.0.1", "10.0.0.1", "172.16.0.1", "192.168.1.1",
            "169.254.169.254", "100.64.0.1", "0.0.0.0", "224.0.0.1",
            "192.0.2.1", "::1", "fc00::1", "fe80::1", "2001:db8::1",
        ):
            with self.subTest(address=address):
                self.assertFalse(is_public_address(address))
        self.assertTrue(is_public_address("93.184.216.34"))
        self.assertTrue(is_public_address("2606:2800:220:1:248:1893:25c8:1946"))

    def test_bounded_decoder_rejects_compression_bombs_and_unknown_encodings(self) -> None:
        self.assertEqual(_decode_bounded(gzip.compress(b"hello"), "gzip"), b"hello")
        compressor = __import__("zlib").compressobj(wbits=-__import__("zlib").MAX_WBITS)
        raw_deflate = compressor.compress(b"hello") + compressor.flush()
        self.assertEqual(_decode_bounded(raw_deflate, "deflate"), b"hello")
        bomb = gzip.compress(b"x" * (1024 * 1024 + 1))
        with self.assertRaisesRegex(IdentityFetchError, "response_body_too_large"):
            _decode_bounded(bomb, "gzip")
        with self.assertRaisesRegex(IdentityFetchError, "unsupported_content_encoding"):
            _decode_bounded(b"hello", "br")
        with self.assertRaisesRegex(IdentityFetchError, "invalid_compressed_response"):
            _decode_bounded(b"not-gzip", "gzip")


class CompanyIdentityNetworkTests(unittest.IsolatedAsyncioTestCase):
    async def test_each_redirect_is_renormalized_reresolved_and_connection_pinned(self) -> None:
        resolutions: list[tuple[str, int]] = []

        async def resolver(host: str, port: int) -> list[str]:
            resolutions.append((host, port))
            return ["93.184.216.34"]

        request = AsyncMock(side_effect=[
            (301, {"location": "https://www.example.com/home"}, b""),
            (200, {"content-type": "text/html"}, b"<html>ok</html>"),
        ])
        with patch("leadpoet_verifier.identity.network._request_one_hop", request):
            page = await fetch_page("http://example.com", resolve_host=resolver)

        self.assertEqual(resolutions, [("example.com", 80), ("www.example.com", 443)])
        self.assertEqual(page.final_url, "https://www.example.com/home")
        self.assertEqual(len(page.redirects), 1)
        self.assertEqual(request.call_args_list[0].args[1], ["93.184.216.34"])

    async def test_private_or_mixed_dns_answers_never_reach_transport(self) -> None:
        request = AsyncMock()

        async def resolver(_host: str, _port: int) -> list[str]:
            return ["93.184.216.34", "169.254.169.254"]

        with (
            patch("leadpoet_verifier.identity.network._request_one_hop", request),
            self.assertRaisesRegex(IdentityFetchError, "dns_non_public_address"),
        ):
            await fetch_page("https://example.com", resolve_host=resolver)
        request.assert_not_awaited()

    async def test_https_downgrade_redirect_fails_closed(self) -> None:
        async def resolver(_host: str, _port: int) -> list[str]:
            return ["93.184.216.34"]

        request = AsyncMock(return_value=(301, {"location": "http://example.com"}, b""))
        with (
            patch("leadpoet_verifier.identity.network._request_one_hop", request),
            self.assertRaisesRegex(IdentityFetchError, "https_downgrade_redirect"),
        ):
            await fetch_page("https://example.com", resolve_host=resolver)

    async def test_redirect_loops_and_nonstandard_ports_fail_closed(self) -> None:
        async def resolver(_host: str, _port: int) -> list[str]:
            return ["93.184.216.34"]

        request = AsyncMock(return_value=(302, {"location": "/"}, b""))
        with (
            patch("leadpoet_verifier.identity.network._request_one_hop", request),
            self.assertRaisesRegex(IdentityFetchError, "redirect_loop"),
        ):
            await fetch_page("https://example.com", resolve_host=resolver)
        with self.assertRaisesRegex(IdentityFetchError, "non_standard_port_forbidden"):
            await fetch_page("https://example.com:8443", resolve_host=resolver)


if __name__ == "__main__":
    unittest.main()
