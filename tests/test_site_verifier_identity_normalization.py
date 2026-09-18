from __future__ import annotations

import unittest
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

if __name__ == "__main__":
    unittest.main()
