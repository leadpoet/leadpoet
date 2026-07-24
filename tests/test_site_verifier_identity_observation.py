from __future__ import annotations

import unittest

from leadpoet_verifier.identity.network import FetchedPage
from leadpoet_verifier.identity.observation import observation_from_page


class CompanyIdentityObservationTests(unittest.TestCase):
    def test_ordinary_meta_and_script_tags_without_identity_attributes_are_ignored(
        self,
    ) -> None:
        page = FetchedPage(
            requested_url="https://acme.example/",
            final_url="https://acme.example/",
            status=200,
            headers={"content-type": "text/html"},
            body=(
                b'<html><head><meta charset="utf-8">'
                b'<script>window.example = true;</script>'
                b'<meta property="og:site_name" content="Acme">'
                b'</head><body></body></html>'
            ),
            redirects=[],
            fetched_at_epoch_ms=1_753_161_600_000,
        )

        observation = observation_from_page(
            page, evidence_ref="identity-homepage-fetch:acme.example"
        )

        self.assertEqual(observation.names, ["Acme"])
        self.assertEqual(observation.status, 200)
        self.assertFalse(observation.transient_failure)


if __name__ == "__main__":
    unittest.main()
