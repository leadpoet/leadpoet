from __future__ import annotations

import hashlib
import unittest
from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

from leadpoet_verifier.contracts import VerificationDecision
from leadpoet_verifier.identity.binding import compare_identity_bound_claim
from leadpoet_verifier.identity.canonical import canonical_sha256
from leadpoet_verifier.identity.models import (
    AnchorSet,
    LinkedInCompanyIdentity,
    PriorReceiptSummary,
    RedirectHop,
    ResolverInput,
    WebsiteObservation,
)
from leadpoet_verifier.identity.policy import resolve_identity


NOW = datetime(2026, 7, 20, 12, 0, tzinfo=timezone.utc)
EMPTY_SHA = hashlib.sha256(b"").hexdigest()


def anchor(
    *, domain: str = "acme.com", website: str | None = None,
    linkedin: str = "https://linkedin.com/company/acme",
) -> AnchorSet:
    return AnchorSet(
        submitted_name="Acme, Inc.",
        submitted_domain=domain,
        submitted_website=website or f"https://{domain}",
        submitted_linkedin_url=linkedin,
    )


def site(
    host: str = "acme.com",
    *,
    names: list[str] | None = None,
    linkedin_urls: list[str] | None = None,
    outbound_hosts: list[str] | None = None,
    parked: bool = False,
    aggregator: bool = False,
    shared: bool = False,
    transient: bool = False,
    unsafe: bool = False,
    requested_host: str | None = None,
    redirects: list[RedirectHop] | None = None,
) -> WebsiteObservation:
    requested = requested_host or host
    return WebsiteObservation(
        requested_url=f"https://{requested}/",
        final_url=f"https://{host}/",
        status=599 if transient or unsafe else 200,
        fetched_at=NOW,
        content_sha256=EMPTY_SHA,
        names=names if names is not None else ["Acme"],
        linkedin_company_urls=linkedin_urls or [],
        outbound_hosts=outbound_hosts or [],
        redirects=redirects or [],
        parked=parked,
        aggregator=aggregator,
        shared_infrastructure=shared,
        transient_failure=transient,
        unsafe_target=unsafe,
        source_evidence_ref=f"site:{requested}",
    )


def linkedin(
    slug: str = "acme", *, company_id: str | None = None, domain: str = "acme.com"
) -> LinkedInCompanyIdentity:
    route = company_id or slug
    return LinkedInCompanyIdentity(
        company_id=company_id,
        canonical_url=f"https://linkedin.com/company/{route}",
        normalized_slug=route,
        listed_website_host=domain,
        source_evidence_ref=f"linkedin:{route}",
        observed_at=NOW,
    )


def resolver_input(
    value: AnchorSet,
    *,
    sites: list[WebsiteObservation] | None = None,
    linkedins: list[LinkedInCompanyIdentity] | None = None,
    prior: PriorReceiptSummary | None = None,
    primary_domains: list[str] | None = None,
    relationship_conflict: bool = False,
    country_conflict: bool = False,
) -> ResolverInput:
    return ResolverInput(
        request_id=uuid4(),
        run_id=uuid4(),
        candidate_id=uuid4(),
        anchor_set_hash=canonical_sha256(value),
        candidate_hash="a" * 64,
        anchor_set=value,
        website_observations=sites or [],
        linkedin_identities=linkedins or [],
        prior_receipt=prior,
        verified_primary_domains=primary_domains or [],
        relationship_conflict=relationship_conflict,
        country_conflict=country_conflict,
        observed_at=NOW,
    )


class CompanyIdentityPolicyTests(unittest.TestCase):
    def assert_rule(self, decision, rule: str, tier: str) -> None:
        self.assertEqual(decision.outcome, "resolved")
        self.assertEqual(decision.identity_tier, tier)
        self.assertEqual(decision.positive_rule_ids, [rule])
        self.assertTrue(decision.canonical_identity.identity_match_key.startswith("idmk1:"))

    def test_a1_numeric_linkedin_id_and_live_named_website_resolve_exact(self) -> None:
        decision = resolve_identity(resolver_input(
            anchor(linkedin="https://linkedin.com/company/12345"),
            sites=[site()], linkedins=[linkedin(company_id="12345")],
        ))
        self.assert_rule(decision, "ID-A1-LINKEDIN-ID-WEBSITE", "exact")

    def test_same_resolver_input_produces_identical_identity_commitment(self) -> None:
        value = resolver_input(
            anchor(linkedin="https://linkedin.com/company/12345"),
            sites=[site()],
            linkedins=[linkedin(company_id="12345")],
        )

        first = resolve_identity(value)
        second = resolve_identity(value)

        self.assertEqual(
            first.canonical_identity.company_identity_id,
            second.canonical_identity.company_identity_id,
        )
        self.assertEqual(
            first.canonical_identity.identity_snapshot_hash,
            second.canonical_identity.identity_snapshot_hash,
        )
        self.assertEqual(
            first.model_dump(mode="json"),
            second.model_dump(mode="json"),
        )

    def test_a2_reciprocal_slug_and_live_named_website_resolve_exact(self) -> None:
        decision = resolve_identity(resolver_input(
            anchor(),
            sites=[site(linkedin_urls=["https://linkedin.com/company/acme"])],
            linkedins=[linkedin()],
        ))
        self.assert_rule(decision, "ID-A2-RECIPROCAL-LINKEDIN-SLUG", "exact")

    def test_a3_fresh_same_tenant_receipt_requires_current_safe_fetch(self) -> None:
        prior = PriorReceiptSummary(
            receipt_id=uuid4(), same_tenant=True, decision="resolved",
            positive_rule_ids=["ID-A2-RECIPROCAL-LINKEDIN-SLUG"],
            stable_linkedin_key="acme", verified_domain="acme.com",
            identity_match_key="idmk1:" + "b" * 64,
            valid_until=NOW + timedelta(days=1),
        )
        decision = resolve_identity(resolver_input(
            anchor(), sites=[site()], linkedins=[linkedin()], prior=prior,
        ))
        self.assert_rule(decision, "ID-A3-FRESH-PRIOR-RECEIPT", "exact")
        self.assertEqual(decision.canonical_identity.identity_match_key, prior.identity_match_key)

    def test_b1_verified_primary_and_controlled_subdomain_resolve_strong(self) -> None:
        value = anchor(domain="careers.acme.com")
        decision = resolve_identity(resolver_input(
            value, sites=[site("careers.acme.com")], primary_domains=["acme.com"],
        ))
        self.assert_rule(decision, "ID-B1-CONTROLLED-SUBDOMAIN", "strong")

    def test_b2_reciprocal_regional_alias_resolves_strong(self) -> None:
        value = anchor(domain="acme.de")
        decision = resolve_identity(resolver_input(
            value,
            sites=[
                site(
                    "acme.de", outbound_hosts=["acme.com"],
                    linkedin_urls=["https://linkedin.com/company/acme"],
                ),
                site(
                    "acme.com", outbound_hosts=["acme.de"],
                    linkedin_urls=["https://linkedin.com/company/acme"],
                ),
            ],
            linkedins=[linkedin(domain="acme.com")],
            primary_domains=["acme.com"],
        ))
        self.assert_rule(decision, "ID-B2-RECIPROCAL-REGIONAL-ALIAS", "strong")

    def test_b3_verified_domain_migration_requires_redirect_and_backlink(self) -> None:
        prior = PriorReceiptSummary(
            receipt_id=uuid4(), same_tenant=True, decision="resolved",
            positive_rule_ids=["ID-A1-LINKEDIN-ID-WEBSITE"], stable_linkedin_key="123",
            verified_domain="old-acme.com", identity_match_key="idmk1:" + "c" * 64,
            valid_until=NOW + timedelta(days=1),
        )
        redirect = RedirectHop(
            source_url="https://old-acme.com/", target_url="https://acme.com/",
            status=301, elapsed_ms=12,
        )
        decision = resolve_identity(resolver_input(
            anchor(),
            sites=[site(
                "acme.com", requested_host="old-acme.com", redirects=[redirect],
                outbound_hosts=["old-acme.com"],
                linkedin_urls=["https://linkedin.com/company/123"],
            )],
            linkedins=[linkedin(company_id="123", domain="old-acme.com")], prior=prior,
        ))
        self.assert_rule(decision, "ID-B3-VERIFIED-DOMAIN-MIGRATION", "strong")

    def test_prior_receipt_changed_linkedin_anchor_is_ambiguous(self) -> None:
        prior = PriorReceiptSummary(
            receipt_id=uuid4(), same_tenant=True, decision="resolved",
            positive_rule_ids=["ID-A2-RECIPROCAL-LINKEDIN-SLUG"],
            stable_linkedin_key="other-company", verified_domain="acme.com",
            identity_match_key="idmk1:" + "e" * 64,
            valid_until=NOW + timedelta(days=1),
        )
        decision = resolve_identity(resolver_input(
            anchor(), sites=[site()], linkedins=[linkedin()], prior=prior,
        ))
        self.assertEqual(decision.outcome, "ambiguous")
        self.assertIn("prior_receipt_linkedin_identity_changed", decision.conflicts)

    def test_a3_rejects_a_current_cross_domain_redirect(self) -> None:
        prior = PriorReceiptSummary(
            receipt_id=uuid4(), same_tenant=True, decision="resolved",
            positive_rule_ids=["ID-A2-RECIPROCAL-LINKEDIN-SLUG"],
            stable_linkedin_key="acme", verified_domain="acme.com",
            identity_match_key="idmk1:" + "f" * 64,
            valid_until=NOW + timedelta(days=1),
        )
        redirect = RedirectHop(
            source_url="https://acme.com/", target_url="https://other.example/",
            status=301, elapsed_ms=1,
        )
        decision = resolve_identity(resolver_input(
            anchor(),
            sites=[site("other.example", requested_host="acme.com", redirects=[redirect])],
            linkedins=[linkedin()], prior=prior,
        ))
        self.assertNotEqual(decision.outcome, "resolved")

    def test_b2_and_b3_require_exact_linkedin_on_both_sides(self) -> None:
        regional = resolve_identity(resolver_input(
            anchor(domain="acme.de"),
            sites=[
                site(
                    "acme.de", outbound_hosts=["acme.com"],
                    linkedin_urls=["https://linkedin.com/company/acme"],
                ),
                site("acme.com", outbound_hosts=["acme.de"]),
            ],
            linkedins=[linkedin(domain="acme.com")], primary_domains=["acme.com"],
        ))
        self.assertNotEqual(regional.outcome, "resolved")

        prior = PriorReceiptSummary(
            receipt_id=uuid4(), same_tenant=True, decision="resolved",
            positive_rule_ids=["ID-A1-LINKEDIN-ID-WEBSITE"],
            stable_linkedin_key="123", verified_domain="old-acme.com",
            identity_match_key="idmk1:" + "1" * 64,
            valid_until=NOW + timedelta(days=1),
        )
        redirect = RedirectHop(
            source_url="https://old-acme.com/", target_url="https://acme.com/",
            status=301, elapsed_ms=1,
        )
        migration = resolve_identity(resolver_input(
            anchor(),
            sites=[site(
                "acme.com", requested_host="old-acme.com", redirects=[redirect],
                outbound_hosts=["old-acme.com"],
            )],
            linkedins=[linkedin(company_id="123", domain="old-acme.com")], prior=prior,
        ))
        self.assertNotEqual(migration.outcome, "resolved")

    def test_tier_c_signals_never_accumulate_into_resolution(self) -> None:
        decision = resolve_identity(resolver_input(anchor(), sites=[site()], linkedins=[]))
        self.assertEqual(decision.outcome, "rejected")
        self.assertEqual(decision.identity_tier, "provisional")
        self.assertEqual(decision.reason_codes, ["insufficient_verified_identity_anchors"])
        self.assertIsNone(decision.canonical_identity.identity_match_key)

    def test_name_resemblance_unreachable_site_sibling_tld_and_redirect_alone_reject(self) -> None:
        cases = [
            resolver_input(anchor(), sites=[]),
            resolver_input(anchor(domain="acme.net"), sites=[site("acme.net")]),
            resolver_input(anchor(), sites=[site(requested_host="acme-old.com")]),
        ]
        for value in cases:
            with self.subTest(value=value.anchor_set.submitted_domain):
                self.assertNotEqual(resolve_identity(value).outcome, "resolved")

    def test_parked_aggregator_shared_and_unsafe_targets_are_deterministic_rejections(self) -> None:
        cases = [
            (anchor(), site(parked=True), "ID-N-PARKED-DOMAIN"),
            (anchor(), site(aggregator=True), "ID-N-AGGREGATOR-HOST"),
            (anchor(domain="tenant.blogspot.com"), site("tenant.blogspot.com"), "ID-N-SHARED-INFRASTRUCTURE"),
            (anchor(), site(unsafe=True), "ID-N-UNSAFE-TARGET"),
        ]
        for value, observation, rule in cases:
            with self.subTest(rule=rule):
                decision = resolve_identity(resolver_input(
                    value, sites=[observation], linkedins=[linkedin(domain=value.submitted_domain or "acme.com")]
                ))
                self.assertEqual(decision.outcome, "rejected")
                self.assertIn(rule, decision.negative_rule_ids)

    def test_transient_required_source_failure_is_unavailable_not_rejected(self) -> None:
        decision = resolve_identity(resolver_input(
            anchor(), sites=[site(transient=True)], linkedins=[linkedin()]
        ))
        self.assertEqual(decision.outcome, "unavailable")
        self.assertEqual(decision.reason_codes, ["required_first_party_source_unavailable"])

    def test_conflicting_stable_ids_country_or_relationship_fail_closed_as_ambiguous(self) -> None:
        cases = [
            resolver_input(anchor(), sites=[site()], linkedins=[
                linkedin(company_id="111"), linkedin(company_id="222"),
            ]),
            resolver_input(anchor(), sites=[site()], linkedins=[linkedin()], country_conflict=True),
            resolver_input(anchor(), sites=[site()], linkedins=[linkedin()], relationship_conflict=True),
        ]
        for value in cases:
            decision = resolve_identity(value)
            self.assertEqual(decision.outcome, "ambiguous")
            self.assertTrue(decision.conflicts)

    def test_cross_tenant_expired_or_superseded_prior_receipts_never_resolve(self) -> None:
        for overrides in (
            {"same_tenant": False},
            {"valid_until": NOW - timedelta(seconds=1)},
            {"superseded": True},
        ):
            fields = {
                "receipt_id": uuid4(), "same_tenant": True, "decision": "resolved",
                "positive_rule_ids": ["ID-A1-LINKEDIN-ID-WEBSITE"],
                "stable_linkedin_key": "123", "verified_domain": "acme.com",
                "identity_match_key": "idmk1:" + "d" * 64,
                "valid_until": NOW + timedelta(days=1), "superseded": False,
            }
            fields.update(overrides)
            decision = resolve_identity(resolver_input(
                anchor(), sites=[site()], prior=PriorReceiptSummary(**fields)
            ))
            self.assertNotEqual(decision.outcome, "resolved")

    def test_anchor_hash_mismatch_is_an_integrity_error(self) -> None:
        value = resolver_input(anchor(), sites=[site()])
        tampered = value.model_copy(update={"anchor_set_hash": "f" * 64})
        with self.assertRaisesRegex(ValueError, "anchor_set_hash"):
            resolve_identity(tampered)

    def test_snapshot_digest_is_stable_and_excludes_only_its_digest_field(self) -> None:
        decision = resolve_identity(resolver_input(
            anchor(), sites=[site(linkedin_urls=["https://linkedin.com/company/acme"])],
            linkedins=[linkedin()],
        ))
        identity = decision.canonical_identity
        self.assertEqual(
            identity.identity_snapshot_hash,
            canonical_sha256(identity.model_dump(mode="python", exclude={"identity_snapshot_hash"})),
        )

    def test_identity_is_a_conjunctive_gate_and_never_changes_claim_score(self) -> None:
        exact = resolve_identity(resolver_input(
            anchor(), sites=[site(linkedin_urls=["https://linkedin.com/company/acme"])],
            linkedins=[linkedin()],
        )).canonical_identity
        rejected_claim = VerificationDecision(
            outcome="rejected", reason_code="unsupported_claim",
            reason_message="The page does not support the claim", verifier_score=0,
        )
        comparison = compare_identity_bound_claim(
            rejected_claim, exact, ["https://acme.com/news"], identity_receipt_id=str(uuid4())
        )
        self.assertEqual(comparison.identity_bound_outcome, "rejected")
        self.assertTrue(comparison.claim_score_unchanged)
        self.assertEqual(comparison.evidence_bindings[0].binding, "same_entity")

        ambiguous = resolve_identity(resolver_input(
            anchor(), sites=[site()], linkedins=[linkedin()], country_conflict=True,
        )).canonical_identity
        accepted_claim = VerificationDecision(
            outcome="accepted", reason_code="verified", reason_message="supported",
            verifier_score=99, verified_payload={"company_name": "Acme", "domain": "acme.com"},
        )
        blocked = compare_identity_bound_claim(
            accepted_claim, ambiguous, ["https://acme.com/news"], identity_receipt_id=str(uuid4())
        )
        self.assertEqual(blocked.identity_bound_outcome, "rejected")
        self.assertTrue(blocked.claim_score_unchanged)


if __name__ == "__main__":
    unittest.main()
