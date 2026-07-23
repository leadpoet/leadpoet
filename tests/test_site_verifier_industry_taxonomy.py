from __future__ import annotations

import importlib.util
import re
import unittest
from pathlib import Path

from leadpoet_verifier.industry_taxonomy import (
    CONTEXTUAL_BROAD_CONCEPTS,
    INDUSTRY_CONCEPT_PATTERNS,
    LEADPOET_PARENT_INDUSTRIES,
    LEADPOET_SUBINDUSTRY_PARENTS,
    industry_concepts,
    industry_tokens,
    leadpoet_taxonomy_match,
    normalized_industry_text,
    requested_industry_concepts,
)


def repository_industry_taxonomy():
    # Lab adaptation: the authoritative taxonomy source in this repository is
    # gateway/utils/industry_taxonomy.py (the AGENTS.md authority file). This
    # consistency check proves the ported verifier JSON snapshot matches the
    # lab's own canonical taxonomy.
    from gateway.utils.industry_taxonomy import INDUSTRY_TAXONOMY

    return INDUSTRY_TAXONOMY


class IndustryTaxonomyTests(unittest.TestCase):
    def assert_canonical_match(
        self,
        requested: str,
        candidate_industry: str,
        candidate_subindustry: str,
        expected_concept: str,
    ) -> None:
        requested_concepts, _suppressed = requested_industry_concepts(requested)
        candidate_concepts = industry_concepts(candidate_industry, candidate_subindustry)
        self.assertIn(expected_concept, requested_concepts)
        self.assertIn(expected_concept, candidate_concepts)
        self.assertTrue(requested_concepts & candidate_concepts)

    def test_every_canonical_pattern_compiles(self):
        for concept, patterns in INDUSTRY_CONCEPT_PATTERNS.items():
            with self.subTest(concept=concept):
                self.assertTrue(patterns)
                for pattern in patterns:
                    re.compile(pattern)

    def test_contextual_broad_concepts_have_explicit_registry_entries(self):
        self.assertTrue(CONTEXTUAL_BROAD_CONCEPTS)
        self.assertLessEqual(CONTEXTUAL_BROAD_CONCEPTS, INDUSTRY_CONCEPT_PATTERNS.keys())

    def test_embedded_taxonomy_exactly_matches_repository_source(self):
        source = repository_industry_taxonomy()
        expected_parents = {
            normalized_industry_text(parent): parent
            for entry in source.values()
            for parent in entry["industries"]
        }
        expected_subindustries = {
            normalized_industry_text(subindustry): (
                subindustry,
                frozenset(
                    normalized_industry_text(parent)
                    for parent in entry["industries"]
                ),
            )
            for subindustry, entry in source.items()
        }

        # Counts derive from the lab authority file (848 sub-industries at
        # port time; the site snapshot carried 725 — regenerate the JSON via
        # scripts/generate_verifier_industry_taxonomy.py when authority moves).
        self.assertEqual(len(expected_parents), 50)
        self.assertEqual(len(expected_subindustries), len(source))
        self.assertEqual(LEADPOET_PARENT_INDUSTRIES, expected_parents)
        self.assertEqual(LEADPOET_SUBINDUSTRY_PARENTS, expected_subindustries)

    def test_exact_taxonomy_rejects_cross_parent_aliases_and_inconsistent_labels(self):
        cases = [
            ("Payments", "Financial Services", "Accounting", False),
            ("Gaming", "Media and Entertainment", "Advice", False),
            ("Apps", "Software", "3D Technology", False),
            ("Hardware", "Software", "3D Technology", True),
            ("Software", "Hardware", "3D Technology", True),
            ("Software", "Hardware", "Software", False),
            ("Health Care", "Financial Services", "Health Insurance", False),
            ("Professional Services", "Financial Services", "Accounting", True),
            ("Financial Services", "Banking", "Treasury Operations", True),
            ("Payments", "Banking", "Treasury Operations", False),
            ("Software", "Enterprise Applications", "Revenue Enablement", True),
            ("Apps", "Enterprise Applications", "Revenue Enablement", True),
        ]
        for requested, industry, subindustry, expected in cases:
            with self.subTest(requested=requested, subindustry=subindustry):
                decision, detail = leadpoet_taxonomy_match(
                    requested,
                    industry,
                    subindustry,
                )
                self.assertIs(decision, expected, detail)

        decision, detail = leadpoet_taxonomy_match(
            "Software",
            "Software Development",
            "Revenue intelligence platform",
        )
        self.assertIsNone(decision, detail)

    def test_semantic_alias_matrix_matches_common_provider_taxonomies(self):
        cases = [
            ("Software", "Software Development", "Sales enablement software", "software"),
            (
                "Information Technology",
                "Technology, Information and Internet",
                "Enterprise systems",
                "information_technology",
            ),
            ("Manufacturing", "Industrial Machinery Manufacturing", "Factory equipment", "manufacturing"),
            ("AI companies", "Software Development", "AI platform", "artificial_intelligence"),
            (
                "Machine-learning vendors",
                "Software Development",
                "Machine learning platform",
                "artificial_intelligence",
            ),
            ("Biotech", "Biotechnology Research", "Drug discovery", "biotechnology_pharmaceuticals"),
            ("Pharma", "Pharmaceutical Manufacturing", "Therapeutics", "biotechnology_pharmaceuticals"),
            ("Education technology", "E-Learning Providers", "Learning platform", "education"),
            ("HR software", "Software Development", "Human resources management software", "human_resources"),
            ("Insurtech", "Insurance", "Digital insurance platform", "insurance"),
            ("Hospitality", "Hotels and Resorts", "Boutique hotels", "hospitality_travel"),
            ("Agriculture", "Farming", "Crop production", "agriculture"),
            ("AgTech", "Software Development", "Farm technology", "agriculture"),
            ("Professional services", "Business Consulting and Services", "Advisory", "professional_services"),
            ("Marketing agencies", "Advertising Services", "Digital advertising agency", "marketing_advertising"),
            ("Restaurants", "Food and Beverage Services", "Restaurant group", "restaurants_food_service"),
            ("Nonprofits", "Non-profit Organizations", "Charitable foundation", "nonprofit_social_impact"),
            ("Construction", "Construction", "Commercial construction", "construction"),
            ("PropTech", "Real Estate", "Property management technology", "real_estate"),
            ("Legal tech", "Legal Services", "Legal technology", "legal_services"),
            ("Retail", "Retail", "Specialty retail", "retail_ecommerce"),
            ("E-commerce", "Internet Marketplace Platforms", "E-commerce marketplace", "retail_ecommerce"),
            ("Telecom", "Telecommunications", "Wireless services", "telecommunications"),
            ("Web3", "Software Development", "Blockchain infrastructure", "blockchain_crypto"),
            ("Data analytics", "Software Development", "Business intelligence analytics", "data_analytics"),
            ("Consumer packaged goods", "Manufacturing", "CPG household products", "consumer_goods"),
            ("Automotive technology", "Motor Vehicle Manufacturing", "Electric vehicles", "automotive_mobility"),
            (
                "Government technology",
                "Government Administration",
                "Public sector technology",
                "government_public_sector",
            ),
            ("Healthcare technology", "Hospitals and Health Care", "Digital health", "healthcare"),
            ("Fintech", "Financial Services", "Payments infrastructure", "financial_infrastructure"),
            ("Food and beverage", "Food Production", "Beverage manufacturing", "food_beverage"),
            ("Sales technology", "Software Development", "Sales enablement platform", "sales_revenue"),
            (
                "Aerospace",
                "Aviation and Aerospace Component Manufacturing",
                "Aircraft components",
                "aerospace",
            ),
            (
                "Industrial robotics",
                "Automation Machinery Manufacturing",
                "Collaborative robots",
                "industrial_automation",
            ),
            (
                "Industrial manufacturing",
                "Industrial Machinery Manufacturing",
                "Industrial equipment",
                "industrial_manufacturing",
            ),
            ("Logistics", "Transportation, Logistics, Supply Chain and Storage", "Freight", "logistics"),
            ("Semiconductor equipment", "Semiconductor Manufacturing", "Wafer metrology", "semiconductor_equipment"),
            ("Cybersecurity", "Computer and Network Security", "Threat detection", "cybersecurity"),
            ("Energy infrastructure", "Electric Power Generation", "Grid modernization", "energy_infrastructure"),
        ]
        for requested, industry, subindustry, expected in cases:
            with self.subTest(requested=requested, industry=industry):
                self.assert_canonical_match(requested, industry, subindustry, expected)

    def test_cross_industry_pairs_do_not_share_canonical_concepts(self):
        cases = [
            ("Healthcare", "Retail Apparel and Fashion", "E-commerce"),
            ("Biotech", "Hospitals and Health Care", "Primary care clinic"),
            ("Agriculture", "Food and Beverage Retail", "Grocery stores"),
            ("Education technology", "Technology, Information and Internet", "Social media platform"),
            ("HR software", "Software Development", "Project management software"),
            ("Insurtech", "Financial Services", "Consumer banking"),
            ("Hospitality", "Transportation and Logistics", "Freight forwarding"),
            ("Professional services", "Retail", "Consumer marketplace"),
            ("Marketing agencies", "Business Consulting and Services", "Operations advisory"),
            ("Restaurants", "Food Production", "Packaged food manufacturing"),
            ("Nonprofits", "Government Administration", "Public agency"),
            ("PropTech", "Construction", "Industrial contractor"),
            ("Legal tech", "Human Resources Services", "Recruiting"),
            ("Telecom", "Computer Hardware Manufacturing", "Laptops"),
            ("Web3", "Financial Services", "Traditional commercial banking"),
            ("Data analytics", "Market Research", "Qualitative interviews"),
            ("Automotive", "Transportation and Logistics", "Freight brokerage"),
            ("Semiconductor equipment", "Industrial Machinery Manufacturing", "Food processing equipment"),
            ("Cybersecurity", "Security and Investigations", "Physical security services"),
            ("Energy infrastructure", "Financial Services", "Energy trading desk"),
        ]
        for requested, industry, subindustry in cases:
            with self.subTest(requested=requested, industry=industry):
                requested_concepts, _suppressed = requested_industry_concepts(requested)
                candidate_concepts = industry_concepts(industry, subindustry)
                self.assertFalse(requested_concepts & candidate_concepts)

    def test_specific_requests_suppress_broad_modifier_concepts(self):
        cases = [
            ("climate software", "energy_infrastructure", "software"),
            ("education information technology", "education", "information_technology"),
            ("pharmaceutical manufacturing", "biotechnology_pharmaceuticals", "manufacturing"),
            ("AI hardware", "artificial_intelligence", "hardware_electronics"),
            ("HR software", "human_resources", "software"),
        ]
        for requested, expected, suppressed in cases:
            with self.subTest(requested=requested):
                active, removed = requested_industry_concepts(requested)
                self.assertIn(expected, active)
                self.assertNotIn(suppressed, active)
                self.assertIn(suppressed, removed)

    def test_explicit_broad_options_remain_active_in_multi_industry_request(self):
        active, suppressed = requested_industry_concepts(
            "Software, manufacturing, or healthcare"
        )
        self.assertIn("software", active)
        self.assertIn("manufacturing", active)
        self.assertIn("healthcare", active)
        self.assertEqual(suppressed, set())

    def test_generic_modifiers_cannot_cross_specific_industries(self):
        cases = [
            (
                "B2B climate, grid, and energy infrastructure software",
                "Software Development",
                "Human resources management platform",
            ),
            (
                "B2B cybersecurity software and cloud-security platforms",
                "Software Development",
                "Sales enablement software",
            ),
            (
                "Semiconductor manufacturing",
                "Food Production",
                "Food manufacturing",
            ),
            (
                "Education technology",
                "Technology, Information and Internet",
                "Enterprise collaboration platform",
            ),
        ]
        for requested, industry, subindustry in cases:
            with self.subTest(requested=requested):
                active, _suppressed = requested_industry_concepts(requested)
                candidate = industry_concepts(industry, subindustry)
                self.assertFalse(active & candidate)

    def test_security_and_automation_terms_remain_domain_specific(self):
        self.assertNotIn(
            "cybersecurity",
            industry_concepts("Security and Investigations", "Physical security operations"),
        )
        self.assertNotIn(
            "industrial_automation",
            industry_concepts("Software Development", "Marketing automation"),
        )
        self.assertIn(
            "industrial_automation",
            industry_concepts("Automation Machinery Manufacturing", "Factory automation"),
        )

    def test_software_supply_chain_security_is_not_logistics(self):
        concepts = industry_concepts("Software Development", "Software supply chain security")
        self.assertIn("cybersecurity", concepts)
        self.assertNotIn("logistics", concepts)

    def test_unknown_exact_non_generic_industry_can_use_token_fallback(self):
        self.assertEqual(industry_concepts("Maritime technology"), set())
        self.assertIn("maritime", industry_tokens("Maritime technology"))
        self.assertIn("maritime", industry_tokens("Maritime Services"))
