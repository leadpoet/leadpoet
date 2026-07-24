from __future__ import annotations

import unittest

from qualification.scoring.lead_scorer import (
    _normalize_icp_employee_buckets,
    _normalize_linkedin_employee_bucket,
)
from research_lab.employee_buckets import LINKEDIN_EMPLOYEE_BUCKETS


class EmployeeSizeBucketTests(unittest.TestCase):
    def test_every_linkedin_bucket_round_trips_exactly(self):
        for bucket in LINKEDIN_EMPLOYEE_BUCKETS:
            with self.subTest(bucket=bucket):
                self.assertEqual(_normalize_linkedin_employee_bucket(bucket), bucket)
                self.assertEqual(_normalize_icp_employee_buckets([bucket]), ({bucket}, True))

    def test_comma_ranges_are_not_split_as_lists(self):
        self.assertEqual(
            _normalize_icp_employee_buckets("501-1,000 | 1,001-5,000"),
            ({"501-1,000", "1,001-5,000"}, True),
        )

    def test_mixed_legacy_and_canonical_default_bands_are_deduplicated(self):
        production_default = [
            "11-50", "51-200", "201-500", "501-1000", "1001-5000",
            "5001-10000", "10000+", "1-10", "501-1,000",
            "1,001-5,000", "5,001-10,000", "10,001+",
        ]
        self.assertEqual(
            _normalize_icp_employee_buckets(production_default),
            ({
                "2-10", "11-50", "51-200", "201-500", "501-1,000",
                "1,001-5,000", "5,001-10,000", "10,001+",
            }, True),
        )

    def test_known_legacy_bands_map_to_exact_linkedin_buckets(self):
        self.assertEqual(
            _normalize_icp_employee_buckets(["1-10", "501-1000", "10000+"]),
            ({"2-10", "501-1,000", "10,001+"}, True),
        )

    def test_unknown_and_malformed_icp_sizes_fail_closed(self):
        for value in (None, "", "any", "about 500", "500ish", "eleven to fifty"):
            with self.subTest(value=value):
                self.assertEqual(_normalize_icp_employee_buckets(value), (set(), False))


if __name__ == "__main__":
    unittest.main()
