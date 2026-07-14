"""Scorer-side employee-band gate parsing.

The stored ICP carries allowed bands as a list; the scoring payload joins them
with "|"; LinkedIn band labels contain thousands separators ("1,001-5,000").
The parser previously split on commas too, shredding every large band into
garbage — the set came back empty and the size gate silently disabled for
exactly the ICPs carrying large bands.
"""

from qualification.scoring.lead_scorer import _normalize_icp_employee_buckets


def test_piped_small_bands():
    assert _normalize_icp_employee_buckets("11-50|51-200") == {"11-50", "51-200"}


def test_piped_large_bands_with_thousands_separators():
    got = _normalize_icp_employee_buckets("1,001-5,000|5,001-10,000|10,001+")
    assert got == {"1,001-5,000", "5,001-10,000", "10,001+"}


def test_list_input_accepted_directly():
    assert _normalize_icp_employee_buckets(["11-50", "51-200"]) == {"11-50", "51-200"}
    got = _normalize_icp_employee_buckets(["1,001-5,000", "10,001+"])
    assert got == {"1,001-5,000", "10,001+"}


def test_plain_number_forms_canonicalized():
    got = _normalize_icp_employee_buckets("1001-5000|5001-10000")
    assert got == {"1,001-5,000", "5,001-10,000"}


def test_unconstrained_forms_stay_empty():
    assert _normalize_icp_employee_buckets("any") == set()
    assert _normalize_icp_employee_buckets("") == set()
    assert _normalize_icp_employee_buckets(None) == set()
