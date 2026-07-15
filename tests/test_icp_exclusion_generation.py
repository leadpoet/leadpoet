"""Generator-side lab exclusion lists: real matching companies per ICP,
fail-open everywhere, hash-covered in the set."""

import os
from unittest import mock

import gateway.tasks.icp_generator as gen


def _sonar_response(payload):
    resp = mock.Mock()
    resp.status_code = 200
    resp.json.return_value = {"choices": [{"message": {"content": payload}}]}
    return resp


def test_entries_prefer_domain_and_validate():
    with mock.patch.object(gen, "OPENROUTER_API_KEY", "k"), \
         mock.patch.object(gen.httpx, "post", return_value=_sonar_response(
             '[{"name": "Acme Inc", "domain": "https://www.acme.com/x"},'
             ' {"name": "NoDomain Co", "domain": "not a domain"}]')):
        entries = gen.generate_exclusions_for_icp({"industry": "Software"}, count=2)
    assert entries == ["acme.com", "NoDomain Co"]


def test_fail_open_on_provider_error_and_bad_json():
    with mock.patch.object(gen, "OPENROUTER_API_KEY", "k"), \
         mock.patch.object(gen.httpx, "post", side_effect=RuntimeError("boom")):
        assert gen.generate_exclusions_for_icp({}, count=1) == []
    bad = _sonar_response("no json here")
    with mock.patch.object(gen, "OPENROUTER_API_KEY", "k"), \
         mock.patch.object(gen.httpx, "post", return_value=bad):
        assert gen.generate_exclusions_for_icp({}, count=1) == []


def test_no_key_skips_without_call():
    with mock.patch.object(gen, "OPENROUTER_API_KEY", ""):
        assert gen.generate_exclusions_for_icp({}, count=1) == []


def test_icp_set_carries_exclusions_and_hash_covers_them():
    with mock.patch.object(gen, "generate_exclusions_for_icp",
                           side_effect=lambda icp, count=None, timeout_s=45.0: ["excluded.example"]):
        icps, _d, h1 = gen.generate_icp_set(20260716, base_seed=9)
    assert all(icp["excluded_companies"] == ["excluded.example"] for icp in icps)
    with mock.patch.object(gen, "generate_exclusions_for_icp",
                           side_effect=lambda icp, count=None, timeout_s=45.0: []):
        _i2, _d2, h2 = gen.generate_icp_set(20260716, base_seed=9)
    assert h1 != h2  # exclusions are part of the audited set hash


def test_disabled_env_yields_empty_lists():
    with mock.patch.dict(os.environ, {"RESEARCH_LAB_ICP_EXCLUSIONS_ENABLED": "0"}):
        icps, _d, _h = gen.generate_icp_set(20260717, base_seed=3)
    assert all(icp["excluded_companies"] == [] for icp in icps)
