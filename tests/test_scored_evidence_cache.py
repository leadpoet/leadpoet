from research_lab.eval.scored_evidence_cache import scoring_cache_key


COMPANIES = [{"company_name": "Acme", "intent_signals": []}]


def _icp() -> dict:
    return {
        "industry": "Software",
        "intent_signals": ["Funding raised: round type series a"],
        "intent_signal_evidence_types": ["FUNDING"],
        "intent_signal_max_age_days": [90],
        "intent_category": "FUNDING",
        "intent_max_age_days": 90,
        "_signal_profile_identity": {
            "signal_catalog_sha256": "b" * 64,
            "primary_definition_sha256": "a" * 64,
        },
    }


def test_scoring_cache_key_is_stable_under_mapping_key_order() -> None:
    icp = _icp()
    reordered = dict(reversed(list(icp.items())))
    assert scoring_cache_key(icp, COMPANIES, False) == scoring_cache_key(
        reordered, COMPANIES, False
    )


def test_scoring_cache_key_binds_profile_category_and_freshness() -> None:
    baseline = scoring_cache_key(_icp(), COMPANIES, False)
    variants = []
    profile = _icp()
    profile["_signal_profile_identity"] = {
        **profile["_signal_profile_identity"],
        "primary_definition_sha256": "c" * 64,
    }
    variants.append(profile)
    category = _icp()
    category["intent_category"] = "HIRING"
    variants.append(category)
    freshness = _icp()
    freshness["intent_max_age_days"] = 120
    variants.append(freshness)
    assert all(
        scoring_cache_key(variant, COMPANIES, False) != baseline
        for variant in variants
    )


def test_scoring_cache_key_ignores_monitor_only_profile_changes() -> None:
    baseline = _icp()
    with_monitor_change = {
        **baseline,
        "signal_profile": {
            "bindings": [
                {"role": "monitor_only", "definition": {"technology": "Databricks"}}
            ]
        },
    }
    assert scoring_cache_key(baseline, COMPANIES, False) == scoring_cache_key(
        with_monitor_change, COMPANIES, False
    )
