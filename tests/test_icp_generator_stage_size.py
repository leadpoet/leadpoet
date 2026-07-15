"""Stage <-> employee-size coherence in generated ICPs.

Stage and size were previously independent random draws, which produced
impossible ICPs (a Seed company with 5,001-10,000 employees) that zero out at
the size gate regardless of sourcing quality. A stage-pinned generated ICP must
now carry the FULL band list allowed for that stage, and nothing outside it.
"""

from gateway.tasks.icp_generator import (
    COMPANY_STAGES,
    STAGE_EMPLOYEE_BUCKETS,
    generate_single_icp,
)
from research_lab.employee_buckets import GENERATED_EMPLOYEE_BUCKETS


def test_every_stage_has_coherent_buckets():
    for stage in COMPANY_STAGES:
        buckets = STAGE_EMPLOYEE_BUCKETS[stage]
        assert buckets, stage
        for bucket in buckets:
            assert bucket in GENERATED_EMPLOYEE_BUCKETS, (stage, bucket)


def test_generated_icp_carries_full_stage_band_list():
    for seed in range(60):
        icp = generate_single_icp(f"t-{seed}", "Software", seed=seed)
        stage = icp["company_stage"]
        expected = set(STAGE_EMPLOYEE_BUCKETS[stage])
        got = set(icp["employee_count"])
        # The canonical ICP's employee_count is the allowed band list: exactly
        # the stage's full coherent list — no band outside it, none missing.
        assert got == expected, (stage, sorted(got), sorted(expected))


def test_generated_prompt_displays_stage_bands():
    icp = generate_single_icp("t-prompt", "Software", seed=7)
    # At least one of the stage's bands should appear in the human prompt after
    # canonicalization rewrites the size text to the allowed-band display.
    assert any(bucket in icp["prompt"] for bucket in icp["employee_count"])


def test_all_stage_size_pairs_possible_are_coherent():
    seen_stages = set()
    for seed in range(300):
        icp = generate_single_icp(f"s-{seed}", "Software", seed=seed)
        seen_stages.add(icp["company_stage"])
        for bucket in icp["employee_count"]:
            assert bucket in STAGE_EMPLOYEE_BUCKETS[icp["company_stage"]]
    assert seen_stages == set(COMPANY_STAGES)  # every stage still generated


def test_company_goal_allocation_invariants():
    import random as _random
    from gateway.tasks.icp_generator import (
        COMPANY_GOAL_MAX,
        COMPANY_GOAL_MIN,
        allocate_company_goals,
    )
    for seed in range(50):
        _random.seed(seed)
        goals = allocate_company_goals(20)
        assert len(goals) == 20
        assert sum(goals) == 100  # 5-per-ICP average -> benchmark volume unchanged
        assert all(COMPANY_GOAL_MIN <= g <= COMPANY_GOAL_MAX for g in goals)
    # reproducible under the same seed
    _random.seed(7); a = allocate_company_goals(20)
    _random.seed(7); b = allocate_company_goals(20)
    assert a == b


def test_icp_set_pins_uniform_goal_of_five(monkeypatch):
    monkeypatch.setenv("RESEARCH_LAB_ICP_EXCLUSIONS_ENABLED", "0")  # no live LLM in tests
    from gateway.tasks.icp_generator import generate_icp_set
    icps, _dist, _h = generate_icp_set(20260715, base_seed=42)
    goals = [icp["max_companies"] for icp in icps]
    # Uniform 5 per ICP for now: benchmark volume/cost unchanged (100 total).
    assert goals == [5] * len(icps)
    assert sum(goals) == 5 * len(icps)
    # reproducible: same base_seed -> same hash
    _icps1, _d1, h1 = generate_icp_set(20260715, base_seed=42)
    _icps2, _d2, h2 = generate_icp_set(20260715, base_seed=42)
    assert h1 == h2


def test_apply_generated_icp_contract_openrouter_path():
    """The production OpenRouter path builds ICP dicts directly; the set-level
    contract must attach max_companies, widen bands to the stage's full range
    (union, never replacing the realism band), and leave unknown stages alone."""
    from gateway.tasks.icp_generator import (
        apply_generated_icp_contract,
        COMPANY_GOAL_AVERAGE,
        STAGE_EMPLOYEE_BUCKETS,
    )
    icps = [
        {"icp_id": "a", "company_stage": "Series A", "employee_count": ["51-200"]},
        {"icp_id": "b", "company_stage": "Public", "employee_count": "5,001-10,000"},
        {"icp_id": "c", "company_stage": "Bootstrapped", "employee_count": ["11-50"]},
        {"icp_id": "d", "company_stage": "Series A", "employee_count": ["201-500"]},
    ]
    apply_generated_icp_contract(icps)
    assert all(i["max_companies"] == COMPANY_GOAL_AVERAGE for i in icps)
    assert set(icps[0]["employee_count"]) == set(STAGE_EMPLOYEE_BUCKETS["Series A"])
    assert set(icps[1]["employee_count"]) == set(STAGE_EMPLOYEE_BUCKETS["Public"]) | {"5,001-10,000"}
    assert icps[2]["employee_count"] == ["11-50"]  # unknown stage untouched
    assert "201-500" in icps[3]["employee_count"]  # realism band preserved (union)


def test_generated_icp_contract_matches_llm_stage_vocabulary():
    """Every stage the LLM is instructed to use must be in the coherence dict,
    or stage widening silently no-ops in production."""
    from gateway.tasks.icp_generator import STAGE_EMPLOYEE_BUCKETS
    llm_stages = {"Seed", "Series A", "Series B", "Series C+", "Private Equity", "Public"}
    assert llm_stages <= set(STAGE_EMPLOYEE_BUCKETS), (
        "LLM stage vocabulary drifted from STAGE_EMPLOYEE_BUCKETS")
