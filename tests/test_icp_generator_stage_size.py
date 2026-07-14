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
