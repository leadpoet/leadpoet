"""Delay fixes preserve the company-only runner-to-reward transition."""

import pytest

from tests.lab_arena import proxy_model_capacity401_e2e_test as prior
from tests.lab_arena.lab_arena_pg_harness import current_service_database


@pytest.fixture()
def database():
    # Current service dependencies are required for the full HTTP lifecycle.
    # Historical406/407 migration and security tests keep their pinned schemas.
    yield from current_service_database()


def test_delay_fixes_preserve_baseline_miners_costs_promotion_and_weights(
    database, tmp_path,
):
    prior.test_company_only_baseline_miners_costs_promotion_and_publication(
        database, True, tmp_path,
    )
