"""Delay fixes preserve the company-only runner-to-reward transition."""

from pathlib import Path

import pytest

from tests.lab_arena import proxy_model_capacity401_e2e_test as prior


database = prior.database
migrated399 = prior.migrated399


@pytest.fixture(scope="module")
def migrated(database, migrated399):
    prior.migrated.__wrapped__(database, migrated399)
    psycopg, dsn = database
    with psycopg.connect(**dsn) as connection:
        connection.autocommit = True
        with connection.cursor() as cursor:
            for name in (
                "406-lab-arena-round-host-fault-quarantine.sql",
                "407-lab-arena-cost-run-lookup.sql",
                "408-lab-arena-cost-run-index.sql",
                "409-lab-arena-publication-cost-document-reuse.sql",
                "410-lab-arena-publication-transition-timeout.sql",
                "411-lab-arena-settlement-success-json-once.sql",
            ):
                cursor.execute((Path(__file__).parents[2] / "scripts" / name).read_text())


def test_delay_fixes_preserve_baseline_miners_costs_promotion_and_weights(
    database, migrated, tmp_path,
):
    prior.test_company_only_baseline_miners_costs_promotion_and_publication(
        database, migrated, tmp_path,
    )
