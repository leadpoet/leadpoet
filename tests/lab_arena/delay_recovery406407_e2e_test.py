"""Delay fixes preserve the company-only runner-to-reward transition."""

from pathlib import Path

import pytest

from tests.lab_arena import proxy_model_capacity401_e2e_test as prior
from tests.lab_arena.proxy_model_capacity401_e2e_test import complete_application_database


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
    complete_application_database, tmp_path,
):
    # Exact delay-migration replay fixtures remain available above. This flow
    # invokes the current service and its complete guarded schema.
    prior._exercise_company_only_round(
        complete_application_database, True, tmp_path,
    )
