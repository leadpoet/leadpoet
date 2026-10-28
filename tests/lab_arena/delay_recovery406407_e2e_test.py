"""Delay fixes preserve the company-only runner-to-reward transition."""

from pathlib import Path

import pytest

from tests.lab_arena import proxy_model_capacity401_e2e_test as prior
from tests.lab_arena import proxy_model_capacity401_postgres_test as frozen401
from tests.lab_arena.historical_service_harness import HistoricalStoreHarnessMixin, freeze_historical_intake_dates


database = frozen401.database
migrated399 = frozen401.migrated399


@pytest.fixture(scope="module")
def migrated(database, migrated399):
    frozen401.migrated.__wrapped__(database, migrated399)
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
    database, migrated, tmp_path, monkeypatch,
):
    monkeypatch.setattr(prior, "QualityHarness", _FrozenQualityHarness)
    prior.test_company_only_baseline_miners_costs_promotion_and_publication(
        database, migrated, tmp_path,
    )


class _FrozenQualityHarness(HistoricalStoreHarnessMixin, prior.QualityHarness):
    def build_service(self):
        service = super().build_service()
        commit = service.commit_benchmark

        def commit_historical(round_id):
            freeze_historical_intake_dates(self, "2026-10-07")
            return commit(round_id)

        service.commit_benchmark = commit_historical
        return service
