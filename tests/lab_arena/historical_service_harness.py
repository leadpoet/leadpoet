"""Client projections for fixtures that deliberately pin pre-474 SQL.

Historical migration tests must retain their original function bodies. They do
not have the later host-retry attempt counter, generation floor, or compact
public read projections. Use this client only in explicitly historical harnesses;
current-service tests must use ArenaStore and current_service_database instead.
"""

import json
from copy import deepcopy
from datetime import date, timedelta

from lab_arena import contracts

from lab_arena.store import (ArenaStore, ArenaStoreError, PsycopgTransport,
                             PUBLIC_RESULTS_CONFIGURATION_FIELDS,
                             CODE_REVIEW_SCHEMA_VERSION, _require_mapping)


class HistoricalArenaStore(ArenaStore):
    def code_review_schema(self):
        """Require the exact bounded-review capability from pre-586 fixtures."""

        result = _require_mapping(
            self._transport.rpc("lab_arena_code_review_schema_v1", {}),
            "code_review_schema",
        )
        if (
            result.get("schema_version") != CODE_REVIEW_SCHEMA_VERSION
            or result.get("version") != 207
            or result.get("claim_ttl_seconds") != 600
            or result.get("retry_backoff_seconds") != 60
            or result.get("max_attempts") != 3
        ):
            raise ArenaStoreError("code review schema mismatch")
        retry_policy = result.get("retry_policy")
        if (
            retry_policy != "bounded_transient_v1"
            or result.get("max_transient_attempts") != 6
            or result.get("transient_retry_backoff_seconds")
                != [60, 120, 240, 480, 900]
            or result.get("legacy_max_attempts") != 3
            or result.get("retry_window")
                != "replacement_freeze_to_benchmark_deadline"
        ):
            raise ArenaStoreError("code review schema mismatch")
        return result

    def list_runs(self, round_id, *, columns="*", **kwargs):
        if columns == "*,execution_attempt:lab_arena_execution_attempt_count":
            # Pre-474 SQL counts every stored attempt. Host-retry exclusions
            # did not exist; the persisted attempt is the historical authority.
            rows = super().list_runs(round_id, columns="*", **kwargs)
            return [{**row, "execution_attempt": row["attempt"]} for row in rows]
        columns = ",".join(column for column in columns.split(",")
                           if column not in (
                               "scoring_generation_floor",
                               "native_failure_reason:lab_arena_public_execution_failure_reason_v1",
                           ))
        return super().list_runs(round_id, columns=columns, **kwargs)

    def list_rounds(self, *, columns="*", **kwargs):
        # Compact competition publication was added after these pinned schemas.
        columns = columns.replace("publication_doc:lab_arena_competition_publication_v1", "publication_doc")
        columns = columns.replace("configuration_doc:lab_arena_public_configuration_v1", "configuration_doc")
        columns = ",".join(column for column in columns.split(",")
                           if column != "scoring_generation_floor")
        return super().list_rounds(columns=columns, **kwargs)

    def recover_baseline(self, round_id, work_items=()):
        # Recovery did not exist on these schemas. A healthy historical baseline
        # requires no recovery; unexpected missing work must fail this fixture.
        assert not work_items, "historical schema has no baseline recovery API"
        row = self.get_round(round_id)
        baseline = next(p["submission_id"] for p in row["participants"] if p["is_king"])
        runs = self.list_runs(round_id, stage=1, kind="execute", submission_id=baseline)
        accepted = {int(run["icp_position"]) for run in runs
                    if run["status"] == "accepted" and run["per_icp_score"] is not None}
        assert accepted == set(range(contracts.benchmark_icp_count(row["configuration_doc"])))
        return {"status": "not_applicable"}

    def scoring_policy_history(self, round_id):
        # These schemas predate model-policy supersession and have no revisions.
        return {"round_id": round_id, "revisions": []}

    def get_public_submissions_round(self, round_id):
        return self.get_round(round_id)

    def get_published_results_round(self, round_id):
        row = self.get_round(round_id)
        if row is None or row["status"] != "published":
            return None
        configuration = row["configuration_doc"]
        return {
            **row,
            **{"cfg_" + key: json.dumps(configuration[key])
               for key in PUBLIC_RESULTS_CONFIGURATION_FIELDS if key in configuration},
        }


class HistoricalStoreHarnessMixin:
    def make_store(self):
        return HistoricalArenaStore(PsycopgTransport(self.connect), lease_ttl_seconds=420)


def freeze_historical_intake_dates(harness, evaluation_day):
    """Reconstruct an old intake window while retaining runnable local leases.

    A dated round can resume execution later. Keep its Day0/Day1 authority
    dates frozen so current date-based continuations do not alter old policy.
    This fixture helper changes only the disposable test database before commit.
    """
    configuration = deepcopy(harness.service.store.get_round(harness.round_id)["configuration_doc"])
    original_cutoff = configuration["schedule"]["submission_cutoff"]
    bank_day = date.fromisoformat(evaluation_day) - timedelta(days=1)
    configuration["schedule"].update(
        submission_open=bank_day.isoformat() + "T00:00:00Z",
        submission_cutoff=evaluation_day + "T00:00:00Z",
    )
    contracts.validate_round_configuration(configuration)
    with harness.connect() as connection, connection.cursor() as cursor:
        cursor.execute("SET LOCAL session_replication_role=replica")
        cursor.execute(
            "UPDATE public.lab_arena_rounds SET configuration_doc=%s::jsonb WHERE round_id=%s",
            (json.dumps(configuration), harness.round_id),
        )
    harness.service._invalidate_hot_round()
    return original_cutoff
