"""Client projections for fixtures that deliberately pin pre-474 SQL.

Historical migration tests must retain their original function bodies. They do
not have the later host-retry attempt counter, generation floor, or compact
public read projections. Use this client only in explicitly historical harnesses;
current-service tests must use ArenaStore and current_service_database instead.
"""

import json

from lab_arena.store import ArenaStore, PsycopgTransport, PUBLIC_RESULTS_CONFIGURATION_FIELDS


class HistoricalArenaStore(ArenaStore):
    def list_runs(self, round_id, *, columns="*", **kwargs):
        if columns == "*,execution_attempt:lab_arena_execution_attempt_count":
            # Pre-474 SQL counts every stored attempt. Host-retry exclusions
            # did not exist; the persisted attempt is the historical authority.
            rows = super().list_runs(round_id, columns="*", **kwargs)
            return [{**row, "execution_attempt": row["attempt"]} for row in rows]
        return super().list_runs(round_id, columns=columns, **kwargs)

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
