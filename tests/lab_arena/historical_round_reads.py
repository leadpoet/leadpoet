"""Pin public read projections to the real pre-517 migration test schema.

Those schemas predate scorer supersession and have one implicit generation
(zero). Keep their actual tables, views, RPCs and permissions unchanged while
allowing the current public formatter to verify their original result bodies.
Current-schema service tests use current_service_database without this adapter.
"""
from functools import wraps


def pre517_round_select(select):
    @wraps(select)
    def historical_select(table, *, columns='*', **kwargs):
        old_round = table in ('lab_arena_rounds', 'lab_arena_published_results_v1')
        fields = columns.split(',')
        if old_round and 'scoring_generation_floor' in fields:
            fields.remove('scoring_generation_floor')
            rows = select(table, columns=','.join(fields), **kwargs)
            return [{**row, 'scoring_generation_floor': 0} for row in rows]
        return select(table, columns=columns, **kwargs)
    return historical_select
