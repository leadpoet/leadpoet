"""Historical projections keep actual SQL, filters and errors intact."""
import pytest
from tests.lab_arena.historical_round_reads import pre517_round_select


def test_only_pre517_round_floor_projection_is_adapted():
    calls = []
    row = {'round_id': 'historical', 'publication_doc': {'unchanged': True}}
    def select(table, **kwargs):
        calls.append((table, kwargs))
        return [row]
    read = pre517_round_select(select)
    assert read('lab_arena_published_results_v1',
                columns='round_id,scoring_generation_floor,publication_doc',
                filters={'status': 'published'}, limit=1) == [{**row, 'scoring_generation_floor': 0}]
    assert calls == [('lab_arena_published_results_v1', {
        'columns': 'round_id,publication_doc', 'filters': {'status': 'published'}, 'limit': 1})]
    assert read('lab_arena_runs', columns='scoring_generation_floor')[0] is row
    assert calls[-1][1]['columns'] == 'scoring_generation_floor'


def test_historical_read_error_is_never_replaced_with_a_result():
    def denied(*_args, **_kwargs):
        raise PermissionError('actual SQL remains denied')
    with pytest.raises(PermissionError, match='actual SQL remains denied'):
        pre517_round_select(denied)('lab_arena_rounds',
                                   columns='round_id,scoring_generation_floor')
