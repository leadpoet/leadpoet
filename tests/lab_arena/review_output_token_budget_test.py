"""Output bytes are not a measured provider token requirement."""
import json

import pytest

from lab_arena import code_review
from tests.lab_arena.test_lab_arena_code_review import _archive, _openrouter_response, _pydantic_style_source


def long_path_archive():
    members = _pydantic_style_source()
    members.update({
        'research/' + 'repeated_module_' * 12 + f'{index:03}.txt': b'Read actual research evidence.\n'
        for index in range(220)
    })
    assert all(len(path.encode()) <= 255 for path in members)
    return _archive(members), members


def complete_response(prepared):
    response = _openrouter_response(prepared)
    document = json.loads(response['choices'][0]['message']['content'])
    document['reviewed_comparisons'] = []
    response['choices'][0]['message']['content'] = json.dumps(document)
    return response


def test_long_compressible_coverage_bytes_do_not_prove_default_token_budget_impossible():
    payload, members = long_path_archive()
    prepared = code_review.prepare_request(payload)
    response = complete_response(prepared)
    assert len(response['choices'][0]['message']['content'].encode()) > 32_768
    assert prepared.parameters['max_tokens'] == 32_768
    submitted = json.loads(prepared.parameters['messages'][1]['content'])
    assert {item['path']: item['content'].encode() for item in submitted['submission_files']} == members
    assert prepared.parameters['response_format']['json_schema']['strict'] is True
    assert code_review.parse_response(response, prepared).passed


@pytest.mark.parametrize('damage', ['missing', 'duplicate'])
def test_long_coverage_still_rejects_missing_or_duplicate_paths(damage):
    payload, _ = long_path_archive()
    prepared = code_review.prepare_request(payload)
    response = complete_response(prepared)
    document = json.loads(response['choices'][0]['message']['content'])
    if damage == 'missing':
        document['reviewed_files'].pop()
    else:
        document['reviewed_files'][1] = document['reviewed_files'][0]
    response['choices'][0]['message']['content'] = json.dumps(document)
    with pytest.raises(code_review.CodeReviewError) as failure:
        code_review.parse_response(response, prepared)
    assert failure.value.response_reason == 'coverage'


@pytest.mark.parametrize('max_tokens,reason', [(32_768, 'truncated'), (65_536, 'truncated_at_limit')])
def test_actual_truncation_stays_incomplete_at_both_output_capacities(max_tokens, reason):
    payload, _ = long_path_archive()
    prepared = code_review.prepare_request(payload, max_output_tokens=max_tokens)
    response = complete_response(prepared)
    response['choices'][0]['finish_reason'] = 'length'
    with pytest.raises(code_review.CodeReviewError) as failure:
        code_review.parse_response(response, prepared)
    assert failure.value.response_reason == reason
