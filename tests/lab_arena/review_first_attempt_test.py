"""First-attempt contract and source-verification regression checks."""
import json

import pytest

from lab_arena import code_review
from tests.lab_arena.test_lab_arena_code_review import (
    _archive, _openrouter_response, _pydantic_style_source,
)


def test_complete_coverage_order_is_semantic_and_normalized():
    prepared = code_review.prepare_request(
        _archive(_pydantic_style_source()),
        similarity_contexts=({'comparison_id': 'first'}, {'comparison_id': 'second'}),
    )
    response = _openrouter_response(prepared)
    content = json.loads(response['choices'][0]['message']['content'])
    content['reviewed_files'].reverse()
    content['reviewed_comparisons'] = ['second', 'first']
    response['choices'][0]['message']['content'] = json.dumps(content)
    result = code_review.parse_response(response, prepared)
    assert result.passed
    assert result.reviewed_files == prepared.reviewed_files


@pytest.mark.parametrize('field', ['reviewed_files', 'reviewed_comparisons'])
@pytest.mark.parametrize('change', ['duplicate', 'missing', 'unknown'])
def test_coverage_still_requires_every_unique_expected_item(field, change):
    prepared = code_review.prepare_request(
        _archive(_pydantic_style_source()),
        similarity_contexts=({'comparison_id': 'first'}, {'comparison_id': 'second'}),
    )
    response = _openrouter_response(prepared)
    content = json.loads(response['choices'][0]['message']['content'])
    content['reviewed_comparisons'] = ['first', 'second']
    values = content[field]
    if change == 'duplicate':
        values[1] = values[0]
    elif change == 'missing':
        values.pop()
    else:
        values[0] = 'unknown'
    response['choices'][0]['message']['content'] = json.dumps(content)
    with pytest.raises(code_review.CodeReviewError) as failure:
        code_review.parse_response(response, prepared)
    assert failure.value.response_reason == 'coverage'


def test_truncated_output_has_distinct_authoritative_reason():
    prepared = code_review.prepare_request(_archive(_pydantic_style_source()))
    response = _openrouter_response(prepared, finish_reason='length')
    with pytest.raises(code_review.CodeReviewError) as failure:
        code_review.parse_response(response, prepared)
    assert failure.value.response_reason == 'truncated'


@pytest.mark.parametrize('research', [False, True])
@pytest.mark.parametrize('model', [code_review.DEFAULT_REVIEW_MODEL, code_review.FALLBACK_REVIEW_MODEL])
def test_supported_pinned_routes_request_strict_schema_without_relaxing_privacy(model, research):
    prepared = code_review.prepare_request(_archive(_pydantic_style_source()), model=model, research_control=research)
    request = prepared.parameters
    assert request['max_tokens'] == 32_768
    response_format = request['response_format']
    assert response_format['type'] == 'json_schema'
    assert response_format['json_schema']['strict'] is True
    schema = response_format['json_schema']['schema']
    assert schema['additionalProperties'] is False
    assert set(schema['required']) == set(schema['properties'])
    assert ('research_control' in schema['properties']) is research
    assert request['provider']['zdr'] is True
    assert request['provider']['data_collection'] == 'deny'
    assert request['provider']['allow_fallbacks'] is False
    assert request['provider']['require_parameters'] is True
    assert request['provider']['only'] == ['azure' if model == code_review.FALLBACK_REVIEW_MODEL else 'google-vertex']


def test_strict_schema_guides_shape_without_claiming_source_evidence():
    from jsonschema import Draft202012Validator
    from tests.lab_arena.research_control_code_review_test import SOURCE, _assessment, _response
    prepared = code_review.prepare_request(_archive(SOURCE), research_control=True)
    schema = prepared.parameters['response_format']['json_schema']['schema']
    Draft202012Validator.check_schema(schema)
    response = _response(prepared, _assessment(prepared))
    content = json.loads(response['choices'][0]['message']['content'])
    content['reviewed_comparisons'] = []
    content['summary'] = 'Full source review.'
    content['research_control']['evidence'][0]['excerpt'] = 'fabricated excerpt unavailable in source'
    Draft202012Validator(schema).validate(content)
    response['choices'][0]['message']['content'] = json.dumps(content)
    with pytest.raises(code_review.CodeReviewError) as failure:
        code_review.parse_response(response, prepared)
    assert failure.value.response_reason == 'research_evidence_mismatch'








def test_schema_binds_exact_source_and_comparison_identifiers():
    from jsonschema import Draft202012Validator, ValidationError
    from tests.lab_arena.research_control_code_review_test import SOURCE, _assessment, _response
    prepared = code_review.prepare_request(_archive(SOURCE), research_control=True,
        similarity_contexts=({'comparison_id': 'bound-comparison'},))
    schema = prepared.parameters['response_format']['json_schema']['schema']
    response = _response(prepared, _assessment(prepared))
    content = json.loads(response['choices'][0]['message']['content'])
    content.update(summary='Review complete.', reviewed_comparisons=['bound-comparison'])
    validator = Draft202012Validator(schema)
    validator.validate(content)
    content['research_control']['source_archive_sha256'] = 'sha256:' + 'f' * 64
    with pytest.raises(ValidationError):
        validator.validate(content)


def test_schema_required_empty_comparisons_are_in_output_capacity_bound():
    # This bound previously omitted a required field when there are no contexts.
    payload = _archive(_pydantic_style_source())
    prepared = code_review.prepare_request(payload)
    minimum = {'verdict': 'pass', 'summary': 'review complete',
               'reviewed_files': list(prepared.reviewed_files),
               'reviewed_comparisons': [], 'findings': []}
    from lab_arena import contracts
    bound = len(contracts.canonical_json(minimum).encode())
    with pytest.raises(code_review.CodeReviewError, match='review_output_cannot_report_coverage'):
        code_review.prepare_request(payload, max_output_tokens=bound-1)


@pytest.mark.parametrize('native_finish', ['length', 'max_tokens'])
def test_native_output_limit_still_classifies_truncation_when_router_says_stop(native_finish):
    prepared = code_review.prepare_request(_archive(_pydantic_style_source()))
    response = _openrouter_response(prepared)
    response['choices'][0]['native_finish_reason'] = native_finish
    with pytest.raises(code_review.CodeReviewError) as failure:
        code_review.parse_response(response, prepared)
    assert failure.value.response_reason == 'truncated'
