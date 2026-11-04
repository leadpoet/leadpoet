"""Native null placeholders contain no opaque state and retain exact evidence."""
import base64
from copy import deepcopy

import pytest

from lab_arena import contracts, training_eligibility, training_export, training_steps, training_thinking
from tests.lab_arena.training_eligibility_test import SOURCE_HASH, refresh_response_links, submission
from tests.lab_arena.training_hosted_search_test import hosted_example
from tests.lab_arena.training_native_responses_test import _eligible_example, example
from tests.lab_arena.training_steps_test import bind_selected_output, seal


@pytest.mark.parametrize('hosted', [False, True])
@pytest.mark.parametrize('historical', [False, True])
def test_null_output_placeholder_preserves_observed_and_verified_evidence(hosted, historical):
    value = hosted_example() if hosted else _eligible_example()
    value.update(source={'sha256': SOURCE_HASH}, output={'companies': []},
                 input_snapshots={'run': {'input': {'icp': 'retained input'}}})
    value['configuration']['training_maximum_missing_reasoning_percent'] = 1
    call = value['provider_calls'][0]
    call.update(accounting_state='settlement', actual_microusd=1, response_status=200)
    response = call['response']
    for item in response['output']:
        if item['type'] == 'reasoning':
            item['encrypted_content'] = None
    if historical:
        call.pop('reasoning_capture')
        value['configuration'] = {}
    bind_selected_output(value)
    refresh_response_links(value, 0)
    seal(value)
    original = deepcopy(value)

    # Exercise the real export privacy projection before the target parser.
    retained, redactions = training_export._response_value({
        'body_b64': base64.b64encode(contracts.canonical_json(response).encode()).decode(),
    }, 'openrouter')
    assert retained == response and redactions == []
    for build in (training_steps._build, training_steps.build):
        splits, manifest = build([value])
        rows = splits['train'] + splits['eval']
        if hosted and historical and build is training_steps._build:
            # The historical strict parser already excludes hosted tools.
            # The observed exporter below can retain their exposed bytes.
            assert rows == []
            assert manifest['counts']['exclusion_reasons'] == {'hosted_tool_shape_unsupported': 1}
            continue
        assert len(rows) == 1, manifest
        expected = response['output'] if hosted else response['output'][1:]
        assert rows[0]['target'] == expected
        assert rows[0]['source_refs'][0]['delivered_response_sha256'] == contracts.document_hash(response)

    strict, manifest = training_thinking.build([value])
    rows = strict['train'] + strict['eval']
    assert len(rows) == (0 if historical else 1), manifest
    if rows:
        assert rows[0]['target'] == (response['output'] if hosted else response['output'][1:])
    verdict = training_eligibility.assess_capture(
        submission(), {'run': value}, ['run'], enabled=True,
        minimum_coverage_percent=80, maximum_missing_reasoning_percent=1)
    assert verdict['eligible'] is (not historical), verdict
    trace = training_thinking.readable_trace(value)
    native = [event['value'] for event in trace['calls'][0]['events']
              if event.get('delivery') == 'selected_terminal'
              and event.get('source_path', [])[:4] == ['provider_calls', 0, 'response', 'output']
              and len(event['source_path']) == 5]
    assert native == response['output']
    assert value == original


@pytest.mark.parametrize('allow_reasoning_history', [False, True])
@pytest.mark.parametrize('opaque', ['opaque-state', '', False, 0, {}, []])
def test_nonnull_output_placeholder_stays_unsupported(allow_reasoning_history, opaque):
    value = example()
    call = value['provider_calls'][0]
    call['response']['output'][0]['encrypted_content'] = opaque
    with pytest.raises(ValueError, match='reasoning_shape_unsupported'):
        training_steps._target(call['request']['outbound']['body'], call['response'],
            'openrouter.responses', allow_reasoning_history=allow_reasoning_history)
