"""Retained reasoning occurrences and summaries survive offline projection."""
from copy import deepcopy

import pytest

from lab_arena import contracts, training_eligibility, training_thinking
from tests.lab_arena.training_eligibility_test import SOURCE_HASH, refresh_response_links, submission
from tests.lab_arena.training_native_responses_test import _eligible_example
from tests.lab_arena.training_steps_test import bind_selected_output, seal
from tests.lab_arena.training_thinking_test import full_example


def _example(operation):
    if operation == 'openrouter.responses':
        return _eligible_example()
    value = full_example()
    value.update(source={'sha256': SOURCE_HASH}, output={'companies': []},
                 input_snapshots={'run': {'input': {'icp': 'retained input'}}})
    value['configuration']['training_maximum_missing_reasoning_percent'] = 1
    value['provider_calls'][0].update(accounting_state='settlement', actual_microusd=1,
                                      response_status=200)
    return bind_selected_output(value)


def _project(value):
    refresh_response_links(value, 0)
    seal(value)
    original = deepcopy(value)
    splits, _ = training_thinking.build([value])
    verdict = training_eligibility.assess_capture(
        submission(), {'run': value}, ['run'], enabled=True,
        minimum_coverage_percent=80, maximum_missing_reasoning_percent=1)
    trace = training_thinking.readable_trace(value)
    assert value == original
    return splits['train'] + splits['eval'], verdict, trace


@pytest.mark.parametrize('operation', ['openrouter.chat', 'openrouter.responses'])
def test_equal_summary_cannot_hide_reviewed_full_text(operation):
    value = _example(operation)
    response = value['provider_calls'][0]['response']
    if operation == 'openrouter.chat':
        item = response['choices'][0]['message']
        text = item['reasoning_content']
        item['reasoning_details'] = [{'type': 'reasoning.summary', 'summary': text}]
    else:
        item = response['output'][0]
        text = item['content'][0]['text']
        item['summary'] = [{'type': 'summary_text', 'text': text}]
    records, verdict, trace = _project(value)
    assert len(records) == 1
    assert [segment['text'] for segment in records[0]['generated_thinking']] == [text]
    assert verdict['eligible'] is True, verdict
    assert verdict['counts']['full_thinking_calls'] == 1
    events = [event for event in trace['calls'][0]['events']
              if event.get('delivery') == 'selected_terminal' and event.get('reasoning_field')]
    assert [(event['kind'], event['value']) for event in events] == (
        [('full_generated_thinking', text), ('summary', text)]
        if operation == 'openrouter.chat' else
        [('summary', text), ('full_generated_thinking', text)])


@pytest.mark.parametrize('summary_matches_full', [False, True])
def test_chat_summary_text_keeps_reviewed_full_thinking_and_exact_summary_path(summary_matches_full):
    value = _example('openrouter.chat')
    response = value['provider_calls'][0]['response']
    message = response['choices'][0]['message']
    full_text = message['reasoning_content']
    summary_text = full_text if summary_matches_full else 'An ancillary summary.'
    message['reasoning_details'] = [{'type': 'summary_text', 'text': summary_text}]
    saved_response = deepcopy(response)
    records, verdict, trace = _project(value)
    assert verdict['eligible'] is True, verdict
    assert verdict['counts']['full_thinking_calls'] == len(records) == 1
    assert [segment['text'] for segment in records[0]['generated_thinking']] == [full_text]
    assert response == saved_response
    events = [event for event in trace['calls'][0]['events']
              if event.get('delivery') == 'selected_terminal' and event.get('reasoning_field')]
    assert [(event['kind'], event['value']) for event in events] == [
        ('full_generated_thinking', full_text), ('summary', summary_text)]
    assert events[1]['source_path'] == [
        'provider_calls', 0, 'response', 'choices', 0, 'message', 'reasoning_details', 0, 'text']


def test_chat_summary_text_alone_cannot_supply_full_thinking():
    value = _example('openrouter.chat')
    message = value['provider_calls'][0]['response']['choices'][0]['message']
    message.pop('reasoning_content')
    message['reasoning_details'] = [{'type': 'summary_text', 'text': 'A summary alone.'}]
    records, verdict, trace = _project(value)
    assert records == [] and verdict['eligible'] is False
    assert verdict['counts']['full_thinking_calls'] == 0
    events = [event for event in trace['calls'][0]['events']
              if event.get('delivery') == 'selected_terminal' and event.get('reasoning_field')]
    assert [(event['kind'], event['value']) for event in events] == [('summary', 'A summary alone.')]


@pytest.mark.parametrize('operation', ['openrouter.chat', 'openrouter.responses'])
@pytest.mark.parametrize('separate_items', [False, True])
def test_repeated_segments_keep_occurrences_order_and_raw_paths(operation, separate_items):
    value = _example(operation)
    call = value['provider_calls'][0]
    texts = ['Repeat.', 'Middle.', 'Repeat.']
    if operation == 'openrouter.chat':
        item = call['response']['choices'][0]['message']
        item.pop('reasoning_content')
        item['reasoning_details'] = [{'type': 'reasoning.text', 'text': text} for text in texts]
        fields = ['reasoning_details.text']
        expected = [['choices', 0, 'message', 'reasoning_details', index, 'text']
                    for index in range(3)]
        if not separate_items:
            # Chat's typed blocks are separate reasoning_details entries;
            # repeat the same case with summaries between those full blocks.
            item['reasoning_details'].insert(1, {'type': 'reasoning.summary', 'summary': 'Summary.'})
            expected[1][-2] = 2
            expected[2][-2] = 3
    else:
        fields = ['output.content.reasoning_text']
        if separate_items:
            template = call['response']['output'][0]
            call['response']['output'][:1] = [
                dict(deepcopy(template), id=f'r{index}',
                     content=[{'type': 'reasoning_text', 'text': text}])
                for index, text in enumerate(texts)]
            expected = [['output', index, 'content', 0, 'text'] for index in range(3)]
        else:
            call['response']['output'][0]['content'] = [
                {'type': 'reasoning_text', 'text': text} for text in texts]
            expected = [['output', 0, 'content', index, 'text'] for index in range(3)]
    registry = value['configuration']['reasoning_route_registry']
    registry['approved_routes'][0]['full_text_fields'] = fields
    call['reasoning_capture'].update(full_text_fields=fields,
        registry_sha256=contracts.document_hash(registry).split(':', 1)[1])
    records, verdict, trace = _project(value)
    assert len(records) == 1
    segments = records[0]['generated_thinking']
    assert [segment['text'] for segment in segments] == texts
    assert [segment['source_paths'] for segment in segments] == [[path] for path in expected]
    assert verdict['eligible'] is True, verdict
    events = [event for event in trace['calls'][0]['events']
              if event.get('delivery') == 'selected_terminal'
              and event['kind'] == 'full_generated_thinking']
    assert [event['value'] for event in events] == texts
    assert [event['source_path'] for event in events] == [
        ['provider_calls', 0, 'response'] + path for path in expected]


def test_equivalent_chat_aliases_keep_one_text_and_all_sources():
    value = _example('openrouter.chat')
    call = value['provider_calls'][0]
    message = call['response']['choices'][0]['message']
    text = message['reasoning_content']
    message['reasoning'] = text
    message['reasoning_details'] = [{'type': 'reasoning.text', 'text': text}]
    fields = ['reasoning', 'reasoning_content', 'reasoning_details.text']
    registry = value['configuration']['reasoning_route_registry']
    registry['approved_routes'][0]['full_text_fields'] = fields
    call['reasoning_capture'].update(full_text_fields=fields,
        registry_sha256=contracts.document_hash(registry).split(':', 1)[1])
    records, verdict, _ = _project(value)
    assert verdict['eligible'] is True, verdict
    assert len(records) == 1
    segments = records[0]['generated_thinking']
    assert [segment['text'] for segment in segments] == [text]
    assert segments[0]['source_paths'] == [
        ['choices', 0, 'message', 'reasoning'],
        ['choices', 0, 'message', 'reasoning_content'],
        ['choices', 0, 'message', 'reasoning_details', 0, 'text']]


def test_partial_chat_alias_match_cannot_erase_or_reorder_typed_occurrences():
    response = {'choices': [{'message': {
        'reasoning': 'Middle.',
        'reasoning_details': [{'type': 'reasoning.text', 'text': text}
                              for text in ['Repeat.', 'Middle.', 'Repeat.']]}}]}
    segments, _, _ = training_thinking._segments(response, 'openrouter.chat')
    assert [segment['text'] for segment in segments] == [
        'Middle.', 'Repeat.', 'Middle.', 'Repeat.']
    assert all(len(segment['source_paths']) == 1 for segment in segments)


def test_matching_chat_choices_are_not_aliases_of_each_other():
    response = {'choices': [{'message': {'reasoning': 'Same.'}},
                            {'message': {'reasoning': 'Same.'}}]}
    segments, _, _ = training_thinking._segments(response, 'openrouter.chat')
    assert [segment['text'] for segment in segments] == ['Same.', 'Same.']
    assert [segment['source_paths'] for segment in segments] == [
        [['choices', 0, 'message', 'reasoning']], [['choices', 1, 'message', 'reasoning']]]
