"""Deliver full-claim guidance without weakening semantic paraphrases.

These are native prompt-delivery controls, not measured model verdicts.
"""
import pytest

from qualification.scoring import intent_verification_three_stage as intent
from qualification.scoring.prompts import _common


def prompt(claim, source, *, target='Launched a new product or capability in the last 365 days.', category='PRODUCT_LAUNCH'):
    row = {'id': 'signal-1', 'company': 'Acme', 'website': 'https://acme.example',
        'company_linkedin': '', 'contact_linkedin': '', 'claim': claim,
        'signal_date': '2026-03-03', 'signal_type': 'intent',
        'claimed_source_urls': ['https://acme.example/news/event'],
        '_target_signal_text': target, '_evidence_type': category,
        '_integrity_policy': True, '_buyer_max_age_days': 365}
    return intent._build_final_judge_prompt(row, {'results': [{
        'url': row['claimed_source_urls'][0], 'title': 'Exact submitted event', 'text': source,
    }], 'statuses': []})


@pytest.mark.parametrize('addition', [
    'and released separate agents on October 6, 2026.',
    'and opened a second manufacturing plant on October 6, 2026.',
])
def test_supported_launch_does_not_establish_an_added_material_event(addition):
    claim = 'Acme launched AI fluency solutions and an MCP capability on March 3, 2026, ' + addition
    source = 'On March 3, 2026, Acme launched AI fluency solutions and an MCP capability.'
    result = prompt(claim, source)
    assert claim in result and source in result
    assert 'Check every distinct event and material factual detail asserted within' in result
    assert 'dates asserted for additional events' in result
    assert 'even when the supported part meets the target' in result
    assert 'return partially_supported' in result
    assert 'and name it in unsupported_parts' in result
    assert _common.SIGNAL_STATUS_DECISION_BLOCK in result


def test_same_event_wording_keeps_semantic_fidelity_and_opening_requirement():
    claim = 'Acme announced and celebrated the opening of its new center in July 2026.'
    source = 'On July 6, 2026, the community gathered to celebrate the opening of Acme Center.'
    result = prompt(claim, source, target='Opened a new facility in the last 365 days.', category='FACILITY_OPENING')
    assert claim in result and source in result
    assert 'Do not count faithful wording for the same supported event as a separate event.' in result
    assert 'Require semantic fidelity, not verbatim wording.' in result
    assert _common.FACILITY_OPENING_BLOCK in result
    assert _common.FACILITY_EXPANSION_ALTERNATIVE_BLOCK not in result
