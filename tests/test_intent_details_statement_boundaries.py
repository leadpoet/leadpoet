"""Keep abbreviation continuations inside lossless paragraph review units."""
import asyncio
import json

import pytest

from qualification.scoring import intent_details, verification_helpers
from tests.test_arena_intent_details_grounding import inputs, _review_response


FLYWIRE_PARAGRAPH = (
    "On 2026-08-12 Flywire announced expanded support for open banking 'Pay by Bank' "
    "payments in the U.S. and Canada, a new North American platform capability "
    "letting payers authorize large domestic and cross-border payments from their "
    "bank accounts in local currency. The 2026-08-12 announcement is a strategic "
    "partnership between Flywire and payments provider Trustly covering open "
    "banking payments in the U.S. and Canada. This may support Flywire's growth "
    "in cross-border and vertical payment processing."
)


def texts(paragraph):
    units = intent_details._statement_units(paragraph)
    assert [unit['unit_id'] for unit in units] == list(range(len(units)))
    assert ' '.join(unit['text'] for unit in units) == paragraph
    assert len(units) <= intent_details._MAX_STATEMENT_UNITS
    return [unit['text'] for unit in units]


def test_saved_flywire_country_abbreviations_keep_their_complete_assertions():
    parts = texts(FLYWIRE_PARAGRAPH)
    assert len(parts) == 3
    assert parts[0].endswith('in local currency.')
    assert parts[1].endswith('in the U.S. and Canada.')
    assert parts[2].startswith('This may support')


@pytest.mark.parametrize('phrase', [
    'the U.S. and Canada', 'the U.K. and Ireland', 'B.C. and Alberta',
    'the U.S. Air Force', 'Dr. Ada Smith', 'J. P. Morgan',
    'Acme Inc. and its subsidiaries', 'tools, e.g. reporting software',
    'software, i.e. workflow tools', '“U.S.” and Canadian teams',
])
def test_common_abbreviations_do_not_create_fragment_units(phrase):
    first = f'Acme supports {phrase} through its platform.'
    assert texts(first + ' This may expand its offering.') == [first, 'This may expand its offering.']


def test_ambiguous_abbreviation_sentence_end_can_share_one_lossless_unit():
    # These are review units, not a linguistic sentence parser. Keeping an
    # ambiguous abbreviation with the following sentence avoids a false fragment.
    paragraph = 'Acme launched in the U.S. It also opened an office.'
    assert texts(paragraph) == [paragraph]


@pytest.mark.parametrize('paragraph,expected', [
    ('Acme launched. It expanded! Could that help?', ['Acme launched.', 'It expanded!', 'Could that help?']),
    ('Acme said “Launched.” It expanded.', ['Acme said “Launched.”', 'It expanded.']),
    ('Acme released version 2.5. It supports 1.2 million records.', ['Acme released version 2.5.', 'It supports 1.2 million records.']),
    ('Acme completed phase 1. Phase 2 remains planned.', ['Acme completed phase 1.', 'Phase 2 remains planned.']),
    ('Acme raised a Series A. It launched a product.', ['Acme raised a Series A.', 'It launched a product.']),
])
def test_ordinary_sentence_ends_and_numbers_preserve_existing_indexes(paragraph, expected):
    assert texts(paragraph) == expected


def test_many_sentences_keep_the_existing_six_unit_bound():
    paragraph = ' '.join(f'Acme completed phase {index}.' for index in range(1, 12))
    assert len(texts(paragraph)) == 6


@pytest.mark.parametrize('paragraph', [
    'J.B. Hunt is hiring local Class A drivers in its Final Mile Services business unit. '
    'This hiring may help J.B. Hunt Transport Services, Inc. expand its trucking and freight logistics offering.',
    'Typewriter announced a $56 million Series A financing on September 2, 2026. '
    'This funding may help Typewriter Therapeutics, Inc. expand its genetic medicines and therapeutics offering.',
    'Pyramid Healthcare opened a licensed outpatient treatment facility in Franklin, North Carolina, in April 2026. '
    'This expansion may help Pyramid Healthcare, Inc. expand its outpatient Mental Health Services offering.',
    'Aspire announced its launch in the U.S. market in an April 2026 newsroom post, expanding into a new country. '
    'This expansion may help Aspire expand its cross-Border and Merchant Payment Processing offering.',
])
def test_other_saved_historical_abbreviations_keep_complete_statements(paragraph):
    assert len(texts(paragraph)) == 2


def test_unsupported_compound_clause_after_abbreviation_is_not_lost_or_accepted():
    company, icp, results, fit = inputs()
    company.intent_details = (
        'Acme launched in the U.S. and doubled payment volume by 80 percent. '
        'It opened a Berlin office.'
    )
    document = intent_details.review_evidence(company, icp, results, fit)
    assert document['intent_details_units'][0]['text'] == (
        'Acme launched in the U.S. and doubled payment volume by 80 percent.'
    )
    response = _review_response(
        {name: True for name in intent_details._CHECKS},
        [{'matched_icp_signal': index, 'covered': True} for index in (0, 1)],
        document,
    )
    response['facts_supported'] = False
    response['unit_grounding'][0]['status'] = 'UNPROVEN'
    response['unit_grounding'][0]['evidence'] = []
    checks = intent_details._validate_review_response(json.dumps(response), document)
    assert checks['facts_supported'] is False
    assert checks['verified_signals_covered'] is True


def test_native_review_retains_unsupported_claim_rejection_and_bonus_coverage(monkeypatch):
    company, icp, results, fit = inputs()
    company.intent_details = FLYWIRE_PARAGRAPH + ' This increased payment volume by 80 percent.'
    calls = []

    async def judge(prompt, **kwargs):
        document = json.loads(prompt)
        calls.append(document)
        assert len(document['intent_details_units']) == 4
        assert document['intent_details_units'][1]['text'].endswith('in the U.S. and Canada.')
        checks = {name: True for name in intent_details._CHECKS}
        response = _review_response(checks, [
            {'matched_icp_signal': 0, 'covered': True},
            {'matched_icp_signal': 1, 'covered': True},
        ], document)
        response['facts_supported'] = False
        # The provider's terminal unsupported factual finding must survive.
        for unit in response['unit_grounding']:
            unit['status'] = 'UNPROVEN' if unit['unit_id'] == 3 else 'VERIFIED'
            if unit['unit_id'] == 3:
                unit['evidence'] = []
        return json.dumps(response)

    monkeypatch.setattr(verification_helpers, 'openrouter_chat', judge)
    receipt = asyncio.run(intent_details.review_intent_details(company, icp, results, fit))
    assert len(calls) == 1
    assert receipt['decision'] == 'mismatch'
    assert receipt['checks']['facts_supported'] is False
    assert receipt['checks']['verified_signals_covered'] is True
