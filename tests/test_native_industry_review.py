import asyncio
import json

import pytest

from lab_arena import scoring
from qualification.scoring import company_evidence_investigator as c, lead_scorer
from tests.test_company_evidence_investigator import _finding
from tests.test_company_evidence_refusal_fallback import _install_loop, FEATURES, URL


MODEL = 'openai/gpt-6.1-sol'


def install(monkeypatch, *, decision='match', missing=None, mutation=None,
            locator_transport=False, **kwargs):
    finding = kwargs.pop('finding', _finding('industry'))
    run, calls, clock = _install_loop(
        monkeypatch, target='industry', status=200,
        body='Coreworks connects account workflows to its analysis and reporting product.',
        urls=('https://www.coreworks.ai/product', URL), finding=finding,
        **kwargs)
    monkeypatch.setenv('ARENA_INDUSTRY_EVIDENCE_MODEL', MODEL)
    original = c._post_json
    review_calls = []
    original_search = c._search_web

    async def search(session, query, **options):
        # The empty-corpus discovery now precedes the first model turn. Keep
        # this offline locator lookup fast; later searches retain this fixture's
        # deliberate deadline crossing. Deadline controls also test slow reads.
        if options.get('read_source_text') and not review_calls:
            calls['search'].append(query)
            return {'results': [], 'notice': 'discovery_only_not_evidence'}
        return await original_search(session, query, **options)

    monkeypatch.setattr(c, '_search_web', search)

    if not locator_transport:
        # These loop fixtures share an offline search stub. Real Sonar request
        # parsing and source admission are exercised by the discovery tests.
        async def locators(session, query, *, key, company_locator, include_domains):
            result = await c._search_web(session, query, key=key,
                discovery_highlights=True, include_domains=include_domains)
            return {k: v for k, v in result.items() if k != '_source_pages'}
        monkeypatch.setattr(c, '_search_industry_locators', locators)

    async def post(session, url, *, headers, payload):
        if 'response_format' not in payload:
            if payload.get('tool_choice') == {
                    'type': 'function', 'function': {'name': 'submit_findings'}}:
                from tests.test_company_evidence_judgment_reserve import _response
                calls['model'].append(payload)
                return 200, _response('submit_findings', {
                    'findings': finding if isinstance(finding, list) else [finding]})
            return await original(session, url, headers=headers, payload=payload)
        review_calls.append(payload)
        pages = [page for message in payload['messages'][2:]
                 for page in json.loads(message['content'])['untrusted_source_pages']]
        spans = [span for page in pages for span in page['spans']]
        result = {'observed_activity': 'Analysis, reporting and account workflow software',
                  'reason': 'The connected product is supported by the two pages.',
                  'missing_requirements': missing or [], 'role': 'supplier_operator',
                  'evidence_span_ids': [span['span_id'] for span in spans], 'decision': decision}
        if mutation:
            mutation(result)
        if 'research_query' in payload['response_format']['json_schema']['schema']['properties']:
            result['research_query'] = ('account workflow integration'
                if result['decision'] != 'match' and result['missing_requirements'] else '')
        return 200, {'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(result)}}]}

    monkeypatch.setattr(c, '_post_json', post)
    return run, calls, clock, review_calls


def test_source_selection_sees_prefetched_tails_seen_by_the_industry_reviewer(monkeypatch):
    """The bounded metadata message must not hide already-loaded source facts."""
    def bounded_review(result):
        result['evidence_span_ids'] = result['evidence_span_ids'][:8]
        result.update(decision='unavailable', missing_requirements=['Account workflow'])

    run, calls, _, reviews = install(monkeypatch, mutation=bounded_review)
    urls = [f'https://www.coreworks.ai/source-{index}' for index in range(3)]
    tails = [f'Coreworks uniquely documented capability number {index}.' for index in range(3)]
    pages = {url: {'final_url': url, 'text': 'Coreworks documentation. ' * 950 + tail}
             for url, tail in zip(urls, tails)}
    asyncio.run(run(prefetched_pages=pages,
                    prior_observations={'submitted_source_urls': urls}))
    assert calls['model'] and reviews
    first = calls['model'][0]['messages']
    metadata = next(m['content'] for m in first if m['role'] == 'user')
    assert not any(tail in metadata for tail in tails)  # Existing message cap is real.
    research = json.dumps(first)
    initial_review = json.dumps(reviews[0]['messages'])
    assert all(tail in initial_review for tail in tails)
    assert all(tail in research for tail in tails)


def test_industry_research_uses_the_final_review_criterion_rules(monkeypatch):
    """Source selection must understand the same capabilities as its reviewer."""
    from qualification.scoring import industry_classification as industry

    run, calls, _, reviews = install(monkeypatch)
    asyncio.run(run())
    assert calls['model'] and reviews
    criterion_rules = '\n'.join(industry.SYSTEM.splitlines()[1:5])
    for request in calls['model']:
        instructions = ''.join(message['content'] for message in request['messages']
                               if message['role'] == 'system')
        assert criterion_rules in instructions
        assert 'judge other targets normally' in instructions
        assert request['reasoning']['effort'] == 'low'
        assert request['max_tokens'] == c.REASONING_MAX_TOKENS
    assert all(criterion_rules in request['messages'][0]['content']
               for request in reviews)
    for request in reviews:
        research = 'research_query' in request['response_format']['json_schema']['schema']['properties']
        assert request['reasoning']['effort'] == ('low' if research else 'medium')
    assert all(request['max_tokens'] == 2000 for request in reviews)


@pytest.mark.parametrize('positive_review', [False, True])
def test_native_loop_retains_both_sources_through_fit_evidence(monkeypatch, positive_review):
    run, calls, clock, reviews = install(monkeypatch)
    # Exercise both normal and prior-positive modes through the same entrypoint.
    original = c.investigate_company_evidence

    async def invoke(**kwargs):
        kwargs['positive_semantic_review'] = positive_review
        return await original(**kwargs)

    monkeypatch.setattr(c, 'investigate_company_evidence', invoke)
    result = asyncio.run(run(requested_product_service='Supplies reporting and account workflows'))
    assert result['_completed_submit'] is True
    finding = result['claims']['industry']
    assert finding['status'] == 'VERIFIED'
    assert len(finding['_industry_review']['evidence']) == 2
    assert len(finding['supporting_evidence']) == 1
    projected = lead_scorer._project_investigator_industry({'attribute_satisfied': False}, finding)
    stored = json.loads(json.dumps(lead_scorer._dimension_web_evidence(projected, 'industry')))
    assert stored['quote'] == FEATURES
    assert stored['supporting_evidence'] == finding['supporting_evidence']
    assert projected['attribute_satisfied'] is False
    assert len(reviews) == result['usage']['industry_review_calls'] == 1
    assert result['usage']['reasoning_turns'] <= c.MAX_REASONING_TURNS
    assert result['usage']['fetch_calls'] <= c.MAX_FETCH_CALLS
    assert result['usage']['search_calls'] <= c.MAX_SEARCH_CALLS


def test_native_discovery_requests_pass_the_real_arena_request_validator(monkeypatch):
    from lab_arena.operations import validate_operation_request
    run, calls, _, reviews = install(monkeypatch)
    result = asyncio.run(run())
    assert result['_completed_submit'] is True
    for request in [*calls['model'], *reviews]:
        validate_operation_request('openrouter.chat', request)


@pytest.mark.parametrize('enabled', [False, True])
def test_final_shared_packet_uses_stronger_model_only_under_new_industry_policy(monkeypatch, enabled):
    run, calls, clock, _ = install(monkeypatch)
    monkeypatch.setenv('ARENA_INDUSTRY_EVIDENCE_MODEL', MODEL if enabled else '')
    original_fetch = c._fetch_page

    async def fetch(*args, **kwargs):
        result = await original_fetch(*args, **kwargs)
        if len(calls['primary']) == 2:
            clock.now = c.ADMISSION_DEADLINE_SECONDS - c.JUDGMENT_ADMISSION_RESERVE_SECONDS
        return result

    monkeypatch.setattr(c, '_fetch_page', fetch)
    asyncio.run(run())
    final = calls['model'][-1]
    assert final['tool_choice'] == {'type': 'function', 'function': {'name': 'submit_findings'}}
    assert final['model'] == (MODEL if enabled else c.POSITIVE_SEMANTIC_REVIEW_MODEL)
    assert final['max_tokens'] == c.REASONING_MAX_TOKENS
    assert final['reasoning']['effort'] == 'low'


@pytest.mark.parametrize('initial_decision', ['unavailable', 'mismatch'])
@pytest.mark.parametrize('missing', [
    ['Account workflow integration'],
    ['General reporting capability', 'Account workflow integration'],
])
def test_native_source_selection_receives_verified_source_gaps_before_research(monkeypatch, missing, initial_decision):
    """The source picker must see the actual gap, not only a broad category."""
    seen = []
    search_options = []

    def initial_gap(result):
        seen.append(result)
        if len(seen) == 1:
            result.update(decision=initial_decision, missing_requirements=missing)

    run, calls, _, reviews = install(monkeypatch, mutation=initial_gap)
    original_post = c._post_json

    async def planned_post(session, url, *, headers, payload):
        if ('response_format' not in payload
                and payload.get('tool_choice') == {
                    'type': 'function', 'function': {'name': 'search_web'}}):
            from tests.test_company_evidence_judgment_reserve import _response
            calls['model'].append(payload)
            return 200, _response('search_web', {'query': 'Coreworks account workflow integration'})
        return await original_post(session, url, headers=headers, payload=payload)

    async def gap_search(session, query, **kwargs):
        calls['search'].append(query)
        search_options.append(kwargs)
        return {'results': [{'url': URL, 'text': 'Account workflows'}],
                'notice': 'discovery_only_not_evidence'}

    monkeypatch.setattr(c, '_post_json', planned_post)
    monkeypatch.setattr(c, '_search_web', gap_search)
    homepage = 'https://www.coreworks.ai/'
    result = asyncio.run(run(
        prior_observations={'submitted_source_urls': [homepage]},
        prefetched_pages={homepage: {'final_url': homepage, 'text': FEATURES}},
        requested_product_service='Supplies reporting and account workflows'))
    first = next(m['content'] for m in calls['model'][0]['messages'] if m['role'] == 'user')
    document = json.loads(first[first.index('{'):])
    brief = document['industry_source_research_brief']
    assert brief['missing_requirements'] == missing
    assert brief['decision'] == initial_decision
    assert brief['research_query'] == 'account workflow integration'
    assert search_options[0]['include_domains'] == ('coreworks.ai',)
    assert search_options[0]['discovery_highlights'] is True
    assert 'not proof' in brief['notice']
    assert calls['search'][0] == (
        "Find Coreworks's own product documentation or customer projects about: "
        'account workflow integration'
    )
    assert document['server_industry_discovery']['notice'] == 'discovery_only_not_evidence'
    assert calls['model'][0]['tool_choice'] == 'required'
    assert calls['model'][0]['model'] == MODEL
    assert calls['model'][1]['model'] == MODEL
    assert all(request['reasoning']['effort'] == 'low' for request in calls['model'][:2])
    # Selecting the next source is still industry research. Do not hand it
    # back to the weaker general reviewer after just one page was fetched.
    assert len(calls['model']) == 3  # Two fetches and final judgment; no query-only model call.
    assert all(request['model'] == MODEL
               for request in calls['model']
               if request.get('tool_choice') == {
                   'type': 'function', 'function': {'name': 'submit_findings'}})
    assert result['claims']['industry']['status'] == 'VERIFIED'
    assert len(reviews) == result['usage']['industry_review_calls'] == 2
    assert result['usage']['reasoning_turns'] <= c.MAX_REASONING_TURNS
    assert document['investigation_limits']['reasoning_turns'] == c.MAX_REASONING_TURNS - 2
    assert result['usage']['fetch_calls'] <= c.MAX_FETCH_CALLS
    assert result['usage']['search_calls'] <= c.MAX_SEARCH_CALLS
    from lab_arena.operations import validate_operation_request
    for request in [*calls['model'], *reviews]:
        validate_operation_request('openrouter.chat', request)


@pytest.mark.parametrize('decision', ['unavailable', 'mismatch'])
@pytest.mark.parametrize('limit', ['deadline', 'search_cap', 'fetch_cap'])
def test_initial_gap_query_obeys_existing_discovery_admission(monkeypatch, limit, decision):
    run, calls, clock, reviews = install(
        monkeypatch, decision=decision, missing=['Account workflow integration'])
    original = c._post_json

    async def review(session, url, **kwargs):
        response = await original(session, url, **kwargs)
        if 'response_format' in kwargs['payload']:
            if limit == 'deadline':
                clock.now = c.ADMISSION_DEADLINE_SECONDS - c.JUDGMENT_ADMISSION_RESERVE_SECONDS
            elif limit == 'search_cap':
                monkeypatch.setattr(c, 'MAX_SEARCH_CALLS', 0)
            else:
                monkeypatch.setattr(c, 'MAX_FETCH_CALLS', 0)
        return response

    async def no_search(*args, **kwargs):
        pytest.fail('Initial query must respect search/fetch/time admission')

    monkeypatch.setattr(c, '_post_json', review)
    monkeypatch.setattr(c, '_search_web', no_search)
    homepage = 'https://www.coreworks.ai/'
    asyncio.run(run(prior_observations={'submitted_source_urls': [homepage]},
        prefetched_pages={homepage: {'final_url': homepage, 'text': FEATURES}}))
    assert reviews
    assert not calls['search']


def test_initial_gap_discovery_is_not_evidence_and_does_not_force_an_empty_fetch(monkeypatch):
    run, calls, _, reviews = install(
        monkeypatch, decision='unavailable', missing=['Account workflow integration'])

    async def empty_search(session, query, **kwargs):
        calls['search'].append(query)
        return {'results': [], 'notice': 'discovery_only_not_evidence'}

    monkeypatch.setattr(c, '_search_web', empty_search)
    homepage = 'https://www.coreworks.ai/'
    result = asyncio.run(run(prior_observations={'submitted_source_urls': [homepage]},
        prefetched_pages={homepage: {'final_url': homepage, 'text': FEATURES}}))
    assert len(calls['search']) == 1
    assert calls['model'][0]['tool_choice'] == 'required'
    assert result['claims']['industry']['status'] == 'UNPROVEN'
    assert all('discovery_only_not_evidence' not in message['content']
               for review in reviews for message in review['messages'][2:])


@pytest.mark.parametrize('enabled', [False, True])
@pytest.mark.parametrize('positive_review', [False, True])
def test_industry_research_keeps_the_frozen_model_even_without_a_gap(monkeypatch, enabled, positive_review):
    run, calls, _, _ = install(monkeypatch, finding=_finding('industry',
        observed_industry='Software', activity_role='supplier_operator',
        evidence_url=URL, evidence_quote='Coreworks connects account workflows to its analysis and reporting product.'))
    monkeypatch.setenv('ARENA_INDUSTRY_EVIDENCE_MODEL', MODEL if enabled else '')
    original = c.investigate_company_evidence

    async def invoke(**kwargs):
        return await original(**dict(kwargs, positive_semantic_review=positive_review))

    monkeypatch.setattr(c, 'investigate_company_evidence', invoke)
    result = asyncio.run(run())
    assert result['_completed_submit']
    assert calls['model']
    expected = c.POSITIVE_SEMANTIC_REVIEW_MODEL if positive_review else c.INVESTIGATOR_MODEL
    for request in calls['model']:
        assert request['model'] == (MODEL if enabled else expected)
    assert result['usage']['reasoning_turns'] <= c.MAX_REASONING_TURNS
    assert result['usage']['fetch_calls'] <= c.MAX_FETCH_CALLS
    assert result['usage']['search_calls'] <= c.MAX_SEARCH_CALLS


def test_verified_industry_only_request_enters_final_submission_without_new_research(monkeypatch):
    run, calls, _, reviews = install(monkeypatch, finding=_finding('industry',
        observed_industry='Software', activity_role='supplier_operator',
        evidence_url=URL, evidence_quote=FEATURES))
    result = asyncio.run(run(prefetched_pages={URL: {'final_url': URL, 'text': FEATURES}},
                            prior_observations={'submitted_source_urls': [URL]}))
    assert result['_completed_submit'] is True
    assert calls['model'][0]['model'] == MODEL
    assert calls['model'][0]['tool_choice'] == {'type': 'function', 'function': {'name': 'submit_findings'}}
    assert not calls['search'] and not calls['primary'] and not calls['contents']
    assert result['claims']['industry']['status'] == 'VERIFIED'
    assert reviews[-1]['response_format']['json_schema']['name'] == 'industry_proposal'


def test_initial_match_does_not_override_review_of_changed_sources(monkeypatch):
    seen = []
    def final_missing(result):
        seen.append(result)
        if len(seen) > 1:
            result.update(decision='unavailable', missing_requirements=['Account workflows'])
    run, calls, _, _ = install(monkeypatch, mutation=final_missing)
    original_review = c._review_industry_sources

    async def review_changed_evidence(*args, **kwargs):
        if not kwargs.get('research'):
            update = 'https://www.coreworks.ai/product-update'
            kwargs['fetched_pages'][update] = 'Coreworks no longer supplies account workflows.'
            kwargs['fetched_final_urls'][update] = update
        return await original_review(*args, **kwargs)

    monkeypatch.setattr(c, '_review_industry_sources', review_changed_evidence)
    result = asyncio.run(run(prefetched_pages={URL: {'final_url': URL, 'text': FEATURES}},
                            prior_observations={'submitted_source_urls': [URL]}))
    assert result['_completed_submit'] is True
    assert result['claims']['industry']['status'] == 'UNPROVEN'
    assert not calls['search'] and not calls['primary']


def test_initial_industry_match_does_not_skip_other_requested_checks(monkeypatch):
    from tests.test_company_evidence_investigator import _attribute_finding
    run, calls, _, _ = install(monkeypatch, finding=[_finding('industry'), _attribute_finding()])
    original = c.investigate_company_evidence
    async def invoke(**kwargs):
        kwargs['targets'] = ('industry', 'required_attribute')
        return await original(**kwargs)
    monkeypatch.setattr(c, 'investigate_company_evidence', invoke)
    result = asyncio.run(run(prefetched_pages={URL: {'final_url': URL, 'text': FEATURES}},
                            prior_observations={'submitted_source_urls': [URL]}))
    assert calls['model'][0]['tool_choice'] == 'required'
    assert result['_completed_submit'] is True
    assert result['claims']['industry']['status'] == 'VERIFIED'
    assert result['claims']['required_attribute']['status'] == 'UNPROVEN'


def test_unproven_industry_review_cannot_leave_an_old_pass(monkeypatch):
    run, _, _, _ = install(monkeypatch, decision='unavailable', missing=['Voice support'])
    result = asyncio.run(run())
    finding = result['claims']['industry']
    assert finding['status'] == 'UNPROVEN'
    old = {'industry_matches': True, 'industry_activity_role': 'supplier_operator',
           'industry_evidence_url': URL, 'industry_evidence_quote': FEATURES,
           'dimension_evidence': {'industry': {'url': URL, 'quote': FEATURES}},
           'stage_matches': True}
    projected = lead_scorer._project_investigator_industry(old, finding)
    assert projected['industry_matches'] is None
    assert projected['stage_matches'] is True
    assert lead_scorer._dimension_web_evidence(projected, 'industry') == {'url': '', 'quote': ''}
    assert old['industry_matches'] is True


@pytest.mark.parametrize('decision', ['match', 'unavailable'])
def test_final_industry_review_replaces_a_rejected_legacy_quote(monkeypatch, decision):
    run, _, _, reviews = install(monkeypatch, decision=decision,
        missing=[] if decision == 'match' else ['Account workflow integration'],
        finding=_finding('industry', status='VERIFIED', evidence_url=URL,
            evidence_quote='An invented quote absent from every fetched page.',
            activity_role='supplier_operator', observed_industry='Software'))
    result = asyncio.run(run())
    assert reviews
    assert result['claims']['industry']['_industry_review']['decision'] == decision
    assert result['_completed_submit'] is True
    assert result['failure_reason'] == ''
    assert result['claims']['industry']['status'] == (
        'VERIFIED' if decision == 'match' else 'UNPROVEN')


def test_final_industry_review_preserves_unrelated_quote_failure(monkeypatch):
    from tests.test_company_evidence_investigator import _attribute_finding
    run, _, _, reviews = install(monkeypatch, finding=[
        _finding('industry'),
        _attribute_finding('VERIFIED', url=URL,
                           quote='An invented subscription quote absent from every source.'),
    ])
    original = c.investigate_company_evidence

    async def invoke(**kwargs):
        kwargs['targets'] = ('industry', 'required_attribute')
        return await original(**kwargs)

    monkeypatch.setattr(c, 'investigate_company_evidence', invoke)
    result = asyncio.run(run())
    assert reviews
    assert result['claims']['industry']['status'] == 'VERIFIED'
    assert result['claims']['required_attribute']['status'] == 'UNPROVEN'
    assert result['_completed_submit'] is False
    assert result['failure_reason'] == 'malformed_response'


@pytest.mark.parametrize('mutation', [
    lambda result: result['evidence_span_ids'].append('foreign-span'),
    lambda result: result.update(missing_requirements=['Unsupported capability']),
    lambda result: result.update(role='customer_user'),
])
def test_invalid_strong_positive_is_not_an_admission(monkeypatch, mutation):
    run, _, _, _ = install(monkeypatch, mutation=mutation)
    result = asyncio.run(run())
    assert result['failure_reason'] == 'malformed_response'
    assert not result.get('_completed_submit')


def test_reserved_industry_judgment_finishes_after_admitted_fetch(monkeypatch):
    run, calls, _, reviews = install(monkeypatch, case='deadline')
    result = asyncio.run(run())
    assert len(reviews) == result['usage']['industry_review_calls'] == 1
    assert result['_completed_submit'] is True
    assert result['failure_reason'] == ''
    assert len(calls['primary']) == 2 and len(calls['search']) == 1


@pytest.mark.parametrize('fault', ['truncated', 'invalid_json', 'missing_role', 'foreign_citation'])
def test_invalid_advisory_does_not_abort_valid_final_industry_review(monkeypatch, fault):
    run, calls, _, reviews = install(monkeypatch)
    original_post = c._post_json

    async def post(session, url, *, headers, payload):
        status, body = await original_post(session, url, headers=headers, payload=payload)
        if 'response_format' in payload and len(reviews) == 1:
            choice = body['choices'][0]
            if fault == 'truncated':
                choice['finish_reason'] = 'length'
            elif fault == 'invalid_json':
                choice['message']['content'] = '{'
            else:
                parsed = json.loads(choice['message']['content'])
                if fault == 'missing_role':
                    parsed.pop('role')
                else:
                    parsed['evidence_span_ids'] = ['foreign']
                choice['message']['content'] = json.dumps(parsed)
        return status, body

    monkeypatch.setattr(c, '_post_json', post)
    homepage = 'https://www.coreworks.ai/'
    result = asyncio.run(run(
        prior_observations={'submitted_source_urls': [homepage]},
        prefetched_pages={homepage: {'final_url': homepage, 'text': FEATURES}},
        requested_product_service='Supplies reporting and account workflows'))
    assert result.get('_completed_submit') is True
    review = result['claims']['industry']['_industry_review']
    assert result['claims']['industry']['status'] == 'VERIFIED'
    assert review['research_brief']['advisory_failure_reason'] == 'malformed_response'
    # An invalid advisory cannot erase source discovery for the original
    # criterion. It supplies neither a verdict nor partial model-generated gaps.
    assert review['research_brief']['decision'] == 'unavailable'
    assert review['research_brief']['research_query'].startswith(
        'Supplies reporting and account workflows')
    assert any('Supplies reporting and account workflows' in query
               for query in calls['search'])
    assert len(reviews) == result['usage']['industry_review_calls'] == 2
    assert result['usage']['reasoning_turns'] <= c.MAX_REASONING_TURNS
    assert result['usage']['fetch_calls'] <= c.MAX_FETCH_CALLS
    assert result['usage']['search_calls'] <= c.MAX_SEARCH_CALLS


def test_an_unbound_source_does_not_reach_strong_judge(monkeypatch):
    run, _, _, reviews = install(monkeypatch, identity={
        'normalized_name': 'Other', 'registrable_dns_domain': 'other.example'})
    result = asyncio.run(run())
    assert not reviews
    assert result['claims']['industry']['status'] == 'UNPROVEN'


def test_identity_navigation_metadata_alone_is_not_industry_evidence(monkeypatch):
    import time

    async def forbidden(*args, **kwargs):
        pytest.fail('Identity-only metadata must not be sent to the industry model')

    monkeypatch.setattr(c, '_post_json', forbidden)
    url = 'https://acme.example/'
    finding, failure, calls = asyncio.run(c._review_industry_sources(
        None, model=MODEL, openrouter_key='offline-key', started=time.monotonic(),
        company_locator={'name':'Acme','website':url},
        criteria={'industry':'Software','subindustry':'','product_service':'Reporting'},
        fetched_pages={url:c._IDENTITY_LINK_CONTEXT_MARKER + '\nhttps://acme.example/reporting'},
        fetched_final_urls={url:url}, first_party_domains={'acme.example'},
        identity_anchor={'submitted_domain':'acme.example','observed_domain':'acme.example',
                         'verified_domain':'acme.example'}, identity_names={'acme'}, evaluation_date_value='2026-10-24'))
    assert finding['status'] == 'UNPROVEN'
    assert calls == 0 and not failure


@pytest.mark.parametrize('enabled', [False, True])
def test_frozen_policy_controls_industry_model_and_blocks_ambient_override(enabled):
    models = dict(scoring.DEFAULT_JUDGE_MODELS)
    if not enabled:
        models.pop('industry_evidence_review')
    policy = scoring.build_scorer_policy(judge_models=models)
    credentials = {name: 'offline-secret' for name in scoring.CREDENTIAL_ENV_NAMES}
    environment = {}
    scoring.apply_policy_to_environment(policy, environ=environment, credentials=credentials)
    assert environment['ARENA_INDUSTRY_EVIDENCE_MODEL'] == (MODEL if enabled else '')
    with pytest.raises(scoring.ScorerPolicyConflict):
        scoring.apply_policy_to_environment(policy,
            environ={'ARENA_INDUSTRY_EVIDENCE_MODEL': 'another/model'}, credentials=credentials)


@pytest.mark.parametrize('model,expected', [('', False), (MODEL, True)])
def test_positive_industry_review_does_not_require_a_separate_attribute(monkeypatch, model, expected):
    from types import SimpleNamespace
    from qualification.scoring.company_fit_decision import company_fit_match
    monkeypatch.setenv('ARENA_INDUSTRY_EVIDENCE_MODEL', model)
    result = company_fit_match('All active company requirements passed', details={
        'dimension_decisions': {'industry': 'match'},
        'required_attribute_decision': 'not_required',
    })
    icp = SimpleNamespace(required_attribute='', industry='Software', product_service='Reporting')
    assert lead_scorer._positive_semantic_review_needed(result, icp, ()) is expected


def test_new_industry_review_preserves_an_independent_final_rejection(monkeypatch):
    from types import SimpleNamespace
    from qualification.scoring.company_fit_decision import company_fit_mismatch
    monkeypatch.setenv('ARENA_INDUSTRY_EVIDENCE_MODEL', MODEL)
    result = company_fit_mismatch('Country mismatch', details={
        'dimension_decisions': {'industry': 'match', 'geography': 'mismatch'},
        'required_attribute_decision': 'not_required',
    })
    assert lead_scorer._positive_semantic_review_needed(
        result, SimpleNamespace(required_attribute=''), ()) is False


@pytest.mark.parametrize('support_count', [1, 8])
def test_cached_judgment_preserves_all_industry_sources_and_separates_frozen_policies(monkeypatch, support_count):
    from lab_arena import company_judgments, contracts
    from tests.lab_arena.company_judgments_test import _input, _refs, _raw, _evidence
    document = _input()
    current = _refs(document)[0]
    old_document = json.loads(json.dumps(document))
    old_document['scorer_policy']['judge_models'].pop('industry_evidence_review')
    previous = _refs(old_document)[0]
    assert current['cache_key'] != previous['cache_key']
    raw = _raw()
    evidence = {'url': URL, 'quote': FEATURES, 'supporting_evidence': [
        {'url': f'https://www.coreworks.ai/workflows/{i}', 'quote': 'Coreworks connects account workflows.'}
        for i in range(support_count)]}
    assert lead_scorer._dimension_web_evidence({'dimension_evidence': {'industry': evidence}}, 'industry') == evidence
    receipt = raw['verifier_gate_receipts'][0]
    receipt['dimension_evidence']['industry'] = evidence
    receipt['investigation_receipt'] = {'claims': {'industry': {'_industry_review': {
        'decision': 'match', 'evidence': [
            {'url': URL, 'quote': FEATURES}, *evidence['supporting_evidence']]}}}}
    # Use the unchanged native cache envelope, with its normal provenance checks.
    import tests.lab_arena.company_judgments_test as fixtures
    monkeypatch.setattr(fixtures, '_raw', lambda score=81.0: raw)
    snapshot = _evidence(current)
    restored = company_judgments.validate_evidence_snapshot(
        json.loads(json.dumps(snapshot)), cache_key=current['cache_key'],
        company_input_hash=current['company_input_hash'], authority_slot=0,
        evidence_hash=contracts.document_hash(snapshot))
    saved_receipt = restored['raw_judgment']['verifier_gate_receipts'][0]
    assert saved_receipt['dimension_evidence']['industry'] == evidence
    assert len(saved_receipt['investigation_receipt']['claims']['industry']['_industry_review']['evidence']) == 1 + support_count


@pytest.mark.parametrize('decision,expected', [('match', 'match'), ('unavailable', 'unavailable')])
@pytest.mark.parametrize('identity_footer', [False, True])
def test_full_company_reverification_retains_multi_source_industry_receipt(monkeypatch, decision, expected, identity_footer):
    """Run the real caller, source review, projection, fit result and JSON receipt."""
    import time
    from tests.test_company_evidence_investigator import _company, _icp, _complete_verdict
    from qualification.scoring.company_fit_decision import company_fit_match
    from lab_arena.contracts import check_strict_document
    monkeypatch.setenv('ARENA_INDUSTRY_EVIDENCE_MODEL', MODEL)
    monkeypatch.setenv('OPENROUTER_API_KEY', 'offline-key')
    pages = {'https://acme.example/tickets': 'Acme supplies ticket management software.',
             'https://acme.example/accounts': 'Acme connects customer accounts to its ticket workflows.'}
    visible_pages = dict(pages)
    if identity_footer:
        pages = {url: text + '\n' + c._IDENTITY_LINK_CONTEXT_MARKER + '\nhttps://acme.example/'
                 for url, text in pages.items()}
    calls = []

    async def provider(**kwargs):
        return _complete_verdict(), ''

    async def keep_size(verdict, *args, **kwargs):
        return verdict

    async def review_post(session, url, *, headers, payload):
        calls.append(payload)
        ids = [span['span_id'] for msg in payload['messages'][2:]
               for page in json.loads(msg['content'])['untrusted_source_pages']
               for span in page['spans']]
        response = {'observed_activity': 'Ticket and account workflow software',
                    'reason': 'The two company pages describe the connected offering.',
                    'missing_requirements': [] if decision == 'match' else ['Account workflow integration'],
                    'role': 'supplier_operator', 'evidence_span_ids': ids, 'decision': decision}
        return 200, {'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(response)}}]}

    async def investigate(**kwargs):
        assert kwargs['targets'] == ('industry',)
        finding, failure, count = await c._review_industry_sources(
            None, model=MODEL, openrouter_key='offline-key', started=time.monotonic(),
            company_locator=kwargs['company_locator'],
            criteria={'industry': kwargs['requested_industry'],
                      'subindustry': kwargs['requested_subindustry'],
                      'product_service': kwargs['requested_product_service']},
            fetched_pages=pages, fetched_final_urls={u:u for u in pages},
            first_party_domains={'acme.example'}, identity_names={'acme'},
            identity_anchor={'submitted_domain':'acme.example','observed_domain':'acme.example',
                             'verified_domain':'acme.example','submitted_name':'Acme',
                             'observed_name':'Acme','verified_name':'Acme'}, evaluation_date_value='2026-10-24')
        assert not failure and count == 1
        return {'claims': {'industry': finding}, '_completed_submit': True,
                'failure_reason': '', 'usage': {'reasoning_turns': 1, 'search_calls': 0, 'fetch_calls': 0},
                c.PRIVATE_FETCHED_PAGES_KEY: {u:{'final_url':u,'text':s} for u,s in pages.items()}}

    monkeypatch.setattr(lead_scorer, '_request_company_reverify_json', provider)
    monkeypatch.setattr(lead_scorer, '_refresh_linkedin_employee_size_observation', keep_size)
    monkeypatch.setattr(lead_scorer, 'investigate_company_evidence', investigate)
    monkeypatch.setattr(c, '_post_json', review_post)
    identity = company_fit_match('Fixture homepage binding', details={
        'identity': {'decision':'match','evidence_source':'company_homepage','observed_name':'acme',
                     'observed_domain':'acme.example','observed_linkedin_slug':'acme'},
        'verified_homepage_transport_domain':'acme.example'})
    result = asyncio.run(lead_scorer._llm_reverify_company(
        _company(), _icp(product_service='Supplies ticket and account workflow software.'),
        require_company_fit_dimensions=True, evidence_investigator=True,
        verified_homepage_identity=identity))
    stored = json.loads(json.dumps(result.receipt('company_fit')))
    check_strict_document(stored)
    assert stored['decision'] == expected
    assert stored['dimension_decisions']['industry'] == expected
    assert stored['dimension_decisions']['employee_size'] == 'match'
    assert stored['dimension_decisions']['geography'] == 'match'
    evidence = stored['dimension_evidence']['industry']
    if decision == 'match':
        assert evidence['quote'] == visible_pages['https://acme.example/tickets']
        assert evidence['supporting_evidence'] == [
            {'url':'https://acme.example/accounts','quote':visible_pages['https://acme.example/accounts']}]
    else:
        assert not evidence['quote']
    assert len(stored['investigation_receipt']['claims']['industry']['_industry_review']['evidence']) == 2
    assert len(calls) == 1


def test_industry_source_brief_exposes_the_exact_reviewed_source_set(monkeypatch):
    """A source usable for another target must not silently close an industry gap."""
    assessments = []

    def missing(result):
        assessments.append(result)
        if len(assessments) == 1:
            result.update(decision='unavailable', missing_requirements=['Account workflows'])

    run, calls, _, reviews = install(monkeypatch, mutation=missing)
    original_post = c._post_json

    async def post(session, url, *, headers, payload):
        if ('response_format' not in payload and payload.get('tool_choice') == {
                'type': 'function', 'function': {'name': 'search_web'}}):
            from tests.test_company_evidence_judgment_reserve import _response
            calls['model'].append(payload)
            return 200, _response('search_web', {'query': 'Coreworks account workflows'})
        return await original_post(session, url, headers=headers, payload=payload)

    async def search(*args, **kwargs):
        return {'results': [{'url': URL, 'title': 'Workflows'}],
                'notice': 'discovery_only_not_evidence'}

    monkeypatch.setattr(c, '_post_json', post)
    monkeypatch.setattr(c, '_search_web', search)
    homepage = 'https://www.coreworks.ai/'
    marketplace = 'https://marketplace.example/coreworks'
    result = asyncio.run(run(
        prior_observations={'submitted_source_urls': [homepage, marketplace]},
        prefetched_pages={
            homepage: {'final_url': homepage, 'text': FEATURES},
            marketplace: {'final_url': marketplace,
                          'text': 'Coreworks supplies account workflows through this listing.'},
        }))
    document_text = next(m['content'] for m in calls['model'][0]['messages'] if m['role'] == 'user')
    document = json.loads(document_text[document_text.index('{'):])
    brief = document['industry_source_research_brief']
    assert brief['source_urls'] == [homepage]
    assert brief['excluded_source_urls'] == [marketplace]
    assert 'remain available for other targets' in brief['notice']
    # The original loaded page survives for independent attribute/stage checks.
    assert any(page['url'] == marketplace for page in document['prefetched_sources'])
    reviewed_urls = {page['source_url'] for request in reviews
                    for message in request['messages'][2:]
                    for page in json.loads(message['content'])['untrusted_source_pages']}
    assert marketplace not in reviewed_urls
    assert result['_completed_submit']
    assert result['claims']['industry']['status'] == 'VERIFIED'
    assert result['usage']['reasoning_turns'] <= c.MAX_REASONING_TURNS


@pytest.mark.parametrize('cached', [False, True])
def test_existing_industry_source_is_reviewed_only_when_already_loaded(monkeypatch, cached):
    run, calls, _, reviews = install(monkeypatch)
    homepage = 'https://www.coreworks.ai/'
    industry_url = 'https://www.coreworks.ai/workflows'
    pages = {homepage: {'final_url': homepage, 'text': FEATURES}}
    if cached:
        pages[industry_url] = {'final_url': industry_url, 'text': FEATURES}
    result = asyncio.run(run(
        prior_observations={'submitted_source_urls': list(pages),
                            'industry_evidence_url': industry_url,
                            'industry_evidence_quote': 'Untrusted prior assertion.'},
        prefetched_pages=pages))
    initial_sources = [page for msg in reviews[0]['messages'][2:]
                       for page in json.loads(msg['content'])['untrusted_source_pages']]
    assert (industry_url in {p['source_url'] for p in initial_sources}) is cached
    assert industry_url not in calls['primary']
    assert 'Untrusted prior assertion.' not in json.dumps(initial_sources)
    document_text = next(m['content'] for m in calls['model'][0]['messages'] if m['role'] == 'user')
    document = json.loads(document_text[document_text.index('{'):])
    assert any(p['url'] == industry_url for p in document['prefetched_sources']) is cached
    assert document['prior_observations']['industry_evidence_url'] == industry_url
    assert document['investigation_limits']['remaining_fetch_calls'] == c.MAX_FETCH_CALLS
    assert result['_completed_submit']
    assert result['usage']['fetch_calls'] <= c.MAX_FETCH_CALLS


@pytest.mark.parametrize('case', ['old_policy', 'third_party', 'unbound_identity', 'deadline', 'fetch_cap'])
def test_existing_industry_locator_prefetch_keeps_identity_policy_and_budgets(monkeypatch, case):
    run, calls, clock, _ = install(monkeypatch)
    industry_url = ('https://marketplace.example/coreworks' if case == 'third_party'
                    else 'https://www.coreworks.ai/workflows')
    if case == 'old_policy':
        monkeypatch.delenv('ARENA_INDUSTRY_EVIDENCE_MODEL')
    if case == 'unbound_identity':
        original = c.investigate_company_evidence

        async def unbound(**kwargs):
            kwargs['verified_homepage_identity'] = {}
            return await original(**kwargs)

        monkeypatch.setattr(c, 'investigate_company_evidence', unbound)
    if case in {'deadline', 'fetch_cap'}:
        original_fetch = c._fetch_page

        async def consume_priority(*args, **kwargs):
            result = await original_fetch(*args, **kwargs)
            if case == 'deadline':
                clock.now = c.ADMISSION_DEADLINE_SECONDS - c.JUDGMENT_ADMISSION_RESERVE_SECONDS
            return result

        monkeypatch.setattr(c, '_fetch_page', consume_priority)
        if case == 'fetch_cap':
            monkeypatch.setattr(c, 'MAX_FETCH_CALLS', 1)
    prior = {'industry_evidence_url': industry_url}
    if case in {'deadline', 'fetch_cap'}:
        prior['submitted_source_urls'] = ['https://www.coreworks.ai/product']
    asyncio.run(run(prior_observations=prior))
    assert industry_url not in calls['primary']
    if case in {'deadline', 'fetch_cap'}:
        assert calls['primary'][0] == 'https://www.coreworks.ai/product'


@pytest.mark.parametrize('failed', [False, True])
def test_existing_industry_locator_does_not_trust_failure_or_cross_company_redirect(monkeypatch, failed):
    run, calls, _, reviews = install(monkeypatch)
    industry_url = 'https://www.coreworks.ai/workflows'
    original_fetch = c._fetch_page

    async def fetch(session, url, **kwargs):
        if url != industry_url:
            return await original_fetch(session, url, **kwargs)
        calls['primary'].append(url)
        return ({'ok': False, 'error': 'http_403'} if failed else {
            'ok': True, 'url': url, 'final_url': 'https://other.example/workflows',
            'text': 'Other Company supplies all required software.'})

    original_post = c._post_json
    selected = False
    async def choose(session, url, *, headers, payload):
        nonlocal selected
        if 'response_format' not in payload and not selected:
            from tests.test_company_evidence_judgment_reserve import _response
            selected = True
            return 200, _response('fetch_page', {'url': industry_url})
        return await original_post(session, url, headers=headers, payload=payload)
    monkeypatch.setattr(c, '_post_json', choose)
    monkeypatch.setattr(c, '_fetch_page', fetch)
    result = asyncio.run(run(prior_observations={'industry_evidence_url': industry_url}))
    reviewed_urls = {page['source_url'] for request in reviews
                     for message in request['messages'][2:]
                     for page in json.loads(message['content'])['untrusted_source_pages']}
    assert industry_url not in reviewed_urls
    assert calls['primary'].count(industry_url) == 1
    assert result['usage']['fetch_calls'] <= c.MAX_FETCH_CALLS
    assert len(result['usage']['fetch_outcomes']) == len(calls['primary'])


@pytest.mark.parametrize('source_kind', ['bound_redirect', 'third_party_redirect', 'different_final_url'])
def test_retained_industry_redirects_keep_the_original_request_binding(monkeypatch, source_kind):
    run, calls, _, reviews = install(monkeypatch)
    requested_url = ('https://other.example/article' if source_kind == 'third_party_redirect'
                     else 'https://coreworks.ai/')
    industry_url = 'https://www.coreworks.ai/'
    final_url = (industry_url + '?campaign=other' if source_kind == 'different_final_url'
                 else industry_url)
    result = asyncio.run(run(
        prior_observations={'submitted_source_urls': [requested_url],
                            'industry_evidence_url': industry_url},
        prefetched_pages={requested_url: {'final_url': final_url, 'text': FEATURES}}))
    reused = source_kind == 'bound_redirect'
    assert industry_url not in calls['primary']
    first_sources = {page['source_url'] for message in reviews[0]['messages'][2:]
                     for page in json.loads(message['content'])['untrusted_source_pages']}
    if reused:
        assert requested_url in first_sources
        assert industry_url not in first_sources  # Keep the real request binding.
        assert industry_url not in result[c.PRIVATE_FETCHED_PAGES_KEY]
    else:
        assert industry_url not in first_sources
        assert (requested_url in first_sources) is (source_kind != 'third_party_redirect')
    assert result['_completed_submit']
    assert result['usage']['fetch_calls'] <= c.MAX_FETCH_CALLS
