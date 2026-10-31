"""Industry-only conditional provenance; semantic model replies are hermetic.

These checks prove source integrity/control flow, not live ownership semantics.
"""
import asyncio
import copy
import hashlib
import json

import pytest

from qualification.scoring import company_evidence_investigator as c
from qualification.scoring import industry_classification as industry, lead_scorer
from tests.test_company_evidence_investigator import _finding
from tests.test_company_evidence_judgment_reserve import _response
from tests.test_native_industry_review import install


PARENT = 'https://www.coreworks.ai/brands'
HOME = 'https://workflowco.example/'
CHILD = HOME + 'product'
RELATION = 'Coreworks operates its WorkflowCo brand as part of our connected commercial software offering.'
FACT = 'WorkflowCo supplies account workflow management software to commercial customers.'
PROOF = {'parent_source_url': PARENT, 'parent_quote': RELATION,
         'child_brand': 'WorkflowCo', 'official_child_homepage': HOME}


def native_loop(monkeypatch, *, proof=PROOF, parent_text=RELATION, href=HOME,
                child_text=FACT, child_final=None, decision='match', omit_relation=False,
                child_first=False):
    monkeypatch.setenv('LEADPOET_COMPETITION_EVALUATION_DATE', '2026-10-24')
    run, _, _, _ = install(monkeypatch)
    reads, research, reviews = [], [], []
    child_final = CHILD if child_final is None else child_final
    plan = [('fetch_page', {'url': PARENT, 'industry_brand_source': None}),
            ('fetch_page', {'url': CHILD, 'industry_brand_source': copy.deepcopy(proof)}),
            ('submit_findings', {'findings': [_finding('industry', status='VERIFIED',
                evidence_url=PARENT, evidence_quote=parent_text if len(parent_text) <= 2000 else RELATION,
                activity_role='supplier_operator',
                observed_industry='Software')]})]

    async def html(session, url, **kwargs):
        target = kwargs.get('params', {}).get('url', url)
        reads.append(target)
        if target == PARENT:
            link = '<a href="' + href + '">WorkflowCo</a>' if href else ''
            return 200, PARENT, '<main>' + parent_text + link + '</main>'
        assert target == CHILD
        return 200, child_final, '<main>' + child_text + '</main>'

    async def search(*args, **kwargs):
        return {'results': [], 'notice': 'discovery_only_not_evidence'}

    async def post(session, url, *, headers, payload):
        if 'response_format' not in payload:
            research.append(payload)
            name, args = plan.pop(0) if plan else ('submit_findings', {'findings': [
                _finding('industry', status='UNPROVEN', evidence_url='', evidence_quote='')]})
            return 200, _response(name, args)
        reviews.append(payload)
        pages = [page for message in payload['messages'][2:]
                 for page in json.loads(message['content'])['untrusted_source_pages']]
        available = any(page['source_url'] == CHILD for page in pages)
        parent_spans = [span for page in pages if page['source_url'] == PARENT for span in page['spans']]
        parent_body = ''.join(span['text'] for span in parent_spans)
        parent_quote = proof.get('parent_quote', RELATION) if isinstance(proof, dict) else RELATION
        quote_start = parent_body.find(parent_quote)
        start = len(parent_body[:quote_start].encode()) if quote_start >= 0 else 0
        end = start + len(parent_quote.encode())
        selected = [span['span_id'] for span in parent_spans
                    if not omit_relation and int(span['span_id'].rsplit(':', 2)[2]) > start
                    and int(span['span_id'].rsplit(':', 2)[1]) < end]
        selected += [span['span_id'] for page in pages if page['source_url'] == CHILD for span in page['spans']]
        selected = list(dict.fromkeys(selected))
        if child_first:
            selected = sorted(selected, key=lambda key: next(
                page['source_url'] != CHILD for page in pages for span in page['spans']
                if span['span_id'] == key))
        actual_decision = decision if available else 'unavailable'
        result = {'decision': actual_decision, 'role': 'supplier_operator',
            'observed_activity': 'Connected commercial software',
            'reason': 'Assess the supplied parent relationship and child offering together.',
            'missing_requirements': [] if actual_decision == 'match' else ['Connected operating relationship'],
            'evidence_span_ids': selected[:8]}
        if 'research_query' in payload['response_format']['json_schema']['schema']['properties']:
            result['research_query'] = '' if actual_decision == 'match' else 'operating relationship'
        return 200, {'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(result)}}]}

    monkeypatch.setattr(c, '_fetch_bounded_html', html)
    monkeypatch.setattr(c, '_search_web', search)
    monkeypatch.setattr(c, '_post_json', post)
    result = asyncio.run(run(requested_product_service='Supplies analysis, reporting and account workflows',
        prior_observations={'observed_company_name': 'Coreworks',
            'observed_company_website': 'https://www.coreworks.ai/',
            'observed_company_linkedin': 'https://www.linkedin.com/company/coreworks-ai'}))
    return result, reads, research, reviews


def test_native_parent_official_link_and_child_reach_same_final_industry_packet(monkeypatch):
    result, reads, research, reviews = native_loop(monkeypatch)
    assert reads == [PARENT, CHILD]
    assert result['claims']['industry']['status'] == 'VERIFIED'
    assert result['_completed_submit'] is True
    assert len(research) == 3 and len(reviews) == 1
    assert result['usage']['fetch_calls'] == 2 <= c.MAX_FETCH_CALLS
    assert result['usage']['search_calls'] == 1
    assert c._industry_source_is_bound(CHILD, CHILD, domains={'coreworks.ai'}, identity={
        'submitted_domain': 'coreworks.ai', 'observed_domain': 'coreworks.ai'}) is False
    evidence = result['claims']['industry']['_industry_review']['evidence']
    assert any(item['url'] == PARENT and RELATION in item['quote'] for item in evidence)
    assert any(item['url'] == CHILD and FACT in item['quote'] for item in evidence)
    from lab_arena.operations import validate_operation_request
    for request in [*research, *reviews]:
        validate_operation_request('openrouter.chat', request)


@pytest.mark.parametrize('change', ['quote', 'quote_case', 'link', 'foreign_home', 'private_home', 'missing_proof'])
def test_unproved_parent_link_or_quote_cannot_authorize_industry_child(monkeypatch, change):
    proof = copy.deepcopy(PROOF)
    href = HOME
    if change == 'quote': proof['parent_quote'] = 'Invented current ownership claim.'
    if change == 'quote_case': proof['parent_quote'] = RELATION.lower()
    if change == 'link': href = ''
    if change == 'foreign_home': proof['official_child_homepage'] = 'https://other.example/'
    if change == 'private_home': proof['official_child_homepage'] = 'https://127.0.0.1/'
    if change == 'missing_proof': proof = None
    result, reads, _, _ = native_loop(monkeypatch, proof=proof, href=href)
    assert result['claims']['industry']['status'] != 'VERIFIED'
    if change != 'missing_proof':
        assert reads == [PARENT]


@pytest.mark.parametrize('change', ['foreign_redirect', 'wrong_brand', 'lookalike_domain'])
def test_child_source_must_retain_exact_brand_scope_and_self_identity(monkeypatch, change):
    options = {'child_final': 'https://other.example/product'} if change == 'foreign_redirect' else {
        'child_text': 'UnrelatedCo supplies account workflows.'} if change == 'wrong_brand' else {
        'child_text': 'notworkflowco.example supplies account workflows.'}
    result, _, _, _ = native_loop(monkeypatch, **options)
    assert result['claims']['industry']['status'] != 'VERIFIED'


def test_child_printed_exact_official_domain_is_identity_not_parent_rebrand(monkeypatch):
    result, _, _, _ = native_loop(monkeypatch,
        child_text='WorkflowLegacy product documentation. WorkflowCo.example supplies account workflows.')
    assert result['claims']['industry']['status'] == 'VERIFIED'
    assert 'rebrand' not in result['claims']


def test_child_positive_without_model_selected_relation_citation_fails_closed(monkeypatch):
    result, _, _, _ = native_loop(monkeypatch, omit_relation=True)
    assert result['claims']['industry']['status'] != 'VERIFIED'
    assert result['failure_reason'] == 'malformed_response'


@pytest.mark.parametrize('printed', [False, True])
@pytest.mark.parametrize('relation', [
    'Coreworks invests in its portfolio company WorkflowCo.',
    'WorkflowCo acquired Coreworks; WorkflowCo is now the parent.',
    'Coreworks plans to acquire WorkflowCo after the evaluation date.',
    'Coreworks divested WorkflowCo before the evaluation date.',
    'WorkflowCo is an independent partner and dealer, not a Coreworks operating brand.',
    'WorkflowCo is a competitor of Coreworks, not part of its supplied offering.',
])
def test_provenance_cannot_override_final_relationship_semantics(monkeypatch, relation, printed):
    if printed:
        relation += ' WorkflowCo homepage: ' + HOME
    proof = {**PROOF, 'parent_quote': relation}
    result, _, _, reviews = native_loop(monkeypatch, proof=proof, parent_text=relation,
        href='' if printed else HOME, decision='unavailable')
    assert result['claims']['industry']['status'] == 'UNPROVEN'
    assert result['claims']['industry']['_industry_review']['decision'] == 'unavailable'
    assert CHILD in json.dumps(reviews[-1])  # It is the final model control, not a lexical veto.


@pytest.mark.parametrize('scheme', ['http', 'https'])
def test_complete_printed_reference_is_conditional_not_an_observed_hyperlink(monkeypatch, scheme):
    printed = HOME.replace('https:', scheme + ':')
    relation = RELATION + ' Official WorkflowCo homepage: ' + printed
    proof = {**PROOF, 'parent_quote': relation}
    result, reads, _, reviews = native_loop(monkeypatch, proof=proof, parent_text=relation, href='')
    assert result['claims']['industry']['status'] == 'VERIFIED'
    assert reads == [PARENT, CHILD]
    bindings = result[c.PRIVATE_INDUSTRY_BINDINGS_KEY]
    assert bindings['links'] == {}
    assert bindings['brands'][CHILD]['pointer_kind'] == 'printed_reference'
    assert bindings['brands'][CHILD]['printed_reference'] == printed
    assert json.loads(reviews[-1]['messages'][1]['content'])['conditional_industry_sources'][0][
        'pointer_kind'] == 'printed_reference'


@pytest.mark.parametrize('reference', ['workflowco.example', 'contact@workflowco.example',
    'https://workflowco.example/wrong-path', 'https://other.example/',
    'javascript:location=workflowco.example', 'https://workflowco.example/?tenant=other'])
def test_printed_reference_cannot_infer_or_change_url_components(monkeypatch, reference):
    relation = RELATION + ' WorkflowCo contact: ' + reference
    result, _, _, _ = native_loop(monkeypatch, proof={**PROOF, 'parent_quote': relation},
                                  parent_text=relation, href='')
    assert result['claims']['industry']['status'] != 'VERIFIED'


@pytest.mark.parametrize('home,target', [
    ('https://shared.example/brand-one', 'https://shared.example/brand-two/product'),
    ('https://shared.example/?tenant=one', 'https://shared.example/?tenant=two'),
    ('https://workflowco.example/', 'https://workflowco.example.evil.example/product'),
    ('https://workflowco.example/', 'https://other.workflowco.example/product'),
])
def test_official_brand_scope_never_trusts_other_tenant_or_host(home, target):
    assert c._industry_brand_scope(target, home) is False


def fit_and_cache(investigation):
    from tests.test_matched_company_retry_source_continuity import _fit_result
    fit = _fit_result()
    identity = fit.details['dimension_evidence']['identity']['web_identity_receipt']
    identity.update(observed_name='Coreworks', observed_domain='coreworks.ai',
                    observed_linkedin_slug='coreworks-ai')
    projection = lead_scorer._project_investigator_industry({}, investigation['claims']['industry'])
    fit.details['dimension_evidence']['industry']['web_evidence'] = lead_scorer._dimension_web_evidence(
        projection, 'industry')
    sink, cache = {}, {}
    lead_scorer._retain_matched_investigator_source_contexts(sink, fit, investigation)
    lead_scorer._retain_unresolved_industry_retry_sources(cache, fit, investigation)
    lead_scorer._retain_matched_company_retry_sources(cache, sink, fit)
    return fit, sink, cache


IDENTITY = {'normalized_name': 'Coreworks', 'registrable_dns_domain': 'coreworks.ai',
            'linkedin_company_slug': 'coreworks-ai'}


@pytest.mark.parametrize('printed', [False, True])
def test_existing_retry_cache_preserves_sources_and_reclassifies_with_bound_witness(monkeypatch, printed):
    relation = RELATION + (' WorkflowCo homepage: ' + HOME if printed else '')
    result, reads, _, reviews = native_loop(monkeypatch, parent_text=relation,
        proof={**PROOF, 'parent_quote': relation}, href='' if printed else HOME)
    fit, sink, cache = fit_and_cache(result)
    bindings = {}
    pages = lead_scorer._matched_company_retry_prefetched_pages(cache, IDENTITY,
        industry_bindings_out=bindings)
    assert set(pages) == {PARENT, CHILD} == set(sink)
    assert bindings['brands'][CHILD]['pointer_kind'] == ('printed_reference' if printed else 'observed_href')
    before_reviews = len(reviews)
    retry = asyncio.run(c.investigate_company_evidence(
        company_locator={'name': 'Coreworks', 'website': 'https://www.coreworks.ai/'},
        targets=('industry',), requested_industry='Software',
        requested_product_service='Supplies analysis, reporting and account workflows',
        verified_homepage_identity=IDENTITY, positive_semantic_review=True,
        prior_observations={'submitted_source_urls': [PARENT, CHILD]},
        prefetched_pages=pages, retained_industry_bindings=bindings))
    assert reads == [PARENT, CHILD]  # No fresh fetch was fabricated on retry.
    assert len(reviews) == before_reviews + 1  # Facts are reviewed; no cached fit flag.
    assert retry['claims']['industry']['status'] == 'VERIFIED'
    assert retry[c.PRIVATE_INDUSTRY_BINDINGS_KEY]['brands'][CHILD] == bindings['brands'][CHILD]
    assert identity_from_fit(fit)['observed_domain'] == 'coreworks.ai'


def identity_from_fit(fit):
    return fit.details['dimension_evidence']['identity']['web_identity_receipt']


@pytest.mark.parametrize('change', ['parent_text', 'child_text', 'parent_missing', 'date',
                                   'parent_identity', 'pointer_bytes', 'foreign_final'])
def test_retry_cannot_reuse_changed_or_missing_relationship_authority(monkeypatch, change):
    relation = RELATION + ' WorkflowCo homepage: http://workflowco.example/'
    result, _, _, _ = native_loop(monkeypatch, parent_text=relation, href='',
        proof={**PROOF, 'parent_quote': relation})
    _, _, cache = fit_and_cache(result)
    if change == 'parent_text': cache['pages'][PARENT]['text'] += ' Changed.'
    if change == 'child_text': cache['pages'][CHILD]['text'] += ' Changed.'
    if change == 'parent_missing': del cache['pages'][PARENT]
    if change == 'date': monkeypatch.setenv('LEADPOET_COMPETITION_EVALUATION_DATE', '2026-10-25')
    if change == 'parent_identity': cache[c.PRIVATE_INDUSTRY_BINDINGS_KEY]['parent_domain'] = 'other.example'
    if change == 'pointer_bytes':
        cache[c.PRIVATE_INDUSTRY_BINDINGS_KEY]['brands'][CHILD]['printed_reference'] = HOME
    if change == 'foreign_final': cache['pages'][CHILD]['final_url'] = 'https://other.example/product'
    bindings = {}
    lead_scorer._matched_company_retry_prefetched_pages(cache, IDENTITY, industry_bindings_out=bindings)
    assert not bindings.get('brands')


def test_parent_relationship_and_child_quotes_reach_existing_paragraph_handoff(monkeypatch):
    from qualification.scoring import intent_details
    from tests.test_intent_details_industry_support_handoff import _inputs
    result, _, _, _ = native_loop(monkeypatch)
    fit, sink, _ = fit_and_cache(result)
    contexts = lead_scorer._matched_investigator_source_contexts(fit, sink, include_industry_support=True)
    assert {entry['url'] for entry in contexts} == {PARENT, CHILD}
    company, icp, signals, _ = _inputs()
    company.company_name = 'Coreworks'
    company.company_website = 'https://www.coreworks.ai/'
    company.intent_details = FACT + ' ' + RELATION
    icp.prompt, icp.product_service = 'Commercial software', 'Account workflows'
    selected = lead_scorer._matched_company_source_contexts(fit, None, None, sink,
        paragraph=company.intent_details, company=company)
    assert len(selected or []) <= 2
    assert sum(context['dimension'] == 'industry' for context in selected or []) <= 1
    document = intent_details.review_evidence(company, icp, signals, fit.receipt('company_fit'),
        company_source_contexts=selected)
    encoded = json.dumps(document)
    assert RELATION in encoded and FACT in encoded
    assert HOME in result[c.PRIVATE_FETCHED_PAGES_KEY][PARENT]['text']
    assert identity_from_fit(fit)['observed_name'] == 'Coreworks'
    assert 'headcount' not in result['claims'] and 'rebrand' not in result['claims']


def test_packet_hash_binds_pointer_and_missing_bounded_parent_drops_child(monkeypatch):
    result, _, _, _ = native_loop(monkeypatch)
    bindings = result[c.PRIVATE_INDUSTRY_BINDINGS_KEY]
    sources = result[c.PRIVATE_FETCHED_PAGES_KEY]
    pages = [{'source_id': f's{i}', 'source_url': url, 'body': c._visible_quote_surface(page['text']).strip()}
             for i, (url, page) in enumerate(sources.items())]
    criterion = {'industry': 'Software', 'subindustry': '', 'product_service': 'Account workflows'}
    company = {'name': 'Coreworks', 'website': 'https://www.coreworks.ai/'}
    original = industry.prepare(criterion, company, pages, evaluation_date='2026-10-24',
        conditional_sources=bindings['brands'])
    changed = copy.deepcopy(bindings['brands'])
    changed[CHILD]['child_body_sha256'] = 'a' * 64
    other = industry.prepare(criterion, company, pages, evaluation_date='2026-10-24',
        conditional_sources=changed)
    assert original.digest != other.digest
    # An identity-only parent fragment cannot carry the relationship.
    cropped = [{**p, 'body': 'Coreworks is an operating company.'} if p['source_url'] == PARENT else p
               for p in pages]
    missing = industry.prepare(criterion, company, cropped, evaluation_date='2026-10-24',
        conditional_sources=bindings['brands'])
    assert CHILD not in {url for _, _, url in missing.pages}
    assert not missing.conditional_sources


def test_identity_only_parent_citation_cannot_replace_selected_relation(monkeypatch):
    parent_text = ('Coreworks is an operating company. ' * 70) + RELATION
    result, _, _, reviews = native_loop(monkeypatch, parent_text=parent_text)
    assert result['claims']['industry']['status'] == 'VERIFIED'
    bindings = result[c.PRIVATE_INDUSTRY_BINDINGS_KEY]
    sources = result[c.PRIVATE_FETCHED_PAGES_KEY]
    packet = industry.prepare({'industry': 'Software', 'subindustry': '', 'product_service': 'Workflows'},
        {'name': 'Coreworks', 'website': 'https://www.coreworks.ai/'},
        [{'source_id': f's{i}', 'source_url': url, 'body': c._visible_quote_surface(page['text']).strip()}
         for i, (url, page) in enumerate(sources.items())], evaluation_date='2026-10-24',
        conditional_sources=bindings['brands'])
    spans = packet.spans()
    identity = next(key for key, span in spans.items() if span['source_url'] == PARENT
                    and RELATION not in span['text'])
    child = next(key for key, span in spans.items() if span['source_url'] == CHILD)
    with pytest.raises(ValueError, match='selected parent relationship'):
        industry.resolve(packet, {'decision': 'match', 'role': 'supplier_operator',
            'observed_activity': 'Software', 'reason': 'Uses child facts.', 'missing_requirements': [],
            'evidence_span_ids': [identity, child]})


def maximum_conditional_packet():
    pages, candidates = [], {}
    for i in range(4):
        parent = f'https://p{i}.coreworks.ai/'
        parent += 'p' * (2000 - len(parent))
        child = f'https://brand{i}.example/'
        child += 'c' * (2000 - len(child))
        brand = 'Brand' + str(i) + 'B' * 194
        prefix = 'Coreworks operates ' + brand + '. '
        quote = prefix + '\x01' * (2000 - len(prefix))
        body = brand + ' supplies account workflows.'
        pages += [{'source_id': 'p' + str(i) + 'p' * 118, 'source_url': parent, 'body': quote},
                  {'source_id': 'c' + str(i) + 'c' * 118, 'source_url': child, 'body': body}]
        candidates[child] = {'parent_source_url': parent, 'parent_quote': quote, 'child_brand': brand,
            'official_child_homepage': child, 'parent_body_sha256': hashlib.sha256(quote.encode()).hexdigest(),
            'child_body_sha256': hashlib.sha256(body.encode()).hexdigest(), 'child_final_url': child,
            'pointer_kind': 'observed_href', 'printed_reference': ''}
    packet = industry.prepare({'industry': 'I' * 300, 'subindustry': 'S' * 300,
        'product_service': 'P' * 500}, {'name': 'N' * 200, 'website': pages[0]['source_url'],
        'linkedin': 'https://www.linkedin.com/company/' + 'l' * 200}, pages,
        evaluation_date='2026-10-24', conditional_sources=candidates)
    return packet, pages, candidates


def test_maximum_four_conditional_quotes_and_urls_obey_native_wire_bounds():
    from lab_arena.operations import OPENROUTER_MAX_CONTENT_CHARS, validate_operation_request
    packet, _, _ = maximum_conditional_packet()
    assert len(packet.conditional_sources) == 4
    request = industry.request_payload(packet, model=industry.DEFAULT_MODEL)
    validate_operation_request('openrouter.chat', request)
    assert all(len(message['content']) < OPENROUTER_MAX_CONTENT_CHARS for message in request['messages'])
    metadata = json.loads(request['messages'][1]['content'])['conditional_industry_sources']
    assert all(item['parent_relationship_span_ids'] for item in metadata)
    assert all('parent_quote' not in item for item in metadata)  # Already present in cited source spans.


def test_closed_packet_input_does_not_allow_child_key_override():
    _, pages, candidates = maximum_conditional_packet()
    key = next(iter(candidates))
    candidates[key]['child_source_url'] = 'https://other.example/forged'
    with pytest.raises(ValueError, match='key cannot be overridden'):
        industry.prepare({'industry': 'Software', 'subindustry': '', 'product_service': ''},
            {'name': 'Coreworks', 'website': 'https://www.coreworks.ai/'}, pages,
            evaluation_date='2026-10-24', conditional_sources=candidates)


def test_body_budget_never_adds_child_without_cropped_parent_relationship(monkeypatch):
    result, _, _, _ = native_loop(monkeypatch)
    proof = copy.deepcopy(result[c.PRIVATE_INDUSTRY_BINDINGS_KEY]['brands'])
    # Saturate the existing body allowance; the late relationship cannot fit
    # under equal bounding, so child contribution must drop rather than expand.
    pages = [{'source_id': 'parent', 'source_url': PARENT,
              'body': 'Coreworks company information. ' * 700 + RELATION},
             {'source_id': 'child', 'source_url': CHILD, 'body': FACT}]
    proof[CHILD]['parent_body_sha256'] = hashlib.sha256(pages[0]['body'].encode()).hexdigest()
    pages += [{'source_id': f'other{i}', 'source_url': f'https://www.coreworks.ai/source{i}',
               'body': 'x' * industry.MAX_PAGE_CHARS} for i in range(28)]
    packet = industry.prepare({'industry': 'Software', 'subindustry': '', 'product_service': 'Workflows'},
        {'name': 'Coreworks', 'website': 'https://www.coreworks.ai/'}, pages,
        evaluation_date='2026-10-24', conditional_sources=proof)
    assert sum(len(body) for _, body, _ in packet.pages) <= industry.MAX_TOTAL_CHARS
    assert CHILD not in {url for _, _, url in packet.pages} and not packet.conditional_sources


def test_legitimate_parent_only_complete_finding_does_not_require_unnecessary_child(monkeypatch):
    result, _, _, _ = native_loop(monkeypatch)
    proof = result[c.PRIVATE_INDUSTRY_BINDINGS_KEY]['brands']
    parent_body = RELATION + ' Coreworks itself supplies analysis, reporting and account workflows.'
    packet = industry.prepare({'industry': 'Software', 'subindustry': '', 'product_service': 'Workflows'},
        {'name': 'Coreworks', 'website': 'https://www.coreworks.ai/'},
        [{'source_id': 'parent', 'source_url': PARENT, 'body': parent_body},
         {'source_id': 'child', 'source_url': CHILD, 'body': FACT}],
        evaluation_date='2026-10-24', conditional_sources=proof)
    parent_ref = next(key for key, span in packet.spans().items() if span['source_url'] == PARENT)
    resolved = industry.resolve(packet, {'decision': 'match', 'role': 'supplier_operator',
        'observed_activity': 'Software', 'reason': 'The parent itself supplies the complete offering.',
        'missing_requirements': [], 'evidence_span_ids': [parent_ref]})
    assert resolved['decision'] == 'match'


def test_wrong_identity_cache_read_clears_previous_binding_output(monkeypatch):
    result, _, _, _ = native_loop(monkeypatch)
    _, _, cache = fit_and_cache(result)
    output = copy.deepcopy(result[c.PRIVATE_INDUSTRY_BINDINGS_KEY])
    assert lead_scorer._matched_company_retry_prefetched_pages(cache,
        {**IDENTITY, 'registrable_dns_domain': 'other.example'}, industry_bindings_out=output) == {}
    assert output == {}


def test_industry_search_scope_does_not_expand_query_tenant_or_oversize_path():
    identity = {'submitted_domain': 'coreworks.ai', 'observed_domain': 'coreworks.ai'}
    for home in ['https://shared.example/?tenant=one', 'https://shared.example/' + 'x' * 300]:
        conditional = {home: {'official_child_homepage': home, 'child_final_url': home}}
        assert c._industry_search_scopes({'coreworks.ai'}, identity, (), conditional) == ('coreworks.ai',)


def test_child_first_citations_keep_bound_parent_attribution_without_inventing_alias(monkeypatch):
    relation = 'We operate our WorkflowCo brand as part of our connected software offering. ' + HOME
    result, _, _, _ = native_loop(monkeypatch, parent_text=relation, href='',
        proof={**PROOF, 'parent_quote': relation}, child_first=True)
    finding = result['claims']['industry']
    assert finding['status'] == 'VERIFIED'
    assert finding['evidence_url'] == PARENT
    assert relation in finding['evidence_quote']


@pytest.mark.parametrize('path', [
    '/tenant-one/../tenant-two/page', '/tenant-one/./../tenant-two/page',
    '/tenant-one/%2e%2e/tenant-two/page', '/tenant-one/.%2E/tenant-two/page',
    '/tenant-one/%2e./tenant-two/page', '/tenant-one/%252e%252e/tenant-two/page',
    '/tenant-one/%2e%2e%2ftenant-two/page', '/tenant-one/%2E%2E%5Ctenant-two/page',
    '/tenant-one/\\..\\tenant-two/page', '/tenant-one/%5c..%5ctenant-two/page',
    '/tenant-one/%255c%252e%252e%255ctenant-two/page',
    '/tenant-one/%252f..%252ftenant-two/page', '/tenant-one/%2f..%2ftenant-two/page',
    '/tenant-one/..;ignored/tenant-two/page', '/tenant-one/%2e%2e%3bignored/tenant-two/page',
    '/tenant-one/%EF%BC%8E%EF%BC%8E%EF%BC%8Ftenant-two/page',
])
def test_conditional_tenant_scope_rejects_literal_encoded_and_normalized_traversal(path):
    assert not c._industry_brand_scope('https://shared.example' + path,
                                        'https://shared.example/tenant-one/')


@pytest.mark.parametrize('home', [
    'https://shared.example/tenant-one/../tenant-two/',
    'https://shared.example/tenant-one/%252e%252e/tenant-two/',
    'https://shared.example/tenant-one%2ftenant-two/',
])
def test_observed_homepage_cannot_itself_contain_ambiguous_tenant_path(home):
    assert not c._industry_brand_scope(home + 'product', home)


@pytest.mark.parametrize('path', [
    '/tenant-one/spec%20sheet.pdf', '/tenant-one/caf%C3%A9.pdf',
    '/tenant-one/v1.2/guide.pdf', '/tenant-one/..backup.pdf',
    '/tenant-one/100%25-complete.pdf', '/tenant-one/a%2Eb.pdf',
    '/tenant-one/sub;v=2/file.pdf', '/tenant-one/%E6%96%87%E6%A1%A3.pdf',
])
def test_ordinary_encoded_paths_stay_supported_without_rewriting(path):
    url = 'https://shared.example' + path
    assert c._industry_brand_scope(url, 'https://shared.example/tenant-one/')
    assert path == url.split('shared.example', 1)[1]


@pytest.mark.parametrize('proof', [None, {**PROOF, 'parent_quote': 'Unproved but irrelevant proposal.'}, PROOF])
def test_unnecessary_conditional_proposal_cannot_block_ordinary_first_party_fetch(monkeypatch, proof):
    own_product = 'https://www.coreworks.ai/product'
    monkeypatch.setitem(globals(), 'CHILD', own_product)
    result, reads, _, _ = native_loop(monkeypatch, proof=proof)
    assert reads == [PARENT, own_product]
    assert result['claims']['industry']['status'] == 'VERIFIED'
    assert own_product not in result.get(c.PRIVATE_INDUSTRY_BINDINGS_KEY, {}).get('brands', {})


def test_self_reference_cannot_erase_ordinary_parent_only_positive(monkeypatch):
    monkeypatch.setitem(globals(), 'CHILD', PARENT)
    text = 'Coreworks itself supplies analysis, reporting and account workflows. Coreworks homepage: ' + PARENT
    proposal = {'parent_source_url': PARENT, 'parent_quote': text,
        'child_brand': 'Coreworks', 'official_child_homepage': PARENT}
    result, reads, _, _ = native_loop(monkeypatch, proof=proposal, parent_text=text)
    assert reads == [PARENT]
    assert result['claims']['industry']['status'] == 'VERIFIED'
    assert result['claims']['industry']['evidence_url'] == PARENT
    assert PARENT not in result.get(c.PRIVATE_INDUSTRY_BINDINGS_KEY, {}).get('brands', {})


@pytest.mark.parametrize('url', [PARENT, 'https://www.coreworks.ai/product'])
def test_retained_conditional_records_never_reclassify_bound_first_party_pages(url):
    text = 'Coreworks itself supplies account workflows. Coreworks homepage: ' + url
    pages = {PARENT: text, url: text}
    binding = {'schema': 'industry_source_bindings_v1', 'parent_domain': 'coreworks.ai',
        'parent_name': 'coreworks', 'evaluation_date': '2026-10-24', 'links': {},
        'brands': {url: {'parent_source_url': PARENT, 'parent_quote': text,
            'child_brand': 'Coreworks', 'official_child_homepage': url,
            'parent_body_sha256': hashlib.sha256(text.encode()).hexdigest(),
            'child_body_sha256': hashlib.sha256(text.encode()).hexdigest(),
            'child_final_url': url, 'pointer_kind': 'printed_reference', 'printed_reference': url}}}
    validated = c._validated_industry_bindings(binding, pages, {key: key for key in pages},
        domains={'coreworks.ai'}, identity={'submitted_domain': 'coreworks.ai',
            'observed_domain': 'coreworks.ai', 'observed_name': 'Coreworks'},
        evaluation_date_value='2026-10-24')
    assert validated['brands'] == {}
    packet = industry.prepare({'industry': 'Software', 'subindustry': '', 'product_service': 'Workflows'},
        {'name': 'Coreworks', 'website': 'https://www.coreworks.ai/'},
        [{'source_id': f'page{i}', 'source_url': key, 'body': body}
         for i, (key, body) in enumerate(pages.items())], evaluation_date='2026-10-24',
        conditional_sources=validated['brands'])
    assert {item[2] for item in packet.pages} == set(pages)
    assert not packet.conditional_sources


@pytest.mark.parametrize('at_final', [False, True])
def test_actual_native_binding_rejects_tenant_traversal_even_if_provider_echoes_it(monkeypatch, at_final):
    home = 'https://shared.example/tenant-one/'
    unsafe = home + '%252e%252e/tenant-two/product'
    target = home + 'product' if at_final else unsafe
    monkeypatch.setitem(globals(), 'CHILD', target)
    proof = {**PROOF, 'official_child_homepage': home}
    result, reads, _, _ = native_loop(monkeypatch, proof=proof, href=home,
        child_final=unsafe)
    assert result['claims']['industry']['status'] != 'VERIFIED'
    assert not result.get(c.PRIVATE_INDUSTRY_BINDINGS_KEY, {}).get('brands')
    assert reads == [PARENT, target] if at_final else reads == [PARENT]
