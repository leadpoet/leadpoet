"""Metadata exclusion follows the verified quote's actual source attribution."""
from copy import deepcopy
import json
import pytest
from qualification.scoring import intent_details as review
from tests.test_intent_details_industry_support_handoff import _inputs
from tests.test_intent_details_locator_metadata_boundary import MARKER, response
ARCHIVE='https://source.example/archive'
RELEASE='https://source.example/actual-release'
QUOTE='Microsoft'

def trace(quote=QUOTE, *, field='evidence_urls_used', cited=RELEASE, suffix=None, retained_release=False):
    ev={'signal_status':'supported','same_entity_check':'pass','supporting_quotes':[quote]}
    if field:ev[field]=[cited]
    ctx=[{'url':ARCHIVE,'text':'Archive index. '+MARKER+' '+(suffix or 'https://source.example/'+quote)}]
    if retained_release:ctx.append({'url':RELEASE,'text':'An optional retained window without the mandatory observation.'})
    return {'verified_source_context':ctx,'intent_verdict':{'signal_evaluations':[ev]}}

@pytest.mark.parametrize('field',['evidence_urls_used','source_urls_supplied'])
@pytest.mark.parametrize('quote,suffix',[(QUOTE,None),('Microsoft announced a contract.','Microsoft announced a contract.'),('https://source.example/one https://source.example/two','https://source.example/one\n\t https://source.example/two')])
@pytest.mark.parametrize('retained_release',[False,True])
def test_unrelated_metadata_does_not_invalidate_attributed_mandatory_quote(field,quote,suffix,retained_release):
    t=trace(quote,field=field,suffix=suffix,retained_release=retained_release);before=deepcopy(t)
    cleaned=review._source_context_trace(t)
    assert cleaned['intent_verdict']['signal_evaluations'][0]['supporting_quotes']==[quote]
    assert t==before
    assert cleaned['verified_source_context'][0]['text']=='Archive index. '

@pytest.mark.parametrize('field',['evidence_urls_used','source_urls_supplied'])
@pytest.mark.parametrize('retained_release',[False,True])
def test_full_native_projection_keeps_valid_observation_without_retained_attributed_body(field,retained_release):
    inputs=_inputs();t=trace(field=field,retained_release=retained_release)
    inputs[2][0]['evidence_urls']=[RELEASE];inputs[2][0]['judge_verdict']['verification_trace']=t
    inputs[0].intent_signals[0].url=RELEASE
    doc=review.review_evidence(*inputs)
    source=next(x for x in doc['admitted_evidence']if x.get('evidence_kind')=='verified_signal_observation')
    assert source['source_url']==RELEASE and source['admitted_text']==[QUOTE]
    verdict=review._validate_review_response(json.dumps(response(doc,source['source_index'],QUOTE)),doc,initial_review=False)
    assert verdict['facts_supported']is True

@pytest.mark.parametrize('field',['evidence_urls_used','source_urls_supplied',None])
def test_known_same_source_metadata_is_still_excluded_in_full_native_assembly(field):
    inputs=_inputs();t=trace(field=field,cited=ARCHIVE)
    inputs[2][0]['evidence_urls']=[ARCHIVE];inputs[0].intent_signals[0].url=ARCHIVE
    inputs[2][0]['judge_verdict']['verification_trace']=t
    assert review._source_context_trace(t)['intent_verdict']['signal_evaluations'][0]['supporting_quotes']==[]
    with pytest.raises(ValueError,match='verified signal lacks supporting source quotes'):review.review_evidence(*inputs)

@pytest.mark.parametrize('field',['evidence_urls_used','source_urls_supplied',None])
def test_explicit_marker_spanning_quote_cannot_use_different_source_as_bypass(field):
    quote='Archive index. '+MARKER+' https://source.example/Microsoft';t=trace(quote,field=field)
    assert review._source_context_trace(t)['intent_verdict']['signal_evaluations'][0]['supporting_quotes']==[]


def test_primary_evidence_urls_take_precedence_over_supplied_fallback():
    t=trace(cited=ARCHIVE);e=t['intent_verdict']['signal_evaluations'][0];e['source_urls_supplied']=[RELEASE]
    assert review._source_context_trace(t)['intent_verdict']['signal_evaluations'][0]['supporting_quotes']==[]
    e['evidence_urls_used']=[RELEASE];e['source_urls_supplied']=[ARCHIVE]
    assert review._source_context_trace(t)['intent_verdict']['signal_evaluations'][0]['supporting_quotes']==[QUOTE]


def test_empty_inner_url_fields_keep_existing_conservative_outer_source_behavior():
    t=trace(field=None);t['evidence_url']=ARCHIVE;t['intent_verdict']['signal_evaluations'][0].update(evidence_urls_used=[],source_urls_supplied=[])
    assert review._source_context_trace(t)['intent_verdict']['signal_evaluations'][0]['supporting_quotes']==[]


def test_cited_body_outside_five_contexts_does_not_expand_or_change_source_allowance():
    t=trace();t['verified_source_context'] += [{'url':'https://source.example/extra'+str(i),'text':'Other ordinary body.'}for i in range(4)]
    t['verified_source_context'].append({'url':RELEASE,'text':'Microsoft announced a contract.'})
    out=review._source_context_trace(t)
    assert len(out['verified_source_context'])==5
    assert out['intent_verdict']['signal_evaluations'][0]['supporting_quotes']==[QUOTE]


@pytest.mark.parametrize('field',['supporting_quotes','contradicting_quotes'])
def test_nonqualifying_native_route_preserves_other_source_observation(field):
    inputs=_inputs();t=trace()
    evaluation=t['intent_verdict']['signal_evaluations'][0]
    evaluation.update(verification_mode='source_grounded',
                      source_urls_supplied=[RELEASE],signal_status='partially_supported')
    evaluation[field]=[QUOTE]
    # Keep a valid primary. A rejected optional signal may add context but
    # cannot substitute for the existing required primary evidence.
    optional=deepcopy(inputs[2][0])
    optional.update(after_decay=0,evidence_urls=[RELEASE],matched_icp_signal=1)
    optional['judge_verdict'].update(client_ready=False,verification_trace=t)
    inputs[2].append(optional)
    inputs[1].intent_signals.append('Other optional intent')
    doc=review.review_evidence(*inputs)
    kind='non_qualifying_'+field
    assert any(source.get('evidence_kind')==kind and source['admitted_text']==[QUOTE]
               for source in doc['admitted_evidence'])
    assert all(MARKER not in value for source in doc['admitted_evidence']
               for value in source.get('admitted_text',[]))


@pytest.mark.parametrize('outer',['result','trace'])
def test_omitted_inner_urls_reuse_existing_outer_source_attribution(outer):
    inputs=_inputs();t=trace(field=None)
    inputs[2][0]['evidence_urls']=[RELEASE]
    inputs[0].intent_signals[0].url=RELEASE
    inputs[2][0]['judge_verdict']['verification_trace']=t
    if outer=='trace':
        inputs[2][0].pop('evidence_urls')
        t['evidence_url']=RELEASE
    doc=review.review_evidence(*inputs)
    source=next(x for x in doc['admitted_evidence']if x.get('evidence_kind')=='verified_signal_observation')
    assert source['source_url']==RELEASE and source['admitted_text']==[QUOTE]
    assert review._validate_review_response(json.dumps(response(doc,source['source_index'],QUOTE)),
        doc,initial_review=False)['facts_supported']is True
