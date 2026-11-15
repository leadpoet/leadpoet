"""Shared recruiting platforms need observed tenant-specific relationships."""
import pytest
from qualification.scoring import intent_verification_three_stage as v

GUID = '3986a99a-17ae-446b-835f-782cb565047f'
OTHER_GUID = '4986a99a-17ae-446b-835f-782cb565047f'
PAY_BOARD = f'https://recruiting.paylocity.com/recruiting/jobs/All/{GUID}/Heartland-Business-Systems-LLC'
PAY_JOB = 'https://recruiting.paylocity.com/Recruiting/Jobs/Details/4606587'
ICIMS_BOARD = 'https://careers-tierpoint.icims.com/jobs/intro?hashed=-626007049'
ICIMS_JOB = 'https://careers-tierpoint.icims.com/jobs/3074/linux-engineer-i/job'


def page(source, link, label='Open Positions'):
    return {'url': source, 'text': 'Fetched recruiting page', 'meta': {
        'observed_ownership_links': [{'url': link, 'label': label}]}}


def test_icims_official_link_binds_exact_tenant_posting():
    assert v._ats_board_identity(ICIMS_BOARD) == v._ats_board_identity(ICIMS_JOB) != None
    anchors = v._ats_ownership_anchors([
        page('https://www.tierpoint.com/about-us/careers/', ICIMS_BOARD, 'Join our team'),
    ], [ICIMS_JOB], {'observed_domain': 'tierpoint.com'})
    assert anchors and anchors[0]['posting_url'] == ICIMS_JOB


@pytest.mark.parametrize('url', [
    'https://careers-tierpoint.icims.com.evil.example/jobs/intro',
    'https://evil.careers-tierpoint.icims.com/jobs/intro',
    'https://careers-tierpoint.icims.com@evil.example/jobs/intro',
    'https://evil@careers-tierpoint.icims.com/jobs/intro',
    'http://careers-tierpoint.icims.com/jobs/intro',
    'https://careers-tierpoint.icims.com:8443/jobs/intro',
    'https://careers-tierpoint.icims.com/not-jobs/intro',
    'https://careers-tierpoint.icims.com/jobs/intro?tenant=other',
    'https://careers-tierpoint.icims.com/jobs/intro#other',
    'https://icims.com/jobs/intro',
])
def test_icims_untrusted_or_non_board_shapes_are_not_tenant_identity(url):
    assert v._ats_board_identity(url) is None


def test_icims_shared_provider_and_other_tenant_do_not_bind():
    identity = {'observed_domain': 'tierpoint.com'}
    for link in ['https://icims.com/jobs/intro', ICIMS_BOARD.replace('tierpoint', 'another')]:
        assert not v._ats_ownership_anchors([
            page('https://www.tierpoint.com/about-us/careers/', link),
        ], [ICIMS_JOB], identity)


def test_paylocity_selected_posting_observed_all_jobs_binds_official_tenant():
    assert v._ats_board_identity(PAY_BOARD) == ('paylocity', GUID)
    assert v._ats_board_identity(PAY_JOB) is None
    anchors = v._ats_ownership_anchors([
        page('https://www.hbs.net/careers', PAY_BOARD),
        page(PAY_JOB, PAY_BOARD, 'All Jobs'),
    ], [PAY_JOB], {'observed_domain': 'hbs.net'})
    assert anchors and anchors[0]['posting_url'] == PAY_JOB
    assert anchors[0]['association_source_url'] == PAY_JOB
    assert anchors[0]['association_linked_url'] == PAY_BOARD


@pytest.mark.parametrize('selected_source,selected_link,official_link', [
    (PAY_JOB, PAY_BOARD.replace(GUID, OTHER_GUID), PAY_BOARD),
    (PAY_JOB + '0', PAY_BOARD, PAY_BOARD),
    ('https://unrelated.example/', PAY_BOARD, PAY_BOARD),
    (PAY_JOB, 'https://recruiting.paylocity.com/', PAY_BOARD),
    (PAY_JOB, PAY_BOARD, 'https://recruiting.paylocity.com/'),
    (PAY_JOB, PAY_BOARD, PAY_BOARD.replace('paylocity.com', 'paylocity.com.evil.example')),
    (PAY_JOB, PAY_BOARD.replace('paylocity.com', 'paylocity.com.evil.example'), PAY_BOARD),
])
def test_paylocity_requires_two_observed_matching_tenant_edges(selected_source, selected_link, official_link):
    assert not v._ats_ownership_anchors([
        page('https://www.hbs.net/careers', official_link),
        page(selected_source, selected_link, 'All Jobs'),
    ], [PAY_JOB], {'observed_domain': 'hbs.net'})


def test_paylocity_unobserved_selected_association_does_not_bind():
    assert not v._ats_ownership_anchors([
        page('https://www.hbs.net/careers', PAY_BOARD),
    ], [PAY_JOB], {'observed_domain': 'hbs.net'})


@pytest.mark.parametrize('url', [
    PAY_BOARD.replace(GUID, 'not-a-guid'),
    PAY_BOARD.replace('/All/', '/Company/'),
    PAY_BOARD + '?company=' + OTHER_GUID,
    PAY_BOARD.replace('https://', 'http://'),
    PAY_BOARD.replace('recruiting.', 'other.'),
    PAY_BOARD.replace('https://', 'https://evil@'),
    PAY_BOARD + '#another',
])
def test_paylocity_only_exact_all_jobs_guid_is_identity(url):
    assert v._ats_board_identity(url) is None


def test_paylocity_conflicting_selected_tenant_links_do_not_bind():
    selected = page(PAY_JOB, PAY_BOARD, 'All Jobs')
    selected['meta']['observed_ownership_links'].append({
        'url': PAY_BOARD.replace(GUID, OTHER_GUID), 'label': 'All Jobs',
    })
    assert not v._ats_ownership_anchors([
        page('https://www.hbs.net/careers', PAY_BOARD), selected,
    ], [PAY_JOB], {'observed_domain': 'hbs.net'})


def test_paylocity_unrelated_tenant_link_label_is_not_selected_job_association():
    assert not v._ats_ownership_anchors([
        page('https://www.hbs.net/careers', PAY_BOARD),
        page(PAY_JOB, PAY_BOARD, 'Customer site'),
    ], [PAY_JOB], {'observed_domain': 'hbs.net'})


def test_paylocity_exact_official_posting_link_still_binds_without_board_association():
    assert v._ats_ownership_anchors([
        page('https://www.hbs.net/careers', PAY_JOB, 'Apply'),
    ], [PAY_JOB], {'observed_domain': 'hbs.net'})
