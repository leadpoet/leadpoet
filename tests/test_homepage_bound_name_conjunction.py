"""A punctuation variant is not an independently proven historical rebrand."""
import json
from types import SimpleNamespace

import pytest

from qualification.scoring.lead_scorer import _web_identity_receipt

URL = 'https://www.linkedin.com/company/kratos-defense-and-security-solutions'
ANCHOR = {'normalized_name': 'kratosdefense', 'registrable_dns_domain': 'kratosdefense.com',
          'linkedin_company_slug': 'kratos-defense-and-security-solutions'}


def receipt(submitted='Kratos Defense and Security Solutions',
            observed='Kratos Defense & Security Solutions, Inc.',
            website='https://kratosdefense.com', linkedin=URL, anchor=ANCHOR,
            company_quality=False):
    return _web_identity_receipt(
        SimpleNamespace(company_name=submitted, company_website='https://kratosdefense.com',
                        company_linkedin=URL),
        {'observed_company_name': observed, 'observed_company_website': website,
         'observed_company_linkedin': linkedin},
        verified_homepage_identity=anchor,
        verified_rebrand_identity={'status': 'UNPROVEN'},
        company_quality=company_quality,
    )


@pytest.mark.parametrize("round_trip", [False, True])
def test_saved_kratos_observation_uses_existing_homepage_binding(round_trip):
    kwargs = {"anchor": ANCHOR}
    if round_trip:
        kwargs = json.loads(json.dumps(kwargs))
    result = receipt(**kwargs)
    assert result['decision'] == 'match'
    assert result['reason_code'] == 'verified_homepage_name_conjunction'
    assert result['observed_name'] == 'kratosdefensesecuritysolutions'


@pytest.mark.parametrize('kwargs', [
    {'website': 'https://other.com'}, {'linkedin': URL+'-other'}, {'linkedin': ''},
    {'anchor': None}, {'anchor': {**ANCHOR, 'registrable_dns_domain': 'other.com'}},
    {'anchor': {**ANCHOR, 'linkedin_company_slug': 'other'}},
    {'observed': 'Kratos Defense & Security Solutions Federal, Inc.'},
    {'observed': 'Kratos Defense & Security Solutions Subsidiary'},
    {'observed': 'Kratos Defense'},
    {'submitted': 'Kratos Defenseand Security Solutions',
     'observed': 'Kratos Defense& Security Solutions'},
    {'submitted': 'Kratos Defense andSecurity Solutions',
     'observed': 'Kratos Defense &Security Solutions'},
    {'company_quality': True},
])
def test_conjunction_does_not_create_an_identity_binding(kwargs):
    assert receipt(**kwargs)['decision'] != 'match'


@pytest.mark.parametrize('submitted,observed', [
    ('Kratos Defense & Security Solutions', 'Kratos Defense and Security Solutions, Inc.'),
    ('Kratos Defense and Security Solutions', 'Kratos Defense & Security Solutions, Inc.'),
])
def test_standalone_conjunction_spelling_is_symmetric(submitted, observed):
    assert receipt(submitted=submitted, observed=observed)['decision'] == 'match'
