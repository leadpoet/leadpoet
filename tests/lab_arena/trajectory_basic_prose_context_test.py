"""Credential context must not redact unrelated Basic prose in full-source input."""
import base64
import json

import pytest

from lab_arena import trajectory, training_capture


@pytest.mark.parametrize('serialize', [False, True])
@pytest.mark.parametrize('redact', [trajectory.redact_auth_schemes, trajectory.redact_text,
                                     lambda text: training_capture.sanitize_training_value(text)[0]])
def test_basic_permissions_survives_unrelated_quoted_auth_context(serialize, redact):
    source = {'submission_files': [
        {'path': 'agent.py', 'content': 'headers = {"Authorization": "Bearer " + api_key}\n'},
        {'path': 'LICENSE', 'content': 'Basic Permissions.'},
    ]}
    text = json.dumps(source) if serialize else source['submission_files'][0]['content'] + '\nBasic Permissions.'
    cleaned = redact(text)
    assert 'Basic Permissions.' in cleaned
    # The ambiguous source expression keeps its existing conservative mask.
    assert '"Bearer "' not in cleaned and '\\"Bearer \\"' not in cleaned


@pytest.mark.parametrize('padding', [True, False])
@pytest.mark.parametrize('style', ['plain', 'quoted', 'escaped', 'multiline'])
def test_real_basic_credentials_stay_private_with_unrelated_prose(padding, style):
    token = base64.b64encode(b'user:secret').decode()
    if not padding:
        token = token.rstrip('=')
    separator = '\n' if 'multiline' in style else ' '
    value = 'Basic' + separator + token
    text = ('"authorization": "' + value + '"' if style in {'quoted', 'escaped'}
            else 'Authorization: ' + value)
    if style == 'escaped':
        text = json.dumps(text)
    text += '\nBasic Permissions.'
    for cleaned in (trajectory.redact_auth_schemes(text), trajectory.redact_text(text),
                    training_capture.sanitize_training_value(text)[0]):
        assert token not in cleaned
        assert 'Basic Permissions.' in cleaned


def test_quoted_nonstandard_basic_secret_stays_private():
    text = '"password": "Basic opaquealphabetictoken"; Basic Permissions.'
    cleaned = training_capture.sanitize_training_value(text)[0]
    assert 'opaquealphabetictoken' not in cleaned
    assert 'Basic Permissions.' in cleaned

@pytest.mark.parametrize('escapes', [1, 2, 3])
@pytest.mark.parametrize('inner', ['prefix" Basic opaquealphabetictoken',
                                   'prefix\\ Basic opaquealphabetictoken'])
def test_escaped_quote_or_backslash_keeps_entire_sensitive_value_private(escapes, inner):
    text = json.dumps({'secret': inner, 'public': 'Basic Permissions.'})
    for _ in range(escapes - 1):
        text = json.dumps(text)
    for cleaned in (trajectory.redact_auth_schemes(text), trajectory.redact_text(text),
                    training_capture.sanitize_training_value(text)[0]):
        assert 'opaquealphabetictoken' not in cleaned
        assert 'Basic Permissions.' in cleaned

@pytest.mark.parametrize('depth', [0, 1, 2, 3])
@pytest.mark.parametrize('field', ['secret', 'authorization', 'api_key'])
def test_empty_quoted_sensitive_values_remain_exact(depth, field):
    text = json.dumps({field: ''})
    for _ in range(depth):
        text = json.dumps(text)
    assert trajectory.redact_auth_schemes(text) == text
    assert trajectory.redact_text(text) == text
    assert training_capture.sanitize_training_value(text) == (text, [])
