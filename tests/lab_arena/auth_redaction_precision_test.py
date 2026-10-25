"""Natural Basic prose survives; authentication material stays private."""
import base64
import json
import re

import pytest

from lab_arena import training_capture, training_export, trajectory


PUBLIC_CASE = ('positioning the combined platform for organisations that have historically '
               'fallen between basic security products and the more demanding tools used '
               'by larger security operations centres.')


def response(value):
    return {'status': 200, 'body_b64': base64.b64encode(json.dumps(value).encode()).decode()}


@pytest.mark.parametrize('text', [PUBLIC_CASE, 'basic security products',
    'Basic capabilities and BASIC SECURITY controls', 'basic understanding of workflows',
    'basic bm9jb2xvbg==', 'basic bm9jb2xvbg', 'basic dT=pw',
    'basic https://example.test/public', 'basic http://example.test/public'])
def test_public_prose_and_noncredential_basic_values_remain_exact(text):
    assert trajectory.redact_text(text) == text
    assert training_capture.sanitize_training_value(text) == (text, [])
    assert training_export._response_value(response({'text': text}), 'deepline') == ({'text': text}, [])


@pytest.mark.parametrize('credentials', [b'u:p', b':', b'u:', b':p',
    'utilisateur:mot-de-passe'.encode(), 'usér:päss'.encode('utf-8'),
    b'us\xe9r:p\xe4ss', b'user:one:two', b'user:\x00\r\nprivate', b'user:' + b'p' * 2048])
@pytest.mark.parametrize('padded', [False, True])
@pytest.mark.parametrize('scheme', ['Basic', 'basic', 'bAsIc'])
def test_actual_basic_credentials_redact_in_both_active_paths(credentials, padded, scheme):
    token = base64.b64encode(credentials).decode()
    if not padded:
        token = token.rstrip('=')
    text = 'before ' + scheme + '\t' + token + ' after'
    assert trajectory.redact_text(text) == 'before [REDACTED] after'
    cleaned, reasons = training_capture.sanitize_training_value(text)
    assert (cleaned, reasons) == ('before [REDACTED] after', ['credential'])
    assert training_capture.sanitize_training_value(cleaned)[0] == cleaned


@pytest.mark.parametrize('label', ['Authorization', 'authorization', 'PROXY-AUTHORIZATION', 'proxy_authorization'])
@pytest.mark.parametrize('credential', ['Basic x', 'basic invalid-value!', 'Basic dTpw===',
    'Bearer x', 'Bearer odd/value!', 'secret', 'x'])
@pytest.mark.parametrize('formatting', ['{label}: {credential}', '{label}={credential}',
    '{label}: "{credential}"', '{label}: \'{credential}\''])
def test_explicit_authentication_context_redacts_even_malformed_or_short_values(label, credential, formatting):
    text = formatting.format(label=label, credential=credential)
    for cleaned in (trajectory.redact_text(text), training_capture.sanitize_training_value(text)[0]):
        assert re.search(r'[:=]\s*\[REDACTED\]$', cleaned)
        assert training_capture.sanitize_training_value(cleaned)[0] == cleaned
    for document in ({label: credential}, {'headers': {label: credential}}):
        cleaned, reasons = training_capture.sanitize_training_value(document)
        assert (cleaned.get('headers') or cleaned)[label] == '[REDACTED]'
        assert reasons == ['credential']


@pytest.mark.parametrize('scheme', ['Bearer', 'bearer', 'bEaReR'])
@pytest.mark.parametrize('token', ['abcdefghijklmnop', 'AbCdEf0123._~+/=-', 'securitytoken'])
def test_bearer_tokens_remain_opaque_and_redacted(scheme, token):
    value = scheme + ' ' + token
    assert trajectory.redact_text(value) == '[REDACTED]'
    assert training_capture.sanitize_training_value(value) == ('[REDACTED]', ['credential'])


def test_known_secrets_sensitive_fields_and_url_secrets_keep_priority():
    secret = 'not-valid-basic-prose'
    text = 'basic ' + secret + ' https://user:password@example.test/?auth=private-auth&keep=visible'
    cleaned, reasons = training_capture.sanitize_training_value(text, secrets=(secret,))
    for value in (secret, 'user:password', 'private-auth'):
        assert value not in cleaned
    assert 'keep=visible' in cleaned and reasons == ['credential']
    assert training_capture.sanitize_training_value(cleaned)[0] == cleaned
    assert training_capture.sanitize_training_value({'api_key': 'x', 'cookie': 'x'}) == (
        {'api_key': '[REDACTED]', 'cookie': '[REDACTED]'}, ['credential'])


@pytest.mark.parametrize('label', ['Authorization', 'Proxy-Authorization', 'proxy_authorization', 'api_key', 'password'])
@pytest.mark.parametrize('credential', ['Basic x', 'Basic invalid!', 'Basic dTpw===', 'Bearer x', 'x'])
def test_quoted_json_credential_values_remain_parseable_and_private(label, credential):
    value = json.dumps({label: credential, 'public': 'basic security products'})
    for cleaned in (trajectory.redact_text(value), training_capture.sanitize_training_value(value)[0]):
        assert json.loads(cleaned) == {label: '[REDACTED]', 'public': 'basic security products'}
        assert training_capture.sanitize_training_value(cleaned)[0] == cleaned


@pytest.mark.parametrize('label', ['api_key', 'password', 'access_token', 'lease_token'])
@pytest.mark.parametrize('scheme', ['Basic', 'Bearer'])
def test_explicit_sensitive_assignments_do_not_leave_short_scheme_payloads(label, scheme):
    value = label + '=' + scheme + ' x'
    assert trajectory.redact_text(value) == label + '=[REDACTED]'
    assert training_capture.sanitize_training_value(value) == (label + '=[REDACTED]', ['credential'])


@pytest.mark.parametrize('separator', [' ', '\t', '\r\n '])
def test_wrapped_authentication_whitespace_still_redacts(separator):
    token = base64.b64encode(b'user:password').decode()
    value = 'Basic' + separator + token
    assert trajectory.redact_text(value) == '[REDACTED]'
    assert training_capture.sanitize_training_value(value) == ('[REDACTED]', ['credential'])
    value = 'Authorization: ' + value
    assert token not in trajectory.redact_text(value)
    assert token not in training_capture.sanitize_training_value(value)[0]


@pytest.mark.parametrize('token', ['Oh==', 'dTp=', 'dTpw===', 'dTpw====', 'dTpwYXNz====',
    'dXNlcjpwYXN=', 'dXNlcjpwYXO=', 'dXNlcjpwYXP='])
def test_noncanonical_but_decodable_basic_credentials_still_redact(token):
    assert trajectory.redact_text('Basic ' + token) == '[REDACTED]'
    assert training_capture.sanitize_training_value('Basic ' + token) == ('[REDACTED]', ['credential'])


@pytest.mark.parametrize('depth', [1, 2, 3, 4])
@pytest.mark.parametrize('label', ['authorization', 'Authorization', 'Proxy-Authorization', 'proxy_authorization'])
@pytest.mark.parametrize('credential', ['Basic invalid-value!', 'Basic security', 'Basic x', 'Bearer x', 'x'])
def test_serialized_authentication_headers_remain_private_and_parseable(depth, label, credential):
    value = {label: credential, 'public': 'basic security products'}
    for _ in range(depth):
        value = json.dumps(value)
    for cleaned in (trajectory.redact_text(value), training_capture.sanitize_training_value(value)[0]):
        decoded = cleaned
        for _ in range(depth):
            decoded = json.loads(decoded)
        assert decoded == {label: '[REDACTED]', 'public': 'basic security products'}
        assert training_capture.sanitize_training_value(cleaned)[0] == cleaned
