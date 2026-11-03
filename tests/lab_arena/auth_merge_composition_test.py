"""Main's Basic precision composes with exact and attached PR markers."""
import base64
import json

import pytest

from lab_arena import training_capture, trajectory


def test_public_basic_prose_literal_header_unpadded_secret_and_attached_marker_compose():
    marker = 'Authorization: "[REDACTED]"'
    assert trajectory.redact_auth_schemes(marker) == marker
    token = base64.b64encode(b'u:p').decode().rstrip('=')
    value = f'basic security products; {marker}; Basic {token}; Bearer [REDACTED]private-tail'
    expected = f'basic security products; {marker}; [REDACTED]; [REDACTED]'
    assert trajectory.redact_text(value) == expected
    assert training_capture.sanitize_training_value(value) == (expected, ['credential'])
    assert training_capture.sanitize_training_value(expected) == (expected, [])


@pytest.mark.parametrize('depth', [1, 2, 3, 4])
def test_serialized_private_context_keeps_literal_marker_and_removes_attached_value(depth):
    value = {'Authorization': 'Basic dTpw', 'public': 'basic security products',
             'literal': 'Bearer [REDACTED]', 'attached': 'Bearer [REDACTED]private-tail'}
    for _ in range(depth):
        value = json.dumps(value)
    expected = {'Authorization': '[REDACTED]', 'public': '[REDACTED] products',
                'literal': 'Bearer [REDACTED]', 'attached': '[REDACTED]'}
    for cleaned in (trajectory.redact_text(value), training_capture.sanitize_training_value(value)[0]):
        decoded = cleaned
        for _ in range(depth):
            decoded = json.loads(decoded)
        assert decoded == expected
        assert training_capture.sanitize_training_value(cleaned)[0] == cleaned
