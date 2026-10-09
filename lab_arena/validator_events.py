"""Private, bounded self-reported validator host observations.

These events convey diagnostics only. They grant no job or recovery authority.
"""
from __future__ import annotations

import json
import math
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Mapping, Optional

from lab_arena import trajectory

MAX_EVENTS_PER_REQUEST = 16
MAX_REQUEST_BYTES = 32 * 1024
MAX_EVENT_BYTES = 4 * 1024
KINDS = frozenset({
    'validator.startup', 'validator.ready', 'validator.state',
    'validator.error', 'validator.recovered', 'validator.stopping',
})
TEXT_KEYS = frozenset({
    'phase', 'state', 'reason', 'operation', 'error_class', 'denial_code',
    'scoring_state', 'weights_state', 'last_progress_at', 'last_poll_at',
    'last_completion_at', 'launch_stderr', 'validator_source_commit',
    'validator_source_dirty', 'validator_source_origin', 'session_id',
})
INTEGER_KEYS = frozenset({'http_status', 'attempt', 'active_runs', 'ready_slots',
                          'proxy_count', 'launch_exit_code'})
BOOLEAN_KEYS = frozenset({'retryable', 'launch_timed_out', 'launch_stderr_truncated'})
CONTENT_KEYS = TEXT_KEYS | INTEGER_KEYS | BOOLEAN_KEYS | {'delay_seconds'}
_QUOTED_SECRET = re.compile(
    r"(?i)(\b(?:api[-_ ]?key|authorization|password|secret|access[-_ ]?token|"
    r"refresh[-_ ]?token|lease[-_ ]?token)\s*[:=]\s*)([\"'])(.*?)(\2)"
)
_URL = re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)


def redact_text(value: str) -> str:
    # Keep credential-bearing URLs out of private host diagnostics too.
    value = _URL.sub('[URL]', value)
    value = _QUOTED_SECRET.sub(lambda match: match.group(1) + '[REDACTED]', value)
    return trajectory.redact_text(value)


_ID = re.compile(r'^[A-Za-z0-9._:-]{1,200}$')


class ValidatorEventError(ValueError):
    """An observation does not meet the private bounded contract."""


def validate_event(value: Any) -> dict:
    if not isinstance(value, Mapping) or not {'event_id', 'kind', 'occurred_at', 'content'} <= set(value):
        raise ValidatorEventError('validator event fields invalid')
    if set(value) - {'event_id', 'kind', 'occurred_at', 'content', 'run_id', 'round_id'}:
        raise ValidatorEventError('validator event fields invalid')
    try:
        event_id = str(uuid.UUID(value['event_id']))
        occurred = datetime.fromisoformat(value['occurred_at'].replace('Z', '+00:00'))
    except (AttributeError, TypeError, ValueError):
        raise ValidatorEventError('validator event identity or time invalid') from None
    if occurred.tzinfo is None or not isinstance(value['kind'], str) or value['kind'] not in KINDS:
        raise ValidatorEventError('validator event kind or time invalid')
    content = value['content']
    if not isinstance(content, Mapping) or set(content) - CONTENT_KEYS:
        raise ValidatorEventError('validator event content fields invalid')
    sanitized = {}
    for key, item in content.items():
        if item is None:
            sanitized[key] = None
        elif key in TEXT_KEYS and isinstance(item, str):
            limit = 2048 if key == 'launch_stderr' else 200
            if len(item.encode('utf-8')) > limit:
                raise ValidatorEventError('validator event content too large')
            if key == 'session_id':
                try:
                    item = str(uuid.UUID(item))
                except ValueError:
                    raise ValidatorEventError('validator session invalid') from None
            sanitized[key] = redact_text(item)
        elif key in BOOLEAN_KEYS and isinstance(item, bool):
            sanitized[key] = item
        elif key in INTEGER_KEYS and isinstance(item, int) and not isinstance(item, bool):
            if not -2147483648 <= item <= 2147483647:
                raise ValidatorEventError('validator event integer out of bounds')
            sanitized[key] = item
        elif key == 'delay_seconds' and isinstance(item, (int, float)) and not isinstance(item, bool) and 0 <= item <= 86400 and math.isfinite(item):
            sanitized[key] = item
        else:
            raise ValidatorEventError('validator event content type invalid')
    result = {'event_id': event_id, 'kind': value['kind'],
              'occurred_at': occurred.astimezone(timezone.utc).isoformat(timespec='milliseconds').replace('+00:00', 'Z'),
              'content': sanitized}
    for key in ('run_id', 'round_id'):
        if key in value:
            if not isinstance(value[key], str) or not _ID.fullmatch(value[key]):
                raise ValidatorEventError('validator event correlation invalid')
            result[key] = value[key]
    if len(json.dumps(result, ensure_ascii=False).encode('utf-8')) > MAX_EVENT_BYTES:
        raise ValidatorEventError('validator event too large')
    return result


def event(kind: str, content: Mapping[str, Any], *, run_id: Optional[str] = None,
          round_id: Optional[str] = None) -> dict:
    document = {'event_id': str(uuid.uuid4()), 'kind': kind,
                'occurred_at': datetime.now(timezone.utc).isoformat(), 'content': content}
    if run_id is not None:
        document['run_id'] = run_id
    if round_id is not None:
        document['round_id'] = round_id
    return validate_event(document)


def validate_events(events: Any) -> list[dict]:
    if not isinstance(events, list) or not 1 <= len(events) <= MAX_EVENTS_PER_REQUEST:
        raise ValidatorEventError('validator event count invalid')
    result = [validate_event(item) for item in events]
    if len(json.dumps(result, ensure_ascii=False).encode('utf-8')) > MAX_REQUEST_BYTES:
        raise ValidatorEventError('validator events too large')
    return result
