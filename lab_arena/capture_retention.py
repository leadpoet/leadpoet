"""Private immutable capture files and bounded replay of original runtime inputs."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import stat
import tempfile
import threading
import time
from typing import Any, Iterator, Mapping

from lab_arena import contracts, training_input

INPUT_GRACE_SECONDS = 180
INPUT_REPLAY_TIMEOUT_SECONDS = 5.0
_INPUT_RECORD_SCHEMA = 'leadpoet.lab_arena.pending_input.v1'
_INPUT_RECORD_LIMIT = training_input.MAX_INPUT_BYTES + 64 * 1024
_RECORD_NAME = re.compile(r'[0-9a-f]{64}\.json')


def private_directory(root: Path, name: str, *, create=False, error=ValueError, label='capture'):
    directory = root / name
    if create:
        root.mkdir(parents=True, exist_ok=True)
        directory.mkdir(mode=0o700, exist_ok=True)
    try:
        details = directory.lstat()
    except FileNotFoundError:
        return None
    if (not stat.S_ISDIR(details.st_mode) or details.st_uid != os.getuid()
            or stat.S_IMODE(details.st_mode) != 0o700):
        raise error('pending %s directory is not private' % label)
    return directory


def sync_directory(directory: Path):
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def locked_record(path: Path, maximum: int, *, error=ValueError, label='capture'):
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except FileNotFoundError:
        yield None
        return
    try:
        details = os.fstat(descriptor)
        if (not stat.S_ISREG(details.st_mode) or details.st_uid != os.getuid()
                or stat.S_IMODE(details.st_mode) != 0o600 or details.st_size > maximum):
            raise error('pending %s file is invalid' % label)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield None
            return
        try:
            current = path.lstat()
        except FileNotFoundError:
            yield None
            return
        if (current.st_dev, current.st_ino) != (details.st_dev, details.st_ino):
            yield None
            return
        yield descriptor
    finally:
        os.close(descriptor)


@contextmanager
def retained_record(path: Path, raw: bytes, maximum: int, *, error=ValueError, label='capture'):
    """Publish a locked, fully fsynced inode; never overwrite an existing record."""
    if len(raw) > maximum:
        raise error('%s document is too large' % label)
    descriptor, temporary = tempfile.mkstemp(prefix='.' + label + '-', dir=path.parent)
    try:
        os.fchmod(descriptor, 0o600)
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with os.fdopen(os.dup(descriptor), 'wb') as destination:
            destination.write(raw)
            destination.flush()
            os.fsync(destination.fileno())
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError:
            with locked_record(path, maximum, error=error, label=label) as existing:
                if existing is None:
                    raise error('%s delivery is already active' % label)
                yield existing, False
        else:
            os.unlink(temporary)
            sync_directory(path.parent)
            yield descriptor, True
    finally:
        os.close(descriptor)
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _time(value):
    parsed = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    if parsed.tzinfo is None:
        raise ValueError('input capture timestamp lacks a zone')
    return parsed.astimezone(timezone.utc)


class InputRetention:
    """One immutable input per lease; replay never owns execution or completion."""
    def __init__(self, config):
        self.config = config
        self._next_poll = 0.0
        self._cursor = ''
        self._thread = None
        self._poll_lock = threading.Lock()
        self._stop = threading.Event()

    def _directory(self, *, create=False):
        return private_directory(self.config.work_dir, 'pending-inputs', create=create)

    @staticmethod
    def _name(record):
        keys = ('run_id', 'round_id', 'validator_hotkey', 'lease_generation', 'claim_request_hash', 'lease_token')
        return contracts.document_hash({key: record[key] for key in keys})[7:] + '.json'

    def _prepare(self, lease, token, document):
        run_id, generation, claim = lease['run_id'], lease['lease_generation'], lease['claim_request_hash']
        training_input.object_ref(run_id, generation)
        training_input._validate_claim_hash(claim)
        expiry = _time(lease['lease_expires_at'])
        if not isinstance(token, str) or not re.fullmatch(r'[0-9a-f]{64}', token):
            raise ValueError('input capture token is invalid')
        if not isinstance(lease.get('round_id'), str) or not lease['round_id']:
            raise ValueError('input capture round is invalid')
        expected, _ = training_input.build_snapshot(run_id, lease.get('kind', 'execute'), document,
            lease_generation=generation, claim_request_hash=claim, secrets=(token,))
        return {'schema_version': _INPUT_RECORD_SCHEMA, 'validator_hotkey': self.config.identity.hotkey,
            'round_id': lease['round_id'], 'run_id': run_id, 'kind': lease.get('kind', 'execute'),
            'lease_generation': generation, 'claim_request_hash': claim, 'lease_token': token,
            'lease_expires_at': expiry.isoformat(), 'input': document,
            'original_input_sha256': contracts.document_hash(document),
            'expected_input_sha256': expected['input_sha256'], 'expected_privacy': expected['privacy']}

    def _read(self, descriptor, path):
        os.lseek(descriptor, 0, os.SEEK_SET)
        with os.fdopen(os.dup(descriptor), 'rb') as source:
            raw = source.read(_INPUT_RECORD_LIMIT + 1)
        if len(raw) > _INPUT_RECORD_LIMIT:
            raise ValueError('pending input exceeds limit')
        record = json.loads(raw)
        if (record.get('schema_version') != _INPUT_RECORD_SCHEMA
                or record.get('validator_hotkey') != self.config.identity.hotkey
                or self.config.round_id is not None and record.get('round_id') != self.config.round_id
                or path.name != self._name(record)
                or record.get('original_input_sha256') != contracts.document_hash(record.get('input'))):
            raise ValueError('pending input identity is invalid')
        expected = self._prepare(record, record['lease_token'], record['input'])
        if expected != record:
            raise ValueError('pending input is invalid')
        return record

    def retain(self, lease, token, document):
        record = self._prepare(lease, token, document)
        path = self._directory(create=True) / self._name(record)
        raw = contracts.canonical_json(record).encode('utf-8')
        with retained_record(path, raw, _INPUT_RECORD_LIMIT) as (descriptor, created):
            if not created and self._read(descriptor, path) != record:
                raise ValueError('pending input identity conflict')
        return path

    def contains(self, lease, token, document):
        record = self._prepare(lease, token, document)
        directory = self._directory()
        if directory is None:
            return False
        path = directory / self._name(record)
        with locked_record(path, _INPUT_RECORD_LIMIT) as descriptor:
            present = descriptor is not None and self._read(descriptor, path) == record
            if present:
                sync_directory(directory)
            return present

    def deliver(self, path, *, document=None, max_attempts=1, replay=False, replay_deadline=None):
        if replay_deadline is not None and time.monotonic() >= replay_deadline:
            return False
        with locked_record(path, _INPUT_RECORD_LIMIT) as descriptor:
            if descriptor is None:
                return False
            try:
                record = self._read(descriptor, path)
            except (ValueError, TypeError, KeyError, AttributeError):
                self._park(path, 'invalid', descriptor)
                raise
            if document is not None and contracts.document_hash(document) != record['original_input_sha256']:
                raise ValueError('retained runtime input differs')
            deadline = _time(record['lease_expires_at']) + timedelta(seconds=INPUT_GRACE_SECONDS)
            remaining = (deadline - self.config.clock()).total_seconds()
            if replay and remaining <= 0:
                self._park(path, 'expired', descriptor)
                return False
            if replay_deadline is not None:
                remaining = min(remaining, replay_deadline - time.monotonic())
                if remaining <= 0:
                    return False
            if replay and self._stop.is_set():
                return False
            for attempt in range(max_attempts):
                try:
                    args = (record['run_id'], record['lease_token'], record['input'] if document is None else document)
                    response = (self.config.api.training_input(*args,
                        timeout_seconds=min(INPUT_REPLAY_TIMEOUT_SECONDS, remaining)) if replay
                        else self.config.api.training_input(*args))
                except Exception as exc:
                    if replay and getattr(exc, 'http_status', None) in {400, 401, 403, 404, 409, 413, 422}:
                        self._park(path, 'rejected', descriptor)
                    if attempt + 1 < max_attempts:
                        continue
                    raise
                if self.matches(record, response):
                    path.unlink()
                    sync_directory(path.parent)
                    return True
                status = getattr(response, 'http_status', None)
                if status is None and isinstance(response, Mapping):
                    status = response.get('http_status')
                if replay and status in {400, 401, 403, 404, 409, 413, 422}:
                    self._park(path, 'rejected', descriptor)
                return False
            return False

    @staticmethod
    def matches(record, response):
        return (isinstance(response, Mapping) and response.get('status') == 'stored'
            and response.get('schema_version') == training_input.SCHEMA_VERSION
            and response.get('ref') == training_input.object_ref(record['run_id'], record['lease_generation'])
            and type(response.get('lease_generation')) is int
            and response['lease_generation'] == record['lease_generation']
            and response.get('claim_request_hash') == record['claim_request_hash']
            and response.get('input_sha256') == record['expected_input_sha256']
            and isinstance(response.get('privacy'), Mapping)
            and type(response['privacy'].get('redacted')) is bool
            and isinstance(response['privacy'].get('redactions'), list)
            and response['privacy'] == record['expected_privacy'])

    def _park(self, path, reason, descriptor):
        directory = private_directory(path.parent, 'audit', create=True)
        target = directory / (path.stem + '.' + reason)
        try:
            os.link(path, target, follow_symlinks=False)
        except FileExistsError:
            source, saved = os.fstat(descriptor), target.lstat()
            if (source.st_dev, source.st_ino) != (saved.st_dev, saved.st_ino):
                with locked_record(target, _INPUT_RECORD_LIMIT) as existing:
                    if existing is None:
                        raise ValueError('input audit record is active')
                    os.lseek(descriptor, 0, os.SEEK_SET)
                    os.lseek(existing, 0, os.SEEK_SET)
                    if os.read(existing, _INPUT_RECORD_LIMIT + 1) != os.read(descriptor, _INPUT_RECORD_LIMIT + 1):
                        raise ValueError('input audit identity conflict')
        # Make the audit link durable before removing the pending name.
        sync_directory(directory)
        path.unlink()
        sync_directory(path.parent)
        import sys
        print("Arena pending input capture unavailable: " + reason, file=sys.stderr, flush=True)

    def _replay_batch(self):
        deadline = time.monotonic() + INPUT_REPLAY_TIMEOUT_SECONDS
        try:
            directory = self._directory()
            if directory is None or self._stop.is_set():
                return
            paths = sorted(path for path in directory.iterdir() if _RECORD_NAME.fullmatch(path.name))
            paths = [p for p in paths if p.name > self._cursor] + [p for p in paths if p.name <= self._cursor]
            for path in paths:
                if self._stop.is_set() or time.monotonic() >= deadline:
                    break
                self._cursor = path.name
                try:
                    self.deliver(path, replay=True, replay_deadline=deadline)
                except Exception as exc:
                    import sys
                    print('Arena pending input deferred: %s' % type(exc).__name__, file=sys.stderr, flush=True)
        except Exception as exc:
            # Never print input, lease credentials or provider content.
            import sys
            print('Arena pending input deferred: %s' % type(exc).__name__, file=sys.stderr, flush=True)

    def poll(self):
        with self._poll_lock:
            if (self._stop.is_set() or self._thread is not None and self._thread.is_alive()
                    or time.monotonic() < self._next_poll):
                return
            self._next_poll = time.monotonic() + max(5.0, float(self.config.claim_poll_seconds))
            try:
                self._thread = threading.Thread(target=self._replay_batch, name='arena-input-replay', daemon=True)
                self._thread.start()
            except Exception as exc:
                if self._thread is None or not self._thread.is_alive():
                    self._thread = None
                import sys
                print('Arena pending input deferred: %s' % type(exc).__name__, file=sys.stderr, flush=True)

    def stop(self):
        self._stop.set()
