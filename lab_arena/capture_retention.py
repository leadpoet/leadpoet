"""Private immutable inputs/final observations, with one bounded replay operation."""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
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

from lab_arena import contracts, training_input, training_observations

INPUT_GRACE_SECONDS = 180
INPUT_REPLAY_TIMEOUT_SECONDS = 5.0
# Retention precedes foreground capture and bounded completion recovery. Keep
# one canonical minimum lease horizon plus the existing capture grace locally;
# the server still owns current lease/finish authority and permanent denials.
FINAL_OBSERVATION_REPLAY_SECONDS = contracts.LEASE_TTL_SECONDS + INPUT_GRACE_SECONDS
_INPUT_RECORD_SCHEMA = 'leadpoet.lab_arena.pending_input.v1'
_INPUT_RECORD_LIMIT = training_input.MAX_INPUT_BYTES + 64 * 1024
_LEGACY_OBSERVATION_RECORD_SCHEMA = 'leadpoet.lab_arena.pending_observations.v1'
_OBSERVATION_RECORD_SCHEMA = 'leadpoet.lab_arena.pending_observations.v2'
_OBSERVATION_RECORD_LIMIT = training_observations.MAX_OBJECT_BYTES + 64 * 1024
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
    """One controller for retained captures; replay never owns execution/completion."""
    _record_limit = _INPUT_RECORD_LIMIT
    _label = 'input'

    def __init__(self, config):
        self.config = config
        self._next_poll = 0.0
        self._cursor = ''
        self._thread = None
        self._poll_lock = threading.Lock()
        self._stop = threading.Event()
        self.observations = _FinalObservationRetention(config, self._stop)

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
                with locked_record(target, self._record_limit) as existing:
                    if existing is None:
                        raise ValueError(self._label + ' audit record is active')
                    os.lseek(descriptor, 0, os.SEEK_SET)
                    os.lseek(existing, 0, os.SEEK_SET)
                    if os.read(existing, self._record_limit + 1) != os.read(descriptor, self._record_limit + 1):
                        raise ValueError(self._label + ' audit identity conflict')
        # Make the audit link durable before removing the pending name.
        sync_directory(directory)
        path.unlink()
        sync_directory(path.parent)
        import sys
        print("Arena pending " + self._label + " capture unavailable: " + reason, file=sys.stderr, flush=True)

    def _replay_batch(self):
        # Drain one finite snapshot under the existing grace-sized admission
        # bound. A five-second sweep otherwise admits only one large final
        # between the runner's thirty-second polls. Each record still gets
        # its own bounded HTTP attempt, never a retry inside this sweep.
        deadline = time.monotonic() + INPUT_GRACE_SECONDS
        try:
            if self._stop.is_set():
                return
            pending = []
            for owner in (self, self.observations):
                try:
                    directory = owner._directory()
                    if directory is not None:
                        pending.extend((directory.name + '/' + path.name, owner, path)
                            for path in directory.iterdir() if _RECORD_NAME.fullmatch(path.name))
                except Exception as exc:
                    import sys
                    print('Arena pending capture deferred: %s' % type(exc).__name__, file=sys.stderr, flush=True)
            pending.sort(key=lambda item: item[0])
            pending = [item for item in pending if item[0] > self._cursor] + [item for item in pending if item[0] <= self._cursor]
            for key, owner, path in pending:
                if self._stop.is_set() or time.monotonic() >= deadline:
                    break
                self._cursor = key
                try:
                    owner.deliver(path, replay=True, replay_deadline=(
                        time.monotonic() + INPUT_REPLAY_TIMEOUT_SECONDS))
                except Exception as exc:
                    import sys
                    print('Arena pending %s deferred: %s' % (owner._label, type(exc).__name__), file=sys.stderr, flush=True)
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


class _FinalObservationRetention(InputRetention):
    """Final files share the input controller's one bounded replay operation."""
    _record_limit = _OBSERVATION_RECORD_LIMIT
    _label = 'observations'

    def __init__(self, config, stop):
        self.config, self._stop = config, stop

    def _directory(self, *, create=False):
        return private_directory(self.config.work_dir, 'pending-observations', create=create)

    def _prepare(self, lease, token, document, expected_digest=None, *, retained_at=None):
        training_observations.object_ref(lease['run_id'], lease['lease_generation'], lease['claim_request_hash'])
        expiry = _time(lease['lease_expires_at'])
        now = _time(self.config.clock())
        retained_at = now if retained_at is None else _time(retained_at)
        if retained_at > now:
            raise ValueError('observation capture clock is invalid')
        if not isinstance(token, str) or not re.fullmatch(r'[0-9a-f]{64}', token):
            raise ValueError('observation capture token is invalid')
        if not isinstance(lease.get('round_id'), str) or not lease['round_id']:
            raise ValueError('observation capture round is invalid')
        if expected_digest is None:
            expected_digest = contracts.document_hash(training_observations._validate_payload(document, secrets=(token,)))
        if lease.get('kind', 'execute') not in {'execute', 'score'}:
            raise ValueError('observation capture kind is invalid')
        if not contracts.SHA256_RE.fullmatch(expected_digest):
            raise ValueError('observation capture digest is invalid')
        return {'schema_version': _OBSERVATION_RECORD_SCHEMA, 'validator_hotkey': self.config.identity.hotkey,
            'round_id': lease['round_id'], 'run_id': lease['run_id'], 'kind': lease.get('kind', 'execute'),
            'lease_generation': lease['lease_generation'], 'claim_request_hash': lease['claim_request_hash'],
            'lease_token': token, 'lease_expires_at': expiry.isoformat(), 'retained_at': retained_at.isoformat(),
            'observations': document,
            'original_observations_sha256': contracts.document_hash(document), 'expected_payload_sha256': expected_digest,
            'expected_snapshot_schema': training_observations.SCHEMA_VERSION if 'response_receipts' in document else training_observations.LEGACY_SCHEMA_VERSION}

    def _read(self, descriptor, path):
        os.lseek(descriptor, 0, os.SEEK_SET)
        with os.fdopen(os.dup(descriptor), 'rb') as source:
            raw = source.read(self._record_limit + 1)
        if len(raw) > self._record_limit:
            raise ValueError('pending observations exceed limit')
        record = json.loads(raw)
        schema = record.get('schema_version')
        if (schema not in {_OBSERVATION_RECORD_SCHEMA, _LEGACY_OBSERVATION_RECORD_SCHEMA}
                or record.get('validator_hotkey') != self.config.identity.hotkey
                or self.config.round_id is not None and record.get('round_id') != self.config.round_id
                or path.name != self._name(record)
                or (schema == _OBSERVATION_RECORD_SCHEMA) != ('retained_at' in record)):
            raise ValueError('pending observation identity is invalid')
        expected = self._prepare(record, record['lease_token'], record['observations'],
                                 retained_at=record.get('retained_at'))
        if schema == _LEGACY_OBSERVATION_RECORD_SCHEMA:
            # Existing files retain their old claim-expiry replay fence. Never
            # manufacture a new finalization time when an old file is read.
            expected['schema_version'] = schema
            expected.pop('retained_at')
        if expected != record:
            raise ValueError('pending observation identity is invalid')
        return record

    @contextmanager
    def retained_delivery(self, lease, token, document, *, expected_digest=None):
        """Keep replay outside the foreground's publish-to-ack operation."""
        record = self._prepare(lease, token, document, expected_digest)
        path = self._directory(create=True) / self._name(record)
        with retained_record(path, contracts.canonical_json(record).encode('utf-8'), self._record_limit,
                             label=self._label) as (descriptor, created):
            if not created:
                existing = self._read(descriptor, path)
                # Repeated foreground delivery cannot refresh the replay fence.
                expected = dict(record, schema_version=existing['schema_version'])
                if 'retained_at' in existing:
                    expected['retained_at'] = existing['retained_at']
                else:
                    expected.pop('retained_at')
                if existing != expected:
                    raise ValueError('pending observation identity conflict')
            yield path, descriptor

    def retain(self, lease, token, document, *, expected_digest=None):
        with self.retained_delivery(lease, token, document, expected_digest=expected_digest) as (path, _):
            return path

    @staticmethod
    def matches(record, response):
        return (isinstance(response, Mapping) and response.get('status') == 'stored'
            and getattr(response, 'http_status', 200) == 200 and response.get('http_status', 200) == 200
            and response.get('schema_version') == record['expected_snapshot_schema']
            and response.get('ref') == training_observations.object_ref(record['run_id'], record['lease_generation'], record['claim_request_hash'])
            and type(response.get('lease_generation')) is int and response['lease_generation'] == record['lease_generation']
            and response.get('claim_request_hash') == record['claim_request_hash']
            and response.get('payload_sha256') == record['expected_payload_sha256'])

    def acknowledge_verified(self, path, payload_sha256, *, _descriptor=None):
        """Called only after the authenticated exact-digest HEAD succeeds."""
        lock = nullcontext(_descriptor) if _descriptor is not None else locked_record(path, self._record_limit, label=self._label)
        with lock as descriptor:
            if descriptor is None or not path.exists() or self._read(descriptor, path)['expected_payload_sha256'] != payload_sha256:
                return False
            path.unlink()
            sync_directory(path.parent)
            return True

    def deliver(self, path, *, replay=False, replay_deadline=None, _descriptor=None):
        if replay_deadline is not None and time.monotonic() >= replay_deadline:
            return False
        # Foreground callers already own the retained inode lock. Reopening
        # and flocking it here would conflict with that same operation.
        lock = nullcontext(_descriptor) if _descriptor is not None else locked_record(path, self._record_limit, label=self._label)
        with lock as descriptor:
            if descriptor is None or not path.exists():
                return False
            validation_started = time.monotonic()
            try:
                record = self._read(descriptor, path)
                validation_seconds = time.monotonic() - validation_started
            except (ValueError, TypeError, KeyError, AttributeError):
                self._park(path, 'invalid', descriptor)
                raise
            anchor = record.get('retained_at', record['lease_expires_at'])
            lifetime = (FINAL_OBSERVATION_REPLAY_SECONDS if 'retained_at' in record
                        else INPUT_GRACE_SECONDS)
            remaining = (_time(anchor) + timedelta(seconds=lifetime) - self.config.clock()).total_seconds()
            # Replay gets one bounded window from final snapshot retention;
            # legacy files keep their original claim-expiry fence. Foreground
            # delivery uses current server authority after provider renewals.
            if replay and remaining <= 0:
                self._park(path, 'expired', descriptor)
                return False
            if replay_deadline is not None:
                # This final was admitted before the batch deadline. Its
                # local privacy/identity validation can exceed the network
                # budget for large documents. Exclude only that work here;
                # the outer deadline still stops admission of later records.
                remaining = min(remaining, replay_deadline + validation_seconds - time.monotonic())
                if remaining <= 0:
                    return False
            if replay and self._stop.is_set():
                return False
            try:
                args = (record['run_id'], record['lease_token'], record['observations'])
                response = (self.config.api.training_observations(*args,
                    timeout_seconds=min(INPUT_REPLAY_TIMEOUT_SECONDS, remaining)) if replay
                    else self.config.api.training_observations(*args))
            except Exception as exc:
                if getattr(exc, 'http_status', None) in {400, 401, 403, 404, 409, 413, 422}:
                    self._park(path, 'rejected', descriptor)
                raise
            if self.matches(record, response):
                path.unlink()
                sync_directory(path.parent)
                return True
            status = getattr(response, 'http_status', None)
            if status is None and isinstance(response, Mapping):
                status = response.get('http_status')
            if status in {400, 401, 403, 404, 409, 413, 422}:
                self._park(path, 'rejected', descriptor)
            return False
