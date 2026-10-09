"""Bounded, best-effort operational events over the validator's existing identity.

The queue is memory-only. Gateway failure never changes scoring or weights.
No exception messages, environment values or provider call bodies enter events.
"""
from __future__ import annotations

import json
import uuid
import threading
import time
import urllib.request
import urllib.error
from collections import deque
from datetime import datetime, timezone
from urllib.parse import urlsplit

from lab_arena import contracts
from lab_arena.runtime_host import RuntimeHostError
from lab_arena.scoring_startup import ScoringStartupError


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class ValidatorOperationalLogger:
    QUEUE_LIMIT = 64
    BATCH_LIMIT = 16
    TRANSPORT_SECONDS = 3.0
    HEARTBEAT_SECONDS = 300.0
    ERROR_INTERVAL_SECONDS = 300.0

    def __init__(self, base_url, *, keypair, network, netuid,
                 now=time.time, monotonic=time.monotonic, transport=None):
        self.base_url = str(base_url).rstrip('/')
        try:
            parsed = urlsplit(self.base_url)
            parsed.port
            secure = parsed.scheme == "https" and bool(parsed.hostname)
            loopback = parsed.scheme == "http" and parsed.hostname in {"localhost", "127.0.0.1", "::1"}
            self._valid_origin = (secure or loopback) and parsed.username is None and parsed.password is None and not parsed.fragment
        except ValueError:
            self._valid_origin = False
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
        self.keypair = keypair
        self.network = str(network)
        self.netuid = int(netuid)
        self.session_id = str(uuid.uuid4())
        self._now = now
        self._monotonic = monotonic
        self._transport = transport or self._post
        self._lock = threading.Lock()
        self._queue = deque(maxlen=self.QUEUE_LIMIT)
        self._errors = {}
        self._state = {"scoring_state": "starting", "weights_state": "starting", "active_runs": 0}
        self._progress = self._monotonic()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        if self._thread is None:
            try:
                self._thread = threading.Thread(target=self._send_loop, name="arena-validator-events", daemon=True)
                self._thread.start()
            except Exception:
                self._thread = None  # Missing telemetry cannot prevent validator startup.

    def emit(self, kind, content, *, run_id=None, round_id=None):
        try:
            from lab_arena.validator_events import event
            from lab_arena.runtime_version import SOURCE_METADATA
            value = event(kind, {**SOURCE_METADATA, **content, "session_id": self.session_id}, run_id=run_id, round_id=round_id)
            with self._lock:
                self._queue.append(value)
            self._wake.set()
        except Exception:
            # Observability is never a prerequisite for executing work.
            pass

    def state(self, *, progress=False, **fields):
        with self._lock:
            self._state.update(fields)
            if progress:
                self._progress = self._monotonic()
                self._state["last_progress_at"] = datetime.fromtimestamp(self._now(), timezone.utc).isoformat()

    def activity(self, delta):
        with self._lock:
            self._state["active_runs"] = max(0, self._state.get("active_runs", 0) + delta)
            self._state["scoring_state"] = "active" if self._state["active_runs"] else "claiming"
            self._progress = self._monotonic()
            self._state["last_progress_at"] = datetime.fromtimestamp(self._now(), timezone.utc).isoformat()

    def snapshot(self):
        with self._lock:
            fields = dict(self._state)
        # An active sandbox may legitimately run for an hour. Lack of poll
        # progress is separately visible without falsely declaring it dead.
        if fields.get("active_runs") and fields.get("scoring_state") in {"discovering", "claiming"}:
            fields["scoring_state"] = "active"
        return fields

    def error(self, phase, exc, *, run_id=None, round_id=None, reason=None, http_status=None, denial_code=None):
        fields = {"phase": phase, "error_class": type(exc).__name__[:128]}
        if isinstance(exc, (RuntimeHostError, ScoringStartupError)):
            fields["reason"] = exc.reason
        elif reason:
            fields["reason"] = reason
        if type(http_status) is int and 100 <= http_status <= 599:
            fields["http_status"] = http_status
        if denial_code:
            fields["denial_code"] = denial_code
        if isinstance(exc, ScoringStartupError):
            if exc.operation:
                fields["operation"] = exc.operation
            if exc.http_status is not None:
                fields["http_status"] = exc.http_status
        if isinstance(exc, RuntimeHostError):
            fields.update({"launch_timed_out": exc.launch_timed_out,
                           "launch_stderr_truncated": exc.launch_stderr_truncated})
            if exc.launch_exit_code is not None:
                fields["launch_exit_code"] = exc.launch_exit_code
            if exc.launch_stderr:
                encoded = exc.launch_stderr.encode("utf-8")
                fields["launch_stderr"] = encoded[:2048].decode("utf-8", errors="ignore")
                fields["launch_stderr_truncated"] = exc.launch_stderr_truncated or len(encoded) > 2048
        key = (phase, fields.get("reason"), fields["error_class"],
               fields.get("operation"), fields.get("http_status"))
        with self._lock:
            previous = self._errors.get(key)
            current = self._monotonic()
            if previous is not None and current - previous < self.ERROR_INTERVAL_SECONDS:
                return
            if len(self._errors) >= self.QUEUE_LIMIT:
                self._errors.pop(next(iter(self._errors)))
            self._errors[key] = current
        self.emit("validator.error", fields, run_id=run_id, round_id=round_id)

    def recovered(self, phase, *, run_id=None, round_id=None):
        with self._lock:
            keys = [key for key in self._errors if key[0] == phase]
            for key in keys:
                del self._errors[key]
        if keys:
            self.emit("validator.recovered", {"phase": phase}, run_id=run_id, round_id=round_id)

    def _post(self, envelope):
        if not self._valid_origin:
            raise OSError("validator event origin invalid")
        request = urllib.request.Request(
            self.base_url + "/arena/v1/validators/events",
            data=json.dumps(envelope, sort_keys=True, separators=(",", ":")).encode(),
            method="POST", headers={"Content-Type": "application/json"},
        )
        # No response body is needed, and proxy errors may include credentials.
        with self._opener.open(request, timeout=self.TRANSPORT_SECONDS) as response:
            if response.status != 200:
                raise OSError("validator event delivery failed")

    def _send_loop(self):
        heartbeat_at = self._monotonic() + self.HEARTBEAT_SECONDS
        retry_at = 0.0
        failures = 0
        while True:
            current = self._monotonic()
            closing = self._stop.is_set()
            if current >= heartbeat_at and not self._stop.is_set():
                self.emit("validator.state", self.snapshot())
                heartbeat_at = current + self.HEARTBEAT_SECONDS
            self._wake.clear()
            with self._lock:
                batch = []
                if current >= retry_at or self._stop.is_set():
                    for value in list(self._queue)[:self.BATCH_LIMIT]:
                        if len(json.dumps(batch + [value]).encode()) > 27 * 1024:
                            break
                        batch.append(value)
            if batch:
                try:
                    envelope = contracts.build_signed_request(
                        scope=contracts.SCOPE_VALIDATOR_EVENTS, round_id="validator-events",
                        hotkey=self.keypair.ss58_address,
                        body={"network": self.network, "netuid": self.netuid, "events": batch},
                        timestamp=int(self._now()),
                        sign_message=lambda message: self.keypair.sign(message.encode()).hex(),
                    )
                    self._transport(envelope)
                except Exception as exc:
                    if isinstance(exc, urllib.error.HTTPError):
                        exc.close()  # Never consume upstream error bodies.
                    failures += 1
                    retry_at = self._monotonic() + min(60.0, 5.0 * 2 ** min(failures - 1, 4))
                    delivered = failures >= 3 or (isinstance(exc, urllib.error.HTTPError) and exc.code in {400, 401, 403, 413, 429})
                    if delivered:
                        failures = 0
                else:
                    failures = 0
                    retry_at = self._monotonic() + 20.0
                    delivered = True
                if delivered:
                    ids = {value["event_id"] for value in batch}
                    with self._lock:
                        # New arrivals can evict old entries while HTTP blocks.
                        self._queue = deque((value for value in self._queue if value["event_id"] not in ids), maxlen=self.QUEUE_LIMIT)
            if self._stop.is_set():
                if closing:
                    return  # Exactly one final batch; no retry or spool.
                # close() can run while HTTP is in flight. Capture events that
                # arrived during that request before ending the sender.
                continue
            with self._lock:
                pending = bool(self._queue)
            delay = min(heartbeat_at - self._monotonic(), max(0.0, retry_at - self._monotonic()) if pending else self.HEARTBEAT_SECONDS)
            self._wake.wait(max(0.01, delay))

    def close(self):
        self.emit("validator.stopping", {"phase": "shutdown"})
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(2 * self.TRANSPORT_SECONDS)
