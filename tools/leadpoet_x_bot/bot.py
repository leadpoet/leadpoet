#!/usr/bin/env python3
"""Small, durable X reply bot using only the Python standard library."""

from __future__ import annotations

import argparse
import base64
import contextlib
import dataclasses
import datetime as dt
import email.utils
import fcntl
import hashlib
import hmac
import json
import logging
import math
import os
import random
import re
import signal
import sqlite3
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping


API_BASE = "https://api.x.com"
DEFAULT_REPLY_TEXT = (
    "Book a quick demo, we’ll define your ICP, and get you 100 free lead credits:\n\n"
    "https://cal.com/team/leadpoet/chat"
)
DEFAULT_PUBLIC_REPLY = "Sending!"
DEFAULT_DM_UNAVAILABLE_REPLY = "sending, please open your Dms!"
DEFAULT_POET_ALIASES = ("poett", "poeet", "poat", "poer", "poey", "peot")
UTC = dt.timezone.utc


class BotError(RuntimeError):
    """Base error suitable for a concise startup/runtime message."""


class ConfigurationError(BotError):
    pass


class FatalApiError(BotError):
    pass


class TransientApiError(BotError):
    def __init__(self, message: str, retry_at: float | None = None):
        super().__init__(message)
        self.retry_at = retry_at


class RateLimitedCycle(BotError):
    def __init__(self, retry_at: float):
        super().__init__("X rate limit is active")
        self.retry_at = retry_at


@dataclasses.dataclass(frozen=True)
class Credentials:
    api_key: str
    api_secret: str
    access_token: str
    access_token_secret: str

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> "Credentials":
        names = ("X_API_KEY", "X_API_SECRET", "X_ACCESS_TOKEN", "X_ACCESS_TOKEN_SECRET")
        missing = [name for name in names if not env.get(name)]
        if missing:
            raise ConfigurationError("Missing X credentials: " + ", ".join(missing))
        return cls(*(env[name] for name in names))


def _strict_bool(value: str, name: str) -> bool:
    lowered = value.strip().lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    raise ConfigurationError(f"{name} must be exactly true or false")


def _csv_ids(value: str) -> tuple[str, ...]:
    result = tuple(dict.fromkeys(item.strip() for item in value.split(",") if item.strip()))
    if any(not item.isdigit() for item in result):
        raise ConfigurationError("Post and user IDs must contain decimal digits only")
    return result


def parse_post_url(value: str) -> str:
    parsed = urllib.parse.urlsplit(value.strip())
    if (
        parsed.scheme != "https"
        or parsed.netloc.lower() not in {"x.com", "www.x.com", "twitter.com", "www.twitter.com"}
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise ConfigurationError(f"Invalid monitored X post URL: {value}")
    match = re.fullmatch(r"/[^/]+/status/(\d+)/?", parsed.path)
    if not match:
        raise ConfigurationError(f"Invalid monitored X post URL: {value}")
    return match.group(1)


@dataclasses.dataclass(frozen=True)
class Config:
    monitored_post_ids: tuple[str, ...]
    trigger_word: str = "Poet"
    trigger_aliases: tuple[str, ...] = DEFAULT_POET_ALIASES
    reply_text: str = DEFAULT_PUBLIC_REPLY
    dm_message: str = DEFAULT_REPLY_TEXT
    dm_unavailable_reply: str = DEFAULT_DM_UNAVAILABLE_REPLY
    poll_interval_seconds: float = 300.0
    dry_run: bool = True
    state_db: Path = Path("state.sqlite3")
    denied_user_ids: frozenset[str] = frozenset()
    request_timeout_seconds: float = 30.0

    @property
    def namespace(self) -> str:
        return "dry" if self.dry_run else "live"

    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str] | None = None,
    ) -> "Config":
        values = os.environ if env is None else env
        url_ids = tuple(
            parse_post_url(item)
            for item in values.get("MONITORED_POST_URLS", "").split(",")
            if item.strip()
        )
        ids = tuple(dict.fromkeys((*url_ids, *_csv_ids(values.get("MONITORED_POST_IDS", "")))))
        if not ids:
            raise ConfigurationError("MONITORED_POST_URLS or MONITORED_POST_IDS must configure a post")
        trigger = values.get("TRIGGER_WORD", "Poet").strip()
        if not trigger:
            raise ConfigurationError("TRIGGER_WORD must not be empty")
        if "TRIGGER_ALIASES" in values:
            aliases = tuple(dict.fromkeys(item.strip() for item in values["TRIGGER_ALIASES"].split(",") if item.strip()))
        else:
            aliases = DEFAULT_POET_ALIASES if trigger.casefold() == "poet" else ()
        try:
            poll = float(values.get("POLL_INTERVAL_SECONDS", "300"))
        except ValueError as exc:
            raise ConfigurationError("POLL_INTERVAL_SECONDS must be a number") from exc
        if not math.isfinite(poll) or poll <= 0:
            raise ConfigurationError("POLL_INTERVAL_SECONDS must be positive")
        reply_text = values.get("REPLY_TEXT", DEFAULT_PUBLIC_REPLY)
        dm_message = values.get("DM_MESSAGE", DEFAULT_REPLY_TEXT)
        unavailable_reply = values.get("DM_UNAVAILABLE_REPLY", DEFAULT_DM_UNAVAILABLE_REPLY)
        if not all(item.strip() for item in (reply_text, dm_message, unavailable_reply)):
            raise ConfigurationError("Reply and DM messages must not be empty")
        return cls(
            monitored_post_ids=tuple(ids),
            trigger_word=trigger,
            trigger_aliases=aliases,
            reply_text=reply_text,
            dm_message=dm_message,
            dm_unavailable_reply=unavailable_reply,
            poll_interval_seconds=poll,
            dry_run=_strict_bool(values.get("DRY_RUN", "true"), "DRY_RUN"),
            state_db=Path(values.get("STATE_DB", "state.sqlite3")).expanduser(),
            denied_user_ids=frozenset(_csv_ids(values.get("DENIED_USER_IDS", ""))),
        )


def contains_trigger(text: str, trigger_word: str = "Poet", aliases: Iterable[str] = ()) -> bool:
    """Return true for a case-insensitive whole-word trigger."""
    words = tuple(dict.fromkeys((trigger_word, *aliases)))
    pattern = "|".join(re.escape(word) for word in words)
    return re.search(rf"(?<!\w)(?:{pattern})(?!\w)", text, re.IGNORECASE) is not None


def _oauth_quote(value: object) -> str:
    return urllib.parse.quote(str(value), safe="~-._")


def oauth1_authorization(
    method: str,
    url: str,
    credentials: Credentials,
    *,
    query: Mapping[str, object] | None = None,
    nonce: str | None = None,
    timestamp: int | None = None,
) -> str:
    """Create an OAuth 1.0a HMAC-SHA1 Authorization header."""
    oauth = {
        "oauth_consumer_key": credentials.api_key,
        "oauth_nonce": nonce or uuid.uuid4().hex,
        "oauth_signature_method": "HMAC-SHA1",
        "oauth_timestamp": str(int(time.time() if timestamp is None else timestamp)),
        "oauth_token": credentials.access_token,
        "oauth_version": "1.0",
    }
    pairs: list[tuple[str, str]] = []
    for source in (urllib.parse.parse_qsl(urllib.parse.urlsplit(url).query), (query or {}).items(), oauth.items()):
        pairs.extend((_oauth_quote(key), _oauth_quote(value)) for key, value in source)
    normalized = "&".join(f"{key}={value}" for key, value in sorted(pairs))
    clean_url = urllib.parse.urlunsplit((*urllib.parse.urlsplit(url)[:2], urllib.parse.urlsplit(url).path, "", ""))
    base = "&".join(_oauth_quote(part) for part in (method.upper(), clean_url, normalized))
    key = f"{_oauth_quote(credentials.api_secret)}&{_oauth_quote(credentials.access_token_secret)}"
    oauth["oauth_signature"] = base64.b64encode(
        hmac.new(key.encode(), base.encode(), hashlib.sha1).digest()
    ).decode()
    return "OAuth " + ", ".join(
        f'{_oauth_quote(key)}="{_oauth_quote(value)}"' for key, value in sorted(oauth.items())
    )


def _retry_delay(headers: Mapping[str, str], now: float, fallback: float) -> float:
    normalized = {str(key).lower(): value for key, value in headers.items()}
    candidates = [fallback]
    retry_after = normalized.get("retry-after")
    if retry_after:
        try:
            candidates.append(max(0.0, float(retry_after)))
        except ValueError:
            try:
                parsed = email.utils.parsedate_to_datetime(retry_after)
            except (TypeError, ValueError, OverflowError):
                parsed = None
            if parsed is not None:
                candidates.append(max(0.0, parsed.timestamp() - now))
    reset = normalized.get("x-rate-limit-reset")
    if reset:
        try:
            candidates.append(max(0.0, float(reset) - now + 1.0))
        except ValueError:
            pass
    return max(candidates)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> None:
        return None


class XApiClient:
    def __init__(
        self,
        credentials: Credentials,
        *,
        api_base: str = API_BASE,
        opener: Callable[..., Any] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        now: Callable[[], float] = time.time,
        timeout: float = 30.0,
        get_max_attempts: int = 5,
    ) -> None:
        self.credentials = credentials
        self.api_base = api_base.rstrip("/")
        self.opener = opener or urllib.request.build_opener(_NoRedirect()).open
        self.sleep = sleep
        self.now = now
        self.timeout = timeout
        self.get_max_attempts = get_max_attempts

    def _request_once(
        self, method: str, path: str, query: Mapping[str, object] | None = None, payload: object = None
    ) -> tuple[int, Mapping[str, str], object]:
        url = self.api_base + path
        encoded_query = urllib.parse.urlencode(query or {})
        request_url = url + ("?" + encoded_query if encoded_query else "")
        body = None if payload is None else json.dumps(payload, separators=(",", ":")).encode()
        headers = {
            "Accept": "application/json",
            "Authorization": oauth1_authorization(method, url, self.credentials, query=query),
            "User-Agent": "leadpoet-x-reply-bot/1",
        }
        if body is not None:
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(request_url, data=body, headers=headers, method=method)
        try:
            response = self.opener(request, timeout=self.timeout)
            with contextlib.closing(response):
                raw = response.read()
                status = int(getattr(response, "status", response.getcode()))
                response_headers = dict(response.headers.items())
        except urllib.error.HTTPError as exc:
            # Parse structured errors for routing, but never log or expose the body.
            try:
                raw = exc.read(65536)
                decoded = json.loads(raw) if raw else None
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                decoded = None
            with contextlib.suppress(Exception):
                exc.close()
            return int(exc.code), dict(exc.headers.items()), decoded
        if not raw:
            decoded: object = None
        else:
            try:
                decoded = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError):
                decoded = None
        return status, response_headers, decoded

    def get_json(self, path: str, query: Mapping[str, object] | None = None) -> Mapping[str, Any]:
        last_reason = "request failed"
        for attempt in range(self.get_max_attempts):
            try:
                status, headers, body = self._request_once("GET", path, query)
            except (OSError, TimeoutError) as exc:
                status, headers, body = 0, {}, None
                last_reason = type(exc).__name__
            if status in (401, 403):
                raise FatalApiError(f"X GET {path} was rejected with HTTP {status}")
            if 200 <= status < 300:
                if isinstance(body, Mapping):
                    return body
                raise FatalApiError(f"X GET {path} returned malformed JSON")
            retryable = status in (0, 408, 425, 429) or status >= 500
            if not retryable:
                raise FatalApiError(f"X GET {path} failed with HTTP {status}")
            last_reason = f"HTTP {status}" if status else last_reason
            if status == 429 and attempt + 1 == self.get_max_attempts:
                raise RateLimitedCycle(self.now() + _retry_delay(headers, self.now(), 60.0))
            if attempt + 1 < self.get_max_attempts:
                backoff = min(60.0, 2.0**attempt) + random.random() * 0.25
                self.sleep(_retry_delay(headers, self.now(), backoff))
        raise TransientApiError(f"X GET {path} failed after retries ({last_reason})")

    def get_me(self) -> Mapping[str, Any]:
        body = self.get_json("/2/users/me")
        data = body.get("data")
        if not isinstance(data, Mapping) or not str(data.get("id", "")):
            raise FatalApiError("X GET /2/users/me omitted the user ID")
        return data

    def get_tweet(self, tweet_id: str) -> Mapping[str, Any]:
        body = self.get_json(
            f"/2/tweets/{urllib.parse.quote(tweet_id, safe='')}",
            {"tweet.fields": "author_id,conversation_id,created_at"},
        )
        data = body.get("data")
        if not isinstance(data, Mapping):
            raise FatalApiError(f"Configured post {tweet_id} was not returned by X")
        return data

    def search_replies(
        self, conversation_id: str, start_time: str, end_time: str
    ) -> Iterable[Mapping[str, Any]]:
        token: str | None = None
        seen_tokens: set[str] = set()
        while True:
            query: dict[str, object] = {
                "query": f"conversation_id:{conversation_id} is:reply -is:retweet",
                "tweet.fields": "author_id,referenced_tweets,conversation_id,created_at",
                "max_results": 100,
                "start_time": start_time,
                "end_time": end_time,
            }
            if token:
                query["next_token"] = token
            body = self.get_json("/2/tweets/search/recent", query)
            if body.get("errors"):
                raise TransientApiError("X recent search returned errors")
            if "data" not in body:
                data = []
            else:
                data = body["data"]
            if data is not None and not isinstance(data, list):
                raise FatalApiError("X recent search returned malformed data")
            for tweet in data or []:
                if not isinstance(tweet, Mapping):
                    raise FatalApiError("X recent search returned a malformed post")
                yield tweet
            meta = body.get("meta")
            if not isinstance(meta, Mapping) or ("result_count" not in meta and "next_token" not in meta):
                raise FatalApiError("X recent search returned malformed metadata")
            next_token = meta.get("next_token")
            if not next_token:
                return
            token = str(next_token)
            if token in seen_tokens:
                raise TransientApiError("X recent search repeated a pagination token")
            seen_tokens.add(token)

    @staticmethod
    def _dm_unavailable(body: object) -> bool:
        if not isinstance(body, Mapping):
            return False
        errors = body.get("errors", [])
        if isinstance(errors, Mapping):
            errors = [errors]
        if not isinstance(errors, list):
            return False
        candidates = [body, *errors]
        for error in candidates:
            if not isinstance(error, Mapping):
                continue
            try:
                code = int(error.get("code"))
            except (TypeError, ValueError):
                code = None
            if code in (150, 349):
                return True
            if any(
                str(error.get(field, "")).strip().rstrip(".").casefold()
                == "you cannot send messages to this user"
                for field in ("message", "detail", "title")
            ):
                return True
        return False

    def _send(
        self, path: str, payload: Mapping[str, Any], success_key: str, *, dm: bool = False
    ) -> tuple[str, float | str | None]:
        try:
            status, headers, body = self._request_once("POST", path, payload=payload)
        except (OSError, TimeoutError) as exc:
            return "unknown", type(exc).__name__
        if status == 401:
            raise FatalApiError(f"X POST {path} was rejected with HTTP 401")
        if status == 429:
            return "rate_limited", self.now() + _retry_delay(headers, self.now(), 60.0)
        if status == 403 and dm and self._dm_unavailable(body):
            return "unavailable", "recipient unavailable"
        if status == 403:
            return "rejected", "HTTP 403"
        if 200 <= status < 300:
            data = body.get("data") if isinstance(body, Mapping) else None
            result_id = data.get(success_key) if isinstance(data, Mapping) else None
            return ("sent", str(result_id)) if result_id else ("unknown", "malformed success")
        if status == 0 or status >= 500:
            return "unknown", f"HTTP {status}" if status else "transport error"
        return "rejected", f"HTTP {status}"

    def send_public_reply(self, reply_to_id: str, text: str) -> tuple[str, float | str | None]:
        return self._send(
            "/2/tweets", {"text": text, "reply": {"in_reply_to_tweet_id": reply_to_id}}, "id"
        )

    def send_dm(self, user_id: str, text: str) -> tuple[str, float | str | None]:
        path = f"/2/dm_conversations/with/{urllib.parse.quote(user_id, safe='')}/messages"
        return self._send(path, {"text": text}, "dm_event_id", dm=True)


class StateStore:
    def __init__(self, path: Path | str, namespace: str) -> None:
        self.path = Path(path)
        self.namespace = namespace
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, timeout=30.0)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS comments (
                namespace TEXT NOT NULL,
                reply_id TEXT NOT NULL,
                parent_id TEXT NOT NULL,
                author_id TEXT NOT NULL,
                completed INTEGER NOT NULL DEFAULT 0,
                discovered_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                PRIMARY KEY (namespace, reply_id)
            );
            CREATE TABLE IF NOT EXISTS actions (
                namespace TEXT NOT NULL,
                reply_id TEXT NOT NULL,
                action TEXT NOT NULL,
                status TEXT NOT NULL,
                claimed_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                retry_at REAL,
                detail TEXT,
                PRIMARY KEY (namespace, reply_id, action)
            );
            CREATE TABLE IF NOT EXISTS cursors (
                namespace TEXT NOT NULL,
                post_id TEXT NOT NULL,
                completed_through REAL NOT NULL,
                PRIMARY KEY (namespace, post_id)
            );
            CREATE TABLE IF NOT EXISTS metadata (
                namespace TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                PRIMARY KEY (namespace, key)
            );
            """
        )
        self.db.commit()

    def close(self) -> None:
        self.db.close()

    def cursor(self, post_id: str) -> float | None:
        row = self.db.execute(
            "SELECT completed_through FROM cursors WHERE namespace=? AND post_id=?",
            (self.namespace, post_id),
        ).fetchone()
        return float(row[0]) if row else None

    def checkpoint(self, post_id: str, completed_through: float) -> None:
        with self.db:
            self.db.execute(
                "INSERT INTO cursors(namespace,post_id,completed_through) VALUES(?,?,?) "
                "ON CONFLICT(namespace,post_id) DO UPDATE SET completed_through=excluded.completed_through",
                (self.namespace, post_id, completed_through),
            )

    def add_comment(self, reply_id: str, parent_id: str, author_id: str, now: float) -> bool:
        with self.db:
            inserted = self.db.execute(
                "INSERT OR IGNORE INTO comments "
                "(namespace,reply_id,parent_id,author_id,completed,discovered_at,updated_at) "
                "VALUES(?,?,?,?,0,?,?)",
                (self.namespace, reply_id, parent_id, author_id, now, now),
            )
        return bool(inserted.rowcount)

    def pending_comments(self, parent_ids: Iterable[str]) -> list[dict[str, str]]:
        ids = tuple(parent_ids)
        if not ids:
            return []
        marks = ",".join("?" for _ in ids)
        rows = self.db.execute(
            f"SELECT reply_id,parent_id,author_id FROM comments "
            f"WHERE namespace=? AND completed=0 AND parent_id IN ({marks}) ORDER BY discovered_at,reply_id",
            (self.namespace, *ids),
        ).fetchall()
        return [dict(zip(("reply_id", "parent_id", "author_id"), row)) for row in rows]

    def complete_comment(self, reply_id: str, now: float) -> None:
        with self.db:
            self.db.execute(
                "UPDATE comments SET completed=1,updated_at=? WHERE namespace=? AND reply_id=?",
                (now, self.namespace, reply_id),
            )

    def claim_action(self, reply_id: str, action: str, now: float) -> str:
        with self.db:
            cursor = self.db.execute(
                "INSERT OR IGNORE INTO actions "
                "(namespace,reply_id,action,status,claimed_at,updated_at) VALUES(?,?,?,'sending',?,?)",
                (self.namespace, reply_id, action, now, now),
            )
            if cursor.rowcount:
                return "claimed"
            row = self.db.execute(
                "SELECT status,retry_at FROM actions WHERE namespace=? AND reply_id=? AND action=?",
                (self.namespace, reply_id, action),
            ).fetchone()
            if row and row[0] == "retry" and (row[1] is None or float(row[1]) <= now):
                changed = self.db.execute(
                    "UPDATE actions SET status='sending',updated_at=?,retry_at=NULL "
                    "WHERE namespace=? AND reply_id=? AND action=? AND status='retry'",
                    (now, self.namespace, reply_id, action),
                )
                if changed.rowcount:
                    return "claimed"
        return str(row[0]) if row else "unknown"

    def set_action(
        self,
        reply_id: str,
        action: str,
        status: str,
        now: float,
        detail: str | None = None,
        retry_at: float | None = None,
    ) -> None:
        with self.db:
            self.db.execute(
                "UPDATE actions SET status=?,updated_at=?,detail=?,retry_at=? "
                "WHERE namespace=? AND reply_id=? AND action=?",
                (status, now, detail, retry_at, self.namespace, reply_id, action),
            )

    def cooldown_until(self) -> float | None:
        row = self.db.execute(
            "SELECT value FROM metadata WHERE namespace=? AND key='post_cooldown_until'", (self.namespace,)
        ).fetchone()
        return float(row[0]) if row else None

    def set_cooldown(self, retry_at: float) -> None:
        with self.db:
            self.db.execute(
                "INSERT INTO metadata(namespace,key,value) VALUES(?,'post_cooldown_until',?) "
                "ON CONFLICT(namespace,key) DO UPDATE SET value=excluded.value",
                (self.namespace, str(retry_at)),
            )


class ProcessLock:
    def __init__(self, state_path: Path | str) -> None:
        self.path = Path(str(state_path) + ".lock")
        self.file: Any = None

    def __enter__(self) -> "ProcessLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("a+")
        try:
            fcntl.flock(self.file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.file.close()
            raise BotError(f"Another bot process holds {self.path}") from exc
        self.file.seek(0)
        self.file.truncate()
        self.file.write(str(os.getpid()))
        self.file.flush()
        return self

    def __exit__(self, *_: object) -> None:
        if self.file is not None:
            fcntl.flock(self.file.fileno(), fcntl.LOCK_UN)
            self.file.close()


def _to_timestamp(value: str) -> float:
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def _rfc3339(value: float) -> str:
    return dt.datetime.fromtimestamp(value, UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def direct_parent_id(tweet: Mapping[str, Any]) -> str | None:
    references = tweet.get("referenced_tweets", [])
    if not isinstance(references, list):
        return None
    replied = [ref for ref in references if isinstance(ref, Mapping) and ref.get("type") == "replied_to"]
    return str(replied[0].get("id")) if len(replied) == 1 and replied[0].get("id") else None


class FixtureClient:
    def __init__(self, fixture: Mapping[str, Any]) -> None:
        self.own_user_id = str(fixture["own_user_id"])
        unavailable = fixture.get("dm_unavailable_user_ids", [])
        if not isinstance(unavailable, list):
            raise ConfigurationError("Fixture dm_unavailable_user_ids must be a JSON list")
        self.dm_unavailable_user_ids = frozenset(str(item) for item in unavailable)
        self.posts: list[Mapping[str, Any]] = []
        for item in fixture.get("posts", []):
            if not isinstance(item, Mapping):
                raise ConfigurationError("Fixture posts must be JSON objects")
            root = {key: value for key, value in item.items() if key != "replies"}
            self.posts.append(root)
            replies = item.get("replies", [])
            if not isinstance(replies, list):
                raise ConfigurationError("Fixture replies must be a JSON list")
            self.posts.extend(reply for reply in replies if isinstance(reply, Mapping))

    def get_me(self) -> Mapping[str, Any]:
        return {"id": self.own_user_id}

    def get_tweet(self, tweet_id: str) -> Mapping[str, Any]:
        for post in self.posts:
            if str(post.get("id")) == tweet_id:
                return post
        raise FatalApiError(f"Configured post {tweet_id} is absent from fixture")

    def search_replies(self, conversation_id: str, start_time: str, end_time: str) -> Iterable[Mapping[str, Any]]:
        start, end = _to_timestamp(start_time), _to_timestamp(end_time)
        for post in self.posts:
            created = post.get("created_at")
            created_ts = _to_timestamp(str(created)) if created else end
            if str(post.get("conversation_id")) == conversation_id and start <= created_ts <= end:
                yield post

    def send_public_reply(self, reply_to_id: str, text: str) -> tuple[str, str]:
        raise AssertionError("Fixture mode cannot post")

    def send_dm(self, user_id: str, text: str) -> tuple[str, str]:
        raise AssertionError("Fixture mode cannot send DMs")


def load_fixture(path: Path | str) -> Mapping[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"Could not load fixture {path}: {type(exc).__name__}") from exc
    if not isinstance(value, Mapping) or not str(value.get("own_user_id", "")) or not isinstance(value.get("posts"), list):
        raise ConfigurationError("Fixture requires own_user_id and a posts list")
    return value


class Bot:
    def __init__(
        self,
        config: Config,
        client: Any,
        store: StateStore,
        *,
        now: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
        logger: logging.Logger | None = None,
    ) -> None:
        self.config = config
        self.client = client
        self.store = store
        self.now = now
        self.sleep = sleep
        self.log = logger or logging.getLogger("leadpoet_x_bot")
        self.stopping = False
        self.own_user_id = ""
        self.roots: dict[str, Mapping[str, Any]] = {}

    def stop(self, *_: object) -> None:
        self.stopping = True

    def validate_startup(self) -> None:
        self.own_user_id = str(self.client.get_me().get("id", ""))
        if not self.own_user_id:
            raise FatalApiError("Authenticated X user ID is missing")
        roots: dict[str, Mapping[str, Any]] = {}
        for post_id in self.config.monitored_post_ids:
            post = self.client.get_tweet(post_id)
            if str(post.get("id", "")) != post_id:
                raise FatalApiError(f"Configured post {post_id} has an unexpected ID")
            if str(post.get("author_id", "")) != self.own_user_id:
                raise ConfigurationError(f"Configured post {post_id} is not owned by the authenticated account")
            if str(post.get("conversation_id", "")) != post_id:
                raise ConfigurationError(f"Configured post {post_id} is not a root post")
            roots[post_id] = post
        self.roots = roots
        self.log.info("Monitoring %s as account %s; dry_run=%s", ",".join(roots), self.own_user_id, self.config.dry_run)

    def _window(self, post_id: str, end: float) -> tuple[float, float]:
        oldest = float(int(self.now() - 7 * 24 * 60 * 60 + 60))
        cursor = self.store.cursor(post_id)
        if cursor is None:
            root_created = self.roots[post_id].get("created_at")
            root_time = _to_timestamp(str(root_created)) if root_created else oldest
            start = max(oldest, root_time)
            if root_time < oldest:
                self.log.warning("Configured post %s predates the recent-search window; older replies cannot be recovered", post_id)
        else:
            if cursor - 120 < oldest:
                self.log.warning("Search checkpoint for %s has a gap beyond X's seven-day recent-search limit", post_id)
            start = max(oldest, cursor - 120)
        return start, end

    def _handle_reply(self, post_id: str, tweet: Mapping[str, Any]) -> None:
        reply_id = str(tweet.get("id", ""))
        author_id = str(tweet.get("author_id", ""))
        if not reply_id or not author_id or author_id == self.own_user_id or author_id in self.config.denied_user_ids:
            return
        if str(tweet.get("conversation_id", "")) != post_id or direct_parent_id(tweet) != post_id:
            return
        if not contains_trigger(
            str(tweet.get("text", "")), self.config.trigger_word, self.config.trigger_aliases
        ):
            return
        if not self.store.add_comment(reply_id, post_id, author_id, self.now()):
            return
        self.log.info("Detected qualifying comment %s under post %s", reply_id, post_id)
        self._process_comment({"reply_id": reply_id, "parent_id": post_id, "author_id": author_id})

    def _run_action(self, comment: Mapping[str, str], action: str, public_text: str | None = None) -> str:
        reply_id = comment["reply_id"]
        author_id = comment["author_id"]
        if self.stopping:
            return "stopping"
        current = self.now()
        state = self.store.claim_action(reply_id, action, current)
        if state == "retry":
            retry_at = self.store.cooldown_until() or current + self.config.poll_interval_seconds
            raise RateLimitedCycle(retry_at)
        if state != "claimed":
            return state
        if self.config.dry_run:
            unavailable = getattr(self.client, "dm_unavailable_user_ids", frozenset())
            if not isinstance(unavailable, (set, frozenset, list, tuple)):
                unavailable = frozenset()
            status = "unavailable" if action == "dm" and author_id in unavailable else "sent"
            detail: float | str | None = "dry_run"
            self.log.info("Dry run would perform %s for post %s", action, reply_id)
        elif action == "dm":
            self.log.info("Sending DM for eligible reply %s", reply_id)
            status, detail = self.client.send_dm(author_id, self.config.dm_message)
        else:
            text = public_text if public_text is not None else self.config.reply_text
            self.log.info("Sending %s public reply for eligible reply %s", action, reply_id)
            status, detail = self.client.send_public_reply(reply_id, text)
        finished = self.now()
        if status == "rate_limited":
            retry_at = float(detail)
            self.store.set_action(reply_id, action, "retry", finished, "HTTP 429", retry_at)
            self.store.set_cooldown(retry_at)
            raise RateLimitedCycle(retry_at)
        self.store.set_action(reply_id, action, status, finished, str(detail))
        if not self.config.dry_run and status == "sent":
            self.log.info("%s succeeded for comment %s; event_id=%s", action, reply_id, detail)
        elif status == "unavailable":
            self.log.warning("DM unavailable for comment %s; scheduling open-DMs reply", reply_id)
        if status in ("unknown", "rejected"):
            self.log.warning("%s for post %s ended as %s and will not be retried", action, reply_id, status)
        return status

    def _process_comment(self, comment: Mapping[str, str]) -> None:
        reply_id = comment["reply_id"]
        if comment["author_id"] in self.config.denied_user_ids:
            self.store.complete_comment(reply_id, self.now())
            return
        dm = self._run_action(comment, "dm")
        if dm == "stopping":
            return
        if dm in ("sent", "unavailable"):
            text = self.config.reply_text if dm == "sent" else self.config.dm_unavailable_reply
            # Both outcomes share one public-action claim. Retain the existing
            # key so an acknowledgement sent before this update is not repeated.
            if self._run_action(comment, "ack", public_text=text) == "stopping":
                return
        self.store.complete_comment(reply_id, self.now())

    def run_once(self) -> None:
        if not self.roots:
            self.validate_startup()
        current = self.now()
        cooldown = self.store.cooldown_until()
        if not self.config.dry_run and cooldown is not None and cooldown > current:
            raise RateLimitedCycle(cooldown)
        for comment in self.store.pending_comments(self.config.monitored_post_ids):
            if self.stopping:
                return
            self._process_comment(comment)
        end = current - 10.0
        for post_id in self.config.monitored_post_ids:
            if self.stopping:
                return
            start, fixed_end = self._window(post_id, end)
            if start >= fixed_end:
                continue
            for tweet in self.client.search_replies(post_id, _rfc3339(start), _rfc3339(fixed_end)):
                if self.stopping:
                    return
                self._handle_reply(post_id, tweet)
            if self.stopping:
                return
            self.store.checkpoint(post_id, fixed_end)

    def run_forever(self) -> None:
        while not self.stopping:
            delay = self.config.poll_interval_seconds
            try:
                self.run_once()
            except RateLimitedCycle as exc:
                delay = max(delay, exc.retry_at - self.now())
                self.log.warning("X rate limit is active; next attempt is delayed")
            except (FatalApiError, ConfigurationError):
                raise
            except Exception as exc:
                self.log.error("Polling cycle failed (%s); no incomplete search checkpoint was saved", type(exc).__name__)
            deadline = self.now() + delay
            while not self.stopping and self.now() < deadline:
                self.sleep(min(1.0, deadline - self.now()))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true", help="poll once and exit (dry run only)")
    parser.add_argument("--fixture", type=Path, help="use an offline JSON fixture (dry run only)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("leadpoet_x_bot")
    store: StateStore | None = None
    try:
        fixture = load_fixture(args.fixture) if args.fixture else None
        config = Config.from_env()
        if args.once and not config.dry_run:
            raise ConfigurationError("--once is allowed only when DRY_RUN=true")
        if fixture is not None and not config.dry_run:
            raise ConfigurationError("--fixture is allowed only when DRY_RUN=true")
        if fixture is None:
            credentials = Credentials.from_env(os.environ)
            client: Any = XApiClient(
                credentials,
                timeout=config.request_timeout_seconds,
            )
        else:
            client = FixtureClient(fixture)
        with ProcessLock(config.state_db):
            store = StateStore(config.state_db, config.namespace)
            bot = Bot(config, client, store)
            signal.signal(signal.SIGTERM, bot.stop)
            signal.signal(signal.SIGINT, bot.stop)
            if args.once or fixture is not None:
                bot.run_once()
            else:
                bot.run_forever()
        return 0
    except BotError as exc:
        log.error("%s", exc)
        return 2
    finally:
        if store is not None:
            store.close()


if __name__ == "__main__":
    raise SystemExit(main())
