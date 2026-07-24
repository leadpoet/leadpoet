"""Guard: every gateway Supabase sync client must be HTTP/1-pinned.

postgrest-py enables HTTP/2 by default, and httpx's shared HPACK encoder is
not thread-safe: two threadpool workers encoding headers on the same client
concurrently raise ``RuntimeError: deque mutated during iteration``. That is
exactly how epoch 24123's weight publication died with a 503
("Authoritative V2 persistence/publication failed closed").

gateway/db/client.py owns the fix (`_create_sync_client` builds the client on
an ``httpx.Client(http1=True, http2=False)``). Every other gateway module that
keeps a module-level or cached Supabase client must build it through that
factory. This test fails the moment someone reintroduces a bare
``create_client(...)`` in the gateway tree, which would silently reopen the
HPACK race.
"""

from __future__ import annotations

import re
from pathlib import Path

GATEWAY_ROOT = Path(__file__).resolve().parent.parent / "gateway"

# The only module allowed to call supabase.create_client directly: it is where
# the HTTP/1-pinned factory lives.
ALLOWED = {GATEWAY_ROOT / "db" / "client.py"}

# Matches sync-client construction. create_async_client is a different symbol
# and does not match; _create_sync_client does not match either.
BARE_CREATE = re.compile(r"(?<![\w.])create_client\s*\(")


def test_gateway_has_no_unpinned_supabase_clients():
    offenders: list[str] = []
    for path in sorted(GATEWAY_ROOT.rglob("*.py")):
        if path in ALLOWED:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for lineno, line in enumerate(text.splitlines(), start=1):
            if BARE_CREATE.search(line):
                offenders.append(f"{path.relative_to(GATEWAY_ROOT.parent)}:{lineno}: {line.strip()}")
    assert not offenders, (
        "Bare supabase.create_client() found in gateway/ — these clients "
        "default to HTTP/2, whose shared HPACK encoder is not thread-safe "
        "(RuntimeError: deque mutated during iteration under concurrent "
        "threadpool use). Build them via gateway.db.client._create_sync_client "
        "instead:\n" + "\n".join(offenders)
    )


def test_allowed_factory_still_exists_and_pins_http1():
    # If db/client.py is refactored, the allowlist and the pinning must move
    # together — this assertion keeps the guard honest.
    text = (GATEWAY_ROOT / "db" / "client.py").read_text(encoding="utf-8")
    assert "_create_sync_client" in text
    assert "http1=True" in text and "http2=False" in text
