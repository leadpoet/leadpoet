"""Guard: shared gateway Supabase clients must be HTTP/1-pinned.

postgrest-py enables HTTP/2 by default, and httpx's shared HPACK encoder is
not thread-safe: two threadpool workers encoding headers on the same client
concurrently raise ``RuntimeError: deque mutated during iteration``. That is
exactly how epoch 24123's weight publication died with a 503
("Authoritative V2 persistence/publication failed closed").

The bug class requires a SHARED client (module-level or cached singleton).
Policy enforced here:

- ``gateway/db/client.py`` owns the pinned factory (``_create_sync_client``
  on ``httpx.Client(http1=True, http2=False)``) and is the only module that
  may call ``supabase.create_client`` freely.
- A small set of PER-CALL sites construct a fresh client inside a function,
  never share it across threads, and are exercised by tests that stub
  ``supabase.create_client`` (e.g. the cutover-fence public-authority
  contract). Those files may keep bare ``create_client`` — but only at
  indented (function-local) positions. Hoisting one to module level turns it
  into a shared client and fails this guard.
- Every other gateway module must build clients through the pinned factory.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
GATEWAY_ROOT = REPO / "gateway"

# The pinned factory lives here; unrestricted use allowed.
FACTORY = GATEWAY_ROOT / "db" / "client.py"

# Files allowed to keep bare create_client at FUNCTION-LOCAL positions only.
PER_CALL_ALLOWED = {
    GATEWAY_ROOT / "utils" / "epoch.py",
    GATEWAY_ROOT / "tasks" / "epoch_monitor.py",
    GATEWAY_ROOT / "qualification" / "api" / "submit.py",
    GATEWAY_ROOT / "qualification" / "api" / "payment.py",
}

# Matches sync-client construction. create_async_client is a different symbol
# and does not match; _create_sync_client does not match either.
BARE_CREATE = re.compile(r"(?<![\w.])create_client\s*\(")


def _offending_lines(path: Path, *, indented_ok: bool) -> list[str]:
    offenders = []
    text = path.read_text(encoding="utf-8", errors="replace")
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not BARE_CREATE.search(line):
            continue
        if indented_ok and line[:1] in (" ", "\t"):
            continue
        offenders.append(
            f"{path.relative_to(REPO)}:{lineno}: {line.strip()}"
        )
    return offenders


def test_gateway_has_no_unpinned_shared_supabase_clients():
    offenders: list[str] = []
    for path in sorted(GATEWAY_ROOT.rglob("*.py")):
        if path == FACTORY:
            continue
        indented_ok = path in PER_CALL_ALLOWED
        offenders.extend(_offending_lines(path, indented_ok=indented_ok))
    assert not offenders, (
        "Bare supabase.create_client() in a position that creates a SHARED "
        "client — these default to HTTP/2, whose HPACK encoder is not "
        "thread-safe (RuntimeError: deque mutated during iteration under "
        "concurrent threadpool use). Build shared clients via "
        "gateway.db.client._create_sync_client instead. Per-call, "
        "function-local constructions are allowed only in PER_CALL_ALLOWED "
        "files:\n" + "\n".join(offenders)
    )


def test_per_call_allowlist_files_still_exist():
    # If an allowlisted file is moved or deleted, the allowlist must move
    # with it rather than silently allowlisting nothing.
    for path in PER_CALL_ALLOWED:
        assert path.exists(), f"PER_CALL_ALLOWED file missing: {path}"


def test_allowed_factory_still_exists_and_pins_http1():
    # If db/client.py is refactored, the allowlist and the pinning must move
    # together — this assertion keeps the guard honest.
    text = FACTORY.read_text(encoding="utf-8")
    assert "_create_sync_client" in text
    assert "http1=True" in text and "http2=False" in text
