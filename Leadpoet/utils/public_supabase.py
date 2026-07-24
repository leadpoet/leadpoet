"""Dependency-free access to Leadpoet's fixed public Supabase authority."""

from __future__ import annotations

import json
from typing import Any, Mapping
from urllib.request import Request, urlopen


SUPABASE_URL = "https://qplwoislplkcegvdmbim.supabase.co"
SUPABASE_ANON_KEY = "sb_publishable_YU7GBMSX-fwEsSH7MnhSBQ_l5ACuFVf"


def call_public_rpc(
    function_name: str,
    params: Mapping[str, Any] | None = None,
    *,
    timeout_seconds: float = 30.0,
) -> Any:
    """Call one read-only PostgREST RPC with the repository's public key."""

    request = Request(
        f"{SUPABASE_URL}/rest/v1/rpc/{function_name}",
        data=json.dumps(dict(params or {}), separators=(",", ":")).encode("utf-8"),
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {SUPABASE_ANON_KEY}",
            "Content-Type": "application/json",
            "apikey": SUPABASE_ANON_KEY,
        },
        method="POST",
    )
    with urlopen(request, timeout=float(timeout_seconds)) as response:
        body = response.read()
    return json.loads(body.decode("utf-8"))
