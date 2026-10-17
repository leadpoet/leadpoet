"""Raw reviewer JSON transport; request and evidence meaning stay in grading."""

from __future__ import annotations

from typing import Any, Mapping

import aiohttp


async def post_json(
    session: aiohttp.ClientSession,
    url: str,
    *,
    headers: Mapping[str, str],
    payload: Mapping[str, Any],
) -> tuple[int, Any]:
    async with session.post(url, headers=dict(headers), json=dict(payload)) as response:
        status = response.status
        try:
            body = await response.json()
        except (aiohttp.ContentTypeError, ValueError):
            body = None
    return status, body
