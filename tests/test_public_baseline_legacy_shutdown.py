from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from gateway.research_lab import api


@pytest.mark.asyncio
async def test_public_baseline_retires_autoresearch_without_storage_reads(
) -> None:
    with pytest.raises(HTTPException) as raised:
        await api._require_autoresearch_not_paused(
            SimpleNamespace(public_baseline_rebenchmark_enabled=True)
        )

    assert raised.value.status_code == 410
    assert raised.value.detail["code"] == "research_lab_autoresearch_retired"
