from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor

from gateway.utils import role_translate


def test_role_translation_lru_is_shared_across_event_loop_threads():
    role_translate._lru.clear()

    def write_then_read(index: int):
        key = f"role-{index}"
        value = f"Role {index}"

        async def exercise():
            await role_translate._lru_put(key, value)
            return await role_translate._lru_get(key)

        return asyncio.run(exercise())

    with ThreadPoolExecutor(max_workers=8) as executor:
        observed = list(executor.map(write_then_read, range(64)))

    assert observed == [f"Role {index}" for index in range(64)]
    assert len(role_translate._lru) == 64
