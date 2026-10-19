"""Bounded recovery of transient chain connections before gateway readiness."""

from __future__ import annotations

import asyncio
import socket


# Leave two minutes of the controller's ten-minute health deadline for the
# identity, crypto and background-service startup checks.
STARTUP_TIMEOUT_SECONDS = 480
ATTEMPT_TIMEOUT_SECONDS = 30
CLEANUP_TIMEOUT_SECONDS = 5
MAX_BACKOFF_SECONDS = 30
_NETWORK_ERRORS = (ConnectionError, asyncio.TimeoutError, socket.gaierror)


async def connect_async_subtensor(
    client_factory, *, network, runtime_sha, record_retry, capture_failure
):
    """Return one connected client; close every unsuccessful client first.

    SDK initialization can raise ConnectionError well before our per-attempt
    timeout. Recover for a fixed elapsed-time budget, rather than a fixed
    attempt count. Config/programming errors and unsuccessful cleanup fail
    closed immediately. No app state or background work is published here.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + STARTUP_TIMEOUT_SECONDS
    attempt = 0
    backoff = 5
    last_error = None
    fields = dict(
        component="gateway",
        stage="chain_runtime_startup",
        dependency="finney",
        runtime_sha=runtime_sha,
    )

    def terminal(error, *, retryable, timed_out=False, cleanup_result=None):
        capture_failure(
            "authority.dependency_unreadable",
            **fields,
            exception=error,
            terminal=True,
            retryable=retryable,
            fail_closed=True,
            attempts=attempt,
            timed_out=timed_out,
            cleanup_result=cleanup_result,
        )

    while True:
        remaining = deadline - loop.time()
        # Reserve a bounded close even if initialization consumes its timeout.
        if remaining <= CLEANUP_TIMEOUT_SECONDS:
            if remaining > 0:
                await asyncio.sleep(remaining)
            terminal(last_error, retryable=True, timed_out=True)
            raise RuntimeError(
                f"AsyncSubtensor connection failed within {STARTUP_TIMEOUT_SECONDS}s "
                f"after {attempt} attempts"
            ) from last_error

        attempt += 1
        client = None
        try:
            print(f"Chain startup attempt {attempt}: connecting to {network}", flush=True)
            client = client_factory(network=network)
            return await asyncio.wait_for(
                client.__aenter__(),
                timeout=min(ATTEMPT_TIMEOUT_SECONDS, remaining - CLEANUP_TIMEOUT_SECONDS),
            )
        except BaseException as error:
            # Keep the constructed client even when __aenter__ fails or is
            # cancelled: the SDK can already own a socket at that point.
            if client is not None:
                try:
                    cleanup_task = asyncio.create_task(
                        asyncio.wait_for(
                            client.close(),
                            timeout=min(CLEANUP_TIMEOUT_SECONDS, max(0, deadline - loop.time())),
                        )
                    )
                    try:
                        await asyncio.shield(cleanup_task)
                    except asyncio.CancelledError as cancelled:
                        # A shutdown arriving during cleanup must not abandon
                        # the socket. The same close timeout still applies.
                        try:
                            await cleanup_task
                        except Exception as cleanup_error:
                            terminal(cleanup_error, retryable=False, cleanup_result="failed")
                        raise cancelled
                except Exception as cleanup_error:
                    terminal(
                        cleanup_error,
                        retryable=False,
                        timed_out=isinstance(cleanup_error, asyncio.TimeoutError),
                        cleanup_result="failed",
                    )
                    if isinstance(error, asyncio.CancelledError):
                        raise error
                    raise RuntimeError("AsyncSubtensor failed-client cleanup failed") from cleanup_error
            if isinstance(error, asyncio.CancelledError):
                raise
            if not isinstance(error, Exception):
                raise
            if not isinstance(error, _NETWORK_ERRORS):
                terminal(error, retryable=False)
                raise
            last_error = error

        remaining = deadline - loop.time()
        if remaining <= 0:
            terminal(last_error, retryable=True, timed_out=True)
            raise RuntimeError(
                f"AsyncSubtensor connection failed within {STARTUP_TIMEOUT_SECONDS}s "
                f"after {attempt} attempts"
            ) from last_error
        delay = min(backoff, remaining)
        record_retry(
            "authority.dependency_unreadable",
            **fields,
            attempt=attempt,
            exception_class=type(last_error).__name__,
            timed_out=isinstance(last_error, asyncio.TimeoutError),
        )
        print(
            f"Chain startup attempt {attempt} failed ({type(last_error).__name__}); "
            f"retrying in {delay:g}s with {remaining:g}s remaining",
            flush=True,
        )
        await asyncio.sleep(delay)
        backoff = min(backoff * 2, MAX_BACKOFF_SECONDS)
