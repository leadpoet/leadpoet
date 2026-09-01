"""Research Lab chain helpers used by gateway workers and reports."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import subprocess
import sys
from typing import Any, Iterable

logger = logging.getLogger(__name__)
_DIRECT_EPOCH_TIMEOUT_SECONDS_ENV = "RESEARCH_LAB_DIRECT_EPOCH_TIMEOUT_SECONDS"
_DIRECT_EPOCH_ATTEMPTS_ENV = "RESEARCH_LAB_DIRECT_EPOCH_ATTEMPTS"
_DEFAULT_DIRECT_EPOCH_TIMEOUT_SECONDS = 60.0
_DEFAULT_DIRECT_EPOCH_ATTEMPTS = 3
_DIRECT_EPOCH_STOP_SECONDS = 5.0
_DIRECT_EPOCH_RESULT_PREFIX = "LEADPOET_EPOCH_RESULT="


def _bind_probe_lifetime_to_parent() -> None:
    """Kill a Linux epoch probe if its gateway parent exits mid-request."""
    if not sys.platform.startswith("linux"):
        return

    parent_pid = os.getppid()
    if parent_pid <= 1:
        os._exit(70)

    import ctypes
    import signal

    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:  # PR_SET_PDEATHSIG
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number))

    # PR_SET_PDEATHSIG is not retroactive. Close the race where the parent
    # exited after the first getppid() but before prctl() completed.
    if os.getppid() != parent_pid:
        os._exit(70)


async def resolve_research_lab_evaluation_epoch(configured_epoch: int | str | None = None) -> tuple[int, int | None, str]:
    """Resolve the live Bittensor epoch without requiring an operator override."""
    from gateway.utils.epoch import (
        get_current_epoch_context_async,
    )

    try:
        configured = int(configured_epoch or 0)
    except (TypeError, ValueError):
        configured = 0
    if configured > 0:
        raise RuntimeError(
            "configured Research Lab epoch overrides are forbidden"
        )

    try:
        timeout_seconds = _direct_epoch_timeout_seconds()
        snapshot, epoch = await asyncio.wait_for(
            get_current_epoch_context_async(finalized=True),
            timeout=timeout_seconds,
        )
        block = snapshot.current_block
        source = "gateway_epoch_utils:finalized"
    except Exception as exc:
        logger.warning(
            "research_lab_epoch_gateway_utils_failed_direct_probe: %s",
            str(exc)[:200],
        )
        try:
            epoch, block, network = await _fetch_current_chain_epoch_direct()
            source = f"direct_subtensor_official:{network}"
        except Exception as direct_exc:
            raise RuntimeError(
                "Research Lab evaluation epoch could not be resolved from "
                "an exact-hash official SN71 snapshot"
            ) from direct_exc

    if epoch <= 0:
        raise RuntimeError("Research Lab evaluation epoch resolved to 0")
    return epoch, block, source


def _direct_epoch_timeout_seconds() -> float:
    try:
        return max(1.0, float(os.getenv(_DIRECT_EPOCH_TIMEOUT_SECONDS_ENV, _DEFAULT_DIRECT_EPOCH_TIMEOUT_SECONDS)))
    except (TypeError, ValueError):
        return float(_DEFAULT_DIRECT_EPOCH_TIMEOUT_SECONDS)


def _direct_epoch_attempts() -> int:
    try:
        return max(
            1,
            int(os.getenv(_DIRECT_EPOCH_ATTEMPTS_ENV, _DEFAULT_DIRECT_EPOCH_ATTEMPTS)),
        )
    except (TypeError, ValueError):
        return int(_DEFAULT_DIRECT_EPOCH_ATTEMPTS)


async def resolve_hotkey_uids(hotkeys: Iterable[str]) -> dict[str, int]:
    """Resolve registered hotkeys to current subnet UIDs using one metagraph read."""
    unique_hotkeys = {str(hotkey) for hotkey in hotkeys if str(hotkey or "").strip()}
    if not unique_hotkeys:
        return {}
    metagraph = await _get_metagraph()
    resolved: dict[str, int] = {}
    for uid, hotkey in enumerate(getattr(metagraph, "hotkeys", []) or []):
        if hotkey in unique_hotkeys:
            resolved[str(hotkey)] = int(uid)
    return resolved


async def _get_metagraph() -> Any:
    try:
        from gateway.utils.registry import get_metagraph_async

        return await get_metagraph_async()
    except Exception as exc:
        logger.warning("research_lab_metagraph_gateway_registry_failed_fallback_direct: %s", str(exc)[:200])
        return await asyncio.to_thread(_fetch_metagraph_direct)


async def _stop_direct_epoch_probe(
    process: Any,
    communication_task: asyncio.Task[tuple[bytes, bytes]],
) -> None:
    """Reap one isolated epoch probe and all of its descendants."""

    def signal_owned_group(signum: int) -> None:
        try:
            if os.name == "posix":
                os.killpg(process.pid, signum)
            elif process.returncode is None and signum == signal.SIGTERM:
                process.terminate()
            elif process.returncode is None:
                process.kill()
        except ProcessLookupError:
            pass

    def owned_group_exists() -> bool:
        if os.name != "posix":
            return process.returncode is None
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    signal_owned_group(signal.SIGTERM)
    communication_timed_out = False
    try:
        await asyncio.wait_for(
            asyncio.shield(communication_task),
            timeout=_DIRECT_EPOCH_STOP_SECONDS,
        )
    except asyncio.TimeoutError:
        communication_timed_out = True
    except BaseException:
        # The caller retains the original communication/cancellation failure;
        # reaching this branch proves the pipe tasks themselves are complete.
        pass
    if communication_timed_out or owned_group_exists():
        signal_owned_group(signal.SIGKILL)
    if not communication_task.done():
        try:
            await asyncio.wait_for(
                asyncio.shield(communication_task),
                timeout=_DIRECT_EPOCH_STOP_SECONDS,
            )
        except asyncio.TimeoutError:
            communication_task.cancel()
            await asyncio.gather(communication_task, return_exceptions=True)
            raise RuntimeError("direct epoch probe output did not close")
        except BaseException:
            pass
    else:
        try:
            communication_task.result()
        except BaseException:
            pass
    await process.wait()
    if os.name == "posix":
        deadline = asyncio.get_running_loop().time() + _DIRECT_EPOCH_STOP_SECONDS
        while owned_group_exists() and asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(0.05)
        if owned_group_exists():
            raise RuntimeError("direct epoch probe process group survived cleanup")


async def _stop_direct_epoch_probe_atomic(
    process: Any,
    communication_task: asyncio.Task[tuple[bytes, bytes]],
) -> bool:
    """Finish cleanup despite repeated cancellation of the owning task."""

    cleanup_task = asyncio.create_task(
        _stop_direct_epoch_probe(process, communication_task)
    )
    cancelled_during_cleanup = False
    while not cleanup_task.done():
        try:
            await asyncio.shield(cleanup_task)
        except asyncio.CancelledError:
            cancelled_during_cleanup = True
    cleanup_task.result()
    return cancelled_during_cleanup


async def _fetch_current_chain_epoch_direct() -> tuple[int, int, str]:
    network = os.getenv("BITTENSOR_NETWORK", "finney")
    proxy_keys = ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy")
    child_env = {key: value for key, value in os.environ.items() if key not in proxy_keys}
    probe = """
import json
import os
import sys
import bittensor as bt
from gateway.research_lab.chain import _bind_probe_lifetime_to_parent
from Leadpoet.utils.subnet_epoch import (
    load_subnet_epoch_cutover,
    read_subnet_epoch_snapshot,
)
from gateway.utils.epoch import validate_stateful_cutover_authority
from gateway.utils.subnet_epoch_archive import (
    validate_cutover_anchor_from_archive,
)

_bind_probe_lifetime_to_parent()
network = os.getenv("BITTENSOR_NETWORK", "finney")
netuid = int(os.getenv("BITTENSOR_NETUID", "71"))
subtensor = bt.Subtensor(network=network)
snapshot = read_subnet_epoch_snapshot(
    subtensor,
    netuid=netuid,
    finalized=True,
)
cutover = load_subnet_epoch_cutover()
epoch = snapshot.settlement_epoch_id(cutover)
block = snapshot.current_block
validate_stateful_cutover_authority(cutover)
validate_cutover_anchor_from_archive(cutover)
official = {
    "official_subnet_epoch_id": snapshot.subnet_epoch_index,
    "epoch_ref": snapshot.epoch_ref,
}
result = {
    "epoch": epoch,
    "block": block,
    "network": network,
}
result.update(official)
sys.stdout.write(%r + json.dumps(result, separators=(",", ":")) + "\\n")
sys.stdout.flush()
# This process exists only to make the SDK call killable. Some SDK versions
# hang while closing their synchronous WebSocket, so let process teardown close
# its private descriptors after the verified result has been flushed.
os._exit(0)
""" % _DIRECT_EPOCH_RESULT_PREFIX
    timeout_seconds = max(1.0, _direct_epoch_timeout_seconds() - 1.0)
    completed = None
    failures: list[str] = []
    attempts = _direct_epoch_attempts()
    for attempt in range(1, attempts + 1):
        command = [sys.executable, "-c", probe]
        candidate = None
        communication_task = None
        try:
            candidate = await asyncio.create_subprocess_exec(
                *command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=child_env,
                start_new_session=(os.name == "posix"),
            )
            communication_task = asyncio.create_task(candidate.communicate())
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                asyncio.shield(communication_task),
                timeout=timeout_seconds,
            )
        except asyncio.TimeoutError:
            if candidate is None or communication_task is None:
                raise RuntimeError("direct epoch probe ownership is unavailable")
            cancelled = await _stop_direct_epoch_probe_atomic(
                candidate,
                communication_task,
            )
            if cancelled:
                raise asyncio.CancelledError()
            failures.append(f"attempt {attempt} timed out after {timeout_seconds:.1f}s")
        except BaseException as exc:
            if candidate is None:
                raise
            if communication_task is None:
                communication_task = asyncio.create_task(candidate.communicate())
            cancelled = isinstance(exc, asyncio.CancelledError)
            cancelled = (
                await _stop_direct_epoch_probe_atomic(
                    candidate,
                    communication_task,
                )
                or cancelled
            )
            if cancelled:
                raise asyncio.CancelledError()
            raise
        else:
            try:
                stdout = stdout_bytes.decode("utf-8")
                stderr = stderr_bytes.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise RuntimeError(
                    "direct subtensor epoch probe returned invalid output"
                ) from exc
            if candidate.returncode == 0:
                completed = subprocess.CompletedProcess(
                    command,
                    candidate.returncode,
                    stdout=stdout,
                    stderr=stderr,
                )
                break
            detail = (stderr or stdout or "").strip().splitlines()
            failures.append(
                "attempt %d failed: %s"
                % (
                    attempt,
                    detail[-1][:200] if detail else f"exit {candidate.returncode}",
                )
            )
        if attempt < attempts:
            logger.warning(
                "research_lab_direct_epoch_probe_retry attempt=%d/%d error=%s",
                attempt,
                attempts,
                failures[-1],
            )
    if completed is None:
        raise RuntimeError(
            "direct subtensor epoch probe exhausted exact-hash attempts: "
            + "; ".join(failures)
        )
    result_line = next(
        (
            line[len(_DIRECT_EPOCH_RESULT_PREFIX) :]
            for line in reversed(completed.stdout.splitlines())
            if line.startswith(_DIRECT_EPOCH_RESULT_PREFIX)
        ),
        "",
    )
    try:
        result = json.loads(result_line)
        epoch = int(result["epoch"])
        block = int(result["block"])
        result_network = str(result["network"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError("direct subtensor epoch probe returned invalid output") from exc
    if block <= 0 or result_network != network:
        raise RuntimeError("direct subtensor epoch probe returned inconsistent output")
    if epoch <= 0:
        raise RuntimeError("direct subtensor epoch probe returned an invalid epoch")
    return epoch, block, network


def _fetch_metagraph_direct() -> Any:
    import bittensor as bt

    network = os.getenv("BITTENSOR_NETWORK", "finney")
    netuid = int(os.getenv("BITTENSOR_NETUID", "71"))
    proxy_keys = ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy")
    saved_proxy_env = {key: os.environ.pop(key) for key in proxy_keys if key in os.environ}
    try:
        subtensor = bt.Subtensor(network=network)
        try:
            return subtensor.metagraph(netuid=netuid)
        finally:
            close = getattr(subtensor, "close", None)
            if callable(close):
                close()
    finally:
        os.environ.update(saved_proxy_env)
