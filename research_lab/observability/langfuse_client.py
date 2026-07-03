"""Small Langfuse wrapper that cannot affect Research Lab execution."""

from __future__ import annotations

from contextlib import contextmanager
import logging
import os
import random
from typing import Any, Iterator

from .redaction import RedactionBlocked, redact_for_langfuse


logger = logging.getLogger(__name__)
TRUTHY = {"1", "true", "yes", "on"}


def langfuse_enabled() -> bool:
    return os.getenv("LANGFUSE_ENABLED", "false").strip().lower() in TRUTHY


def langfuse_project() -> str:
    return os.getenv("LANGFUSE_PROJECT", "leadpoet-lab-prod-redacted").strip() or "leadpoet-lab-prod-redacted"


def langfuse_sampled() -> bool:
    try:
        sample_rate = float(os.getenv("LANGFUSE_SAMPLE_RATE", "0.10"))
    except ValueError:
        sample_rate = 0.10
    return random.random() <= max(0.0, min(1.0, sample_rate))


def redaction_mode() -> str:
    return os.getenv("LANGFUSE_REDACTION_LEVEL", "prod").strip().lower() or "prod"


def get_langfuse_client() -> Any | None:
    if not langfuse_enabled() or not langfuse_sampled():
        return None
    try:
        from langfuse import get_client  # type: ignore
    except Exception:
        get_client = None
    if get_client is not None:
        try:
            return get_client()
        except Exception as exc:
            logger.warning("langfuse_client_unavailable error=%s", str(exc)[:200])
            return None
    try:
        from langfuse import Langfuse  # type: ignore

        return Langfuse()
    except Exception as exc:
        logger.warning("langfuse_client_unavailable error=%s", str(exc)[:200])
        return None


@contextmanager
def observation(
    name: str,
    *,
    as_type: str = "span",
    metadata: dict[str, Any] | None = None,
    input: Any | None = None,
) -> Iterator[Any | None]:
    """Start a best-effort Langfuse observation.

    A Langfuse or redaction failure never propagates to caller code.
    """

    client = get_langfuse_client()
    if client is None:
        yield None
        return
    try:
        safe_metadata = redact_for_langfuse(
            {
                **(metadata or {}),
                "langfuse_project": langfuse_project(),
            },
            mode=redaction_mode(),
        )
        kwargs: dict[str, Any] = {
            "as_type": as_type,
            "name": name,
            "metadata": safe_metadata,
        }
        if input is not None:
            kwargs["input"] = redact_for_langfuse(input, mode=redaction_mode())
        if hasattr(client, "start_as_current_observation"):
            with client.start_as_current_observation(**kwargs) as span:
                yield span
            return

        factory_name = "generation" if as_type == "generation" else "trace" if as_type == "trace" else "span"
        factory = getattr(client, factory_name, None) or getattr(client, "span", None)
        if factory is None:
            yield None
            return
        span = factory(
            name=name,
            metadata=kwargs.get("metadata"),
            input=kwargs.get("input"),
        )
        try:
            yield span
        finally:
            try:
                if hasattr(span, "end"):
                    span.end()
            except Exception as exc:
                logger.warning("langfuse_observation_end_failed trace_name=%s error=%s", name, str(exc)[:200])
    except RedactionBlocked as exc:
        logger.warning(
            "langfuse_redaction_blocked_event=true trace_name=%s blocked_reason=%s",
            name,
            str(exc)[:200],
        )
        yield None
    except Exception as exc:
        logger.warning("langfuse_observation_failed trace_name=%s error=%s", name, str(exc)[:200])
        yield None


def update_observation(obs: Any | None, *, output: Any | None = None, metadata: dict[str, Any] | None = None) -> None:
    if obs is None:
        return
    try:
        payload: dict[str, Any] = {}
        if output is not None:
            payload["output"] = redact_for_langfuse(output, mode=redaction_mode())
        if metadata is not None:
            payload["metadata"] = redact_for_langfuse(metadata, mode=redaction_mode())
        if payload:
            obs.update(**payload)
    except RedactionBlocked as exc:
        logger.warning("langfuse_redaction_blocked_update=true blocked_reason=%s", str(exc)[:200])
    except Exception as exc:
        logger.warning("langfuse_observation_update_failed error=%s", str(exc)[:200])


def flush_langfuse() -> None:
    client = get_langfuse_client()
    if client is None:
        return
    try:
        client.flush()
    except Exception as exc:
        logger.warning("langfuse_flush_failed error=%s", str(exc)[:200])
