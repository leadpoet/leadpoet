"""Small Langfuse wrapper that cannot affect Research Lab execution."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import logging
import os
import random
import sys
from typing import Any, Iterator

from .redaction import RedactionBlocked, redact_for_langfuse


logger = logging.getLogger(__name__)
TRUTHY = {"1", "true", "yes", "on"}

# Salt for the deterministic per-run sampling decision. Distinct from the
# trace-id derivation salt so the sampling bits are not a prefix of the id.
_SAMPLE_SEED_SALT = "research-lab-langfuse-sample:"
_TRACE_ID_SALT = "research-lab-langfuse-trace:"


def langfuse_enabled() -> bool:
    return os.getenv("LANGFUSE_ENABLED", "false").strip().lower() in TRUTHY


def langfuse_project() -> str:
    return os.getenv("LANGFUSE_PROJECT", "leadpoet-lab-prod-redacted").strip() or "leadpoet-lab-prod-redacted"


def _sample_rate() -> float:
    # Default = capture everything. Spans are pointers/hashes only (content
    # stays in SSE-KMS S3), so full capture is cheap; an unset env var must
    # not silently drop 90% of runs. Set LANGFUSE_SAMPLE_RATE < 1.0 only to
    # deliberately throttle.
    try:
        sample_rate = float(os.getenv("LANGFUSE_SAMPLE_RATE", "1.0"))
    except ValueError:
        sample_rate = 1.0
    return max(0.0, min(1.0, sample_rate))


def langfuse_sampled() -> bool:
    return random.random() <= _sample_rate()


def deterministic_sampled(seed: str) -> bool:
    """Per-run sampling: every span seeded with the same run_id gets the same
    keep/drop decision, so a sampled run is captured end-to-end instead of as
    a random 10% scatter of unrelated spans."""
    digest = hashlib.sha256((_SAMPLE_SEED_SALT + str(seed)).encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16**12) <= _sample_rate()


def run_trace_id(run_id: str) -> str:
    """Deterministic W3C-format Langfuse trace id for a Research Lab run.

    A pure function of ``run_id`` so the loop engine, the scoring worker, and
    any later backfill land observations on the same trace without passing a
    trace id through candidate records. Empty when ``run_id`` is empty.
    """
    if not str(run_id or "").strip():
        return ""
    return hashlib.sha256((_TRACE_ID_SALT + str(run_id)).encode("utf-8")).hexdigest()[:32]


def redaction_mode() -> str:
    return os.getenv("LANGFUSE_REDACTION_LEVEL", "prod").strip().lower() or "prod"


def get_langfuse_client(sample_seed: str | None = None, *, skip_sampling: bool = False) -> Any | None:
    if not langfuse_enabled():
        return None
    if not skip_sampling:
        sampled = deterministic_sampled(sample_seed) if sample_seed else langfuse_sampled()
        if not sampled:
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
    trace_id: str | None = None,
    sample_seed: str | None = None,
) -> Iterator[Any | None]:
    """Start a best-effort Langfuse observation.

    A Langfuse or redaction failure never propagates to caller code, and —
    just as important — an exception raised by the caller's body propagates
    unchanged (it is never swallowed or re-typed by this wrapper).

    ``trace_id`` attaches the observation to an existing trace (see
    ``run_trace_id``); ``sample_seed`` switches the sampling decision from
    random-per-call to deterministic-per-seed so one run's spans are kept or
    dropped together.
    """

    try:
        client = get_langfuse_client(sample_seed=sample_seed)
    except TypeError:
        # Monkeypatched/legacy zero-arg client factories.
        client = get_langfuse_client()
    if client is None:
        yield None
        return
    span: Any | None = None
    exit_cm: Any | None = None
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
            if trace_id:
                try:
                    exit_cm = client.start_as_current_observation(
                        **kwargs, trace_context={"trace_id": trace_id}
                    )
                    span = exit_cm.__enter__()
                except TypeError:
                    # Client without trace_context support: fresh trace.
                    exit_cm = None
                    span = None
            if exit_cm is None:
                exit_cm = client.start_as_current_observation(**kwargs)
                span = exit_cm.__enter__()
        else:
            factory_name = "generation" if as_type == "generation" else "trace" if as_type == "trace" else "span"
            factory = getattr(client, factory_name, None) or getattr(client, "span", None)
            if factory is not None:
                span = factory(
                    name=name,
                    metadata=kwargs.get("metadata"),
                    input=kwargs.get("input"),
                )
    except RedactionBlocked as exc:
        logger.warning(
            "langfuse_redaction_blocked_event=true trace_name=%s blocked_reason=%s",
            name,
            str(exc)[:200],
        )
        span = None
        exit_cm = None
    except Exception as exc:
        logger.warning("langfuse_observation_failed trace_name=%s error=%s", name, str(exc)[:200])
        span = None
        exit_cm = None
    if span is None and exit_cm is None:
        yield None
        return
    # Caller-body exceptions must pass through untouched: only the Langfuse
    # close/end is guarded here. (Catching the body's exception and yielding
    # again would make contextlib raise "generator didn't stop after throw()"
    # in place of the real exception — e.g. StaleParentDuringScoring.)
    try:
        yield span
    finally:
        try:
            if exit_cm is not None:
                exit_cm.__exit__(*sys.exc_info())
            elif span is not None and hasattr(span, "end"):
                span.end()
        except Exception as exc:
            logger.warning("langfuse_observation_end_failed trace_name=%s error=%s", name, str(exc)[:200])


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
    # Flush must bypass sampling: it drains whatever earlier (sampled-in)
    # observations are still buffered, so rolling the sampling dice again
    # here would silently drop the tail ~90% of the time.
    try:
        client = get_langfuse_client(skip_sampling=True)
    except TypeError:
        client = get_langfuse_client()
    if client is None:
        return
    try:
        client.flush()
    except Exception as exc:
        logger.warning("langfuse_flush_failed error=%s", str(exc)[:200])
