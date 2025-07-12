"""
Datadog APM Integration for Leadpoet Intent Model v1.1
Provides distributed tracing and APM monitoring for FastAPI application.
"""

import os
from typing import Optional, Dict, Any
from contextlib import contextmanager
import time

from ddtrace import tracer
from ddtrace.contrib.fastapi import FastAPIMiddleware
from ddtrace import patch_all

from app.core.config import settings
from loguru import logger

# Configure Datadog tracer
def configure_datadog_apm():
    """Configure Datadog APM tracing."""
    try:
        # Set service name and environment
        tracer.configure(
            service="leadpoet-intent-model",
            env=settings.ENVIRONMENT,
            version="1.1.0"
        )
        
        # Enable auto-instrumentation for common libraries
        patch_all()
        
        # Set sampling rate based on environment
        if settings.ENVIRONMENT == "production":
            tracer.sampler.sample_rate = 0.1  # 10% sampling in production
        else:
            tracer.sampler.sample_rate = 1.0  # 100% sampling in development
        
        logger.info(f"Datadog APM configured for service: leadpoet-intent-model, env: {settings.ENVIRONMENT}")
        
    except Exception as e:
        logger.error(f"Failed to configure Datadog APM: {e}")


class DatadogAPMMiddleware:
    """Custom Datadog APM middleware for FastAPI."""
    
    def __init__(self, app):
        self.app = app
        self.fastapi_middleware = FastAPIMiddleware(app)
    
    async def __call__(self, scope, receive, send):
        return await self.fastapi_middleware(scope, receive, send)


@contextmanager
def trace_pipeline_stage(stage: str, sub_stage: Optional[str] = None, **tags):
    """
    Context manager for tracing pipeline stages.
    
    Args:
        stage: Main pipeline stage (parsing, retrieval, scoring, filtering)
        sub_stage: Sub-stage or component name
        **tags: Additional tags for the span
    """
    span_name = f"pipeline.{stage}"
    if sub_stage:
        span_name += f".{sub_stage}"
    
    with tracer.trace(span_name, service="leadpoet-intent-model") as span:
        # Set standard tags
        span.set_tag("pipeline.stage", stage)
        if sub_stage:
            span.set_tag("pipeline.sub_stage", sub_stage)
        
        # Set additional tags
        for key, value in tags.items():
            span.set_tag(key, value)
        
        try:
            yield span
        except Exception as e:
            span.error = True
            span.set_tag("error.message", str(e))
            raise


@contextmanager
def trace_database_operation(operation: str, table: str, query: Optional[str] = None):
    """
    Context manager for tracing database operations.
    
    Args:
        operation: Database operation (select, insert, update, delete)
        table: Table name
        query: SQL query (optional, for debugging)
    """
    span_name = f"db.{operation}"
    
    with tracer.trace(span_name, service="leadpoet-intent-model") as span:
        span.set_tag("db.operation", operation)
        span.set_tag("db.table", table)
        if query:
            span.set_tag("db.query", query)
        
        try:
            yield span
        except Exception as e:
            span.error = True
            span.set_tag("error.message", str(e))
            raise


@contextmanager
def trace_cache_operation(operation: str, cache_type: str, key: Optional[str] = None):
    """
    Context manager for tracing cache operations.
    
    Args:
        operation: Cache operation (get, set, delete)
        cache_type: Cache type (features, llm_responses, candidates)
        key: Cache key (optional, for debugging)
    """
    span_name = f"cache.{operation}"
    
    with tracer.trace(span_name, service="leadpoet-intent-model") as span:
        span.set_tag("cache.operation", operation)
        span.set_tag("cache.type", cache_type)
        if key:
            span.set_tag("cache.key", key)
        
        try:
            yield span
        except Exception as e:
            span.error = True
            span.set_tag("error.message", str(e))
            raise


@contextmanager
def trace_llm_call(model: str, operation: str, input_tokens: Optional[int] = None):
    """
    Context manager for tracing LLM API calls.
    
    Args:
        model: LLM model name (gpt-4o, gpt-3.5-turbo)
        operation: Operation type (completion, embedding)
        input_tokens: Number of input tokens (optional)
    """
    span_name = f"llm.{operation}"
    
    with tracer.trace(span_name, service="leadpoet-intent-model") as span:
        span.set_tag("llm.model", model)
        span.set_tag("llm.operation", operation)
        if input_tokens:
            span.set_tag("llm.input_tokens", input_tokens)
        
        try:
            yield span
        except Exception as e:
            span.error = True
            span.set_tag("error.message", str(e))
            raise


def trace_function(name: str, **tags):
    """
    Decorator for tracing functions.
    
    Args:
        name: Span name
        **tags: Tags to add to the span
    """
    def decorator(func):
        import asyncio
        import functools

        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                with tracer.trace(name, service="leadpoet-intent-model") as span:
                    for key, value in tags.items():
                        span.set_tag(key, value)
                    try:
                        result = await func(*args, **kwargs)
                        return result
                    except Exception as e:
                        span.error = True
                        span.set_tag("error.message", str(e))
                        raise
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                with tracer.trace(name, service="leadpoet-intent-model") as span:
                    for key, value in tags.items():
                        span.set_tag(key, value)
                    try:
                        result = func(*args, **kwargs)
                        return result
                    except Exception as e:
                        span.error = True
                        span.set_tag("error.message", str(e))
                        raise
            return sync_wrapper

    return decorator


def add_custom_span_tags(span, **tags):
    """
    Add custom tags to the current span.
    
    Args:
        **tags: Tags to add
    """
    try:
        for key, value in tags.items():
            span.set_tag(key, value)
    except Exception as e:
        logger.warning(f"Failed to add span tags: {e}")


def get_current_span():
    """Get the current active span."""
    return tracer.current_span()


def set_span_metric(metric_name: str, value: float):
    """
    Set a metric on the current span.
    
    Args:
        metric_name: Name of the metric
        value: Metric value
    """
    try:
        span = get_current_span()
        if span:
            span.set_metric(metric_name, value)
    except Exception as e:
        logger.warning(f"Failed to set span metric: {e}")


# Initialize Datadog APM when needed
_initialized = False

def ensure_datadog_apm_initialized():
    """Ensure Datadog APM is initialized (singleton pattern)."""
    global _initialized
    if not _initialized:
        configure_datadog_apm()
        _initialized = True


# Initialize Datadog APM on module import
configure_datadog_apm() 