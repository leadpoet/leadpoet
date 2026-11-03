"""A failed stage names the rule that rejected the catalog, not just the class."""

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from gateway.observability.otel_bootstrap import configure_arena_otel
from lab_arena import telemetry
from lab_arena.deepline_catalog import CatalogError


@pytest.fixture
def exporter():
    exporter = InMemorySpanExporter()
    telemetry.install_recorder(configure_arena_otel(None, span_exporter=exporter))
    try:
        yield exporter
    finally:
        telemetry.install_recorder(None)


def _error_types(exporter):
    return [
        span.attributes["arena.error_type"]
        for span in exporter.get_finished_spans()
        if span.name == "arena.ensure_daily_round"
    ]


def test_catalog_failure_carries_its_reason_through_the_boundary(exporter):
    with pytest.raises(CatalogError):
        with telemetry.stage("ensure_daily_round"):
            raise CatalogError("invalid_public_company_sql")
    assert _error_types(exporter) == ["deepline_catalog_invalid_public_company_sql"]


def test_an_ordinary_exception_still_records_its_class_name(exporter):
    with pytest.raises(TimeoutError):
        with telemetry.stage("ensure_daily_round"):
            raise TimeoutError("some message")
    assert _error_types(exporter) == ["TimeoutError"]


@pytest.mark.parametrize("label", ["", "has:colon", "has space", "9leading", "x" * 65, 17])
def test_a_label_the_boundary_would_reject_falls_back_to_the_class_name(exporter, label):
    class Weird(RuntimeError):
        pass

    weird = Weird()
    weird.arena_error_type = label
    with pytest.raises(Weird):
        with telemetry.stage("ensure_daily_round"):
            raise weird
    assert _error_types(exporter) == ["Weird"]


def test_every_catalog_reason_is_a_label_the_boundary_accepts():
    from lab_arena import deepline_catalog

    reasons = [
        "catalog_too_large", "invalid_catalog", "missing_schema", "invalid_schema",
        "unsupported_schema_reference", "contextdev_news_filter_schema_changed",
        "invalid_pricing_hint", "invalid_public_company_sql", "catalog_identity_changed",
    ]
    for reason in reasons:
        label = CatalogError(reason).arena_error_type
        assert telemetry._ERROR_TYPE_RE.fullmatch(label), label
    assert deepline_catalog.CatalogError("invalid_catalog").args == ("invalid_catalog",)
