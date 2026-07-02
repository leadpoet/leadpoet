"""Tests for gateway/research_lab/capture_health.py (trajectoryimprovements.md P5).

The invariant chain per capture channel is enabled ⇒ S3 prefix set ⇒ KMS key
set; production workers must refuse to start on a degraded config unless the
explicit ALLOW_DEGRADED override is set.
"""

from __future__ import annotations

import logging

import pytest

from gateway.research_lab import capture_health as ch
from gateway.research_lab.config import ResearchLabGatewayConfig


CAPTURE_ENVS = (
    ch.RAW_TRACE_CAPTURE_ENABLED_ENV,
    ch.RAW_TRACE_S3_PREFIX_ENV,
    ch.SCORER_TRACE_CAPTURE_ENV,
    ch.SCORER_TRACE_S3_PREFIX_ENV,
    ch.INCONTAINER_TRACE_CAPTURE_ENV,
    ch.INCONTAINER_TRACE_S3_PREFIX_ENV,
    ch.INCONTAINER_TRACE_KMS_KEY_ENV,
    ch.PROJECTOR_ENABLED_ENV,
    ch.ALLOW_DEGRADED_ENV,
    ch.ENFORCE_ENV,
)


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    for name in CAPTURE_ENVS:
        monkeypatch.delenv(name, raising=False)


def _config(**overrides) -> ResearchLabGatewayConfig:
    defaults = dict(
        production_writes_enabled=False,
        private_model_manifest_uri="",
        score_bundle_kms_key_id="alias/test-key",
    )
    defaults.update(overrides)
    return ResearchLabGatewayConfig(**defaults)


def _fully_configured(monkeypatch) -> None:
    monkeypatch.setenv(ch.RAW_TRACE_S3_PREFIX_ENV, "s3://bucket/raw")
    monkeypatch.setenv(ch.SCORER_TRACE_S3_PREFIX_ENV, "s3://bucket/scorer")
    monkeypatch.setenv(ch.INCONTAINER_TRACE_S3_PREFIX_ENV, "s3://bucket/incontainer")
    monkeypatch.setenv(ch.INCONTAINER_TRACE_KMS_KEY_ENV, "kms-key-1")
    monkeypatch.setenv(ch.PROJECTOR_ENABLED_ENV, "true")


def test_env_names_match_recorder_modules():
    """The health module duplicates env names to avoid import cycles — pin
    them against the authoritative definitions."""
    from gateway.research_lab import worker as w
    from gateway.research_lab import scoring_worker as sw
    from gateway.research_lab import trajectory_projector as tp
    from research_lab.eval import evaluator as ev

    assert ch.RAW_TRACE_CAPTURE_ENABLED_ENV == w._RAW_TRACE_CAPTURE_ENABLED_ENV
    assert ch.RAW_TRACE_S3_PREFIX_ENV == w._RAW_TRACE_S3_PREFIX_ENV
    assert ch.SCORER_TRACE_CAPTURE_ENV == sw._SCORER_TRACE_CAPTURE_ENV
    assert ch.SCORER_TRACE_S3_PREFIX_ENV == sw._SCORER_TRACE_S3_PREFIX_ENV
    assert ch.INCONTAINER_TRACE_S3_PREFIX_ENV == ev.INCONTAINER_TRACE_S3_PREFIX_ENV
    assert ch.INCONTAINER_TRACE_KMS_KEY_ENV == ev.INCONTAINER_TRACE_KMS_KEY_ENV
    assert ch.PROJECTOR_ENABLED_ENV == tp.PROJECTOR_ENABLED_ENV


def test_fully_configured_reports_ok_and_no_violations(monkeypatch):
    _fully_configured(monkeypatch)
    health = ch.collect_capture_health(_config())
    assert {c["status"] for c in health["channels"].values()} == {"ok"}
    assert ch.capture_health_violations(health) == []


def test_missing_prefix_is_misconfigured_not_green(monkeypatch):
    _fully_configured(monkeypatch)
    monkeypatch.delenv(ch.RAW_TRACE_S3_PREFIX_ENV)
    health = ch.collect_capture_health(_config())
    assert health["channels"]["raw_trace"]["status"] == "misconfigured_missing_s3_prefix"
    assert any(v.startswith("raw_trace:misconfigured") for v in ch.capture_health_violations(health))


def test_prefix_without_kms_is_misconfigured(monkeypatch):
    _fully_configured(monkeypatch)
    monkeypatch.delenv(ch.INCONTAINER_TRACE_KMS_KEY_ENV)
    health = ch.collect_capture_health(_config())
    assert (
        health["channels"]["incontainer_trace"]["status"]
        == "misconfigured_missing_kms_key"
    )


def test_disabled_channel_and_projector_off_are_violations(monkeypatch):
    _fully_configured(monkeypatch)
    monkeypatch.setenv(ch.SCORER_TRACE_CAPTURE_ENV, "false")
    monkeypatch.setenv(ch.PROJECTOR_ENABLED_ENV, "false")
    health = ch.collect_capture_health(_config())
    violations = ch.capture_health_violations(health)
    assert "scorer_trace:disabled" in violations
    assert any(v.startswith("trajectory_projector:disabled") for v in violations)


def test_manifest_uri_fallback_resolves_raw_and_scorer_prefixes(monkeypatch):
    monkeypatch.setenv(ch.INCONTAINER_TRACE_S3_PREFIX_ENV, "s3://bucket/incontainer")
    monkeypatch.setenv(ch.INCONTAINER_TRACE_KMS_KEY_ENV, "kms-key-1")
    monkeypatch.setenv(ch.PROJECTOR_ENABLED_ENV, "true")
    config = _config(private_model_manifest_uri="s3://bucket/lab/manifest.json")
    health = ch.collect_capture_health(config)
    assert health["channels"]["raw_trace"]["status"] == "ok"
    assert health["channels"]["raw_trace"]["s3_prefix"] == "s3://bucket/lab"
    assert health["channels"]["scorer_trace"]["status"] == "ok"
    # in-container has no manifest fallback by design
    assert health["channels"]["incontainer_trace"]["status"] == "ok"


def test_production_refuses_start_on_degraded_config(monkeypatch):
    # No prefixes at all: every channel misconfigured.
    config = _config(production_writes_enabled=True, private_model_manifest_uri="")
    with pytest.raises(ch.CaptureHealthError):
        ch.enforce_capture_health(config, worker_kind="hosted_worker")


def test_production_allow_degraded_override_starts_with_error_log(monkeypatch, caplog):
    config = _config(production_writes_enabled=True, private_model_manifest_uri="")
    monkeypatch.setenv(ch.ALLOW_DEGRADED_ENV, "true")
    with caplog.at_level(logging.ERROR, logger="gateway.research_lab.capture_health"):
        health = ch.enforce_capture_health(config, worker_kind="hosted_worker")
    assert health["violations"]
    assert any(
        "research_lab_capture_health_degraded" in record.getMessage()
        for record in caplog.records
    )


def test_non_production_logs_warning_but_starts(monkeypatch, caplog):
    config = _config(production_writes_enabled=False)
    with caplog.at_level(logging.INFO, logger="gateway.research_lab.capture_health"):
        health = ch.enforce_capture_health(config, worker_kind="scoring_worker")
    assert health["violations"]  # nothing configured in this env
    assert any(
        "research_lab_capture_health" in record.getMessage() for record in caplog.records
    )


def test_enforce_env_forces_refusal_in_non_production(monkeypatch):
    config = _config(production_writes_enabled=False)
    monkeypatch.setenv(ch.ENFORCE_ENV, "true")
    with pytest.raises(ch.CaptureHealthError):
        ch.enforce_capture_health(config, worker_kind="scoring_worker")


def test_healthy_production_start_passes(monkeypatch):
    _fully_configured(monkeypatch)
    config = _config(production_writes_enabled=True)
    health = ch.enforce_capture_health(config, worker_kind="hosted_worker")
    assert health["violations"] == []
