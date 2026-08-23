"""Fail-closed artifact/version binding for read-only Lab diagnostics."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from research_lab.canonical import sha256_json
from research_lab.eval.private_runtime import PrivateModelRuntimeError
from research_lab.sourcing_model_contract_check import (
    QUALIFICATION_SCORING_ADAPTER_VERSION_V1,
    QUALIFICATION_SCORING_ADAPTER_VERSION_V2,
)
import research_lab.eval.diagnostic_artifact as diagnostic_artifact
import scripts.verify_research_lab_parallel_benchmark as parallel_diagnostic
import scripts.verify_research_lab_private_baseline_live as baseline_diagnostic


IMAGE = (
    "123456789012.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model"
    "@sha256:" + "a" * 64
)


def _artifact(*, scoring_adapter_version: str, image_digest: str = IMAGE):
    payload = {
        "model_artifact_hash": "sha256:" + "b" * 64,
        "git_commit_sha": "c" * 40,
        "image_digest": image_digest,
        "config_hash": "sha256:" + "d" * 64,
        "component_registry_version": "sourcing-model-components:v2",
        "scoring_adapter_version": scoring_adapter_version,
        "manifest_uri": "s3://test-private-artifacts/model/manifest.json",
        "signature_ref": "s3://test-private-artifacts/model/manifest.sig",
        "build_id": "diagnostic-test",
    }
    return {**payload, "manifest_hash": sha256_json(payload)}


@pytest.mark.parametrize(
    "scoring_adapter_version",
    [
        QUALIFICATION_SCORING_ADAPTER_VERSION_V1,
        QUALIFICATION_SCORING_ADAPTER_VERSION_V2,
    ],
)
def test_diagnostic_artifact_derives_supported_signed_scorer_version(
    monkeypatch,
    scoring_adapter_version,
):
    payload = _artifact(scoring_adapter_version=scoring_adapter_version)
    verified = []
    monkeypatch.setattr(
        diagnostic_artifact,
        "load_private_artifact_manifest",
        lambda _uri: payload,
    )
    monkeypatch.setattr(
        diagnostic_artifact,
        "verify_private_artifact_manifest_signature",
        lambda artifact, *, key_id: verified.append(
            (artifact.manifest_hash, key_id)
        ),
    )

    artifact = (
        diagnostic_artifact.load_verified_diagnostic_private_model_artifact(
            "s3://test-private-artifacts/model/manifest.json",
            expected_image_digest=IMAGE,
        )
    )

    assert artifact.scoring_adapter_version == scoring_adapter_version
    assert verified == [
        (
            payload["manifest_hash"],
            diagnostic_artifact.DEFAULT_PRIVATE_MODEL_ARTIFACT_SIGNING_KMS_KEY_ID,
        )
    ]


@pytest.mark.parametrize(
    ("scoring_adapter_version", "image_digest", "error"),
    [
        ("qualification-company-scorer:v3", IMAGE, "unsupported"),
        (
            QUALIFICATION_SCORING_ADAPTER_VERSION_V2,
            IMAGE.replace("a" * 64, "e" * 64),
            "differs from the signed",
        ),
    ],
)
def test_diagnostic_artifact_rejects_unknown_version_or_image_mismatch(
    monkeypatch,
    scoring_adapter_version,
    image_digest,
    error,
):
    monkeypatch.setattr(
        diagnostic_artifact,
        "load_private_artifact_manifest",
        lambda _uri: _artifact(
            scoring_adapter_version=scoring_adapter_version,
            image_digest=image_digest,
        ),
    )
    monkeypatch.setattr(
        diagnostic_artifact,
        "verify_private_artifact_manifest_signature",
        lambda *_args, **_kwargs: {},
    )

    with pytest.raises(PrivateModelRuntimeError, match=error):
        diagnostic_artifact.load_verified_diagnostic_private_model_artifact(
            "s3://test-private-artifacts/model/manifest.json",
            expected_image_digest=IMAGE,
        )


@pytest.mark.parametrize(
    "module",
    [baseline_diagnostic, parallel_diagnostic],
)
def test_diagnostic_cli_fails_before_execution_when_artifact_is_not_admitted(
    monkeypatch,
    module,
):
    for name in (
        "EXA_API_KEY",
        "SCRAPINGDOG_API_KEY",
        "QUALIFICATION_SCRAPINGDOG_API_KEY",
        "OPENROUTER_API_KEY",
        "QUALIFICATION_OPENROUTER_API_KEY",
        "OPENROUTER_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(
        module,
        "load_verified_diagnostic_private_model_artifact",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            PrivateModelRuntimeError("unsupported scorer version")
        ),
    )
    monkeypatch.setattr(
        module.sys,
        "argv",
        [
            module.__file__,
            "--image",
            IMAGE,
            "--artifact-manifest-uri",
            "s3://test-private-artifacts/model/manifest.json",
        ],
    )

    assert module.main() == 2


def test_private_baseline_threads_signed_scorer_version(monkeypatch):
    scorer_kwargs = []

    class FakeRunner:
        def __init__(self, _spec):
            pass

        def __call__(self, _icp, _context):
            return []

    class FakeScorer:
        def __init__(self, **kwargs):
            scorer_kwargs.append(kwargs)

        async def score_with_breakdowns(self, _outputs, _icp, _is_reference):
            return []

    monkeypatch.setattr(baseline_diagnostic, "DockerPrivateModelRunner", FakeRunner)
    monkeypatch.setattr(
        baseline_diagnostic,
        "QualificationStyleCompanyScorer",
        FakeScorer,
    )

    result = asyncio.run(
        baseline_diagnostic._run(
            IMAGE,
            1,
            1,
            False,
            scoring_adapter_version=QUALIFICATION_SCORING_ADAPTER_VERSION_V2,
        )
    )

    assert result == 1
    assert scorer_kwargs == [
        {
            "reference_scoring_adapter_version": (
                QUALIFICATION_SCORING_ADAPTER_VERSION_V2
            ),
            "candidate_scoring_adapter_version": (
                QUALIFICATION_SCORING_ADAPTER_VERSION_V2
            ),
        }
    ]


def test_parallel_benchmark_threads_signed_scorer_version(monkeypatch):
    scorer_kwargs = []

    class FakeRunner:
        def __init__(self, _spec):
            pass

        def __call__(self, _icp, _context):
            return []

    class FakeScorer:
        def __init__(self, **kwargs):
            scorer_kwargs.append(kwargs)

    monkeypatch.setattr(parallel_diagnostic, "DockerPrivateModelRunner", FakeRunner)
    monkeypatch.setattr(
        parallel_diagnostic,
        "QualificationStyleCompanyScorer",
        FakeScorer,
    )
    args = SimpleNamespace(
        concurrency=1,
        exa_api_key="",
        exa_max_rps=0.8,
        image=IMAGE,
        retry_concurrency=1,
        retry_rounds=0,
        scoring_adapter_version=QUALIFICATION_SCORING_ADAPTER_VERSION_V2,
        timeout_seconds=1,
    )

    result = asyncio.run(
        parallel_diagnostic._run(
            args,
            [{"icp_id": "diagnostic"}],
        )
    )

    assert result == 0
    assert scorer_kwargs == [
        {
            "reference_scoring_adapter_version": (
                QUALIFICATION_SCORING_ADAPTER_VERSION_V2
            ),
            "candidate_scoring_adapter_version": (
                QUALIFICATION_SCORING_ADAPTER_VERSION_V2
            ),
        }
    ]
