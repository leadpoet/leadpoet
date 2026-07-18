from __future__ import annotations

import base64
import json

import pytest

from gateway.research_lab.provider_profiles_v2 import (
    BENCHMARK_MODEL_PROFILE,
    bind_provider_profile_envelopes_v2,
    load_provider_profile_v2,
    provision_provider_profile_v2,
    require_worker_proxy_profile_v2,
    verify_required_worker_proxy_profiles_v2,
)
from gateway.tee.provider_broker_v2 import credential_value_hash
from gateway.utils.tee_kms_provision_v2 import (
    PROVIDER_ENVELOPE_SCHEMA_VERSION,
    kms_key_reference_hash,
    validate_job_provider_envelope,
)
from leadpoet_canonical.attested_v2 import sha256_bytes, sha256_json


def _envelope(provider_id: str, credential: str) -> dict:
    ciphertext = ("kms:" + credential).encode("utf-8")
    context = {"service": "leadpoet", "profile": "benchmark", "slot": provider_id}
    return {
        "schema_version": PROVIDER_ENVELOPE_SCHEMA_VERSION,
        "credential_slot": provider_id,
        # Job-scoped envelopes bind the exact plaintext value hash used by the
        # Nitro recipient, not the boot-wide provider reference domain.
        "credential_ref_hash": credential_value_hash(credential),
        "ciphertext_blob_b64": base64.b64encode(ciphertext).decode("ascii"),
        "ciphertext_blob_hash": sha256_bytes(ciphertext),
        "kms_key_id_hash": kms_key_reference_hash("kms-key-1"),
        "encryption_context": context,
        "encryption_context_hash": sha256_json(context),
    }


def test_benchmark_profile_is_optional_and_binds_only_encrypted_commitments(tmp_path):
    empty = load_provider_profile_v2(BENCHMARK_MODEL_PROFILE, config_dir=tmp_path)
    assert empty["credential_ref_hashes"] == {}
    assert empty["envelopes"] == []

    envelope = _envelope("exa", "benchmark-exa-secret")
    (tmp_path / "benchmark_exa.json").write_text(
        json.dumps(envelope),
        encoding="utf-8",
    )
    profile = load_provider_profile_v2(
        BENCHMARK_MODEL_PROFILE,
        config_dir=tmp_path,
    )
    assert profile["credential_ref_hashes"] == {
        "exa": credential_value_hash("benchmark-exa-secret")
    }
    encoded = json.dumps(profile, default=str)
    assert "benchmark-exa-secret" not in encoded

    bound = bind_provider_profile_envelopes_v2(
        profile,
        job_id="scoring-v2:model:1",
    )
    assert len(bound) == 1
    assert bound[0]["job_id"] == "scoring-v2:model:1"
    assert bound[0]["credential_slot"] == "exa"
    assert bound[0]["credential_value_hash"] == profile[
        "credential_ref_hashes"
    ]["exa"]
    assert "ciphertext_blob" not in bound[0]
    assert validate_job_provider_envelope(bound[0]) == bound[0]


@pytest.mark.asyncio
async def test_partial_profile_provision_is_released_on_failure(tmp_path):
    for provider_id, filename in (
        ("openrouter", "benchmark_openrouter.json"),
        ("scrapingdog", "benchmark_scrapingdog.json"),
    ):
        (tmp_path / filename).write_text(
            json.dumps(_envelope(provider_id, provider_id + "-secret")),
            encoding="utf-8",
        )
    profile = load_provider_profile_v2(
        "benchmark_scorer",
        config_dir=tmp_path,
    )
    calls = []

    class _Client:
        async def v2_release_job_credentials(self, job_id):
            calls.append(("release", job_id))
            return {"status": "released"}

    async def _provision(envelope, *, client):
        del client
        calls.append(("provision", envelope["credential_slot"]))
        if envelope["credential_slot"] == "scrapingdog":
            raise RuntimeError("injected KMS failure")
        return {"status": "ready"}

    with pytest.raises(RuntimeError, match="KMS failure"):
        await provision_provider_profile_v2(
            profile,
            job_id="scoring-v2:score:1",
            client=_Client(),
            provision=_provision,
        )
    assert calls[-1] == ("release", "scoring-v2:score:1")


def test_worker_proxy_profile_is_encrypted_scoped_and_required(tmp_path):
    proxy_url = "https://worker-7:password@proxy.example.com:443"
    (tmp_path / "scoring_proxy_07.json").write_text(
        json.dumps(_envelope("egress_proxy", proxy_url)),
        encoding="utf-8",
    )

    profile = load_provider_profile_v2(
        "default",
        config_dir=tmp_path,
        execution_role="gateway_scoring",
        worker_index=7,
        require_egress_proxy=True,
    )

    assert profile["execution_role"] == "gateway_scoring"
    assert profile["worker_index"] == 7
    assert profile["credential_ref_hashes"] == {
        "egress_proxy": credential_value_hash(proxy_url)
    }
    assert proxy_url not in json.dumps(profile, default=str)
    with pytest.raises(Exception, match="proxy envelope"):
        load_provider_profile_v2(
            "default",
            config_dir=tmp_path,
            execution_role="gateway_scoring",
            worker_index=8,
            require_egress_proxy=True,
        )

    required = require_worker_proxy_profile_v2(
        config_dir=tmp_path,
        execution_role="gateway_scoring",
        worker_index=7,
    )
    assert required["credential_ref_hashes"] == profile["credential_ref_hashes"]


def test_release_verifier_requires_all_worker_and_provider_profiles(tmp_path):
    proxy_url = "https://worker:password@proxy.example.com:443"
    for prefix, count in (("scoring", 7), ("autoresearch", 3)):
        for worker_index in range(count):
            (tmp_path / f"{prefix}_proxy_{worker_index:02d}.json").write_text(
                json.dumps(_envelope("egress_proxy", proxy_url)),
                encoding="utf-8",
            )
    for provider_id, filename in (
        ("exa", "benchmark_exa.json"),
        ("openrouter", "benchmark_openrouter.json"),
        ("scrapingdog", "benchmark_scrapingdog.json"),
        ("openrouter", "stale_parent_openrouter.json"),
        ("openrouter", "source_add_judge_openrouter.json"),
    ):
        (tmp_path / filename).write_text(
            json.dumps(_envelope(provider_id, filename + "-secret")),
            encoding="utf-8",
        )

    result = verify_required_worker_proxy_profiles_v2(config_dir=tmp_path)

    assert result["status"] == "ready"
    assert result["profile_count"] == 10
    assert result["worker_counts"] == {
        "gateway_autoresearch": 3,
        "gateway_scoring": 7,
    }


def test_release_verifier_rejects_noncontiguous_worker_profiles(tmp_path):
    proxy_url = "https://worker:password@proxy.example.com:443"
    for filename in (
        "scoring_proxy_00.json",
        "scoring_proxy_02.json",
        "autoresearch_proxy_00.json",
    ):
        (tmp_path / filename).write_text(
            json.dumps(_envelope("egress_proxy", proxy_url)),
            encoding="utf-8",
        )

    with pytest.raises(Exception, match="must be contiguous"):
        verify_required_worker_proxy_profiles_v2(config_dir=tmp_path)
