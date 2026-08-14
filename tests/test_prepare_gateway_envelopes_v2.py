from __future__ import annotations

import json

import pytest

from gateway.research_lab.worker_autostart import (
    build_research_lab_worker_autostart_plan,
)
from gateway.tee import prepare_gateway_envelopes_v2 as envelope_module
from gateway.tee.prepare_gateway_envelopes_v2 import (
    install_gateway_envelopes_v2,
    load_environment_file,
    prepare_gateway_envelopes_v2,
    scrub_parent_environment_file_v2,
)
from gateway.tee.proxy_transport_preflight_v2 import (
    WorkerProxyTransportPreflightV2Error,
)
from gateway.tee.provider_broker_v2 import (
    credential_reference_hash,
    credential_value_hash,
)
from gateway.utils.tee_kms_provision_v2 import validate_provider_envelope


class KMS:
    def __init__(self):
        self.requests = []

    def encrypt(self, **request):
        self.requests.append(request)
        return {
            "KeyId": "arn:aws:kms:us-east-1:123:key/gateway-v2",
            "CiphertextBlob": ("ciphertext:%03d" % len(self.requests)).encode(),
        }


def _skip_proxy_probe(_fleets):
    return None


def _environment():
    return {
        "RESEARCH_LAB_HOSTED_RUNS_ENABLED": "true",
        "RESEARCH_LAB_EVALUATION_BUNDLES_ENABLED": "true",
        "RESEARCH_LAB_V2_OPENROUTER_API_KEY": "openrouter-secret",
        "RESEARCH_LAB_V2_EXA_API_KEY": "exa-secret",
        "RESEARCH_LAB_V2_SCRAPINGDOG_API_KEY": "scrapingdog-secret",
        "RESEARCH_LAB_V2_DEEPLINE_API_KEY": "deepline-secret",
        "SUPABASE_SERVICE_ROLE_KEY": "supabase-secret",
        "TRUELIST_API_KEY": "truelist-secret",
        "RESEARCH_LAB_V2_AUTORESEARCH_HTTPS_PROXY_1": "https://hosted-1.example.com",
        "RESEARCH_LAB_V2_AUTORESEARCH_HTTPS_PROXY_2": "https://hosted-2.example.com",
        "RESEARCH_LAB_V2_SCORING_HTTPS_PROXY_1": "https://scoring-1.example.com",
        "RESEARCH_LAB_V2_SCORING_HTTPS_PROXY_2": "https://scoring-2.example.com",
        "RESEARCH_LAB_V2_SCORING_HTTPS_PROXY_3": "https://scoring-3.example.com",
    }


def _legacy_http_proxy_environment():
    environment = _environment()
    environment.update(
        {
            "RESEARCH_LAB_AUTO_RESEARCH_WEBSHARE_PROXY_1": (
                "http://user:password@hosted-1.example.com:12431"
            ),
            "RESEARCH_LAB_AUTO_RESEARCH_WEBSHARE_PROXY_2": (
                "http://user:password@hosted-2.example.com:12432"
            ),
            "RESEARCH_LAB_QUALIFICATION_WEBSHARE_PROXY_1": (
                "http://user:password@scoring-1.example.com:13431"
            ),
            "RESEARCH_LAB_QUALIFICATION_WEBSHARE_PROXY_2": (
                "http://user:password@scoring-2.example.com:13432"
            ),
            "RESEARCH_LAB_QUALIFICATION_WEBSHARE_PROXY_3": (
                "http://user:password@scoring-3.example.com:13433"
            ),
        }
    )
    return environment


def test_prepares_complete_dynamic_gateway_envelope_set(tmp_path):
    kms = KMS()
    output = tmp_path / "v2"
    result = prepare_gateway_envelopes_v2(
        environment=_environment(),
        kms_key_id="alias/gateway-v2",
        deploy_commit="1" * 40,
        output_dir=output,
        kms_client=kms,
        proxy_fleet_probe=_skip_proxy_probe,
    )

    assert result["hosted_worker_count"] == 2
    assert result["scoring_worker_count"] == 3
    assert result["worker_proxy_source"] == {
        "gateway_autoresearch": "v2_tls",
        "gateway_scoring": "v2_tls",
    }
    assert (output / "autoresearch_proxy_01.json").is_file()
    assert (output / "scoring_proxy_02.json").is_file()
    assert result["required_count_environment"] == {
        "RESEARCH_LAB_HOSTED_WORKER_PROCESS_COUNT": "2",
        "RESEARCH_LAB_SCORING_WORKER_PROCESS_COUNT": "3",
    }
    assert "SUPABASE_SERVICE_ROLE_KEY" not in result["plaintext_environment_names_to_remove"]
    assert "TRUELIST_API_KEY" not in result["plaintext_environment_names_to_remove"]
    documents = [
        json.loads(path.read_text())
        for path in output.glob("*.json")
        if path.name != "gateway-v2-env-transition.json"
    ]
    assert len(documents) == 7 + 5 + 5
    assert all(validate_provider_envelope(document) for document in documents)
    assert json.loads((output / "openrouter.json").read_text())[
        "credential_ref_hash"
    ] == credential_reference_hash("openrouter-secret")
    assert json.loads((output / "benchmark_openrouter.json").read_text())[
        "credential_ref_hash"
    ] == credential_value_hash("openrouter-secret")
    assert json.loads((output / "autoresearch_proxy_00.json").read_text())[
        "credential_ref_hash"
    ] == credential_value_hash("https://hosted-1.example.com")
    assert credential_reference_hash("openrouter-secret") != (
        credential_value_hash("openrouter-secret")
    )
    assert not any(
        secret in json.dumps(documents)
        for secret in (
            "openrouter-secret",
            "exa-secret",
            "scrapingdog-secret",
            "https://hosted-1.example.com",
        )
    )


def test_v2_proxy_fleets_replace_legacy_values_and_scrub_every_alias(tmp_path):
    environment = _environment()
    environment.update(
        {
            "RESEARCH_LAB_AUTO_RESEARCH_WEBSHARE_PROXY_1": (
                "http://legacy-hosted.example.com:18080"
            ),
            "RESEARCH_LAB_QUALIFICATION_WEBSHARE_PROXY_1": (
                "http://legacy-scoring.example.com:19090"
            ),
        }
    )
    plan = build_research_lab_worker_autostart_plan(environment)
    assert plan.hosted.proxy_source == "v2_tls"
    assert plan.scoring.proxy_source == "v2_tls"
    assert plan.hosted.proxy_values == (
        "https://hosted-1.example.com",
        "https://hosted-2.example.com",
    )
    assert plan.scoring.proxy_values == (
        "https://scoring-1.example.com",
        "https://scoring-2.example.com",
        "https://scoring-3.example.com",
    )

    result = prepare_gateway_envelopes_v2(
        environment=environment,
        kms_key_id="alias/gateway-v2",
        deploy_commit="1" * 40,
        output_dir=tmp_path / "v2",
        kms_client=KMS(),
        proxy_fleet_probe=_skip_proxy_probe,
    )
    removal_names = set(result["plaintext_environment_names_to_remove"])
    assert {
        "RESEARCH_LAB_V2_AUTORESEARCH_HTTPS_PROXY_1",
        "RESEARCH_LAB_V2_SCORING_HTTPS_PROXY_1",
        "RESEARCH_LAB_AUTO_RESEARCH_WEBSHARE_PROXY_1",
        "RESEARCH_LAB_QUALIFICATION_WEBSHARE_PROXY_1",
    } <= removal_names


def test_production_shaped_authenticated_http_connect_fleet_is_sealed(tmp_path):
    environment = {
        name: value
        for name, value in _environment().items()
        if "V2_AUTORESEARCH_HTTPS_PROXY" not in name
        and "V2_SCORING_HTTPS_PROXY" not in name
    }
    environment.update(
        {
            "RESEARCH_LAB_AUTO_RESEARCH_WEBSHARE_PROXY_1": (
                "http://user:password@legacy-hosted.example.com:6162"
            ),
            "RESEARCH_LAB_QUALIFICATION_WEBSHARE_PROXY_1": (
                "http://user:password@legacy-scoring.example.com:7421"
            ),
        }
    )
    kms = KMS()
    probe_calls = []

    result = prepare_gateway_envelopes_v2(
        environment=environment,
        kms_key_id="alias/gateway-v2",
        deploy_commit="1" * 40,
        output_dir=tmp_path / "v2",
        kms_client=kms,
        proxy_fleet_probe=lambda fleets: probe_calls.append(fleets),
    )

    assert result["worker_proxy_source"] == {
        "gateway_autoresearch": "legacy",
        "gateway_scoring": "legacy",
    }
    assert result["hosted_worker_count"] == 1
    assert result["scoring_worker_count"] == 1
    assert probe_calls == [
        {
            "gateway_autoresearch": (
                "http://user:password@legacy-hosted.example.com:6162",
            ),
            "gateway_scoring": (
                "http://user:password@legacy-scoring.example.com:7421",
            ),
        }
    ]
    assert (tmp_path / "v2" / "autoresearch_proxy_00.json").is_file()
    assert (tmp_path / "v2" / "scoring_proxy_00.json").is_file()


def _production_sized_proxy_environment() -> dict[str, str]:
    environment = {
        name: value
        for name, value in _environment().items()
        if "V2_AUTORESEARCH_HTTPS_PROXY" not in name
        and "V2_SCORING_HTTPS_PROXY" not in name
    }
    environment.update(
        {
            "RESEARCH_LAB_AUTO_RESEARCH_WEBSHARE_PROXY_%d" % index: (
                "http://user:password@8.8.8.8:%d"
                % (6100 + index)
            )
            for index in range(1, 11)
        }
    )
    environment.update(
        {
            "RESEARCH_LAB_QUALIFICATION_WEBSHARE_PROXY_%d" % index: (
                "http://user:password@1.1.1.1:%d"
                % (7100 + index)
            )
            for index in range(1, 26)
        }
    )
    return environment


def test_production_sized_legacy_http_fleets_preserve_all_worker_slots(
    tmp_path,
):
    environment = _production_sized_proxy_environment()
    output = tmp_path / "v2"
    observed_fleets = []

    result = prepare_gateway_envelopes_v2(
        environment=environment,
        kms_key_id="alias/gateway-v2",
        deploy_commit="1" * 40,
        output_dir=output,
        kms_client=KMS(),
        proxy_fleet_probe=lambda fleets: observed_fleets.append(fleets),
    )

    assert result["worker_proxy_source"] == {
        "gateway_autoresearch": "legacy",
        "gateway_scoring": "legacy",
    }
    assert result["hosted_worker_count"] == 10
    assert result["scoring_worker_count"] == 25
    assert len(observed_fleets) == 1
    assert len(observed_fleets[0]["gateway_autoresearch"]) == 10
    assert len(observed_fleets[0]["gateway_scoring"]) == 25
    assert len(tuple(output.glob("autoresearch_proxy_*.json"))) == 10
    assert len(tuple(output.glob("scoring_proxy_*.json"))) == 25
    assert result["required_count_environment"] == {
        "RESEARCH_LAB_HOSTED_WORKER_PROCESS_COUNT": "10",
        "RESEARCH_LAB_SCORING_WORKER_PROCESS_COUNT": "25",
    }
    assert result["worker_proxy_profile_counts"] == {
        "gateway_autoresearch": {
            "configured": 10,
            "verified": 10,
            "quarantined": 0,
            "sealed_worker_slots": 10,
        },
        "gateway_scoring": {
            "configured": 25,
            "verified": 25,
            "quarantined": 0,
            "sealed_worker_slots": 25,
        },
    }


def test_failed_live_profiles_are_not_sealed_but_worker_capacity_is_preserved(
    tmp_path,
):
    environment = _production_sized_proxy_environment()
    output = tmp_path / "v2"
    failed_hosted = environment[
        "RESEARCH_LAB_AUTO_RESEARCH_WEBSHARE_PROXY_3"
    ]
    failed_scoring = environment[
        "RESEARCH_LAB_QUALIFICATION_WEBSHARE_PROXY_3"
    ]

    def verified_fleets(fleets):
        return {
            "gateway_autoresearch": tuple(
                value
                for value in fleets["gateway_autoresearch"]
                if value != failed_hosted
            ),
            "gateway_scoring": tuple(
                value
                for value in fleets["gateway_scoring"]
                if value != failed_scoring
            ),
        }

    kms = KMS()
    result = prepare_gateway_envelopes_v2(
        environment=environment,
        kms_key_id="alias/gateway-v2",
        deploy_commit="1" * 40,
        output_dir=output,
        kms_client=kms,
        proxy_fleet_probe=verified_fleets,
    )

    assert result["hosted_worker_count"] == 10
    assert result["scoring_worker_count"] == 25
    assert result["worker_proxy_profile_counts"] == {
        "gateway_autoresearch": {
            "configured": 10,
            "verified": 9,
            "quarantined": 1,
            "sealed_worker_slots": 10,
        },
        "gateway_scoring": {
            "configured": 25,
            "verified": 24,
            "quarantined": 1,
            "sealed_worker_slots": 25,
        },
    }
    assert len(tuple(output.glob("autoresearch_proxy_*.json"))) == 10
    assert len(tuple(output.glob("scoring_proxy_*.json"))) == 25
    worker_plaintexts = {
        request["Plaintext"].decode("utf-8")
        for request in kms.requests
        if request["EncryptionContext"].get("leadpoet:purpose")
        == "gateway-worker-egress-v2"
    }
    assert failed_hosted not in worker_plaintexts
    assert failed_scoring not in worker_plaintexts
    assert len(result["worker_proxy_credential_ref_hashes"][
        "gateway_autoresearch"
    ]) == 10
    assert len(result["worker_proxy_credential_ref_hashes"][
        "gateway_scoring"
    ]) == 25


def test_install_carries_verified_selection_into_generated_envelopes(tmp_path):
    environment = _environment()
    failed_scoring = environment[
        "RESEARCH_LAB_V2_SCORING_HTTPS_PROXY_2"
    ]
    kms = KMS()

    def verified_fleets(fleets):
        return {
            **fleets,
            "gateway_scoring": tuple(
                value
                for value in fleets["gateway_scoring"]
                if value != failed_scoring
            ),
        }

    result = install_gateway_envelopes_v2(
        environment=environment,
        kms_key_id="alias/gateway-v2",
        deploy_commit="1" * 40,
        install_dir=tmp_path / "v2",
        kms_client=kms,
        proxy_fleet_probe=verified_fleets,
    )

    assert result["status"] == "installed"
    assert result["worker_proxy_profile_counts"]["gateway_scoring"] == {
        "configured": 3,
        "verified": 2,
        "quarantined": 1,
        "sealed_worker_slots": 3,
    }
    scoring_plaintexts = [
        request["Plaintext"].decode("utf-8")
        for request in kms.requests
        if request["EncryptionContext"].get("leadpoet:role")
        == "gateway_scoring"
    ]
    assert len(scoring_plaintexts) == 3
    assert failed_scoring not in scoring_plaintexts


def test_invalid_probe_selection_fails_before_kms(tmp_path):
    kms = KMS()

    with pytest.raises(
        Exception,
        match="worker proxy preflight returned an invalid fleet selection",
    ):
        prepare_gateway_envelopes_v2(
            environment=_environment(),
            kms_key_id="alias/gateway-v2",
            deploy_commit="1" * 40,
            output_dir=tmp_path / "v2",
            kms_client=kms,
            proxy_fleet_probe=lambda fleets: {
                **fleets,
                "gateway_scoring": ("https://unknown.example.com",),
            },
        )

    assert kms.requests == []
    assert not (tmp_path / "v2").exists()


def test_v2_proxy_migration_requires_explicit_production_worker_counts(
    tmp_path,
):
    environment = _production_sized_proxy_environment()
    environment.update(
        {
            "RESEARCH_LAB_V2_AUTORESEARCH_HTTPS_PROXY_1": (
                "https://hosted-v2.example.com:443"
            ),
            "RESEARCH_LAB_V2_SCORING_HTTPS_PROXY_1": (
                "https://scoring-v2.example.com:443"
            ),
        }
    )
    kms = KMS()

    with pytest.raises(
        Exception,
        match=(
            "gateway_autoresearch V2 proxy migration would reduce worker "
            "coverage from 10 legacy slots to 1 selected proxy profile"
        ),
    ):
        prepare_gateway_envelopes_v2(
            environment=environment,
            kms_key_id="alias/gateway-v2",
            deploy_commit="1" * 40,
            output_dir=tmp_path / "v2",
            kms_client=kms,
            proxy_fleet_probe=_skip_proxy_probe,
        )

    assert kms.requests == []
    assert not (tmp_path / "v2").exists()


def test_one_v2_proxy_per_role_preserves_production_worker_capacity(
    tmp_path,
):
    environment = _production_sized_proxy_environment()
    environment.update(
        {
            "RESEARCH_LAB_V2_AUTORESEARCH_HTTPS_PROXY_1": (
                "https://hosted-v2.example.com:443"
            ),
            "RESEARCH_LAB_V2_SCORING_HTTPS_PROXY_1": (
                "https://scoring-v2.example.com:443"
            ),
            "RESEARCH_LAB_HOSTED_WORKER_PROCESS_COUNT": "10",
            "RESEARCH_LAB_SCORING_WORKER_PROCESS_COUNT": "25",
        }
    )
    kms = KMS()
    observed = []

    result = prepare_gateway_envelopes_v2(
        environment=environment,
        kms_key_id="alias/gateway-v2",
        deploy_commit="1" * 40,
        output_dir=tmp_path / "v2",
        kms_client=kms,
        proxy_fleet_probe=lambda fleets: observed.append(fleets),
    )

    assert observed == [
        {
            "gateway_autoresearch": (
                "https://hosted-v2.example.com:443",
            ),
            "gateway_scoring": (
                "https://scoring-v2.example.com:443",
            ),
        }
    ]
    assert result["hosted_worker_count"] == 10
    assert result["scoring_worker_count"] == 25
    assert result["required_count_environment"] == {
        "RESEARCH_LAB_HOSTED_WORKER_PROCESS_COUNT": "10",
        "RESEARCH_LAB_SCORING_WORKER_PROCESS_COUNT": "25",
    }
    assert result["worker_proxy_source"] == {
        "gateway_autoresearch": "v2_tls",
        "gateway_scoring": "v2_tls",
    }
    assert len(list((tmp_path / "v2").glob("autoresearch_proxy_*.json"))) == 10
    assert len(list((tmp_path / "v2").glob("scoring_proxy_*.json"))) == 25
    assert len(kms.requests) == 47


def test_tls_connect_capability_failure_blocks_before_kms(tmp_path):
    environment = _environment()
    kms = KMS()
    observed = []

    def fail_probe(fleets):
        observed.append(fleets)
        raise WorkerProxyTransportPreflightV2Error(
            "gateway_autoresearch worker proxy 1 failed V2 TLS CONNECT preflight"
        )

    with pytest.raises(
        Exception,
        match="failed V2 TLS CONNECT preflight",
    ):
        prepare_gateway_envelopes_v2(
            environment=environment,
            kms_key_id="alias/gateway-v2",
            deploy_commit="1" * 40,
            output_dir=tmp_path / "v2",
            kms_client=kms,
            proxy_fleet_probe=fail_probe,
        )

    assert len(observed) == 1
    assert set(observed[0]) == {"gateway_autoresearch", "gateway_scoring"}
    assert kms.requests == []
    assert not (tmp_path / "v2").exists()


def test_partial_v2_fleet_reuses_tls_profile_without_legacy_fallback(tmp_path):
    environment = _environment()
    environment.update(
        {
            "RESEARCH_LAB_HOSTED_WORKER_PROCESS_COUNT": "2",
            "RESEARCH_LAB_AUTO_RESEARCH_WEBSHARE_PROXY_1": (
                "http://legacy-hosted-1.example.com:6162"
            ),
            "RESEARCH_LAB_AUTO_RESEARCH_WEBSHARE_PROXY_2": (
                "http://legacy-hosted-2.example.com:6162"
            ),
        }
    )
    del environment["RESEARCH_LAB_V2_AUTORESEARCH_HTTPS_PROXY_2"]
    kms = KMS()
    observed = []

    output = tmp_path / "v2"
    result = prepare_gateway_envelopes_v2(
        environment=environment,
        kms_key_id="alias/gateway-v2",
        deploy_commit="1" * 40,
        output_dir=output,
        kms_client=kms,
        proxy_fleet_probe=lambda fleets: observed.append(fleets),
    )

    assert observed == [
        {
            "gateway_autoresearch": (
                "https://hosted-1.example.com",
            ),
            "gateway_scoring": (
                "https://scoring-1.example.com",
                "https://scoring-2.example.com",
                "https://scoring-3.example.com",
            ),
        }
    ]
    hosted_hash = credential_value_hash("https://hosted-1.example.com")
    assert result["worker_proxy_credential_ref_hashes"][
        "gateway_autoresearch"
    ] == [hosted_hash, hosted_hash]
    assert json.loads((output / "autoresearch_proxy_00.json").read_text())[
        "credential_ref_hash"
    ] == hosted_hash
    assert json.loads((output / "autoresearch_proxy_01.json").read_text())[
        "credential_ref_hash"
    ] == hosted_hash
    hosted_contexts = [
        request["EncryptionContext"]
        for request in kms.requests
        if request["EncryptionContext"].get("leadpoet:role")
        == "gateway_autoresearch"
    ]
    assert [context["leadpoet:worker_index"] for context in hosted_contexts] == [
        "0",
        "1",
    ]
    assert all(
        context["leadpoet:commit"] == "1" * 40
        and context["leadpoet:purpose"] == "gateway-worker-egress-v2"
        for context in hosted_contexts
    )
    assert result["worker_proxy_source"]["gateway_autoresearch"] == "v2_tls"
    assert {
        "RESEARCH_LAB_AUTO_RESEARCH_WEBSHARE_PROXY_1",
        "RESEARCH_LAB_AUTO_RESEARCH_WEBSHARE_PROXY_2",
    } <= set(result["plaintext_environment_names_to_remove"])

    request_count = len(kms.requests)
    reused = install_gateway_envelopes_v2(
        environment=environment,
        kms_key_id="alias/gateway-v2",
        deploy_commit="1" * 40,
        install_dir=output,
        kms_client=kms,
        proxy_fleet_probe=_skip_proxy_probe,
    )
    assert reused["status"] == "reused"
    assert len(kms.requests) == request_count


def test_install_uses_canonical_secret_names_and_reuses_exact_commit(tmp_path):
    environment = _environment()
    environment["OPENROUTER_API_KEY"] = environment.pop(
        "RESEARCH_LAB_V2_OPENROUTER_API_KEY"
    )
    environment["EXA_API_KEY"] = environment.pop("RESEARCH_LAB_V2_EXA_API_KEY")
    environment["SCRAPINGDOG_API_KEY"] = environment.pop(
        "RESEARCH_LAB_V2_SCRAPINGDOG_API_KEY"
    )
    environment["DEEPLINE_API_KEY"] = environment.pop(
        "RESEARCH_LAB_V2_DEEPLINE_API_KEY"
    )
    destination = tmp_path / "v2"
    destination.mkdir()
    (destination / "acceptance-corpus-v2.json").write_text("{}")
    kms = KMS()
    installed = install_gateway_envelopes_v2(
        environment=environment,
        kms_key_id="alias/gateway-v2",
        deploy_commit="1" * 40,
        install_dir=destination,
        kms_client=kms,
        proxy_fleet_probe=_skip_proxy_probe,
    )
    request_count = len(kms.requests)
    assert installed["status"] == "installed"
    assert "OPENROUTER_API_KEY" in installed["plaintext_environment_names_to_remove"]
    assert (destination / "acceptance-corpus-v2.json").read_text() == "{}"

    reused = install_gateway_envelopes_v2(
        environment=environment,
        kms_key_id="alias/gateway-v2",
        deploy_commit="1" * 40,
        install_dir=destination,
        kms_client=kms,
        proxy_fleet_probe=_skip_proxy_probe,
    )
    assert reused["status"] == "reused"
    assert len(kms.requests) == request_count


def test_install_preserves_artifact_master_key_across_release_and_rollback(
    tmp_path,
):
    destination = tmp_path / "v2"
    kms = KMS()
    first = install_gateway_envelopes_v2(
        environment=_environment(),
        kms_key_id="alias/gateway-v2",
        deploy_commit="1" * 40,
        install_dir=destination,
        kms_client=kms,
        proxy_fleet_probe=_skip_proxy_probe,
    )
    first_envelope = json.loads(
        (destination / "artifact_master_key.json").read_text()
    )
    first_key_encryptions = [
        request
        for request in kms.requests
        if request["EncryptionContext"].get("leadpoet:slot")
        == "artifact_master_key"
    ]

    second = install_gateway_envelopes_v2(
        environment=_environment(),
        kms_key_id="alias/gateway-v2",
        deploy_commit="2" * 40,
        install_dir=destination,
        kms_client=kms,
        proxy_fleet_probe=_skip_proxy_probe,
    )
    second_envelope = json.loads(
        (destination / "artifact_master_key.json").read_text()
    )
    rollback = install_gateway_envelopes_v2(
        environment=_environment(),
        kms_key_id="alias/gateway-v2",
        deploy_commit="1" * 40,
        install_dir=destination,
        kms_client=kms,
        proxy_fleet_probe=_skip_proxy_probe,
    )
    rollback_envelope = json.loads(
        (destination / "artifact_master_key.json").read_text()
    )

    assert first["artifact_master_key_ref_hash"] == second[
        "artifact_master_key_ref_hash"
    ] == rollback["artifact_master_key_ref_hash"]
    assert first_envelope == second_envelope == rollback_envelope
    assert len(first_key_encryptions) == 1
    assert len(
        [
            request
            for request in kms.requests
            if request["EncryptionContext"].get("leadpoet:slot")
            == "artifact_master_key"
        ]
    ) == 1


def test_install_fails_closed_on_invalid_existing_artifact_key(tmp_path):
    destination = tmp_path / "v2"
    destination.mkdir()
    (destination / "artifact_master_key.json").write_text("{}")

    with pytest.raises(
        Exception, match="existing artifact master key envelope is invalid"
    ):
        install_gateway_envelopes_v2(
            environment=_environment(),
            kms_key_id="alias/gateway-v2",
            deploy_commit="1" * 40,
            install_dir=destination,
            kms_client=KMS(),
            proxy_fleet_probe=_skip_proxy_probe,
        )

    assert (destination / "artifact_master_key.json").read_text() == "{}"


def test_install_fails_closed_on_artifact_key_symlink(tmp_path):
    destination = tmp_path / "v2"
    destination.mkdir()
    external = tmp_path / "external-key.json"
    external.write_text("{}")
    (destination / "artifact_master_key.json").symlink_to(external)

    with pytest.raises(Exception, match="not a regular file"):
        install_gateway_envelopes_v2(
            environment=_environment(),
            kms_key_id="alias/gateway-v2",
            deploy_commit="1" * 40,
            install_dir=destination,
            kms_client=KMS(),
            proxy_fleet_probe=_skip_proxy_probe,
        )

    assert external.read_text() == "{}"


def test_transition_removes_every_alias_of_sealed_parent_plaintext(tmp_path):
    environment = _environment()
    environment.update(
        {
            "FULFILLMENT_OPENROUTER_API_KEY": "openrouter-secret",
            "QUALIFICATION_OPENROUTER_API_KEY": "openrouter-secret",
            "QUALIFICATION_SCRAPINGDOG_API_KEY": "scrapingdog-secret",
            "UNRELATED_RUNTIME_VALUE": "keep-me",
        }
    )
    result = prepare_gateway_envelopes_v2(
        environment=environment,
        kms_key_id="alias/gateway-v2",
        deploy_commit="1" * 40,
        output_dir=tmp_path / "v2",
        kms_client=KMS(),
        proxy_fleet_probe=_skip_proxy_probe,
    )

    removal_names = set(result["plaintext_environment_names_to_remove"])
    assert {
        "RESEARCH_LAB_V2_OPENROUTER_API_KEY",
        "FULFILLMENT_OPENROUTER_API_KEY",
        "QUALIFICATION_OPENROUTER_API_KEY",
        "RESEARCH_LAB_V2_SCRAPINGDOG_API_KEY",
        "QUALIFICATION_SCRAPINGDOG_API_KEY",
    } <= removal_names
    assert "SUPABASE_SERVICE_ROLE_KEY" not in removal_names
    assert "TRUELIST_API_KEY" not in removal_names
    assert "UNRELATED_RUNTIME_VALUE" not in removal_names
    assert len(result["plaintext_credential_ref_hashes_to_remove"]) == 9

    environment["LATE_OPENROUTER_ALIAS"] = "openrouter-secret"
    environment["LATE_PROXY_ALIAS"] = "https://hosted-1.example.com"
    parent_environment = tmp_path / "gateway-parent.env"
    parent_environment.write_text(
        "\n".join(
            f"export {name}={value}"
            for name, value in environment.items()
        )
        + "\n"
    )
    transition = scrub_parent_environment_file_v2(
        environment_path=parent_environment,
        transition_report_path=tmp_path / "v2" / "gateway-v2-env-transition.json",
    )
    scrubbed = load_environment_file(parent_environment)
    assert "FULFILLMENT_OPENROUTER_API_KEY" not in scrubbed
    assert "QUALIFICATION_OPENROUTER_API_KEY" not in scrubbed
    assert "QUALIFICATION_SCRAPINGDOG_API_KEY" not in scrubbed
    assert "LATE_OPENROUTER_ALIAS" not in scrubbed
    assert "LATE_PROXY_ALIAS" not in scrubbed
    assert scrubbed["SUPABASE_SERVICE_ROLE_KEY"] == "supabase-secret"
    assert scrubbed["TRUELIST_API_KEY"] == "truelist-secret"
    assert scrubbed["UNRELATED_RUNTIME_VALUE"] == "keep-me"
    assert transition["installed_count_environment"] == {
        "RESEARCH_LAB_HOSTED_WORKER_PROCESS_COUNT": "2",
        "RESEARCH_LAB_SCORING_WORKER_PROCESS_COUNT": "3",
    }
    plan = build_research_lab_worker_autostart_plan(scrubbed)
    assert plan.hosted.enabled
    assert plan.hosted.worker_count == 2
    assert plan.hosted.proxy_values == ()
    assert plan.scoring.enabled
    assert plan.scoring.worker_count == 3
    assert plan.scoring.proxy_values == ()


def test_transition_rejects_worker_counts_not_bound_to_sealed_profiles(tmp_path):
    output = tmp_path / "v2"
    prepare_gateway_envelopes_v2(
        environment=_environment(),
        kms_key_id="alias/gateway-v2",
        deploy_commit="1" * 40,
        output_dir=output,
        kms_client=KMS(),
        proxy_fleet_probe=_skip_proxy_probe,
    )
    report_path = output / "gateway-v2-env-transition.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["required_count_environment"][
        "RESEARCH_LAB_HOSTED_WORKER_PROCESS_COUNT"
    ] = "99"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    parent_environment = tmp_path / "gateway-parent.env"
    parent_environment.write_text("export KEEP_ME=true\n", encoding="utf-8")

    with pytest.raises(
        Exception,
        match="differs from sealed profiles",
    ):
        scrub_parent_environment_file_v2(
            environment_path=parent_environment,
            transition_report_path=report_path,
        )


@pytest.mark.parametrize(
    "proxy_value",
    (
        "socks5://proxy.example.com:6162",
        "https://proxy.invalid",
    ),
)
def test_rejects_worker_proxy_outside_measured_connect_contract(
    tmp_path,
    proxy_value,
):
    environment = _environment()
    environment["RESEARCH_LAB_V2_SCORING_HTTPS_PROXY_1"] = proxy_value
    kms = KMS()

    with pytest.raises(
        Exception,
        match="incompatible with V2 provider transport",
    ):
        prepare_gateway_envelopes_v2(
            environment=environment,
            kms_key_id="alias/gateway-v2",
            deploy_commit="1" * 40,
            output_dir=tmp_path / "v2",
            kms_client=kms,
            proxy_fleet_probe=_skip_proxy_probe,
        )

    assert kms.requests == []
    assert not (tmp_path / "v2").exists()


def test_explicit_deferral_seals_only_validated_v2_tls_proxies(
    tmp_path,
):
    environment = _legacy_http_proxy_environment()
    environment["GATEWAY_V2_DEFER_WORKER_FLEETS"] = "all"
    output = tmp_path / "v2"

    result = prepare_gateway_envelopes_v2(
        environment=environment,
        kms_key_id="alias/gateway-v2",
        deploy_commit="1" * 40,
        output_dir=output,
        kms_client=KMS(),
        proxy_fleet_probe=_skip_proxy_probe,
    )

    assert result["deferred_worker_fleet_roles"] == [
        "gateway_autoresearch",
        "gateway_scoring",
    ]
    assert len(
        result["worker_proxy_credential_ref_hashes"][
            "gateway_autoresearch"
        ]
    ) == 2
    assert len(
        result["worker_proxy_credential_ref_hashes"]["gateway_scoring"]
    ) == 3
    assert (output / "autoresearch_proxy_00.json").is_file()
    assert (output / "scoring_proxy_02.json").is_file()
    persisted_documents = {
        path.name: path.read_text(encoding="utf-8")
        for path in output.glob("*.json")
    }
    assert not any(
        plaintext in document
        for document in persisted_documents.values()
        for plaintext in (
            "rehearsal-password",
            "user:password",
            "http://",
        )
    )

    parent_environment = tmp_path / "gateway-parent.env"
    parent_environment.write_text(
        "\n".join(
            "export %s=%s" % (name, value)
            for name, value in environment.items()
        )
        + "\n",
        encoding="utf-8",
    )
    transition = scrub_parent_environment_file_v2(
        environment_path=parent_environment,
        transition_report_path=output / "gateway-v2-env-transition.json",
    )
    scrubbed = load_environment_file(parent_environment)
    assert scrubbed["GATEWAY_V2_DEFER_WORKER_FLEETS"] == (
        "gateway_autoresearch,gateway_scoring"
    )
    assert not any(
        name.startswith("RESEARCH_LAB_AUTO_RESEARCH_WEBSHARE_PROXY")
        or name.startswith("RESEARCH_LAB_QUALIFICATION_WEBSHARE_PROXY")
        for name in scrubbed
    )
    assert transition["installed_deferred_worker_fleet_roles"] == [
        "gateway_autoresearch",
        "gateway_scoring",
    ]


def test_explicit_deferral_seals_legacy_authenticated_http_connect_proxies(
    tmp_path,
):
    environment = _legacy_http_proxy_environment()
    environment["GATEWAY_V2_DEFER_WORKER_FLEETS"] = "all"
    for name in tuple(environment):
        if name.startswith("RESEARCH_LAB_V2_AUTORESEARCH_HTTPS_PROXY_") or (
            name.startswith("RESEARCH_LAB_V2_SCORING_HTTPS_PROXY_")
        ):
            environment.pop(name)
    output = tmp_path / "v2"
    result = prepare_gateway_envelopes_v2(
        environment=environment,
        kms_key_id="alias/gateway-v2",
        deploy_commit="1" * 40,
        output_dir=output,
        kms_client=KMS(),
        proxy_fleet_probe=_skip_proxy_probe,
    )

    assert result["worker_proxy_source"] == {
        "gateway_autoresearch": "legacy",
        "gateway_scoring": "legacy",
    }
    assert (output / "autoresearch_proxy_01.json").is_file()
    assert (output / "scoring_proxy_02.json").is_file()


def test_one_deferred_role_does_not_relax_the_other_role(tmp_path):
    environment = _environment()
    environment["GATEWAY_V2_DEFER_WORKER_FLEETS"] = "gateway_autoresearch"
    environment["RESEARCH_LAB_V2_SCORING_HTTPS_PROXY_3"] = (
        "socks5://scoring-3.example.com:13433"
    )

    with pytest.raises(
        Exception,
        match="gateway_scoring worker proxy 3 from v2_tls configuration",
    ):
        prepare_gateway_envelopes_v2(
            environment=environment,
            kms_key_id="alias/gateway-v2",
            deploy_commit="1" * 40,
            output_dir=tmp_path / "v2",
            kms_client=KMS(),
            proxy_fleet_probe=_skip_proxy_probe,
        )


def test_exact_commit_reuse_is_bound_to_deferred_role_set(tmp_path):
    environment = _environment()
    destination = tmp_path / "v2"
    kms = KMS()
    first = install_gateway_envelopes_v2(
        environment=environment,
        kms_key_id="alias/gateway-v2",
        deploy_commit="1" * 40,
        install_dir=destination,
        kms_client=kms,
        proxy_fleet_probe=_skip_proxy_probe,
    )
    first_request_count = len(kms.requests)
    assert first["status"] == "installed"

    environment["GATEWAY_V2_DEFER_WORKER_FLEETS"] = "gateway_autoresearch"
    second = install_gateway_envelopes_v2(
        environment=environment,
        kms_key_id="alias/gateway-v2",
        deploy_commit="1" * 40,
        install_dir=destination,
        kms_client=kms,
        proxy_fleet_probe=_skip_proxy_probe,
    )

    assert second["status"] == "installed"
    assert second["deferred_worker_fleet_roles"] == [
        "gateway_autoresearch"
    ]
    assert len(kms.requests) > first_request_count


def test_exact_commit_reuse_revalidates_worker_proxy_contract(tmp_path):
    environment = _environment()
    destination = tmp_path / "v2"
    kms = KMS()
    installed = install_gateway_envelopes_v2(
        environment=environment,
        kms_key_id="alias/gateway-v2",
        deploy_commit="1" * 40,
        install_dir=destination,
        kms_client=kms,
        proxy_fleet_probe=_skip_proxy_probe,
    )
    request_count = len(kms.requests)
    assert installed["status"] == "installed"

    environment["RESEARCH_LAB_V2_SCORING_HTTPS_PROXY_1"] = (
        "socks5://proxy.example.com:6162"
    )
    with pytest.raises(
        Exception,
        match="incompatible with V2 provider transport",
    ):
        install_gateway_envelopes_v2(
            environment=environment,
            kms_key_id="alias/gateway-v2",
            deploy_commit="1" * 40,
            install_dir=destination,
            kms_client=kms,
            proxy_fleet_probe=_skip_proxy_probe,
        )

    assert len(kms.requests) == request_count
    assert json.loads(
        (destination / "gateway-v2-env-transition.json").read_text()
    )[
        "worker_proxy_transport_policy"
    ] == "authenticated_http_or_https_connect.v2"


def test_install_cli_checks_schema_before_writing_envelopes(
    monkeypatch,
    tmp_path,
    capsys,
):
    environment = {
        "SUPABASE_URL": "https://project.supabase.co",
        "SUPABASE_SERVICE_ROLE_KEY": "service-role-value",
    }
    env_file = tmp_path / "gateway.env"
    env_file.write_text(json.dumps(environment), encoding="utf-8")
    calls = []

    monkeypatch.setenv(
        "GATEWAY_DEPLOY_STAGE",
        "v2_credential_envelope_preparation",
    )

    def cleanup_stale_probes():
        calls.append("cleanup")
        return []

    def verify_schema(observed_environment):
        assert observed_environment == environment
        calls.append("schema")
        return {
            "status": "ready",
            "probe_count": 19,
            "migration_files": ["scripts/95.sql", "scripts/97.sql"],
        }

    def install(**kwargs):
        assert kwargs["environment"] == environment
        calls.append("install")
        return {"status": "installed"}

    monkeypatch.setattr(
        envelope_module,
        "verify_required_supabase_v2_schema",
        verify_schema,
    )
    monkeypatch.setattr(
        envelope_module,
        "cleanup_stale_vsock_probes",
        cleanup_stale_probes,
    )
    monkeypatch.setattr(envelope_module, "install_gateway_envelopes_v2", install)

    assert envelope_module.main(
        [
            "--install",
            "--env-file",
            str(env_file),
            "--kms-key-id",
            "alias/gateway-v2",
            "--deploy-commit",
            "1" * 40,
            "--output-dir",
            str(tmp_path / "v2"),
        ]
    ) == 0

    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert calls == ["cleanup", "schema", "install"]
    assert "GATEWAY_RESTART_STALE_PROBE_CLEANUP" in captured.err
    assert result["supabase_v2_schema"]["status"] == "ready"


def test_install_cli_carries_explicit_inherited_deferral_into_env_transition(
    monkeypatch,
    tmp_path,
    capsys,
):
    environment = {"SUPABASE_URL": "https://project.supabase.co"}
    env_file = tmp_path / "gateway.env"
    env_file.write_text(json.dumps(environment), encoding="utf-8")
    observed = {}

    monkeypatch.setenv("GATEWAY_V2_DEFER_WORKER_FLEETS", "all")
    monkeypatch.setattr(
        envelope_module,
        "verify_required_supabase_v2_schema",
        lambda _environment: {"status": "ready"},
    )

    def install(**kwargs):
        observed.update(kwargs["environment"])
        return {
            "status": "installed",
            "deferred_worker_fleet_roles": [
                "gateway_autoresearch",
                "gateway_scoring",
            ],
        }

    monkeypatch.setattr(
        envelope_module,
        "install_gateway_envelopes_v2",
        install,
    )

    assert envelope_module.main(
        [
            "--install",
            "--env-file",
            str(env_file),
            "--kms-key-id",
            "alias/gateway-v2",
            "--deploy-commit",
            "1" * 40,
            "--output-dir",
            str(tmp_path / "v2"),
        ]
    ) == 0
    assert observed["GATEWAY_V2_DEFER_WORKER_FLEETS"] == (
        "gateway_autoresearch,gateway_scoring"
    )
    assert json.loads(capsys.readouterr().out)[
        "deferred_worker_fleet_roles"
    ] == ["gateway_autoresearch", "gateway_scoring"]
