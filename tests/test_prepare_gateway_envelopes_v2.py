from __future__ import annotations

import json

import pytest

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
            "CiphertextBlob": (
                "ciphertext:%03d" % len(self.requests)
            ).encode(),
        }


def _skip_proxy_probe(_fleets):
    return None


def _environment() -> dict[str, str]:
    return {
        "RESEARCH_LAB_V2_OPENROUTER_API_KEY": "openrouter-secret",
        "RESEARCH_LAB_V2_EXA_API_KEY": "exa-secret",
        "RESEARCH_LAB_V2_SCRAPINGDOG_API_KEY": "scrapingdog-secret",
        "RESEARCH_LAB_V2_DEEPLINE_API_KEY": "deepline-secret",
        "SUPABASE_SERVICE_ROLE_KEY": "supabase-secret",
        "TRUELIST_API_KEY": "truelist-secret",
        "RESEARCH_LAB_V2_AUTORESEARCH_HTTPS_PROXY_1": (
            "https://retired-autoresearch.example.com"
        ),
        "RESEARCH_LAB_V2_SCORING_HTTPS_PROXY_1": (
            "https://scoring-1.example.com"
        ),
        "RESEARCH_LAB_V2_SCORING_HTTPS_PROXY_2": (
            "https://scoring-2.example.com"
        ),
        "RESEARCH_LAB_V2_SCORING_HTTPS_PROXY_3": (
            "https://scoring-3.example.com"
        ),
    }


def _legacy_scoring_environment(count: int = 3) -> dict[str, str]:
    environment = {
        name: value
        for name, value in _environment().items()
        if "V2_SCORING_HTTPS_PROXY" not in name
    }
    environment.update(
        {
            "RESEARCH_LAB_QUALIFICATION_WEBSHARE_PROXY_%d" % index: (
                "http://user:password@1.1.1.1:%d" % (7100 + index)
            )
            for index in range(1, count + 1)
        }
    )
    return environment


def _v2_scoring_environment(count: int) -> dict[str, str]:
    environment = {
        name: value
        for name, value in _environment().items()
        if "V2_SCORING_HTTPS_PROXY" not in name
    }
    environment.update(
        {
            "RESEARCH_LAB_V2_SCORING_HTTPS_PROXY_%d" % index: (
                "https://scoring-%d.example.com" % index
            )
            for index in range(1, count + 1)
        }
    )
    return environment


def test_prepares_boot_and_scoring_envelopes_without_autoresearch_fleet(
    tmp_path,
):
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

    assert result["schema_version"] == "leadpoet.gateway_envelope_transition.v3"
    assert result["scoring_worker_count"] == 3
    assert result["worker_proxy_source"] == {
        "gateway_scoring": "v2_tls"
    }
    assert result["required_count_environment"] == {
        "RESEARCH_LAB_SCORING_WORKER_PROCESS_COUNT": "3"
    }
    assert (output / "scoring_proxy_02.json").is_file()
    assert not tuple(output.glob("autoresearch_proxy_*.json"))
    assert "hosted_worker_count" not in result
    assert "deferred_worker_fleet_roles" not in result
    documents = [
        json.loads(path.read_text())
        for path in output.glob("*.json")
        if path.name != "gateway-v2-env-transition.json"
    ]
    assert len(documents) == 13
    assert not (output / "stale_parent_openrouter.json").exists()
    assert all(validate_provider_envelope(document) for document in documents)
    assert json.loads((output / "openrouter.json").read_text())[
        "credential_ref_hash"
    ] == credential_reference_hash("openrouter-secret")
    assert json.loads((output / "benchmark_openrouter.json").read_text())[
        "credential_ref_hash"
    ] == credential_value_hash("openrouter-secret")
    assert not any(
        secret in json.dumps(documents)
        for secret in (
            "openrouter-secret",
            "scrapingdog-secret",
            "https://scoring-1.example.com",
        )
    )


def test_retired_autoresearch_and_legacy_scoring_aliases_are_scrubbed(tmp_path):
    environment = _environment()
    environment.update(
        {
            "RESEARCH_LAB_AUTO_RESEARCH_WEBSHARE_PROXY_1": (
                "http://retired:user@legacy-auto.example.com:18080"
            ),
            "RESEARCH_LAB_QUALIFICATION_WEBSHARE_PROXY_1": (
                "http://legacy:user@legacy-scoring.example.com:19090"
            ),
            "GATEWAY_V2_DEFER_WORKER_FLEETS": "all",
            "RESEARCH_LAB_HOSTED_WORKER_PROCESS_COUNT": "10",
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
        "RESEARCH_LAB_V2_AUTORESEARCH_HTTPS_PROXY_1",
        "RESEARCH_LAB_AUTO_RESEARCH_WEBSHARE_PROXY_1",
        "RESEARCH_LAB_V2_SCORING_HTTPS_PROXY_1",
        "RESEARCH_LAB_QUALIFICATION_WEBSHARE_PROXY_1",
        "GATEWAY_V2_DEFER_WORKER_FLEETS",
        "RESEARCH_LAB_HOSTED_WORKER_PROCESS_COUNT",
    } <= removal_names


def test_legacy_scoring_fleet_is_sealed_and_probed(tmp_path):
    environment = _legacy_scoring_environment()
    observed = []

    result = prepare_gateway_envelopes_v2(
        environment=environment,
        kms_key_id="alias/gateway-v2",
        deploy_commit="1" * 40,
        output_dir=tmp_path / "v2",
        kms_client=KMS(),
        proxy_fleet_probe=lambda fleets: observed.append(fleets),
    )

    assert result["worker_proxy_source"] == {"gateway_scoring": "legacy"}
    assert result["scoring_worker_count"] == 3
    assert observed == [
        {
            "gateway_scoring": tuple(
                environment[
                    "RESEARCH_LAB_QUALIFICATION_WEBSHARE_PROXY_%d" % index
                ]
                for index in range(1, 4)
            )
        }
    ]


def test_scoring_capacity_follows_verified_profiles(tmp_path):
    environment = _legacy_scoring_environment(25)
    failed = environment["RESEARCH_LAB_QUALIFICATION_WEBSHARE_PROXY_3"]

    def verified_fleets(fleets):
        return {
            "gateway_scoring": tuple(
                value for value in fleets["gateway_scoring"] if value != failed
            )
        }

    result = prepare_gateway_envelopes_v2(
        environment=environment,
        kms_key_id="alias/gateway-v2",
        deploy_commit="1" * 40,
        output_dir=tmp_path / "v2",
        kms_client=KMS(),
        proxy_fleet_probe=verified_fleets,
    )

    assert result["scoring_worker_count"] == 24
    assert result["worker_proxy_profile_counts"] == {
        "gateway_scoring": {
            "configured": 25,
            "verified": 24,
            "quarantined": 1,
            "sealed_worker_slots": 24,
        }
    }
    assert len(tuple((tmp_path / "v2").glob("scoring_proxy_*.json"))) == 24


@pytest.mark.parametrize("proxy_count", (1, 7, 15))
def test_scoring_capacity_follows_distinct_configured_proxies(
    tmp_path,
    proxy_count,
):
    environment = _v2_scoring_environment(proxy_count)
    environment["RESEARCH_LAB_SCORING_WORKER_PROCESS_COUNT"] = "10"

    result = prepare_gateway_envelopes_v2(
        environment=environment,
        kms_key_id="alias/gateway-v2",
        deploy_commit="1" * 40,
        output_dir=tmp_path / "v2",
        kms_client=KMS(),
        proxy_fleet_probe=_skip_proxy_probe,
    )

    assert result["scoring_worker_count"] == proxy_count
    assert result["required_count_environment"] == {
        "RESEARCH_LAB_SCORING_WORKER_PROCESS_COUNT": str(proxy_count)
    }
    assert len(
        tuple((tmp_path / "v2").glob("scoring_proxy_*.json"))
    ) == proxy_count


@pytest.mark.parametrize("stale_override", ("15", "malformed"))
def test_v2_precedence_ignores_stale_worker_override(
    tmp_path,
    stale_override,
):
    environment = _legacy_scoring_environment(25)
    environment[
        "RESEARCH_LAB_V2_SCORING_HTTPS_PROXY_1"
    ] = "https://scoring-v2.example.com:443"
    environment["RESEARCH_LAB_SCORING_WORKER_PROCESS_COUNT"] = stale_override

    result = prepare_gateway_envelopes_v2(
        environment=environment,
        kms_key_id="alias/gateway-v2",
        deploy_commit="1" * 40,
        output_dir=tmp_path / "v2",
        kms_client=KMS(),
        proxy_fleet_probe=_skip_proxy_probe,
    )

    assert result["worker_proxy_source"] == {"gateway_scoring": "v2_tls"}
    assert result["scoring_worker_count"] == 1
    assert len(tuple((tmp_path / "v2").glob("scoring_proxy_*.json"))) == 1


def test_duplicate_and_missing_indexed_proxy_slots_are_compacted(tmp_path):
    environment = _v2_scoring_environment(1)
    first = environment.pop("RESEARCH_LAB_V2_SCORING_HTTPS_PROXY_1")
    environment["RESEARCH_LAB_V2_SCORING_HTTPS_PROXY_2"] = first
    environment["RESEARCH_LAB_V2_SCORING_HTTPS_PROXY_7"] = first
    environment[
        "RESEARCH_LAB_V2_SCORING_HTTPS_PROXY_15"
    ] = "https://scoring-15.example.com"

    result = prepare_gateway_envelopes_v2(
        environment=environment,
        kms_key_id="alias/gateway-v2",
        deploy_commit="1" * 40,
        output_dir=tmp_path / "v2",
        kms_client=KMS(),
        proxy_fleet_probe=_skip_proxy_probe,
    )

    assert result["scoring_worker_count"] == 2
    assert result["worker_proxy_profile_counts"]["gateway_scoring"] == {
        "configured": 2,
        "verified": 2,
        "quarantined": 0,
        "sealed_worker_slots": 2,
    }
    assert sorted(
        path.name for path in (tmp_path / "v2").glob("scoring_proxy_*.json")
    ) == ["scoring_proxy_00.json", "scoring_proxy_01.json"]


def test_missing_scoring_proxy_fails_closed_before_kms(tmp_path):
    environment = {
        name: value
        for name, value in _environment().items()
        if "V2_SCORING_HTTPS_PROXY" not in name
    }
    kms = KMS()

    with pytest.raises(Exception, match="scoring proxy values are required"):
        prepare_gateway_envelopes_v2(
            environment=environment,
            kms_key_id="alias/gateway-v2",
            deploy_commit="1" * 40,
            output_dir=tmp_path / "v2",
            kms_client=kms,
            proxy_fleet_probe=_skip_proxy_probe,
        )

    assert kms.requests == []


def test_scoring_proxy_capacity_above_bound_fails_before_kms(tmp_path):
    environment = _v2_scoring_environment(500)
    environment[
        "RESEARCH_LAB_V2_SCORING_HTTPS_PROXY"
    ] = "https://scoring-unindexed.example.com"
    kms = KMS()

    with pytest.raises(Exception, match="verified scoring worker capacity is invalid"):
        prepare_gateway_envelopes_v2(
            environment=environment,
            kms_key_id="alias/gateway-v2",
            deploy_commit="1" * 40,
            output_dir=tmp_path / "v2",
            kms_client=kms,
            proxy_fleet_probe=_skip_proxy_probe,
        )

    assert kms.requests == []


def test_invalid_or_unverified_scoring_proxy_fails_before_kms(tmp_path):
    environment = _environment()
    environment[
        "RESEARCH_LAB_V2_SCORING_HTTPS_PROXY_1"
    ] = "socks5://proxy.example.com:6162"
    kms = KMS()
    with pytest.raises(Exception, match="incompatible with V2 provider transport"):
        prepare_gateway_envelopes_v2(
            environment=environment,
            kms_key_id="alias/gateway-v2",
            deploy_commit="1" * 40,
            output_dir=tmp_path / "v2",
            kms_client=kms,
            proxy_fleet_probe=_skip_proxy_probe,
        )
    assert kms.requests == []

    environment = _environment()
    with pytest.raises(Exception, match="invalid fleet selection"):
        prepare_gateway_envelopes_v2(
            environment=environment,
            kms_key_id="alias/gateway-v2",
            deploy_commit="1" * 40,
            output_dir=tmp_path / "another-v2",
            kms_client=kms,
            proxy_fleet_probe=lambda _fleets: {
                "gateway_scoring": ("https://unknown.example.com",)
            },
        )
    assert kms.requests == []


def test_tls_connect_failure_fails_before_kms(tmp_path):
    kms = KMS()

    def fail_probe(_fleets):
        raise WorkerProxyTransportPreflightV2Error(
            "gateway_scoring worker proxy failed V2 TLS CONNECT preflight"
        )

    with pytest.raises(Exception, match="failed V2 TLS CONNECT preflight"):
        prepare_gateway_envelopes_v2(
            environment=_environment(),
            kms_key_id="alias/gateway-v2",
            deploy_commit="1" * 40,
            output_dir=tmp_path / "v2",
            kms_client=kms,
            proxy_fleet_probe=fail_probe,
        )
    assert kms.requests == []


def test_install_reuses_exact_scoring_configuration(tmp_path):
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
    request_count = len(kms.requests)

    second = install_gateway_envelopes_v2(
        environment=_environment(),
        kms_key_id="alias/gateway-v2",
        deploy_commit="1" * 40,
        install_dir=destination,
        kms_client=kms,
        proxy_fleet_probe=_skip_proxy_probe,
    )

    assert first["status"] == "installed"
    assert second["status"] == "reused"
    assert len(kms.requests) == request_count


def test_install_preserves_artifact_master_key_across_release(tmp_path):
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
    second = install_gateway_envelopes_v2(
        environment=_environment(),
        kms_key_id="alias/gateway-v2",
        deploy_commit="2" * 40,
        install_dir=destination,
        kms_client=kms,
        proxy_fleet_probe=_skip_proxy_probe,
    )

    assert first["artifact_master_key_ref_hash"] == second[
        "artifact_master_key_ref_hash"
    ]
    assert json.loads(
        (destination / "artifact_master_key.json").read_text()
    ) == first_envelope


def test_install_removes_stale_proxy_slots_when_fleet_shrinks(tmp_path):
    destination = tmp_path / "v2"
    kms = KMS()
    first = install_gateway_envelopes_v2(
        environment=_v2_scoring_environment(7),
        kms_key_id="alias/gateway-v2",
        deploy_commit="1" * 40,
        install_dir=destination,
        kms_client=kms,
        proxy_fleet_probe=_skip_proxy_probe,
    )
    second = install_gateway_envelopes_v2(
        environment=_v2_scoring_environment(3),
        kms_key_id="alias/gateway-v2",
        deploy_commit="2" * 40,
        install_dir=destination,
        kms_client=kms,
        proxy_fleet_probe=_skip_proxy_probe,
    )

    assert first["scoring_worker_count"] == 7
    assert second["scoring_worker_count"] == 3
    assert sorted(
        path.name for path in destination.glob("scoring_proxy_*.json")
    ) == [
        "scoring_proxy_00.json",
        "scoring_proxy_01.json",
        "scoring_proxy_02.json",
    ]
    report = json.loads(
        (destination / "gateway-v2-env-transition.json").read_text()
    )
    assert report["required_count_environment"] == {
        "RESEARCH_LAB_SCORING_WORKER_PROCESS_COUNT": "3"
    }


@pytest.mark.parametrize("kind", ("invalid", "symlink"))
def test_install_fails_closed_on_invalid_existing_artifact_key(tmp_path, kind):
    destination = tmp_path / "v2"
    destination.mkdir()
    artifact_path = destination / "artifact_master_key.json"
    if kind == "invalid":
        artifact_path.write_text("{}")
    else:
        external = tmp_path / "external-key.json"
        external.write_text("{}")
        artifact_path.symlink_to(external)

    with pytest.raises(Exception, match="artifact master key envelope"):
        install_gateway_envelopes_v2(
            environment=_environment(),
            kms_key_id="alias/gateway-v2",
            deploy_commit="1" * 40,
            install_dir=destination,
            kms_client=KMS(),
            proxy_fleet_probe=_skip_proxy_probe,
        )


def test_transition_scrubs_secrets_and_installs_only_scoring_count(tmp_path):
    environment = _environment()
    environment.update(
        {
            "FULFILLMENT_OPENROUTER_API_KEY": "openrouter-secret",
            "UNRELATED_RUNTIME_VALUE": "keep-me",
        }
    )
    output = tmp_path / "v2"
    result = prepare_gateway_envelopes_v2(
        environment=environment,
        kms_key_id="alias/gateway-v2",
        deploy_commit="1" * 40,
        output_dir=output,
        kms_client=KMS(),
        proxy_fleet_probe=_skip_proxy_probe,
    )
    parent_environment = tmp_path / "gateway-parent.env"
    parent_environment.write_text(
        "\n".join(
            "export %s=%s" % (name, value)
            for name, value in environment.items()
        )
        + "\n"
    )

    transition = scrub_parent_environment_file_v2(
        environment_path=parent_environment,
        transition_report_path=output / "gateway-v2-env-transition.json",
    )
    scrubbed = load_environment_file(parent_environment)

    assert "FULFILLMENT_OPENROUTER_API_KEY" not in scrubbed
    assert "RESEARCH_LAB_V2_AUTORESEARCH_HTTPS_PROXY_1" not in scrubbed
    assert scrubbed["SUPABASE_SERVICE_ROLE_KEY"] == "supabase-secret"
    assert scrubbed["TRUELIST_API_KEY"] == "truelist-secret"
    assert scrubbed["UNRELATED_RUNTIME_VALUE"] == "keep-me"
    assert transition["installed_count_environment"] == {
        "RESEARCH_LAB_SCORING_WORKER_PROCESS_COUNT": "3"
    }
    assert result["plaintext_credential_ref_hashes_to_remove"]


def test_transition_rejects_scoring_count_not_bound_to_profiles(tmp_path):
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
    report = json.loads(report_path.read_text())
    report["required_count_environment"][
        "RESEARCH_LAB_SCORING_WORKER_PROCESS_COUNT"
    ] = "99"
    report_path.write_text(json.dumps(report))
    parent_environment = tmp_path / "gateway-parent.env"
    parent_environment.write_text("export KEEP_ME=true\n")

    with pytest.raises(Exception, match="differs from sealed profiles"):
        scrub_parent_environment_file_v2(
            environment_path=parent_environment,
            transition_report_path=report_path,
        )


def test_install_cli_checks_schema_before_writing_envelopes(
    monkeypatch,
    tmp_path,
    capsys,
):
    environment = {"SUPABASE_URL": "https://project.supabase.co"}
    env_file = tmp_path / "gateway.env"
    env_file.write_text(json.dumps(environment))
    calls = []
    monkeypatch.setenv(
        "GATEWAY_DEPLOY_STAGE", "v2_credential_envelope_preparation"
    )
    monkeypatch.setattr(
        envelope_module,
        "cleanup_stale_vsock_probes",
        lambda: calls.append("cleanup") or [],
    )
    def verify_schema(observed, *, defer_incentive_retirement):
        assert observed == environment
        assert defer_incentive_retirement is False
        calls.append("schema")
        return {"status": "ready"}

    monkeypatch.setattr(
        envelope_module,
        "verify_required_supabase_v2_schema",
        verify_schema,
    )
    monkeypatch.setattr(
        envelope_module,
        "install_gateway_envelopes_v2",
        lambda **_kwargs: calls.append("install") or {"status": "installed"},
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

    assert calls == ["cleanup", "schema", "install"]
    assert json.loads(capsys.readouterr().out)["supabase_v2_schema"] == {
        "status": "ready"
    }


def test_install_cli_does_not_inherit_retired_worker_deferral(
    monkeypatch,
    tmp_path,
):
    environment = {"SUPABASE_URL": "https://project.supabase.co"}
    env_file = tmp_path / "gateway.env"
    env_file.write_text(json.dumps(environment))
    observed = {}
    monkeypatch.setenv("GATEWAY_V2_DEFER_WORKER_FLEETS", "all")
    def verify_schema(_environment, *, defer_incentive_retirement):
        assert defer_incentive_retirement is False
        return {"status": "ready"}

    monkeypatch.setattr(
        envelope_module,
        "verify_required_supabase_v2_schema",
        verify_schema,
    )
    monkeypatch.setattr(
        envelope_module,
        "install_gateway_envelopes_v2",
        lambda **kwargs: observed.update(kwargs["environment"])
        or {"status": "installed"},
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
    assert "GATEWAY_V2_DEFER_WORKER_FLEETS" not in observed
