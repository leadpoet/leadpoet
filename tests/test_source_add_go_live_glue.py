"""Go-live glue: docker trial runner, ablation harness, probe-guard wiring."""

from __future__ import annotations

import http.client
import json
import time
from pathlib import Path
from typing import Any

import pytest

from gateway.research_lab.provider_evidence_proxy import (
    ProviderRegistryEntry,
    serve_evidence_proxy,
)
from gateway.research_lab.provider_probe import probe_guard_terms_from_icp_items
from gateway.research_lab.source_add_ablation import (
    AdapterAblationResult,
    arm_leg2_for_merge,
    disabled_registry,
    run_adapter_ablation,
)
from gateway.research_lab.source_add_trial_runner import (
    build_source_add_sandbox_runner,
    build_trial_registry_entry,
)
from research_lab.eval.provider_evidence_cache import canonical_request_fingerprint
from research_lab.source_add_execution import intake_source_add_submission, run_sandboxed_trial


def _manifest_doc(**overrides: Any) -> dict[str, Any]:
    doc = {
        "adapter_id": "adapter:glue-test-1",
        "miner_ref": "miner:hotkey-g",
        "source_name": "Glue Test Source",
        "source_kind": "news",
        "declared_base_domains": ["gluefeed.example"],
        "output_schema_ref": "schema:source-add-output:v1",
        "allowed_output_fields": ["evidence_refs", "snapshot_refs", "content_hashes", "normalized_text_hashes"],
        "submitted_artifact_ref": "artifact:glue",
        "code_bundle_hash": "sha256:" + "a" * 64,
        "sandbox_policy_ref": "policy:sandbox-v1",
        "max_trial_cost_cents": 500,
        "max_request_cost_cents": 5,
        "max_latency_ms": 30_000,
        "fixture_refs": ["fixture:glue"],
    }
    doc.update(overrides)
    return doc


def _submission(**overrides: Any):
    record, errors = intake_source_add_submission(_manifest_doc(**overrides), miner_hotkey="hk-glue")
    assert errors == []
    return record


def _output_doc(adapter_id: str, icp_ref: str) -> dict[str, Any]:
    return {
        "output_ref": f"output:{icp_ref}",
        "adapter_id": adapter_id,
        "icp_ref": icp_ref,
        "evidence_refs": [f"evidence:{icp_ref}:0"],
        "snapshot_refs": [f"snapshot:{icp_ref}"],
        "content_hashes": ["sha256:" + "b" * 64],
        "normalized_text_hashes": ["sha256:" + "c" * 64],
    }


class TestTrialRegistryEntry:
    def test_entry_derives_from_manifest(self):
        record = _submission()
        entry = build_trial_registry_entry(record)
        assert entry.base_url == "https://gluefeed.example"
        assert entry.auth_kind == "none"  # no_credentials manifest
        assert entry.id.startswith("trial_")

    def test_entry_with_credential_uses_header_auth(self):
        record = _submission(
            adapter_id="adapter:glue-cred-1",
            credential_policy="credential_ref_only",
            credential_ref="encrypted_ref:source_add:abc",
        )
        entry = build_trial_registry_entry(record)
        assert entry.auth_kind == "header"
        assert entry.auth_name == "x-api-key"


class TestSandboxRunner:
    def _run(self, tmp_path: Path, docker_exec, record=None, miner_credential=""):
        record = record or _submission()
        bundle = tmp_path / "bundle"
        bundle.mkdir()
        (bundle / "adapter.py").write_text("print('{}')\n", encoding="utf-8")
        runner, shutdown = build_source_add_sandbox_runner(
            record=record,
            bundle_dir=bundle,
            work_dir=tmp_path / "work",
            miner_credential=miner_credential,
            docker_exec=docker_exec,
        )
        try:
            return record, runner(record, "icp:1")
        finally:
            shutdown()

    def test_successful_run_parses_last_json_line(self, tmp_path):
        record = _submission()
        output = _output_doc(record.adapter_id, "icp:1")

        def _fake_docker(argv, timeout):
            assert argv[:3] == ["docker", "run", "--rm"]
            assert any("RESEARCH_LAB_EVIDENCE_PROXY_URL=" in arg for arg in argv)
            return 0, "progress line\n" + json.dumps(output) + "\n", ""

        _, result = self._run(tmp_path, _fake_docker, record=record)
        assert result["output"] == output
        assert result["cost_cents"] == 0

    def test_nonzero_exit_maps_to_error(self, tmp_path):
        _, result = self._run(tmp_path, lambda argv, timeout: (3, "", "boom"))
        assert result["error"] == "adapter_exit_3"

    def test_missing_output_json_maps_to_error(self, tmp_path):
        _, result = self._run(tmp_path, lambda argv, timeout: (0, "no json here", ""))
        assert result["error"] == "no_output_json"

    def test_timeout_maps_to_timeout_error(self, tmp_path):
        import subprocess

        def _hang(argv, timeout):
            raise subprocess.TimeoutExpired(cmd=argv, timeout=timeout)

        _, result = self._run(tmp_path, _hang)
        assert result["error"] == "timeout"

    def test_bundle_without_entrypoint_rejected(self, tmp_path):
        record = _submission()
        empty_bundle = tmp_path / "empty"
        empty_bundle.mkdir()
        with pytest.raises(ValueError, match="adapter.py"):
            build_source_add_sandbox_runner(
                record=record,
                bundle_dir=empty_bundle,
                work_dir=tmp_path / "work",
            )

    def test_trial_proxy_serves_only_the_adapter_provider_with_override_credential(self, tmp_path):
        """The container-facing guarantee: adapter's provider replays through
        the per-trial proxy with the in-memory miner key; everything else 404s."""

        record = _submission(
            adapter_id="adapter:glue-proxy-1",
            credential_policy="credential_ref_only",
            credential_ref="encrypted_ref:source_add:xyz",
        )
        entry = build_trial_registry_entry(record)
        proxy_url_holder: dict[str, str] = {}

        def _probe_proxy_docker(argv, timeout):
            proxy_env = next(arg for arg in argv if arg.startswith("RESEARCH_LAB_EVIDENCE_PROXY_URL="))
            base = proxy_env.split("=", 1)[1]  # http://127.0.0.1:PORT/<entry.id>
            host_port = base.split("://", 1)[1].split("/", 1)[0]
            host, port = host_port.split(":")
            connection = http.client.HTTPConnection(host, int(port), timeout=5)
            try:
                # The adapter's own provider route resolves (recorded below).
                connection.request("GET", f"/{entry.id}/feed?q=x")
                own = connection.getresponse()
                own_body = own.read()
                connection.close()
                connection = http.client.HTTPConnection(host, int(port), timeout=5)
                # Any other provider is unreachable from the trial.
                connection.request("GET", "/exa/search?q=x")
                other = connection.getresponse()
                other.read()
                proxy_url_holder["own_status"] = str(own.status)
                proxy_url_holder["own_body"] = own_body.decode()
                proxy_url_holder["other_status"] = str(other.status)
            finally:
                connection.close()
            return 0, json.dumps(_output_doc(record.adapter_id, "icp:1")), ""

        bundle = tmp_path / "bundle"
        bundle.mkdir()
        (bundle / "adapter.py").write_text("print('{}')\n", encoding="utf-8")
        runner, shutdown = build_source_add_sandbox_runner(
            record=record,
            bundle_dir=bundle,
            work_dir=tmp_path / "work",
            registry_entry=entry,
            miner_credential="miner-secret-key-123",
            docker_exec=_probe_proxy_docker,
        )
        try:
            # Pre-record the adapter-provider response so no live call happens.
            from gateway.research_lab import provider_evidence_proxy as proxy_module  # noqa: F401

            # Reach into the spawned proxy store via a replay: record first.
            fingerprint = canonical_request_fingerprint("GET", "https://gluefeed.example/feed?q=x", None)
            # The store is owned by the runner's proxy; simplest path: make the
            # "live" call fail closed is not needed — record via the day cache
            # is internal, so instead let the request go "live" and stub it.
            import urllib.request as _urllib_request

            class _FakeUpstream:
                status = 200
                headers: dict[str, str] = {}

                def read(self):
                    return b'{"feed": []}'

                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    return False

            seen_keys: list[str] = []
            real_urlopen = _urllib_request.urlopen

            def _routed(request, timeout=0):
                url = request.full_url if hasattr(request, "full_url") else str(request)
                if "gluefeed.example" in url:
                    seen_keys.append(request.headers.get("X-api-key", ""))
                    return _FakeUpstream()
                return real_urlopen(request, timeout=timeout)

            _urllib_request.urlopen = _routed
            try:
                result = runner(record, "icp:1")
            finally:
                _urllib_request.urlopen = real_urlopen
        finally:
            shutdown()
        assert "output" in result
        assert proxy_url_holder["own_status"] == "200"
        assert proxy_url_holder["own_body"] == '{"feed": []}'
        # Upstream saw the miner's override key (proxy memory, not env).
        assert seen_keys == ["miner-secret-key-123"]
        # Other providers do not exist inside the trial.
        assert proxy_url_holder["other_status"] == "404"


class TestFullTrialThroughRunner:
    def test_run_sandboxed_trial_with_docker_runner_and_cost_metering(self, tmp_path):
        record = _submission(adapter_id="adapter:glue-cost-1")
        bundle = tmp_path / "bundle"
        bundle.mkdir()
        (bundle / "adapter.py").write_text("print('{}')\n", encoding="utf-8")

        calls: list[str] = []

        def _fake_docker(argv, timeout):
            icp_ref = next(arg for arg in argv if arg.startswith("SOURCE_ADD_ICP_REF=")).split("=", 1)[1]
            calls.append(icp_ref)
            return 0, json.dumps(_output_doc(record.adapter_id, icp_ref)), ""

        runner, shutdown = build_source_add_sandbox_runner(
            record=record,
            bundle_dir=bundle,
            work_dir=tmp_path / "work",
            docker_exec=_fake_docker,
        )
        try:
            trial = run_sandboxed_trial(
                record,
                trial_icp_refs=("icp:1", "icp:2"),
                sandbox_runner=runner,
                evidence_classifier=lambda _ref: "news",
            )
        finally:
            shutdown()
        assert trial.failure_reason == ""
        assert trial.measured_trial_yield == 1.0
        assert calls == ["icp:1", "icp:2"]


class TestAblationHarness:
    ENTRIES = [
        ProviderRegistryEntry(id="exa", base_url="https://api.exa.ai", auth_kind="none"),
        ProviderRegistryEntry(id="gluefeed", base_url="https://gluefeed.example", auth_kind="none"),
    ]

    def test_disabled_registry_removes_only_the_adapter(self):
        remaining = disabled_registry(self.ENTRIES, "gluefeed")
        assert [entry.id for entry in remaining] == ["exa"]

    def test_disabled_registry_rejects_unknown_provider(self):
        with pytest.raises(ValueError, match="not in the eval registry"):
            disabled_registry(self.ENTRIES, "nope")

    @pytest.mark.asyncio
    async def test_ablation_passes_when_delta_reaches_threshold(self):
        async def _evaluator(candidate_ref, entries, holdout_ref):
            return 6.2 if any(entry.id == "gluefeed" for entry in entries) else 5.4

        result = await run_adapter_ablation(
            adapter_id="adapter:glue-test-1",
            registry_provider_id="gluefeed",
            candidate_ref="candidate:1",
            registry_entries=self.ENTRIES,
            evaluator=_evaluator,
            holdout_ref="holdout:lab:1",
        )
        assert result.passed
        assert result.delta_points == pytest.approx(0.8)

    @pytest.mark.asyncio
    async def test_dead_code_routing_fails_attribution(self):
        # Adapter present or absent, score identical → routing is dead code.
        async def _evaluator(candidate_ref, entries, holdout_ref):
            return 6.0

        result = await run_adapter_ablation(
            adapter_id="adapter:glue-test-1",
            registry_provider_id="gluefeed",
            candidate_ref="candidate:1",
            registry_entries=self.ENTRIES,
            evaluator=_evaluator,
        )
        assert not result.passed
        assert result.delta_points == 0.0


class TestArmLeg2:
    def _ablation(self, on: float, off: float) -> AdapterAblationResult:
        return AdapterAblationResult(
            adapter_id="adapter:glue-test-1",
            registry_provider_id="gluefeed",
            candidate_ref="candidate:1",
            holdout_ref="holdout:lab:1",
            adapter_on_score=on,
            adapter_off_score=off,
            delta_points=on - off,
            threshold_points=0.5,
            passed=(on - off) >= 0.5,
            evaluated_at="2026-08-01T00:00:00Z",
        )

    @pytest.mark.asyncio
    async def test_arms_and_returns_reward_doc(self):
        reward, blockers = await arm_leg2_for_merge(
            adapter_id="adapter:glue-test-1",
            adapter_owner_miner_ref="hk-owner",
            catalog_id="source_catalog:" + "0" * 16,
            catalog_registry_ids=("gluefeed",),
            merged=True,
            merged_diff_routed_registry_ids=("gluefeed",),
            merge_cleared_score_bar=True,
            shadow_monitor_live=True,
            shadow_window_days_elapsed=7.2,
            shadow_window_survived=True,
            ablation=self._ablation(6.2, 5.4),
            start_epoch=900,
            accepted_at="2026-07-06T00:00:00Z",
            market_open_at="2026-07-20T00:00:00Z",
            persist=False,
        )
        assert blockers == []
        assert reward is not None
        assert reward["alpha_percent"] == 5.0
        assert reward["miner_ref"] == "hk-owner"
        assert reward["trigger_evidence"]["ablation_result"]["delta_points"] == pytest.approx(0.8)

    @pytest.mark.asyncio
    async def test_missing_ablation_blocks(self):
        reward, blockers = await arm_leg2_for_merge(
            adapter_id="adapter:glue-test-1",
            adapter_owner_miner_ref="hk-owner",
            catalog_id="",
            catalog_registry_ids=("gluefeed",),
            merged=True,
            merged_diff_routed_registry_ids=("gluefeed",),
            merge_cleared_score_bar=True,
            shadow_monitor_live=True,
            shadow_window_days_elapsed=7.2,
            shadow_window_survived=True,
            ablation=None,
            start_epoch=900,
            accepted_at="2026-07-06T00:00:00Z",
            market_open_at="2026-07-20T00:00:00Z",
            persist=False,
        )
        assert reward is None
        assert "ablation_not_run" in blockers


class TestProbeGuardTermExtraction:
    def test_extracts_identifier_shaped_terms_only(self):
        items = [
            {
                "icp_description": "B2B SaaS companies in Europe hiring RevOps leaders with intent signals",
                "target_companies": [{"company_name": "Acme Robotics", "domain": "acme-robotics.example"}],
                "account_hints": {"employer": "Globex"},
            }
        ]
        terms = probe_guard_terms_from_icp_items(items)
        assert terms == {"Acme Robotics", "acme-robotics.example", "Globex"}
