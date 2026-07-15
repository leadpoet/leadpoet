"""Drop-in Research Lab model runner backed by the measured scoring EIF.

The parent may extract bytes from an immutable compatibility image, but the
scoring enclave independently reconstructs the source tree and requires it to
match the signed model artifact before executing it in a fresh runsc sandbox.
No parent credential or host filesystem path crosses the authority boundary.
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
import threading
from typing import Any, Mapping, Sequence

from gateway.research_lab.attested_scoring_v2 import execute_scoring_v2
from gateway.research_lab.code_build import _extract_parent_image_source
from gateway.research_lab.v2_authority import load_source_add_catalog_snapshot_v2
from gateway.research_lab.tee_protocol import legacy_v1_enabled
from gateway.tee.model_sandbox_v2 import (
    MODEL_SANDBOX_REQUEST_SCHEMA_VERSION,
    provider_evidence_tape_input_root,
)
from gateway.tee.scoring_executor_v2 import OP_RUN_MODEL_SANDBOX_V2
from gateway.tee.source_add_runtime_v2 import (
    build_source_add_job_envelope_v2,
    build_source_add_runtime_catalog_v2,
    source_add_runtime_credential_refs_v2,
    validate_source_add_runtime_catalog_v2,
)
from gateway.tee.source_bundle_v2 import build_source_bundle_v2
from leadpoet_canonical.attested_v2 import canonical_json, sha256_json
from research_lab.eval import (
    DockerPrivateModelSpec,
    PrivateModelArtifactManifest,
    PrivateModelRuntimeError,
    ensure_private_model_outputs,
    validate_private_model_artifact_manifest,
)
from research_lab.eval.private_runtime import (
    DockerPrivateModelRunner,
    PROVIDER_COST_EVALUATION_SCOPE_ENV,
    _redacted_context,
    canonicalize_private_model_icp,
    publish_incontainer_trace_entries,
)
from research_lab.eval.provider_costs import summarize_provider_cost_trace_entries
from research_lab.eval.provider_evidence_cache import icp_evidence_cache_key


_SOURCE_BUNDLE_CACHE_SIZE = 8
_SOURCE_BUNDLE_CACHE: "OrderedDict[tuple[str, str], dict[str, Any]]" = OrderedDict()
_SOURCE_BUNDLE_CACHE_LOCK = threading.Lock()
_SOURCE_BUNDLE_BUILD_LOCKS: dict[tuple[str, str], threading.Lock] = {}
V2_PROVIDER_PROFILE_ENV = "LEADPOET_V2_PROVIDER_CREDENTIAL_PROFILE"
PROVIDER_EVIDENCE_TAPE_ARTIFACT_KIND = "provider_evidence_tape_v2"
_CREDENTIAL_ENV_NAMES = frozenset(
    {
        "DEEPLINE_API_KEY",
        "EXA_API_KEY",
        "OPENROUTER_API_KEY",
        "OPENROUTER_KEY",
        "QUALIFICATION_OPENROUTER_API_KEY",
        "QUALIFICATION_SCRAPINGDOG_API_KEY",
        "SCRAPINGDOG_API_KEY",
    }
)
_HOST_ONLY_ENV_NAMES = frozenset(
    {
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "NO_PROXY",
        "RESEARCH_LAB_PROVIDER_EVIDENCE_CACHE_DIR",
        "RESEARCH_LAB_PROVIDER_EVIDENCE_CACHE_PATH",
        "RESEARCH_LAB_SCORING_CACHE_DIR",
        "http_proxy",
        "https_proxy",
        "no_proxy",
        V2_PROVIDER_PROFILE_ENV,
    }
)


class AttestedPrivateModelRunnerV2Error(PrivateModelRuntimeError):
    """The measured model result or one of its commitments is invalid."""


class _LegacyPrivateModelRunnerAdapter:
    """Current-commit host runner with the V2 runner's public interface."""

    def __init__(
        self,
        *,
        artifact: PrivateModelArtifactManifest | Mapping[str, Any],
        spec: DockerPrivateModelSpec | Mapping[str, Any],
        model_kind: str,
        worker_index: int,
        epoch_id: int | None = None,
        parent_graphs: Sequence[Mapping[str, Any]] = (),
        **_kwargs: Any,
    ) -> None:
        self.artifact = (
            artifact
            if isinstance(artifact, PrivateModelArtifactManifest)
            else PrivateModelArtifactManifest.from_mapping(artifact)
        )
        errors = validate_private_model_artifact_manifest(self.artifact)
        if errors:
            raise AttestedPrivateModelRunnerV2Error(
                "model artifact is invalid: " + "; ".join(errors)
            )
        self.spec = (
            spec
            if isinstance(spec, DockerPrivateModelSpec)
            else DockerPrivateModelSpec.from_mapping(spec)
        )
        if self.spec.image_digest != self.artifact.image_digest:
            raise AttestedPrivateModelRunnerV2Error(
                "legacy model runner image differs from the signed artifact"
            )
        if model_kind not in {"private", "candidate"}:
            raise AttestedPrivateModelRunnerV2Error(
                "legacy model runner kind is invalid"
            )
        self.model_kind = model_kind
        self.worker_index = int(worker_index)
        self.epoch_id = int(epoch_id) if epoch_id is not None else None
        self.parent_graphs = tuple(dict(item) for item in parent_graphs)
        self._runner = DockerPrivateModelRunner(self.spec)

    def with_spec(self, spec: DockerPrivateModelSpec) -> "_LegacyPrivateModelRunnerAdapter":
        return _LegacyPrivateModelRunnerAdapter(
            artifact=self.artifact,
            spec=spec,
            model_kind=self.model_kind,
            worker_index=self.worker_index,
            epoch_id=self.epoch_id,
            parent_graphs=self.parent_graphs,
        )

    def attested_receipts(self) -> list[dict[str, Any]]:
        return []

    def attested_authorities(self) -> list[dict[str, Any]]:
        return []

    async def __call__(
        self,
        icp: Mapping[str, Any],
        context: Mapping[str, Any],
    ) -> list[Mapping[str, Any]]:
        return list(await asyncio.to_thread(self._runner, icp, context))

    def metadata(self) -> Mapping[str, Any]:
        return self._runner.metadata()


def _source_bundle_for_artifact(
    artifact: PrivateModelArtifactManifest,
    *,
    timeout_seconds: int,
) -> dict[str, Any]:
    cache_key = (artifact.model_artifact_hash, artifact.image_digest)
    with _SOURCE_BUNDLE_CACHE_LOCK:
        cached = _SOURCE_BUNDLE_CACHE.get(cache_key)
        if cached is not None:
            _SOURCE_BUNDLE_CACHE.move_to_end(cache_key)
            return dict(cached)
        build_lock = _SOURCE_BUNDLE_BUILD_LOCKS.setdefault(
            cache_key,
            threading.Lock(),
        )
    with build_lock:
        with _SOURCE_BUNDLE_CACHE_LOCK:
            cached = _SOURCE_BUNDLE_CACHE.get(cache_key)
            if cached is not None:
                _SOURCE_BUNDLE_CACHE.move_to_end(cache_key)
                return dict(cached)
        with tempfile.TemporaryDirectory(prefix="research-lab-model-v2-source-") as tmp:
            source_root = Path(tmp) / "app"
            observed_tree_hash, _paths = _extract_parent_image_source(
                image_digest=artifact.image_digest,
                source_dir=source_root,
                timeout_seconds=max(120, int(timeout_seconds)),
            )
            if observed_tree_hash != artifact.model_artifact_hash:
                raise AttestedPrivateModelRunnerV2Error(
                    "immutable model image source differs from its signed artifact"
                )
            bundle = build_source_bundle_v2(source_root)
        if bundle.get("source_tree_hash") != artifact.model_artifact_hash:
            raise AttestedPrivateModelRunnerV2Error(
                "model source bundle differs from its signed artifact"
            )
        with _SOURCE_BUNDLE_CACHE_LOCK:
            _SOURCE_BUNDLE_CACHE[cache_key] = dict(bundle)
            _SOURCE_BUNDLE_CACHE.move_to_end(cache_key)
            while len(_SOURCE_BUNDLE_CACHE) > _SOURCE_BUNDLE_CACHE_SIZE:
                _SOURCE_BUNDLE_CACHE.popitem(last=False)
            _SOURCE_BUNDLE_BUILD_LOCKS.pop(cache_key, None)
        return dict(bundle)


async def source_bundle_for_artifact_v2(
    artifact: PrivateModelArtifactManifest,
    *,
    timeout_seconds: int,
) -> dict[str, Any]:
    return await asyncio.to_thread(
        _source_bundle_for_artifact,
        artifact,
        timeout_seconds=timeout_seconds,
    )


def _provider_evidence_cache(
    spec: DockerPrivateModelSpec,
    *,
    canonical_icp: Mapping[str, Any] | None,
) -> dict[str, Any]:
    extra_env = dict(spec.extra_env or {})
    cache_path = ""
    cache_dir = str(
        extra_env.get("RESEARCH_LAB_PROVIDER_EVIDENCE_CACHE_DIR") or ""
    ).strip()
    if cache_dir and canonical_icp is not None:
        cache_path = str(
            Path(cache_dir) / (icp_evidence_cache_key(canonical_icp) + ".json")
        )
    if not cache_path:
        cache_path = str(
            extra_env.get("RESEARCH_LAB_PROVIDER_EVIDENCE_CACHE_PATH") or ""
        ).strip()
    if not cache_path or not Path(cache_path).is_file():
        return {}
    try:
        document = json.loads(Path(cache_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AttestedPrivateModelRunnerV2Error(
            "provider evidence cache is unreadable"
        ) from exc
    if not isinstance(document, Mapping):
        raise AttestedPrivateModelRunnerV2Error(
            "provider evidence cache is not an object"
        )
    return dict(document)


def _provider_evidence_cache_ref(
    canonical_icp: Mapping[str, Any] | None,
) -> str:
    return icp_evidence_cache_key(canonical_icp) if canonical_icp is not None else ""


def _require_tape_receipt(
    graph: Mapping[str, Any],
    *,
    cache_ref: str,
    cache_hash: str,
) -> Mapping[str, Any]:
    expected_input_root = provider_evidence_tape_input_root(cache_ref, cache_hash)
    matches = [
        item
        for item in graph.get("receipts") or ()
        if isinstance(item, Mapping)
        and item.get("role") == "gateway_scoring"
        and item.get("purpose") == "research_lab.provider_evidence_tape.v2"
        and item.get("status") == "succeeded"
        and item.get("input_root") == expected_input_root
        and item.get("output_root") == cache_hash
    ]
    if len(matches) != 1:
        raise AttestedPrivateModelRunnerV2Error(
            "provider evidence cache has no unique measured tape receipt"
        )
    return matches[0]


async def _load_provider_evidence_tape_graph(
    *,
    cache_ref: str,
    cache_hash: str,
) -> dict[str, Any]:
    from gateway.research_lab.attested_v2_store import (
        load_business_artifact_graph_v2,
    )

    graph = await load_business_artifact_graph_v2(
        artifact_kind=PROVIDER_EVIDENCE_TAPE_ARTIFACT_KIND,
        artifact_ref=cache_ref,
        artifact_hash=cache_hash,
    )
    _require_tape_receipt(graph, cache_ref=cache_ref, cache_hash=cache_hash)
    return dict(graph)


async def _persist_provider_evidence_tape_link(
    *,
    receipt_hash: str,
    cache_ref: str,
    cache_hash: str,
) -> dict[str, Any]:
    from gateway.research_lab.attested_v2_store import (
        persist_business_artifact_links_v2,
    )

    return await persist_business_artifact_links_v2(
        receipt_hash=receipt_hash,
        artifacts=(
            {
                "artifact_kind": PROVIDER_EVIDENCE_TAPE_ARTIFACT_KIND,
                "artifact_ref": cache_ref,
                "artifact_hash": cache_hash,
            },
        ),
    )


def _write_provider_evidence_cache(
    *,
    cache_ref: str,
    cache_document: Mapping[str, Any],
) -> str:
    cache_dir = str(
        os.getenv("RESEARCH_LAB_PROVIDER_EVIDENCE_CACHE_DIR") or ""
    ).strip()
    if not cache_dir:
        return ""
    destination_dir = Path(cache_dir)
    if not destination_dir.is_dir():
        raise AttestedPrivateModelRunnerV2Error(
            "provider evidence cache directory is unavailable"
        )
    destination = destination_dir / (cache_ref + ".json")
    encoded = canonical_json(dict(cache_document)).encode("utf-8")
    with tempfile.NamedTemporaryFile(
        mode="wb",
        prefix=destination.name + ".tmp.",
        dir=str(destination_dir),
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        temporary.chmod(0o600)
        os.replace(temporary, destination)
        directory_fd = os.open(str(destination_dir), os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    return str(destination)


def _measured_environment(
    spec: DockerPrivateModelSpec,
    *,
    additional_credential_env_names: Sequence[str] = (),
) -> dict[str, str]:
    credential_env_names = _CREDENTIAL_ENV_NAMES | frozenset(
        str(item) for item in additional_credential_env_names
    )
    environment = {}
    for name, value in dict(spec.extra_env or {}).items():
        normalized_name = str(name)
        if normalized_name in credential_env_names or normalized_name in _HOST_ONLY_ENV_NAMES:
            continue
        environment[normalized_name] = str(value)
    for name in spec.env_passthrough:
        if name in credential_env_names or name in _HOST_ONLY_ENV_NAMES:
            continue
        # The legacy runner only forwards names present in the process env.
        import os

        if name in os.environ:
            environment[str(name)] = str(os.environ[name])
    return environment


class AttestedPrivateModelRunnerV2:
    """The existing model-runner interface with V2 enclave authority."""

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        if cls is AttestedPrivateModelRunnerV2 and legacy_v1_enabled():
            return _LegacyPrivateModelRunnerAdapter(*args, **kwargs)
        return super().__new__(cls)

    def __init__(
        self,
        *,
        artifact: PrivateModelArtifactManifest | Mapping[str, Any],
        spec: DockerPrivateModelSpec | Mapping[str, Any],
        model_kind: str,
        worker_index: int,
        epoch_id: int | None = None,
        parent_graphs: Sequence[Mapping[str, Any]] = (),
        execute: Any = execute_scoring_v2,
        catalog_snapshot_loader: Any = None,
        _shared_state: dict[str, Any] | None = None,
    ) -> None:
        self.artifact = (
            artifact
            if isinstance(artifact, PrivateModelArtifactManifest)
            else PrivateModelArtifactManifest.from_mapping(artifact)
        )
        errors = validate_private_model_artifact_manifest(self.artifact)
        if errors:
            raise AttestedPrivateModelRunnerV2Error(
                "model artifact is invalid: " + "; ".join(errors)
            )
        self.spec = (
            spec
            if isinstance(spec, DockerPrivateModelSpec)
            else DockerPrivateModelSpec.from_mapping(spec)
        )
        if self.spec.image_digest != self.artifact.image_digest:
            raise AttestedPrivateModelRunnerV2Error(
                "model runner image differs from the signed artifact"
            )
        if model_kind not in {"private", "candidate"}:
            raise AttestedPrivateModelRunnerV2Error("model runner kind is invalid")
        self.model_kind = model_kind
        self.worker_index = int(worker_index)
        self.epoch_id = int(epoch_id) if epoch_id is not None else None
        if self.epoch_id is not None and self.epoch_id < 0:
            raise AttestedPrivateModelRunnerV2Error("model authority epoch is invalid")
        self.parent_graphs = tuple(dict(item) for item in parent_graphs)
        self._execute = execute
        self._catalog_snapshot_loader = catalog_snapshot_loader
        self._shared_state = _shared_state or {
            "sequence": 0,
            "receipts": [],
            "authorities": [],
            "generated_caches": {},
            "evidence_summaries": {},
            "lock": threading.Lock(),
        }

    def with_spec(self, spec: DockerPrivateModelSpec) -> "AttestedPrivateModelRunnerV2":
        return AttestedPrivateModelRunnerV2(
            artifact=self.artifact,
            spec=spec,
            model_kind=self.model_kind,
            worker_index=self.worker_index,
            epoch_id=self.epoch_id,
            parent_graphs=self.parent_graphs,
            execute=self._execute,
            catalog_snapshot_loader=self._catalog_snapshot_loader,
            _shared_state=self._shared_state,
        )

    def attested_receipts(self) -> list[dict[str, Any]]:
        with self._shared_state["lock"]:
            return [dict(item) for item in self._shared_state["receipts"]]

    def attested_authorities(self) -> list[dict[str, Any]]:
        with self._shared_state["lock"]:
            return [dict(item) for item in self._shared_state["authorities"]]

    async def __call__(
        self,
        icp: Mapping[str, Any],
        context: Mapping[str, Any],
    ) -> list[Mapping[str, Any]]:
        canonical_icp = canonicalize_private_model_icp(icp)
        cache_ref = _provider_evidence_cache_ref(canonical_icp)
        cache_document = _provider_evidence_cache(
            self.spec,
            canonical_icp=canonical_icp,
        )
        cache_parent_graphs = ()
        if cache_document:
            cache_parent_graphs = (
                await _load_provider_evidence_tape_graph(
                    cache_ref=cache_ref,
                    cache_hash=sha256_json(cache_document),
                ),
            )
        run_mode = str(dict(context or {}).get("mode") or "")
        evidence_mode = (
            "record"
            if self.model_kind == "private" and run_mode == "private_baseline"
            else "cache_live"
            if cache_document
            else "live"
        )
        result = await self._execute_operation(
            operation="run_icp",
            input_doc={
                "icp": canonical_icp,
                "context": _redacted_context(context),
            },
            provider_evidence_cache=cache_document,
            provider_evidence_cache_ref=cache_ref,
            provider_evidence_mode=evidence_mode,
            provider_snapshot_bundle={},
            provider_snapshot_tree_hash="",
            provider_snapshot_manifest_hash="",
            provider_cost_scope_override="",
            provider_cost_cap_microusd=0,
            provider_call_cap=0,
            publish_provider_evidence_cache=True,
            additional_parent_graphs=cache_parent_graphs,
        )
        return list(
            ensure_private_model_outputs(
                result,
                context_label="V2 measured private model",
                require_non_empty=False,
            )
        )

    async def run_with_provider_evidence(
        self,
        icp: Mapping[str, Any],
        context: Mapping[str, Any],
        *,
        provider_evidence_cache: Mapping[str, Any],
        provider_evidence_mode: str,
        cache_parent_graphs: Sequence[Mapping[str, Any]] = (),
        provider_snapshot_bundle: Mapping[str, Any] | None = None,
        provider_snapshot_tree_hash: str = "",
        provider_snapshot_manifest_hash: str = "",
        provider_cost_scope: str = "",
        provider_cost_cap_microusd: int = 0,
        provider_call_cap: int = 0,
    ) -> list[Mapping[str, Any]]:
        """Run one ICP under an explicitly committed tree-evaluation tape mode."""

        canonical_icp = canonicalize_private_model_icp(icp)
        result = await self._execute_operation(
            operation="run_icp",
            input_doc={
                "icp": canonical_icp,
                "context": _redacted_context(context),
            },
            provider_evidence_cache=dict(provider_evidence_cache),
            provider_evidence_cache_ref=_provider_evidence_cache_ref(canonical_icp),
            provider_evidence_mode=str(provider_evidence_mode),
            provider_snapshot_bundle=dict(provider_snapshot_bundle or {}),
            provider_snapshot_tree_hash=str(provider_snapshot_tree_hash or ""),
            provider_snapshot_manifest_hash=str(
                provider_snapshot_manifest_hash or ""
            ),
            provider_cost_scope_override=str(provider_cost_scope or ""),
            provider_cost_cap_microusd=int(provider_cost_cap_microusd),
            provider_call_cap=int(provider_call_cap),
            publish_provider_evidence_cache=False,
            additional_parent_graphs=cache_parent_graphs,
        )
        return list(
            ensure_private_model_outputs(
                result,
                context_label="V2 measured tree evaluation",
                require_non_empty=False,
            )
        )

    def generated_provider_evidence_cache(
        self, cache_ref: str
    ) -> dict[str, Any]:
        with self._shared_state["lock"]:
            return dict(
                self._shared_state.get("generated_caches", {}).get(cache_ref) or {}
            )

    def provider_evidence_summary(self, cache_ref: str) -> dict[str, Any]:
        with self._shared_state["lock"]:
            return dict(
                self._shared_state.get("evidence_summaries", {}).get(cache_ref)
                or {}
            )

    def metadata(self) -> Mapping[str, Any]:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                self._execute_operation(
                    operation="metadata",
                    input_doc={},
                    provider_evidence_cache={},
                    provider_evidence_cache_ref="",
                    provider_evidence_mode="live",
                    provider_snapshot_bundle={},
                    provider_snapshot_tree_hash="",
                    provider_snapshot_manifest_hash="",
                    provider_cost_scope_override="",
                    provider_cost_cap_microusd=0,
                    provider_call_cap=0,
                    publish_provider_evidence_cache=False,
                )
            )
        raise AttestedPrivateModelRunnerV2Error(
            "synchronous model metadata cannot run on an active event loop"
        )

    async def _execute_operation(
        self,
        *,
        operation: str,
        input_doc: Mapping[str, Any],
        provider_evidence_cache: Mapping[str, Any],
        provider_evidence_cache_ref: str,
        provider_evidence_mode: str,
        provider_snapshot_bundle: Mapping[str, Any],
        provider_snapshot_tree_hash: str,
        provider_snapshot_manifest_hash: str,
        provider_cost_scope_override: str,
        provider_cost_cap_microusd: int,
        provider_call_cap: int,
        publish_provider_evidence_cache: bool,
        additional_parent_graphs: Sequence[Mapping[str, Any]] = (),
    ) -> Any:
        source_bundle = await asyncio.to_thread(
            _source_bundle_for_artifact,
            self.artifact,
            timeout_seconds=self.spec.timeout_seconds,
        )
        argv = [
            self.spec.module_name,
            self.spec.callable_name if operation == "run_icp" else "adapter_metadata",
        ]
        scope_doc: dict[str, Any] = {
            "image_digest": self.spec.image_digest,
            "argv": argv,
            "stdin_payload": dict(input_doc),
        }
        evaluation_scope = str(
            dict(self.spec.extra_env or {}).get(
                PROVIDER_COST_EVALUATION_SCOPE_ENV,
                "",
            )
            or ""
        ).strip()
        if evaluation_scope:
            scope_doc["evaluation_scope"] = evaluation_scope
        provider_cost_scope = str(provider_cost_scope_override or "") or sha256_json(
            scope_doc
        )
        cache_hash = sha256_json(dict(provider_evidence_cache))
        image_hash = "sha256:" + self.artifact.image_digest.rsplit("@sha256:", 1)[1]
        execution_epoch = (
            self.epoch_id
            if self.epoch_id is not None
            else int(
                dict(input_doc.get("context") or {}).get("evaluation_epoch") or 0
            )
        )
        catalog_loader = (
            self._catalog_snapshot_loader or load_source_add_catalog_snapshot_v2
        )
        catalog_outcome = await catalog_loader(epoch_id=int(execution_epoch))
        catalog_result = catalog_outcome.get("result")
        catalog_graph = catalog_outcome.get("receipt_graph")
        catalog_receipt = catalog_outcome.get("receipt") or catalog_outcome.get(
            "execution_receipt"
        )
        if (
            not isinstance(catalog_result, Mapping)
            or not isinstance(catalog_graph, Mapping)
            or not isinstance(catalog_receipt, Mapping)
        ):
            raise AttestedPrivateModelRunnerV2Error(
                "measured SOURCE_ADD catalog authority is unavailable"
            )
        provisioned_sources = catalog_result.get("provisioned_sources")
        private_registry_rows = catalog_result.get("private_registry_rows")
        if (
            catalog_result.get("schema_version")
            != "leadpoet.source_add_catalog_snapshot.v2"
            or not isinstance(provisioned_sources, list)
            or any(not isinstance(item, Mapping) for item in provisioned_sources)
            or not isinstance(private_registry_rows, list)
            or any(not isinstance(item, Mapping) for item in private_registry_rows)
        ):
            raise AttestedPrivateModelRunnerV2Error(
                "measured SOURCE_ADD catalog result is invalid"
            )
        try:
            runtime_catalog = validate_source_add_runtime_catalog_v2(
                catalog_result.get("runtime_catalog") or {}
            )
            derived_runtime_catalog = build_source_add_runtime_catalog_v2(
                [dict(item) for item in provisioned_sources]
            )
        except Exception as exc:
            raise AttestedPrivateModelRunnerV2Error(
                "measured SOURCE_ADD runtime catalog is invalid"
            ) from exc
        catalog_root = str(catalog_graph.get("root_receipt_hash") or "")
        if (
            runtime_catalog != derived_runtime_catalog
            or catalog_result.get("provisioned_sources_hash")
            != sha256_json([dict(item) for item in provisioned_sources])
            or catalog_result.get("private_registry_rows_hash")
            != sha256_json([dict(item) for item in private_registry_rows])
            or catalog_result.get("runtime_catalog_hash")
            != runtime_catalog["catalog_hash"]
            or catalog_root != catalog_receipt.get("receipt_hash")
            or catalog_receipt.get("role") != "gateway_coordinator"
            or catalog_receipt.get("purpose")
            != "research_lab.source_add_catalog_snapshot.v2"
            or catalog_receipt.get("status") != "succeeded"
            or catalog_receipt.get("output_root")
            != sha256_json(dict(catalog_result))
        ):
            raise AttestedPrivateModelRunnerV2Error(
                "measured SOURCE_ADD catalog commitment differs"
            )
        dynamic_provider_refs = source_add_runtime_credential_refs_v2(
            runtime_catalog
        )
        dynamic_credential_env_names = tuple(
            str(env_name)
            for route in runtime_catalog["routes"]
            for env_name in route["credential_env_refs"]
        )
        purpose = (
            "research_lab.private_model_run.v2"
            if self.model_kind == "private"
            else "research_lab.candidate_hybrid_discovery.v2"
            if provider_evidence_mode == "record"
            else "research_lab.candidate_model_run.v2"
        )
        with self._shared_state["lock"]:
            sequence = int(self._shared_state["sequence"])
            self._shared_state["sequence"] = sequence + 1
        parent_graph_by_root = {
            str(graph.get("root_receipt_hash") or ""): dict(graph)
            for graph in (
                *self.parent_graphs,
                *additional_parent_graphs,
                catalog_graph,
            )
        }
        if "" in parent_graph_by_root:
            raise AttestedPrivateModelRunnerV2Error(
                "model authority parent graph root is missing"
            )
        outcome = await self._execute(
            operation=OP_RUN_MODEL_SANDBOX_V2,
            purpose=purpose,
            epoch_id=int(execution_epoch),
            sequence=sequence,
            payload={
                "schema_version": MODEL_SANDBOX_REQUEST_SCHEMA_VERSION,
                "model_kind": self.model_kind,
                "operation": operation,
                "artifact": self.artifact.to_dict(),
                "source_bundle": source_bundle,
                "module_name": self.spec.module_name,
                "callable_name": self.spec.callable_name,
                "input": dict(input_doc),
                "environment": _measured_environment(
                    self.spec,
                    additional_credential_env_names=dynamic_credential_env_names,
                ),
                "provider_evidence_cache": dict(provider_evidence_cache),
                "provider_evidence_cache_ref": provider_evidence_cache_ref,
                "provider_evidence_mode": provider_evidence_mode,
                "provider_snapshot_bundle": dict(provider_snapshot_bundle),
                "provider_snapshot_tree_hash": provider_snapshot_tree_hash,
                "provider_snapshot_manifest_hash": provider_snapshot_manifest_hash,
                "provider_cost_scope": provider_cost_scope,
                "provider_cost_cap_microusd": int(provider_cost_cap_microusd),
                "provider_call_cap": int(provider_call_cap),
                "provider_runtime_catalog": runtime_catalog,
                "provider_catalog_evidence": {
                    "result": dict(catalog_result),
                    "root_receipt_hash": catalog_root,
                },
            },
            worker_index=self.worker_index,
            provider_credential_profile=str(
                dict(self.spec.extra_env or {}).get(
                    V2_PROVIDER_PROFILE_ENV,
                    "default",
                )
                or "default"
            ),
            provider_credential_ref_hashes=dynamic_provider_refs,
            additional_job_credential_envelope_builder=lambda job_id: [
                envelope
                for source_row in provisioned_sources
                for envelope in (
                    build_source_add_job_envelope_v2(
                        source_row,
                        job_id=job_id,
                    ),
                )
                if envelope is not None
            ],
            parent_graphs=tuple(
                parent_graph_by_root[key] for key in sorted(parent_graph_by_root)
            ),
            input_artifact_hashes=(
                self.artifact.model_artifact_hash,
                self.artifact.manifest_hash,
                image_hash,
                str(source_bundle["archive_sha256"]),
                cache_hash,
                *(
                    (
                        str(provider_snapshot_bundle["archive_sha256"]),
                        provider_snapshot_tree_hash,
                        provider_snapshot_manifest_hash,
                    )
                    if provider_snapshot_bundle
                    else ()
                ),
                catalog_root,
                str(catalog_result["provisioned_sources_hash"]),
                str(catalog_result["private_registry_rows_hash"]),
                str(catalog_result["runtime_catalog_hash"]),
                *dynamic_provider_refs.values(),
            ),
            timeout_seconds=max(1.0, float(self.spec.timeout_seconds) + 120.0),
        )
        result = outcome.get("result")
        if not isinstance(result, Mapping):
            raise AttestedPrivateModelRunnerV2Error(
                "measured model result is missing"
            )
        expected = {
            "schema_version": "leadpoet.model_sandbox_result.v2",
            "model_kind": self.model_kind,
            "operation": operation,
            "model_artifact_hash": self.artifact.model_artifact_hash,
            "model_manifest_hash": self.artifact.manifest_hash,
            "compatibility_image_digest": self.artifact.image_digest,
            "source_bundle_hash": source_bundle["archive_sha256"],
            "input_hash": sha256_json(dict(input_doc)),
            "provider_evidence_cache_hash": cache_hash,
            "provider_evidence_cache_ref": provider_evidence_cache_ref,
            "provider_evidence_mode": provider_evidence_mode,
            "provider_snapshot_archive_hash": (
                str(provider_snapshot_bundle.get("archive_sha256") or "")
                if provider_snapshot_bundle
                else sha256_json({})
            ),
            "provider_snapshot_tree_hash": (
                provider_snapshot_tree_hash or sha256_json({})
            ),
            "provider_snapshot_manifest_hash": (
                provider_snapshot_manifest_hash or sha256_json({})
            ),
            "provider_cost_cap_microusd": int(provider_cost_cap_microusd),
            "provider_call_cap": int(provider_call_cap),
            "provider_runtime_catalog_hash": runtime_catalog["catalog_hash"],
        }
        if any(result.get(name) != value for name, value in expected.items()):
            raise AttestedPrivateModelRunnerV2Error(
                "measured model result commitments differ"
            )
        trace_entries = result.get("trace_entries")
        if not isinstance(trace_entries, list) or sha256_json(trace_entries) != result.get(
            "trace_entries_hash"
        ):
            raise AttestedPrivateModelRunnerV2Error(
                "measured model trace commitment differs"
            )
        cost_summary = summarize_provider_cost_trace_entries(trace_entries)
        if result.get("output_hash") != sha256_json(result.get("output")):
            raise AttestedPrivateModelRunnerV2Error(
                "measured model output commitment differs"
            )
        generated_cache = result.get("generated_provider_evidence_cache")
        generated_cache_hash = str(
            result.get("generated_provider_evidence_cache_hash") or ""
        )
        if not isinstance(generated_cache, Mapping) or generated_cache_hash != sha256_json(
            dict(generated_cache)
        ):
            raise AttestedPrivateModelRunnerV2Error(
                "measured provider evidence tape commitment differs"
            )
        publish_incontainer_trace_entries(trace_entries)
        receipt = outcome.get("receipt")
        if not isinstance(receipt, Mapping):
            raise AttestedPrivateModelRunnerV2Error(
                "measured model receipt is missing"
            )
        if generated_cache:
            graph = outcome.get("receipt_graph")
            if not isinstance(graph, Mapping):
                raise AttestedPrivateModelRunnerV2Error(
                    "measured provider evidence tape graph is missing"
                )
            _require_tape_receipt(
                graph,
                cache_ref=provider_evidence_cache_ref,
                cache_hash=generated_cache_hash,
            )
            await _persist_provider_evidence_tape_link(
                receipt_hash=str(receipt.get("receipt_hash") or ""),
                cache_ref=provider_evidence_cache_ref,
                cache_hash=generated_cache_hash,
            )
            if publish_provider_evidence_cache:
                _write_provider_evidence_cache(
                    cache_ref=provider_evidence_cache_ref,
                    cache_document=generated_cache,
                )
            with self._shared_state["lock"]:
                self._shared_state.setdefault("generated_caches", {})[
                    provider_evidence_cache_ref
                ] = dict(generated_cache)
                self._shared_state.setdefault("evidence_summaries", {})[
                    provider_evidence_cache_ref
                ] = {
                    "cache_hash": generated_cache_hash,
                    "trace_entries_hash": str(result.get("trace_entries_hash") or ""),
                    "cost_summary": dict(cost_summary),
                }
        with self._shared_state["lock"]:
            receipt_hash = str(receipt.get("receipt_hash") or "")
            receipts = self._shared_state["receipts"]
            if not any(item.get("receipt_hash") == receipt_hash for item in receipts):
                receipts.append(dict(receipt))
            authorities = self._shared_state["authorities"]
            if not any(
                item.get("receipt", {}).get("receipt_hash") == receipt_hash
                for item in authorities
            ):
                authorities.append(dict(outcome))
        return result.get("output")


def retry_attested_model_runner_v2(
    runner: AttestedPrivateModelRunnerV2,
    *,
    extra_env: Mapping[str, str],
) -> AttestedPrivateModelRunnerV2:
    return runner.with_spec(
        replace(
            runner.spec,
            extra_env=dict(extra_env),
            pull_before_run=False,
        )
    )
