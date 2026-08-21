"""Reviewed provider-binding compiler for measured routing experiments.

The Lab never accepts a provider URL or body from an API, CLI, experiment
specification, or model profile.  A KMS-signed runtime catalog binds the full
model-owned :class:`ProviderBindingIdentity` to one compiler family and one
reviewed action.  The compiler resolves inputs from a separate immutable,
signed unit dataset and emits the exact ProviderBrokerV2 request.

Adding a model SourceAdd therefore does not add a provider branch to routing
semantics.  It adds a reviewed signed catalog entry which may select only an
action already present in ``DEEPLINE_ACTION_POLICIES``.
"""

from __future__ import annotations

import base64
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from decimal import Decimal, InvalidOperation, ROUND_CEILING
import json
import os
import re
from typing import Any, Callable, Mapping, Sequence

from gateway.research_lab.routing_experiment_artifacts import (
    RoutingArtifactAuthorityError,
    verify_routing_json_kms_signature,
)
from gateway.tee.provider_broker_v2 import PROVIDER_BROKER_SCHEMA_VERSION
from gateway.research_lab.routing_predictleads_workflows import (
    ROUTE_ACTION_ORDER,
    ROUTE_CONNECTIONS,
    ROUTE_CREDIT_CEILINGS,
    ROUTE_MAX_CALLS,
    ROUTE_NEWS,
    ROUTE_TECHNOLOGY,
    PredictLeadsWorkflowError,
    validate_workflow_input,
    workflow_manifest,
)
from gateway.research_lab.routing_execution_authorization import (
    routing_provider_logical_operation_id_v2,
)
from research_lab.canonical import sha256_json
from research_lab.eval import load_private_artifact_manifest
from research_lab.routing_experiments import (
    ProviderBindingIdentity,
    validate_provider_binding_identity,
)


ROUTING_BINDING_CATALOG_ENV = "RESEARCH_LAB_ROUTING_BINDING_CATALOG_URI"
ROUTING_UNIT_DATASET_ENV = "RESEARCH_LAB_ROUTING_UNIT_DATASET_URI"
ROUTING_BINDING_CATALOG_KEY_ENV = "RESEARCH_LAB_ROUTING_BINDING_CATALOG_KMS_KEY_ID"
ROUTING_UNIT_DATASET_KEY_ENV = "RESEARCH_LAB_ROUTING_UNIT_DATASET_KMS_KEY_ID"
ROUTING_BINDING_CATALOG_SCHEMA = "leadpoet.routing_provider_binding_catalog.v1"
ROUTING_UNIT_DATASET_SCHEMA = "leadpoet.routing_experiment_units.v1"
ROUTING_BINDING_SCHEMA = "leadpoet.routing_provider_binding.v1"
DEEPLINE_COMPILER_FAMILY = "deepline_tool_action_v1"
DEEPLINE_BASE_URL = "https://code.deepline.com"
DEEPLINE_PROVIDER_PURPOSE = "research_lab.routing_provider_evidence.v2"
_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_REF_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,191}$")
_DOMAIN_RE = re.compile(
    r"^(?=.{1,253}$)(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,63}$"
)
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_COUNTRIES_RE = re.compile(r"^[A-Z]{2}(?:;[A-Z]{2})*$")
_SAFE_TEXT_RE = re.compile(r"^[^\x00-\x1f\x7f]{1,512}$")
_SENSITIVE_KEYS = frozenset(
    {
        "authorization",
        "api_key",
        "apikey",
        "access_token",
        "token",
        "password",
        "private_key",
        "client_secret",
        "credential",
        "service_role",
        "request_body",
        "response_text",
    }
)


class RoutingProviderBindingError(RuntimeError):
    """A runtime binding, signed input, or broker result is not exact."""


def _require_hash(value: Any, name: str) -> str:
    text = str(value or "").strip().lower()
    if not _HASH_RE.fullmatch(text):
        raise RoutingProviderBindingError(f"routing {name} is not a sha256 digest")
    return text


def _require_immutable_s3_uri(value: Any, name: str) -> str:
    uri = str(value or "").strip()
    if (
        not uri.startswith("s3://")
        or uri.endswith("/current.json")
        or "/branches/" in uri
    ):
        raise RoutingProviderBindingError(f"routing {name} URI is not immutable")
    return uri


def _canonical_manifest_hash(document: Mapping[str, Any]) -> str:
    payload = dict(document)
    manifest_hash = _require_hash(payload.pop("manifest_hash", ""), "manifest hash")
    if sha256_json(payload) != manifest_hash:
        raise RoutingProviderBindingError("routing signed manifest hash differs")
    return manifest_hash


def _verified_document(
    *,
    uri: str,
    loader: Callable[[str], Mapping[str, Any]],
    verifier: Callable[[Mapping[str, Any], str], Mapping[str, Any]],
    key_id: str,
) -> tuple[dict[str, Any], str]:
    try:
        document = dict(loader(uri))
        manifest_hash = _canonical_manifest_hash(document)
        verification = dict(verifier(document, key_id))
    except RoutingProviderBindingError:
        raise
    except Exception as exc:  # noqa: BLE001 - authority boundary
        raise RoutingProviderBindingError("routing signed document verification failed") from exc
    if (
        verification.get("verified") is not True
        or verification.get("manifest_hash") != manifest_hash
        or verification.get("signature_ref") != document.get("signature_ref")
        or verification.get("key_id") != key_id
        or verification.get("signing_algorithm") != "ECDSA_SHA_256"
    ):
        raise RoutingProviderBindingError("routing signed document binding is incomplete")
    if document.get("manifest_uri") != uri:
        raise RoutingProviderBindingError("routing signed document URI differs")
    return document, manifest_hash


@dataclass(frozen=True)
class DeeplineActionPolicy:
    """Code-reviewed constraints for one Deepline tool action."""

    action_id: str
    provider: str
    operation: str
    allowed_input_fields: frozenset[str]
    validation_context_fields: frozenset[str]
    required_any: tuple[frozenset[str], ...]
    fixed_inputs: Mapping[str, Any]
    maximum_results: int
    hard_timeout_ms: int
    hard_credit_microunits: int
    fixed_credit_per_result_microunits: int | None
    fixed_credit_per_call_microunits: int | None
    result_paths: tuple[tuple[str, ...], ...]
    evidence_fields: tuple[str, ...]


def _policy(
    action_id: str,
    *,
    allowed: Sequence[str],
    context: Sequence[str] = (),
    required_any: Sequence[Sequence[str]],
    fixed: Mapping[str, Any],
    maximum_results: int,
    credit_per_result: int | None = None,
    credit_per_call: int | None = None,
    result_paths: Sequence[Sequence[str]],
    evidence_fields: Sequence[str],
    timeout_ms: int = 30_000,
) -> DeeplineActionPolicy:
    return DeeplineActionPolicy(
        action_id=action_id,
        provider=action_id.split("_", 1)[0],
        operation=action_id,
        allowed_input_fields=frozenset(allowed),
        validation_context_fields=frozenset(context),
        required_any=tuple(frozenset(group) for group in required_any),
        fixed_inputs=dict(fixed),
        maximum_results=maximum_results,
        hard_timeout_ms=timeout_ms,
        hard_credit_microunits=(
            credit_per_call
            if credit_per_call is not None
            else (credit_per_result or 0) * maximum_results
        ),
        fixed_credit_per_result_microunits=credit_per_result,
        fixed_credit_per_call_microunits=credit_per_call,
        result_paths=tuple(tuple(path) for path in result_paths),
        evidence_fields=tuple(evidence_fields),
    )


# This registry is intentionally closed.  A signed binding catalog can map a
# new model SourceAdd to a reviewed action without adding routing branches,
# but it cannot name an arbitrary Deepline endpoint or request field.
DEEPLINE_ACTION_POLICIES: Mapping[str, DeeplineActionPolicy] = {
    "bloomberry_search_job_postings": _policy(
        "bloomberry_search_job_postings",
        allowed=(
            "domain", "keyword", "normalized_job_titles", "begin_date", "end_date",
            "countries", "region_countries", "company_size_range", "company_industry",
            "active_only", "remote_only",
        ),
        context=("minimum_date", "maximum_date"),
        required_any=(("domain",), ("keyword", "normalized_job_titles")),
        fixed={"limit": 1, "exact_match": True, "show_facets": False, "active_only": True},
        maximum_results=1,
        credit_per_result=90_000,
        result_paths=(("result", "data", "jobs"), ("result", "jobs"), ("jobs",)),
        evidence_fields=(
            "id", "title", "normalized_job_title", "company_domain", "snapshot_date",
            "displayed_url", "inactive", "regions", "remote",
        ),
    ),
    "bloomberry_get_tech_stack_changes": _policy(
        "bloomberry_get_tech_stack_changes",
        allowed=(
            "technology_name", "technology_category", "begin_date", "end_date", "countries",
            "company_size_range", "company_industry",
        ),
        context=("expected_domain", "expected_country", "minimum_date", "maximum_date"),
        required_any=(("technology_name", "technology_category"),),
        fixed={"limit": 1, "omit_linkedin_data": "true"},
        maximum_results=1,
        credit_per_result=4_210_000,
        result_paths=(("result", "data", "signals"), ("result", "signals"), ("signals",)),
        evidence_fields=(
            "company_domain", "company_name", "vendor_name", "category", "first_seen",
            "last_seen", "event_type", "date", "change_date", "vendor_url", "vendor_source",
            "company_country", "country",
        ),
        timeout_ms=45_000,
    ),
    "podscan_episodes_search": _policy(
        "podscan_episodes_search",
        allowed=("query", "language", "include_transcript"),
        context=("expected_company", "expected_person", "minimum_date", "maximum_date"),
        required_any=(("query",),),
        fixed={"page": 1, "per_page": 1, "include_transcript": False},
        maximum_results=1,
        credit_per_result=140_000,
        result_paths=(
            ("data", "episodes"),
            ("result", "data", "episodes"),
            ("result", "episodes"),
            ("episodes",),
        ),
        evidence_fields=(
            "id", "episode_id", "title", "episode_title", "episode_url", "url", "podcast_name",
            "podcast.podcast_name", "published_at", "posted_at", "metadata.guests",
            "transcript_url", "search_highlight", "transcript_highlight",
            "_search_highlight", "guest_company", "guest_name", "guests",
        ),
    ),
    **{
        action_id: _policy(
            action_id,
            allowed=("domain", "begin_date", "end_date", "active_only", "categories"),
            context=("role_keyword", "minimum_date", "maximum_date"),
            required_any=(("domain",),),
            fixed={
                "page": 1,
                "limit": 25,
                **(
                    {"active_only": True, "not_closed": True}
                    if action_id == "predictleads_company_job_openings"
                    else {}
                ),
            },
            maximum_results=25,
            credit_per_call=560_000,
            result_paths=(("result", "data", "data"), ("result", "data"), ("data",)),
            evidence_fields=(
                "id", "type", "attributes.title", "attributes.normalized_title",
                "attributes.first_seen_at", "attributes.last_seen_at", "attributes.found_at",
                "attributes.posted_at", "attributes.effective_date", "attributes.announced_on",
                "attributes.status", "attributes.description", "attributes.url",
                "attributes.source_url", "attributes.source_urls", "attributes.domain",
                "attributes.category", "attributes.amount", "attributes.funding_type",
                "attributes.financing_type", "attributes.financing_type_normalized",
                "attributes.categories", "attributes.amount_normalized",
            ),
        )
        for action_id in (
            "predictleads_company_connections",
            "predictleads_company_financing_events",
            "predictleads_company_job_openings",
            "predictleads_company_news_events",
            "predictleads_company_technology_detections",
        )
    },
    "builtwith_domain_lookup": _policy(
        "builtwith_domain_lookup",
        allowed=(
            "domain", "first_detected_from", "first_detected_until",
            "last_detected_from", "last_detected_until",
        ),
        context=("requested_technology", "parent_intent_event_hash"),
        required_any=(("domain",),),
        fixed={
            "live_only": False,
            "no_pii": True,
            "no_meta": True,
            "no_attr": True,
            "hide_text": True,
            "trust": False,
        },
        maximum_results=1,
        # BuiltWith pricing is usage-calculated.  The signed binding must set
        # a conservative lower ceiling no larger than this hard stop, and the
        # Deepline response must return authoritative billing.
        credit_per_call=10_000_000,
        result_paths=(("result", "data", "Results"), ("result", "Results"), ("Results",)),
        evidence_fields=(
            "Lookup", "FirstIndexed", "LastIndexed", "Result.Spend",
            "Result.Paths",
        ),
        timeout_ms=45_000,
    ),
}


_MODEL_TOOL_ACTIONS: Mapping[str, frozenset[str]] = {
    "intent.source_add.bloomberry": frozenset({"bloomberry_get_tech_stack_changes"}),
    "intent.source_add.bloomberry_jobs": frozenset({"bloomberry_search_job_postings"}),
    "intent.source_add.podscan": frozenset({"podscan_episodes_search"}),
    "intent.source_add.predictleads_connections": frozenset({"predictleads_company_connections"}),
    "intent.source_add.predictleads_financing": frozenset({"predictleads_company_financing_events"}),
    "intent.source_add.predictleads_jobs": frozenset({"predictleads_company_job_openings"}),
    "intent.source_add.predictleads_news": frozenset({"predictleads_company_news_events"}),
    "intent.source_add.predictleads_technology": frozenset(
        {"predictleads_company_technology_detections"}
    ),
    "intent.source_add.builtwith": frozenset({"builtwith_domain_lookup"}),
}

_MODEL_TOOL_PROVIDER_IDS: Mapping[str, str] = {
    "intent.source_add.bloomberry": "bloomberry",
    "intent.source_add.bloomberry_jobs": "bloomberry_jobs",
    "intent.source_add.podscan": "podscan",
    "intent.source_add.predictleads_connections": "predictleads_connections",
    "intent.source_add.predictleads_financing": "predictleads_financing",
    "intent.source_add.predictleads_jobs": "predictleads_jobs",
    "intent.source_add.predictleads_news": "predictleads_news",
    "intent.source_add.predictleads_technology": "predictleads_technology",
    "intent.source_add.builtwith": "builtwith",
}

# The Model registers Sumble for deterministic routing and replay, but the Lab
# has no reviewed, provider-bounded execution contract for it. A signed host
# catalog cannot make this tool available until that contract is added here.
EXPLICITLY_UNAVAILABLE_MODEL_TOOLS = frozenset(
    {"intent.source_add.sumble"}
)

# These model-owned tools are multi-call workflows.  A direct action is not a
# safe substitute.  The legacy direct-action admission projection remains
# fail-closed; the composite workflow compiler below consumes the separate
# code-reviewed workflow manifest and never dispatches a provider.
UNAVAILABLE_COMPOSITE_SOURCE_TOOLS = frozenset(
    {
        "intent.source_add.predictleads_connections",
        "intent.source_add.predictleads_news",
        "intent.source_add.predictleads_technology",
    }
)

# A composite row is admitted only when it binds one of these model tools to
# the exact workflow manifest exported by routing_predictleads_workflows.
# The old unavailable projection is retained until the parser below is
# updated; it must never be used as the workflow compiler's route map.
_COMPOSITE_SOURCE_WORKFLOWS: Mapping[str, str] = {
    "intent.source_add.predictleads_connections": ROUTE_CONNECTIONS,
    "intent.source_add.predictleads_news": ROUTE_NEWS,
    "intent.source_add.predictleads_technology": ROUTE_TECHNOLOGY,
}
COMPOSITE_WORKFLOW_TOOL_IDS = frozenset(_COMPOSITE_SOURCE_WORKFLOWS)
MEASURED_UNAVAILABLE_COMPOSITE_SOURCE_TOOLS = frozenset(
    {"intent.source_add.predictleads_news"}
)
_COMPOSITE_INPUT_FIELDS: Mapping[str, frozenset[str]] = {
    ROUTE_CONNECTIONS: frozenset({"company_domain", "minimum_date", "maximum_date"}),
    ROUTE_NEWS: frozenset(
        {"company_domain", "intent_category", "minimum_date", "maximum_date"}
    ),
    ROUTE_TECHNOLOGY: frozenset(
        {"company_domain", "technology", "minimum_date", "maximum_date"}
    ),
}


@dataclass(frozen=True)
class RoutingBindingManifest:
    binding: ProviderBindingIdentity
    compiler_family: str
    transport_id: str
    action_id: str
    input_projection: Mapping[str, str]
    input_constants: Mapping[str, Any]
    model_binding_requirements_hash: str
    output_contract_hash: str
    evidence_contract_hash: str
    retry_policy_hash: str
    max_results: int
    timeout_ms: int
    credit_ceiling_microunits: int
    execution_kind: str = "direct_action"
    workflow_id: str | None = None
    workflow_manifest_hash: str | None = None

    def identity_key(self) -> str:
        return sha256_json(
            {
                "schema_version": ROUTING_BINDING_SCHEMA,
                "binding": self.binding.to_dict(),
            }
        )


@dataclass(frozen=True)
class VerifiedRoutingBindingCatalog:
    manifest_uri: str
    manifest_hash: str
    signature_ref: str
    signing_key_id: str
    catalog_version: str
    bindings: Mapping[str, RoutingBindingManifest]

    def resolve(self, binding: ProviderBindingIdentity) -> RoutingBindingManifest:
        key = sha256_json(
            {"schema_version": ROUTING_BINDING_SCHEMA, "binding": binding.to_dict()}
        )
        manifest = self.bindings.get(key)
        if manifest is None or manifest.binding != binding:
            raise RoutingProviderBindingError(
                "routing provider binding is absent from the signed runtime catalog"
            )
        return manifest


class SignedRoutingBindingCatalogLoader:
    def __init__(
        self,
        *,
        manifest_uri: str | None = None,
        key_id: str | None = None,
        loader: Callable[[str], Mapping[str, Any]] = load_private_artifact_manifest,
        verifier: Callable[[Mapping[str, Any], str], Mapping[str, Any]] = verify_routing_json_kms_signature,
    ) -> None:
        self.manifest_uri = _require_immutable_s3_uri(
            manifest_uri or os.getenv(ROUTING_BINDING_CATALOG_ENV, ""),
            "binding catalog",
        )
        self.key_id = str(
            key_id or os.getenv(ROUTING_BINDING_CATALOG_KEY_ENV, "")
        ).strip()
        if not self.key_id:
            raise RoutingProviderBindingError("routing binding catalog signing key is missing")
        self._loader = loader
        self._verifier = verifier

    def load(self) -> VerifiedRoutingBindingCatalog:
        document, manifest_hash = _verified_document(
            uri=self.manifest_uri,
            loader=self._loader,
            verifier=self._verifier,
            key_id=self.key_id,
        )
        if set(document) != {
            "schema_version", "manifest_uri", "catalog_version", "bindings",
            "signature_ref", "manifest_hash",
        } or document.get("schema_version") != ROUTING_BINDING_CATALOG_SCHEMA:
            raise RoutingProviderBindingError("routing binding catalog fields are invalid")
        rows = document.get("bindings")
        if not isinstance(rows, list) or not rows:
            raise RoutingProviderBindingError("routing binding catalog is empty")
        bindings: dict[str, RoutingBindingManifest] = {}
        for row in rows:
            manifest = self._parse_binding(row)
            key = manifest.identity_key()
            if key in bindings:
                raise RoutingProviderBindingError("routing binding identity is duplicated")
            bindings[key] = manifest
        return VerifiedRoutingBindingCatalog(
            manifest_uri=self.manifest_uri,
            manifest_hash=manifest_hash,
            signature_ref=str(document["signature_ref"]),
            signing_key_id=self.key_id,
            catalog_version=str(document.get("catalog_version") or ""),
            bindings=bindings,
        )

    def load_composite_workflows(self) -> VerifiedRoutingBindingCatalog:
        """Load a catalog containing only signed, preparation-only workflows.

        The ordinary ``load`` path remains fail-closed for the legacy direct
        projection.  This explicit loader is used by the workflow compiler
        and cannot create a direct action or dispatch a provider.
        """

        document, manifest_hash = _verified_document(
            uri=self.manifest_uri,
            loader=self._loader,
            verifier=self._verifier,
            key_id=self.key_id,
        )
        if set(document) != {
            "schema_version", "manifest_uri", "catalog_version", "bindings",
            "signature_ref", "manifest_hash",
        } or document.get("schema_version") != ROUTING_BINDING_CATALOG_SCHEMA:
            raise RoutingProviderBindingError("routing binding catalog fields are invalid")
        rows = document.get("bindings")
        if not isinstance(rows, list) or not rows:
            raise RoutingProviderBindingError("routing binding catalog is empty")
        bindings: dict[str, RoutingBindingManifest] = {}
        for row in rows:
            manifest = self._parse_composite_binding(row)
            key = manifest.identity_key()
            if key in bindings:
                raise RoutingProviderBindingError("routing binding identity is duplicated")
            bindings[key] = manifest
        return VerifiedRoutingBindingCatalog(
            manifest_uri=self.manifest_uri,
            manifest_hash=manifest_hash,
            signature_ref=str(document["signature_ref"]),
            signing_key_id=self.key_id,
            catalog_version=str(document.get("catalog_version") or ""),
            bindings=bindings,
        )

    def load_reviewed_bindings(self) -> VerifiedRoutingBindingCatalog:
        """Load one signed catalog containing direct and composite rows.

        This is the explicit mixed-waterfall path.  The legacy ``load`` path
        is unchanged and remains the compatibility gate for direct bindings.
        Each row declares its exact execution kind, and duplicate binding
        identities are rejected across both kinds.
        """

        document, manifest_hash = _verified_document(
            uri=self.manifest_uri,
            loader=self._loader,
            verifier=self._verifier,
            key_id=self.key_id,
        )
        if set(document) != {
            "schema_version", "manifest_uri", "catalog_version", "bindings",
            "signature_ref", "manifest_hash",
        } or document.get("schema_version") != ROUTING_BINDING_CATALOG_SCHEMA:
            raise RoutingProviderBindingError("routing binding catalog fields are invalid")
        rows = document.get("bindings")
        if not isinstance(rows, list) or not rows:
            raise RoutingProviderBindingError("routing binding catalog is empty")
        bindings: dict[str, RoutingBindingManifest] = {}
        for row in rows:
            kind = row.get("execution_kind") if isinstance(row, Mapping) else None
            if kind == "direct_action":
                manifest = self._parse_binding(row)
            elif kind == "composite_workflow":
                manifest = self._parse_composite_binding(row)
            else:
                raise RoutingProviderBindingError(
                    "routing binding execution kind is invalid"
                )
            key = manifest.identity_key()
            if key in bindings:
                raise RoutingProviderBindingError("routing binding identity is duplicated")
            bindings[key] = manifest
        return VerifiedRoutingBindingCatalog(
            manifest_uri=self.manifest_uri,
            manifest_hash=manifest_hash,
            signature_ref=str(document["signature_ref"]),
            signing_key_id=self.key_id,
            catalog_version=str(document.get("catalog_version") or ""),
            bindings=bindings,
        )

    @staticmethod
    def _parse_binding(row: Any) -> RoutingBindingManifest:
        expected = {
            "binding", "compiler_family", "transport_id", "execution_kind", "action_id",
            "workflow_id", "workflow_manifest_hash", "input_projection",
            "input_constants", "model_binding_requirements_hash",
            "output_contract_hash", "evidence_contract_hash",
            "retry_policy_hash", "max_results", "timeout_ms", "credit_ceiling_microunits",
        }
        if not isinstance(row, Mapping) or set(row) != expected:
            raise RoutingProviderBindingError("routing binding manifest fields are invalid")
        raw_binding = row.get("binding")
        if not isinstance(raw_binding, Mapping):
            raise RoutingProviderBindingError("routing provider binding is invalid")
        binding = ProviderBindingIdentity.from_mapping(raw_binding)
        if (
            set(raw_binding) != set(binding.to_dict())
            or validate_provider_binding_identity(binding)
        ):
            raise RoutingProviderBindingError("routing provider binding is invalid")
        if binding.tool_id in EXPLICITLY_UNAVAILABLE_MODEL_TOOLS:
            raise RoutingProviderBindingError(
                "routing model tool is explicitly unavailable"
            )
        action_id = str(row.get("action_id") or "")
        policy = DEEPLINE_ACTION_POLICIES.get(action_id)
        execution_kind = row.get("execution_kind")
        workflow_id = row.get("workflow_id")
        workflow_manifest_hash = row.get("workflow_manifest_hash")
        if execution_kind not in {"direct_action", "composite_workflow"}:
            raise RoutingProviderBindingError("routing binding execution kind is invalid")
        if execution_kind == "direct_action":
            if workflow_id is not None or workflow_manifest_hash is not None or not action_id:
                raise RoutingProviderBindingError(
                    "routing direct-action workflow fields are invalid"
                )
        elif (
            action_id
            or not isinstance(workflow_id, str)
            or not isinstance(workflow_manifest_hash, str)
            or binding.tool_id not in _COMPOSITE_SOURCE_WORKFLOWS
            or workflow_id != _COMPOSITE_SOURCE_WORKFLOWS[binding.tool_id]
        ):
            raise RoutingProviderBindingError(
                "routing composite-workflow identity is invalid"
            )
        if execution_kind == "composite_workflow":
            reviewed_workflow = workflow_manifest(workflow_id)
            if workflow_manifest_hash != reviewed_workflow.manifest_hash:
                raise RoutingProviderBindingError(
                    "routing workflow manifest hash differs from reviewed workflow"
                )
        if (
            row.get("transport_id") != "deepline"
            or binding.tool_id in UNAVAILABLE_COMPOSITE_SOURCE_TOOLS
            or row.get("compiler_family") != DEEPLINE_COMPILER_FAMILY
            or policy is None
            or binding.tool_id not in _MODEL_TOOL_ACTIONS
            or binding.provider_id != _MODEL_TOOL_PROVIDER_IDS.get(binding.tool_id)
            or action_id not in _MODEL_TOOL_ACTIONS[binding.tool_id]
        ):
            raise RoutingProviderBindingError("routing binding selects an unreviewed action")
        projection = row.get("input_projection")
        constants = row.get("input_constants")
        if not isinstance(projection, Mapping) or not isinstance(constants, Mapping):
            raise RoutingProviderBindingError("routing binding input projection is invalid")
        normalized_projection = {str(key): str(value) for key, value in projection.items()}
        if (
            not set(normalized_projection).issubset(
                policy.allowed_input_fields | policy.validation_context_fields
            )
            or any(not _REF_RE.fullmatch(value) for value in normalized_projection.values())
            or not set(constants).issubset(policy.allowed_input_fields)
            or set(constants).intersection(normalized_projection)
            or set(constants).intersection(policy.fixed_inputs)
            or set(normalized_projection).intersection(policy.fixed_inputs)
            or any(str(key).lower() in _SENSITIVE_KEYS for key in (*projection, *constants))
        ):
            raise RoutingProviderBindingError("routing binding input fields are not reviewed")
        max_results = row.get("max_results")
        timeout_ms = row.get("timeout_ms")
        credit = row.get("credit_ceiling_microunits")
        if (
            type(max_results) is not int or not 1 <= max_results <= policy.maximum_results
            or type(timeout_ms) is not int or not 1 <= timeout_ms <= policy.hard_timeout_ms
            or type(credit) is not int or not 0 < credit <= policy.hard_credit_microunits
        ):
            raise RoutingProviderBindingError("routing binding limits exceed reviewed action")
        for field_name in (
            "model_binding_requirements_hash", "output_contract_hash",
            "evidence_contract_hash", "retry_policy_hash"
        ):
            _require_hash(row.get(field_name), field_name)
        return RoutingBindingManifest(
            binding=binding,
            compiler_family=DEEPLINE_COMPILER_FAMILY,
            transport_id="deepline",
            action_id=action_id,
            input_projection=normalized_projection,
            input_constants=dict(constants),
            model_binding_requirements_hash=str(row["model_binding_requirements_hash"]),
            output_contract_hash=str(row["output_contract_hash"]),
            evidence_contract_hash=str(row["evidence_contract_hash"]),
            retry_policy_hash=str(row["retry_policy_hash"]),
            max_results=max_results,
            timeout_ms=timeout_ms,
            credit_ceiling_microunits=credit,
        )

    @staticmethod
    def _parse_composite_binding(row: Any) -> RoutingBindingManifest:
        """Parse one exact composite row without accepting transport details."""

        expected = {
            "binding", "compiler_family", "transport_id", "execution_kind", "action_id",
            "workflow_id", "workflow_manifest_hash", "input_projection",
            "input_constants", "model_binding_requirements_hash",
            "output_contract_hash", "evidence_contract_hash", "retry_policy_hash",
            "max_results", "timeout_ms", "credit_ceiling_microunits",
        }
        if not isinstance(row, Mapping) or set(row) != expected:
            raise RoutingProviderBindingError("routing binding manifest fields are invalid")
        raw_binding = row.get("binding")
        if not isinstance(raw_binding, Mapping):
            raise RoutingProviderBindingError("routing provider binding is invalid")
        binding = ProviderBindingIdentity.from_mapping(raw_binding)
        if (
            set(raw_binding) != set(binding.to_dict())
            or validate_provider_binding_identity(binding)
        ):
            raise RoutingProviderBindingError("routing provider binding is invalid")
        if binding.tool_id in EXPLICITLY_UNAVAILABLE_MODEL_TOOLS:
            raise RoutingProviderBindingError(
                "routing model tool is explicitly unavailable"
            )
        workflow_id = row.get("workflow_id")
        workflow_hash = row.get("workflow_manifest_hash")
        if (
            row.get("execution_kind") != "composite_workflow"
            or row.get("transport_id") != "deepline"
            or row.get("compiler_family") != DEEPLINE_COMPILER_FAMILY
            or row.get("action_id") is not None
            or not isinstance(workflow_id, str)
            or not isinstance(workflow_hash, str)
            or binding.tool_id not in _COMPOSITE_SOURCE_WORKFLOWS
            or workflow_id != _COMPOSITE_SOURCE_WORKFLOWS[binding.tool_id]
        ):
            raise RoutingProviderBindingError("routing composite-workflow identity is invalid")
        try:
            reviewed = workflow_manifest(workflow_id)
        except PredictLeadsWorkflowError as exc:
            raise RoutingProviderBindingError("routing workflow is not reviewed") from exc
        if (
            workflow_hash != reviewed.manifest_hash
            or tuple(item.action_id for item in reviewed.ordered_actions)
            != ROUTE_ACTION_ORDER[workflow_id]
            or reviewed.max_calls != ROUTE_MAX_CALLS[workflow_id]
            or reviewed.credit_ceiling_microcredits
            != ROUTE_CREDIT_CEILINGS[workflow_id]
        ):
            raise RoutingProviderBindingError(
                "routing workflow manifest differs from reviewed workflow"
            )
        if binding.provider_id != _MODEL_TOOL_PROVIDER_IDS.get(binding.tool_id):
            raise RoutingProviderBindingError("routing composite provider is not reviewed")
        projection = row.get("input_projection")
        constants = row.get("input_constants")
        if not isinstance(projection, Mapping) or not isinstance(constants, Mapping):
            raise RoutingProviderBindingError("routing composite input projection is invalid")
        normalized_projection = {str(key): str(value) for key, value in projection.items()}
        if (
            set(normalized_projection) != _COMPOSITE_INPUT_FIELDS[workflow_id]
            or constants
            or any(not _REF_RE.fullmatch(value) for value in normalized_projection.values())
            or any(str(key).lower() in _SENSITIVE_KEYS for key in (*projection, *constants))
        ):
            raise RoutingProviderBindingError("routing composite input projection is invalid")
        max_results = row.get("max_results")
        timeout_ms = row.get("timeout_ms")
        credit = row.get("credit_ceiling_microunits")
        if (
            type(max_results) is not int
            or type(timeout_ms) is not int
            or type(credit) is not int
            or max_results != 1
            or timeout_ms != reviewed.timeout_ms
            or credit != ROUTE_CREDIT_CEILINGS[workflow_id]
        ):
            raise RoutingProviderBindingError(
                "routing composite workflow limits exceed reviewed route"
            )
        for field_name in (
            "model_binding_requirements_hash", "output_contract_hash",
            "evidence_contract_hash", "retry_policy_hash",
        ):
            _require_hash(row.get(field_name), field_name)
        return RoutingBindingManifest(
            binding=binding,
            compiler_family=DEEPLINE_COMPILER_FAMILY,
            transport_id="deepline",
            action_id="",
            input_projection=normalized_projection,
            input_constants={},
            model_binding_requirements_hash=str(row["model_binding_requirements_hash"]),
            output_contract_hash=str(row["output_contract_hash"]),
            evidence_contract_hash=str(row["evidence_contract_hash"]),
            retry_policy_hash=str(row["retry_policy_hash"]),
            max_results=1,
            timeout_ms=reviewed.timeout_ms,
            credit_ceiling_microunits=credit,
            execution_kind="composite_workflow",
            workflow_id=workflow_id,
            workflow_manifest_hash=workflow_hash,
        )


@dataclass(frozen=True)
class VerifiedRoutingUnitDataset:
    manifest_uri: str
    manifest_hash: str
    signature_ref: str
    signing_key_id: str
    unit_set_hash: str
    provenance_hash: str
    units: Mapping[str, Mapping[str, Any]]

    def resolve(self, unit_ref: str) -> tuple[Mapping[str, Any], str]:
        if unit_ref not in self.units:
            raise RoutingProviderBindingError("routing unit is absent from signed dataset")
        value = dict(self.units[unit_ref])
        return value, sha256_json(
            {
                "schema_version": "leadpoet.routing_unit_input.v1",
                "unit_ref": unit_ref,
                "input": value,
            }
        )


class SignedRoutingUnitDatasetLoader:
    def __init__(
        self,
        *,
        manifest_uri: str | None = None,
        key_id: str | None = None,
        loader: Callable[[str], Mapping[str, Any]] = load_private_artifact_manifest,
        verifier: Callable[[Mapping[str, Any], str], Mapping[str, Any]] = verify_routing_json_kms_signature,
    ) -> None:
        self.manifest_uri = _require_immutable_s3_uri(
            manifest_uri or os.getenv(ROUTING_UNIT_DATASET_ENV, ""),
            "unit dataset",
        )
        self.key_id = str(
            key_id or os.getenv(ROUTING_UNIT_DATASET_KEY_ENV, "")
        ).strip()
        if not self.key_id:
            raise RoutingProviderBindingError("routing unit dataset signing key is missing")
        self._loader = loader
        self._verifier = verifier

    def load(
        self,
        *,
        expected_unit_refs: Sequence[str],
        expected_unit_set_hash: str,
    ) -> VerifiedRoutingUnitDataset:
        dataset = self.load_reviewed_dataset()
        if tuple(dataset.units) != tuple(sorted(set(expected_unit_refs))):
            raise RoutingProviderBindingError(
                "routing unit refs differ from signed dataset"
            )
        if _require_hash(expected_unit_set_hash, "unit set hash") != (
            dataset.unit_set_hash
        ):
            raise RoutingProviderBindingError("routing unit dataset hash differs")
        return dataset

    def load_reviewed_dataset(self) -> VerifiedRoutingUnitDataset:
        """Load the complete signed immutable dataset for protected execution."""

        document, manifest_hash = _verified_document(
            uri=self.manifest_uri,
            loader=self._loader,
            verifier=self._verifier,
            key_id=self.key_id,
        )
        if set(document) != {
            "schema_version", "manifest_uri", "units", "unit_set_hash",
            "provenance_hash", "signature_ref", "manifest_hash",
        } or document.get("schema_version") != ROUTING_UNIT_DATASET_SCHEMA:
            raise RoutingProviderBindingError("routing unit dataset fields are invalid")
        raw_units = document.get("units")
        if not isinstance(raw_units, Mapping):
            raise RoutingProviderBindingError("routing unit dataset is invalid")
        units: dict[str, Mapping[str, Any]] = {}
        for raw_ref, raw_input in sorted(raw_units.items()):
            ref = str(raw_ref)
            if not _REF_RE.fullmatch(ref) or not isinstance(raw_input, Mapping):
                raise RoutingProviderBindingError("routing unit dataset entry is invalid")
            _validate_unit_input(raw_input)
            units[ref] = dict(raw_input)
        unit_set_hash = sha256_json(
            {
                "schema_version": ROUTING_UNIT_DATASET_SCHEMA,
                "units": [{"unit_ref": key, "input": units[key]} for key in units],
            }
        )
        if document.get("unit_set_hash") != unit_set_hash:
            raise RoutingProviderBindingError("routing unit dataset hash differs")
        return VerifiedRoutingUnitDataset(
            manifest_uri=self.manifest_uri,
            manifest_hash=manifest_hash,
            signature_ref=str(document["signature_ref"]),
            signing_key_id=self.key_id,
            unit_set_hash=unit_set_hash,
            provenance_hash=_require_hash(document.get("provenance_hash"), "unit provenance hash"),
            units=units,
        )


def _validate_unit_input(value: Mapping[str, Any]) -> None:
    if len(value) > 24:
        raise RoutingProviderBindingError("routing unit input has too many fields")
    for key, item in value.items():
        name = str(key)
        if name == "model_input":
            _validate_signed_model_input(item, depth=0)
            continue
        if (
            not _REF_RE.fullmatch(name)
            or name.lower() in _SENSITIVE_KEYS
            or isinstance(item, Mapping)
            or isinstance(item, (bytes, bytearray))
        ):
            raise RoutingProviderBindingError("routing unit input field is invalid")
        if isinstance(item, Sequence) and not isinstance(item, str):
            if len(item) > 32 or any(not isinstance(part, str) for part in item):
                raise RoutingProviderBindingError("routing unit input list is invalid")
        elif item is not None and type(item) not in {str, bool, int}:
            raise RoutingProviderBindingError("routing unit input value is invalid")


def _validate_signed_model_input(value: Any, *, depth: int) -> None:
    """Allow one bounded JSON Model start input in the signed unit artifact."""

    if depth > 8:
        raise RoutingProviderBindingError("routing Model input is too deep")
    if isinstance(value, Mapping):
        if len(value) > 64:
            raise RoutingProviderBindingError(
                "routing Model input has too many fields"
            )
        for raw_key, child in value.items():
            key = str(raw_key)
            if (
                not _REF_RE.fullmatch(key)
                or key.casefold() in _SENSITIVE_KEYS
            ):
                raise RoutingProviderBindingError(
                    "routing Model input field is invalid"
                )
            _validate_signed_model_input(child, depth=depth + 1)
        return
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        if len(value) > 256:
            raise RoutingProviderBindingError(
                "routing Model input list is too large"
            )
        for child in value:
            _validate_signed_model_input(child, depth=depth + 1)
        return
    if value is None or type(value) in {str, bool, int}:
        if isinstance(value, str) and len(value) > 8_192:
            raise RoutingProviderBindingError(
                "routing Model input text is too large"
            )
        return
    raise RoutingProviderBindingError("routing Model input value is invalid")


@dataclass(frozen=True)
class PreparedRoutingProviderCall:
    binding: ProviderBindingIdentity
    binding_manifest_hash: str
    binding_catalog_manifest_hash: str
    binding_catalog_version: str
    unit_ref: str
    unit_input_hash: str
    unit_dataset_manifest_hash: str
    unit_set_hash: str
    model_binding_requirements_hash: str
    action_id: str
    transport_id: str
    provider: str
    operation: str
    payload: Mapping[str, Any]
    validation_context: Mapping[str, Any]
    validation_context_hash: str
    request_body_hash: str
    timeout_ms: int
    credit_ceiling_microunits: int
    max_results: int
    retry_policy_hash: str
    evidence_contract_hash: str
    output_contract_hash: str

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any]
    ) -> "PreparedRoutingProviderCall":
        if not isinstance(value, Mapping) or set(value) != set(cls.__annotations__):
            raise RoutingProviderBindingError(
                "prepared routing provider call fields are invalid"
            )
        binding = value.get("binding")
        payload = value.get("payload")
        validation_context = value.get("validation_context")
        if (
            not isinstance(binding, Mapping)
            or not isinstance(payload, Mapping)
            or not isinstance(validation_context, Mapping)
        ):
            raise RoutingProviderBindingError(
                "prepared routing provider call documents are invalid"
            )
        values = dict(value)
        values["binding"] = ProviderBindingIdentity.from_mapping(binding)
        values["payload"] = dict(payload)
        values["validation_context"] = dict(validation_context)
        try:
            prepared = cls(**values)
        except (TypeError, ValueError) as exc:
            raise RoutingProviderBindingError(
                "prepared routing provider call is invalid"
            ) from exc
        # Reuse the existing closed projection validators.  The exact catalog
        # and unit authorities still perform the final equality check before
        # dispatch; this seam only rejects malformed or non-canonical input.
        for name in (
            "binding_manifest_hash",
            "binding_catalog_manifest_hash",
            "unit_input_hash",
            "unit_dataset_manifest_hash",
            "unit_set_hash",
            "model_binding_requirements_hash",
            "validation_context_hash",
            "request_body_hash",
            "retry_policy_hash",
            "evidence_contract_hash",
            "output_contract_hash",
        ):
            _require_hash(getattr(prepared, name), name)
        for name in (
            "binding_catalog_version",
            "unit_ref",
            "action_id",
            "transport_id",
            "provider",
            "operation",
        ):
            if not _REF_RE.fullmatch(str(getattr(prepared, name) or "")):
                raise RoutingProviderBindingError(
                    f"prepared routing provider {name} is invalid"
                )
        for name, maximum in (
            ("timeout_ms", 900_000),
            ("credit_ceiling_microunits", 100_000_000),
            ("max_results", 10_000),
        ):
            number = getattr(prepared, name)
            if type(number) is not int or not 1 <= number <= maximum:
                raise RoutingProviderBindingError(
                    f"prepared routing provider {name} is invalid"
                )
        return prepared

    def authorization_projection(self) -> Mapping[str, Any]:
        data = asdict(self)
        data.pop("payload")
        data.pop("validation_context")
        return data


class ReviewedDeeplineActionCompiler:
    """Compile one exact action from signed binding and unit authorities."""

    def __init__(
        self,
        *,
        binding_catalog: VerifiedRoutingBindingCatalog,
        unit_dataset: VerifiedRoutingUnitDataset,
    ) -> None:
        self.binding_catalog = binding_catalog
        self.unit_dataset = unit_dataset

    def prepare(
        self,
        *,
        binding: ProviderBindingIdentity,
        unit_ref: str,
        authorization_credit_microunits: int,
        authorization_timeout_ms: int,
        expected_model_binding_requirements_hash: str,
        phase: str = "initial",
        execution_mode: str = "measured_lab",
    ) -> PreparedRoutingProviderCall:
        manifest = self.binding_catalog.resolve(binding)
        policy = DEEPLINE_ACTION_POLICIES[manifest.action_id]
        if (
            _require_hash(
                expected_model_binding_requirements_hash,
                "model binding requirements hash",
            )
            != manifest.model_binding_requirements_hash
        ):
            raise RoutingProviderBindingError(
                "routing model binding requirements differ from signed runtime catalog"
            )
        if manifest.action_id == "builtwith_domain_lookup" and phase != "conditional_confirmation":
            raise RoutingProviderBindingError(
                "routing BuiltWith action is confirmation-only"
            )
        if manifest.action_id == "builtwith_domain_lookup" and execution_mode == "measured_lab":
            # The upstream action has usage-calculated billing and no
            # provider-enforced pre-call cap. A local reservation cannot stop
            # an unbounded provider charge, so only offline replay may use the
            # confirmation validator.
            raise RoutingProviderBindingError(
                "routing BuiltWith measured execution has no provider-enforced cost cap"
            )
        if (
            manifest.action_id == "bloomberry_get_tech_stack_changes"
            and execution_mode == "measured_lab"
        ):
            # The current action has no domain request field. Post-filtering a
            # global result cannot satisfy the model's domain-scoped request
            # contract and would spend 4.21 credits on an ineligible call.
            raise RoutingProviderBindingError(
                "routing Bloomberry technology changes have no domain-scoped action"
            )
        unit, unit_input_hash = self.unit_dataset.resolve(unit_ref)
        payload = dict(manifest.input_constants)
        payload.update(policy.fixed_inputs)
        validation_context: dict[str, Any] = {}
        for target, source in manifest.input_projection.items():
            if source in unit and unit[source] is not None:
                if target in policy.validation_context_fields:
                    validation_context[target] = unit[source]
                else:
                    payload[target] = unit[source]
        payload = _normalize_action_payload(policy, payload, max_results=manifest.max_results)
        _validate_action_context(policy, payload, validation_context)
        timeout_ms = min(manifest.timeout_ms, authorization_timeout_ms)
        credit = min(manifest.credit_ceiling_microunits, authorization_credit_microunits)
        if timeout_ms < 1 or credit < 1:
            raise RoutingProviderBindingError("routing action has no authorized time or credit")
        body = {
            "provider": policy.provider,
            "operation": policy.operation,
            "payload": payload,
        }
        return PreparedRoutingProviderCall(
            binding=binding,
            binding_manifest_hash=binding.manifest_hash,
            binding_catalog_manifest_hash=self.binding_catalog.manifest_hash,
            binding_catalog_version=self.binding_catalog.catalog_version,
            unit_ref=unit_ref,
            unit_input_hash=unit_input_hash,
            unit_dataset_manifest_hash=self.unit_dataset.manifest_hash,
            unit_set_hash=self.unit_dataset.unit_set_hash,
            model_binding_requirements_hash=manifest.model_binding_requirements_hash,
            action_id=manifest.action_id,
            transport_id=manifest.transport_id,
            provider=policy.provider,
            operation=policy.operation,
            payload=payload,
            validation_context=validation_context,
            validation_context_hash=sha256_json(
                {
                    "schema_version": "leadpoet.routing_validation_context.v1",
                    "action_id": manifest.action_id,
                    "context": validation_context,
                }
            ),
            request_body_hash=sha256_json(body),
            timeout_ms=timeout_ms,
            credit_ceiling_microunits=credit,
            max_results=manifest.max_results,
            retry_policy_hash=manifest.retry_policy_hash,
            evidence_contract_hash=manifest.evidence_contract_hash,
            output_contract_hash=manifest.output_contract_hash,
        )

    @staticmethod
    def broker_request(
        *,
        prepared: PreparedRoutingProviderCall,
        experiment_hash: str,
        dispatch_job_id: str,
        variant_id: str,
        attempt_number: int,
        core_request_fingerprint: str,
        authorization_hash: str,
        authorization_proof_hash: str,
    ) -> Mapping[str, Any]:
        """Build the transport request bound to the protected dispatch job.

        The admission and authorization jobs are separate durable identities
        and must never be copied into the provider request. The caller derives
        this dispatch ID from the complete signed authorization proof.
        """
        for name, value in (
            ("experiment_hash", experiment_hash),
            ("core_request_fingerprint", core_request_fingerprint),
            ("authorization_hash", authorization_hash),
            ("authorization_proof_hash", authorization_proof_hash),
        ):
            _require_hash(value, name)
        if type(attempt_number) is not int or attempt_number < 0:
            raise RoutingProviderBindingError("routing broker attempt is invalid")
        if not isinstance(dispatch_job_id, str) or not _REF_RE.fullmatch(
            dispatch_job_id
        ):
            raise RoutingProviderBindingError(
                "routing broker dispatch job is invalid"
            )
        body = {
            "provider": prepared.provider,
            "operation": prepared.operation,
            "payload": dict(prepared.payload),
        }
        if sha256_json(body) != prepared.request_body_hash:
            raise RoutingProviderBindingError("routing prepared request body changed")
        logical_operation_id = routing_provider_logical_operation_id_v2(
            experiment_hash=experiment_hash,
            variant_id=variant_id,
            unit_ref=prepared.unit_ref,
            tool_id=prepared.binding.tool_id,
            attempt=attempt_number,
            core_request_fingerprint=core_request_fingerprint,
            request_body_hash=prepared.request_body_hash,
        )
        return {
            "schema_version": PROVIDER_BROKER_SCHEMA_VERSION,
            "logical_operation_id": logical_operation_id,
            "job_id": dispatch_job_id,
            "purpose": DEEPLINE_PROVIDER_PURPOSE,
            "provider_id": prepared.transport_id,
            "attempt_number": attempt_number,
            "method": "POST",
            "url": f"{DEEPLINE_BASE_URL}/api/v2/integrations/{prepared.action_id}/execute",
            "headers": {
                "Content-Type": "application/json",
                "x-deepline-execute-response-intent": "raw",
            },
            "body_b64": base64.b64encode(
                json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).decode("ascii"),
            "timeout_ms": prepared.timeout_ms,
            "retry_policy_hash": prepared.retry_policy_hash,
            "routing_authorization": {
                "authorization_hash": authorization_hash,
                "authorization_proof_hash": authorization_proof_hash,
                "request_body_hash": prepared.request_body_hash,
                "action_id": prepared.action_id,
                "credit_cap_microunits": prepared.credit_ceiling_microunits,
                "timeout_ms": prepared.timeout_ms,
            },
        }

    @staticmethod
    def project_result(
        *,
        prepared: PreparedRoutingProviderCall,
        broker_request: Mapping[str, Any],
        broker_result: Mapping[str, Any],
        core_request_fingerprint: str,
    ) -> Mapping[str, Any]:
        policy = DEEPLINE_ACTION_POLICIES[prepared.action_id]
        if broker_result.get("terminal_status") != "authenticated_response":
            return _projected_failure(
                prepared, core_request_fingerprint, broker_result, billing_state="uncertain"
            )
        attempt = broker_result.get("transport_attempt")
        if not isinstance(attempt, Mapping):
            raise RoutingProviderBindingError("routing broker transport attempt is missing")
        expected_identity = {
            "logical_operation_id": broker_request["logical_operation_id"],
            "job_id": broker_request["job_id"],
            "purpose": DEEPLINE_PROVIDER_PURPOSE,
            "provider_id": prepared.transport_id,
            "attempt_number": broker_request["attempt_number"],
            "method": "POST",
            "timeout_ms": prepared.timeout_ms,
            "retry_policy_hash": prepared.retry_policy_hash,
        }
        if any(attempt.get(key) != value for key, value in expected_identity.items()):
            raise RoutingProviderBindingError("routing broker transport identity differs")
        http_status = broker_result.get("http_status")
        if type(http_status) is not int:
            raise RoutingProviderBindingError("routing Deepline HTTP status is invalid")
        try:
            raw = base64.b64decode(str(broker_result.get("body_b64") or ""), validate=True)
            if len(raw) > 8 * 1024 * 1024:
                raise ValueError("oversize")
            response = json.loads(raw)
        except Exception as exc:
            raise RoutingProviderBindingError("routing Deepline response is malformed") from exc
        if not isinstance(response, Mapping):
            raise RoutingProviderBindingError("routing Deepline response is not an object")
        credit, billing_state = _billing(response, prepared)
        if credit > prepared.credit_ceiling_microunits:
            raise RoutingProviderBindingError("routing Deepline billing exceeds signed ceiling")
        latency_ms = _attempt_latency_ms(attempt)
        if not 200 <= http_status < 300:
            return _projected_failure(
                prepared,
                core_request_fingerprint,
                {"http_status": http_status, "billing": response.get("billing")},
                billing_state=billing_state,
                credit=credit,
                latency_ms=latency_ms,
            )
        rows = _extract_rows(response, policy.result_paths)
        qualifying = _qualifying_direct_rows(
            prepared=prepared,
            response=response,
            rows=rows,
        )
        evidence = [
            _project_evidence_row(row, policy.evidence_fields)
            for row in qualifying[: prepared.max_results]
        ]
        return {
            "outcome": "verified" if evidence else "source_miss",
            "evidence_hash": sha256_json(
                {
                    "schema_version": "leadpoet.routing_deepline_evidence_projection.v1",
                    "action_id": prepared.action_id,
                    "evidence_contract_hash": prepared.evidence_contract_hash,
                    "rows": evidence,
                }
            ),
            "credit_microunits": credit,
            "latency_ms": latency_ms,
            "billing_state": billing_state,
            "binding_id": prepared.binding.binding_id,
            "provider_id": prepared.binding.provider_id,
            "tool_id": prepared.binding.tool_id,
            "request_fingerprint": core_request_fingerprint,
        }


@dataclass(frozen=True)
class PreparedRoutingProviderWorkflow:
    """A signed composite route prepared without dispatching a provider.

    The workflow input is derived only from the immutable signed unit dataset.
    It is deliberately not a Deepline request body.  A later reviewed
    dispatcher may use the ordered action IDs and branch projection after it
    has attached per-action authorization and durable receipts.
    """

    binding: ProviderBindingIdentity
    binding_manifest_hash: str
    binding_catalog_manifest_hash: str
    binding_catalog_version: str
    unit_ref: str
    unit_input_hash: str
    unit_dataset_manifest_hash: str
    unit_set_hash: str
    model_binding_requirements_hash: str
    workflow_id: str
    workflow_manifest_hash: str
    workflow_input: Mapping[str, Any]
    workflow_input_hash: str
    ordered_actions: tuple[str, ...]
    branch_optional_actions: tuple[str, ...]
    max_calls: int
    timeout_ms: int
    credit_ceiling_microunits: int
    max_results: int
    retry_policy_hash: str
    evidence_contract_hash: str
    output_contract_hash: str

    def authorization_projection(self) -> Mapping[str, Any]:
        data = asdict(self)
        data.pop("workflow_input")
        return data


class ReviewedDeeplineWorkflowCompiler:
    """Compile one exact, code-reviewed PredictLeads workflow.

    This compiler prepares a route only.  It does not call Deepline and does
    not produce an endpoint, HTTP body, or caller-selected branch.
    """

    def __init__(
        self,
        *,
        binding_catalog: VerifiedRoutingBindingCatalog,
        unit_dataset: VerifiedRoutingUnitDataset,
    ) -> None:
        self.binding_catalog = binding_catalog
        self.unit_dataset = unit_dataset

    def prepare(
        self,
        *,
        binding: ProviderBindingIdentity,
        unit_ref: str,
        authorization_credit_microunits: int,
        authorization_timeout_ms: int,
        expected_model_binding_requirements_hash: str,
        execution_mode: str = "measured_lab",
    ) -> PreparedRoutingProviderWorkflow:
        manifest = self.binding_catalog.resolve(binding)
        if manifest.execution_kind != "composite_workflow":
            raise RoutingProviderBindingError(
                "routing binding is not a composite workflow"
            )
        workflow_id = manifest.workflow_id
        workflow_hash = manifest.workflow_manifest_hash
        if not isinstance(workflow_id, str) or not isinstance(workflow_hash, str):
            raise RoutingProviderBindingError("routing workflow identity is incomplete")
        try:
            reviewed = workflow_manifest(workflow_id)
        except PredictLeadsWorkflowError as exc:
            raise RoutingProviderBindingError("routing workflow is not reviewed") from exc
        if workflow_hash != reviewed.manifest_hash:
            raise RoutingProviderBindingError(
                "routing workflow manifest hash differs from reviewed workflow"
            )
        if binding.tool_id not in _COMPOSITE_SOURCE_WORKFLOWS:
            raise RoutingProviderBindingError(
                "routing binding is not a reviewed composite source"
            )
        if _COMPOSITE_SOURCE_WORKFLOWS[binding.tool_id] != workflow_id:
            raise RoutingProviderBindingError(
                "routing workflow does not match model source tool"
            )
        if (
            tuple(item.action_id for item in reviewed.ordered_actions)
            != ROUTE_ACTION_ORDER[workflow_id]
            or reviewed.max_calls != ROUTE_MAX_CALLS[workflow_id]
            or reviewed.credit_ceiling_microcredits
            != ROUTE_CREDIT_CEILINGS[workflow_id]
        ):
            raise RoutingProviderBindingError(
                "routing workflow manifest has an unexpected action or cap"
            )
        if (
            type(manifest.max_results) is not int
            or type(manifest.timeout_ms) is not int
            or type(manifest.credit_ceiling_microunits) is not int
            or manifest.max_results != 1
            or manifest.timeout_ms != reviewed.timeout_ms
            or manifest.credit_ceiling_microunits
            != ROUTE_CREDIT_CEILINGS[workflow_id]
        ):
            raise RoutingProviderBindingError(
                "routing composite workflow limits differ from reviewed workflow"
            )
        if (
            execution_mode == "measured_lab"
            and binding.tool_id in MEASURED_UNAVAILABLE_COMPOSITE_SOURCE_TOOLS
        ):
            raise RoutingProviderBindingError(
                "routing PredictLeads news workflow has no provider-enforced cost cap"
            )
        if type(authorization_timeout_ms) is not int or authorization_timeout_ms < reviewed.timeout_ms:
            raise RoutingProviderBindingError(
                "routing workflow authorization timeout is below reviewed route timeout"
            )
        if (
            type(authorization_credit_microunits) is not int
            or authorization_credit_microunits < ROUTE_CREDIT_CEILINGS[workflow_id]
        ):
            raise RoutingProviderBindingError(
                "routing workflow authorization credit is below reviewed route cap"
            )
        if (
            _require_hash(
                expected_model_binding_requirements_hash,
                "model binding requirements hash",
            )
            != manifest.model_binding_requirements_hash
        ):
            raise RoutingProviderBindingError(
                "routing model binding requirements differ from signed runtime catalog"
            )
        unit, unit_input_hash = self.unit_dataset.resolve(unit_ref)
        projection = manifest.input_projection
        if (
            set(projection) != _COMPOSITE_INPUT_FIELDS[workflow_id]
            or manifest.input_constants
            or any(not _REF_RE.fullmatch(value) for value in projection.values())
        ):
            raise RoutingProviderBindingError(
                "routing composite workflow input projection is invalid"
            )
        workflow_input: dict[str, Any] = {}
        for target, source in projection.items():
            if source not in unit or unit[source] is None:
                raise RoutingProviderBindingError(
                    "routing composite workflow input is absent from signed unit"
                )
            workflow_input[target] = unit[source]
        try:
            normalized_input = validate_workflow_input(workflow_id, workflow_input)
        except PredictLeadsWorkflowError as exc:
            raise RoutingProviderBindingError(
                "routing composite workflow input is invalid"
            ) from exc
        return PreparedRoutingProviderWorkflow(
            binding=binding,
            binding_manifest_hash=binding.manifest_hash,
            binding_catalog_manifest_hash=self.binding_catalog.manifest_hash,
            binding_catalog_version=self.binding_catalog.catalog_version,
            unit_ref=unit_ref,
            unit_input_hash=unit_input_hash,
            unit_dataset_manifest_hash=self.unit_dataset.manifest_hash,
            unit_set_hash=self.unit_dataset.unit_set_hash,
            model_binding_requirements_hash=manifest.model_binding_requirements_hash,
            workflow_id=workflow_id,
            workflow_manifest_hash=workflow_hash,
            workflow_input=normalized_input,
            workflow_input_hash=sha256_json(
                {
                    "schema_version": "leadpoet.routing_workflow_input.v1",
                    "workflow_id": workflow_id,
                    "workflow_manifest_hash": workflow_hash,
                    "input": normalized_input,
                }
            ),
            ordered_actions=tuple(item.action_id for item in reviewed.ordered_actions),
            branch_optional_actions=tuple(reviewed.branch_optional_actions),
            max_calls=reviewed.max_calls,
            timeout_ms=reviewed.timeout_ms,
            credit_ceiling_microunits=reviewed.credit_ceiling_microcredits,
            max_results=manifest.max_results,
            retry_policy_hash=manifest.retry_policy_hash,
            evidence_contract_hash=manifest.evidence_contract_hash,
            output_contract_hash=manifest.output_contract_hash,
        )


def _normalize_action_payload(
    policy: DeeplineActionPolicy,
    payload: Mapping[str, Any],
    *,
    max_results: int,
) -> dict[str, Any]:
    if not set(payload).issubset(policy.allowed_input_fields | frozenset(policy.fixed_inputs)):
        raise RoutingProviderBindingError("routing Deepline payload field is not reviewed")
    value = dict(payload)
    # Model vocabulary is normalized into exact provider field names here.
    aliases = {
        "technology_name": "vendor_name",
        "technology_category": "category",
        "technology_upper_category": "upper_level_category",
        "domain": "company_id_or_domain" if policy.action_id.startswith("predictleads_") else "domain",
        "begin_date": (
            "first_seen_at_from" if policy.action_id.startswith("predictleads_") else "begin_date"
        ),
        "end_date": (
            "first_seen_at_until" if policy.action_id.startswith("predictleads_") else "end_date"
        ),
        "countries": (
            "company_country"
            if policy.action_id == "bloomberry_get_tech_stack_changes"
            else "countries"
        ),
    }
    normalized: dict[str, Any] = {}
    for key, item in value.items():
        target = aliases.get(key, key)
        normalized[target] = item
    limit_field = "per_page" if policy.action_id == "podscan_episodes_search" else "limit"
    if limit_field in normalized:
        normalized[limit_field] = min(
            int(normalized[limit_field]), policy.maximum_results
        )
    for requirement in policy.required_any:
        targets = {aliases.get(item, item) for item in requirement}
        if not any(
            normalized.get(item) is not None
            and normalized.get(item) != ""
            and normalized.get(item) != ()
            and normalized.get(item) != []
            for item in targets
        ):
            raise RoutingProviderBindingError("routing Deepline required input is missing")
    domain_key = "company_id_or_domain" if policy.action_id.startswith("predictleads_") else "domain"
    if domain_key in normalized:
        domain = str(normalized[domain_key]).strip().lower()
        if not _DOMAIN_RE.fullmatch(domain):
            raise RoutingProviderBindingError("routing Deepline domain is invalid")
        normalized[domain_key] = domain
    for key in ("begin_date", "end_date", "first_seen_at_from", "first_seen_at_until"):
        if key in normalized and not _DATE_RE.fullmatch(str(normalized[key])):
            raise RoutingProviderBindingError("routing Deepline date is invalid")
    for key in ("countries", "region_countries"):
        if key in normalized and not _COUNTRIES_RE.fullmatch(str(normalized[key])):
            raise RoutingProviderBindingError("routing Deepline country filter is invalid")
    for key, item in normalized.items():
        if isinstance(item, str) and key not in {
            "domain", "company_id_or_domain", "begin_date", "end_date",
            "first_seen_at_from", "first_seen_at_to", "countries", "region_countries",
        } and not _SAFE_TEXT_RE.fullmatch(item):
            raise RoutingProviderBindingError("routing Deepline text input is invalid")
    if policy.action_id == "builtwith_domain_lookup":
        first_from = normalized.pop("first_detected_from", None)
        first_until = normalized.pop("first_detected_until", None)
        last_from = normalized.pop("last_detected_from", None)
        last_until = normalized.pop("last_detected_until", None)
        if (first_from is None) != (first_until is None) or (last_from is None) != (last_until is None):
            raise RoutingProviderBindingError(
                "routing BuiltWith detection range is incomplete"
            )
        if first_from is not None:
            if not _DATE_RE.fullmatch(str(first_from)) or not _DATE_RE.fullmatch(str(first_until)):
                raise RoutingProviderBindingError("routing BuiltWith detection date is invalid")
            normalized["first_detected_range"] = {"from": first_from, "to": first_until}
        if last_from is not None:
            if not _DATE_RE.fullmatch(str(last_from)) or not _DATE_RE.fullmatch(str(last_until)):
                raise RoutingProviderBindingError("routing BuiltWith detection date is invalid")
            normalized["last_detected_range"] = {"from": last_from, "to": last_until}
        # These privacy controls cannot be relaxed by a signed runtime entry.
        normalized.update(
            {
                "no_pii": True,
                "no_meta": True,
                "no_attr": True,
                "hide_text": True,
                "trust": False,
            }
        )
    if policy.action_id == "bloomberry_get_tech_stack_changes":
        if ("vendor_name" in normalized) == ("category" in normalized):
            raise RoutingProviderBindingError(
                "routing Bloomberry change requires exactly one technology selector"
            )
    return normalized


def _validate_action_context(
    policy: DeeplineActionPolicy,
    payload: Mapping[str, Any],
    context: Mapping[str, Any],
) -> None:
    """Reject paid requests that cannot pass the direct-source evidence gate."""

    action = policy.action_id
    required_context: tuple[str, ...]
    if action == "bloomberry_search_job_postings":
        required_context = ("minimum_date", "maximum_date")
    elif action == "bloomberry_get_tech_stack_changes":
        required_context = ("expected_domain", "minimum_date", "maximum_date")
    elif action == "podscan_episodes_search":
        required_context = ("minimum_date", "maximum_date")
        expected = str(
            context.get("expected_person") or context.get("expected_company") or ""
        ).strip()
        query = str(payload.get("query") or "").strip()
        if (
            not expected
            or len(query) > 256
            or query.count('"') < 2
            or f'"{expected.casefold()}"' not in query.casefold()
        ):
            raise RoutingProviderBindingError(
                "routing Podscan query is not bound to the signed subject"
            )
    elif action == "predictleads_company_job_openings":
        required_context = ("role_keyword", "minimum_date", "maximum_date")
    elif action == "predictleads_company_financing_events":
        required_context = ("minimum_date", "maximum_date")
    elif action == "builtwith_domain_lookup":
        required_context = ("requested_technology", "parent_intent_event_hash")
        if not (
            isinstance(payload.get("first_detected_range"), Mapping)
            or isinstance(payload.get("last_detected_range"), Mapping)
        ):
            raise RoutingProviderBindingError(
                "routing BuiltWith detection range is required"
            )
    else:
        raise RoutingProviderBindingError("routing direct action has no reviewed context gate")
    if any(context.get(field) in {None, ""} for field in required_context):
        raise RoutingProviderBindingError("routing action validation context is incomplete")
    if "expected_domain" in required_context:
        if not _DOMAIN_RE.fullmatch(str(context["expected_domain"]).strip().lower()):
            raise RoutingProviderBindingError("routing validation domain is invalid")
    if "parent_intent_event_hash" in required_context:
        _require_hash(context["parent_intent_event_hash"], "parent intent event hash")
    if "minimum_date" in required_context:
        minimum = str(context["minimum_date"])
        maximum = str(context["maximum_date"])
        if (
            not _DATE_RE.fullmatch(minimum)
            or not _DATE_RE.fullmatch(maximum)
            or minimum > maximum
        ):
            raise RoutingProviderBindingError("routing validation date range is invalid")


def _extract_rows(root: Mapping[str, Any], paths: Sequence[Sequence[str]]) -> list[Any]:
    for path in paths:
        node: Any = root
        for part in path:
            if not isinstance(node, Mapping) or part not in node:
                node = None
                break
            node = node[part]
        if isinstance(node, list):
            return node
    return []


def _qualifying_direct_rows(
    *,
    prepared: PreparedRoutingProviderCall,
    response: Mapping[str, Any],
    rows: Sequence[Any],
) -> list[Mapping[str, Any]]:
    """Apply the model-exported direct-source evidence gates.

    This function is deliberately conservative.  Unknown output shapes are a
    source miss, never verified evidence.  Composite PredictLeads workflows
    never reach this function because catalog admission rejects them.
    """

    context = {**dict(prepared.payload), **dict(prepared.validation_context)}
    action = prepared.action_id
    candidates = [row for row in rows if isinstance(row, Mapping)]
    accepted: list[Mapping[str, Any]] = []
    included = _included_resources(response)
    for row in candidates:
        if action == "bloomberry_search_job_postings":
            domain = str(context.get("domain") or "").casefold()
            title = _flatten_text(
                row.get("title"), row.get("normalized_job_title")
            )
            keyword = context.get("keyword") or context.get("normalized_job_titles") or ""
            inactive = row.get("inactive")
            company = row.get("company")
            row_domain = str(
                row.get("company_domain")
                or row.get("domain")
                or (company.get("domain") if isinstance(company, Mapping) else "")
                or ""
            ).casefold()
            if (
                row_domain != domain
                or not (
                    inactive in {0, False, "0"}
                    or row.get("active") is True
                    or str(row.get("status") or "").casefold() in {"active", "open"}
                )
                or not _role_relevant(title, keyword)
                or not _date_within(
                    row.get("snapshot_date") or row.get("created_at"),
                    context.get("minimum_date") or context.get("begin_date"),
                    context.get("maximum_date") or context.get("end_date"),
                )
                or not _public_source_url(
                    row.get("displayed_url") or row.get("url"),
                    forbidden_hosts={"bloomberry.com", "revealera.com", "api.revealera.com"},
                )
            ):
                continue
        elif action == "bloomberry_get_tech_stack_changes":
            requested_vendor = " ".join(
                str(context.get("vendor_name") or "").casefold().split()
            )
            requested_category = " ".join(
                str(context.get("category") or "").casefold().split()
            )
            actual_vendor = " ".join(
                str(row.get("vendor_name") or row.get("technology_name") or "")
                .casefold()
                .split()
            )
            actual_category = " ".join(
                str(row.get("category") or row.get("technology_category") or "")
                .casefold()
                .split()
            )
            if (
                not (
                    (requested_vendor and requested_vendor == actual_vendor)
                    or (requested_category and requested_category == actual_category)
                )
                or
                str(row.get("company_domain") or row.get("domain") or "").casefold()
                != str(context.get("expected_domain") or "").casefold()
                or not _date_within(
                    row.get("change_date") or row.get("date") or row.get("first_seen"),
                    context.get("minimum_date") or context.get("begin_date"),
                    context.get("maximum_date") or context.get("end_date"),
                )
                or (
                    context.get("expected_country")
                    and str(row.get("company_country") or row.get("country") or "").upper()
                    != str(context["expected_country"]).upper()
                )
                or not str(row.get("vendor_source") or "").strip()
                or not _public_source_url(row.get("vendor_url"))
            ):
                continue
        elif action == "podscan_episodes_search":
            expected = str(
                context.get("expected_person") or context.get("expected_company") or ""
            ).strip()
            relationship = _searchable_text(
                row.get("guest_name"),
                row.get("guest_company"),
                row.get("guests"),
                (row.get("metadata") or {}).get("guests")
                if isinstance(row.get("metadata"), Mapping)
                else None,
            )
            highlight = _searchable_text(
                row.get("search_highlight")
                or row.get("_search_highlight")
                or row.get("transcript_highlight")
                or row.get("highlight")
                or ""
            )
            if (
                not expected
                or _searchable_text(expected) not in relationship
                or _searchable_text(expected) not in highlight
                or not _date_within(
                    row.get("posted_at")
                    or row.get("published_at")
                    or row.get("published_date"),
                    context.get("minimum_date"),
                    context.get("maximum_date"),
                )
                or not _public_source_url(
                    row.get("episode_url") or row.get("url"),
                    forbidden_hosts={"podscan.fm", "www.podscan.fm"},
                )
            ):
                continue
        elif action in {
            "predictleads_company_financing_events",
            "predictleads_company_job_openings",
        }:
            attributes = row.get("attributes")
            relationships = row.get("relationships")
            if not isinstance(attributes, Mapping) or not isinstance(relationships, Mapping):
                continue
            requested_domain = str(
                context.get("company_id_or_domain") or context.get("domain") or ""
            ).casefold()
            company_relationship = relationships.get("company")
            company_data = (
                company_relationship.get("data")
                if isinstance(company_relationship, Mapping)
                else None
            )
            company_id = (
                str(company_data.get("id") or "")
                if isinstance(company_data, Mapping)
                else ""
            )
            company = included.get(("company", company_id))
            company_attributes = (
                company.get("attributes") if isinstance(company, Mapping) else None
            )
            if (
                not isinstance(company_attributes, Mapping)
                or str(company_attributes.get("domain") or "").casefold()
                != requested_domain
            ):
                continue
            if action == "predictleads_company_job_openings":
                status = str(attributes.get("status") or "").casefold()
                title_text = _flatten_text(
                    attributes.get("title"),
                    attributes.get("normalized_title"),
                    attributes.get("description"),
                )
                role = str(context.get("role_keyword") or "")
                # PredictLeads' declared response contract represents an open
                # posting with ``status: null``.  ``closed`` is the only
                # non-null status in the live schema.  Do not rely on the
                # request-side ``active_only``/``not_closed`` filters alone:
                # provider output is untrusted and must prove a recent
                # last-seen date before it becomes intent evidence.
                last_seen = attributes.get("last_seen_at")
                observation_cutoff = str(
                    context.get("maximum_date")
                    or context.get("first_seen_at_until")
                    or ""
                )[:10]
                observation_floor = (
                    date.fromisoformat(observation_cutoff) - timedelta(days=5)
                ).isoformat()
                if (
                    row.get("type") != "job_opening"
                    or status not in {"", "active", "open", "published"}
                    or not _date_within(
                        last_seen,
                        observation_floor,
                        observation_cutoff,
                    )
                    or not role
                    or not _role_relevant(title_text, role)
                    or not _date_within(
                        attributes.get("posted_at") or attributes.get("first_seen_at"),
                        context.get("minimum_date") or context.get("first_seen_at_from"),
                        context.get("maximum_date") or context.get("first_seen_at_until"),
                    )
                    or not _public_source_url(attributes.get("url"))
                ):
                    continue
            else:
                source = _first_public_url(
                    attributes.get("source_urls"),
                )
                effective_date = (
                    attributes.get("effective_date")
                    or attributes.get("announced_on")
                    or attributes.get("first_seen_at")
                )
                if (
                    row.get("type") != "financing_event"
                    or not _date_within(
                        effective_date,
                        context.get("minimum_date") or context.get("first_seen_at_from"),
                        context.get("maximum_date") or context.get("first_seen_at_until"),
                    )
                    or not source
                ):
                    continue
        elif action == "builtwith_domain_lookup":
            expected_domain = str(context.get("domain") or "").casefold()
            requested_technology = str(context.get("requested_technology") or "").strip()
            if (
                not _HASH_RE.fullmatch(str(context.get("parent_intent_event_hash") or ""))
                or str(row.get("Lookup") or "").casefold() != expected_domain
                or not requested_technology
                or not _builtwith_technology_history_matches(
                    row,
                    expected_domain=expected_domain,
                    technology=requested_technology,
                    first_detected_range=context.get("first_detected_range"),
                    last_detected_range=context.get("last_detected_range"),
                )
            ):
                continue
        else:
            # No generic "row exists" path is permitted.
            continue
        accepted.append(row)
        if len(accepted) >= prepared.max_results:
            break
    return accepted


def _included_resources(response: Mapping[str, Any]) -> dict[tuple[str, str], Mapping[str, Any]]:
    nodes: Any = response
    for path in (("result", "data", "included"), ("result", "included"), ("included",)):
        node: Any = response
        for part in path:
            if not isinstance(node, Mapping) or part not in node:
                node = None
                break
            node = node[part]
        if isinstance(node, list):
            nodes = node
            break
    if not isinstance(nodes, list):
        return {}
    result: dict[tuple[str, str], Mapping[str, Any]] = {}
    for item in nodes:
        if isinstance(item, Mapping):
            key = (str(item.get("type") or ""), str(item.get("id") or ""))
            if all(key):
                result[key] = item
    return result


def _role_relevant(text: str, query: Any) -> bool:
    normalized_text = " ".join(str(text).casefold().split())
    raw_queries = (
        list(query)
        if isinstance(query, Sequence) and not isinstance(query, (str, bytes, bytearray))
        else re.split(r"\s+OR\s+|;", str(query), flags=re.IGNORECASE)
    )
    phrases = [" ".join(str(item).casefold().split()) for item in raw_queries if str(item).strip()]
    return bool(phrases) and any(phrase in normalized_text for phrase in phrases)


def _date_within(value: Any, minimum: Any, maximum: Any) -> bool:
    date = str(value or "")[:10]
    if not _DATE_RE.fullmatch(date):
        return False
    if minimum and (not _DATE_RE.fullmatch(str(minimum)) or date < str(minimum)):
        return False
    if maximum and (not _DATE_RE.fullmatch(str(maximum)) or date > str(maximum)):
        return False
    return True


def _public_source_url(value: Any, *, forbidden_hosts: set[str] | None = None) -> bool:
    from urllib.parse import urlsplit

    text = str(value or "").strip()
    try:
        parsed = urlsplit(text)
    except Exception:
        return False
    host = str(parsed.hostname or "").casefold()
    return bool(
        parsed.scheme == "https"
        and host
        and "." in host
        and host not in (forbidden_hosts or set())
        and parsed.username is None
        and parsed.password is None
    )


def _first_public_url(*values: Any) -> str:
    for value in values:
        candidates = value if isinstance(value, list) else [value]
        for candidate in candidates:
            if _public_source_url(candidate):
                return str(candidate)
    return ""


def _flatten_text(*values: Any) -> str:
    parts: list[str] = []
    for value in values:
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, Mapping):
            parts.extend(_flatten_text(*value.values()).split(" "))
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            parts.extend(_flatten_text(*value).split(" "))
    return " ".join(part for part in parts if part)


def _searchable_text(*values: Any) -> str:
    """Normalize bounded provider text for exact subject comparisons."""

    import html

    flattened = html.unescape(_flatten_text(*values))
    without_markup = re.sub(r"</?[A-Za-z][^>]{0,127}>", " ", flattened)
    return " ".join(without_markup.casefold().split())


def _builtwith_technology_history_matches(
    row: Mapping[str, Any],
    *,
    expected_domain: str,
    technology: str,
    first_detected_range: Any,
    last_detected_range: Any,
) -> bool:
    result = row.get("Result")
    paths = result.get("Paths") if isinstance(result, Mapping) else None
    if not isinstance(paths, list):
        return False
    wanted = " ".join(technology.casefold().split())
    for path in paths:
        if not isinstance(path, Mapping):
            continue
        domain = str(path.get("Domain") or "").casefold()
        if domain != expected_domain:
            continue
        technologies = path.get("Technologies")
        if not isinstance(technologies, list):
            continue
        for item in technologies:
            if not isinstance(item, Mapping):
                continue
            first_detected = str(item.get("FirstDetected") or "")[:10]
            last_detected = str(item.get("LastDetected") or "")[:10]
            if (
                " ".join(str(item.get("Name") or "").casefold().split()) == wanted
                and _date_in_range(first_detected, first_detected_range)
                and _date_in_range(last_detected, last_detected_range)
            ):
                return True
    return False


def _date_in_range(value: str, requested_range: Any) -> bool:
    if not _DATE_RE.fullmatch(value):
        return False
    if requested_range is None:
        return True
    if not isinstance(requested_range, Mapping) or set(requested_range) != {"from", "to"}:
        return False
    minimum = str(requested_range.get("from") or "")
    maximum = str(requested_range.get("to") or "")
    return bool(
        _DATE_RE.fullmatch(minimum)
        and _DATE_RE.fullmatch(maximum)
        and minimum <= value <= maximum
    )


def _path_value(row: Mapping[str, Any], path: str) -> Any:
    node: Any = row
    for part in path.split("."):
        if not isinstance(node, Mapping) or part not in node:
            return None
        node = node[part]
    return node


def _bounded_projection(value: Any, *, depth: int = 0) -> Any:
    if depth > 4:
        return "[depth-limit]"
    if value is None or type(value) in {bool, int, float}:
        return value
    if isinstance(value, str):
        return value[:1024]
    if isinstance(value, Mapping):
        return {
            str(key)[:128]: _bounded_projection(item, depth=depth + 1)
            for key, item in sorted(value.items())[:64]
            if str(key).lower() not in _SENSITIVE_KEYS
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_bounded_projection(item, depth=depth + 1) for item in value[:64]]
    return str(type(value).__name__)


def _project_evidence_row(row: Mapping[str, Any], fields: Sequence[str]) -> Mapping[str, Any]:
    projected = {
        field: _bounded_projection(_path_value(row, field))
        for field in fields
        if _path_value(row, field) is not None
    }
    return projected


def _billing(
    response: Mapping[str, Any], prepared: PreparedRoutingProviderCall
) -> tuple[int, str]:
    billing = response.get("billing")
    if not isinstance(billing, Mapping):
        return 0, "uncertain"
    raw = billing.get("credits_charged", billing.get("creditsCharged"))
    if raw is None:
        return 0, "uncertain"
    try:
        credits = Decimal(str(raw))
        if not credits.is_finite() or credits < 0:
            raise InvalidOperation
        microunits = int((credits * 1_000_000).to_integral_value(rounding=ROUND_CEILING))
    except (InvalidOperation, ValueError) as exc:
        raise RoutingProviderBindingError("routing Deepline billing is invalid") from exc
    return microunits, "known"


def _attempt_latency_ms(attempt: Mapping[str, Any]) -> int:
    try:
        from datetime import datetime

        started = datetime.fromisoformat(str(attempt["started_at"]).replace("Z", "+00:00"))
        completed = datetime.fromisoformat(str(attempt["completed_at"]).replace("Z", "+00:00"))
        value = int(max(0.0, (completed - started).total_seconds()) * 1000)
    except Exception as exc:
        raise RoutingProviderBindingError("routing broker latency evidence is invalid") from exc
    return value


def _projected_failure(
    prepared: PreparedRoutingProviderCall,
    core_request_fingerprint: str,
    evidence: Mapping[str, Any],
    *,
    billing_state: str,
    credit: int = 0,
    latency_ms: int = 0,
) -> Mapping[str, Any]:
    return {
        "outcome": "adapter_failure",
        "evidence_hash": sha256_json(
            {
                "schema_version": "leadpoet.routing_deepline_failure.v1",
                "action_id": prepared.action_id,
                "evidence": _bounded_projection(evidence),
            }
        ),
        "credit_microunits": credit,
        "latency_ms": latency_ms,
        "billing_state": billing_state,
        "binding_id": prepared.binding.binding_id,
        "provider_id": prepared.binding.provider_id,
        "tool_id": prepared.binding.tool_id,
        "request_fingerprint": core_request_fingerprint,
    }


__all__ = [
    "ROUTING_BINDING_CATALOG_ENV",
    "ROUTING_UNIT_DATASET_ENV",
    "ROUTING_BINDING_CATALOG_SCHEMA",
    "ROUTING_UNIT_DATASET_SCHEMA",
    "DEEPLINE_COMPILER_FAMILY",
    "DEEPLINE_ACTION_POLICIES",
    "EXPLICITLY_UNAVAILABLE_MODEL_TOOLS",
    "RoutingProviderBindingError",
    "DeeplineActionPolicy",
    "RoutingBindingManifest",
    "VerifiedRoutingBindingCatalog",
    "SignedRoutingBindingCatalogLoader",
    "VerifiedRoutingUnitDataset",
    "SignedRoutingUnitDatasetLoader",
    "PreparedRoutingProviderCall",
    "ReviewedDeeplineActionCompiler",
    "PreparedRoutingProviderWorkflow",
    "ReviewedDeeplineWorkflowCompiler",
    "MEASURED_UNAVAILABLE_COMPOSITE_SOURCE_TOOLS",
    "COMPOSITE_WORKFLOW_TOOL_IDS",
]
