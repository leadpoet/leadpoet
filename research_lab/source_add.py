"""SOURCE_ADD manifest, output, and credential validation contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import re
from typing import Any, Mapping, Sequence

SOURCE_ADD_OUTPUT_FIELDS: tuple[str, ...] = (
    "evidence_refs",
    "snapshot_refs",
    "content_hashes",
    "normalized_text_hashes",
    "metadata_refs",
)

RAW_OUTPUT_FIELDS: tuple[str, ...] = (
    "content",
    "raw_content",
    "html",
    "raw_html",
    "text",
    "page_text",
    "scraped_content",
    "body",
    "raw_response",
)

# These are defense-in-depth tripwires for declared schemas. The hard guarantee
# is the ref/hash-only adapter contract plus the disabled sandbox execution path.
RAW_CREDENTIAL_KEYS: tuple[str, ...] = (
    "access_key",
    "access_token",
    "api-key",
    "api_key",
    "apikey",
    "authorization",
    "client_secret",
    "credential",
    "credentials",
    "key",
    "password",
    "proxy-authorization",
    "private_key",
    "secret",
    "token",
    "credential_value",
    "raw_credential",
)

_SOURCE_ADD_CREDENTIAL_MARKERS: tuple[str, ...] = (
    "sk-or-",
    "openrouter_api_key",
    "openrouter_management_key",
    "raw_openrouter_key",
    "raw_secret",
    "service_role",
)
_SOURCE_ADD_CREDENTIAL_KEY_SUFFIXES: tuple[str, ...] = (
    "access_key",
    "access_token",
    "api_key",
    "apikey",
    "auth_token",
    "authorization",
    "client_secret",
    "credential",
    "credentials",
    "password",
    "private_key",
    "proxy_authorization",
    "secret",
    "secret_key",
    "subscription_key",
    "token",
)
_SOURCE_ADD_CREDENTIAL_KEY_COMPACT_SUFFIXES: tuple[str, ...] = (
    "accesskey",
    "accesstoken",
    "apikey",
    "apitoken",
    "authkey",
    "auth",
    "authtoken",
    "authorization",
    "clientkey",
    "clientcredential",
    "clientcredentials",
    "clientsecret",
    "clienttoken",
    "password",
    "credential",
    "credentials",
    "privatekey",
    "providerkey",
    "providersecret",
    "providertoken",
    "proxyauthorization",
    "refreshtoken",
    "secretkey",
    "subscriptionkey",
)
_SOURCE_ADD_CREDENTIAL_ASSIGNMENT_RE = re.compile(
    r"(?:^|[^A-Za-z0-9_])[\"']?"
    r"(?:key|token|password|credentials?|authorization|proxy-authorization|"
    r"[A-Za-z0-9_-]*(?:api|access|auth|client|private|provider|refresh|secret|subscription)"
    r"[A-Za-z0-9_-]*(?:key|token|secret|password|credentials?|auth|authorization)|"
    r"x[A-Za-z0-9_-]*(?:key|token|secret|auth|authorization))"
    r"[\"']?\s*(?:=|:|%3d)\s*[\"']?"
    r"[^\s\"'&,;}]{8,}"
    r"|\b(?:authorization|proxy-authorization)\s*:\s*"
    r"(?:bearer|basic|api(?:[\s_-]*key)?)\s+[^\s\"',;}]{8,}",
    re.IGNORECASE,
)
_SOURCE_ADD_CREDENTIAL_VALUE_RE = re.compile(
    r"\b(?:sk-[A-Za-z0-9_-]{12,}|AKIA[A-Z0-9]{16}|AIza[A-Za-z0-9_-]{30,})\b"
)


def source_add_text_contains_credential_material(value: str) -> bool:
    """Detect credential-bearing text that SOURCE_ADD must never persist."""

    text = str(value or "")
    lowered = text.lower()
    return (
        any(marker in lowered for marker in _SOURCE_ADD_CREDENTIAL_MARKERS)
        or bool(_SOURCE_ADD_CREDENTIAL_ASSIGNMENT_RE.search(text))
        or bool(_SOURCE_ADD_CREDENTIAL_VALUE_RE.search(text))
    )


def source_add_contains_credential_material(value: Any) -> bool:
    """Return whether any SOURCE_ADD request field carries credentials."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized_key = re.sub(
                r"(?<=[a-z0-9])(?=[A-Z])",
                "_",
                str(key).strip(),
            ).lower()
            tokenized_key = re.sub(r"[^a-z0-9]+", "_", normalized_key).strip(
                "_"
            )
            compact_key = re.sub(r"[^a-z0-9]+", "", normalized_key)
            if (
                normalized_key in RAW_CREDENTIAL_KEYS
                or tokenized_key == "key"
                or any(
                    compact_key == suffix or compact_key.endswith(suffix)
                    for suffix in _SOURCE_ADD_CREDENTIAL_KEY_COMPACT_SUFFIXES
                )
                or (
                    tokenized_key.endswith("_key")
                    and any(
                        marker in tokenized_key
                        for marker in (
                            "access",
                            "api",
                            "auth",
                            "private",
                            "secret",
                            "subscription",
                        )
                    )
                )
                or any(
                    tokenized_key == suffix
                    or tokenized_key.endswith("_" + suffix)
                    for suffix in _SOURCE_ADD_CREDENTIAL_KEY_SUFFIXES
                )
            ):
                return True
            if source_add_text_contains_credential_material(str(key)):
                return True
            if source_add_contains_credential_material(item):
                return True
        return False
    if isinstance(value, (list, tuple)):
        return any(source_add_contains_credential_material(item) for item in value)
    if isinstance(value, str):
        return source_add_text_contains_credential_material(value)
    return False


class SourceAddSourceKind(str, Enum):
    WEB = "web"
    FILING = "filing"
    NEWS = "news"
    REGISTRY = "registry"
    PROCUREMENT = "procurement"
    SOCIAL = "social"
    HIRING = "hiring"
    TECH_STACK = "tech_stack"
    FUNDING = "funding"
    FIRMOGRAPHIC = "firmographic"
    PEOPLE = "people"
    INTENT = "intent"
    REVIEWS = "reviews"
    EVENTS = "events"


@dataclass(frozen=True)
class SourceAddAdapterManifest:
    adapter_id: str
    miner_ref: str
    source_name: str
    source_kind: str
    declared_base_domains: tuple[str, ...]
    output_schema_ref: str
    allowed_output_fields: tuple[str, ...]
    submitted_artifact_ref: str
    code_bundle_hash: str
    sandbox_policy_ref: str
    max_trial_cost_cents: int
    max_request_cost_cents: int
    max_latency_ms: int
    credential_policy: str = "no_credentials"
    credential_ref: str = ""
    fixture_refs: tuple[str, ...] = ()
    arbitrary_code_execution_enabled: bool = False
    live_network_enabled: bool = False
    component_publicly_accepted: bool = False
    visibility_policy: str = "default_private"
    artifact_release_state: str = "private_live_champion"

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "SourceAddAdapterManifest":
        return cls(
            adapter_id=str(data["adapter_id"]),
            miner_ref=str(data["miner_ref"]),
            source_name=str(data["source_name"]),
            source_kind=str(data["source_kind"]),
            declared_base_domains=tuple(str(item) for item in data.get("declared_base_domains", [])),
            output_schema_ref=str(data["output_schema_ref"]),
            allowed_output_fields=tuple(str(item) for item in data.get("allowed_output_fields", [])),
            submitted_artifact_ref=str(data["submitted_artifact_ref"]),
            code_bundle_hash=str(data["code_bundle_hash"]),
            sandbox_policy_ref=str(data["sandbox_policy_ref"]),
            max_trial_cost_cents=int(data["max_trial_cost_cents"]),
            max_request_cost_cents=int(data["max_request_cost_cents"]),
            max_latency_ms=int(data["max_latency_ms"]),
            credential_policy=str(data.get("credential_policy", "no_credentials")),
            credential_ref=str(data.get("credential_ref", "")),
            fixture_refs=tuple(str(item) for item in data.get("fixture_refs", [])),
            arbitrary_code_execution_enabled=bool(data.get("arbitrary_code_execution_enabled", False)),
            live_network_enabled=bool(data.get("live_network_enabled", False)),
            component_publicly_accepted=bool(data.get("component_publicly_accepted", False)),
            visibility_policy=str(data.get("visibility_policy", "default_private")),
            artifact_release_state=str(
                data.get("artifact_release_state", "private_live_champion")
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["declared_base_domains"] = list(self.declared_base_domains)
        data["allowed_output_fields"] = list(self.allowed_output_fields)
        data["fixture_refs"] = list(self.fixture_refs)
        return data


@dataclass(frozen=True)
class SourceAddTrialOutputRecord:
    output_ref: str
    adapter_id: str
    icp_ref: str
    evidence_refs: tuple[str, ...]
    snapshot_refs: tuple[str, ...]
    content_hashes: tuple[str, ...]
    normalized_text_hashes: tuple[str, ...]
    metadata_refs: tuple[str, ...] = ()
    output_schema_ref: str = "schema:source-add-output:v1"

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "SourceAddTrialOutputRecord":
        return cls(
            output_ref=str(data["output_ref"]),
            adapter_id=str(data["adapter_id"]),
            icp_ref=str(data["icp_ref"]),
            evidence_refs=tuple(str(item) for item in data.get("evidence_refs", [])),
            snapshot_refs=tuple(str(item) for item in data.get("snapshot_refs", [])),
            content_hashes=tuple(str(item) for item in data.get("content_hashes", [])),
            normalized_text_hashes=tuple(str(item) for item in data.get("normalized_text_hashes", [])),
            metadata_refs=tuple(str(item) for item in data.get("metadata_refs", [])),
            output_schema_ref=str(data.get("output_schema_ref", "schema:source-add-output:v1")),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for field in SOURCE_ADD_OUTPUT_FIELDS:
            data[field] = list(getattr(self, field))
        return data


def validate_source_add_adapter_manifest(
    manifest: SourceAddAdapterManifest | Mapping[str, Any],
) -> list[str]:
    raw = manifest if isinstance(manifest, Mapping) else manifest.to_dict()
    if not isinstance(manifest, SourceAddAdapterManifest):
        manifest = SourceAddAdapterManifest.from_mapping(manifest)
    errors: list[str] = []
    if manifest.source_kind not in {kind.value for kind in SourceAddSourceKind}:
        errors.append(f"unknown source_kind: {manifest.source_kind}")
    if not manifest.declared_base_domains:
        errors.append("declared_base_domains must not be empty")
    if not manifest.output_schema_ref:
        errors.append("output_schema_ref is required")
    if not manifest.code_bundle_hash.startswith("sha256:"):
        errors.append("code_bundle_hash must be sha256-prefixed")
    if not manifest.submitted_artifact_ref:
        errors.append("submitted_artifact_ref is required")
    if not manifest.fixture_refs:
        errors.append("fixture_refs must not be empty")
    if manifest.max_trial_cost_cents <= 0 or manifest.max_request_cost_cents <= 0:
        errors.append("cost caps must be positive")
    if manifest.max_request_cost_cents > manifest.max_trial_cost_cents:
        errors.append("max_request_cost_cents cannot exceed max_trial_cost_cents")
    if manifest.max_trial_cost_cents > 5000:
        errors.append("max_trial_cost_cents exceeds P1.5 launch cap")
    if manifest.max_latency_ms <= 0:
        errors.append("max_latency_ms must be positive")
    disallowed = sorted(set(manifest.allowed_output_fields) - set(SOURCE_ADD_OUTPUT_FIELDS))
    if disallowed:
        errors.append("allowed_output_fields contains disallowed fields: " + ", ".join(disallowed))
    if "evidence_refs" not in manifest.allowed_output_fields:
        errors.append("allowed_output_fields must include evidence_refs")
    if any(field in manifest.allowed_output_fields for field in RAW_OUTPUT_FIELDS):
        errors.append("allowed_output_fields must not include raw scraped content fields")
    if manifest.credential_policy not in {"no_credentials", "credential_ref_only"}:
        errors.append("credential_policy must be no_credentials or credential_ref_only")
    if manifest.credential_policy == "no_credentials" and manifest.credential_ref:
        errors.append("credential_ref must be empty when credential_policy=no_credentials")
    if _contains_raw_credential_key(raw):
        errors.append("manifest must not contain raw credential fields")
    if manifest.arbitrary_code_execution_enabled:
        errors.append("arbitrary_code_execution_enabled must remain false")
    if manifest.live_network_enabled:
        errors.append("live_network_enabled must remain false")
    if manifest.component_publicly_accepted:
        errors.append("component_publicly_accepted must remain false in P1.5")
    if manifest.visibility_policy != "default_private":
        errors.append("SOURCE_ADD adapter manifest visibility must remain private")
    if manifest.artifact_release_state != "private_live_champion":
        errors.append("SOURCE_ADD adapter manifest must remain private")
    return errors


def validate_source_add_trial_output(output: SourceAddTrialOutputRecord | Mapping[str, Any]) -> list[str]:
    raw = output if isinstance(output, Mapping) else output.to_dict()
    if not isinstance(output, SourceAddTrialOutputRecord):
        output = SourceAddTrialOutputRecord.from_mapping(output)
    errors: list[str] = []
    if _contains_any_key(raw, RAW_OUTPUT_FIELDS):
        errors.append("adapter trial outputs must not contain raw scraped content fields")
    if not output.evidence_refs:
        errors.append("evidence_refs must not be empty")
    if not output.snapshot_refs:
        errors.append("snapshot_refs must not be empty")
    if not output.content_hashes:
        errors.append("content_hashes must not be empty")
    if not output.normalized_text_hashes:
        errors.append("normalized_text_hashes must not be empty")
    for field in ("content_hashes", "normalized_text_hashes"):
        bad = [value for value in getattr(output, field) if not value.startswith("sha256:")]
        if bad:
            errors.append(f"{field} must be sha256-prefixed")
    for evidence_ref in output.evidence_refs:
        if not (evidence_ref.startswith("evidence:") or evidence_ref.startswith("sha256:")):
            errors.append("evidence_refs must be evidence: or sha256: references")
    return errors


def _contains_any_key(value: Any, keys: Sequence[str]) -> bool:
    key_set = {key.lower() for key in keys}
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized_key = str(key).lower()
            if normalized_key in key_set:
                return True
            if _contains_any_key(nested, keys):
                return True
    elif isinstance(value, list):
        return any(_contains_any_key(item, keys) for item in value)
    return False


def _contains_raw_credential_key(value: Any) -> bool:
    return source_add_contains_credential_material(value)
