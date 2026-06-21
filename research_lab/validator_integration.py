"""Validator-side Research Lab fetch/verify/weight integration.

This module is production-oriented but defaults to shadow/read-only behavior.
It lets validators fetch official Research Lab shadow bundles, verify them
locally, and compute the Research Lab weight component without submitting it
on-chain unless an explicit future live mutation flag is enabled.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping
from urllib.request import Request, urlopen

from leadpoet_verifier.golden_vectors import run_golden_vectors

from .canonical import sha256_json
from .production_shadow import (
    assert_shadow_output_read_only,
    build_controlled_production_shadow,
    compute_verifier_divergence,
    load_production_shadow_fixture,
)


FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "validator_integration_fixtures.json"
TRUTHY_VALUES = {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class ResearchLabValidatorFlags:
    fetch_enabled: bool = False
    shadow_verify_enabled: bool = False
    require_shadow_verification_before_submit: bool = False
    weight_mutation_enabled: bool = False
    production_writes_enabled: bool = False
    submit_on_chain_enabled: bool = False
    fulfillment_mutation_enabled: bool = False

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None = None) -> "ResearchLabValidatorFlags":
        data = data or {}
        return cls(
            fetch_enabled=_truthy(data.get("RESEARCH_LAB_VALIDATOR_FETCH_ENABLED", data.get("fetch_enabled", False))),
            shadow_verify_enabled=_truthy(
                data.get("RESEARCH_LAB_VALIDATOR_SHADOW_VERIFY_ENABLED", data.get("shadow_verify_enabled", False))
            ),
            require_shadow_verification_before_submit=_truthy(
                data.get(
                    "RESEARCH_LAB_REQUIRE_SHADOW_VERIFICATION_BEFORE_SUBMIT",
                    data.get("require_shadow_verification_before_submit", False),
                )
            ),
            weight_mutation_enabled=_truthy(
                data.get("RESEARCH_LAB_WEIGHT_MUTATION_ENABLED", data.get("weight_mutation_enabled", False))
            ),
            production_writes_enabled=_truthy(
                data.get("RESEARCH_LAB_PRODUCTION_WRITES_ENABLED", data.get("production_writes_enabled", False))
            ),
            submit_on_chain_enabled=_truthy(
                data.get("RESEARCH_LAB_SUBMIT_ON_CHAIN_ENABLED", data.get("submit_on_chain_enabled", False))
            ),
            fulfillment_mutation_enabled=_truthy(
                data.get("RESEARCH_LAB_FULFILLMENT_MUTATION_ENABLED", data.get("fulfillment_mutation_enabled", False))
            ),
        )

    def to_dict(self) -> dict[str, bool]:
        return asdict(self)

    def enabled_mutation_flags(self) -> list[str]:
        return [
            name
            for name, enabled in {
                "RESEARCH_LAB_WEIGHT_MUTATION_ENABLED": self.weight_mutation_enabled,
                "RESEARCH_LAB_PRODUCTION_WRITES_ENABLED": self.production_writes_enabled,
                "RESEARCH_LAB_SUBMIT_ON_CHAIN_ENABLED": self.submit_on_chain_enabled,
                "RESEARCH_LAB_FULFILLMENT_MUTATION_ENABLED": self.fulfillment_mutation_enabled,
            }.items()
            if enabled
        ]


def fetch_research_lab_shadow_bundle(gateway_url: str, epoch: int, *, timeout_seconds: int = 20) -> dict[str, Any]:
    """Fetch a Research Lab shadow report from the gateway.

    This performs a read-only HTTP GET. Callers still need to verify the bundle
    with ``verify_research_lab_shadow_bundle`` before trusting any weight field.
    """
    base = gateway_url.rstrip("/")
    request = Request(
        f"{base}/research-lab/reports/shadow/{int(epoch)}",
        headers={"Accept": "application/json"},
        method="GET",
    )
    with urlopen(request, timeout=timeout_seconds) as response:
        return json.loads(response.read().decode("utf-8"))


def verify_research_lab_shadow_bundle(
    bundle: Mapping[str, Any],
    *,
    flags: ResearchLabValidatorFlags | Mapping[str, Any] | None = None,
    official_weight_hash: str | None = None,
) -> dict[str, Any]:
    validator_flags = flags if isinstance(flags, ResearchLabValidatorFlags) else ResearchLabValidatorFlags.from_mapping(flags)
    errors: list[str] = []
    if not validator_flags.shadow_verify_enabled:
        errors.append("validator_shadow_verify_disabled")
    errors.extend(validator_flags.enabled_mutation_flags())

    read_only = assert_shadow_output_read_only(bundle)
    if not read_only["passed"]:
        errors.extend(read_only["errors"])

    if _contains_secret_material(bundle):
        errors.append("bundle_contains_raw_secret_material")

    source_state = bundle.get("source_state")
    source_state_hash = bundle.get("source_state_hash")
    if source_state is not None or source_state_hash is not None:
        if not source_state or not source_state_hash:
            errors.append("source_state_and_hash_must_be_present_together")
        elif sha256_json(source_state) != source_state_hash:
            errors.append("source_state_hash_diverged")

    golden_errors = run_golden_vectors()
    if golden_errors:
        errors.extend(f"open_verifier:{error}" for error in golden_errors)

    weight_vector = bundle.get("weight_vector", {})
    if not isinstance(weight_vector, Mapping):
        errors.append("weight_vector_must_be_object")
        weight_vector = {}
    divergence = compute_verifier_divergence(
        weight_vector,
        official_weight_hash=official_weight_hash or bundle.get("weight_vector_hash"),
    )
    if divergence["diverged"]:
        errors.append("weight_vector_hash_diverged")

    return {
        "passed": not errors,
        "errors": errors,
        "shadow_only": bool(bundle.get("shadow_only", False)),
        "read_only": bool(bundle.get("read_only", False)),
        "validator_flags": validator_flags.to_dict(),
        "bundle_id": bundle.get("bundle_id"),
        "epoch": bundle.get("epoch"),
        "weight_vector_hash": sha256_json(weight_vector) if weight_vector else None,
        "verifier_divergence": divergence,
        "on_chain_submission_allowed": False,
    }


def build_research_lab_weight_component(
    bundle: Mapping[str, Any],
    *,
    flags: ResearchLabValidatorFlags | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    verification = verify_research_lab_shadow_bundle(bundle, flags=flags)
    if not verification["passed"]:
        raise ValueError("; ".join(verification["errors"]))
    weight_vector = bundle["weight_vector"]
    return {
        "epoch": int(bundle["epoch"]),
        "shadow_only": True,
        "read_only": True,
        "submission_allowed": False,
        "on_chain_submission_allowed": False,
        "bundle_id": bundle.get("bundle_id"),
        "source_state_hash": bundle.get("source_state_hash"),
        "u16_weights": dict(weight_vector.get("u16_weights", {})),
        "weight_sum": int(weight_vector.get("weight_sum", 0)),
        "weight_vector_hash": verification["weight_vector_hash"],
    }


def build_on_chain_weight_payload(_component: Mapping[str, Any]) -> dict[str, Any]:
    raise RuntimeError("Research Lab shadow verification cannot build on-chain weight payloads")


def write_research_lab_validator_artifact(
    *,
    output_dir: Path | str,
    epoch: int,
    bundle: Mapping[str, Any],
    verification: Mapping[str, Any],
    component: Mapping[str, Any] | None = None,
) -> Path:
    """Write the validator's local Research Lab verification artifact.

    This is a local file only; it is not a production database write and never
    contains raw provider keys.
    """
    if _contains_secret_material(bundle) or _contains_secret_material(verification) or _contains_secret_material(component or {}):
        raise ValueError("Research Lab validator artifact contains raw secret material")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    artifact_path = output_path / f"research_lab_shadow_epoch_{int(epoch)}.json"
    payload = {
        "schema_version": "1.0",
        "artifact_type": "research_lab_validator_shadow_verification",
        "epoch": int(epoch),
        "bundle": dict(bundle),
        "verification": dict(verification),
        "component": dict(component or {}),
    }
    artifact_path.write_text(json.dumps(payload, sort_keys=True, indent=2, default=str), encoding="utf-8")
    return artifact_path


def verify_research_lab_validator_integration(path: Path | str | None = None) -> dict[str, Any]:
    fixture_path = Path(path) if path else FIXTURE_PATH
    with fixture_path.open("r", encoding="utf-8") as handle:
        fixture = json.load(handle)

    shadow_fixture = load_production_shadow_fixture()
    shadow_result = build_controlled_production_shadow(shadow_fixture["shadow_case"])
    bundle = shadow_result["bundle"]
    bundle = {
        **bundle,
        "weight_vector_hash": sha256_json(bundle["weight_vector"]),
    }

    verification = verify_research_lab_shadow_bundle(bundle, flags=fixture["validator_flags"])
    if not verification["passed"]:
        raise AssertionError("; ".join(verification["errors"]))

    component = build_research_lab_weight_component(bundle, flags=fixture["validator_flags"])
    try:
        build_on_chain_weight_payload(component)
        raise AssertionError("shadow component built an on-chain payload")
    except RuntimeError:
        pass

    unsafe = verify_research_lab_shadow_bundle(bundle, flags=fixture["unsafe_validator_flags"])
    if unsafe["passed"] or not unsafe["errors"]:
        raise AssertionError("unsafe mutation flags were not rejected")

    tampered_source = {
        **bundle,
        "source_state": {"tampered": True},
        "source_state_hash": sha256_json({"tampered": False}),
    }
    tampered = verify_research_lab_shadow_bundle(tampered_source, flags=fixture["validator_flags"])
    if tampered["passed"] or "source_state_hash_diverged" not in tampered["errors"]:
        raise AssertionError("source-state hash tamper was not rejected")

    return {
        "bundle_id": bundle["bundle_id"],
        "epoch": bundle["epoch"],
        "weight_vector_hash": component["weight_vector_hash"],
        "weight_sum": component["weight_sum"],
        "unsafe_errors": unsafe["errors"],
    }


def _contains_secret_material(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, item in value.items():
            lowered_key = str(key).lower()
            if any(marker in lowered_key for marker in ("api_key", "raw_secret", "raw_openrouter", "credential")):
                return True
            if _contains_secret_material(item):
                return True
    elif isinstance(value, list):
        return any(_contains_secret_material(item) for item in value)
    elif isinstance(value, str):
        lowered = value.lower()
        return any(marker in lowered for marker in ("sk-or-", "raw_openrouter_key", "openrouter_api_key", "raw_secret"))
    return False


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in TRUTHY_VALUES
