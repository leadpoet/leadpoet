"""Production Research Lab gateway flags.

Most live workflow flags default on only for the production subnet
(`BITTENSOR_NETWORK=finney`, `BITTENSOR_NETUID=71`). Reimbursements and weight
mutation remain explicit opt-ins because they directly affect incentives.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import os
from typing import Any, Mapping, Optional


TRUTHY = {"1", "true", "yes", "on"}
logger = logging.getLogger(__name__)

# Single code-level default for the Research Lab loop-start fee. Operators can
# still override it at runtime with RESEARCH_LAB_LOOP_START_FEE_USD.
DEFAULT_LOOP_START_FEE_USD = 0.2
DEFAULT_ACTIVE_LOOP_STALE_AFTER_SECONDS = 300
DEFAULT_HOSTED_WORKER_RETRYABLE_FAILURE_LIMIT = 3
DEFAULT_PRIVATE_REPO_URL = ""
DEFAULT_PRIVATE_TEST_CMD = r"""
python3 - <<'PY'
import research_lab_adapter
import sourcing_model

metadata = research_lab_adapter.adapter_metadata()
assert metadata.get("adapter_version")
assert sourcing_model is not None
PY
""".strip()
DEFAULT_PRIVATE_ARTIFACT_MANIFEST_OUTPUT = ".research_lab/candidate_manifest.json"
DEFAULT_PRIVATE_BUILD_CMD = r"""
bash <<'BASH'
set -euo pipefail

AWS_REGION="${AWS_REGION:-${AWS_DEFAULT_REGION:-us-east-1}}"
ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
ECR_REPOSITORY="${RESEARCH_LAB_PRIVATE_ECR_REPOSITORY:-leadpoet/sourcing-model}"
REGISTRY="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
COMMIT_SHA="${RESEARCH_LAB_PRIVATE_COMMIT_SHA:?RESEARCH_LAB_PRIVATE_COMMIT_SHA is required}"
RUN_ID="${RESEARCH_LAB_RUN_ID:?RESEARCH_LAB_RUN_ID is required}"
CANDIDATE_INDEX="${RESEARCH_LAB_CANDIDATE_INDEX:?RESEARCH_LAB_CANDIDATE_INDEX is required}"
IMAGE_TAG_RAW="research-lab-${RUN_ID}-${CANDIDATE_INDEX}-${COMMIT_SHA}"
IMAGE_TAG="$(printf '%s' "${IMAGE_TAG_RAW}" | tr -c 'A-Za-z0-9_.-' '-' | cut -c1-120)"
IMAGE_URI="${REGISTRY}/${ECR_REPOSITORY}:${IMAGE_TAG}"
IMAGE_DIGEST=""
cleanup_candidate_image() {
  if [ -n "${IMAGE_DIGEST:-}" ]; then
    docker image rm "${IMAGE_DIGEST}" >/dev/null 2>&1 || true
  fi
  if [ -n "${IMAGE_URI:-}" ]; then
    docker image rm "${IMAGE_URI}" >/dev/null 2>&1 || true
  fi
}
trap cleanup_candidate_image EXIT
PLATFORM="${RESEARCH_LAB_PRIVATE_MODEL_DOCKER_PLATFORM:-linux/amd64}"
PARENT_MANIFEST_URI="${RESEARCH_LAB_PRIVATE_MODEL_MANIFEST_URI:-s3://leadpoet-private-model-artifacts-493765492819/research-lab/sourcing-model/current.json}"
MANIFEST_BASE="${PARENT_MANIFEST_URI%/*}"
MANIFEST_URI="${MANIFEST_BASE}/candidates/${RUN_ID}/${CANDIDATE_INDEX}/${COMMIT_SHA}.json"
SIGNATURE_URI="${MANIFEST_BASE}/candidates/${RUN_ID}/${CANDIDATE_INDEX}/${COMMIT_SHA}.sig.b64"
KMS_KEY_ID="${RESEARCH_LAB_SCORE_BUNDLE_KMS_KEY_ID:?RESEARCH_LAB_SCORE_BUNDLE_KMS_KEY_ID is required}"
OUTPUT_PATH="${RESEARCH_LAB_PRIVATE_ARTIFACT_MANIFEST_OUTPUT:-.research_lab/candidate_manifest.json}"

aws ecr get-login-password --region "${AWS_REGION}" \
  | docker login --username AWS --password-stdin "${REGISTRY}" >/dev/null

docker build --platform "${PLATFORM}" -f Dockerfile.research-lab -t "${IMAGE_URI}" .
docker push "${IMAGE_URI}"
DIGEST="$(aws ecr describe-images \
  --repository-name "${ECR_REPOSITORY}" \
  --image-ids imageTag="${IMAGE_TAG}" \
  --query 'imageDetails[0].imageDigest' \
  --output text)"
IMAGE_DIGEST="${REGISTRY}/${ECR_REPOSITORY}@${DIGEST}"

python3 - "${OUTPUT_PATH}" "${IMAGE_DIGEST}" "${MANIFEST_URI}" "${SIGNATURE_URI}" "${COMMIT_SHA}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

output = Path(sys.argv[1])
image_digest, manifest_uri, signature_ref, git_commit_sha = sys.argv[2:6]
excluded_parts = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".venv", "venv", ".research_lab"}
excluded_suffixes = (".pyc", ".pyo", ".env", ".pem", ".key")

def sha256_json(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()

def sha256_bytes(value):
    return "sha256:" + hashlib.sha256(value).hexdigest()

def excluded(rel):
    parts = rel.split("/")
    return any(part in excluded_parts for part in parts) or rel.endswith(excluded_suffixes) or rel == ".env" or rel.startswith(".env.")

digest_inputs = []
for path in sorted(Path(".").rglob("*")):
    if not path.is_file():
        continue
    rel = path.as_posix()
    if excluded(rel):
        continue
    digest_inputs.append((rel, sha256_bytes(path.read_bytes())))

config_payload = {
    "component_registry_version": "sourcing-model-components:v1",
    "scoring_adapter_version": "qualification-company-scorer:v1",
    "adapter_version": "sourcing-model-research-lab-adapter:v1",
}
payload = {
    "model_artifact_hash": sha256_json(digest_inputs),
    "git_commit_sha": git_commit_sha,
    "image_digest": image_digest,
    "config_hash": sha256_json(config_payload),
    "component_registry_version": config_payload["component_registry_version"],
    "scoring_adapter_version": config_payload["scoring_adapter_version"],
    "manifest_uri": manifest_uri,
    "signature_ref": signature_ref,
    "build_id": f"gateway-code-edit:{git_commit_sha}",
}
manifest = {**payload, "manifest_hash": sha256_json(payload)}
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(manifest["manifest_hash"])
PY

MANIFEST_HASH="$(python3 -c "import json; print(json.load(open('${OUTPUT_PATH}'))['manifest_hash'])")"
printf '%s' "${MANIFEST_HASH}" > /tmp/research_lab_candidate_manifest_hash.txt
aws kms sign \
  --key-id "${KMS_KEY_ID}" \
  --message fileb:///tmp/research_lab_candidate_manifest_hash.txt \
  --message-type RAW \
  --signing-algorithm ECDSA_SHA_256 \
  --query Signature \
  --output text > /tmp/research_lab_candidate_manifest.sig.b64
aws s3 cp /tmp/research_lab_candidate_manifest.sig.b64 "${SIGNATURE_URI}" >/dev/null
aws s3 cp "${OUTPUT_PATH}" "${MANIFEST_URI}" >/dev/null
BASH
""".strip()


def _is_production_subnet() -> bool:
    network = (
        os.getenv("BITTENSOR_NETWORK")
        or os.getenv("SUBTENSOR_NETWORK")
        or ""
    ).strip().lower()
    netuid = (
        os.getenv("BITTENSOR_NETUID")
        or os.getenv("NETUID")
        or ""
    ).strip()
    return network == "finney" and netuid == "71"


def _prod_default(default_for_prod: bool, default_for_non_prod: bool = False) -> str:
    return (
        "true"
        if (_is_production_subnet() and default_for_prod) or (not _is_production_subnet() and default_for_non_prod)
        else "false"
    )


def _truthy(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in TRUTHY


def _normalize_auto_research_reasoning_effort(value: Any) -> str:
    effort = str(value or "").strip().lower()
    if not effort:
        return ""
    allowed = {"minimal", "low", "medium", "high", "xhigh", "max"}
    if effort not in allowed:
        logger.warning("Ignoring unsupported RESEARCH_LAB_AUTO_RESEARCH_REASONING_EFFORT: %s", effort)
        return ""
    return effort


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


# Floor for the promotion improvement threshold. A threshold of 0 (or below)
# would promote unmeasured / 0.0-basis candidates — explicitly forbidden by
# fableanalysis §8.3 and §0-N3. The clamp is defense in depth alongside the
# promotion metric's hard-reject of unmeasured candidates.
MIN_IMPROVEMENT_THRESHOLD_POINTS = 0.1


def _improvement_threshold_points(default: float = 1.0) -> float:
    value = _float("RESEARCH_LAB_IMPROVEMENT_THRESHOLD_POINTS", default)
    if value < MIN_IMPROVEMENT_THRESHOLD_POINTS:
        logger.warning(
            "RESEARCH_LAB_IMPROVEMENT_THRESHOLD_POINTS=%s is below the minimum %s; "
            "clamping. A zero or negative threshold promotes unmeasured/0.0-basis "
            "candidates (fableanalysis §8.3, §0-N3).",
            value,
            MIN_IMPROVEMENT_THRESHOLD_POINTS,
        )
        return MIN_IMPROVEMENT_THRESHOLD_POINTS
    return value


def _count_configured_proxy_values(prefixes: tuple[str, ...]) -> int:
    seen: set[str] = set()
    count = 0
    for index in range(1, 501):
        for prefix in prefixes:
            value = os.getenv(f"{prefix}_{index}", "").strip()
            if value and value not in seen:
                seen.add(value)
                count += 1
                break
    return count


def _worker_total_from_proxy_count(
    *,
    prefixes: tuple[str, ...],
    legacy_total_env: str,
    default: int = 1,
) -> int:
    proxy_count = _count_configured_proxy_values(prefixes)
    if proxy_count > 0:
        return proxy_count
    return max(1, _int(legacy_total_env, default))


def _optional_int(name: str) -> Optional[int]:
    value = os.getenv(name)
    if value is None or not value.strip():
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _normalized_csv(name: str, default: str) -> tuple[str, ...]:
    raw = os.getenv(name, default).strip()
    if raw.startswith("["):
        try:
            decoded = json.loads(raw)
            items = decoded if isinstance(decoded, list) else []
        except json.JSONDecodeError:
            items = []
    else:
        items = raw.split(",")
    normalized: list[str] = []
    for item in items:
        value = str(item or "").strip().lower().replace("-", "_").replace(" ", "_")
        if value and value not in normalized:
            normalized.append(value)
    return tuple(normalized or ("generalist",))


def _tuple_from_json_or_csv(raw: str, default: tuple[str, ...]) -> tuple[str, ...]:
    text = str(raw or "").strip()
    if not text:
        return default
    if text.startswith("["):
        try:
            decoded = json.loads(text)
        except json.JSONDecodeError:
            return default
        if not isinstance(decoded, list):
            return default
        values = [str(item).strip() for item in decoded if str(item).strip()]
    else:
        values = [item.strip() for item in text.split(",") if item.strip()]
    return tuple(values or default)


@dataclass(frozen=True)
class ResearchLabGatewayConfig:
    api_enabled: bool = False
    production_writes_enabled: bool = False
    miner_submissions_enabled: bool = False
    paid_loops_enabled: bool = False
    loop_topups_enabled: bool = False
    probes_enabled: bool = False
    hosted_runs_enabled: bool = False
    receipts_enabled: bool = False
    evaluation_bundles_enabled: bool = False
    reports_enabled: bool = False
    public_activity_enabled: bool = False
    shadow_bundles_enabled: bool = False
    shadow_weights_enabled: bool = False
    shadow_reimbursements_enabled: bool = False
    crowning_enabled: bool = True
    reimbursements_enabled: bool = False
    weight_mutation_enabled: bool = False
    fulfillment_mutation_enabled: bool = False
    auto_promotion_enabled: bool = True
    auto_commit_enabled: bool = False
    miner_openrouter_key_required: bool = True
    max_open_tickets_per_hotkey: int = 3
    loop_start_fee_usd: float = DEFAULT_LOOP_START_FEE_USD
    allowed_research_islands: tuple[str, ...] = ("generalist",)
    internal_api_key: str = ""
    hosted_worker_enabled: bool = False
    hosted_worker_poll_seconds: int = 15
    hosted_worker_max_runs: int = 0
    hosted_worker_max_candidates: int = 1
    hosted_worker_dry_run: bool = False
    hosted_worker_id: str = ""
    hosted_worker_index: int = 0
    hosted_worker_total_workers: int = 1
    hosted_worker_queue_fetch_limit: int = 20
    hosted_worker_require_proxy: bool = False
    hosted_worker_proxy_url: str = ""
    active_loop_stale_after_seconds: int = DEFAULT_ACTIVE_LOOP_STALE_AFTER_SECONDS
    hosted_worker_retryable_failure_limit: int = DEFAULT_HOSTED_WORKER_RETRYABLE_FAILURE_LIMIT
    scoring_worker_enabled: bool = False
    scoring_worker_poll_seconds: int = 15
    scoring_worker_max_candidates: int = 1
    scoring_worker_id: str = ""
    scoring_worker_index: int = 0
    scoring_worker_total_workers: int = 1
    scoring_worker_require_proxy: bool = False
    scoring_worker_proxy_url: str = ""
    scoring_worker_model_timeout_seconds: int = 1800
    scoring_worker_max_claim_requeues: int = 3
    scoring_worker_baseline_not_ready_retry_seconds: int = 900
    scoring_worker_retryable_failure_retry_seconds: int = 300
    private_model_docker_global_proxy_enabled: bool = False
    scoring_worker_allow_partial_icp_window: bool = False
    # Scoring health is audit metadata only. Promotion is score-only against the
    # stored daily baseline; provider/runtime health must not veto champions.
    scoring_health_gate_enabled: bool = False
    scoring_health_max_reference_runtime_failure_rate: float = 0.25
    scoring_health_max_candidate_runtime_failure_rate: float = 0.25
    scoring_health_max_reference_zero_company_rate: float = 0.50
    scoring_health_max_candidate_zero_company_rate: float = 0.50
    scoring_health_max_provider_error_rate: float = 0.10
    scoring_health_max_timeout_rate: float = 0.10
    private_baseline_rebenchmark_enabled: bool = False
    private_baseline_concurrency: int = 1
    private_baseline_retry_concurrency: int = 2
    private_baseline_provider_retry_rounds: int = 2
    benchmark_exa_api_key: str = ""
    benchmark_exa_max_rps: float = 0.0
    auto_research_min_seconds: int = 600
    auto_research_max_seconds: int = 2700
    auto_research_min_iterations: int = 3
    auto_research_max_iterations: int = 6
    # 180s: 12k-token draft diffs routinely exceed the old 90s (fableanalysis §6.4).
    auto_research_draft_timeout_seconds: int = 180
    auto_research_reflection_timeout_seconds: int = 90
    auto_research_max_tokens: int = 12000
    auto_research_temperature: float = 0.35
    auto_research_estimated_iteration_cost_usd: float = 0.5
    loop_planner_enabled: bool = True
    loop_alignment_judge_enabled: bool = True
    loop_novelty_strict: bool = True
    loop_planner_model: str = ""
    loop_planner_fallback_models: tuple[str, ...] = ()
    loop_planner_reasoning_effort: str = ""
    # Matches the from_env default (12000) so direct constructions exercise the
    # same behavior prod gets (fableanalysis bug #32).
    loop_planner_max_tokens: int = 12000
    loop_planner_temperature: float = 0.40
    loop_planner_allow_non_zdr: bool = False
    loop_executor_model: str = ""
    loop_executor_reasoning_effort: str = ""
    loop_alignment_judge_model: str = ""
    loop_alignment_judge_reasoning_effort: str = ""
    loop_alignment_judge_max_tokens: int = 1600
    loop_alignment_judge_temperature: float = 0.0
    reimbursement_policy_id: str = "alpha-reimbursement-production-v1"
    reimbursement_min_rebate_rate: float = 0.25
    reimbursement_base_rebate_rate: float = 0.50
    reimbursement_max_rebate_rate: float = 1.0
    reimbursement_high_participation_target: float = 10.0
    reimbursement_epochs: int = 20
    reimbursement_max_usd_per_run: float = 100.0
    reimbursement_max_usd_per_hotkey_day: float = 100.0
    reimbursement_max_usd_per_island_day: float = 1000.0
    reimbursement_global_budget_usd: float = 5000.0
    reimbursement_material_spend_ratio: float = 0.80
    reimbursement_default_island: str = "generalist"
    reimbursement_usd_per_0_1_percent_epoch: float = 0.162
    lab_emission_percent: float = 10.0
    fulfillment_emission_percent: float = 80.5
    fulfillment_leaderboard_emission_percent: float = 9.5
    lab_reward_epochs: int = 20
    lab_reimbursement_allow_overpay_without_champions: bool = True
    lab_reimbursement_max_cost_multiplier_with_champions: float = 1.0
    lab_reimbursement_min_alpha_percent: float = 0.0
    lab_champion_min_alpha_percent: float = 2.0
    lab_champion_extra_alpha_percent_per_point: float = 0.1
    lab_champion_max_alpha_percent: float = 5.0
    lab_champion_placeholder_alpha_percent: float = 0.0001
    lab_champion_queue_trigger_ratio: float = 0.50
    lab_champion_threshold_points: float = 2.0
    lab_champion_eval_days: int = 10
    # 2 matches the from_env default (fableanalysis bug #32); §6.4 recommends
    # raising to 4-6 only after benchmark-concurrency parity is verified.
    lab_champion_icps_per_day: int = 2
    public_benchmark_public_icps_per_day: int = 3
    public_benchmark_public_weak_per_day: int = 2
    # Ratio rule (fableanalysis §6.4): the public split must expose at most 1/3
    # of the private scoring window (lab_champion_eval_days *
    # lab_champion_icps_per_day = 10 * 2 = 20 ICPs by default), so the default
    # public total is 6 (was 10, i.e. half the sealed set exposed daily). The
    # weak share keeps roughly the per-day 2/3 proportion and must stay <=
    # public_benchmark_public_total_icps (validate_public_benchmark_split).
    public_benchmark_public_total_icps: Optional[int] = 6
    public_benchmark_public_weak_total: Optional[int] = 4
    improvement_threshold_points: float = 1.0
    private_model_manifest_uri: str = (
        "s3://leadpoet-private-model-artifacts-493765492819/research-lab/sourcing-model/current.json"
    )
    private_repo_url: str = DEFAULT_PRIVATE_REPO_URL
    private_repo_branch: str = "main"
    private_patch_applier_cmd: str = ""
    private_test_cmd: str = DEFAULT_PRIVATE_TEST_CMD
    private_build_cmd: str = DEFAULT_PRIVATE_BUILD_CMD
    private_artifact_manifest_output: str = DEFAULT_PRIVATE_ARTIFACT_MANIFEST_OUTPUT
    private_benchmark_path: str = ""
    code_edit_candidates_enabled: bool = True
    # 1800s: the build includes the cold docker pull, and 900s unfairly killed
    # first candidates on cold workers (fableanalysis §6.4).
    code_edit_build_timeout_seconds: int = 1800
    code_edit_allowed_paths_json: str = ""
    code_edit_allowed_exact_paths_json: str = ""
    code_edit_allowed_suffixes_json: str = ""
    code_edit_source_inspection_rounds: int = 3
    # 12 cumulative file reads: 8 caps patch quality on ~300-file trees
    # (fableanalysis §6.4 recommends 12-16).
    code_edit_source_inspection_max_files: int = 12
    code_edit_source_inspection_file_bytes: int = 24_000
    code_edit_source_inspection_total_bytes: int = 120_000
    code_edit_source_inspection_search_matches: int = 30
    code_edit_patch_repair_attempts: int = 2
    stale_parent_rebase_enabled: bool = True
    stale_parent_rebase_repair_enabled: bool = True
    stale_parent_rebase_repair_model: str = "anthropic/claude-sonnet-4.6"
    stale_parent_rebase_repair_timeout_seconds: int = 120
    stale_parent_check_interval_seconds: int = 30
    stale_parent_rebase_max_depth: int = 3
    score_bundle_kms_key_id: str = "alias/leadpoet-research-lab-artifact-signing"
    score_bundle_signature_uri_prefix: str = ""
    auto_research_model: str = ""
    auto_research_reasoning_effort: str = ""
    approved_auto_research_models_json: str = ""
    default_auto_research_model_tier: str = "default"
    default_compute_budget_usd: float = 5.0
    min_compute_budget_usd: float = 1.0
    max_compute_budget_usd: float = 100.0
    topup_promising_delta_threshold: float = 0.5
    miner_openrouter_key_env_var: str = ""
    miner_openrouter_key_ref_env_map_json: str = ""
    openrouter_key_kms_key_id: str = ""
    evaluation_epoch: int = 0
    arweave_audit_enabled: bool = True
    arweave_audit_shadow_enabled: bool = False

    @classmethod
    def from_env(cls) -> "ResearchLabGatewayConfig":
        prod_on = _prod_default(True)
        total_workers = _worker_total_from_proxy_count(
            prefixes=(
                "RESEARCH_LAB_AUTO_RESEARCH_WEBSHARE_PROXY",
                "RESEARCH_LAB_WORKER_PROXY",
                "RESEARCH_LAB_WORKER_HTTPS_PROXY",
            ),
            legacy_total_env="RESEARCH_LAB_HOSTED_WORKER_TOTAL_WORKERS",
        )
        worker_index = _int("RESEARCH_LAB_HOSTED_WORKER_INDEX", 0)
        if worker_index < 0:
            worker_index = 0
        if worker_index >= total_workers:
            worker_index = worker_index % total_workers
        scoring_total_workers = _worker_total_from_proxy_count(
            prefixes=(
                "RESEARCH_LAB_QUALIFICATION_WEBSHARE_PROXY",
                "QUALIFICATION_WEBSHARE_PROXY",
                "RESEARCH_LAB_SCORING_WORKER_PROXY",
            ),
            legacy_total_env="RESEARCH_LAB_SCORING_WORKER_TOTAL_WORKERS",
        )
        scoring_worker_index = _int("RESEARCH_LAB_SCORING_WORKER_INDEX", 0)
        if scoring_worker_index < 0:
            scoring_worker_index = 0
        if scoring_worker_index >= scoring_total_workers:
            scoring_worker_index = scoring_worker_index % scoring_total_workers
        return cls(
            api_enabled=_truthy("RESEARCH_LAB_GATEWAY_API_ENABLED", prod_on),
            production_writes_enabled=_truthy("RESEARCH_LAB_PRODUCTION_WRITES_ENABLED", prod_on),
            miner_submissions_enabled=_truthy("RESEARCH_LAB_MINER_SUBMISSIONS_ENABLED", prod_on),
            paid_loops_enabled=_truthy("RESEARCH_LAB_PAID_LOOPS_ENABLED", prod_on),
            loop_topups_enabled=_truthy("RESEARCH_LAB_LOOP_TOPUPS_ENABLED", "false"),
            probes_enabled=_truthy("RESEARCH_LAB_PROBES_ENABLED", prod_on),
            hosted_runs_enabled=_truthy("RESEARCH_LAB_HOSTED_RUNS_ENABLED", prod_on),
            receipts_enabled=_truthy("RESEARCH_LAB_RECEIPTS_ENABLED", prod_on),
            evaluation_bundles_enabled=_truthy("RESEARCH_LAB_EVALUATION_BUNDLES_ENABLED", prod_on),
            reports_enabled=_truthy("RESEARCH_LAB_REPORTS_ENABLED", prod_on),
            public_activity_enabled=_truthy("RESEARCH_LAB_PUBLIC_ACTIVITY_ENABLED", prod_on),
            shadow_bundles_enabled=_truthy("RESEARCH_LAB_SHADOW_BUNDLES_ENABLED", prod_on),
            shadow_weights_enabled=_truthy("RESEARCH_LAB_SHADOW_WEIGHTS_ENABLED", prod_on),
            shadow_reimbursements_enabled=_truthy("RESEARCH_LAB_SHADOW_REIMBURSEMENTS_ENABLED"),
            crowning_enabled=_truthy("RESEARCH_LAB_CROWNING_ENABLED", "true"),
            reimbursements_enabled=_truthy("RESEARCH_LAB_REIMBURSEMENTS_ENABLED"),
            weight_mutation_enabled=_truthy("RESEARCH_LAB_WEIGHT_MUTATION_ENABLED"),
            fulfillment_mutation_enabled=_truthy("RESEARCH_LAB_FULFILLMENT_MUTATION_ENABLED"),
            auto_promotion_enabled=_truthy("RESEARCH_LAB_AUTO_PROMOTION_ENABLED", "true"),
            auto_commit_enabled=_truthy("RESEARCH_LAB_AUTO_COMMIT_ENABLED"),
            miner_openrouter_key_required=_truthy("RESEARCH_LAB_MINER_OPENROUTER_KEY_REQUIRED", "true"),
            max_open_tickets_per_hotkey=max(
                1,
                _int("RESEARCH_LAB_MAX_OPEN_TICKETS_PER_HOTKEY", 3),
            ),
            loop_start_fee_usd=_float("RESEARCH_LAB_LOOP_START_FEE_USD", DEFAULT_LOOP_START_FEE_USD),
            allowed_research_islands=_normalized_csv("RESEARCH_LAB_ALLOWED_ISLANDS", "generalist"),
            internal_api_key=os.getenv("RESEARCH_LAB_INTERNAL_API_KEY", ""),
            hosted_worker_enabled=_truthy("RESEARCH_LAB_HOSTED_WORKER_ENABLED"),
            hosted_worker_poll_seconds=_int("RESEARCH_LAB_HOSTED_WORKER_POLL_SECONDS", 15),
            hosted_worker_max_runs=_int("RESEARCH_LAB_HOSTED_WORKER_MAX_RUNS", 0),
            hosted_worker_max_candidates=max(1, _int("RESEARCH_LAB_HOSTED_WORKER_MAX_CANDIDATES", 1)),
            hosted_worker_dry_run=_truthy("RESEARCH_LAB_HOSTED_WORKER_DRY_RUN", "false"),
            hosted_worker_id=os.getenv("RESEARCH_LAB_HOSTED_WORKER_ID", ""),
            hosted_worker_index=worker_index,
            hosted_worker_total_workers=total_workers,
            hosted_worker_queue_fetch_limit=max(1, _int("RESEARCH_LAB_HOSTED_WORKER_QUEUE_FETCH_LIMIT", 20)),
            hosted_worker_require_proxy=(
                _truthy("RESEARCH_LAB_REQUIRE_WORKER_PROXY")
                or _truthy("RESEARCH_LAB_HOSTED_WORKER_REQUIRE_PROXY")
            ),
            hosted_worker_proxy_url=os.getenv("RESEARCH_LAB_HOSTED_WORKER_PROXY", ""),
            active_loop_stale_after_seconds=max(
                60,
                _int("RESEARCH_LAB_ACTIVE_LOOP_STALE_AFTER_SECONDS", DEFAULT_ACTIVE_LOOP_STALE_AFTER_SECONDS),
            ),
            hosted_worker_retryable_failure_limit=max(
                0,
                _int(
                    "RESEARCH_LAB_HOSTED_WORKER_RETRYABLE_FAILURE_LIMIT",
                    DEFAULT_HOSTED_WORKER_RETRYABLE_FAILURE_LIMIT,
                ),
            ),
            scoring_worker_enabled=_truthy("RESEARCH_LAB_SCORING_WORKER_ENABLED"),
            scoring_worker_poll_seconds=max(1, _int("RESEARCH_LAB_SCORING_WORKER_POLL_SECONDS", 15)),
            scoring_worker_max_candidates=max(1, _int("RESEARCH_LAB_SCORING_WORKER_MAX_CANDIDATES", 1)),
            scoring_worker_id=os.getenv("RESEARCH_LAB_SCORING_WORKER_ID", ""),
            scoring_worker_index=scoring_worker_index,
            scoring_worker_total_workers=scoring_total_workers,
            scoring_worker_require_proxy=(
                _truthy("RESEARCH_LAB_REQUIRE_QUALIFICATION_PROXY")
                or _truthy("RESEARCH_LAB_SCORING_WORKER_REQUIRE_PROXY")
            ),
            scoring_worker_proxy_url=os.getenv("RESEARCH_LAB_SCORING_WORKER_PROXY", ""),
            scoring_worker_model_timeout_seconds=max(
                30,
                _int("RESEARCH_LAB_SCORING_WORKER_MODEL_TIMEOUT_SECONDS", 1800),
            ),
            scoring_worker_max_claim_requeues=max(
                1,
                _int("RESEARCH_LAB_SCORING_WORKER_MAX_CLAIM_REQUEUES", 3),
            ),
            scoring_worker_baseline_not_ready_retry_seconds=max(
                60,
                _int("RESEARCH_LAB_SCORING_BASELINE_NOT_READY_RETRY_SECONDS", 900),
            ),
            scoring_worker_retryable_failure_retry_seconds=max(
                60,
                _int("RESEARCH_LAB_SCORING_RETRYABLE_FAILURE_RETRY_SECONDS", 300),
            ),
            private_model_docker_global_proxy_enabled=_truthy(
                "RESEARCH_LAB_PRIVATE_MODEL_DOCKER_GLOBAL_PROXY_ENABLED",
                "false",
            ),
            scoring_worker_allow_partial_icp_window=_truthy(
                "RESEARCH_LAB_SCORING_ALLOW_PARTIAL_ICP_WINDOW",
                "false",
            ),
            scoring_health_gate_enabled=_truthy("RESEARCH_LAB_SCORING_HEALTH_GATE_ENABLED", "false"),
            scoring_health_max_reference_runtime_failure_rate=min(
                1.0,
                max(0.0, _float("RESEARCH_LAB_SCORING_HEALTH_MAX_REFERENCE_RUNTIME_FAILURE_RATE", 0.25)),
            ),
            scoring_health_max_candidate_runtime_failure_rate=min(
                1.0,
                max(0.0, _float("RESEARCH_LAB_SCORING_HEALTH_MAX_CANDIDATE_RUNTIME_FAILURE_RATE", 0.25)),
            ),
            scoring_health_max_reference_zero_company_rate=min(
                1.0,
                max(0.0, _float("RESEARCH_LAB_SCORING_HEALTH_MAX_REFERENCE_ZERO_COMPANY_RATE", 0.50)),
            ),
            scoring_health_max_candidate_zero_company_rate=min(
                1.0,
                max(0.0, _float("RESEARCH_LAB_SCORING_HEALTH_MAX_CANDIDATE_ZERO_COMPANY_RATE", 0.50)),
            ),
            scoring_health_max_provider_error_rate=min(
                1.0,
                max(0.0, _float("RESEARCH_LAB_SCORING_HEALTH_MAX_PROVIDER_ERROR_RATE", 0.10)),
            ),
            scoring_health_max_timeout_rate=min(
                1.0,
                max(0.0, _float("RESEARCH_LAB_SCORING_HEALTH_MAX_TIMEOUT_RATE", 0.10)),
            ),
            private_baseline_rebenchmark_enabled=_truthy(
                "RESEARCH_LAB_PRIVATE_BASELINE_REBENCHMARK_ENABLED",
                prod_on,
            ),
            private_baseline_concurrency=max(
                1,
                _int("RESEARCH_LAB_BENCHMARK_CONCURRENCY", 1),
            ),
            private_baseline_retry_concurrency=max(
                1,
                _int("RESEARCH_LAB_BENCHMARK_RETRY_CONCURRENCY", 2),
            ),
            private_baseline_provider_retry_rounds=max(
                0,
                _int("RESEARCH_LAB_BENCHMARK_PROVIDER_RETRY_ROUNDS", 2),
            ),
            benchmark_exa_api_key=os.getenv("RESEARCH_LAB_BENCHMARK_EXA_API_KEY", ""),
            benchmark_exa_max_rps=max(
                0.0,
                _float("RESEARCH_LAB_BENCHMARK_EXA_MAX_RPS", 0.0),
            ),
            auto_research_min_seconds=max(0, _int("RESEARCH_LAB_AUTO_RESEARCH_MIN_SECONDS", 600)),
            auto_research_max_seconds=max(1, _int("RESEARCH_LAB_AUTO_RESEARCH_MAX_SECONDS", 2700)),
            auto_research_min_iterations=max(1, _int("RESEARCH_LAB_AUTO_RESEARCH_MIN_ITERATIONS", 3)),
            auto_research_max_iterations=max(1, _int("RESEARCH_LAB_AUTO_RESEARCH_MAX_ITERATIONS", 6)),
            auto_research_draft_timeout_seconds=max(
                10,
                _int("RESEARCH_LAB_AUTO_RESEARCH_DRAFT_TIMEOUT_SECONDS", 180),
            ),
            auto_research_reflection_timeout_seconds=max(
                10,
                _int("RESEARCH_LAB_AUTO_RESEARCH_REFLECTION_TIMEOUT_SECONDS", 90),
            ),
            auto_research_max_tokens=max(
                700,
                _int("RESEARCH_LAB_AUTO_RESEARCH_MAX_TOKENS", 12000),
            ),
            auto_research_temperature=min(
                2.0,
                max(0.0, _float("RESEARCH_LAB_AUTO_RESEARCH_TEMPERATURE", 0.35)),
            ),
            auto_research_estimated_iteration_cost_usd=max(
                0.01,
                _float("RESEARCH_LAB_AUTO_RESEARCH_ESTIMATED_ITERATION_COST_USD", 0.5),
            ),
            loop_planner_enabled=_truthy("RESEARCH_LAB_LOOP_PLANNER_ENABLED", "true"),
            loop_alignment_judge_enabled=_truthy("RESEARCH_LAB_LOOP_ALIGNMENT_JUDGE_ENABLED", "true"),
            loop_novelty_strict=_truthy("RESEARCH_LAB_LOOP_NOVELTY_STRICT", "true"),
            loop_planner_model=os.getenv("RESEARCH_LAB_LOOP_PLANNER_MODEL", ""),
            loop_planner_fallback_models=_tuple_from_json_or_csv(
                os.getenv("RESEARCH_LAB_LOOP_PLANNER_FALLBACK_MODELS", ""),
                (),
            ),
            loop_planner_reasoning_effort=os.getenv("RESEARCH_LAB_LOOP_PLANNER_REASONING_EFFORT", ""),
            loop_planner_max_tokens=max(
                512,
                _int("RESEARCH_LAB_LOOP_PLANNER_MAX_TOKENS", 12000),
            ),
            loop_planner_temperature=min(
                2.0,
                max(0.0, _float("RESEARCH_LAB_LOOP_PLANNER_TEMPERATURE", 0.40)),
            ),
            loop_planner_allow_non_zdr=_truthy("RESEARCH_LAB_LOOP_PLANNER_ALLOW_NON_ZDR"),
            loop_executor_model=os.getenv("RESEARCH_LAB_LOOP_EXECUTOR_MODEL", ""),
            loop_executor_reasoning_effort=os.getenv("RESEARCH_LAB_LOOP_EXECUTOR_REASONING_EFFORT", ""),
            loop_alignment_judge_model=os.getenv("RESEARCH_LAB_LOOP_ALIGNMENT_JUDGE_MODEL", ""),
            loop_alignment_judge_reasoning_effort=os.getenv("RESEARCH_LAB_LOOP_ALIGNMENT_JUDGE_REASONING_EFFORT", ""),
            loop_alignment_judge_max_tokens=max(
                256,
                _int("RESEARCH_LAB_LOOP_ALIGNMENT_JUDGE_MAX_TOKENS", 1600),
            ),
            loop_alignment_judge_temperature=min(
                2.0,
                max(0.0, _float("RESEARCH_LAB_LOOP_ALIGNMENT_JUDGE_TEMPERATURE", 0.0)),
            ),
            reimbursement_policy_id=os.getenv(
                "RESEARCH_LAB_REIMBURSEMENT_POLICY_ID",
                "alpha-reimbursement-production-v1",
            ),
            reimbursement_min_rebate_rate=max(0.0, _float("RESEARCH_LAB_REIMBURSEMENT_MIN_REBATE_RATE", 0.25)),
            reimbursement_base_rebate_rate=max(0.0, _float("RESEARCH_LAB_REIMBURSEMENT_BASE_REBATE_RATE", 0.50)),
            reimbursement_max_rebate_rate=max(0.0, _float("RESEARCH_LAB_REIMBURSEMENT_MAX_REBATE_RATE", 1.0)),
            reimbursement_high_participation_target=max(
                0.01,
                _float("RESEARCH_LAB_REIMBURSEMENT_HIGH_PARTICIPATION_TARGET", 10.0),
            ),
            reimbursement_epochs=max(
                1,
                _int(
                    "RESEARCH_LAB_REIMBURSEMENT_EPOCHS",
                    _int("RESEARCH_LAB_REWARD_EPOCHS", 20),
                ),
            ),
            reimbursement_max_usd_per_run=max(0.0, _float("RESEARCH_LAB_REIMBURSEMENT_MAX_USD_PER_RUN", 100.0)),
            reimbursement_max_usd_per_hotkey_day=max(
                0.0,
                _float("RESEARCH_LAB_REIMBURSEMENT_MAX_USD_PER_HOTKEY_DAY", 100.0),
            ),
            reimbursement_max_usd_per_island_day=max(
                0.0,
                _float("RESEARCH_LAB_REIMBURSEMENT_MAX_USD_PER_ISLAND_DAY", 1000.0),
            ),
            reimbursement_global_budget_usd=max(
                0.0,
                _float("RESEARCH_LAB_REIMBURSEMENT_GLOBAL_BUDGET_USD", 5000.0),
            ),
            reimbursement_material_spend_ratio=min(
                1.0,
                max(0.0, _float("RESEARCH_LAB_REIMBURSEMENT_MATERIAL_SPEND_RATIO", 0.80)),
            ),
            reimbursement_default_island=os.getenv("RESEARCH_LAB_REIMBURSEMENT_DEFAULT_ISLAND", "generalist"),
            reimbursement_usd_per_0_1_percent_epoch=max(
                0.000001,
                _float("RESEARCH_LAB_REIMBURSEMENT_USD_PER_0_1_PERCENT_EPOCH", 0.162),
            ),
            lab_emission_percent=max(0.0, _float("RESEARCH_LAB_EMISSION_PERCENT", 10.0)),
            fulfillment_emission_percent=max(0.0, _float("RESEARCH_LAB_FULFILLMENT_EMISSION_PERCENT", 80.5)),
            fulfillment_leaderboard_emission_percent=max(
                0.0,
                _float("RESEARCH_LAB_FULFILLMENT_LEADERBOARD_EMISSION_PERCENT", 9.5),
            ),
            lab_reward_epochs=max(1, _int("RESEARCH_LAB_REWARD_EPOCHS", 20)),
            lab_reimbursement_allow_overpay_without_champions=_truthy(
                "RESEARCH_LAB_REIMBURSEMENT_ALLOW_OVERPAY_WITHOUT_CHAMPIONS",
                "true",
            ),
            lab_reimbursement_max_cost_multiplier_with_champions=max(
                0.0,
                _float("RESEARCH_LAB_REIMBURSEMENT_MAX_COST_MULTIPLIER_WITH_CHAMPIONS", 1.0),
            ),
            lab_reimbursement_min_alpha_percent=max(
                0.0,
                _float("RESEARCH_LAB_REIMBURSEMENT_MIN_ALPHA_PERCENT", 0.0),
            ),
            lab_champion_min_alpha_percent=max(0.0, _float("RESEARCH_LAB_CHAMPION_MIN_ALPHA_PERCENT", 2.0)),
            lab_champion_extra_alpha_percent_per_point=max(
                0.0,
                _float("RESEARCH_LAB_CHAMPION_EXTRA_ALPHA_PERCENT_PER_POINT", 0.1),
            ),
            lab_champion_max_alpha_percent=max(0.0, _float("RESEARCH_LAB_CHAMPION_MAX_ALPHA_PERCENT", 5.0)),
            lab_champion_placeholder_alpha_percent=max(
                0.0,
                _float("RESEARCH_LAB_CHAMPION_PLACEHOLDER_ALPHA_PERCENT", 0.0001),
            ),
            lab_champion_queue_trigger_ratio=min(
                1.0,
                max(0.0, _float("RESEARCH_LAB_CHAMPION_QUEUE_TRIGGER_RATIO", 0.50)),
            ),
            lab_champion_threshold_points=max(0.0, _float("RESEARCH_LAB_CHAMPION_THRESHOLD_POINTS", 2.0)),
            lab_champion_eval_days=max(1, _int("RESEARCH_LAB_CHAMPION_EVAL_DAYS", 10)),
            lab_champion_icps_per_day=max(1, _int("RESEARCH_LAB_CHAMPION_ICPS_PER_DAY", 2)),
            public_benchmark_public_icps_per_day=max(
                1,
                _int("RESEARCH_LAB_PUBLIC_BENCHMARK_PUBLIC_ICPS_PER_DAY", 3),
            ),
            public_benchmark_public_weak_per_day=max(
                0,
                _int("RESEARCH_LAB_PUBLIC_BENCHMARK_PUBLIC_WEAK_PER_DAY", 2),
            ),
            # Ratio rule (fableanalysis §6.4): expose at most 1/3 of the private
            # window (eval_days * icps_per_day = 10 * 2 = 20 by default), hence 6
            # (the old default of 10 exposed half the sealed set daily). The weak
            # fallback keeps roughly the per-day 2/3 proportion and must stay <=
            # the public total (validate_public_benchmark_split enforces this).
            public_benchmark_public_total_icps=(
                max(1, value)
                if (value := _optional_int("RESEARCH_LAB_PUBLIC_BENCHMARK_PUBLIC_TOTAL_ICPS")) is not None
                else 6
            ),
            public_benchmark_public_weak_total=(
                max(0, value)
                if (value := _optional_int("RESEARCH_LAB_PUBLIC_BENCHMARK_PUBLIC_WEAK_TOTAL")) is not None
                else 4
            ),
            improvement_threshold_points=_improvement_threshold_points(),
            private_model_manifest_uri=os.getenv(
                "RESEARCH_LAB_PRIVATE_MODEL_MANIFEST_URI",
                "s3://leadpoet-private-model-artifacts-493765492819/research-lab/sourcing-model/current.json",
            ),
            private_repo_url=os.getenv("RESEARCH_LAB_PRIVATE_REPO_URL", DEFAULT_PRIVATE_REPO_URL),
            private_repo_branch=os.getenv("RESEARCH_LAB_PRIVATE_REPO_BRANCH", "main") or "main",
            private_patch_applier_cmd=os.getenv("RESEARCH_LAB_PRIVATE_PATCH_APPLIER_CMD", ""),
            private_test_cmd=os.getenv("RESEARCH_LAB_PRIVATE_TEST_CMD", DEFAULT_PRIVATE_TEST_CMD),
            private_build_cmd=os.getenv("RESEARCH_LAB_PRIVATE_BUILD_CMD", DEFAULT_PRIVATE_BUILD_CMD),
            private_artifact_manifest_output=os.getenv(
                "RESEARCH_LAB_PRIVATE_ARTIFACT_MANIFEST_OUTPUT",
                DEFAULT_PRIVATE_ARTIFACT_MANIFEST_OUTPUT,
            ),
            private_benchmark_path=os.getenv("RESEARCH_LAB_PRIVATE_BENCHMARK_PATH", ""),
            code_edit_candidates_enabled=_truthy("RESEARCH_LAB_CODE_EDIT_CANDIDATES_ENABLED", "true"),
            code_edit_build_timeout_seconds=max(
                60,
                _int("RESEARCH_LAB_CODE_EDIT_BUILD_TIMEOUT_SECONDS", 1800),
            ),
            code_edit_allowed_paths_json=os.getenv("RESEARCH_LAB_CODE_EDIT_ALLOWED_PATHS_JSON", ""),
            code_edit_allowed_exact_paths_json=os.getenv("RESEARCH_LAB_CODE_EDIT_ALLOWED_EXACT_PATHS_JSON", ""),
            code_edit_allowed_suffixes_json=os.getenv("RESEARCH_LAB_CODE_EDIT_ALLOWED_SUFFIXES_JSON", ""),
            code_edit_source_inspection_rounds=max(
                1,
                _int("RESEARCH_LAB_CODE_EDIT_SOURCE_INSPECTION_ROUNDS", 3),
            ),
            code_edit_source_inspection_max_files=max(
                1,
                _int("RESEARCH_LAB_CODE_EDIT_SOURCE_INSPECTION_MAX_FILES", 12),
            ),
            code_edit_source_inspection_file_bytes=max(
                1024,
                _int("RESEARCH_LAB_CODE_EDIT_SOURCE_INSPECTION_FILE_BYTES", 24_000),
            ),
            code_edit_source_inspection_total_bytes=max(
                4096,
                _int("RESEARCH_LAB_CODE_EDIT_SOURCE_INSPECTION_TOTAL_BYTES", 120_000),
            ),
            code_edit_source_inspection_search_matches=max(
                1,
                _int("RESEARCH_LAB_CODE_EDIT_SOURCE_INSPECTION_SEARCH_MATCHES", 30),
            ),
            code_edit_patch_repair_attempts=max(
                0,
                _int("RESEARCH_LAB_CODE_EDIT_PATCH_REPAIR_ATTEMPTS", 2),
            ),
            stale_parent_rebase_enabled=_truthy("RESEARCH_LAB_STALE_PARENT_REBASE_ENABLED", "true"),
            stale_parent_rebase_repair_enabled=_truthy("RESEARCH_LAB_STALE_PARENT_REBASE_REPAIR_ENABLED", "true"),
            stale_parent_rebase_repair_model=os.getenv(
                "RESEARCH_LAB_STALE_PARENT_REBASE_REPAIR_MODEL",
                "anthropic/claude-sonnet-4.6",
            ),
            stale_parent_rebase_repair_timeout_seconds=max(
                30,
                _int("RESEARCH_LAB_STALE_PARENT_REBASE_REPAIR_TIMEOUT_SECONDS", 120),
            ),
            stale_parent_check_interval_seconds=max(
                1,
                _int("RESEARCH_LAB_STALE_PARENT_CHECK_INTERVAL_SECONDS", 30),
            ),
            stale_parent_rebase_max_depth=max(
                0,
                _int("RESEARCH_LAB_STALE_PARENT_REBASE_MAX_DEPTH", 3),
            ),
            score_bundle_kms_key_id=os.getenv(
                "RESEARCH_LAB_SCORE_BUNDLE_KMS_KEY_ID",
                "alias/leadpoet-research-lab-artifact-signing",
            ),
            score_bundle_signature_uri_prefix=os.getenv("RESEARCH_LAB_SCORE_BUNDLE_SIGNATURE_URI_PREFIX", ""),
            auto_research_model=os.getenv("RESEARCH_LAB_AUTO_RESEARCH_MODEL", ""),
            auto_research_reasoning_effort=os.getenv("RESEARCH_LAB_AUTO_RESEARCH_REASONING_EFFORT", ""),
            approved_auto_research_models_json=os.getenv("RESEARCH_LAB_APPROVED_AUTO_RESEARCH_MODELS_JSON", ""),
            default_auto_research_model_tier=os.getenv("RESEARCH_LAB_DEFAULT_AUTO_RESEARCH_MODEL_TIER", "default"),
            default_compute_budget_usd=_float("RESEARCH_LAB_DEFAULT_COMPUTE_BUDGET_USD", 5.0),
            min_compute_budget_usd=_float("RESEARCH_LAB_MIN_COMPUTE_BUDGET_USD", 1.0),
            max_compute_budget_usd=_float("RESEARCH_LAB_MAX_COMPUTE_BUDGET_USD", 100.0),
            topup_promising_delta_threshold=_float("RESEARCH_LAB_TOPUP_PROMISING_DELTA_THRESHOLD", 0.5),
            miner_openrouter_key_env_var=os.getenv("RESEARCH_LAB_MINER_OPENROUTER_KEY_ENV_VAR", ""),
            miner_openrouter_key_ref_env_map_json=os.getenv("RESEARCH_LAB_OPENROUTER_KEY_REF_ENV_MAP_JSON", ""),
            openrouter_key_kms_key_id=os.getenv("RESEARCH_LAB_OPENROUTER_KEY_KMS_KEY_ID", ""),
            evaluation_epoch=_int("RESEARCH_LAB_EVALUATION_EPOCH", 0),
            arweave_audit_enabled=_truthy("RESEARCH_LAB_ARWEAVE_AUDIT_ENABLED", "true"),
            arweave_audit_shadow_enabled=_truthy("RESEARCH_LAB_ARWEAVE_AUDIT_SHADOW_ENABLED"),
        )

    def approved_auto_research_models(self) -> dict[str, dict[str, Any]]:
        configured = self._decode_model_tiers()
        if configured:
            return configured
        if not self.auto_research_model:
            return {}
        default_doc = {
            "model": self.auto_research_model,
            "max_candidates": self.hosted_worker_max_candidates,
            "max_tokens": self.auto_research_max_tokens,
            "description": "Default hosted auto-research model",
        }
        reasoning_effort = _normalize_auto_research_reasoning_effort(self.auto_research_reasoning_effort)
        if reasoning_effort:
            default_doc["reasoning_effort"] = reasoning_effort
        return {
            self.default_auto_research_model_tier: default_doc
        }

    def resolve_auto_research_model(self, tier: str | None) -> tuple[str, str, dict[str, Any]]:
        tiers = self.approved_auto_research_models()
        if not tiers:
            if self.auto_research_model:
                return self.default_auto_research_model_tier, self.auto_research_model, {
                    "model": self.auto_research_model,
                    "max_candidates": self.hosted_worker_max_candidates,
                    "max_tokens": self.auto_research_max_tokens,
                }
            raise ValueError("no hosted auto-research model is configured")
        effective_tier = str(tier or self.default_auto_research_model_tier or "default")
        if not tier and effective_tier not in tiers:
            effective_tier = sorted(tiers)[0]
        if effective_tier not in tiers:
            raise ValueError(f"auto-research model tier is not approved: {effective_tier}")
        doc = dict(tiers[effective_tier])
        model = str(doc.get("model") or "")
        if not model:
            raise ValueError(f"approved auto-research model tier has no model: {effective_tier}")
        if "reasoning_effort" not in doc:
            reasoning_effort = _normalize_auto_research_reasoning_effort(self.auto_research_reasoning_effort)
            if reasoning_effort:
                doc["reasoning_effort"] = reasoning_effort
        return effective_tier, model, doc

    def clamp_compute_budget_usd(self, value: float | int | str | None) -> float:
        try:
            budget = float(self.default_compute_budget_usd if value is None else value)
        except (TypeError, ValueError):
            budget = float(self.default_compute_budget_usd)
        lower = max(0.0, float(self.min_compute_budget_usd))
        upper = max(lower, float(self.max_compute_budget_usd))
        return min(max(budget, lower), upper)

    def _decode_model_tiers(self) -> dict[str, dict[str, Any]]:
        if not self.approved_auto_research_models_json:
            return {}
        try:
            decoded = json.loads(self.approved_auto_research_models_json)
        except json.JSONDecodeError as exc:
            logger.warning("Invalid RESEARCH_LAB_APPROVED_AUTO_RESEARCH_MODELS_JSON: %s", exc)
            return {}
        if not isinstance(decoded, Mapping):
            logger.warning("RESEARCH_LAB_APPROVED_AUTO_RESEARCH_MODELS_JSON must decode to an object")
            return {}
        tiers: dict[str, dict[str, Any]] = {}
        for name, value in decoded.items():
            if not isinstance(value, Mapping):
                logger.warning("Skipping invalid auto-research model tier %s: expected object", name)
                continue
            model = str(value.get("model") or "")
            if not model:
                logger.warning("Skipping invalid auto-research model tier %s: missing model", name)
                continue
            tiers[str(name)] = {
                key: item
                for key, item in dict(value).items()
                if key
                in {
                    "model",
                    "max_candidates",
                    "max_compute_budget_usd",
                    "max_tokens",
                    "description",
                    "reasoning_effort",
                }
            }
            reasoning_effort = _normalize_auto_research_reasoning_effort(tiers[str(name)].get("reasoning_effort"))
            if reasoning_effort:
                tiers[str(name)]["reasoning_effort"] = reasoning_effort
            else:
                tiers[str(name)].pop("reasoning_effort", None)
        return tiers

    def code_edit_allowed_path_prefixes(self) -> tuple[str, ...]:
        from research_lab.code_editing import DEFAULT_ALLOWED_PATH_PREFIXES

        return _tuple_from_json_or_csv(
            self.code_edit_allowed_paths_json,
            DEFAULT_ALLOWED_PATH_PREFIXES,
        )

    def code_edit_allowed_exact_paths(self) -> tuple[str, ...]:
        from research_lab.code_editing import DEFAULT_ALLOWED_EXACT_PATHS

        return _tuple_from_json_or_csv(
            self.code_edit_allowed_exact_paths_json,
            DEFAULT_ALLOWED_EXACT_PATHS,
        )

    def code_edit_allowed_suffixes(self) -> tuple[str, ...]:
        from research_lab.code_editing import DEFAULT_ALLOWED_SUFFIXES

        return _tuple_from_json_or_csv(
            self.code_edit_allowed_suffixes_json,
            DEFAULT_ALLOWED_SUFFIXES,
        )

    def live_mutation_flags(self) -> dict[str, bool]:
        return {
            "RESEARCH_LAB_MINER_SUBMISSIONS_ENABLED": self.miner_submissions_enabled,
            "RESEARCH_LAB_PAID_LOOPS_ENABLED": self.paid_loops_enabled,
            "RESEARCH_LAB_LOOP_TOPUPS_ENABLED": self.loop_topups_enabled,
            "RESEARCH_LAB_HOSTED_RUNS_ENABLED": self.hosted_runs_enabled,
            "RESEARCH_LAB_PROBES_ENABLED": self.probes_enabled,
            "RESEARCH_LAB_CROWNING_ENABLED": self.crowning_enabled,
            "RESEARCH_LAB_REIMBURSEMENTS_ENABLED": self.reimbursements_enabled,
            "RESEARCH_LAB_WEIGHT_MUTATION_ENABLED": self.weight_mutation_enabled,
            "RESEARCH_LAB_FULFILLMENT_MUTATION_ENABLED": self.fulfillment_mutation_enabled,
            "RESEARCH_LAB_AUTO_PROMOTION_ENABLED": self.auto_promotion_enabled,
            "RESEARCH_LAB_AUTO_COMMIT_ENABLED": self.auto_commit_enabled,
        }

    def reimbursement_policy_doc(self, *, enabled: bool | None = None) -> dict[str, object]:
        return {
            "policy_id": self.reimbursement_policy_id,
            "enabled": self.reimbursements_enabled if enabled is None else bool(enabled),
            "min_rebate_rate": self.reimbursement_min_rebate_rate,
            "base_rebate_rate": self.reimbursement_base_rebate_rate,
            "max_rebate_rate": self.reimbursement_max_rebate_rate,
            "high_participation_target": self.reimbursement_high_participation_target,
            "reimbursement_epochs": self.reimbursement_epochs,
            "max_usd_per_run": self.reimbursement_max_usd_per_run,
            "max_usd_per_hotkey_day": self.reimbursement_max_usd_per_hotkey_day,
            "max_usd_per_island_day": self.reimbursement_max_usd_per_island_day,
            "global_budget_usd": self.reimbursement_global_budget_usd,
            "include_loop_start_fee_in_base": False,
            "material_spend_ratio": self.reimbursement_material_spend_ratio,
            "default_island": self.reimbursement_default_island,
            "usd_per_0_1_percent_epoch": self.reimbursement_usd_per_0_1_percent_epoch,
            "distinct_funded_hotkey_weight": 1,
            "paid_loop_weight": 1,
            "unique_brief_weight": 1,
            "research_lab_emission_percent": self.lab_emission_percent,
            "fulfillment_emission_percent": self.fulfillment_emission_percent,
            "fulfillment_leaderboard_emission_percent": self.fulfillment_leaderboard_emission_percent,
            "reward_epochs": self.lab_reward_epochs,
            "reimbursement_allow_overpay_without_champions": (
                self.lab_reimbursement_allow_overpay_without_champions
            ),
            "reimbursement_max_cost_multiplier_with_champions": (
                self.lab_reimbursement_max_cost_multiplier_with_champions
            ),
            "reimbursement_min_alpha_percent": self.lab_reimbursement_min_alpha_percent,
            "champion_min_alpha_percent": self.lab_champion_min_alpha_percent,
            "champion_extra_alpha_percent_per_point": self.lab_champion_extra_alpha_percent_per_point,
            "champion_max_alpha_percent": self.lab_champion_max_alpha_percent,
            "champion_placeholder_alpha_percent": self.lab_champion_placeholder_alpha_percent,
            "champion_queue_trigger_ratio": self.lab_champion_queue_trigger_ratio,
            "champion_threshold_points": self.lab_champion_threshold_points,
            "champion_eval_days": self.lab_champion_eval_days,
            "champion_icps_per_day": self.lab_champion_icps_per_day,
            "public_benchmark_public_icps_per_day": self.public_benchmark_public_icps_per_day,
            "public_benchmark_public_weak_per_day": self.public_benchmark_public_weak_per_day,
            "public_benchmark_public_total_icps": self.public_benchmark_public_total_icps,
            "public_benchmark_public_weak_total": self.public_benchmark_public_weak_total,
        }

    def validate_public_benchmark_split(self) -> None:
        total_icps = self.lab_champion_eval_days * self.lab_champion_icps_per_day
        if self.public_benchmark_public_total_icps is not None:
            if self.public_benchmark_public_total_icps >= total_icps:
                raise ValueError("RESEARCH_LAB_PUBLIC_BENCHMARK_PUBLIC_TOTAL_ICPS must leave private holdout ICPs")
            weak_total = self.public_benchmark_public_weak_total
            if weak_total is None:
                weak_total = self.public_benchmark_public_total_icps // 2
            if weak_total > self.public_benchmark_public_total_icps:
                raise ValueError(
                    "RESEARCH_LAB_PUBLIC_BENCHMARK_PUBLIC_WEAK_TOTAL must be <= "
                    "RESEARCH_LAB_PUBLIC_BENCHMARK_PUBLIC_TOTAL_ICPS"
                )
            return
        if self.public_benchmark_public_icps_per_day > self.lab_champion_icps_per_day:
            raise ValueError(
                "RESEARCH_LAB_PUBLIC_BENCHMARK_PUBLIC_ICPS_PER_DAY must be <= "
                "RESEARCH_LAB_CHAMPION_ICPS_PER_DAY"
            )
        if self.public_benchmark_public_weak_per_day > self.public_benchmark_public_icps_per_day:
            raise ValueError(
                "RESEARCH_LAB_PUBLIC_BENCHMARK_PUBLIC_WEAK_PER_DAY must be <= "
                "RESEARCH_LAB_PUBLIC_BENCHMARK_PUBLIC_ICPS_PER_DAY"
            )
        if self.public_benchmark_public_icps_per_day >= self.lab_champion_icps_per_day:
            raise ValueError(
                "RESEARCH_LAB_PUBLIC_BENCHMARK_PUBLIC_ICPS_PER_DAY must leave private holdout ICPs"
            )

    def public_status(self) -> dict[str, object]:
        return {
            "api_enabled": self.api_enabled,
            "production_writes_enabled": self.production_writes_enabled,
            "miner_submissions_enabled": self.miner_submissions_enabled,
            "paid_loops_enabled": self.paid_loops_enabled,
            "loop_topups_enabled": self.loop_topups_enabled,
            "probes_enabled": self.probes_enabled,
            "hosted_runs_enabled": self.hosted_runs_enabled,
            "receipts_enabled": self.receipts_enabled,
            "evaluation_bundles_enabled": self.evaluation_bundles_enabled,
            "reports_enabled": self.reports_enabled,
            "public_activity_enabled": self.public_activity_enabled,
            "shadow_flags": {
                "RESEARCH_LAB_SHADOW_BUNDLES_ENABLED": self.shadow_bundles_enabled,
                "RESEARCH_LAB_SHADOW_WEIGHTS_ENABLED": self.shadow_weights_enabled,
                "RESEARCH_LAB_SHADOW_REIMBURSEMENTS_ENABLED": self.shadow_reimbursements_enabled,
            },
            "live_mutation_flags": self.live_mutation_flags(),
            "loop_start_fee_usd": self.loop_start_fee_usd,
            "allowed_research_areas": list(self.allowed_research_islands),
            "miner_openrouter_key_required": self.miner_openrouter_key_required,
            "openrouter_key_registration_enabled": bool(self.openrouter_key_kms_key_id),
            "reimbursement": {
                "enabled": self.reimbursements_enabled,
                "shadow_enabled": self.shadow_reimbursements_enabled,
                "policy_id": self.reimbursement_policy_id,
                "default_island": self.reimbursement_default_island,
                "min_rebate_rate": self.reimbursement_min_rebate_rate,
                "base_rebate_rate": self.reimbursement_base_rebate_rate,
                "max_rebate_rate": self.reimbursement_max_rebate_rate,
                "default_rebate_rate_estimate": self.reimbursement_base_rebate_rate,
                "reimbursement_epochs": self.reimbursement_epochs,
                "material_spend_ratio": self.reimbursement_material_spend_ratio,
                "usd_per_0_1_percent_epoch": self.reimbursement_usd_per_0_1_percent_epoch,
                "loop_start_fee_included": False,
                "lab_emission_percent": self.lab_emission_percent,
                "fulfillment_emission_percent": self.fulfillment_emission_percent,
                "fulfillment_leaderboard_emission_percent": self.fulfillment_leaderboard_emission_percent,
                "reward_epochs": self.lab_reward_epochs,
                "allow_overpay_without_champions": self.lab_reimbursement_allow_overpay_without_champions,
                "max_cost_multiplier_with_champions": self.lab_reimbursement_max_cost_multiplier_with_champions,
                "champion_min_alpha_percent": self.lab_champion_min_alpha_percent,
                "champion_extra_alpha_percent_per_point": self.lab_champion_extra_alpha_percent_per_point,
                "champion_max_alpha_percent": self.lab_champion_max_alpha_percent,
                "champion_placeholder_alpha_percent": self.lab_champion_placeholder_alpha_percent,
                "champion_threshold_points": self.lab_champion_threshold_points,
                "champion_eval_days": self.lab_champion_eval_days,
                "champion_icps_per_day": self.lab_champion_icps_per_day,
                "public_benchmark_public_icps_per_day": self.public_benchmark_public_icps_per_day,
                "public_benchmark_public_weak_per_day": self.public_benchmark_public_weak_per_day,
                "public_benchmark_public_total_icps": self.public_benchmark_public_total_icps,
                "public_benchmark_public_weak_total": self.public_benchmark_public_weak_total,
                "improvement_threshold_points": self.improvement_threshold_points,
            },
            "arweave_audit": {
                "enabled": self.arweave_audit_enabled,
                "shadow_enabled": self.arweave_audit_shadow_enabled,
                "event_type": "RESEARCH_LAB_EPOCH_AUDIT",
            },
            "hosted_worker": {
                "enabled": self.hosted_worker_enabled,
                "dry_run": self.hosted_worker_dry_run,
                "poll_seconds": self.hosted_worker_poll_seconds,
                "max_candidates": self.hosted_worker_max_candidates,
                "worker_id": self.hosted_worker_id,
                "worker_index": self.hosted_worker_index,
                "total_workers": self.hosted_worker_total_workers,
                "queue_fetch_limit": self.hosted_worker_queue_fetch_limit,
                "require_proxy": self.hosted_worker_require_proxy,
                "worker_proxy_configured": bool(self.hosted_worker_proxy_url),
                "auto_research_min_seconds": self.auto_research_min_seconds,
                "auto_research_max_seconds": self.auto_research_max_seconds,
                "auto_research_min_iterations": self.auto_research_min_iterations,
                "auto_research_max_iterations": self.auto_research_max_iterations,
                "auto_research_draft_timeout_seconds": self.auto_research_draft_timeout_seconds,
                "auto_research_reflection_timeout_seconds": self.auto_research_reflection_timeout_seconds,
                "auto_research_max_tokens": self.auto_research_max_tokens,
                "auto_research_temperature": self.auto_research_temperature,
                "auto_research_estimated_iteration_cost_usd": self.auto_research_estimated_iteration_cost_usd,
                "loop_planner": {
                    "enabled": self.loop_planner_enabled,
                    "model_configured": bool(self.loop_planner_model),
                    "fallback_model_count": len(self.loop_planner_fallback_models),
                    "reasoning_effort": _normalize_auto_research_reasoning_effort(
                        self.loop_planner_reasoning_effort
                    ),
                    "max_tokens": self.loop_planner_max_tokens,
                    "temperature": self.loop_planner_temperature,
                    "allow_non_zdr": self.loop_planner_allow_non_zdr,
                },
                "loop_executor": {
                    "model_configured": bool(self.loop_executor_model),
                    "reasoning_effort": _normalize_auto_research_reasoning_effort(
                        self.loop_executor_reasoning_effort
                    ),
                },
                "loop_alignment_judge": {
                    "enabled": self.loop_alignment_judge_enabled,
                    "novelty_strict": self.loop_novelty_strict,
                    "model_configured": bool(self.loop_alignment_judge_model),
                    "reasoning_effort": _normalize_auto_research_reasoning_effort(
                        self.loop_alignment_judge_reasoning_effort
                    ),
                    "max_tokens": self.loop_alignment_judge_max_tokens,
                    "temperature": self.loop_alignment_judge_temperature,
                },
                "private_model_manifest_uri_configured": bool(self.private_model_manifest_uri),
                "scoring_owner": "gateway_qualification_workers",
                "scoring_worker_enabled": self.scoring_worker_enabled,
                "scoring_worker_poll_seconds": self.scoring_worker_poll_seconds,
                "scoring_worker_max_candidates": self.scoring_worker_max_candidates,
                "scoring_worker_id": self.scoring_worker_id,
                "scoring_worker_index": self.scoring_worker_index,
                "scoring_worker_total_workers": self.scoring_worker_total_workers,
                "scoring_worker_require_proxy": self.scoring_worker_require_proxy,
                "scoring_worker_proxy_configured": bool(self.scoring_worker_proxy_url),
                "scoring_worker_baseline_not_ready_retry_seconds": (
                    self.scoring_worker_baseline_not_ready_retry_seconds
                ),
                "scoring_worker_retryable_failure_retry_seconds": (
                    self.scoring_worker_retryable_failure_retry_seconds
                ),
                "private_model_docker_global_proxy_enabled": self.private_model_docker_global_proxy_enabled,
                "scoring_worker_allow_partial_icp_window": self.scoring_worker_allow_partial_icp_window,
                "scoring_health_gate": {
                    "enabled": self.scoring_health_gate_enabled,
                    "max_reference_runtime_failure_rate": self.scoring_health_max_reference_runtime_failure_rate,
                    "max_candidate_runtime_failure_rate": self.scoring_health_max_candidate_runtime_failure_rate,
                    "max_reference_zero_company_rate": self.scoring_health_max_reference_zero_company_rate,
                    "max_candidate_zero_company_rate": self.scoring_health_max_candidate_zero_company_rate,
                    "max_provider_error_rate": self.scoring_health_max_provider_error_rate,
                    "max_timeout_rate": self.scoring_health_max_timeout_rate,
                },
                "private_baseline_rebenchmark_enabled": self.private_baseline_rebenchmark_enabled,
                "private_baseline_concurrency": self.private_baseline_concurrency,
                "private_baseline_retry_concurrency": self.private_baseline_retry_concurrency,
                "private_baseline_provider_retry_rounds": self.private_baseline_provider_retry_rounds,
                "benchmark_exa_key_configured": bool(self.benchmark_exa_api_key),
                "benchmark_exa_max_rps": self.benchmark_exa_max_rps,
                "auto_promotion_enabled": self.auto_promotion_enabled,
                "auto_commit_enabled": self.auto_commit_enabled,
                "private_repo_configured": bool(self.private_repo_url),
                "private_patch_applier_configured": bool(self.private_patch_applier_cmd),
                "private_test_cmd_configured": bool(self.private_test_cmd),
                "private_build_cmd_configured": bool(self.private_build_cmd),
                "private_artifact_manifest_output_configured": bool(self.private_artifact_manifest_output),
                "code_edit_source_inspection": {
                    "rounds": self.code_edit_source_inspection_rounds,
                    "max_files": self.code_edit_source_inspection_max_files,
                    "file_bytes": self.code_edit_source_inspection_file_bytes,
                    "total_bytes": self.code_edit_source_inspection_total_bytes,
                    "search_matches": self.code_edit_source_inspection_search_matches,
                },
                "code_edit_patch_repair_attempts": self.code_edit_patch_repair_attempts,
                "stale_parent_rebase": {
                    "enabled": self.stale_parent_rebase_enabled,
                    "repair_enabled": self.stale_parent_rebase_repair_enabled,
                    "repair_model_configured": bool(self.stale_parent_rebase_repair_model),
                    "check_interval_seconds": self.stale_parent_check_interval_seconds,
                    "max_depth": self.stale_parent_rebase_max_depth,
                    "repair_operator_key_configured": bool(
                        os.getenv("RESEARCH_LAB_STALE_PARENT_REBASE_OPENROUTER_API_KEY")
                        or os.getenv("OPENROUTER_API_KEY")
                    ),
                },
                "auto_research_model_configured": bool(self.auto_research_model),
                "auto_research_reasoning_effort": _normalize_auto_research_reasoning_effort(
                    self.auto_research_reasoning_effort
                ),
                "approved_model_tiers": {
                    tier: {
                        "model_configured": bool(doc.get("model")),
                        "max_candidates": doc.get("max_candidates", self.hosted_worker_max_candidates),
                        "max_compute_budget_usd": doc.get("max_compute_budget_usd", self.max_compute_budget_usd),
                        "max_tokens": doc.get("max_tokens", self.auto_research_max_tokens),
                        "reasoning_effort": doc.get("reasoning_effort"),
                        "description": doc.get("description"),
                    }
                    for tier, doc in self.approved_auto_research_models().items()
                },
                "default_model_tier": self.default_auto_research_model_tier,
                "default_compute_budget_usd": self.default_compute_budget_usd,
                "min_compute_budget_usd": self.min_compute_budget_usd,
                "max_compute_budget_usd": self.max_compute_budget_usd,
            },
        }
