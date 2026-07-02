"""Post-merge production shadow monitor (audit §9.2 / follow-up item 4.2).

After a candidate-derived merge (``active_version_created``), the previous
champion artifact is re-evaluated in a read-only shadow run on the SAME daily
benchmark window the new champion was scored on, for
``RESEARCH_LAB_SHADOW_WINDOW_DAYS`` benchmark days. The per-day
``shadow_live_diff`` (live new-champion aggregate minus shadow previous-champion
aggregate on identical ICPs) is the revert evidence the merge metric cannot
provide on its own: with same-model noise of sd ~= 1.9 points per benchmark run
(audit §5.1) a bad promotion will eventually clear the merge bar, and without a
shadow window it is permanent.

Strictly read-only against lab state:

* The monitor NEVER writes promotion, version, benchmark, or dispatch events.
  Its only outputs are S3 report/state docs and structured log lines. All DB
  access goes through the injected read-only ``select_many``.
* Window state is persisted as an S3 doc (derived from prior shadow report
  artifacts on re-poll) — deliberately NO new tables.
* Every emitted doc carries the ``production_shadow`` read-only markers and is
  checked with ``assert_shadow_output_read_only``.

ENV HYGIENE — RUN AS A SEPARATE PROCESS (verified hazard, audit §9.2):
``assert_shadow_output_read_only`` hard-fails when any env in
``research_lab.production_shadow.LIVE_MUTATION_FLAG_ENV_NAMES`` is truthy —
notably ``RESEARCH_LAB_PAID_LOOPS_ENABLED``, which the live worker fleet sets.
Do NOT run this monitor inside the gateway/worker environment. Launch it as its
own process (cron/systemd) with:

* required: Supabase read credentials (gateway DB client env), AWS credentials
  (S3 reports + manifest loads, docker ECR pulls), docker, and the provider
  budget envs a normal benchmark uses (``EXA_API_KEY``/
  ``RESEARCH_LAB_BENCHMARK_EXA_API_KEY``, ``RESEARCH_LAB_BENCHMARK_EXA_MAX_RPS``,
  ``SCRAPINGDOG_API_KEY``/``QUALIFICATION_SCRAPINGDOG_API_KEY``, an OpenRouter
  key for the scorer path) plus ``RESEARCH_LAB_SHADOW_MONITOR_ENABLED=true``;
* forbidden: EVERY name in ``LIVE_MUTATION_FLAG_ENV_NAMES`` must be unset or
  falsy. The monitor refuses to start otherwise (exit code 3) and re-checks
  before each shadow evaluation.

CLI::

    python3 -m gateway.research_lab.shadow_monitor --watch
    python3 -m gateway.research_lab.shadow_monitor --once
    python3 -m gateway.research_lab.shadow_monitor --window <active_version_id>

Flags (all monitor-local, read from the environment):

* ``RESEARCH_LAB_SHADOW_MONITOR_ENABLED`` — default FALSE (entrypoint gate).
* ``RESEARCH_LAB_SHADOW_WINDOW_DAYS`` — benchmark days to shadow (default 5).
* ``RESEARCH_LAB_SHADOW_ALERT_THRESHOLD_POINTS`` — cumulative window diff more
  negative than this alerts (default 2.0 ~= 1 se per §5.1).
* ``RESEARCH_LAB_SHADOW_EARLY_ALERT_DAY_POINTS`` — 2 consecutive observed days
  each more negative than this alert early (default 1.0).
* ``RESEARCH_LAB_SHADOW_MONITOR_POLL_SECONDS`` — watch poll interval (900).
* ``RESEARCH_LAB_SHADOW_MONITOR_CONCURRENCY`` — ICP eval concurrency (1).
* ``RESEARCH_LAB_SHADOW_MODEL_TIMEOUT_SECONDS`` — per-ICP docker timeout (1800).
* ``RESEARCH_LAB_SHADOW_REPORT_URI_PREFIX`` — optional s3:// prefix for reports;
  defaults to deriving from the merged champion's manifest URI.
* ``RESEARCH_LAB_SHADOW_WINDOW_GRACE_DAYS`` — calendar grace beyond the window
  before an under-observed window is closed (3).
* ``RESEARCH_LAB_SHADOW_DISCOVERY_LIMIT`` — merge events polled per tick (20).

Failure model: a crash mid-day loses only that day's in-flight evaluation; the
next poll re-runs the day from the recorded state doc (day reports are
overwritten idempotently). Per-ICP checkpointing is intentionally out of scope
for a read-only monitor.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
import json
import logging
import math
import os
from typing import Any, Awaitable, Callable, Mapping, Protocol, Sequence

from gateway.research_lab.logging_utils import compact_ref, format_worker_line
from research_lab.canonical import sha256_json
from research_lab.eval import (
    DockerPrivateModelRunner,
    DockerPrivateModelSpec,
    PrivateModelArtifactManifest,
    PrivateModelRuntimeError,
    ensure_private_model_outputs,
    load_private_artifact_manifest,
    private_model_env_passthrough,
    validate_private_model_artifact_manifest,
)
from research_lab.eval.evaluator import QualificationStyleCompanyScorer
from research_lab.production_shadow import (
    LIVE_MUTATION_FLAG_ENV_NAMES,
    ProductionShadowFlags,
    assert_shadow_output_read_only,
    compute_shadow_live_diff,
)


logger = logging.getLogger(__name__)

TRUTHY = {"1", "true", "yes", "on"}

SHADOW_MONITOR_VERSION = "post-merge-shadow-monitor-v0.1.0"
SHADOW_REGRESSION_ALERT_LINE = "research_lab_shadow_regression_alert"
SHADOW_RUN_CONTEXT = {"mode": "post_merge_shadow"}

SHADOW_MONITOR_ENABLED_ENV = "RESEARCH_LAB_SHADOW_MONITOR_ENABLED"
SHADOW_WINDOW_DAYS_ENV = "RESEARCH_LAB_SHADOW_WINDOW_DAYS"
SHADOW_ALERT_THRESHOLD_ENV = "RESEARCH_LAB_SHADOW_ALERT_THRESHOLD_POINTS"
SHADOW_EARLY_ALERT_DAY_POINTS_ENV = "RESEARCH_LAB_SHADOW_EARLY_ALERT_DAY_POINTS"
SHADOW_POLL_SECONDS_ENV = "RESEARCH_LAB_SHADOW_MONITOR_POLL_SECONDS"
SHADOW_CONCURRENCY_ENV = "RESEARCH_LAB_SHADOW_MONITOR_CONCURRENCY"
SHADOW_MODEL_TIMEOUT_ENV = "RESEARCH_LAB_SHADOW_MODEL_TIMEOUT_SECONDS"
SHADOW_REPORT_URI_PREFIX_ENV = "RESEARCH_LAB_SHADOW_REPORT_URI_PREFIX"
SHADOW_WINDOW_GRACE_DAYS_ENV = "RESEARCH_LAB_SHADOW_WINDOW_GRACE_DAYS"
SHADOW_DISCOVERY_LIMIT_ENV = "RESEARCH_LAB_SHADOW_DISCOVERY_LIMIT"

# Observed empirically on 2026-07-01: the same base model scored sd ~= 1.9 on
# the same public ICPs across 11 evaluations in one day (audit §5.1). Any
# alert threshold below ~1 se would page on pure benchmark noise.
SAME_MODEL_BENCHMARK_SD_POINTS = 1.9
EARLY_ALERT_CONSECUTIVE_DAYS = 2

_MILLIPOINTS = 1000


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in TRUTHY


def _env_int(env: Mapping[str, str], name: str, default: int) -> int:
    try:
        return int(str(env.get(name, default)).strip())
    except (TypeError, ValueError):
        return default


def _env_float(env: Mapping[str, str], name: str, default: float) -> float:
    try:
        return float(str(env.get(name, default)).strip())
    except (TypeError, ValueError):
        return default


class ShadowMonitorEnvHygieneError(RuntimeError):
    """A live-mutation env flag is set in the shadow monitor process.

    ``assert_shadow_output_read_only`` treats the env names in
    ``LIVE_MUTATION_FLAG_ENV_NAMES`` as proof the process can mutate live lab
    state; the monitor must run in a separate process where all of them are
    unset (module docstring, audit §9.2 hazard).
    """


class ShadowWindowSetupError(RuntimeError):
    """A shadow window cannot be constructed (missing lineage/manifest/prefix)."""


@dataclass(frozen=True)
class ShadowMonitorSettings:
    """Runtime knobs for the shadow monitor. Defaults are inert (enabled=False)."""

    enabled: bool = False
    window_days: int = 5
    alert_threshold_points: float = 2.0
    early_alert_day_points: float = 1.0
    early_alert_consecutive_days: int = EARLY_ALERT_CONSECUTIVE_DAYS
    poll_seconds: int = 900
    concurrency: int = 1
    model_timeout_seconds: int = 1800
    report_uri_prefix: str = ""
    grace_days: int = 3
    discovery_limit: int = 20

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "ShadowMonitorSettings":
        env = os.environ if env is None else env
        return cls(
            enabled=_truthy(env.get(SHADOW_MONITOR_ENABLED_ENV, "false")),
            window_days=max(1, _env_int(env, SHADOW_WINDOW_DAYS_ENV, 5)),
            alert_threshold_points=abs(_env_float(env, SHADOW_ALERT_THRESHOLD_ENV, 2.0)),
            early_alert_day_points=abs(_env_float(env, SHADOW_EARLY_ALERT_DAY_POINTS_ENV, 1.0)),
            poll_seconds=max(30, _env_int(env, SHADOW_POLL_SECONDS_ENV, 900)),
            concurrency=max(1, _env_int(env, SHADOW_CONCURRENCY_ENV, 1)),
            model_timeout_seconds=max(60, _env_int(env, SHADOW_MODEL_TIMEOUT_ENV, 1800)),
            report_uri_prefix=str(env.get(SHADOW_REPORT_URI_PREFIX_ENV, "") or "").strip(),
            grace_days=max(0, _env_int(env, SHADOW_WINDOW_GRACE_DAYS_ENV, 3)),
            discovery_limit=max(1, _env_int(env, SHADOW_DISCOVERY_LIMIT_ENV, 20)),
        )


# ---------------------------------------------------------------------------
# Read-only guarantees
# ---------------------------------------------------------------------------


def ensure_shadow_process_read_only(env: Mapping[str, str] | None = None) -> dict[str, bool]:
    """Fail hard when the process carries any live-mutation env flag.

    Returns the (all-false) live mutation flag map used to stamp reports.
    """

    env = os.environ if env is None else env
    flags = ProductionShadowFlags.from_env(env)
    enabled = [name for name, value in flags.live_mutation_flags().items() if value]
    if enabled:
        raise ShadowMonitorEnvHygieneError(
            "shadow monitor must run in a separate process without live-mutation env flags; "
            f"unset {', '.join(sorted(enabled))} (guarded set: {', '.join(LIVE_MUTATION_FLAG_ENV_NAMES)})"
        )
    return flags.live_mutation_flags()


def read_only_stamp(env: Mapping[str, str] | None = None) -> dict[str, Any]:
    """The production_shadow read-only marker fields for every emitted doc."""

    flags = ProductionShadowFlags.from_env(os.environ if env is None else env)
    return {
        "shadow_only": True,
        "read_only": True,
        "submission_allowed": False,
        "on_chain_submission_allowed": False,
        "mutation_guards": {
            "live_mutation_flags": flags.live_mutation_flags(),
            "forbidden_runtime_actions": [
                "write_promotion_events",
                "write_version_events",
                "write_benchmark_bundles",
                "write_dispatch_events",
                "write_production_supabase",
            ],
        },
    }


def stamped_read_only_doc(doc: Mapping[str, Any], env: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Stamp ``doc`` with read-only markers and embed passing guard evidence.

    Honors ``assert_shadow_output_read_only`` exactly as the dormant apparatus
    does: with a truthy live-mutation env (e.g. RESEARCH_LAB_PAID_LOOPS_ENABLED)
    the guard fails and this raises instead of emitting the artifact.
    """

    stamped = {**dict(doc), **read_only_stamp(env)}
    evidence = assert_shadow_output_read_only(stamped)
    if not evidence["passed"]:
        raise ShadowMonitorEnvHygieneError(
            f"shadow output failed read-only guard: {evidence['errors']} "
            f"(enabled live flags: {evidence['enabled_live_mutation_flags']})"
        )
    return {**stamped, "read_only_evidence": evidence}


# ---------------------------------------------------------------------------
# Injectable dependencies (evaluation/DB/S3 are fakeable in tests)
# ---------------------------------------------------------------------------


class ShadowReportStore(Protocol):
    def get_json(self, uri: str) -> dict[str, Any] | None: ...

    def put_json(self, uri: str, doc: Mapping[str, Any]) -> None: ...


SelectMany = Callable[..., Awaitable[list[dict[str, Any]]]]
RunnerFactory = Callable[[PrivateModelArtifactManifest], Callable[..., Any]]


@dataclass(frozen=True)
class ShadowMonitorDeps:
    """Side-effect channels. The DB channel is read-only by construction —
    the monitor is never handed an event/row writer."""

    select_many: SelectMany
    load_manifest: Callable[[str], Mapping[str, Any]]
    runner_factory: RunnerFactory
    scorer_factory: Callable[[], Any]
    report_store: ShadowReportStore
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc)


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    text = str(uri or "")
    if not text.startswith("s3://"):
        raise ShadowWindowSetupError(f"expected s3:// URI, got {text[:80]!r}")
    bucket, sep, key = text[5:].partition("/")
    if not bucket or not sep or not key:
        raise ShadowWindowSetupError(f"malformed s3 URI: {text[:80]!r}")
    return bucket, key


class S3ShadowReportStore:
    """Default report/state store; mirrors the scoring-progress S3 doc pattern."""

    def get_json(self, uri: str) -> dict[str, Any] | None:
        import boto3  # type: ignore

        bucket, key = _parse_s3_uri(uri)
        try:
            body = boto3.client("s3").get_object(Bucket=bucket, Key=key)["Body"].read()
            doc = json.loads(body.decode("utf-8"))
        except Exception:
            return None
        return dict(doc) if isinstance(doc, Mapping) else None

    def put_json(self, uri: str, doc: Mapping[str, Any]) -> None:
        import boto3  # type: ignore

        bucket, key = _parse_s3_uri(uri)
        boto3.client("s3").put_object(
            Bucket=bucket,
            Key=key,
            Body=json.dumps(doc, sort_keys=True, default=str).encode("utf-8"),
            ContentType="application/json",
        )


def _shadow_scoring_env() -> dict[str, str]:
    """Provider budget env for the shadow container, like a normal benchmark.

    Mirrors the daily baseline's env (host provider keys, with the dedicated
    benchmark Exa budget overriding when configured) without importing
    scoring_worker internals.
    """

    env: dict[str, str] = {}
    for name in (
        "EXA_API_KEY",
        "EXA_MAX_RPS",
        "SCRAPINGDOG_API_KEY",
        "QUALIFICATION_SCRAPINGDOG_API_KEY",
        "OPENROUTER_API_KEY",
        "QUALIFICATION_OPENROUTER_API_KEY",
        "OPENROUTER_KEY",
    ):
        value = os.getenv(name)
        if value:
            env[name] = value
    benchmark_exa_key = os.getenv("RESEARCH_LAB_BENCHMARK_EXA_API_KEY")
    if benchmark_exa_key:
        env["EXA_API_KEY"] = benchmark_exa_key
    try:
        benchmark_exa_rps = float(os.getenv("RESEARCH_LAB_BENCHMARK_EXA_MAX_RPS", "0") or 0.0)
    except ValueError:
        benchmark_exa_rps = 0.0
    if benchmark_exa_rps > 0:
        env["EXA_MAX_RPS"] = str(benchmark_exa_rps)
    return env


def _default_runner_factory(settings: ShadowMonitorSettings) -> RunnerFactory:
    def _build(artifact: PrivateModelArtifactManifest) -> Callable[..., Any]:
        include_proxy = _truthy(os.getenv("RESEARCH_LAB_PRIVATE_MODEL_DOCKER_GLOBAL_PROXY_ENABLED"))
        return DockerPrivateModelRunner(
            DockerPrivateModelSpec(
                image_digest=artifact.image_digest,
                timeout_seconds=settings.model_timeout_seconds,
                env_passthrough=private_model_env_passthrough(include_proxy=include_proxy),
                extra_env=_shadow_scoring_env(),
            )
        )

    return _build


def default_shadow_monitor_deps(settings: ShadowMonitorSettings) -> ShadowMonitorDeps:
    from gateway.research_lab.store import select_many

    return ShadowMonitorDeps(
        select_many=select_many,
        load_manifest=load_private_artifact_manifest,
        runner_factory=_default_runner_factory(settings),
        scorer_factory=QualificationStyleCompanyScorer,
        report_store=S3ShadowReportStore(),
        now=lambda: datetime.now(timezone.utc),
    )


# ---------------------------------------------------------------------------
# S3 layout
# ---------------------------------------------------------------------------


def _version_tag(version_id: str) -> str:
    text = str(version_id or "").strip()
    tail = text.rsplit(":", 1)[-1]
    if tail and all(ch in "0123456789abcdef" for ch in tail.lower()) and len(tail) >= 16:
        return tail[:16]
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in text)
    return safe[:48] or "version"


def shadow_window_prefix(
    *,
    active_version_id: str,
    live_manifest_uri: str,
    settings: ShadowMonitorSettings,
) -> str:
    """s3://... prefix for one window's state/day/window docs.

    Prefers the explicit env prefix; otherwise derives from the merged
    champion's manifest URI (same anchoring as candidate scoring progress).
    """

    tag = _version_tag(active_version_id)
    if settings.report_uri_prefix:
        base = settings.report_uri_prefix.rstrip("/")
        _parse_s3_uri(base + "/x")  # validate shape early
        return f"{base}/shadow-windows/{tag}"
    uri = str(live_manifest_uri or "")
    if not uri.startswith("s3://"):
        raise ShadowWindowSetupError(
            "cannot derive shadow report location: champion manifest URI is not s3:// and "
            f"{SHADOW_REPORT_URI_PREFIX_ENV} is unset"
        )
    bucket, key = _parse_s3_uri(uri)
    base_prefix = key.rsplit("/", 1)[0] if "/" in key else "research-lab/sourcing-model"
    return f"s3://{bucket}/{base_prefix}/shadow-windows/{tag}"


def window_state_uri(prefix: str) -> str:
    return f"{prefix}/window-state.json"


def day_report_uri(prefix: str, benchmark_date: str) -> str:
    return f"{prefix}/day-{benchmark_date}.json"


def window_report_uri(prefix: str) -> str:
    return f"{prefix}/window-report.json"


# ---------------------------------------------------------------------------
# Discovery: candidate-derived merges without a completed shadow window
# ---------------------------------------------------------------------------


def _parse_iso(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


async def _version_row(deps: ShadowMonitorDeps, version_id: str) -> dict[str, Any] | None:
    rows = await deps.select_many(
        "research_lab_private_model_version_current",
        columns=(
            "private_model_version_id,model_artifact_hash,private_model_manifest_hash,"
            "private_model_manifest_uri,current_version_status,current_status_at"
        ),
        filters=(("private_model_version_id", version_id),),
        order_by=(("current_status_at", True),),
        limit=1,
    )
    return rows[0] if rows else None


async def discover_unshadowed_merges(
    deps: ShadowMonitorDeps,
    settings: ShadowMonitorSettings,
) -> list[dict[str, Any]]:
    """Candidate-derived ``active_version_created`` events lacking a completed
    shadow window (state derived from the S3 window-state doc, oldest first)."""

    events = await deps.select_many(
        "research_lab_candidate_promotion_events",
        columns=(
            "promotion_event_id,candidate_id,event_type,promotion_status,"
            "private_model_version_id,active_parent_artifact_hash,created_at,event_doc"
        ),
        filters=(("event_type", "active_version_created"),),
        order_by=(("created_at", True),),
        limit=settings.discovery_limit,
    )
    pending: list[dict[str, Any]] = []
    seen_versions: set[str] = set()
    for event in sorted(events, key=lambda row: str(row.get("created_at") or "")):
        version_id = str(event.get("private_model_version_id") or "")
        candidate_id = str(event.get("candidate_id") or "")
        if not version_id or not candidate_id or version_id in seen_versions:
            continue
        seen_versions.add(version_id)
        version_row = await _version_row(deps, version_id)
        if version_row is None:
            logger.warning(
                "research_lab_shadow_monitor_version_row_missing: version=%s event=%s",
                compact_ref(version_id),
                compact_ref(event.get("promotion_event_id")),
            )
            continue
        try:
            prefix = shadow_window_prefix(
                active_version_id=version_id,
                live_manifest_uri=str(version_row.get("private_model_manifest_uri") or ""),
                settings=settings,
            )
        except ShadowWindowSetupError as exc:
            logger.warning(
                "research_lab_shadow_monitor_window_prefix_unavailable: version=%s error=%s",
                compact_ref(version_id),
                str(exc)[:200],
            )
            continue
        state = deps.report_store.get_json(window_state_uri(prefix))
        if isinstance(state, Mapping) and str(state.get("status") or "") in {"completed", "aborted"}:
            continue
        pending.append(
            {
                "event": dict(event),
                "version_row": dict(version_row),
                "prefix": prefix,
                "state": dict(state) if isinstance(state, Mapping) else None,
            }
        )
    return pending


async def resolve_shadow_artifact(
    deps: ShadowMonitorDeps,
    *,
    previous_artifact_hash: str,
) -> PrivateModelArtifactManifest:
    """Load the superseded champion's manifest from its lineage row.

    Verifies the loaded manifest still hashes to the lineage-recorded artifact
    (the mutable-manifest hazard promotion guards against) before any docker
    execution.
    """

    rows = await deps.select_many(
        "research_lab_private_model_version_current",
        columns=(
            "private_model_version_id,model_artifact_hash,private_model_manifest_hash,"
            "private_model_manifest_uri,current_version_status,current_status_at"
        ),
        filters=(("model_artifact_hash", previous_artifact_hash),),
        order_by=(("current_status_at", True),),
        limit=5,
    )
    if not rows:
        raise ShadowWindowSetupError(
            f"previous champion version not found for artifact {compact_ref(previous_artifact_hash)}"
        )
    row = rows[0]
    manifest_uri = str(row.get("private_model_manifest_uri") or "")
    manifest_doc = deps.load_manifest(manifest_uri)
    artifact = PrivateModelArtifactManifest.from_mapping(manifest_doc)
    errors = validate_private_model_artifact_manifest(artifact)
    if errors:
        raise ShadowWindowSetupError(
            f"previous champion manifest failed validation: {'; '.join(errors)}"
        )
    if artifact.model_artifact_hash != str(previous_artifact_hash):
        raise ShadowWindowSetupError(
            "previous champion manifest no longer matches its lineage row "
            f"(row={compact_ref(previous_artifact_hash)} "
            f"loaded={compact_ref(artifact.model_artifact_hash)}); refusing to shadow a mutated artifact"
        )
    return artifact


def new_window_state(
    *,
    event: Mapping[str, Any],
    version_row: Mapping[str, Any],
    settings: ShadowMonitorSettings,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    merged_at = _parse_iso(event.get("created_at")) or datetime.now(timezone.utc)
    merge_date = merged_at.date()
    deadline = merge_date + timedelta(days=settings.window_days + settings.grace_days)
    return stamped_read_only_doc(
        {
            "schema_version": "1.0",
            "artifact_type": "research_lab_post_merge_shadow_window_state",
            "monitor_version": SHADOW_MONITOR_VERSION,
            "active_version_id": str(event.get("private_model_version_id") or ""),
            "merge_promotion_event_id": str(event.get("promotion_event_id") or ""),
            "merge_candidate_id": str(event.get("candidate_id") or ""),
            "merged_at": merged_at.isoformat(),
            "merge_date": merge_date.isoformat(),
            "deadline_date": deadline.isoformat(),
            "window_days": settings.window_days,
            "live_model_artifact_hash": str(version_row.get("model_artifact_hash") or ""),
            "shadow_model_artifact_hash": str(event.get("active_parent_artifact_hash") or ""),
            "status": "open",
            "abort_reason": None,
            "days": [],
            "alerts": [],
        },
        env,
    )


# ---------------------------------------------------------------------------
# Daily shadow evaluation
# ---------------------------------------------------------------------------


def _benchmark_row_is_valid(row: Mapping[str, Any]) -> bool:
    """Local mirror of the day's-reference validity used by the promotion gate."""

    status = str(row.get("current_benchmark_status") or row.get("benchmark_status") or "")
    if status and status != "completed":
        return False
    doc = row.get("score_summary_doc") if isinstance(row.get("score_summary_doc"), Mapping) else {}
    summaries = doc.get("per_icp_summaries") if isinstance(doc, Mapping) else None
    if not isinstance(summaries, list) or not summaries:
        return False
    if str(row.get("benchmark_quality") or "") == "passed":
        return True
    try:
        return int(row.get("evaluation_epoch") or 0) > 0
    except (TypeError, ValueError):
        return False


def _per_icp_live_scores(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    doc = row.get("score_summary_doc") if isinstance(row.get("score_summary_doc"), Mapping) else {}
    summaries = doc.get("per_icp_summaries") if isinstance(doc, Mapping) else None
    rows: list[dict[str, Any]] = []
    if not isinstance(summaries, list):
        return rows
    for item in summaries:
        if not isinstance(item, Mapping):
            continue
        ref = str(item.get("icp_ref") or "").strip()
        icp_hash = str(item.get("icp_hash") or "").strip()
        try:
            score = float(item.get("score"))
        except (TypeError, ValueError):
            continue
        if ref:
            rows.append({"icp_ref": ref, "icp_hash": icp_hash, "score": score})
    return rows


def _parse_icp_ref(icp_ref: str) -> tuple[int, str] | None:
    parts = str(icp_ref or "").split(":")
    if len(parts) != 3 or parts[0] != "qualification_private_icp_sets":
        return None
    try:
        return int(parts[1]), parts[2]
    except ValueError:
        return None


async def _resolve_benchmark_icps(
    deps: ShadowMonitorDeps,
    live_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Reconstruct the day's exact ICP payloads from their stored refs.

    Every resolved ICP is re-hashed and checked against the benchmark's
    recorded ``icp_hash`` so the shadow provably runs the identical inputs;
    unresolvable or drifted ICPs are excluded (and mirrored out of the live
    side by the caller) instead of silently comparing different windows.
    """

    wanted: dict[str, Mapping[str, Any]] = {str(row["icp_ref"]): row for row in live_rows}
    set_ids = sorted(
        {parsed[0] for parsed in (_parse_icp_ref(ref) for ref in wanted) if parsed is not None}
    )
    set_rows = (
        await deps.select_many(
            "qualification_private_icp_sets",
            columns="set_id,icps,icp_set_hash",
            filters=(("set_id", "in", set_ids),),
            limit=max(len(set_ids), 1),
        )
        if set_ids
        else []
    )
    icps_by_set: dict[int, list[Mapping[str, Any]]] = {}
    for row in set_rows:
        try:
            set_id = int(row.get("set_id"))
        except (TypeError, ValueError):
            continue
        icps = row.get("icps") or []
        icps_by_set[set_id] = [icp for icp in icps if isinstance(icp, Mapping)]

    items: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for ref, live_row in wanted.items():
        parsed = _parse_icp_ref(ref)
        if parsed is None:
            excluded.append({"icp_ref": ref, "reason": "icp_ref_unparseable"})
            continue
        set_id, icp_id = parsed
        icp = next(
            (
                candidate
                for candidate in icps_by_set.get(set_id, [])
                if str(candidate.get("icp_id") or "") == icp_id
            ),
            None,
        )
        if icp is None:
            excluded.append({"icp_ref": ref, "reason": "icp_not_found_in_set"})
            continue
        recorded_hash = str(live_row.get("icp_hash") or "")
        recomputed = sha256_json({"icp": dict(icp)})
        if recorded_hash and recomputed != recorded_hash:
            excluded.append({"icp_ref": ref, "reason": "icp_hash_mismatch"})
            continue
        items.append({"icp": dict(icp), "icp_ref": ref, "icp_hash": recorded_hash or recomputed})
    return items, excluded


def _average(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _short_error(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {str(exc)[:280]}"


async def _score_shadow_icp(
    *,
    runner: Callable[..., Any],
    scorer: Any,
    item: Mapping[str, Any],
) -> dict[str, Any]:
    label = str(item.get("icp_ref") or "unknown_icp")
    try:
        outputs = ensure_private_model_outputs(
            await asyncio.to_thread(runner, item["icp"], dict(SHADOW_RUN_CONTEXT)),
            context_label=f"post-merge shadow for {label}",
            require_non_empty=False,
        )
    except PrivateModelRuntimeError as exc:
        return {"icp_ref": label, "score": None, "runtime_error": _short_error(exc)}
    breakdowns = await scorer.score_with_breakdowns(outputs, item["icp"], True) if outputs else []
    scores = [float(row.get("final_score", 0.0) or 0.0) for row in breakdowns]
    return {
        "icp_ref": label,
        "score": _average(scores),
        "company_count": len(scores),
        "runtime_error": "",
    }


def _noise_context(comparable_days: int) -> dict[str, Any]:
    day_diff_se = SAME_MODEL_BENCHMARK_SD_POINTS * math.sqrt(2)
    days = max(1, int(comparable_days))
    return {
        "same_model_benchmark_sd_points": SAME_MODEL_BENCHMARK_SD_POINTS,
        "day_diff_se_points": round(day_diff_se, 3),
        "cumulative_se_points": round(day_diff_se * math.sqrt(days), 3),
        "source": "audit section 5.1: same model scored sd~=1.9 across 11 same-day evaluations",
    }


async def run_shadow_day(
    deps: ShadowMonitorDeps,
    settings: ShadowMonitorSettings,
    *,
    state: Mapping[str, Any],
    benchmark_row: Mapping[str, Any],
    shadow_artifact: PrivateModelArtifactManifest,
    env: Mapping[str, str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Evaluate the previous champion on one recorded champion-benchmark day.

    Returns ``(day_entry, day_report)``. The diff is computed over the ICPs
    scored on BOTH sides (shadow runtime errors and unresolvable ICPs are
    excluded symmetrically, mirroring the promotion gate's §5.2-1 discipline).
    """

    # Re-check hygiene before spending on docker/provider execution.
    ensure_shadow_process_read_only(env)
    benchmark_date = str(benchmark_row.get("benchmark_date") or "")
    live_rows = _per_icp_live_scores(benchmark_row)
    items, excluded = await _resolve_benchmark_icps(deps, live_rows)

    runner = deps.runner_factory(shadow_artifact)
    scorer = deps.scorer_factory()
    semaphore = asyncio.Semaphore(settings.concurrency)

    async def _bounded(item: Mapping[str, Any]) -> dict[str, Any]:
        async with semaphore:
            return await _score_shadow_icp(runner=runner, scorer=scorer, item=item)

    results = await asyncio.gather(*(_bounded(item) for item in items))

    shadow_scores: dict[str, float] = {}
    for result in results:
        if result.get("runtime_error"):
            excluded.append(
                {
                    "icp_ref": result["icp_ref"],
                    "reason": "shadow_runtime_error",
                    "error": result["runtime_error"],
                }
            )
            continue
        shadow_scores[str(result["icp_ref"])] = float(result["score"])

    live_scores = {str(row["icp_ref"]): float(row["score"]) for row in live_rows}
    shared_refs = sorted(ref for ref in shadow_scores if ref in live_scores)
    live_aggregate = _average([live_scores[ref] for ref in shared_refs])
    shadow_aggregate = _average([shadow_scores[ref] for ref in shared_refs])
    diff_points = round(live_aggregate - shadow_aggregate, 6) if shared_refs else None

    # Reuse the dormant per-key diff API (its delta convention is shadow-live);
    # ICP refs are mapped to stable integer indexes because it sorts int keys.
    index_by_ref = {ref: str(index) for index, ref in enumerate(shared_refs)}
    per_icp_diff = compute_shadow_live_diff(
        {index_by_ref[ref]: int(round(shadow_scores[ref] * _MILLIPOINTS)) for ref in shared_refs},
        {index_by_ref[ref]: int(round(live_scores[ref] * _MILLIPOINTS)) for ref in shared_refs},
    )

    evaluated_at = deps.now().isoformat()
    day_report = stamped_read_only_doc(
        {
            "schema_version": "1.0",
            "artifact_type": "research_lab_post_merge_shadow_day_report",
            "monitor_version": SHADOW_MONITOR_VERSION,
            "active_version_id": str(state.get("active_version_id") or ""),
            "benchmark_date": benchmark_date,
            "benchmark_bundle_id": str(benchmark_row.get("benchmark_bundle_id") or ""),
            "live_model_artifact_hash": str(state.get("live_model_artifact_hash") or ""),
            "shadow_model_artifact_hash": str(state.get("shadow_model_artifact_hash") or ""),
            "live_aggregate_score": round(live_aggregate, 6),
            "shadow_aggregate_score": round(shadow_aggregate, 6),
            "shadow_live_diff_points": diff_points,
            "compared_icp_count": len(shared_refs),
            "benchmark_icp_count": len(live_rows),
            "excluded_icps": excluded,
            "per_icp_scores": {
                ref: {"live": round(live_scores[ref], 6), "shadow": round(shadow_scores[ref], 6)}
                for ref in shared_refs
            },
            "per_icp_diff_millipoints": per_icp_diff,
            "per_icp_diff_note": "delta values are shadow-minus-live milli-points (dormant API convention); shadow_live_diff_points is live-minus-shadow",
            "icp_index_legend": {index: ref for ref, index in index_by_ref.items()},
            "noise_context": _noise_context(1),
            "evaluated_at": evaluated_at,
        },
        env,
    )
    day_report = {
        **day_report,
        "report_id": "research_lab_post_merge_shadow_day:" + sha256_json(day_report),
    }
    day_entry = {
        "benchmark_date": benchmark_date,
        "benchmark_bundle_id": str(benchmark_row.get("benchmark_bundle_id") or ""),
        "live_aggregate_score": round(live_aggregate, 6),
        "shadow_aggregate_score": round(shadow_aggregate, 6),
        "shadow_live_diff_points": diff_points,
        "compared_icp_count": len(shared_refs),
        "excluded_icp_count": len(excluded),
        "day_report_id": day_report["report_id"],
        "evaluated_at": evaluated_at,
    }
    return day_entry, day_report


# ---------------------------------------------------------------------------
# Alerts + window report
# ---------------------------------------------------------------------------


def _comparable_days(state: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    days = state.get("days") if isinstance(state.get("days"), list) else []
    ordered = sorted(
        (day for day in days if isinstance(day, Mapping)),
        key=lambda day: str(day.get("benchmark_date") or ""),
    )
    return [day for day in ordered if day.get("shadow_live_diff_points") is not None]


def evaluate_window_alerts(
    state: Mapping[str, Any],
    settings: ShadowMonitorSettings,
) -> list[dict[str, Any]]:
    """Alert conditions currently true for the window (full recompute).

    * ``cumulative_regression`` — cumulative diff over the observed window is
      more negative than ``-alert_threshold_points`` (default 2.0 ~= 1 se —
      §5.1's sd~=1.9 makes any single-day dip smaller than that pure noise).
    * ``consecutive_negative_days`` — 2 consecutive observed comparable days
      each more negative than ``-early_alert_day_points`` (default 1.0).
    """

    comparable = _comparable_days(state)
    diffs = [float(day["shadow_live_diff_points"]) for day in comparable]
    if not diffs:
        return []
    cumulative = round(sum(diffs), 6)
    mean = round(cumulative / len(diffs), 6)
    noise = _noise_context(len(diffs))
    base = {
        "active_version_id": str(state.get("active_version_id") or ""),
        "triggered_on_date": str(comparable[-1].get("benchmark_date") or ""),
        "comparable_day_count": len(diffs),
        "day_diffs": diffs,
        "cumulative_shadow_live_diff_points": cumulative,
        "mean_shadow_live_diff_points": mean,
        "noise_context": noise,
    }
    alerts: list[dict[str, Any]] = []
    if cumulative < -settings.alert_threshold_points:
        alerts.append(
            {
                **base,
                "alert_type": "cumulative_regression",
                "threshold_points": settings.alert_threshold_points,
                "cumulative_diff_over_se": round(
                    abs(cumulative) / float(noise["cumulative_se_points"]), 3
                ),
            }
        )
    run: list[Mapping[str, Any]] = []
    for day in comparable:
        if float(day["shadow_live_diff_points"]) < -settings.early_alert_day_points:
            run.append(day)
            if len(run) >= settings.early_alert_consecutive_days:
                alerts.append(
                    {
                        **base,
                        "alert_type": "consecutive_negative_days",
                        "day_points_threshold": settings.early_alert_day_points,
                        "consecutive_dates": [str(item.get("benchmark_date") or "") for item in run],
                    }
                )
                break
        else:
            run = []
    return alerts


def _log_new_alerts(state: dict[str, Any], settings: ShadowMonitorSettings) -> list[dict[str, Any]]:
    """Append newly-true alert conditions to state and log each exactly once."""

    existing_types = {
        str(alert.get("alert_type") or "")
        for alert in (state.get("alerts") or [])
        if isinstance(alert, Mapping)
    }
    new_alerts = [
        alert for alert in evaluate_window_alerts(state, settings)
        if alert["alert_type"] not in existing_types
    ]
    for alert in new_alerts:
        logger.warning(
            format_worker_line(
                SHADOW_REGRESSION_ALERT_LINE,
                alert_type=alert["alert_type"],
                active_version_id=compact_ref(alert["active_version_id"], keep=28),
                cumulative_points=f"{alert['cumulative_shadow_live_diff_points']:+.4f}",
                mean_points=f"{alert['mean_shadow_live_diff_points']:+.4f}",
                threshold_points=alert.get("threshold_points"),
                day_points_threshold=alert.get("day_points_threshold"),
                comparable_days=alert["comparable_day_count"],
                day_diffs="|".join(f"{value:+.3f}" for value in alert["day_diffs"]),
                day_diff_se_points=alert["noise_context"]["day_diff_se_points"],
                cumulative_se_points=alert["noise_context"]["cumulative_se_points"],
                triggered_on=alert["triggered_on_date"],
            )
        )
        state.setdefault("alerts", []).append(alert)
    return new_alerts


def build_window_report(
    state: Mapping[str, Any],
    settings: ShadowMonitorSettings,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    comparable = _comparable_days(state)
    diffs = [float(day["shadow_live_diff_points"]) for day in comparable]
    days = [dict(day) for day in state.get("days") or [] if isinstance(day, Mapping)]
    blockers: list[str] = []
    if len(days) < int(state.get("window_days") or settings.window_days):
        blockers.append("observed_fewer_benchmark_days_than_window")
    if len(comparable) < len(days):
        blockers.append("days_without_comparable_icps")
    report = stamped_read_only_doc(
        {
            "schema_version": "1.0",
            "artifact_type": "research_lab_post_merge_shadow_window_report",
            "monitor_version": SHADOW_MONITOR_VERSION,
            "active_version_id": str(state.get("active_version_id") or ""),
            "merge_promotion_event_id": str(state.get("merge_promotion_event_id") or ""),
            "merge_candidate_id": str(state.get("merge_candidate_id") or ""),
            "merge_date": str(state.get("merge_date") or ""),
            "deadline_date": str(state.get("deadline_date") or ""),
            "window_days": int(state.get("window_days") or settings.window_days),
            "live_model_artifact_hash": str(state.get("live_model_artifact_hash") or ""),
            "shadow_model_artifact_hash": str(state.get("shadow_model_artifact_hash") or ""),
            "observed_day_count": len(days),
            "comparable_day_count": len(comparable),
            "days": days,
            "cumulative_shadow_live_diff_points": round(sum(diffs), 6) if diffs else None,
            "mean_shadow_live_diff_points": round(_average(diffs), 6) if diffs else None,
            "min_day_diff_points": min(diffs) if diffs else None,
            "max_day_diff_points": max(diffs) if diffs else None,
            "alert_threshold_points": settings.alert_threshold_points,
            "early_alert_day_points": settings.early_alert_day_points,
            "alerts": [dict(alert) for alert in state.get("alerts") or [] if isinstance(alert, Mapping)],
            "report_blockers": blockers,
            "noise_context": _noise_context(len(comparable)),
        },
        env,
    )
    return {
        **report,
        "report_id": "research_lab_post_merge_shadow_window:" + sha256_json(report),
    }


# ---------------------------------------------------------------------------
# Window orchestration
# ---------------------------------------------------------------------------


async def _pending_benchmark_rows(
    deps: ShadowMonitorDeps,
    *,
    state: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Valid champion-benchmark rows for un-shadowed dates in the window.

    One row per benchmark date (newest recording wins, matching the promotion
    gate's reference selection), earliest dates first, capped at the window's
    remaining day budget.
    """

    live_hash = str(state.get("live_model_artifact_hash") or "")
    merge_date = str(state.get("merge_date") or "")
    deadline_date = str(state.get("deadline_date") or "")
    done_dates = {
        str(day.get("benchmark_date") or "")
        for day in state.get("days") or []
        if isinstance(day, Mapping)
    }
    rows = await deps.select_many(
        "research_lab_private_model_benchmark_current",
        columns=(
            "benchmark_bundle_id,benchmark_date,private_model_artifact_hash,"
            "private_model_manifest_hash,rolling_window_hash,benchmark_quality,"
            "evaluation_epoch,current_benchmark_status,aggregate_score,score_summary_doc,created_at"
        ),
        filters=(
            ("private_model_artifact_hash", live_hash),
            ("current_benchmark_status", "completed"),
            ("benchmark_date", "gte", merge_date),
        ),
        order_by=(("benchmark_date", False), ("created_at", True)),
        limit=100,
    )
    by_date: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not _benchmark_row_is_valid(row):
            continue
        row_date = str(row.get("benchmark_date") or "")
        if not row_date or row_date in done_dates or (deadline_date and row_date > deadline_date):
            continue
        if row_date not in by_date:
            by_date[row_date] = dict(row)
    remaining = max(0, int(state.get("window_days") or 0) - len(done_dates))
    return [by_date[key] for key in sorted(by_date)][:remaining]


def _finalize_window(
    deps: ShadowMonitorDeps,
    settings: ShadowMonitorSettings,
    *,
    state: dict[str, Any],
    prefix: str,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    report = build_window_report(state, settings, env)
    deps.report_store.put_json(window_report_uri(prefix), report)
    state["status"] = "completed"
    state["window_report_id"] = report["report_id"]
    state["updated_at"] = deps.now().isoformat()
    deps.report_store.put_json(window_state_uri(prefix), state)
    logger.info(
        format_worker_line(
            "research_lab_shadow_window_completed",
            active_version_id=compact_ref(state.get("active_version_id"), keep=28),
            observed_days=report["observed_day_count"],
            comparable_days=report["comparable_day_count"],
            cumulative_points=report["cumulative_shadow_live_diff_points"],
            alerts=len(report["alerts"]),
        )
    )
    return report


async def process_window(
    deps: ShadowMonitorDeps,
    settings: ShadowMonitorSettings,
    *,
    event: Mapping[str, Any],
    version_row: Mapping[str, Any],
    prefix: str,
    state: dict[str, Any] | None = None,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Advance one shadow window: evaluate any newly-recorded benchmark days,
    update S3 state, log alerts, and finalize when the window is done."""

    if state is None:
        state = new_window_state(event=event, version_row=version_row, settings=settings, env=env)
        state["updated_at"] = deps.now().isoformat()
        deps.report_store.put_json(window_state_uri(prefix), state)
    if str(state.get("status") or "") != "open":
        return {"status": state.get("status"), "active_version_id": state.get("active_version_id")}

    if not str(state.get("shadow_model_artifact_hash") or ""):
        state["status"] = "aborted"
        state["abort_reason"] = "merge_event_missing_active_parent_artifact_hash"
        state["updated_at"] = deps.now().isoformat()
        deps.report_store.put_json(window_state_uri(prefix), state)
        return {"status": "aborted", "reason": state["abort_reason"]}

    pending = await _pending_benchmark_rows(deps, state=state)
    evaluated = 0
    if pending:
        try:
            shadow_artifact = await resolve_shadow_artifact(
                deps,
                previous_artifact_hash=str(state["shadow_model_artifact_hash"]),
            )
        except ShadowWindowSetupError as exc:
            state["status"] = "aborted"
            state["abort_reason"] = str(exc)[:400]
            state["updated_at"] = deps.now().isoformat()
            deps.report_store.put_json(window_state_uri(prefix), state)
            logger.error(
                "research_lab_shadow_window_aborted: version=%s reason=%s",
                compact_ref(state.get("active_version_id")),
                str(exc)[:300],
            )
            return {"status": "aborted", "reason": state["abort_reason"]}
        for benchmark_row in pending:
            day_entry, day_report = await run_shadow_day(
                deps,
                settings,
                state=state,
                benchmark_row=benchmark_row,
                shadow_artifact=shadow_artifact,
                env=env,
            )
            deps.report_store.put_json(
                day_report_uri(prefix, day_entry["benchmark_date"]), day_report
            )
            state.setdefault("days", []).append(day_entry)
            _log_new_alerts(state, settings)
            state["updated_at"] = deps.now().isoformat()
            deps.report_store.put_json(window_state_uri(prefix), state)
            evaluated += 1
            logger.info(
                format_worker_line(
                    "research_lab_shadow_day_completed",
                    active_version_id=compact_ref(state.get("active_version_id"), keep=28),
                    benchmark_date=day_entry["benchmark_date"],
                    diff_points=day_entry["shadow_live_diff_points"],
                    compared_icps=day_entry["compared_icp_count"],
                )
            )

    observed = len(state.get("days") or [])
    today = deps.now().date().isoformat()
    deadline_passed = bool(state.get("deadline_date")) and today > str(state["deadline_date"])
    if observed >= int(state.get("window_days") or settings.window_days) or deadline_passed:
        report = _finalize_window(deps, settings, state=state, prefix=prefix, env=env)
        return {
            "status": "completed",
            "active_version_id": state.get("active_version_id"),
            "evaluated_days": evaluated,
            "observed_day_count": observed,
            "window_report_id": report["report_id"],
        }
    return {
        "status": "open",
        "active_version_id": state.get("active_version_id"),
        "evaluated_days": evaluated,
        "observed_day_count": observed,
    }


async def watch_once(
    deps: ShadowMonitorDeps,
    settings: ShadowMonitorSettings,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """One poll tick: discover unshadowed merges and advance each window."""

    ensure_shadow_process_read_only(env)
    pending = await discover_unshadowed_merges(deps, settings)
    results = []
    for entry in pending:
        results.append(
            await process_window(
                deps,
                settings,
                event=entry["event"],
                version_row=entry["version_row"],
                prefix=entry["prefix"],
                state=entry["state"],
                env=env,
            )
        )
    return {"discovered_windows": len(pending), "windows": results}


async def watch(
    deps: ShadowMonitorDeps,
    settings: ShadowMonitorSettings,
    env: Mapping[str, str] | None = None,
) -> None:
    if not settings.enabled:
        raise RuntimeError(f"shadow monitor disabled; set {SHADOW_MONITOR_ENABLED_ENV}=true")
    while True:
        try:
            summary = await watch_once(deps, settings, env)
            logger.info(
                format_worker_line(
                    "research_lab_shadow_monitor_tick",
                    discovered_windows=summary["discovered_windows"],
                )
            )
        except ShadowMonitorEnvHygieneError:
            raise
        except Exception as exc:
            logger.exception("research_lab_shadow_monitor_tick_failed: %s", str(exc)[:300])
        await asyncio.sleep(settings.poll_seconds)


async def run_window(
    deps: ShadowMonitorDeps,
    settings: ShadowMonitorSettings,
    *,
    active_version_id: str,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Run one observation-window pass for a specific merged version."""

    ensure_shadow_process_read_only(env)
    events = await deps.select_many(
        "research_lab_candidate_promotion_events",
        columns=(
            "promotion_event_id,candidate_id,event_type,promotion_status,"
            "private_model_version_id,active_parent_artifact_hash,created_at,event_doc"
        ),
        filters=(
            ("event_type", "active_version_created"),
            ("private_model_version_id", active_version_id),
        ),
        order_by=(("created_at", True),),
        limit=1,
    )
    if not events:
        return {
            "status": "error",
            "error": "active_version_created_event_not_found",
            "active_version_id": active_version_id,
        }
    event = events[0]
    version_row = await _version_row(deps, active_version_id)
    if version_row is None:
        return {
            "status": "error",
            "error": "private_model_version_not_found",
            "active_version_id": active_version_id,
        }
    prefix = shadow_window_prefix(
        active_version_id=active_version_id,
        live_manifest_uri=str(version_row.get("private_model_manifest_uri") or ""),
        settings=settings,
    )
    stored = deps.report_store.get_json(window_state_uri(prefix))
    state = dict(stored) if isinstance(stored, Mapping) else None
    return await process_window(
        deps,
        settings,
        event=event,
        version_row=version_row,
        prefix=prefix,
        state=state,
        env=env,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python3 -m gateway.research_lab.shadow_monitor",
        description="Read-only post-merge shadow monitor (separate process; see module docstring).",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--watch", action="store_true", help="poll loop over unshadowed merges")
    mode.add_argument("--once", action="store_true", help="single discovery/evaluation tick")
    mode.add_argument("--window", metavar="ACTIVE_VERSION_ID", help="run one observation window")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = _build_parser().parse_args(argv)
    settings = ShadowMonitorSettings.from_env()
    if not settings.enabled:
        logger.error(
            "research_lab_shadow_monitor_disabled: set %s=true to run (default off)",
            SHADOW_MONITOR_ENABLED_ENV,
        )
        return 2
    try:
        ensure_shadow_process_read_only()
    except ShadowMonitorEnvHygieneError as exc:
        logger.error("research_lab_shadow_monitor_env_hygiene_failed: %s", exc)
        return 3
    deps = default_shadow_monitor_deps(settings)
    try:
        if args.watch:
            asyncio.run(watch(deps, settings))
            return 0
        if args.once:
            summary = asyncio.run(watch_once(deps, settings))
            print(json.dumps(summary, sort_keys=True, default=str))
            return 0
        result = asyncio.run(run_window(deps, settings, active_version_id=str(args.window)))
        print(json.dumps(result, sort_keys=True, default=str))
        return 0 if result.get("status") != "error" else 1
    except ShadowMonitorEnvHygieneError as exc:
        logger.error("research_lab_shadow_monitor_env_hygiene_failed: %s", exc)
        return 3
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
