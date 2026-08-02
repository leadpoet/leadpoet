"""Best-effort one-shot Sentry bridge for host shell workflows."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import re
from typing import Iterable, Optional
import urllib.request

from .sentry_bootstrap import flush_sentry, init_sentry
from .sentry_operations import emit_release_summary, emit_restart_summary


_RELEASE_STEP_NAMES = {
    "Remove stale root-owned V2 builder temp directories": "stale_cleanup",
    "Check out exact release source": "source_checkout",
    "Guard gateway builder host memory": "host_memory_guard",
    "Reclaim unreferenced gateway build space": "storage_reclaim",
    "Reclaim unreferenced validator build space": "storage_reclaim",
    "Prepare immutable build inputs": "source_prepare",
    "Build gateway and validator evidence": "gateway_validator_build",
    "Upload gateway-parent evidence": "evidence_upload",
    "Upload validator-parent evidence": "evidence_upload",
    "Reclaim gateway-parent storage after evidence generation": "final_cleanup",
    "Reclaim validator-parent storage after evidence generation": "final_cleanup",
    "Download independent release evidence": "evidence_download",
    "Assemble both six-build manifests": "manifest_assembly",
    "Publish with COMPLIANCE Object Lock": "immutable_publication",
}


def _sha(value: str) -> Optional[str]:
    candidate = str(value or "").strip().lower()
    return candidate if re.fullmatch(r"[0-9a-f]{40}", candidate) else None


def _existing_paths(values: Iterable[str]) -> list[Path]:
    output = []
    for value in values:
        path = Path(str(value))
        if path.is_absolute() and path.is_file():
            output.append(path)
    return output[:8]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    restart = subparsers.add_parser("restart-summary")
    restart.add_argument("--component", choices=("gateway", "validator"), required=True)
    restart.add_argument("--status", choices=("passed", "failed"), required=True)
    restart.add_argument("--stage", required=True)
    restart.add_argument("--ledger", required=True)
    restart.add_argument("--restart-invocation-id", required=True)
    restart.add_argument("--candidate-sha", default="")
    restart.add_argument("--shutdown-started", action="store_true")
    restart.add_argument("--release-attempts", type=int)
    restart.add_argument("--evidence", action="append", default=[])
    release = subparsers.add_parser("release-summary")
    release.add_argument("--component", required=True)
    release.add_argument("--physical-role", required=True)
    release.add_argument("--status", choices=("passed", "failed"), required=True)
    release.add_argument("--candidate-sha", required=True)
    release.add_argument("--stage-status", action="append", default=[])
    release.add_argument("--github-job-name", default="")
    release.add_argument("--duration-seconds", type=float)
    return parser


def _stage_statuses(values: Iterable[str]) -> list[tuple[str, str]]:
    output = []
    for value in values:
        stage, separator, status = str(value).partition("=")
        if separator and stage and status:
            output.append((stage[:120], status[:40]))
    return output[:40]


def _parse_timestamp(value: object) -> Optional[dt.datetime]:
    try:
        text = str(value or "").strip().replace("Z", "+00:00")
        parsed = dt.datetime.fromisoformat(text)
        return parsed if parsed.tzinfo is not None else None
    except (TypeError, ValueError):
        return None


def _stage_durations_from_actions_document(
    document: object, job_name: str
) -> dict[str, float]:
    """Extract only semantic step durations from a bounded Actions response."""

    jobs = document.get("jobs") if isinstance(document, dict) else None
    if not isinstance(jobs, list):
        return {}
    selected = next(
        (
            job
            for job in jobs[:100]
            if isinstance(job, dict) and str(job.get("name")) == str(job_name)
        ),
        None,
    )
    if not isinstance(selected, dict):
        return {}
    output: dict[str, float] = {}
    for step in (selected.get("steps") or [])[:100]:
        if not isinstance(step, dict):
            continue
        semantic_name = _RELEASE_STEP_NAMES.get(str(step.get("name") or ""))
        started = _parse_timestamp(step.get("started_at"))
        completed = _parse_timestamp(step.get("completed_at"))
        if semantic_name and started is not None and completed is not None:
            duration = (completed - started).total_seconds()
            if 0.0 <= duration <= 43_200.0:
                output[semantic_name] = round(duration, 3)
    return output


def _github_stage_durations(job_name: str) -> dict[str, float]:
    """Best-effort read of this run's native step timings (telemetry only)."""

    try:
        api_url = os.environ["GITHUB_API_URL"]
        repository = os.environ["GITHUB_REPOSITORY"]
        run_id = os.environ["GITHUB_RUN_ID"]
        token = os.environ["LEADPOET_GITHUB_JOB_TOKEN"]
        request = urllib.request.Request(
            f"{api_url}/repos/{repository}/actions/runs/{run_id}/jobs?per_page=100",
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {token}",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        with urllib.request.urlopen(request, timeout=3) as response:
            body = response.read(2_000_001)
        if len(body) > 2_000_000:
            return {}
        document = json.loads(body.decode("utf-8"))
    except BaseException:
        return {}
    return _stage_durations_from_actions_document(document, job_name)


def main(argv=None) -> int:
    try:
        args = build_parser().parse_args(argv)
        if args.command == "restart-summary":
            init_sentry(
                component=f"{args.component}-restart",
                tags={
                    "physical_role": args.component,
                    "role": "restart-controller",
                },
            )
            emit_restart_summary(
                component=args.component,
                status=args.status,
                stage=str(args.stage)[:120],
                ledger_path=Path(args.ledger),
                restart_invocation_id=str(args.restart_invocation_id)[:200],
                candidate_sha=_sha(args.candidate_sha),
                shutdown_started=bool(args.shutdown_started),
                release_attempts=args.release_attempts,
                evidence_paths=_existing_paths(args.evidence),
            )
        else:
            candidate_sha = _sha(args.candidate_sha)
            if candidate_sha is None:
                return 0
            sentry_active = init_sentry(
                component=args.component,
                tags={
                    "physical_role": args.physical_role,
                    "role": "release-builder",
                },
            )
            emit_release_summary(
                component=str(args.component)[:120],
                physical_role=str(args.physical_role)[:120],
                status=args.status,
                candidate_sha=candidate_sha,
                stage_statuses=_stage_statuses(args.stage_status),
                stage_durations=(
                    _github_stage_durations(args.github_job_name)
                    if sentry_active
                    else {}
                ),
                duration_seconds=args.duration_seconds,
            )
        flush_sentry(timeout=1.0)
    except BaseException as exc:
        # No values: arguments may contain local paths and the DSN is ambient.
        try:
            print(
                "leadpoet_sentry_cli_skipped error=%s" % type(exc).__name__,
                flush=True,
            )
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
