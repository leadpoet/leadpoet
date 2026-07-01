#!/usr/bin/env python3
"""Stress-test the parallel Research Lab daily benchmark path locally.

Runs N fixture (or Supabase read-only) ICPs concurrently through the same
DockerPrivateModelRunner + QualificationStyleCompanyScorer boundary as the
gateway scoring worker, with the dedicated benchmark Exa key and a
per-container EXA_MAX_RPS budget. Mirrors the production fan-out (semaphore +
dedicated executor, settle-then-raise, retry rounds at lower concurrency with
the aggregate Exa budget re-spread).

Purpose: answer "does N-at-once trip the Exa key's QPS ceiling?" before any
deploy. No Supabase/Arweave/chain writes.

Exit codes: 0 = clean; 1 = at least one Exa 429 observed (first pass or
retries); 2 = bad invocation/environment.
"""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import functools
import itertools
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gateway.research_lab.scoring_worker import (  # noqa: E402
    _baseline_error_is_retryable,
    _runtime_error_diagnostics,
)
from research_lab.eval import (  # noqa: E402
    DockerPrivateModelRunner,
    DockerPrivateModelSpec,
    PrivateModelRuntimeError,
    ensure_private_model_outputs,
    private_model_env_passthrough,
)
from research_lab.eval.evaluator import QualificationStyleCompanyScorer  # noqa: E402


DEFAULT_ICPS: tuple[dict[str, Any], ...] = (
    {
        "icp_id": "stress-software-funding",
        "industry": "Software Development",
        "sub_industry": "AI-powered Accounts Receivable Automation",
        "geography": "United States",
        "employee_count": "11-50 employees",
        "company_stage": "Any",
        "required_attribute": (
            "The company sells AI-powered accounts receivable, finance operations, billing, "
            "or revenue automation software"
        ),
        "product_service": "AI-powered accounts receivable automation software",
        "intent_signal": (
            "The company recently raised a named seed, Series A, Series B, or growth funding "
            "round to expand its finance automation software business"
        ),
        "intent_category": "FUNDING",
        "intent_max_age_days": 365,
    },
    {
        "icp_id": "stress-cloud-product-launch",
        "industry": "Information Technology",
        "sub_industry": "Cloud Infrastructure and IT Services",
        "target_geography": "United States",
        "company_size": "200-500",
        "product_service": "managed cloud infrastructure services",
        "intent_signals": [
            "Announced a new managed cloud infrastructure, Kubernetes, security, or AI infrastructure product for enterprise customers"
        ],
        "intent_max_age_days": 365,
    },
    {
        "icp_id": "stress-biotech-regulatory",
        "industry": "Biotechnology",
        "sub_industry": "Therapeutics and Drug Development",
        "geography": "United States",
        "employee_count": "50-200",
        "product_service": "biotech therapeutics",
        "intent_signal": (
            "Announced FDA clearance, FDA approval, CE mark certification, or another named "
            "regulatory milestone for a therapeutic or diagnostic product"
        ),
        "intent_category": "REGULATORY_CLEARANCE",
        "intent_max_age_days": 365,
    },
)

_RATE_LIMIT_MARKERS = ("too many requests", "rate limit")
_INFRA_MARKERS = ("exit status 137", "killed", "docker daemon", "no space left on device")


def _is_rate_limited_429(error_text: str) -> bool:
    diagnostics = _runtime_error_diagnostics(error_text)
    if int(diagnostics.get("status") or 0) == 429:
        return True
    lowered = error_text.lower()
    return any(marker in lowered for marker in _RATE_LIMIT_MARKERS)


def _is_oom_or_infra(error_text: str) -> bool:
    lowered = error_text.lower()
    return any(marker in lowered for marker in _INFRA_MARKERS)


def _load_icps(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.icp_fixtures:
        payload = json.loads(Path(args.icp_fixtures).read_text())
        icps = payload if isinstance(payload, list) else payload.get("icps")
        if not isinstance(icps, list) or not icps:
            raise ValueError("--icp-fixtures must be a JSON list of ICP objects (or {'icps': [...]})")
        source = [dict(icp) for icp in icps]
    elif args.icp_source == "supabase":
        source = _load_supabase_icps()
    else:
        source = [dict(icp) for icp in DEFAULT_ICPS]
    # Cycle the fixture pool up to the requested count; suffix repeated ids so
    # per-ICP rows stay distinguishable in the report.
    icps: list[dict[str, Any]] = []
    for index, icp in enumerate(itertools.islice(itertools.cycle(source), args.icp_count), start=1):
        item = dict(icp)
        if index > len(source):
            item["icp_id"] = f"{item.get('icp_id') or 'icp'}-repeat{index}"
        icps.append(item)
    return icps


def _load_supabase_icps() -> list[dict[str, Any]]:
    from supabase import create_client

    from gateway.research_lab.icp_window import select_rolling_icp_window_from_sets

    url = os.getenv("SUPABASE_URL", "").strip()
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip() or os.getenv("SUPABASE_ANON_KEY", "").strip()
    if not url or not key:
        raise ValueError("--icp-source supabase needs SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY/SUPABASE_ANON_KEY")
    response = (
        create_client(url, key)
        .table("qualification_private_icp_sets")
        .select("set_id,icps,icp_set_hash,active_from,active_until,is_active")
        .order("set_id", desc=True)
        .limit(10)
        .execute()
    )
    rows = getattr(response, "data", None) or []
    window = select_rolling_icp_window_from_sets(rows, days=10, icps_per_day=2, allow_partial=True)
    icps = [dict(item["icp"]) for item in window.benchmark_items if isinstance(item.get("icp"), Mapping)]
    if not icps:
        raise ValueError("Supabase ICP window returned no ICPs")
    return icps


def _container_env(exa_api_key: str, exa_max_rps: float) -> dict[str, str]:
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
    if exa_api_key:
        env["EXA_API_KEY"] = exa_api_key
    if exa_max_rps > 0:
        env["EXA_MAX_RPS"] = str(exa_max_rps)
    return env


async def _run_one(
    *,
    runner: DockerPrivateModelRunner,
    scorer: QualificationStyleCompanyScorer,
    icp: Mapping[str, Any],
    index: int,
    executor: concurrent.futures.Executor,
    run_start: float,
) -> dict[str, Any]:
    loop = asyncio.get_running_loop()
    started = time.time()
    row: dict[str, Any] = {"index": index, "icp_id": str(icp.get("icp_id") or f"icp-{index}")}
    error_text = ""
    try:
        outputs = ensure_private_model_outputs(
            await loop.run_in_executor(
                executor,
                functools.partial(runner, icp, {"mode": "parallel_benchmark_stress"}),
            ),
            context_label=f"parallel benchmark stress ICP {row['icp_id']}",
            require_non_empty=False,
        )
    except PrivateModelRuntimeError as exc:
        outputs = []
        error_text = str(exc)
    score_breakdowns = await scorer.score_with_breakdowns(outputs, icp, True) if outputs else []
    scores = [float(item.get("final_score", 0.0) or 0.0) for item in score_breakdowns]
    row.update(
        {
            "status": "provider_error" if error_text else "completed",
            "companies": len(outputs),
            "scored_companies": len(scores),
            "score": round(sum(scores) / len(scores), 4) if scores else 0.0,
            "runtime_seconds": round(time.time() - started, 3),
            "elapsed_seconds": round(time.time() - run_start, 3),
            "rate_limited_429": _is_rate_limited_429(error_text) if error_text else False,
            "oom_or_infra": _is_oom_or_infra(error_text) if error_text else False,
            "retryable": _baseline_error_is_retryable(error_text) if error_text else False,
            "error": error_text[:500],
        }
    )
    print(json.dumps({"progress": row}, sort_keys=True), file=sys.stderr, flush=True)
    return row


async def _run(args: argparse.Namespace, icps: list[dict[str, Any]]) -> int:
    run_start = time.time()
    passthrough = private_model_env_passthrough(include_proxy=False)
    runner = DockerPrivateModelRunner(
        DockerPrivateModelSpec(
            image_digest=args.image,
            timeout_seconds=args.timeout_seconds,
            env_passthrough=passthrough,
            extra_env=_container_env(args.exa_api_key, args.exa_max_rps),
        )
    )
    retry_rps = (
        round(args.exa_max_rps * args.concurrency / max(1, args.retry_concurrency), 3)
        if args.exa_max_rps > 0
        else 0.0
    )
    retry_runner = DockerPrivateModelRunner(
        DockerPrivateModelSpec(
            image_digest=args.image,
            timeout_seconds=args.timeout_seconds,
            env_passthrough=passthrough,
            extra_env=_container_env(args.exa_api_key, retry_rps),
            pull_before_run=False,
        )
    )
    scorer = QualificationStyleCompanyScorer()
    executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=max(args.concurrency, args.retry_concurrency),
        thread_name_prefix="stress-icp",
    )
    saw_429 = False
    try:
        semaphore = asyncio.Semaphore(args.concurrency)

        async def first_pass(index: int, icp: Mapping[str, Any]) -> dict[str, Any]:
            async with semaphore:
                return await _run_one(
                    runner=runner, scorer=scorer, icp=icp, index=index,
                    executor=executor, run_start=run_start,
                )

        settled = await asyncio.gather(
            *(first_pass(index, icp) for index, icp in enumerate(icps, start=1)),
            return_exceptions=True,
        )
        fatal = [entry for entry in settled if isinstance(entry, BaseException)]
        if fatal:
            raise fatal[0]
        results = {row["index"]: row for row in settled}
        saw_429 = any(row["rate_limited_429"] for row in results.values())

        retry_rounds_run = 0
        for round_no in range(1, args.retry_rounds + 1):
            pending = sorted(index for index, row in results.items() if row.get("retryable"))
            if not pending:
                break
            retry_rounds_run = round_no
            print(
                json.dumps({"retry_round": round_no, "retrying": pending}, sort_keys=True),
                file=sys.stderr,
                flush=True,
            )
            retry_semaphore = asyncio.Semaphore(args.retry_concurrency)

            async def retry_pass(index: int) -> dict[str, Any]:
                async with retry_semaphore:
                    return await _run_one(
                        runner=retry_runner, scorer=scorer, icp=icps[index - 1], index=index,
                        executor=executor, run_start=run_start,
                    )

            retried = await asyncio.gather(*(retry_pass(index) for index in pending), return_exceptions=True)
            fatal = [entry for entry in retried if isinstance(entry, BaseException)]
            if fatal:
                raise fatal[0]
            for row in retried:
                saw_429 = saw_429 or row["rate_limited_429"]
                results[row["index"]] = row
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    rows = [results[index] for index in sorted(results)]
    provider_errors = sum(1 for row in rows if row["status"] != "completed")
    report = {
        "image": args.image,
        "concurrency": args.concurrency,
        "exa_max_rps": args.exa_max_rps,
        "retry_concurrency": args.retry_concurrency,
        "retry_rps": retry_rps,
        "retry_rounds_run": retry_rounds_run,
        "icp_count": len(rows),
        "wall_clock_seconds": round(time.time() - run_start, 3),
        "provider_errors_final": provider_errors,
        "rate_limited_429_seen": saw_429,
        "oom_or_infra_final": sum(1 for row in rows if row["oom_or_infra"]),
        "nonempty_icps": sum(1 for row in rows if row["companies"] > 0),
        "mean_score": round(sum(row["score"] for row in rows) / len(rows), 4) if rows else 0.0,
        "results": rows,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if saw_429 else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--image",
        default=os.getenv("RESEARCH_LAB_PRIVATE_MODEL_IMAGE_DIGEST", ""),
        help="Immutable private model image digest. Defaults to RESEARCH_LAB_PRIVATE_MODEL_IMAGE_DIGEST.",
    )
    parser.add_argument("--concurrency", type=int, default=10, help="ICPs in flight at once.")
    parser.add_argument("--icp-count", type=int, default=10, help="Total ICPs to run.")
    parser.add_argument(
        "--exa-api-key",
        default=os.getenv("RESEARCH_LAB_BENCHMARK_EXA_API_KEY", ""),
        help="Dedicated benchmark Exa key. Defaults to RESEARCH_LAB_BENCHMARK_EXA_API_KEY; empty inherits EXA_API_KEY.",
    )
    parser.add_argument("--exa-max-rps", type=float, default=0.8, help="Per-container EXA_MAX_RPS for the first pass.")
    parser.add_argument("--retry-concurrency", type=int, default=2, help="ICPs in flight during retry rounds.")
    parser.add_argument("--retry-rounds", type=int, default=2, help="Retry rounds over retryable failures.")
    parser.add_argument("--timeout-seconds", type=int, default=1800, help="Per-ICP model timeout.")
    parser.add_argument("--icp-fixtures", default="", help="JSON file with a list of ICP objects to run.")
    parser.add_argument(
        "--icp-source",
        choices=("fixtures", "supabase"),
        default="fixtures",
        help="Where to get ICPs: built-in fixtures or a read-only Supabase window fetch.",
    )
    args = parser.parse_args()

    if not args.image or "@sha256:" not in args.image:
        print("ERROR: --image or RESEARCH_LAB_PRIVATE_MODEL_IMAGE_DIGEST must be an immutable digest")
        return 2
    if args.concurrency < 1 or args.icp_count < 1:
        print("ERROR: --concurrency and --icp-count must be >= 1")
        return 2
    missing = [
        name
        for name in ("SCRAPINGDOG_API_KEY", "OPENROUTER_API_KEY")
        if not os.getenv(name)
    ]
    if not (args.exa_api_key or os.getenv("EXA_API_KEY")):
        missing.append("EXA_API_KEY (or --exa-api-key / RESEARCH_LAB_BENCHMARK_EXA_API_KEY)")
    if missing:
        print("ERROR: missing required provider env vars: " + ", ".join(missing))
        return 2

    try:
        icps = _load_icps(args)
    except Exception as exc:
        print(f"ERROR: could not load ICPs: {exc}")
        return 2
    return asyncio.run(_run(args, icps))


if __name__ == "__main__":
    raise SystemExit(main())
