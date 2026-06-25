#!/usr/bin/env python3
"""Run a small live Research Lab private-model baseline check.

This is an operator diagnostic. It uses the same DockerPrivateModelRunner and
QualificationStyleCompanyScorer boundary as the gateway scoring worker, but it
does not write to Supabase, Arweave, or chain state.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_lab.eval import DockerPrivateModelRunner, DockerPrivateModelSpec, ensure_private_model_outputs  # noqa: E402
from research_lab.eval.evaluator import QualificationStyleCompanyScorer  # noqa: E402


DEFAULT_ICPS: tuple[dict[str, Any], ...] = (
    {
        "icp_id": "live-fintech-funding",
        "industry": "Financial Services",
        "sub_industry": "Fintech",
        "geography": "United States",
        "employee_count": "201-500",
        "required_attribute": "The company is privately held and sells financial software",
        "product_service": "financial software",
        "intent_signal": "The company recently raised a funding round",
        "intent_category": "FUNDING",
        "intent_max_age_days": 365,
    },
    {
        "icp_id": "live-software-product-launch",
        "industry": "Information Technology",
        "sub_industry": "Cloud Infrastructure and IT Services",
        "target_geography": "United States",
        "company_size": "200-500",
        "product_service": "managed cloud infrastructure services",
        "intent_signals": ["Launched or announced a new product"],
        "intent_max_age_days": 365,
    },
    {
        "icp_id": "live-biotech-regulatory",
        "industry": "Biotechnology",
        "sub_industry": "Therapeutics and Drug Development",
        "geography": "United States",
        "employee_count": "50-200",
        "product_service": "biotech therapeutics",
        "intent_signal": "Achieved regulatory clearance or certification",
        "intent_category": "REGULATORY_CLEARANCE",
        "intent_max_age_days": 365,
    },
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image",
        default=os.getenv("RESEARCH_LAB_PRIVATE_MODEL_IMAGE_DIGEST", ""),
        help="Immutable private model image digest. Defaults to RESEARCH_LAB_PRIVATE_MODEL_IMAGE_DIGEST.",
    )
    parser.add_argument("--max-icps", type=int, default=1, help="Number of fixture ICPs to run.")
    parser.add_argument("--timeout-seconds", type=int, default=900, help="Per-ICP model timeout.")
    args = parser.parse_args()

    missing = [
        name
        for name in ("EXA_API_KEY", "SCRAPINGDOG_API_KEY", "OPENROUTER_API_KEY")
        if not os.getenv(name)
    ]
    if missing:
        print("ERROR: missing required provider env vars: " + ", ".join(missing))
        return 2
    if not args.image or "@sha256:" not in args.image:
        print("ERROR: --image or RESEARCH_LAB_PRIVATE_MODEL_IMAGE_DIGEST must be an immutable digest")
        return 2

    return asyncio.run(_run(args.image, args.max_icps, args.timeout_seconds))


async def _run(image: str, max_icps: int, timeout_seconds: int) -> int:
    runner = DockerPrivateModelRunner(
        DockerPrivateModelSpec(
            image_digest=image,
            timeout_seconds=timeout_seconds,
            pull_before_run=False,
        )
    )
    scorer = QualificationStyleCompanyScorer()
    positive_icps = 0
    rows: list[dict[str, Any]] = []
    for index, icp in enumerate(DEFAULT_ICPS[: max(1, max_icps)], start=1):
        started = time.time()
        try:
            outputs = ensure_private_model_outputs(
                await asyncio.to_thread(runner, icp, {"mode": "private_baseline_live_verify"}),
                context_label=f"live private baseline ICP {icp['icp_id']}",
                require_non_empty=False,
            )
            score_breakdowns = await scorer.score_with_breakdowns(outputs, icp, True)
        except Exception as exc:
            rows.append(
                {
                    "index": index,
                    "icp_id": icp["icp_id"],
                    "status": "failed",
                    "error_type": exc.__class__.__name__,
                    "error": str(exc)[:500],
                    "runtime_seconds": round(time.time() - started, 3),
                }
            )
            continue
        scores = [float(row.get("final_score", 0.0) or 0.0) for row in score_breakdowns]
        score = sum(scores) / len(scores) if scores else 0.0
        if outputs and score > 0:
            positive_icps += 1
        rows.append(
            {
                "index": index,
                "icp_id": icp["icp_id"],
                "status": "completed",
                "companies": len(outputs),
                "score": round(score, 4),
                "runtime_seconds": round(time.time() - started, 3),
            }
        )
    print(json.dumps({"positive_icps": positive_icps, "results": rows}, indent=2, sort_keys=True))
    return 0 if positive_icps > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
