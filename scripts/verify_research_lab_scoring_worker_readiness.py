#!/usr/bin/env python3
"""Verify scoring workers do not claim candidates before provider env is ready."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gateway.research_lab.config import ResearchLabGatewayConfig  # noqa: E402
from gateway.research_lab.scoring_worker import ResearchLabGatewayScoringWorker  # noqa: E402


PRIVATE_MODEL_ENV = (
    "EXA_API_KEY",
    "SCRAPINGDOG_API_KEY",
    "QUALIFICATION_SCRAPINGDOG_API_KEY",
    "OPENROUTER_API_KEY",
    "QUALIFICATION_OPENROUTER_API_KEY",
    "OPENROUTER_KEY",
)


class ProbeScoringWorker(ResearchLabGatewayScoringWorker):
    def __init__(self, config: ResearchLabGatewayConfig):
        super().__init__(config, worker_ref="readiness-test-scorer")
        self.recover_calls = 0
        self.claim_calls = 0

    async def _recover_stale_candidate_claims(self) -> int:
        self.recover_calls += 1
        return 0

    async def _claim_next_candidate(self) -> dict[str, Any] | None:
        self.claim_calls += 1
        return None


def _config() -> ResearchLabGatewayConfig:
    return ResearchLabGatewayConfig(
        production_writes_enabled=True,
        evaluation_bundles_enabled=True,
        scoring_worker_enabled=True,
        private_baseline_rebenchmark_enabled=False,
        scoring_worker_max_candidates=1,
    )


async def _verify() -> None:
    saved_env = {name: os.environ.get(name) for name in PRIVATE_MODEL_ENV}
    try:
        for name in PRIVATE_MODEL_ENV:
            os.environ.pop(name, None)

        missing_worker = ProbeScoringWorker(_config())
        missing_result = await missing_worker.run_once()
        assert missing_result["status"] == "idle"
        assert missing_result["private_model_env_ready"] is False
        assert "EXA_API_KEY" in missing_result["missing_private_model_env"]
        assert missing_worker.recover_calls == 0
        assert missing_worker.claim_calls == 0

        os.environ["EXA_API_KEY"] = "test-exa"
        os.environ["SCRAPINGDOG_API_KEY"] = "test-scrapingdog"
        os.environ["OPENROUTER_API_KEY"] = "test-openrouter"

        ready_worker = ProbeScoringWorker(_config())
        ready_result = await ready_worker.run_once()
        assert ready_result["status"] == "idle"
        assert "private_model_env_ready" not in ready_result
        assert ready_worker.recover_calls == 1
        assert ready_worker.claim_calls == 1
    finally:
        for name, value in saved_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def main() -> int:
    asyncio.run(_verify())
    print("Research Lab scoring worker readiness verified: missing provider env leaves candidates queued.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
