#!/usr/bin/env python3
"""Verify gateway Research Lab worker autostart planning."""

from __future__ import annotations

from pathlib import Path
import sys
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gateway.research_lab.worker_autostart import build_research_lab_worker_autostart_plan


def _base_env() -> dict[str, str]:
    return {
        "RESEARCH_LAB_HOSTED_RUNS_ENABLED": "true",
        "RESEARCH_LAB_EVALUATION_BUNDLES_ENABLED": "true",
    }


def main() -> int:
    env = {
        **_base_env(),
        "RESEARCH_LAB_AUTO_RESEARCH_WEBSHARE_PROXY_1": "http://proxy-a.example:1111",
        "RESEARCH_LAB_AUTO_RESEARCH_WEBSHARE_PROXY_2": "http://proxy-b.example:2222",
        "RESEARCH_LAB_QUALIFICATION_WEBSHARE_PROXY_1": "http://proxy-c.example:3333",
        "RESEARCH_LAB_QUALIFICATION_WEBSHARE_PROXY_2": "http://proxy-c.example:3333",
        "QUALIFICATION_WEBSHARE_PROXY_3": "http://proxy-d.example:4444",
    }
    plan = build_research_lab_worker_autostart_plan(env)
    assert plan.auto_start_enabled is True
    assert plan.hosted.enabled is True
    assert plan.hosted.worker_count == 2
    assert len(plan.hosted.proxy_refs) == 2
    assert plan.scoring.enabled is True
    assert plan.scoring.worker_count == 2
    assert len(plan.scoring.proxy_refs) == 2
    assert all(ref.startswith("sha256:") for ref in (*plan.hosted.proxy_refs, *plan.scoring.proxy_refs))

    disabled = build_research_lab_worker_autostart_plan({**env, "RESEARCH_LAB_AUTO_START_WORKERS": "false"})
    assert disabled.auto_start_enabled is False
    assert disabled.hosted.enabled is False
    assert disabled.scoring.enabled is False

    override = build_research_lab_worker_autostart_plan(
        {
            **_base_env(),
            "RESEARCH_LAB_HOSTED_WORKER_PROCESS_COUNT": "3",
            "RESEARCH_LAB_SCORING_WORKER_PROCESS_COUNT": "4",
        }
    )
    assert override.hosted.enabled is True
    assert override.hosted.worker_count == 3
    assert override.scoring.enabled is True
    assert override.scoring.worker_count == 4

    sparse = build_research_lab_worker_autostart_plan(
        {
            **_base_env(),
            "RESEARCH_LAB_QUALIFICATION_WEBSHARE_PROXY_1": "http://proxy-a.example:1111",
            "RESEARCH_LAB_QUALIFICATION_WEBSHARE_PROXY_3": "http://proxy-b.example:2222",
            "QUALIFICATION_WEBSHARE_PROXY_4": "http://proxy-a.example:1111",
        }
    )
    assert sparse.scoring.enabled is True
    assert sparse.scoring.worker_count == 2
    assert len(sparse.scoring.proxy_refs) == 2

    no_flags = build_research_lab_worker_autostart_plan(
        {"RESEARCH_LAB_AUTO_RESEARCH_WEBSHARE_PROXY_1": "http://proxy-a.example:1111"}
    )
    assert no_flags.hosted.enabled is False
    assert no_flags.hosted.reason == "hosted_runs_disabled"

    with TemporaryDirectory() as tmp:
        proxy_file = Path(tmp) / ".env.docker"
        proxy_file.write_text(
            "\n".join(
                [
                    *(f'RESEARCH_LAB_AUTO_RESEARCH_WEBSHARE_PROXY_{idx}="http://hosted-{idx}.example:8000"' for idx in range(1, 6)),
                    *(f'RESEARCH_LAB_QUALIFICATION_WEBSHARE_PROXY_{idx}="http://scoring-{idx}.example:9000"' for idx in range(1, 6)),
                    'OPENROUTER_API_KEY="not-loaded-by-proxy-loader"',
                ]
            ),
            encoding="utf-8",
        )
        file_env_plan = build_research_lab_worker_autostart_plan(
            {
                **_base_env(),
                "RESEARCH_LAB_WORKER_PROXY_ENV_FILE": str(proxy_file),
            }
        )
        assert file_env_plan.hosted.enabled is True
        assert file_env_plan.hosted.worker_count == 5
        assert len(file_env_plan.hosted.proxy_refs) == 5
        assert file_env_plan.scoring.enabled is True
        assert file_env_plan.scoring.worker_count == 5
        assert len(file_env_plan.scoring.proxy_refs) == 5

    print("research lab worker autostart verification passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
