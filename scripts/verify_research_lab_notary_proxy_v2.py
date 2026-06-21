#!/usr/bin/env python3
"""Verify Research Lab Notary Proxy v2 local/staged contract."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_lab.notary_proxy_v2 import verify_research_lab_notary_proxy_v2


def main() -> int:
    summary = verify_research_lab_notary_proxy_v2()
    routes = summary["provider_routes"]
    print(
        "Research Lab Notary Proxy v2 verified: "
        f"bundle {summary['public_bundle_hash']}, "
        f"trace {summary['execution_trace_hash']}, "
        f"{summary['successful_call_count']} calls, "
        f"{summary['blocked_call_count']} blocked, "
        f"{summary['evidence_bundle_count']} evidence bundles, "
        f"{summary['total_cost_cents']} cents, "
        f"deletion {summary['deletion_bundle_hash']}, "
        f"routes openrouter={routes['openrouter']} "
        f"exa={routes['exa']} scrapingdog={routes['scrapingdog']}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
