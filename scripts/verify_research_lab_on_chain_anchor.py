#!/usr/bin/env python3
"""Verify Research Lab Phase 2.7 on-chain anchor contracts."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_lab.on_chain_anchor import verify_research_lab_on_chain_anchor


def main() -> int:
    summary = verify_research_lab_on_chain_anchor()
    print(
        "Research Lab Phase 2.7 on-chain anchor verified: "
        f"proposal={summary['proposal_id']}, "
        f"tx_stub={summary['tx_stub_id']}, "
        f"inclusion={summary['inclusion_id']}, "
        f"payload_hashes={summary['payload_hashes']}, "
        f"merkle_root={summary['merkle_root']}, "
        f"merkle_convention={summary['merkle_convention']}, "
        f"sealing_gate={summary['sealing_gate']}, "
        f"market gates={summary['market_dependency_gates']}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
