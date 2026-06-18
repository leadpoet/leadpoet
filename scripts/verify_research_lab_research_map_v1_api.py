#!/usr/bin/env python3
"""Verify Research Lab Phase 2.5 Research Map v1 API contracts."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_lab.research_map_v1_api import verify_research_lab_research_map_v1_api


def main() -> int:
    summary = verify_research_lab_research_map_v1_api()
    print(
        "Research Lab Phase 2.5 Research Map v1 API verified: "
        f"projection={summary['projection_id']}, "
        f"quote={summary['quote_id']}, "
        f"artifact={summary['artifact_ref']}, "
        f"public_cells={summary['public_cells']}, "
        f"endpoint={summary['endpoint_path']}, "
        f"market gates={summary['market_dependency_gates']}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
