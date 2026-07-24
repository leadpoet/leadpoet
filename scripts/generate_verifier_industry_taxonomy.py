"""Regenerate the verifier's industry-taxonomy JSON from the repo authority.

The canonical taxonomy lives in ``gateway/utils/industry_taxonomy.py`` (the
authority file).  The verifier's ``leadpoet_verifier/leadpoet_industry_taxonomy
.json`` is a derived snapshot consumed by ``leadpoet_verifier.industry_taxonomy``;
regenerate it with this script whenever the authority changes.  The ported
consistency test (tests/test_site_verifier_industry_taxonomy.py::
test_embedded_taxonomy_exactly_matches_repository_source) fails on any drift,
so a stale snapshot cannot ship silently.

Mirrors the site verifier's generator (same version-1 schema) so future
site<->lab syncs stay 1:1 diffable.
"""

from __future__ import annotations

import json
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = (
    REPOSITORY_ROOT / "leadpoet_verifier/leadpoet_industry_taxonomy.json"
)


def main() -> None:
    import sys

    sys.path.insert(0, str(REPOSITORY_ROOT))
    from gateway.utils.industry_taxonomy import INDUSTRY_TAXONOMY

    payload = {
        "version": 1,
        "source": "gateway/utils/industry_taxonomy.py",
        "parent_industries": sorted({
            parent
            for entry in INDUSTRY_TAXONOMY.values()
            for parent in entry["industries"]
        }),
        "subindustry_parents": {
            subindustry: sorted(entry["industries"])
            for subindustry, entry in sorted(INDUSTRY_TAXONOMY.items())
        },
    }
    OUTPUT_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"Wrote {len(payload['parent_industries'])} parents and "
        f"{len(payload['subindustry_parents'])} subindustries to {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
