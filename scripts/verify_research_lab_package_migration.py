#!/usr/bin/env python3
"""Verify Research Lab Phase 4.1 package-migration audit contracts."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_lab.package_migration_audit import verify_package_migration_audit


def main() -> int:
    summary = verify_package_migration_audit()
    print(
        "Research Lab Phase 4.1 package migration verified: "
        f"audit={summary['current_audit_id']}, "
        f"native_package_present={summary['leadpoet_native_package_present']}, "
        f"research_lab_runtime_legacy_imports={summary['research_lab_runtime_legacy_imports']}, "
        f"operational_script_legacy_imports={summary['operational_script_legacy_imports']}, "
        f"package_migration_ready={summary['package_migration_ready']}, "
        f"blockers={summary['blockers']}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
