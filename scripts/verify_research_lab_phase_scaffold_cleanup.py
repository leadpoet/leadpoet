#!/usr/bin/env python3
"""Verify Phase 4.3 phase-scaffold cleanup contracts."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_lab.phase_scaffold_cleanup import verify_phase_scaffold_cleanup


def main() -> int:
    summary = verify_phase_scaffold_cleanup()
    print(
        "Research Lab Phase 4.3 phase-scaffold cleanup verified: "
        f"audit={summary['current_audit_id']}, "
        f"module_paths={summary['phase_module_path_refs']}, "
        f"verifier_scripts={summary['phase_verifier_script_refs']}, "
        f"fixtures={summary['phase_fixture_path_refs']}, "
        f"public_symbols={summary['phase_public_symbol_refs']}, "
        f"env_vars={summary['phase_env_var_refs']}, "
        f"phase_scaffold_cleanup_ready={summary['phase_scaffold_cleanup_ready']}, "
        f"blockers={summary['blockers']}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
