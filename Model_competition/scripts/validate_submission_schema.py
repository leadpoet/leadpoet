"""Validate qual_engine output against the production submission schema.

Runs qualify() on one ICP, serializes each CompanyMatch via to_submission_dict(),
then calls production CompanyOutput.model_validate() to confirm zero parse errors.
"""

from __future__ import annotations
import asyncio
import json
import sys
from pathlib import Path

# Make both qual_engine AND the gateway production schema importable
sys.path.insert(0, str(Path(__file__).parent.parent))           # qual_engine
sys.path.insert(0, "/Users/tasnimul/Desktop/leadpoet")          # gateway (parent repo)

from qual_engine.entry import qualify
from qual_engine.models import ICPPrompt
from qual_engine.layers.l8_output import to_submission_dict

# Production submission schema
from gateway.qualification.models import (
    CompanyOutput as ProdCompanyOutput,
)


ICP = ICPPrompt(
    icp_id="schema_validate",
    prompt="I need Series B AI companies on the US West Coast that recently raised funding.",
    industry="Software",
    sub_industry="Artificial Intelligence",
    geography="United States, West Coast",
    country="United States",
    employee_count="50-200",
    company_stage="Series B",
    product_service="AI/ML platform",
    intent_signals=["Recently raised funding"],
)


async def main():
    print("Running qualify() on one ICP for schema validation...")
    result = await qualify(ICP)
    print(f"matches={result.total_matches}  cost=${result.cost_breakdown.get('total_usd', 0):.4f}")
    if not result.matches:
        print("No matches — cannot test serialization.")
        return

    print("\n" + "=" * 80)
    print("VALIDATING EACH MATCH AGAINST PRODUCTION CompanyOutput")
    print("=" * 80)

    fail_count = 0
    for i, m in enumerate(result.matches, 1):
        try:
            submission = to_submission_dict(m.company)
        except Exception as e:
            print(f"[{i}] ✗ serializer crashed: {e}")
            fail_count += 1
            continue

        try:
            prod = ProdCompanyOutput.model_validate(submission)
            print(f"[{i}] ✓ {submission['company_name']:30} "
                  f"({len(submission['intent_signals'])} signals)")
        except Exception as e:
            fail_count += 1
            print(f"[{i}] ✗ {submission.get('company_name','?'):30} — validation failed:")
            print(f"     {type(e).__name__}: {str(e)[:400]}")
            print(f"     submission payload:")
            print("    ", json.dumps(submission, indent=4, default=str).replace("\n", "\n    ")[:2000])

    print(f"\n{'='*80}")
    print(f"VALIDATION RESULT: {len(result.matches) - fail_count}/{len(result.matches)} pass")
    print(f"{'='*80}")

    # Show one full passing submission as proof
    if result.matches and fail_count < len(result.matches):
        first = to_submission_dict(result.matches[0].company)
        print("\nFULL PRODUCTION-SHAPED SUBMISSION (1st match):")
        print(json.dumps(first, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())
