"""Per-evidence-type prompt builders for the three-stage intent verifier.

Submodules:
  ``_common``       — PART 0 entity, PART A claim-ICP, PART B URL-supports-claim,
                      signal_status decision rules, final-judge rules,
                      miner-date consistency check.  Shared assemblers.
  ``social_posting`` — adds PART D author-role check.
  ``default``       — shared builder for HIRING, FUNDING, missing, and other
                      evidence types without a specialised builder. PART D
                      includes its own applicability condition.
"""
