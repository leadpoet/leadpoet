"""Per-evidence-type prompt builders for the three-stage intent verifier.

Submodules:
  ``_common``       — PART 0 entity, PART A claim-ICP, PART B URL-supports-claim,
                      signal_status decision rules, final-judge rules,
                      miner-date consistency check.  Shared assemblers.
  ``social_posting`` — adds PART D author-role check.
  ``default``       — legacy compatibility builder used for any evidence_type
                      that is not yet specialised (HIRING / FUNDING / None /
                      unrecognised values).  During this refactor it still
                      includes PART D so the dispatcher's output is
                      byte-identical to the pre-refactor prompt.
"""
