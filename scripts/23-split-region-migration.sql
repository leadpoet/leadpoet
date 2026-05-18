-- 23-split-region-migration.sql
-- Two-phase data migration for the split-region feature.
--
-- Goal: rename ``country``/``geography`` keys inside fulfillment_requests.icp_details
-- to the new ``company_country``/``company_region``, plus reclassify Adedeji 5's
-- location filter from company-side to contact-side.
--
-- ZERO-DOWNTIME DEPLOY STRATEGY (EXPAND → CODE DEPLOY → CONTRACT):
-- ════════════════════════════════════════════════════════════════
-- The current production code reads ``country`` / ``geography``.  The new
-- code reads ``company_country`` / ``company_region``.  A naive rename
-- (drop old keys, add new keys) BREAKS the still-serving old code mid-
-- deploy because rows would have neither old (just dropped) nor be readable
-- by old code (doesn't know the new names).
--
-- Instead use the standard expand-migrate-contract pattern:
--
--   ┌─────────────────────────────────────────────────────────────────────┐
--   │ PHASE 1 (EXPAND) — run THIS, BEFORE or DURING code deploy           │
--   │ Safe while old code is still serving traffic.                       │
--   ├─────────────────────────────────────────────────────────────────────┤
--   │ ADD ``company_country`` = ``country`` to every row that has the     │
--   │ legacy key.  ADD ``company_region`` = ``geography`` likewise.       │
--   │ KEEP the legacy keys so old code keeps reading them.                │
--   │                                                                      │
--   │ After this step every row has BOTH shapes.  Old code reads the      │
--   │ legacy keys, new code reads the new keys — both work in parallel    │
--   │ throughout the deploy.                                              │
--   └─────────────────────────────────────────────────────────────────────┘
--
--   ┌─────────────────────────────────────────────────────────────────────┐
--   │ CODE DEPLOY — push commit + rsync gateway + run deploy_dynamic.sh  │
--   │ on validator.  No DB ordering coupling.                             │
--   └─────────────────────────────────────────────────────────────────────┘
--
--   ┌─────────────────────────────────────────────────────────────────────┐
--   │ PHASE 2 (CONTRACT) — run AFTER all old containers are gone.        │
--   ├─────────────────────────────────────────────────────────────────────┤
--   │ Drop the legacy ``country`` / ``geography`` keys (storage hygiene). │
--   │ Apply Adedeji 5 reclassification (move filter to contact side).    │
--   │                                                                      │
--   │ After this step rows have only the new shape.                       │
--   └─────────────────────────────────────────────────────────────────────┘
--
-- Both phases are idempotent: re-running matches nothing once applied.

-- ══════════════════════════════════════════════════════════════════════
-- ## PHASE 1 — EXPAND (run before / during code deploy)
-- ══════════════════════════════════════════════════════════════════════

BEGIN;

-- For every row that still has a legacy ``country`` or ``geography``,
-- duplicate the value into the new key.  COALESCE ensures empty defaults
-- ("[]"::jsonb for country, '""'::jsonb for region) if the legacy key
-- existed but was null.  The WHERE clause skips rows that already have
-- BOTH new keys (i.e. were created post-deploy with the new shape) so
-- re-running this is a no-op.
UPDATE fulfillment_requests
SET icp_details = icp_details
    || jsonb_build_object(
        'company_country',
        COALESCE(icp_details->'country', '[]'::jsonb)
    )
    || jsonb_build_object(
        'company_region',
        COALESCE(icp_details->'geography', '""'::jsonb)
    )
WHERE (icp_details ? 'country' OR icp_details ? 'geography')
  AND NOT (icp_details ? 'company_country' AND icp_details ? 'company_region');

-- Phase 1 verify: every row that had legacy keys now also has the new ones.
SELECT
    'Phase 1 verify — rows with legacy but not new keys (must be 0)' AS check_name,
    COUNT(*) AS count
FROM fulfillment_requests
WHERE (icp_details ? 'country' OR icp_details ? 'geography')
  AND NOT (icp_details ? 'company_country' AND icp_details ? 'company_region');

COMMIT;

-- ⚠️ STOP HERE.  Deploy the code change.  Verify the new validator-main
-- and gateway are running the new commit (ad805539+).  Confirm at least
-- one full FF dispatch cycle completes cleanly under the new code before
-- moving to Phase 2.


-- ══════════════════════════════════════════════════════════════════════
-- ## PHASE 2 — CONTRACT (run only AFTER all old containers are gone)
-- ══════════════════════════════════════════════════════════════════════

-- BEGIN;
--
-- -- Step 2a: remove the legacy keys now that no live code reads them.
-- UPDATE fulfillment_requests
-- SET icp_details = icp_details - 'country' - 'geography'
-- WHERE icp_details ? 'country' OR icp_details ? 'geography';
--
-- -- Step 2b: Adedeji 5 — move location filter from company side to
-- -- contact side.  Their existing ``company_region`` (= "New Jersey,
-- -- Pennsylvania") was always intended as a PERSON-location filter, but
-- -- pre-rename code matched it against company HQ.  Move it to
-- -- contact_region + populate contact_country so the new contact gate
-- -- replaces the expensive Sonar attribute-verification path their
-- -- required_attributes.contact[] currently does.
-- --
-- -- Targets only the chain HEAD (no successor yet); the lifecycle.py
-- -- recycle code (fixed in commit 46fdca9a) carries the new fields
-- -- forward into all subsequent successors automatically.
-- UPDATE fulfillment_requests
-- SET icp_details = (
--     icp_details
--     - 'company_region'  -- remove from company side
--     || jsonb_build_object(
--         'contact_country',
--         COALESCE(icp_details->'company_country', '["United States"]'::jsonb)
--     )
--     || jsonb_build_object(
--         'contact_region',
--         '"New Jersey, Pennsylvania"'::jsonb
--     )
-- )
-- WHERE internal_label = 'Adedeji 5'
--   AND status IN ('open','continued_open','commit_closed','scoring','partially_fulfilled')
--   AND successor_request_id IS NULL;
--
-- -- Phase 2 verify: zero legacy keys remain.
-- SELECT
--     'Phase 2 verify — rows still using legacy keys (must be 0)' AS check_name,
--     COUNT(*) AS count
-- FROM fulfillment_requests
-- WHERE icp_details ? 'country' OR icp_details ? 'geography';
--
-- -- Phase 2 verify: Adedeji 5 chain head now has contact_* fields.
-- SELECT
--     request_id, internal_label, status,
--     icp_details->'company_country' AS company_country,
--     icp_details->'company_region'  AS company_region,
--     icp_details->'contact_country' AS contact_country,
--     icp_details->'contact_region'  AS contact_region
-- FROM fulfillment_requests
-- WHERE internal_label = 'Adedeji 5'
--   AND status IN ('open','continued_open','commit_closed','scoring','partially_fulfilled')
--   AND successor_request_id IS NULL;
--
-- -- Sample post-migration shape across all 7 active clients (info only):
-- SELECT
--     internal_label,
--     icp_details->'company_country' AS cc,
--     icp_details->'company_region'  AS cr,
--     icp_details->'contact_country' AS pc,
--     icp_details->'contact_region'  AS pr
-- FROM fulfillment_requests
-- WHERE internal_label IN (
--     'Bruce Callahan 5', 'Daniel iMove 10', 'Leadpoet 20 Large ICP',
--     'Adedeji 5', '9 Vicks', '5 Stan', '5 Kyle Big'
-- )
-- AND status IN ('open','continued_open','commit_closed','scoring')
-- ORDER BY internal_label
-- LIMIT 20;
--
-- COMMIT;
