-- ============================================================
-- Delete the stale 25-industry active ICP set so the rotation
-- task regenerates a fresh 20-industry set on the next tick.
--
-- Run this AFTER the new code (TOTAL_ICPS=20, 20-industry list)
-- is deployed to the gateway and the gateway is restarted.
-- Order:
--   1. git push (already done by the agent)
--   2. Restart the gateway so the new INDUSTRY_DISTRIBUTION + new
--      get_total_leads()=100 / SUBMISSION_COST=$10 take effect
--   3. Run this SQL in the Supabase SQL editor (service_role)
--   4. Within ~60s the icp_rotation_task() detects the missing
--      active set and regenerates one with the new 20-industry list
--   5. (Optional) verify with the SELECT block at the bottom
--
-- Idempotent: safe to run repeatedly. If no active set exists, the
-- DELETE statement matches 0 rows and the COMMIT is a no-op.
--
-- Atomic via BEGIN/COMMIT so partial deletes can't leave the DB in
-- a half-state.
-- ============================================================

BEGIN;

-- =====================================================================
-- Step 1: Snapshot what we're about to delete (for the operator's log).
-- =====================================================================
-- This RAISE NOTICE will print to the SQL editor's output pane. If it
-- says "industry_count = 25" you're deleting the stale set; if it says
-- "industry_count = 20" you're already on the new set and this SQL is
-- a no-op (the DELETE below will match 0 rows).
-- =====================================================================

DO $$
DECLARE
    r RECORD;
    industry_count_dist INTEGER;
BEGIN
    FOR r IN
        SELECT
            set_id,
            is_active,
            active_from,
            active_until,
            industry_distribution,
            jsonb_array_length(icps) AS icp_count
        FROM qualification_private_icp_sets
        WHERE is_active = TRUE
    LOOP
        -- industry_distribution is a jsonb OBJECT ({"Software": 1, ...}),
        -- not an array.  Count its keys defensively; handle null / wrong
        -- shape without crashing.
        IF r.industry_distribution IS NULL THEN
            industry_count_dist := -1;  -- sentinel: missing column
        ELSIF jsonb_typeof(r.industry_distribution) = 'object' THEN
            SELECT count(*)::int INTO industry_count_dist
            FROM jsonb_object_keys(r.industry_distribution);
        ELSIF jsonb_typeof(r.industry_distribution) = 'array' THEN
            industry_count_dist := jsonb_array_length(r.industry_distribution);
        ELSE
            industry_count_dist := -2;  -- sentinel: unexpected shape
        END IF;

        RAISE NOTICE
            'About to delete: set_id=%, active_from=%, active_until=%, icp_count=%, industry_keys_in_distribution=%',
            r.set_id, r.active_from, r.active_until, r.icp_count, industry_count_dist;
    END LOOP;
END $$;

-- =====================================================================
-- Step 2: Delete the active set(s).  In practice there should be
-- exactly 0 or 1 row — the rotation task only ever activates the
-- newest today_set_id.
-- =====================================================================

DELETE FROM qualification_private_icp_sets
WHERE is_active = TRUE;

-- =====================================================================
-- Step 3: (Defensive) also delete any future-dated set that was
-- pre-generated with the old 25-industry distribution.  An ICP set is
-- "future" relative to NOW if active_from > now().
-- =====================================================================

DELETE FROM qualification_private_icp_sets
WHERE active_from > NOW()
  AND jsonb_typeof(icps) = 'array'
  AND jsonb_array_length(icps) <> 20;

COMMIT;

-- =====================================================================
-- Step 4: Verification — run after the rotation task fires.  Expect
-- one row with icp_count=20 and the active_from / active_until window
-- spanning today (UTC).
-- =====================================================================

-- SELECT
--   set_id,
--   is_active,
--   active_from,
--   active_until,
--   jsonb_array_length(icps) AS icp_count
-- FROM qualification_private_icp_sets
-- ORDER BY active_from DESC
-- LIMIT 5;
