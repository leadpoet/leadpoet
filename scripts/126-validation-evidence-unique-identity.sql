-- Enforce one validation-evidence row per (epoch_id, validator_hotkey, lead_id).
--
-- The gateway's duplicate-submission guard (gateway/api/validate.py) is a
-- check-then-insert, and the insert writes a fresh uuid per row with no
-- ON CONFLICT, so a concurrent double-submit (or a retry after a partial
-- commit) can create duplicate evidence rows for the same identity. Consensus
-- already dedups by validator in-app (gateway/utils/consensus.py, one vote per
-- validator), so this is not a live payout double-count today; this index is
-- the belt-and-suspenders DB-level guard so any future consumer cannot
-- double-count a validator's v_trust*stake, and so the retry path fails closed
-- instead of duplicating.
--
-- ⚠️ OPERATOR-RUN, NOT auto-apply. This mutates a live weight-critical table.
-- Apply the three steps IN ORDER, reviewing counts before the DELETE.
-- CONCURRENTLY cannot run inside a transaction, so keep Step 3 separate.

-- ── Step 1: how many duplicate rows exist? (read-only pre-check) ────────────
-- SELECT count(*) AS duplicate_rows FROM (
--   SELECT epoch_id, validator_hotkey, lead_id, count(*) c
--   FROM public.validation_evidence_private
--   GROUP BY 1,2,3 HAVING count(*) > 1
-- ) d;

-- ── Step 2: remove duplicates, keeping the most recent row per identity ─────
-- Review the Step-1 count first. Run inside a transaction.
-- BEGIN;
-- WITH ranked AS (
--   SELECT ctid,
--          row_number() OVER (
--            PARTITION BY epoch_id, validator_hotkey, lead_id
--            ORDER BY created_at DESC NULLS LAST, ctid DESC
--          ) AS rn
--   FROM public.validation_evidence_private
-- )
-- DELETE FROM public.validation_evidence_private v
--   USING ranked r
--   WHERE v.ctid = r.ctid AND r.rn > 1;
-- COMMIT;

-- ── Step 3: enforce uniqueness going forward (OUTSIDE any transaction) ──────
CREATE UNIQUE INDEX CONCURRENTLY IF NOT EXISTS
  uq_validation_evidence_epoch_validator_lead
  ON public.validation_evidence_private (epoch_id, validator_hotkey, lead_id);
