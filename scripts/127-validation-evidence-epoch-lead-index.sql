-- Index validation_evidence_private on (epoch_id, lead_id) for the per-lead
-- consensus read.
--
-- compute_weighted_consensus's DB fallback (gateway/utils/consensus.py) filters
-- .eq("lead_id", ...).eq("epoch_id", ...); the epoch-lifecycle consensus pass
-- resolves per-lead evidence the same way. The only migration index on this
-- table is idx_validation_evidence_epoch_identity_v1 on (epoch_id) alone
-- (migration 100), so a per-lead lookup scans every row for the epoch
-- (thousands per epoch). The SEC1 unique index (migration 126) on
-- (epoch_id, validator_hotkey, lead_id) does not serve an (epoch_id, lead_id)
-- filter efficiently (validator_hotkey is the second key, not lead_id).
--
-- Additive, idempotent, CONCURRENTLY (run OUTSIDE a transaction). No-op if an
-- equivalent index already exists on the base table.

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_validation_evidence_epoch_lead
  ON public.validation_evidence_private (epoch_id, lead_id);
