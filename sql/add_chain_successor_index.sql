-- Applied to production 2026-06-23 (live, via session pooler; CONCURRENTLY, no lock).
-- This file documents the change for reproducibility / DB rebuilds.
--
-- Root cause it fixes:
--   The chain-walk SQL functions get_chain_winners(), get_chain_held_count(),
--   and get_chain_root_num_leads() use a RECURSIVE CTE that joins
--     fulfillment_requests fr ON fr.successor_request_id = chain.request_id
--   to walk a request chain. fulfillment_requests had NO index on
--   successor_request_id, so every recursive step did a full ~6,000-row
--   Seq Scan. PostgREST calls these RPCs heavily during consensus (15-23
--   concurrent), so the seq scans saturated the connection pool — making the
--   REST API stall 5-35s for ALL callers (miner /fulfillment/requests/active
--   timed out; the validator's /rewards/active exceeded its 45s budget and
--   dropped fulfillment rewards, spiking burn). Postgres itself was idle and
--   the queries' execution was sub-ms; the cost was entirely the unindexed
--   recursive join under concurrency.
--
-- Effect (measured): recursive chain walk 25.3ms -> 0.11ms (222x); the three
-- RPCs 3-8s -> 0.002s; concurrent slow queries 14-30 -> 0; REST latency
-- 5-35s -> ~0.05s; miner endpoint 8/8 timeouts -> 10/10 fast.

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_fr_successor
  ON fulfillment_requests (successor_request_id);

ANALYZE fulfillment_requests;
