-- Operator reconciliation for one usage epoch. Supply -v epoch=<number> to psql
-- on the read replica or in a read-only owner session. Each finalized call is
-- one immutable source row. Use call_id to deduplicate exports and reconcile
-- against external payments. These are accruals; this report transfers no funds.
SELECT s.call_id, s.epoch, k.miner_hotkey, c.tool, c.operation,
       c.ask_microusd, s.units, s.charge_microusd AS buyer_charge_microusd,
       s.owed_microusd AS miner_accrued_microusd,
       s.charge_microusd - s.owed_microusd AS gross_fee_microusd,
       s.settled_at
FROM public.tool_market_call_settlements s
JOIN public.tool_market_calls c USING (call_id)
JOIN public.tool_market_keys k USING (key_id)
WHERE s.epoch = :'epoch'::bigint AND s.status = 'settled' AND s.owed_microusd > 0
ORDER BY k.miner_hotkey, s.call_id;
