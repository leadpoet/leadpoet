-- Read-only reconciliation: psql -v epoch=EPOCH -f scripts/tool_market_epoch_targets.sql.
-- Each row is an accepted target. Gateway reports identify reveal evidence to
-- verify against Finney; neither a report count nor a weight is a token receipt.
SELECT w.epoch, w.state_hash, w.state_doc->>'valid_from_block' AS boundary_block,
       a.key AS miner_hotkey, a.value::bigint AS scheduled_microusd,
       w.state_doc#>>'{tool_market,emission_microusd}' AS epoch_emission_microusd,
       coalesce(w.state_doc#>'{tool_market,valuation}', w.state_doc#>'{reward_basis,valuation}') AS valuation,
       (SELECT count(*) FROM public.lab_arena_chain_outcomes o
         WHERE (o.network,o.netuid,o.epoch)=(w.network,w.netuid,w.epoch)
           AND o.outcome_doc->>'state_hash'=w.state_hash) AS matching_validator_reports
FROM public.lab_arena_accepted_weight_states w
CROSS JOIN LATERAL jsonb_each_text(w.state_doc#>'{tool_market,allocations_microusd}') a
WHERE w.network='finney' AND w.netuid=71 AND w.epoch=:'epoch'::bigint AND w.state_doc ? 'tool_market'
ORDER BY a.key;
