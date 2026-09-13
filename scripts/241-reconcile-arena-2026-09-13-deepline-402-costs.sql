-- Reconcile eight exact Deepline HTTP 402 calls whose authenticated billing
-- histories were read through every provider-directed recent_offset page at
-- 2026-09-13T16:43Z. Every page reported incomplete=false and pagination
-- reached has_more=false. For 5dff the newest returned charge was
-- 2026-09-13T10:19:29.846Z, before uncertainties at 10:19:30.971Z through
-- 10:46:49.814Z.  For caf0 the newest charge was 2026-09-12T00:22:34.373Z,
-- before its uncertainty at 2026-09-13T00:04:39.662Z.  Both histories had no
-- later charge. This appends zero settlements; it never changes prior rows.

BEGIN;
LOCK TABLE public.lab_arena_ledger IN SHARE ROW EXCLUSIVE MODE;

DO $reconcile_deepline_402$
DECLARE
  v_expected RECORD;
  v_head public.lab_arena_ledger%ROWTYPE;
  v_reservation public.lab_arena_ledger%ROWTYPE;
BEGIN
  FOR v_expected IN
    SELECT * FROM (VALUES
      (315826::BIGINT, 'sub-caf0e1ef30c9712e6385afe24a75375e', 'arena-2026-09-13:sub-caf0e1ef30c9712e6385afe24a75375e:1:0:1', 1, 'sha256:223335eda1ccf9fcf8b0d4d3a00f85d957cf896611b8c845fea331be83c8c14d', 'deepline.execute', 79982421::BIGINT, 1653),
      (331371, 'sub-5dffdbaa2b96e8dc78160aea8f80a7b9', 'arena-2026-09-13:sub-5dffdbaa2b96e8dc78160aea8f80a7b9:2:12:1', 2, 'sha256:6bc891aeb6faf7f07d1fe97e3f9cc12e3cac686211097473456b77613a524f2c', 'deepline.execute', 56000, 1928),
      (331374, 'sub-5dffdbaa2b96e8dc78160aea8f80a7b9', 'arena-2026-09-13:sub-5dffdbaa2b96e8dc78160aea8f80a7b9:2:12:1', 2, 'sha256:bbeb448fa9f273c0b58b6a82ad8ae7d7154e20a8789d39b596f0f5dcce7bbb76', 'deepline.execute', 56000, 1968),
      (331383, 'sub-5dffdbaa2b96e8dc78160aea8f80a7b9', 'arena-2026-09-13:sub-5dffdbaa2b96e8dc78160aea8f80a7b9:2:12:1', 2, 'sha256:070df2985bf72f35fdc6d01c93786769f6fb8f7d976294ddf1fc14fe756d3496', 'deepline.execute', 56000, 1918),
      (331388, 'sub-5dffdbaa2b96e8dc78160aea8f80a7b9', 'arena-2026-09-13:sub-5dffdbaa2b96e8dc78160aea8f80a7b9:2:12:1', 2, 'sha256:231f8c7f751019140a87611f800b4cc1fc6256b440aad576ba3d31899310a7da', 'deepline.execute', 56000, 1928),
      (331398, 'sub-5dffdbaa2b96e8dc78160aea8f80a7b9', 'arena-2026-09-13:sub-5dffdbaa2b96e8dc78160aea8f80a7b9:2:12:1', 2, 'sha256:83b62e720fd19d4bafa5bab2d2dd09345b591f28e576f4d87c7bab19198894f1', 'deepline.execute', 56000, 1968),
      (331411, 'sub-5dffdbaa2b96e8dc78160aea8f80a7b9', 'arena-2026-09-13:sub-5dffdbaa2b96e8dc78160aea8f80a7b9:2:12:1', 2, 'sha256:bf2cd93f1511916c9c2c5967605d058d830c4797f0cdc055d60d29f339749fb1', 'deepline.execute', 73321638, 1653),
      (336996, 'sub-5dffdbaa2b96e8dc78160aea8f80a7b9', 'arena-2026-09-13:sub-5dffdbaa2b96e8dc78160aea8f80a7b9:2:10:score:1', 2, 'sha256:94484384d8b1dfcb5de6ec34755d9c55abb4ef2cf8fc826e624de541d5d29a93', 'scrapingdog.scrape', 49049470, 1743)
    ) AS expected(uncertain_entry_id, submission_id, run_id, stage, call_identity, operation_id, reserved_microusd, body_bytes)
  LOOP
    SELECT * INTO v_head FROM public.lab_arena__ledger_head(v_expected.call_identity);
    IF v_head.entry_kind = 'settlement'
       AND v_head.entry_doc ->> 'deepline_402_history_reconciliation' = 'true'
       AND (v_head.entry_doc ->> 'reconciled_uncertainty_entry_id')::BIGINT = v_expected.uncertain_entry_id
       AND v_head.amount_microusd = 0 THEN
      CONTINUE;
    END IF;
    IF v_head.entry_id IS DISTINCT FROM v_expected.uncertain_entry_id
       OR v_head.entry_kind IS DISTINCT FROM 'uncertain'
       OR v_head.round_id IS DISTINCT FROM 'arena-2026-09-13'
       OR v_head.submission_id IS DISTINCT FROM v_expected.submission_id
       OR v_head.run_id IS DISTINCT FROM v_expected.run_id
       OR v_head.stage IS DISTINCT FROM v_expected.stage
       OR v_head.call_identity IS DISTINCT FROM v_expected.call_identity
       OR v_head.provider IS DISTINCT FROM 'deepline'
       OR v_head.operation_id IS DISTINCT FROM v_expected.operation_id
       OR v_head.funding_source IS DISTINCT FROM 'miner_key'
       OR v_head.amount_microusd IS DISTINCT FROM v_expected.reserved_microusd
       OR v_head.entry_doc ->> 'reason' IS DISTINCT FROM 'worker_reported'
       OR v_head.entry_doc #>> '{call,reason}' IS DISTINCT FROM 'missing_provider_cost'
       OR v_head.entry_doc #>> '{call,provider_status}' IS DISTINCT FROM '402'
       OR v_head.entry_doc #>> '{call,call_succeeded}' IS DISTINCT FROM 'false'
       OR v_head.entry_doc #>> '{call,body_is_mapping}' IS DISTINCT FROM 'true'
       OR v_head.entry_doc #>> '{call,billing_present}' IS DISTINCT FROM 'true'
       OR v_head.entry_doc #>> '{call,usage_present}' IS DISTINCT FROM 'false'
       OR (v_head.entry_doc #>> '{call,body_bytes}')::INTEGER IS DISTINCT FROM v_expected.body_bytes THEN
      RAISE EXCEPTION 'arena_20260913_deepline_402_head_mismatch:%', v_expected.uncertain_entry_id;
    END IF;
    SELECT * INTO v_reservation FROM public.lab_arena_ledger
     WHERE call_identity=v_expected.call_identity AND entry_kind='reservation';
    IF v_reservation.entry_id IS NULL
       OR v_reservation.amount_microusd IS DISTINCT FROM v_expected.reserved_microusd
       OR v_reservation.run_id IS DISTINCT FROM v_expected.run_id
       OR NOT EXISTS (SELECT 1 FROM public.lab_arena_ledger d WHERE d.call_identity=v_expected.call_identity AND d.entry_kind='dispatch') THEN
      RAISE EXCEPTION 'arena_20260913_deepline_402_chain_mismatch:%', v_expected.uncertain_entry_id;
    END IF;
    INSERT INTO public.lab_arena_ledger(entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,call_identity,provider,operation_id,funding_source,amount_microusd,entry_doc,terminal_response)
    VALUES ('settlement',v_head.miner_hotkey,v_head.round_id,v_head.submission_id,v_head.run_id,v_head.stage,v_head.call_identity,v_head.provider,v_head.operation_id,v_head.funding_source,0,
      pg_catalog.jsonb_build_object('reserved_microusd',v_head.amount_microusd,'released_microusd',v_head.amount_microusd,'variance_microusd',-v_head.amount_microusd,'late_reconciliation',true,'deepline_402_history_reconciliation',true,'reconciled_uncertainty_entry_id',v_head.entry_id,'provider_history_checked_at','2026-09-13T16:43:00Z'),
      pg_catalog.jsonb_build_object('status',502,'headers',pg_catalog.jsonb_build_object('content-type','application/json','content-length','41'),'body_b64','eyJlcnJvciI6eyJjb2RlIjoicHJvdmlkZXJfdW5hdmFpbGFibGUifX0=','call_succeeded',false,'provider_cost',pg_catalog.jsonb_build_object('basis','deepline_authenticated_history_no_charge','units','0','unit_name','credits','operation',v_head.operation_id)));
  END LOOP;
  IF (SELECT count(*) FROM public.lab_arena_ledger
      WHERE entry_kind='settlement'
        AND entry_doc->>'deepline_402_history_reconciliation'='true'
        AND (entry_doc->>'reconciled_uncertainty_entry_id')::BIGINT IN
          (315826,331371,331374,331383,331388,331398,331411,336996)) <> 8 THEN
    RAISE EXCEPTION 'arena_20260913_deepline_402_settlement_count_mismatch';
  END IF;
END;
$reconcile_deepline_402$;

COMMIT;
