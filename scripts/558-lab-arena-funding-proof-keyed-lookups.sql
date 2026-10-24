-- Keep exact call and native-request keys as parameterized index conditions.
-- Lookup barriers prevent per-head scans of every reservation in the run;
-- every existing funding, owner, floor, history and duplicate guard remains.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';
DO $funding_lookup558$
DECLARE d TEXT; metadata JSONB;
  target OID:='public.lab_arena_miner_credit_failure_v1(text)'::regprocedure;
BEGIN
  IF NOT EXISTS(SELECT 1 FROM pg_index x JOIN pg_class i ON i.oid=x.indexrelid
      JOIN pg_class t ON t.oid=x.indrelid JOIN pg_am a ON a.oid=i.relam
      WHERE i.oid=to_regclass('public.lab_arena_ledger_paid_native_request_idx')
        AND t.oid='public.lab_arena_ledger'::regclass AND a.amname='btree'
        AND pg_get_userbyid(i.relowner)='lab_arena_owner'
        AND x.indisvalid AND x.indisready AND NOT x.indisunique
        AND pg_get_indexdef(i.oid) = $index_definition$CREATE INDEX lab_arena_ledger_paid_native_request_idx ON public.lab_arena_ledger USING btree (round_id, submission_id, provider, ((terminal_response #>> '{provider_cost,request_id}'::text[]))) WHERE ((entry_kind = 'settlement'::text) AND (amount_microusd > 0) AND ((length((terminal_response #>> '{provider_cost,request_id}'::text[])) >= 1) AND (length((terminal_response #>> '{provider_cost,request_id}'::text[])) <= 256)))$index_definition$) THEN
    RAISE EXCEPTION 'funding throughput native index differs' USING ERRCODE='55000';
  END IF;
  SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc' INTO STRICT d,metadata
    FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang WHERE p.oid=target
      AND pg_get_userbyid(p.proowner)='lab_arena_owner'
      AND p.proacl::TEXT='{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}'
      AND p.prosecdef AND p.provolatile='s'
      AND p.proconfig=ARRAY['search_path=pg_catalog, public']::TEXT[] AND l.lanname='plpgsql';
  IF encode(extensions.digest(d,'sha256'),'hex')='23fbb6e5d40450916b75110d16d9b3c9c369264014dea51e17db7f3f978786c6' THEN RETURN; END IF;
  IF encode(extensions.digest(d,'sha256'),'hex')<>'a962ddae017d3e1efbbe05d296bd1b9c895523d4a5081b0a3fd2c78b9c7f3d73' THEN
    RAISE EXCEPTION 'funding throughput preimage differs' USING ERRCODE='55000';
  END IF;
  IF (length(d)-length(replace(d,$old_lookup558_1$      FROM required k
      JOIN public.lab_arena_ledger head ON head.provider=k.provider
        AND head.created_at>k.original_denial
        AND head.round_id=requested_run.round_id
        AND head.submission_id=requested_run.submission_id
        AND head.miner_hotkey=requested_run.miner_hotkey
        AND head.entry_kind IN ('settlement','uncertain')
        AND head.funding_source='miner_key'
$old_lookup558_1$,'')))<>length($old_lookup558_1$      FROM required k
      JOIN public.lab_arena_ledger head ON head.provider=k.provider
        AND head.created_at>k.original_denial
        AND head.round_id=requested_run.round_id
        AND head.submission_id=requested_run.submission_id
        AND head.miner_hotkey=requested_run.miner_hotkey
        AND head.entry_kind IN ('settlement','uncertain')
        AND head.funding_source='miner_key'
$old_lookup558_1$) THEN
    RAISE EXCEPTION 'funding keyed lookup anchor differs' USING ERRCODE='55000';
  END IF;
  d:=replace(d,$old_lookup558_1$      FROM required k
      JOIN public.lab_arena_ledger head ON head.provider=k.provider
        AND head.created_at>k.original_denial
        AND head.round_id=requested_run.round_id
        AND head.submission_id=requested_run.submission_id
        AND head.miner_hotkey=requested_run.miner_hotkey
        AND head.entry_kind IN ('settlement','uncertain')
        AND head.funding_source='miner_key'
$old_lookup558_1$,$new_lookup558_1$      FROM required k
      CROSS JOIN LATERAL(SELECT scoped_head.* FROM public.lab_arena_ledger scoped_head
        WHERE scoped_head.provider=k.provider AND scoped_head.created_at>k.original_denial
          AND scoped_head.round_id=requested_run.round_id AND scoped_head.submission_id=requested_run.submission_id
          AND scoped_head.miner_hotkey=requested_run.miner_hotkey
          AND scoped_head.entry_kind IN ('settlement','uncertain')
          AND scoped_head.funding_source='miner_key'
        OFFSET 0) head
$new_lookup558_1$);
  IF (length(d)-length(replace(d,$old_lookup558_2$      JOIN public.lab_arena_ledger reservation ON reservation.run_id=head.run_id
        AND reservation.call_identity=head.call_identity AND reservation.entry_kind='reservation'
$old_lookup558_2$,'')))<>length($old_lookup558_2$      JOIN public.lab_arena_ledger reservation ON reservation.run_id=head.run_id
        AND reservation.call_identity=head.call_identity AND reservation.entry_kind='reservation'
$old_lookup558_2$) THEN
    RAISE EXCEPTION 'funding keyed lookup anchor differs' USING ERRCODE='55000';
  END IF;
  d:=replace(d,$old_lookup558_2$      JOIN public.lab_arena_ledger reservation ON reservation.run_id=head.run_id
        AND reservation.call_identity=head.call_identity AND reservation.entry_kind='reservation'
$old_lookup558_2$,$new_lookup558_2$      JOIN LATERAL(SELECT original_reservation.* FROM public.lab_arena_ledger original_reservation
        WHERE original_reservation.run_id=head.run_id
          AND original_reservation.call_identity=head.call_identity
          AND original_reservation.entry_kind='reservation'
        OFFSET 0) reservation ON TRUE
$new_lookup558_2$);
  IF (length(d)-length(replace(d,$old_lookup558_3$FROM public.lab_arena_ledger duplicate
$old_lookup558_3$,'')))<>length($old_lookup558_3$FROM public.lab_arena_ledger duplicate
$old_lookup558_3$) THEN
    RAISE EXCEPTION 'funding keyed lookup anchor differs' USING ERRCODE='55000';
  END IF;
  d:=replace(d,$old_lookup558_3$FROM public.lab_arena_ledger duplicate
$old_lookup558_3$,$new_lookup558_3$FROM LATERAL(
            SELECT native.* FROM public.lab_arena_ledger native
            WHERE native.round_id=requested_run.round_id AND native.submission_id=requested_run.submission_id
              AND native.provider=success.provider AND native.miner_hotkey=requested_run.miner_hotkey
              AND native.entry_kind='settlement' AND native.amount_microusd>0
              AND length(native.terminal_response#>>'{provider_cost,request_id}') BETWEEN 1 AND 256
              AND native.terminal_response#>>'{provider_cost,request_id}'=success.native_request_id
            OFFSET 0
          ) duplicate
$new_lookup558_3$);
  IF (length(d)-length(replace(d,$old_lookup558_4$          JOIN public.lab_arena_ledger original ON original.run_id=duplicate.run_id
            AND original.call_identity=duplicate.call_identity AND original.entry_kind='reservation'
$old_lookup558_4$,'')))<>length($old_lookup558_4$          JOIN public.lab_arena_ledger original ON original.run_id=duplicate.run_id
            AND original.call_identity=duplicate.call_identity AND original.entry_kind='reservation'
$old_lookup558_4$) THEN
    RAISE EXCEPTION 'funding keyed lookup anchor differs' USING ERRCODE='55000';
  END IF;
  d:=replace(d,$old_lookup558_4$          JOIN public.lab_arena_ledger original ON original.run_id=duplicate.run_id
            AND original.call_identity=duplicate.call_identity AND original.entry_kind='reservation'
$old_lookup558_4$,$new_lookup558_4$          JOIN LATERAL(SELECT duplicate_reservation.* FROM public.lab_arena_ledger duplicate_reservation
            WHERE duplicate_reservation.run_id=duplicate.run_id
              AND duplicate_reservation.call_identity=duplicate.call_identity
              AND duplicate_reservation.entry_kind='reservation'
            OFFSET 0) original ON TRUE
$new_lookup558_4$);
  EXECUTE d;
  IF (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=target) IS DISTINCT FROM metadata
    OR encode(extensions.digest(pg_get_functiondef(target),'sha256'),'hex')<>'23fbb6e5d40450916b75110d16d9b3c9c369264014dea51e17db7f3f978786c6' THEN
    RAISE EXCEPTION 'funding keyed lookup readback differs' USING ERRCODE='55000';
  END IF;
END;
$funding_lookup558$;
COMMIT;
