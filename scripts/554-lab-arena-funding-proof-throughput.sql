-- Preserve the funding proof while reading only potentially relevant heads.
-- Every historical positive native identity remains covered by the exact
-- latest-head, owner, payer, floor and required-key duplicate check.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';
DO $funding_throughput554$
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
  IF encode(extensions.digest(d,'sha256'),'hex')='a962ddae017d3e1efbbe05d296bd1b9c895523d4a5081b0a3fd2c78b9c7f3d73' THEN RETURN; END IF;
  IF encode(extensions.digest(d,'sha256'),'hex')<>'46acfd38b92eb7a03613a4d55c72cc9af57339cf6602ed4611955af22b511cf6' THEN
    RAISE EXCEPTION 'funding throughput preimage differs' USING ERRCODE='55000';
  END IF;
  IF (length(d)-length(replace(d,$old_outcomes554$    ), scoped AS MATERIALIZED (
      SELECT reservation.run_id,reservation.round_id,reservation.submission_id,reservation.miner_hotkey,
        reservation.stage,reservation.call_identity,reservation.provider,reservation.operation_id,
        reservation.funding_source,reservation.created_at,owned.scoring_generation_floor AS run_floor,
        jsonb_build_object('base_call_identity',reservation_fields.base_call_identity,
          'provider_attempt',reservation_fields.provider_attempt,
          'action_sequence',reservation_fields.action_sequence) AS accounting_doc,
        reservation_fields.credential_fingerprint AS fingerprint,
        reservation.entry_doc ?| ARRAY['judgment_cache_source_call_identity','judgment_cache_source_run_id'] AS reservation_cached
      FROM public.lab_arena_ledger reservation
      JOIN public.lab_arena_runs owned ON owned.run_id=reservation.run_id
      CROSS JOIN LATERAL jsonb_to_record(CASE WHEN jsonb_typeof(reservation.entry_doc)='object' THEN reservation.entry_doc ELSE '{}'::JSONB END) AS reservation_fields(base_call_identity JSONB,provider_attempt JSONB,action_sequence JSONB,credential_fingerprint TEXT,judgment_cache_source_call_identity JSONB,judgment_cache_source_run_id JSONB)
      WHERE reservation.round_id=requested_run.round_id
        AND reservation.submission_id=requested_run.submission_id
        AND reservation.miner_hotkey=requested_run.miner_hotkey
        AND reservation.entry_kind='reservation' AND reservation.funding_source='miner_key'
        AND owned.kind IN ('execute','score')
        AND owned.scoring_generation_floor<=requested_run.scoring_generation_floor
        AND (owned.round_id,owned.submission_id,owned.miner_hotkey,owned.stage)
          IS NOT DISTINCT FROM (reservation.round_id,reservation.submission_id,
            reservation.miner_hotkey,reservation.stage)
        AND EXISTS(SELECT 1 FROM required k WHERE k.provider=reservation.provider
          AND k.fingerprint=reservation_fields.credential_fingerprint)
    ), outcomes AS MATERIALIZED (
      SELECT reservation.*,head.entry_kind AS head_kind,head.created_at AS head_at,
        head.amount_microusd AS head_amount,
        response_fields.call_succeeded AS succeeded,
        response_fields.credit_failure_proof AS credit_proof,
        jsonb_typeof(response_fields.provider_cost->'request_id') AS native_id_type,
        response_fields.provider_cost->>'request_id' AS native_request_id,
        head.entry_doc ?| ARRAY['judgment_cache_source_call_identity','judgment_cache_source_run_id'] AS head_cached,
        head.terminal_response ?| ARRAY['judgment_cache_source_call_identity','judgment_cache_source_run_id'] AS response_cached,
        CASE head.entry_kind WHEN 'uncertain' THEN head.entry_doc#>'{call,account_failure_evidence}'
          ELSE response_fields.account_failure_evidence END AS account_evidence,
        (response_fields.call_succeeded='false'::JSONB
          OR head.entry_doc#>'{call,call_succeeded}'='false'::JSONB) AS failed
      FROM scoped reservation
      JOIN LATERAL(SELECT terminal.* FROM public.lab_arena_ledger terminal
        WHERE terminal.run_id=reservation.run_id AND terminal.call_identity=reservation.call_identity
        ORDER BY terminal.entry_id DESC LIMIT 1) head ON TRUE
      CROSS JOIN LATERAL jsonb_to_record(CASE WHEN jsonb_typeof(head.terminal_response)='object' THEN head.terminal_response ELSE '{}'::JSONB END) AS response_fields(call_succeeded JSONB,credit_failure_proof JSONB,provider_cost JSONB,account_failure_evidence JSONB,judgment_cache_source_call_identity JSONB,judgment_cache_source_run_id JSONB)
      WHERE (head.round_id,head.submission_id,head.miner_hotkey,head.stage,head.provider,
        head.operation_id,head.funding_source) IS NOT DISTINCT FROM
        (reservation.round_id,reservation.submission_id,reservation.miner_hotkey,reservation.stage,
         reservation.provider,reservation.operation_id,reservation.funding_source)
$old_outcomes554$,'')))<>length($old_outcomes554$    ), scoped AS MATERIALIZED (
      SELECT reservation.run_id,reservation.round_id,reservation.submission_id,reservation.miner_hotkey,
        reservation.stage,reservation.call_identity,reservation.provider,reservation.operation_id,
        reservation.funding_source,reservation.created_at,owned.scoring_generation_floor AS run_floor,
        jsonb_build_object('base_call_identity',reservation_fields.base_call_identity,
          'provider_attempt',reservation_fields.provider_attempt,
          'action_sequence',reservation_fields.action_sequence) AS accounting_doc,
        reservation_fields.credential_fingerprint AS fingerprint,
        reservation.entry_doc ?| ARRAY['judgment_cache_source_call_identity','judgment_cache_source_run_id'] AS reservation_cached
      FROM public.lab_arena_ledger reservation
      JOIN public.lab_arena_runs owned ON owned.run_id=reservation.run_id
      CROSS JOIN LATERAL jsonb_to_record(CASE WHEN jsonb_typeof(reservation.entry_doc)='object' THEN reservation.entry_doc ELSE '{}'::JSONB END) AS reservation_fields(base_call_identity JSONB,provider_attempt JSONB,action_sequence JSONB,credential_fingerprint TEXT,judgment_cache_source_call_identity JSONB,judgment_cache_source_run_id JSONB)
      WHERE reservation.round_id=requested_run.round_id
        AND reservation.submission_id=requested_run.submission_id
        AND reservation.miner_hotkey=requested_run.miner_hotkey
        AND reservation.entry_kind='reservation' AND reservation.funding_source='miner_key'
        AND owned.kind IN ('execute','score')
        AND owned.scoring_generation_floor<=requested_run.scoring_generation_floor
        AND (owned.round_id,owned.submission_id,owned.miner_hotkey,owned.stage)
          IS NOT DISTINCT FROM (reservation.round_id,reservation.submission_id,
            reservation.miner_hotkey,reservation.stage)
        AND EXISTS(SELECT 1 FROM required k WHERE k.provider=reservation.provider
          AND k.fingerprint=reservation_fields.credential_fingerprint)
    ), outcomes AS MATERIALIZED (
      SELECT reservation.*,head.entry_kind AS head_kind,head.created_at AS head_at,
        head.amount_microusd AS head_amount,
        response_fields.call_succeeded AS succeeded,
        response_fields.credit_failure_proof AS credit_proof,
        jsonb_typeof(response_fields.provider_cost->'request_id') AS native_id_type,
        response_fields.provider_cost->>'request_id' AS native_request_id,
        head.entry_doc ?| ARRAY['judgment_cache_source_call_identity','judgment_cache_source_run_id'] AS head_cached,
        head.terminal_response ?| ARRAY['judgment_cache_source_call_identity','judgment_cache_source_run_id'] AS response_cached,
        CASE head.entry_kind WHEN 'uncertain' THEN head.entry_doc#>'{call,account_failure_evidence}'
          ELSE response_fields.account_failure_evidence END AS account_evidence,
        (response_fields.call_succeeded='false'::JSONB
          OR head.entry_doc#>'{call,call_succeeded}'='false'::JSONB) AS failed
      FROM scoped reservation
      JOIN LATERAL(SELECT terminal.* FROM public.lab_arena_ledger terminal
        WHERE terminal.run_id=reservation.run_id AND terminal.call_identity=reservation.call_identity
        ORDER BY terminal.entry_id DESC LIMIT 1) head ON TRUE
      CROSS JOIN LATERAL jsonb_to_record(CASE WHEN jsonb_typeof(head.terminal_response)='object' THEN head.terminal_response ELSE '{}'::JSONB END) AS response_fields(call_succeeded JSONB,credit_failure_proof JSONB,provider_cost JSONB,account_failure_evidence JSONB,judgment_cache_source_call_identity JSONB,judgment_cache_source_run_id JSONB)
      WHERE (head.round_id,head.submission_id,head.miner_hotkey,head.stage,head.provider,
        head.operation_id,head.funding_source) IS NOT DISTINCT FROM
        (reservation.round_id,reservation.submission_id,reservation.miner_hotkey,reservation.stage,
         reservation.provider,reservation.operation_id,reservation.funding_source)
$old_outcomes554$) THEN RAISE EXCEPTION 'funding throughput anchor differs'; END IF;
  d:=replace(d,$old_outcomes554$    ), scoped AS MATERIALIZED (
      SELECT reservation.run_id,reservation.round_id,reservation.submission_id,reservation.miner_hotkey,
        reservation.stage,reservation.call_identity,reservation.provider,reservation.operation_id,
        reservation.funding_source,reservation.created_at,owned.scoring_generation_floor AS run_floor,
        jsonb_build_object('base_call_identity',reservation_fields.base_call_identity,
          'provider_attempt',reservation_fields.provider_attempt,
          'action_sequence',reservation_fields.action_sequence) AS accounting_doc,
        reservation_fields.credential_fingerprint AS fingerprint,
        reservation.entry_doc ?| ARRAY['judgment_cache_source_call_identity','judgment_cache_source_run_id'] AS reservation_cached
      FROM public.lab_arena_ledger reservation
      JOIN public.lab_arena_runs owned ON owned.run_id=reservation.run_id
      CROSS JOIN LATERAL jsonb_to_record(CASE WHEN jsonb_typeof(reservation.entry_doc)='object' THEN reservation.entry_doc ELSE '{}'::JSONB END) AS reservation_fields(base_call_identity JSONB,provider_attempt JSONB,action_sequence JSONB,credential_fingerprint TEXT,judgment_cache_source_call_identity JSONB,judgment_cache_source_run_id JSONB)
      WHERE reservation.round_id=requested_run.round_id
        AND reservation.submission_id=requested_run.submission_id
        AND reservation.miner_hotkey=requested_run.miner_hotkey
        AND reservation.entry_kind='reservation' AND reservation.funding_source='miner_key'
        AND owned.kind IN ('execute','score')
        AND owned.scoring_generation_floor<=requested_run.scoring_generation_floor
        AND (owned.round_id,owned.submission_id,owned.miner_hotkey,owned.stage)
          IS NOT DISTINCT FROM (reservation.round_id,reservation.submission_id,
            reservation.miner_hotkey,reservation.stage)
        AND EXISTS(SELECT 1 FROM required k WHERE k.provider=reservation.provider
          AND k.fingerprint=reservation_fields.credential_fingerprint)
    ), outcomes AS MATERIALIZED (
      SELECT reservation.*,head.entry_kind AS head_kind,head.created_at AS head_at,
        head.amount_microusd AS head_amount,
        response_fields.call_succeeded AS succeeded,
        response_fields.credit_failure_proof AS credit_proof,
        jsonb_typeof(response_fields.provider_cost->'request_id') AS native_id_type,
        response_fields.provider_cost->>'request_id' AS native_request_id,
        head.entry_doc ?| ARRAY['judgment_cache_source_call_identity','judgment_cache_source_run_id'] AS head_cached,
        head.terminal_response ?| ARRAY['judgment_cache_source_call_identity','judgment_cache_source_run_id'] AS response_cached,
        CASE head.entry_kind WHEN 'uncertain' THEN head.entry_doc#>'{call,account_failure_evidence}'
          ELSE response_fields.account_failure_evidence END AS account_evidence,
        (response_fields.call_succeeded='false'::JSONB
          OR head.entry_doc#>'{call,call_succeeded}'='false'::JSONB) AS failed
      FROM scoped reservation
      JOIN LATERAL(SELECT terminal.* FROM public.lab_arena_ledger terminal
        WHERE terminal.run_id=reservation.run_id AND terminal.call_identity=reservation.call_identity
        ORDER BY terminal.entry_id DESC LIMIT 1) head ON TRUE
      CROSS JOIN LATERAL jsonb_to_record(CASE WHEN jsonb_typeof(head.terminal_response)='object' THEN head.terminal_response ELSE '{}'::JSONB END) AS response_fields(call_succeeded JSONB,credit_failure_proof JSONB,provider_cost JSONB,account_failure_evidence JSONB,judgment_cache_source_call_identity JSONB,judgment_cache_source_run_id JSONB)
      WHERE (head.round_id,head.submission_id,head.miner_hotkey,head.stage,head.provider,
        head.operation_id,head.funding_source) IS NOT DISTINCT FROM
        (reservation.round_id,reservation.submission_id,reservation.miner_hotkey,reservation.stage,
         reservation.provider,reservation.operation_id,reservation.funding_source)
$old_outcomes554$,$new_outcomes554$    ), outcomes AS MATERIALIZED (
      -- Only heads newer than this key's original denial can supply a new
      -- dispatch or strengthen its denial watermark. Historical duplicate
      -- native identities are checked separately below, without losing history.
      SELECT reservation.run_id,reservation.round_id,reservation.submission_id,reservation.miner_hotkey,
        reservation.stage,reservation.call_identity,reservation.provider,reservation.operation_id,
        reservation.funding_source,reservation.created_at,owned.scoring_generation_floor AS run_floor,
        jsonb_build_object('base_call_identity',reservation_fields.base_call_identity,
          'provider_attempt',reservation_fields.provider_attempt,
          'action_sequence',reservation_fields.action_sequence) AS accounting_doc,
        reservation_fields.credential_fingerprint AS fingerprint,
        reservation.entry_doc ?| ARRAY['judgment_cache_source_call_identity','judgment_cache_source_run_id'] AS reservation_cached,
        head.entry_kind AS head_kind,head.created_at AS head_at,
        head.amount_microusd AS head_amount,
        response_fields.call_succeeded AS succeeded,
        response_fields.credit_failure_proof AS credit_proof,
        jsonb_typeof(response_fields.provider_cost->'request_id') AS native_id_type,
        response_fields.provider_cost->>'request_id' AS native_request_id,
        head.entry_doc ?| ARRAY['judgment_cache_source_call_identity','judgment_cache_source_run_id'] AS head_cached,
        head.terminal_response ?| ARRAY['judgment_cache_source_call_identity','judgment_cache_source_run_id'] AS response_cached,
        CASE head.entry_kind WHEN 'uncertain' THEN head.entry_doc#>'{call,account_failure_evidence}'
          ELSE response_fields.account_failure_evidence END AS account_evidence,
        (response_fields.call_succeeded='false'::JSONB
          OR head.entry_doc#>'{call,call_succeeded}'='false'::JSONB) AS failed
      FROM required k
      JOIN public.lab_arena_ledger head ON head.provider=k.provider
        AND head.created_at>k.original_denial
        AND head.round_id=requested_run.round_id
        AND head.submission_id=requested_run.submission_id
        AND head.miner_hotkey=requested_run.miner_hotkey
        AND head.entry_kind IN ('settlement','uncertain')
        AND head.funding_source='miner_key'
      JOIN public.lab_arena_runs owned ON owned.run_id=head.run_id
      JOIN public.lab_arena_ledger reservation ON reservation.run_id=head.run_id
        AND reservation.call_identity=head.call_identity AND reservation.entry_kind='reservation'
      CROSS JOIN LATERAL jsonb_to_record(CASE WHEN jsonb_typeof(reservation.entry_doc)='object' THEN reservation.entry_doc ELSE '{}'::JSONB END) AS reservation_fields(base_call_identity JSONB,provider_attempt JSONB,action_sequence JSONB,credential_fingerprint TEXT,judgment_cache_source_call_identity JSONB,judgment_cache_source_run_id JSONB)
      CROSS JOIN LATERAL jsonb_to_record(CASE WHEN jsonb_typeof(head.terminal_response)='object' THEN head.terminal_response ELSE '{}'::JSONB END) AS response_fields(call_succeeded JSONB,credit_failure_proof JSONB,provider_cost JSONB,account_failure_evidence JSONB,judgment_cache_source_call_identity JSONB,judgment_cache_source_run_id JSONB)
      WHERE reservation.round_id=requested_run.round_id
        AND reservation.submission_id=requested_run.submission_id
        AND reservation.miner_hotkey=requested_run.miner_hotkey
        AND reservation.funding_source='miner_key'
        AND owned.kind IN ('execute','score')
        AND owned.scoring_generation_floor<=requested_run.scoring_generation_floor
        AND (owned.round_id,owned.submission_id,owned.miner_hotkey,owned.stage)
          IS NOT DISTINCT FROM (reservation.round_id,reservation.submission_id,
            reservation.miner_hotkey,reservation.stage)
        AND (head.round_id,head.submission_id,head.miner_hotkey,head.stage,head.provider,
          head.operation_id,head.funding_source) IS NOT DISTINCT FROM
          (reservation.round_id,reservation.submission_id,reservation.miner_hotkey,reservation.stage,
           reservation.provider,reservation.operation_id,reservation.funding_source)
        AND reservation_fields.credential_fingerprint=k.fingerprint
        AND NOT EXISTS(SELECT 1 FROM public.lab_arena_ledger newer
          WHERE newer.run_id=head.run_id AND newer.call_identity=head.call_identity
            AND newer.entry_id>head.entry_id)
$new_outcomes554$);
  IF (length(d)-length(replace(d,$old_duplicates554$        AND NOT EXISTS(SELECT 1 FROM outcomes duplicate
          WHERE duplicate.provider=success.provider
            AND duplicate.native_request_id=success.native_request_id
            AND (duplicate.run_id,duplicate.call_identity) IS DISTINCT FROM (success.run_id,success.call_identity)
            AND duplicate.head_kind='settlement' AND duplicate.head_amount>0)$old_duplicates554$,'')))<>length($old_duplicates554$        AND NOT EXISTS(SELECT 1 FROM outcomes duplicate
          WHERE duplicate.provider=success.provider
            AND duplicate.native_request_id=success.native_request_id
            AND (duplicate.run_id,duplicate.call_identity) IS DISTINCT FROM (success.run_id,success.call_identity)
            AND duplicate.head_kind='settlement' AND duplicate.head_amount>0)$old_duplicates554$) THEN RAISE EXCEPTION 'funding throughput anchor differs'; END IF;
  d:=replace(d,$old_duplicates554$        AND NOT EXISTS(SELECT 1 FROM outcomes duplicate
          WHERE duplicate.provider=success.provider
            AND duplicate.native_request_id=success.native_request_id
            AND (duplicate.run_id,duplicate.call_identity) IS DISTINCT FROM (success.run_id,success.call_identity)
            AND duplicate.head_kind='settlement' AND duplicate.head_amount>0)$old_duplicates554$,$new_duplicates554$        AND NOT EXISTS(
          SELECT 1 FROM public.lab_arena_ledger duplicate
          JOIN public.lab_arena_runs duplicate_owned ON duplicate_owned.run_id=duplicate.run_id
          JOIN public.lab_arena_ledger original ON original.run_id=duplicate.run_id
            AND original.call_identity=duplicate.call_identity AND original.entry_kind='reservation'
          CROSS JOIN LATERAL jsonb_to_record(CASE WHEN jsonb_typeof(original.entry_doc)='object' THEN original.entry_doc ELSE '{}'::JSONB END) AS original_fields(credential_fingerprint TEXT)
          WHERE duplicate.round_id=requested_run.round_id
            AND duplicate.submission_id=requested_run.submission_id
            AND duplicate.miner_hotkey=requested_run.miner_hotkey
            AND duplicate.provider=success.provider
            AND duplicate.entry_kind='settlement' AND duplicate.amount_microusd>0
            AND length(duplicate.terminal_response#>>'{provider_cost,request_id}') BETWEEN 1 AND 256
            AND duplicate.terminal_response#>>'{provider_cost,request_id}'=success.native_request_id
            AND (duplicate.run_id,duplicate.call_identity) IS DISTINCT FROM (success.run_id,success.call_identity)
            AND original.round_id=requested_run.round_id
            AND original.submission_id=requested_run.submission_id
            AND original.miner_hotkey=requested_run.miner_hotkey
            AND original.funding_source='miner_key'
            AND duplicate_owned.kind IN ('execute','score')
            AND duplicate_owned.scoring_generation_floor<=requested_run.scoring_generation_floor
            AND (duplicate_owned.round_id,duplicate_owned.submission_id,duplicate_owned.miner_hotkey,duplicate_owned.stage)
              IS NOT DISTINCT FROM (original.round_id,original.submission_id,original.miner_hotkey,original.stage)
            AND (duplicate.round_id,duplicate.submission_id,duplicate.miner_hotkey,duplicate.stage,duplicate.provider,
              duplicate.operation_id,duplicate.funding_source) IS NOT DISTINCT FROM
              (original.round_id,original.submission_id,original.miner_hotkey,original.stage,original.provider,
               original.operation_id,original.funding_source)
            AND EXISTS(SELECT 1 FROM required duplicate_key WHERE duplicate_key.provider=original.provider
              AND duplicate_key.fingerprint=original_fields.credential_fingerprint)
            AND NOT EXISTS(SELECT 1 FROM public.lab_arena_ledger newer
              WHERE newer.run_id=duplicate.run_id AND newer.call_identity=duplicate.call_identity
                AND newer.entry_id>duplicate.entry_id))$new_duplicates554$);
  EXECUTE d;
  IF (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=target) IS DISTINCT FROM metadata
    OR encode(extensions.digest(pg_get_functiondef(target),'sha256'),'hex')<>'a962ddae017d3e1efbbe05d296bd1b9c895523d4a5081b0a3fd2c78b9c7f3d73' THEN
    RAISE EXCEPTION 'funding throughput readback differs' USING ERRCODE='55000';
  END IF;
END;
$funding_throughput554$;
COMMIT;
