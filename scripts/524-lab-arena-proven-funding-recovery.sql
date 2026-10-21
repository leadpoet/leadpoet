-- Retire a proved funding stop only after a newer paid, same-frozen-key dispatch.
-- No judgment, ledger entry, credential, provider request or source output is changed.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';
DO $funded_recovery$
DECLARE d TEXT; metadata JSONB;
 target OID:='public.lab_arena_miner_credit_failure_v1(text)'::regprocedure;
BEGIN
 SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc' INTO STRICT d,metadata
 FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang WHERE p.oid=target
  AND pg_get_userbyid(p.proowner)='lab_arena_owner'
  AND p.proacl::TEXT='{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}'
  AND p.prosecdef AND p.provolatile='s'
  AND p.proconfig=ARRAY['search_path=pg_catalog, public']::TEXT[] AND l.lanname='plpgsql';
 IF encode(extensions.digest(d,'sha256'),'hex')='41fd019031415dd9c162500e233d58fadfaf54288c05676fc20e7f9390596b62' THEN RETURN; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')<>'7ae6b0fd1e77c2dee07f677bb50e0d014a16d1b62040dc5791f2df7b65f1d20c' THEN RAISE EXCEPTION 'Funded recovery preimage differs' USING ERRCODE='55000'; END IF;
 IF (length(d)-length(replace(d,$old_funded524_0$  h RECORD; evidence JSONB; funding JSONB; proved BOOLEAN:=FALSE;$old_funded524_0$,'')))<>length($old_funded524_0$  h RECORD; evidence JSONB; funding JSONB; proved BOOLEAN:=FALSE;$old_funded524_0$) THEN RAISE EXCEPTION 'Funded recovery anchor differs' USING ERRCODE='55000'; END IF;
 d:=replace(d,$old_funded524_0$  h RECORD; evidence JSONB; funding JSONB; proved BOOLEAN:=FALSE;$old_funded524_0$,$new_funded524_0$  h RECORD; evidence JSONB; funding JSONB; proved BOOLEAN:=FALSE;
  requested_run public.lab_arena_runs; credit_keys JSONB:='[]'::JSONB;$new_funded524_0$);
 IF (length(d)-length(replace(d,$old_funded524_1$  SELECT * INTO r FROM public.lab_arena_runs WHERE run_id=p_run_id;$old_funded524_1$,'')))<>length($old_funded524_1$  SELECT * INTO r FROM public.lab_arena_runs WHERE run_id=p_run_id;$old_funded524_1$) THEN RAISE EXCEPTION 'Funded recovery anchor differs' USING ERRCODE='55000'; END IF;
 d:=replace(d,$old_funded524_1$  SELECT * INTO r FROM public.lab_arena_runs WHERE run_id=p_run_id;$old_funded524_1$,$new_funded524_1$  SELECT * INTO r FROM public.lab_arena_runs WHERE run_id=p_run_id;
  requested_run:=r;$new_funded524_1$);
 IF (length(d)-length(replace(d,$old_funded524_2$    RETURN public.lab_arena_miner_credit_failure_v1(previous.run_id);$old_funded524_2$,'')))<>length($old_funded524_2$    RETURN public.lab_arena_miner_credit_failure_v1(previous.run_id);$old_funded524_2$) THEN RAISE EXCEPTION 'Funded recovery anchor differs' USING ERRCODE='55000'; END IF;
 d:=replace(d,$old_funded524_2$    RETURN public.lab_arena_miner_credit_failure_v1(previous.run_id);$old_funded524_2$,$new_funded524_2$    IF NOT public.lab_arena_miner_credit_failure_v1(previous.run_id) THEN RETURN FALSE; END IF;
    -- Keep the original native proof, but evaluate any funding recovery for
    -- the current, already-authorized inherited root rather than its old floor.
    r:=previous;$new_funded524_2$);
 IF (length(d)-length(replace(d,$old_funded524_3$      THEN proved:=TRUE; CONTINUE; END IF;
    evidence:=$old_funded524_3$,'')))<>length($old_funded524_3$      THEN proved:=TRUE; CONTINUE; END IF;
    evidence:=$old_funded524_3$) THEN RAISE EXCEPTION 'Funded recovery anchor differs' USING ERRCODE='55000'; END IF;
 d:=replace(d,$old_funded524_3$      THEN proved:=TRUE; CONTINUE; END IF;
    evidence:=$old_funded524_3$,$new_funded524_3$      THEN credit_keys:=credit_keys||jsonb_build_array(jsonb_build_object(
        'provider',h.provider,'fingerprint',h.reservation_doc->>'credential_fingerprint',
        'denied_at',h.created_at)); proved:=TRUE; CONTINUE; END IF;
    evidence:=$new_funded524_3$);
 IF (length(d)-length(replace(d,$old_funded524_4$      THEN proved:=TRUE; CONTINUE; END IF;
    -- No HTTP$old_funded524_4$,'')))<>length($old_funded524_4$      THEN proved:=TRUE; CONTINUE; END IF;
    -- No HTTP$old_funded524_4$) THEN RAISE EXCEPTION 'Funded recovery anchor differs' USING ERRCODE='55000'; END IF;
 d:=replace(d,$old_funded524_4$      THEN proved:=TRUE; CONTINUE; END IF;
    -- No HTTP$old_funded524_4$,$new_funded524_4$      THEN credit_keys:=credit_keys||jsonb_build_array(jsonb_build_object(
        'provider',h.provider,'fingerprint',h.reservation_doc->>'credential_fingerprint',
        'denied_at',h.created_at)); proved:=TRUE; CONTINUE; END IF;
    -- No HTTP$new_funded524_4$);
 IF (length(d)-length(replace(d,$old_funded524_5$  RETURN proved;
$old_funded524_5$,'')))<>length($old_funded524_5$  RETURN proved;
$old_funded524_5$) THEN RAISE EXCEPTION 'Funded recovery anchor differs' USING ERRCODE='55000'; END IF;
 d:=replace(d,$old_funded524_5$  RETURN proved;
$old_funded524_5$,$new_funded524_5$  IF NOT proved THEN RETURN FALSE; END IF;
  -- A paid success proves only funding availability for one normal retry.
  -- It is never source evidence, a verdict, or permission to change a payer.
  IF NOT EXISTS(SELECT 1 FROM public.lab_arena_rounds current_round
    WHERE current_round.round_id=requested_run.round_id
      AND current_round.scoring_generation_floor=requested_run.scoring_generation_floor
      AND current_round.status NOT IN ('published','cancelled')
      AND current_round.cancel_reason IS NULL) THEN RETURN TRUE; END IF;
  RETURN NOT COALESCE((
    WITH required AS MATERIALIZED (
      SELECT key->>'provider' AS provider,key->>'fingerprint' AS fingerprint,
        max((key->>'denied_at')::TIMESTAMPTZ) AS original_denial
      FROM jsonb_array_elements(credit_keys) key
      GROUP BY key->>'provider',key->>'fingerprint'
    ), scoped AS MATERIALIZED (
      SELECT reservation.*,owned.scoring_generation_floor AS run_floor
      FROM public.lab_arena_ledger reservation
      JOIN public.lab_arena_runs owned ON owned.run_id=reservation.run_id
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
          AND k.fingerprint=reservation.entry_doc->>'credential_fingerprint')
    ), outcomes AS MATERIALIZED (
      SELECT reservation.*,head.entry_kind AS head_kind,head.created_at AS head_at,
        head.amount_microusd AS head_amount,head.terminal_response AS response,
        head.entry_doc AS head_doc,
        CASE head.entry_kind WHEN 'uncertain' THEN head.entry_doc#>'{call,account_failure_evidence}'
          ELSE head.terminal_response->'account_failure_evidence' END AS account_evidence,
        (head.terminal_response->'call_succeeded'='false'::JSONB
          OR head.entry_doc#>'{call,call_succeeded}'='false'::JSONB) AS failed
      FROM scoped reservation
      JOIN LATERAL(SELECT terminal.* FROM public.lab_arena_ledger terminal
        WHERE terminal.run_id=reservation.run_id AND terminal.call_identity=reservation.call_identity
        ORDER BY terminal.entry_id DESC LIMIT 1) head ON TRUE
      WHERE (head.round_id,head.submission_id,head.miner_hotkey,head.stage,head.provider,
        head.operation_id,head.funding_source) IS NOT DISTINCT FROM
        (reservation.round_id,reservation.submission_id,reservation.miner_hotkey,reservation.stage,
         reservation.provider,reservation.operation_id,reservation.funding_source)
    ), denials AS MATERIALIZED (
      SELECT o.*,o.response->'credit_failure_proof'=jsonb_build_object(
          'schema_version','leadpoet.lab_arena.credit_failure_proof.v1','provider',o.provider,
          'reason','out_of_credit','provider_status',402,'actual_microusd',0)
          AND o.head_kind='settlement' AND o.head_amount=0 AS exact_credit_proof,
        public.lab_arena__champion_account_failure_matches(o.entry_doc,o.account_evidence)
          AS bound_account_failure
      FROM outcomes o WHERE o.failed AND o.head_kind IN ('settlement','uncertain')
    )
    SELECT bool_and(k.fingerprint~'^sha256:[0-9a-f]{64}$' AND EXISTS(
      SELECT 1 FROM outcomes success
      JOIN LATERAL(SELECT count(*) AS calls,min(dispatched.created_at) AS dispatched_at
        FROM public.lab_arena_ledger dispatched
        WHERE dispatched.run_id=success.run_id AND dispatched.call_identity=success.call_identity
          AND dispatched.entry_kind='dispatch'
          AND (dispatched.round_id,dispatched.submission_id,dispatched.miner_hotkey,dispatched.stage,
            dispatched.provider,dispatched.operation_id,dispatched.funding_source) IS NOT DISTINCT FROM
            (success.round_id,success.submission_id,success.miner_hotkey,success.stage,
             success.provider,success.operation_id,success.funding_source)) dispatch ON TRUE
      WHERE success.provider=k.provider AND success.entry_doc->>'credential_fingerprint'=k.fingerprint
        AND success.run_floor=requested_run.scoring_generation_floor
        AND success.head_kind='settlement' AND success.head_amount>0
        AND success.response->'call_succeeded'='true'::JSONB
        AND jsonb_typeof(success.response#>'{provider_cost,request_id}')='string'
        AND length(success.response#>>'{provider_cost,request_id}') BETWEEN 1 AND 256
        AND (success.response#>>'{provider_cost,request_id}') ~ '[^[:space:]]'
        AND NOT success.entry_doc ?| ARRAY['judgment_cache_source_call_identity','judgment_cache_source_run_id']
        AND NOT success.head_doc ?| ARRAY['judgment_cache_source_call_identity','judgment_cache_source_run_id']
        AND NOT success.response ?| ARRAY['judgment_cache_source_call_identity','judgment_cache_source_run_id']
        AND dispatch.calls=1 AND success.created_at<=dispatch.dispatched_at
        AND dispatch.dispatched_at<=success.head_at
        -- Dispatch time, not late billing time, must be newer than every
        -- same-key native credit denial. A new denial makes old proof unusable.
        AND dispatch.dispatched_at>GREATEST(k.original_denial,COALESCE((
          SELECT max(d.head_at) FROM denials d WHERE d.provider=k.provider
            AND d.entry_doc->>'credential_fingerprint'=k.fingerprint
            AND (d.exact_credit_proof OR (d.bound_account_failure
              AND d.account_evidence->>'reason'='out_of_credit'
              AND d.account_evidence->'provider_status'='402'::JSONB))
        ),k.original_denial))
        -- Funding success on another endpoint cannot retire an authenticated
        -- invalid-key/access/quota failure that followed the exhausted credit.
        AND NOT EXISTS(SELECT 1 FROM denials auth WHERE auth.provider=k.provider
          AND auth.entry_doc->>'credential_fingerprint'=k.fingerprint
          AND auth.head_at>k.original_denial AND auth.bound_account_failure
          AND auth.account_evidence->>'reason' IS DISTINCT FROM 'out_of_credit')
        AND NOT EXISTS(SELECT 1 FROM outcomes duplicate
          WHERE duplicate.provider=success.provider
            AND duplicate.response#>>'{provider_cost,request_id}'=success.response#>>'{provider_cost,request_id}'
            AND (duplicate.run_id,duplicate.call_identity) IS DISTINCT FROM (success.run_id,success.call_identity)
            AND duplicate.head_kind='settlement' AND duplicate.head_amount>0)
    )) FROM required k
  ),FALSE);
$new_funded524_5$);
 EXECUTE d;
 IF (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=target) IS DISTINCT FROM metadata OR encode(extensions.digest(pg_get_functiondef(target),'sha256'),'hex')<>'41fd019031415dd9c162500e233d58fadfaf54288c05676fc20e7f9390596b62' THEN RAISE EXCEPTION 'Funded recovery readback differs' USING ERRCODE='55000'; END IF;
END;
$funded_recovery$;
COMMIT;
