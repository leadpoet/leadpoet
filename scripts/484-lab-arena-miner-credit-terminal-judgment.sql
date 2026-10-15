-- A proved miner funding failure is not an unfinished infrastructure judgment.
-- Retain accepted outputs, NULL unmeasured scores, fees and owner retry gates.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='60s';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;

CREATE OR REPLACE FUNCTION public.lab_arena_miner_credit_failure_v1(p_run_id TEXT)
RETURNS BOOLEAN LANGUAGE plpgsql STABLE SECURITY DEFINER
SET search_path=pg_catalog,public AS $credit$
DECLARE r public.lab_arena_runs; previous public.lab_arena_runs;
  h RECORD; evidence JSONB; funding JSONB; proved BOOLEAN:=FALSE;
BEGIN
  SELECT * INTO r FROM public.lab_arena_runs WHERE run_id=p_run_id;
  IF NOT FOUND OR r.kind<>'score'
    OR NOT public.lab_arena_judge_continuation_enabled_v1(r.round_id)
    OR NOT EXISTS (SELECT 1 FROM public.lab_arena_submissions s
      WHERE s.submission_id=r.submission_id AND s.round_id=r.round_id
        AND s.miner_hotkey=r.miner_hotkey AND s.status='frozen' AND NOT s.is_king)
    OR NOT EXISTS (SELECT 1 FROM public.lab_arena_runs source
      WHERE source.run_id=r.scored_run_id AND source.kind='execute'
        AND source.status='accepted' AND source.terminal_cause='accepted'
        AND source.output_ref IS NOT NULL
        AND r.assignment_id=source.assignment_id||':score'
        AND r.run_id=r.assignment_id||':'||r.attempt::TEXT
        AND (source.round_id,source.submission_id,source.miner_hotkey,source.stage,source.icp_position)
          IS NOT DISTINCT FROM (r.round_id,r.submission_id,r.miner_hotkey,r.stage,r.icp_position))
    OR NOT EXISTS (SELECT 1 FROM public.lab_arena_rounds round_row,
      LATERAL jsonb_array_elements(round_row.participants) participant
      WHERE round_row.round_id=r.round_id AND participant->>'submission_id'=r.submission_id
        AND participant->>'miner_hotkey'=r.miner_hotkey
        AND participant->'is_king'='false'::JSONB
        AND r.judgment_scope_doc->>'round_id'=r.round_id
        AND r.judgment_scope_doc->>'scoring_input_hash'=r.judgment_input_hash
        AND r.judgment_scope_doc->>'cache_key'=r.judgment_cache_key
        AND r.judgment_scope_doc->>'scorer_image_digest'=round_row.configuration_doc->>'scorer_image_digest'
        AND r.judgment_scope_doc->>'scorer_image_reference'=round_row.configuration_doc->>'scorer_image_reference')
    THEN RETURN FALSE; END IF;
  -- Existing accepted cache evidence keeps its normal recovery path. This
  -- predicate never substitutes funding evidence for a verifiable judgment.
  IF r.judgment_cache_key IS NOT NULL AND EXISTS (
    SELECT 1 FROM public.lab_arena_judgment_cache cache
      WHERE cache.cache_key=r.judgment_cache_key) THEN RETURN FALSE; END IF;
  -- Already queued successors have no new provider evidence. Only an untouched
  -- pending/closed successor can inherit the preceding funding failure. A
  -- dispatched attempt with a new infrastructure error never inherits it.
  IF r.status='pending' OR (r.status='failed' AND r.terminal_cause='stage_closed'
      AND r.terminal_doc->>'previous_status'='pending') THEN
    IF r.runner_hotkey IS NOT NULL OR EXISTS (SELECT 1 FROM public.lab_arena_ledger l
      WHERE l.run_id=r.run_id) THEN RETURN FALSE; END IF;
    SELECT * INTO previous FROM public.lab_arena_runs p
      WHERE p.assignment_id=r.assignment_id AND p.attempt<r.attempt
        AND p.status='failed' AND p.terminal_cause='credential_error'
      ORDER BY p.attempt DESC LIMIT 1;
    IF NOT FOUND OR (previous.scored_run_id,previous.miner_hotkey,
        previous.judgment_cache_key,previous.judgment_input_hash,previous.judgment_scope_doc)
      IS DISTINCT FROM (r.scored_run_id,r.miner_hotkey,r.judgment_cache_key,
        r.judgment_input_hash,r.judgment_scope_doc)
      OR EXISTS (SELECT 1 FROM public.lab_arena_runs intervening
        WHERE intervening.assignment_id=r.assignment_id
          AND intervening.attempt>previous.attempt AND intervening.attempt<r.attempt
          AND (intervening.runner_hotkey IS NOT NULL OR intervening.status<>'failed'
            OR intervening.terminal_cause<>'stage_closed'
            OR intervening.terminal_doc->>'previous_status' IS DISTINCT FROM 'pending'
            OR EXISTS (SELECT 1 FROM public.lab_arena_ledger dispatched
              WHERE dispatched.run_id=intervening.run_id)))
      THEN RETURN FALSE; END IF;
    RETURN public.lab_arena_miner_credit_failure_v1(previous.run_id);
  END IF;
  IF r.status<>'failed' OR r.terminal_cause<>'credential_error'
    OR r.result_doc->>'terminal_status' IS DISTINCT FROM 'credential_error'
    THEN RETURN FALSE; END IF;
  FOR h IN SELECT head.*,reservation.entry_doc AS reservation_doc
    FROM (SELECT DISTINCT ON (l.call_identity) l.*
      FROM public.lab_arena_ledger l WHERE l.run_id=r.run_id
      ORDER BY l.call_identity,l.entry_id DESC) head
    LEFT JOIN LATERAL (SELECT entry_doc FROM public.lab_arena_ledger original
      WHERE original.run_id=r.run_id AND original.call_identity=head.call_identity
        AND original.entry_kind='reservation' LIMIT 1) reservation ON TRUE
  LOOP
    IF h.call_identity IS NULL OR
      (h.round_id,h.submission_id,h.miner_hotkey,h.stage) IS DISTINCT FROM
      (r.round_id,r.submission_id,r.miner_hotkey,r.stage)
      THEN RETURN FALSE; END IF;
    IF h.entry_kind IN ('reservation','dispatch')
      OR (h.entry_kind='uncertain' AND
        h.entry_doc#>'{call,call_succeeded}' IS DISTINCT FROM 'false'::JSONB)
      THEN RETURN FALSE; END IF;
    IF h.entry_kind='settlement' AND h.terminal_response->'call_succeeded'='true'::JSONB
      THEN CONTINUE; END IF;
    -- Only a successful retry of this exact bound request can supersede an
    -- earlier failure. A different target on the same operation proves nothing.
    IF h.reservation_doc->>'provider_attempt' IN ('1','2','3','4')
      AND h.reservation_doc->>'base_call_identity' ~ '^sha256:[0-9a-f]{64}$'
      AND EXISTS (SELECT 1 FROM public.lab_arena_ledger later
        JOIN LATERAL (SELECT terminal.entry_kind,terminal.terminal_response
          FROM public.lab_arena_ledger terminal
          WHERE terminal.run_id=r.run_id AND terminal.call_identity=later.call_identity
          ORDER BY terminal.entry_id DESC LIMIT 1) outcome ON TRUE
        WHERE later.run_id=r.run_id AND later.entry_kind='reservation'
          AND later.provider=h.provider AND later.operation_id=h.operation_id
          AND later.entry_doc->'base_call_identity'=h.reservation_doc->'base_call_identity'
          AND later.entry_doc->'action_sequence'=h.reservation_doc->'action_sequence'
          AND later.entry_doc->>'provider_attempt' IN ('1','2','3','4')
          AND CASE WHEN later.entry_doc->>'provider_attempt' IN ('1','2','3','4')
            THEN (later.entry_doc->>'provider_attempt')::INTEGER END
            > CASE WHEN h.reservation_doc->>'provider_attempt' IN ('1','2','3','4')
              THEN (h.reservation_doc->>'provider_attempt')::INTEGER END
          AND outcome.entry_kind='settlement'
          AND outcome.terminal_response->'call_succeeded'='true'::JSONB
      ) THEN CONTINUE; END IF;
    IF h.entry_kind='settlement' AND h.terminal_response->'call_succeeded'='false'::JSONB
      AND h.terminal_response->>'status' IN ('404','422') THEN CONTINUE; END IF;
    IF h.funding_source IS DISTINCT FROM 'miner_key' OR h.provider NOT IN ('openrouter','scrapingdog','deepline')
      THEN RETURN FALSE; END IF;
    funding:=public.lab_arena_provider_funding(r.run_id,h.provider);
    IF funding->>'funding_source' IS DISTINCT FROM 'miner_key'
      OR funding->>'credential_submission_id' IS DISTINCT FROM r.submission_id
      OR funding->>'credential_miner_hotkey' IS DISTINCT FROM r.miner_hotkey
      THEN RETURN FALSE; END IF;
    IF h.entry_kind='settlement' AND h.amount_microusd=0
      AND h.terminal_response->'call_succeeded'='false'::JSONB
      AND h.terminal_response->'credit_failure_proof'=jsonb_build_object(
        'schema_version','leadpoet.lab_arena.credit_failure_proof.v1',
        'provider',h.provider,'reason','out_of_credit','provider_status',402,'actual_microusd',0)
      THEN proved:=TRUE; CONTINUE; END IF;
    evidence:=CASE h.entry_kind WHEN 'uncertain'
      THEN h.entry_doc#>'{call,account_failure_evidence}'
      WHEN 'settlement' THEN h.terminal_response->'account_failure_evidence' END;
    IF h.provider='deepline' AND evidence->>'reason'='out_of_credit'
      AND evidence->'provider_status'='402'::JSONB
      AND ((h.entry_kind='uncertain' AND h.entry_doc#>'{call,call_succeeded}'='false'::JSONB)
        OR (h.entry_kind='settlement' AND h.terminal_response->'call_succeeded'='false'::JSONB))
      AND EXISTS (SELECT 1 FROM public.lab_arena_ledger reservation
        WHERE reservation.run_id=r.run_id AND reservation.call_identity=h.call_identity
          AND reservation.entry_kind='reservation' AND reservation.provider=h.provider
          AND reservation.funding_source='miner_key'
          AND (reservation.round_id,reservation.submission_id,reservation.miner_hotkey,reservation.stage)
            IS NOT DISTINCT FROM (r.round_id,r.submission_id,r.miner_hotkey,r.stage)
          AND public.lab_arena__champion_account_failure_matches(reservation.entry_doc,evidence))
      THEN proved:=TRUE; CONTINUE; END IF;
    -- No HTTP status, diagnostic text, unbound receipt or unrelated failure
    -- grants terminal funding authority.
    RETURN FALSE;
  END LOOP;
  RETURN proved;
END;
$credit$;
ALTER FUNCTION public.lab_arena_miner_credit_failure_v1(TEXT) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_miner_credit_failure_v1(TEXT)
  FROM PUBLIC,anon,authenticated,service_role;
GRANT EXECUTE ON FUNCTION public.lab_arena_miner_credit_failure_v1(TEXT) TO lab_arena_service;

CREATE OR REPLACE FUNCTION public.lab_arena__miner_credit_judgment_terminal_v1(p_source_id TEXT)
RETURNS BOOLEAN LANGUAGE sql STABLE SECURITY DEFINER SET search_path=pg_catalog,public AS $terminal$
  SELECT COALESCE((SELECT effective.status='failed'
      AND public.lab_arena_miner_credit_failure_v1(effective.run_id)
      AND clock_timestamp()>=(round_row.configuration_doc#>>ARRAY['schedule',
        CASE source.stage WHEN 1 THEN 'stage_1_scoring_close' ELSE 'final_scoring_close' END])::TIMESTAMPTZ
      AND NOT EXISTS (SELECT 1 FROM public.lab_arena_runs active
        WHERE active.scored_run_id=source.run_id AND active.kind='score'
          AND active.round_id=source.round_id AND active.submission_id=source.submission_id
          AND active.stage=source.stage AND active.icp_position=source.icp_position
          AND active.status IN ('pending','leased','submitted'))
    FROM public.lab_arena_runs source JOIN public.lab_arena_rounds round_row USING(round_id)
    JOIN LATERAL (SELECT judge.* FROM public.lab_arena_runs judge
      WHERE judge.scored_run_id=source.run_id AND judge.kind='score'
        AND judge.round_id=source.round_id AND judge.submission_id=source.submission_id
        AND judge.stage=source.stage AND judge.icp_position=source.icp_position
      ORDER BY (judge.status='accepted') DESC,judge.stage_generation DESC,judge.attempt DESC,judge.run_id DESC
      LIMIT 1) effective ON TRUE WHERE source.run_id=p_source_id),FALSE);
$terminal$;
ALTER FUNCTION public.lab_arena__miner_credit_judgment_terminal_v1(TEXT) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena__miner_credit_judgment_terminal_v1(TEXT)
  FROM PUBLIC,anon,authenticated,service_role;
GRANT EXECUTE ON FUNCTION public.lab_arena__miner_credit_judgment_terminal_v1(TEXT) TO lab_arena_service;

DO $patch$
DECLARE p RECORD; definition TEXT; owner_name TEXT; secure BOOLEAN;
BEGIN
  FOR p IN SELECT * FROM (VALUES
    ('public.lab_arena__judge_continuation_v1(public.lab_arena_rounds,public.lab_arena_runs)',
      $old$  IF p_run.kind<>'score'$old$,
      $new$  IF public.lab_arena_miner_credit_failure_v1(p_run.run_id) THEN RETURN FALSE; END IF;
  IF p_run.kind<>'score'$new$),
    ('public.lab_arena_claim_assignment(text,text,integer,integer,text[],text,text,text,integer)',
      $old$AND runs.status = 'pending'$old$,
      $new$AND runs.status = 'pending'
    AND (NOT public.lab_arena_miner_credit_failure_v1(runs.run_id)
      OR (runs.attempt <= (v_round.configuration_doc->>'max_attempts_per_assignment')::INTEGER
        AND clock_timestamp() < (v_round.configuration_doc#>>ARRAY['schedule',
          CASE runs.stage WHEN 1 THEN 'stage_1_scoring_close' ELSE 'final_scoring_close' END])::TIMESTAMPTZ))$new$),
    ('public.lab_arena_expire_leases(text)',
      $old$  -- Materialize successors for failures that predate this migration or$old$,
      $new$  -- Close only untouched credit successors which no longer have ordinary
  -- admission authority. Existing leases and every prior failure stay intact.
  UPDATE public.lab_arena_runs pending SET status='failed',terminal_cause='stage_closed',
    terminal_doc=jsonb_build_object('closed_at',clock_timestamp(),'previous_status','pending')
  WHERE pending.round_id=p_round_id AND pending.kind='score' AND pending.status='pending'
    AND public.lab_arena_miner_credit_failure_v1(pending.run_id)
    AND (pending.attempt>(v_round.configuration_doc->>'max_attempts_per_assignment')::INTEGER
      OR clock_timestamp()>=(v_round.configuration_doc#>>ARRAY['schedule',
        CASE pending.stage WHEN 1 THEN 'stage_1_scoring_close' ELSE 'final_scoring_close' END])::TIMESTAMPTZ);
  -- Materialize successors for failures that predate this migration or$new$),
    ('public.lab_arena_close_scoring(text,smallint)',
      $old$  v_generation := v_round.stage_generation + 1;$old$,
      $new$  -- Keep the original owner-recovery window. Never close an active lease
  -- to manufacture a terminal funding outcome.
  IF public.lab_arena_judge_continuation_enabled_v1(p_round_id) AND (
    EXISTS (SELECT 1 FROM public.lab_arena_runs active WHERE active.round_id=p_round_id
      AND active.kind='score' AND active.status IN ('leased','submitted')
      AND (active.stage=p_stage OR (p_stage=2
        AND public.lab_arena__baseline_recovery_run_v1(active))))
    OR (clock_timestamp() < (v_round.configuration_doc#>>ARRAY['schedule',
        CASE p_stage WHEN 1 THEN 'stage_1_scoring_close' ELSE 'final_scoring_close' END])::TIMESTAMPTZ
      AND EXISTS (SELECT 1 FROM public.lab_arena_runs credit WHERE credit.round_id=p_round_id
        AND credit.stage=p_stage AND credit.kind='score'
        AND public.lab_arena_miner_credit_failure_v1(credit.run_id)
        AND NOT EXISTS (SELECT 1 FROM public.lab_arena_runs accepted
          WHERE accepted.scored_run_id=credit.scored_run_id AND accepted.kind='score'
            AND accepted.status='accepted')))) THEN
    RETURN jsonb_build_object('status','draining','round_status',v_round.status,
      'stage_generation',v_round.stage_generation);
  END IF;
  v_generation := v_round.stage_generation + 1;$new$),
    ('public.lab_arena__all_judgments_ready_v1(text)',
      $old$AND source.stage IN (1,2) AND source.status='accepted'$old$,
      $new$AND source.stage IN (1,2) AND source.status='accepted'
      AND NOT public.lab_arena__miner_credit_judgment_terminal_v1(source.run_id)$new$),
    ('public.lab_arena__publication_execution_incomplete_v1(text,text)',
      $old$    IF FOUND THEN
      v_close_key := CASE v_execution.stage$old$,
      $new$    IF FOUND THEN
      IF public.lab_arena__miner_credit_judgment_terminal_v1(v_execution.run_id) THEN
        CONTINUE;
      END IF;
      v_close_key := CASE v_execution.stage$new$)
  ) AS patches(signature,old_text,new_text) LOOP
    SELECT pg_get_functiondef(proc.oid),pg_get_userbyid(proc.proowner),proc.prosecdef
      INTO definition,owner_name,secure FROM pg_proc proc WHERE proc.oid=to_regprocedure(p.signature);
    IF definition IS NULL OR owner_name<>'lab_arena_owner' OR secure IS DISTINCT FROM TRUE THEN
      RAISE EXCEPTION 'miner credit function authority differs: %',p.signature; END IF;
    IF strpos(definition,p.new_text)>0 THEN CONTINUE; END IF;
    IF (length(definition)-length(replace(definition,p.old_text,'')))/length(p.old_text)<>1 THEN
      RAISE EXCEPTION 'miner credit anchor differs: %',p.signature; END IF;
    EXECUTE replace(definition,p.old_text,p.new_text);
  END LOOP;
END;
$patch$;
REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;
NOTIFY pgrst,'reload schema';
COMMIT;
