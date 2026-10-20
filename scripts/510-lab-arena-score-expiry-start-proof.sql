-- Infer a failed score runner only after authenticated lease handling began.
-- Preserve explicit host errors, ordinary expiry, TTL, retries and all other gates.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';
DO $score_expiry_start_proof510$
DECLARE
  target OID:=to_regprocedure('public.lab_arena_claim_assignment(text,text,integer,integer,text[],text,text,text,integer)');
  definition TEXT;
  updated TEXT;
  metadata JSONB;
  identity JSONB;
  patch RECORD;
BEGIN
  SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc',
    jsonb_build_array(pg_get_userbyid(p.proowner),p.proacl::TEXT,p.prosecdef,
      p.provolatile,p.proconfig,l.lanname)
  INTO definition,metadata,identity FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang
  WHERE p.oid=target;
  IF definition IS NULL OR identity IS DISTINCT FROM
    '["lab_arena_owner","{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}",true,"v",["search_path=pg_catalog, public"],"plpgsql"]'::JSONB THEN
    RAISE EXCEPTION 'Score expiry start proof security shape differs' USING ERRCODE='55000';
  END IF;
  IF encode(extensions.digest(definition,'sha256'),'hex')='ccf306a3d6fddbe92f22f90cc2c0c401847eb4d80eb3a160cf06fa6ba6aa9987' THEN RETURN; END IF;
  IF encode(extensions.digest(definition,'sha256'),'hex')<>'6d9bdb660a476838c42bee9dd9de5588be9f19f41ee32ea167b1f47bcafbf2ef' THEN
    RAISE EXCEPTION 'Score expiry start proof preimage differs' USING ERRCODE='55000';
  END IF;
  updated:=definition;
  FOR patch IN SELECT * FROM (VALUES
    ($cooldown_old$         AND NOT EXISTS (
           SELECT 1 FROM public.lab_arena_ledger AS prior_cost
           WHERE prior_cost.run_id=expired_score.run_id
         )
$cooldown_old$,$cooldown_new$         AND NOT EXISTS (
           SELECT 1 FROM public.lab_arena_ledger AS prior_cost
           WHERE prior_cost.run_id=expired_score.run_id
         )
         -- A committed claim alone is not evidence that this runner began it.
         AND expired_score.lease_generation>0
         AND expired_score.lease_token_hash IS NOT NULL
         AND expired_score.claim_request_id IS NOT NULL
         AND expired_score.claim_response->>'status'='leased'
         AND expired_score.claim_response->>'run_id'=expired_score.run_id
         AND expired_score.claim_response->'lease_generation'=to_jsonb(expired_score.lease_generation)
         AND EXISTS (
           SELECT 1 FROM public.lab_arena_trajectory_events AS started
           WHERE started.run_id=expired_score.run_id
             AND (started.runner_hotkey,started.round_id,started.submission_id,
               started.miner_hotkey,started.assignment_id,started.stage,
               started.icp_position,started.attempt,started.run_kind)
               IS NOT DISTINCT FROM
               (expired_score.runner_hotkey,expired_score.round_id,expired_score.submission_id,
                expired_score.miner_hotkey,expired_score.assignment_id,expired_score.stage,
                expired_score.icp_position,expired_score.attempt,expired_score.kind)
             AND started.event_kind='runtime.started'
             AND started.content->>'status'='starting'
             AND started.content->'lease_generation'=to_jsonb(expired_score.lease_generation)
             AND started.created_at BETWEEN
               (expired_score.claim_response->>'lease_expires_at')::TIMESTAMPTZ
                 -make_interval(secs=>(v_round.configuration_doc->>'lease_ttl_seconds')::INTEGER)
               AND (expired_score.claim_response->>'lease_expires_at')::TIMESTAMPTZ
             AND started.occurred_at BETWEEN date_trunc('second',
               (expired_score.claim_response->>'lease_expires_at')::TIMESTAMPTZ
                 -make_interval(secs=>(v_round.configuration_doc->>'lease_ttl_seconds')::INTEGER))
               AND (expired_score.claim_response->>'lease_expires_at')::TIMESTAMPTZ
         )
$cooldown_new$),
    ($handoff_old$          AND NOT EXISTS (
            SELECT 1 FROM public.lab_arena_ledger AS prior_cost
            WHERE prior_cost.run_id=prior_score.run_id
          )
$handoff_old$,$handoff_new$          AND NOT EXISTS (
            SELECT 1 FROM public.lab_arena_ledger AS prior_cost
            WHERE prior_cost.run_id=prior_score.run_id
          )
          -- A committed claim alone is not evidence that this runner began it.
          AND prior_score.lease_generation>0
          AND prior_score.lease_token_hash IS NOT NULL
          AND prior_score.claim_request_id IS NOT NULL
          AND prior_score.claim_response->>'status'='leased'
          AND prior_score.claim_response->>'run_id'=prior_score.run_id
          AND prior_score.claim_response->'lease_generation'=to_jsonb(prior_score.lease_generation)
          AND EXISTS (
            SELECT 1 FROM public.lab_arena_trajectory_events AS started
            WHERE started.run_id=prior_score.run_id
              AND (started.runner_hotkey,started.round_id,started.submission_id,
                started.miner_hotkey,started.assignment_id,started.stage,
                started.icp_position,started.attempt,started.run_kind)
                IS NOT DISTINCT FROM
                (prior_score.runner_hotkey,prior_score.round_id,prior_score.submission_id,
                 prior_score.miner_hotkey,prior_score.assignment_id,prior_score.stage,
                 prior_score.icp_position,prior_score.attempt,prior_score.kind)
              AND started.event_kind='runtime.started'
              AND started.content->>'status'='starting'
              AND started.content->'lease_generation'=to_jsonb(prior_score.lease_generation)
              AND started.created_at BETWEEN
                (prior_score.claim_response->>'lease_expires_at')::TIMESTAMPTZ
                  -make_interval(secs=>(v_round.configuration_doc->>'lease_ttl_seconds')::INTEGER)
                AND (prior_score.claim_response->>'lease_expires_at')::TIMESTAMPTZ
              AND started.occurred_at BETWEEN date_trunc('second',
                (prior_score.claim_response->>'lease_expires_at')::TIMESTAMPTZ
                  -make_interval(secs=>(v_round.configuration_doc->>'lease_ttl_seconds')::INTEGER))
                AND (prior_score.claim_response->>'lease_expires_at')::TIMESTAMPTZ
          )
$handoff_new$)
  ) AS patches(old_text,new_text) LOOP
    IF length(updated)-length(replace(updated,patch.old_text,''))<>length(patch.old_text) THEN
      RAISE EXCEPTION 'Score expiry start proof anchor differs' USING ERRCODE='55000';
    END IF;
    updated:=replace(updated,patch.old_text,patch.new_text);
  END LOOP;
  IF encode(extensions.digest(updated,'sha256'),'hex')<>'ccf306a3d6fddbe92f22f90cc2c0c401847eb4d80eb3a160cf06fa6ba6aa9987' THEN
    RAISE EXCEPTION 'Score expiry start proof postimage differs' USING ERRCODE='55000';
  END IF;
  EXECUTE updated;
  IF encode(extensions.digest(pg_get_functiondef(target),'sha256'),'hex')<>'ccf306a3d6fddbe92f22f90cc2c0c401847eb4d80eb3a160cf06fa6ba6aa9987'
    OR (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=target) IS DISTINCT FROM metadata THEN
    RAISE EXCEPTION 'Score expiry start proof readback differs' USING ERRCODE='55000';
  END IF;
END;
$score_expiry_start_proof510$;
NOTIFY pgrst,'reload schema';
COMMIT;
