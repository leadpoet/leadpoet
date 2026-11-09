-- Expired score leases with closed accounting and proved transport failures are not judge verdicts.
-- Change only the existing shared attempt counter. Claim, completion and expiry
-- retain their existing deadlines, cumulative score admission cap, backoff,
-- physical attempt IDs, checkpoints, payer rules and protected recovery paths.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';
SELECT pg_advisory_xact_lock(hashtextextended('lab-arena-claim-control',0));
DO $judge_credit680$
DECLARE target OID:='public.lab_arena__score_attempt_count_v1(public.lab_arena_runs)'::regprocedure;
 p RECORD; d TEXT; original_metadata JSONB; original_hash TEXT; updated_hash TEXT;
 old_text TEXT:='GREATEST(0, p_run.attempt - (';
 new_text TEXT:=$new680$GREATEST(0, p_run.attempt - (
   SELECT count(*)::INTEGER FROM public.lab_arena_runs lost
   JOIN public.lab_arena_rounds frozen ON frozen.round_id=lost.round_id
   WHERE p_run.kind='score' AND lost.kind='score'
     AND lost.assignment_id=p_run.assignment_id AND lost.attempt<=p_run.attempt
     AND lost.scoring_generation_floor=p_run.scoring_generation_floor
     AND (lost.round_id,lost.submission_id,lost.miner_hotkey,lost.stage,lost.icp_position)
       IS NOT DISTINCT FROM
       (p_run.round_id,p_run.submission_id,p_run.miner_hotkey,p_run.stage,p_run.icp_position)
     AND frozen.configuration_doc->>'infrastructure_retry_cost_policy'='trusted_infrastructure_v1'
     AND frozen.evaluation_date::DATE>=DATE '2026-10-14'
     AND lost.status='failed' AND NOT lost.champion_restart_required
     AND lost.lease_generation>0 AND lost.runner_hotkey IS NOT NULL
     AND lost.lease_token_hash IS NOT NULL AND lost.claim_request_id IS NOT NULL
     AND lost.claim_response->>'status'='leased'
     AND lost.claim_response->'lease_generation'=to_jsonb(lost.lease_generation)
     AND (
       -- A naturally expired score lease has no miner execution and no judge
       -- completion. The database expired it after the captured claim window;
       -- every call in this run must have a final priced settlement. Shortened
       -- early-host leases retain their separate authenticated proof below.
       (lost.terminal_cause='lease_expired' AND lost.result_doc IS NULL
         AND lost.output_ref IS NULL
         AND lost.terminal_doc=jsonb_build_object('expired_at',lost.terminal_doc->'expired_at')
         AND lost.lease_expires_at>=(lost.claim_response->>'lease_expires_at')::TIMESTAMPTZ
         AND (lost.terminal_doc->>'expired_at')::TIMESTAMPTZ>=lost.lease_expires_at
         AND (lost.terminal_doc->>'expired_at')::TIMESTAMPTZ<=clock_timestamp()
         AND public.lab_arena__judge_calls_clear_v1(lost.assignment_id)
         AND NOT EXISTS (SELECT 1 FROM public.lab_arena_ledger unidentified
           WHERE unidentified.run_id=lost.run_id AND (unidentified.call_identity IS NULL
             OR unidentified.call_identity !~ '^sha256:[0-9a-f]{64}$'))
         AND NOT EXISTS (SELECT 1 FROM (
           SELECT DISTINCT ON (cost.call_identity) cost.entry_kind
           FROM public.lab_arena_ledger cost WHERE cost.run_id=lost.run_id
           ORDER BY cost.call_identity,cost.entry_id DESC
         ) heads WHERE heads.entry_kind<>'settlement'))
       OR (
       -- The existing expiry path authored these receipts only after its exact
       -- signaled-host, cleanup and closed-accounting proof. Natural expiry and
       -- generic worker diagnostics cannot create them.
       (lost.terminal_cause='lease_expired' AND lost.result_doc IS NULL
         AND lost.output_ref IS NULL
         AND lost.terminal_doc->>'recovery_reason' IN (
           'authenticated_settled_score_runtime_host_error',
           'authenticated_completed_unpriced_score_runtime_host_error',
           'authenticated_closed_failed_score_runtime_host_error')
         AND (lost.terminal_doc->>'original_lease_expires_at')::TIMESTAMPTZ>=
           (lost.claim_response->>'lease_expires_at')::TIMESTAMPTZ
         AND lost.lease_expires_at<(lost.terminal_doc->>'original_lease_expires_at')::TIMESTAMPTZ
         AND EXISTS (SELECT 1 FROM public.lab_arena_ledger cost WHERE cost.run_id=lost.run_id)
         AND NOT EXISTS (SELECT 1 FROM public.lab_arena_trajectory_events done
           WHERE done.run_id=lost.run_id AND (done.event_kind IN ('runtime.finished','runtime.cleanup_error')
             OR done.event_kind='runtime.error' AND done.content->>'failure_stage'='cleanup'))
         AND EXISTS (SELECT 1 FROM public.lab_arena_trajectory_events error
           WHERE error.run_id=lost.run_id
             AND (error.runner_hotkey,error.round_id,error.submission_id,error.miner_hotkey,
               error.assignment_id,error.stage,error.icp_position,error.attempt,error.run_kind)
               IS NOT DISTINCT FROM
               (lost.runner_hotkey,lost.round_id,lost.submission_id,lost.miner_hotkey,
                 lost.assignment_id,lost.stage,lost.icp_position,lost.attempt,lost.kind)
             AND error.event_kind='runtime.error' AND error.content->>'status'='abandoned'
             AND error.content->>'failure_stage'='runtime'
             AND error.content->>'error_class'='RuntimeHostError'
             AND error.content->>'runtime_host_reason'='sandbox_launcher_signaled'
             AND jsonb_typeof(error.content->'launch_exit_code')='number'
             AND error.content->>'launch_exit_code' ~ '^-[1-9][0-9]*$'
             AND error.content->'launch_timed_out'='false'::JSONB
             AND error.created_at BETWEEN (lost.claim_response->>'lease_expires_at')::TIMESTAMPTZ
               -make_interval(secs=>(frozen.configuration_doc->>'lease_ttl_seconds')::INTEGER)
               AND (lost.claim_response->>'lease_expires_at')::TIMESTAMPTZ
             AND error.occurred_at BETWEEN date_trunc('second',
               (lost.claim_response->>'lease_expires_at')::TIMESTAMPTZ
               -make_interval(secs=>(frozen.configuration_doc->>'lease_ttl_seconds')::INTEGER))
               AND (lost.claim_response->>'lease_expires_at')::TIMESTAMPTZ)))
       OR (
         -- Only a final broker failure qualifies. A scorer's provider_error
         -- cannot bind a required failed source and is deliberately excluded.
         lost.terminal_cause='judge_error' AND lost.result_doc->>'terminal_status'='judge_error'
         AND lost.result_doc#>>'{failure_diagnostic,stage}'='provider_call'
         AND lost.result_doc#>>'{failure_diagnostic,error_class}'='provider_unavailable'
         AND lost.result_doc#>>'{failure_diagnostic,reason}'='provider_error'
         AND jsonb_typeof(lost.result_doc#>'{resource_summary,provider_unresolved_failure_count}')='number'
         AND (lost.result_doc#>>'{resource_summary,provider_unresolved_failure_count}')::NUMERIC>0
         AND EXISTS (
           SELECT 1 FROM public.lab_arena_ledger failed
           WHERE failed.run_id=lost.run_id
             AND (failed.round_id,failed.submission_id,failed.miner_hotkey,failed.stage)
               IS NOT DISTINCT FROM (lost.round_id,lost.submission_id,lost.miner_hotkey,lost.stage)
             AND ((failed.entry_kind='settlement' AND failed.terminal_response->'call_succeeded'='false'::JSONB)
               OR (failed.entry_kind='uncertain' AND failed.entry_doc#>'{call,call_succeeded}'='false'::JSONB))
             AND public.lab_arena__retained_provider_transient_v1(failed)
             -- No later attempted broker call may have healed the failure.
             -- Closed cost reconciliation of this same failed call is allowed.
             AND NOT EXISTS (SELECT 1 FROM public.lab_arena_ledger later
               WHERE later.run_id=lost.run_id AND later.entry_id>failed.entry_id
                 AND (later.call_identity IS DISTINCT FROM failed.call_identity
                   OR later.entry_kind IN ('reservation','dispatch','recovery')
                   OR later.terminal_response->'call_succeeded'='true'::JSONB
                   OR later.entry_doc#>'{call,call_succeeded}'='true'::JSONB))
         )
       )
     )
 ) - ($new680$;
BEGIN
 SELECT * INTO STRICT p FROM pg_proc WHERE oid=target;
 original_metadata:=to_jsonb(p)-'prosrc';d:=pg_get_functiondef(target);
 IF pg_get_userbyid(p.proowner)<>'lab_arena_owner'
   OR p.proacl::TEXT IS DISTINCT FROM '{lab_arena_owner=X/lab_arena_owner}'
   OR NOT p.prosecdef OR p.provolatile<>'s'
   OR p.prolang<>(SELECT oid FROM pg_language WHERE lanname='sql')
   OR p.proconfig IS DISTINCT FROM ARRAY['search_path=pg_catalog, public']::TEXT[]
   OR p.proisstrict OR p.proleakproof OR p.proparallel<>'u' THEN
   RAISE EXCEPTION 'judge credit680 counter security differs'; END IF;
 IF NOT EXISTS(SELECT 1 FROM pg_proc proof
   WHERE proof.oid=to_regprocedure('public.lab_arena__retained_provider_transient_v1(public.lab_arena_ledger)')
     AND encode(extensions.digest(proof.prosrc,'sha256'),'hex')='e8aab74f0d9a3ddd37d01cd3e4dc8df375030ab3fa6f4e532bdd829e3326f235'
     AND pg_get_userbyid(proof.proowner)='lab_arena_owner' AND proof.prosecdef
     AND proof.provolatile='s' AND NOT proof.proisstrict AND NOT proof.proleakproof
     AND proof.proparallel='u' AND proof.prolang=(SELECT oid FROM pg_language WHERE lanname='plpgsql')
     AND proof.proacl::TEXT='{lab_arena_owner=X/lab_arena_owner}'
     AND proof.proconfig=ARRAY['search_path=pg_catalog, public']::TEXT[]) THEN
   RAISE EXCEPTION 'judge credit680 retained provider proof differs'; END IF;
 IF NOT EXISTS(SELECT 1 FROM pg_proc clear
   WHERE clear.oid=to_regprocedure('public.lab_arena__judge_calls_clear_v1(text)')
     AND encode(extensions.digest(pg_get_functiondef(clear.oid),'sha256'),'hex')='aa8250cbfe7b62e0ba7802eb1b8c2a38c5b4a012a3e0044a04cb09692318bae7'
     AND pg_get_userbyid(clear.proowner)='lab_arena_owner' AND clear.prosecdef
     AND clear.proacl::TEXT='{lab_arena_owner=X/lab_arena_owner}'
     AND clear.proconfig=ARRAY['search_path=pg_catalog, public']::TEXT[]
     AND clear.provolatile='s' AND NOT clear.proisstrict AND NOT clear.proleakproof
     AND clear.proparallel='u' AND clear.prolang=(SELECT oid FROM pg_language WHERE lanname='sql')) THEN
   RAISE EXCEPTION 'judge credit680 closed accounting proof differs'; END IF;
 original_hash:=encode(extensions.digest(d,'sha256'),'hex');
 IF strpos(d,'-- Only a final broker failure qualifies.')>0 THEN
   IF replace(d,new_text,old_text)=d OR encode(extensions.digest(replace(d,new_text,old_text),'sha256'),'hex')
     NOT IN ('41f2a3f5fa3a5abf0e6106df52e98c59121a55e99e22daad88673ff8f45d9b84',
       '5bfb39967f4b3aeb60396560b1ef52564a68fcaf550893fdc760a0f03a9d9571') THEN
     RAISE EXCEPTION 'judge credit680 counter replay differs'; END IF;
   RETURN;
 END IF;
 IF original_hash NOT IN ('41f2a3f5fa3a5abf0e6106df52e98c59121a55e99e22daad88673ff8f45d9b84',
   '5bfb39967f4b3aeb60396560b1ef52564a68fcaf550893fdc760a0f03a9d9571')
   OR length(d)-length(replace(d,old_text,''))<>length(old_text) THEN
   RAISE EXCEPTION 'judge credit680 counter preimage differs'; END IF;
 d:=replace(d,old_text,new_text);updated_hash:=encode(extensions.digest(d,'sha256'),'hex');
 EXECUTE d;
 IF (SELECT to_jsonb(q)-'prosrc' FROM pg_proc q WHERE q.oid=target)IS DISTINCT FROM original_metadata
   OR encode(extensions.digest(pg_get_functiondef(target),'sha256'),'hex')<>updated_hash THEN
   RAISE EXCEPTION 'judge credit680 counter readback differs'; END IF;
END;
$judge_credit680$;
NOTIFY pgrst,'reload schema';
COMMIT;
