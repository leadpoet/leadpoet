-- Restore execution capacity lost to a second paid validator restart and its gateway release drain.
-- Only this round's existing upper time bounds change. Intake stays closed.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='15s';
SET LOCAL TIME ZONE 'UTC';
SET LOCAL ROLE lab_arena_owner;
SELECT pg_advisory_xact_lock(hashtextextended('lab_arena.baseline_promotion',0));
SELECT pg_advisory_xact_lock(hashtextextended('lab-arena-claim-control',0));
DO $window714$
DECLARE
 rid CONSTANT text := 'arena-2026-11-21';
 request CONSTANT text := 'nov21-second-validator-loss-window714-v1';
 r public.lab_arena_rounds; revised public.lab_arena_rounds;
 prior public.lab_arena_scoring_supersessions;
 bounds jsonb; evidence jsonb;
BEGIN
 SELECT * INTO STRICT r FROM public.lab_arena_rounds WHERE round_id=rid FOR UPDATE;
 SELECT jsonb_build_object('stage_2_close','2026-11-23T06:02:26Z',
   'final_scoring_close','2026-11-23T09:02:26Z','publication_deadline','2026-11-23T09:02:27Z') INTO bounds;
 IF EXISTS (SELECT 1 FROM public.lab_arena_scoring_supersessions WHERE round_id=rid AND request_id=request
       AND recovery_doc->>'operation'='round_operational_update_v1' AND recovery_doc->'windows'=bounds
       AND (new_round_doc#>'{configuration_doc,schedule}') @> bounds) THEN RETURN; END IF;
 SELECT * INTO STRICT prior FROM public.lab_arena_scoring_supersessions
   WHERE round_id=rid AND request_id='nov21-validator-capacity-window712-v1' AND recovery_doc->>'operation'='round_operational_update_v1';
 IF r.status<>'stage2' OR r.published_at IS NOT NULL OR r.reward_activated_at IS NOT NULL
    OR current_setting('session_replication_role')<>'origin'
    OR (public.lab_arena_restart_guard_state_v1()->>'paused')::boolean IS DISTINCT FROM false
    OR r.configuration_doc IS DISTINCT FROM prior.new_round_doc->'configuration_doc'
    OR r.participants IS DISTINCT FROM prior.new_round_doc->'participants'
    OR NOT ((r.configuration_doc->'schedule') @> '{"stage_2_close":"2026-11-23T04:02:26Z","final_scoring_close":"2026-11-23T07:02:26Z","publication_deadline":"2026-11-23T07:02:27Z"}')
    OR clock_timestamp()>=(r.configuration_doc#>>'{schedule,stage_2_close}')::timestamptz
    OR NOT EXISTS (SELECT 1 FROM public.lab_arena_runs WHERE round_id=rid AND stage=2 AND kind='execute' AND status='pending')
    OR NOT EXISTS (SELECT 1 FROM public.lab_arena_validator_events v
      WHERE v.validator_hotkey='5Chnr6Y72gdfTFdoZnsCvkndKpMk8jAtt9JAYKaNG3LmU4BW'
        AND v.event_kind='validator.startup' AND v.content->>'session_id'='38fdc318-482c-4d10-92f7-8e51bf89ab43'
        AND v.content->>'active_runs'='0' AND v.occurred_at>='2026-11-23T01:09:00Z'
        AND v.occurred_at<'2026-11-23T01:10:00Z')
    OR (SELECT count(DISTINCT h.assignment_id) FROM public.lab_arena_runs h
      WHERE h.round_id=rid AND h.kind='execute' AND h.stage=2 AND h.status='failed' AND h.terminal_cause='lease_expired'
        AND h.runner_hotkey='5Chnr6Y72gdfTFdoZnsCvkndKpMk8jAtt9JAYKaNG3LmU4BW'
        AND h.lease_expires_at>='2026-11-23T02:24:00Z' AND h.lease_expires_at<'2026-11-23T02:25:00Z'
        AND h.lease_expires_at<=clock_timestamp()
        AND h.lease_expires_at>=(h.claim_response->>'lease_expires_at')::timestamptz
        AND h.result_doc IS NULL AND h.output_ref IS NULL
        AND EXISTS(SELECT 1 FROM public.lab_arena_ledger l WHERE l.run_id=h.run_id
          AND l.entry_kind='settlement' AND l.amount_microusd>0)
        AND NOT EXISTS(SELECT 1 FROM public.lab_arena_trajectory_events e WHERE e.run_id=h.run_id
          AND e.event_kind='runtime.finished'))<>14
 THEN RAISE EXCEPTION 'Recovery window preconditions differ'; END IF;
 revised:=r;
 revised.configuration_doc:=jsonb_set(r.configuration_doc,'{schedule}',(r.configuration_doc->'schedule')||bounds,false);
 evidence:=jsonb_build_object('operation','round_operational_update_v1',
   'reason','nov21_second_paid_validator_restart','windows',bounds);
 INSERT INTO public.lab_arena_scoring_supersessions(round_id,generation_floor,request_id,request_hash,
   actor_ref,transaction_id,backend_pid,old_round_doc,new_round_doc,old_score_projections,recovery_doc)
 SELECT rid,coalesce(max(generation_floor),0)+1,request,'sha256:'||encode(extensions.digest(evidence::text,'sha256'),'hex'),
   'authorized-nov21-nov22-recovery',txid_current(),pg_backend_pid(),to_jsonb(r)-'updated_at',
   to_jsonb(revised)-'updated_at','{}',evidence FROM public.lab_arena_scoring_supersessions WHERE round_id=rid;
 UPDATE public.lab_arena_rounds SET configuration_doc=revised.configuration_doc WHERE round_id=rid;
 IF (SELECT to_jsonb(check_row)-'updated_at' FROM public.lab_arena_rounds check_row WHERE round_id=rid)
    IS DISTINCT FROM to_jsonb(revised)-'updated_at' THEN RAISE EXCEPTION 'Recovery window readback differs'; END IF;
END;
$window714$;
COMMIT;
