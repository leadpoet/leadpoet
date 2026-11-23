-- Extend the native runner capacity check. Three paid executions that
-- naturally expired without a final worker event warrant a cooldown of one
-- frozen lease TTL. This allocates no retries and does not stop existing work.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';
DO $cooldown713$
DECLARE
 target OID := 'public.lab_arena_claim_assignment(text,text,integer,integer,text[],text,text,text,integer)'::regprocedure;
 definition TEXT; metadata JSONB;
 pre CONSTANT TEXT := '704e7b48d229f9400d8ce460bc5bd410b205ee724cfb6d13a8f55bd865047abd';
 post CONSTANT TEXT := 'dfae2ae84a92c2d90f15a10d5f2ea283bdda1e4f1d359e03c132e7953ad612dd';
 anchor CONSTANT TEXT := $old$    GROUP BY f.runner_hotkey HAVING count(DISTINCT f.assignment_id)>=3$old$;
 extra CONSTANT TEXT := $extra$
    UNION ALL
    SELECT lost.runner_hotkey, 'execute'::TEXT
    FROM public.lab_arena_runs lost
    WHERE lost.round_id=p_round_id AND lost.kind='execute' AND lost.status='failed'
      AND lost.terminal_cause='lease_expired' AND lost.result_doc IS NULL AND lost.output_ref IS NULL
      AND NOT lost.champion_restart_required
      AND lost.lease_generation>0 AND lost.claim_request_hash IS NOT NULL
      AND lost.claim_response->>'status'='leased' AND lost.claim_response->>'run_id'=lost.run_id
      AND lost.claim_response->'lease_generation'=to_jsonb(lost.lease_generation)
      AND lost.lease_expires_at >= (lost.claim_response->>'lease_expires_at')::timestamptz
      AND lost.lease_expires_at <= statement_timestamp()
      AND lost.lease_expires_at > statement_timestamp()
        - make_interval(secs => (v_round.configuration_doc->>'lease_ttl_seconds')::integer)
      AND EXISTS (SELECT 1 FROM public.lab_arena_runs demand
        WHERE demand.round_id=p_round_id AND demand.kind='execute' AND demand.status='pending')
      AND EXISTS (SELECT 1 FROM public.lab_arena_ledger cost WHERE cost.run_id=lost.run_id
        AND cost.entry_kind='settlement' AND cost.amount_microusd>0 AND cost.created_at<=lost.lease_expires_at)
      AND NOT EXISTS (SELECT 1 FROM public.lab_arena_trajectory_events finish
        WHERE finish.run_id=lost.run_id AND finish.runner_hotkey=lost.runner_hotkey
          AND finish.run_kind='execute' AND finish.event_kind='runtime.finished')
    GROUP BY lost.runner_hotkey HAVING count(DISTINCT lost.assignment_id)>=3$extra$;
BEGIN
 SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc' INTO STRICT definition,metadata
 FROM pg_proc p WHERE p.oid=target;
 IF pg_get_userbyid((metadata->>'proowner')::oid)<>'lab_arena_owner'
    OR metadata->'proacl' IS DISTINCT FROM '["lab_arena_owner=X/lab_arena_owner","lab_arena_service=X/lab_arena_owner"]'::jsonb
    OR metadata->>'prosecdef'<>'true' THEN RAISE EXCEPTION 'Claim cooldown authority differs'; END IF;
 IF encode(extensions.digest(definition,'sha256'),'hex')=post THEN RETURN; END IF;
 IF encode(extensions.digest(definition,'sha256'),'hex')<>pre THEN RAISE EXCEPTION 'Claim cooldown preimage differs'; END IF;
 definition:=replace(definition,anchor,anchor||extra);
 IF encode(extensions.digest(definition,'sha256'),'hex')<>post THEN RAISE EXCEPTION 'Claim cooldown postimage differs'; END IF;
 EXECUTE definition;
 IF NOT EXISTS(SELECT 1 FROM pg_proc p WHERE p.oid=target AND to_jsonb(p)-'prosrc'=metadata
    AND encode(extensions.digest(pg_get_functiondef(p.oid),'sha256'),'hex')=post)
 THEN RAISE EXCEPTION 'Claim cooldown readback differs'; END IF;
END;
$cooldown713$;
COMMIT;
