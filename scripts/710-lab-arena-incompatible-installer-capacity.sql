-- Extend the existing claim quarantine to a proved pre-sandbox prerequisite failure.
-- No new control table, retry budget, or mutation of existing leases.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';
DO $claim710$
DECLARE
 target OID := 'public.lab_arena_claim_assignment(text,text,integer,integer,text[],text,text,text,integer)'::regprocedure;
 d TEXT; metadata JSONB;
 pre CONSTANT TEXT := 'f7c1ed03241656ed0111749c04c18ad5ceab019d51dd93ec957a3f52a0f86da0';
 post CONSTANT TEXT := '704e7b48d229f9400d8ce460bc5bd410b205ee724cfb6d13a8f55bd865047abd';
BEGIN
 SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc' INTO STRICT d,metadata
 FROM pg_proc p WHERE p.oid=target;
 IF pg_get_userbyid((metadata->>'proowner')::OID)<>'lab_arena_owner'
    OR metadata->'proacl' IS DISTINCT FROM '["lab_arena_owner=X/lab_arena_owner","lab_arena_service=X/lab_arena_owner"]'::jsonb
    OR metadata->>'prosecdef'<>'true' THEN RAISE EXCEPTION 'Claim710 authority differs'; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')=post THEN RETURN; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')<>pre THEN RAISE EXCEPTION 'Claim710 preimage differs'; END IF;
 d:=replace(d,$old$    SELECT host_fault.runner_hotkey FROM public.lab_arena_runs AS host_fault$old$,$new$    SELECT host_fault.runner_hotkey, NULL::TEXT AS restricted_kind FROM public.lab_arena_runs AS host_fault$new$);
 d:=replace(d,$old$    HAVING COUNT(DISTINCT host_fault.assignment_id)>=3$old$,$new$    HAVING COUNT(DISTINCT host_fault.assignment_id)>=3
    -- A missing host interpreter prevents new execution. Trusted scorer jobs
    -- use their own image and keep their ordinary eligibility.
    UNION ALL
    SELECT f.runner_hotkey, 'execute'::TEXT
    FROM public.lab_arena_runs f
    WHERE f.round_id=p_round_id AND f.kind='execute' AND f.status='failed'
      AND f.terminal_cause='provider_error' AND f.output_ref IS NULL
      AND f.result_doc->>'terminal_status'='provider_error'
      AND f.result_doc#>'{resource_summary,provider_call_count}'='0'::JSONB
      AND f.result_doc#>'{resource_summary,wall_seconds}'='0'::JSONB
      AND f.claim_request_hash IS NOT NULL AND f.lease_generation>0
      AND EXISTS (SELECT 1 FROM public.lab_arena_runs demand
        WHERE demand.round_id=p_round_id AND demand.kind='execute' AND demand.status='pending')
      AND NOT EXISTS (SELECT 1 FROM public.lab_arena_ledger l WHERE l.run_id=f.run_id)
      AND EXISTS (SELECT 1 FROM public.lab_arena_trajectory_events e
        WHERE e.run_id=f.run_id AND e.runner_hotkey=f.runner_hotkey AND e.run_kind='execute'
          AND e.event_kind='runtime.error' AND e.content->>'status'='provider_error'
          AND e.content->>'failure_stage'='setup'
          AND e.content->>'error_class'='DependencyInstallInfrastructureError'
          AND e.content->>'failure_kind'='installer_error'
          AND e.content->>'diagnostic_detail'='compatible dependency installer unavailable: CPython 3.11 required')
    GROUP BY f.runner_hotkey HAVING count(DISTINCT f.assignment_id)>=3$new$);
 d:=replace(d,$old$WHERE quarantined.runner_hotkey=p_runner_hotkey
$old$,$new$WHERE quarantined.runner_hotkey=p_runner_hotkey
        AND (quarantined.restricted_kind IS NULL OR quarantined.restricted_kind=runs.kind)
$new$);
 d:=replace(d,$old$WHERE quarantined.runner_hotkey=others.runner_hotkey
$old$,$new$WHERE quarantined.runner_hotkey=others.runner_hotkey
                 AND (quarantined.restricted_kind IS NULL OR quarantined.restricted_kind=runs.kind)
$new$);
 IF encode(extensions.digest(d,'sha256'),'hex')<>post THEN RAISE EXCEPTION 'Claim710 postimage differs'; END IF;
 EXECUTE d;
 IF NOT EXISTS(SELECT 1 FROM pg_proc p WHERE p.oid=target AND to_jsonb(p)-'prosrc'=metadata
    AND encode(extensions.digest(pg_get_functiondef(p.oid),'sha256'),'hex')=post)
 THEN RAISE EXCEPTION 'Claim710 readback differs'; END IF;
END;
$claim710$;
COMMIT;
