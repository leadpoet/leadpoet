-- Apply the existing proved, zero-call installer fault at assignment scope.
-- No additional fault scan, retry permit, or paid call is created.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';
DO $migration$
DECLARE
 target oid := 'public.lab_arena_claim_assignment(text,text,integer,integer,text[],text,text,text,integer)'::regprocedure;
 definition text; metadata jsonb;
 pre constant text := 'dfae2ae84a92c2d90f15a10d5f2ea283bdda1e4f1d359e03c132e7953ad612dd';
 post constant text := '525e540ab8236c256c07f6fed0d13f10f660f32598707c043abb9a8770edfa43';
BEGIN
 SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc' INTO STRICT definition,metadata
 FROM pg_proc p WHERE p.oid=target;
 IF pg_get_userbyid((metadata->>'proowner')::oid)<>'lab_arena_owner'
   OR metadata->'proacl' IS DISTINCT FROM '["lab_arena_owner=X/lab_arena_owner","lab_arena_service=X/lab_arena_owner"]'::jsonb
   OR metadata->>'prosecdef'<>'true' THEN RAISE EXCEPTION 'Claim assignment authority differs'; END IF;
 IF encode(sha256(convert_to(definition,'UTF8')),'hex')=post THEN RETURN; END IF;
 IF encode(sha256(convert_to(definition,'UTF8')),'hex')<>pre THEN RAISE EXCEPTION 'Claim assignment preimage differs'; END IF;
 definition:=replace(definition,$old$NULL::TEXT AS restricted_kind FROM public.lab_arena_runs AS host_fault$old$,$new$NULL::TEXT AS restricted_kind, NULL::TEXT AS restricted_assignment FROM public.lab_arena_runs AS host_fault$new$);
 definition:=replace(definition,$old$SELECT f.runner_hotkey, 'execute'::TEXT
    FROM public.lab_arena_runs f$old$,$new$SELECT f.runner_hotkey, 'execute'::TEXT, f.assignment_id
    FROM public.lab_arena_runs f$new$);
 definition:=replace(definition,$old$GROUP BY f.runner_hotkey HAVING count(DISTINCT f.assignment_id)>=3$old$,$new$-- One authenticated miss excludes this host for this assignment only.
    -- Three distinct misses retain the existing round-wide execution guard.
    GROUP BY GROUPING SETS ((f.runner_hotkey,f.assignment_id),(f.runner_hotkey))
    HAVING grouping(f.assignment_id)=0 OR count(DISTINCT f.assignment_id)>=3$new$);
 definition:=replace(definition,$old$SELECT lost.runner_hotkey, 'execute'::TEXT
    FROM public.lab_arena_runs lost$old$,$new$SELECT lost.runner_hotkey, 'execute'::TEXT, NULL::TEXT
    FROM public.lab_arena_runs lost$new$);
 definition:=replace(definition,$old$WHERE quarantined.runner_hotkey=p_runner_hotkey
$old$,$new$WHERE quarantined.runner_hotkey=p_runner_hotkey
        AND (quarantined.restricted_assignment IS NULL OR quarantined.restricted_assignment=runs.assignment_id)
$new$);
 definition:=replace(definition,$old$WHERE quarantined.runner_hotkey=others.runner_hotkey
$old$,$new$WHERE quarantined.runner_hotkey=others.runner_hotkey
                 AND (quarantined.restricted_assignment IS NULL OR quarantined.restricted_assignment=runs.assignment_id)
$new$);
 IF encode(sha256(convert_to(definition,'UTF8')),'hex')<>post THEN RAISE EXCEPTION 'Claim assignment postimage differs'; END IF;
 EXECUTE definition;
 IF NOT EXISTS(SELECT FROM pg_proc p WHERE p.oid=target AND to_jsonb(p)-'prosrc'=metadata
   AND encode(sha256(convert_to(pg_get_functiondef(p.oid),'UTF8')),'hex')=post)
 THEN RAISE EXCEPTION 'Claim assignment readback differs'; END IF;
END;
$migration$;
COMMIT;
