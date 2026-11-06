-- Bound existing cost evidence reads to the exact run and call already selected.
-- OFFSET 0 retains parameterized primary-key lookups instead of global scans.
-- Preserve all latest-head, recovered response, NULL, cancellation, infrastructure,
-- retirement and delayed-settlement rules. No data, index, timeout or grant change.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';
DO $cost_lookup662$
DECLARE
  item RECORD;
  target OID;
  definition TEXT;
  updated TEXT;
  metadata JSONB;
  identity JSONB;
  before_hash TEXT;
  after_hash TEXT;
  prerequisite RECORD;
BEGIN
  FOR prerequisite IN SELECT * FROM (VALUES
    ('public.lab_arena_runs_pkey','public.lab_arena_runs',
     'CREATE UNIQUE INDEX lab_arena_runs_pkey ON public.lab_arena_runs USING btree (run_id)'),
    ('public.lab_arena_deepline_call_responses_pkey','public.lab_arena_deepline_call_responses',
     'CREATE UNIQUE INDEX lab_arena_deepline_call_responses_pkey ON public.lab_arena_deepline_call_responses USING btree (call_identity)'),
    ('public.lab_arena_ledger_cost_run_idx','public.lab_arena_ledger',
     'CREATE INDEX lab_arena_ledger_cost_run_idx ON public.lab_arena_ledger USING btree (submission_id, run_id) WHERE (call_identity IS NOT NULL)')
  ) required(name,relation,definition) LOOP
    IF NOT EXISTS(SELECT 1 FROM pg_index i JOIN pg_class c ON c.oid=i.indexrelid
      WHERE i.indexrelid=to_regclass(prerequisite.name)
        AND i.indrelid=to_regclass(prerequisite.relation)
        AND i.indisvalid AND i.indisready
        AND pg_get_userbyid(c.relowner)='lab_arena_owner'
        AND pg_get_indexdef(i.indexrelid)=prerequisite.definition) THEN
      RAISE EXCEPTION 'Cost keyed lookup662 index differs: %',prerequisite.name USING ERRCODE='55000';
    END IF;
  END LOOP;
  FOR item IN SELECT * FROM (VALUES
    ('public.lab_arena__successful_call_cost_state(text,text,text)','cc0e7370f7cb650ea92b3420ad573f089e4164cd136e28741c95f780b2543a3e','ad8d70f9b22b83610a26aaf1af8508eaec2f63129ff30d2865a58766c99643be',
     $old_run662$    JOIN public.lab_arena_runs AS runs ON runs.run_id = scoped.run_id
    WHERE runs.kind = p_kind$old_run662$,
     $new_run662$    CROSS JOIN LATERAL (
      SELECT found.run_id FROM public.lab_arena_runs AS found
      WHERE found.run_id = scoped.run_id AND found.kind = p_kind
      OFFSET 0
    ) AS runs$new_run662$,
     '{lab_arena_owner=X/lab_arena_owner}'),
    ('public.lab_arena__successful_icp_cost_state(text,text,integer)','05d1386f6c557f63c163d4865b4af634eeee2b0fc40e68d35a4c7d8dae2013e3','03163f8ece8c6755df514f9ef1e6dcbbcd5ce54d01dfe2fc76cf5c12aef462be',
     $old_run662$    JOIN public.lab_arena_runs AS runs ON runs.run_id = scoped.run_id
    WHERE runs.kind = 'execute'
      AND runs.icp_position = p_icp_position$old_run662$,
     $new_run662$    CROSS JOIN LATERAL (
      SELECT found.* FROM public.lab_arena_runs AS found
      WHERE found.run_id = scoped.run_id AND found.kind = 'execute'
        AND found.icp_position = p_icp_position
      OFFSET 0
    ) AS runs$new_run662$,
     '{lab_arena_owner=X/lab_arena_owner}'),
    ('public.lab_arena_submission_costs(text)','71ddd536b6069f80904f0863a16c27d0e990871b961597c84c66c4d35ce7d128','22cdebb6fa8779cc191f8164be6c3d2c52989cda4687412ebd998a2ca67f04fd',
     $old_run662$      JOIN public.lab_arena_runs AS runs ON runs.run_id = scoped.run_id$old_run662$,
     $new_run662$      CROSS JOIN LATERAL (
        SELECT found.run_id, found.kind FROM public.lab_arena_runs AS found
        WHERE found.run_id = scoped.run_id
        OFFSET 0
      ) AS runs$new_run662$,
     '{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}')
  ) required(signature,preimage,postimage,old_run,new_run,acl) LOOP
    target:=to_regprocedure(item.signature);
    SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc',
      jsonb_build_array(pg_get_userbyid(p.proowner),p.proacl::TEXT,p.prosecdef,
        p.provolatile,p.proconfig,l.lanname),
      encode(pg_catalog.sha256(pg_catalog.convert_to(p.prosrc,'UTF8')),'hex')
    INTO definition,metadata,identity,before_hash
    FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang WHERE p.oid=target;
    IF definition IS NULL OR identity IS DISTINCT FROM jsonb_build_array(
      'lab_arena_owner',item.acl,TRUE,'s',ARRAY['search_path=pg_catalog, public'],
      CASE WHEN item.signature='public.lab_arena_submission_costs(text)' THEN 'plpgsql' ELSE 'sql' END) THEN
      RAISE EXCEPTION 'Cost keyed lookup662 security differs: %',item.signature USING ERRCODE='55000';
    END IF;
    IF before_hash=item.postimage THEN CONTINUE; END IF;
    IF before_hash IS DISTINCT FROM item.preimage THEN
      RAISE EXCEPTION 'Cost keyed lookup662 preimage differs: %',item.signature USING ERRCODE='55000';
    END IF;
    updated:=replace(definition,item.old_run,item.new_run);
    IF item.signature<>'public.lab_arena_submission_costs(text)' THEN
      updated:=replace(updated,
        $old_response662$              AND response.terminal_response -> 'call_succeeded' = 'true'::JSONB)$old_response662$,
        $new_response662$              AND response.terminal_response -> 'call_succeeded' = 'true'::JSONB
            OFFSET 0)$new_response662$);
    END IF;
    EXECUTE updated;
    SELECT encode(pg_catalog.sha256(pg_catalog.convert_to(p.prosrc,'UTF8')),'hex')
      INTO after_hash FROM pg_proc p WHERE p.oid=target;
    IF after_hash IS DISTINCT FROM item.postimage
      OR (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=target) IS DISTINCT FROM metadata THEN
      RAISE EXCEPTION 'Cost keyed lookup662 readback differs: %',item.signature USING ERRCODE='55000';
    END IF;
  END LOOP;
END;
$cost_lookup662$;
COMMIT;
