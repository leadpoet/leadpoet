-- Preserve paid responses while excluding multi-tool replies from single-tool reuse.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';
DO $single_tool_cache$
DECLARE
  v_definition TEXT;
  v_hash TEXT;
  v_old CONSTANT TEXT := $old$      IF pg_catalog.jsonb_typeof(v_body) = 'object' THEN$old$;
  v_new CONSTANT TEXT := $new$      -- A native success is still paid and retained when it violates the
      -- explicit single-tool request. It cannot become another run's input.
      IF pg_catalog.jsonb_typeof(v_body) = 'object'
         AND (
           p_call_doc -> 'judgment_cache_single_tool_required'
             IS DISTINCT FROM 'true'::JSONB
           OR NOT EXISTS (
             SELECT 1 FROM pg_catalog.jsonb_array_elements(
               CASE WHEN pg_catalog.jsonb_typeof(v_body -> 'choices') = 'array'
                 THEN v_body -> 'choices' ELSE '[]'::JSONB END
             ) AS choice
             WHERE pg_catalog.jsonb_array_length(
               CASE WHEN pg_catalog.jsonb_typeof(choice #> '{message,tool_calls}') = 'array'
                 THEN choice #> '{message,tool_calls}' ELSE '[]'::JSONB END
             ) > 1
           )
         ) THEN$new$;
BEGIN
  SELECT pg_catalog.pg_get_functiondef(p.oid),
    pg_catalog.encode(pg_catalog.sha256(pg_catalog.convert_to(
      pg_catalog.pg_get_functiondef(p.oid),'UTF8')),'hex')
  INTO STRICT v_definition,v_hash
  FROM pg_catalog.pg_proc p
  WHERE p.oid='public.lab_arena_reserve_judgment_call(text,text,text,text,text,text,bigint,jsonb,integer)'::pg_catalog.regprocedure
    AND pg_catalog.pg_get_userbyid(p.proowner)='lab_arena_owner'
    AND p.prosecdef AND p.provolatile='v'
    AND p.proconfig=ARRAY['search_path=pg_catalog, public']::TEXT[]
    AND p.proacl::TEXT='{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}';
  IF v_hash='6970e47b0b3aa9308a3a38471ce0c70fc5498f4336c9277994546134ecb28d3b' THEN RETURN; END IF;
  IF v_hash<>'15959c759d07df235d9d8fa4ce844ec4113c3b442b679a06fdc20f6a52336e35' OR
     (pg_catalog.length(v_definition)-pg_catalog.length(
       pg_catalog.replace(v_definition,v_old,'')))/pg_catalog.length(v_old)<>1 THEN
    RAISE EXCEPTION 'single-tool cache source differs';
  END IF;
  EXECUTE pg_catalog.replace(v_definition,v_old,v_new);
  SELECT pg_catalog.encode(pg_catalog.sha256(pg_catalog.convert_to(
    pg_catalog.pg_get_functiondef('public.lab_arena_reserve_judgment_call(text,text,text,text,text,text,bigint,jsonb,integer)'::pg_catalog.regprocedure),'UTF8')),'hex')
  INTO v_hash;
  IF v_hash<>'6970e47b0b3aa9308a3a38471ce0c70fc5498f4336c9277994546134ecb28d3b' THEN
    RAISE EXCEPTION 'single-tool cache postimage differs';
  END IF;
END;
$single_tool_cache$;
COMMIT;
