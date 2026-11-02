-- Extend only private reviewer observations for the verified Sol route.
-- Admission decisions, retry policy, costs and historical rows are unchanged.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';

DO $private_review599$
DECLARE
  v_signature CONSTANT TEXT := 'public.lab_arena_finish_submission_review(text,text,text,text,jsonb,bigint)';
  v_preimage CONSTANT TEXT := 'cd09eb1c882c3be43b78415901eddb462feeb860b25e6affd7b86278c072d777';
  v_postimage CONSTANT TEXT := '6bf2741c49b649f47a846c7556e38770bb7ab5dbc3e94294aac1b3aeb04795ce';
  v_oid OID;
  v_definition TEXT;
  v_updated TEXT;
  v_metadata JSONB;
  v_identity JSONB;
  v_old TEXT;
  v_new TEXT;
BEGIN
  v_oid:=pg_catalog.to_regprocedure(v_signature);
  SELECT pg_catalog.pg_get_functiondef(p.oid),pg_catalog.to_jsonb(p)-'prosrc',
    pg_catalog.jsonb_build_array(owner.rolname,p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig)
  INTO v_definition,v_metadata,v_identity
  FROM pg_catalog.pg_proc p JOIN pg_catalog.pg_roles owner ON owner.oid=p.proowner
  WHERE p.oid=v_oid;
  IF v_definition IS NULL OR v_identity IS DISTINCT FROM pg_catalog.jsonb_build_array(
    'lab_arena_owner',
    '{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}',
    TRUE,'v',ARRAY['search_path=pg_catalog, public']::TEXT[]) THEN
    RAISE EXCEPTION 'Arena review599 security shape differs' USING ERRCODE='55000';
  END IF;
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')=v_postimage THEN
    RETURN;
  END IF;
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')<>v_preimage THEN
    RAISE EXCEPTION 'Arena review599 preimage differs' USING ERRCODE='55000';
  END IF;
  v_updated:=v_definition;
  v_old:=$old599_model$'anthropic/claude-sonnet-5','anthropic/claude-sonnet-5-20260630')$old599_model$;
  v_new:=$new599_model$'anthropic/claude-sonnet-5','anthropic/claude-sonnet-5-20260630',
                'openai/gpt-6.1-sol','openai/gpt-6.1-sol-20260929')$new599_model$;
  IF length(v_updated)-length(pg_catalog.replace(v_updated,v_old,''))<>length(v_old) THEN
    RAISE EXCEPTION 'Arena review599 model anchor differs' USING ERRCODE='55000';
  END IF;
  v_updated:=pg_catalog.replace(v_updated,v_old,v_new);
  v_old:=$old599_provider$('Anthropic','Amazon Bedrock','Google','Google Vertex')$old599_provider$;
  v_new:=$new599_provider$('Anthropic','Amazon Bedrock','Google','Google Vertex','Azure')$new599_provider$;
  IF length(v_updated)-length(pg_catalog.replace(v_updated,v_old,''))<>length(v_old) THEN
    RAISE EXCEPTION 'Arena review599 provider anchor differs' USING ERRCODE='55000';
  END IF;
  v_updated:=pg_catalog.replace(v_updated,v_old,v_new);
  IF pg_catalog.encode(extensions.digest(v_updated,'sha256'),'hex')<>v_postimage THEN
    RAISE EXCEPTION 'Arena review599 postimage differs' USING ERRCODE='55000';
  END IF;
  EXECUTE v_updated;
  IF pg_catalog.encode(extensions.digest(pg_catalog.pg_get_functiondef(v_oid),'sha256'),'hex')<>v_postimage
     OR (SELECT pg_catalog.to_jsonb(p)-'prosrc' FROM pg_catalog.pg_proc p WHERE p.oid=v_oid)
       IS DISTINCT FROM v_metadata THEN
    RAISE EXCEPTION 'Arena review599 readback differs' USING ERRCODE='55000';
  END IF;
END;
$private_review599$;

COMMIT;
