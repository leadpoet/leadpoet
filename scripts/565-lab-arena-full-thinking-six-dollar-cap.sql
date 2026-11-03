-- Only frozen full-generated-thinking rounds admit and score up to $6 per ICP.
-- Current-policy admission still uses confirmed competition cost without holds.
-- Historical $4 flat rounds and $0.80 qualified-pair rounds keep their rules.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';
DO $full_thinking_cap564$
DECLARE
  item RECORD;
  source TEXT;
  updated TEXT;
  identity_before JSONB;
  identity_after JSONB;
  target OID;
  old_icp TEXT := $old_icp$IS DISTINCT FROM 4000000$old_icp$;
  new_icp TEXT := $new_icp$IS DISTINCT FROM (CASE WHEN v_round.configuration_doc ->> 'reasoning_capture_policy'
           = 'full_generated_thinking_v1' THEN 6000000 ELSE 4000000 END)$new_icp$;
  old_company TEXT := $old_company$= 'successful_calls_flat_per_icp_v1' THEN 4000000 ELSE 800000 END$old_company$;
  new_company TEXT := $new_company$= 'successful_calls_flat_per_icp_v1' THEN
             CASE WHEN v_round.configuration_doc ->> 'reasoning_capture_policy'
               = 'full_generated_thinking_v1' THEN 6000000 ELSE 4000000 END
           ELSE 800000 END$new_company$;
BEGIN
  FOR item IN SELECT * FROM (VALUES
    ('public.lab_arena_reserve_call(text,text,text,text,text,text,bigint,jsonb,integer)',
     '8cd8236c6f185956035c327357f5c7d7ea67bee9f83bbab8f7976c97e1b5b4f8',
     'b800eaf326ae18e3090fbeb35447c7f516db5104782750bdb9d64b7af1803104'),
    ('public.lab_arena_icp_cost_eligibility(text,text,integer,integer)',
     'f190d9727c331c967966f8552a43e12f6cbafad8e8281be7d68749d5b8704571',
     '97b5b528c0f81f902f7ce1ed6e0390dbce2822d0224a60b28b6cab9397f353a6')
  ) AS t(signature,preimage,postimage) LOOP
    target := to_regprocedure(item.signature);
    SELECT pg_get_functiondef(p.oid),
           jsonb_build_array(pg_get_userbyid(p.proowner),p.proacl::TEXT,
             p.prosecdef,p.provolatile,p.proconfig,l.lanname)
      INTO source,identity_before
      FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang WHERE p.oid=target;
    IF source IS NULL OR identity_before IS DISTINCT FROM
       jsonb_build_array('lab_arena_owner',
         '{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}',
         true, CASE WHEN item.signature LIKE '%reserve_call%' THEN 'v' ELSE 's' END,
         ARRAY['search_path=pg_catalog, public']::TEXT[], 'plpgsql') THEN
      RAISE EXCEPTION 'full thinking cost security identity differs: %',item.signature
        USING ERRCODE='55000';
    END IF;
    IF encode(extensions.digest(source,'sha256'),'hex')=item.postimage THEN CONTINUE; END IF;
    IF encode(extensions.digest(source,'sha256'),'hex')<>item.preimage
       OR (length(source)-length(replace(source,old_icp,'')))<>length(old_icp)
       OR (length(source)-length(replace(source,old_company,'')))<>length(old_company) THEN
      RAISE EXCEPTION 'full thinking cost preimage differs: %',item.signature
        USING ERRCODE='55000';
    END IF;
    updated := replace(replace(source,old_icp,new_icp),old_company,new_company);
    IF encode(extensions.digest(updated,'sha256'),'hex')<>item.postimage THEN
      RAISE EXCEPTION 'full thinking cost target differs: %',item.signature
        USING ERRCODE='55000';
    END IF;
    EXECUTE updated;
    SELECT jsonb_build_array(pg_get_userbyid(p.proowner),p.proacl::TEXT,
             p.prosecdef,p.provolatile,p.proconfig,l.lanname)
      INTO identity_after
      FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang WHERE p.oid=target;
    IF identity_after IS DISTINCT FROM identity_before
       OR encode(extensions.digest(pg_get_functiondef(target),'sha256'),'hex')<>item.postimage THEN
      RAISE EXCEPTION 'full thinking cost readback differs: %',item.signature
        USING ERRCODE='55000';
    END IF;
  END LOOP;
END;
$full_thinking_cap564$;
NOTIFY pgrst,'reload schema';
COMMIT;
