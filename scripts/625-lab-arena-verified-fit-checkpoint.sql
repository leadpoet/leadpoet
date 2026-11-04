-- Typed positive company-fit evidence in the existing private signal authority.
-- Preserve normal signal positions, costs, provenance, leases, and historical rows.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';
DO $fit625_constraint$
DECLARE definition TEXT;
BEGIN
 SELECT pg_get_constraintdef(oid) INTO definition FROM pg_constraint
 WHERE conrelid='public.lab_arena_signal_judgment_uses'::regclass
  AND conname='lab_arena_signal_judgment_uses_signal_index_check';
 IF definition='CHECK (((signal_index >= ''-1''::integer) AND (signal_index <= 99)))' THEN RETURN; END IF;
 IF definition IS DISTINCT FROM 'CHECK (((signal_index >= 0) AND (signal_index <= 99)))' THEN
  RAISE EXCEPTION 'Fit625 signal position constraint differs' USING ERRCODE='55000'; END IF;
 ALTER TABLE public.lab_arena_signal_judgment_uses DROP CONSTRAINT lab_arena_signal_judgment_uses_signal_index_check;
 ALTER TABLE public.lab_arena_signal_judgment_uses ADD CONSTRAINT lab_arena_signal_judgment_uses_signal_index_check CHECK (signal_index BETWEEN -1 AND 99);
END;
$fit625_constraint$;
DO $fit625_claim$
DECLARE signature CONSTANT TEXT:='public.lab_arena_claim_signal_judgment_v1(text,text,integer,integer,jsonb)';
 definition TEXT; updated TEXT; metadata JSONB; identity JSONB;
BEGIN
 SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc',
  jsonb_build_array(pg_get_userbyid(p.proowner),p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig,l.lanname)
 INTO definition,metadata,identity FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang
 WHERE p.oid=to_regprocedure(signature);
 IF identity IS DISTINCT FROM '["lab_arena_owner","{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}",true,"v",["search_path=pg_catalog, public"],"plpgsql"]'::JSONB THEN
  RAISE EXCEPTION 'Fit625 security shape differs' USING ERRCODE='55000'; END IF;
 IF encode(extensions.digest(definition,'sha256'),'hex')='63a7c34ed921254c562943507cb527b4ece4ad9079789d7092972206d5a420c5' THEN RETURN; END IF;
 IF encode(extensions.digest(definition,'sha256'),'hex') IS DISTINCT FROM '47a767d79cc8a47ad3d653f940e5987ce1ae210701ec41f97e781b498b1ce250' THEN
  RAISE EXCEPTION 'Fit625 preimage differs' USING ERRCODE='55000'; END IF;
 updated:=definition;
  IF (length(updated)-length(replace(updated,$old6250$p_signal_index NOT BETWEEN 0 AND 99$old6250$,'')))<>length($old6250$p_signal_index NOT BETWEEN 0 AND 99$old6250$) THEN RAISE EXCEPTION 'Fit625 patch shape differs' USING ERRCODE='55000'; END IF;
  updated:=replace(updated,$old6250$p_signal_index NOT BETWEEN 0 AND 99$old6250$,$new6250$(p_signal_index NOT BETWEEN -1 AND 99
    OR (p_signal_index=-1) IS DISTINCT FROM COALESCE(p_scope->>'checkpoint_kind'='verified_company_fit_v1',FALSE)
    OR (p_signal_index>=0 AND p_scope ? 'checkpoint_kind'))$new6250$);
  IF (length(updated)-length(replace(updated,$old6251$ v_key:=p_scope$old6251$,'')))<>length($old6251$ v_key:=p_scope$old6251$) THEN RAISE EXCEPTION 'Fit625 patch shape differs' USING ERRCODE='55000'; END IF;
  updated:=replace(updated,$old6251$ v_key:=p_scope$old6251$,$new6251$ IF p_signal_index=-1 AND (
  jsonb_typeof(p_scope->'checkpoint_context') IS DISTINCT FROM 'object'
  OR p_scope#>>'{checkpoint_context,assignment_id}' IS DISTINCT FROM v_run.assignment_id
  OR p_scope#>>'{checkpoint_context,scored_run_id}' IS DISTINCT FROM v_run.scored_run_id
  OR p_scope#>'{checkpoint_context,scoring_generation_floor}' IS DISTINCT FROM to_jsonb(v_run.scoring_generation_floor)
  OR COALESCE(p_scope#>>'{checkpoint_context,execution_output_hash}','') !~ '^sha256:[0-9a-f]{64}$'
  OR COALESCE(p_scope#>>'{checkpoint_context,scoring_input_hash}','') !~ '^sha256:[0-9a-f]{64}$'
  OR NOT EXISTS (SELECT 1 FROM public.lab_arena_runs e
    JOIN public.lab_arena_rounds r ON r.round_id=v_run.round_id
    WHERE e.run_id=v_run.scored_run_id AND e.kind='execute' AND e.status='accepted'
      AND e.round_id=v_run.round_id AND e.submission_id=v_run.submission_id
      AND e.miner_hotkey=v_run.miner_hotkey AND e.stage=v_run.stage AND e.icp_position=v_run.icp_position
      AND e.output_ref=p_scope#>>'{checkpoint_context,execution_output_ref}'
      AND p_scope#>>'{checkpoint_context,runtime_image_digest}'=COALESCE(
        v_run.reviewer_runtime_doc->>'runtime_image_digest',r.configuration_doc->>'scorer_image_digest')
      AND p_scope#>>'{checkpoint_context,runtime_image_reference}'=COALESCE(
        v_run.reviewer_runtime_doc->>'runtime_image_reference',r.configuration_doc->>'scorer_image_reference'))
 ) THEN RAISE EXCEPTION 'lab_arena_fit_checkpoint_context_invalid' USING ERRCODE='22023'; END IF;
 v_key:=p_scope$new6251$);
 IF encode(extensions.digest(updated,'sha256'),'hex')<>'63a7c34ed921254c562943507cb527b4ece4ad9079789d7092972206d5a420c5' THEN
  RAISE EXCEPTION 'Fit625 postimage differs' USING ERRCODE='55000'; END IF;
 EXECUTE updated;
 IF (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=to_regprocedure(signature)) IS DISTINCT FROM metadata
  OR encode(extensions.digest(pg_get_functiondef(to_regprocedure(signature)),'sha256'),'hex')<>'63a7c34ed921254c562943507cb527b4ece4ad9079789d7092972206d5a420c5' THEN
  RAISE EXCEPTION 'Fit625 readback differs' USING ERRCODE='55000'; END IF;
END;
$fit625_claim$;

DO $fit625_complete$
DECLARE signature CONSTANT TEXT:='public.lab_arena_complete_signal_judgment_v1(text,text,integer,integer,jsonb,jsonb,jsonb,text)';
 definition TEXT; updated TEXT; metadata JSONB; identity JSONB;
BEGIN
 SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc',
  jsonb_build_array(pg_get_userbyid(p.proowner),p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig,l.lanname)
 INTO definition,metadata,identity FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang
 WHERE p.oid=to_regprocedure(signature);
 IF identity IS DISTINCT FROM '["lab_arena_owner","{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}",true,"v",["search_path=pg_catalog, public"],"plpgsql"]'::JSONB THEN
  RAISE EXCEPTION 'Fit625 security shape differs' USING ERRCODE='55000'; END IF;
 IF encode(extensions.digest(definition,'sha256'),'hex')='eb40ae7ccb5b5791bcfc3ff0764922f401158dbe314275a0a421f1f22fc12c30' THEN RETURN; END IF;
 IF encode(extensions.digest(definition,'sha256'),'hex') IS DISTINCT FROM '71c7da21e23a7a6da67d00344cdfd250bb2b10a8ea8cd115145bcee20d642ff9' THEN
  RAISE EXCEPTION 'Fit625 preimage differs' USING ERRCODE='55000'; END IF;
 updated:=definition;
  IF (length(updated)-length(replace(updated,$old6250$p_signal_index NOT BETWEEN 0 AND 99$old6250$,'')))<>length($old6250$p_signal_index NOT BETWEEN 0 AND 99$old6250$) THEN RAISE EXCEPTION 'Fit625 patch shape differs' USING ERRCODE='55000'; END IF;
  updated:=replace(updated,$old6250$p_signal_index NOT BETWEEN 0 AND 99$old6250$,$new6250$(p_signal_index NOT BETWEEN -1 AND 99
    OR (p_signal_index=-1) IS DISTINCT FROM COALESCE(p_scope->>'checkpoint_kind'='verified_company_fit_v1',FALSE)
    OR (p_signal_index>=0 AND p_scope ? 'checkpoint_kind'))$new6250$);
  IF (length(updated)-length(replace(updated,$old6251$ v_ref:=jsonb_build_object$old6251$,'')))<>length($old6251$ v_ref:=jsonb_build_object$old6251$) THEN RAISE EXCEPTION 'Fit625 patch shape differs' USING ERRCODE='55000'; END IF;
  updated:=replace(updated,$old6251$ v_ref:=jsonb_build_object$old6251$,$new6251$ IF p_signal_index=-1 AND (
  jsonb_typeof(p_scope->'checkpoint_context') IS DISTINCT FROM 'object'
  OR p_scope#>>'{checkpoint_context,assignment_id}' IS DISTINCT FROM v_run.assignment_id
  OR p_scope#>>'{checkpoint_context,scored_run_id}' IS DISTINCT FROM v_run.scored_run_id
  OR p_scope#>'{checkpoint_context,scoring_generation_floor}' IS DISTINCT FROM to_jsonb(v_run.scoring_generation_floor)
  OR COALESCE(p_scope#>>'{checkpoint_context,execution_output_hash}','') !~ '^sha256:[0-9a-f]{64}$'
  OR COALESCE(p_scope#>>'{checkpoint_context,scoring_input_hash}','') !~ '^sha256:[0-9a-f]{64}$'
  OR NOT EXISTS (SELECT 1 FROM public.lab_arena_runs e
    JOIN public.lab_arena_rounds r ON r.round_id=v_run.round_id
    WHERE e.run_id=v_run.scored_run_id AND e.kind='execute' AND e.status='accepted'
      AND e.round_id=v_run.round_id AND e.submission_id=v_run.submission_id
      AND e.miner_hotkey=v_run.miner_hotkey AND e.stage=v_run.stage AND e.icp_position=v_run.icp_position
      AND e.output_ref=p_scope#>>'{checkpoint_context,execution_output_ref}'
      AND p_scope#>>'{checkpoint_context,runtime_image_digest}'=COALESCE(
        v_run.reviewer_runtime_doc->>'runtime_image_digest',r.configuration_doc->>'scorer_image_digest')
      AND p_scope#>>'{checkpoint_context,runtime_image_reference}'=COALESCE(
        v_run.reviewer_runtime_doc->>'runtime_image_reference',r.configuration_doc->>'scorer_image_reference'))
 ) THEN RAISE EXCEPTION 'lab_arena_fit_checkpoint_context_invalid' USING ERRCODE='22023'; END IF;
 v_ref:=jsonb_build_object$new6251$);
  IF (length(updated)-length(replace(updated,$old6252$ -- This immutable verdict carries no cost authority.$old6252$,'')))<>length($old6252$ -- This immutable verdict carries no cost authority.$old6252$) THEN RAISE EXCEPTION 'Fit625 patch shape differs' USING ERRCODE='55000'; END IF;
  updated:=replace(updated,$old6252$ -- This immutable verdict carries no cost authority.$old6252$,$new6252$ IF p_signal_index=-1 AND (
  octet_length(p_judgment::TEXT)>1048576
  OR p_judgment->>'schema_version' IS DISTINCT FROM 'leadpoet.lab_arena.verified_company_fit.v1'
  OR p_judgment#>>'{company_fit_receipt,gate}' IS DISTINCT FROM 'company_fit'
  OR p_judgment#>>'{company_fit_receipt,contract_id}' IS DISTINCT FROM 'company-fit-decision:v1'
  OR p_judgment#>>'{company_fit_receipt,contract_version}' IS DISTINCT FROM 'company-fit-decision:v1'
  OR p_judgment#>>'{company_fit_receipt,decision}' IS DISTINCT FROM 'match'
  OR p_judgment#>>'{company_fit_receipt,company_fit_decision}' IS DISTINCT FROM 'match'
  OR jsonb_typeof(p_judgment#>'{company_fit_receipt,company_fit_stage_required}') IS DISTINCT FROM 'boolean'
  OR p_judgment#>>'{company_fit_receipt,required_attribute_decision}' IS DISTINCT FROM 'match'
  OR COALESCE(p_judgment#>>'{company_fit_receipt,failure_class}','')<>''
  OR COALESCE(p_judgment#>>'{company_fit_receipt,failure_reason_code}','')<>''
  OR jsonb_typeof(p_judgment->'linkedin_profile_source') IS DISTINCT FROM 'object'
  OR jsonb_typeof(p_judgment->'matched_company_sources') IS DISTINCT FROM 'object'
  OR jsonb_typeof(p_judgment->'required_attribute_sources') IS DISTINCT FROM 'object'
  OR jsonb_typeof(p_judgment#>'{company_fit_receipt,company_fit_dimensions}') IS DISTINCT FROM 'object'
  OR NOT (p_judgment#>'{company_fit_receipt,company_fit_dimensions}') ?& ARRAY['identity','employee_size','industry','geography','stage']
  OR EXISTS (SELECT 1 FROM unnest(ARRAY['identity','employee_size','industry','geography']) dimension
    WHERE p_judgment#>>ARRAY['company_fit_receipt','company_fit_dimensions',dimension] IS DISTINCT FROM 'match'
      OR p_judgment#>>ARRAY['company_fit_receipt','dimension_evidence',dimension,'decision'] IS DISTINCT FROM 'match')
  OR (p_judgment#>'{company_fit_receipt,company_fit_stage_required}'='true'::JSONB
    AND (p_judgment#>>'{company_fit_receipt,company_fit_dimensions,stage}' IS DISTINCT FROM 'match'
      OR p_judgment#>>'{company_fit_receipt,dimension_evidence,stage,decision}' IS DISTINCT FROM 'match'))
  OR p_judgment#>>'{company_fit_receipt,dimension_evidence,identity,web_identity_receipt,decision}' IS DISTINCT FROM 'match'
  OR COALESCE(p_judgment#>>'{company_fit_receipt,dimension_evidence,identity,web_identity_receipt,observed_name}','')=''
  OR COALESCE(p_judgment#>>'{company_fit_receipt,dimension_evidence,identity,web_identity_receipt,observed_domain}','')=''
 ) THEN RAISE EXCEPTION 'lab_arena_fit_checkpoint_judgment_invalid' USING ERRCODE='22023'; END IF;
 -- This immutable verdict carries no cost authority.$new6252$);
 IF encode(extensions.digest(updated,'sha256'),'hex')<>'eb40ae7ccb5b5791bcfc3ff0764922f401158dbe314275a0a421f1f22fc12c30' THEN
  RAISE EXCEPTION 'Fit625 postimage differs' USING ERRCODE='55000'; END IF;
 EXECUTE updated;
 IF (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=to_regprocedure(signature)) IS DISTINCT FROM metadata
  OR encode(extensions.digest(pg_get_functiondef(to_regprocedure(signature)),'sha256'),'hex')<>'eb40ae7ccb5b5791bcfc3ff0764922f401158dbe314275a0a421f1f22fc12c30' THEN
  RAISE EXCEPTION 'Fit625 readback differs' USING ERRCODE='55000'; END IF;
END;
$fit625_complete$;
COMMIT;
