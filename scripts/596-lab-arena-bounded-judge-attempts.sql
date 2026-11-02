-- Bound the existing score queue to five attempts per assignment/generation.
-- A supersession retains its physical attempt sequence but gets a new bound.
-- Do not change continuation proofs used by already leased judges, their settlement,
-- accepted judgments, execution retries, backoff, or frozen round results.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';
DO $bounded_judge_attempts596$
DECLARE target RECORD; patch RECORD; function_oid OID;
  definition TEXT; updated TEXT; metadata JSONB; identity JSONB;
BEGIN
  FOR target IN SELECT * FROM (VALUES
    ('public.lab_arena_claim_assignment(text,text,integer,integer,text[],text,text,text,integer)','5c7c9194d06d1d1920e24b8e85426d403cbd4b6f398afbcd6fe5f86f0dc5dacd','fd89e06bf90317d579504869210632457f6fb9bcb79aabca16048e5bc119bc22'),
    ('public.lab_arena_complete_attempt(text,text,jsonb,text,text)','20da61fe353b6b7ceba81eb6deabf6597e1b6db42efd5ceef0219539d004171b','1f0f2372abcbaee30424af6ba76258d39e081aa8d197d82707b550ad0ad4abf0'),
    ('public.lab_arena_expire_leases(text)','b14f3018d626ae76fdd4151d9106161a89cfc264ded814c7abb7d49c48da42c4','2e3f75408b96884640580e8c4f4cd8855750a9fd66a0f9acd95f29b113770307')
  ) AS targets(signature,preimage,postimage) LOOP
    function_oid:=to_regprocedure(target.signature);
    SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc',
      jsonb_build_array(pg_get_userbyid(p.proowner),p.proacl::TEXT,
        p.prosecdef,p.provolatile,p.proconfig,l.lanname)
      INTO definition,metadata,identity
    FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang WHERE p.oid=function_oid;
    IF definition IS NULL OR identity IS DISTINCT FROM jsonb_build_array(
      'lab_arena_owner','{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}',
      TRUE,'v',ARRAY['search_path=pg_catalog, public']::TEXT[],'plpgsql') THEN
      RAISE EXCEPTION 'bounded judge attempts security differs: %',target.signature USING ERRCODE='55000';
    END IF;
    IF encode(extensions.digest(definition,'sha256'),'hex')=target.postimage THEN CONTINUE; END IF;
    IF encode(extensions.digest(definition,'sha256'),'hex')<>target.preimage THEN
      RAISE EXCEPTION 'bounded judge attempts preimage differs: %',target.signature USING ERRCODE='55000';
    END IF;
    updated:=definition;
    FOR patch IN SELECT * FROM (VALUES
      ('public.lab_arena_claim_assignment(text,text,integer,integer,text[],text,text,text,integer)',$old596$    AND runs.round_id = p_round_id
    AND (runs.kind<>'score' OR runs.scoring_generation_floor=v_round.scoring_generation_floor)$old596$,$new596$    AND runs.round_id = p_round_id
    AND (runs.kind<>'score' OR runs.attempt - (
      SELECT min(generation_root.attempt) FROM public.lab_arena_runs generation_root
      WHERE generation_root.assignment_id=runs.assignment_id
        AND generation_root.scoring_generation_floor=runs.scoring_generation_floor) < 5)
    AND (runs.kind<>'score' OR runs.scoring_generation_floor=v_round.scoring_generation_floor)$new596$),
      ('public.lab_arena_complete_attempt(text,text,jsonb,text,text)',$old596$  IF v_status = 'failed'
     AND ((v_run.kind='score' AND v_run.attempt<2147483647$old596$,$new596$  IF v_status = 'failed'
     AND (v_run.kind<>'score' OR v_run.attempt - (
       SELECT min(generation_root.attempt) FROM public.lab_arena_runs generation_root
       WHERE generation_root.assignment_id=v_run.assignment_id
         AND generation_root.scoring_generation_floor=v_run.scoring_generation_floor) < 4)
     AND ((v_run.kind='score' AND v_run.attempt<2147483647$new596$),
      ('public.lab_arena_expire_leases(text)',$old596$    IF (
      (v_run.kind='score' AND v_run.attempt<2147483647$old596$,$new596$    IF (v_run.kind<>'score' OR v_run.attempt - (
      SELECT min(generation_root.attempt) FROM public.lab_arena_runs generation_root
      WHERE generation_root.assignment_id=v_run.assignment_id
        AND generation_root.scoring_generation_floor=v_run.scoring_generation_floor) < 4) AND (
      (v_run.kind='score' AND v_run.attempt<2147483647$new596$),
      ('public.lab_arena_expire_leases(text)',$old596$      AND candidate.status='failed' AND candidate.attempt<2147483647$old596$,$new596$      AND candidate.status='failed' AND candidate.attempt<2147483647
      AND candidate.attempt - (
        SELECT min(generation_root.attempt) FROM public.lab_arena_runs generation_root
        WHERE generation_root.assignment_id=candidate.assignment_id
          AND generation_root.scoring_generation_floor=candidate.scoring_generation_floor) < 4$new596$)
    ) AS patches(signature,old_text,new_text) WHERE signature=target.signature LOOP
      IF length(updated)-length(replace(updated,patch.old_text,''))<>length(patch.old_text) THEN
        RAISE EXCEPTION 'bounded judge attempts anchor differs: %',target.signature USING ERRCODE='55000';
      END IF;
      updated:=replace(updated,patch.old_text,patch.new_text);
    END LOOP;
    IF encode(extensions.digest(updated,'sha256'),'hex')<>target.postimage THEN
      RAISE EXCEPTION 'bounded judge attempts postimage differs: %',target.signature USING ERRCODE='55000';
    END IF;
    EXECUTE updated;
    IF encode(extensions.digest(pg_get_functiondef(function_oid),'sha256'),'hex')<>target.postimage
      OR (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=function_oid) IS DISTINCT FROM metadata THEN
      RAISE EXCEPTION 'bounded judge attempts readback differs: %',target.signature USING ERRCODE='55000';
    END IF;
  END LOOP;
END;
$bounded_judge_attempts596$;
NOTIFY pgrst,'reload schema';
COMMIT;
