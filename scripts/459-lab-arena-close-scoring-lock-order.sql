-- Close scoring in the same lock order as claims: claim-control, then round.
-- Keep the status transition trigger as the final operator-hold guard.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

DO $close_scoring_lock_order$
DECLARE
  v_signature pg_catalog.regprocedure :=
    'public.lab_arena_close_scoring(text,smallint)'::pg_catalog.regprocedure;
  v_existing TEXT;
  v_updated TEXT;
  v_old TEXT := $old$  IF p_stage NOT IN (1, 2) THEN
    RAISE EXCEPTION 'lab_arena_stage_invalid' USING ERRCODE = '22023';
  END IF;
  SELECT * INTO v_round$old$;
  v_new TEXT := $new$  IF p_stage NOT IN (1, 2) THEN
    RAISE EXCEPTION 'lab_arena_stage_invalid' USING ERRCODE = '22023';
  END IF;
  -- Serialize with claims before locking the round row. The status trigger
  -- remains the final, fail-closed operator-hold check.
  PERFORM pg_catalog.pg_advisory_xact_lock(
    pg_catalog.hashtextextended('lab-arena-claim-control', 0));
  SELECT * INTO v_round$new$;
  v_preimage TEXT := '3bc8179fb87fdeadf200d4f3f53a9550fc2e56ee781584a327b1e6ab00b7bd50';
  v_postimage TEXT := '7f6069b56d9c8a6c8f1b9aa8803e61fb08aaf3005c9e852d81fb2bd89493d14b';
BEGIN
  SELECT pg_catalog.pg_get_functiondef(v_signature) INTO v_existing;
  IF v_existing IS NULL THEN
    RAISE EXCEPTION 'Arena close-scoring function missing' USING ERRCODE='55000';
  END IF;
  IF pg_catalog.encode(extensions.digest(v_existing,'sha256'),'hex') = v_postimage THEN
    RETURN;
  END IF;
  IF pg_catalog.encode(extensions.digest(v_existing,'sha256'),'hex') <> v_preimage
     OR pg_catalog.length(v_existing) - pg_catalog.length(pg_catalog.replace(v_existing,v_old,''))
        <> pg_catalog.length(v_old) THEN
    RAISE EXCEPTION 'Arena close-scoring lock-order preimage differs' USING ERRCODE='55000';
  END IF;
  v_updated := pg_catalog.replace(v_existing,v_old,v_new);
  IF pg_catalog.encode(extensions.digest(v_updated,'sha256'),'hex') <> v_postimage THEN
    RAISE EXCEPTION 'Arena close-scoring lock-order postimage differs' USING ERRCODE='55000';
  END IF;
  EXECUTE v_updated;
  IF pg_catalog.encode(extensions.digest(pg_catalog.pg_get_functiondef(v_signature),'sha256'),'hex')
     <> v_postimage THEN
    RAISE EXCEPTION 'Arena close-scoring lock-order readback differs' USING ERRCODE='55000';
  END IF;
END;
$close_scoring_lock_order$;
NOTIFY pgrst, 'reload schema';
COMMIT;
