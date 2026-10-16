-- Runtime deployment is separate from immutable round grading/cache identity.
-- Registration is an operator action after verification of image composition.
-- No round, terminal result, accepted evidence, or cache row is rewritten.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;

CREATE TABLE IF NOT EXISTS public.lab_arena_reviewer_runtimes (
  grading_image_digest TEXT NOT NULL CHECK (grading_image_digest ~ '^sha256:[0-9a-f]{64}$'),
  grading_image_reference TEXT NOT NULL CHECK (length(grading_image_reference) <= 512
    AND grading_image_reference ~ '^[a-z0-9][a-z0-9._:/-]*@sha256:[0-9a-f]{64}$'
    AND right(grading_image_reference,72)='@'||grading_image_digest),
  runtime_image_digest TEXT NOT NULL CHECK (runtime_image_digest ~ '^sha256:[0-9a-f]{64}$'),
  runtime_image_reference TEXT NOT NULL CHECK (length(runtime_image_reference) <= 512
    AND runtime_image_reference ~ '^[a-z0-9][a-z0-9._:/-]*@sha256:[0-9a-f]{64}$'
    AND right(runtime_image_reference,72)='@'||runtime_image_digest),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  PRIMARY KEY (grading_image_digest,grading_image_reference)
);
ALTER TABLE public.lab_arena_reviewer_runtimes OWNER TO lab_arena_owner;
ALTER TABLE public.lab_arena_reviewer_runtimes ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON public.lab_arena_reviewer_runtimes
  FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;

ALTER TABLE public.lab_arena_runs ADD COLUMN IF NOT EXISTS reviewer_runtime_doc JSONB;

CREATE OR REPLACE FUNCTION public.lab_arena_register_reviewer_runtime_v1(
  p_grading_image_digest TEXT,p_grading_image_reference TEXT,
  p_runtime_image_digest TEXT,p_runtime_image_reference TEXT
) RETURNS JSONB
LANGUAGE plpgsql VOLATILE SECURITY DEFINER
SET search_path=pg_catalog,public AS $register_runtime$
DECLARE v_mapping public.lab_arena_reviewer_runtimes;
BEGIN
  INSERT INTO public.lab_arena_reviewer_runtimes AS deployed (
    grading_image_digest,grading_image_reference,runtime_image_digest,runtime_image_reference
  ) VALUES (p_grading_image_digest,p_grading_image_reference,p_runtime_image_digest,p_runtime_image_reference)
  ON CONFLICT (grading_image_digest,grading_image_reference) DO UPDATE
    SET runtime_image_digest=EXCLUDED.runtime_image_digest,
        runtime_image_reference=EXCLUDED.runtime_image_reference,
        updated_at=clock_timestamp()
    WHERE (deployed.runtime_image_digest,deployed.runtime_image_reference)
      IS DISTINCT FROM (EXCLUDED.runtime_image_digest,EXCLUDED.runtime_image_reference);
  SELECT * INTO STRICT v_mapping FROM public.lab_arena_reviewer_runtimes
    WHERE grading_image_digest=p_grading_image_digest
      AND grading_image_reference=p_grading_image_reference;
  RETURN to_jsonb(v_mapping);
END;
$register_runtime$;
ALTER FUNCTION public.lab_arena_register_reviewer_runtime_v1(TEXT,TEXT,TEXT,TEXT) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_register_reviewer_runtime_v1(TEXT,TEXT,TEXT,TEXT)
  FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;
GRANT EXECUTE ON FUNCTION public.lab_arena_register_reviewer_runtime_v1(TEXT,TEXT,TEXT,TEXT)
  TO lab_arena_service;

CREATE OR REPLACE FUNCTION public.lab_arena__pin_reviewer_runtime_v1()
RETURNS TRIGGER LANGUAGE plpgsql SECURITY DEFINER
SET search_path=pg_catalog,public AS $pin_runtime$
DECLARE
  v_round public.lab_arena_rounds;
  v_mapping public.lab_arena_reviewer_runtimes;
BEGIN
  IF NEW.kind='score' AND NEW.status='leased'
     AND (TG_OP='INSERT' OR NEW.lease_generation IS DISTINCT FROM OLD.lease_generation) THEN
    IF (TG_OP='UPDATE' AND OLD.status IN ('accepted','failed'))
       OR NEW.lease_generation < 1
       OR NEW.runner_hotkey IS NULL OR NEW.lease_token_hash IS NULL
       OR NEW.claim_response->>'run_id' IS DISTINCT FROM NEW.run_id
       OR NEW.claim_response->'lease_generation' IS DISTINCT FROM to_jsonb(NEW.lease_generation) THEN
      RAISE EXCEPTION 'reviewer runtime requires an ordinary exact score lease' USING ERRCODE='55000';
    END IF;
    SELECT * INTO STRICT v_round FROM public.lab_arena_rounds WHERE round_id=NEW.round_id;
    IF v_round.status IN ('published','cancelled') THEN
      RAISE EXCEPTION 'reviewer runtime cannot lease a terminal round' USING ERRCODE='55000';
    END IF;
    -- This executes inside the ordinary claim UPDATE, before its commit.
    -- Mapping changes cannot replace the image for an existing generation.
    SELECT * INTO v_mapping FROM public.lab_arena_reviewer_runtimes
      WHERE grading_image_digest=v_round.configuration_doc->>'scorer_image_digest'
        AND grading_image_reference=v_round.configuration_doc->>'scorer_image_reference';
    NEW.reviewer_runtime_doc := NULL;
    NEW.claim_response := NEW.claim_response-'reviewer_runtime_doc';
    IF FOUND THEN
      NEW.reviewer_runtime_doc := jsonb_build_object(
        'schema_version','leadpoet.lab_arena.reviewer_runtime.v1',
        'run_id',NEW.run_id,'lease_generation',NEW.lease_generation,
        'grading_image_digest',v_mapping.grading_image_digest,
        'grading_image_reference',v_mapping.grading_image_reference,
        'runtime_image_digest',v_mapping.runtime_image_digest,
        'runtime_image_reference',v_mapping.runtime_image_reference);
      NEW.claim_response := NEW.claim_response || jsonb_build_object(
        'reviewer_runtime_doc',NEW.reviewer_runtime_doc);
    END IF;
  ELSIF TG_OP='INSERT' THEN
    IF NEW.reviewer_runtime_doc IS NOT NULL THEN
      RAISE EXCEPTION 'reviewer runtime cannot be supplied without a new score lease' USING ERRCODE='55000';
    END IF;
  ELSIF NEW.reviewer_runtime_doc IS DISTINCT FROM OLD.reviewer_runtime_doc THEN
    RAISE EXCEPTION 'reviewer runtime lease pin is immutable' USING ERRCODE='55000';
  END IF;
  RETURN NEW;
END;
$pin_runtime$;
ALTER FUNCTION public.lab_arena__pin_reviewer_runtime_v1() OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena__pin_reviewer_runtime_v1()
  FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;
DROP TRIGGER IF EXISTS lab_arena_runs_reviewer_runtime ON public.lab_arena_runs;
CREATE TRIGGER lab_arena_runs_reviewer_runtime
  BEFORE INSERT OR UPDATE ON public.lab_arena_runs
  FOR EACH ROW EXECUTE FUNCTION public.lab_arena__pin_reviewer_runtime_v1();

REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;
NOTIFY pgrst,'reload schema';
COMMIT;
