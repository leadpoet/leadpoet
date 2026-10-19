-- A score lease may freeze one independently terminal intent signal. The
-- database owns reservations, compatible authority, and immutable provenance.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

-- This is an atomic one-time migration. A repeat or partial schema must stop
-- before any write; the release helper verifies the installed SQL by hash.
DO $signal_preflight$
BEGIN
 IF to_regclass('public.lab_arena_signal_judgments') IS NOT NULL
  OR to_regclass('public.lab_arena_signal_judgment_reservations') IS NOT NULL
  OR to_regclass('public.lab_arena_signal_judgment_uses') IS NOT NULL
  OR to_regprocedure('public.lab_arena_signal_authority_exclusions_v1(text)') IS NOT NULL
  OR to_regprocedure('public.lab_arena__signal_lease_valid_v1(public.lab_arena_runs,text)') IS NOT NULL
  OR to_regprocedure('public.lab_arena_claim_signal_judgment_v1(text,text,integer,integer,jsonb)') IS NOT NULL
  OR to_regprocedure('public.lab_arena_complete_signal_judgment_v1(text,text,integer,integer,jsonb,jsonb,jsonb,text)') IS NOT NULL
  OR to_regprocedure('public.lab_arena_signal_judgment_lease_sync_v1()') IS NOT NULL
  OR EXISTS (SELECT 1 FROM pg_trigger WHERE tgname='lab_arena_signal_judgment_lease_sync') THEN
  RAISE EXCEPTION 'signal migration503 already or partly installed' USING ERRCODE='55000';
 END IF;
END; $signal_preflight$;
GRANT CREATE ON SCHEMA public TO lab_arena_owner;

CREATE TABLE public.lab_arena_signal_judgments (
  cache_key TEXT NOT NULL CHECK (cache_key ~ '^sha256:[0-9a-f]{64}$'),
  authority_slot INTEGER NOT NULL CHECK (authority_slot >= 0),
  signal_input_hash TEXT NOT NULL CHECK (signal_input_hash ~ '^sha256:[0-9a-f]{64}$'),
  scope_doc JSONB NOT NULL,
  judgment_hash TEXT NOT NULL CHECK (judgment_hash ~ '^sha256:[0-9a-f]{64}$'),
  judgment_doc JSONB NOT NULL,
  source_score_run_id TEXT NOT NULL REFERENCES public.lab_arena_runs(run_id),
  source_scored_run_id TEXT NOT NULL REFERENCES public.lab_arena_runs(run_id),
  source_runner_hotkey TEXT NOT NULL,
  source_claim_request_hash TEXT NOT NULL,
  source_lease_generation BIGINT NOT NULL CHECK (source_lease_generation >= 1),
  runner_authority_exclusions JSONB NOT NULL CHECK (jsonb_typeof(runner_authority_exclusions) = 'array'),
  created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  PRIMARY KEY (cache_key, authority_slot),
  UNIQUE (source_score_run_id, cache_key),
  CHECK (jsonb_typeof(scope_doc) = 'object' AND scope_doc->>'cache_key' = cache_key
    AND scope_doc->>'signal_input_hash' = signal_input_hash),
  CHECK (jsonb_typeof(judgment_doc) = 'object'),
  CHECK (runner_authority_exclusions @> jsonb_build_array(source_runner_hotkey))
);
ALTER TABLE public.lab_arena_signal_judgments OWNER TO lab_arena_owner;
CREATE TRIGGER lab_arena_signal_judgments_append_only BEFORE UPDATE OR DELETE
  ON public.lab_arena_signal_judgments FOR EACH ROW
  EXECUTE FUNCTION public.lab_arena_append_only_v1();
ALTER TABLE public.lab_arena_signal_judgments ENABLE ROW LEVEL SECURITY;
CREATE POLICY lab_arena_signal_judgments_read ON public.lab_arena_signal_judgments
  FOR SELECT TO lab_arena_service USING (TRUE);
REVOKE ALL ON public.lab_arena_signal_judgments FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;
GRANT SELECT ON public.lab_arena_signal_judgments TO lab_arena_service;

CREATE TABLE public.lab_arena_signal_judgment_reservations (
  cache_key TEXT PRIMARY KEY CHECK (cache_key ~ '^sha256:[0-9a-f]{64}$'),
  authority_slot INTEGER NOT NULL CHECK (authority_slot >= 0),
  signal_input_hash TEXT NOT NULL,
  run_id TEXT NOT NULL REFERENCES public.lab_arena_runs(run_id),
  lease_generation BIGINT NOT NULL CHECK (lease_generation >= 1),
  lease_expires_at TIMESTAMPTZ NOT NULL,
  UNIQUE (cache_key, authority_slot), UNIQUE (run_id, cache_key)
);
ALTER TABLE public.lab_arena_signal_judgment_reservations OWNER TO lab_arena_owner;
REVOKE ALL ON public.lab_arena_signal_judgment_reservations FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;

CREATE TABLE public.lab_arena_signal_judgment_uses (
  run_id TEXT NOT NULL REFERENCES public.lab_arena_runs(run_id),
  company_index INTEGER NOT NULL CHECK (company_index BETWEEN 0 AND 4),
  signal_index INTEGER NOT NULL CHECK (signal_index BETWEEN 0 AND 99),
  cache_key TEXT NOT NULL,
  authority_slot INTEGER NOT NULL,
  signal_input_hash TEXT NOT NULL,
  source_score_run_id TEXT NOT NULL,
  judgment_hash TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  PRIMARY KEY (run_id, company_index, signal_index),
  FOREIGN KEY (cache_key, authority_slot) REFERENCES public.lab_arena_signal_judgments(cache_key, authority_slot)
);
ALTER TABLE public.lab_arena_signal_judgment_uses OWNER TO lab_arena_owner;
CREATE TRIGGER lab_arena_signal_judgment_uses_append_only BEFORE UPDATE OR DELETE
  ON public.lab_arena_signal_judgment_uses FOR EACH ROW
  EXECUTE FUNCTION public.lab_arena_append_only_v1();
ALTER TABLE public.lab_arena_signal_judgment_uses ENABLE ROW LEVEL SECURITY;
CREATE POLICY lab_arena_signal_judgment_uses_read ON public.lab_arena_signal_judgment_uses
  FOR SELECT TO lab_arena_service USING (TRUE);
REVOKE ALL ON public.lab_arena_signal_judgment_uses FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;
GRANT SELECT ON public.lab_arena_signal_judgment_uses TO lab_arena_service;

CREATE FUNCTION public.lab_arena_signal_authority_exclusions_v1(p_run_id TEXT)
RETURNS JSONB LANGUAGE sql STABLE SECURITY DEFINER SET search_path = pg_catalog, public AS $$
 SELECT CASE WHEN round_row.configuration_doc#>>'{scorer_policy,signal_judgment_policy}'
       ='signal_judgment_v1' THEN (
   SELECT COALESCE(jsonb_agg(DISTINCT hotkey ORDER BY hotkey),'[]'::jsonb)
   FROM (
     SELECT jsonb_array_elements_text(score.claim_response->'runner_authority_exclusions') AS hotkey
     UNION
     SELECT jsonb_array_elements_text(j.runner_authority_exclusions) AS hotkey
     FROM public.lab_arena_signal_judgment_uses u
     JOIN public.lab_arena_signal_judgments j
      ON j.cache_key=u.cache_key AND j.authority_slot=u.authority_slot
     WHERE u.run_id=p_run_id
   ) sources
 ) ELSE score.claim_response->'runner_authority_exclusions' END
 FROM public.lab_arena_runs score
 JOIN public.lab_arena_rounds round_row ON round_row.round_id=score.round_id
 WHERE score.run_id=p_run_id;
$$;
ALTER FUNCTION public.lab_arena_signal_authority_exclusions_v1(TEXT) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_signal_authority_exclusions_v1(TEXT) FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;
GRANT EXECUTE ON FUNCTION public.lab_arena_signal_authority_exclusions_v1(TEXT) TO lab_arena_service;

CREATE FUNCTION public.lab_arena__signal_lease_valid_v1(p_run public.lab_arena_runs, p_lease_hash TEXT)
RETURNS BOOLEAN LANGUAGE sql STABLE SECURITY DEFINER SET search_path = pg_catalog, public AS $$
 SELECT p_run.kind='score' AND p_run.status='leased'
  AND p_run.lease_token_hash=p_lease_hash
  AND p_run.lease_expires_at>clock_timestamp()
  AND EXISTS (SELECT 1 FROM public.lab_arena_rounds r WHERE r.round_id=p_run.round_id
    AND r.configuration_doc#>>'{scorer_policy,signal_judgment_policy}'='signal_judgment_v1'
    AND r.configuration_doc->>'company_judgment_policy'='company_judgment_v1');
$$;
ALTER FUNCTION public.lab_arena__signal_lease_valid_v1(public.lab_arena_runs,TEXT) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena__signal_lease_valid_v1(public.lab_arena_runs,TEXT) FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;

CREATE FUNCTION public.lab_arena_claim_signal_judgment_v1(
  p_run_id TEXT, p_lease_token_hash TEXT, p_company_index INTEGER, p_signal_index INTEGER,
  p_scope JSONB
) RETURNS JSONB LANGUAGE plpgsql VOLATILE SECURITY DEFINER SET search_path = pg_catalog, public AS $$
DECLARE v_run public.lab_arena_runs; v_hit public.lab_arena_signal_judgments;
 v_use public.lab_arena_signal_judgment_uses; v_res public.lab_arena_signal_judgment_reservations;
 v_slot INTEGER; v_key TEXT; v_hash TEXT; v_inserted INTEGER; v_reference JSONB;
BEGIN
 SELECT * INTO v_run FROM public.lab_arena_runs WHERE run_id=p_run_id FOR UPDATE;
 IF NOT FOUND OR NOT public.lab_arena__signal_lease_valid_v1(v_run,p_lease_token_hash)
  OR p_company_index NOT BETWEEN 0 AND 4 OR p_signal_index NOT BETWEEN 0 AND 99
  OR jsonb_typeof(p_scope) IS DISTINCT FROM 'object' THEN
   RAISE EXCEPTION 'lab_arena_signal_claim_invalid' USING ERRCODE='22023'; END IF;
 v_key:=p_scope->>'cache_key'; v_hash:=p_scope->>'signal_input_hash';
 IF COALESCE(v_key,'') !~ '^sha256:[0-9a-f]{64}$' OR COALESCE(v_hash,'') !~ '^sha256:[0-9a-f]{64}$'
  OR p_scope->>'schema_version' IS DISTINCT FROM 'leadpoet.lab_arena.signal_judgment_scope.v1'
  OR p_scope->>'round_id' IS DISTINCT FROM v_run.round_id
  OR p_scope->>'signal_judgment_policy' IS DISTINCT FROM 'signal_judgment_v1'
  OR EXISTS (SELECT 1 FROM public.lab_arena_rounds r WHERE r.round_id=v_run.round_id
    AND (p_scope->>'evaluation_date' IS DISTINCT FROM r.evaluation_date
      OR p_scope->>'scorer_image_digest' IS DISTINCT FROM r.configuration_doc->>'scorer_image_digest'
      OR p_scope->>'scorer_image_reference' IS DISTINCT FROM r.configuration_doc->>'scorer_image_reference')) THEN
   RAISE EXCEPTION 'lab_arena_signal_scope_invalid' USING ERRCODE='22023'; END IF;
 SELECT * INTO v_use FROM public.lab_arena_signal_judgment_uses
  WHERE run_id=p_run_id AND company_index=p_company_index AND signal_index=p_signal_index;
 IF FOUND THEN
  IF v_use.cache_key<>v_key OR v_use.signal_input_hash<>v_hash THEN
   RAISE EXCEPTION 'lab_arena_signal_use_conflict' USING ERRCODE='23505'; END IF;
  SELECT * INTO STRICT v_hit FROM public.lab_arena_signal_judgments
   WHERE cache_key=v_use.cache_key AND authority_slot=v_use.authority_slot;
  RETURN jsonb_build_object('status','hit','reference',jsonb_build_object(
   'schema_version','leadpoet.lab_arena.signal_judgment_reference.v1',
   'cache_key',v_key,'signal_input_hash',v_hash,'authority_slot',v_hit.authority_slot),
   'judgment',v_hit.judgment_doc);
 END IF;
 SELECT * INTO v_hit FROM public.lab_arena_signal_judgments j WHERE j.cache_key=v_key
  AND j.signal_input_hash=v_hash
  AND NOT j.runner_authority_exclusions @> jsonb_build_array(v_run.miner_hotkey)
  ORDER BY j.authority_slot LIMIT 1;
 IF FOUND THEN
  INSERT INTO public.lab_arena_signal_judgment_uses
   (run_id,company_index,signal_index,cache_key,authority_slot,signal_input_hash,source_score_run_id,judgment_hash)
   VALUES (p_run_id,p_company_index,p_signal_index,v_key,v_hit.authority_slot,v_hash,
    v_hit.source_score_run_id,v_hit.judgment_hash);
  RETURN jsonb_build_object('status','hit','reference',jsonb_build_object(
   'schema_version','leadpoet.lab_arena.signal_judgment_reference.v1',
   'cache_key',v_key,'signal_input_hash',v_hash,'authority_slot',v_hit.authority_slot),
   'judgment',v_hit.judgment_doc);
 END IF;
 DELETE FROM public.lab_arena_signal_judgment_reservations reservation
  WHERE reservation.cache_key=v_key AND reservation.lease_expires_at<=clock_timestamp()
   AND NOT EXISTS (
    SELECT 1 FROM public.lab_arena_runs owner_run
    WHERE owner_run.run_id=reservation.run_id AND owner_run.status='leased'
     AND owner_run.lease_generation=reservation.lease_generation
     AND owner_run.lease_expires_at>clock_timestamp()
   );
 SELECT * INTO v_res FROM public.lab_arena_signal_judgment_reservations WHERE cache_key=v_key;
 IF FOUND THEN
  IF v_res.run_id<>p_run_id OR v_res.lease_generation<>v_run.lease_generation THEN
   RETURN jsonb_build_object('status','busy'); END IF;
  v_slot:=v_res.authority_slot;
 ELSE
  SELECT COALESCE(MAX(authority_slot),-1)+1 INTO v_slot FROM public.lab_arena_signal_judgments WHERE cache_key=v_key;
  INSERT INTO public.lab_arena_signal_judgment_reservations
   (cache_key,authority_slot,signal_input_hash,run_id,lease_generation,lease_expires_at)
   VALUES (v_key,v_slot,v_hash,p_run_id,v_run.lease_generation,v_run.lease_expires_at)
   ON CONFLICT (cache_key) DO NOTHING;
  GET DIAGNOSTICS v_inserted=ROW_COUNT;
  IF v_inserted<>1 THEN RETURN jsonb_build_object('status','busy'); END IF;
 END IF;
 RETURN jsonb_build_object('status','miss','reference',jsonb_build_object(
  'schema_version','leadpoet.lab_arena.signal_judgment_reference.v1',
  'cache_key',v_key,'signal_input_hash',v_hash,'authority_slot',v_slot));
END; $$;
ALTER FUNCTION public.lab_arena_claim_signal_judgment_v1(TEXT,TEXT,INTEGER,INTEGER,JSONB) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_claim_signal_judgment_v1(TEXT,TEXT,INTEGER,INTEGER,JSONB) FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;
GRANT EXECUTE ON FUNCTION public.lab_arena_claim_signal_judgment_v1(TEXT,TEXT,INTEGER,INTEGER,JSONB) TO lab_arena_service;

CREATE FUNCTION public.lab_arena_complete_signal_judgment_v1(
 p_run_id TEXT,p_lease_token_hash TEXT,p_company_index INTEGER,p_signal_index INTEGER,
 p_scope JSONB,p_reference JSONB,p_judgment JSONB,p_judgment_hash TEXT
) RETURNS JSONB LANGUAGE plpgsql VOLATILE SECURITY DEFINER SET search_path = pg_catalog, public AS $$
DECLARE v_run public.lab_arena_runs; v_res public.lab_arena_signal_judgment_reservations;
 v_use public.lab_arena_signal_judgment_uses; v_hit public.lab_arena_signal_judgments;
 v_ref JSONB;
BEGIN
 SELECT * INTO v_run FROM public.lab_arena_runs WHERE run_id=p_run_id FOR UPDATE;
 IF NOT FOUND OR NOT public.lab_arena__signal_lease_valid_v1(v_run,p_lease_token_hash)
  OR p_company_index NOT BETWEEN 0 AND 4 OR p_signal_index NOT BETWEEN 0 AND 99 THEN
  RAISE EXCEPTION 'lab_arena_signal_completion_invalid' USING ERRCODE='22023'; END IF;
 v_ref:=jsonb_build_object('schema_version','leadpoet.lab_arena.signal_judgment_reference.v1',
  'cache_key',p_scope->>'cache_key','signal_input_hash',p_scope->>'signal_input_hash',
  'authority_slot',p_reference->'authority_slot');
 IF p_reference IS DISTINCT FROM v_ref THEN
  RAISE EXCEPTION 'lab_arena_signal_reference_invalid' USING ERRCODE='22023'; END IF;
 SELECT * INTO v_use FROM public.lab_arena_signal_judgment_uses
  WHERE run_id=p_run_id AND company_index=p_company_index AND signal_index=p_signal_index;
 IF FOUND THEN
  IF v_use.cache_key<>p_scope->>'cache_key' OR v_use.authority_slot<>(p_reference->>'authority_slot')::INTEGER
   OR (p_judgment IS NOT NULL AND v_use.judgment_hash<>p_judgment_hash) THEN
   RAISE EXCEPTION 'lab_arena_signal_use_conflict' USING ERRCODE='23505'; END IF;
  RETURN jsonb_build_object('status','existing','reference',p_reference);
 END IF;
 SELECT * INTO v_res FROM public.lab_arena_signal_judgment_reservations
  WHERE cache_key=p_scope->>'cache_key' AND authority_slot=(p_reference->>'authority_slot')::INTEGER
   AND signal_input_hash=p_scope->>'signal_input_hash' AND run_id=p_run_id
   AND lease_generation=v_run.lease_generation FOR UPDATE;
 IF NOT FOUND THEN RAISE EXCEPTION 'lab_arena_signal_reservation_stale' USING ERRCODE='P0003'; END IF;
 IF p_judgment IS NULL THEN
  DELETE FROM public.lab_arena_signal_judgment_reservations WHERE cache_key=v_res.cache_key;
  RETURN jsonb_build_object('status','released','reference',p_reference); END IF;
 IF jsonb_typeof(p_judgment) IS DISTINCT FROM 'object'
  OR COALESCE(p_judgment_hash,'') !~ '^sha256:[0-9a-f]{64}$' THEN
  RAISE EXCEPTION 'lab_arena_signal_judgment_invalid' USING ERRCODE='22023'; END IF;
 -- This immutable verdict carries no cost authority. Original provider
 -- ledger and score-publication eligibility remain bound to the source run.
 INSERT INTO public.lab_arena_signal_judgments
  (cache_key,authority_slot,signal_input_hash,scope_doc,judgment_hash,judgment_doc,
   source_score_run_id,source_scored_run_id,source_runner_hotkey,source_claim_request_hash,
   source_lease_generation,runner_authority_exclusions)
 VALUES (v_res.cache_key,v_res.authority_slot,v_res.signal_input_hash,p_scope,p_judgment_hash,p_judgment,
  p_run_id,v_run.scored_run_id,v_run.runner_hotkey,v_run.claim_request_hash,
  v_run.lease_generation,public.lab_arena_signal_authority_exclusions_v1(p_run_id));
 INSERT INTO public.lab_arena_signal_judgment_uses
  (run_id,company_index,signal_index,cache_key,authority_slot,signal_input_hash,source_score_run_id,judgment_hash)
 VALUES (p_run_id,p_company_index,p_signal_index,v_res.cache_key,v_res.authority_slot,
  v_res.signal_input_hash,p_run_id,p_judgment_hash);
 DELETE FROM public.lab_arena_signal_judgment_reservations WHERE cache_key=v_res.cache_key;
 RETURN jsonb_build_object('status','stored','reference',p_reference);
END; $$;
ALTER FUNCTION public.lab_arena_complete_signal_judgment_v1(TEXT,TEXT,INTEGER,INTEGER,JSONB,JSONB,JSONB,TEXT) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_complete_signal_judgment_v1(TEXT,TEXT,INTEGER,INTEGER,JSONB,JSONB,JSONB,TEXT) FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;
GRANT EXECUTE ON FUNCTION public.lab_arena_complete_signal_judgment_v1(TEXT,TEXT,INTEGER,INTEGER,JSONB,JSONB,JSONB,TEXT) TO lab_arena_service;

CREATE FUNCTION public.lab_arena_signal_judgment_lease_sync_v1() RETURNS trigger
 LANGUAGE plpgsql SECURITY DEFINER SET search_path = pg_catalog, public AS $$
BEGIN
 IF NEW.status<>'leased' THEN
  DELETE FROM public.lab_arena_signal_judgment_reservations WHERE run_id=NEW.run_id;
 ELSIF NEW.lease_expires_at IS DISTINCT FROM OLD.lease_expires_at THEN
  UPDATE public.lab_arena_signal_judgment_reservations SET lease_expires_at=NEW.lease_expires_at
   WHERE run_id=NEW.run_id AND lease_generation=NEW.lease_generation;
 END IF;
 RETURN NULL;
END; $$;
ALTER FUNCTION public.lab_arena_signal_judgment_lease_sync_v1() OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_signal_judgment_lease_sync_v1() FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;
CREATE TRIGGER lab_arena_signal_judgment_lease_sync AFTER UPDATE OF status,lease_expires_at
 ON public.lab_arena_runs FOR EACH ROW WHEN (OLD.kind='score')
 EXECUTE FUNCTION public.lab_arena_signal_judgment_lease_sync_v1();

-- A company result containing reused signals inherits the original source
-- exclusions. This is the only change to the established company completion.
DO $signal_company_authority$
DECLARE old_def TEXT; new_def TEXT;
BEGIN
 SELECT pg_get_functiondef('public.lab_arena_complete_attempt_v3(text,text,jsonb,text,text,text,jsonb,text)'::regprocedure)
 INTO old_def;
 IF old_def IS NULL OR strpos(old_def,'v_run.claim_response -> ''runner_authority_exclusions''')=0 THEN
  RAISE EXCEPTION 'company completion authority anchor differs' USING ERRCODE='55000'; END IF;
 new_def:=replace(old_def,
  'v_run.claim_response -> ''runner_authority_exclusions''',
  'public.lab_arena_signal_authority_exclusions_v1(p_run_id)');
 EXECUTE new_def;
END; $signal_company_authority$;

REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;
NOTIFY pgrst,'reload schema';
COMMIT;
