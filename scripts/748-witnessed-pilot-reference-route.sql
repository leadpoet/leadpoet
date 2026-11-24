-- A successor is a private pilot cohort with its own immutable route identity.
-- The unchanged slot numbers preserve the unassigned schedule of the parent.
BEGIN;
SET LOCAL statement_timeout='60s';
SET LOCAL lock_timeout='5s';
SET LOCAL search_path=pg_catalog;
CREATE TEMP TABLE prev748_acl ON COMMIT DROP AS SELECT nspacl,has_schema_privilege('lab_arena_owner','public','CREATE') had_create
 FROM pg_namespace WHERE nspname='public';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;

CREATE TABLE public.collection_reference_successions (
 parent_id text PRIMARY KEY REFERENCES public.collection_batches(batch_id),
 successor_id text NOT NULL UNIQUE CHECK(successor_id ~ '^rs-[a-f0-9]{64}$'),
 next_slot bigint NOT NULL CHECK(next_slot>0),
 model text NOT NULL CHECK(model='deepseek/deepseek-v4-pro-0813'),
 provider_tag text NOT NULL CHECK(provider_tag='coreweave/fp8'),
 model_revision text NOT NULL CHECK(model_revision='deepseek/deepseek-v4-pro-20260813'),
 witness_sha256 text NOT NULL CHECK(witness_sha256 ~ '^sha256:[0-9a-f]{64}$'),
 recorded_at timestamptz NOT NULL DEFAULT clock_timestamp(),
 CHECK(parent_id<>successor_id AND parent_id !~ '^rs-')
);
ALTER TABLE public.collection_reference_successions OWNER TO lab_arena_owner;
ALTER TABLE public.collection_reference_successions ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON public.collection_reference_successions FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;

CREATE FUNCTION public.collection_reference_succession_guard_v1() RETURNS trigger
LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog AS $$
DECLARE parent public.collection_batches; committed bigint;
BEGIN
 IF TG_OP<>'INSERT' THEN RAISE EXCEPTION 'reference succession is immutable'; END IF;
 PERFORM pg_advisory_xact_lock(hashtextextended('collection_claim:finney:71',0));
 SELECT * INTO STRICT parent FROM public.collection_batches WHERE batch_id=NEW.parent_id AND mode='pilot'
   AND experiment_terms_hash IS NULL;
 SELECT max(slot)+1 INTO committed FROM public.collection_jobs WHERE batch_id=NEW.parent_id;
 IF committed IS DISTINCT FROM NEW.next_slot OR NEW.next_slot>=jsonb_array_length(parent.scenario_ids)*6
  OR parent.models->>(NEW.next_slot%6)::int IS DISTINCT FROM NEW.model
  OR EXISTS(SELECT FROM public.collection_jobs WHERE batch_id=NEW.parent_id AND status IN ('running','ready'))
  OR EXISTS(SELECT FROM public.collection_batches WHERE batch_id=NEW.successor_id)
  OR public.collection_reward_schedule_v1()#>>'{pilot_evidence,batch_id}'=NEW.parent_id
 THEN RAISE EXCEPTION 'parent frozen slot or reference eligibility changed'; END IF;
 RETURN NEW;
END $$;
CREATE TRIGGER immutable_reference_succession BEFORE INSERT OR UPDATE OR DELETE ON public.collection_reference_successions
 FOR EACH ROW EXECUTE FUNCTION public.collection_reference_succession_guard_v1();
CREATE TRIGGER no_truncate_reference_succession BEFORE TRUNCATE ON public.collection_reference_successions
 FOR EACH STATEMENT EXECUTE FUNCTION public.collection_reference_succession_guard_v1();
ALTER FUNCTION public.collection_reference_succession_guard_v1() OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.collection_reference_succession_guard_v1() FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;

-- Preserve every prior constraint, quota, receipt and pre-approved pilot retry.
DO $migration$
DECLARE target regprocedure := 'public.collection_claim_v1(uuid,text,text,text,text,jsonb,jsonb,jsonb,jsonb,text)'::regprocedure;
 source text; pre jsonb; post jsonb;
 declaration constant text := 'attempt_number integer := 1;';
 admission constant text := ' INSERT INTO public.collection_batches(batch_id,mode,harness_commit,scenario_ids,models,prices,budgets,experiment_terms_hash)';
 select_next constant text := 'SELECT coalesce(max(slot)+1,0) INTO n FROM public.collection_jobs WHERE batch_id=p_batch;';
 privaterefs constant text := $patch$
 IF EXISTS(SELECT FROM public.collection_reference_successions WHERE parent_id=p_batch) THEN
  RETURN NULL;
 END IF;
 IF p_batch ~ '^rs-' THEN
  SELECT s.next_slot INTO successor_first FROM public.collection_reference_successions s
  JOIN public.collection_batches parent ON parent.batch_id=s.parent_id
  WHERE s.successor_id=p_batch AND p_mode='pilot' AND NOT experiment AND
        p_terms_hash='sha256:bea1b86f92cde71382507dac407fa56a2c87576c5a1d7753212b5ff4c0841b0b' AND
        (parent.harness_commit,parent.scenario_ids,parent.models,parent.prices,parent.budgets)
        IS NOT DISTINCT FROM (p_commit,p_scenarios,p_models,p_prices,p_budgets);
  IF NOT FOUND THEN RETURN jsonb_build_object('error','unwitnessed_reference_successor'); END IF;
 END IF;
$patch$;
BEGIN
 SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc' INTO STRICT source,pre FROM pg_proc p
 WHERE p.oid=target AND p.prosecdef AND p.proowner='lab_arena_owner'::regrole;
 IF cardinality(string_to_array(source,declaration))<>2 OR cardinality(string_to_array(source,admission))<>2
    OR cardinality(string_to_array(source,select_next))<>2 THEN RAISE EXCEPTION '748 unexpected claim predecessor'; END IF;
 source:=replace(source,declaration,declaration || ' successor_first bigint := 0;');
 source:=replace(source,admission,privaterefs || admission);
 EXECUTE replace(source,select_next,replace(select_next,'max(slot)+1,0','max(slot)+1,successor_first'));
 SELECT to_jsonb(p)-'prosrc' INTO post FROM pg_proc p WHERE p.oid=target;
 IF post IS DISTINCT FROM pre THEN RAISE EXCEPTION '748 altered prior claim metadata'; END IF;
 IF NOT (SELECT had_create FROM pg_temp.prev748_acl) THEN REVOKE CREATE ON SCHEMA public FROM lab_arena_owner; END IF;
 IF (SELECT nspacl FROM pg_namespace WHERE nspname='public') IS DISTINCT FROM (SELECT nspacl FROM pg_temp.prev748_acl)
 THEN RAISE EXCEPTION 'collection schema privileges changed'; END IF;
END $migration$;
COMMIT;
