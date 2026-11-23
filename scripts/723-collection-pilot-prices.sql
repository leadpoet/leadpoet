-- Complete economic activation only after the genuine reviewed pilot.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='60s';
SET LOCAL search_path=pg_catalog;
CREATE TEMP TABLE collection716_acl ON COMMIT DROP AS
 SELECT nspacl,has_schema_privilege('lab_arena_owner','public','CREATE') AS had_create
 FROM pg_namespace WHERE nspname='public';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;

-- Lock DDL against accidental intermediate activation. An existing active
-- schedule cannot silently acquire new terms in a later migration.
DO $$ BEGIN
 IF EXISTS(SELECT FROM public.collection_reward_cutover) THEN
  RAISE EXCEPTION '714 activated before 716; reconcile cutover first'; END IF;
END $$;
ALTER TABLE public.collection_reward_cutover ADD COLUMN pilot_evidence jsonb NOT NULL;

CREATE FUNCTION public.collection_fee_class_v1(p_scenario text) RETURNS text
LANGUAGE sql IMMUTABLE STRICT SET search_path=pg_catalog AS $$
 SELECT CASE WHEN p_scenario ~ '^research-[A-Za-z0-9_.:-]+$' THEN 'research'
 WHEN p_scenario ~ '^(C[0-9]{2}|K[0-9]{2}|B[1-8]|E[1-7]|P[1-6])-s[1-9][0-9]{0,7}$' THEN left(p_scenario,1)
 END
$$;

-- A scoped release reader returns identifiers and immutable artifact hashes only.
CREATE FUNCTION public.collection_batch_jobs_v1(p_batch text) RETURNS jsonb
LANGUAGE sql STABLE SECURITY DEFINER SET search_path=pg_catalog AS $$
 SELECT coalesce(jsonb_agg(to_jsonb(job) ORDER BY job.slot),'[]') FROM (
    SELECT j.job_id,j.batch_id,j.mode,j.slot,j.scenario_id,j.model,j.harness_commit,j.miner_hotkey,
           j.artifact_sha256,j.status,j.budget_microusd,j.price_microusd
    FROM public.collection_jobs j WHERE j.batch_id=p_batch AND j.network='finney' AND j.netuid=71
 ) job
$$;

CREATE FUNCTION public.collection_pilot_costs_v1(p_batch text) RETURNS jsonb
LANGUAGE sql STABLE SECURITY DEFINER SET search_path=pg_catalog AS $$
 WITH measured AS (
  SELECT j.job_id,j.batch_id,j.slot,j.scenario_id,j.model,j.harness_commit,j.miner_hotkey,j.artifact_sha256,
    j.status,j.budget_microusd,
    count(*) FILTER (WHERE c.request_id IS NOT NULL AND r.cost_microusd IS NULL) unresolved,
    count(*) FILTER (WHERE c.provider='openrouter' AND r.http_status=200 AND r.cost_microusd>0) generator_calls,
    coalesce(sum(r.cost_microusd) FILTER (WHERE c.provider IN ('openrouter','deepline') AND c.key_id IS NOT NULL),0) miner_cost_microusd
  FROM public.collection_jobs j
  LEFT JOIN public.collection_calls c ON j.job_id=c.job_id
  LEFT JOIN public.collection_call_receipts r ON (c.request_id,c.job_id)=(r.request_id,r.job_id)
  WHERE j.batch_id=p_batch AND j.mode='pilot' AND j.network='finney' AND j.netuid=71
  GROUP BY j.job_id
 ) SELECT coalesce(jsonb_agg(to_jsonb(m) ORDER BY job_id),'[]') FROM measured m
$$;

CREATE FUNCTION public.collection_pilot_fees_v1(p_batch text,p_controls jsonb) RETURNS jsonb
LANGUAGE plpgsql STABLE SECURITY DEFINER SET search_path=pg_catalog AS $$
DECLARE measurements jsonb; amounts jsonb; caps jsonb; bad boolean; real_count bigint;
BEGIN
 measurements:=public.collection_pilot_costs_v1(p_batch);
 IF jsonb_typeof(p_controls) IS DISTINCT FROM 'array' OR jsonb_array_length(p_controls)>4096
    OR jsonb_array_length(measurements)=0 THEN RAISE EXCEPTION 'invalid pilot jobs or controls'; END IF;
 IF EXISTS (SELECT FROM jsonb_array_elements_text(p_controls) c(id)
    WHERE NOT EXISTS (SELECT FROM jsonb_array_elements(measurements) m(doc) WHERE m.doc->>'job_id'=c.id))
    OR EXISTS (SELECT FROM jsonb_array_elements_text(p_controls) c(id) GROUP BY c.id HAVING count(*)>1)
 THEN RAISE EXCEPTION 'foreign or duplicate controls'; END IF;

 WITH data AS (
  SELECT p.*,public.collection_fee_class_v1(scenario_id) AS family
  FROM jsonb_to_recordset(measurements) p(job_id text,status text,artifact_sha256 text,scenario_id text,model text,
       budget_microusd bigint,unresolved bigint,generator_calls bigint,miner_cost_microusd bigint)
  WHERE NOT p_controls ? job_id
 ), by_model AS (
  SELECT family,model,count(*) jobs,ceil(sum(miner_cost_microusd)*5/(4*count(*))) fee,
         max(budget_microusd) cap,bool_or(status NOT IN ('accepted','rejected') OR artifact_sha256 IS NULL
           OR family IS NULL OR unresolved>0 OR generator_calls=0 OR miner_cost_microusd<=0) OR
           min(budget_microusd)<>max(budget_microusd) invalid
  FROM data GROUP BY family,model
 ), by_family AS (
  SELECT coalesce(family,'invalid') family,sum(jobs) jobs,jsonb_object_agg(model,fee) prices,
         jsonb_object_agg(model,cap) budgets,bool_or(invalid OR fee>10000000000) OR count(*)<>6 invalid
  FROM by_model GROUP BY family
 )
 SELECT jsonb_object_agg(family,prices),jsonb_object_agg(family,budgets),bool_or(invalid),sum(jobs)
 INTO amounts,caps,bad,real_count FROM by_family;

 IF amounts IS NULL OR bad OR real_count<100 OR EXISTS (
  SELECT FROM jsonb_each(amounts) t(family,prices)
  WHERE NOT prices ?& ARRAY['moonshotai/kimi-k3.1-pro','deepseek/deepseek-v4-pro-0813',
    'thinkingmachines/inkling-2-small','tencent/hy4','z-ai/glm-5.3','deepseek/deepseek-v4.1-flash-1017']
 ) THEN RAISE EXCEPTION 'pilot measurement incomplete'; END IF;
 RETURN jsonb_build_object('prices',amounts,'budgets',caps,'job_count',real_count,
   'measurement_sha256',encode(sha256(convert_to(measurements::text,'UTF8')),'hex'));
END $$;

CREATE FUNCTION public.collection_approved_pilot_guard_v1() RETURNS trigger
LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog AS $$
DECLARE evidence jsonb:=NEW.pilot_evidence; calculated jsonb; batch public.collection_batches;
 prior public.lab_arena_accepted_weight_states; observation jsonb:=NEW.pilot_evidence->'cutover_anchor';
BEGIN
 SELECT * INTO STRICT batch FROM public.collection_batches WHERE batch_id=evidence->>'batch_id' AND mode='pilot';
 SELECT * INTO prior FROM public.lab_arena_accepted_weight_states WHERE network='finney' AND netuid=71 AND epoch=NEW.first_epoch-1;
 IF NOT FOUND OR jsonb_typeof(observation) IS DISTINCT FROM 'object' OR
    NOT observation ?& ARRAY['epoch','state_hash','head_block','head_hash','observed_at'] OR
    (observation->>'epoch')::bigint IS DISTINCT FROM prior.epoch OR
    observation->>'state_hash' IS DISTINCT FROM prior.state_hash OR
    coalesce(observation->>'head_hash','') !~ '^0x[0-9a-f]{64}$' OR
    coalesce((observation->>'head_block')::bigint,-1) NOT BETWEEN
        (prior.state_doc->>'valid_from_block')::bigint AND (prior.state_doc->>'valid_until_block')::bigint OR
    coalesce(extract(epoch from clock_timestamp())-(observation->>'observed_at')::bigint,-1) NOT BETWEEN 0 AND 120 THEN
  RAISE EXCEPTION 'current finalized legacy epoch readback required for cutover';
 END IF;
 IF evidence->>'schema' IS DISTINCT FROM 'leadpoet.collection.approved_pilot.v1' OR
    NOT (evidence ?& ARRAY['review_sha256','review_bundle_sha256','judge_policy_sha256','judge_prompt_sha256',
                     'builder_commit','harness_commit','prices','budgets','control_ids','job_count','measurement_sha256'])
    OR coalesce(evidence->>'review_sha256','') !~ '^[0-9a-f]{64}$' OR coalesce(evidence->>'review_bundle_sha256','') !~ '^[0-9a-f]{64}$'
    OR coalesce(evidence->>'judge_policy_sha256','') !~ '^[0-9a-f]{64}$' OR coalesce(evidence->>'judge_prompt_sha256','') !~ '^[0-9a-f]{64}$'
    OR coalesce(evidence->>'builder_commit','') !~ '^[0-9a-f]{40}$' OR evidence->>'harness_commit' IS DISTINCT FROM batch.harness_commit
 THEN RAISE EXCEPTION 'invalid signed pilot authority'; END IF;
 calculated:=public.collection_pilot_fees_v1(batch.batch_id,evidence->'control_ids');
 IF jsonb_build_object('prices',evidence->'prices','budgets',evidence->'budgets',
     'measurement_sha256',evidence->'measurement_sha256','job_count',evidence->'job_count') IS DISTINCT FROM calculated THEN RAISE EXCEPTION 'pilot-approved prices or budget differs'; END IF;
 RETURN NEW;
END $$;

CREATE FUNCTION public.collection_approved_job_price_v1() RETURNS trigger
LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog AS $$
DECLARE approval jsonb; family text;
BEGIN
 IF NEW.mode='live' THEN
  SELECT pilot_evidence INTO STRICT approval FROM public.collection_reward_cutover;
  family:=public.collection_fee_class_v1(NEW.scenario_id);
  IF family IS NULL OR approval->>'harness_commit' IS DISTINCT FROM NEW.harness_commit OR
     (approval->'prices'->family->>NEW.model)::bigint IS DISTINCT FROM NEW.price_microusd OR
     (approval->'budgets'->family->>NEW.model)::bigint IS DISTINCT FROM NEW.budget_microusd
  THEN RAISE EXCEPTION 'unapproved live quote or spend ceiling'; END IF;
 END IF;
 RETURN NEW;
END $$;

CREATE FUNCTION public.collection_reward_schedule_v1() RETURNS jsonb
LANGUAGE sql STABLE SECURITY DEFINER SET search_path=pg_catalog AS $$
 SELECT coalesce((SELECT jsonb_build_object('first_epoch',first_epoch,'pilot_evidence',pilot_evidence)
                  FROM public.collection_reward_cutover),'{}')
$$;

CREATE TRIGGER collection_approved_pilot_guard BEFORE INSERT ON public.collection_reward_cutover
 FOR EACH ROW EXECUTE FUNCTION public.collection_approved_pilot_guard_v1();
CREATE TRIGGER collection_approved_job_price BEFORE INSERT OR UPDATE OF price_microusd,budget_microusd,scenario_id,model,harness_commit,mode
 ON public.collection_jobs FOR EACH ROW EXECUTE FUNCTION public.collection_approved_job_price_v1();

DO $acl$
DECLARE f regprocedure;
BEGIN
 FOREACH f IN ARRAY ARRAY['public.collection_fee_class_v1(text)'::regprocedure,
  'public.collection_batch_jobs_v1(text)'::regprocedure,'public.collection_pilot_costs_v1(text)'::regprocedure,
  'public.collection_pilot_fees_v1(text,jsonb)'::regprocedure,'public.collection_approved_pilot_guard_v1()'::regprocedure,
  'public.collection_approved_job_price_v1()'::regprocedure,'public.collection_reward_schedule_v1()'::regprocedure] LOOP
  EXECUTE format('ALTER FUNCTION %s OWNER TO lab_arena_owner',f);
  EXECUTE format('REVOKE ALL ON FUNCTION %s FROM PUBLIC,anon,authenticated,service_role,lab_arena_service',f);
 END LOOP;
 GRANT EXECUTE ON FUNCTION public.collection_batch_jobs_v1(text),public.collection_pilot_costs_v1(text),public.collection_pilot_fees_v1(text,jsonb),
                           public.collection_reward_schedule_v1() TO lab_arena_service;
 IF NOT (SELECT had_create FROM pg_temp.collection716_acl) THEN REVOKE CREATE ON SCHEMA public FROM lab_arena_owner; END IF;
 IF (SELECT nspacl FROM pg_namespace WHERE nspname='public') IS DISTINCT FROM (SELECT nspacl FROM pg_temp.collection716_acl)
 THEN RAISE EXCEPTION 'collection schema ACL differs'; END IF;
END $acl$;
NOTIFY pgrst,'reload schema';
COMMIT;
