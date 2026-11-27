-- Standing grants share the existing allocator, call reservations and finalized epoch ledger.
-- Activate prospectively with INSERT INTO public.collection_pool_activation(singleton) VALUES(true).
BEGIN;
SET LOCAL search_path=pg_catalog;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='60s';
SELECT pg_advisory_xact_lock(hashtextextended('collection_claim:finney:71',0));
CREATE TEMP TABLE pool_previous_acl ON COMMIT DROP AS
 SELECT nspacl,has_schema_privilege('lab_arena_owner','public','CREATE') had_create
 FROM pg_namespace WHERE nspname='public';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;

CREATE TABLE public.collection_pool_activation (
 singleton boolean PRIMARY KEY CHECK(singleton),
 activated_at timestamptz NOT NULL DEFAULT clock_timestamp()
);
ALTER TABLE public.collection_pool_activation OWNER TO lab_arena_owner;
ALTER TABLE public.collection_pool_activation ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON public.collection_pool_activation FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;
CREATE FUNCTION public.collection_pool_active_v1() RETURNS boolean
 LANGUAGE sql STABLE SECURITY DEFINER SET search_path=pg_catalog AS $$
 SELECT EXISTS(SELECT FROM public.collection_pool_activation)
$$;
CREATE FUNCTION public.collection_pool_activate_v1() RETURNS trigger
 LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog AS $$
BEGIN
 PERFORM pg_advisory_xact_lock(hashtextextended('collection_claim:finney:71',0));
 PERFORM pg_advisory_xact_lock(hashtextextended('collection_pool_registration',0));
 IF NOT EXISTS(SELECT FROM public.collection_reward_cutover) THEN RAISE EXCEPTION 'paid collection not active'; END IF;
 RETURN NEW;
END $$;
CREATE TRIGGER validate_pool_activation BEFORE INSERT ON public.collection_pool_activation
 FOR EACH ROW EXECUTE FUNCTION public.collection_pool_activate_v1();
CREATE TRIGGER immutable_pool_activation BEFORE UPDATE OR DELETE ON public.collection_pool_activation
 FOR EACH ROW EXECUTE FUNCTION public.lab_arena_append_only_v1();
CREATE TRIGGER no_truncate_pool_activation BEFORE TRUNCATE ON public.collection_pool_activation
 FOR EACH STATEMENT EXECUTE FUNCTION public.lab_arena_append_only_v1();

-- At claim time, hold each running lease's ceiling, including concurrent experiments.
-- At reservation time the caller may use its own remaining lease allowance.
-- A new, explicitly signed grant starts a fresh balance; unsettled calls from an older
-- registration of the same credential still hold capacity until their native bills arrive.
CREATE FUNCTION public.collection_pool_available_v1(p_key uuid,p_job uuid) RETURNS bigint
 LANGUAGE sql STABLE SECURITY DEFINER SET search_path=pg_catalog AS $$
 WITH k AS (
  SELECT * FROM public.collection_keys WHERE key_id=p_key AND provider='openrouter'
 ), jobs AS (
  SELECT j.job_id,j.status,j.lease_expires_at,j.budget_microusd,
   coalesce(sum(coalesce(r.cost_microusd,c.reserved_microusd)),0) spent
  FROM public.collection_jobs j LEFT JOIN public.collection_calls c ON c.job_id=j.job_id AND c.key_id=p_key
  LEFT JOIN public.collection_call_receipts r ON (r.job_id,r.request_id)=(c.job_id,c.request_id)
  WHERE j.openrouter_key_id=p_key GROUP BY j.job_id
 ), historic AS (
  SELECT coalesce(sum(c.reserved_microusd),0) held FROM k
  JOIN public.collection_keys old ON old.fingerprint=k.fingerprint AND old.provider=k.provider AND old.key_id<>p_key
  JOIN public.collection_calls c ON c.key_id=old.key_id
  LEFT JOIN public.collection_call_receipts r USING(job_id,request_id)
  WHERE r.cost_microusd IS NULL
 )
 SELECT greatest(0,k.daily_cap_microusd-historic.held-coalesce((SELECT sum(
   greatest(spent,CASE WHEN status='running' AND lease_expires_at>statement_timestamp()
            AND job_id IS DISTINCT FROM p_job THEN budget_microusd ELSE 0 END)) FROM jobs),0))::bigint
 FROM k CROSS JOIN historic
$$;

-- The published target is computed once from the ledger's integer micro-USD charges.
-- A provider cost exceeding its authorized reservation or a reused generation cannot earn a target.
CREATE INDEX collection_receipt_provider_identity ON public.collection_call_receipts(provider_id) WHERE provider_id IS NOT NULL;
CREATE INDEX collection_jobs_funded_key ON public.collection_jobs(openrouter_key_id);
CREATE FUNCTION public.collection_pool_target_v1(p_job uuid) RETURNS bigint
 LANGUAGE sql STABLE SECURITY DEFINER SET search_path=pg_catalog AS $$
 SELECT ((bill->>'cost_microusd')::bigint*5+3)/4
 FROM public.collection_verified_generator_bill_v1(p_job) bill
 WHERE bill IS NOT NULL AND NOT EXISTS(
  SELECT FROM public.collection_calls c JOIN public.collection_call_receipts r USING(job_id,request_id)
  WHERE c.job_id=p_job AND c.key_id IS NOT NULL AND (
    r.cost_microusd>c.reserved_microusd OR r.provider_id !~ '^[A-Za-z0-9][A-Za-z0-9._:-]{0,199}$'
    OR EXISTS(SELECT FROM public.collection_call_receipts other
      WHERE other.provider_id=r.provider_id AND (other.job_id,other.request_id)<>(r.job_id,r.request_id))))
$$;

ALTER TABLE public.collection_jobs DROP CONSTRAINT collection_jobs_check;
ALTER TABLE public.collection_jobs ADD CHECK(price_microusd>=0 AND (
 price_microusd>0 OR mode='pilot' OR (batch_id ~ '^pool-[a-f0-9]{64}$' AND status<>'accepted')));
ALTER TABLE public.collection_jobs DROP CONSTRAINT collection_synthetic_key_scope;
ALTER TABLE public.collection_jobs ADD CONSTRAINT collection_synthetic_key_scope CHECK (
 deepline_key_id IS NOT NULL OR scenario_id ~ '^(C[0-9]{2}|K[0-9]{2}|B[1-8]|E[1-7]|P[1-7])-s[1-9][0-9]{0,7}$'
 OR (mode='live' AND batch_id ~ '^dv-[a-f0-9]{64}$' AND scenario_id ~ '^P8-s[1-9][0-9]{0,7}$')
 OR (mode='live' AND batch_id ~ '^pool-[a-f0-9]{64}$' AND scenario_id ~ '^P(7|8|9|10)-s[1-9][0-9]{0,7}$'));
ALTER TABLE public.collection_batches DROP CONSTRAINT collection_batches_check;
ALTER TABLE public.collection_batches ADD CONSTRAINT collection_batches_check CHECK (
 jsonb_typeof(models)='array' AND
 CASE WHEN batch_id ~ '^pool-[a-f0-9]{64}$' THEN mode='live'
      AND experiment_terms_hash IS NOT DISTINCT FROM 'sha256:1655fcba4ba721aa1559c073e1430853450b1771ebf2dea7c74c5a74c4768df1'
      AND jsonb_array_length(models)=jsonb_array_length(scenario_ids) AND jsonb_array_length(models) BETWEEN 1 AND 4096
      WHEN experiment_terms_hash IS NULL AND batch_id ~ '^dv-[a-f0-9]{64}$'
      THEN mode='live' AND jsonb_array_length(models) BETWEEN 1 AND 6
      WHEN experiment_terms_hash IS NULL THEN jsonb_array_length(models)=6
      ELSE mode IN ('pilot','live') AND batch_id ~ '^xp-[0-9a-f]{64}$' AND experiment_terms_hash ~ '^sha256:[0-9a-f]{64}$'
      AND jsonb_array_length(models)=jsonb_array_length(scenario_ids) AND jsonb_array_length(models) BETWEEN 1 AND 4096 END);

-- Update exact predecessor fragments, retaining the canonical lease locks, retry
-- behavior, signed epoch guard and scoring/weight publication paths.
CREATE FUNCTION pg_temp.pool_patch(target regprocedure, old_sql text, new_sql text) RETURNS void
 LANGUAGE plpgsql AS $$ DECLARE definition text; prior jsonb; BEGIN
 SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc' INTO STRICT definition, prior FROM pg_proc p
 WHERE p.oid=target AND p.prosecdef AND p.proowner='lab_arena_owner'::regrole;
 IF cardinality(string_to_array(definition,old_sql))<>2 THEN RAISE EXCEPTION '758 unexpected predecessor: %',target; END IF;
 EXECUTE replace(definition,old_sql,new_sql);
 IF (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=target) IS DISTINCT FROM prior THEN
   RAISE EXCEPTION '758 changed function authority: %',target; END IF;
END $$;

DO $migration$
DECLARE reg regprocedure:='public.collection_register_v1(uuid,text,text,text,jsonb,bigint,jsonb,jsonb)'::regprocedure;
 claim regprocedure:='public.collection_priced_claim_v1(uuid,text,text,text,text,jsonb,jsonb,jsonb,jsonb,text,text)'::regprocedure;
 reserve regprocedure:='public.collection_reserve_call_v1(uuid,text,text,text,text,bigint)'::regprocedure;
 price regprocedure:='public.collection_approved_job_price_v1()'::regprocedure;
 reward regprocedure:='public.collection_reward_job_guard_v1()'::regprocedure;
 auth regprocedure:='public.collection_training_consents_v1(text)'::regprocedure;
 statement regprocedure:='public.collection_statement_v1(text,text)'::regprocedure;
 marker text; old text;
BEGIN
 marker:=' IF EXISTS(SELECT FROM public.collection_registrations WHERE miner_hotkey=p_hotkey';
 PERFORM pg_temp.pool_patch(reg,marker,$body$ IF p_mode='live' THEN
   PERFORM pg_advisory_xact_lock_shared(hashtextextended('collection_pool_registration',0));
   IF public.collection_pool_active_v1() IS DISTINCT FROM
     coalesce(p_consent#>>'{body,task_scope}'='sales_sandbox',false) THEN
     RETURN jsonb_build_object('error','collection_consent_invalid');
   END IF;
 END IF;
$body$||marker);
 marker:=' IF NOT coalesce((';
 PERFORM pg_temp.pool_patch(reg,marker,$body$ IF p_consent#>>'{body,task_scope}'='sales_sandbox' THEN
   IF NOT coalesce(p_mode='live'
      AND p_consent->>'round_id'='finney:71:locked-collection:live'
      AND p_consent->>'scope'='lab_arena.collection.register.v1' AND p_consent->>'hotkey'=p_hotkey
      AND jsonb_typeof(p_consent->'body')='object'
      AND (p_consent->'body')-ARRAY['terms_sha256','task_scope','credentials_sha256','priority','spend_limit_microusd']='{}'::jsonb
      AND p_consent#>>'{body,terms_sha256}'='sha256:1655fcba4ba721aa1559c073e1430853450b1771ebf2dea7c74c5a74c4768df1'
      AND p_consent#>>'{body,credentials_sha256}' ~ '^sha256:[a-f0-9]{64}$'
      AND jsonb_typeof(p_consent#>'{body,spend_limit_microusd}')='number'
      AND p_consent#>>'{body,spend_limit_microusd}'=p_cap::text AND p_cap BETWEEN 1 AND 10000000000
      AND jsonb_typeof(p_consent#>'{body,priority}')='number' AND p_consent#>>'{body,priority}' IN ('0','1')
      AND jsonb_typeof(p_openrouter)='object' AND (p_deepline IS NULL OR p_deepline='null'::jsonb),false)
   THEN RETURN jsonb_build_object('error','collection_consent_invalid'); END IF;
 ELSIF NOT coalesce(($body$);
 -- Keep the original register insert; other consent hashes retain their original slots.

 marker:=$body$ experiment boolean := p_batch ~ '^xp-[0-9a-f]{64}$';$body$;
 PERFORM pg_temp.pool_patch(claim,marker,$body$ pooled boolean := p_batch ~ '^pool-[0-9a-f]{64}$';
 experiment boolean := p_batch ~ '^(xp|pool)-[0-9a-f]{64}$';$body$);
 marker:=' IF jsonb_typeof(p_scenarios)';
 PERFORM pg_temp.pool_patch(claim,marker,$body$ IF p_mode='live' AND pooled IS DISTINCT FROM public.collection_pool_active_v1()
   AND NOT EXISTS(SELECT FROM public.collection_jobs WHERE job_id=p_job AND batch_id=p_batch)
 THEN RETURN NULL; END IF;
$body$||marker);
 old:=$body$(p_mode<>'live' OR p_batch !~ '^dv-[a-f0-9]{64}$' OR s.id !~ '^P8-s[1-9][0-9]{0,7}$')$body$;
 PERFORM pg_temp.pool_patch(claim,old,'('||old||$body$ AND NOT (pooled AND p_mode='live' AND s.id ~ '^P(7|8|9|10)-s[1-9][0-9]{0,7}$'))$body$);
 marker:=$body$ IF experiment AND p_mode='live' THEN$body$;
 PERFORM pg_temp.pool_patch(claim,marker,$body$ IF pooled THEN
   IF p_mode IS DISTINCT FROM 'live' OR p_terms_hash IS DISTINCT FROM
       'sha256:1655fcba4ba721aa1559c073e1430853450b1771ebf2dea7c74c5a74c4768df1'
      OR jsonb_typeof(p_models) IS DISTINCT FROM 'array' OR jsonb_typeof(p_prices) IS DISTINCT FROM 'object'
      OR jsonb_typeof(p_budgets) IS DISTINCT FROM 'object'
      OR jsonb_array_length(p_models) NOT BETWEEN 1 AND 4096 OR jsonb_array_length(p_models)<>jsonb_array_length(p_scenarios)
      OR EXISTS(SELECT FROM jsonb_array_elements_text(p_scenarios) s(id) WHERE s.id IS NULL OR s.id !~ '^P(7|8|9|10)-s[1-9][0-9]{0,7}$')
      OR EXISTS(SELECT FROM jsonb_array_elements_text(p_models) m(id) WHERE m.id IS NULL OR
        m.id NOT IN ('moonshotai/kimi-k3.1-pro','deepseek/deepseek-v4-pro-0813','mistralai/mistral-large-4-0',
         'qwen/qwen3.8-27b','nvidia/nemotron-3.5-super-vl','deepseek/deepseek-v4.1-flash-1017')
        OR p_prices->m.id IS DISTINCT FROM '0'::jsonb OR jsonb_typeof(p_budgets->m.id) IS DISTINCT FROM 'number'
        OR (p_budgets->>m.id) !~ '^[1-9][0-9]{0,10}$' OR (p_budgets->>m.id)::numeric>10000000000)
   THEN RETURN jsonb_build_object('error','invalid_pool_experiment'); END IF;
 ELSIF experiment AND p_mode='live' THEN$body$);
 marker:=' SELECT daily_slots_per_owner INTO STRICT daily_limit';
 SELECT substr(prosrc,strpos(prosrc,marker),strpos(prosrc,'r.created_at,r.registration_id LIMIT 1 FOR UPDATE OF r;')
    -strpos(prosrc,marker)+length('r.created_at,r.registration_id LIMIT 1 FOR UPDATE OF r;')) INTO STRICT old
 FROM pg_proc WHERE oid=claim;
 PERFORM pg_temp.pool_patch(claim,old,$body$ IF pooled THEN
   day_utc := (clock_timestamp() AT TIME ZONE 'UTC')::date;
   SELECT r.registration_id,r.miner_hotkey,o.key_id openrouter_key_id INTO enrollment
   FROM public.collection_registrations r JOIN public.collection_keys o ON o.registration_id=r.registration_id
   CROSS JOIN LATERAL (SELECT public.collection_pool_available_v1(o.key_id,NULL) AS available OFFSET 0) capacity
   WHERE r.mode='live' AND o.provider='openrouter' AND o.miner_hotkey=r.miner_hotkey
     AND r.revoked_at IS NULL AND o.revoked_at IS NULL AND r.consent#>>'{body,task_scope}'='sales_sandbox'
     AND r.consent#>>'{body,terms_sha256}'=p_terms_hash AND capacity.available>=spending
     -- Give a refused credential time to be restored or topped up. Its unknown bills still hold budget.
     AND NOT EXISTS(SELECT FROM public.collection_calls declined JOIN public.collection_call_receipts rc USING(job_id,request_id)
       WHERE declined.key_id=o.key_id AND declined.provider='openrouter' AND rc.http_status IN (401,402)
         AND declined.created_at>statement_timestamp()-interval '1 hour')
   ORDER BY (r.consent#>>'{body,priority}')::int DESC,capacity.available DESC,r.created_at,r.registration_id
   LIMIT 1 FOR UPDATE OF r,o;
 ELSE
$body$||old||E'\n END IF;');

 marker:=' SELECT coalesce(sum(coalesce(r.cost_microusd,c.reserved_microusd)),0) INTO key_used';
 SELECT substr(prosrc,strpos(prosrc,marker),strpos(prosrc,$body$c.created_at>clock_timestamp()-interval '24 hours';$body$)
    -strpos(prosrc,marker)+length($body$c.created_at>clock_timestamp()-interval '24 hours';$body$)) INTO STRICT old
 FROM pg_proc WHERE oid=reserve;
 PERFORM pg_temp.pool_patch(reserve,old,$body$ IF j.batch_id ~ '^pool-[a-f0-9]{64}$' AND p_provider<>'judge' THEN
   IF p_provider<>'openrouter' OR public.collection_pool_available_v1(k.key_id,j.job_id)<p_amount THEN
      RETURN jsonb_build_object('status','budget_exhausted'); END IF;
   key_used:=0;
 ELSE
$body$||old||E'\n END IF;');

 -- Serialize matching native IDs at the receipt's natural insertion point. A retry
 -- of the same request remains idempotent; one generation cannot finance two jobs.
 marker:=' INSERT INTO public.collection_call_receipts VALUES(p_job,p_request,p_hash,p_provider_id,p_cost,p_status)';
 PERFORM pg_temp.pool_patch('public.collection_record_call_v1(uuid,text,text,text,text,bigint,integer)',marker,$body$ IF p_provider_id IS NOT NULL THEN
   PERFORM pg_advisory_xact_lock(hashtextextended('collection_provider_generation:'||p_provider_id,0));
   IF EXISTS(SELECT FROM public.collection_call_receipts prior
    JOIN public.collection_jobs previous ON previous.job_id=prior.job_id
    JOIN public.collection_jobs current_job ON current_job.job_id=p_job
    WHERE prior.provider_id=p_provider_id AND (prior.job_id,prior.request_id)<>(p_job,p_request)
     AND (current_job.batch_id ~ '^pool-' OR previous.batch_id ~ '^pool-'))
   THEN RAISE EXCEPTION 'ambiguous collection provider generation'; END IF;
 END IF;
$body$||marker);

 marker:=$body$ IF NEW.mode='live' THEN$body$;
 PERFORM pg_temp.pool_patch(price,marker,$body$ IF NEW.batch_id ~ '^pool-[a-f0-9]{64}$' THEN
   IF NEW.mode<>'live' OR NEW.price_microusd<>0 OR NEW.deepline_key_id IS NOT NULL OR NOT EXISTS(
    SELECT FROM public.collection_batches b JOIN public.collection_registrations r ON r.registration_id=NEW.registration_id
    JOIN public.collection_keys k ON k.registration_id=r.registration_id AND k.key_id=NEW.openrouter_key_id
    WHERE b.batch_id=NEW.batch_id AND b.mode='live' AND b.harness_commit=NEW.harness_commit
     AND b.experiment_terms_hash='sha256:1655fcba4ba721aa1559c073e1430853450b1771ebf2dea7c74c5a74c4768df1'
     AND b.scenario_ids->>(NEW.slot::int)=NEW.scenario_id AND b.models->>(NEW.slot::int)=NEW.model
     AND (b.budgets->>NEW.model)::bigint=NEW.budget_microusd AND b.prices->NEW.model='0'::jsonb
     AND r.mode='live' AND r.miner_hotkey=NEW.miner_hotkey AND k.provider='openrouter' AND k.miner_hotkey=r.miner_hotkey
     AND r.consent#>>'{body,task_scope}'='sales_sandbox' AND r.consent#>>'{body,terms_sha256}'=b.experiment_terms_hash
     AND r.revoked_at IS NULL AND k.revoked_at IS NULL AND public.collection_pool_active_v1()
   ) THEN RAISE EXCEPTION 'invalid standing-funded job'; END IF;
   RETURN NEW;
 END IF;
$body$||marker);

 marker:=$body$ IF TG_OP<>'DELETE' AND NEW.batch_id ~ '^dv-$body$;
 PERFORM pg_temp.pool_patch(reward,marker,$body$ IF TG_OP='UPDATE' AND (OLD.batch_id ~ '^pool-' OR NEW.batch_id ~ '^pool-')
   AND (to_jsonb(OLD)-ARRAY['status','quality','artifact_sha256','finished_at','accepted_epoch']) IS DISTINCT FROM
       (to_jsonb(NEW)-ARRAY['status','quality','artifact_sha256','finished_at','accepted_epoch']) THEN
   RAISE EXCEPTION 'standing-funded assignment is immutable';
 END IF;
 IF TG_OP<>'DELETE' AND NEW.batch_id ~ '^pool-' AND NEW.status IN ('ready','accepted') THEN
   IF TG_OP<>'UPDATE' OR OLD.status<>'running' OR NEW.status<>'accepted'
      OR NEW.quality->>'payment_rule' IS DISTINCT FROM 'verified_actual_cost_v1'
      OR NOT coalesce(NEW.quality->>'training_status' IN ('accepted','code_rejected'),false)
      OR NEW.artifact_sha256 IS NULL
   THEN RAISE EXCEPTION 'verified complete pool work required'; END IF;
   NEW.price_microusd:=public.collection_pool_target_v1(NEW.job_id);
   IF NEW.price_microusd IS NULL OR NEW.price_microusd<=0 THEN
     RAISE EXCEPTION 'verified settled generator receipts required'; END IF;
 END IF;
$body$||marker);

 marker:='r.consent,r.max_slots_per_utc_day,';
 PERFORM pg_temp.pool_patch(auth,marker,marker||$body$
          CASE WHEN r.consent#>>'{body,task_scope}'='sales_sandbox' THEN jsonb_build_object(
           'spend_limit_microusd',(SELECT daily_cap_microusd FROM public.collection_keys WHERE key_id=j.openrouter_key_id),
           'priority',(r.consent#>>'{body,priority}')::int) ELSE '{}'::jsonb END funding,$body$);

 marker:='SELECT coalesce(jsonb_agg(to_jsonb(rows) ORDER BY job_id)';
 PERFORM pg_temp.pool_patch(auth,marker,
  'SELECT coalesce(jsonb_agg((to_jsonb(rows)-''funding'')||rows.funding ORDER BY job_id)');
 marker:=$body$AS daily_cap_microusd
   FROM public.collection_registrations r$body$;
 PERFORM pg_temp.pool_patch(statement,marker,$body$AS daily_cap_microusd,
          CASE WHEN r.consent#>>'{body,task_scope}'='sales_sandbox' THEN
           (SELECT jsonb_build_object('spend_limit_microusd',k.daily_cap_microusd,'priority',(r.consent#>>'{body,priority}')::int,
                                  'available_microusd',CASE WHEN r.revoked_at IS NULL AND k.revoked_at IS NULL
                                      THEN public.collection_pool_available_v1(k.key_id,NULL) ELSE 0 END)
            FROM public.collection_keys k WHERE k.registration_id=r.registration_id AND k.provider='openrouter') END pool_info
   FROM public.collection_registrations r$body$);
 marker:='SELECT jsonb_agg(to_jsonb(e) ORDER BY created_at DESC,registration_id)';
 PERFORM pg_temp.pool_patch(statement,marker,$body$SELECT jsonb_agg(CASE WHEN e.pool_info IS NULL THEN to_jsonb(e)-'pool_info'
   ELSE (to_jsonb(e)-ARRAY['pool_info','daily_cap_microusd','max_slots_per_utc_day','effective_max_slots_per_utc_day'])||e.pool_info
   END ORDER BY created_at DESC,registration_id)$body$);
 PERFORM pg_temp.pool_patch('public.collection_participation_v1()', 'WHEN ''synthetic_corrections'' THEN ''correction''',
    'WHEN ''sales_sandbox'' THEN ''pool'' WHEN ''synthetic_corrections'' THEN ''correction''');
 -- The same classification is used by the work and billing CTEs.
 SELECT pg_get_functiondef('public.collection_participation_v1()'::regprocedure) INTO old;
 marker:=$body$CASE WHEN j.batch_id LIKE 'xp-%' THEN 'correction' ELSE 'comparison' END arm$body$;
 IF cardinality(string_to_array(old,marker))<>3 THEN RAISE EXCEPTION '758 unexpected participation classification'; END IF;
 EXECUTE replace(old,marker,$body$CASE WHEN j.batch_id LIKE 'pool-%' THEN 'pool'
    WHEN j.batch_id LIKE 'xp-%' THEN 'correction' ELSE 'comparison' END arm$body$);
END $migration$;

DO $$ DECLARE f regprocedure; BEGIN
 FOREACH f IN ARRAY ARRAY['public.collection_pool_active_v1()'::regprocedure,
  'public.collection_pool_activate_v1()'::regprocedure,'public.collection_pool_available_v1(uuid,uuid)'::regprocedure,
  'public.collection_pool_target_v1(uuid)'::regprocedure] LOOP
   EXECUTE format('ALTER FUNCTION %s OWNER TO lab_arena_owner',f);
   EXECUTE format('REVOKE ALL ON FUNCTION %s FROM PUBLIC,anon,authenticated,service_role,lab_arena_service',f);
 END LOOP;
 GRANT EXECUTE ON FUNCTION public.collection_pool_active_v1() TO lab_arena_service;
 IF NOT (SELECT had_create FROM pg_temp.pool_previous_acl) THEN REVOKE CREATE ON SCHEMA public FROM lab_arena_owner; END IF;
 IF (SELECT nspacl FROM pg_namespace WHERE nspname='public') IS DISTINCT FROM (SELECT nspacl FROM pg_temp.pool_previous_acl) THEN
  RAISE EXCEPTION 'collection schema privileges changed'; END IF;
END $$;
NOTIFY pgrst,'reload schema';
COMMIT;
