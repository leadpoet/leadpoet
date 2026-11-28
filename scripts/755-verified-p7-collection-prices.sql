-- Price native, replayable P7 work from completed funded runs. Existing judged
-- cohorts retain their original admission, price and export rules.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='60s';
SET LOCAL search_path=pg_catalog;
SELECT pg_advisory_xact_lock(hashtextextended('collection_claim:finney:71',0));
CREATE TEMP TABLE pre755_acl ON COMMIT DROP AS SELECT nspacl,
 has_schema_privilege('lab_arena_owner','public','CREATE') AS had_create FROM pg_namespace WHERE nspname='public';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;

-- A settled generator bill belongs to this miner's exact enrolled credential.
-- Failed, duplicated, unpriced and out-of-scope calls cannot set a fee.
CREATE FUNCTION public.collection_verified_generator_bill_v1(p_job uuid) RETURNS jsonb
LANGUAGE sql STABLE SECURITY DEFINER SET search_path=pg_catalog AS $$
 SELECT jsonb_build_object('calls',count(*),'cost_microusd',sum(r.cost_microusd))
 FROM public.collection_jobs j JOIN public.collection_calls c ON c.job_id=j.job_id
 LEFT JOIN public.collection_call_receipts r ON (r.job_id,r.request_id)=(c.job_id,c.request_id)
 WHERE j.job_id=p_job AND c.key_id IS NOT NULL
 GROUP BY j.job_id
 HAVING bool_and(c.provider='openrouter' AND c.key_id=j.openrouter_key_id
    AND r.http_status IS NOT DISTINCT FROM 200 AND r.cost_microusd IS NOT NULL AND r.provider_id IS NOT NULL)
   AND count(*)=count(DISTINCT r.provider_id) AND sum(r.cost_microusd) BETWEEN 1 AND j.budget_microusd
$$;

CREATE FUNCTION public.collection_verified_p7_fees_v1(p_batch text,p_jobs jsonb) RETURNS jsonb
LANGUAGE plpgsql STABLE SECURITY DEFINER SET search_path=pg_catalog AS $$
DECLARE samples jsonb; fees jsonb; caps jsonb;
BEGIN
 IF jsonb_typeof(p_jobs) IS DISTINCT FROM 'array' OR jsonb_array_length(p_jobs) NOT BETWEEN 1 AND 4096 THEN
  RAISE EXCEPTION 'missing P7 price observations'; END IF;
 SELECT jsonb_agg(jsonb_build_object('job_id',j.job_id,'model',j.model,'scenario',j.scenario_id,
    'artifact_sha256',j.artifact_sha256,'budget',original.budget_microusd,'bill',bill.doc) ORDER BY j.job_id)
 INTO samples
 FROM public.collection_jobs j JOIN public.collection_batches b ON b.batch_id=j.batch_id
 JOIN public.collection_jobs original ON original.batch_id=j.batch_id AND original.slot=j.slot AND original.attempt=1
 CROSS JOIN LATERAL (SELECT public.collection_verified_generator_bill_v1(j.job_id) doc) bill
 WHERE j.batch_id=p_batch AND p_jobs ? j.job_id::text AND b.mode='pilot' AND j.mode='pilot'
   AND j.status='accepted' AND j.artifact_sha256 IS NOT NULL AND j.price_microusd=0 AND
   j.scenario_id ~ '^P7-s[1-9][0-9]{0,7}$' AND j.harness_commit=b.harness_commit AND bill.doc IS NOT NULL;
 IF coalesce(jsonb_array_length(samples),0)<>jsonb_array_length(p_jobs) THEN
  RAISE EXCEPTION 'P7 price observation is missing, repeated, incomplete or unsettled'; END IF;
 SELECT jsonb_object_agg(model,fee),jsonb_object_agg(model,cap) INTO fees,caps FROM (
  SELECT s->>'model' model,ceil(avg((s#>>'{bill,cost_microusd}')::bigint)*1.25) fee,
         min((s->>'budget')::bigint) cap
  FROM jsonb_array_elements(samples) s GROUP BY s->>'model'
 ) rates;
 RETURN jsonb_build_object('prices',jsonb_build_object('P',fees),'budgets',jsonb_build_object('P',caps),
  'measurement_sha256',encode(sha256(convert_to(samples::text,'UTF8')),'hex'),'job_count',jsonb_array_length(samples));
END $$;

-- Preserve the immutable launch schedule, fresh signed chain anchor, escrow,
-- and first-job transaction. Only the source of the initial fixed fees changes.
DO $migration$
DECLARE fn regprocedure; definition text; marker text; replacement text; change record;
BEGIN
 FOR change IN SELECT * FROM (VALUES
 ('public.collection_approved_pilot_guard_v1()',
  ' IF evidence->>''schema'' IS DISTINCT FROM ''leadpoet.collection.approved_pilot.v1'' OR',
  $patch$ IF evidence->>'schema'='leadpoet.collection.verified_p7_prices.v1' THEN
  calculated:=public.collection_verified_p7_fees_v1(batch.batch_id,evidence->'measurement_jobs');
  IF evidence->>'payment_rule' IS DISTINCT FROM 'verified_p7_v1' OR NOT (evidence @> calculated) OR evidence->>'harness_commit' IS DISTINCT FROM batch.harness_commit
     OR coalesce(evidence->>'live_batch_id','') !~ '^dv-[a-f0-9]{64}$'
     OR coalesce(evidence->>'builder_commit','') !~ '^[a-f0-9]{40}$'
     OR coalesce(evidence->>'split_manifest_sha256','') !~ '^[a-f0-9]{64}$'
     OR coalesce(evidence->>'judge_prompt_sha256','') !~ '^[a-f0-9]{64}$'
     OR evidence->>'terms_sha256' IS DISTINCT FROM 'sha256:813ddcb2d471870fc142a36bee50725a0818487e31fdd1608e2acf106db4ee8d'
     OR evidence->>'capture_version' IS DISTINCT FROM 'collection.workflow-events.v2'
     OR jsonb_typeof(evidence->'routes') IS DISTINCT FROM 'object'
     OR (SELECT jsonb_object_agg(k,true) FROM jsonb_object_keys(evidence->'routes') k)
      IS DISTINCT FROM (SELECT jsonb_object_agg(k,true) FROM jsonb_object_keys(calculated#>'{prices,P}') k)
     OR evidence->'prices' IS DISTINCT FROM calculated->'prices' OR evidence->'budgets' IS DISTINCT FROM calculated->'budgets'
  THEN RAISE EXCEPTION 'unverified P7 price schedule'; END IF;
  RETURN NEW;
 END IF;
$patch$),
 ('public.collection_approved_job_price_v1()',
  '  family:=public.collection_fee_class_v1(NEW.scenario_id);',
  $patch$  IF approval->>'schema'='leadpoet.collection.verified_p7_prices.v1' AND
     (NEW.batch_id IS DISTINCT FROM approval->>'live_batch_id' OR NEW.scenario_id !~ '^P7-s[1-9][0-9]{0,7}$'
     OR NOT EXISTS(SELECT FROM public.collection_registrations r WHERE r.registration_id=NEW.registration_id
      AND r.mode='live' AND r.miner_hotkey=NEW.miner_hotkey AND r.consent#>>'{body,offer_id}'=NEW.batch_id
      AND r.consent#>>'{body,terms_sha256}'=approval->>'terms_sha256' AND r.consent#>>'{body,task_scope}'='synthetic_workflows'))
  THEN RAISE EXCEPTION 'unapproved P7 batch or miner grant'; END IF;
$patch$),
 ('public.collection_register_v1(uuid,text,text,text,jsonb,bigint,jsonb,jsonb)',
  '       OR (p_consent#>>''{body,task_scope}''=''synthetic_workflows''',
  $patch$       OR (p_mode='live' AND p_consent#>>'{body,task_scope}'='synthetic_workflows'
           AND p_consent->>'round_id'='finney:71:locked-collection:live'
           AND p_consent->>'scope'='lab_arena.collection.register.v1' AND p_consent->>'hotkey'=p_hotkey
           AND p_consent#>>'{body,daily_cap_microusd}'=p_cap::text
           AND EXISTS (SELECT FROM public.collection_reward_schedule_v1() schedule
            WHERE schedule#>>'{pilot_evidence,schema}'='leadpoet.collection.verified_p7_prices.v1'
             AND schedule#>>'{pilot_evidence,live_batch_id}'=p_consent#>>'{body,offer_id}'
             AND schedule#>>'{pilot_evidence,terms_sha256}'=p_consent#>>'{body,terms_sha256}'))
$patch$),
 ('public.collection_priced_claim_v1(uuid,text,text,text,text,jsonb,jsonb,jsonb,jsonb,text,text)',
  ' AND r.consent->''body''->>''terms_sha256''=p_terms_hash',
  $patch$ AND r.consent#>>'{body,offer_id}' IS NOT DISTINCT FROM
   (CASE WHEN p_batch ~ '^dv-[a-f0-9]{64}$' THEN p_batch END)
$patch$),
 ('public.collection_reward_job_guard_v1()',
  ' IF TG_OP <> ''INSERT'' AND OLD.mode=''live'' AND OLD.status=''accepted'' THEN',
  $patch$ IF TG_OP<>'DELETE' AND NEW.batch_id ~ '^dv-[a-f0-9]{64}$' AND NEW.status IN ('ready','accepted')
   AND (NEW.mode<>'live' OR NEW.quality->>'payment_rule' IS DISTINCT FROM 'verified_p7_v1'
    OR NEW.quality->>'training_status' IS DISTINCT FROM 'needs_review'
    OR public.collection_verified_generator_bill_v1(NEW.job_id) IS NULL)
  THEN RAISE EXCEPTION 'settled verified P7 execution required'; END IF;
$patch$)
 ) attrs(sig,needle,insert_sql) LOOP
  fn:=change.sig::regprocedure;
  SELECT pg_get_functiondef(oid) INTO STRICT definition FROM pg_proc
   WHERE oid=fn AND prosecdef AND proowner='lab_arena_owner'::regrole;
  marker:=change.needle; replacement:=change.insert_sql||marker;
  IF cardinality(string_to_array(definition,marker))<>2 THEN RAISE EXCEPTION '755 unexpected predecessor %',fn; END IF;
  EXECUTE replace(definition,marker,replacement);
 END LOOP;
 fn:='public.collection_register_v1(uuid,text,text,text,jsonb,bigint,jsonb,jsonb)'::regprocedure;
 SELECT pg_get_functiondef(fn) INTO definition;
 marker:=$old$p_consent#>>'{body,terms_sha256}'='sha256:bea1b86f92cde71382507dac407fa56a2c87576c5a1d7753212b5ff4c0841b0b'
     AND jsonb_typeof$old$;
 IF cardinality(string_to_array(definition,marker))<>2 THEN RAISE EXCEPTION '755 unexpected signed quota'; END IF;
 EXECUTE replace(definition,marker,$new$p_consent#>>'{body,terms_sha256}' IN
       ('sha256:bea1b86f92cde71382507dac407fa56a2c87576c5a1d7753212b5ff4c0841b0b',
        'sha256:813ddcb2d471870fc142a36bee50725a0818487e31fdd1608e2acf106db4ee8d')
     AND jsonb_typeof$new$);
 fn:='public.collection_priced_claim_v1(uuid,text,text,text,text,jsonb,jsonb,jsonb,jsonb,text,text)'::regprocedure;
 SELECT pg_get_functiondef(fn) INTO definition;
 IF cardinality(string_to_array(definition,'n%6'))<>2 OR cardinality(string_to_array(definition,'n/6'))<>2 THEN
  RAISE EXCEPTION '755 unexpected ordinary model rotation'; END IF;
 EXECUTE replace(replace(definition,'n%6','n%jsonb_array_length(p_models)'),'n/6','n/jsonb_array_length(p_models)');
END $migration$;
ALTER TABLE public.collection_batches DROP CONSTRAINT collection_batches_check;
ALTER TABLE public.collection_batches ADD CONSTRAINT collection_batches_check CHECK (
 jsonb_typeof(models)='array' AND
 CASE WHEN experiment_terms_hash IS NULL AND batch_id ~ '^dv-[a-f0-9]{64}$'
      THEN mode='live' AND jsonb_array_length(models) BETWEEN 1 AND 6
      WHEN experiment_terms_hash IS NULL THEN jsonb_array_length(models)=6
      ELSE mode IN ('pilot','live') AND batch_id ~ '^xp-[0-9a-f]{64}$' AND experiment_terms_hash ~ '^sha256:[0-9a-f]{64}$'
       AND jsonb_array_length(models)=jsonb_array_length(scenario_ids) AND jsonb_array_length(models) BETWEEN 1 AND 4096 END
);

DO $$ DECLARE fn regprocedure; BEGIN
 FOREACH fn IN ARRAY ARRAY['public.collection_verified_generator_bill_v1(uuid)'::regprocedure,
                           'public.collection_verified_p7_fees_v1(text,jsonb)'::regprocedure] LOOP
  EXECUTE format('ALTER FUNCTION %s OWNER TO lab_arena_owner',fn);
  EXECUTE format('REVOKE ALL ON FUNCTION %s FROM PUBLIC,anon,authenticated,service_role,lab_arena_service',fn);
 END LOOP;
 GRANT EXECUTE ON FUNCTION public.collection_verified_p7_fees_v1(text,jsonb) TO lab_arena_service;
 IF NOT (SELECT had_create FROM pg_temp.pre755_acl) THEN REVOKE CREATE ON SCHEMA public FROM lab_arena_owner; END IF;
 IF (SELECT nspacl FROM pg_namespace WHERE nspname='public') IS DISTINCT FROM (SELECT nspacl FROM pg_temp.pre755_acl) THEN
  RAISE EXCEPTION 'collection schema privileges changed'; END IF;
END $$;
NOTIFY pgrst,'reload schema';
COMMIT;
