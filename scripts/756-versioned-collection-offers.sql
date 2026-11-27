-- The first paid epoch stays immutable. Later, miners can sign separately priced
-- batches and continue existing grants at exactly the rates they accepted.
BEGIN;
SET LOCAL search_path=pg_catalog;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='60s';
SELECT pg_advisory_xact_lock(hashtextextended('collection_reward_cutover',0));
CREATE TEMP TABLE pre756_acl ON COMMIT DROP AS SELECT nspacl,
 has_schema_privilege('lab_arena_owner','public','CREATE') had_create FROM pg_namespace WHERE nspname='public';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;

CREATE TABLE public.collection_paid_offers (
 id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
 offer_id text UNIQUE NOT NULL CHECK(offer_id ~ '^dv-[a-f0-9]{64}$'),
 previous_offer_id text UNIQUE NOT NULL CHECK(previous_offer_id ~ '^dv-[a-f0-9]{64}$' AND previous_offer_id<>offer_id),
 approval jsonb NOT NULL CHECK(approval->>'live_batch_id' IS NOT DISTINCT FROM offer_id)
);
ALTER TABLE public.collection_paid_offers OWNER TO lab_arena_owner;
ALTER TABLE public.collection_paid_offers ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON public.collection_paid_offers FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;

CREATE FUNCTION public.collection_offer_schedule_v1(p_offer text) RETURNS jsonb
LANGUAGE sql STABLE SECURITY DEFINER SET search_path=pg_catalog AS $$
 WITH initial AS (SELECT public.collection_reward_schedule_v1() s), schedules AS (
  SELECT 0::bigint id,s FROM initial WHERE s<>'{}'::jsonb
  UNION ALL
  SELECT o.id,jsonb_build_object('first_epoch',c.first_epoch,'pilot_evidence',o.approval-'observations')
  FROM public.collection_reward_cutover c CROSS JOIN public.collection_paid_offers o
 ), choices AS (
  SELECT DISTINCT ON (s#>>'{pilot_evidence,payment_rule}') id,
    jsonb_build_object('offer_id',s#>>'{pilot_evidence,live_batch_id}',
                      'task',CASE s#>>'{pilot_evidence,payment_rule}' WHEN 'verified_p7_v1' THEN 'P7' ELSE 'P8' END) choice
  FROM schedules WHERE s#>>'{pilot_evidence,live_batch_id}' IS NOT NULL
  ORDER BY s#>>'{pilot_evidence,payment_rule}',id DESC
 )
 SELECT coalesce((SELECT s || CASE WHEN p_offer IS NULL AND EXISTS(SELECT FROM public.collection_paid_offers)
   THEN jsonb_build_object('available_offers',(SELECT jsonb_agg(choice ORDER BY id DESC) FROM choices))
   ELSE '{}'::jsonb END
 FROM schedules WHERE p_offer IS NULL OR s#>>'{pilot_evidence,live_batch_id}'=p_offer ORDER BY id DESC LIMIT 1),'{}')

$$;

-- Use every accepted run from this comparable, completed prefix; failed runs
-- are kept as data and never counted as price observations. Verifier charges
-- have no miner key and cannot inflate the reference cost.
CREATE FUNCTION public.collection_offer_samples_v1(p_source text,p_through_slot bigint) RETURNS jsonb
LANGUAGE plpgsql STABLE SECURITY DEFINER SET search_path=pg_catalog AS $$
DECLARE b public.collection_batches; samples jsonb;
BEGIN
 SELECT * INTO STRICT b FROM public.collection_batches WHERE batch_id=p_source AND mode='live' AND experiment_terms_hash IS NULL;
 IF p_through_slot IS NULL OR p_through_slot NOT BETWEEN 0 AND 24575 OR NOT EXISTS(
    SELECT FROM public.collection_jobs WHERE batch_id=p_source AND slot=p_through_slot) OR
    EXISTS(SELECT FROM public.collection_jobs WHERE batch_id=p_source AND slot<=p_through_slot
     AND (status IN ('running','ready') OR (status='accepted' AND public.collection_verified_generator_bill_v1(job_id) IS NULL))) THEN
  RAISE EXCEPTION 'incomplete production price observations'; END IF;
 SELECT jsonb_agg(jsonb_build_object('job_id',j.job_id,'model',j.model,'scenario_id',j.scenario_id,
  'harness_commit',j.harness_commit,'artifact_sha256',j.artifact_sha256,'budget_microusd',j.budget_microusd,
  'cost_microusd',(v.bill->>'cost_microusd')::bigint,'calls',(v.bill->>'calls')::bigint) ORDER BY j.slot,j.attempt)
 INTO samples FROM public.collection_jobs j
 CROSS JOIN LATERAL (SELECT public.collection_verified_generator_bill_v1(j.job_id) bill) v
 WHERE j.batch_id=p_source AND j.slot<=p_through_slot AND j.status='accepted' AND j.mode='live';
 IF samples IS NULL THEN RAISE EXCEPTION 'no accepted production costs'; END IF;
 RETURN samples;
END $$;

CREATE FUNCTION public.collection_paid_offer_guard_v1() RETURNS trigger
LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog AS $$
DECLARE a jsonb; source jsonb; samples jsonb; family text; task text; fees jsonb; caps jsonb; n integer;
BEGIN
 IF TG_OP<>'INSERT' THEN RAISE EXCEPTION 'collection offers are immutable'; END IF;
 PERFORM pg_advisory_xact_lock(hashtextextended('collection_reward_cutover',0));
 a:=NEW.approval;
 IF NOT EXISTS(SELECT FROM public.collection_reward_cutover)
   OR public.collection_offer_schedule_v1(NULL)#>>'{pilot_evidence,live_batch_id}' IS DISTINCT FROM NEW.previous_offer_id
   OR public.collection_offer_schedule_v1(NEW.offer_id)<>'{}'::jsonb THEN
  RAISE EXCEPTION 'paid offer predecessor changed'; END IF;
 family:=CASE a->>'payment_rule' WHEN 'verified_p7_v1' THEN 'P' WHEN 'verified_sales_evidence_v1' THEN 'P8' END;
 task:=CASE family WHEN 'P' THEN 'P7' WHEN 'P8' THEN 'P8' END;
 IF family IS NULL OR a->>'schema' IS DISTINCT FROM 'leadpoet.collection.verified_task_prices.v1'
   OR a->>'terms_sha256' IS DISTINCT FROM 'sha256:813ddcb2d471870fc142a36bee50725a0818487e31fdd1608e2acf106db4ee8d'
   OR a->>'capture_version' IS DISTINCT FROM 'collection.workflow-events.v2'
   OR coalesce(a->>'harness_commit','') !~ '^[a-f0-9]{40}$' OR coalesce(a->>'builder_commit','') !~ '^[a-f0-9]{40}$'
   OR coalesce(a->>'judge_prompt_sha256','') !~ '^[a-f0-9]{64}$' OR coalesce(a->>'split_manifest_sha256','') !~ '^[a-f0-9]{64}$'
   OR jsonb_typeof(a->'routes') IS DISTINCT FROM 'object' THEN RAISE EXCEPTION 'invalid paid offer policy'; END IF;
 IF a ? 'historical_realization' THEN
  IF jsonb_typeof(a->'historical_realization') IS DISTINCT FROM 'object'
    OR (a->'historical_realization') - ARRAY['quote_epoch','payment_block','block_hash','quoted_microusd','received_microusd_at_signed_price'] <> '{}'::jsonb
    OR coalesce(a#>>'{historical_realization,block_hash}','') !~ '^0x[a-f0-9]{64}$'
    OR coalesce((a#>>'{historical_realization,quote_epoch}')::bigint,0)<=0
    OR coalesce((a#>>'{historical_realization,payment_block}')::bigint,0)<=0
    OR coalesce((a#>>'{historical_realization,received_microusd_at_signed_price}')::bigint,-1)<0
    OR coalesce((a#>>'{historical_realization,quoted_microusd}')::bigint,0)<=0
    OR (SELECT sum(value::bigint) FROM jsonb_each_text(public.collection_reward_quotes_v1(
          (a#>>'{historical_realization,quote_epoch}')::bigint)->'quotes_microusd'))
       IS DISTINCT FROM (a#>>'{historical_realization,quoted_microusd}')::numeric
  THEN RAISE EXCEPTION 'invalid prior payout observation'; END IF;
 END IF;
 samples:=a->'observations';
 IF a->>'measurement_kind'='settled_production' THEN
  source:=public.collection_offer_schedule_v1(a->>'batch_id')->'pilot_evidence';
  IF source IS NULL OR source->>'payment_rule' IS DISTINCT FROM a->>'payment_rule'
     OR source->>'harness_commit' IS DISTINCT FROM a->>'harness_commit' OR source->>'builder_commit' IS DISTINCT FROM a->>'builder_commit'
     OR source->>'split_manifest_sha256' IS DISTINCT FROM a->>'split_manifest_sha256'
     OR NOT ((source->'routes') @> (a->'routes'))
     OR samples IS DISTINCT FROM public.collection_offer_samples_v1(a->>'batch_id',(a->>'through_slot')::bigint)
  THEN RAISE EXCEPTION 'paid offer differs from settled production source'; END IF;
 ELSIF a->>'measurement_kind'='owner_provider_receipts' THEN
  IF family<>'P8' OR coalesce(a->>'batch_id','') !~ '^owner-p8-[a-f0-9]{64}$'
     OR EXISTS(SELECT FROM public.collection_jobs WHERE batch_id=a->>'batch_id') THEN
   RAISE EXCEPTION 'invalid owner measurement source'; END IF;
 ELSE RAISE EXCEPTION 'unknown price observation source'; END IF;
 IF jsonb_typeof(samples) IS DISTINCT FROM 'array' OR jsonb_array_length(samples) NOT BETWEEN 1 AND 24576 THEN
  RAISE EXCEPTION 'missing paid offer measurements'; END IF;
 SELECT count(DISTINCT s->>'job_id') INTO n FROM jsonb_array_elements(samples) s;
 IF n<>jsonb_array_length(samples) OR (a->>'job_count')::int IS DISTINCT FROM n
  OR EXISTS(SELECT FROM jsonb_array_elements(samples) s
   WHERE coalesce(s->>'job_id','') !~ '^[a-f0-9]{8}(-[a-f0-9]{4}){3}-[a-f0-9]{12}$'
     OR s->>'harness_commit' IS DISTINCT FROM a->>'harness_commit'
     OR coalesce(s->>'artifact_sha256','') !~ '^sha256:[a-f0-9]{64}$'
     OR coalesce(s->>'scenario_id','') !~ ('^'||task||'-s[1-9][0-9]{0,7}$')
     OR NOT coalesce((a->'routes') ? (s->>'model'),false)
     OR coalesce((s->>'calls')::bigint,0) NOT BETWEEN 1 AND 100000
     OR coalesce((s->>'budget_microusd')::bigint,0) NOT BETWEEN 1 AND 10000000000
     OR coalesce((s->>'cost_microusd')::bigint,0) NOT BETWEEN 1 AND (s->>'budget_microusd')::bigint
     OR (a->>'measurement_kind'='owner_provider_receipts' AND
       coalesce(s->>'provider_receipts_sha256','') !~ '^[a-f0-9]{64}$'))
 THEN RAISE EXCEPTION 'invalid settled price observations'; END IF;
 SELECT jsonb_object_agg(model,fee),jsonb_object_agg(model,cap) INTO fees,caps FROM (
  SELECT s->>'model' model,ceil(avg((s->>'cost_microusd')::bigint)*1.25) fee,
    min((s->>'budget_microusd')::bigint) cap FROM jsonb_array_elements(samples) s GROUP BY 1
 ) rates;
 IF a->'prices' IS DISTINCT FROM jsonb_build_object(family,fees) OR a->'budgets' IS DISTINCT FROM jsonb_build_object(family,caps)
   OR EXISTS(SELECT FROM jsonb_object_keys(a->'routes') k WHERE NOT fees ? k)
 THEN RAISE EXCEPTION 'fixed fees or spend caps differ from observations'; END IF;
 RETURN NEW;
END $$;
CREATE TRIGGER check_paid_offer BEFORE INSERT OR UPDATE OR DELETE ON public.collection_paid_offers
 FOR EACH ROW EXECUTE FUNCTION public.collection_paid_offer_guard_v1();
CREATE TRIGGER no_truncate_paid_offers BEFORE TRUNCATE ON public.collection_paid_offers
 FOR EACH STATEMENT EXECUTE FUNCTION public.collection_paid_offer_guard_v1();

ALTER TABLE public.collection_jobs DROP CONSTRAINT collection_synthetic_key_scope;
ALTER TABLE public.collection_jobs ADD CONSTRAINT collection_synthetic_key_scope CHECK (
 deepline_key_id IS NOT NULL OR scenario_id ~ '^(C[0-9]{2}|K[0-9]{2}|B[1-8]|E[1-7]|P[1-7])-s[1-9][0-9]{0,7}$'
 OR (mode='live' AND batch_id ~ '^dv-[a-f0-9]{64}$' AND scenario_id ~ '^P8-s[1-9][0-9]{0,7}$')
);

-- Keep the canonical allocator, enrollment restrictions and settlement
-- transaction. Their previously global reads now resolve the exact signed offer.
DO $migration$
DECLARE r record; src text; target regprocedure;
BEGIN
 FOR r IN SELECT * FROM (VALUES
 ('public.collection_register_v1(uuid,text,text,text,jsonb,bigint,jsonb,jsonb)',
  'FROM public.collection_reward_schedule_v1() schedule','FROM public.collection_offer_schedule_v1(p_consent#>>''{body,offer_id}'') schedule'),
 ('public.collection_register_v1(uuid,text,text,text,jsonb,bigint,jsonb,jsonb)',
  'schedule#>>''{pilot_evidence,schema}''=''leadpoet.collection.verified_p7_prices.v1''',
  'schedule#>>''{pilot_evidence,schema}'' IN (''leadpoet.collection.verified_p7_prices.v1'',''leadpoet.collection.verified_task_prices.v1'')'),
 ('public.collection_approved_job_price_v1()',
  'approval:=public.collection_reward_schedule_v1()->''pilot_evidence'';',
  'approval:=CASE WHEN NEW.batch_id ~ ''^dv-[a-f0-9]{64}$'' THEN public.collection_offer_schedule_v1(NEW.batch_id)
                  ELSE public.collection_reward_schedule_v1() END->''pilot_evidence'';'),
 ('public.collection_approved_job_price_v1()',
  'approval->>''schema''=''leadpoet.collection.verified_p7_prices.v1'' AND',
  'approval->>''schema'' IN (''leadpoet.collection.verified_p7_prices.v1'',''leadpoet.collection.verified_task_prices.v1'') AND'),
 ('public.collection_approved_job_price_v1()',
  'NEW.scenario_id !~ ''^P7-s[1-9][0-9]{0,7}$''',
  'NEW.scenario_id !~ (CASE approval->>''payment_rule'' WHEN ''verified_p7_v1'' THEN ''^P7-s[1-9][0-9]{0,7}$''
                          WHEN ''verified_sales_evidence_v1'' THEN ''^P8-s[1-9][0-9]{0,7}$'' ELSE ''(?!)'' END)'),
 ('public.collection_approved_job_price_v1()','family:=public.collection_fee_class_v1(NEW.scenario_id);',
  'family:=CASE WHEN approval->>''payment_rule''=''verified_sales_evidence_v1'' THEN ''P8'' ELSE public.collection_fee_class_v1(NEW.scenario_id) END;'),
 ('public.collection_reward_job_guard_v1()',
  'NEW.quality->>''payment_rule'' IS DISTINCT FROM ''verified_p7_v1''',
  'NEW.quality->>''payment_rule'' IS DISTINCT FROM public.collection_offer_schedule_v1(NEW.batch_id)#>>''{pilot_evidence,payment_rule}'''),
 ('public.collection_priced_claim_v1(uuid,text,text,text,text,jsonb,jsonb,jsonb,jsonb,text,text)',
  's.id !~ ''^(C[0-9]{2}|K[0-9]{2}|B[1-8]|E[1-7]|P[1-7])-s[1-9][0-9]{0,7}$''',
  '(s.id !~ ''^(C[0-9]{2}|K[0-9]{2}|B[1-8]|E[1-7]|P[1-7])-s[1-9][0-9]{0,7}$'' AND
     (p_mode<>''live'' OR p_batch !~ ''^dv-[a-f0-9]{64}$'' OR s.id !~ ''^P8-s[1-9][0-9]{0,7}$''))')
 ) AS changes(sig,from_sql,to_sql) LOOP
  target:=r.sig::regprocedure;
  SELECT pg_get_functiondef(oid) INTO STRICT src FROM pg_proc WHERE oid=target AND proowner='lab_arena_owner'::regrole AND prosecdef;
  IF cardinality(string_to_array(src,r.from_sql))<>2 THEN RAISE EXCEPTION '756 unexpected function predecessor %',target; END IF;
  EXECUTE replace(src,r.from_sql,r.to_sql);
 END LOOP;
END $migration$;
DO $$ DECLARE f regprocedure; BEGIN
 FOREACH f IN ARRAY ARRAY['public.collection_offer_schedule_v1(text)'::regprocedure,'public.collection_offer_samples_v1(text,bigint)'::regprocedure,
                          'public.collection_paid_offer_guard_v1()'::regprocedure] LOOP
   EXECUTE format('ALTER FUNCTION %s OWNER TO lab_arena_owner',f);
   EXECUTE format('REVOKE ALL ON FUNCTION %s FROM PUBLIC,anon,authenticated,service_role,lab_arena_service',f);
 END LOOP;
 GRANT EXECUTE ON FUNCTION public.collection_offer_schedule_v1(text),public.collection_offer_samples_v1(text,bigint) TO lab_arena_service;
 IF NOT (SELECT had_create FROM pg_temp.pre756_acl) THEN REVOKE CREATE ON SCHEMA public FROM lab_arena_owner; END IF;
 IF (SELECT nspacl FROM pg_namespace WHERE nspname='public') IS DISTINCT FROM (SELECT nspacl FROM pg_temp.pre756_acl) THEN
  RAISE EXCEPTION 'collection schema privileges changed'; END IF;
END $$;
NOTIFY pgrst,'reload schema';
COMMIT;
