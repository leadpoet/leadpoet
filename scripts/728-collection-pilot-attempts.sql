-- Keep the logical assignment stable, preserve every physical attempt and its bills.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='60s';
SET LOCAL search_path=pg_catalog;

ALTER TABLE public.collection_jobs ADD COLUMN attempt smallint NOT NULL DEFAULT 1
  CHECK (attempt=1 OR (mode='pilot' AND attempt=2));
ALTER TABLE public.collection_jobs DROP CONSTRAINT collection_jobs_batch_id_slot_key,
  ADD UNIQUE (batch_id,slot,attempt);
CREATE UNIQUE INDEX collection_one_active_slot ON public.collection_jobs(batch_id,slot)
  WHERE status<>'failed';


CREATE OR REPLACE FUNCTION public.collection_claim_v1(p_job uuid,p_token_hash text,p_batch text,p_mode text,p_commit text,
 p_scenarios jsonb,p_models jsonb,p_prices jsonb,p_budgets jsonb,p_terms_hash text)
RETURNS jsonb LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog AS $$
DECLARE b public.collection_batches; enrollment record; n bigint; generator text; spending bigint; price bigint;
 day_utc date := (clock_timestamp() AT TIME ZONE 'UTC')::date;
 job public.collection_jobs; retry_slot bigint; retry_model text; remaining bigint; attempt_number integer := 1;
BEGIN
 PERFORM pg_advisory_xact_lock(hashtextextended('lab-arena-claim-control',0));
 PERFORM pg_advisory_xact_lock(hashtextextended('collection_claim:finney:71',0));
 IF jsonb_typeof(p_scenarios) IS DISTINCT FROM 'array'
    OR EXISTS (SELECT FROM jsonb_array_elements_text(p_scenarios) s(id)
               WHERE s.id IS NULL OR s.id !~ '^(C[0-9]{2}|K[0-9]{2}|B[1-8]|E[1-7]|P[1-6])-s[1-9][0-9]{0,7}$')
 THEN RETURN jsonb_build_object('error','collection_synthetic_scope_required'); END IF;
 INSERT INTO public.collection_batches VALUES(p_batch,p_mode,p_commit,p_scenarios,p_models,p_prices,p_budgets) ON CONFLICT DO NOTHING;
 SELECT * INTO STRICT b FROM public.collection_batches WHERE batch_id=p_batch;
 IF (b.mode,b.harness_commit,b.scenario_ids,b.models,b.prices,b.budgets)
 IS DISTINCT FROM (p_mode,p_commit,p_scenarios,p_models,p_prices,p_budgets)
 THEN RETURN jsonb_build_object('error','batch_changed'); END IF;
 SELECT * INTO job FROM public.collection_jobs WHERE job_id=p_job;
 IF FOUND THEN
  IF job.lease_token_hash<>p_token_hash OR job.batch_id<>p_batch OR job.status<>'running' OR job.lease_expires_at<=clock_timestamp()
  THEN RETURN jsonb_build_object('error','claim_conflict'); END IF;
  RETURN (to_jsonb(job)-'lease_token_hash') || jsonb_build_object('owner_coldkey',
   (SELECT owner_coldkey FROM public.collection_registrations WHERE registration_id=job.registration_id));
 END IF;
 -- Existing leases above can finish while canonical release or operator holds block new work.
 IF NOT EXISTS (SELECT FROM public.lab_arena_restart_claim_control
                WHERE singleton AND NOT operator_paused AND guard_commitment='') THEN RETURN NULL; END IF;
 UPDATE public.collection_jobs SET status='failed',finished_at=clock_timestamp(),quality='{}'
 WHERE status='running' AND lease_expires_at<=clock_timestamp();
 -- A failed pilot attempt can consume only the original slot's unused budget.
 -- A later bill can exceed its reservation, so wait for every original charge to settle.
 IF p_mode='pilot' THEN
   PERFORM pg_advisory_xact_lock_shared(hashtextextended('collection_reward_cutover',0));
   IF public.collection_reward_schedule_v1()#>>'{pilot_evidence,batch_id}'=p_batch THEN RETURN NULL; END IF;
   SELECT j.slot, j.model, j.budget_microusd-cost.committed INTO retry_slot, retry_model, remaining
   FROM public.collection_jobs j
   CROSS JOIN LATERAL (
     SELECT coalesce(sum(r.cost_microusd),0) committed, count(*)=count(r.cost_microusd) settled
     FROM public.collection_calls c LEFT JOIN public.collection_call_receipts r USING(job_id,request_id)
     WHERE c.job_id=j.job_id
   ) cost
   WHERE j.batch_id=p_batch AND j.attempt=1 AND j.status='failed' AND j.artifact_sha256 IS NULL
     AND cost.settled AND j.budget_microusd>cost.committed AND NOT EXISTS (
       SELECT FROM public.collection_jobs retry WHERE retry.batch_id=p_batch AND retry.slot=j.slot AND retry.attempt=2)
   ORDER BY j.slot LIMIT 1;
 END IF;
 IF retry_slot IS NOT NULL THEN
   n:=retry_slot; generator:=retry_model; spending:=remaining; price:=0; attempt_number:=2;
 ELSE
   SELECT coalesce(max(slot)+1,0) INTO n FROM public.collection_jobs WHERE batch_id=p_batch;
   IF n >= jsonb_array_length(p_scenarios)::bigint * jsonb_array_length(p_models) THEN RETURN NULL; END IF;
   generator:=p_models->>(n%6)::integer; spending:=(p_budgets->>generator)::bigint; price:=(p_prices->>generator)::bigint;
 END IF;
 SELECT r.registration_id,r.miner_hotkey,o.key_id openrouter_key_id INTO enrollment
 FROM public.collection_registrations r JOIN public.collection_keys o ON o.registration_id=r.registration_id AND o.provider='openrouter'
 WHERE r.mode=p_mode AND r.revoked_at IS NULL AND o.revoked_at IS NULL
 AND r.consent->'body'->>'task_scope'='synthetic_workflows'
 AND r.consent->'body'->>'terms_sha256'=p_terms_hash
 AND NOT EXISTS (
   SELECT FROM public.collection_jobs j
   JOIN public.collection_registrations owner ON owner.registration_id=j.registration_id
   JOIN public.collection_keys funding ON funding.key_id=j.openrouter_key_id
   WHERE ((j.mode=p_mode AND j.collection_day=day_utc) OR j.status='running') AND
         (j.miner_hotkey=r.miner_hotkey OR owner.owner_coldkey=r.owner_coldkey OR funding.fingerprint=o.fingerprint)
 )
 ORDER BY (SELECT max(j.created_at) FROM public.collection_jobs j WHERE j.miner_hotkey=r.miner_hotkey) ASC NULLS FIRST,
 r.created_at,r.registration_id LIMIT 1 FOR UPDATE OF r;
 IF NOT FOUND THEN RETURN NULL; END IF;
 INSERT INTO public.collection_jobs(job_id,batch_id,mode,registration_id,miner_hotkey,openrouter_key_id,deepline_key_id,
   slot,attempt,collection_day,scenario_id,model,harness_commit,budget_microusd,price_microusd,lease_token_hash,lease_expires_at)
 VALUES(p_job,p_batch,p_mode,enrollment.registration_id,enrollment.miner_hotkey,enrollment.openrouter_key_id,NULL,
   n,attempt_number,day_utc,p_scenarios->>(n/6)::integer,generator,p_commit,spending,price,p_token_hash,clock_timestamp()+interval '2 hours')
 RETURNING * INTO job;
 RETURN (to_jsonb(job)-'lease_token_hash') || jsonb_build_object('owner_coldkey',
   (SELECT owner_coldkey FROM public.collection_registrations WHERE registration_id=job.registration_id));
END; $$;

CREATE OR REPLACE FUNCTION public.collection_batch_jobs_v1(p_batch text) RETURNS jsonb
LANGUAGE sql STABLE SECURITY DEFINER SET search_path=pg_catalog AS $$
 SELECT coalesce(jsonb_agg(to_jsonb(job) ORDER BY job.slot,job.attempt),'[]') FROM (
    SELECT j.job_id,j.batch_id,j.mode,j.slot,j.attempt,j.scenario_id,j.model,j.harness_commit,j.miner_hotkey,
           j.artifact_sha256,j.status,j.budget_microusd,j.price_microusd
    FROM public.collection_jobs j WHERE j.batch_id=p_batch AND j.network='finney' AND j.netuid=71
 ) job
$$;

CREATE OR REPLACE FUNCTION public.collection_pilot_costs_v1(p_batch text) RETURNS jsonb
LANGUAGE sql STABLE SECURITY DEFINER SET search_path=pg_catalog AS $$
 WITH measured AS (
  SELECT j.job_id,j.batch_id,j.slot,j.attempt,j.scenario_id,j.model,j.harness_commit,j.miner_hotkey,j.artifact_sha256,
    j.status,j.budget_microusd,
    count(*) FILTER (WHERE c.key_id IS NOT NULL AND r.cost_microusd IS NULL) unresolved,
    count(*) FILTER (WHERE c.provider='openrouter' AND r.http_status=200 AND r.cost_microusd>0) generator_calls,
    coalesce(sum(r.cost_microusd) FILTER (WHERE c.provider IN ('openrouter','deepline') AND c.key_id IS NOT NULL),0) miner_cost_microusd
  FROM public.collection_jobs j
  LEFT JOIN public.collection_calls c ON j.job_id=c.job_id
  LEFT JOIN public.collection_call_receipts r ON (c.request_id,c.job_id)=(r.request_id,r.job_id)
  WHERE j.batch_id=p_batch AND j.mode='pilot' AND j.network='finney' AND j.netuid=71
  GROUP BY j.job_id
 ) SELECT coalesce(jsonb_agg(to_jsonb(m) ORDER BY job_id),'[]') FROM measured m
$$;

CREATE OR REPLACE FUNCTION public.collection_pilot_fees_v1(p_batch text,p_controls jsonb) RETURNS jsonb
LANGUAGE plpgsql STABLE SECURITY DEFINER SET search_path=pg_catalog AS $$
DECLARE measurements jsonb; amounts jsonb; caps jsonb; bad boolean; real_count bigint; fingerprint jsonb; expected_slots integer;
BEGIN
 measurements:=public.collection_pilot_costs_v1(p_batch);
 IF jsonb_typeof(p_controls) IS DISTINCT FROM 'array' OR jsonb_array_length(p_controls)>4096
    OR jsonb_array_length(measurements)=0 THEN RAISE EXCEPTION 'invalid pilot jobs or controls'; END IF;
 IF EXISTS (SELECT FROM jsonb_array_elements_text(p_controls) c(id)
    WHERE NOT EXISTS (SELECT FROM jsonb_array_elements(measurements) m(doc) WHERE m.doc->>'job_id'=c.id))
    OR EXISTS (SELECT FROM jsonb_array_elements_text(p_controls) c(id) GROUP BY c.id HAVING count(*)>1)
 THEN RAISE EXCEPTION 'foreign or duplicate controls'; END IF;

 -- A missing attempt, a running attempt, or an unknown miner bill cannot be omitted.
 SELECT jsonb_array_length(scenario_ids)*jsonb_array_length(models) INTO STRICT expected_slots
 FROM public.collection_batches WHERE batch_id=p_batch AND mode='pilot';
 IF (SELECT count(DISTINCT (m->>'slot')::bigint) FROM jsonb_array_elements(measurements) m) <> expected_slots
    OR EXISTS (SELECT FROM jsonb_array_elements(measurements) m
      WHERE (m->>'slot')::bigint>=expected_slots OR (m->>'unresolved')::bigint<>0
      OR m->>'status' NOT IN ('accepted','rejected','failed')
      OR (m->>'status'='failed' AND (m->>'artifact_sha256' IS NOT NULL OR p_controls ? (m->>'job_id'))))
 THEN RAISE EXCEPTION 'pilot measurement incomplete'; END IF;

 WITH data AS (
  SELECT p.*,original.budget_microusd AS slot_budget, public.collection_fee_class_v1(p.scenario_id) AS family
  FROM jsonb_to_recordset(measurements) p(job_id text,slot bigint,status text,artifact_sha256 text,scenario_id text,model text,
       budget_microusd bigint,unresolved bigint,generator_calls bigint,miner_cost_microusd bigint)
  JOIN public.collection_jobs original ON original.batch_id=p_batch AND original.slot=p.slot AND original.attempt=1
  WHERE p.status<>'failed' AND NOT p_controls ? p.job_id
 ), by_model AS (
  SELECT family,model,count(*) jobs,ceil(sum(miner_cost_microusd)*5/(4*count(*))) fee,
         max(slot_budget) cap,bool_or(status NOT IN ('accepted','rejected') OR artifact_sha256 IS NULL
           OR family IS NULL OR unresolved>0 OR generator_calls=0 OR miner_cost_microusd<=0) OR
           min(slot_budget)<>max(slot_budget) invalid
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
    'mistralai/mistral-large-4-0','qwen/qwen3.8-27b','nvidia/nemotron-3.5-super-vl','deepseek/deepseek-v4.1-flash-1017']
 ) THEN RAISE EXCEPTION 'pilot measurement incomplete'; END IF;
 -- Preserve the pre-migration commitment for a cohort with no replacement attempts.
 SELECT jsonb_agg(CASE WHEN (doc->>'attempt')::int=1 THEN doc-'attempt' ELSE doc END ORDER BY i)
 INTO fingerprint FROM jsonb_array_elements(measurements) WITH ORDINALITY rows(doc,i);
 RETURN jsonb_build_object('prices',amounts,'budgets',caps,'job_count',real_count,
   'measurement_sha256',encode(sha256(convert_to(fingerprint::text,'UTF8')),'hex'));
END $$;

CREATE OR REPLACE FUNCTION public.collection_verify_funding_v1() RETURNS trigger
LANGUAGE plpgsql SET search_path=pg_catalog AS $fn$
BEGIN
 IF NEW.status='accepted' AND OLD.status IS DISTINCT FROM NEW.status AND (
   NOT EXISTS(SELECT FROM public.collection_calls c JOIN public.collection_call_receipts r USING(job_id,request_id)
     WHERE c.job_id=NEW.job_id AND c.provider='openrouter' AND r.http_status=200 AND r.cost_microusd>0)
   OR EXISTS(SELECT FROM public.collection_calls c LEFT JOIN public.collection_call_receipts r USING(job_id,request_id)
     WHERE c.job_id=NEW.job_id AND c.key_id IS NOT NULL AND r.cost_microusd IS NULL)
 ) THEN RAISE EXCEPTION 'missing verified collection funding'; END IF;
 RETURN NEW;
END $fn$;

CREATE OR REPLACE FUNCTION public.collection_rank_pack_v1(p_job uuid,p_lease text,p_packet text,p_assessment text)
RETURNS jsonb LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog AS $$
DECLARE j public.collection_jobs; m public.collection_ranking_members; first_slot bigint; ready boolean; members jsonb;
BEGIN
 SELECT * INTO STRICT j FROM public.collection_jobs WHERE job_id=p_job FOR UPDATE;
 IF j.status<>'running' OR j.lease_token_hash IS DISTINCT FROM p_lease OR j.lease_expires_at<=clock_timestamp()
 THEN RAISE EXCEPTION 'inactive_collection_lease'; END IF;
 first_slot:=(j.slot / 6)*6;
 PERFORM pg_advisory_xact_lock(hashtextextended('collection_rank:'||j.batch_id||':'||first_slot,0));
 SELECT * INTO m FROM public.collection_ranking_members WHERE job_id=p_job;
 IF FOUND THEN
   IF (m.packet_sha256,m.assessment_sha256) IS DISTINCT FROM (p_packet,p_assessment)
   THEN RAISE EXCEPTION 'collection_ranking_input_changed'; END IF;
 ELSE
   SELECT count(*)=6 AND count(DISTINCT q.model)=6 AND bool_and(q.scenario_id=j.scenario_id)
     AND bool_and(q.job_id=p_job OR r.job_id IS NOT NULL OR q.status='failed') INTO ready
   FROM public.collection_jobs q LEFT JOIN public.collection_ranking_members r ON r.job_id=q.job_id
   WHERE q.batch_id=j.batch_id AND q.slot BETWEEN first_slot AND first_slot+5
       AND (j.mode<>'pilot' OR q.status<>'failed');
   INSERT INTO public.collection_ranking_members VALUES(p_job,p_packet,p_assessment,ready) RETURNING * INTO m;
 END IF;
 IF NOT m.ranks_pack THEN RETURN jsonb_build_object('status','waiting'); END IF;
 SELECT jsonb_agg(jsonb_build_object('job_id',q.job_id,'packet_sha256',r.packet_sha256,
   'assessment_sha256',r.assessment_sha256) ORDER BY q.slot) INTO members
 FROM public.collection_jobs q JOIN public.collection_ranking_members r ON r.job_id=q.job_id
 WHERE q.batch_id=j.batch_id AND q.scenario_id=j.scenario_id AND q.slot BETWEEN first_slot AND first_slot+5
       AND (j.mode<>'pilot' OR q.status<>'failed');
 RETURN jsonb_build_object('status','ready','members',members);
END $$;

NOTIFY pgrst,'reload schema';
COMMIT;
