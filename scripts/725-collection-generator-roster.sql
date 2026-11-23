-- Refresh the six models eligible for prospective pilot pricing.
-- CREATE OR REPLACE retains the existing function owner, grants and trigger dependencies.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='60s';
SET LOCAL search_path=pg_catalog;

CREATE OR REPLACE FUNCTION public.collection_pilot_fees_v1(p_batch text,p_controls jsonb) RETURNS jsonb
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
    'mistralai/mistral-large-4-0','qwen/qwen3.8-27b','nvidia/nemotron-3.5-super-vl','deepseek/deepseek-v4.1-flash-1017']
 ) THEN RAISE EXCEPTION 'pilot measurement incomplete'; END IF;
 RETURN jsonb_build_object('prices',amounts,'budgets',caps,'job_count',real_count,
   'measurement_sha256',encode(sha256(convert_to(measurements::text,'UTF8')),'hex'));
END $$;

COMMIT;
