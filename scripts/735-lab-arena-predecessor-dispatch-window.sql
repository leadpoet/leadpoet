-- Apply the advertised execution budget once when a prior champion decision
-- consumed the available baseline window. Stage opening already holds the row
-- lock. Both the operational receipt and new jobs commit in that transaction.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';
CREATE TEMP TABLE dispatch735_acl ON COMMIT DROP AS
 SELECT nspacl,has_schema_privilege('lab_arena_owner','public','CREATE') had_create
 FROM pg_namespace WHERE nspname='public';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;
SET LOCAL ROLE lab_arena_owner;

CREATE OR REPLACE FUNCTION public.lab_arena__predecessor_dispatch_window_v1()
RETURNS trigger LANGUAGE plpgsql SECURITY INVOKER SET search_path=pg_catalog,public
AS $window$
DECLARE
  predecessor_id text;
  schedule jsonb := NEW.configuration_doc->'schedule';
  release_at timestamptz;
  opens_at timestamptz := date_trunc('second',clock_timestamp());
  nominal_start timestamptz := (schedule->>'stage_1_start')::timestamptz;
  latest_start timestamptz := (schedule->>'stage_1_close')::timestamptz
    - make_interval(secs=>(NEW.configuration_doc->>'icp_wall_clock_seconds')::integer);
  delta interval := opens_at - nominal_start;
  windows jsonb;
  receipt jsonb;
  revised public.lab_arena_rounds;
BEGIN
  IF opens_at <= latest_start THEN RETURN NULL; END IF;
  SELECT p.round_id,greatest(p.published_at,p.baseline_promoted_at)
    INTO predecessor_id,release_at FROM public.lab_arena_rounds p
   WHERE p.arena_network_name=NEW.arena_network_name AND p.arena_netuid=NEW.arena_netuid
     AND p.rewards_enabled AND p.configuration_doc->>'mode'='live'
     AND p.evaluation_date<NEW.evaluation_date AND p.status='published'
   ORDER BY p.evaluation_date DESC,p.round_id DESC LIMIT 1;
  IF release_at IS NULL OR release_at <= latest_start OR release_at > clock_timestamp()
     OR delta <= interval '0' OR EXISTS(
       SELECT FROM public.lab_arena_scoring_supersessions WHERE round_id=NEW.round_id
         AND request_id='late-predecessor-dispatch-v1') THEN RETURN NULL; END IF;

  SELECT jsonb_object_agg(key,to_char(
    ((schedule->>key)::timestamptz + delta) AT TIME ZONE 'UTC','YYYY-MM-DD"T"HH24:MI:SS"Z"'))
    INTO windows FROM unnest(ARRAY['stage_1_start','stage_1_close','stage_1_scoring_close',
      'stage_2_start','stage_2_close','final_scoring_close','publication_deadline']) AS key;
  revised := NEW;
  revised.configuration_doc := jsonb_set(NEW.configuration_doc,'{schedule}',schedule||windows);
  receipt := jsonb_build_object('operation','round_operational_update_v1',
    'reason','late_predecessor_dispatch','predecessor',predecessor_id,
    'released_at',release_at,'windows',windows);
  INSERT INTO public.lab_arena_scoring_supersessions(round_id,generation_floor,request_id,
    request_hash,actor_ref,transaction_id,backend_pid,old_round_doc,new_round_doc,
    old_score_projections,recovery_doc)
  SELECT NEW.round_id,coalesce(max(generation_floor),0)+1,'late-predecessor-dispatch-v1',
    'sha256:'||encode(sha256(convert_to(receipt::text,'UTF8')),'hex'),'arena:open_stage',
    txid_current(),pg_backend_pid(),to_jsonb(NEW)-'updated_at',to_jsonb(revised)-'updated_at','{}',receipt
    FROM public.lab_arena_scoring_supersessions WHERE round_id=NEW.round_id;
  UPDATE public.lab_arena_rounds SET configuration_doc=revised.configuration_doc WHERE round_id=NEW.round_id;
  RETURN NULL;
END;
$window$;
REVOKE ALL ON FUNCTION public.lab_arena__predecessor_dispatch_window_v1()
  FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;
CREATE OR REPLACE TRIGGER lab_arena_predecessor_dispatch_window
  AFTER UPDATE OF status ON public.lab_arena_rounds FOR EACH ROW
  WHEN (OLD.status='committed' AND NEW.status='stage1' AND NEW.rewards_enabled
    AND NEW.configuration_doc->>'dispatch_window_policy'='late_predecessor_v1')
  EXECUTE FUNCTION public.lab_arena__predecessor_dispatch_window_v1();
RESET ROLE;
DO $acl$
BEGIN
  IF NOT (SELECT had_create FROM pg_temp.dispatch735_acl) THEN
    REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;
  END IF;
  IF (SELECT nspacl FROM pg_namespace WHERE nspname='public')
    IS DISTINCT FROM (SELECT nspacl FROM pg_temp.dispatch735_acl) THEN
    RAISE EXCEPTION 'Arena schema ACL differs';
  END IF;
END;
$acl$;
COMMIT;
