-- Limit new funded slots per UTC day, independently of the rolling key spending cap.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '60s';
SET LOCAL search_path = pg_catalog;
SELECT pg_advisory_xact_lock(hashtextextended('collection_claim:finney:71',0));

CREATE TEMP TABLE dispatch745_schema ON COMMIT DROP AS
 SELECT nspacl,has_schema_privilege('lab_arena_owner','public','CREATE') AS had_create
 FROM pg_catalog.pg_namespace WHERE nspname='public';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;

CREATE TABLE public.collection_dispatch_settings (
 singleton boolean PRIMARY KEY DEFAULT true CHECK (singleton),
 daily_slots_per_owner integer NOT NULL DEFAULT 1 CHECK (daily_slots_per_owner BETWEEN 1 AND 4096)
);
ALTER TABLE public.collection_dispatch_settings OWNER TO lab_arena_owner;
ALTER TABLE public.collection_dispatch_settings ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON public.collection_dispatch_settings FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;
INSERT INTO public.collection_dispatch_settings DEFAULT VALUES;

-- The claim advisory lock serializes admission, including different hotkeys of the same owner.
ALTER TABLE public.collection_jobs DROP CONSTRAINT collection_jobs_mode_collection_day_miner_hotkey_key;
CREATE INDEX collection_jobs_daily_usage ON public.collection_jobs(mode,collection_day,miner_hotkey);

DO $migration$
DECLARE
 target regprocedure := 'public.collection_claim_v1(uuid,text,text,text,text,jsonb,jsonb,jsonb,jsonb,text)'::regprocedure;
 definition text; original_metadata jsonb; updated_metadata jsonb;
 old_day constant text := $day$day_utc date := (clock_timestamp() AT TIME ZONE 'UTC')::date;$day$;
 choose_funder constant text := ' SELECT r.registration_id,r.miner_hotkey,o.key_id openrouter_key_id INTO enrollment';
 old_limit constant text := $old$ AND NOT EXISTS (
   SELECT FROM public.collection_jobs j
   JOIN public.collection_registrations owner ON owner.registration_id=j.registration_id
   JOIN public.collection_keys funding ON funding.key_id=j.openrouter_key_id
   WHERE ((j.mode=p_mode AND j.collection_day=day_utc) OR j.status='running') AND
         (j.miner_hotkey=r.miner_hotkey OR owner.owner_coldkey=r.owner_coldkey OR funding.fingerprint=o.fingerprint)
 )$old$;
 new_limit constant text := $new$ AND NOT EXISTS (
   SELECT FROM public.collection_jobs j
   JOIN public.collection_registrations owner ON owner.registration_id=j.registration_id
   JOIN public.collection_keys funding ON funding.key_id=j.openrouter_key_id
   WHERE ((j.mode=p_mode AND j.collection_day=day_utc) OR j.status='running')
     AND (j.miner_hotkey=r.miner_hotkey OR owner.owner_coldkey=r.owner_coldkey OR funding.fingerprint=o.fingerprint)
     -- The one settled pilot replacement reuses its slot. Running work always excludes a funder.
     AND (j.status='running' OR (j.batch_id,j.slot) <> (p_batch,n))
   HAVING bool_or(j.status='running') OR greatest(
     count(DISTINCT (j.batch_id,j.slot)) FILTER (WHERE j.miner_hotkey=r.miner_hotkey),
     count(DISTINCT (j.batch_id,j.slot)) FILTER (WHERE owner.owner_coldkey=r.owner_coldkey),
     count(DISTINCT (j.batch_id,j.slot)) FILTER (WHERE funding.fingerprint=o.fingerprint)
   ) >= daily_limit
 )$new$;
BEGIN
 SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc' INTO STRICT definition,original_metadata
 FROM pg_proc p WHERE p.oid=target AND p.prosecdef AND p.proowner='lab_arena_owner'::regrole;
 IF cardinality(string_to_array(definition,old_limit))<>2
    OR cardinality(string_to_array(definition,old_day))<>2
    OR cardinality(string_to_array(definition,choose_funder))<>2 THEN
  RAISE EXCEPTION '745 unexpected collection claim daily policy';
 END IF;
 definition := replace(definition,old_day,'day_utc date; daily_limit integer;');
 definition := replace(definition,choose_funder,$setting$ SELECT daily_slots_per_owner INTO STRICT daily_limit
 FROM public.collection_dispatch_settings WHERE singleton FOR SHARE;
 day_utc := (clock_timestamp() AT TIME ZONE 'UTC')::date;
$setting$ || choose_funder);
 EXECUTE replace(definition,old_limit,new_limit);
 SELECT to_jsonb(p)-'prosrc' INTO STRICT updated_metadata FROM pg_proc p WHERE p.oid=target;
 IF updated_metadata IS DISTINCT FROM original_metadata THEN RAISE EXCEPTION '745 changed claim privileges'; END IF;
 IF NOT (SELECT had_create FROM pg_temp.dispatch745_schema) THEN REVOKE CREATE ON SCHEMA public FROM lab_arena_owner; END IF;
 IF (SELECT nspacl FROM pg_namespace WHERE nspname='public') IS DISTINCT FROM
    (SELECT nspacl FROM pg_temp.dispatch745_schema) THEN RAISE EXCEPTION '745 changed schema privileges'; END IF;
END $migration$;
COMMIT;
