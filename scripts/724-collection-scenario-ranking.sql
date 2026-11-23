-- Register each immutable, already-stored judge input and result before closing the job.
-- The last ready worker owns the single comparison call inside its existing spend cap.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='60s';
CREATE TEMP TABLE collection717_acl ON COMMIT DROP AS
 SELECT nspacl,has_schema_privilege('lab_arena_owner','public','CREATE') AS had_create
 FROM pg_namespace WHERE nspname='public';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;
CREATE TABLE public.collection_ranking_members (
 job_id uuid PRIMARY KEY REFERENCES public.collection_jobs(job_id),
 packet_sha256 text NOT NULL CHECK(packet_sha256 ~ '^sha256:[0-9a-f]{64}$'),
 assessment_sha256 text NOT NULL CHECK(assessment_sha256 ~ '^sha256:[0-9a-f]{64}$'),
 ranks_pack boolean NOT NULL
);
ALTER TABLE public.collection_ranking_members OWNER TO lab_arena_owner;
ALTER TABLE public.collection_ranking_members ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON public.collection_ranking_members FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;
CREATE TRIGGER collection_ranking_members_immutable BEFORE UPDATE OR DELETE ON public.collection_ranking_members
 FOR EACH ROW EXECUTE FUNCTION public.lab_arena_append_only_v1();
CREATE FUNCTION public.collection_rank_pack_v1(p_job uuid,p_lease text,p_packet text,p_assessment text)
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
   WHERE q.batch_id=j.batch_id AND q.slot BETWEEN first_slot AND first_slot+5;
   INSERT INTO public.collection_ranking_members VALUES(p_job,p_packet,p_assessment,ready) RETURNING * INTO m;
 END IF;
 IF NOT m.ranks_pack THEN RETURN jsonb_build_object('status','waiting'); END IF;
 SELECT jsonb_agg(jsonb_build_object('job_id',q.job_id,'packet_sha256',r.packet_sha256,
   'assessment_sha256',r.assessment_sha256) ORDER BY q.slot) INTO members
 FROM public.collection_jobs q JOIN public.collection_ranking_members r ON r.job_id=q.job_id
 WHERE q.batch_id=j.batch_id AND q.scenario_id=j.scenario_id AND q.slot BETWEEN first_slot AND first_slot+5;
 RETURN jsonb_build_object('status','ready','members',members);
END $$;
ALTER FUNCTION public.collection_rank_pack_v1(uuid,text,text,text) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.collection_rank_pack_v1(uuid,text,text,text) FROM PUBLIC,anon,authenticated,service_role;
GRANT EXECUTE ON FUNCTION public.collection_rank_pack_v1(uuid,text,text,text) TO lab_arena_service;
DO $$ BEGIN
 IF NOT (SELECT had_create FROM pg_temp.collection717_acl) THEN REVOKE CREATE ON SCHEMA public FROM lab_arena_owner; END IF;
 IF (SELECT nspacl FROM pg_namespace WHERE nspname='public') IS DISTINCT FROM (SELECT nspacl FROM pg_temp.collection717_acl)
 THEN RAISE EXCEPTION 'collection schema ACL differs'; END IF;
END $$;
NOTIFY pgrst,'reload schema';
COMMIT;
