-- Render only after the Sep20 rerun326 round is terminal and every protected
-- database seal below matches. The .template suffix prevents application
-- before the final scorer image and live schedule are fixed.
BEGIN;
SET LOCAL TIME ZONE 'UTC';
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='120s';

-- Keep the immutable score run IDs in the archive. Patch only the exact
-- scoring initializer preimage so fresh score rows use a new namespace.
DO $patch_sep20_rerun328_scoring$
DECLARE
 definition TEXT;
 definition_hash TEXT;
 anchor TEXT:=$anchor$    v_status := CASE WHEN v_cache.cache_key IS NULL THEN 'pending' ELSE 'accepted' END;$anchor$;
 replacement TEXT:=$replacement$    IF p_round_id='arena-2026-09-20'
       AND EXISTS(SELECT 1 FROM public.lab_arena_rounds
         WHERE round_id='arena-2026-09-20-r328archive') THEN
      v_assignment:=p_round_id||':'||v_scored.submission_id||':'||p_stage::TEXT||':'||
        v_scored.icp_position::TEXT||':score:rerun328';
    END IF;
    v_status := CASE WHEN v_cache.cache_key IS NULL THEN 'pending' ELSE 'accepted' END;$replacement$;
BEGIN
 definition:=pg_catalog.pg_get_functiondef(
  'public.lab_arena_open_scoring_v2(text,smallint,jsonb)'::pg_catalog.regprocedure);
 definition_hash:=pg_catalog.encode(extensions.digest(definition,'sha256'),'hex');
 IF definition_hash='30c941d0b704577219842ebe76433093c7052a21fb182ea5352422ece45274a6' THEN
  IF (pg_catalog.length(definition)-pg_catalog.length(pg_catalog.replace(
      definition,anchor,'')))<>pg_catalog.length(anchor) THEN
   RAISE EXCEPTION 'Sep20 rerun328 scoring seam differs';
  END IF;
  EXECUTE pg_catalog.replace(definition,anchor,replacement);
 ELSIF definition_hash<>'0d2a69c1a4dda9a23ab36ba60219fe36da2e729dde971699eddf829b7034e3f9' THEN
  RAISE EXCEPTION 'Sep20 rerun328 scoring definition differs';
 END IF;
 IF pg_catalog.encode(extensions.digest(pg_catalog.pg_get_functiondef(
   'public.lab_arena_open_scoring_v2(text,smallint,jsonb)'::pg_catalog.regprocedure),
   'sha256'),'hex')<>'0d2a69c1a4dda9a23ab36ba60219fe36da2e729dde971699eddf829b7034e3f9' THEN
  RAISE EXCEPTION 'Sep20 rerun328 scoring patch differs';
 END IF;
END $patch_sep20_rerun328_scoring$;

DO $sep20_rerun328$
DECLARE
 active public.lab_arena_rounds;
 baseline public.lab_arena_submissions;
 archive_config JSONB;
 archive_participants JSONB;
 new_participants JSONB;
 expected_submission JSONB;
 miner_submissions_before JSONB;
 miner_submissions_after JSONB;
 miner_runs_before JSONB;
 miner_runs_after JSONB;
 miner_ledger_before JSONB;
 miner_ledger_after JSONB;
 archived_execution_judgments JSONB;
 prior_archive_before JSONB;
 prior_archive_after JSONB;
 judgments_before JSONB;
 judgments_after JSONB;
 cache_before JSONB;
 cache_after JSONB;
 active_count BIGINT;
 unsettled_count BIGINT;
 reservation_count BIGINT;
 moved_baseline_runs BIGINT;
 moved_baseline_ledger BIGINT;
 moved_score_runs BIGINT;
 moved_score_ledger BIGINT;
 inserted_runs BIGINT:=0;
 position INTEGER;
 stage SMALLINT;
 assignment TEXT;
BEGIN
 PERFORM pg_catalog.pg_advisory_xact_lock(
  pg_catalog.hashtextextended('arena-2026-09-20-rerun328',0));
 LOCK TABLE public.lab_arena_rounds IN ACCESS EXCLUSIVE MODE;
 LOCK TABLE public.lab_arena_submissions IN ACCESS EXCLUSIVE MODE;
 LOCK TABLE public.lab_arena_runs IN ACCESS EXCLUSIVE MODE;
 LOCK TABLE public.lab_arena_ledger IN ACCESS EXCLUSIVE MODE;
 LOCK TABLE public.lab_arena_company_judgments IN SHARE MODE;
 LOCK TABLE public.lab_arena_judgment_cache IN SHARE MODE;
 LOCK TABLE public.lab_arena_company_judgment_reservations IN SHARE MODE;

 SELECT * INTO STRICT active FROM public.lab_arena_rounds
  WHERE round_id='arena-2026-09-20' FOR UPDATE;
 SELECT * INTO STRICT baseline FROM public.lab_arena_submissions
  WHERE round_id='arena-2026-09-20'
   AND submission_id='baseline-2026-09-20' FOR UPDATE;

 IF EXISTS(SELECT 1 FROM public.lab_arena_rounds
   WHERE round_id='arena-2026-09-20-r328archive') THEN
  IF NOT EXISTS(SELECT 1 FROM public.lab_arena_rounds archived
      WHERE archived.round_id='arena-2026-09-20-r328archive'
       AND archived.status='cancelled' AND archived.rewards_enabled IS FALSE
       AND archived.cancel_reason=
           'authorized_sep20_90m_verifier_baseline_archive'
       AND archived.configuration_doc->>'archived_terminal_round_sha256'=
           'f331321b87552c25a282d02218c986d8fb11185582b8df7e60761675bb19c394'
       AND archived.configuration_doc->>'archived_terminal_submissions_sha256'=
           'e19d327d61490a3082bd45e24a109b7f881040ce4f1b0890db3f111ebe5b023a'
       AND archived.configuration_doc->>'archived_terminal_runs_sha256'=
           '58fd363dd809da7f76e3a66947205ad7cd3db63e4770e47918075e61e4a4e831'
       AND archived.configuration_doc->>'archived_terminal_ledger_sha256'=
           '2c37013a3dbcaee2cb37faa1e01c6972fff3c9e572c70aca17efadf34a3dc0d0'
       AND archived.configuration_doc->>'archived_terminal_company_judgments_sha256'=
           '0ef317eb17e2318bae2ef9d4bc8de770cc8cbbae941ab9456941b2c44b9c5c41'
       AND archived.configuration_doc->>'archived_terminal_judgment_cache_sha256'=
           'd3e38c13968388ebf31f062d1ae47c27d58970348513f6a69bf0f63d34f0c2d9'
       AND archived.configuration_doc->>'archived_bank_object_sha256'=
           '7aaa7fbb4b4ce078dd62217c4d594dafdc82161d30c7718bd4500fa659fe5ad9')
    OR baseline.source_ref IS DISTINCT FROM
       'arena/arena-2026-09-20/sources/baseline-2026-09-20-rerun328-5d492f17.tar.gz'
    OR baseline.source_size_bytes IS DISTINCT FROM 854708
    OR baseline.submission_doc->>'source_sha256' IS DISTINCT FROM
       'f1f24e24e0cd736d8c9695640093f4d5f28c360e2773b9824dd5010fe5100f3e'
    OR baseline.submission_doc->>'source_commit' IS DISTINCT FROM
       '5d492f1715e27c8d9b919bcc3ad69d1be5160524'
    OR active.configuration_doc->>'checkpoint_deadline_policy' IS DISTINCT FROM
       'atomic_checkpoint_90m_v1'
    OR (active.configuration_doc->>'icp_wall_clock_seconds')::INTEGER<>5400
    OR (active.configuration_doc->>'lease_ttl_seconds')::INTEGER<>6300
    OR active.configuration_doc->>'scorer_image_digest' IS DISTINCT FROM
       'sha256:333ae499ede5eb51d60385bc2d11c80fed2c2a9ce6922111adde5fa52b40236f'
    OR active.configuration_doc->>'scorer_image_reference' IS DISTINCT FROM
       '493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@sha256:333ae499ede5eb51d60385bc2d11c80fed2c2a9ce6922111adde5fa52b40236f'
    OR active.configuration_doc->'schedule' IS DISTINCT FROM
       '{"benchmark_deadline":"2026-09-20T05:08:23Z","final_scoring_close":"2026-09-20T19:08:23Z","publication_deadline":"2026-09-20T19:08:24Z","stage_1_close":"2026-09-20T12:08:23Z","stage_1_scoring_close":"2026-09-20T15:08:23Z","stage_1_start":"2026-09-20T05:08:24Z","stage_2_close":"2026-09-20T16:08:23Z","stage_2_start":"2026-09-20T15:08:24Z","submission_cutoff":"2026-09-20T00:00:00Z","submission_open":"2026-09-19T00:00:00Z"}'::JSONB
    OR (SELECT pg_catalog.count(*) FROM public.lab_arena_submissions
        WHERE round_id='arena-2026-09-20-r328archive')<>
       8
    OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
        WHERE round_id='arena-2026-09-20-r328archive' AND kind='execute'
         AND submission_id='baseline-2026-09-20:r328archive')<>20
    OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
        WHERE round_id='arena-2026-09-20-r328archive' AND kind='score')<>100
    OR (SELECT pg_catalog.count(DISTINCT assignment_id)
        FROM public.lab_arena_runs WHERE round_id='arena-2026-09-20'
         AND kind='execute' AND submission_id='baseline-2026-09-20'
         AND assignment_id LIKE '%:rerun328')<>20
    OR EXISTS(SELECT 1 FROM public.lab_arena_runs
        WHERE round_id='arena-2026-09-20' AND kind='execute'
         AND submission_id='baseline-2026-09-20'
         AND assignment_id NOT LIKE '%:rerun328')
    OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
        WHERE round_id='arena-2026-09-20' AND kind='execute'
         AND submission_id<>'baseline-2026-09-20' AND status='accepted'
         AND terminal_cause='accepted' AND output_ref IS NOT NULL)<>80 THEN
   RAISE EXCEPTION 'existing Sep20 rerun328 differs' USING ERRCODE='55000';
  END IF;
  RETURN;
 END IF;

 IF 'sha256:333ae499ede5eb51d60385bc2d11c80fed2c2a9ce6922111adde5fa52b40236f'!~'^sha256:[0-9a-f]{64}$'
  OR '493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@sha256:333ae499ede5eb51d60385bc2d11c80fed2c2a9ce6922111adde5fa52b40236f' NOT LIKE '%@sha256:333ae499ede5eb51d60385bc2d11c80fed2c2a9ce6922111adde5fa52b40236f'
  OR active.configuration_doc->>'scorer_image_digest'='sha256:333ae499ede5eb51d60385bc2d11c80fed2c2a9ce6922111adde5fa52b40236f' THEN
  RAISE EXCEPTION 'Sep20 rerun328 scorer identity differs' USING ERRCODE='22023';
 END IF;
 IF ('{"benchmark_deadline":"2026-09-20T05:08:23Z","final_scoring_close":"2026-09-20T19:08:23Z","publication_deadline":"2026-09-20T19:08:24Z","stage_1_close":"2026-09-20T12:08:23Z","stage_1_scoring_close":"2026-09-20T15:08:23Z","stage_1_start":"2026-09-20T05:08:24Z","stage_2_close":"2026-09-20T16:08:23Z","stage_2_start":"2026-09-20T15:08:24Z","submission_cutoff":"2026-09-20T00:00:00Z","submission_open":"2026-09-19T00:00:00Z"}'::JSONB->>'submission_open') IS DISTINCT FROM
      '2026-09-19T00:00:00Z'
 OR ('{"benchmark_deadline":"2026-09-20T05:08:23Z","final_scoring_close":"2026-09-20T19:08:23Z","publication_deadline":"2026-09-20T19:08:24Z","stage_1_close":"2026-09-20T12:08:23Z","stage_1_scoring_close":"2026-09-20T15:08:23Z","stage_1_start":"2026-09-20T05:08:24Z","stage_2_close":"2026-09-20T16:08:23Z","stage_2_start":"2026-09-20T15:08:24Z","submission_cutoff":"2026-09-20T00:00:00Z","submission_open":"2026-09-19T00:00:00Z"}'::JSONB->>'submission_cutoff') IS DISTINCT FROM
      '2026-09-20T00:00:00Z'
  OR ('{"benchmark_deadline":"2026-09-20T05:08:23Z","final_scoring_close":"2026-09-20T19:08:23Z","publication_deadline":"2026-09-20T19:08:24Z","stage_1_close":"2026-09-20T12:08:23Z","stage_1_scoring_close":"2026-09-20T15:08:23Z","stage_1_start":"2026-09-20T05:08:24Z","stage_2_close":"2026-09-20T16:08:23Z","stage_2_start":"2026-09-20T15:08:24Z","submission_cutoff":"2026-09-20T00:00:00Z","submission_open":"2026-09-19T00:00:00Z"}'::JSONB->>'benchmark_deadline')::TIMESTAMPTZ
      <=pg_catalog.clock_timestamp()
  OR ('{"benchmark_deadline":"2026-09-20T05:08:23Z","final_scoring_close":"2026-09-20T19:08:23Z","publication_deadline":"2026-09-20T19:08:24Z","stage_1_close":"2026-09-20T12:08:23Z","stage_1_scoring_close":"2026-09-20T15:08:23Z","stage_1_start":"2026-09-20T05:08:24Z","stage_2_close":"2026-09-20T16:08:23Z","stage_2_start":"2026-09-20T15:08:24Z","submission_cutoff":"2026-09-20T00:00:00Z","submission_open":"2026-09-19T00:00:00Z"}'::JSONB->>'stage_1_start')::TIMESTAMPTZ
      IS DISTINCT FROM
      ('{"benchmark_deadline":"2026-09-20T05:08:23Z","final_scoring_close":"2026-09-20T19:08:23Z","publication_deadline":"2026-09-20T19:08:24Z","stage_1_close":"2026-09-20T12:08:23Z","stage_1_scoring_close":"2026-09-20T15:08:23Z","stage_1_start":"2026-09-20T05:08:24Z","stage_2_close":"2026-09-20T16:08:23Z","stage_2_start":"2026-09-20T15:08:24Z","submission_cutoff":"2026-09-20T00:00:00Z","submission_open":"2026-09-19T00:00:00Z"}'::JSONB->>'benchmark_deadline')::TIMESTAMPTZ+
       INTERVAL '1 second'
  OR ('{"benchmark_deadline":"2026-09-20T05:08:23Z","final_scoring_close":"2026-09-20T19:08:23Z","publication_deadline":"2026-09-20T19:08:24Z","stage_1_close":"2026-09-20T12:08:23Z","stage_1_scoring_close":"2026-09-20T15:08:23Z","stage_1_start":"2026-09-20T05:08:24Z","stage_2_close":"2026-09-20T16:08:23Z","stage_2_start":"2026-09-20T15:08:24Z","submission_cutoff":"2026-09-20T00:00:00Z","submission_open":"2026-09-19T00:00:00Z"}'::JSONB->>'stage_1_close')::TIMESTAMPTZ
      IS DISTINCT FROM
      ('{"benchmark_deadline":"2026-09-20T05:08:23Z","final_scoring_close":"2026-09-20T19:08:23Z","publication_deadline":"2026-09-20T19:08:24Z","stage_1_close":"2026-09-20T12:08:23Z","stage_1_scoring_close":"2026-09-20T15:08:23Z","stage_1_start":"2026-09-20T05:08:24Z","stage_2_close":"2026-09-20T16:08:23Z","stage_2_start":"2026-09-20T15:08:24Z","submission_cutoff":"2026-09-20T00:00:00Z","submission_open":"2026-09-19T00:00:00Z"}'::JSONB->>'benchmark_deadline')::TIMESTAMPTZ+
       INTERVAL '7 hours'
  OR ('{"benchmark_deadline":"2026-09-20T05:08:23Z","final_scoring_close":"2026-09-20T19:08:23Z","publication_deadline":"2026-09-20T19:08:24Z","stage_1_close":"2026-09-20T12:08:23Z","stage_1_scoring_close":"2026-09-20T15:08:23Z","stage_1_start":"2026-09-20T05:08:24Z","stage_2_close":"2026-09-20T16:08:23Z","stage_2_start":"2026-09-20T15:08:24Z","submission_cutoff":"2026-09-20T00:00:00Z","submission_open":"2026-09-19T00:00:00Z"}'::JSONB->>'stage_1_scoring_close')::TIMESTAMPTZ
      IS DISTINCT FROM
      ('{"benchmark_deadline":"2026-09-20T05:08:23Z","final_scoring_close":"2026-09-20T19:08:23Z","publication_deadline":"2026-09-20T19:08:24Z","stage_1_close":"2026-09-20T12:08:23Z","stage_1_scoring_close":"2026-09-20T15:08:23Z","stage_1_start":"2026-09-20T05:08:24Z","stage_2_close":"2026-09-20T16:08:23Z","stage_2_start":"2026-09-20T15:08:24Z","submission_cutoff":"2026-09-20T00:00:00Z","submission_open":"2026-09-19T00:00:00Z"}'::JSONB->>'benchmark_deadline')::TIMESTAMPTZ+
       INTERVAL '10 hours'
  OR ('{"benchmark_deadline":"2026-09-20T05:08:23Z","final_scoring_close":"2026-09-20T19:08:23Z","publication_deadline":"2026-09-20T19:08:24Z","stage_1_close":"2026-09-20T12:08:23Z","stage_1_scoring_close":"2026-09-20T15:08:23Z","stage_1_start":"2026-09-20T05:08:24Z","stage_2_close":"2026-09-20T16:08:23Z","stage_2_start":"2026-09-20T15:08:24Z","submission_cutoff":"2026-09-20T00:00:00Z","submission_open":"2026-09-19T00:00:00Z"}'::JSONB->>'stage_2_start')::TIMESTAMPTZ
      IS DISTINCT FROM
      ('{"benchmark_deadline":"2026-09-20T05:08:23Z","final_scoring_close":"2026-09-20T19:08:23Z","publication_deadline":"2026-09-20T19:08:24Z","stage_1_close":"2026-09-20T12:08:23Z","stage_1_scoring_close":"2026-09-20T15:08:23Z","stage_1_start":"2026-09-20T05:08:24Z","stage_2_close":"2026-09-20T16:08:23Z","stage_2_start":"2026-09-20T15:08:24Z","submission_cutoff":"2026-09-20T00:00:00Z","submission_open":"2026-09-19T00:00:00Z"}'::JSONB->>'benchmark_deadline')::TIMESTAMPTZ+
       INTERVAL '10 hours 1 second'
  OR ('{"benchmark_deadline":"2026-09-20T05:08:23Z","final_scoring_close":"2026-09-20T19:08:23Z","publication_deadline":"2026-09-20T19:08:24Z","stage_1_close":"2026-09-20T12:08:23Z","stage_1_scoring_close":"2026-09-20T15:08:23Z","stage_1_start":"2026-09-20T05:08:24Z","stage_2_close":"2026-09-20T16:08:23Z","stage_2_start":"2026-09-20T15:08:24Z","submission_cutoff":"2026-09-20T00:00:00Z","submission_open":"2026-09-19T00:00:00Z"}'::JSONB->>'stage_2_close')::TIMESTAMPTZ
      IS DISTINCT FROM
      ('{"benchmark_deadline":"2026-09-20T05:08:23Z","final_scoring_close":"2026-09-20T19:08:23Z","publication_deadline":"2026-09-20T19:08:24Z","stage_1_close":"2026-09-20T12:08:23Z","stage_1_scoring_close":"2026-09-20T15:08:23Z","stage_1_start":"2026-09-20T05:08:24Z","stage_2_close":"2026-09-20T16:08:23Z","stage_2_start":"2026-09-20T15:08:24Z","submission_cutoff":"2026-09-20T00:00:00Z","submission_open":"2026-09-19T00:00:00Z"}'::JSONB->>'benchmark_deadline')::TIMESTAMPTZ+
       INTERVAL '11 hours'
  OR ('{"benchmark_deadline":"2026-09-20T05:08:23Z","final_scoring_close":"2026-09-20T19:08:23Z","publication_deadline":"2026-09-20T19:08:24Z","stage_1_close":"2026-09-20T12:08:23Z","stage_1_scoring_close":"2026-09-20T15:08:23Z","stage_1_start":"2026-09-20T05:08:24Z","stage_2_close":"2026-09-20T16:08:23Z","stage_2_start":"2026-09-20T15:08:24Z","submission_cutoff":"2026-09-20T00:00:00Z","submission_open":"2026-09-19T00:00:00Z"}'::JSONB->>'final_scoring_close')::TIMESTAMPTZ
      IS DISTINCT FROM
      ('{"benchmark_deadline":"2026-09-20T05:08:23Z","final_scoring_close":"2026-09-20T19:08:23Z","publication_deadline":"2026-09-20T19:08:24Z","stage_1_close":"2026-09-20T12:08:23Z","stage_1_scoring_close":"2026-09-20T15:08:23Z","stage_1_start":"2026-09-20T05:08:24Z","stage_2_close":"2026-09-20T16:08:23Z","stage_2_start":"2026-09-20T15:08:24Z","submission_cutoff":"2026-09-20T00:00:00Z","submission_open":"2026-09-19T00:00:00Z"}'::JSONB->>'benchmark_deadline')::TIMESTAMPTZ+
       INTERVAL '14 hours'
  OR ('{"benchmark_deadline":"2026-09-20T05:08:23Z","final_scoring_close":"2026-09-20T19:08:23Z","publication_deadline":"2026-09-20T19:08:24Z","stage_1_close":"2026-09-20T12:08:23Z","stage_1_scoring_close":"2026-09-20T15:08:23Z","stage_1_start":"2026-09-20T05:08:24Z","stage_2_close":"2026-09-20T16:08:23Z","stage_2_start":"2026-09-20T15:08:24Z","submission_cutoff":"2026-09-20T00:00:00Z","submission_open":"2026-09-19T00:00:00Z"}'::JSONB->>'publication_deadline')::TIMESTAMPTZ
      IS DISTINCT FROM
      ('{"benchmark_deadline":"2026-09-20T05:08:23Z","final_scoring_close":"2026-09-20T19:08:23Z","publication_deadline":"2026-09-20T19:08:24Z","stage_1_close":"2026-09-20T12:08:23Z","stage_1_scoring_close":"2026-09-20T15:08:23Z","stage_1_start":"2026-09-20T05:08:24Z","stage_2_close":"2026-09-20T16:08:23Z","stage_2_start":"2026-09-20T15:08:24Z","submission_cutoff":"2026-09-20T00:00:00Z","submission_open":"2026-09-19T00:00:00Z"}'::JSONB->>'benchmark_deadline')::TIMESTAMPTZ+
       INTERVAL '14 hours 1 second' THEN
  RAISE EXCEPTION 'Sep20 rerun328 schedule differs' USING ERRCODE='22023';
 END IF;

 SELECT pg_catalog.count(*) INTO active_count FROM public.lab_arena_runs
  WHERE round_id='arena-2026-09-20'
   AND status IN('pending','leased','submitted');
 SELECT pg_catalog.count(*) INTO unsettled_count
 FROM public.lab_arena_submissions s
 CROSS JOIN(VALUES('execute'::TEXT),('score'::TEXT)) k(kind)
 CROSS JOIN LATERAL(SELECT public.lab_arena__successful_call_cost_state(
  s.submission_id,k.kind,NULL) state)c
 WHERE s.round_id='arena-2026-09-20'
  AND (COALESCE((c.state->>'inflight_calls')::BIGINT,-1)<>0
   OR COALESCE((c.state->>'success_unresolved_calls')::BIGINT,-1)<>0);
 SELECT pg_catalog.count(*) INTO reservation_count
 FROM public.lab_arena_company_judgment_reservations reservation
 JOIN public.lab_arena_runs run ON run.run_id=reservation.run_id
 WHERE run.round_id='arena-2026-09-20';

 IF active.status IS DISTINCT FROM 'published'
  OR active.status IS DISTINCT FROM 'published'
  OR active.cancel_reason IS DISTINCT FROM NULL
  OR active_count<>0 OR unsettled_count<>0 OR reservation_count<>0
  OR active.benchmark_ref IS DISTINCT FROM
     'arena/arena-2026-09-20/benchmark.json'
  OR active.evaluation_date IS DISTINCT FROM '2026-09-20'
  OR active.icp_set_date IS DISTINCT FROM DATE '2026-09-19'
  OR pg_catalog.jsonb_array_length(active.participants)<>5
  OR (SELECT pg_catalog.count(*) FROM public.lab_arena_submissions
      WHERE round_id='arena-2026-09-20')<>8
  OR (SELECT pg_catalog.count(*) FROM public.lab_arena_submissions
      WHERE round_id='arena-2026-09-20' AND status='frozen')<>5
  OR active.configuration_doc->'call_quotas' IS DISTINCT FROM
     '{"deepline":200,"openrouter":500,"scrapingdog":200}'::JSONB
  OR active.configuration_doc->'scoring_call_quotas' IS DISTINCT FROM
     '{"deepline":40,"openrouter":120,"scrapingdog":150}'::JSONB
  OR active.configuration_doc->>'sourcing_cost_eligibility_policy'
     IS DISTINCT FROM 'successful_calls_per_icp_v1'
  OR (active.configuration_doc->>'execution_icp_cap_microusd')::BIGINT<>4000000
  OR (active.configuration_doc->>'cost_per_company_microusd')::BIGINT<>800000
  OR active.configuration_doc->>'checkpoint_deadline_policy' IS DISTINCT FROM
     'atomic_checkpoint_45m_v1'
  OR (active.configuration_doc->>'icp_wall_clock_seconds')::INTEGER<>2700
  OR (active.configuration_doc->>'lease_ttl_seconds')::INTEGER<>3600
  OR pg_catalog.encode(extensions.digest(pg_catalog.to_jsonb(active)::TEXT,
       'sha256'),'hex')<>'f331321b87552c25a282d02218c986d8fb11185582b8df7e60761675bb19c394'
  OR pg_catalog.encode(extensions.digest(pg_catalog.to_jsonb(baseline)::TEXT,
       'sha256'),'hex')<>'0b47d8e1b89978be5846105599e337c1721c48ca2e5459d45b5c62b7cb98f558'
  OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
      WHERE round_id='arena-2026-09-20')<>200
  OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
      WHERE round_id='arena-2026-09-20' AND kind='execute')<>100
  OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
      WHERE round_id='arena-2026-09-20' AND kind='execute'
       AND status='accepted' AND terminal_cause='accepted'
       AND output_ref IS NOT NULL)<>100
  OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
      WHERE round_id='arena-2026-09-20' AND kind='score')<>100
  OR EXISTS(SELECT 1 FROM public.lab_arena_runs
      WHERE round_id='arena-2026-09-20' AND kind='score'
       AND (status<>'accepted' OR terminal_cause<>'accepted'
        OR assignment_id NOT LIKE '%:score:rerun326'))
  OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
      WHERE round_id='arena-2026-09-20' AND kind='execute'
       AND submission_id='baseline-2026-09-20')<>20
  OR EXISTS(SELECT 1 FROM public.lab_arena_runs
      WHERE round_id='arena-2026-09-20' AND kind='execute'
       AND submission_id='baseline-2026-09-20'
       AND assignment_id NOT LIKE '%:rerun326')
  OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
      WHERE round_id='arena-2026-09-20' AND kind='execute'
       AND submission_id<>'baseline-2026-09-20')<>80
  OR NOT EXISTS(SELECT 1 FROM public.lab_arena_rounds archived
      WHERE archived.round_id='arena-2026-09-20-r326archive'
       AND archived.configuration_doc->>'archived_bank_object_sha256'=
       '7aaa7fbb4b4ce078dd62217c4d594dafdc82161d30c7718bd4500fa659fe5ad9') THEN
  RAISE EXCEPTION 'Sep20 rerun328 terminal preimage differs' USING ERRCODE='55000';
 END IF;

 IF pg_catalog.encode(extensions.digest(COALESCE((SELECT pg_catalog.string_agg(
    pg_catalog.encode(extensions.digest(pg_catalog.to_jsonb(s)::TEXT,'sha256'),'hex'),''
    ORDER BY submission_id) FROM public.lab_arena_submissions s
    WHERE round_id='arena-2026-09-20'),''),'sha256'),'hex')<>
       'e19d327d61490a3082bd45e24a109b7f881040ce4f1b0890db3f111ebe5b023a'
  OR pg_catalog.encode(extensions.digest(COALESCE((SELECT pg_catalog.string_agg(
    pg_catalog.encode(extensions.digest(pg_catalog.to_jsonb(r)::TEXT,'sha256'),'hex'),''
    ORDER BY run_id) FROM public.lab_arena_runs r
    WHERE round_id='arena-2026-09-20'),''),'sha256'),'hex')<>
       '58fd363dd809da7f76e3a66947205ad7cd3db63e4770e47918075e61e4a4e831'
  OR pg_catalog.encode(extensions.digest(COALESCE((SELECT pg_catalog.string_agg(
    pg_catalog.encode(extensions.digest(pg_catalog.to_jsonb(l)::TEXT,'sha256'),'hex'),''
    ORDER BY entry_id) FROM public.lab_arena_ledger l
    WHERE round_id='arena-2026-09-20'),''),'sha256'),'hex')<>
       '2c37013a3dbcaee2cb37faa1e01c6972fff3c9e572c70aca17efadf34a3dc0d0'
  OR (SELECT pg_catalog.count(*) FROM public.lab_arena_ledger
      WHERE round_id='arena-2026-09-20')<>19922
  OR pg_catalog.encode(extensions.digest(COALESCE((SELECT pg_catalog.string_agg(
    pg_catalog.encode(extensions.digest(pg_catalog.to_jsonb(j)::TEXT,'sha256'),'hex'),''
    ORDER BY cache_key,authority_slot) FROM public.lab_arena_company_judgments j),''),
    'sha256'),'hex')<>'0ef317eb17e2318bae2ef9d4bc8de770cc8cbbae941ab9456941b2c44b9c5c41'
  OR (SELECT pg_catalog.count(*) FROM public.lab_arena_company_judgments)<>
       67
  OR pg_catalog.encode(extensions.digest(COALESCE((SELECT pg_catalog.string_agg(
    pg_catalog.encode(extensions.digest(pg_catalog.to_jsonb(c)::TEXT,'sha256'),'hex'),''
    ORDER BY cache_key) FROM public.lab_arena_judgment_cache c),''),
    'sha256'),'hex')<>'d3e38c13968388ebf31f062d1ae47c27d58970348513f6a69bf0f63d34f0c2d9'
  OR (SELECT pg_catalog.count(*) FROM public.lab_arena_judgment_cache)<>
       811 THEN
  RAISE EXCEPTION 'Sep20 rerun328 terminal history differs' USING ERRCODE='55000';
 END IF;

 SELECT pg_catalog.jsonb_build_object(
  'round',(SELECT pg_catalog.encode(extensions.digest(pg_catalog.to_jsonb(r)::TEXT,
    'sha256'),'hex') FROM public.lab_arena_rounds r
    WHERE round_id='arena-2026-09-20-r326archive'),
  'submissions',(SELECT pg_catalog.jsonb_build_array(pg_catalog.count(*),
    pg_catalog.encode(extensions.digest(COALESCE(pg_catalog.string_agg(
     pg_catalog.encode(extensions.digest(pg_catalog.to_jsonb(s)::TEXT,'sha256'),'hex'),''
     ORDER BY submission_id),''),'sha256'),'hex'))
    FROM public.lab_arena_submissions s
    WHERE round_id='arena-2026-09-20-r326archive'),
  'runs',(SELECT pg_catalog.jsonb_build_array(pg_catalog.count(*),
    pg_catalog.encode(extensions.digest(COALESCE(pg_catalog.string_agg(
     pg_catalog.encode(extensions.digest(pg_catalog.to_jsonb(r)::TEXT,'sha256'),'hex'),''
     ORDER BY run_id),''),'sha256'),'hex')) FROM public.lab_arena_runs r
    WHERE round_id='arena-2026-09-20-r326archive'),
  'ledger',(SELECT pg_catalog.jsonb_build_array(pg_catalog.count(*),
    pg_catalog.encode(extensions.digest(COALESCE(pg_catalog.string_agg(
     pg_catalog.encode(extensions.digest(pg_catalog.to_jsonb(l)::TEXT,'sha256'),'hex'),''
     ORDER BY entry_id),''),'sha256'),'hex')) FROM public.lab_arena_ledger l
    WHERE round_id='arena-2026-09-20-r326archive')) INTO prior_archive_before;
 SELECT pg_catalog.jsonb_build_object('count',pg_catalog.count(*),'sha256',
   pg_catalog.encode(extensions.digest(COALESCE(pg_catalog.string_agg(
    pg_catalog.encode(extensions.digest(pg_catalog.to_jsonb(j)::TEXT,'sha256'),'hex'),''
    ORDER BY cache_key,authority_slot),''),'sha256'),'hex'))
  INTO judgments_before FROM public.lab_arena_company_judgments j;
 SELECT pg_catalog.jsonb_build_object('count',pg_catalog.count(*),'sha256',
   pg_catalog.encode(extensions.digest(COALESCE(pg_catalog.string_agg(
    pg_catalog.encode(extensions.digest(pg_catalog.to_jsonb(c)::TEXT,'sha256'),'hex'),''
    ORDER BY cache_key),''),'sha256'),'hex'))
  INTO cache_before FROM public.lab_arena_judgment_cache c;

 expected_submission:=baseline.submission_doc||pg_catalog.jsonb_build_object(
  'source_ref','arena/arena-2026-09-20/sources/baseline-2026-09-20-rerun328-5d492f17.tar.gz',
  'source_size_bytes',854708,
  'source_sha256','f1f24e24e0cd736d8c9695640093f4d5f28c360e2773b9824dd5010fe5100f3e',
  'source_commit','5d492f1715e27c8d9b919bcc3ad69d1be5160524');
 SELECT pg_catalog.jsonb_agg(CASE
   WHEN item->>'submission_id'='baseline-2026-09-20' THEN
    item||pg_catalog.jsonb_build_object(
     'source_ref','arena/arena-2026-09-20/sources/baseline-2026-09-20-rerun328-5d492f17.tar.gz',
     'source_size_bytes',854708)
   ELSE item END ORDER BY ordinal)
  INTO new_participants FROM pg_catalog.jsonb_array_elements(active.participants)
   WITH ORDINALITY entries(item,ordinal);
 SELECT pg_catalog.jsonb_agg(item||pg_catalog.jsonb_build_object(
   'submission_id',(item->>'submission_id')||':r328archive') ORDER BY ordinal)
  INTO archive_participants FROM pg_catalog.jsonb_array_elements(active.participants)
   WITH ORDINALITY entries(item,ordinal);
 SELECT pg_catalog.jsonb_agg(pg_catalog.jsonb_build_object(
   'run_id',run_id,'per_icp_score',per_icp_score,
   'qualification_doc',qualification_doc,'updated_at',updated_at) ORDER BY run_id)
  INTO archived_execution_judgments FROM public.lab_arena_runs
  WHERE round_id='arena-2026-09-20' AND kind='execute';
 archive_config:=active.configuration_doc||pg_catalog.jsonb_build_object(
  'round_id','arena-2026-09-20-r328archive','mode','shadow','rewards_enabled',FALSE,
  'archived_terminal_status',active.status,
  'archived_terminal_cancel_reason',active.cancel_reason,
  'archived_terminal_round_sha256','f331321b87552c25a282d02218c986d8fb11185582b8df7e60761675bb19c394',
  'archived_terminal_baseline_sha256','0b47d8e1b89978be5846105599e337c1721c48ca2e5459d45b5c62b7cb98f558',
  'archived_terminal_submissions_sha256','e19d327d61490a3082bd45e24a109b7f881040ce4f1b0890db3f111ebe5b023a',
  'archived_terminal_runs_sha256','58fd363dd809da7f76e3a66947205ad7cd3db63e4770e47918075e61e4a4e831',
  'archived_terminal_ledger_sha256','2c37013a3dbcaee2cb37faa1e01c6972fff3c9e572c70aca17efadf34a3dc0d0',
  'archived_terminal_company_judgments_sha256',
   '0ef317eb17e2318bae2ef9d4bc8de770cc8cbbae941ab9456941b2c44b9c5c41',
  'archived_terminal_judgment_cache_sha256','d3e38c13968388ebf31f062d1ae47c27d58970348513f6a69bf0f63d34f0c2d9',
  'archived_rewards_enabled',active.rewards_enabled,
  'archived_effective_reward_epoch',active.effective_reward_epoch,
  'archived_reward_basis_hash',active.reward_basis_hash,
  'archived_reward_basis_doc',active.reward_basis_doc,
  'archived_signing_key_doc',active.signing_key_doc,
  'archived_reward_activated_at',active.reward_activated_at,
  'archived_king_outcome',active.king_outcome,
  'archived_king_hotkey',active.king_hotkey,
  'archived_king_start_epoch',active.king_start_epoch,
  'archived_promotion_required',active.promotion_required,
  'archived_promotion_doc',active.promotion_doc,
  'archived_baseline_promoted_at',active.baseline_promoted_at,
  'archived_execution_judgments',archived_execution_judgments,
  'archived_bank_object_sha256',
   '7aaa7fbb4b4ce078dd62217c4d594dafdc82161d30c7718bd4500fa659fe5ad9');
 SELECT pg_catalog.jsonb_agg(pg_catalog.to_jsonb(s) ORDER BY submission_id)
  INTO miner_submissions_before FROM public.lab_arena_submissions s
  WHERE round_id='arena-2026-09-20' AND submission_id<>'baseline-2026-09-20';
 SELECT pg_catalog.jsonb_agg(pg_catalog.to_jsonb(r)-'per_icp_score'-
   'qualification_doc'-'updated_at' ORDER BY run_id) INTO miner_runs_before
  FROM public.lab_arena_runs r WHERE round_id='arena-2026-09-20'
   AND kind='execute' AND submission_id<>'baseline-2026-09-20';
 SELECT pg_catalog.jsonb_agg(pg_catalog.to_jsonb(l) ORDER BY entry_id)
  INTO miner_ledger_before FROM public.lab_arena_ledger l
  WHERE round_id='arena-2026-09-20'
   AND submission_id<>'baseline-2026-09-20'
   AND NOT EXISTS(SELECT 1 FROM public.lab_arena_runs s
    WHERE s.round_id='arena-2026-09-20' AND s.kind='score' AND s.run_id=l.run_id);

 ALTER TABLE public.lab_arena_rounds DISABLE TRIGGER USER;
 ALTER TABLE public.lab_arena_submissions DISABLE TRIGGER USER;
 ALTER TABLE public.lab_arena_runs DISABLE TRIGGER USER;
 ALTER TABLE public.lab_arena_ledger DISABLE TRIGGER USER;
 INSERT INTO public.lab_arena_rounds(
  round_id,status,status_generation,stage_generation,configuration_doc,
  rewards_enabled,participants,benchmark_ref,evaluation_date,
  stage1_scoring_plan_doc,stage2_scoring_plan_doc,finalists,publication_doc,
  king_outcome,king_hotkey,king_start_epoch,effective_reward_epoch,
  reward_basis_hash,reward_basis_doc,signing_key_doc,reward_activated_at,
  cancel_reason,published_at,created_at,updated_at,promotion_required,
  promotion_doc,baseline_promoted_at,icp_set_date,confirmation_bank_ref,
  confirmation_bank_hash,confirmation_cohort,stage3_scoring_plan_doc,
  champion_funding_frozen,champion_submission_id,champion_hotkey,
  champion_fallback_providers)
 SELECT round_id,status,status_generation,stage_generation,configuration_doc,
  rewards_enabled,participants,benchmark_ref,evaluation_date,
  stage1_scoring_plan_doc,stage2_scoring_plan_doc,finalists,publication_doc,
  king_outcome,king_hotkey,king_start_epoch,effective_reward_epoch,
  reward_basis_hash,reward_basis_doc,signing_key_doc,reward_activated_at,
  cancel_reason,published_at,created_at,updated_at,promotion_required,
  promotion_doc,baseline_promoted_at,icp_set_date,confirmation_bank_ref,
  confirmation_bank_hash,confirmation_cohort,stage3_scoring_plan_doc,
  champion_funding_frozen,champion_submission_id,champion_hotkey,
  champion_fallback_providers FROM pg_catalog.jsonb_populate_record(
   NULL::public.lab_arena_rounds,pg_catalog.to_jsonb(active)||
   pg_catalog.jsonb_build_object(
    'round_id','arena-2026-09-20-r328archive','status','cancelled',
    'configuration_doc',archive_config,'participants',archive_participants,
    'rewards_enabled',FALSE,'reward_basis_hash',NULL,'reward_basis_doc',NULL,
    'signing_key_doc',NULL,'effective_reward_epoch',NULL,'reward_activated_at',NULL,
    'cancel_reason','authorized_sep20_90m_verifier_baseline_archive'));
 INSERT INTO public.lab_arena_submissions SELECT (pg_catalog.jsonb_populate_record(
  NULL::public.lab_arena_submissions,pg_catalog.to_jsonb(s)||
  pg_catalog.jsonb_build_object('round_id','arena-2026-09-20-r328archive',
   'submission_id',s.submission_id||':r328archive'))).*
 FROM public.lab_arena_submissions s WHERE s.round_id='arena-2026-09-20';

 UPDATE public.lab_arena_runs SET round_id='arena-2026-09-20-r328archive',
  submission_id=submission_id||':r328archive'
 WHERE round_id='arena-2026-09-20' AND kind='score';
 GET DIAGNOSTICS moved_score_runs=ROW_COUNT;
 UPDATE public.lab_arena_ledger l SET round_id='arena-2026-09-20-r328archive',
  submission_id=submission_id||':r328archive'
 WHERE round_id='arena-2026-09-20' AND EXISTS(SELECT 1
  FROM public.lab_arena_runs s WHERE s.round_id='arena-2026-09-20-r328archive'
   AND s.kind='score' AND s.run_id=l.run_id);
 GET DIAGNOSTICS moved_score_ledger=ROW_COUNT;
 UPDATE public.lab_arena_runs SET round_id='arena-2026-09-20-r328archive',
  submission_id='baseline-2026-09-20:r328archive'
 WHERE round_id='arena-2026-09-20' AND kind='execute'
  AND submission_id='baseline-2026-09-20';
 GET DIAGNOSTICS moved_baseline_runs=ROW_COUNT;
 UPDATE public.lab_arena_ledger SET round_id='arena-2026-09-20-r328archive',
  submission_id='baseline-2026-09-20:r328archive'
 WHERE round_id='arena-2026-09-20'
  AND submission_id='baseline-2026-09-20';
 GET DIAGNOSTICS moved_baseline_ledger=ROW_COUNT;
 UPDATE public.lab_arena_runs SET per_icp_score=NULL,qualification_doc=NULL
 WHERE round_id='arena-2026-09-20' AND kind='execute'
  AND submission_id<>'baseline-2026-09-20';
 UPDATE public.lab_arena_submissions SET
  source_ref='arena/arena-2026-09-20/sources/baseline-2026-09-20-rerun328-5d492f17.tar.gz',
  source_size_bytes=854708,submission_doc=expected_submission,
  updated_at=pg_catalog.clock_timestamp()
 WHERE round_id='arena-2026-09-20' AND submission_id='baseline-2026-09-20';
 IF NOT FOUND THEN RAISE EXCEPTION 'Sep20 rerun328 baseline update missing'; END IF;
 UPDATE public.lab_arena_rounds SET status='stage1',
  status_generation=status_generation+1,stage_generation=stage_generation+1,
  configuration_doc=configuration_doc||pg_catalog.jsonb_build_object(
   'schedule','{"benchmark_deadline":"2026-09-20T05:08:23Z","final_scoring_close":"2026-09-20T19:08:23Z","publication_deadline":"2026-09-20T19:08:24Z","stage_1_close":"2026-09-20T12:08:23Z","stage_1_scoring_close":"2026-09-20T15:08:23Z","stage_1_start":"2026-09-20T05:08:24Z","stage_2_close":"2026-09-20T16:08:23Z","stage_2_start":"2026-09-20T15:08:24Z","submission_cutoff":"2026-09-20T00:00:00Z","submission_open":"2026-09-19T00:00:00Z"}'::JSONB,
   'scorer_image_digest','sha256:333ae499ede5eb51d60385bc2d11c80fed2c2a9ce6922111adde5fa52b40236f',
   'scorer_image_reference','493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@sha256:333ae499ede5eb51d60385bc2d11c80fed2c2a9ce6922111adde5fa52b40236f',
   'checkpoint_deadline_policy','atomic_checkpoint_90m_v1',
   'icp_wall_clock_seconds',5400,'lease_ttl_seconds',6300),
  participants=new_participants,stage1_scoring_plan_doc=NULL,
  stage2_scoring_plan_doc=NULL,stage3_scoring_plan_doc=NULL,finalists=NULL,
  publication_doc=NULL,published_at=NULL,cancel_reason=NULL,
  king_outcome=NULL,king_hotkey=NULL,king_start_epoch=NULL,
  effective_reward_epoch=NULL,reward_basis_hash=NULL,reward_basis_doc=NULL,
  signing_key_doc=NULL,reward_activated_at=NULL,
  promotion_required=TRUE,promotion_doc=NULL,baseline_promoted_at=NULL,
  updated_at=pg_catalog.clock_timestamp()
 WHERE round_id='arena-2026-09-20';
 IF NOT FOUND THEN RAISE EXCEPTION 'Sep20 rerun328 round update missing'; END IF;
 FOR position IN 0..19 LOOP
  stage:=CASE WHEN position<10 THEN 1 ELSE 2 END;
  assignment:='arena-2026-09-20:baseline-2026-09-20:'||stage::TEXT||':'||
   position::TEXT||':rerun328';
  INSERT INTO public.lab_arena_runs(run_id,assignment_id,round_id,submission_id,
   miner_hotkey,stage,icp_position,attempt,kind,status,stage_generation)
  VALUES(assignment||':1',assignment,'arena-2026-09-20','baseline-2026-09-20',
   baseline.miner_hotkey,stage,position,1,'execute','pending',
   active.stage_generation+1);
  inserted_runs:=inserted_runs+1;
 END LOOP;
 ALTER TABLE public.lab_arena_ledger ENABLE TRIGGER USER;
 ALTER TABLE public.lab_arena_runs ENABLE TRIGGER USER;
 ALTER TABLE public.lab_arena_submissions ENABLE TRIGGER USER;
 ALTER TABLE public.lab_arena_rounds ENABLE TRIGGER USER;

 SELECT pg_catalog.jsonb_agg(pg_catalog.to_jsonb(s) ORDER BY submission_id)
  INTO miner_submissions_after FROM public.lab_arena_submissions s
  WHERE round_id='arena-2026-09-20' AND submission_id<>'baseline-2026-09-20';
 SELECT pg_catalog.jsonb_agg(pg_catalog.to_jsonb(r)-'per_icp_score'-
   'qualification_doc'-'updated_at' ORDER BY run_id) INTO miner_runs_after
  FROM public.lab_arena_runs r WHERE round_id='arena-2026-09-20'
   AND kind='execute' AND submission_id<>'baseline-2026-09-20';
 SELECT pg_catalog.jsonb_agg(pg_catalog.to_jsonb(l) ORDER BY entry_id)
  INTO miner_ledger_after FROM public.lab_arena_ledger l
  WHERE round_id='arena-2026-09-20'
   AND submission_id<>'baseline-2026-09-20';
 SELECT pg_catalog.jsonb_build_object(
  'round',(SELECT pg_catalog.encode(extensions.digest(pg_catalog.to_jsonb(r)::TEXT,
    'sha256'),'hex') FROM public.lab_arena_rounds r
    WHERE round_id='arena-2026-09-20-r326archive'),
  'submissions',(SELECT pg_catalog.jsonb_build_array(pg_catalog.count(*),
    pg_catalog.encode(extensions.digest(COALESCE(pg_catalog.string_agg(
     pg_catalog.encode(extensions.digest(pg_catalog.to_jsonb(s)::TEXT,'sha256'),'hex'),''
     ORDER BY submission_id),''),'sha256'),'hex'))
    FROM public.lab_arena_submissions s
    WHERE round_id='arena-2026-09-20-r326archive'),
  'runs',(SELECT pg_catalog.jsonb_build_array(pg_catalog.count(*),
    pg_catalog.encode(extensions.digest(COALESCE(pg_catalog.string_agg(
     pg_catalog.encode(extensions.digest(pg_catalog.to_jsonb(r)::TEXT,'sha256'),'hex'),''
     ORDER BY run_id),''),'sha256'),'hex')) FROM public.lab_arena_runs r
    WHERE round_id='arena-2026-09-20-r326archive'),
  'ledger',(SELECT pg_catalog.jsonb_build_array(pg_catalog.count(*),
    pg_catalog.encode(extensions.digest(COALESCE(pg_catalog.string_agg(
     pg_catalog.encode(extensions.digest(pg_catalog.to_jsonb(l)::TEXT,'sha256'),'hex'),''
     ORDER BY entry_id),''),'sha256'),'hex')) FROM public.lab_arena_ledger l
    WHERE round_id='arena-2026-09-20-r326archive')) INTO prior_archive_after;
 SELECT pg_catalog.jsonb_build_object('count',pg_catalog.count(*),'sha256',
   pg_catalog.encode(extensions.digest(COALESCE(pg_catalog.string_agg(
    pg_catalog.encode(extensions.digest(pg_catalog.to_jsonb(j)::TEXT,'sha256'),'hex'),''
    ORDER BY cache_key,authority_slot),''),'sha256'),'hex'))
  INTO judgments_after FROM public.lab_arena_company_judgments j;
 SELECT pg_catalog.jsonb_build_object('count',pg_catalog.count(*),'sha256',
   pg_catalog.encode(extensions.digest(COALESCE(pg_catalog.string_agg(
    pg_catalog.encode(extensions.digest(pg_catalog.to_jsonb(c)::TEXT,'sha256'),'hex'),''
    ORDER BY cache_key),''),'sha256'),'hex'))
  INTO cache_after FROM public.lab_arena_judgment_cache c;
 IF moved_score_runs<>100
  OR moved_score_ledger<>147
  OR moved_baseline_runs<>20
  OR moved_baseline_ledger<>15070
  OR inserted_runs<>20
  OR miner_submissions_after IS DISTINCT FROM miner_submissions_before
  OR miner_runs_after IS DISTINCT FROM miner_runs_before
  OR miner_ledger_after IS DISTINCT FROM miner_ledger_before
  OR prior_archive_after IS DISTINCT FROM prior_archive_before
  OR judgments_after IS DISTINCT FROM judgments_before
  OR cache_after IS DISTINCT FROM cache_before
  OR EXISTS(SELECT 1 FROM public.lab_arena_runs
      WHERE round_id='arena-2026-09-20' AND kind='score') THEN
  RAISE EXCEPTION 'Sep20 rerun328 preservation differs' USING ERRCODE='55000';
 END IF;
END $sep20_rerun328$;
COMMIT;
